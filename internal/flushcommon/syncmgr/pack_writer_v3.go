// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package syncmgr

import (
	"context"
	"fmt"
	"path"
	"strconv"
	"strings"

	"github.com/cockroachdb/errors"
	"github.com/google/uuid"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/flushcommon/metacache"
	storage "github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// packedBatchWriter is the common subset both storage.packedRecordBatchWriter
// and storage.packedTextBatchWriter expose to the V3 sync path. Close
// returns a packed.WriterOutput regardless of which concrete writer is
// in use; the V3 sync path treats both uniformly.
type packedBatchWriter interface {
	Write(r storage.Record) error
	Close() (packed.WriterOutput, error)
	GetWrittenUncompressed() uint64
	GetColumnGroupWrittenCompressed(columnGroup typeutil.UniqueID) uint64
	GetColumnGroupWrittenUncompressed(columnGroup typeutil.UniqueID) uint64
	GetWrittenPaths(columnGroup typeutil.UniqueID) string
	GetWrittenRowNum() int64
}

type BulkPackWriterV3 struct {
	*BulkPackWriterV2

	manifestPath string

	// initialManifestPath captures the manifest path observed when Write is
	// invoked. Read-based merges (existing bloom filter / BM25 file lists)
	// reference this stable value instead of bw.manifestPath so they cannot
	// drift after Phase 2's commit updates manifestPath.
	initialManifestPath string

	// statsBlobSize tracks the correction this sync applies to the cumulative
	// manifest-visible bloom-filter / BM25 footprint. Normally it is just this
	// sync's delta. After recovery from a pre-Statistics DataNode it also carries
	// the missing prior-manifest baseline. A non-L0 final flush replaces the
	// bloom entry's prior batch files with one compound file, so its bloom delta
	// is compoundSize-existingBloomSize and may be negative. Reset at the top of
	// Write and fed into the cumulative StatisticsCollector Digest.
	statsBlobSize int64
}

func NewBulkPackWriterV3(metaCache metacache.MetaCache, schema *schemapb.CollectionSchema, chunkManager storage.ChunkManager,
	allocator allocator.Interface, bufferSize, multiPartUploadSize int64,
	storageConfig *indexpb.StorageConfig, columnGroups []storagecommon.ColumnGroup, curManifestPath string, writeRetryOpts ...retry.Option,
) *BulkPackWriterV3 {
	bwV2 := NewBulkPackWriterV2(metaCache, schema, chunkManager, allocator, bufferSize,
		multiPartUploadSize, storageConfig, columnGroups, writeRetryOpts...)
	return &BulkPackWriterV3{
		BulkPackWriterV2: bwV2,
		manifestPath:     curManifestPath,
	}
}

// perBatchStatPaths drops the compound (merged) stat blob from a manifest stat
// entry's paths, keeping only the per-batch blobs. The flush merge rebuilds the
// compound from the per-batch blobs, so a compound written by an EARLIER flush
// of the same Flushing segment (a segment in SegmentState_Flushing emits a flush
// pack on every sync) must be excluded: DeserializeBloomFilterStats short-circuits
// on a compound path and would drop the per-batch blobs written since that flush
// (PK undercount), and the BM25 merge would additively re-count it (double-count).
func perBatchStatPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if _, logidx := path.Split(p); logidx == storage.CompoundStatsType.LogIdx() {
			continue
		}
		out = append(out, p)
	}
	return out
}

// bloomMergeSourcePaths returns the committed bloom blobs that represent all
// rows before the current sync, without double-counting them.
//
// A pre-H22 final manifest may contain both the complete per-batch set and a
// compound blob; use the batch set there to avoid counting the same rows twice.
// A compound-only manifest is the layout emitted by H22; deserializing that
// list yields the same prior per-batch PrimaryKeyStats needed to build a new
// compound if a later final flush is ever scheduled for the segment.
//
// There is no metadata that can distinguish the valid pre-H22 mixed layout
// from an invalid "H22 compound, then a later incremental batch" layout. The
// write path therefore rejects incremental/L0 batch writes when an existing
// compound path is present; this helper relies on that segment-lifecycle
// invariant.
func bloomMergeSourcePaths(paths []string) []string {
	batchPaths := perBatchStatPaths(paths)
	if len(batchPaths) > 0 {
		return batchPaths
	}
	for _, p := range paths {
		if _, logidx := path.Split(p); logidx == storage.CompoundStatsType.LogIdx() {
			return []string{p}
		}
	}
	return nil
}

func hasCompoundStatPath(paths []string) bool {
	for _, p := range paths {
		if _, logidx := path.Split(p); logidx == storage.CompoundStatsType.LogIdx() {
			return true
		}
	}
	return false
}

func compoundStatPath(paths []string) string {
	for _, p := range paths {
		if _, logidx := path.Split(p); logidx == storage.CompoundStatsType.LogIdx() {
			return p
		}
	}
	return ""
}

// newCompoundBloomPath gives every final writer its own object key. The final
// component stays "1" so existing readers still recognize the compound stats
// format, while the unique parent prevents a late writer (for example, one
// left running across a channel handoff) from overwriting another writer's
// manifest-visible bloom object.
func newCompoundBloomPath(basePath string, fieldID int64) string {
	return path.Join(
		basePath,
		fmt.Sprintf("_stats/bloom_filter.%d", fieldID),
		"compound-"+uuid.NewString(),
		storage.CompoundStatsType.LogIdx(),
	)
}

func isIsolatedCompoundBloomPath(compoundPath string) bool {
	if path.Base(compoundPath) != storage.CompoundStatsType.LogIdx() {
		return false
	}
	dirName := path.Base(path.Dir(compoundPath))
	const prefix = "compound-"
	if !strings.HasPrefix(dirName, prefix) {
		return false
	}
	_, err := uuid.Parse(strings.TrimPrefix(dirName, prefix))
	return err == nil
}

// loadPriorPkStats reads the bloom-filter source selected from the pre-commit
// manifest and expands it into per-batch PrimaryKeyStats. The source is either
// the complete per-batch path set or one compound-only blob. Returns nil when
// empty.
func (bw *BulkPackWriterV3) loadPriorPkStats(ctx context.Context, paths []string) ([]*storage.PrimaryKeyStats, error) {
	if len(paths) == 0 {
		return nil, nil
	}
	values, err := bw.chunkManager.MultiRead(ctx, paths)
	if err != nil {
		return nil, err
	}
	blobs := make([]*storage.Blob, len(values))
	for i := range values {
		blobs[i] = &storage.Blob{Value: values[i]}
	}
	return storage.DeserializeBloomFilterStats(paths, blobs)
}

// loadPriorBM25Stats reads the per-batch BM25 blobs already persisted for this
// segment (grouped by field) and merges them into one combined SegmentBM25Stats.
func (bw *BulkPackWriterV3) loadPriorBM25Stats(ctx context.Context, fieldPaths map[int64][]string) (*metacache.SegmentBM25Stats, error) {
	combined := metacache.NewEmptySegmentBM25Stats()
	for fieldID, paths := range fieldPaths {
		if len(paths) == 0 {
			continue
		}
		values, err := bw.chunkManager.MultiRead(ctx, paths)
		if err != nil {
			return nil, err
		}
		for _, v := range values {
			stats, err := storage.NewBM25StatsWithBytes(v)
			if err != nil {
				return nil, err
			}
			combined.Merge(map[int64]*storage.BM25Stats{fieldID: stats})
		}
	}
	return combined, nil
}

// Write executes a SyncPack as do-then-commit:
//
//  1. Phase 1 (slow, runs once): write all data files — parquet/LOB via the
//     loon writer, deltalog via the deltalog writer, stat blobs via the
//     filesystem FFI — and assemble a single packed.ManifestUpdates payload
//     describing every change the segment needs to register.
//  2. Phase 2 (fast, retried on transient loon errors): call
//     packed.CommitManifestUpdates once. The loon transaction handle is
//     opened, all changes are staged, and the transaction is committed in
//     one shot, producing exactly one manifest version bump.
//
// The metaCache is not mutated by Write. The only metaCache state this sync
// produces is bw.preparedStats — this sync's cumulative stats clone — which
// SyncTask.Run installs after the DataCoord ack. The bloom-filter / BM25 merged
// blobs are built at flush from the per-batch blobs already persisted in the
// manifest (see writeStats / writeBM25Stasts), so no per-sync roll into the
// metaCache is needed.
func (bw *BulkPackWriterV3) Write(ctx context.Context, pack *SyncPack) (
	inserts map[int64]*datapb.FieldBinlog,
	deltas *datapb.FieldBinlog,
	stats map[int64]*datapb.FieldBinlog,
	bm25Stats map[int64]*datapb.FieldBinlog,
	manifest string,
	size int64,
	segmentStats *datapb.Statistics,
	err error,
) {
	bw.initialManifestPath = bw.manifestPath
	bw.statsBlobSize = 0

	basePath, baseVersion, parseErr := packed.UnmarshalManifestPath(bw.initialManifestPath)
	if parseErr != nil {
		err = parseErr
		return
	}
	if err = bw.recoverStatsBlobSizeBaseline(pack); err != nil {
		return
	}

	// Phase 1: write files. Each helper returns its contribution to the
	// final ManifestUpdates instead of mutating shared state.
	var (
		insertFiles  packed.WriterOutput
		statEntries  []packed.StatEntry
		deltaEntries []packed.DeltaLogEntry
		bm25Entries  []packed.StatEntry
	)
	defer func() {
		if insertFiles != nil {
			insertFiles.Destroy()
		}
	}()

	if inserts, insertFiles, err = bw.writeInserts(ctx, pack, basePath); err != nil {
		mlog.Warn(ctx, "failed to write insert data", mlog.Err(err))
		return
	}
	if statEntries, err = bw.writeStats(ctx, pack, basePath); err != nil {
		mlog.Warn(ctx, "failed to process stats blob", mlog.Err(err))
		return
	}
	if deltas, deltaEntries, err = bw.writeDelta(ctx, pack, basePath); err != nil {
		mlog.Warn(ctx, "failed to process delta blob", mlog.Err(err))
		return
	}
	if bm25Entries, err = bw.writeBM25Stasts(ctx, pack, basePath); err != nil {
		mlog.Warn(ctx, "failed to process bm25 stats blob", mlog.Err(err))
		return
	}

	updates := &packed.ManifestUpdates{
		NewFiles:  insertFiles,
		DeltaLogs: deltaEntries,
		Stats:     append(statEntries, bm25Entries...),
	}

	// Phase 2: commit the assembled updates. CommitManifestUpdates short-
	// circuits to the unchanged manifest path when updates carry nothing.
	// The outer retry.Do handles transient FFI errors classified as
	// packed.ErrLoonTransient; loon's own optimistic retry covers
	// manifest-version conflicts within a single attempt.
	err = retry.Do(ctx, func() error {
		newPath, commitErr := packed.CommitManifestUpdates(basePath, baseVersion, bw.storageConfig, updates)
		if commitErr != nil {
			return classifyLoonErr(commitErr)
		}
		bw.manifestPath = newPath
		return nil
	}, bw.writeRetryOpts...)
	if err != nil {
		return
	}

	digested := len(inserts) > 0 || bw.statsBlobSize != 0 || len(deltas.GetBinlogs()) > 0

	manifest = bw.manifestPath
	size = bw.sizeWritten

	// V3 feeds the tracked statsBlobSize instead of summing a stats array (it
	// returns no stats array); finalizeStats produces the cumulative Statistics.
	segmentStats, err = bw.finalizeStats(pack, digested, inserts, deltas, bw.statsBlobSize)
	return
}

// classifyLoonErr maps loon FFI failures to retryable errors and everything
// else to retry.Unrecoverable so the outer retry loop terminates immediately.
//
// NOTE: today milvus-storage does not reliably preserve structured error codes
// through every FFI path, so packed.ErrLoonTransient covers ALL loon errors,
// including non-recoverable IO failures. The bounded retry budget keeps the
// worst case finite. Once error codes survive end-to-end, narrow the retryable
// set here.
func classifyLoonErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, packed.ErrLoonTransient) {
		return err
	}
	return retry.Unrecoverable(err)
}

// writeInserts writes the insert data files for the SyncPack and returns
// the produced WriterOutput plus the per-column-group binlog metadata.
// The returned WriterOutput is owned by the caller and must be Destroy'd
// after the surrounding commit (success or failure).
func (bw *BulkPackWriterV3) writeInserts(ctx context.Context, pack *SyncPack, basePath string) (logs map[int64]*datapb.FieldBinlog, files packed.WriterOutput, err error) {
	if len(pack.insertData) == 0 {
		return make(map[int64]*datapb.FieldBinlog), nil, nil
	}

	rec, err := bw.serializeBinlog(ctx, pack)
	if err != nil {
		return nil, nil, err
	}
	defer rec.Release()

	tsFrom, tsTo := bw.getTsRange(rec)
	pluginContextPtr := bw.getPluginContext(pack.collectionID)
	writerFormat, schemaBasedFormats, err := bw.resolveInsertWriterFormats()
	if err != nil {
		return nil, nil, err
	}

	// LOB base path is at partition level: {basePath}/.. = {root}/insert_log/{coll}/{part}
	partitionBasePath := path.Dir(basePath)
	textColumnConfigs := buildTextColumnConfigs(bw.schema, partitionBasePath)
	var w packedBatchWriter
	if len(textColumnConfigs) > 0 {
		mlog.Info(ctx, "using TEXT-aware writer for import",
			mlog.Int("textFieldCount", len(textColumnConfigs)),
			mlog.String("basePath", basePath))
		w, err = storage.NewPackedTextBatchWriter("", basePath, bw.schema,
			bw.bufferSize, bw.multiPartUploadSize, bw.columnGroups, bw.storageConfig, textColumnConfigs, writerFormat, schemaBasedFormats)
	} else {
		w, err = storage.NewPackedRecordBatchWriter(basePath, bw.schema,
			bw.bufferSize, bw.multiPartUploadSize, bw.columnGroups, bw.storageConfig, pluginContextPtr, writerFormat, schemaBasedFormats)
	}
	if err != nil {
		return nil, nil, err
	}

	// Ensure the FFI writer's C resources are reclaimed even if Write fails
	// before we reach the explicit Close below. Once Close has been called
	// (success or failure) the writer's own defer has already released its
	// handle and properties, so closeAttempted gates against a redundant
	// second Close from this defer.
	closeAttempted := false
	defer func() {
		if !closeAttempted {
			if out, closeErr := w.Close(); closeErr == nil && out != nil {
				out.Destroy()
			}
		}
	}()

	if err = w.Write(rec); err != nil {
		mlog.Warn(ctx, "failed to write inserts",
			mlog.FieldCollectionID(pack.collectionID),
			mlog.FieldSegmentID(pack.segmentID),
			mlog.Err(err))
		return nil, nil, err
	}

	closeAttempted = true
	if files, err = w.Close(); err != nil {
		return nil, nil, err
	}

	getFieldNullCounts := func(columnGroup storagecommon.ColumnGroup) map[int64]int64 {
		result := make(map[int64]int64, len(columnGroup.Fields))
		for _, fieldID := range columnGroup.Fields {
			if col := rec.Column(fieldID); col != nil {
				result[fieldID] = int64(col.NullN())
			}
		}
		return result
	}

	logs = buildV3ColumnGroupFieldBinlogs(
		bw.columnGroups,
		w.GetWrittenRowNum(),
		tsFrom,
		tsTo,
		func(columnGroupID int64) int64 { return int64(w.GetColumnGroupWrittenCompressed(columnGroupID)) },
		func(columnGroupID int64) int64 { return int64(w.GetColumnGroupWrittenUncompressed(columnGroupID)) },
		nil,
		w.GetWrittenPaths,
		getFieldNullCounts,
	)
	return logs, files, nil
}

func buildV3ColumnGroupFieldBinlogs(
	columnGroups []storagecommon.ColumnGroup,
	entriesNum int64,
	tsFrom uint64,
	tsTo uint64,
	compressedSize func(columnGroupID int64) int64,
	memorySize func(columnGroupID int64) int64,
	logID func(columnGroupID int64) int64,
	logPath func(columnGroupID int64) string,
	fieldNullCounts func(columnGroup storagecommon.ColumnGroup) map[int64]int64,
) map[int64]*datapb.FieldBinlog {
	logs := make(map[int64]*datapb.FieldBinlog, len(columnGroups))
	for _, columnGroup := range columnGroups {
		columnGroupID := columnGroup.GroupID
		fieldBinlog := &datapb.FieldBinlog{
			FieldID:     columnGroupID,
			ChildFields: columnGroup.Fields,
			Format:      columnGroup.Format,
		}
		if entriesNum > 0 {
			binlog := &datapb.Binlog{
				EntriesNum:    entriesNum,
				TimestampFrom: tsFrom,
				TimestampTo:   tsTo,
			}
			if compressedSize != nil {
				binlog.LogSize = compressedSize(columnGroupID)
			}
			if memorySize != nil {
				binlog.MemorySize = memorySize(columnGroupID)
			}
			if logID != nil {
				binlog.LogID = logID(columnGroupID)
			}
			if logPath != nil {
				binlog.LogPath = logPath(columnGroupID)
			}
			if fieldNullCounts != nil {
				binlog.FieldNullCounts = fieldNullCounts(columnGroup)
			}
			fieldBinlog.Binlogs = []*datapb.Binlog{binlog}
		}
		logs[columnGroupID] = fieldBinlog
	}
	return logs
}

func (bw *BulkPackWriterV3) resolveInsertWriterFormats() (string, []string, error) {
	writerFormat := paramtable.Get().DataNodeCfg.StorageFormat.GetValue()
	if bw.initialManifestPath != "" {
		_, version, err := packed.UnmarshalManifestPath(bw.initialManifestPath)
		if err != nil {
			return "", nil, err
		}
		if version != packed.ManifestEarliest {
			for _, columnGroup := range bw.columnGroups {
				if columnGroup.Format == "" {
					return "", nil, merr.WrapErrDataIntegrityMsg("column group %d fields %v missing format for existing manifest %s",
						columnGroup.GroupID, columnGroup.Fields, bw.initialManifestPath)
				}
			}
		}
	}
	schemaBasedFormats := storagecommon.ColumnGroupFormats(bw.columnGroups, writerFormat)
	return writerFormat, schemaBasedFormats, nil
}

// writeDelta writes the deltalog file and returns the DeltaLogEntry list
// the caller will fold into ManifestUpdates plus a pathless delta summary
// FieldBinlog (EntriesNum + MemorySize) used by compaction-trigger
// decisions.
func (bw *BulkPackWriterV3) writeDelta(ctx context.Context, pack *SyncPack, basePath string) (*datapb.FieldBinlog, []packed.DeltaLogEntry, error) {
	if pack.deltaData == nil || pack.deltaData.RowCount == 0 {
		return nil, nil, nil
	}

	pkField, err := typeutil.GetPrimaryFieldSchema(bw.schema)
	if err != nil {
		return nil, nil, merr.Wrap(err, "primary key field not found")
	}

	logID, err := bw.allocator.AllocOne()
	if err != nil {
		return nil, nil, err
	}
	deltaPath := metautil.BuildDeltaLogPathV3(basePath, logID)

	writer, err := storage.NewDeltalogWriter(
		ctx, pack.collectionID, pack.partitionID, pack.segmentID, logID, pkField.DataType, deltaPath,
		storage.WithVersion(storage.StorageV2),
		storage.WithStorageConfig(bw.storageConfig),
	)
	if err != nil {
		return nil, nil, merr.Wrap(err, "failed to create deltalog writer")
	}

	record, tsFrom, tsTo, err := storage.BuildDeleteRecord(pack.deltaData.Pks, pack.deltaData.Tss)
	if err != nil {
		return nil, nil, merr.Wrap(err, "failed to build delete record")
	}
	defer record.Release()

	if err := writer.Write(record); err != nil {
		return nil, nil, merr.Wrap(err, "failed to write delta record")
	}
	if err := writer.Close(); err != nil {
		return nil, nil, merr.Wrap(err, "failed to close delta writer")
	}

	bw.sizeWritten += pack.deltaData.Size()
	deltaMemSize := int64(writer.GetWrittenUncompressed())

	summary := &datapb.FieldBinlog{
		Binlogs: []*datapb.Binlog{{
			LogID:         logID,
			EntriesNum:    pack.deltaData.RowCount,
			MemorySize:    deltaMemSize,
			TimestampFrom: tsFrom,
			TimestampTo:   tsTo,
		}},
	}
	return summary, []packed.DeltaLogEntry{{Path: deltaPath, NumEntries: pack.deltaData.RowCount}}, nil
}

// hasExistingManifest reports whether initialManifestPath points at a committed
// manifest that can carry prior-batch stats (a version past earliest). A
// brand-new segment (empty path or earliest version) has no prior stats to
// preserve, so the prior-stats read is skipped for it.
func (bw *BulkPackWriterV3) hasExistingManifest() bool {
	if bw.initialManifestPath == "" {
		return false
	}
	_, version, err := packed.UnmarshalManifestPath(bw.initialManifestPath)
	if err != nil {
		return false
	}
	return version != packed.ManifestEarliest
}

// recoverStatsBlobSizeBaseline repairs the one supported recovery case where
// the manifest already contains V3 bloom/BM25 objects but the recovered
// collector reports no usable footprint. This happens during a rolling upgrade
// from a pre-Statistics DataNode: V3 has no stats-log arrays, so DataCoord's
// legacy array fallback persists StatsBinlogSize=0 even though the manifest is
// not empty.
//
// The repair is kept in statsBlobSize instead of mutating the live collector.
// finalizeStats folds it into the prepared clone, which SyncTask installs only
// after DataCoord acknowledges the same committed manifest. Reading before any
// object writes also makes a manifest failure side-effect free.
func (bw *BulkPackWriterV3) recoverStatsBlobSizeBaseline(pack *SyncPack) error {
	if !bw.hasExistingManifest() {
		return nil
	}
	segment, ok := bw.metaCache.GetSegmentByID(pack.segmentID)
	if !ok {
		return merr.WrapErrSegmentNotFound(pack.segmentID)
	}
	if stats := segment.Statistics().Publish(); stats != nil && stats.GetStatsBinlogSize() > 0 {
		return nil
	}

	manifestSize, err := packed.StatsBinlogSizeFromManifest(bw.initialManifestPath, bw.storageConfig)
	if err != nil {
		return merr.Wrap(err, "failed to recover stats footprint from prior manifest")
	}
	bw.statsBlobSize = manifestSize
	return nil
}

// writeStats writes bloom filter stat blobs under basePath/_stats and
// returns the resulting StatEntry list. The caller folds the entries into
// a ManifestUpdates that commits atomically with inserts / delta / bm25.
func (bw *BulkPackWriterV3) writeStats(ctx context.Context, pack *SyncPack, basePath string) ([]packed.StatEntry, error) {
	finalCompound := pack.isFlush && pack.level != datapb.SegmentLevel_L0
	if len(pack.insertData) == 0 && !finalCompound {
		return nil, nil
	}

	serializer, err := NewStorageSerializer(bw.metaCache, bw.schema)
	if err != nil {
		return nil, err
	}
	singlePKStats, err := serializer.buildPrimaryKeyStats(pack)
	if err != nil {
		return nil, err
	}

	pkFieldID := serializer.pkField.GetFieldID()

	var files []string
	var existingMemorySize int64
	// footprintDelta is this sync's change to the manifest-visible bloom
	// footprint. Incremental/L0 writes append one batch blob. A non-L0 final
	// flush replaces every prior path with one compound path, so the delta is
	// compoundSize-existingMemorySize rather than the number of bytes written.
	var footprintDelta int64

	// Preserve existing bloom filter files from previous batches.
	// loon_transaction_update_stat uses replace semantics, so we must
	// merge previously written files into the new entry.
	statKey := fmt.Sprintf("bloom_filter.%d", pkFieldID)
	var priorBloomPaths []string
	var existingHasCompound bool
	var existingCompoundPath string
	if bw.hasExistingManifest() {
		// A transient manifest read failure must NOT be swallowed: under loon
		// replace semantics the committed StatEntry would then drop the prior
		// per-batch paths, so the flush compound would hold only this sync's PKs
		// — losing prior-batch PKs for delete-routing / PK-pruning. Propagate so
		// the sync retries instead of persisting a truncated bloom set.
		existingStats, err := packed.GetManifestStats(bw.initialManifestPath, bw.storageConfig)
		if err != nil {
			return nil, merr.Wrap(err, "failed to read prior bloom stats from manifest")
		}
		if existing, ok := existingStats[statKey]; ok && len(existing.Paths) > 0 {
			// Select a complete, non-duplicated history source from the committed
			// paths. Mixed layouts use their per-batch set; compound-only layouts
			// expand the compound back into its original per-batch stats.
			priorBloomPaths = bloomMergeSourcePaths(existing.Paths)
			existingHasCompound = hasCompoundStatPath(existing.Paths)
			existingCompoundPath = compoundStatPath(existing.Paths)
			files = append(files, existing.Paths...)
			if memStr, ok := existing.Metadata["memory_size"]; ok {
				existingMem, _ := strconv.ParseInt(memStr, 10, 64)
				existingMemorySize += existingMem
			}
		}
	}

	// A non-L0 final flush writes only the compound object. It is built from
	// stats referenced by the exact committed manifest version passed to this
	// writer plus the current in-memory batch stats. No batch log ID is
	// allocated, no batch blob is encoded, and the replacement manifest entry
	// registers only the compound path.
	if finalCompound {
		// A repeated empty final over an already isolated compound is a no-op.
		// Legacy compound-only manifests still use the shared direct "/1" key;
		// rewrite those once so a late handoff writer can no longer mutate the
		// object referenced by this manifest.
		if singlePKStats == nil && len(files) == 1 && isIsolatedCompoundBloomPath(existingCompoundPath) {
			return nil, nil
		}

		priorStats, err := bw.loadPriorPkStats(ctx, priorBloomPaths)
		if err != nil {
			return nil, err
		}
		mergedStats := priorStats
		if singlePKStats != nil {
			mergedStats = append(mergedStats, singlePKStats)
		}
		if len(mergedStats) == 0 {
			return nil, nil
		}
		segment, ok := bw.metaCache.GetSegmentByID(pack.segmentID)
		if !ok {
			return nil, merr.WrapErrSegmentNotFound(pack.segmentID)
		}
		mergedStatsBlob, err := serializer.serializeMergedPkStatsList(mergedStats, segment.NumOfRows())
		if err != nil {
			return nil, err
		}
		mergedFullPath := newCompoundBloomPath(basePath, pkFieldID)
		if err := packed.WriteFile(bw.storageConfig, mergedFullPath, mergedStatsBlob.Value); err != nil {
			return nil, err
		}
		compoundSize := int64(len(mergedStatsBlob.Value))
		bw.sizeWritten += compoundSize
		bw.statsBlobSize += compoundSize - existingMemorySize

		return []packed.StatEntry{{
			Key:      statKey,
			Files:    []string{mergedFullPath},
			Metadata: map[string]string{"memory_size": strconv.FormatInt(compoundSize, 10)},
		}}, nil
	}

	// Incremental syncs and L0 final flushes preserve the per-batch format.
	// A compound path means this segment has already crossed its non-L0 final
	// boundary (or carries the legacy final layout). Appending a batch after that
	// point would create an ambiguous compound+batch manifest in which the
	// resolver returns the old compound and hides the new batch.
	if existingHasCompound {
		return nil, merr.WrapErrDataIntegrityMsg(
			"segment %d cannot append batch bloom stats after compound stats were committed",
			pack.segmentID)
	}
	batchStatsBlob, err := serializer.serializePrimaryKeyStats(singlePKStats, pack.batchRows)
	if err != nil {
		return nil, err
	}
	id, err := bw.allocator.AllocOne()
	if err != nil {
		return nil, err
	}
	relPath := fmt.Sprintf("_stats/bloom_filter.%d/%d", pkFieldID, id)
	fullPath := path.Join(basePath, relPath)
	if err := packed.WriteFile(bw.storageConfig, fullPath, batchStatsBlob.Value); err != nil {
		return nil, err
	}
	batchSize := int64(len(batchStatsBlob.Value))
	bw.sizeWritten += batchSize
	memorySize := existingMemorySize + batchSize
	footprintDelta += batchSize
	files = append(files, fullPath)

	// Feed the collector this sync's manifest-footprint change; memorySize stays
	// cumulative for the manifest StatEntry below.
	bw.statsBlobSize += footprintDelta

	return []packed.StatEntry{{
		Key:      statKey,
		Files:    files,
		Metadata: map[string]string{"memory_size": fmt.Sprintf("%d", memorySize)},
	}}, nil
}

// writeBM25Stasts writes BM25 stat blobs under basePath/_stats and returns
// the resulting StatEntry list. The caller folds the entries into a
// ManifestUpdates that commits atomically with inserts / delta / stats.
func (bw *BulkPackWriterV3) writeBM25Stasts(ctx context.Context, pack *SyncPack, basePath string) ([]packed.StatEntry, error) {
	if len(pack.bm25Stats) == 0 {
		return nil, nil
	}

	serializer, err := NewStorageSerializer(bw.metaCache, bw.schema)
	if err != nil {
		return nil, err
	}
	bm25Blobs, err := serializer.serializeBM25Stats(pack)
	if err != nil {
		return nil, err
	}

	// Track per-field files and memory sizes, then build stat entries at the end
	type fieldStats struct {
		files      []string
		memorySize int64
	}
	fieldMap := make(map[int64]*fieldStats)

	// newBlobBytes is only the BM25 blob bytes WRITTEN this sync across all
	// fields (NOT the preserved existing footprint). The collector accumulates
	// these per-sync deltas; the merged blob writes only at flush and per-sync
	// blobs use unique keys, so Σ(newBlobBytes) == final cumulative memory_size.
	var newBlobBytes int64

	// Preserve existing BM25 stat files from previous batches. priorBM25Paths
	// captures them per field (already absolute, chunkManager-readable) for the
	// flush merge below, so we don't re-read the manifest via a StatsResolver.
	priorBM25Paths := make(map[int64][]string)
	if bw.hasExistingManifest() {
		// Same as the bloom path: propagate a transient read failure rather than
		// silently dropping prior BM25 blobs and committing a truncated set.
		existingStats, err := packed.GetManifestStats(bw.initialManifestPath, bw.storageConfig)
		if err != nil {
			return nil, merr.Wrap(err, "failed to read prior bm25 stats from manifest")
		}
		for key, existing := range existingStats {
			prefix, fieldID, ok := packed.ParseStatKey(key)
			if !ok || prefix != "bm25" || len(existing.Paths) == 0 {
				continue
			}
			priorBM25Paths[fieldID] = perBatchStatPaths(existing.Paths)
			fs := &fieldStats{files: existing.Paths}
			if memStr, ok := existing.Metadata["memory_size"]; ok {
				fs.memorySize, _ = strconv.ParseInt(memStr, 10, 64)
			}
			fieldMap[fieldID] = fs
		}
	}

	for fieldID, blob := range bm25Blobs {
		id, err := bw.allocator.AllocOne()
		if err != nil {
			return nil, err
		}

		relPath := fmt.Sprintf("_stats/bm25.%d/%d", fieldID, id)
		fullPath := path.Join(basePath, relPath)
		if err := packed.WriteFile(bw.storageConfig, fullPath, blob.Value); err != nil {
			return nil, err
		}
		bw.sizeWritten += int64(len(blob.Value))

		fs := fieldMap[fieldID]
		if fs == nil {
			fs = &fieldStats{}
			fieldMap[fieldID] = fs
		}
		fs.files = append(fs.files, fullPath)
		fs.memorySize += int64(len(blob.Value))
		newBlobBytes += int64(len(blob.Value))
	}

	// Write merged BM25 stats on flush. Build the merged blob from the
	// per-batch BM25 blobs already persisted in the pre-commit manifest
	// (priorBM25Paths, read once above) plus this flush batch — no metaCache
	// history is consulted.
	if pack.isFlush && pack.level != datapb.SegmentLevel_L0 && hasBM25Function(bw.schema) {
		combined, err := bw.loadPriorBM25Stats(ctx, priorBM25Paths)
		if err != nil {
			return nil, err
		}
		combined.Merge(pack.bm25Stats)
		mergedBM25Blob, err := serializer.serializeMergedBM25StatsFrom(combined)
		if err != nil {
			return nil, err
		}
		for fieldID, blob := range mergedBM25Blob {
			mergedRelPath := fmt.Sprintf("_stats/bm25.%d/%d", fieldID, int64(storage.CompoundStatsType))
			mergedFullPath := path.Join(basePath, mergedRelPath)
			if err := packed.WriteFile(bw.storageConfig, mergedFullPath, blob.Value); err != nil {
				return nil, err
			}
			bw.sizeWritten += int64(len(blob.Value))

			fs := fieldMap[fieldID]
			if fs == nil {
				fs = &fieldStats{}
				fieldMap[fieldID] = fs
			}
			fs.files = append(fs.files, mergedFullPath)
			fs.memorySize += int64(len(blob.Value))
			newBlobBytes += int64(len(blob.Value))
		}
	}

	var entries []packed.StatEntry
	for fieldID, fs := range fieldMap {
		entries = append(entries, packed.StatEntry{
			Key:      fmt.Sprintf("bm25.%d", fieldID),
			Files:    fs.files,
			Metadata: map[string]string{"memory_size": fmt.Sprintf("%d", fs.memorySize)},
		})
	}

	// Feed the collector only THIS SYNC's newly-written BM25 blob bytes (summed
	// across all fields); fs.memorySize stays cumulative for the StatEntry above.
	bw.statsBlobSize += newBlobBytes

	return entries, nil
}

// buildTextColumnConfigs builds TextColumnConfig for all TEXT fields in the schema.
// partitionBasePath is the partition-level path: {root}/insert_log/{coll}/{part}
// Per-column LOB path: {partitionBasePath}/lobs/{field_id}
func buildTextColumnConfigs(schema *schemapb.CollectionSchema, partitionBasePath string) []packed.TextColumnConfig {
	var configs []packed.TextColumnConfig
	for _, field := range schema.GetFields() {
		if field.GetDataType() == schemapb.DataType_Text {
			fieldID := field.GetFieldID()
			configs = append(configs, packed.TextColumnConfig{
				FieldID:             fieldID,
				LobBasePath:         path.Join(partitionBasePath, "lobs", strconv.FormatInt(fieldID, 10)),
				InlineThreshold:     paramtable.Get().DataNodeCfg.TextInlineThreshold.GetAsInt64(),
				MaxLobFileBytes:     paramtable.Get().DataNodeCfg.TextMaxLobFileBytes.GetAsInt64(),
				FlushThresholdBytes: paramtable.Get().DataNodeCfg.TextFlushThresholdBytes.GetAsInt64(),
			})
		}
	}
	return configs
}
