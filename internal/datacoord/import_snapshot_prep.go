// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datacoord

import (
	"context"
	"errors"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/snapshotio/storage"
	milvusstorage "github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/hookutil"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// errSnapshotExpansionRetry marks an expansion failure that must be retried
// without failing the job. File-ID allocation is the only such step: the plan
// has not been published yet, so the next tick retries from scratch.
var errSnapshotExpansionRetry = errors.New("snapshot expansion retry")

// expandSnapshotJob expands a pending snapshot job's metadata reference into
// its executable file plan. It performs source I/O and allocates file IDs but
// persists nothing: callers own the durability boundary and the failure
// transition. The returned plan is fully validated, including the allocated
// IDs, before it is handed back.
func expandSnapshotJob(ctx context.Context, meta *meta, alloc allocator.Allocator, job ImportJob) ([]*internalpb.ImportFile, importutilv2.Options, []*internalpb.SnapshotImportL0Source, error) {
	// Recovered jobs must validate the reference before doing source I/O.
	if err := importutilv2.ValidateSnapshotImportPlan(job.GetFiles(), job.GetOptions(), nil); err != nil {
		return nil, nil, nil, err
	}
	metadata, err := readSnapshotImportMetadata(ctx, meta.chunkManager, job.GetOptions())
	if err != nil {
		return nil, nil, nil, err
	}
	uri, _ := funcutil.GetAttrByKeyFromRepeatedKV(importutilv2.SnapshotSourceURI, job.GetOptions())
	files, options, sources, err := expandSnapshotImportFiles(ctx, job.GetPartitionIDs(), meta.chunkManager,
		job.GetSchema(), uri, job.GetOptions(), metadata)
	if err != nil {
		return nil, nil, nil, err
	}
	if ctx.Err() != nil {
		return nil, nil, nil, ctx.Err()
	}
	start, _, err := alloc.AllocN(int64(len(files)))
	if err != nil {
		return nil, nil, nil, errors.Join(errSnapshotExpansionRetry, err)
	}
	for i, file := range files {
		file.Id = start + int64(i)
	}
	// Preparation can rebase paths and add external/routing descriptors after
	// its intermediate checks. Validate the complete durable plan, including
	// the allocated IDs, before handing it back.
	if err := importutilv2.ValidateSnapshotImportPlan(files, options, sources); err != nil {
		return nil, nil, nil, err
	}
	return files, options, sources, nil
}

// snapshotExpansionTimeout caps one V3 expansion attempt when the job has no
// earlier deadline. Expansion runs on the V3 checker goroutine, so an unbounded
// source read would stall every other V3 job. V2's scheduler runs off the
// checker goroutine and keeps its original deadline behavior.
const snapshotExpansionTimeout = 10 * time.Minute

// snapshotExpansionContext bounds one expansion by the earlier of the job
// deadline and, when capAttempt is set, snapshotExpansionTimeout. capped
// reports that the cap, not the job's own deadline, fired.
func snapshotExpansionContext(parent context.Context, job ImportJob, capAttempt bool) (context.Context, context.CancelFunc, bool) {
	var deadline time.Time
	capped := false
	if capAttempt {
		deadline = time.Now().Add(snapshotExpansionTimeout)
		capped = true
	}
	if timeoutTs := job.GetTimeoutTs(); timeoutTs != 0 && timeoutTs != math.MaxUint64 {
		if jobDeadline, _ := tsoutil.ParseTS(timeoutTs); deadline.IsZero() || jobDeadline.Before(deadline) {
			deadline, capped = jobDeadline, false
		}
	}
	if deadline.IsZero() {
		ctx, cancel := context.WithCancel(parent)
		return ctx, cancel, false
	}
	ctx, cancel := context.WithDeadline(parent, deadline)
	return ctx, cancel, capped
}

// readSnapshotImportMetadata reopens the exact reference validated when the
// job was submitted. Callers must keep the snapshot and all referenced objects
// unchanged and readable through retries and imports in every cluster. Import
// does not detect in-place replacement of otherwise valid source objects.
func readSnapshotImportMetadata(ctx context.Context, cm milvusstorage.ChunkManager, options importutilv2.Options) (*datapb.SnapshotMetadata, error) {
	uri, _ := funcutil.GetAttrByKeyFromRepeatedKV(importutilv2.SnapshotSourceURI, options)
	if !importutilv2.HasExternalSource(options) {
		if err := storage.ValidateInstanceSnapshotImportURI(storage.InstanceConfigFromParamtable(paramtable.Get()), uri); err != nil {
			return nil, err
		}
	}
	cm, _, err := importutilv2.ResolveSnapshotImportStorage(ctx, cm, nil, uri, options)
	if err != nil {
		return nil, err
	}
	return storage.NewSnapshotReader(cm).ReadMetadata(ctx, uri)
}

// prepareSnapshotImportFiles records the top-level metadata URI and layout
// in WAL body options; it never embeds the metadata itself.
// Segment metadata and physical manifest I/O belong to snapshot
// preparation/PreImport, not the user-facing create RPC. A version-9 descriptor
// is not executable by a worker.
// broadcastImport validates the completed descriptor before broadcasting it.
func prepareSnapshotImportFiles(ctx context.Context, partitions []int64, cm milvusstorage.ChunkManager,
	schema *schemapb.CollectionSchema, files []*internalpb.ImportFile, options importutilv2.Options,
) ([]*internalpb.ImportFile, importutilv2.Options, error) {
	if err := importutilv2.ValidateSnapshotSourceOptions(options); err != nil {
		return nil, nil, err
	}
	if !importutilv2.IsSnapshotSource(options) {
		return files, options, nil
	}
	if cm == nil {
		return nil, nil, merr.WrapErrServiceInternalMsg("chunk manager cannot be nil")
	}
	if len(files) != 1 || len(files[0].GetPaths()) != 1 {
		return nil, nil, merr.WrapErrImportFailedMsg("snapshot-source import requires exactly one metadata URI")
	}
	uri := strings.TrimSpace(files[0].GetPaths()[0])
	bucket, _, _, err := storage.ParseForeignURI(uri)
	if err != nil {
		return nil, nil, err
	}
	if bucket == "" {
		return nil, nil, merr.WrapErrImportFailedMsg("snapshot-source import requires a complete metadata URI")
	}
	if !importutilv2.HasExternalSource(options) {
		if err := storage.ValidateInstanceSnapshotImportURI(storage.InstanceConfigFromParamtable(paramtable.Get()), uri); err != nil {
			return nil, nil, err
		}
	}
	cm, _, err = importutilv2.ResolveSnapshotImportStorage(ctx, cm, nil, uri, options)
	if err != nil {
		return nil, nil, err
	}
	reader := storage.NewSnapshotReader(cm)
	metadata, err := reader.ReadMetadata(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	snapshot, err := reader.ReadSnapshotFromMetadata(ctx, uri, metadata, false)
	if err != nil {
		return nil, nil, err
	}
	if err := storage.ValidateSnapshotMetadataLocation(uri, snapshot.SnapshotInfo); err != nil {
		return nil, nil, err
	}
	if err := storage.ValidateExternalSnapshotPaths(uri, snapshot, nil); err != nil {
		return nil, nil, err
	}
	options, err = prepareSnapshotImportOptions(schema, snapshot.Collection.GetSchema(), options)
	if err != nil {
		return nil, nil, err
	}
	if _, err := resolveSnapshotPartitionMapping(snapshot.Collection.GetPartitions(), partitions, schema, options); err != nil {
		return nil, nil, err
	}
	layout := "referenced"
	switch snapshot.Layout {
	case datapb.SnapshotLayout_SnapshotLayoutReferenced:
	case datapb.SnapshotLayout_SnapshotLayoutSelfContained:
		layout = "self-contained"
	default:
		return nil, nil, merr.WrapErrImportFailedMsg("unsupported snapshot layout: %s", snapshot.Layout.String())
	}
	options = append(append(importutilv2.Options(nil), options...),
		&commonpb.KeyValuePair{Key: importutilv2.SnapshotSourceURI, Value: uri},
		&commonpb.KeyValuePair{Key: importutilv2.SnapshotLayout, Value: layout})
	files = []*internalpb.ImportFile{{SnapshotSource: &internalpb.SnapshotImportSource{
		Version: importutilv2.SnapshotPreparationVersion,
	}}}
	return files, options, nil
}

// expandSnapshotImportFiles converts parsed snapshot metadata into one ImportFile
// per selected source segment in the preparation phase. Callers must keep source
// objects immutable; preparation uses their exact manifest versions, never latest.
// The complete plan is persisted before task creation. Preparation never reopens
// the top-level object. Only snapshot preparation jobs reach this
// function; ordinary Import passthrough belongs to prepareSnapshotImportFiles.
// Caller-owned options are never modified.
func expandSnapshotImportFiles(
	ctx context.Context,
	targetPartitionIDs []int64,
	cm milvusstorage.ChunkManager,
	targetSchema *schemapb.CollectionSchema,
	metadataURI string,
	options importutilv2.Options,
	metadata *datapb.SnapshotMetadata,
) ([]*internalpb.ImportFile, importutilv2.Options, []*internalpb.SnapshotImportL0Source, error) {
	if err := importutilv2.ValidateSnapshotSourceOptions(options); err != nil {
		return nil, nil, nil, err
	}
	if cm == nil {
		return nil, nil, nil, merr.WrapErrServiceInternalMsg("chunk manager cannot be nil")
	}

	metadataPath := strings.TrimSpace(metadataURI)
	// Keep the captured URI's storage identity when resolving source readers
	// for pending jobs, including retries and recovery. Already-prepared jobs
	// skip preparation and retain their existing representation.
	bucket, _, _, err := storage.ParseForeignURI(metadataPath)
	if err != nil {
		return nil, nil, nil, err
	}
	if bucket == "" {
		return nil, nil, nil, merr.WrapErrParameterInvalidMsg("snapshot-source import requires a complete metadata URI; bare object keys are not supported")
	}
	if !importutilv2.HasExternalSource(options) {
		if err := storage.ValidateInstanceSnapshotImportURI(storage.InstanceConfigFromParamtable(paramtable.Get()), metadataPath); err != nil {
			return nil, nil, nil, err
		}
	}
	cm, _, err = importutilv2.ResolveSnapshotImportStorage(ctx, cm, nil, metadataPath, options)
	if err != nil {
		return nil, nil, nil, err
	}
	if err := storage.ValidateSnapshotObjectPathForBucket(cm, "snapshot_source", metadataPath, ""); err != nil {
		return nil, nil, nil, err
	}
	reader := storage.NewSnapshotReader(cm)
	snapshot, err := reader.ReadSnapshotFromMetadata(ctx, metadataPath, metadata, true)
	if err != nil {
		return nil, nil, nil, merr.Wrap(err, "failed to read snapshot import source")
	}
	if snapshot == nil {
		return nil, nil, nil, merr.WrapErrImportSysFailed("snapshot reader returned no data")
	}
	if err := storage.ValidateSnapshotMetadataLocation(metadataPath, snapshot.SnapshotInfo); err != nil {
		return nil, nil, nil, err
	}
	if snapshot.Layout != datapb.SnapshotLayout_SnapshotLayoutReferenced &&
		snapshot.Layout != datapb.SnapshotLayout_SnapshotLayoutSelfContained {
		return nil, nil, nil, merr.WrapErrImportFailedMsg("unsupported snapshot layout: %s", snapshot.Layout.String())
	}
	// Import accepts old snapshots and uses captured commit timestamps as-is.
	// Old producers omitted commit timestamps, so a zero may mean either
	// an ordinary segment or a lost commit-time override. In the latter case,
	// deletes are compared against raw row timestamps and can remove rows that
	// should survive (row=100, actual commit=300, delete=200). This is an accepted
	// historical snapshot limitation, not permission to skip any delete input.
	// Neither scanning manifests nor re-exporting can recover the missing time;
	// do not probe manifests or infer it from the live catalog/destination job.
	options, err = prepareSnapshotImportOptions(targetSchema, snapshot.Collection.GetSchema(), options)
	if err != nil {
		return nil, nil, nil, err
	}
	partitionMapping, err := resolveSnapshotPartitionMapping(snapshot.Collection.GetPartitions(), targetPartitionIDs, targetSchema, options)
	if err != nil {
		return nil, nil, nil, err
	}

	// The snapshot defines the source scope. Import all data partitions while
	// retaining their original IDs/channels for L0 matching; target placement
	// is independent and follows ordinary Import target partition routing.
	segments := make([]*datapb.SegmentDescription, 0)
	for _, segment := range snapshot.Segments {
		if segment == nil {
			return nil, nil, nil, merr.WrapErrImportFailed("snapshot contains a nil segment")
		}
		if segment.GetSegmentLevel() != datapb.SegmentLevel_L0 {
			if partitionMapping != nil && partitionMapping[segment.GetPartitionId()] == 0 {
				return nil, nil, nil, merr.WrapErrImportFailedMsg("snapshot segment %d belongs to an unknown source partition %d", segment.GetSegmentId(), segment.GetPartitionId())
			}
			segments = append(segments, segment)
		}
	}
	if len(segments) == 0 {
		return nil, nil, nil, merr.WrapErrImportFailedMsg("snapshot contains no data segments")
	}
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].GetSegmentId() < segments[j].GetSegmentId()
	})
	if len(segments) > paramtable.Get().DataCoordCfg.MaxFilesPerImportReq.GetAsInt() {
		return nil, nil, nil, merr.WrapErrImportFailedMsg("The max number of import files should not exceed %d, but got %d",
			paramtable.Get().DataCoordCfg.MaxFilesPerImportReq.GetAsInt(), len(segments))
	}

	seenManifests := make(map[string]struct{}, len(segments))
	result := make([]*internalpb.ImportFile, 0, len(segments))
	for _, segment := range segments {
		if segment.GetStorageVersion() != milvusstorage.StorageV3 {
			return nil, nil, nil, merr.WrapErrOperationNotSupportedMsg(
				"snapshot-source import only supports StorageV3 segments, segment %d uses storage version %d",
				segment.GetSegmentId(), segment.GetStorageVersion(),
			)
		}
		manifestPath := segment.GetManifestPath()
		if manifestPath == "" {
			return nil, nil, nil, merr.WrapErrImportFailedMsg(
				"StorageV3 segment %d has no manifest path", segment.GetSegmentId(),
			)
		}
		_, version, err := packed.UnmarshalManifestPath(manifestPath)
		if err != nil {
			return nil, nil, nil, merr.WrapErrImportFailedMsg(
				"invalid manifest path for StorageV3 segment %d: %s", segment.GetSegmentId(), err,
			)
		}
		if version == packed.ManifestLatest {
			return nil, nil, nil, merr.WrapErrImportFailedMsg(
				"snapshot segment %d must reference an exact manifest version", segment.GetSegmentId(),
			)
		}
		if _, ok := seenManifests[manifestPath]; ok {
			return nil, nil, nil, merr.WrapErrImportFailedMsg(
				"snapshot source contains duplicate manifest for segment %d", segment.GetSegmentId(),
			)
		}
		seenManifests[manifestPath] = struct{}{}
		result = append(result, &internalpb.ImportFile{Paths: []string{manifestPath}})
	}
	applicable, err := snapshotImportL0Segments(snapshot.Segments, segments)
	if err != nil {
		return nil, nil, nil, err
	}
	validationSegments := append([]*datapb.SegmentDescription(nil), segments...)
	validationSegments = append(validationSegments, applicable...)
	// Validate metadata references without listing prefixes or opening physical
	// manifests. Batch the references so the metadata manifest list is scanned
	// only once, not once per segment. Workers validate discovered references.
	refs := make([]storage.SnapshotFileRef, 0, len(validationSegments))
	for _, segment := range validationSegments {
		if segment.GetManifestPath() == "" {
			continue
		}
		base, _, err := packed.UnmarshalManifestPath(segment.GetManifestPath())
		if err != nil {
			return nil, nil, nil, err
		}
		refs = append(refs, storage.SnapshotFileRef{Path: base, NormalizedPath: storage.NormalizeSnapshotObjectPath(base)})
	}
	if err := storage.ValidateExternalSnapshotPaths(metadataPath, snapshot, refs); err != nil {
		return nil, nil, nil, err
	}
	needsSourceContext := len(applicable) != 0
	for _, segment := range segments {
		// Compaction can move L0 deletes into a data segment's manifest. The
		// commit-time override still applies after the standalone L0 disappears;
		// never choose delete semantics based on where the deletes are stored.
		needsSourceContext = needsSourceContext || segment.GetCommitTimestamp() != 0
	}
	var l0Sources []*internalpb.SnapshotImportL0Source
	if needsSourceContext {
		l0Sources, err = attachSnapshotImportSources(ctx, cm, metadataPath, snapshot, segments, applicable, result)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	external := importutilv2.HasExternalSource(options)
	if external || partitionMapping != nil {
		// Select the complete contract once: external storage adds 1, target
		// routing adds 2, and shared L0 adds 4 to the base version. Older workers
		// must reject capabilities they cannot preserve, never ignore them.
		version := uint32(1)
		if external {
			version++
		}
		if partitionMapping != nil {
			version += 2
		}
		if len(l0Sources) != 0 {
			version += 4
		}
		for i, file := range result {
			if file.SnapshotSource == nil {
				file.SnapshotSource = &internalpb.SnapshotImportSource{ManifestPath: file.GetPaths()[0]}
			}
			// Preserve the commit timestamp and source scope attached above.
			file.SnapshotSource.Version = version
			if partitionMapping != nil {
				file.SnapshotSource.TargetPartitionId = partitionMapping[segments[i].GetPartitionId()]
			}
			file.Paths = nil
		}
	}
	return result, options, l0Sources, nil
}

func resolveSnapshotPartitionMapping(sourcePartitions map[string]int64, targetIDs []int64,
	schema *schemapb.CollectionSchema, options importutilv2.Options,
) (map[int64]int64, error) {
	mapping, err := importutilv2.GetPartitionMapping(options)
	if err != nil || mapping == nil {
		return nil, err
	}
	if typeutil.HasPartitionKey(schema) {
		return nil, merr.WrapErrImportFailedMsg("partition_mapping does not support a partition-key target collection")
	}
	if len(mapping) != len(sourcePartitions) {
		return nil, merr.WrapErrImportFailedMsg("partition_mapping must cover every source snapshot partition")
	}
	names := importutilv2.PartitionMappingTargets(mapping)
	if len(names) != len(targetIDs) {
		return nil, merr.WrapErrServiceInternalMsg("partition_mapping target IDs do not match Proxy's resolved destinations")
	}
	targets := make(map[string]int64, len(names))
	// Keep both bounds explicit for static analysis; the equality check above
	// still rejects mismatched metadata instead of accepting a partial mapping.
	for i := 0; i < len(names) && i < len(targetIDs); i++ {
		if targetIDs[i] <= 0 {
			return nil, merr.WrapErrServiceInternalMsg("partition_mapping contains an invalid resolved target ID")
		}
		targets[names[i]] = targetIDs[i]
	}
	result := make(map[int64]int64, len(mapping))
	for source, target := range mapping {
		id, ok := sourcePartitions[source]
		if !ok {
			return nil, merr.WrapErrImportFailedMsg("partition_mapping source partition %s does not exist in the snapshot", source)
		}
		if id <= 0 || result[id] != 0 {
			return nil, merr.WrapErrImportFailedMsg("snapshot contains invalid or duplicate source partition IDs")
		}
		result[id] = targets[target]
	}
	return result, nil
}

// Recheck Proxy's name-to-ID resolution while holding the collection broadcast
// lock. A drop/recreate during snapshot I/O must not redirect already-bound
// files to another partition with the same name.
func (s *Server) validateSnapshotPartitionTargets(ctx context.Context, collectionID int64,
	partitionIDs []int64, options importutilv2.Options,
) error {
	mapping, err := importutilv2.GetPartitionMapping(options)
	if err != nil || mapping == nil {
		return err
	}
	partitions, err := s.broker.ShowPartitions(ctx, collectionID)
	if err != nil {
		return err
	}
	if len(partitions.GetPartitionNames()) != len(partitions.GetPartitionIDs()) {
		return merr.WrapErrServiceInternalMsg("partition metadata has mismatched names and IDs")
	}
	current := make(map[string]int64)
	for i, name := range partitions.GetPartitionNames() {
		current[name] = partitions.GetPartitionIDs()[i]
	}
	names := importutilv2.PartitionMappingTargets(mapping)
	if len(names) != len(partitionIDs) {
		return merr.WrapErrServiceInternalMsg("partition_mapping destination count changed")
	}
	for i, name := range names {
		if current[name] != partitionIDs[i] {
			return merr.WrapErrServiceUnavailableMsg("target partition %s changed while preparing snapshot import; retry the request", name)
		}
	}
	return nil
}

// Match L0 against original source partitions/channels, including collection-wide
// deletes. Empty markers still activate the job.
func snapshotImportL0Segments(all, data []*datapb.SegmentDescription) ([]*datapb.SegmentDescription, error) {
	var result []*datapb.SegmentDescription
	for _, delta := range all {
		if delta.GetSegmentLevel() != datapb.SegmentLevel_L0 {
			continue
		}
		for _, segment := range data {
			if delta.GetPartitionId() != common.AllPartitionsID && delta.GetPartitionId() != segment.GetPartitionId() {
				continue
			}
			if delta.GetChannelName() == "" || segment.GetChannelName() == "" {
				return nil, merr.WrapErrImportFailedMsg("snapshot L0 matching requires source channel identity")
			}
			if delta.GetChannelName() != segment.GetChannelName() {
				continue
			}
			switch delta.GetStorageVersion() {
			case milvusstorage.StorageV1, milvusstorage.StorageV2:
			case milvusstorage.StorageV3:
				base, version, err := packed.UnmarshalManifestPath(delta.GetManifestPath())
				if err != nil || base == "" || version == packed.ManifestLatest {
					return nil, merr.WrapErrImportFailedMsg("snapshot L0 requires an exact manifest")
				}
			default:
				return nil, merr.WrapErrOperationNotSupportedMsg("unsupported snapshot L0 storage version %d", delta.GetStorageVersion())
			}
			result = append(result, delta)
			break
		}
	}
	return result, nil
}

// Attach timestamp context per segment, but retain L0 paths once per source
// scope. Channel-wide deletes stay separate in the durable job and are merged
// with partition-local deletes only when constructing a worker task.
// Enforce incremental size limits here; expandSnapshotJob validates the final
// plan with real options after external/routing transforms and ID allocation.
func attachSnapshotImportSources(ctx context.Context, cm milvusstorage.ChunkManager, metadataPath string,
	snapshot *storage.SnapshotData, data, deltas []*datapb.SegmentDescription, files []*internalpb.ImportFile,
) ([]*internalpb.SnapshotImportL0Source, error) {
	type scope struct {
		channel     string
		partitionID int64
	}
	groups := make(map[scope]*internalpb.SnapshotImportL0Source)
	seen := make(map[scope]map[string]bool)
	planSize := 0
	groupFor := func(channel string, partitionID int64) *internalpb.SnapshotImportL0Source {
		key := scope{channel, partitionID}
		if group := groups[key]; group != nil {
			return group
		}
		group := &internalpb.SnapshotImportL0Source{SourceChannel: channel, SourcePartitionId: partitionID}
		groups[key] = group
		seen[key] = make(map[string]bool)
		planSize += proto.Size(group) + 16
		return group
	}
	for i, segment := range data {
		source := &internalpb.SnapshotImportSource{
			Version: 1, ManifestPath: segment.GetManifestPath(), SourceCommitTimestamp: segment.GetCommitTimestamp(),
		}
		if len(deltas) != 0 {
			source.Version = 5
			source.SourceChannel = segment.GetChannelName()
			source.SourcePartitionId = segment.GetPartitionId()
			groupFor(source.SourceChannel, source.SourcePartitionId)
		}
		files[i].SnapshotSource = source
		files[i].Paths = nil // Old workers must not ignore timestamp/delete semantics.
		planSize += proto.Size(files[i]) + 16
		if planSize > importutilv2.SnapshotSourcePlanMaxBytes {
			return nil, merr.WrapErrImportFailedMsg("snapshot source plan exceeds 256 KiB")
		}
	}
	kinds := make(map[string]bool)
	// The caller has already checked the full metadata manifest list. Retain
	// the layout/root boundary for each L0 reference without rescanning it.
	pathBoundary := &storage.SnapshotData{Layout: snapshot.Layout}
	for _, delta := range deltas {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		group := groupFor(delta.GetChannelName(), delta.GetPartitionId())
		groupSeen := seen[scope{delta.GetChannelName(), delta.GetPartitionId()}]
		packedDelta := delta.GetStorageVersion() == milvusstorage.StorageV3
		var paths []string
		if packedDelta {
			base, version, err := packed.UnmarshalManifestPath(delta.GetManifestPath())
			if err != nil || base == "" || version == packed.ManifestLatest {
				return nil, merr.WrapErrDataIntegrityMsg("snapshot L0 requires an exact manifest version")
			}
			paths = []string{delta.GetManifestPath()}
		} else {
			for _, field := range delta.GetDeltalogs() {
				for _, log := range field.GetBinlogs() {
					// A legacy zero count does not imply an empty object.
					if strings.TrimSpace(log.GetLogPath()) == "" {
						return nil, merr.WrapErrDataIntegrityMsg("snapshot L0 deltalog has no object path")
					}
					paths = append(paths, log.GetLogPath())
				}
			}
		}
		for _, path := range paths {
			objectPath := path
			if packedDelta {
				objectPath, _, _ = packed.UnmarshalManifestPath(path)
			}
			if err := storage.ValidateSnapshotObjectPathForBucket(cm, "snapshot L0", objectPath, ""); err != nil {
				return nil, err
			}
			if err := storage.ValidateExternalSnapshotPaths(metadataPath, pathBoundary, []storage.SnapshotFileRef{{
				Path: objectPath, NormalizedPath: storage.NormalizeSnapshotObjectPath(objectPath), Type: storage.SnapshotFileTypeDeltaBinlog,
			}}); err != nil {
				return nil, err
			}
			// Validate the URI before normalizing it; both phases read the same
			// object key and must agree on its decoder even across source scopes.
			if packedDelta {
				_, version, _ := packed.UnmarshalManifestPath(path)
				path = packed.MarshalManifestPath(storage.NormalizeSnapshotObjectPath(objectPath), version)
			} else {
				path = storage.NormalizeSnapshotObjectPath(path)
			}
			if previous, ok := kinds[path]; ok && previous != packedDelta {
				return nil, merr.WrapErrDataIntegrityMsg("snapshot delete object has conflicting decoder contracts")
			}
			kinds[path] = packedDelta
			if _, ok := groupSeen[path]; ok {
				continue
			}
			planSize += len(path) + 16
			if planSize > importutilv2.SnapshotSourcePlanMaxBytes {
				return nil, merr.WrapErrImportFailedMsg("snapshot source plan exceeds 256 KiB")
			}
			groupSeen[path] = packedDelta
			if packedDelta {
				group.ManifestL0Paths = append(group.ManifestL0Paths, path)
			} else {
				group.LegacyL0Deltalogs = append(group.LegacyL0Deltalogs, path)
			}
		}
	}
	sources := make([]*internalpb.SnapshotImportL0Source, 0, len(groups))
	for _, group := range groups {
		sort.Strings(group.LegacyL0Deltalogs)
		sort.Strings(group.ManifestL0Paths)
		sources = append(sources, group)
	}
	sort.Slice(sources, func(i, j int) bool {
		if sources[i].SourceChannel != sources[j].SourceChannel {
			return sources[i].SourceChannel < sources[j].SourceChannel
		}
		return sources[i].SourcePartitionId < sources[j].SourcePartitionId
	})
	return sources, nil
}

// normalizeSnapshotImportEncryption uses source metadata, not the presence of
// an EZK or the target schema, to decide whether source decryption is needed.
func normalizeSnapshotImportEncryption(sourceSchema *schemapb.CollectionSchema, options importutilv2.Options) (importutilv2.Options, error) {
	var (
		sourceEzID int64
		encrypted  bool
	)
	for _, property := range sourceSchema.GetProperties() {
		if property.GetKey() != common.EncryptionEzIDKey {
			continue
		}
		parsed, err := strconv.ParseInt(property.GetValue(), 10, 64)
		if err != nil {
			return nil, merr.WrapErrImportFailedMsg(
				"snapshot source has an invalid %s property", common.EncryptionEzIDKey,
			)
		}
		sourceEzID = parsed
		encrypted = true
		break
	}

	if !encrypted {
		// A plaintext snapshot needs no source key. Remove even a malformed EZK
		// before WAL/job persistence so both phases and retries select plaintext
		// reading, including TEXT/LOB, without retaining an unused credential.
		// Allocate a new slice: the request options may be shared by callers.
		normalized := make(importutilv2.Options, 0, len(options))
		for _, option := range options {
			if option.GetKey() != importutilv2.EZK {
				normalized = append(normalized, option)
			}
		}
		return normalized, nil
	}
	importEzk, _ := importutilv2.GetEZK(options)
	if importEzk == "" {
		return nil, merr.WrapErrImportFailedMsg(
			"CMEK-protected snapshot-source import requires ezk",
		)
	}

	importEzID, err := hookutil.GetEzIDByImportEzk(importEzk)
	if err != nil {
		// EZK is request content, so malformed base64/JSON is an input error. Do
		// not leak the opaque key or parser details into logs or client errors.
		return nil, merr.WrapErrImportFailedMsg("snapshot-source import received an invalid ezk")
	}
	if importEzID != sourceEzID {
		return nil, merr.WrapErrImportFailedMsg(
			"snapshot-source ezk belongs to encryption zone %d, source requires zone %d",
			importEzID,
			sourceEzID,
		)
	}

	// StorageV3 stores TEXT values as physical LOB references. Resolving those
	// references requires milvus-storage SegmentReader, whose current C API
	// cannot receive Milvus's source key-retriever context. Non-TEXT schemas use
	// PackedReader, which already accepts that context, so reject only this
	// unsupported combination instead of rejecting all CMEK snapshot imports.
	if typeutil.HasTextField(sourceSchema) {
		return nil, merr.WrapErrOperationNotSupportedMsg(
			"CMEK-protected snapshot-source import does not support TEXT/LOB fields",
		)
	}
	return options, nil
}

// prepareSnapshotImportOptions applies the same schema admission and source-key
// normalization at RPC admission and preparation recovery. Both boundaries must
// validate their input; neither requires source/target schema equality.
func prepareSnapshotImportOptions(target, source *schemapb.CollectionSchema, options importutilv2.Options) (importutilv2.Options, error) {
	if source == nil {
		return nil, merr.WrapErrImportFailedMsg("snapshot source schema is missing")
	}
	if typeutil.IsExternalCollection(source) || source.GetExternalSource() != "" || source.GetExternalSpec() != "" {
		return nil, merr.WrapErrOperationNotSupportedMsg("external collection snapshots cannot be used as import sources")
	}
	options, err := normalizeSnapshotImportEncryption(source, options)
	if err != nil {
		return nil, err
	}
	if target == nil {
		return nil, merr.WrapErrImportSysFailed("target collection schema is missing")
	}
	if typeutil.IsExternalCollection(target) ||
		target.GetExternalSource() != "" ||
		target.GetExternalSpec() != "" {
		return nil, merr.WrapErrOperationNotSupportedMsg(
			"snapshot-source import does not support an external target collection",
		)
	}
	// Match ordinary backup import: the target schema interprets physical field
	// IDs, and the reader validates required columns against each source segment.
	// Snapshot metadata is not an extra schema-equality or name-mapping contract.
	// Renaming a field does not remap its data; callers must retain the intended
	// field-ID correspondence, just as when importing binlog backups. Optional
	// target columns, AutoID and target routing use the existing Import behavior.
	return options, nil
}
