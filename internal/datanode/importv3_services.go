// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. See the NOTICE file distributed with this work for
// additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

package datanode

import (
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"sort"
	"strconv"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/datanode/importv2"
	"github.com/milvus-io/milvus/internal/datanode/importv3"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/bloomfilter"
	"github.com/milvus-io/milvus/internal/util/function/embedding"
	"github.com/milvus-io/milvus/internal/util/hookutil"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/internal/util/importutilv2/binlog"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexcgopb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v3/taskcommon"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
	"google.golang.org/protobuf/proto"
)

func (node *DataNode) createImportV3WorkerTask(
	_ context.Context,
	taskID, runID, slot int64,
	execute importv3.Run,
) (*commonpb.Status, error) {
	if node.importV3TaskMgr == nil {
		return merr.Status(merr.WrapErrServiceNotReadyMsg("import V3 task manager is not initialized")), nil
	}
	if err := node.importV3TaskMgr.Add(taskID, runID, slot, execute); err != nil {
		return merr.Status(err), nil
	}
	return merr.Success(), nil
}

func (node *DataNode) queryImportV3WorkerTask(
	ctx context.Context,
	taskID, runID int64,
	taskType taskcommon.Type,
) (*workerpb.QueryTaskResponse, error) {
	if node.importV3TaskMgr == nil {
		return &workerpb.QueryTaskResponse{Status: merr.Status(merr.WrapErrServiceNotReadyMsg("import V3 task manager is not initialized"))}, nil
	}
	snapshot, ok := node.importV3TaskMgr.Query(taskID, runID)
	if !ok {
		return &workerpb.QueryTaskResponse{Status: merr.Status(merr.WrapErrNodeNotFound(node.GetNodeID(),
			"cannot find current import V3 task run"))}, nil
	}
	properties := taskcommon.NewProperties(nil)
	properties.AppendTaskState(importV3TaskCommonState(snapshot.State))
	properties.AppendReason(snapshot.Reason)

	var payload any
	switch taskType {
	case taskcommon.Reshard:
		payload = &datapb.QueryReshardTaskResponse{
			Status: merr.Success(), State: reshardTaskState(snapshot.State), Reason: snapshot.Reason,
		}
	case taskcommon.ImportV3:
		response := &datapb.QueryImportTaskV3Response{
			Status: merr.Success(), State: importTaskV3State(snapshot.State), Reason: snapshot.Reason,
			Segments: snapshot.Segments,
		}
		if snapshot.State == importv3.StateCompleted {
			resultBytes := proto.Size(response)
			grpcLimitBytes := min(
				paramtable.Get().DataNodeGrpcServerCfg.ServerMaxSendSize.GetAsInt(),
				paramtable.Get().DataNodeGrpcClientCfg.ClientMaxRecvSize.GetAsInt(),
			)
			if resultBytes >= grpcLimitBytes/2 {
				mlog.Warn(ctx, "import V3 result is close to the gRPC result limit",
					mlog.Int64("taskID", taskID),
					mlog.Int64("runID", runID),
					mlog.Int("segmentCount", len(snapshot.Segments)),
					mlog.Int("resultBytes", resultBytes),
					mlog.Int("grpcResultLimitBytes", grpcLimitBytes))
			}
		}
		payload = response
	default:
		return &workerpb.QueryTaskResponse{Status: merr.Status(merr.WrapErrServiceInternalMsg(
			"invalid V3 task type %q", taskType))}, nil
	}
	// Keep the concrete proto types at the boundary so wrapQueryTaskResult can
	// enforce the existing GetStatus payload contract.
	switch result := payload.(type) {
	case *datapb.QueryReshardTaskResponse:
		return wrapQueryTaskResult(result, properties)
	case *datapb.QueryImportTaskV3Response:
		return wrapQueryTaskResult(result, properties)
	default:
		panic("unreachable import V3 query payload")
	}
}

func (node *DataNode) dropImportV3WorkerTask(taskID, runID int64) (*commonpb.Status, error) {
	if node.importV3TaskMgr == nil {
		return merr.Success(), nil
	}
	// Best effort and idempotent.  A stale run must not cancel a newer run;
	// TaskManager.Drop returns false for both stale and already-absent tasks.
	node.importV3TaskMgr.Drop(taskID, runID)
	return merr.Success(), nil
}

func (node *DataNode) executeReshardTask(ctx context.Context, req *datapb.ReshardTaskRequest, runID int64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if req == nil || req.GetRunId() != runID || req.GetStorageConfig() == nil || req.GetSlot() <= 0 {
		return merr.WrapErrImportSysFailedMsg("invalid or incomplete ReshardTask request")
	}
	plan, err := node.readReshardTaskPlan(ctx, req.GetStorageConfig(), req.GetJobId(), req.GetTaskId())
	if err != nil {
		return err
	}
	cm, err := node.storageFactory.NewChunkManager(ctx, req.GetStorageConfig())
	if err != nil {
		return err
	}
	pluginContext, err := hookutil.GetCPluginContext(req.GetPluginContext(), plan.GetCollectionId())
	if err != nil {
		return err
	}
	return executeReshardPlan(ctx, cm, req, plan, pluginContext)
}

type reshardBucket struct {
	vchannelOrdinal  int
	partitionOrdinal int
	batches          []*storage.InsertData
	spillChunks      []string
	bytes            int64
	logicalBytes     int64
	rows             int64
	nextSpillSeq     int
}

func executeReshardPlan(ctx context.Context, cm storage.ChunkManager, req *datapb.ReshardTaskRequest, plan *datapb.ReshardTaskPlan, pluginContext *indexcgopb.StoragePluginContext) error {
	temporarySchema := plan.GetTempSchema()
	if plan.GetSchema() == nil || temporarySchema == nil ||
		len(plan.GetVchannels()) == 0 || len(plan.GetPartitions()) == 0 || len(plan.GetSources()) == 0 {
		return merr.WrapErrDataIntegrityMsg("invalid ReshardTask plan")
	}
	sortFields, err := importv3.SortFields(plan.GetSort(), temporarySchema)
	if err != nil {
		return err
	}
	bufferSize := paramtable.Get().DataNodeCfg.ImportBaseBufferSize.GetAsInt64()
	maxFileSize := int64(paramtable.Get().DataNodeCfg.MaxImportFileSizeInGB.GetAsFloat() * 1024 * 1024 * 1024)
	fragmentTarget := plan.GetFragmentSize()
	if fragmentTarget <= 0 {
		fragmentTarget = 128 * 1024 * 1024
	}
	slot := req.GetSlot()
	if slot <= 0 {
		slot = 1
	}
	memoryBudget := slot * paramtable.Get().DataCoordCfg.ImportMemoryLimitPerSlot.GetAsInt64()
	spillRoot := path.Join(paramtable.Get().LocalStorageCfg.Path.GetValue(), "import_v3",
		strconv.FormatInt(req.GetJobId(), 10), strconv.FormatInt(req.GetTaskId(), 10))
	if err := os.MkdirAll(spillRoot, 0o755); err != nil {
		return merr.Wrapf(err, "create reshard spill directory")
	}
	defer os.RemoveAll(spillRoot)

	manifest := &datapb.ReshardManifest{}
	buckets := make(map[reshardBucketKey]*reshardBucket)
	var residentBytes int64
	var fragmentSeq int64

	flushBucket := func(b *reshardBucket) error {
		descriptor, err := writeReshardFragment(ctx, req, plan, temporarySchema, b, fragmentSeq, bufferSize, sortFields, pluginContext)
		if err != nil {
			return err
		}
		fragmentSeq++
		manifest.Fragments = append(manifest.Fragments, descriptor)
		for _, chunk := range b.spillChunks {
			_ = os.Remove(chunk)
		}
		*b = reshardBucket{vchannelOrdinal: b.vchannelOrdinal, partitionOrdinal: b.partitionOrdinal}
		return nil
	}

	spillLargest := func() (int64, error) {
		var target *reshardBucket
		for _, b := range buckets {
			if b.bytes == 0 {
				continue
			}
			if target == nil || b.bytes > target.bytes ||
				(b.bytes == target.bytes && (b.vchannelOrdinal < target.vchannelOrdinal ||
					(b.vchannelOrdinal == target.vchannelOrdinal && b.partitionOrdinal < target.partitionOrdinal))) {
				target = b
			}
		}
		if target == nil {
			return 0, nil
		}
		spillPath := path.Join(spillRoot, fmt.Sprintf("%d_%d_%d_%d.arrow", req.GetRunId(),
			target.vchannelOrdinal, plan.GetPartitions()[target.partitionOrdinal], target.nextSpillSeq))
		writer, err := importv3.NewSpillWriter(spillPath, temporarySchema)
		if err != nil {
			return 0, err
		}
		for _, batch := range target.batches {
			if err := writer.Write(batch); err != nil {
				_ = writer.Close()
				return 0, err
			}
		}
		if err := writer.Close(); err != nil {
			return 0, err
		}
		target.spillChunks = append(target.spillChunks, spillPath)
		target.batches = nil
		target.nextSpillSeq++
		spilled := target.bytes
		target.bytes = 0
		return spilled, nil
	}

	for _, source := range plan.GetSources() {
		if err := ctx.Err(); err != nil {
			return err
		}
		if source == nil || source.GetFile() == nil {
			return merr.WrapErrDataIntegrityMsg("nil ReshardTask source")
		}
		reader, err := newReshardSourceReader(ctx, cm, plan.GetSchema(), source, bufferSize, req.GetStorageConfig(), pluginContext)
		if err != nil {
			return err
		}
		size, err := reader.Size()
		if err != nil {
			reader.Close()
			return merr.Wrapf(err, "get import source %d size", source.GetOrdinal())
		}
		if size > maxFileSize {
			reader.Close()
			return merr.WrapErrParameterInvalidMsg("import file size (%d bytes) exceeds the maximum allowed size (%d bytes)", size, maxFileSize)
		}

		var idOffset int64
		for {
			batch, readErr := reader.Read()
			if readErr == io.EOF {
				break
			}
			if readErr != nil {
				reader.Close()
				return merr.Wrapf(readErr, "read import source %d", source.GetOrdinal())
			}
			rowNum, _ := importv2.GetInsertDataRowCount(batch, plan.GetSchema())
			if err := normalizeReshardBatch(source, plan.GetSchema(), batch, rowNum, &idOffset); err != nil {
				reader.Close()
				return err
			}
			if rowNum == 0 {
				continue
			}
			hashed, err := importv2.HashDataBySchema(temporarySchema, plan.GetVchannels(), plan.GetPartitions(), batch)
			if err != nil {
				reader.Close()
				return err
			}
			for channelOrdinal := range hashed {
				for partitionOrdinal, bucketData := range hashed[channelOrdinal] {
					if bucketData.GetRowNum() == 0 {
						continue
					}
					key := reshardBucketKey{channelOrdinal: channelOrdinal, partitionOrdinal: partitionOrdinal}
					b := buckets[key]
					if b == nil {
						b = &reshardBucket{vchannelOrdinal: channelOrdinal, partitionOrdinal: partitionOrdinal}
						buckets[key] = b
					}
					appendReshardBatch(b, bucketData)
					residentBytes += int64(bucketData.GetMemorySize())
					if b.logicalBytes >= fragmentTarget {
						freed := b.bytes
						if err := flushBucket(b); err != nil {
							reader.Close()
							return err
						}
						residentBytes -= freed
					} else if residentBytes > memoryBudget {
						spilled, err := spillLargest()
						if err != nil {
							reader.Close()
							return err
						}
						residentBytes -= spilled
					}
				}
			}
		}
		reader.Close()
	}

	keys := make([]reshardBucketKey, 0, len(buckets))
	for key := range buckets {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].channelOrdinal != keys[j].channelOrdinal {
			return keys[i].channelOrdinal < keys[j].channelOrdinal
		}
		return keys[i].partitionOrdinal < keys[j].partitionOrdinal
	})
	for _, key := range keys {
		b := buckets[key]
		if b.rows == 0 && len(b.spillChunks) == 0 {
			continue
		}
		if err := flushBucket(b); err != nil {
			return err
		}
	}
	return publishReshardManifest(ctx, cm, req, manifest)
}

type reshardBucketKey struct {
	channelOrdinal   int
	partitionOrdinal int
}

func appendReshardBatch(b *reshardBucket, data *storage.InsertData) {
	mem := int64(data.GetMemorySize())
	b.batches = append(b.batches, data)
	b.bytes += mem
	b.logicalBytes += mem
	b.rows += int64(data.GetRowNum())
}

func newReshardSourceReader(ctx context.Context, cm storage.ChunkManager, schema *schemapb.CollectionSchema, source *datapb.SourceFileSpec, bufferSize int64, storageConfig *indexpb.StorageConfig, pluginContext *indexcgopb.StoragePluginContext) (importutilv2.Reader, error) {
	if source == nil || source.GetFile() == nil {
		return nil, merr.WrapErrDataIntegrityMsg("nil ReshardTask source")
	}
	options := make(importutilv2.Options, 0, 6)
	appendOption := func(key, value string) {
		options = append(options, &commonpb.KeyValuePair{Key: key, Value: value})
	}
	readerOptions := source.GetOptions()
	if source.GetFormat() == datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_CSV {
		if readerOptions.GetSeparator() != "" {
			appendOption(importutilv2.CSVSep, readerOptions.GetSeparator())
		}
		if readerOptions.GetNullKey() != "" {
			appendOption(importutilv2.CSVNullKey, readerOptions.GetNullKey())
		}
	}
	if source.GetBackup() {
		insertLogs := make(map[int64][]string, len(source.GetInsertFields()))
		for _, field := range source.GetInsertFields() {
			if field == nil || len(field.GetPaths()) == 0 {
				return nil, merr.WrapErrDataIntegrityMsg("backup source has an empty expanded insert field")
			}
			if _, ok := insertLogs[field.GetId()]; ok {
				return nil, merr.WrapErrDataIntegrityMsg("backup source has duplicate expanded field/group %d", field.GetId())
			}
			insertLogs[field.GetId()] = append([]string(nil), field.GetPaths()...)
		}
		return binlog.NewExplicitReader(ctx, cm, schema, storageConfig, readerOptions.GetStorageVersion(), insertLogs,
			append([]string(nil), source.GetDeltaPaths()...), readerOptions.GetStartTs(), readerOptions.GetEndTs(), int(bufferSize), pluginContext)
	}
	return importutilv2.NewReader(ctx, cm, schema, source.GetFile(), options, int(bufferSize), storageConfig)
}

func normalizeReshardBatch(source *datapb.SourceFileSpec, schema *schemapb.CollectionSchema, data *storage.InsertData, rowNum int, idOffset *int64) error {
	if data == nil {
		return merr.WrapErrDataIntegrityMsg("source reader returned nil data")
	}
	if err := importv2.CheckRowsEqual(schema, data); err != nil {
		return err
	}
	if err := importv2.CheckStructArrayConsistency(schema, data); err != nil {
		return err
	}
	if err := importv2.AppendNullableDefaultFieldsData(schema, data, rowNum); err != nil {
		return err
	}
	if err := importv2.FillDynamicData(schema, data, rowNum); err != nil {
		return err
	}
	if source.GetBackup() {
		return nil
	}
	return importv2.AppendPreallocatedSystemFields(schema, data, rowNum, source.GetFile().GetPreAllocatedAutoIds(), idOffset)
}

func writeReshardFragment(ctx context.Context, req *datapb.ReshardTaskRequest, plan *datapb.ReshardTaskPlan, temporarySchema *schemapb.CollectionSchema, bucket *reshardBucket, seq, bufferSize int64, sortFields []int64, pluginContext *indexcgopb.StoragePluginContext) (*datapb.FragmentDescriptor, error) {
	fragmentPath := path.Join(metautil.BuildImportV3ReshardOutputPath(req.GetJobId(), req.GetTaskId()), "fragments", strconv.Itoa(bucket.vchannelOrdinal), strconv.FormatInt(plan.GetPartitions()[bucket.partitionOrdinal], 10), fmt.Sprintf("%d_%d.parquet", req.GetRunId(), seq))
	fields := typeutil.GetAllFieldSchemas(temporarySchema)
	columns := make([]int, len(fields))
	fieldIDs := make([]int64, len(fields))
	for index, field := range fields {
		columns[index], fieldIDs[index] = index, field.GetFieldID()
	}
	writer, err := storage.NewPackedRecordWriter(req.GetStorageConfig().GetBucketName(), []string{fragmentPath}, temporarySchema, bufferSize, packed.DefaultMultiPartUploadSize, []storagecommon.ColumnGroup{{GroupID: 0, Columns: columns, Fields: fieldIDs}}, req.GetStorageConfig(), pluginContext)
	if err != nil {
		return nil, err
	}

	readers := make([]storage.RecordReader, 0, len(bucket.spillChunks)+len(bucket.batches))
	for _, chunk := range bucket.spillChunks {
		reader, err := importv3.NewSpillReader(chunk, temporarySchema)
		if err != nil {
			_ = writer.Close()
			return nil, err
		}
		readers = append(readers, reader)
	}
	for _, batch := range bucket.batches {
		reader, err := storage.NewInsertDataRecordReader(batch, temporarySchema)
		if err != nil {
			_ = writer.Close()
			closeReshardReaders(readers)
			return nil, err
		}
		readers = append(readers, reader)
	}
	defer closeReshardReaders(readers)

	rows, _, err := storage.Sort(uint64(bufferSize), temporarySchema, readers, writer, func(storage.Record, int, int) bool { return true }, sortFields)
	if err != nil {
		_ = writer.Close()
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	if int64(rows) != bucket.rows || writer.GetWrittenRowNum() != int64(rows) {
		return nil, merr.WrapErrDataIntegrityMsg("fragment row count mismatch: input=%d sorted=%d written=%d", bucket.rows, rows, writer.GetWrittenRowNum())
	}
	return &datapb.FragmentDescriptor{
		ChannelIndex: int32(bucket.vchannelOrdinal), PartitionIndex: int32(bucket.partitionOrdinal), Seq: seq,
		Path: writer.GetWrittenPaths(0), Rows: int64(rows), LogicalBytes: int64(writer.GetWrittenUncompressed()),
	}, nil
}

func closeReshardReaders(readers []storage.RecordReader) {
	for _, reader := range readers {
		if reader != nil {
			_ = reader.Close()
		}
	}
}

func publishReshardManifest(ctx context.Context, cm storage.ChunkManager, req *datapb.ReshardTaskRequest, manifest *datapb.ReshardManifest) error {
	payload, err := proto.Marshal(manifest)
	if err != nil {
		return merr.WrapErrSerializationFailed(err, "marshal ReshardManifest")
	}
	resultPath := metautil.BuildImportV3ReshardResultPath(req.GetJobId(), req.GetTaskId(), req.GetRunId())
	if err := cm.Write(ctx, resultPath, payload); err != nil {
		return merr.Wrap(err, "write ReshardManifest")
	}
	return nil
}

func (node *DataNode) executeImportTaskV3(ctx context.Context, req *datapb.ImportTaskV3Request, runID int64) ([]*datapb.SegmentResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if req == nil || req.GetRunId() != runID || req.GetStorageConfig() == nil || req.GetSlot() <= 0 {
		return nil, merr.WrapErrImportSysFailedMsg("invalid or incomplete ImportTaskV3 request")
	}
	plan, err := node.readImportTaskPlan(ctx, req.GetStorageConfig(), req.GetJobId(), req.GetTaskId())
	if err != nil {
		return nil, err
	}
	if plan.GetFanIn() < 2 || plan.GetFanIn() > 1024 {
		return nil, merr.WrapErrImportSysFailedMsg("ImportTaskV3 plan merge fan-in is invalid: %d", plan.GetFanIn())
	}
	if plan.GetCollectionId() == 0 || plan.GetSchema() == nil || plan.GetTempSchema() == nil || plan.GetWriter() == nil || len(req.GetSegments()) == 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 plan/output segments are incomplete")
	}
	seenSegments := make(map[int64]struct{}, len(req.GetSegments()))
	for _, segmentID := range req.GetSegments() {
		if segmentID <= 0 {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 output segment ID is invalid: %d", segmentID)
		}
		if _, ok := seenSegments[segmentID]; ok {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 output segment ID is duplicated: %d", segmentID)
		}
		seenSegments[segmentID] = struct{}{}
	}
	pluginContext, err := hookutil.GetCPluginContext(req.GetPluginContext(), plan.GetCollectionId())
	if err != nil {
		return nil, err
	}
	cm, err := node.storageFactory.NewChunkManager(ctx, req.GetStorageConfig())
	if err != nil {
		return nil, err
	}
	return executeImportPlan(ctx, cm, req, plan, pluginContext)
}

func executeImportPlan(ctx context.Context, cm storage.ChunkManager, req *datapb.ImportTaskV3Request, plan *datapb.ImportTaskPlan, pluginContext *indexcgopb.StoragePluginContext) ([]*datapb.SegmentResult, error) {
	temporarySchema := plan.GetTempSchema()
	targetSchema := plan.GetSchema()
	if err := validateImportV3Schemas(temporarySchema, targetSchema); err != nil {
		return nil, err
	}
	sortFields, err := importv3.SortFields(plan.GetSort(), temporarySchema)
	if err != nil {
		return nil, err
	}
	if len(plan.GetSegments()) == 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 plan has no segment plans")
	}
	if plan.GetCollectionId() == 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 plan collection ID is empty")
	}
	if len(req.GetSegments()) != len(plan.GetSegments()) {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 output segment IDs/segment plans mismatch: ids=%d plans=%d", len(req.GetSegments()), len(plan.GetSegments()))
	}
	if req.GetLogRange() == nil || req.GetLogRange().GetEnd() <= req.GetLogRange().GetBegin() {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 log ID range is empty")
	}
	logAllocator := allocator.NewLocalAllocator(req.GetLogRange().GetBegin(), req.GetLogRange().GetEnd())
	results := make([]*datapb.SegmentResult, 0, len(plan.GetSegments()))
	writerSpec := plan.GetWriter()
	for segmentIndex, segment := range plan.GetSegments() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if segment == nil {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 segment plan is nil")
		}
		segmentID := req.GetSegments()[segmentIndex]
		if segmentID <= 0 {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 physical segment ID is invalid: %d", segmentID)
		}
		writerOptions, err := buildImportV3WriterOptions(req.GetStorageConfig(), plan.GetCollectionId(), segment, targetSchema, writerSpec, pluginContext)
		if err != nil {
			return nil, err
		}
		sources := make([]importv3.Source, 0, len(segment.GetFragments()))
		for _, ref := range segment.GetFragments() {
			source, err := importv3.SourceFromFragment(ref, temporarySchema, paramtable.Get().DataNodeCfg.ImportBaseBufferSize.GetAsInt64(), req.GetStorageConfig(), pluginContext)
			if err != nil {
				return nil, err
			}
			sources = append(sources, source)
		}
		var writer storage.BinlogRecordWriter
		finalWriter := func(_ context.Context) (storage.RecordWriter, error) {
			if writer != nil {
				return writer, nil
			}
			writer, err = storage.NewBinlogRecordWriter(ctx, plan.GetCollectionId(), segment.GetPartitionId(), segmentID, targetSchema, logAllocator, uint64(paramtable.Get().DataNodeCfg.BinLogMaxSize.GetAsInt64()), segment.GetRows(), writerOptions...)
			return writer, err
		}
		predicate := importv3.NewTTLOnlyPredicate(temporarySchema, writerSpec.GetTtlNanos(), plan.GetDataTs())
		executor := &importv3.MergeExecutor{
			FanIn: int(plan.GetFanIn()), BatchSize: uint64(paramtable.Get().DataNodeCfg.ImportBaseBufferSize.GetAsInt64()),
			Schema: temporarySchema, SortFields: sortFields, Predicate: predicate,
			FinalWriter: func(output storage.RecordWriter) storage.RecordWriter {
				return newImportV3FinalWriter(ctx, output, temporarySchema, targetSchema, plan.GetDataTs(), plan.GetClusterId())
			},
			Intermediate: newImportV3IntermediateFactory(req, segmentIndex, temporarySchema, pluginContext),
		}
		rows, err := executor.Execute(ctx, sources, finalWriter)
		if err != nil {
			return nil, err
		}
		segmentResult := &datapb.SegmentResult{Rows: rows}
		if writer != nil {
			fieldBinlogs, statsLog, bm25Logs, manifestPath, expiration := writer.GetLogs()
			segmentResult.InsertLogs = storage.SortFieldBinlogs(fieldBinlogs)
			segmentResult.PkLog = statsLog
			segmentResult.Bm25Logs = storage.SortFieldBinlogs(bm25Logs)
			segmentResult.ManifestPath = manifestPath
			segmentResult.ExpirationQuantiles = expiration
			segmentResult.Statistics = storage.BuildStatsFromFieldBinlogs(segmentResult.InsertLogs, nil, segmentResult.Bm25Logs, nil)
			segmentResult.Statistics.StatsBinlogSize = writer.GetStatsBlobSize()
			if writer.GetRowNum() != rows {
				return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 writer/result row mismatch: writer=%d result=%d", writer.GetRowNum(), rows)
			}
		} else if rows != 0 {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 kept rows without materializing a writer: %d", rows)
		}
		results = append(results, segmentResult)
	}
	if len(results) != len(plan.GetSegments()) {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 result segment count mismatch")
	}
	return results, nil
}

func validateImportV3Schemas(temporary, target *schemapb.CollectionSchema) error {
	if temporary == nil || target == nil {
		return merr.WrapErrDataIntegrityMsg("ImportTaskV3 plan schema is nil")
	}
	temporaryFields := make(map[int64]*schemapb.FieldSchema)
	for _, field := range typeutil.GetAllFieldSchemas(temporary) {
		temporaryFields[field.GetFieldID()] = field
	}
	targetFields := make(map[int64]*schemapb.FieldSchema)
	for _, field := range typeutil.GetAllFieldSchemas(target) {
		targetFields[field.GetFieldID()] = field
	}
	for _, systemField := range []int64{common.RowIDField, common.TimeStampField} {
		field := targetFields[systemField]
		if field == nil || field.GetDataType() != schemapb.DataType_Int64 {
			return merr.WrapErrDataIntegrityMsg("ImportTaskV3 target schema is missing int64 system field %d", systemField)
		}
	}
	if field := temporaryFields[common.RowIDField]; field == nil || field.GetDataType() != schemapb.DataType_Int64 {
		return merr.WrapErrDataIntegrityMsg("ImportTaskV3 temporary schema is missing RowID")
	}
	for fieldID, sourceField := range temporaryFields {
		targetField := targetFields[fieldID]
		if targetField == nil || targetField.GetDataType() != sourceField.GetDataType() {
			return merr.WrapErrDataIntegrityMsg("ImportTaskV3 temporary field %d is incompatible with target schema", fieldID)
		}
	}
	return nil
}

func buildImportV3WriterOptions(storageConfig *indexpb.StorageConfig, collectionID int64, segment *datapb.SegmentPlan, targetSchema *schemapb.CollectionSchema, spec *datapb.WriterSpec, pluginContext *indexcgopb.StoragePluginContext) ([]storage.RwOption, error) {
	if spec == nil {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec is nil")
	}
	if spec.GetStorageVersion() != storage.StorageV2 && spec.GetStorageVersion() != storage.StorageV3 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec storage version is unsupported: %d", spec.GetStorageVersion())
	}
	if spec.GetSchemaVersion() != int64(targetSchema.GetVersion()) {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec schema version mismatch: spec=%d plan=%d", spec.GetSchemaVersion(), targetSchema.GetVersion())
	}
	if spec.GetPkCapacity() <= 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec PK stats capacity must be positive")
	}
	if segment.GetRows() <= 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 planned rows must be positive")
	}
	if spec.GetPkCapacity() < segment.GetRows() {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec PK stats capacity is smaller than planned rows: capacity=%d rows=%d", spec.GetPkCapacity(), segment.GetRows())
	}
	bfType := bloomfilter.BFTypeFromString(spec.GetBloomType())
	if (bfType != bloomfilter.BasicBF && bfType != bloomfilter.BlockedBF) || spec.GetBloomFpp() <= 0 || spec.GetBloomFpp() >= 1 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec Bloom filter config is invalid")
	}
	if spec.GetFormat() == "" {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec writer format is empty")
	}
	groups, err := importV3Groups(targetSchema, spec.GetGroups())
	if err != nil {
		return nil, err
	}
	bufferSize := int64(packed.DefaultWriteBufferSize)
	multipartSize := int64(packed.DefaultMultiPartUploadSize)
	if spec.GetV2() != nil {
		if spec.GetV2().GetBufferSize() <= 0 || spec.GetV2().GetMultipartSize() <= 0 {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec V2 IO sizes must be positive")
		}
		bufferSize = spec.GetV2().GetBufferSize()
		multipartSize = spec.GetV2().GetMultipartSize()
	}
	options := []storage.RwOption{
		storage.WithVersion(spec.GetStorageVersion()),
		storage.WithBufferSize(bufferSize),
		storage.WithMultiPartUploadSize(multipartSize),
		storage.WithColumnGroups(groups),
		storage.WithStorageConfig(storageConfig),
		storage.WithPluginContext(pluginContext),
		storage.WithWriterFormat(spec.GetFormat()),
		storage.WithPkStatsConfig(storage.PkStatsConfig{
			Capacity: spec.GetPkCapacity(), BloomFilterType: spec.GetBloomType(), MaxBloomFalsePositive: spec.GetBloomFpp(),
		}),
	}
	if len(spec.GetText()) > 0 {
		if spec.GetStorageVersion() != storage.StorageV3 {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 TEXT columns require StorageV3")
		}
		partitionBase := path.Join(storageConfig.GetRootPath(), common.SegmentInsertLogPath, strconv.FormatInt(collectionID, 10), strconv.FormatInt(segment.GetPartitionId(), 10))
		textConfigs := make([]packed.TextColumnConfig, 0, len(spec.GetText()))
		for _, text := range spec.GetText() {
			if text == nil || text.GetFieldId() < common.StartOfUserFieldID || text.GetInlineLimit() < 0 || text.GetLobLimit() <= 0 || text.GetFlushLimit() <= 0 {
				return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec TEXT config is invalid")
			}
			textConfigs = append(textConfigs, packed.TextColumnConfig{
				FieldID: text.GetFieldId(), LobBasePath: path.Join(partitionBase, "lobs", strconv.FormatInt(text.GetFieldId(), 10)),
				InlineThreshold: text.GetInlineLimit(), MaxLobFileBytes: text.GetLobLimit(), FlushThresholdBytes: text.GetFlushLimit(),
			})
		}
		options = append(options, storage.WithTextColumnConfigs(textConfigs))
	}
	if spec.GetStorageVersion() == storage.StorageV3 && len(groups) == 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 StorageV3 WriterSpec has no column groups")
	}
	return options, nil
}

func importV3Groups(schema *schemapb.CollectionSchema, specs []*datapb.ColumnGroupSpec) ([]storagecommon.ColumnGroup, error) {
	fields := typeutil.GetAllFieldSchemas(schema)
	fieldColumns := make(map[int64]int, len(fields))
	for column, field := range fields {
		fieldColumns[field.GetFieldID()] = column
	}
	if len(specs) == 0 {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec column groups are empty")
	}
	groups := make([]storagecommon.ColumnGroup, 0, len(specs))
	covered := make(map[int64]struct{}, len(fields))
	for _, spec := range specs {
		if spec == nil || len(spec.GetFields()) == 0 || spec.GetFormat() == "" {
			return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec column group is incomplete")
		}
		group := storagecommon.ColumnGroup{GroupID: spec.GetId(), Format: spec.GetFormat()}
		for _, fieldID := range spec.GetFields() {
			column, ok := fieldColumns[fieldID]
			if !ok {
				return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec column group references unknown field %d", fieldID)
			}
			if _, ok := covered[fieldID]; ok {
				return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec column groups overlap at field %d", fieldID)
			}
			covered[fieldID] = struct{}{}
			group.Fields = append(group.Fields, fieldID)
			group.Columns = append(group.Columns, column)
		}
		groups = append(groups, group)
	}
	if len(covered) != len(fields) {
		return nil, merr.WrapErrDataIntegrityMsg("ImportTaskV3 WriterSpec column groups do not cover target schema: covered=%d fields=%d", len(covered), len(fields))
	}
	return groups, nil
}

type importV3FinalWriter struct {
	ctx             context.Context
	output          storage.RecordWriter
	temporarySchema *schemapb.CollectionSchema
	targetSchema    *schemapb.CollectionSchema
	dataTS          uint64
	clusterID       string
	runFunctions    bool
}

func newImportV3FinalWriter(ctx context.Context, output storage.RecordWriter, temporarySchema, targetSchema *schemapb.CollectionSchema, dataTS uint64, clusterID string) storage.RecordWriter {
	return &importV3FinalWriter{ctx: ctx, output: output, temporarySchema: temporarySchema, targetSchema: targetSchema, dataTS: dataTS, clusterID: clusterID, runFunctions: !temporarySchemaContainsFunctionOutput(temporarySchema, targetSchema)}
}

func temporarySchemaContainsFunctionOutput(temporarySchema, targetSchema *schemapb.CollectionSchema) bool {
	temporaryFields := make(map[int64]struct{})
	for _, field := range typeutil.GetAllFieldSchemas(temporarySchema) {
		temporaryFields[field.GetFieldID()] = struct{}{}
	}
	for _, field := range typeutil.GetAllFieldSchemas(targetSchema) {
		if field.GetIsFunctionOutput() {
			if _, ok := temporaryFields[field.GetFieldID()]; ok {
				return true
			}
		}
	}
	return false
}

func (w *importV3FinalWriter) Write(record storage.Record) error {
	required := typeutil.NewSet[int64]()
	for _, field := range typeutil.GetAllFieldSchemas(w.temporarySchema) {
		required.Insert(field.GetFieldID())
	}
	data, err := storage.RecordToInsertData(record, w.targetSchema, required)
	if err != nil {
		return merr.WrapErrDataIntegrity(err, "convert ImportTaskV3 merged record")
	}
	rows := data.GetRowNum()
	if rows == 0 {
		return nil
	}
	if ts := data.Data[common.TimeStampField]; ts == nil || ts.RowNum() == 0 {
		timestamps := make([]int64, rows)
		for index := range timestamps {
			timestamps[index] = int64(w.dataTS)
		}
		data.Data[common.TimeStampField] = &storage.Int64FieldData{Data: timestamps}
	} else if ts.RowNum() != rows {
		return merr.WrapErrDataIntegrityMsg("ImportTaskV3 source timestamp rows mismatch: timestamps=%d rows=%d", ts.RowNum(), rows)
	}
	if w.runFunctions && len(w.targetSchema.GetFunctions()) > 0 {
		if err := embedding.RunAll(w.ctx, w.targetSchema, data, embedding.RunOptions{
			ClusterID: w.clusterID, DBName: w.targetSchema.GetDbName(),
			AllowNonBM25Outputs: common.GetCollectionAllowInsertNonBM25FunctionOutputs(w.targetSchema.GetProperties()),
		}); err != nil {
			return merr.Wrap(err, "run ImportTaskV3 functions")
		}
	}
	reader, err := storage.NewInsertDataRecordReader(data, w.targetSchema)
	if err != nil {
		return err
	}
	defer reader.Close()
	finalRecord, err := reader.Next()
	if err != nil {
		return merr.Wrap(err, "build ImportTaskV3 final record")
	}
	return w.output.Write(finalRecord)
}

func (w *importV3FinalWriter) GetWrittenUncompressed() uint64 {
	return w.output.GetWrittenUncompressed()
}

func (w *importV3FinalWriter) Close() error {
	return w.output.Close()
}

func newImportV3IntermediateFactory(req *datapb.ImportTaskV3Request, segmentIndex int, schema *schemapb.CollectionSchema, pluginContext *indexcgopb.StoragePluginContext) importv3.IntermediateWriterFactory {
	return func(_ context.Context, round, group int, _ []importv3.Source) (storage.RecordWriter, func(int64) (importv3.Source, error), error) {
		fields := typeutil.GetAllFieldSchemas(schema)
		columns := make([]int, len(fields))
		fieldIDs := make([]int64, len(fields))
		for index, field := range fields {
			columns[index], fieldIDs[index] = index, field.GetFieldID()
		}
		intermediatePath := path.Join(metautil.BuildImportV3ImportOutputPath(req.GetJobId(), req.GetTaskId()), "merge", fmt.Sprintf("%d_%d_%d_%d.parquet", req.GetRunId(), segmentIndex, round, group))
		writer, err := storage.NewPackedRecordWriter(req.GetStorageConfig().GetBucketName(), []string{intermediatePath}, schema,
			paramtable.Get().DataNodeCfg.ImportBaseBufferSize.GetAsInt64(), packed.DefaultMultiPartUploadSize,
			[]storagecommon.ColumnGroup{{GroupID: 0, Columns: columns, Fields: fieldIDs}}, req.GetStorageConfig(), pluginContext)
		if err != nil {
			return nil, nil, err
		}
		commit := func(rows int64) (importv3.Source, error) {
			if rows <= 0 || writer.GetWrittenRowNum() != rows {
				return importv3.Source{}, merr.WrapErrDataIntegrityMsg("ImportTaskV3 intermediate rows mismatch: writer=%d merge=%d", writer.GetWrittenRowNum(), rows)
			}
			return importv3.Source{
				ID: intermediatePath, Rows: rows,
				Open: func(ctx context.Context) (storage.RecordReader, error) {
					return storage.NewImportFragmentRecordReader(ctx, storage.ImportFragmentReaderSpec{Path: intermediatePath, Format: storage.ImportFragmentFormatParquet, StartRow: 0, EndRow: rows, Rows: rows}, schema,
						storage.WithBufferSize(paramtable.Get().DataNodeCfg.ImportBaseBufferSize.GetAsInt64()), storage.WithStorageConfig(req.GetStorageConfig()), storage.WithPluginContext(pluginContext))
				},
			}, nil
		}
		return writer, commit, nil
	}
}

func (node *DataNode) readReshardTaskPlan(ctx context.Context, cfg *indexpb.StorageConfig, jobID, taskID int64) (*datapb.ReshardTaskPlan, error) {
	cm, err := node.storageFactory.NewChunkManager(ctx, cfg)
	if err != nil {
		return nil, err
	}
	planPath := metautil.BuildImportV3ReshardPlanPath(jobID, taskID)
	payload, err := cm.Read(ctx, planPath)
	if err != nil {
		return nil, err
	}
	plan := &datapb.ReshardTaskPlan{}
	if err := proto.Unmarshal(payload, plan); err != nil {
		return nil, merr.WrapErrDataIntegrityMsg("decode ReshardTask plan %s: %s", planPath, err.Error())
	}
	return plan, nil
}

func (node *DataNode) readImportTaskPlan(ctx context.Context, cfg *indexpb.StorageConfig, jobID, taskID int64) (*datapb.ImportTaskPlan, error) {
	cm, err := node.storageFactory.NewChunkManager(ctx, cfg)
	if err != nil {
		return nil, err
	}
	planPath := metautil.BuildImportV3ImportPlanPath(jobID, taskID)
	payload, err := cm.Read(ctx, planPath)
	if err != nil {
		return nil, err
	}
	plan := &datapb.ImportTaskPlan{}
	if err := proto.Unmarshal(payload, plan); err != nil {
		return nil, merr.WrapErrDataIntegrityMsg("decode ImportTaskV3 plan %s: %s", planPath, err.Error())
	}
	return plan, nil
}

func importV3TaskCommonState(state importv3.State) taskcommon.State {
	switch state {
	case importv3.StatePending:
		return taskcommon.Init
	case importv3.StateRunning:
		return taskcommon.InProgress
	case importv3.StateRetry:
		return taskcommon.Retry
	case importv3.StateCompleted:
		return taskcommon.Finished
	case importv3.StateFailed:
		return taskcommon.Failed
	default:
		return taskcommon.None
	}
}

func reshardTaskState(state importv3.State) datapb.ImportTaskStateV2 {
	switch state {
	case importv3.StatePending:
		return datapb.ImportTaskStateV2_Pending
	case importv3.StateRunning:
		return datapb.ImportTaskStateV2_InProgress
	case importv3.StateRetry:
		return datapb.ImportTaskStateV2_Retry
	case importv3.StateCompleted:
		return datapb.ImportTaskStateV2_Completed
	case importv3.StateFailed:
		return datapb.ImportTaskStateV2_Failed
	default:
		return datapb.ImportTaskStateV2_None
	}
}

func importTaskV3State(state importv3.State) datapb.ImportTaskStateV2 {
	switch state {
	case importv3.StatePending:
		return datapb.ImportTaskStateV2_Pending
	case importv3.StateRunning:
		return datapb.ImportTaskStateV2_InProgress
	case importv3.StateRetry:
		return datapb.ImportTaskStateV2_Retry
	case importv3.StateCompleted:
		return datapb.ImportTaskStateV2_Completed
	case importv3.StateFailed:
		return datapb.ImportTaskStateV2_Failed
	default:
		return datapb.ImportTaskStateV2_None
	}
}
