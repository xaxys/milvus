package datacoord

// The V3 planner is deliberately small and deterministic. A restart keeps
// ready tasks, fills the missing logical tasks, and then advances the job
// state without depending on map iteration.

import (
	"context"
	"math"
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	importbinlog "github.com/milvus-io/milvus/internal/util/importutilv2/binlog"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func writeImportV3Proto(ctx context.Context, cm storage.ChunkManager, objectPath string, msg proto.Message) error {
	payload, err := proto.Marshal(msg)
	if err != nil {
		return merr.WrapErrSerializationFailed(err, "marshal import v3 object")
	}
	if err := cm.Write(ctx, objectPath, payload); err != nil {
		return merr.Wrap(err, "write import v3 object")
	}
	return nil
}

func calculateV3Slots(workingSet, memoryPerSlot int64) int64 {
	return max((workingSet+memoryPerSlot-1)/memoryPerSlot, 1)
}

func calculateV3ReshardTaskSlot(readBuffer, fragmentTarget, writerBuffer, memoryPerSlot int64) int64 {
	workingSet := 3*readBuffer + 2*fragmentTarget + writerBuffer
	return calculateV3Slots(workingSet, memoryPerSlot)
}

func calculateV3ImportTaskSlot(readBuffer, writerBuffer, memoryPerSlot int64, fanIn int) int64 {
	workingSet := int64(fanIn)*2*readBuffer + 3*readBuffer + writerBuffer
	return calculateV3Slots(workingSet, memoryPerSlot)
}

func importV3SourceFormat(file *internalpb.ImportFile) datapb.ImportSourceFormat {
	ft, _ := importutilv2.GetFileType(file)
	switch ft {
	case importutilv2.JSON:
		return datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_JSON
	case importutilv2.JSONLines:
		return datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_JSON_LINES
	case importutilv2.CSV:
		return datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_CSV
	case importutilv2.Parquet:
		return datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_PARQUET
	case importutilv2.Numpy:
		return datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_NUMPY
	default:
		return datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_BACKUP_BINLOG
	}
}

func (c *importCheckerV3) createV3ReshardTasks(job ImportJob) error {
	files := job.GetFiles()
	if len(files) == 0 {
		state := internalpb.ImportJobState_Uncommitted
		if job.GetAutoCommit() {
			state = internalpb.ImportJobState_Completed
		}
		return c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(state))
	}
	sortSpec, err := v3DefaultSortSpec(job.GetSchema())
	if err != nil {
		return err
	}
	covered, err := c.loadV3ReshardSourceIDs(job)
	if err != nil {
		return err
	}
	missingFiles := make([]*internalpb.ImportFile, 0, len(files)-len(covered))
	for _, file := range files {
		if _, ok := covered[file.GetId()]; !ok {
			missingFiles = append(missingFiles, file)
		}
	}
	missing, err := c.groupV3ReshardSources(missingFiles)
	if err != nil {
		return err
	}
	if len(missing) > 0 {
		start, _, err := c.alloc.AllocN(int64(len(missing)))
		if err != nil {
			return err
		}
		for i, bin := range missing {
			if err := c.createV3ReshardTask(job, sortSpec, start+int64(i), bin); err != nil {
				return err
			}
		}
	}
	return c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Resharding))
}

func (c *importCheckerV3) createV3ReshardTask(job ImportJob, sortSpec *datapb.SortSpec, taskID int64, bin v3ReshardBin) error {
	sources := make([]*datapb.SourceFileSpec, 0, len(bin.sources))
	for _, source := range bin.sources {
		spec, err := c.buildV3SourceFileSpec(job, source)
		if err != nil {
			return err
		}
		sources = append(sources, spec)
	}
	fragmentSize := Params.DataCoordCfg.ImportFragmentSize.GetAsInt64() * 1024 * 1024
	plan := &datapb.ReshardTaskPlan{
		CollectionId: job.GetCollectionID(),
		Schema:       proto.Clone(job.GetSchema()).(*schemapb.CollectionSchema),
		TempSchema:   buildImportV3TempSchema(job.GetSchema(), importutilv2.IsBackup(job.GetOptions())),
		Vchannels:    append([]string(nil), job.GetVchannels()...),
		Partitions:   append([]int64(nil), job.GetPartitionIDs()...),
		Sort:         proto.Clone(sortSpec).(*datapb.SortSpec),
		FragmentSize: fragmentSize,
		Sources:      sources,
	}
	if err := writeImportV3Proto(c.ctx, c.meta.chunkManager, metautil.BuildImportV3ReshardPlanPath(job.GetJobID(), taskID), plan); err != nil {
		return err
	}
	slot := calculateV3ReshardTaskSlot(
		Params.DataNodeCfg.ImportBaseBufferSize.GetAsInt64(),
		fragmentSize,
		packed.DefaultWriteBufferSize,
		Params.DataCoordCfg.ImportMemoryLimitPerSlot.GetAsInt64(),
	)
	task := newReshardTask(&datapb.ReshardTask{JobId: job.GetJobID(), TaskId: taskID, CollectionId: job.GetCollectionID(), State: datapb.ImportTaskStateV2_Pending, RunId: 1, NodeId: NullNodeID, Slot: slot}, c.importMeta, c.meta, c.alloc)
	return c.importMeta.AddTask(c.ctx, task)
}

func (c *importCheckerV3) buildV3SourceFileSpec(job ImportJob, source v3ReshardSource) (*datapb.SourceFileSpec, error) {
	format := importV3SourceFormat(source.file)
	spec := &datapb.SourceFileSpec{
		Ordinal: int32(source.ordinal),
		File:    proto.Clone(source.file).(*internalpb.ImportFile),
		Format:  format,
		Backup:  importutilv2.IsBackup(job.GetOptions()),
		Options: &datapb.ReaderOptions{},
	}
	if format == datapb.ImportSourceFormat_IMPORT_SOURCE_FORMAT_CSV {
		separator, err := importutilv2.GetCSVSep(job.GetOptions())
		if err != nil {
			return nil, err
		}
		nullKey, err := importutilv2.GetCSVNullKey(job.GetOptions())
		if err != nil {
			return nil, err
		}
		spec.Options.Separator = string(separator)
		spec.Options.NullKey = nullKey
	}
	if !spec.GetBackup() {
		return spec, nil
	}
	startTS, endTS, err := importutilv2.ParseTimeRange(job.GetOptions())
	if err != nil {
		return nil, err
	}
	storageVersion, err := importutilv2.GetStorageVersion(job.GetOptions())
	if err != nil {
		return nil, err
	}
	insertObjects, deltaObjects, err := importbinlog.ExpandObjects(c.ctx, c.meta.chunkManager, source.file.GetPaths())
	if err != nil {
		return nil, err
	}
	fieldIDs := make([]int64, 0, len(insertObjects))
	for fieldID := range insertObjects {
		fieldIDs = append(fieldIDs, fieldID)
	}
	sort.Slice(fieldIDs, func(i, j int) bool { return fieldIDs[i] < fieldIDs[j] })
	for _, fieldID := range fieldIDs {
		spec.InsertFields = append(spec.InsertFields, &datapb.ExpandedBinlogField{
			Id:    fieldID,
			Paths: append([]string(nil), insertObjects[fieldID]...),
		})
	}
	spec.DeltaPaths = append([]string(nil), deltaObjects...)
	spec.Options.StartTs = startTS
	spec.Options.EndTs = endTS
	spec.Options.StorageVersion = storageVersion
	return spec, nil
}

func (c *importCheckerV3) loadV3ReshardSourceIDs(job ImportJob) (map[int64]struct{}, error) {
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ReshardTaskType), WithJob(job.GetJobID()))
	jobFiles := make(map[int64]struct{}, len(job.GetFiles()))
	for _, file := range job.GetFiles() {
		jobFiles[file.GetId()] = struct{}{}
	}
	covered := make(map[int64]struct{}, len(jobFiles))
	for _, generic := range tasks {
		task, ok := generic.(*reshardTask)
		if !ok {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 reshard task set contains an unexpected task type")
		}
		if task.GetState() == datapb.ImportTaskStateV2_Failed {
			return nil, merr.WrapErrImportSysFailedMsg("import v3 reshard task %d failed: %s", task.GetTaskID(), task.GetReason())
		}
		p := task.task.Load()
		plan := &datapb.ReshardTaskPlan{}
		if err := loadImportV3Proto(c.ctx, c.meta.chunkManager, metautil.BuildImportV3ReshardPlanPath(job.GetJobID(), p.GetTaskId()), plan); err != nil {
			return nil, err
		}
		if plan.GetCollectionId() != job.GetCollectionID() {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 reshard task plan identity mismatch")
		}
		if len(plan.GetSources()) == 0 {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 reshard task plan has no source")
		}
		for _, source := range plan.GetSources() {
			if source == nil || source.GetFile() == nil {
				return nil, merr.WrapErrDataIntegrityMsg("import v3 reshard task plan has a nil source")
			}
			fileID := source.GetFile().GetId()
			if _, ok := jobFiles[fileID]; !ok {
				return nil, merr.WrapErrDataIntegrityMsg("import v3 reshard task contains source %d outside the job", fileID)
			}
			if _, ok := covered[fileID]; ok {
				return nil, merr.WrapErrDataIntegrityMsg("import v3 reshard task set contains duplicate source %d", fileID)
			}
			covered[fileID] = struct{}{}
		}
	}
	return covered, nil
}

type v3ReshardSource struct {
	file    *internalpb.ImportFile
	ordinal int
	size    int64
}

type v3ReshardBin struct {
	sources []v3ReshardSource
	size    int64
}

// groupV3ReshardSources implements stable one-dimensional BFD. ImportFile is
// the atom: no path/file splitting, no backtracking, and an oversized file owns
// one bin. Equal-size files keep their job ordinal; equal-fit bins keep their
// creation ordinal.
func (c *importCheckerV3) groupV3ReshardSources(files []*internalpb.ImportFile) ([]v3ReshardBin, error) {
	sources := make([]v3ReshardSource, 0, len(files))
	for ordinal, file := range files {
		size, err := storage.GetFilesSize(c.ctx, file.GetPaths(), c.meta.chunkManager)
		if err != nil {
			return nil, merr.Wrapf(err, "estimate import v3 source file %d", file.GetId())
		}
		sources = append(sources, v3ReshardSource{file: file, ordinal: ordinal, size: size})
	}
	sort.SliceStable(sources, func(i, j int) bool {
		return sources[i].size > sources[j].size
	})
	target := Params.DataCoordCfg.MaxSizeInMBPerImportTask.GetAsInt64() * 1024 * 1024
	if target <= 0 {
		return nil, merr.WrapErrImportSysFailedMsg("import v3 reshard BFD target must be positive")
	}
	bins := make([]v3ReshardBin, 0)
	for _, source := range sources {
		best := -1
		bestRemaining := int64(math.MaxInt64)
		if source.size <= target {
			for i := range bins {
				remaining := target - bins[i].size - source.size
				if remaining >= 0 && remaining < bestRemaining {
					best, bestRemaining = i, remaining
				}
			}
		}
		if best < 0 {
			bins = append(bins, v3ReshardBin{sources: []v3ReshardSource{source}, size: source.size})
			continue
		}
		bins[best].sources = append(bins[best].sources, source)
		bins[best].size += source.size
	}
	return bins, nil
}

func v3DefaultSortSpec(schema *schemapb.CollectionSchema) (*datapb.SortSpec, error) {
	pk, err := typeutil.GetPrimaryFieldSchema(schema)
	if err != nil {
		return nil, merr.Wrap(err, "import v3 schema has no primary key")
	}
	toSpec := func(field *schemapb.FieldSchema) (*datapb.SortFieldSpec, error) {
		var keyType datapb.SortKeyType
		switch field.GetDataType() {
		case schemapb.DataType_Int64:
			keyType = datapb.SortKeyType_SORT_KEY_TYPE_INT64
		case schemapb.DataType_VarChar:
			keyType = datapb.SortKeyType_SORT_KEY_TYPE_STRING
		default:
			return nil, merr.WrapErrImportSysFailedMsg("import v3 sort field %d has unsupported type %s", field.GetFieldID(), field.GetDataType())
		}
		return &datapb.SortFieldSpec{FieldId: field.GetFieldID(), KeyType: keyType}, nil
	}
	spec := &datapb.SortSpec{}
	if schema.GetEnableNamespace() {
		partitionKey, err := typeutil.GetPartitionKeyFieldSchema(schema)
		if err != nil {
			return nil, merr.Wrap(err, "import v3 namespace schema has no partition key")
		}
		field, err := toSpec(partitionKey)
		if err != nil {
			return nil, err
		}
		spec.Fields = append(spec.Fields, field)
	}
	field, err := toSpec(pk)
	if err != nil {
		return nil, err
	}
	spec.Fields = append(spec.Fields, field)
	return spec, nil
}

// buildImportV3TempSchema describes the fields physically present in
// immutable fragments. Ordinary fragments carry user fields plus materialized
// PK/RowID, but timestamp is supplied by the import data timestamp at final
// merge. Backup fragments retain source timestamp, so they use the full system
// field schema. Function outputs are produced only by the final transform and
// are therefore omitted from the temporary reader schema.
func buildImportV3TempSchema(schema *schemapb.CollectionSchema, backup bool) *schemapb.CollectionSchema {
	cloned := proto.Clone(schema).(*schemapb.CollectionSchema)
	fields := make([]*schemapb.FieldSchema, 0, len(cloned.GetFields())+2)
	for _, field := range cloned.GetFields() {
		if field.GetIsFunctionOutput() && !backup {
			continue
		}
		fields = append(fields, field)
	}
	cloned.Fields = fields
	if backup {
		return typeutil.AppendSystemFields(cloned)
	}
	cloned.Fields = append(cloned.Fields, &schemapb.FieldSchema{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64})
	return cloned
}

type v3PlanningFragment struct {
	sourceID       int64
	channelIndex   int32
	partitionIndex int32
	seq            int64
	path           string
	rows           int64
	bytes          int64
}

func loadImportV3Proto(ctx context.Context, cm storage.ChunkManager, objectPath string, target proto.Message) error {
	data, err := cm.Read(ctx, objectPath)
	if err != nil {
		return merr.Wrap(err, "read import v3 planning object")
	}
	if err := proto.Unmarshal(data, target); err != nil {
		return merr.WrapErrDataIntegrity(err, "unmarshal import v3 planning object")
	}
	return nil
}

func (c *importCheckerV3) summarizeV3ReshardResults(job ImportJob) (bool, int64, error) {
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ReshardTaskType), WithJob(job.GetJobID()))
	if len(tasks) == 0 {
		return false, 0, nil
	}
	var totalRows int64
	for _, generic := range tasks {
		task, ok := generic.(*reshardTask)
		if !ok {
			return false, 0, merr.WrapErrDataIntegrityMsg("import v3 reshard result set contains an unexpected task type")
		}
		if task.GetState() != datapb.ImportTaskStateV2_Completed {
			return false, 0, nil
		}
		p := task.task.Load()
		manifest, err := loadReshardResultManifest(c.ctx, c.meta.chunkManager, p.GetJobId(), p.GetTaskId(), p.GetRunId())
		if err != nil {
			return false, 0, err
		}
		if err := validateReshardManifest(manifest); err != nil {
			return false, 0, err
		}
		for _, fragment := range manifest.GetFragments() {
			totalRows += fragment.GetRows()
		}
	}
	return true, totalRows, nil
}

func (c *importCheckerV3) cleanupPreparingV3ImportTasks(job ImportJob) error {
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ImportTaskV3Type), WithJob(job.GetJobID()))
	for _, generic := range tasks {
		task, ok := generic.(*importTaskV3)
		if !ok {
			return merr.WrapErrDataIntegrityMsg("import v3 planning task set contains an unexpected task type")
		}
		if task.GetState() != datapb.ImportTaskStateV2_None {
			continue
		}
		segmentIDs := task.task.Load().GetSegments()
		if len(segmentIDs) > 0 {
			if err := c.meta.UpdateSegmentsInfo(c.ctx, dropImportV3Segments(segmentIDs, false)); err != nil {
				return err
			}
		}
		if err := c.importMeta.RemoveTask(c.ctx, task.GetTaskID()); err != nil {
			return err
		}
	}
	return nil
}

func (c *importCheckerV3) loadExistingImportV3Tasks(job ImportJob) (map[string]*datapb.ImportTaskPlan, error) {
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ImportTaskV3Type), WithJob(job.GetJobID()))
	existing := make(map[string]*datapb.ImportTaskPlan, len(tasks))
	for _, generic := range tasks {
		task, ok := generic.(*importTaskV3)
		if !ok {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 planning task set contains an unexpected task type")
		}
		if task.GetState() == datapb.ImportTaskStateV2_None {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 task %d is still preparing", task.GetTaskID())
		}
		if task.GetState() == datapb.ImportTaskStateV2_Failed {
			return nil, merr.WrapErrImportSysFailedMsg("import v3 task %d failed: %s", task.GetTaskID(), task.GetReason())
		}
		p := task.task.Load()
		plan := &datapb.ImportTaskPlan{}
		if err := loadImportV3Proto(c.ctx, c.meta.chunkManager, metautil.BuildImportV3ImportPlanPath(job.GetJobID(), p.GetTaskId()), plan); err != nil {
			return nil, err
		}
		if plan.GetCollectionId() != job.GetCollectionID() {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 task plan identity mismatch")
		}
		if len(plan.GetSegments()) == 0 || len(plan.GetSegments()) != len(p.GetSegments()) {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 task plan segment count mismatch")
		}
		channel := plan.GetSegments()[0].GetVchannel()
		for _, segment := range plan.GetSegments() {
			if channel == "" || segment.GetVchannel() != channel {
				return nil, merr.WrapErrDataIntegrityMsg("import v3 task plan must contain exactly one vchannel")
			}
		}
		if _, ok := existing[channel]; ok {
			return nil, merr.WrapErrDataIntegrityMsg("import v3 planning has duplicate tasks for vchannel %s", channel)
		}
		existing[channel] = plan
	}
	return existing, nil
}

func (c *importCheckerV3) planV3Job(job ImportJob) error {
	if _, err := validateImportV3Schema(c.meta, job.GetCollectionID(), job.GetSchema()); err != nil {
		return err
	}
	if err := c.cleanupPreparingV3ImportTasks(job); err != nil {
		return err
	}
	existingTasks, err := c.loadExistingImportV3Tasks(job)
	if err != nil {
		return err
	}
	sortSpec, err := v3DefaultSortSpec(job.GetSchema())
	if err != nil {
		return err
	}
	reshards := c.importMeta.GetTaskBy(c.ctx, WithType(ReshardTaskType), WithJob(job.GetJobID()))
	if len(reshards) == 0 {
		return merr.WrapErrImportSysFailedMsg("import v3 planning has no completed reshard tasks")
	}
	if len(job.GetVchannels()) == 0 {
		return merr.WrapErrDataIntegrityMsg("import v3 job has no vchannels")
	}
	fragments := make([]v3PlanningFragment, 0)
	for _, generic := range reshards {
		t := generic.(*reshardTask)
		if t.GetState() != datapb.ImportTaskStateV2_Completed {
			return nil
		}
		p := t.task.Load()
		manifest, err := loadReshardResultManifest(c.ctx, c.meta.chunkManager, p.GetJobId(), p.GetTaskId(), p.GetRunId())
		if err != nil {
			return err
		}
		if err := validateReshardManifest(manifest); err != nil {
			return err
		}
		for _, f := range manifest.GetFragments() {
			fragments = append(fragments, v3PlanningFragment{
				sourceID: p.GetTaskId(), channelIndex: f.GetChannelIndex(), partitionIndex: f.GetPartitionIndex(),
				seq: f.GetSeq(), path: f.GetPath(), rows: f.GetRows(), bytes: f.GetLogicalBytes(),
			})
		}
	}
	sort.Slice(fragments, func(i, j int) bool {
		a, b := fragments[i], fragments[j]
		if a.channelIndex != b.channelIndex {
			return a.channelIndex < b.channelIndex
		}
		if a.partitionIndex != b.partitionIndex {
			return a.partitionIndex < b.partitionIndex
		}
		if a.sourceID != b.sourceID {
			return a.sourceID < b.sourceID
		}
		return a.seq < b.seq
	})
	targetSchema := typeutil.AppendSystemFields(job.GetSchema())
	temporarySchema := buildImportV3TempSchema(job.GetSchema(), importutilv2.IsBackup(job.GetOptions()))
	segmentPlans := make([]*datapb.SegmentPlan, 0)
	target := getExpectedSegmentSize(c.meta, job.GetCollectionID(), job.GetSchema())
	var current *datapb.SegmentPlan
	var currentBytes int64
	for _, f := range fragments {
		if f.channelIndex < 0 || int(f.channelIndex) >= len(job.GetVchannels()) || f.partitionIndex < 0 || int(f.partitionIndex) >= len(job.GetPartitionIDs()) {
			return merr.WrapErrDataIntegrityMsg("import v3 fragment bucket ordinal is out of range")
		}
		vchannel := job.GetVchannels()[f.channelIndex]
		partitionID := job.GetPartitionIDs()[f.partitionIndex]
		if current == nil || current.GetVchannel() != vchannel || current.GetPartitionId() != partitionID || (currentBytes > 0 && currentBytes+f.bytes > target) {
			current = &datapb.SegmentPlan{Vchannel: vchannel, PartitionId: partitionID}
			segmentPlans = append(segmentPlans, current)
			currentBytes = 0
		}
		current.Fragments = append(current.Fragments, &datapb.FragmentRef{Path: f.path, RowCount: f.rows})
		current.Rows += f.rows
		currentBytes += f.bytes
	}
	if len(segmentPlans) == 0 {
		if len(existingTasks) > 0 {
			return merr.WrapErrDataIntegrityMsg("import v3 planning has tasks but no segment plan")
		}
		return c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_IndexBuilding))
	}
	ttl, ttlErr := common.GetCollectionTTL(job.GetSchema().GetProperties())
	if ttlErr != nil {
		return ttlErr
	}
	writerFormat := Params.DataNodeCfg.StorageFormat.GetValue()
	maxRows := int64(1)
	for _, segment := range segmentPlans {
		maxRows = max(maxRows, segment.GetRows())
	}
	writerSpec := &datapb.WriterSpec{
		StorageVersion: importStorageVersion(false),
		SchemaVersion:  int64(targetSchema.GetVersion()),
		Format:         writerFormat,
		V2:             &datapb.V2PackedIOConfig{BufferSize: packed.DefaultWriteBufferSize, MultipartSize: packed.DefaultMultiPartUploadSize},
		TtlNanos:       ttl.Nanoseconds(),
		PkCapacity:     maxRows,
		BloomType:      Params.CommonCfg.BloomFilterType.GetValue(),
		BloomFpp:       Params.CommonCfg.MaxBloomFalsePositive.GetAsFloat(),
	}
	for _, function := range targetSchema.GetFunctions() {
		if function.GetType() == schemapb.FunctionType_BM25 {
			writerSpec.Bm25Fields = append(writerSpec.Bm25Fields, function.GetOutputFieldIds()...)
		}
	}
	sort.Slice(writerSpec.Bm25Fields, func(i, j int) bool { return writerSpec.Bm25Fields[i] < writerSpec.Bm25Fields[j] })
	for _, field := range typeutil.GetAllFieldSchemas(targetSchema) {
		if field.GetDataType() == schemapb.DataType_Text {
			writerSpec.Text = append(writerSpec.Text, &datapb.TextColumnWriteSpec{
				FieldId: field.GetFieldID(), InlineLimit: Params.DataNodeCfg.TextInlineThreshold.GetAsInt64(),
				LobLimit: Params.DataNodeCfg.TextMaxLobFileBytes.GetAsInt64(), FlushLimit: Params.DataNodeCfg.TextFlushThresholdBytes.GetAsInt64(),
			})
		}
	}
	columnGroups := storagecommon.SplitColumns(typeutil.GetAllFieldSchemas(targetSchema), map[int64]storagecommon.ColumnStats{}, storagecommon.DefaultPolicies()...)
	columnGroups = storagecommon.FillColumnGroupFormats(columnGroups, writerFormat)
	writerSpec.Groups = make([]*datapb.ColumnGroupSpec, 0, len(columnGroups))
	for _, group := range columnGroups {
		writerSpec.Groups = append(writerSpec.Groups, &datapb.ColumnGroupSpec{Id: group.GroupID, Fields: append([]int64(nil), group.Fields...), Format: group.Format})
	}
	taskSpecs := make([]v3ImportTaskSpec, 0, len(job.GetVchannels()))
	for _, channel := range job.GetVchannels() {
		segments := make([]*datapb.SegmentPlan, 0)
		for _, s := range segmentPlans {
			if s.GetVchannel() == channel {
				segments = append(segments, s)
			}
		}
		if len(segments) == 0 {
			continue
		}
		taskSegments := make([]*datapb.SegmentPlan, len(segments))
		for i, segment := range segments {
			taskSegments[i] = proto.Clone(segment).(*datapb.SegmentPlan)
		}
		taskSpecs = append(taskSpecs, v3ImportTaskSpec{channel: channel, segments: taskSegments})
	}

	missing := make([]v3ImportTaskSpec, 0, len(taskSpecs))
	expectedChannels := make(map[string]struct{}, len(taskSpecs))
	for _, spec := range taskSpecs {
		expectedChannels[spec.channel] = struct{}{}
		if existing, ok := existingTasks[spec.channel]; ok {
			if !sameImportV3FragmentCoverage(existing.GetSegments(), spec.segments) {
				return merr.WrapErrDataIntegrityMsg("import v3 task plan for vchannel %s does not cover current planning input", spec.channel)
			}
			continue
		}
		missing = append(missing, spec)
	}
	for channel := range existingTasks {
		if _, ok := expectedChannels[channel]; !ok {
			return merr.WrapErrDataIntegrityMsg("import v3 task plan has unexpected vchannel %s", channel)
		}
	}
	if len(missing) > 0 {
		taskStart, _, err := c.alloc.AllocN(int64(len(missing)))
		if err != nil {
			return err
		}
		for i, spec := range missing {
			if err := c.createImportV3Task(job, sortSpec, targetSchema, temporarySchema, writerSpec, taskStart+int64(i), spec); err != nil {
				return err
			}
		}
	}
	return c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Importing))
}

type v3ImportTaskSpec struct {
	channel  string
	segments []*datapb.SegmentPlan
}

func sameImportV3FragmentCoverage(left, right []*datapb.SegmentPlan) bool {
	flatten := func(segments []*datapb.SegmentPlan) []*datapb.FragmentRef {
		fragments := make([]*datapb.FragmentRef, 0)
		for _, segment := range segments {
			fragments = append(fragments, segment.GetFragments()...)
		}
		return fragments
	}
	return proto.Equal(
		&datapb.SegmentPlan{Fragments: flatten(left)},
		&datapb.SegmentPlan{Fragments: flatten(right)},
	)
}

func (c *importCheckerV3) createImportV3Task(
	job ImportJob,
	sortSpec *datapb.SortSpec,
	targetSchema, temporarySchema *schemapb.CollectionSchema,
	writerSpec *datapb.WriterSpec,
	taskID int64,
	spec v3ImportTaskSpec,
) error {
	taskSegments := spec.segments
	segmentStart, _, err := c.alloc.AllocN(int64(len(taskSegments)))
	if err != nil {
		return err
	}
	outputIDs := make([]int64, len(taskSegments))
	for i := range outputIDs {
		outputIDs[i] = segmentStart + int64(i)
	}
	perSegment := int64(1 + len(writerSpec.GetBm25Fields()))
	if writerSpec.GetStorageVersion() == storage.StorageV2 {
		perSegment += int64(len(writerSpec.GetGroups()))
	}
	required := int64(len(taskSegments)) * perSegment
	if required <= 0 || required > math.MaxUint32 {
		return merr.WrapErrImportSysFailedMsg("import v3 log id budget is invalid")
	}
	logBegin, logEnd, err := c.alloc.AllocN(required)
	if err != nil {
		return err
	}
	fanIn := Params.DataCoordCfg.FragmentMergeFanIn.GetAsInt()
	plan := &datapb.ImportTaskPlan{
		Sort:         proto.Clone(sortSpec).(*datapb.SortSpec),
		Segments:     taskSegments,
		FanIn:        int32(fanIn),
		Writer:       proto.Clone(writerSpec).(*datapb.WriterSpec),
		Schema:       proto.Clone(targetSchema).(*schemapb.CollectionSchema),
		TempSchema:   proto.Clone(temporarySchema).(*schemapb.CollectionSchema),
		DataTs:       job.GetDataTs(),
		CollectionId: job.GetCollectionID(),
		ClusterId:    Params.CommonCfg.ClusterPrefix.GetValue(),
	}
	if err := writeImportV3Proto(c.ctx, c.meta.chunkManager, metautil.BuildImportV3ImportPlanPath(job.GetJobID(), taskID), plan); err != nil {
		return err
	}
	plannedRows := int64(0)
	for _, segment := range taskSegments {
		plannedRows += segment.GetRows()
	}
	slot := calculateV3ImportTaskSlot(
		Params.DataNodeCfg.ImportBaseBufferSize.GetAsInt64(),
		packed.DefaultWriteBufferSize,
		Params.DataCoordCfg.ImportMemoryLimitPerSlot.GetAsInt64(),
		fanIn,
	)
	task := newImportTaskV3(&datapb.ImportTaskV3{JobId: job.GetJobID(), TaskId: taskID, CollectionId: job.GetCollectionID(), State: datapb.ImportTaskStateV2_None, RunId: 1, NodeId: NullNodeID, Segments: outputIDs, LogRange: &datapb.IDRange{Begin: logBegin, End: logEnd}, Slot: slot, Rows: plannedRows}, c.importMeta, c.meta, c.alloc)
	if err := c.importMeta.AddTask(c.ctx, task); err != nil {
		return err
	}
	for i, segment := range taskSegments {
		if job.GetDataTs() == 0 {
			if _, err := c.alloc.AllocTimestamp(c.ctx); err != nil {
				return err
			}
		}
		_, err := addImportSegment(c.ctx, c.meta, outputIDs[i], job.GetJobID(), taskID, job.GetCollectionID(), segment.GetPartitionId(), segment.GetVchannel(), datapb.SegmentLevel_L1, writerSpec.GetStorageVersion(), int32(writerSpec.GetSchemaVersion()))
		if err != nil {
			return err
		}
	}
	return c.importMeta.UpdateTask(c.ctx, taskID, UpdateState(datapb.ImportTaskStateV2_Pending))
}
