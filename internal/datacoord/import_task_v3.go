package datacoord

// This file contains the DataCoord scheduler adapters for the V3 control
// records.  V3 keeps run_id inside the task record; it does not introduce a
// separate attempt proto or catalog level.  The adapters deliberately mirror
// the old ImportTask lifecycle: Create persists Running only after the worker
// accepts the request, Query moves transient failures back to Pending, and
// Drop treats a missing worker as ownership loss.

import (
	"context"
	"time"

	"github.com/cockroachdb/errors"
	"go.uber.org/atomic"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/session"
	"github.com/milvus-io/milvus/internal/json"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/taskcommon"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

type reshardTask struct {
	task       atomic.Pointer[datapb.ReshardTask]
	importMeta ImportMeta
	meta       *meta
	alloc      allocator.Allocator
	tr         *timerecord.TimeRecorder
	times      *taskcommon.Times
}

func newReshardTask(p *datapb.ReshardTask, importMeta ImportMeta, meta *meta, alloc ...allocator.Allocator) *reshardTask {
	t := &reshardTask{importMeta: importMeta, meta: meta, tr: timerecord.NewTimeRecorder("reshard task"), times: taskcommon.NewTimes()}
	if len(alloc) > 0 {
		t.alloc = alloc[0]
	}
	t.task.Store(p)
	return t
}

func (t *reshardTask) GetJobID() int64        { return t.task.Load().GetJobId() }
func (t *reshardTask) GetTaskID() int64       { return t.task.Load().GetTaskId() }
func (t *reshardTask) GetCollectionID() int64 { return t.task.Load().GetCollectionId() }
func (t *reshardTask) GetNodeID() int64       { return t.task.Load().GetNodeId() }
func (t *reshardTask) GetType() TaskType      { return ReshardTaskType }
func (t *reshardTask) GetState() datapb.ImportTaskStateV2 {
	return t.task.Load().GetState()
}
func (t *reshardTask) GetReason() string                       { return t.task.Load().GetReason() }
func (t *reshardTask) GetFileStats() []*datapb.ImportFileStats { return nil }
func (t *reshardTask) GetSource() datapb.ImportTaskSourceV2    { return datapb.ImportTaskSourceV2_Request }
func (t *reshardTask) GetTR() *timerecord.TimeRecorder         { return t.tr }
func (t *reshardTask) GetTaskType() taskcommon.Type            { return taskcommon.Reshard }
func (t *reshardTask) GetTaskState() taskcommon.State {
	return taskcommon.FromImportState(t.task.Load().GetState())
}
func (t *reshardTask) GetTaskNodeID() int64                             { return t.GetNodeID() }
func (t *reshardTask) GetTaskSlot() int64                               { return t.task.Load().GetSlot() }
func (t *reshardTask) GetTaskVersion() int64                            { return t.task.Load().GetRunId() }
func (t *reshardTask) SetTaskTime(tt taskcommon.TimeType, tm time.Time) { t.times.SetTaskTime(tt, tm) }
func (t *reshardTask) GetTaskTime(tt taskcommon.TimeType) time.Time     { return tt.GetTaskTime(t.times) }
func (t *reshardTask) setState(state datapb.ImportTaskStateV2) {
	t.task.Load().State = state
}

func (t *reshardTask) CreateTaskOnWorker(nodeID int64, cluster session.Cluster) {
	p := t.task.Load()
	if p.GetRunId() == 0 {
		t.fail("reshard task has no run")
		return
	}
	job := t.importMeta.GetJob(context.TODO(), p.GetJobId())
	if job == nil {
		t.fail("reshard task job is missing")
		return
	}
	req := &datapb.ReshardTaskRequest{
		JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId(),
		Slot: p.GetSlot(), StorageConfig: createStorageConfig(),
		PluginContext: GetReadPluginContext(job.GetOptions()),
	}
	WrapPluginContext(t.GetCollectionID(), job.GetSchema().GetProperties(), req)
	err := cluster.CreateReshard(nodeID, req, t.GetCollectionID())
	if err != nil {
		mlog.Warn(context.TODO(), "create reshard task failed", WrapTaskLog(t, mlog.Err(err))...)
		if !merr.IsRetryableErr(err) {
			t.fail(err.Error())
		}
		return
	}
	if err := t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_InProgress), UpdateNodeID(nodeID)); err != nil {
		mlog.Warn(context.TODO(), "persist reshard task running state failed", WrapTaskLog(t, mlog.Err(err))...)
	}
}

func (t *reshardTask) QueryTaskOnWorker(cluster session.Cluster) {
	p := t.task.Load()
	resp, err := cluster.QueryReshard(t.GetNodeID(), &datapb.QueryReshardTaskRequest{JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId()})
	if err != nil {
		if errors.Is(err, merr.ErrNodeNotFound) {
			t.prepareRetry(cluster)
		} else if !merr.IsRetryableErr(err) {
			t.fail(err.Error())
		} else {
			t.prepareRetry(cluster)
		}
		return
	}
	if resp.GetState() == datapb.ImportTaskStateV2_Retry {
		t.prepareRetry(cluster)
		return
	}
	if resp.GetState() == datapb.ImportTaskStateV2_Failed {
		t.fail(resp.GetReason())
		return
	}
	if resp.GetState() == datapb.ImportTaskStateV2_Completed {
		if t.GetState() != datapb.ImportTaskStateV2_InProgress {
			return
		}
		if err := t.acceptResult(); err != nil {
			t.fail(err.Error())
			return
		}
		if err := t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Completed)); err != nil {
			mlog.Warn(context.TODO(), "persist accepted reshard result marker failed", WrapTaskLog(t, mlog.Err(err))...)
		}
	}
}

func (t *reshardTask) prepareRetry(cluster session.Cluster) {
	p := t.task.Load()
	if t.GetNodeID() != NullNodeID {
		_ = cluster.DropReshard(t.GetNodeID(), &datapb.DropReshardTaskRequest{JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId()})
	}
	newRun := p.GetRunId() + 1
	_ = t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), updateReshardRun(newRun))
}

func (t *reshardTask) acceptResult() error {
	if t.meta == nil || t.meta.chunkManager == nil {
		return merr.WrapErrImportSysFailedMsg("reshard result storage is unavailable")
	}
	p := t.task.Load()
	manifest, err := loadReshardResultManifest(context.TODO(), t.meta.chunkManager, p.GetJobId(), p.GetTaskId(), p.GetRunId())
	if err != nil {
		return err
	}
	return validateReshardManifest(manifest)
}

func (t *reshardTask) DropTaskOnWorker(cluster session.Cluster) {
	p := t.task.Load()
	if t.GetNodeID() != NullNodeID {
		err := cluster.DropReshard(t.GetNodeID(), &datapb.DropReshardTaskRequest{JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId()})
		if err != nil && !errors.Is(err, merr.ErrNodeNotFound) {
			return
		}
	}
	_ = t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateNodeID(NullNodeID))
}

func (t *reshardTask) fail(reason string) {
	_ = t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Failed), UpdateReason(reason))
	_ = t.importMeta.UpdateJob(context.TODO(), t.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(reason))
}
func (t *reshardTask) Clone() ImportTask {
	c := newReshardTask(proto.Clone(t.task.Load()).(*datapb.ReshardTask), t.importMeta, t.meta, t.alloc)
	c.tr, c.times = t.tr, t.times
	return c
}
func (t *reshardTask) MarshalJSON() ([]byte, error) {
	return json.Marshal(metricsinfo.ImportTask{JobID: t.GetJobID(), TaskID: t.GetTaskID(), CollectionID: t.GetCollectionID(), NodeID: t.GetNodeID(), State: t.GetState().String(), Reason: t.GetReason(), TaskType: t.GetType().String()})
}

type importTaskV3 struct {
	task       atomic.Pointer[datapb.ImportTaskV3]
	importMeta ImportMeta
	meta       *meta
	alloc      allocator.Allocator
	tr         *timerecord.TimeRecorder
	times      *taskcommon.Times
}

func newImportTaskV3(p *datapb.ImportTaskV3, importMeta ImportMeta, meta *meta, alloc ...allocator.Allocator) *importTaskV3 {
	t := &importTaskV3{importMeta: importMeta, meta: meta, tr: timerecord.NewTimeRecorder("import v3 task"), times: taskcommon.NewTimes()}
	if len(alloc) > 0 {
		t.alloc = alloc[0]
	}
	t.task.Store(p)
	return t
}
func (t *importTaskV3) GetJobID() int64        { return t.task.Load().GetJobId() }
func (t *importTaskV3) GetTaskID() int64       { return t.task.Load().GetTaskId() }
func (t *importTaskV3) GetCollectionID() int64 { return t.task.Load().GetCollectionId() }
func (t *importTaskV3) GetNodeID() int64       { return t.task.Load().GetNodeId() }
func (t *importTaskV3) GetType() TaskType      { return ImportTaskV3Type }
func (t *importTaskV3) GetState() datapb.ImportTaskStateV2 {
	return t.task.Load().GetState()
}
func (t *importTaskV3) GetReason() string                       { return t.task.Load().GetReason() }
func (t *importTaskV3) GetFileStats() []*datapb.ImportFileStats { return nil }
func (t *importTaskV3) GetSource() datapb.ImportTaskSourceV2 {
	return datapb.ImportTaskSourceV2_Request
}
func (t *importTaskV3) GetTR() *timerecord.TimeRecorder { return t.tr }
func (t *importTaskV3) GetTaskType() taskcommon.Type    { return taskcommon.ImportV3 }
func (t *importTaskV3) GetTaskState() taskcommon.State {
	return taskcommon.FromImportState(t.task.Load().GetState())
}
func (t *importTaskV3) GetTaskNodeID() int64                             { return t.GetNodeID() }
func (t *importTaskV3) GetTaskSlot() int64                               { return t.task.Load().GetSlot() }
func (t *importTaskV3) GetTaskVersion() int64                            { return t.task.Load().GetRunId() }
func (t *importTaskV3) SetTaskTime(tt taskcommon.TimeType, tm time.Time) { t.times.SetTaskTime(tt, tm) }
func (t *importTaskV3) GetTaskTime(tt taskcommon.TimeType) time.Time     { return tt.GetTaskTime(t.times) }
func (t *importTaskV3) setState(state datapb.ImportTaskStateV2) {
	t.task.Load().State = state
}

func (t *importTaskV3) CreateTaskOnWorker(nodeID int64, cluster session.Cluster) {
	p := t.task.Load()
	if p.GetRunId() == 0 || p.GetLogRange() == nil {
		t.fail("import v3 task has no run resources")
		return
	}
	job := t.importMeta.GetJob(context.TODO(), p.GetJobId())
	if job == nil {
		t.fail("import v3 task job is missing")
		return
	}
	req := &datapb.ImportTaskV3Request{
		JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId(), Segments: p.GetSegments(),
		LogRange: p.GetLogRange(), Slot: p.GetSlot(),
		StorageConfig: createStorageConfig(),
		PluginContext: GetReadPluginContext(job.GetOptions()),
	}
	WrapPluginContext(t.GetCollectionID(), job.GetSchema().GetProperties(), req)
	err := cluster.CreateImportV3(nodeID, req, t.GetCollectionID())
	if err != nil {
		mlog.Warn(context.TODO(), "create import v3 task failed", WrapTaskLog(t, mlog.Err(err))...)
		if !merr.IsRetryableErr(err) {
			t.fail(err.Error())
		}
		return
	}
	if err := t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_InProgress), UpdateNodeID(nodeID)); err != nil {
		mlog.Warn(context.TODO(), "persist import v3 task running state failed", WrapTaskLog(t, mlog.Err(err))...)
	}
}

func (t *importTaskV3) QueryTaskOnWorker(cluster session.Cluster) {
	p := t.task.Load()
	resp, err := cluster.QueryImportV3(t.GetNodeID(), &datapb.QueryImportTaskV3Request{JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId()})
	if err != nil {
		if errors.Is(err, merr.ErrNodeNotFound) {
			t.prepareRetry(cluster)
		} else if !merr.IsRetryableErr(err) {
			t.fail(err.Error())
		} else {
			t.prepareRetry(cluster)
		}
		return
	}
	if resp.GetState() == datapb.ImportTaskStateV2_Retry {
		t.prepareRetry(cluster)
		return
	}
	if resp.GetState() == datapb.ImportTaskStateV2_Failed {
		t.fail(resp.GetReason())
		return
	}
	if resp.GetState() == datapb.ImportTaskStateV2_Completed {
		if t.GetState() != datapb.ImportTaskStateV2_InProgress {
			return
		}
		if err := t.acceptResult(resp.GetSegments()); err != nil {
			t.fail(err.Error())
			return
		}
		if err := t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Completed)); err != nil {
			mlog.Warn(context.TODO(), "persist accepted import v3 result marker failed", WrapTaskLog(t, mlog.Err(err))...)
		}
	}
}

func (t *importTaskV3) prepareRetry(cluster session.Cluster) {
	p := t.task.Load()
	if t.alloc == nil || t.meta == nil || p.GetLogRange() == nil {
		t.fail("import v3 retry resources are unavailable")
		return
	}
	type segmentIdentity struct {
		partitionID, storageVersion int64
		schemaVersion               int32
		vchannel                    string
	}
	identities := make([]segmentIdentity, len(p.GetSegments()))
	for i, segmentID := range p.GetSegments() {
		segment := t.meta.GetSegment(context.TODO(), segmentID)
		if segment == nil {
			t.fail("import v3 retry segment is missing")
			return
		}
		identities[i] = segmentIdentity{
			partitionID: segment.GetPartitionID(), vchannel: segment.GetInsertChannel(),
			schemaVersion: segment.GetSchemaVersion(), storageVersion: segment.GetStorageVersion(),
		}
	}
	if t.GetNodeID() != NullNodeID {
		_ = cluster.DropImportV3(t.GetNodeID(), &datapb.DropImportTaskV3Request{JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId()})
	}
	oldSegments := append([]int64(nil), p.GetSegments()...)
	job := t.importMeta.GetJob(context.TODO(), p.GetJobId())
	if job == nil {
		t.fail("import v3 retry job is missing")
		return
	}
	newSegments := make([]int64, len(oldSegments))
	cleanupNewSegments := func() {
		if len(newSegments) > 0 {
			_ = t.meta.UpdateSegmentsInfo(context.TODO(), dropImportV3Segments(newSegments, false))
		}
	}
	for i, identity := range identities {
		segment, err := AllocImportSegment(context.TODO(), t.alloc, t.meta,
			p.GetJobId(), p.GetTaskId(), p.GetCollectionId(), identity.partitionID,
			identity.vchannel, job.GetDataTs(), datapb.SegmentLevel_L1, identity.storageVersion, identity.schemaVersion)
		if err != nil {
			cleanupNewSegments()
			t.fail(err.Error())
			return
		}
		newSegments[i] = segment.GetID()
	}
	// The existing log range is fixed for a plan. A retried run gets a fresh
	// range so a late old writer cannot reuse current segment log IDs.
	begin, end, err := t.alloc.AllocN(p.GetLogRange().GetEnd() - p.GetLogRange().GetBegin())
	if err != nil {
		cleanupNewSegments()
		t.fail(err.Error())
		return
	}
	newRun := p.GetRunId() + 1
	if err := t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), updateImportV3Run(newRun, newSegments, &datapb.IDRange{Begin: begin, End: end})); err != nil {
		cleanupNewSegments()
		t.fail(err.Error())
		return
	}
	if len(oldSegments) > 0 {
		if err := t.meta.UpdateSegmentsInfo(context.TODO(), dropImportV3Segments(oldSegments, false)); err != nil {
			mlog.Warn(context.TODO(), "drop old import v3 retry segments failed", WrapTaskLog(t, mlog.Err(err))...)
		}
	}
}

func (t *importTaskV3) acceptResult(results []*datapb.SegmentResult) error {
	if t.meta == nil {
		return merr.WrapErrImportSysFailedMsg("import v3 result metadata is unavailable")
	}
	p := t.task.Load()
	job := t.importMeta.GetJob(context.TODO(), p.GetJobId())
	if job == nil {
		return merr.WrapErrImportSysFailedMsg("import v3 job is unavailable during result acceptance")
	}
	schemaVersion, err := validateImportV3Schema(t.meta, p.GetCollectionId(), job.GetSchema())
	if err != nil {
		return err
	}
	if err := validateImportResults(results, len(p.GetSegments())); err != nil {
		return err
	}
	namespaceSorted := job.GetSchema().GetEnableNamespace()
	return applyImportResults(context.TODO(), t.meta, p.GetCollectionId(), schemaVersion, p.GetSegments(), !namespaceSorted, namespaceSorted, results)
}

func (t *importTaskV3) DropTaskOnWorker(cluster session.Cluster) {
	p := t.task.Load()
	if t.GetNodeID() != NullNodeID {
		err := cluster.DropImportV3(t.GetNodeID(), &datapb.DropImportTaskV3Request{JobId: p.GetJobId(), TaskId: p.GetTaskId(), RunId: p.GetRunId()})
		if err != nil && !errors.Is(err, merr.ErrNodeNotFound) {
			return
		}
	}
	if t.meta != nil && len(p.GetSegments()) > 0 {
		zeroOnly := t.GetState() == datapb.ImportTaskStateV2_Completed
		if err := t.meta.UpdateSegmentsInfo(context.TODO(), dropImportV3Segments(p.GetSegments(), zeroOnly)); err != nil {
			mlog.Warn(context.TODO(), "drop import v3 segments failed", WrapTaskLog(t, mlog.Err(err))...)
			return
		}
	}
	_ = t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateNodeID(NullNodeID))
}
func (t *importTaskV3) fail(reason string) {
	_ = t.importMeta.UpdateTask(context.TODO(), t.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Failed), UpdateReason(reason))
	_ = t.importMeta.UpdateJob(context.TODO(), t.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(reason))
}
func (t *importTaskV3) Clone() ImportTask {
	c := newImportTaskV3(proto.Clone(t.task.Load()).(*datapb.ImportTaskV3), t.importMeta, t.meta, t.alloc)
	c.tr, c.times = t.tr, t.times
	return c
}
func (t *importTaskV3) MarshalJSON() ([]byte, error) {
	return json.Marshal(metricsinfo.ImportTask{JobID: t.GetJobID(), TaskID: t.GetTaskID(), CollectionID: t.GetCollectionID(), NodeID: t.GetNodeID(), State: t.GetState().String(), Reason: t.GetReason(), TaskType: t.GetType().String()})
}
