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

package datacoord

import (
	"context"
	"errors"
	"path"
	"testing"
	"time"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/session"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
)

// TestQuiesceImportJobKeepsTasksDuringRetention pins that unbound task records
// survive until cleanupTs passes: they back the terminal job's row accounting
// (getImportRowsInfo) during the retention window, like V2 GC.
func TestQuiesceImportJobKeepsTasksDuringRetention(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	log := mlog.With(mlog.FieldJobID(int64(1)))

	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_Completed, NodeId: NullNodeID, SegmentId: 100, Rows: 42,
	}, importMeta, nil, nil)

	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, State: internalpb.ImportJobState_Completed,
		CleanupTs: tsoutil.ComposeTSByTime(time.Now().Add(time.Hour))}}
	importMeta.EXPECT().GetTaskByJob(mock.Anything, int64(1)).Return([]ImportTask{task}).Once()
	require.False(t, checker.quiesceImportJob(job, log))

	job.ImportJob.CleanupTs = tsoutil.ComposeTSByTime(time.Now().Add(-time.Hour))
	importMeta.EXPECT().GetTaskByJob(mock.Anything, int64(1)).Return([]ImportTask{task}).Once()
	importMeta.EXPECT().RemoveTask(mock.Anything, int64(10)).Return(nil).Once()
	require.True(t, checker.quiesceImportJob(job, log))
}

func TestCleanupPreparingV3ImportTasksIsIdempotent(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_None, NodeId: NullNodeID,
	}, importMeta, nil, nil)

	first := importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	second := importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
	mock.InOrder(first, second)
	importMeta.EXPECT().RemoveTask(mock.Anything, int64(10)).Return(nil).Once()

	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
}

func TestCalculateV3TaskSlots(t *testing.T) {
	const mib = int64(1024 * 1024)

	require.Equal(t, int64(3), calculateReshardTaskSlot(16*mib, 128*mib, 32*mib, 160*mib))
	require.Equal(t, int64(4), calculateV3ImportTaskSlot(16*mib, 32*mib, 160*mib, 16))
	require.Equal(t, int64(1), calculateV3Slots(1, 160*mib))
	require.Equal(t, int64(2), calculateV3Slots(160*mib+1, 160*mib))
	// The slot helper must not divide by zero when the configured per-slot
	// memory limit is invalid; paramtable validation remains the primary guard.
	require.Equal(t, int64(1), calculateV3Slots(1, 0))
	require.Equal(t, int64(1), calculateV3Slots(1, -1))
}

func TestEffectiveImportV3FanIn(t *testing.T) {
	require.Equal(t, 16, effectiveImportV3FanIn(16, 0))
	require.Equal(t, 16, effectiveImportV3FanIn(16, 100))
	require.Equal(t, 5, effectiveImportV3FanIn(16, 5))
	require.Equal(t, 2, effectiveImportV3FanIn(16, 1))
	require.Equal(t, 2, effectiveImportV3FanIn(2, 1))
}

func TestCleanupPreparingV3ImportTasksKeepsReadyTask(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_Pending, NodeId: NullNodeID,
	}, importMeta, nil, nil)
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()

	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
}

func TestCleanupPreparingV3ImportTaskDropsOwnedSegment(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	segment, err := addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	require.Equal(t, commonpb.SegmentState_Importing, segment.GetState())
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: meta, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_None, NodeId: NullNodeID, SegmentId: 100,
	}, importMeta, meta, nil)
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	importMeta.EXPECT().RemoveTask(mock.Anything, int64(10)).Return(nil).Once()

	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
	require.Equal(t, commonpb.SegmentState_Dropped, meta.GetSegment(ctx, 100).GetState())
}

func TestCreateImportV3TaskPublishesPendingAfterSegments(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	meta.chunkManager = storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	importMeta := NewMockImportMeta(t)
	alloc := allocator.NewMockAllocator(t)
	alloc.EXPECT().AllocN(int64(1)).Return(int64(200), int64(201), nil).Once()
	checker := &importCheckerV3{ctx: ctx, meta: meta, importMeta: importMeta, alloc: alloc}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, CollectionID: 2, DataTs: 1}}

	var task *importTaskV3
	add := importMeta.EXPECT().AddTask(mock.Anything, mock.Anything).Run(func(_ context.Context, added ImportTask) {
		task = added.(*importTaskV3)
		require.Equal(t, datapb.ImportTaskStateV2_None, task.GetState())
		require.Equal(t, int64(1), task.GetTaskSlot())
		require.Nil(t, meta.GetSegment(ctx, 100))
	}).Return(nil).Once()
	update := importMeta.EXPECT().UpdateTask(mock.Anything, int64(10), mock.Anything).Run(func(_ context.Context, _ int64, actions ...UpdateAction) {
		require.NotNil(t, meta.GetSegment(ctx, 100))
		for _, action := range actions {
			action(task)
		}
	}).Return(nil).Once()
	mock.InOrder(add, update)

	err = checker.createImportV3Task(
		job,
		&datapb.SortSpec{},
		&schemapb.CollectionSchema{},
		&schemapb.CollectionSchema{},
		&datapb.WriterSpec{StorageVersion: 1, SchemaVersion: 3},
		10, 100,
		v3ImportTaskSpec{channel: "v0", partitionID: 4, fragments: []*datapb.FragmentRef{{Path: "f0", RowCount: 1}}, rows: 5},
		false,
	)
	require.NoError(t, err)
	require.Equal(t, datapb.ImportTaskStateV2_Pending, task.GetState())
	segment := meta.GetSegment(ctx, 100)
	require.True(t, segment.GetIsImporting())
	require.False(t, segment.GetIsInvisible())
	require.Equal(t, int32(3), segment.GetSchemaVersion())
}

func TestCreateReshardTasksKeepsExistingAndAddsMissingSources(t *testing.T) {
	ctx := context.Background()
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	missingPath := path.Join(cm.RootPath(), "missing.json")
	require.NoError(t, cm.Write(ctx, missingPath, []byte("{}")))
	importMeta := NewMockImportMeta(t)
	alloc := allocator.NewMockAllocator(t)
	alloc.EXPECT().AllocN(int64(1)).Return(int64(20), int64(21), nil).Once()
	checker := &importCheckerV3{ctx: ctx, meta: &meta{chunkManager: cm}, importMeta: importMeta, alloc: alloc}
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, DataType: schemapb.DataType_Int64, IsPrimaryKey: true}}}
	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, CollectionID: 2, Schema: schema, Vchannels: []string{"v0"}, PartitionIDs: []int64{3},
		Files: []*internalpb.ImportFile{{Id: 1, Paths: []string{"existing.json"}}, {Id: 2, Paths: []string{missingPath}}},
	}, tr: timerecord.NewTimeRecorder("import job")}
	existingPlan := &datapb.ReshardTaskPlan{
		CollectionId: 2,
		Sources:      []*datapb.SourceFileSpec{{File: job.GetFiles()[0]}},
	}
	require.NoError(t, writeImportV3Proto(ctx, cm, metautil.BuildImportReshardPlanPath(1, 10), existingPlan))
	existing := newReshardTask(&datapb.ReshardTask{
		JobId: 1, TaskId: 10, CollectionId: 2, State: datapb.ImportTaskStateV2_InProgress,
	}, importMeta, checker.meta, alloc)
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{existing}).Once()
	importMeta.EXPECT().AddTask(mock.Anything, mock.Anything).Run(func(_ context.Context, added ImportTask) {
		plan := &datapb.ReshardTaskPlan{}
		p := added.(*reshardTask).task.Load()
		require.Equal(t, int64(3), p.GetSlot())
		require.NoError(t, loadImportV3Proto(ctx, cm, metautil.BuildImportReshardPlanPath(1, p.GetTaskId()), plan))
		require.Len(t, plan.GetSources(), 1)
		require.Equal(t, int64(2), plan.GetSources()[0].GetFile().GetId())
	}).Return(nil).Once()
	importMeta.EXPECT().UpdateJob(mock.Anything, int64(1), mock.Anything).Return(nil).Once()

	require.NoError(t, checker.createReshardTasks(job))
}

func TestCreateReshardTasksRetriesJobStateWhenSourcesCovered(t *testing.T) {
	ctx := context.Background()
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: &meta{chunkManager: cm}, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, CollectionID: 2,
		Schema: &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, DataType: schemapb.DataType_Int64, IsPrimaryKey: true}}},
		Files:  []*internalpb.ImportFile{{Id: 1, Paths: []string{"existing.json"}}},
	}, tr: timerecord.NewTimeRecorder("import job")}
	plan := &datapb.ReshardTaskPlan{
		CollectionId: 2,
		Sources:      []*datapb.SourceFileSpec{{File: job.GetFiles()[0]}},
	}
	require.NoError(t, writeImportV3Proto(ctx, cm, metautil.BuildImportReshardPlanPath(1, 10), plan))
	task := newReshardTask(&datapb.ReshardTask{JobId: 1, TaskId: 10, CollectionId: 2, State: datapb.ImportTaskStateV2_Completed}, importMeta, checker.meta, nil)
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	importMeta.EXPECT().UpdateJob(mock.Anything, int64(1), mock.Anything).Return(nil).Once()

	require.NoError(t, checker.createReshardTasks(job))
}

func TestCreateImportV3TaskRejectsMissingDataTs(t *testing.T) {
	checker := &importCheckerV3{ctx: context.Background()}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, CollectionID: 2}}
	err := checker.createImportV3Task(
		job,
		&datapb.SortSpec{},
		&schemapb.CollectionSchema{},
		&schemapb.CollectionSchema{},
		&datapb.WriterSpec{},
		10, 100,
		v3ImportTaskSpec{channel: "v0", partitionID: 4},
		false,
	)
	require.Error(t, err)
}

func TestSummarizeReshardResultsRejectsFailedTask(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newReshardTask(&datapb.ReshardTask{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_Failed, Reason: "reader failed",
	}, importMeta, nil, nil)
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()

	completed, _, err := checker.summarizeReshardResults(job)
	require.False(t, completed)
	require.Error(t, err)
}

func TestBuildV3SegmentPlans(t *testing.T) {
	fragments := []v3PlanningFragment{
		{sourceID: 1, channelIndex: 0, partitionIndex: 0, seq: 1, path: "f1", rows: 1, bytes: 4},
		{sourceID: 1, channelIndex: 0, partitionIndex: 0, seq: 2, path: "f2", rows: 100, bytes: 4},
		{sourceID: 1, channelIndex: 0, partitionIndex: 0, seq: 3, path: "f3", rows: 100, bytes: 4},
		{sourceID: 1, channelIndex: 1, partitionIndex: 0, seq: 4, path: "f4", rows: 7, bytes: 1},
	}
	plans := buildV3SegmentPlans(fragments, []string{"v0", "v1"}, []int64{10}, 10)
	require.Len(t, plans, 3)
	require.Equal(t, "v0", plans[0].channel)
	require.Equal(t, []string{"f1", "f2"}, []string{plans[0].fragments[0].GetPath(), plans[0].fragments[1].GetPath()})
	require.Equal(t, int64(101), plans[0].rows)
	require.Equal(t, "v0", plans[1].channel)
	require.Equal(t, []string{"f3"}, []string{plans[1].fragments[0].GetPath()})
	require.Equal(t, int64(100), plans[1].rows)
	require.Equal(t, "v1", plans[2].channel)
	require.Equal(t, int64(7), plans[2].rows)
}

func TestMissingV3ImportTaskSpecs(t *testing.T) {
	fragments := []v3PlanningFragment{
		{sourceID: 1, channelIndex: 0, partitionIndex: 0, seq: 1, path: "f1", rows: 1, bytes: 4},
		{sourceID: 1, channelIndex: 0, partitionIndex: 0, seq: 2, path: "f2", rows: 100, bytes: 4},
		{sourceID: 1, channelIndex: 0, partitionIndex: 0, seq: 3, path: "f3", rows: 100, bytes: 4},
	}
	existing := []*datapb.ImportTaskPlan{{
		Fragments: []*datapb.FragmentRef{{Path: "f1", RowCount: 1}},
	}}
	missing, err := missingV3ImportTaskSpecs(fragments, existing, []string{"v0"}, []int64{10}, 10)
	require.NoError(t, err)
	require.Len(t, missing, 1)
	require.Equal(t, []string{"f2", "f3"}, []string{missing[0].fragments[0].GetPath(), missing[0].fragments[1].GetPath()})
	require.Equal(t, int64(200), missing[0].rows)
}

func TestIsTerminalImportV3Err(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"plain non-milvus error", errors.New("plain etcd/io"), false},
		{"ErrIoFailed is terminal", merr.WrapErrIoFailed("x", errors.New("io failed")), true},
		{"ErrServiceUnavailable is transient", merr.WrapErrServiceUnavailable("transient storage outage"), false},
		{"ErrIoUnexpectEOF is transient", merr.WrapErrIoUnexpectEOF("x", errors.New("unexpected eof")), false},
		{"ErrIoTooManyRequests is transient", merr.WrapErrIoTooManyRequests("x", errors.New("too many requests")), false},
		{"ErrIoKeyNotFound is terminal", merr.WrapErrIoKeyNotFound("x", "not found"), true},
		{"ErrImportFailed is terminal", merr.WrapErrImportFailedMsg("bad file"), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.want, isTerminalImportV3Err(c.err))
		})
	}
}

func TestImportCheckerV3CheckGCIgnoresNonTerminalJob(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, State: internalpb.ImportJobState_Importing}}

	// No catalog interaction: checkGC returns before touching tasks or objects.
	checker.checkGC(job)
}

func TestImportCheckerV3QuiesceDropsBoundTask(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	cluster := session.NewMockCluster(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta, cluster: cluster}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, State: internalpb.ImportJobState_Failed}}
	task := newReshardTask(&datapb.ReshardTask{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_InProgress, NodeId: 5,
	}, importMeta, nil, nil)

	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	// The GC loop re-issues a version-aware best-effort Drop every tick so a
	// transient RPC failure or a lost node cannot pin the job forever.
	cluster.EXPECT().DropReshard(int64(5), mock.Anything).Return(nil).Once()
	importMeta.EXPECT().UpdateTask(mock.Anything, int64(10), mock.Anything).Return(nil).Once()

	require.False(t, checker.quiesceImportJob(job, mlog.With(mlog.FieldJobID(job.GetJobID()))))
}

func TestImportCheckerV3QuiesceRemovesFailedTaskAndDropsSegment(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	segment, err := addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	require.Equal(t, commonpb.SegmentState_Importing, segment.GetState())
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: meta, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, State: internalpb.ImportJobState_Failed}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_Completed, NodeId: NullNodeID, SegmentId: 100,
	}, importMeta, meta, nil)

	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	importMeta.EXPECT().RemoveTask(mock.Anything, int64(10)).Return(nil).Once()

	require.True(t, checker.quiesceImportJob(job, mlog.With(mlog.FieldJobID(job.GetJobID()))))
	require.Equal(t, commonpb.SegmentState_Dropped, meta.GetSegment(ctx, 100).GetState())
}

func TestImportCheckerV3CheckGCWaitsForCleanupTs(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, State: internalpb.ImportJobState_Failed, CleanupTs: tsoutil.ComposeTSByTime(time.Now().Add(time.Hour)),
	}}

	// Quiesce is done (no tasks), but the retention window has not elapsed, so
	// deleteImportJob must not run. RemoveJob is un-mocked on purpose: a call
	// would fail the test.
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything).Return(nil).Once()
	checker.checkGC(job)
}

func TestImportCheckerV3CheckGCDeletesPastCleanup(t *testing.T) {
	ctx := context.Background()
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: &meta{chunkManager: cm}, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, State: internalpb.ImportJobState_Failed, AutoCommit: true,
		CleanupTs: tsoutil.ComposeTSByTime(time.Now().Add(-time.Hour)),
	}}

	planPath := path.Join(cm.RootPath(), metautil.BuildImportV3JobPath(1), "plans", "reshard", "10", "plan.pb")
	require.NoError(t, cm.Write(ctx, planPath, []byte("plan")))
	// Quiesce sees no tasks; delete runs RemoveWithPrefix then RemoveJob.
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything).Return(nil).Twice()
	importMeta.EXPECT().RemoveJob(mock.Anything, int64(1)).Return(nil).Once()

	checker.checkGC(job)
	exist, err := cm.Exist(ctx, planPath)
	require.NoError(t, err)
	require.False(t, exist) // prefix removed
}

func TestImportCheckerV3RollbackGateBlocksDelete(t *testing.T) {
	ctx := context.Background()
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: &meta{chunkManager: cm}, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, State: internalpb.ImportJobState_Failed, AutoCommit: false,
		CleanupTs: tsoutil.ComposeTSByTime(time.Now().Add(-time.Hour)),
	}}
	rollbackCalled := false
	checker.hooks.rollbackImport = func(ctx context.Context, j ImportJob) error {
		rollbackCalled = true
		return merr.WrapErrServiceUnavailable("transient")
	}
	checker.hooks.isReplicatingCluster = func(ctx context.Context) (bool, error) { return true, nil }

	planPath := path.Join(cm.RootPath(), metautil.BuildImportV3JobPath(1), "plans", "reshard", "10", "plan.pb")
	require.NoError(t, cm.Write(ctx, planPath, []byte("plan")))

	// Rollback broadcast fails transiently: keep the job, no delete this tick.
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything).Return(nil).Once()
	checker.checkGC(job)
	require.True(t, rollbackCalled)

	// Next tick rollback succeeds; delete proceeds.
	checker.hooks.rollbackImport = func(ctx context.Context, j ImportJob) error { return nil }
	importMeta.EXPECT().GetTaskByJob(mock.Anything, mock.Anything).Return(nil).Twice()
	importMeta.EXPECT().RemoveJob(mock.Anything, int64(1)).Return(nil).Once()
	checker.checkGC(job)
}
