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
	"path"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
)

func TestCleanupPreparingV3ImportTasksIsIdempotent(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_None, NodeId: NullNodeID,
	}, importMeta, nil)

	first := importMeta.EXPECT().GetTaskBy(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	second := importMeta.EXPECT().GetTaskBy(mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
	mock.InOrder(first, second)
	importMeta.EXPECT().RemoveTask(mock.Anything, int64(10)).Return(nil).Once()

	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
}

func TestCalculateV3TaskSlots(t *testing.T) {
	const mib = int64(1024 * 1024)

	require.Equal(t, int64(3), calculateV3ReshardTaskSlot(16*mib, 128*mib, 32*mib, 160*mib))
	require.Equal(t, int64(4), calculateV3ImportTaskSlot(16*mib, 32*mib, 160*mib, 16))
	require.Equal(t, int64(1), calculateV3Slots(1, 160*mib))
	require.Equal(t, int64(2), calculateV3Slots(160*mib+1, 160*mib))
}

func TestCleanupPreparingV3ImportTasksKeepsReadyTask(t *testing.T) {
	ctx := context.Background()
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_Pending, NodeId: NullNodeID,
	}, importMeta, nil)
	importMeta.EXPECT().GetTaskBy(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()

	require.NoError(t, checker.cleanupPreparingV3ImportTasks(job))
}

func TestCleanupPreparingV3ImportTaskDropsOwnedSegment(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	segment, err := addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: meta, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, State: datapb.ImportTaskStateV2_None, NodeId: NullNodeID, Segments: []int64{100},
	}, importMeta, meta)
	importMeta.EXPECT().GetTaskBy(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
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
	alloc.EXPECT().AllocN(int64(1)).Return(int64(100), int64(101), nil).Once()
	alloc.EXPECT().AllocN(int64(1)).Return(int64(200), int64(201), nil).Once()
	checker := &importCheckerV3{ctx: ctx, meta: meta, importMeta: importMeta, alloc: alloc}
	job := &importJob{ImportJob: &datapb.ImportJob{JobID: 1, CollectionID: 2, DataTs: 1}}

	var task *importTaskV3
	add := importMeta.EXPECT().AddTask(mock.Anything, mock.Anything).Run(func(_ context.Context, added ImportTask) {
		task = added.(*importTaskV3)
		require.Equal(t, datapb.ImportTaskStateV2_None, task.GetState())
		require.Equal(t, int64(4), task.GetTaskSlot())
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
		10,
		v3ImportTaskSpec{channel: "v0", segments: []*datapb.SegmentPlan{{Vchannel: "v0", PartitionId: 4, Rows: 5}}},
	)
	require.NoError(t, err)
	require.Equal(t, datapb.ImportTaskStateV2_Pending, task.GetState())
	segment := meta.GetSegment(ctx, 100)
	require.True(t, segment.GetIsImporting())
	require.False(t, segment.GetIsInvisible())
	require.Equal(t, int32(3), segment.GetSchemaVersion())
}

func TestCreateV3ReshardTasksKeepsExistingAndAddsMissingSources(t *testing.T) {
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
	}}
	existingPlan := &datapb.ReshardTaskPlan{
		CollectionId: 2,
		Sources:      []*datapb.SourceFileSpec{{File: job.GetFiles()[0]}},
	}
	require.NoError(t, writeImportV3Proto(ctx, cm, metautil.BuildImportV3ReshardPlanPath(1, 10), existingPlan))
	existing := newReshardTask(&datapb.ReshardTask{
		JobId: 1, TaskId: 10, CollectionId: 2, State: datapb.ImportTaskStateV2_InProgress,
	}, importMeta, checker.meta, alloc)
	importMeta.EXPECT().GetTaskBy(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{existing}).Once()
	importMeta.EXPECT().AddTask(mock.Anything, mock.Anything).Run(func(_ context.Context, added ImportTask) {
		plan := &datapb.ReshardTaskPlan{}
		p := added.(*reshardTask).task.Load()
		require.Equal(t, int64(3), p.GetSlot())
		require.NoError(t, loadImportV3Proto(ctx, cm, metautil.BuildImportV3ReshardPlanPath(1, p.GetTaskId()), plan))
		require.Len(t, plan.GetSources(), 1)
		require.Equal(t, int64(2), plan.GetSources()[0].GetFile().GetId())
	}).Return(nil).Once()
	importMeta.EXPECT().UpdateJob(mock.Anything, int64(1), mock.Anything).Return(nil).Once()

	require.NoError(t, checker.createV3ReshardTasks(job))
}

func TestCreateV3ReshardTasksRetriesJobStateWhenSourcesCovered(t *testing.T) {
	ctx := context.Background()
	cm := storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	importMeta := NewMockImportMeta(t)
	checker := &importCheckerV3{ctx: ctx, meta: &meta{chunkManager: cm}, importMeta: importMeta}
	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, CollectionID: 2,
		Schema: &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, DataType: schemapb.DataType_Int64, IsPrimaryKey: true}}},
		Files:  []*internalpb.ImportFile{{Id: 1, Paths: []string{"existing.json"}}},
	}}
	plan := &datapb.ReshardTaskPlan{
		CollectionId: 2,
		Sources:      []*datapb.SourceFileSpec{{File: job.GetFiles()[0]}},
	}
	require.NoError(t, writeImportV3Proto(ctx, cm, metautil.BuildImportV3ReshardPlanPath(1, 10), plan))
	task := newReshardTask(&datapb.ReshardTask{JobId: 1, TaskId: 10, CollectionId: 2, State: datapb.ImportTaskStateV2_Completed}, importMeta, checker.meta)
	importMeta.EXPECT().GetTaskBy(mock.Anything, mock.Anything, mock.Anything).Return([]ImportTask{task}).Once()
	importMeta.EXPECT().UpdateJob(mock.Anything, int64(1), mock.Anything).Return(nil).Once()

	require.NoError(t, checker.createV3ReshardTasks(job))
}

func TestSameImportV3FragmentCoverage(t *testing.T) {
	f1 := &datapb.FragmentRef{Path: "f1", RowCount: 1}
	f2 := &datapb.FragmentRef{Path: "f2", RowCount: 2}
	require.True(t, sameImportV3FragmentCoverage(
		[]*datapb.SegmentPlan{{Fragments: []*datapb.FragmentRef{f1}}, {Fragments: []*datapb.FragmentRef{f2}}},
		[]*datapb.SegmentPlan{{Fragments: []*datapb.FragmentRef{f1, f2}}},
	))
	require.False(t, sameImportV3FragmentCoverage(
		[]*datapb.SegmentPlan{{Fragments: []*datapb.FragmentRef{f1}}},
		[]*datapb.SegmentPlan{{Fragments: []*datapb.FragmentRef{f1, f2}}},
	))
}
