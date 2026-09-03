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

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/session"
	"github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
)

// TestImportTaskV3PrepareRetryRepointsBeforeDroppingOld pins the crash-safe
// ordering of prepareRetry: the task is repointed to the fresh segment ID
// BEFORE the superseded run is cleaned up, and retry no longer registers the
// new segment (it is built only at acceptance).
func TestImportTaskV3PrepareRetryRepointsBeforeDroppingOld(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	importMeta := NewMockImportMeta(t)
	cluster := session.NewMockCluster(t)
	alloc := allocator.NewMockAllocator(t)

	// An accepted run leaves a registered IsImporting segment behind.
	oldSeg, err := addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	require.Equal(t, commonpb.SegmentState_Importing, oldSeg.GetState())

	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, CollectionID: 2, State: internalpb.ImportJobState_Importing, DataTs: 1, Schema: retryTestSchema,
	}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, CollectionId: 2, PartitionId: 3, Vchannel: "v0",
		State: datapb.ImportTaskStateV2_InProgress,
		RunId: 1, NodeId: 5, SegmentId: 100, LogRange: &datapb.IDRange{Begin: 1000, End: 2000},
	}, importMeta, meta, alloc)

	importMeta.EXPECT().GetJob(mock.Anything, int64(1)).Return(job).Once()
	cluster.EXPECT().DropImportV3(int64(5), mock.Anything).Return(nil).Once()
	alloc.EXPECT().AllocID(mock.Anything).Return(int64(200), nil).Once()
	alloc.EXPECT().AllocN(mock.Anything).Return(int64(5000), int64(6000), nil).Once()

	var repointBeforeDrop bool
	importMeta.EXPECT().UpdateTask(mock.Anything, int64(10),
		mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything,
	).RunAndReturn(func(_ context.Context, _ int64, actions ...UpdateAction) error {
		// At repoint time the old segment must still be Importing: the drop only
		// happens after the task points at the new segment.
		require.Equal(t, commonpb.SegmentState_Importing, meta.GetSegment(ctx, 100).GetState())
		repointBeforeDrop = true
		// Simulate the real UpdateTask: apply the actions to the task in place so
		// the post-repoint assertions below observe the new run/segment/log range.
		clone := task.Clone()
		for _, action := range actions {
			action(clone)
		}
		task.task.Store(clone.(*importTaskV3).task.Load())
		return nil
	}).Once()

	task.prepareRetry(cluster)

	require.True(t, repointBeforeDrop)
	require.Equal(t, commonpb.SegmentState_Dropped, meta.GetSegment(ctx, 100).GetState())
	// The retry never registers the new segment; it is created at acceptance.
	require.Nil(t, meta.GetSegment(ctx, 200))
	require.Equal(t, int64(2), task.task.Load().GetRunId())
	require.Equal(t, int64(200), task.task.Load().GetSegmentId())
	require.Equal(t, int64(5000), task.task.Load().GetLogRange().GetBegin())
	require.Equal(t, int64(6000), task.task.Load().GetLogRange().GetEnd())
	require.Equal(t, datapb.ImportTaskStateV2_Pending, task.task.Load().GetState())
	require.Equal(t, int64(NullNodeID), task.task.Load().GetNodeId())
}

// TestImportTaskV3PrepareRetryUnacceptedRunRemovesBasePath pins that retrying a
// run that never reached acceptance (no registered segment) removes its
// unregistered output prefix directly instead of leaking it until the orphan
// scan ages it out.
func TestImportTaskV3PrepareRetryUnacceptedRunRemovesBasePath(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	meta.chunkManager = storage.NewLocalChunkManager(objectstorage.RootPath(t.TempDir()))
	importMeta := NewMockImportMeta(t)
	cluster := session.NewMockCluster(t)
	alloc := allocator.NewMockAllocator(t)

	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, CollectionID: 2, State: internalpb.ImportJobState_Importing, DataTs: 1, Schema: retryTestSchema,
	}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, CollectionId: 2, PartitionId: 3, Vchannel: "v0",
		State: datapb.ImportTaskStateV2_InProgress,
		RunId: 1, NodeId: NullNodeID, SegmentId: 100, LogRange: &datapb.IDRange{Begin: 1000, End: 2000},
	}, importMeta, meta, alloc)

	// Seed a file under the superseded run's base path.
	basePath := path.Join(meta.chunkManager.RootPath(), common.SegmentInsertLogPath, metautil.JoinIDPath(2, 3, 100))
	require.NoError(t, meta.chunkManager.Write(ctx, path.Join(basePath, "_data", "f.parquet"), []byte("x")))

	importMeta.EXPECT().GetJob(mock.Anything, int64(1)).Return(job).Once()
	alloc.EXPECT().AllocID(mock.Anything).Return(int64(200), nil).Once()
	alloc.EXPECT().AllocN(mock.Anything).Return(int64(5000), int64(6000), nil).Once()
	importMeta.EXPECT().UpdateTask(mock.Anything, int64(10),
		mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything,
	).Return(nil).Once()

	task.prepareRetry(cluster)

	exist, err := meta.chunkManager.Exist(ctx, path.Join(basePath, "_data", "f.parquet"))
	require.NoError(t, err)
	require.False(t, exist)
	require.Nil(t, meta.GetSegment(ctx, 100))
	require.Nil(t, meta.GetSegment(ctx, 200))
}

// TestImportTaskV3PrepareRetryFailsJobOnTaskUpdateFailure pins that a failed
// repoint fails the job and leaves the old segment untouched; nothing is
// registered before the repoint, so there is no new segment to roll back.
func TestImportTaskV3PrepareRetryFailsJobOnTaskUpdateFailure(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	importMeta := NewMockImportMeta(t)
	cluster := session.NewMockCluster(t)
	alloc := allocator.NewMockAllocator(t)

	_, err = addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)

	job := &importJob{ImportJob: &datapb.ImportJob{
		JobID: 1, CollectionID: 2, State: internalpb.ImportJobState_Importing, DataTs: 1, Schema: retryTestSchema,
	}}
	task := newImportTaskV3(&datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, CollectionId: 2, PartitionId: 3, Vchannel: "v0",
		State: datapb.ImportTaskStateV2_InProgress,
		RunId: 1, NodeId: NullNodeID, SegmentId: 100, LogRange: &datapb.IDRange{Begin: 1000, End: 2000},
	}, importMeta, meta, alloc)

	importMeta.EXPECT().GetJob(mock.Anything, int64(1)).Return(job).Once()
	alloc.EXPECT().AllocID(mock.Anything).Return(int64(200), nil).Once()
	alloc.EXPECT().AllocN(mock.Anything).Return(int64(5000), int64(6000), nil).Once()
	importMeta.EXPECT().UpdateTask(mock.Anything, int64(10),
		mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything,
	).Return(errRetryUpdate).Once()
	// t.fail marks the task and job Failed.
	importMeta.EXPECT().UpdateTask(mock.Anything, int64(10), mock.Anything, mock.Anything).Return(nil).Once()
	importMeta.EXPECT().UpdateJob(mock.Anything, int64(1), mock.Anything, mock.Anything).Return(nil).Once()

	task.prepareRetry(cluster)

	// The old segment is untouched and the new one was never registered.
	require.Equal(t, commonpb.SegmentState_Importing, meta.GetSegment(ctx, 100).GetState())
	require.Nil(t, meta.GetSegment(ctx, 200))
}

var errRetryUpdate = errors.New("update task failed")

// retryTestSchema gives prepareRetry's LogRange-width re-derivation
// (buildImportV3WriterSpec) a valid target schema; production jobs always
// carry one, but the retry tests construct minimal jobs.
var retryTestSchema = &schemapb.CollectionSchema{
	Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
	},
}

func TestReconcileOrphanImportSegments(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)

	catalog := mocks.NewDataCoordCatalog(t)
	catalog.EXPECT().ListReshardTasks(mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().ListImportTasksV3(mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().ListImportJobs(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListPreImportTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListImportTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().SaveImportTaskV3(mock.Anything, mock.Anything).Return(nil).Maybe()
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil).Maybe()

	importMeta, err := NewImportMeta(ctx, catalog, nil, meta)
	require.NoError(t, err)

	// Referenced by a V3 task -> kept.
	_, err = addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	// Referenced by a V2 task -> kept.
	_, err = addImportSegment(ctx, meta, 101, 1, 11, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	// Orphan Importing segment (no task references it) -> dropped.
	_, err = addImportSegment(ctx, meta, 102, 999, 999, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	// A Dropped segment stays Dropped and is never re-touched.
	_, err = addImportSegment(ctx, meta, 103, 998, 998, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)
	require.NoError(t, meta.UpdateSegmentsInfo(ctx, UpdateStatusOperator(103, commonpb.SegmentState_Dropped)))

	v3task := newImportTaskV3(&datapb.ImportTaskV3{JobId: 1, TaskId: 10, CollectionId: 2, SegmentId: 100}, importMeta, meta, nil)
	require.NoError(t, importMeta.AddTask(ctx, v3task))
	v2taskProto := &datapb.ImportTaskV2{JobID: 1, TaskID: 11, CollectionID: 2, SegmentIDs: []int64{101}}
	v2task := &importTask{meta: meta, importMeta: importMeta}
	v2task.task.Store(v2taskProto)
	require.NoError(t, importMeta.AddTask(ctx, v2task))

	inspector := &importInspector{ctx: ctx, meta: meta, importMeta: importMeta}
	inspector.reconcileOrphanImportSegments()

	require.Equal(t, commonpb.SegmentState_Importing, meta.GetSegment(ctx, 100).GetState())
	require.Equal(t, commonpb.SegmentState_Importing, meta.GetSegment(ctx, 101).GetState())
	require.Equal(t, commonpb.SegmentState_Dropped, meta.GetSegment(ctx, 102).GetState())
	require.Equal(t, commonpb.SegmentState_Dropped, meta.GetSegment(ctx, 103).GetState())
}
