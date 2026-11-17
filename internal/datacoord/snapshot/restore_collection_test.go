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

package snapshot

import (
	"context"
	"fmt"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/copysegment"
	dcindex "github.com/milvus-io/milvus/internal/datacoord/index"
	"github.com/milvus-io/milvus/internal/datacoord/meta"
	"github.com/milvus-io/milvus/internal/metacache"
	kvdatacoord "github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/internal/rootcoord"
	snapshotstorage "github.com/milvus-io/milvus/internal/snapshotio/storage"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/broadcast"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/messagespb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestRestoreCollectionUsesOneBroadcast(t *testing.T) {
	mockey.PatchConvey("one durable creation recipe", t, func() {
		body := &messagespb.RestoreSnapshotMessageBody{
			CollectionHeader: &message.CreateCollectionMessageHeader{CollectionId: 100, DbId: 1, PartitionIds: []int64{201, 202}},
			Collection: &message.CreateCollectionRequest{
				CollectionID: 100, PartitionIDs: []int64{201, 202}, PartitionNames: []string{"_default", "p1"},
				CollectionSchema: &schemapb.CollectionSchema{Name: "target"}, VirtualChannelNames: []string{"v1", "v2"},
			},
		}
		cleaned := false
		mockey.Mock((*rootcoord.Core).CheckRestoreTargetAbsent).Return(merr.WrapErrCollectionNotFound("target")).Build()
		mockey.Mock((*rootcoord.Core).PrepareRestoreCollection).To(func(_ *rootcoord.Core, _ context.Context, request *milvuspb.CreateCollectionRequest, partitions []string) (*messagespb.RestoreSnapshotMessageBody, func(), error) {
			require.Equal(t, "target", request.GetCollectionName())
			require.ElementsMatch(t, []string{"_default", "p1"}, partitions)
			return body, func() { cleaned = true }, nil
		}).Build()
		mockey.Mock((*restoreAllocatorTarget).AllocID).Return(int64(77), nil).Build()
		capture := &meta.CaptureBroadcastAPI{}
		mockey.Mock(broadcast.StartBroadcastWithResourceKeys).Return(capture, nil).Build()
		calls := 0
		mockey.Mock((*meta.CaptureBroadcastAPI).Broadcast).To(func(_ *meta.CaptureBroadcastAPI, _ context.Context, msg message.BroadcastMutableMessage) (*types.BroadcastAppendResult, error) {
			calls++
			require.Equal(t, message.MessageTypeRestoreSnapshot, msg.MessageType())
			restore := message.MustAsBroadcastRestoreSnapshotMessageV2(msg)
			require.Equal(t, int64(100), restore.Header().GetCollectionId())
			require.Equal(t, int64(77), restore.Header().GetJobId())
			require.True(t, proto.Equal(body, restore.MustBody()))
			require.ElementsMatch(t, []string{"v1", "v2"}, msg.BroadcastHeader().VChannels)
			return &types.BroadcastAppendResult{BroadcastID: 1}, nil
		}).Build()
		legacy := mockey.Mock((*Manager).RestoreCollection).Return(int64(0), merr.ErrServiceInternal).Build()
		sm := &Manager{CollectionRestorer: &rootcoord.Core{}, Allocator: &restoreAllocatorTarget{}}
		job, err := sm.finishRestoreSnapshot(t.Context(), nil, &snapshotstorage.SnapshotData{
			Collection: &datapb.CollectionDescription{Schema: &schemapb.CollectionSchema{Name: "source"}, Partitions: map[string]int64{"_default": 1, "p1": 2}},
		}, "snapshot", 10, "target", "default", 5, false, "", "", nil, nil, nil, nil)
		require.NoError(t, err)
		require.Equal(t, int64(77), job)
		require.Equal(t, 1, calls)
		require.Zero(t, legacy.Times())
		require.False(t, cleaned, "durable broadcast owns creation resources")
	})
}

func TestRestoreCallbackPersistsJobBeforeSegments(t *testing.T) {
	mockey.PatchConvey("callback owns segment allocation; retries reuse durable Job", t, func() {
		ctx := t.Context()
		catalog := kvdatacoord.NewCatalog(meta.NewMetaMemoryKV(), "", "")
		store := metacache.NewMetaStore(nil)
		m := &meta.Meta{MetaStore: store, Catalog: catalog, Segments: meta.NewSegmentsInfo(store)}
		copies, err := copysegment.NewCopySegmentMeta(ctx, catalog, m, nil, nil, nil)
		require.NoError(t, err)
		allocated := mockey.Mock((*restoreAllocatorTarget).AllocN).Return(int64(2001), int64(2002), nil).Build()
		mockey.Mock((*meta.MockHandler).GetCollection).Return(&meta.CollectionInfo{}, nil).Build()
		mockey.Mock((*meta.Meta).UpdateChannelCheckpoint).Return(nil).Build()
		sm := &Manager{meta: m, copySegmentMeta: copies, Allocator: &restoreAllocatorTarget{}, handler: meta.NewMockHandler()}
		snapshot := &snapshotstorage.SnapshotData{
			SnapshotInfo: &datapb.SnapshotInfo{Name: "snapshot", CollectionId: 1},
			Segments:     []*datapb.SegmentDescription{{SegmentId: 1, PartitionId: 1, ChannelName: "source", SegmentLevel: datapb.SegmentLevel_L1}},
		}
		require.NoError(t, sm.createRestoreJob(ctx, 100, map[string]string{"source": "target"}, map[int64]int64{1: 10}, snapshot, 77, 0, false, "", "", "", &restorePublication{starts: map[string]uint64{"target": 123}, createsCollection: true}))
		require.Nil(t, m.GetSegment(ctx, 2001), "registration must follow durable Job creation")
		job := copies.GetJob(ctx, 77)
		require.EqualValues(t, 2001, job.GetIdMappings()[0].GetTargetSegmentId())
		// WAL callback replay after restart resolves the original JobID first.
		sm.copySegmentMeta, err = copysegment.NewCopySegmentMeta(ctx, catalog, m, nil, nil, nil)
		require.NoError(t, err)
		id, err := sm.restoreData(ctx, 1, "snapshot", "", 100, 77, 0, false, "", "")
		require.NoError(t, err)
		require.EqualValues(t, 77, id)
		require.Equal(t, 1, allocated.Times())
	})
}

func TestRestoreCollectionEnforcesMultiIndexGateAndCap(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Name: "source",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, Name: "scalar", DataType: schemapb.DataType_Int64},
			{FieldID: 102, Name: "vec", DataType: schemapb.DataType_FloatVector},
		},
	}
	twoScalar := []*indexpb.IndexInfo{
		{IndexName: "idx_inv", FieldID: 101, IndexParams: []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "INVERTED"}}},
		{IndexName: "idx_sort", FieldID: 101, IndexParams: []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "STL_SORT"}}},
	}

	t.Run("production broadcastRestoreCollection refuses when switch is off", func(t *testing.T) {
		setMultiScalarIndexPerField(t, "false")
		mockey.PatchConvey("switch off", t, func() {
			capture := &meta.CaptureBroadcastAPI{}
			mockey.Mock(broadcast.StartBroadcastWithResourceKeys).Return(capture, nil).Build()
			mockey.Mock((*rootcoord.Core).CheckRestoreTargetAbsent).Return(merr.WrapErrCollectionNotFound("target")).Build()
			prepare := mockey.Mock((*rootcoord.Core).PrepareRestoreCollection).Return(nil, func() {}, nil).Build()

			sm := &Manager{CollectionRestorer: &rootcoord.Core{}}
			_, err := sm.finishRestoreSnapshot(t.Context(), nil, &snapshotstorage.SnapshotData{
				Collection: &datapb.CollectionDescription{Schema: schema},
				Indexes:    twoScalar,
			}, "snapshot", 10, "target", "default", 5, false, "", "", nil, nil, nil, nil)
			require.ErrorIs(t, err, merr.ErrParameterInvalid)
			require.Contains(t, err.Error(), "creating multiple indexes on same field is not supported")
			require.Zero(t, prepare.Times())
		})
	})

	t.Run("production broadcastRestoreCollection refuses when software upgrade gate is pending (auto)", func(t *testing.T) {
		setMultiScalarIndexPerField(t, "auto")
		mockey.PatchConvey("auto gate pending", t, func() {
			capture := &meta.CaptureBroadcastAPI{}
			mockey.Mock(broadcast.StartBroadcastWithResourceKeys).Return(capture, nil).Build()
			mockey.Mock((*rootcoord.Core).CheckRestoreTargetAbsent).Return(merr.WrapErrCollectionNotFound("target")).Build()
			prepare := mockey.Mock((*rootcoord.Core).PrepareRestoreCollection).Return(nil, func() {}, nil).Build()

			sm := &Manager{CollectionRestorer: &rootcoord.Core{}}
			_, err := sm.finishRestoreSnapshot(t.Context(), nil, &snapshotstorage.SnapshotData{
				Collection: &datapb.CollectionDescription{Schema: schema},
				Indexes:    twoScalar,
			}, "snapshot", 10, "target", "default", 5, false, "", "", nil, nil, nil, nil)
			require.ErrorIs(t, err, merr.ErrServiceNotReady)
			require.Contains(t, err.Error(), "cluster software upgrade gate")
			require.Zero(t, prepare.Times())
		})
	})

	t.Run("production broadcastRestoreCollection refuses exceeding per-field scalar index cap", func(t *testing.T) {
		setMultiScalarIndexPerField(t, "true")
		tooMany := make([]*indexpb.IndexInfo, 0, dcindex.MaxScalarIndexesPerField+1)
		for i := 0; i <= dcindex.MaxScalarIndexesPerField; i++ {
			tooMany = append(tooMany, &indexpb.IndexInfo{
				IndexName:   fmt.Sprintf("idx_%d", i),
				FieldID:     101,
				IndexParams: []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "INVERTED"}},
			})
		}
		mockey.PatchConvey("cap exceeded", t, func() {
			capture := &meta.CaptureBroadcastAPI{}
			mockey.Mock(broadcast.StartBroadcastWithResourceKeys).Return(capture, nil).Build()
			mockey.Mock((*rootcoord.Core).CheckRestoreTargetAbsent).Return(merr.WrapErrCollectionNotFound("target")).Build()
			prepare := mockey.Mock((*rootcoord.Core).PrepareRestoreCollection).Return(nil, func() {}, nil).Build()

			sm := &Manager{CollectionRestorer: &rootcoord.Core{}}
			_, err := sm.finishRestoreSnapshot(t.Context(), nil, &snapshotstorage.SnapshotData{
				Collection: &datapb.CollectionDescription{Schema: schema},
				Indexes:    tooMany,
			}, "snapshot", 10, "target", "default", 5, false, "", "", nil, nil, nil, nil)
			require.ErrorIs(t, err, merr.ErrParameterInvalid)
			require.Contains(t, err.Error(), "the limit is")
			require.Zero(t, prepare.Times())
		})
	})
}
