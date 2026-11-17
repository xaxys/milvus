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

package querycoordv2

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querynodev2/qnview"
	balancerapi "github.com/milvus-io/milvus/internal/views/coord/balancer/api"
	"github.com/milvus-io/milvus/internal/views/coord/balancer/cache"
	"github.com/milvus-io/milvus/internal/views/coord/coordview"
	"github.com/milvus-io/milvus/internal/views/coord/loadmgr"
	qvresource "github.com/milvus-io/milvus/internal/views/coord/resource"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type memLoadInfoCatalog struct {
	records []*viewpb.CollectionLoadConfig
}

func (c *memLoadInfoCatalog) List(context.Context) ([]*viewpb.CollectionLoadConfig, error) {
	return c.records, nil
}

func (c *memLoadInfoCatalog) Save(_ context.Context, info *viewpb.CollectionLoadConfig) error {
	c.records = append(c.records, info)
	return nil
}

func (c *memLoadInfoCatalog) RemoveVersions(context.Context, int64, []uint64) error { return nil }
func (c *memLoadInfoCatalog) RemoveCollection(context.Context, int64) error         { return nil }

// indexLoadStatusFixture wires the QueryView runtime surface GetIndexLoadStatus
// reads: LoadConfigStore for desired replicas and load requirements, and
// ShardViewRegistry snapshots for serving placements and per-segment index ids.
type indexLoadStatusFixture struct {
	server *Server
	stats  map[qviews.ShardID]*coordview.ShardStats
	ver    uint64
}

func newIndexLoadStatusFixture(t *testing.T, collID int64, replicaIDs []int64, requiredIndexes ...int64) *indexLoadStatusFixture {
	t.Helper()
	ctx := context.Background()

	var nextVer atomic.Uint64
	indexDefs := make([]*indexpb.IndexInfo, 0, len(requiredIndexes))
	for _, idxID := range requiredIndexes {
		indexDefs = append(indexDefs, &indexpb.IndexInfo{CollectionID: collID, FieldID: 101, IndexID: idxID})
	}
	store, err := loadmgr.RecoverLoadConfigStore(ctx, loadmgr.LoadConfigStoreDependencies{
		Catalog:         &memLoadInfoCatalog{},
		AllocateVersion: func(context.Context) (uint64, error) { return nextVer.Add(1), nil },
		Indexes:         func(int64, ...int64) []*indexpb.IndexInfo { return indexDefs },
	})
	require.NoError(t, err)

	replicas := make([]*loadmgr.ReplicaAssignment, 0, len(replicaIDs))
	for _, repID := range replicaIDs {
		replicas = append(replicas, &loadmgr.ReplicaAssignment{ReplicaID: repID, ResourceGroup: "rg"})
	}
	var requiredFields []*viewpb.LoadFieldConfig
	if len(requiredIndexes) > 0 {
		requiredFields = []*viewpb.LoadFieldConfig{{FieldId: 101, IndexIds: requiredIndexes}}
	}
	require.NoError(t, store.Put(ctx, &loadmgr.LoadConfig{
		CollectionID:   collID,
		PartitionIDs:   []int64{10},
		Replicas:       replicas,
		RequiredFields: requiredFields,
	}))

	stats := make(map[qviews.ShardID]*coordview.ShardStats)
	registry := &coordview.ShardViewRegistry{}
	patch := mockey.Mock((*coordview.ShardViewRegistry).SnapshotForCollection).To(
		func(_ *coordview.ShardViewRegistry, id int64) *coordview.ShardViewSnapshot {
			if id != collID {
				return coordview.NewShardViewSnapshot(1, nil)
			}
			return coordview.NewShardViewSnapshot(1, stats)
		},
	).Build()
	t.Cleanup(func() { patch.UnPatch() })

	server := &Server{
		qviewsRuntime: &qviewsRuntime{
			loadConfigStore:   store,
			shardViewRegistry: registry,
			resources: &qvresource.Resources{
				VChannels: func(context.Context, int64) ([]string, error) {
					seen := typeutil.NewSet[string]()
					for shard := range stats {
						if shard.VChannel != "" {
							seen.Insert(shard.VChannel)
						}
					}
					return seen.Collect(), nil
				},
			},
		},
		broker: meta.NewMockBroker(t),
	}
	server.status.Store(int32(commonpb.StateCode_Healthy))

	return &indexLoadStatusFixture{
		server: server,
		stats:  stats,
		ver:    store.GetConfigVersion(collID),
	}
}

// putServingSegment registers one served segment on a replica's shard view,
// reporting the given index ids as loaded on that segment.
func (f *indexLoadStatusFixture) putServingSegment(replicaID int64, channel string, nodeID, segID int64, indexIDs ...int64) {
	shard := qviews.ShardID{ReplicaID: replicaID, VChannel: channel}
	stat := f.stats[shard]
	if stat == nil {
		stat = &coordview.ShardStats{
			UpVersion:         &qviews.QueryViewVersion{QueryVersion: 1},
			UpLoadInfoVersion: f.ver,
			UpPlacement: &coordview.ViewPlacement{
				Assignments:    make(map[int64]int64),
				SegmentIndexes: make(map[int64]map[int64]struct{}),
			},
		}
		f.stats[shard] = stat
	}
	stat.ResidentNodes = append(stat.ResidentNodes, nodeID)
	stat.UpPlacement.Assignments[segID] = nodeID
	loaded := make(map[int64]struct{}, len(indexIDs))
	for _, id := range indexIDs {
		loaded[id] = struct{}{}
	}
	stat.UpPlacement.SegmentIndexes[segID] = loaded
}

func TestServerGetIndexLoadStatus(t *testing.T) {
	const (
		collID    = int64(100)
		fieldID   = int64(101)
		replicaA  = int64(1)
		replicaB  = int64(2)
		nodeID    = int64(10)
		segA      = int64(1000)
		segB      = int64(1001)
		channelV0 = "v0"
		channelV1 = "v1"
	)

	t.Run("every loaded segment carries the index", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 11, 12)
		f.putServingSegment(replicaA, channelV0, nodeID, segA, 11, 12)
		f.putServingSegment(replicaA, channelV0, nodeID, segB, 12)
		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
			}, nil)

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)

		status := resp.GetStatuses()[0]
		assert.Equal(t, int64(12), status.GetIndexID())
		assert.Equal(t, fieldID, status.GetFieldID())
		assert.Equal(t, "scalar_ngram", status.GetIndexName())
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_LOADED, status.GetState())
		assert.Equal(t, int64(1), status.GetExpectedReplicas())
		assert.Equal(t, int64(1), status.GetReadyReplicas())
		assert.Equal(t, int64(2), status.GetExpectedSegments())
		assert.Equal(t, int64(2), status.GetReadySegments())
		assert.Empty(t, status.GetReason())
	})

	t.Run("one loaded segment is missing the index", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 11, 12)
		f.putServingSegment(replicaA, channelV0, nodeID, segA, 12)
		f.putServingSegment(replicaA, channelV0, nodeID, segB, 11)
		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
			}, nil)

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)

		status := resp.GetStatuses()[0]
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, status.GetState())
		assert.Equal(t, int64(2), status.GetExpectedSegments())
		assert.Equal(t, int64(1), status.GetReadySegments())
		// The reason has to be actionable: it says what is missing, not a distribution dump.
		assert.Contains(t, status.GetReason(), "do not carry the index")
	})

	t.Run("no loaded segment carries the index yet", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 11)
		f.putServingSegment(replicaA, channelV0, nodeID, segA, 11)
		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
			}, nil)

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)

		status := resp.GetStatuses()[0]
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, status.GetState())
		assert.Equal(t, int64(0), status.GetReadySegments())
		assert.Contains(t, status.GetReason(), "no loaded segment")
	})

	t.Run("multiple replicas across channels require full channel coverage per replica", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA, replicaB}, 12)
		// Replica A serves both v0 and v1; Replica B only serves v0 so far (even sharing nodeID).
		f.putServingSegment(replicaA, channelV0, nodeID, segA, 12)
		f.putServingSegment(replicaA, channelV1, nodeID, segB, 12)
		f.putServingSegment(replicaB, channelV0, nodeID, segA, 12)
		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
			}, nil).Twice()

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, resp.GetStatuses()[0].GetState())
		assert.Equal(t, int64(2), resp.GetStatuses()[0].GetExpectedReplicas())
		assert.Equal(t, int64(1), resp.GetStatuses()[0].GetReadyReplicas())
		assert.Equal(t, int64(4), resp.GetStatuses()[0].GetExpectedSegments())
		assert.Equal(t, int64(3), resp.GetStatuses()[0].GetReadySegments())

		// Once Replica B also serves v1, both replicas are ready.
		f.putServingSegment(replicaB, channelV1, nodeID+1, segB, 12)
		resp, err = f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_LOADED, resp.GetStatuses()[0].GetState())
		assert.Equal(t, int64(2), resp.GetStatuses()[0].GetReadyReplicas())
		assert.Equal(t, int64(4), resp.GetStatuses()[0].GetReadySegments())
	})

	t.Run("single replica missing an expected vchannel stays partially loaded", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 12)
		f.server.qviewsRuntime.resources = &qvresource.Resources{
			VChannels: func(context.Context, int64) ([]string, error) {
				return []string{channelV0, channelV1}, nil
			},
		}
		// Only v0 is Up; v1 has no Up view yet.
		f.putServingSegment(replicaA, channelV0, nodeID, segA, 12)
		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
			}, nil)

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, resp.GetStatuses()[0].GetState())
		assert.Equal(t, int64(0), resp.GetStatuses()[0].GetReadyReplicas())
	})

	t.Run("loaded empty collection reports LOADED for active index and NOT_LOADED for invisible or unpropagated index", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 12)
		f.server.qviewsRuntime.resources = &qvresource.Resources{
			VChannels: func(context.Context, int64) ([]string, error) {
				return []string{channelV0, channelV1}, nil
			},
		}
		// Only v0 is Up (0 segments): partially loaded.
		f.stats[qviews.ShardID{ReplicaID: replicaA, VChannel: channelV0}] = &coordview.ShardStats{
			UpVersion:         &qviews.QueryViewVersion{QueryVersion: 1},
			UpLoadInfoVersion: f.ver,
			UpPlacement:       &coordview.ViewPlacement{Assignments: map[int64]int64{}},
		}
		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "active_idx"},
				{
					IndexID:         13,
					FieldID:         fieldID,
					IndexName:       "invisible_idx",
					UserIndexParams: []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "false"}},
				},
				{IndexID: 14, FieldID: fieldID, IndexName: "building_idx"},
			}, nil).Twice()

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 3)
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, resp.GetStatuses()[0].GetState())
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, resp.GetStatuses()[1].GetState())
		assert.Contains(t, resp.GetStatuses()[1].GetReason(), "not visible")
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, resp.GetStatuses()[2].GetState())
		assert.Equal(t, "no loaded segment carries the index yet", resp.GetStatuses()[2].GetReason())

		// Once v1 is also Up with 0 segments, active_idx is LOADED.
		f.stats[qviews.ShardID{ReplicaID: replicaA, VChannel: channelV1}] = &coordview.ShardStats{
			UpVersion:         &qviews.QueryViewVersion{QueryVersion: 1},
			UpLoadInfoVersion: f.ver,
			UpPlacement:       &coordview.ViewPlacement{Assignments: map[int64]int64{}},
		}
		resp, err = f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 3)
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_LOADED, resp.GetStatuses()[0].GetState())
		assert.Equal(t, int64(1), resp.GetStatuses()[0].GetReadyReplicas())
		assert.Equal(t, int64(0), resp.GetStatuses()[0].GetExpectedSegments())
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, resp.GetStatuses()[1].GetState())
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, resp.GetStatuses()[2].GetState())
	})

	t.Run("end-to-end QN report through CoordQueryViewStateMachine and recovery preserves 0-index segment", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 12)
		metaPB := &viewpb.QueryViewMeta{
			CollectionId:    collID,
			ReplicaId:       replicaA,
			Vchannel:        channelV0,
			LoadInfoVersion: f.ver,
			Version:         &viewpb.QueryViewVersion{QueryVersion: 1},
			State:           viewpb.QueryViewState_QueryViewStatePreparing,
		}
		qnPB := &viewpb.QueryViewOfQueryNode{
			NodeId: nodeID,
			Partitions: []*viewpb.QueryViewOfPartition{
				{PartitionId: 10, SegmentIds: []int64{segA, segB}},
			},
		}
		snPB := &viewpb.QueryViewOfStreamingNode{}

		// QN reports segA loaded [12] and segB loaded [] (0 indexes!).
		qnSM := qnview.NewQNQueryViewStateMachine(metaPB, qnPB)
		qnSM.OnSegmentsReady(
			map[int64][]int64{10: {segA, segB}},
			map[int64][]int64{segA: {12}, segB: nil},
		)
		qnReportPB := qnSM.ConsumeReport()
		require.NotNil(t, qnReportPB)

		coordSM := coordview.NewCoordQueryViewStateMachine(&viewpb.QueryViewOfShard{
			Meta:          proto.Clone(metaPB).(*viewpb.QueryViewMeta),
			QueryNode:     []*viewpb.QueryViewOfQueryNode{proto.Clone(qnPB).(*viewpb.QueryViewOfQueryNode)},
			StreamingNode: snPB,
		})
		expected, ok := f.server.qviewsRuntime.loadConfigStore.RequiredIndexIDs(collID, f.ver)
		require.True(t, ok)
		coordSM.SetExpectedIndexes(expected)
		_ = coordSM.ConsumeFlush()
		coordSM.OnNodeStateReported(qviews.NewQueryViewAtQueryNode(qnReportPB.GetMeta(), qnReportPB.GetQueryNode()[0]))
		// sm.View() remains pure placement without mutating SegmentLoadedIndexes.
		for _, n := range coordSM.View().GetQueryNode() {
			for _, part := range n.GetPartitions() {
				assert.Empty(t, part.GetSegmentLoadedIndexes())
			}
		}
		snReadyMeta := proto.Clone(metaPB).(*viewpb.QueryViewMeta)
		snReadyMeta.State = viewpb.QueryViewState_QueryViewStateUp
		coordSM.OnNodeStateReported(qviews.NewFullQueryViewAtStreamingNode(snReadyMeta, snPB, coordSM.View().GetQueryNode()))
		require.Equal(t, qviews.QueryViewStateUp, coordSM.State())

		// Simulate ETCD persistence + Coordinator restart recovery.
		flush := coordSM.ConsumeFlush()
		require.NotNil(t, flush.Persist)
		// segA carries the expected index {12}, so steady-state persistence omits it
		// and persists ONLY deviating segB ([]).
		persistedItems := flush.Persist.GetQueryNode()[0].GetPartitions()[0].GetSegmentLoadedIndexes()
		require.Len(t, persistedItems, 1)
		assert.Equal(t, segB, persistedItems[0].GetSegmentId())
		assert.Empty(t, persistedItems[0].GetIndexIds())
		wire, err := proto.Marshal(flush.Persist)
		require.NoError(t, err)
		var recoveredPB viewpb.QueryViewOfShard
		require.NoError(t, proto.Unmarshal(wire, &recoveredPB))
		recoveredSM := coordview.RecoverCoordQueryViewStateMachine(&recoveredPB)
		for _, n := range recoveredSM.View().GetQueryNode() {
			for _, part := range n.GetPartitions() {
				assert.Empty(t, part.GetSegmentLoadedIndexes())
			}
		}

		// Extract ViewPlacement from recoveredSM's QNSegmentIndexes() (same logic as ShardViewManager).
		upPlacement := &coordview.ViewPlacement{
			Assignments:    map[int64]int64{segA: nodeID, segB: nodeID},
			SegmentIndexes: recoveredSM.QNSegmentIndexes(),
		}
		f.stats[qviews.ShardID{ReplicaID: replicaA, VChannel: channelV0}] = &coordview.ShardStats{
			UpVersion:         &qviews.QueryViewVersion{QueryVersion: 1},
			UpLoadInfoVersion: f.ver,
			UpPlacement:       upPlacement,
		}

		f.server.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{
				{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
			}, nil)

		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Len(t, resp.GetStatuses(), 1)
		// segB loaded 0 indexes, so it must NOT fall back to required={12}!
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, resp.GetStatuses()[0].GetState())
		assert.Equal(t, int64(2), resp.GetStatuses()[0].GetExpectedSegments())
		assert.Equal(t, int64(1), resp.GetStatuses()[0].GetReadySegments())
	})

	t.Run("rejects non-positive collectionID and fails explicitly when qviewsRuntime is uninitialized", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{replicaA}, 12)
		resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: 0,
		})
		require.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrParameterInvalid)

		uninitServer := &Server{broker: meta.NewMockBroker(t)}
		uninitServer.status.Store(int32(commonpb.StateCode_Healthy))
		uninitServer.broker.(*meta.MockBroker).EXPECT().
			ListIndexes(mock.Anything, collID).
			Return([]*indexpb.IndexInfo{{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"}}, nil)

		resp, err = uninitServer.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
			CollectionID: collID,
		})
		require.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceInternal)
	})
}

// A serving query view certifies exactly the indexes its load requirements
// selected: an index the view did not select is not loaded on those segments, and
// a shard with no serving view is ignored completely.
func TestQViewsPlacementsCertifyOnlySelectedIndexes(t *testing.T) {
	serving := &coordview.ShardStats{
		UpVersion:         &qviews.QueryViewVersion{},
		UpLoadInfoVersion: 7,
		UpPlacement: &coordview.ViewPlacement{Assignments: map[int64]int64{
			1: 10,
			2: 11,
		}},
	}
	loading := &coordview.ShardStats{
		// No Up view yet: ignored by qviewsPlacements.
		UpPlacement: &coordview.ViewPlacement{Assignments: map[int64]int64{3: 12}},
	}
	placements := qviewsPlacements(map[qviews.ShardID]*coordview.ShardStats{
		{ReplicaID: 1000, VChannel: "v0"}: serving,
		{ReplicaID: 1000, VChannel: "v1"}: loading,
	}, func(version uint64) (map[int64]struct{}, bool) {
		require.Equal(t, uint64(7), version)
		return map[int64]struct{}{2001: {}}, true
	})

	require.Len(t, placements, 2)
	ready := 0
	nodes := typeutil.NewSet[int64]()
	for _, placement := range placements {
		nodes.Insert(placement.node)
		if placement.carries(2001) {
			ready++
		}
		assert.False(t, placement.carries(2002), "an index the view did not select is not loaded")
	}
	assert.Equal(t, 2, ready, "only the serving shard certifies the index")
	assert.ElementsMatch(t, []int64{10, 11}, nodes.Collect())
}

func TestQViewsPlacementsPreferPerSegmentLoadedIndexes(t *testing.T) {
	servingID := qviews.ShardID{ReplicaID: 1000, VChannel: "v0"}
	serving := &coordview.ShardStats{
		UpVersion:         &qviews.QueryViewVersion{},
		UpLoadInfoVersion: 7,
		UpPlacement: &coordview.ViewPlacement{
			Assignments: map[int64]int64{1: 10, 2: 11},
			// The view requires 2001, but only segment 1 reports it as loaded;
			// segment 2 reports an empty set (it loaded, without that index).
			SegmentIndexes: map[int64]map[int64]struct{}{
				1: {2001: {}},
				2: {},
			},
		},
	}
	placements := qviewsPlacements(map[qviews.ShardID]*coordview.ShardStats{
		servingID: serving,
	}, func(uint64) (map[int64]struct{}, bool) {
		return map[int64]struct{}{2001: {}}, true
	})

	require.Len(t, placements, 2)
	carrying := 0
	for _, placement := range placements {
		if placement.carries(2001) {
			carrying++
		}
	}
	assert.Equal(t, 1, carrying,
		"only the segment that reported the index may be certified, not every segment the requirement covers")
}

func TestQViewsPlacementsFallBackWhenSegmentsReportNoIndexes(t *testing.T) {
	servingID := qviews.ShardID{ReplicaID: 1000, VChannel: "v0"}
	serving := &coordview.ShardStats{
		UpVersion:         &qviews.QueryViewVersion{},
		UpLoadInfoVersion: 7,
		// No SegmentIndexes: an older node sends no per-segment report, so the
		// view's requirement remains the only evidence.
		UpPlacement: &coordview.ViewPlacement{Assignments: map[int64]int64{1: 10}},
	}
	placements := qviewsPlacements(map[qviews.ShardID]*coordview.ShardStats{
		servingID: serving,
	}, func(uint64) (map[int64]struct{}, bool) {
		return map[int64]struct{}{2001: {}}, true
	})

	require.Len(t, placements, 1)
	assert.True(t, placements[0].carries(2001))
}

func TestServerGetIndexLoadStatusIgnoresDrainingScaleInReplica(t *testing.T) {
	const (
		collID          = int64(100)
		fieldID         = int64(101)
		activeReplica   = int64(1)
		drainingReplica = int64(2)
		nodeID          = int64(10)
		segA            = int64(1000)
	)
	// Desired LoadConfig has scaled down to only activeReplica (1), while
	// drainingReplica (2) still has an UpPlacement in ShardViewRegistry until drained.
	f := newIndexLoadStatusFixture(t, collID, []int64{activeReplica}, 12)
	f.putServingSegment(activeReplica, "v0", nodeID, segA, 12)
	f.putServingSegment(drainingReplica, "v0", nodeID, segA, 12)
	f.server.broker.(*meta.MockBroker).EXPECT().
		ListIndexes(mock.Anything, collID).
		Return([]*indexpb.IndexInfo{
			{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"},
		}, nil)

	resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
		CollectionID: collID,
	})
	require.NoError(t, err)
	require.Len(t, resp.GetStatuses(), 1)
	status := resp.GetStatuses()[0]
	assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_LOADED, status.GetState())
	assert.Equal(t, int64(1), status.GetExpectedReplicas())
	assert.Equal(t, int64(1), status.GetReadyReplicas())
	assert.Equal(t, int64(1), status.GetExpectedSegments())
	assert.Equal(t, int64(1), status.GetReadySegments())
}

func TestServerGetIndexLoadStatusZeroVersionAndPreShardViewDataView(t *testing.T) {
	const (
		collID  = int64(200)
		fieldID = int64(101)
	)
	f := newIndexLoadStatusFixture(t, collID, []int64{1}, 12)

	// 1. UpLoadInfoVersion == 0 must not fall back to the latest pending LoadConfig.
	_, ok := f.server.qviewsRuntime.loadConfigStore.RequiredIndexIDs(collID, 0)
	assert.False(t, ok)

	// 2. Before ShardViews exist, ExpectedSegments falls back to DataView in balanceCache.
	f.server.qviewsRuntime.resources.VChannels = func(context.Context, int64) ([]string, error) {
		return []string{"v0"}, nil
	}
	bc := cache.New(nil)
	t.Cleanup(bc.Close)
	bc.PublishDataView(collID, balancerapi.PrepareCollectionDataView(&balancerapi.CollectionDataView{
		CollectionID: collID,
		Shards: []*balancerapi.ShardDataView{
			{
				VChannel: "v0",
				Partitions: []*balancerapi.PartitionDataView{
					{
						PartitionID: 10,
						Segments: []*balancerapi.SegmentDataView{
							{SegmentID: 5001, RowNum: 100},
							{SegmentID: 5002, RowNum: 200},
						},
					},
					{
						PartitionID: 99,
						Segments:    []*balancerapi.SegmentDataView{{SegmentID: 5003, RowNum: 300}},
					},
				},
			},
		},
	}))
	f.server.qviewsRuntime.balanceCache = bc
	f.server.broker.(*meta.MockBroker).EXPECT().
		ListIndexes(mock.Anything, collID).
		Return([]*indexpb.IndexInfo{{IndexID: 12, FieldID: fieldID, IndexName: "scalar_ngram"}}, nil)

	resp, err := f.server.GetIndexLoadStatus(context.Background(), &querypb.GetIndexLoadStatusRequest{
		CollectionID: collID,
	})
	require.NoError(t, err)
	require.Len(t, resp.GetStatuses(), 1)
	assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, resp.GetStatuses()[0].GetState())
	assert.Equal(t, int64(2), resp.GetStatuses()[0].GetExpectedSegments())
	assert.Equal(t, int64(0), resp.GetStatuses()[0].GetReadySegments())
}
