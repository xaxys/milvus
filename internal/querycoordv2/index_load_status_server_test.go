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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metacache"
	metastoremocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/views/coord/coordview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// indexLoadStatusFixture is the smallest server surface GetIndexLoadStatus reads:
// the replica manager (which replicas exist and on which nodes) and the segment
// distribution (which segments each node reports as loaded, with the index ids
// that segment actually carries).
type indexLoadStatusFixture struct {
	server   *Server
	dist     *meta.DistributionManager
	segments map[int64][]*meta.Segment
}

func newIndexLoadStatusFixture(t *testing.T, collID int64, nodeIDs []int64) *indexLoadStatusFixture {
	t.Helper()
	ctx := context.Background()

	nodeMgr := session.NewNodeManager()
	catalog := metastoremocks.NewQueryCoordCatalog(t)
	catalog.EXPECT().GetCollections(mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().GetPartitions(mock.Anything, mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().GetReplicas(mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().GetResourceGroups(mock.Anything).Return(nil, nil).Maybe()
	catalog.EXPECT().SaveReplica(mock.Anything, mock.Anything).Return(nil).Maybe()
	catalog.EXPECT().ReleaseReplicas(mock.Anything, mock.Anything).Return(nil).Maybe()
	catalog.EXPECT().SaveResourceGroup(mock.Anything, mock.Anything).Return(nil).Maybe()

	m := meta.NewMeta(nil, catalog, nodeMgr, metacache.NewMetaStore(nil))
	for _, nodeID := range nodeIDs {
		require.NoError(t, m.Put(ctx, meta.NewReplica(&querypb.Replica{
			ID:           nodeID,
			CollectionID: collID,
			Nodes:        []int64{nodeID},
		})))
	}

	dist := meta.NewDistributionManager(nodeMgr)
	server := &Server{
		meta:   m,
		dist:   dist,
		broker: meta.NewMockBroker(t),
	}
	// GetIndexLoadStatus answers only from a healthy coordinator, like every other
	// QueryCoord service method.
	server.status.Store(int32(commonpb.StateCode_Healthy))

	return &indexLoadStatusFixture{server: server, dist: dist, segments: make(map[int64][]*meta.Segment)}
}

// putLoadedSegment registers one loaded segment on a node, carrying the given
// index ids in its FieldIndexInfo map (indexID -> info), exactly as a QueryNode
// reports it.
func (f *indexLoadStatusFixture) putLoadedSegment(nodeID, collID, segID int64, indexIDs ...int64) {
	indexInfo := make(map[int64]*querypb.FieldIndexInfo, len(indexIDs))
	for _, indexID := range indexIDs {
		indexInfo[indexID] = &querypb.FieldIndexInfo{IndexID: indexID, EnableIndex: true}
	}
	// Update replaces a node's whole list, so keep the accumulated view.
	f.segments[nodeID] = append(f.segments[nodeID], &meta.Segment{
		SegmentInfo: &datapb.SegmentInfo{ID: segID, CollectionID: collID},
		Node:        nodeID,
		IndexInfo:   indexInfo,
	})
	f.server.dist.SegmentDistManager.Update(nodeID, f.segments[nodeID]...)
}

func TestServerGetIndexLoadStatus(t *testing.T) {
	const (
		collID  = int64(100)
		fieldID = int64(101)
		nodeID  = int64(1)
		segA    = int64(1000)
		segB    = int64(1001)
	)

	t.Run("every loaded segment carries the index", func(t *testing.T) {
		f := newIndexLoadStatusFixture(t, collID, []int64{nodeID})
		f.putLoadedSegment(nodeID, collID, segA, 11, 12)
		f.putLoadedSegment(nodeID, collID, segB, 12)
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
		f := newIndexLoadStatusFixture(t, collID, []int64{nodeID})
		f.putLoadedSegment(nodeID, collID, segA, 12)
		f.putLoadedSegment(nodeID, collID, segB, 11)
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
		f := newIndexLoadStatusFixture(t, collID, []int64{nodeID})
		f.putLoadedSegment(nodeID, collID, segA, 11)
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
}

// A serving query view certifies exactly the indexes its load requirements
// selected: an index the view did not select is not loaded on those segments, and
// a shard with no serving view certifies nothing at all.
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
		// No Up view yet: the placements exist, but nothing is certified.
		UpPlacement: &coordview.ViewPlacement{Assignments: map[int64]int64{3: 12}},
	}
	placements := qviewsPlacements(map[qviews.ShardID]*coordview.ShardStats{
		{ReplicaID: 1000, VChannel: "v0"}: serving,
		{ReplicaID: 1000, VChannel: "v1"}: loading,
	}, func(version uint64) (*querypb.GetQueryViewLoadInfoResponse, error) {
		require.Equal(t, uint64(7), version)
		return &querypb.GetQueryViewLoadInfoResponse{
			RequiredFields: []*viewpb.LoadFieldConfig{
				{FieldId: 100, IndexIds: []int64{2001}},
			},
		}, nil
	})

	require.Len(t, placements, 3)
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
	assert.ElementsMatch(t, []int64{10, 11, 12}, nodes.Collect())
}
