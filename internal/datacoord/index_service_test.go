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
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/coordinator/snmanager"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/broker"
	"github.com/milvus-io/milvus/internal/distributed/streaming"
	mockkv "github.com/milvus-io/milvus/internal/kv/mocks"
	"github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/mocks/distributed/mock_streaming"
	"github.com/milvus-io/milvus/internal/mocks/streamingcoord/server/mock_balancer"
	"github.com/milvus-io/milvus/internal/mocks/streamingcoord/server/mock_broadcaster"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/balancer"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/balancer/balance"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/balancer/channel"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/broadcast"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/registry"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v3/streaming/walimpls/impls/rmq"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func initStreamingSystem(t *testing.T) {
	registry.ResetRegistration()
	wal := mock_streaming.NewMockWALAccesser(t)
	wal.EXPECT().ControlChannel().Return(funcutil.GetControlChannel("by-dev-rootcoord-dml_0")).Maybe()
	streaming.SetWALForTest(wal)

	bapi := mock_broadcaster.NewMockBroadcastAPI(t)
	bapi.EXPECT().Broadcast(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, msg message.BroadcastMutableMessage) (*types.BroadcastAppendResult, error) {
		results := make(map[string]*message.AppendResult)
		for _, vchannel := range msg.BroadcastHeader().VChannels {
			results[vchannel] = &message.AppendResult{
				MessageID:              rmq.NewRmqID(1),
				TimeTick:               tsoutil.ComposeTSByTime(time.Now()),
				LastConfirmedMessageID: rmq.NewRmqID(1),
			}
		}
		retry.Do(context.Background(), func() error {
			return registry.CallMessageAckCallback(context.Background(), msg, results)
		}, retry.AttemptAlways())
		return &types.BroadcastAppendResult{}, nil
	}).Maybe()
	bapi.EXPECT().Close().Return().Maybe()

	mb := mock_broadcaster.NewMockBroadcaster(t)
	mb.EXPECT().WithResourceKeys(mock.Anything, mock.Anything).Return(bapi, nil).Maybe()
	mb.EXPECT().WithResourceKeys(mock.Anything, mock.Anything, mock.Anything).Return(bapi, nil).Maybe()
	mb.EXPECT().WithResourceKeys(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(bapi, nil).Maybe()
	mb.EXPECT().Close().Return().Maybe()
	broadcast.Release()
	broadcast.ResetBroadcaster()
	broadcast.Register(mb)

	snmanager.ResetStreamingNodeManager()
	b := mock_balancer.NewMockBalancer(t)
	b.EXPECT().AllocVirtualChannels(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, param balancer.AllocVChannelParam) ([]string, error) {
		vchannels := make([]string, 0, param.Num)
		for i := 0; i < param.Num; i++ {
			vchannels = append(vchannels, funcutil.GetVirtualChannel(fmt.Sprintf("by-dev-rootcoord-dml_%d_100v0", i), param.CollectionID, i))
		}
		return vchannels, nil
	}).Maybe()
	b.EXPECT().WatchChannelAssignments(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, callback balancer.WatchChannelAssignmentsCallback) error {
		<-ctx.Done()
		return ctx.Err()
	}).Maybe()
	b.EXPECT().WaitUntilWALbasedDDLReady(mock.Anything).Return(nil).Maybe()
	b.EXPECT().Close().Return().Maybe()
	balance.Register(b)
	channel.ResetStaticPChannelStatsManager()
	channel.RecoverPChannelStatsManager([]string{})
}

func TestServerId(t *testing.T) {
	s := &Server{session: &sessionutil.Session{SessionRaw: sessionutil.SessionRaw{ServerID: 0}}}
	assert.Equal(t, int64(0), s.serverID())
}

func TestServer_CreateIndex(t *testing.T) {
	initStreamingSystem(t)

	var (
		collID  = UniqueID(1)
		fieldID = UniqueID(10)
		// indexID    = UniqueID(100)
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		req = &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexName:       "",
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			Timestamp:       100,
			IsAutoIndex:     false,
			UserIndexParams: indexParams,
		}
		ctx = context.Background()
	)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(nil).Maybe()

	mock0Allocator := newMockAllocator(t)

	collections := typeutil.NewConcurrentMap[UniqueID, *collectionInfo]()
	collections.Insert(collID, &collectionInfo{
		ID:             collID,
		Partitions:     nil,
		StartPositions: nil,
		Properties:     nil,
		CreatedAt:      0,
	})

	indexMeta := newSegmentIndexMeta(catalog)
	s := &Server{
		meta: &meta{
			catalog:     catalog,
			collections: collections,
			indexMeta:   indexMeta,
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}
	RegisterDDLCallbacks(s)

	s.stateCode.Store(commonpb.StateCode_Healthy)

	b := mocks.NewMixCoord(t)

	t.Run("get field name failed", func(t *testing.T) {
		b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(nil, errors.New("mock error"))

		s.broker = broker.NewCoordinatorBroker(b)
		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
		assert.Equal(t, "mock error", resp.GetReason())
	})

	b.ExpectedCalls = nil
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status: &commonpb.Status{
			ErrorCode: 0,
			Reason:    "",
			Code:      0,
			Retriable: false,
			Detail:    "",
		},
		Schema: &schemapb.CollectionSchema{
			Name:        "test_index",
			Description: "test index",
			AutoID:      false,
			Fields: []*schemapb.FieldSchema{
				{
					FieldID:        0,
					Name:           "pk",
					IsPrimaryKey:   false,
					Description:    "",
					DataType:       schemapb.DataType_Int64,
					TypeParams:     nil,
					IndexParams:    nil,
					AutoID:         false,
					State:          0,
					ElementType:    0,
					DefaultValue:   nil,
					IsDynamic:      false,
					IsPartitionKey: false,
				},
				{
					FieldID:        fieldID,
					Name:           "FieldFloatVector",
					IsPrimaryKey:   false,
					Description:    "",
					DataType:       schemapb.DataType_FloatVector,
					TypeParams:     nil,
					IndexParams:    nil,
					AutoID:         false,
					State:          0,
					ElementType:    0,
					DefaultValue:   nil,
					IsDynamic:      false,
					IsPartitionKey: false,
				},
				{
					FieldID:        fieldID + 1,
					Name:           "json",
					IsPrimaryKey:   false,
					Description:    "",
					DataType:       schemapb.DataType_JSON,
					TypeParams:     nil,
					IndexParams:    nil,
					AutoID:         false,
					State:          0,
					ElementType:    0,
					DefaultValue:   nil,
					IsDynamic:      false,
					IsPartitionKey: false,
				},
			},
			EnableDynamicField: false,
		},
		CollectionID: collID,
	}, nil)

	t.Run("success", func(t *testing.T) {
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("test json path", func(t *testing.T) {
		req := &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID + 1,
			IndexName:       "",
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			Timestamp:       100,
			IsAutoIndex:     false,
			UserIndexParams: indexParams,
		}
		req.IndexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.JSONPathKey,
				Value: "json",
			},
			{
				Key:   common.JSONCastTypeKey,
				Value: "double",
			},
			{
				Key:   common.IndexTypeKey,
				Value: "INVERTED",
			},
		}
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		indexes := s.meta.indexMeta.GetFieldIndexes(req.GetCollectionID(), req.GetFieldID(), req.GetIndexName())
		assert.Equal(t, 1, len(indexes))
		jsonPath, err := funcutil.GetAttrByKeyFromRepeatedKV(common.JSONPathKey, indexes[0].IndexParams)
		assert.NoError(t, err)
		assert.Equal(t, "", jsonPath)
	})

	t.Run("success with index exist", func(t *testing.T) {
		req.IndexName = ""
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("server not healthy", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Abnormal)
		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
	})

	req.IndexName = "FieldFloatVector"
	t.Run("index not consistent", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Healthy)
		req.FieldID++
		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("alloc ID fail", func(t *testing.T) {
		req.FieldID = fieldID
		alloc := allocator.NewMockAllocator(t)
		alloc.EXPECT().AllocID(mock.Anything).Return(0, errors.New("mock")).Maybe()
		alloc.EXPECT().AllocTimestamp(mock.Anything).Return(0, nil).Maybe()
		s.allocator = alloc
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}
		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("disk index", func(t *testing.T) {
		s.allocator = mock0Allocator
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}
		req.IndexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "DISKANN",
			},
		}
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("disk index with mmap", func(t *testing.T) {
		s.allocator = mock0Allocator
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}
		req.IndexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "DISKANN",
			},
			{
				Key:   common.MmapEnabledKey,
				Value: "true",
			},
		}

		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("save index fail", func(t *testing.T) {
		metakv := mockkv.NewMetaKv(t)
		metakv.EXPECT().Save(mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, key string, value string) error {
			if rand.Intn(3) == 0 {
				return errors.New("failed")
			}
			return nil
		}).Maybe()
		metakv.EXPECT().MultiSave(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, kvs map[string]string) error {
			if rand.Intn(3) == 0 {
				return errors.New("failed")
			}
			return nil
		}).Maybe()
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}
		s.meta.catalog = &datacoord.Catalog{MetaKv: metakv}
		s.meta.indexMeta.catalog = s.meta.catalog
		req.IndexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("preserve index ID with valid ID", func(t *testing.T) {
		// Reset catalog and allocator
		s.meta.catalog = catalog
		s.meta.indexMeta.catalog = catalog
		s.allocator = mock0Allocator
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}

		preservedIndexID := int64(12345)
		req := &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexName:       "preserved_index",
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			Timestamp:       100,
			IsAutoIndex:     false,
			UserIndexParams: indexParams,
			PreserveIndexId: true,
			IndexId:         preservedIndexID,
		}

		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		// Verify the index was created with the preserved ID
		indexes := s.meta.indexMeta.GetFieldIndexes(collID, fieldID, "preserved_index")
		assert.Equal(t, 1, len(indexes))
		assert.Equal(t, preservedIndexID, indexes[0].IndexID)
	})

	t.Run("preserve index ID with invalid ID (zero)", func(t *testing.T) {
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}

		req := &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexName:       "invalid_preserved_index",
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			Timestamp:       100,
			IsAutoIndex:     false,
			UserIndexParams: indexParams,
			PreserveIndexId: true,
			IndexId:         0, // Invalid ID
		}

		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
		assert.Contains(t, resp.GetReason(), "index_id must be positive")
	})

	t.Run("preserve index ID with invalid ID (negative)", func(t *testing.T) {
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}

		req := &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexName:       "negative_preserved_index",
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			Timestamp:       100,
			IsAutoIndex:     false,
			UserIndexParams: indexParams,
			PreserveIndexId: true,
			IndexId:         -100, // Invalid negative ID
		}

		resp, err := s.CreateIndex(ctx, req)
		assert.Error(t, merr.CheckRPCCall(resp, err))
		assert.Contains(t, resp.GetReason(), "index_id must be positive")
	})

	t.Run("normal index creation without preserve", func(t *testing.T) {
		// Verify normal path still works (without PreserveIndexId)
		s.meta.catalog = catalog
		s.meta.indexMeta.catalog = catalog
		s.meta.indexMeta.indexes = map[UniqueID]map[UniqueID]*model.Index{}

		req := &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexName:       "normal_index",
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			Timestamp:       100,
			IsAutoIndex:     false,
			UserIndexParams: indexParams,
			PreserveIndexId: false, // Normal allocation
		}

		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		// Verify the index was created with an allocated ID (not zero)
		indexes := s.meta.indexMeta.GetFieldIndexes(collID, fieldID, "normal_index")
		assert.Equal(t, 1, len(indexes))
		assert.NotEqual(t, int64(0), indexes[0].IndexID)
	})
}

func TestServer_AlterIndex(t *testing.T) {
	initStreamingSystem(t)
	var (
		collID       = UniqueID(1)
		partID       = UniqueID(2)
		fieldID      = UniqueID(10)
		indexID      = UniqueID(100)
		segID        = UniqueID(1000)
		invalidSegID = UniqueID(1001)
		buildID      = UniqueID(10000)
		indexName    = "default_idx"
		typeParams   = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.AlterIndexRequest{
			CollectionID: collID,
			IndexName:    "default_idx",
			Params: []*commonpb.KeyValuePair{{
				Key:   common.MmapEnabledKey,
				Value: "true",
			}},
		}
	)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.On("AlterIndexes",
		mock.Anything,
		mock.Anything,
	).Return(nil)

	mock0Allocator := newMockAllocator(t)

	indexMeta := &indexMeta{
		catalog: catalog,
		indexes: map[UniqueID]map[UniqueID]*model.Index{
			collID: {
				// finished
				indexID: {
					TenantID:        "",
					CollectionID:    collID,
					FieldID:         fieldID,
					IndexID:         indexID,
					IndexName:       indexName,
					IsDeleted:       false,
					CreateTime:      createTS,
					TypeParams:      typeParams,
					IndexParams:     indexParams,
					IsAutoIndex:     false,
					UserIndexParams: nil,
				},
				// deleted
				indexID + 1: {
					TenantID:        "",
					CollectionID:    collID,
					FieldID:         fieldID + 1,
					IndexID:         indexID + 1,
					IndexName:       indexName + "_1",
					IsDeleted:       true,
					CreateTime:      createTS,
					TypeParams:      typeParams,
					IndexParams:     indexParams,
					IsAutoIndex:     false,
					UserIndexParams: nil,
				},
				// unissued
				indexID + 2: {
					TenantID:        "",
					CollectionID:    collID,
					FieldID:         fieldID + 2,
					IndexID:         indexID + 2,
					IndexName:       indexName + "_2",
					IsDeleted:       false,
					CreateTime:      createTS,
					TypeParams:      typeParams,
					IndexParams:     indexParams,
					IsAutoIndex:     false,
					UserIndexParams: nil,
				},
				// inProgress
				indexID + 3: {
					TenantID:        "",
					CollectionID:    collID,
					FieldID:         fieldID + 3,
					IndexID:         indexID + 3,
					IndexName:       indexName + "_3",
					IsDeleted:       false,
					CreateTime:      createTS,
					TypeParams:      typeParams,
					IndexParams:     indexParams,
					IsAutoIndex:     false,
					UserIndexParams: nil,
				},
				// failed
				indexID + 4: {
					TenantID:        "",
					CollectionID:    collID,
					FieldID:         fieldID + 4,
					IndexID:         indexID + 4,
					IndexName:       indexName + "_4",
					IsDeleted:       false,
					CreateTime:      createTS,
					TypeParams:      typeParams,
					IndexParams:     indexParams,
					IsAutoIndex:     false,
					UserIndexParams: nil,
				},
				// unissued
				indexID + 5: {
					TenantID:        "",
					CollectionID:    collID,
					FieldID:         fieldID + 5,
					IndexID:         indexID + 5,
					IndexName:       indexName + "_5",
					IsDeleted:       false,
					CreateTime:      createTS,
					TypeParams:      typeParams,
					IndexParams:     indexParams,
					IsAutoIndex:     false,
					UserIndexParams: nil,
				},
			},
		},
		segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
	}
	segIdx1 := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx1.Insert(indexID, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID,
		BuildID:             buildID,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+1, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 1,
		BuildID:             buildID + 1,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+3, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 3,
		BuildID:             buildID + 3,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_InProgress,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+4, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 4,
		BuildID:             buildID + 4,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Failed,
		FailReason:          "mock failed",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+5, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 5,
		BuildID:             buildID + 5,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Unissued,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	indexMeta.segmentIndexes.Insert(segID, segIdx1)

	segIdx2 := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx2.Insert(indexID, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID,
		BuildID:        buildID,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_Finished,
		CreatedUTCTime: createTS,
	})
	segIdx2.Insert(indexID+1, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID + 1,
		BuildID:        buildID + 1,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_Finished,
		CreatedUTCTime: createTS,
	})
	segIdx2.Insert(indexID+3, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID + 3,
		BuildID:        buildID + 3,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_InProgress,
		CreatedUTCTime: createTS,
	})
	segIdx2.Insert(indexID+4, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID + 4,
		BuildID:        buildID + 4,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_Failed,
		FailReason:     "mock failed",
		CreatedUTCTime: createTS,
	})
	segIdx2.Insert(indexID+5, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID + 5,
		BuildID:        buildID + 5,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_Finished,
		CreatedUTCTime: createTS,
	})
	indexMeta.segmentIndexes.Insert(segID-1, segIdx2)

	mockHandler := NewNMockHandler(t)

	mockGetCollectionInfo := func() {
		mockHandler.EXPECT().GetCollection(mock.Anything, collID).Return(&collectionInfo{
			ID: collID,
			Schema: &schemapb.CollectionSchema{
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:  fieldID,
						Name:     "FieldFloatVector",
						DataType: schemapb.DataType_FloatVector,
					},
				},
			},
		}, nil).Once()
	}
	b := broker.NewMockBroker(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status:         merr.Status(nil),
		DbName:         "test_db",
		CollectionName: "test_collection",
	}, nil)

	s := &Server{
		meta: &meta{
			catalog:   catalog,
			indexMeta: indexMeta,
			segments: &SegmentsInfo{
				compactionTo: make(map[int64][]int64),
				segments: map[UniqueID]*SegmentInfo{
					invalidSegID: {
						SegmentInfo: &datapb.SegmentInfo{
							ID:             invalidSegID,
							CollectionID:   collID,
							PartitionID:    partID,
							NumOfRows:      10000,
							State:          commonpb.SegmentState_Flushed,
							MaxRowNum:      65536,
							LastExpireTime: createTS,
							Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
							StartPosition: &msgpb.MsgPosition{
								// timesamp > index start time, will be filtered out
								Timestamp: createTS + 1,
							},
						},
					},
					segID: {
						SegmentInfo: &datapb.SegmentInfo{
							ID:             segID,
							CollectionID:   collID,
							PartitionID:    partID,
							NumOfRows:      10000,
							State:          commonpb.SegmentState_Flushed,
							MaxRowNum:      65536,
							LastExpireTime: createTS,
							Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
							StartPosition: &msgpb.MsgPosition{
								Timestamp: createTS,
							},
							CreatedByCompaction: true,
							CompactionFrom:      []int64{segID - 1},
						},
					},
					segID - 1: {
						SegmentInfo: &datapb.SegmentInfo{
							ID:             segID,
							CollectionID:   collID,
							PartitionID:    partID,
							NumOfRows:      10000,
							State:          commonpb.SegmentState_Dropped,
							MaxRowNum:      65536,
							LastExpireTime: createTS,
							Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
							StartPosition: &msgpb.MsgPosition{
								Timestamp: createTS,
							},
						},
					},
				},
			},
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
		handler:         mockHandler,
		broker:          b,
	}
	RegisterDDLCallbacks(s)

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.AlterIndex(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp), merr.ErrServiceNotReady)
	})

	s.stateCode.Store(commonpb.StateCode_Healthy)

	t.Run("mmap_unsupported", func(t *testing.T) {
		mockGetCollectionInfo()
		indexParams[0].Value = "GPU_CAGRA"

		resp, err := s.AlterIndex(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.CheckRPCCall(resp, err), merr.ErrParameterInvalid)

		indexParams[0].Value = "IVF_FLAT"
	})

	t.Run("param_value_invalied", func(t *testing.T) {
		mockGetCollectionInfo()
		req.Params[0].Value = "abc"
		resp, err := s.AlterIndex(ctx, req)
		assert.ErrorIs(t, merr.CheckRPCCall(resp, err), merr.ErrParameterInvalid)

		req.Params[0].Value = "true"
	})

	t.Run("delete_params", func(t *testing.T) {
		mockGetCollectionInfo()
		deleteReq := &indexpb.AlterIndexRequest{
			CollectionID: collID,
			IndexName:    indexName,
			DeleteKeys:   []string{common.MmapEnabledKey},
		}
		resp, err := s.AlterIndex(ctx, deleteReq)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		describeResp, err := s.DescribeIndex(ctx, &indexpb.DescribeIndexRequest{
			CollectionID: collID,
			IndexName:    indexName,
			Timestamp:    createTS,
		})
		assert.NoError(t, merr.CheckRPCCall(describeResp, err))
		for _, param := range describeResp.IndexInfos[0].GetUserIndexParams() {
			assert.NotEqual(t, common.MmapEnabledKey, param.GetKey())
		}
	})
	t.Run("update_and_delete_params", func(t *testing.T) {
		mockGetCollectionInfo()
		updateAndDeleteReq := &indexpb.AlterIndexRequest{
			CollectionID: collID,
			IndexName:    indexName,
			Params: []*commonpb.KeyValuePair{
				{
					Key:   common.MmapEnabledKey,
					Value: "true",
				},
			},
			DeleteKeys: []string{common.MmapEnabledKey},
		}
		resp, err := s.AlterIndex(ctx, updateAndDeleteReq)
		assert.ErrorIs(t, merr.CheckRPCCall(resp, err), merr.ErrParameterInvalid)
	})

	t.Run("success", func(t *testing.T) {
		resp, err := s.AlterIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		describeResp, err := s.DescribeIndex(ctx, &indexpb.DescribeIndexRequest{
			CollectionID: collID,
			IndexName:    "default_idx",
			Timestamp:    createTS,
		})
		assert.NoError(t, merr.CheckRPCCall(describeResp, err))
		enableMmap, ok := common.IsMmapDataEnabled(describeResp.IndexInfos[0].GetUserIndexParams()...)
		assert.True(t, enableMmap, "indexInfo: %+v", describeResp.IndexInfos[0])
		assert.True(t, ok)
	})
}

func TestServer_GetIndexState(t *testing.T) {
	var (
		collID     = UniqueID(1)
		partID     = UniqueID(2)
		fieldID    = UniqueID(10)
		indexID    = UniqueID(100)
		segID      = UniqueID(1000)
		buildID    = UniqueID(10000)
		indexName  = "default_idx"
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.GetIndexStateRequest{
			CollectionID: collID,
			IndexName:    "",
		}
	)
	mock0Allocator := newMockAllocator(t)
	s := &Server{
		meta: &meta{
			catalog:   &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
			indexMeta: newSegmentIndexMeta(&datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)}),
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.GetIndexState(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	s.stateCode.Store(commonpb.StateCode_Healthy)
	t.Run("index not found", func(t *testing.T) {
		resp, err := s.GetIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_IndexNotExist, resp.GetStatus().GetErrorCode())
	})

	segments := map[UniqueID]*SegmentInfo{
		segID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				InsertChannel:  "",
				NumOfRows:      10250,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS - 1,
				Stats:          &datapb.Statistics{TimestampFrom: createTS - 1, TimestampTo: createTS - 1},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS - 1,
				},
			},
			allocations:     nil,
			lastFlushTime:   time.Time{},
			isCompacting:    false,
			lastWrittenTime: time.Time{},
		},
	}
	s.meta = &meta{
		catalog: &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
		indexMeta: &indexMeta{
			catalog: &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
			indexes: map[UniqueID]map[UniqueID]*model.Index{
				collID: {
					indexID: {
						TenantID:        "",
						CollectionID:    collID,
						FieldID:         fieldID,
						IndexID:         indexID,
						IndexName:       indexName,
						IsDeleted:       false,
						CreateTime:      createTS,
						TypeParams:      typeParams,
						IndexParams:     indexParams,
						IsAutoIndex:     false,
						UserIndexParams: nil,
					},
				},
			},
			segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
		},

		segments: NewSegmentsInfo(),
	}
	for id, segment := range segments {
		s.meta.segments.SetSegment(id, segment)
	}

	t.Run("index state is unissued", func(t *testing.T) {
		resp, err := s.GetIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, commonpb.IndexState_InProgress, resp.GetState())
	})

	segments = map[UniqueID]*SegmentInfo{
		segID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				InsertChannel:  "",
				NumOfRows:      10250,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS - 1,
				Stats:          &datapb.Statistics{TimestampFrom: createTS - 1, TimestampTo: createTS - 1},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS - 1,
				},
			},
			allocations:     nil,
			lastFlushTime:   time.Time{},
			isCompacting:    false,
			lastWrittenTime: time.Time{},
		},
	}
	s.meta = &meta{
		catalog: &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
		indexMeta: &indexMeta{
			indexes: map[UniqueID]map[UniqueID]*model.Index{
				collID: {
					indexID: {
						TenantID:        "",
						CollectionID:    collID,
						FieldID:         fieldID,
						IndexID:         indexID,
						IndexName:       indexName,
						IsDeleted:       false,
						CreateTime:      createTS,
						TypeParams:      typeParams,
						IndexParams:     indexParams,
						IsAutoIndex:     false,
						UserIndexParams: nil,
					},
				},
			},
			segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
		},
		segments: NewSegmentsInfo(),
	}
	segIdx := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx.Insert(indexID, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             3000,
		IndexID:             indexID,
		BuildID:             buildID,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_IndexStateNone,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      0,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	s.meta.indexMeta.segmentIndexes.Insert(segID, segIdx)
	for id, segment := range segments {
		s.meta.segments.SetSegment(id, segment)
	}

	t.Run("index state is none", func(t *testing.T) {
		resp, err := s.GetIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, commonpb.IndexState_IndexStateNone, resp.GetState())
	})

	t.Run("ambiguous index name", func(t *testing.T) {
		s.meta.indexMeta.indexes[collID][indexID+1] = &model.Index{
			TenantID:        "",
			CollectionID:    collID,
			IndexID:         indexID + 1,
			IndexName:       "default_idx_1",
			IsDeleted:       false,
			CreateTime:      createTS,
			TypeParams:      typeParams,
			IndexParams:     indexParams,
			IsAutoIndex:     false,
			UserIndexParams: nil,
		}
		resp, err := s.GetIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.GetStatus().GetErrorCode())
	})
}

func TestServer_GetSegmentIndexState(t *testing.T) {
	var (
		collID     = UniqueID(1)
		partID     = UniqueID(2)
		fieldID    = UniqueID(10)
		indexID    = UniqueID(100)
		segID      = UniqueID(1000)
		indexName  = "default_idx"
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.GetSegmentIndexStateRequest{
			CollectionID: collID,
			IndexName:    "",
			SegmentIDs:   []UniqueID{segID},
		}
	)

	mock0Allocator := newMockAllocator(t)
	indexMeta := newSegmentIndexMeta(&datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)})

	s := &Server{
		meta: &meta{
			catalog:   indexMeta.catalog,
			indexMeta: indexMeta,
			segments:  NewSegmentsInfo(),
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}

	t.Run("server is not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Abnormal)
		resp, err := s.GetSegmentIndexState(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	t.Run("no indexes", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Healthy)
		resp, err := s.GetSegmentIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_IndexNotExist, resp.GetStatus().GetErrorCode())
	})

	t.Run("unfinished", func(t *testing.T) {
		s.meta.indexMeta.indexes[collID] = map[UniqueID]*model.Index{
			indexID: {
				TenantID:        "",
				CollectionID:    collID,
				FieldID:         fieldID,
				IndexID:         indexID,
				IndexName:       indexName,
				IsDeleted:       false,
				CreateTime:      createTS,
				TypeParams:      typeParams,
				IndexParams:     indexParams,
				IsAutoIndex:     false,
				UserIndexParams: nil,
			},
		}
		s.meta.indexMeta.updateSegmentIndex(&model.SegmentIndex{
			SegmentID:           segID,
			CollectionID:        collID,
			PartitionID:         partID,
			NumRows:             10250,
			IndexID:             indexID,
			BuildID:             10,
			NodeID:              0,
			IndexVersion:        1,
			IndexState:          commonpb.IndexState_InProgress,
			FailReason:          "",
			IsDeleted:           false,
			CreatedUTCTime:      createTS,
			IndexFileKeys:       []string{"file1", "file2"},
			IndexSerializedSize: 1025,
			WriteHandoff:        false,
		})
		s.meta.segments.SetSegment(segID, &SegmentInfo{
			SegmentInfo: &datapb.SegmentInfo{
				ID:            segID,
				CollectionID:  collID,
				PartitionID:   partID,
				InsertChannel: "ch",
			},
			allocations:     nil,
			lastFlushTime:   time.Time{},
			isCompacting:    false,
			lastWrittenTime: time.Time{},
		})

		resp, err := s.GetSegmentIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})

	t.Run("finish", func(t *testing.T) {
		s.meta.indexMeta.updateSegmentIndex(&model.SegmentIndex{
			SegmentID:           segID,
			CollectionID:        collID,
			PartitionID:         partID,
			NumRows:             10250,
			IndexID:             indexID,
			BuildID:             10,
			NodeID:              0,
			IndexVersion:        1,
			IndexState:          commonpb.IndexState_Finished,
			FailReason:          "",
			IsDeleted:           false,
			CreatedUTCTime:      createTS,
			IndexFileKeys:       []string{"file1", "file2"},
			IndexSerializedSize: 1025,
			WriteHandoff:        false,
		})
		resp, err := s.GetSegmentIndexState(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestServer_GetIndexBuildProgress(t *testing.T) {
	var (
		collID     = UniqueID(1)
		partID     = UniqueID(2)
		fieldID    = UniqueID(10)
		indexID    = UniqueID(100)
		segID      = UniqueID(1000)
		indexName  = "default_idx"
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.GetIndexBuildProgressRequest{
			CollectionID: collID,
			IndexName:    "",
		}
	)

	mock0Allocator := newMockAllocator(t)

	s := &Server{
		meta: &meta{
			catalog:   &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
			indexMeta: newSegmentIndexMeta(&datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)}),
			segments:  NewSegmentsInfo(),
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}
	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.GetIndexBuildProgress(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	t.Run("no indexes", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Healthy)
		resp, err := s.GetIndexBuildProgress(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_IndexNotExist, resp.GetStatus().GetErrorCode())
	})

	t.Run("unissued", func(t *testing.T) {
		s.meta.indexMeta.indexes[collID] = map[UniqueID]*model.Index{
			indexID: {
				TenantID:        "",
				CollectionID:    collID,
				FieldID:         fieldID,
				IndexID:         indexID,
				IndexName:       indexName,
				IsDeleted:       false,
				CreateTime:      createTS,
				TypeParams:      typeParams,
				IndexParams:     indexParams,
				IsAutoIndex:     false,
				UserIndexParams: nil,
			},
		}
		s.meta.segments = NewSegmentsInfo()
		s.meta.segments.SetSegment(segID, &SegmentInfo{
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				InsertChannel:  "",
				NumOfRows:      10250,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS,
				},
			},
			allocations:     nil,
			lastFlushTime:   time.Time{},
			isCompacting:    false,
			lastWrittenTime: time.Time{},
		})

		resp, err := s.GetIndexBuildProgress(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, int64(10250), resp.GetTotalRows())
		assert.Equal(t, int64(0), resp.GetIndexedRows())
	})

	t.Run("finish", func(t *testing.T) {
		s.meta.indexMeta.updateSegmentIndex(&model.SegmentIndex{
			SegmentID:           segID,
			CollectionID:        collID,
			PartitionID:         partID,
			NumRows:             10250,
			IndexID:             indexID,
			BuildID:             10,
			NodeID:              0,
			IndexVersion:        1,
			IndexState:          commonpb.IndexState_Finished,
			FailReason:          "",
			IsDeleted:           false,
			CreatedUTCTime:      createTS,
			IndexFileKeys:       []string{"file1", "file2"},
			IndexSerializedSize: 0,
			WriteHandoff:        false,
		})
		s.meta.segments = NewSegmentsInfo()
		s.meta.segments.SetSegment(segID, &SegmentInfo{
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				InsertChannel:  "",
				NumOfRows:      10250,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS,
				},
			},
			allocations:     nil,
			lastFlushTime:   time.Time{},
			isCompacting:    false,
			lastWrittenTime: time.Time{},
		})

		resp, err := s.GetIndexBuildProgress(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, int64(10250), resp.GetTotalRows())
		assert.Equal(t, int64(10250), resp.GetIndexedRows())
	})

	t.Run("multiple index", func(t *testing.T) {
		s.meta.indexMeta.indexes[collID] = map[UniqueID]*model.Index{
			indexID: {
				TenantID:        "",
				CollectionID:    collID,
				FieldID:         fieldID,
				IndexID:         indexID,
				IndexName:       indexName,
				IsDeleted:       false,
				CreateTime:      createTS,
				TypeParams:      typeParams,
				IndexParams:     indexParams,
				IsAutoIndex:     false,
				UserIndexParams: nil,
			},
			indexID + 1: {
				TenantID:        "",
				CollectionID:    collID,
				FieldID:         fieldID + 1,
				IndexID:         indexID + 1,
				IndexName:       "_default_idx_102",
				IsDeleted:       false,
				CreateTime:      0,
				TypeParams:      nil,
				IndexParams:     nil,
				IsAutoIndex:     false,
				UserIndexParams: nil,
			},
		}
		resp, err := s.GetIndexBuildProgress(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.GetStatus().GetErrorCode())
	})
}

func TestServer_DescribeIndex(t *testing.T) {
	initStreamingSystem(t)

	var (
		collID       = UniqueID(1)
		partID       = UniqueID(2)
		fieldID      = UniqueID(10)
		indexID      = UniqueID(100)
		segID        = UniqueID(1000)
		invalidSegID = UniqueID(1001)
		buildID      = UniqueID(10000)
		indexName    = "default_idx"
		typeParams   = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.DescribeIndexRequest{
			CollectionID: collID,
			IndexName:    "",
			Timestamp:    createTS,
		}
	)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.On("AlterIndexes",
		mock.Anything,
		mock.Anything,
	).Return(nil)

	mock0Allocator := newMockAllocator(t)

	segments := map[UniqueID]*SegmentInfo{
		invalidSegID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             invalidSegID,
				CollectionID:   collID,
				PartitionID:    partID,
				NumOfRows:      10000,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					// timesamp > index start time, will be filtered out
					Timestamp: createTS + 1,
				},
			},
		},
		segID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				NumOfRows:      10000,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS,
				},
				CreatedByCompaction: true,
				CompactionFrom:      []int64{segID - 1},
			},
		},
		segID - 1: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID - 1,
				CollectionID:   collID,
				PartitionID:    partID,
				NumOfRows:      10000,
				State:          commonpb.SegmentState_Dropped,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS,
				},
			},
		},
	}

	b := broker.NewMockBroker(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status:         merr.Status(nil),
		DbName:         "test_db",
		CollectionName: "test_collection",
	}, nil)
	s := &Server{
		meta: &meta{
			catalog: catalog,
			indexMeta: &indexMeta{
				catalog: catalog,
				indexes: map[UniqueID]map[UniqueID]*model.Index{
					collID: {
						// finished
						indexID: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID,
							IndexName:       indexName,
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// deleted
						indexID + 1: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 1,
							IndexID:         indexID + 1,
							IndexName:       indexName + "_1",
							IsDeleted:       true,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 2: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 2,
							IndexID:         indexID + 2,
							IndexName:       indexName + "_2",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// inProgress
						indexID + 3: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 3,
							IndexID:         indexID + 3,
							IndexName:       indexName + "_3",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// failed
						indexID + 4: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 4,
							IndexID:         indexID + 4,
							IndexName:       indexName + "_4",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 5: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 5,
							IndexID:         indexID + 5,
							IndexName:       indexName + "_5",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
					},
				},
				segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			},

			segments: NewSegmentsInfo(),
		},
		mixCoord:        mocks.NewMixCoord(t),
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
		broker:          b,
	}
	segIdx1 := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx1.Insert(indexID, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID,
		BuildID:             buildID,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
		CurrentIndexVersion: 7,
	})
	segIdx1.Insert(indexID+1, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 1,
		BuildID:             buildID + 1,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
		// deleted index
		CurrentIndexVersion: 6,
	})
	segIdx1.Insert(indexID+3, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 3,
		BuildID:             buildID + 3,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_InProgress,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+4, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 4,
		BuildID:             buildID + 4,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Failed,
		FailReason:          "mock failed",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+5, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 5,
		BuildID:             buildID + 5,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Unissued,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	s.meta.indexMeta.segmentIndexes.Insert(segID, segIdx1)

	segIdx2 := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx2.Insert(indexID, &model.SegmentIndex{
		SegmentID:           segID - 1,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID,
		BuildID:             buildID,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		CreatedUTCTime:      createTS,
		CurrentIndexVersion: 6,
	})
	segIdx2.Insert(indexID+1, &model.SegmentIndex{
		SegmentID:           segID - 1,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 1,
		BuildID:             buildID + 1,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		CreatedUTCTime:      createTS,
		CurrentIndexVersion: 6,
	})
	segIdx2.Insert(indexID+3, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID + 3,
		BuildID:        buildID + 3,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_InProgress,
		CreatedUTCTime: createTS,
	})
	segIdx2.Insert(indexID+4, &model.SegmentIndex{
		SegmentID:      segID - 1,
		CollectionID:   collID,
		PartitionID:    partID,
		NumRows:        10000,
		IndexID:        indexID + 4,
		BuildID:        buildID + 4,
		NodeID:         0,
		IndexVersion:   1,
		IndexState:     commonpb.IndexState_Failed,
		FailReason:     "mock failed",
		CreatedUTCTime: createTS,
	})
	segIdx2.Insert(indexID+5, &model.SegmentIndex{
		SegmentID:           segID - 1,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 5,
		BuildID:             buildID + 5,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		CreatedUTCTime:      createTS,
		CurrentIndexVersion: 6,
	})
	s.meta.indexMeta.segmentIndexes.Insert(segID-1, segIdx2)

	for id, segment := range segments {
		s.meta.segments.SetSegment(id, segment)
	}
	RegisterDDLCallbacks(s)

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.DescribeIndex(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	s.stateCode.Store(commonpb.StateCode_Healthy)

	t.Run("success", func(t *testing.T) {
		resp, err := s.DescribeIndex(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, 5, len(resp.GetIndexInfos()))
		minIndexVersion := int32(math.MaxInt32)
		maxIndexVersion := int32(math.MinInt32)
		for _, indexInfo := range resp.GetIndexInfos() {
			if indexInfo.GetMinIndexVersion() < minIndexVersion {
				minIndexVersion = indexInfo.GetMinIndexVersion()
			}
			if indexInfo.GetMaxIndexVersion() > maxIndexVersion {
				maxIndexVersion = indexInfo.GetMaxIndexVersion()
			}
		}
		assert.Equal(t, int32(7), minIndexVersion)
		assert.Equal(t, int32(7), maxIndexVersion)
	})

	t.Run("describe after drop index", func(t *testing.T) {
		s.mixCoord.(*mocks.MixCoord).EXPECT().ShowLoadCollections(
			mock.Anything,
			mock.Anything,
		).Return(&querypb.ShowCollectionsResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_Success,
			},
			CollectionIDs: []int64{},
		}, nil)
		status, err := s.DropIndex(ctx, &indexpb.DropIndexRequest{
			CollectionID: collID,
			PartitionIDs: nil,
			IndexName:    "",
			DropAll:      true,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())

		resp, err := s.DescribeIndex(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_IndexNotExist, resp.GetStatus().GetErrorCode())
	})
}

func TestServer_ListIndexes(t *testing.T) {
	var (
		collID     = UniqueID(1)
		fieldID    = UniqueID(10)
		indexID    = UniqueID(100)
		indexName  = "default_idx"
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.ListIndexesRequest{
			CollectionID: collID,
		}
	)

	mock0Allocator := newMockAllocator(t)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	s := &Server{
		meta: &meta{
			catalog: catalog,
			indexMeta: &indexMeta{
				catalog: catalog,
				indexes: map[UniqueID]map[UniqueID]*model.Index{
					collID: {
						// finished
						indexID: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID,
							IndexName:       indexName,
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// deleted
						indexID + 1: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 1,
							IndexID:         indexID + 1,
							IndexName:       indexName + "_1",
							IsDeleted:       true,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 2: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 2,
							IndexID:         indexID + 2,
							IndexName:       indexName + "_2",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// inProgress
						indexID + 3: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 3,
							IndexID:         indexID + 3,
							IndexName:       indexName + "_3",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// failed
						indexID + 4: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 4,
							IndexID:         indexID + 4,
							IndexName:       indexName + "_4",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 5: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 5,
							IndexID:         indexID + 5,
							IndexName:       indexName + "_5",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
					},
				},
				segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			},

			segments: NewSegmentsInfo(),
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.ListIndexes(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	s.stateCode.Store(commonpb.StateCode_Healthy)

	t.Run("success", func(t *testing.T) {
		resp, err := s.ListIndexes(ctx, req)
		assert.NoError(t, err)

		// assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, 5, len(resp.GetIndexInfos()))
	})
}

func TestServer_GetIndexStatistics(t *testing.T) {
	initStreamingSystem(t)
	var (
		collID       = UniqueID(1)
		partID       = UniqueID(2)
		fieldID      = UniqueID(10)
		indexID      = UniqueID(100)
		segID        = UniqueID(1000)
		invalidSegID = UniqueID(1001)
		buildID      = UniqueID(10000)
		indexName    = "default_idx"
		typeParams   = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.GetIndexStatisticsRequest{
			CollectionID: collID,
			IndexName:    "",
		}
	)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.On("AlterIndexes",
		mock.Anything,
		mock.Anything,
	).Return(nil)

	mock0Allocator := newMockAllocator(t)

	segments := map[UniqueID]*SegmentInfo{
		invalidSegID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				NumOfRows:      10000,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					// timesamp > index start time, will be filtered out
					Timestamp: createTS + 1,
				},
			},
		},
		segID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:             segID,
				CollectionID:   collID,
				PartitionID:    partID,
				NumOfRows:      10000,
				State:          commonpb.SegmentState_Flushed,
				MaxRowNum:      65536,
				LastExpireTime: createTS,
				Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
				StartPosition: &msgpb.MsgPosition{
					Timestamp: createTS,
				},
			},
		},
	}
	b := broker.NewMockBroker(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status:         merr.Status(nil),
		DbName:         "test_db",
		CollectionName: "test_collection",
	}, nil)
	s := &Server{
		meta: &meta{
			catalog: catalog,
			indexMeta: &indexMeta{
				catalog: catalog,
				indexes: map[UniqueID]map[UniqueID]*model.Index{
					collID: {
						// finished
						indexID: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID,
							IndexName:       indexName,
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// deleted
						indexID + 1: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 1,
							IndexID:         indexID + 1,
							IndexName:       indexName + "_1",
							IsDeleted:       true,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 2: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 2,
							IndexID:         indexID + 2,
							IndexName:       indexName + "_2",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// inProgress
						indexID + 3: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 3,
							IndexID:         indexID + 3,
							IndexName:       indexName + "_3",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// failed
						indexID + 4: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 4,
							IndexID:         indexID + 4,
							IndexName:       indexName + "_4",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 5: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 5,
							IndexID:         indexID + 5,
							IndexName:       indexName + "_5",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
					},
				},
				segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			},

			segments: NewSegmentsInfo(),
		},
		mixCoord:        mocks.NewMixCoord(t),
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
		broker:          b,
	}
	segIdx1 := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx1.Insert(indexID, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID,
		BuildID:             buildID,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+1, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 1,
		BuildID:             buildID + 1,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+3, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 3,
		BuildID:             buildID + 3,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_InProgress,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+4, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 4,
		BuildID:             buildID + 4,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Failed,
		FailReason:          "mock failed",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	segIdx1.Insert(indexID+5, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID + 5,
		BuildID:             buildID + 5,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Unissued,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	s.meta.indexMeta.segmentIndexes.Insert(segID, segIdx1)
	for id, segment := range segments {
		s.meta.segments.SetSegment(id, segment)
	}
	RegisterDDLCallbacks(s)

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.GetIndexStatistics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_NotReadyServe, resp.GetStatus().GetErrorCode())
	})

	s.stateCode.Store(commonpb.StateCode_Healthy)

	t.Run("success", func(t *testing.T) {
		resp, err := s.GetIndexStatistics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, 5, len(resp.GetIndexInfos()))
	})

	t.Run("describe after drop index", func(t *testing.T) {
		s.mixCoord.(*mocks.MixCoord).EXPECT().ShowLoadCollections(mock.Anything, mock.Anything).Return(&querypb.ShowCollectionsResponse{
			Status:        merr.Success(),
			CollectionIDs: []int64{},
		}, nil)
		status, err := s.DropIndex(ctx, &indexpb.DropIndexRequest{
			CollectionID: collID,
			PartitionIDs: nil,
			IndexName:    "",
			DropAll:      true,
		})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())

		resp, err := s.GetIndexStatistics(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_IndexNotExist, resp.GetStatus().GetErrorCode())
	})
}

func TestServer_DropIndex(t *testing.T) {
	initStreamingSystem(t)
	var (
		collID     = UniqueID(1)
		partID     = UniqueID(2)
		fieldID    = UniqueID(10)
		indexID    = UniqueID(100)
		segID      = UniqueID(1000)
		indexName  = "default_idx"
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.DropIndexRequest{
			CollectionID: collID,
			IndexName:    indexName,
		}
	)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.On("AlterIndexes",
		mock.Anything,
		mock.Anything,
	).Return(nil)

	mock0Allocator := newMockAllocator(t)

	b := broker.NewMockBroker(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status:         merr.Status(nil),
		DbName:         "test_db",
		CollectionName: "test_collection",
	}, nil)

	s := &Server{
		meta: &meta{
			catalog: catalog,
			indexMeta: &indexMeta{
				catalog: catalog,
				indexes: map[UniqueID]map[UniqueID]*model.Index{
					collID: {
						// finished
						indexID: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID,
							IndexName:       indexName,
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// deleted
						indexID + 1: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 1,
							IndexID:         indexID + 1,
							IndexName:       indexName + "_1",
							IsDeleted:       true,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// unissued
						indexID + 2: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID + 2,
							IndexID:         indexID + 2,
							IndexName:       indexName + "_2",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// inProgress
						indexID + 3: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID + 3,
							IndexName:       indexName + "_3",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
						// failed
						indexID + 4: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID + 4,
							IndexName:       indexName + "_4",
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
					},
				},
				segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			},

			segments: NewSegmentsInfo(),
		},
		broker:          b,
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}

	mixCoord := mocks.NewMixCoord(t)
	mixCoord.EXPECT().ShowLoadCollections(mock.Anything, mock.Anything).Return(&querypb.ShowCollectionsResponse{
		Status:        merr.Success(),
		CollectionIDs: []int64{},
	}, nil)
	s.mixCoord = mixCoord

	s.meta.segments.SetSegment(segID, &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:             segID,
			CollectionID:   collID,
			PartitionID:    partID,
			NumOfRows:      10000,
			State:          commonpb.SegmentState_Flushed,
			MaxRowNum:      65536,
			LastExpireTime: createTS,
			Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
		},
	})

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.DropIndex(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp), merr.ErrServiceNotReady)
	})

	RegisterDDLCallbacks(s)
	s.stateCode.Store(commonpb.StateCode_Healthy)

	t.Run("drop fail", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		catalog.EXPECT().AlterIndexes(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, indexes []*model.Index) error {
			if rand.Intn(3) == 0 {
				return errors.New("fail")
			}
			return nil
		}).Maybe()
		s.meta.indexMeta.catalog = catalog
		resp, err := s.DropIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("drop one index", func(t *testing.T) {
		s.meta.indexMeta.catalog = catalog
		resp, err := s.DropIndex(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("drop one without indexName", func(t *testing.T) {
		req = &indexpb.DropIndexRequest{
			CollectionID: collID,
			PartitionIDs: nil,
			IndexName:    "",
			DropAll:      false,
		}
		resp, err := s.DropIndex(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.GetErrorCode())
	})

	t.Run("drop all indexes", func(t *testing.T) {
		req = &indexpb.DropIndexRequest{
			CollectionID: collID,
			PartitionIDs: nil,
			IndexName:    "",
			DropAll:      true,
		}
		resp, err := s.DropIndex(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("drop not exist index", func(t *testing.T) {
		req = &indexpb.DropIndexRequest{
			CollectionID: collID,
			PartitionIDs: nil,
			IndexName:    "",
			DropAll:      true,
		}
		resp, err := s.DropIndex(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}

func TestServer_DropIndex_DroppedField(t *testing.T) {
	initStreamingSystem(t)
	var (
		collID    = UniqueID(1)
		fieldID   = UniqueID(10)
		indexID   = UniqueID(100)
		indexName = "idx_dropped_field"
		createTS  = uint64(1000)
		ctx       = context.Background()
	)

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.On("AlterIndexes", mock.Anything, mock.Anything).Return(nil)

	// Broker returns a schema that does NOT contain fieldID=10 (simulating a dropped field)
	b := broker.NewMockBroker(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status:         merr.Status(nil),
		DbName:         "test_db",
		CollectionName: "test_collection",
		Schema: &schemapb.CollectionSchema{
			Name: "test_collection",
			Fields: []*schemapb.FieldSchema{
				{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
				{FieldID: 101, Name: "vec", DataType: schemapb.DataType_FloatVector},
				// fieldID=10 is intentionally absent — it has been dropped
			},
		},
	}, nil)

	s := &Server{
		meta: &meta{
			catalog: catalog,
			indexMeta: &indexMeta{
				catalog: catalog,
				indexes: map[UniqueID]map[UniqueID]*model.Index{
					collID: {
						indexID: {
							CollectionID: collID,
							FieldID:      fieldID, // points to dropped field
							IndexID:      indexID,
							IndexName:    indexName,
							IsDeleted:    false,
							CreateTime:   createTS,
							TypeParams: []*commonpb.KeyValuePair{
								{Key: common.DimKey, Value: "128"},
							},
							IndexParams: []*commonpb.KeyValuePair{
								{Key: common.IndexTypeKey, Value: "IVF_FLAT"},
							},
						},
					},
				},
				segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			},
			segments: NewSegmentsInfo(),
		},
		broker:          b,
		allocator:       newMockAllocator(t),
		notifyIndexChan: make(chan UniqueID, 1),
	}

	// Collection is loaded
	mixCoord := mocks.NewMixCoord(t)
	mixCoord.EXPECT().ShowLoadCollections(mock.Anything, mock.Anything).Return(&querypb.ShowCollectionsResponse{
		Status:        merr.Success(),
		CollectionIDs: []int64{collID},
	}, nil)
	s.mixCoord = mixCoord

	RegisterDDLCallbacks(s)
	s.stateCode.Store(commonpb.StateCode_Healthy)

	// DropIndex should succeed: field not in schema triggers continue, not error
	resp, err := s.DropIndex(ctx, &indexpb.DropIndexRequest{
		CollectionID: collID,
		IndexName:    indexName,
	})
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
}

func TestServer_GetIndexInfos(t *testing.T) {
	var (
		collID     = UniqueID(1)
		partID     = UniqueID(2)
		fieldID    = UniqueID(10)
		indexID    = UniqueID(100)
		segID      = UniqueID(1000)
		buildID    = UniqueID(10000)
		indexName  = "default_idx"
		typeParams = []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: "128",
			},
		}
		indexParams = []*commonpb.KeyValuePair{
			{
				Key:   common.IndexTypeKey,
				Value: "IVF_FLAT",
			},
		}
		createTS = uint64(1000)
		ctx      = context.Background()
		req      = &indexpb.GetIndexInfoRequest{
			CollectionID: collID,
			SegmentIDs:   []UniqueID{segID},
			IndexName:    indexName,
		}
	)

	chunkManagerFactory := storage.NewChunkManagerFactoryWithParam(Params)
	cli, err := chunkManagerFactory.NewPersistentStorageChunkManager(ctx)
	assert.NoError(t, err)

	mock0Allocator := newMockAllocator(t)

	s := &Server{
		meta: &meta{
			catalog: &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
			indexMeta: &indexMeta{
				catalog: &datacoord.Catalog{MetaKv: mockkv.NewMetaKv(t)},
				indexes: map[UniqueID]map[UniqueID]*model.Index{
					collID: {
						// finished
						indexID: {
							TenantID:        "",
							CollectionID:    collID,
							FieldID:         fieldID,
							IndexID:         indexID,
							IndexName:       indexName,
							IsDeleted:       false,
							CreateTime:      createTS,
							TypeParams:      typeParams,
							IndexParams:     indexParams,
							IsAutoIndex:     false,
							UserIndexParams: nil,
						},
					},
				},
				segmentIndexes: typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			},

			segments:     NewSegmentsInfo(),
			chunkManager: cli,
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
	}
	segIdx1 := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	segIdx1.Insert(indexID, &model.SegmentIndex{
		SegmentID:           segID,
		CollectionID:        collID,
		PartitionID:         partID,
		NumRows:             10000,
		IndexID:             indexID,
		BuildID:             buildID,
		NodeID:              0,
		IndexVersion:        1,
		IndexState:          commonpb.IndexState_Finished,
		FailReason:          "",
		IsDeleted:           false,
		CreatedUTCTime:      createTS,
		IndexFileKeys:       nil,
		IndexSerializedSize: 0,
		WriteHandoff:        false,
	})
	s.meta.indexMeta.segmentIndexes.Insert(segID, segIdx1)
	s.meta.segments.SetSegment(segID, &SegmentInfo{
		SegmentInfo: &datapb.SegmentInfo{
			ID:             segID,
			CollectionID:   collID,
			PartitionID:    partID,
			NumOfRows:      10000,
			State:          commonpb.SegmentState_Flushed,
			MaxRowNum:      65536,
			LastExpireTime: createTS,
			Stats:          &datapb.Statistics{TimestampFrom: createTS, TimestampTo: createTS},
		},
	})

	t.Run("server not available", func(t *testing.T) {
		s.stateCode.Store(commonpb.StateCode_Initializing)
		resp, err := s.GetIndexInfos(ctx, req)
		assert.NoError(t, err)
		assert.ErrorIs(t, merr.Error(resp.GetStatus()), merr.ErrServiceNotReady)
	})

	s.stateCode.Store(commonpb.StateCode_Healthy)
	t.Run("get segment index infos", func(t *testing.T) {
		resp, err := s.GetIndexInfos(ctx, req)
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
		assert.Equal(t, 1, len(resp.GetSegmentInfo()))
	})
}

func TestMeta_GetHasUnindexTaskSegments(t *testing.T) {
	var (
		collID    = UniqueID(1)
		partID    = UniqueID(2)
		segID     = UniqueID(1000)
		indexID   = UniqueID(100)
		fieldID   = UniqueID(10)
		indexName = "default_idx"
	)
	segments := map[UniqueID]*SegmentInfo{
		segID: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:            segID,
				CollectionID:  collID,
				PartitionID:   partID,
				InsertChannel: "",
				NumOfRows:     1025,
				State:         commonpb.SegmentState_Flushed,
			},
		},
		segID + 1: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:            segID + 1,
				CollectionID:  collID,
				PartitionID:   partID,
				InsertChannel: "",
				NumOfRows:     1025,
				State:         commonpb.SegmentState_Growing,
			},
		},
		segID + 2: {
			SegmentInfo: &datapb.SegmentInfo{
				ID:            segID + 2,
				CollectionID:  collID,
				PartitionID:   partID,
				InsertChannel: "",
				NumOfRows:     1025,
				State:         commonpb.SegmentState_Dropped,
			},
		},
	}
	m := &meta{
		segments: NewSegmentsInfo(),
		indexMeta: &indexMeta{
			segmentBuildInfo: newSegmentIndexBuildInfo(),
			segmentIndexes:   typeutil.NewConcurrentMap[UniqueID, *typeutil.ConcurrentMap[UniqueID, *model.SegmentIndex]](),
			indexes: map[UniqueID]map[UniqueID]*model.Index{
				collID: {
					indexID: {
						TenantID:        "",
						CollectionID:    collID,
						FieldID:         fieldID,
						IndexID:         indexID,
						IndexName:       indexName,
						IsDeleted:       false,
						CreateTime:      0,
						TypeParams:      nil,
						IndexParams:     nil,
						IsAutoIndex:     false,
						UserIndexParams: nil,
					},
					indexID + 1: {
						TenantID:        "",
						CollectionID:    collID,
						FieldID:         fieldID + 1,
						IndexID:         indexID + 1,
						IndexName:       indexName + "_1",
						IsDeleted:       false,
						CreateTime:      0,
						TypeParams:      nil,
						IndexParams:     nil,
						IsAutoIndex:     false,
						UserIndexParams: nil,
					},
				},
			},
		},
	}
	for id, segment := range segments {
		m.segments.SetSegment(id, segment)
	}
	indexInspector := &indexInspector{
		meta: m,
	}

	t.Run("normal", func(t *testing.T) {
		segments := indexInspector.getUnIndexTaskSegments(context.TODO())
		assert.Equal(t, 1, len(segments))
		assert.Equal(t, segID, segments[0].ID)

		m.indexMeta.segmentIndexes.Insert(segID, typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]())
		m.indexMeta.updateSegmentIndex(&model.SegmentIndex{
			CollectionID: collID,
			SegmentID:    segID,
			IndexID:      indexID + 2,
			IndexState:   commonpb.IndexState_Finished,
		})
		assert.Equal(t, 1, len(segments))
		assert.Equal(t, segID, segments[0].ID)
	})

	t.Run("segment partial field with index", func(t *testing.T) {
		m.indexMeta.updateSegmentIndex(&model.SegmentIndex{
			CollectionID: collID,
			SegmentID:    segID,
			IndexID:      indexID,
			IndexState:   commonpb.IndexState_Finished,
		})

		segments := indexInspector.getUnIndexTaskSegments(context.TODO())
		assert.Equal(t, 1, len(segments))
		assert.Equal(t, segID, segments[0].ID)
	})

	t.Run("segment all vector field with index", func(t *testing.T) {
		m.indexMeta.updateSegmentIndex(&model.SegmentIndex{
			CollectionID: collID,
			SegmentID:    segID,
			IndexID:      indexID + 1,
			IndexState:   commonpb.IndexState_Finished,
		})

		segments := indexInspector.getUnIndexTaskSegments(context.TODO())
		assert.Equal(t, 0, len(segments))
	})
}

func TestJsonIndex(t *testing.T) {
	initStreamingSystem(t)

	collID := UniqueID(1)
	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(nil).Maybe()
	mock0Allocator := newMockAllocator(t)
	indexMeta := newSegmentIndexMeta(catalog)
	b := mocks.NewMixCoord(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status: &commonpb.Status{
			ErrorCode: 0,
			Code:      0,
		},
		Schema: &schemapb.CollectionSchema{
			Name: "test_index",
			Fields: []*schemapb.FieldSchema{
				{
					FieldID:  0,
					Name:     "json",
					DataType: schemapb.DataType_JSON,
				},
				{
					FieldID:  1,
					Name:     "json2",
					DataType: schemapb.DataType_JSON,
				},
				{
					FieldID:   2,
					Name:      "dynamic",
					DataType:  schemapb.DataType_JSON,
					IsDynamic: true,
				},
			},
		},
	}, nil)

	collections := typeutil.NewConcurrentMap[UniqueID, *collectionInfo]()
	collections.Insert(collID, &collectionInfo{
		ID: collID,
	})

	s := &Server{
		meta: &meta{
			catalog:     catalog,
			collections: collections,
			indexMeta:   indexMeta,
		},
		allocator:       mock0Allocator,
		notifyIndexChan: make(chan UniqueID, 1),
		broker:          broker.NewCoordinatorBroker(b),
	}
	RegisterDDLCallbacks(s)
	s.stateCode.Store(commonpb.StateCode_Healthy)

	req := &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "a",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "varchar"}, {Key: common.JSONPathKey, Value: "json[\"a\"]"}},
	}
	resp, err := s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "varchar"}, {Key: common.JSONPathKey, Value: "json[\"c\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// different json field with same json path
	req = &indexpb.CreateIndexRequest{
		FieldID:     1,
		IndexName:   "",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "varchar"}, {Key: common.JSONPathKey, Value: "json2[\"c\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// duplicated index with same params
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "a",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "varchar"}, {Key: common.JSONPathKey, Value: "json[\"a\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// duplicated index with different cast type
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "a",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[\"a\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// duplicated index with different index name
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "b",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[\"a\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// another field json index with same index name
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "a",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[\"b\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// lack of json params
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "a",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONPathKey, Value: "json[\"a\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// incorrect field name in json path
	req = &indexpb.CreateIndexRequest{
		FieldID:     1,
		IndexName:   "c",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "bad_json[\"a\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// dynamic field
	req = &indexpb.CreateIndexRequest{
		FieldID:     2,
		IndexName:   "",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "dynamic_a_field"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// wrong path: missing quotes
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "d",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[a][\"b\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// wrong path: missing closing quote
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "e",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[\"a\"][\"b"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// wrong path: malformed brackets
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "f",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[\"a\"[\"b]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))

	// valid path with array index
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "g",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: "double"}, {Key: common.JSONPathKey, Value: "json[\"a\"][0][\"b\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// test json flat index
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "h",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: strconv.Itoa(int(schemapb.DataType_JSON))}, {Key: common.JSONPathKey, Value: "json[\"a\"][\"b\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// test json flat index with dynamic field
	req = &indexpb.CreateIndexRequest{
		FieldID:     2,
		IndexName:   "i",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: strconv.Itoa(int(schemapb.DataType_JSON))}, {Key: common.JSONPathKey, Value: "dynamic"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.NoError(t, merr.CheckRPCCall(resp, err))

	// duplicated json flat index
	req = &indexpb.CreateIndexRequest{
		FieldID:     0,
		IndexName:   "a",
		IndexParams: []*commonpb.KeyValuePair{{Key: common.JSONCastTypeKey, Value: strconv.Itoa(int(schemapb.DataType_JSON))}, {Key: common.JSONPathKey, Value: "json[\"a\"]"}},
	}
	resp, err = s.CreateIndex(context.Background(), req)
	assert.Error(t, merr.CheckRPCCall(resp, err))
}

// Test_checkFMIndexEngineVersion covers the shared FMINDEX rolling-upgrade gate
// used by BOTH Server.CreateIndex and snapshotManager.RestoreIndexes (so a
// snapshot restore cannot bypass it). MinScalarIndexVersionForFMINDEX is 5:
// resolved version 4 must be rejected, 5 accepted; non-FMINDEX always passes.
func Test_checkFMIndexEngineVersion(t *testing.T) {
	fmParams := []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "FMINDEX"}}
	invertedParams := []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "INVERTED"}}

	t.Run("fmindex below min version rejected", func(t *testing.T) {
		err := checkFMIndexEngineVersion(fmParams, common.MinScalarIndexVersionForFMINDEX-1)
		assert.Error(t, err)
		assert.ErrorIs(t, err, merr.ErrServiceNotReady)
		assert.Contains(t, err.Error(), "FMINDEX requires scalar index engine version")
		assert.True(t, merr.Status(err).GetRetriable())
	})

	t.Run("fmindex at min version accepted", func(t *testing.T) {
		assert.NoError(t, checkFMIndexEngineVersion(fmParams, common.MinScalarIndexVersionForFMINDEX))
	})

	t.Run("fmindex above min version accepted", func(t *testing.T) {
		assert.NoError(t, checkFMIndexEngineVersion(fmParams, common.MinScalarIndexVersionForFMINDEX+1))
	})

	t.Run("non-fmindex always accepted regardless of version", func(t *testing.T) {
		assert.NoError(t, checkFMIndexEngineVersion(invertedParams, 0))
	})
}

// newIndexMetaForMultiIndexTest builds an indexMeta holding exactly the given
// indexes. The checks under test only read the in-memory map, so a nil catalog is
// fine: no metadata write goes through it.
func newIndexMetaForMultiIndexTest(indexes ...*model.Index) *indexMeta {
	m := newSegmentIndexMeta(nil)
	for _, index := range indexes {
		if m.indexes[index.CollectionID] == nil {
			m.indexes[index.CollectionID] = make(map[UniqueID]*model.Index)
		}
		m.indexes[index.CollectionID][index.IndexID] = index
	}
	return m
}

// storedIndex builds an index as it would be persisted by CreateIndex.
func storedIndex(collID, fieldID, indexID int64, name string, params ...*commonpb.KeyValuePair) *model.Index {
	return &model.Index{
		CollectionID:    collID,
		FieldID:         fieldID,
		IndexID:         indexID,
		IndexName:       name,
		IndexParams:     params,
		UserIndexParams: params,
	}
}

// setMultiScalarIndexPerField toggles the operator switch that allows a second
// scalar index on one field (dataCoord.enableMultiScalarIndexPerField), restoring the
// previous value when the (sub)test ends. Same SwapTempValue pattern the
// neighbouring datacoord tests use to flip a ParamItem.
// setMultiScalarIndexPerField sets the three-valued capability switch that allows
// a second scalar index on one field: "auto" resolves from the scalar index engine
// version the cluster reports, "false" refuses it outright and "true" forces it on.
func ptr[T any](v T) *T { return &v }

func setMultiScalarIndexPerField(t *testing.T, value string) {
	orig := Params.DataCoordCfg.EnableMultiScalarIndexPerField.SwapTempValue(value)
	t.Cleanup(func() {
		Params.DataCoordCfg.EnableMultiScalarIndexPerField.SwapTempValue(orig)
	})
}

// TestServer_checkIndexCreationPolicy pins the field-level creation rules:
//
//   - a second index on the same SCALAR field needs BOTH the operator switch
//     (dataCoord.enableMultiScalarIndexPerField, three-valued: auto/false/true) and a whole
//     cluster reporting scalar
//     the version gate has activated, which the switcher confirms cluster-wide (an older
//     QueryNode asserts while loading two scalar indexes on one field);
//   - with the switch off the second scalar index is rejected even when the
//     version is new enough, and the refusal names the config key and is not
//     retriable (only an operator can clear it);
//   - below that version the second scalar index is rejected and the error is
//     merr.ErrServiceNotReady, so the caller can retry after the upgrade;
//   - a second index on a VECTOR field is rejected unconditionally as an
//     input-class error, without consulting the version or the switch;
//   - a second JSON index on the SAME path is gated, a JSON index on a DIFFERENT
//     path is not (the pre-existing multi-path capability).
func TestServer_checkIndexCreationPolicy(t *testing.T) {
	const collID = UniqueID(1)

	invertedParams := []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "INVERTED"}}
	jsonPathA := []*commonpb.KeyValuePair{
		{Key: common.IndexTypeKey, Value: "INVERTED"},
		{Key: common.JSONPathKey, Value: `json["a"]`},
		{Key: common.JSONCastTypeKey, Value: "varchar"},
	}

	m := newIndexMetaForMultiIndexTest(
		storedIndex(collID, 100, 1000, "scalar_first", invertedParams...),
		storedIndex(collID, 101, 1001, "vector_first", invertedParams...),
		storedIndex(collID, 102, 1002, "json_first", jsonPathA...),
	)

	// serverWithVersion returns a Server whose version manager resolves to the
	// given scalar index engine version. version == nil registers no expectation at
	// all, so a ResolveScalarIndexVersion call fails the test -- that is how the
	// "not gated at all" cases assert the version is never read.
	serverWithVersion := func(version *int32) *Server {
		vm := NewMockVersionManager(t)
		if version != nil {
			vm.On("ResolveScalarIndexVersion").Return(*version).Maybe()
		}
		return &Server{
			meta:                      &meta{indexMeta: m},
			indexEngineVersionManager: vm,
		}
	}

	req := func(fieldID int64, params ...*commonpb.KeyValuePair) *indexpb.CreateIndexRequest {
		return &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexName:       "second",
			IndexParams:     params,
			UserIndexParams: params,
		}
	}
	ctx := context.Background()

	t.Run("second scalar index rejected while the switch is off", func(t *testing.T) {
		// the version is new enough, so the explicit "false" is the only reason to
		// refuse: the escape hatch has to work even on a fully upgraded cluster
		setMultiScalarIndexPerField(t, "false")
		version := common.CurrentScalarIndexEngineVersion
		s := serverWithVersion(&version)
		err := s.checkIndexCreationPolicy(ctx, req(100, invertedParams...), false, schemapb.DataType_Int64)
		assert.Error(t, err)
		assert.NotErrorIs(t, err, merr.ErrServiceNotReady)
		assert.Equal(t, merr.Code(merr.ErrOperationNotSupported), merr.Code(err))
		assert.Contains(t, err.Error(), "dataCoord.enableMultiScalarIndexPerField")
		// an operator condition, not a transient one: a client retry cannot clear it
		assert.False(t, merr.Status(err).GetRetriable())
	})

	t.Run("auto stays off until the version gate activates", func(t *testing.T) {
		// "auto" is resolved by VersionGateSwitcher: until the whole cluster reaches
		// the gate version it resolves to PreSwitchValue ("false"), which is the same
		// refusal an explicit false produces. A new-enough QueryNode version alone
		// must NOT flip it -- the switcher owns that decision.
		setMultiScalarIndexPerField(t, "auto")
		version := common.CurrentScalarIndexEngineVersion
		s := serverWithVersion(&version)
		err := s.checkIndexCreationPolicy(ctx, req(100, invertedParams...), false, schemapb.DataType_Int64)
		assert.ErrorIs(t, err, merr.ErrOperationNotSupported)
		assert.Contains(t, err.Error(), "dataCoord.enableMultiScalarIndexPerField")
	})

	t.Run("switch true bypasses the readiness check", func(t *testing.T) {
		// "true" is the documented force-enable path: it must not depend on what the
		// cluster reports, so an empty version still passes.
		setMultiScalarIndexPerField(t, "true")
		s := serverWithVersion(nil)
		err := s.checkIndexCreationPolicy(ctx, req(100, invertedParams...), false, schemapb.DataType_Int64)
		assert.NoError(t, err)
	})

	t.Run("auto refuses for any reported engine version until the gate opens", func(t *testing.T) {
		// The reported version no longer decides anything by itself; the switcher's
		// cluster-version confirmation does. Both an unknown and a new-enough version
		// resolve to the same refusal while the gate is closed.
		setMultiScalarIndexPerField(t, "auto")
		for _, version := range []*int32{nil, ptr(common.CurrentScalarIndexEngineVersion)} {
			s := serverWithVersion(version)
			err := s.checkIndexCreationPolicy(ctx, req(100, invertedParams...), false, schemapb.DataType_Int64)
			assert.ErrorIs(t, err, merr.ErrOperationNotSupported)
		}
	})

	t.Run("second vector index rejected regardless of version", func(t *testing.T) {
		s := serverWithVersion(nil)
		err := s.checkIndexCreationPolicy(ctx, req(101, invertedParams...), false, schemapb.DataType_FloatVector)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Equal(t, merr.Code(merr.ErrParameterInvalid), merr.Code(err))
		assert.Contains(t, err.Error(), "creating multiple indexes on a vector field is not supported")
	})

	t.Run("json index on a different path is never gated", func(t *testing.T) {
		s := serverWithVersion(nil)
		otherPath := []*commonpb.KeyValuePair{
			{Key: common.IndexTypeKey, Value: "INVERTED"},
			{Key: common.JSONPathKey, Value: `json["b"]`},
			{Key: common.JSONCastTypeKey, Value: "varchar"},
		}
		err := s.checkIndexCreationPolicy(ctx, req(102, otherPath...), true, schemapb.DataType_JSON)
		assert.NoError(t, err)
	})

	t.Run("json index on the same path is gated", func(t *testing.T) {
		// Same identity (field + json_path) means the same "second index" shape, so it
		// rides the same switch: auto is off until the version gate activates.
		setMultiScalarIndexPerField(t, "auto")
		s := serverWithVersion(ptr(common.CurrentScalarIndexEngineVersion))
		err := s.checkIndexCreationPolicy(ctx, req(102, jsonPathA...), true, schemapb.DataType_JSON)
		assert.ErrorIs(t, err, merr.ErrOperationNotSupported)

		// forcing it on makes both the scalar and the same-path JSON case pass
		setMultiScalarIndexPerField(t, "true")
		err = s.checkIndexCreationPolicy(ctx, req(102, jsonPathA...), true, schemapb.DataType_JSON)
		assert.NoError(t, err)
	})

	t.Run("field without any index is never gated", func(t *testing.T) {
		s := serverWithVersion(nil)
		assert.NoError(t, s.checkIndexCreationPolicy(ctx, req(103, invertedParams...), false, schemapb.DataType_Int64))
		assert.NoError(t, s.checkIndexCreationPolicy(ctx, req(103, invertedParams...), false, schemapb.DataType_FloatVector))
	})
}

// TestServer_resolveIndexName pins the name-resolution rules. The
// name is the only identity a user has for an index, so resolution must stay
// idempotent for a replay and mint a unique name for a genuinely new index on a
// field that already carries one.
func TestServer_resolveIndexName(t *testing.T) {
	const (
		collID       = UniqueID(1)
		fieldID      = UniqueID(100)
		otherFieldID = UniqueID(101)
		jsonFieldID  = UniqueID(102)
		jsonPathA    = `json["a"]`
	)

	schema := &schemapb.CollectionSchema{
		Name: "multi_index",
		Fields: []*schemapb.FieldSchema{
			{FieldID: fieldID, Name: "f100", DataType: schemapb.DataType_Int64},
			{FieldID: otherFieldID, Name: "f101", DataType: schemapb.DataType_Int64},
			{FieldID: jsonFieldID, Name: "json", DataType: schemapb.DataType_JSON},
		},
	}

	indexType := func(t string) []*commonpb.KeyValuePair {
		return []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: t}}
	}
	jsonParams := func(path, castType string) []*commonpb.KeyValuePair {
		return []*commonpb.KeyValuePair{
			{Key: common.IndexTypeKey, Value: "INVERTED"},
			{Key: common.JSONPathKey, Value: path},
			{Key: common.JSONCastTypeKey, Value: castType},
		}
	}
	req := func(fieldID int64, params []*commonpb.KeyValuePair) *indexpb.CreateIndexRequest {
		return &indexpb.CreateIndexRequest{
			CollectionID:    collID,
			FieldID:         fieldID,
			IndexParams:     params,
			UserIndexParams: params,
		}
	}
	resolve := func(m *indexMeta, r *indexpb.CreateIndexRequest, isJSON bool) (string, error) {
		s := &Server{meta: &meta{indexMeta: m}}
		return s.resolveIndexName(r, schema, isJSON)
	}

	t.Run("identical index on the same field reuses its name", func(t *testing.T) {
		m := newIndexMetaForMultiIndexTest(storedIndex(collID, fieldID, 1000, "user_named", indexType("INVERTED")...))
		name, err := resolve(m, req(fieldID, indexType("INVERTED")), false)
		assert.NoError(t, err)
		assert.Equal(t, "user_named", name)

		// replaying the very same request must resolve to the very same name,
		// otherwise a retry would create a second index on the field.
		replay, err := resolve(m, req(fieldID, indexType("INVERTED")), false)
		assert.NoError(t, err)
		assert.Equal(t, name, replay)
	})

	t.Run("json index on the same path and cast reuses its name", func(t *testing.T) {
		// the stored name is exactly the base name the resolver derives
		// (field name + json path), so a mint would be observable.
		m := newIndexMetaForMultiIndexTest(storedIndex(collID, jsonFieldID, 1000, "json"+jsonPathA, jsonParams(jsonPathA, "varchar")...))
		name, err := resolve(m, req(jsonFieldID, jsonParams(jsonPathA, "varchar")), true)
		assert.NoError(t, err)
		assert.Equal(t, "json"+jsonPathA, name)
	})

	t.Run("json index on the same path with another cast mints a name", func(t *testing.T) {
		m := newIndexMetaForMultiIndexTest(storedIndex(collID, jsonFieldID, 1000, "json"+jsonPathA, jsonParams(jsonPathA, "varchar")...))
		name, err := resolve(m, req(jsonFieldID, jsonParams(jsonPathA, "double")), true)
		assert.NoError(t, err)
		assert.Equal(t, "json"+jsonPathA+"_inverted", name)
	})

	t.Run("taken base name gets the index type suffix", func(t *testing.T) {
		m := newIndexMetaForMultiIndexTest(storedIndex(collID, fieldID, 1000, "f100", indexType("BITMAP")...))
		name, err := resolve(m, req(fieldID, indexType("STL_SORT")), false)
		assert.NoError(t, err)
		assert.Equal(t, "f100_stl_sort", name)
	})

	t.Run("name uniqueness is collection scoped", func(t *testing.T) {
		// the base name is taken by an index on ANOTHER field: names are unique per
		// collection, so the request must not silently reuse "f100".
		m := newIndexMetaForMultiIndexTest(storedIndex(collID, otherFieldID, 1000, "f100", indexType("INVERTED")...))
		name, err := resolve(m, req(fieldID, indexType("INVERTED")), false)
		assert.NoError(t, err)
		assert.Equal(t, "f100_inverted", name)
	})

	t.Run("ordinal fallback when the type suffix is taken", func(t *testing.T) {
		m := newIndexMetaForMultiIndexTest(
			storedIndex(collID, fieldID, 1000, "f100", indexType("INVERTED")...),
			// AUTOINDEX + a different metric: not identical to the request below,
			// so it is just a name occupying "f100_autoindex".
			storedIndex(collID, fieldID, 1001, "f100_autoindex",
				&commonpb.KeyValuePair{Key: common.IndexTypeKey, Value: common.AutoIndexName},
				&commonpb.KeyValuePair{Key: common.MetricTypeKey, Value: "L2"}),
		)
		name, err := resolve(m, req(fieldID, indexType(common.AutoIndexName)), false)
		assert.NoError(t, err)
		assert.Equal(t, "f100_autoindex_2", name)
	})

	t.Run("no index type falls back to the ordinal suffix", func(t *testing.T) {
		m := newIndexMetaForMultiIndexTest(storedIndex(collID, fieldID, 1000, "f100", indexType("INVERTED")...))
		name, err := resolve(m, req(fieldID, nil), false)
		assert.NoError(t, err)
		assert.Equal(t, "f100_2", name)
	})

	t.Run("no index type skips an ordinal that is already taken", func(t *testing.T) {
		m := newIndexMetaForMultiIndexTest(
			storedIndex(collID, fieldID, 1000, "f100", indexType("INVERTED")...),
			storedIndex(collID, fieldID, 1001, "f100_2", indexType("BITMAP")...),
		)
		name, err := resolve(m, req(fieldID, nil), false)
		assert.NoError(t, err)
		assert.Equal(t, "f100_3", name)
	})
}

// The multi-index test schema: one scalar, one vector and one JSON field, served
// by the stubbed broker of newMultiIndexPolicyTestServer.
const (
	multiIndexCollID    = UniqueID(1)
	multiIndexScalarFID = UniqueID(10)
	multiIndexVectorFID = UniqueID(11)
	multiIndexJSONFID   = UniqueID(12)
)

// newMultiIndexPolicyTestServer builds a healthy DataCoord Server over the
// multi-index test schema. The returned setter changes the scalar index engine
// version the (mocked) version manager resolves, so a test can simulate a cluster
// that still has a QueryNode from before the upgrade.
//
// It leaves the capability switch at its "auto" default (restored when the test
// ends): a second scalar index on one field is then allowed exactly when the
// cluster reports a new enough version, which is the half of the gate these tests
// cover.
func newMultiIndexPolicyTestServer(t *testing.T) (*Server, func(int32)) {
	initStreamingSystem(t)
	setMultiScalarIndexPerField(t, "auto")

	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(nil).Maybe()

	collections := typeutil.NewConcurrentMap[UniqueID, *collectionInfo]()
	collections.Insert(multiIndexCollID, &collectionInfo{ID: multiIndexCollID})

	resolvedVersion := common.CurrentScalarIndexEngineVersion
	mockVM := NewMockVersionManager(t)
	mockVM.EXPECT().ResolveScalarIndexVersion().RunAndReturn(func() int32 { return resolvedVersion }).Maybe()

	s := &Server{
		meta: &meta{
			catalog:     catalog,
			collections: collections,
			indexMeta:   newSegmentIndexMeta(catalog),
		},
		allocator:                 newMockAllocator(t),
		notifyIndexChan:           make(chan UniqueID, 1),
		indexEngineVersionManager: mockVM,
	}
	RegisterDDLCallbacks(s)
	s.stateCode.Store(commonpb.StateCode_Healthy)

	b := mocks.NewMixCoord(t)
	b.EXPECT().DescribeCollectionInternal(mock.Anything, mock.Anything).Return(&milvuspb.DescribeCollectionResponse{
		Status: merr.Success(),
		Schema: &schemapb.CollectionSchema{
			Name: "multi_index",
			Fields: []*schemapb.FieldSchema{
				{FieldID: multiIndexScalarFID, Name: "scalar", DataType: schemapb.DataType_Int64},
				{
					FieldID: multiIndexVectorFID, Name: "vec", DataType: schemapb.DataType_FloatVector,
					TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "128"}},
				},
				{FieldID: multiIndexJSONFID, Name: "json", DataType: schemapb.DataType_JSON},
			},
		},
		CollectionID: multiIndexCollID,
	}, nil)
	s.broker = broker.NewCoordinatorBroker(b)

	return s, func(version int32) { resolvedVersion = version }
}

func multiIndexIndexType(indexType string) []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: indexType}}
}

func multiIndexDimParams() []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "128"}}
}

func multiIndexJSONParams(path string) []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{
		{Key: common.JSONPathKey, Value: path},
		{Key: common.JSONCastTypeKey, Value: "varchar"},
	}
}

func multiIndexCreateReq(fieldID int64, name string, params, typeParams []*commonpb.KeyValuePair) *indexpb.CreateIndexRequest {
	return &indexpb.CreateIndexRequest{
		CollectionID:    multiIndexCollID,
		FieldID:         fieldID,
		IndexName:       name,
		TypeParams:      typeParams,
		IndexParams:     params,
		UserIndexParams: params,
		Timestamp:       100,
	}
}

// TestServer_CreateIndex_MultiIndexPolicy covers G1 and G2 end to end through
// Server.CreateIndex: the policy runs before any metadata is written, so the
// rejection is visible to the client and no index is persisted.
func TestServer_CreateIndex_MultiIndexPolicy(t *testing.T) {
	s, setResolvedVersion := newMultiIndexPolicyTestServer(t)
	ctx := context.Background()

	t.Run("first index on the scalar field accepted", func(t *testing.T) {
		resp, err := s.CreateIndex(ctx, multiIndexCreateReq(multiIndexScalarFID, "scalar_inverted", multiIndexIndexType("INVERTED"), nil))
		assert.NoError(t, merr.CheckRPCCall(resp, err))
	})

	t.Run("second index on the scalar field accepted once the gate has opened", func(t *testing.T) {
		// The version gate opening is what turns the switch from its "auto" ->
		// "false" pre-switch value into "true"; a unit test emulates the post-gate
		// state directly, which is exactly what a cluster-wide confirmation produces.
		setMultiScalarIndexPerField(t, "true")
		defer setMultiScalarIndexPerField(t, "auto")

		resp, err := s.CreateIndex(ctx, multiIndexCreateReq(multiIndexScalarFID, "scalar_stl_sort", multiIndexIndexType("STL_SORT"), nil))
		assert.NoError(t, merr.CheckRPCCall(resp, err))
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexScalarFID, ""), 2)
	})

	t.Run("second index on the scalar field rejected while the switch is off", func(t *testing.T) {
		// an explicit "false" is the only reason to refuse here: the cluster is on a
		// version that could serve it
		setResolvedVersion(common.CurrentScalarIndexEngineVersion)
		setMultiScalarIndexPerField(t, "false")

		resp, err := s.CreateIndex(ctx, multiIndexCreateReq(multiIndexScalarFID, "scalar_bitmap_off", multiIndexIndexType("BITMAP"), nil))
		assert.ErrorIs(t, merr.CheckRPCCall(resp, err), merr.ErrOperationNotSupported)
		assert.Contains(t, resp.GetReason(), "dataCoord.enableMultiScalarIndexPerField")
		// an operator condition, not a transient one: a client retry cannot clear it
		assert.False(t, resp.GetRetriable())
		// nothing was written
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexScalarFID, ""), 2)
	})

	t.Run("auto refuses until the gate opens, whatever the engine version reports", func(t *testing.T) {
		// The switcher owns the decision; the engine version a QueryNode reports no
		// longer decides anything on its own. A new-enough report must not be enough.
		setResolvedVersion(common.CurrentScalarIndexEngineVersion)

		resp, err := s.CreateIndex(ctx, multiIndexCreateReq(multiIndexScalarFID, "scalar_bitmap", multiIndexIndexType("BITMAP"), nil))
		assert.ErrorIs(t, merr.CheckRPCCall(resp, err), merr.ErrOperationNotSupported)
		assert.Contains(t, resp.GetReason(), "dataCoord.enableMultiScalarIndexPerField")
		// nothing was written
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexScalarFID, ""), 2)
	})

	t.Run("json indexes on different paths are not gated", func(t *testing.T) {
		setResolvedVersion(common.CurrentScalarIndexEngineVersion - 1)
		defer setResolvedVersion(common.CurrentScalarIndexEngineVersion)

		resp, err := s.CreateIndex(ctx, multiIndexCreateReq(multiIndexJSONFID, "json_a", multiIndexJSONParams(`json["a"]`), nil))
		assert.NoError(t, merr.CheckRPCCall(resp, err))
		// a second path on the same json field is the pre-existing capability
		resp, err = s.CreateIndex(ctx, multiIndexCreateReq(multiIndexJSONFID, "json_b", multiIndexJSONParams(`json["b"]`), nil))
		assert.NoError(t, merr.CheckRPCCall(resp, err))
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexJSONFID, ""), 2)
	})

	t.Run("second index on the vector field rejected", func(t *testing.T) {
		resp, err := s.CreateIndex(ctx, multiIndexCreateReq(multiIndexVectorFID, "vec_ivf", multiIndexIndexType("IVF_FLAT"), multiIndexDimParams()))
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		resp, err = s.CreateIndex(ctx, multiIndexCreateReq(multiIndexVectorFID, "vec_hnsw", multiIndexIndexType("HNSW"), multiIndexDimParams()))
		assert.ErrorIs(t, merr.CheckRPCCall(resp, err), merr.ErrParameterInvalid)
		assert.Contains(t, resp.GetReason(), "creating multiple indexes on a vector field is not supported")
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexVectorFID, ""), 1)
	})
}

// TestServer_CreateIndex_IdenticalReplayIsNotASecondIndex pins the pre-existing
// invariant that replaying a CreateIndex stays idempotent -- same name, same
// field, same params -> a no-op (TestServer_CreateIndex/success_with_index_exist
// has covered the vector case long before this feature). A client retry, an SDK
// retry, a DDL replay and a snapshot restore all rely on it.
//
// It goes through Server.CreateIndex, so it pins the wiring as well: the policy's
// first pass recognises the replay and returns before the readiness gate,
// resolveIndexName reuses the existing name, and canCreateIndex reports
// errIndexOperationIgnored.
func TestServer_CreateIndex_IdenticalReplayIsNotASecondIndex(t *testing.T) {
	s, setResolvedVersion := newMultiIndexPolicyTestServer(t)
	ctx := context.Background()

	t.Run("identical replay on a vector field stays idempotent", func(t *testing.T) {
		req := multiIndexCreateReq(multiIndexVectorFID, "vec_replay", multiIndexIndexType("IVF_FLAT"), multiIndexDimParams())
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		// the very same request again: no new index, the call is a no-op
		replay := multiIndexCreateReq(multiIndexVectorFID, "vec_replay", multiIndexIndexType("IVF_FLAT"), multiIndexDimParams())
		resp, err = s.CreateIndex(ctx, replay)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexVectorFID, ""), 1)
	})

	t.Run("identical replay on a scalar field does not need version 6", func(t *testing.T) {
		req := multiIndexCreateReq(multiIndexScalarFID, "scalar_replay", multiIndexIndexType("INVERTED"), nil)
		resp, err := s.CreateIndex(ctx, req)
		assert.NoError(t, merr.CheckRPCCall(resp, err))

		// a replay creates a second index only if it is NOT recognised as a
		// replay, so the rolling-upgrade gate must not apply to it
		setResolvedVersion(common.CurrentScalarIndexEngineVersion - 1)
		defer setResolvedVersion(common.CurrentScalarIndexEngineVersion)

		replay := multiIndexCreateReq(multiIndexScalarFID, "scalar_replay", multiIndexIndexType("INVERTED"), nil)
		resp, err = s.CreateIndex(ctx, replay)
		assert.NoError(t, merr.CheckRPCCall(resp, err))
		assert.Len(t, s.meta.indexMeta.GetFieldIndexes(multiIndexCollID, multiIndexScalarFID, ""), 1)
	})
}
