package querycoordv2

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/metacache"
	catalogkv "github.com/milvus-io/milvus/internal/metastore/kv/queryview"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/querycoordv2/autoscale"
	"github.com/milvus-io/milvus/internal/querycoordv2/job"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/views/coord/loadmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/messagespb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestQueryViewAutoLoadConfigBroadcastAndManualLoad(t *testing.T) {
	s := &Server{idAllocator: func() (int64, error) { return 1000, nil }}
	coll := &metacache.CollectionInfo{ID: 100, Schema: &schemapb.CollectionSchema{}}
	expected := qviewsExpectedLoadConfig{AutoLoad: true, PartitionIDs: []int64{10}, ReplicaNumber: map[string]int{"rg1": 1}}
	msg, err := s.generateAlterLoadConfigMessageForLoadCollection(t.Context(), coll, nil, expected)
	require.NoError(t, err)
	header := message.MustAsBroadcastAlterLoadConfigMessageV2(msg).Header()
	// The persisted broadcast must retain the source across replay, not a context value.
	wire, err := proto.Marshal(header)
	require.NoError(t, err)
	replayed := &messagespb.AlterLoadConfigMessageHeader{}
	require.NoError(t, proto.Unmarshal(wire, replayed))
	current := loadmgr.FromAlterLoadConfigMessage(replayed)
	require.True(t, current.AutoLoad)
	require.True(t, loadConfigIntoAlterLoadConfigHeader(current).GetAutoLoad())
	msg, err = s.generateAlterLoadConfigMessageForLoadCollection(t.Context(), coll, current, expected)
	require.NoError(t, err)
	require.Nil(t, msg, "an identical automatic load remains a no-op")

	// Replica/partition operations through the admission adapter retain the source.
	adapted := qviewsCurrentLoadConfig(current)
	require.True(t, adapted.AutoLoad)
	require.True(t, adapted.IntoLoadConfigMessageHeader().GetAutoLoad())
	msg, err = job.GenerateAlterLoadConfigMessage(t.Context(), &job.AlterLoadConfigRequest{
		CollectionInfo: coll, Current: adapted,
		Expected: job.ExpectedLoadConfig{
			ExpectedAutoLoad: adapted.AutoLoad, ExpectedPartitionIDs: []int64{10, 20},
			ExpectedReplicaNumber: map[string]int{"rg1": 1},
		},
	})
	require.NoError(t, err)
	require.True(t, message.MustAsBroadcastAlterLoadConfigMessageV2(msg).Header().GetAutoLoad())

	// Explicit manual Load must publish even if only the source changes.
	expected.AutoLoad = false
	msg, err = s.generateAlterLoadConfigMessageForLoadCollection(t.Context(), coll, current, expected)
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.False(t, message.MustAsBroadcastAlterLoadConfigMessageV2(msg).Header().GetAutoLoad())
	require.True(t, current.AutoLoad, "published config remains immutable")
}

func TestLoadCollectionSharedAdmissionAndRefresh(t *testing.T) {
	s := &Server{}
	s.UpdateStateCode(commonpb.StateCode_Abnormal)
	status, err := s.LoadCollection(t.Context(), &querypb.LoadCollectionRequest{CollectionID: 100})
	require.NoError(t, err)
	require.Error(t, merr.Error(status), "the shared helper must retain health admission")

	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "failure"}[fail], func(t *testing.T) {
			s.UpdateStateCode(commonpb.StateCode_Healthy)
			var refreshErr error
			if fail {
				refreshErr = merr.ErrServiceUnavailable
			}
			patch := mockey.Mock((*Server).refreshCollection).Return(refreshErr).Build()
			defer patch.UnPatch()
			// Refresh does not create a new load intent or need a LoadConfigStore.
			status, err := s.LoadCollection(t.Context(), &querypb.LoadCollectionRequest{CollectionID: 100, Refresh: true})
			require.NoError(t, err)
			require.ErrorIs(t, merr.Error(status), refreshErr)
			require.EqualValues(t, 1, patch.Times())
		})
	}
}

func TestLoadCollectionBroadcastsWhenSecondaryIndexVisibleChanges(t *testing.T) {
	catalog := catalogkv.NewCollectionLoadInfoCatalog(nil)
	patchList := mockey.Mock(mockey.GetMethod(catalog, "List")).Return([]*viewpb.CollectionLoadConfig(nil), nil).Build()
	defer patchList.UnPatch()
	patchSave := mockey.Mock(mockey.GetMethod(catalog, "Save")).Return(nil).Build()
	defer patchSave.UnPatch()

	var version uint64
	idxPrimary := &indexpb.IndexInfo{CollectionID: 100, FieldID: 101, IndexID: 200, IndexName: "price_sort"}
	idxSecondary := &indexpb.IndexInfo{CollectionID: 100, FieldID: 101, IndexID: 201, IndexName: "price_ngram"}
	indexes := []*indexpb.IndexInfo{idxPrimary, idxSecondary}
	deps := loadmgr.LoadConfigStoreDependencies{
		Catalog:         catalog,
		AllocateVersion: func(context.Context) (uint64, error) { version++; return version, nil },
		Indexes:         func(int64, ...int64) []*indexpb.IndexInfo { return indexes },
	}

	store, err := loadmgr.RecoverLoadConfigStore(t.Context(), deps)
	require.NoError(t, err)
	s := &Server{
		idAllocator:   func() (int64, error) { return 1000, nil },
		qviewsRuntime: &qviewsRuntime{loadConfigStore: store},
	}
	coll := &metacache.CollectionInfo{
		ID: 100,
		Schema: &schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{{FieldID: 101, DataType: schemapb.DataType_Int64, IsPrimaryKey: true}},
		},
	}
	expected := qviewsExpectedLoadConfig{
		PartitionIDs:  []int64{10},
		ReplicaNumber: map[string]int{"rg1": 1},
		FieldIndexID:  map[int64]int64{101: 200},
	}

	msg, err := s.generateAlterLoadConfigMessageForLoadCollection(t.Context(), coll, nil, expected)
	require.NoError(t, err)
	require.NotNil(t, msg)
	header := message.MustAsBroadcastAlterLoadConfigMessageV2(msg).Header()
	require.NoError(t, store.Put(t.Context(), loadmgr.FromAlterLoadConfigMessage(header)))
	require.Equal(t, []int64{200, 201}, store.GetConfig(100).RequiredFields[0].GetIndexIds())

	// Repeating LoadCollection with no index changes is an idempotent no-op.
	msg, err = s.generateAlterLoadConfigMessageForLoadCollection(t.Context(), coll, store.GetConfig(100), expected)
	require.NoError(t, err)
	require.Nil(t, msg)

	// Hiding the 2nd index keeps FieldIndexID[101]=200 in the legacy request map,
	// yet LoadCollection still detects the RequiredFields delta if not yet refreshed.
	idxSecondary.IndexParams = []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "false"}}
	msg, err = s.generateAlterLoadConfigMessageForLoadCollection(t.Context(), coll, store.GetConfig(100), expected)
	require.NoError(t, err)
	require.NotNil(t, msg)
	header = message.MustAsBroadcastAlterLoadConfigMessageV2(msg).Header()
	require.NoError(t, store.Put(t.Context(), loadmgr.FromAlterLoadConfigMessage(header)))
	require.Equal(t, []int64{200}, store.GetConfig(100).RequiredFields[0].GetIndexIds())

	// LoadPartitions stays idempotent when active RequiredFields match resolved indexes,
	// and broadcasts updated index_ids when visible changes.
	partReq := &job.AlterLoadConfigRequest{
		CollectionInfo: coll,
		Current:        qviewsCurrentLoadConfig(store.GetConfig(100)),
		Expected: job.ExpectedLoadConfig{
			ExpectedPartitionIDs:   []int64{10},
			ExpectedReplicaNumber:  map[string]int{"rg1": 1},
			ExpectedFieldIndexID:   map[int64]int64{101: 200},
			ExpectedLoadFields:     loadmgr.PreloadFieldIDsForSchema(coll.Schema, nil),
			ExpectedRequiredFields: s.resolveExpectedRequiredFields(coll.ID, coll.Schema, map[int64]int64{101: 200}),
		},
	}
	partMsg, err := job.GenerateAlterLoadConfigMessage(t.Context(), partReq)
	require.NoError(t, err)
	require.Nil(t, partMsg)

	idxSecondary.IndexParams = []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "true"}}
	partReq.Expected.ExpectedRequiredFields = s.resolveExpectedRequiredFields(coll.ID, coll.Schema, map[int64]int64{101: 200})
	partMsg, err = job.GenerateAlterLoadConfigMessage(t.Context(), partReq)
	require.NoError(t, err)
	require.NotNil(t, partMsg)
}

func TestRequiredLoadResourceDeltaForVisibleAndActiveIndexes(t *testing.T) {
	catalog := catalogkv.NewCollectionLoadInfoCatalog(nil)
	patchList := mockey.Mock(mockey.GetMethod(catalog, "List")).Return([]*viewpb.CollectionLoadConfig(nil), nil).Build()
	defer patchList.UnPatch()
	patchSave := mockey.Mock(mockey.GetMethod(catalog, "Save")).Return(nil).Build()
	defer patchSave.UnPatch()

	var version uint64
	idx1 := &indexpb.IndexInfo{CollectionID: 100, FieldID: 101, IndexID: 200, IndexName: "idx_1"}
	idx2 := &indexpb.IndexInfo{
		CollectionID: 100,
		FieldID:      101,
		IndexID:      201,
		IndexName:    "idx_2",
		IndexParams:  []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "false"}},
	}
	indexes := []*indexpb.IndexInfo{idx1, idx2}
	store, err := loadmgr.RecoverLoadConfigStore(t.Context(), loadmgr.LoadConfigStoreDependencies{
		Catalog:         catalog,
		AllocateVersion: func(context.Context) (uint64, error) { version++; return version, nil },
		Indexes:         func(int64, ...int64) []*indexpb.IndexInfo { return indexes },
	})
	require.NoError(t, err)

	broker := meta.NewMockBroker(t)
	seg := &datapb.SegmentInfo{ID: 500, CollectionID: 100, PartitionID: 10}
	broker.EXPECT().GetRecoveryInfoV2(mock.Anything, int64(100), int64(10)).Return(nil, []*datapb.SegmentInfo{seg}, nil)
	broker.EXPECT().GetSegmentInfo(mock.Anything, int64(500)).Return([]*datapb.SegmentInfo{seg}, nil)
	broker.EXPECT().GetIndexInfo(mock.Anything, int64(100), int64(500)).Return(map[int64][]*querypb.FieldIndexInfo{
		500: {
			{FieldID: 101, IndexID: 200},
			{FieldID: 101, IndexID: 201},
		},
	}, nil)

	patchEstimate := mockey.Mock(autoscale.EstimateSegmentsLoadResourceForLoadConfig).To(
		func(_ context.Context, _ *schemapb.CollectionSchema, _ []*datapb.SegmentInfo, segIndexes map[int64][]*querypb.FieldIndexInfo, _ []int64, _ autoscale.EstimateOptions) (autoscale.ResourceUsage, error) {
			var mem int64 = 1000 // base raw/segment overhead
			for _, info := range segIndexes[500] {
				if info.GetIndexID() == 200 {
					mem += 300
				}
				if info.GetIndexID() == 201 {
					mem += 700
				}
			}
			return autoscale.ResourceUsage{MemoryBytes: mem}, nil
		},
	).Build()
	defer patchEstimate.UnPatch()

	s := &Server{
		broker:        broker,
		qviewsRuntime: &qviewsRuntime{loadConfigStore: store},
	}
	coll := &metacache.CollectionInfo{
		ID: 100,
		Schema: &schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{{FieldID: 101, DataType: schemapb.DataType_Int64, IsPrimaryKey: true}},
		},
	}

	// 1. Initial load with idx2 invisible: expectedUsage counts only idx1 (1000 + 300 = 1300).
	initialReq := &job.AlterLoadConfigRequest{
		CollectionInfo: coll,
		Expected: job.ExpectedLoadConfig{
			ExpectedPartitionIDs:  []int64{10},
			ExpectedReplicaNumber: map[string]int{"rg1": 1},
		},
	}
	usage, err := s.requiredLoadResourceByRG(t.Context(), initialReq)
	require.NoError(t, err)
	require.Equal(t, autoscaleResourceUsage{memoryBytes: 1300}, usage["rg1"])

	// Commit initial load (active = [200]).
	require.NoError(t, store.Put(t.Context(), &loadmgr.LoadConfig{
		CollectionID:   100,
		PartitionIDs:   []int64{10},
		RequiredFields: []*viewpb.LoadFieldConfig{{FieldId: 101}},
		Replicas:       []*loadmgr.ReplicaAssignment{{ReplicaID: 1, ResourceGroup: "rg1"}},
	}))
	require.Equal(t, []int64{200}, store.GetConfig(100).RequiredFields[0].GetIndexIds())

	// 2. Show idx2 and re-load: expectedUsage=2000 (idx1+idx2), currentUsage=1300 (active idx1) -> delta=700!
	idx2.IndexParams = []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "true"}}
	reloadReq := &job.AlterLoadConfigRequest{
		CollectionInfo: coll,
		Current:        qviewsCurrentLoadConfig(store.GetConfig(100)),
		Expected: job.ExpectedLoadConfig{
			ExpectedPartitionIDs:  []int64{10},
			ExpectedReplicaNumber: map[string]int{"rg1": 1},
		},
	}
	usage, err = s.requiredLoadResourceByRG(t.Context(), reloadReq)
	require.NoError(t, err)
	require.Equal(t, autoscaleResourceUsage{memoryBytes: 700}, usage["rg1"])
}

func TestPrepareAutoLoadRequestSkipsInvisibleAndSelectsSmallestIndexID(t *testing.T) {
	mockMix := mocks.NewMixCoord(t)
	mockMix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&indexpb.DescribeIndexResponse{
		Status: merr.Success(),
		IndexInfos: []*indexpb.IndexInfo{
			// Invisible vector index must be ignored (including UserIndexParams visible=false).
			{
				FieldID:         101,
				IndexID:         10,
				UserIndexParams: []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "false"}},
			},
			// Multiple visible scalar indexes on field 102 -> deterministic smallest IndexID (20).
			{FieldID: 102, IndexID: 30},
			{FieldID: 102, IndexID: 20},
			{
				FieldID:     102,
				IndexID:     5,
				IndexParams: []*commonpb.KeyValuePair{{Key: common.IndexVisibleKey, Value: "false"}},
			},
			// Visible vector index on field 103.
			{FieldID: 103, IndexID: 40},
		},
	}, nil)

	s := &Server{mixCoord: mockMix}
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vec_disabled", DataType: schemapb.DataType_FloatVector},
			{FieldID: 102, Name: "scalar", DataType: schemapb.DataType_Int64},
			{FieldID: 103, Name: "vec_ok", DataType: schemapb.DataType_FloatVector},
		},
	}

	// Disabled vector index on field 101 causes unindexed vector error.
	_, err := s.prepareAutoLoadRequest(t.Context(), &milvuspb.DescribeCollectionResponse{
		CollectionID: 1,
		Schema:       schema,
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "vec_disabled")

	// With only vec_ok as vector field, prepareAutoLoadRequest succeeds and picks smallest enabled IndexID (20) on field 102.
	schema.Fields = []*schemapb.FieldSchema{
		{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		{FieldID: 102, Name: "scalar", DataType: schemapb.DataType_Int64},
		{FieldID: 103, Name: "vec_ok", DataType: schemapb.DataType_FloatVector},
	}
	loadReq, err := s.prepareAutoLoadRequest(t.Context(), &milvuspb.DescribeCollectionResponse{
		DbId:         2,
		CollectionID: 1,
		Schema:       schema,
	})
	require.NoError(t, err)
	require.Equal(t, map[int64]int64{102: 20, 103: 40}, loadReq.GetFieldIndexID())
}

func TestGenerateLoadFieldsNeverRevivesZeroIndexRequiredField(t *testing.T) {
	got := job.GenerateLoadFields(
		[]int64{101, 102, 103},
		map[int64]int64{
			101: 999, // stale fallback must NOT override explicit 0-index RequiredFields entry
			102: 888, // fallback used only when field is absent from RequiredFields
		},
		[]*messagespb.LoadFieldConfig{
			{FieldId: 101}, // explicitly 0 enabled indexes
			{FieldId: 103, IndexId: 301, IndexIds: []int64{301, 302}},
		},
	)
	require.Len(t, got, 3)
	require.EqualValues(t, 101, got[0].GetFieldId())
	require.Zero(t, got[0].GetIndexId())
	require.Empty(t, got[0].GetIndexIds())

	require.EqualValues(t, 102, got[1].GetFieldId())
	require.EqualValues(t, 888, got[1].GetIndexId())
	require.Equal(t, []int64{888}, got[1].GetIndexIds())

	require.EqualValues(t, 103, got[2].GetFieldId())
	require.EqualValues(t, 301, got[2].GetIndexId())
	require.Equal(t, []int64{301, 302}, got[2].GetIndexIds())
}
