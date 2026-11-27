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

package dql

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proxy/metacache"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func kv(key, value string) *commonpb.KeyValuePair {
	return &commonpb.KeyValuePair{Key: key, Value: value}
}

func TestIndexHintNames(t *testing.T) {
	t.Run("no hint", func(t *testing.T) {
		names, strict, err := indexHintNames(nil)
		require.NoError(t, err)
		assert.Empty(t, names)
		assert.False(t, strict)
	})

	t.Run("empty hint is not a hint", func(t *testing.T) {
		names, strict, err := indexHintNames([]*commonpb.KeyValuePair{kv(IndexHintKey, "   ")})
		require.NoError(t, err)
		assert.Empty(t, names)
		assert.False(t, strict)
	})

	t.Run("order is preserved and blanks are dropped", func(t *testing.T) {
		names, strict, err := indexHintNames([]*commonpb.KeyValuePair{kv(IndexHintKey, " idx_a , ,idx_b ")})
		require.NoError(t, err)
		assert.Equal(t, []string{"idx_a", "idx_b"}, names)
		assert.False(t, strict)
	})

	t.Run("strict mode", func(t *testing.T) {
		names, strict, err := indexHintNames([]*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_a"), kv(IndexHintModeKey, " STRICT "),
		})
		require.NoError(t, err)
		assert.Equal(t, []string{"idx_a"}, names)
		assert.True(t, strict)
	})

	t.Run("explicit fallback mode", func(t *testing.T) {
		_, strict, err := indexHintNames([]*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_a"), kv(IndexHintModeKey, "fallback"),
		})
		require.NoError(t, err)
		assert.False(t, strict)
	})

	t.Run("unknown mode is rejected", func(t *testing.T) {
		_, _, err := indexHintNames([]*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_a"), kv(IndexHintModeKey, "sometimes"),
		})
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	})

	t.Run("nested params JSON fallback and top-level precedence", func(t *testing.T) {
		names, strict, err := indexHintNames([]*commonpb.KeyValuePair{
			kv(ParamsKey, `{"nprobe": 10, "index_hint": "idx_a,idx_b", "index_hint_mode": "strict"}`),
		})
		require.NoError(t, err)
		assert.Equal(t, []string{"idx_a", "idx_b"}, names)
		assert.True(t, strict)

		// Top-level KV wins directly without re-scanning nested params JSON.
		names, strict, err = indexHintNames([]*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_a"),
			kv(ParamsKey, `{"index_hint": "idx_b"}`),
		})
		require.NoError(t, err)
		assert.Equal(t, []string{"idx_a"}, names)
		assert.False(t, strict)
	})

	t.Run("hybrid search mergeIndexHint merges outer over sub", func(t *testing.T) {
		outer, err := parseIndexHintParams([]*commonpb.KeyValuePair{kv(IndexHintModeKey, "strict")})
		require.NoError(t, err)
		sub, err := parseIndexHintParams([]*commonpb.KeyValuePair{kv(ParamsKey, `{"index_hint": "idx_sub"}`)})
		require.NoError(t, err)
		names, strict := mergeIndexHint(outer, sub)
		assert.Equal(t, []string{"idx_sub"}, names)
		assert.True(t, strict)

		outerOverride, err := parseIndexHintParams([]*commonpb.KeyValuePair{kv(IndexHintKey, "idx_outer")})
		require.NoError(t, err)
		subOverride, err := parseIndexHintParams([]*commonpb.KeyValuePair{kv(IndexHintKey, "idx_sub")})
		require.NoError(t, err)
		names, _ = mergeIndexHint(outerOverride, subOverride)
		assert.Equal(t, []string{"idx_outer"}, names)

		// Blank outer index_hint explicitly clears sub-request hint.
		outerBlank, err := parseIndexHintParams([]*commonpb.KeyValuePair{kv(IndexHintKey, " ")})
		require.NoError(t, err)
		names, _ = mergeIndexHint(outerBlank, subOverride)
		assert.Empty(t, names)
	})
}

type fakeHintNode struct {
	taskmodel.TaskNode
	mix   types.MixCoordClient
	cache metacache.Cache
}

func (n *fakeHintNode) MixCoord() types.MixCoordClient { return n.mix }
func (n *fakeHintNode) GetMetaCache() metacache.Cache  { return n.cache }

func TestApplyIndexHint(t *testing.T) {
	const collID = int64(1)

	newNodeWithMix := func(t *testing.T, mix types.MixCoordClient) *fakeHintNode {
		t.Helper()
		mc, err := metacache.NewMetaCache(mix)
		require.NoError(t, err)
		return &fakeHintNode{mix: mix, cache: mc}
	}

	newNode := func(t *testing.T, infos ...*indexpb.IndexInfo) *fakeHintNode {
		t.Helper()
		mix := mocks.NewMockMixCoordClient(t)
		mix.EXPECT().ListIndexes(mock.Anything, mock.Anything).Return(&indexpb.ListIndexesResponse{
			Status:     merr.Success(),
			IndexInfos: infos,
		}, nil).Maybe()
		return newNodeWithMix(t, mix)
	}

	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "scalar", DataType: schemapb.DataType_Int64},
			{FieldID: 101, Name: "json_f", DataType: schemapb.DataType_JSON},
			{FieldID: 102, Name: "vec", DataType: schemapb.DataType_FloatVector},
		},
	}

	infos := []*indexpb.IndexInfo{
		{CollectionID: collID, FieldID: 100, IndexID: 1001, IndexName: "idx_a"},
		{CollectionID: collID, FieldID: 100, IndexID: 1002, IndexName: "idx_b"},
		{
			CollectionID: collID,
			FieldID:      101,
			IndexID:      1003,
			IndexName:    "idx_json_path",
			IndexParams: []*commonpb.KeyValuePair{
				kv(common.JSONPathKey, "/a/b"),
				kv(common.JSONCastTypeKey, "DOUBLE"),
			},
		},
		{
			CollectionID: collID,
			FieldID:      101,
			IndexID:      1004,
			IndexName:    "idx_json_flat",
			IndexParams: []*commonpb.KeyValuePair{
				kv(common.JSONPathKey, "/a"),
				kv(common.JSONCastTypeKey, "JSON"),
			},
		},
		{
			CollectionID: collID,
			FieldID:      100,
			IndexID:      1005,
			IndexName:    "idx_invisible",
			UserIndexParams: []*commonpb.KeyValuePair{
				kv(common.IndexVisibleKey, "false"),
			},
		},
		{CollectionID: collID, FieldID: 102, IndexID: 1006, IndexName: "idx_vec"},
	}

	t.Run("names resolve to field-scoped ids in hint order", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		params := []*commonpb.KeyValuePair{kv(IndexHintKey, "idx_b,idx_json_path,idx_json_flat,idx_a")}
		require.NoError(t, applyIndexHint(context.Background(), newNode(t, infos...), collID, schema, plan, params))

		require.Len(t, plan.GetPlanOptions().GetIndexHints(), 4)
		assert.Equal(t, int64(100), plan.PlanOptions.IndexHints[0].GetFieldId())
		assert.Equal(t, int64(1002), plan.PlanOptions.IndexHints[0].GetIndexId())

		assert.Equal(t, int64(101), plan.PlanOptions.IndexHints[1].GetFieldId())
		assert.Equal(t, int64(1003), plan.PlanOptions.IndexHints[1].GetIndexId())

		assert.Equal(t, int64(101), plan.PlanOptions.IndexHints[2].GetFieldId())
		assert.Equal(t, int64(1004), plan.PlanOptions.IndexHints[2].GetIndexId())

		assert.Equal(t, int64(100), plan.PlanOptions.IndexHints[3].GetFieldId())
		assert.Equal(t, int64(1001), plan.PlanOptions.IndexHints[3].GetIndexId())
		assert.False(t, plan.PlanOptions.GetIndexHintStrict())
	})

	t.Run("fallback drops unknown, invisible, and vector index names", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		params := []*commonpb.KeyValuePair{kv(IndexHintKey, "idx_missing,idx_invisible,idx_vec,idx_a")}
		require.NoError(t, applyIndexHint(context.Background(), newNode(t, infos...), collID, schema, plan, params))
		require.Len(t, plan.GetPlanOptions().GetIndexHints(), 1)
		assert.Equal(t, int64(1001), plan.PlanOptions.IndexHints[0].GetIndexId())
	})

	t.Run("strict rejects unknown, invisible, and vector index names with ParameterInvalid", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		err := applyIndexHint(context.Background(), newNode(t, infos...), collID, schema, plan, []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_missing"), kv(IndexHintModeKey, "strict"),
		})
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)

		err = applyIndexHint(context.Background(), newNode(t, infos...), collID, schema, plan, []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_invisible"), kv(IndexHintModeKey, "strict"),
		})
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)

		err = applyIndexHint(context.Background(), newNode(t, infos...), collID, schema, plan, []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_vec"), kv(IndexHintModeKey, "strict"),
		})
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Empty(t, plan.GetPlanOptions().GetIndexHints())
	})

	t.Run("ErrIndexNotFound maps to ParameterInvalid in strict mode", func(t *testing.T) {
		mix := mocks.NewMockMixCoordClient(t)
		mix.EXPECT().ListIndexes(mock.Anything, mock.Anything).Return(&indexpb.ListIndexesResponse{
			Status: merr.Status(merr.ErrIndexNotFound),
		}, nil)
		node := newNodeWithMix(t, mix)
		err := applyIndexHint(context.Background(), node, collID, schema, &planpb.PlanNode{}, []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_a"), kv(IndexHintModeKey, "strict"),
		})
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.False(t, errors.Is(err, merr.ErrIndexNotFound))
	})

	t.Run("MetaCache hits avoid repeat RPCs and DDL invalidation evicts immediately", func(t *testing.T) {
		mix := mocks.NewMockMixCoordClient(t)
		mix.EXPECT().ListIndexes(mock.Anything, mock.Anything).Return(&indexpb.ListIndexesResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{CollectionID: collID, FieldID: 100, IndexID: 1001, IndexName: "idx_a"},
			},
		}, nil).Once()
		node := newNodeWithMix(t, mix)

		// Two consecutive queries for cached idx_a only trigger ListIndexes once.
		require.NoError(t, applyIndexHint(context.Background(), node, collID, schema, &planpb.PlanNode{}, []*commonpb.KeyValuePair{kv(IndexHintKey, "idx_a")}))
		require.NoError(t, applyIndexHint(context.Background(), node, collID, schema, &planpb.PlanNode{}, []*commonpb.KeyValuePair{kv(IndexHintKey, "idx_a")}))

		// Simulate AlterIndex(visible=false) or DropIndex invalidating MetaCache:
		// next query immediately re-fetches and sees idx_a invisible and idx_new present.
		node.cache.InvalidateCollectionIndexes(context.Background(), collID)
		mix.EXPECT().ListIndexes(mock.Anything, mock.Anything).Return(&indexpb.ListIndexesResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{
					CollectionID:    collID,
					FieldID:         100,
					IndexID:         1001,
					IndexName:       "idx_a",
					UserIndexParams: []*commonpb.KeyValuePair{kv(common.IndexVisibleKey, "false")},
				},
				{CollectionID: collID, FieldID: 100, IndexID: 1099, IndexName: "idx_new"},
			},
		}, nil).Once()
		err := applyIndexHint(context.Background(), node, collID, schema, &planpb.PlanNode{}, []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_a"), kv(IndexHintModeKey, "strict"),
		})
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)

		plan := &planpb.PlanNode{}
		require.NoError(t, applyIndexHint(context.Background(), node, collID, schema, plan, []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_new"), kv(IndexHintModeKey, "strict"),
		}))
		require.Len(t, plan.GetPlanOptions().GetIndexHints(), 1)
		assert.Equal(t, int64(1099), plan.PlanOptions.IndexHints[0].GetIndexId())
	})

	t.Run("no hint leaves the plan alone", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		require.NoError(t, applyIndexHint(context.Background(), newNode(t, infos...), collID, schema, plan, nil))
		assert.Nil(t, plan.GetPlanOptions())
	})
}

func TestApplyIndexHintCoordinatorFailure(t *testing.T) {
	plan := &planpb.PlanNode{}
	params := []*commonpb.KeyValuePair{
		{Key: IndexHintKey, Value: "idx_a"},
		{Key: IndexHintModeKey, Value: "strict"},
	}
	mix := mocks.NewMockMixCoordClient(t)
	mix.EXPECT().ListIndexes(mock.Anything, mock.Anything).
		Return(nil, merr.ErrServiceNotReady).Once()
	mc, err := metacache.NewMetaCache(mix)
	require.NoError(t, err)
	node := &fakeHintNode{mix: mix, cache: mc}

	err = applyIndexHint(context.Background(), node, 1, nil, plan, params)
	require.Error(t, err)
	assert.True(t, errors.Is(err, merr.ErrServiceNotReady),
		"the coordinator's retriable code must survive")

	// Fallback mode runs without the hint, and transient failure was not cached.
	fallback := []*commonpb.KeyValuePair{{Key: IndexHintKey, Value: "idx_a"}}
	mix.EXPECT().ListIndexes(mock.Anything, mock.Anything).
		Return(nil, merr.ErrServiceNotReady).Once()
	assert.NoError(t, applyIndexHint(context.Background(), node, 1, nil, plan, fallback))
}
