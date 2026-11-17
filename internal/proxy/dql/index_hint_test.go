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
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/internal/types"
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
		assert.Error(t, err)
	})
}

// fakeHintNode implements just enough of taskmodel.TaskNode for applyIndexHint:
// the hint resolver only asks for the coordinator.
type fakeHintNode struct {
	taskmodel.TaskNode
	mix types.MixCoordClient
}

func (n *fakeHintNode) MixCoord() types.MixCoordClient { return n.mix }

func TestApplyIndexHint(t *testing.T) {
	const collID = int64(1)

	newNode := func(t *testing.T, infos ...*indexpb.IndexInfo) *fakeHintNode {
		t.Helper()
		mix := mocks.NewMockMixCoordClient(t)
		mix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&indexpb.DescribeIndexResponse{
			Status:     merr.Success(),
			IndexInfos: infos,
		}, nil).Maybe()
		return &fakeHintNode{mix: mix}
	}

	infos := []*indexpb.IndexInfo{
		{CollectionID: collID, FieldID: 100, IndexID: 1001, IndexName: "idx_a"},
		{CollectionID: collID, FieldID: 100, IndexID: 1002, IndexName: "idx_b"},
		{CollectionID: collID, FieldID: 101, IndexID: 1003, IndexName: "idx_json"},
	}

	t.Run("names resolve to field-scoped ids in hint order", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		params := []*commonpb.KeyValuePair{kv(IndexHintKey, "idx_b,idx_a")}
		require.NoError(t, applyIndexHint(context.Background(), newNode(t, infos...), collID, plan, params))

		require.Len(t, plan.GetPlanOptions().GetIndexHints(), 2)
		assert.Equal(t, int64(100), plan.PlanOptions.IndexHints[0].GetFieldId())
		assert.Equal(t, int64(1002), plan.PlanOptions.IndexHints[0].GetIndexId())
		assert.Equal(t, int64(1001), plan.PlanOptions.IndexHints[1].GetIndexId())
		assert.False(t, plan.PlanOptions.GetIndexHintStrict())
	})

	t.Run("fallback drops an unknown name", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		params := []*commonpb.KeyValuePair{kv(IndexHintKey, "idx_missing,idx_a")}
		require.NoError(t, applyIndexHint(context.Background(), newNode(t, infos...), collID, plan, params))
		require.Len(t, plan.GetPlanOptions().GetIndexHints(), 1)
		assert.Equal(t, int64(1001), plan.PlanOptions.IndexHints[0].GetIndexId())
	})

	t.Run("strict rejects an unknown name", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		params := []*commonpb.KeyValuePair{
			kv(IndexHintKey, "idx_missing"), kv(IndexHintModeKey, "strict"),
		}
		err := applyIndexHint(context.Background(), newNode(t, infos...), collID, plan, params)
		assert.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Empty(t, plan.GetPlanOptions().GetIndexHints())
	})

	t.Run("no hint leaves the plan alone", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		require.NoError(t, applyIndexHint(context.Background(), newNode(t, infos...), collID, plan, nil))
		assert.Nil(t, plan.GetPlanOptions())
	})
}

// A coordinator failure must not swallow the cause: strict keeps its code, and
// fallback runs the query without the hint instead of failing it.
func TestApplyIndexHintCoordinatorFailure(t *testing.T) {
	plan := &planpb.PlanNode{}
	node := &fakeHintNode{}
	params := []*commonpb.KeyValuePair{
		{Key: IndexHintKey, Value: "idx_a"},
		{Key: IndexHintModeKey, Value: "strict"},
	}
	mix := mocks.NewMockMixCoordClient(t)
	mix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).
		Return(nil, merr.ErrServiceNotReady)
	node.mix = mix

	err := applyIndexHint(context.Background(), node, 1, plan, params)
	require.Error(t, err)
	assert.True(t, errors.Is(err, merr.ErrServiceNotReady),
		"the coordinator's retriable code must survive")

	// Fallback mode runs without the hint.
	fallback := []*commonpb.KeyValuePair{{Key: IndexHintKey, Value: "idx_a"}}
	mix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).
		Return(nil, merr.ErrServiceNotReady)
	assert.NoError(t, applyIndexHint(context.Background(), node, 1, plan, fallback))
}

// When the outer HybridSearch params carry no index_hint, each sub-search's own
// params govern; when the outer params specify index_hint, outer overrides
// sub-search hints across all sub-plans.
func TestHybridSubSearchIndexHintUsesSubParamsAndOuterOverride(t *testing.T) {
	newMockHintNode := func(t *testing.T) *fakeHintNode {
		t.Helper()
		mix := mocks.NewMockMixCoordClient(t)
		mix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&indexpb.DescribeIndexResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{CollectionID: 1, FieldID: 101, IndexID: 3001, IndexName: "outer_idx"},
				{CollectionID: 1, FieldID: 101, IndexID: 3002, IndexName: "sub_idx"},
			},
		}, nil).Maybe()
		return &fakeHintNode{mix: mix}
	}

	t.Run("sub-search strict hint applies when outer omits index_hint", func(t *testing.T) {
		task := newHybridDynamicSearchTask(t, false, []string{"pk"}, nil)
		task.node = newMockHintNode(t)
		require.NotEmpty(t, task.request.GetSubReqs())
		task.request.SubReqs[1].SearchParams = append(task.request.SubReqs[1].SearchParams,
			kv(IndexHintKey, "missing_idx"), kv(IndexHintModeKey, "strict"))

		err := task.initAdvancedSearchRequest(context.Background())
		require.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), "missing_idx")
	})

	t.Run("outer strict hint applies to sub-searches", func(t *testing.T) {
		task := newHybridDynamicSearchTask(t, false, []string{"pk"}, nil)
		task.node = newMockHintNode(t)
		task.request.SearchParams = append(task.request.SearchParams,
			kv(IndexHintKey, "missing_outer"), kv(IndexHintModeKey, "strict"))

		err := task.initAdvancedSearchRequest(context.Background())
		require.ErrorIs(t, err, merr.ErrParameterInvalid)
		assert.Contains(t, err.Error(), "missing_outer")
	})

	t.Run("outer index_hint overrides sub-search index_hint", func(t *testing.T) {
		task := newHybridDynamicSearchTask(t, false, []string{"pk"}, nil)
		task.node = newMockHintNode(t)
		task.request.SearchParams = append(task.request.SearchParams,
			kv(IndexHintKey, "outer_idx"), kv(IndexHintModeKey, "fallback"))
		task.request.SubReqs[0].SearchParams = append(task.request.SubReqs[0].SearchParams,
			kv(IndexHintKey, "sub_idx"), kv(IndexHintModeKey, "strict"))
		task.request.SubReqs[1].SearchParams = append(task.request.SubReqs[1].SearchParams,
			kv(IndexHintKey, "missing_sub"), kv(IndexHintModeKey, "strict"))

		require.NoError(t, task.initAdvancedSearchRequest(context.Background()))
		for i, sub := range task.GetSubReqs() {
			plan := &planpb.PlanNode{}
			require.NoError(t, proto.Unmarshal(sub.GetSerializedExprPlan(), plan), "sub[%d]", i)
			require.Len(t, plan.GetPlanOptions().GetIndexHints(), 1, "sub[%d]", i)
			assert.Equal(t, int64(3001), plan.GetPlanOptions().GetIndexHints()[0].GetIndexId(), "sub[%d]", i)
			assert.False(t, plan.GetPlanOptions().GetIndexHintStrict(), "sub[%d]", i)
		}
	})

	t.Run("sub-search inherits outer index_hint_mode when outer omits index_hint", func(t *testing.T) {
		task := newHybridDynamicSearchTask(t, false, []string{"pk"}, nil)
		task.node = newMockHintNode(t)
		task.request.SearchParams = append(task.request.SearchParams,
			kv(IndexHintModeKey, "strict"))
		task.request.SubReqs[0].SearchParams = append(task.request.SubReqs[0].SearchParams,
			kv(IndexHintKey, "sub_idx"))

		require.NoError(t, task.initAdvancedSearchRequest(context.Background()))
		firstPlan := &planpb.PlanNode{}
		require.NoError(t, proto.Unmarshal(task.GetSubReqs()[0].GetSerializedExprPlan(), firstPlan))
		require.Len(t, firstPlan.GetPlanOptions().GetIndexHints(), 1)
		assert.Equal(t, int64(3002), firstPlan.GetPlanOptions().GetIndexHints()[0].GetIndexId())
		assert.True(t, firstPlan.GetPlanOptions().GetIndexHintStrict())

		secondPlan := &planpb.PlanNode{}
		require.NoError(t, proto.Unmarshal(task.GetSubReqs()[1].GetSerializedExprPlan(), secondPlan))
		assert.Empty(t, secondPlan.GetPlanOptions().GetIndexHints())
	})
}
