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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/stretchr/testify/mock"

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

// A hybrid search carries its hints on each sub-search, not on the outer
// (ranking) params. The plan builder must read the sub-search's own params, or a
// strict hint there is silently ignored.
func TestHybridSubSearchIndexHintUsesSubParams(t *testing.T) {
	task := newHybridDynamicSearchTask(t, false, []string{"pk"}, nil)

	mix := mocks.NewMockMixCoordClient(t)
	mix.EXPECT().DescribeIndex(mock.Anything, mock.Anything).Return(&indexpb.DescribeIndexResponse{
		Status: merr.Success(),
		IndexInfos: []*indexpb.IndexInfo{
			{CollectionID: 1, FieldID: 104, IndexID: 2001, IndexName: "vec_idx"},
		},
	}, nil).Maybe()
	task.node = &fakeHintNode{mix: mix}

	// Only the second sub-search hints, in strict mode, at an unknown name. If
	// plan generation read the outer params this error would never be raised.
	require.NotEmpty(t, task.request.GetSubReqs())
	task.request.SubReqs[1].SearchParams = append(task.request.SubReqs[1].SearchParams,
		kv(IndexHintKey, "missing_idx"), kv(IndexHintModeKey, "strict"))

	err := task.initAdvancedSearchRequest(context.Background())
	require.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "missing_idx")
}
