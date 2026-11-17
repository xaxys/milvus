// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

package dql

import (
	"context"
	"encoding/json"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/util/explain"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func intLiteral(v int64) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: v}}
}

func stringLiteral(v string) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: v}}
}

func TestDescribeExplainLeaf(t *testing.T) {
	fields := map[int64]string{101: "title", 103: "price", 100: "sku_id", 105: "j", 107: "tags"}

	rangeExpr := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 103},
		Op:         planpb.OpType_GreaterThan,
		Value:      intLiteral(100),
	}}}
	condition, typ, fieldID := describeExplainLeaf(rangeExpr, fields)
	assert.Equal(t, "price in (100, +inf)", condition)
	assert.Equal(t, "range", typ)
	assert.Equal(t, int64(103), fieldID)

	binaryRange := &planpb.Expr{Expr: &planpb.Expr_BinaryRangeExpr{BinaryRangeExpr: &planpb.BinaryRangeExpr{
		ColumnInfo:     &planpb.ColumnInfo{FieldId: 103},
		LowerInclusive: true,
		UpperInclusive: false,
		LowerValue:     intLiteral(2),
		UpperValue:     intLiteral(10),
	}}}
	condition, typ, _ = describeExplainLeaf(binaryRange, fields)
	assert.Equal(t, "price in [2, 10)", condition)
	assert.Equal(t, "range", typ)

	term := &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 100},
		Values:     []*planpb.GenericValue{intLiteral(1), intLiteral(2), intLiteral(3)},
	}}}
	condition, typ, fieldID = describeExplainLeaf(term, fields)
	assert.Equal(t, "sku_id in [1, 2, 3]", condition)
	assert.Equal(t, "in", typ)
	assert.Equal(t, int64(100), fieldID)

	// Inverted TermExpr (42 in tags) renders value on the left.
	inFieldTerm := &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 107},
		Values:     []*planpb.GenericValue{intLiteral(42)},
		IsInField:  true,
	}}}
	condition, typ, _ = describeExplainLeaf(inFieldTerm, fields)
	assert.Equal(t, "42 in tags", condition)
	assert.Equal(t, "in", typ)

	// ArrayLength renders as array_length(tags) > 2.
	arrLen := &planpb.Expr{Expr: &planpb.Expr_BinaryArithOpEvalRangeExpr{BinaryArithOpEvalRangeExpr: &planpb.BinaryArithOpEvalRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 107},
		ArithOp:    planpb.ArithOpType_ArrayLength,
		Op:         planpb.OpType_GreaterThan,
		Value:      intLiteral(2),
	}}}
	condition, typ, _ = describeExplainLeaf(arrLen, fields)
	assert.Equal(t, "array_length(tags) > 2", condition)
	assert.Equal(t, "array_length", typ)

	// Membership filters hide internal bloom/roaring details.
	bloom := &planpb.Expr{Expr: &planpb.Expr_BloomFilterExpr{BloomFilterExpr: &planpb.BloomFilterExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 100},
		FilterBlob: []byte{1, 2, 3, 4},
	}}}
	condition, typ, _ = describeExplainLeaf(bloom, fields)
	assert.Equal(t, "sku_id membership_filter", condition)
	assert.Equal(t, "membership", typ)

	roaring := &planpb.Expr{Expr: &planpb.Expr_RoaringFilterExpr{RoaringFilterExpr: &planpb.RoaringFilterExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 100},
	}}}
	condition, typ, _ = describeExplainLeaf(roaring, fields)
	assert.Equal(t, "sku_id membership_filter", condition)
	assert.Equal(t, "membership", typ)

	// Dynamic $meta field unwraps to user dynamic key name.
	dynCond, dynTyp, dynFID, dynUserField := describeExplainLeafWithDynamic(
		&planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
			ColumnInfo: &planpb.ColumnInfo{FieldId: 199, NestedPath: []string{"color"}},
			Op:         planpb.OpType_Equal,
			Value:      stringLiteral("red"),
		}}},
		nil,
		map[int64]bool{199: true},
	)
	assert.Equal(t, `color == "red"`, dynCond)
	assert.Equal(t, "equal", dynTyp)
	assert.Equal(t, int64(199), dynFID)
	assert.Equal(t, "color", dynUserField)
	assert.NotContains(t, dynCond, "$meta")
}

func TestDescribeExplainLeafBoundsMembershipList(t *testing.T) {
	values := make([]*planpb.GenericValue, 12)
	for i := range values {
		values[i] = intLiteral(int64(i))
	}
	term := &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 100},
		Values:     values,
	}}}
	condition, _, _ := describeExplainLeaf(term, nil)
	assert.Equal(t, "unknown in [0, 1, 2, 3, 4, 5, 6, 7, ... <12 values>]", condition)
}

func TestAssignExplainIDsTagsLeavesAndElementFilterScope(t *testing.T) {
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "sku_id"},
		{FieldID: 103, Name: "price"},
		{FieldID: 201, Name: "items[score]"},
	}}
	elemExpr := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 201},
		Op:         planpb.OpType_GreaterThan,
		Value:      intLiteral(10),
	}}}
	docExpr := &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 100},
		Values:     []*planpb.GenericValue{intLiteral(1), intLiteral(2)},
	}}}
	root := &planpb.Expr{Expr: &planpb.Expr_ElementFilterExpr{ElementFilterExpr: &planpb.ElementFilterExpr{
		StructName:  "items",
		ElementExpr: elemExpr,
		Predicate:   docExpr,
	}}}
	plan := &planpb.PlanNode{Node: &planpb.PlanNode_VectorAnns{VectorAnns: &planpb.VectorANNS{
		Predicates: root,
	}}}

	var next int64
	var leaves map[int64]explain.LeafInfo
	assignExplainIDsWithSubReq(plan, 1, schema, &next, &leaves)

	assert.Equal(t, int64(2), next)
	assert.Equal(t, int64(0), root.GetExprId())
	require.Len(t, leaves, 2)
	assert.True(t, leaves[1].ElementFilter)
	assert.True(t, leaves[1].HasSubReq)
	assert.Equal(t, 1, leaves[1].SubReqIndex)
	assert.False(t, leaves[2].ElementFilter)
	assert.Equal(t, "sku_id", leaves[2].Field)
}

func enableExplainKV() []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{{Key: explainKey, Value: explainValueDefault}}
}

func enableJSONExplainKV() []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{
		{Key: explainKey, Value: explainLevelSummary},
		{Key: explainFormatKey, Value: explainFormatJSON},
	}
}

func TestExplainParams(t *testing.T) {
	kv := func(key, value string) *commonpb.KeyValuePair {
		return &commonpb.KeyValuePair{Key: key, Value: value}
	}

	enabled, format, level, err := explainParams(nil)
	require.NoError(t, err)
	assert.False(t, enabled)
	assert.Equal(t, "text", format)
	assert.Empty(t, level)

	enabled, format, level, err = explainParams([]*commonpb.KeyValuePair{kv(explainKey, "default")})
	require.NoError(t, err)
	assert.True(t, enabled)
	assert.Equal(t, "text", format)
	assert.Equal(t, "summary", level)

	enabled, format, level, err = explainParams([]*commonpb.KeyValuePair{
		kv(explainKey, "SUMMARY"),
		kv(explainFormatKey, "JSON"),
	})
	require.NoError(t, err)
	assert.True(t, enabled)
	assert.Equal(t, "json", format)
	assert.Equal(t, "summary", level)

	_, _, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainFormatKey, "yaml")})
	assert.Error(t, err)
	_, _, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainKey, "verbose")})
	assert.Error(t, err)
}

func explainSegment() *internalpb.ExecutionExplain {
	return &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "459123456789012346",
		Counters: map[string]int64{"rows_total": 50000},
		Children: []*internalpb.ExecutionExplain{
			{
				Kind:       explain.KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0"},
				Counters:   map[string]int64{"rows_in": 50000, "rows_out": 31},
				Children: []*internalpb.ExecutionExplain{{
					Kind: explain.KindFilterGroup,
					Attributes: map[string]string{
						"expr_id": "3", "field_id": "101", "family": "ngram",
						"index_id": "459123456789100001", "nested": "false",
					},
					Counters: map[string]int64{"rows_in": 50000, "idx_scan": 50000, "raw_scan": 0, "rows_out": 31},
				}},
			},
			{
				Kind:       explain.KindOperator,
				Attributes: map[string]string{"name": "mvcc", "step": "1"},
				Counters:   map[string]int64{"rows_in": 31, "rows_out": 31},
			},
			{
				Kind:       explain.KindOperator,
				Attributes: map[string]string{"name": "vector_search", "step": "2"},
				Counters:   map[string]int64{"rows_in": 31, "rows_out": 10},
			},
		},
	}
}

func TestRenderExplainLevelsAndFormats(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, Name: "title"}},
	}
	leaves := map[int64]explain.LeafInfo{
		3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101, Field: "title"},
	}

	rendered, err := renderExplain(nil, []*internalpb.ExecutionExplain{explainSegment()}, leaves, schema, nil, nil)
	require.NoError(t, err)
	assert.Empty(t, rendered)

	rendered, err = renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{explainSegment()}, leaves, schema, nil, nil)
	require.NoError(t, err)
	assert.Contains(t, rendered, "query (segments=1, rows_total=50000)")
	assert.Contains(t, rendered, "[op 0] filter (segments=1, rows: 50000 -> 31)")
	assert.Contains(t, rendered, `expr: title contains "%降噪%"`)
	assert.Contains(t, rendered, "index ngram (ngram):")
	assert.NotContains(t, rendered, "459123456789100001")
	assert.NotContains(t, rendered, "expr_id")
	assert.NotContains(t, rendered, "field_id")

	rendered, err = renderExplain(enableJSONExplainKV(), []*internalpb.ExecutionExplain{explainSegment()}, leaves, schema, nil, nil)
	require.NoError(t, err)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(rendered), &decoded))
	assert.Equal(t, "summary", decoded["level"])
	assert.NotContains(t, rendered, "index_id")
	assert.NotContains(t, rendered, "field_id")
	assert.NotContains(t, rendered, "expr_id")
}

func TestRenderExplainUsesIndexNamesAndPrimaryKeyLabel(t *testing.T) {
	leaves := map[int64]explain.LeafInfo{3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101, Field: "title"}}
	names := map[int64]string{459123456789100001: "title_ngram"}
	segments := []*internalpb.ExecutionExplain{explainSegment()}

	rendered, err := renderExplain(enableExplainKV(), segments, leaves, nil, nil, func([]int64) map[int64]string { return names })
	require.NoError(t, err)
	assert.Contains(t, rendered, "index title_ngram (ngram):")

	rendered, err = renderExplain(enableJSONExplainKV(), segments, leaves, nil, nil, func([]int64) map[int64]string { return names })
	require.NoError(t, err)
	assert.Contains(t, rendered, `"index_name":"title_ngram"`)
	assert.NotContains(t, rendered, "index_id")

	// Built-in PK path (index_id=0, family="pk") renders as primary_key without RPC.
	pkLeaves := map[int64]explain.LeafInfo{1: {Condition: "id == 5", Type: "equal", FieldID: 100, Field: "id"}}
	pkSegment := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 100},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       explain.KindOperator,
			Attributes: map[string]string{"name": "filter", "step": "0"},
			Counters:   map[string]int64{"rows_in": 100, "rows_out": 1},
			Children: []*internalpb.ExecutionExplain{{
				Kind: explain.KindFilterGroup,
				Attributes: map[string]string{
					"expr_id": "1", "field_id": "100", "family": "pk",
					"index_id": "0", "nested": "false",
				},
				Counters: map[string]int64{"rows_in": 100, "idx_scan": 100, "raw_scan": 0, "rows_out": 1},
			}},
		}},
	}
	called := false
	pkRendered, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{pkSegment}, pkLeaves, nil, nil, func([]int64) map[int64]string {
		called = true
		return nil
	})
	require.NoError(t, err)
	assert.False(t, called)
	assert.Contains(t, pkRendered, "index primary_key (pk):")
}

func TestRenderExplainUnfilteredVectorSearch(t *testing.T) {
	unfilteredSeg := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 1000},
		Children: []*internalpb.ExecutionExplain{
			{
				Kind:       explain.KindOperator,
				Attributes: map[string]string{"name": "mvcc", "step": "0"},
				Counters:   map[string]int64{"rows_in": 1000, "rows_out": 990},
			},
			{
				Kind:       explain.KindOperator,
				Attributes: map[string]string{"name": "vector_search", "step": "1"},
				Counters:   map[string]int64{"rows_in": 990, "rows_out": 10},
			},
		},
	}
	rendered, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{unfilteredSeg}, nil, nil, nil, nil)
	require.NoError(t, err)
	assert.Contains(t, rendered, "[op 0] mvcc (segments=1, rows: 1000 -> 990)")
	assert.Contains(t, rendered, "[op 1] vector_search (segments=1, rows: 990 -> 10)")
}

func TestRenderExplainNestedCallExprPreservesField(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, Name: "address"}},
	}
	nestedCall := &planpb.Expr{
		Expr: &planpb.Expr_CallExpr{
			CallExpr: &planpb.CallExpr{
				FunctionName: "outer_fn",
				FunctionParameters: []*planpb.Expr{{
					Expr: &planpb.Expr_CallExpr{
						CallExpr: &planpb.CallExpr{
							FunctionName: "inner_fn",
							FunctionParameters: []*planpb.Expr{{
								Expr: &planpb.Expr_ColumnExpr{
									ColumnExpr: &planpb.ColumnExpr{
										Info: &planpb.ColumnInfo{FieldId: 101},
									},
								},
							}},
						},
					},
				}},
			},
		},
	}
	plan := &planpb.PlanNode{
		Node: &planpb.PlanNode_Query{
			Query: &planpb.QueryPlanNode{Predicates: nestedCall},
		},
	}

	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, true, -1, schema, &next, &leaves))
	require.Contains(t, leaves, int64(1))
	assert.Equal(t, int64(101), leaves[1].FieldID)
	assert.Equal(t, "address", leaves[1].Field)
	assert.Equal(t, "outer_fn(...)", leaves[1].Condition)
}

func TestConfigureExplainPlanAndEmptyExprUntagged(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, Name: "title"}},
	}
	newPlan := func() *planpb.PlanNode {
		return &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}},
					},
				},
			},
		}
	}

	// Empty user expression (hasUserExpr=false) enables explain without tagging
	// parser-synthesized AlwaysTrueExpr.
	var next int64
	var leaves map[int64]explain.LeafInfo
	plan := newPlan()
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, false, -1, schema, &next, &leaves))
	assert.True(t, plan.GetPlanOptions().GetEnableExplain())
	assert.Equal(t, int64(0), plan.GetQuery().GetPredicates().GetExprId())
	assert.Empty(t, leaves)

	// User-written AlwaysTrue ("1 == 1", hasUserExpr=true) tags AlwaysTrueExpr.
	plan = newPlan()
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, true, -1, schema, &next, &leaves))
	assert.True(t, plan.GetPlanOptions().GetEnableExplain())
	assert.Equal(t, int64(1), plan.GetQuery().GetPredicates().GetExprId())
	assert.Len(t, leaves, 1)

	// Requery clears EnableExplain.
	require.NoError(t, configureExplainPlan(plan, enableJSONExplainKV(), true, true, -1, schema, &next, &leaves))
	assert.False(t, plan.GetPlanOptions().GetEnableExplain())
}

func TestExplainIndexNamesUsesListIndexes(t *testing.T) {
	defaultExplainIndexCache.purge()
	t.Cleanup(defaultExplainIndexCache.purge)

	const collID = int64(42)
	mix := mocks.NewMockMixCoordClient(t)
	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{
		CollectionID: collID,
	}).Return(&indexpb.ListIndexesResponse{
		Status: merr.Success(),
		IndexInfos: []*indexpb.IndexInfo{
			{CollectionID: collID, FieldID: 101, IndexID: 9001, IndexName: "title_inv"},
			{CollectionID: collID, FieldID: 101, IndexID: 9002, IndexName: "title_ngram"},
		},
	}, nil).Once()

	node := &fakeHintNode{mix: mix}
	names := explainIndexNames(context.Background(), node, collID, []int64{9001, 9002})
	assert.Equal(t, map[int64]string{
		9001: "title_inv",
		9002: "title_ngram",
	}, names)

	cached := explainIndexNames(context.Background(), node, collID, []int64{9001})
	assert.Equal(t, names, cached)
}

func TestExplainIndexCacheRefreshTimeoutAndCooldown(t *testing.T) {
	now := time.Now()
	clock := func() time.Time { return now }

	t.Run("new index triggers refresh while unknown id is rate-limited", func(t *testing.T) {
		cache := newExplainIndexCache(16, 100*time.Millisecond, clock)
		const collID = int64(77)
		mix := mocks.NewMockMixCoordClient(t)

		mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{
			CollectionID: collID,
		}).Return(&indexpb.ListIndexesResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{CollectionID: collID, FieldID: 101, IndexID: 1001, IndexName: "idx_v1"},
			},
		}, nil).Once()

		names := cache.resolve(context.Background(), mix, collID, []int64{1001, 9999})
		assert.Equal(t, map[int64]string{1001: "idx_v1"}, names)

		names = cache.resolve(context.Background(), mix, collID, []int64{1001, 9999})
		assert.Equal(t, map[int64]string{1001: "idx_v1"}, names)

		mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{
			CollectionID: collID,
		}).Return(&indexpb.ListIndexesResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{CollectionID: collID, FieldID: 101, IndexID: 1001, IndexName: "idx_v1"},
				{CollectionID: collID, FieldID: 101, IndexID: 1002, IndexName: "idx_v2"},
			},
		}, nil).Once()

		names = cache.resolve(context.Background(), mix, collID, []int64{1002})
		assert.Equal(t, map[int64]string{
			1001: "idx_v1",
			1002: "idx_v2",
		}, names)
	})

	t.Run("bounded lookup timeout and failure cooldown", func(t *testing.T) {
		cache := newExplainIndexCache(16, 30*time.Millisecond, clock)
		const collID = int64(88)
		mix := mocks.NewMockMixCoordClient(t)

		mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{
			CollectionID: collID,
		}).RunAndReturn(func(ctx context.Context, _ *indexpb.ListIndexesRequest, _ ...grpc.CallOption) (*indexpb.ListIndexesResponse, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}).Once()

		start := time.Now()
		names := cache.resolve(context.Background(), mix, collID, []int64{5001})
		assert.Nil(t, names)
		assert.Less(t, time.Since(start), 500*time.Millisecond)
		assert.Nil(t, cache.resolve(context.Background(), mix, collID, []int64{5001}))
	})
}

func TestDescribeExplainLeafPreservesTextMatchAndGIS(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{
				FieldID:  101,
				Name:     "body",
				DataType: schemapb.DataType_VarChar,
				TypeParams: []*commonpb.KeyValuePair{
					{Key: "max_length", Value: "256"},
					{Key: "enable_match", Value: "true"},
				},
			},
			{FieldID: 102, Name: "geo", DataType: schemapb.DataType_Geometry},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	describe := func(exprStr string, tmpl map[string]*schemapb.TemplateValue) string {
		parsed, err := planparserv2.ParseExpr(helper, exprStr, tmpl)
		require.NoError(t, err)
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{Predicates: parsed},
			},
		}
		var next int64
		var leaves map[int64]explain.LeafInfo
		require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, true, -1, schema, &next, &leaves))
		require.Contains(t, leaves, int64(1))
		return leaves[1].Condition
	}

	assert.Equal(t, `phrase_match(body, "red fox", slop=1)`, describe(`phrase_match(body, "red fox", 1)`, nil))
	assert.Equal(t, `text_match(body, "red fox", minimum_should_match=2)`, describe(`text_match(body, "red fox", minimum_should_match=2)`, nil))
	assert.Equal(t, `text_match_fuzzy(body, "query", max_edit_distance=1)`, describe(`text_match_fuzzy(body, "query", max_edit_distance=1)`, nil))
	assert.Equal(t, `geo dwithin "POINT (12.34 56.78)" distance=25`, describe(`st_dwithin(geo, {pt}, 25)`, map[string]*schemapb.TemplateValue{
		"pt": {Val: &schemapb.TemplateValue_StringVal{StringVal: "POINT (12.34 56.78)"}},
	}))
}

func TestConcurrentEmptyRetrievePlanExplainTaggingIsRaceFree(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{
		Name: "test_empty_expr",
		Fields: []*schemapb.FieldSchema{{
			FieldID:      100,
			Name:         "pk",
			IsPrimaryKey: true,
			DataType:     schemapb.DataType_Int64,
		}},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		workerID := int64(i * 10)
		go func() {
			defer wg.Done()
			plan, err := planparserv2.CreateRetrievePlan(helper, "", nil)
			require.NoError(t, err)
			next := workerID
			var leaves map[int64]explain.LeafInfo
			require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, true, -1, schema, &next, &leaves))
			raw, err := proto.Marshal(plan)
			require.NoError(t, err)
			var decoded planpb.PlanNode
			require.NoError(t, proto.Unmarshal(raw, &decoded))
			assert.Equal(t, workerID+1, decoded.GetQuery().GetPredicates().GetExprId())
		}()
	}
	wg.Wait()
}

func TestAttachExplainPreservesReportValueCompatibility(t *testing.T) {
	var emptyStatus *commonpb.Status
	attachExplain(&emptyStatus, "")
	assert.Nil(t, emptyStatus)

	var nilStatus *commonpb.Status
	attachExplain(&nilStatus, "explain_text")
	require.NotNil(t, nilStatus)
	assert.Equal(t, "0", nilStatus.ExtraInfo[reportValueKey])
	assert.Equal(t, "explain_text", nilStatus.ExtraInfo[explainKey])
}

func TestRewrittenTemplatedPredicatesRenderBoundValuesInExplain(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{
		Name: "test_rewritten_templates",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", IsPrimaryKey: true, DataType: schemapb.DataType_Int64},
			{FieldID: 101, Name: "email", DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "128"}}},
			{FieldID: 102, Name: "age", DataType: schemapb.DataType_Int64},
			{FieldID: 103, Name: "tags", DataType: schemapb.DataType_Array, ElementType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: "max_capacity", Value: "16"}, {Key: "max_length", Value: "64"}}},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	templates := map[string]*schemapb.TemplateValue{
		"ea": {Val: &schemapb.TemplateValue_StringVal{StringVal: "alice@private.example"}},
		"eb": {Val: &schemapb.TemplateValue_StringVal{StringVal: "bob@private.example"}},
		"lo": {Val: &schemapb.TemplateValue_Int64Val{Int64Val: 31337}},
		"hi": {Val: &schemapb.TemplateValue_Int64Val{Int64Val: 99991}},
		"ta": {Val: &schemapb.TemplateValue_StringVal{StringVal: "tag_alpha_bound"}},
		"tb": {Val: &schemapb.TemplateValue_StringVal{StringVal: "tag_beta_bound"}},
	}

	exprStr := `(email == {ea} or email == {eb}) and (age > {lo} and age < {hi}) and (array_contains(tags, {ta}) or array_contains(tags, {tb}))`
	plan, err := planparserv2.CreateRetrievePlan(helper, exprStr, templates)
	require.NoError(t, err)

	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, true, -1, schema, &next, &leaves))
	require.Len(t, leaves, 3)

	var children []*internalpb.ExecutionExplain
	for id, info := range leaves {
		children = append(children, &internalpb.ExecutionExplain{
			Kind: explain.KindFilterGroup,
			Attributes: map[string]string{
				"expr_id":        strconv.FormatInt(id, 10),
				"field_id":       strconv.FormatInt(info.FieldID, 10),
				"decline_reason": "no_index_on_field",
			},
			Counters: map[string]int64{"rows_in": 10, "idx_scan": 0, "raw_scan": 10, "rows_out": 2},
		})
	}

	seg := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 10},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       explain.KindOperator,
			Attributes: map[string]string{"name": "filter", "step": "0"},
			Counters:   map[string]int64{"rows_in": 10, "rows_out": 2},
			Children:   children,
		}},
	}
	textOut, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, schema, nil, nil)
	require.NoError(t, err)
	assert.Contains(t, textOut, "alice@private.example")
	assert.Contains(t, textOut, "31337")
	assert.Contains(t, textOut, "tag_alpha_bound")
}

func TestDescribeExplainLeafMissingSchemaNeverEmitsFieldIdInAnyPosition(t *testing.T) {
	existsExpr := &planpb.Expr{Expr: &planpb.Expr_ExistsExpr{ExistsExpr: &planpb.ExistsExpr{
		Info: &planpb.ColumnInfo{FieldId: 101, NestedPath: []string{"k"}},
	}}}
	arrLenExpr := &planpb.Expr{Expr: &planpb.Expr_BinaryArithOpEvalRangeExpr{BinaryArithOpEvalRangeExpr: &planpb.BinaryArithOpEvalRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 102},
		ArithOp:    planpb.ArithOpType_ArrayLength,
		Op:         planpb.OpType_GreaterThan,
		Value:      intLiteral(2),
	}}}
	inFieldExpr := &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 103},
		IsInField:  true,
		Values:     []*planpb.GenericValue{intLiteral(5)},
	}}}
	cmpExpr := &planpb.Expr{Expr: &planpb.Expr_CompareExpr{CompareExpr: &planpb.CompareExpr{
		LeftColumnInfo:  &planpb.ColumnInfo{FieldId: 104},
		RightColumnInfo: &planpb.ColumnInfo{FieldId: 105},
		Op:              planpb.OpType_LessThan,
	}}}

	for _, e := range []*planpb.Expr{existsExpr, arrLenExpr, inFieldExpr, cmpExpr} {
		cond, _, _ := describeExplainLeaf(e, nil)
		assert.NotContains(t, cond, "field_10")
		assert.Contains(t, cond, "unknown")
	}
}
