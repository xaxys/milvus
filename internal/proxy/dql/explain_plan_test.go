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
	"strings"
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
	"github.com/milvus-io/milvus/internal/proxy/metacache"
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

func renderExplain(
	params []*commonpb.KeyValuePair,
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]explain.LeafInfo,
	schema *schemapb.CollectionSchema,
	subSearches []explain.SubSearchMeta,
	indexNames func([]int64) map[int64]string,
) (string, error) {
	return renderExplainWithAST(params, explains, leaves, nil, schema, subSearches, indexNames)
}

func configureExplainPlan(
	plan *planpb.PlanNode,
	params []*commonpb.KeyValuePair,
	disableExplain bool,
	hasUserExpr bool,
	subReqIndex int,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
) error {
	return configureExplainPlanWithAST(plan, params, disableExplain, hasUserExpr, subReqIndex, schema, next, leaves, nil)
}

func assignExplainIDsWithSubReq(
	plan *planpb.PlanNode,
	subReqIndex int,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
) {
	assignExplainIDsWithAST(plan, subReqIndex, schema, next, leaves, nil)
}

func describeExplainLeaf(node *planpb.Expr, fieldNames map[int64]string) (string, string, int64) {
	cond, typ, fid, _ := describeExplainLeafWithDynamic(node, fieldNames, nil)
	return cond, typ, fid
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

	colExpr := &planpb.Expr{Expr: &planpb.Expr_ColumnExpr{ColumnExpr: &planpb.ColumnExpr{
		Info: &planpb.ColumnInfo{FieldId: 101},
	}}}
	condition, typ, fieldID = describeExplainLeaf(colExpr, fields)
	assert.Equal(t, "title", condition)
	assert.Equal(t, "column", typ)
	assert.Equal(t, int64(101), fieldID)

	arithExpr := &planpb.Expr{Expr: &planpb.Expr_BinaryArithExpr{BinaryArithExpr: &planpb.BinaryArithExpr{
		Left:  colExpr,
		Right: &planpb.Expr{Expr: &planpb.Expr_ColumnExpr{ColumnExpr: &planpb.ColumnExpr{Info: &planpb.ColumnInfo{FieldId: 103}}}},
		Op:    planpb.ArithOpType_Add,
	}}}
	condition, typ, fieldID = describeExplainLeaf(arithExpr, fields)
	assert.Equal(t, "title add price", condition)
	assert.Equal(t, "arith", typ)
	assert.Equal(t, int64(101), fieldID)

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

	// Nested JSON path keys containing quotes or backslashes are escaped cleanly.
	escapedJSONCond, _, _ := describeExplainLeaf(
		&planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
			ColumnInfo: &planpb.ColumnInfo{FieldId: 105, NestedPath: []string{`a"b`, `c\d`}},
			Op:         planpb.OpType_Equal,
			Value:      intLiteral(1),
		}}},
		fields,
	)
	assert.Equal(t, `j["a\"b"]["c\\d"] == 1`, escapedJSONCond)
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

	cfg, err := ParseExplainConfig(nil)
	require.NoError(t, err)
	assert.False(t, cfg.Enabled)
	assert.Equal(t, "text", cfg.Format)

	cfg, err = ParseExplainConfig([]*commonpb.KeyValuePair{kv(explainKey, "default")})
	require.NoError(t, err)
	assert.True(t, cfg.Enabled)
	assert.Equal(t, "text", cfg.Format)

	cfg, err = ParseExplainConfig([]*commonpb.KeyValuePair{
		kv(explainKey, "SUMMARY"),
		kv(explainFormatKey, "JSON"),
	})
	require.NoError(t, err)
	assert.True(t, cfg.Enabled)
	assert.Equal(t, "json", cfg.Format)

	_, err = ParseExplainConfig([]*commonpb.KeyValuePair{kv(explainFormatKey, "yaml")})
	assert.Error(t, err)
	_, err = ParseExplainConfig([]*commonpb.KeyValuePair{kv(explainKey, "verbose")})
	assert.Error(t, err)
	_, err = ParseExplainConfig([]*commonpb.KeyValuePair{kv(explainKey, "detail")})
	assert.Error(t, err)

	// Nested ParamsKey JSON (RESTful v2 / search_params["params"]).
	cfg, err = ParseExplainConfig([]*commonpb.KeyValuePair{
		kv(ParamsKey, `{"explain": "summary", "explain_format": "json"}`),
	})
	require.NoError(t, err)
	assert.True(t, cfg.Enabled)
	assert.Equal(t, "json", cfg.Format)

	cfg, err = ParseExplainConfig([]*commonpb.KeyValuePair{
		kv(ParamsKey, `{"explain": true}`),
	})
	require.NoError(t, err)
	assert.True(t, cfg.Enabled)
	assert.Equal(t, "text", cfg.Format)

	// Top-level KV wins directly without scanning nested ParamsKey JSON.
	cfg, err = ParseExplainConfig([]*commonpb.KeyValuePair{
		kv(explainKey, "false"),
		kv(ParamsKey, `{"explain": "summary"}`),
	})
	require.NoError(t, err)
	assert.False(t, cfg.Enabled)

	cfg, err = ParseExplainConfig([]*commonpb.KeyValuePair{
		kv(explainKey, "summary"),
		kv(explainFormatKey, "text"),
		kv(ParamsKey, `{"explain_format": "json"}`),
	})
	require.NoError(t, err)
	assert.True(t, cfg.Enabled)
	assert.Equal(t, "text", cfg.Format)
}

func explainSegment() *internalpb.ExecutionExplain {
	return &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       1,
			SealedSegments: 1,
			RowsTotal:      50000,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments: 1,
					RowsIn:   50000,
					RowsOut:  31,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:   3,
						FieldId:  101,
						Segments: 1,
						RowsIn:   50000,
						IdxScan:  50000,
						RowsOut:  31,
						Used: []*internalpb.ExplainIndexUseStats{{
							IndexId:  459123456789100001,
							Family:   "ngram",
							Segments: 1,
							RowsIn:   50000,
							IdxScan:  50000,
							RowsOut:  31,
						}},
					}},
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
					Segments: 1,
					RowsIn:   31,
					RowsOut:  31,
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
					Segments: 1,
					RowsIn:   31,
					RowsOut:  10,
				},
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
	assert.Contains(t, rendered, "query (segments=1 [sealed=1, growing=0], rows_total=50000)")
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
	assert.EqualValues(t, 1, decoded["sealed_segments"])
	assert.EqualValues(t, 0, decoded["growing_segments"])
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
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       1,
			SealedSegments: 1,
			RowsTotal:      100,
			Operators: []*internalpb.ExplainOperatorStats{{
				Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
				Segments: 1,
				RowsIn:   100,
				RowsOut:  1,
				Conditions: []*internalpb.ExplainConditionStats{{
					ExprId:   1,
					FieldId:  100,
					Segments: 1,
					RowsIn:   100,
					IdxScan:  100,
					RowsOut:  1,
					Used: []*internalpb.ExplainIndexUseStats{{
						IndexId:  0,
						Family:   "pk",
						Segments: 1,
						RowsIn:   100,
						IdxScan:  100,
						RowsOut:  1,
					}},
				}},
			}},
		},
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
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       1,
			SealedSegments: 1,
			RowsTotal:      1000,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
					Segments: 1,
					RowsIn:   1000,
					RowsOut:  990,
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
					Segments: 1,
					RowsIn:   990,
					RowsOut:  10,
				},
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

func TestExplainIndexNamesUsesMetaCacheAndInvalidation(t *testing.T) {
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

	mc, err := metacache.NewMetaCache(mix)
	require.NoError(t, err)
	node := &fakeHintNode{mix: mix, cache: mc}

	names := explainIndexNames(context.Background(), node, collID, []int64{9001, 9002})
	assert.Equal(t, map[int64]string{
		9001: "title_inv",
		9002: "title_ngram",
	}, names)

	// Second call hits MetaCache without issuing another ListIndexes RPC.
	cached := explainIndexNames(context.Background(), node, collID, []int64{9001})
	assert.Equal(t, map[int64]string{9001: "title_inv"}, cached)

	// DDL invalidation evicts MetaCache so newly created indexes resolve immediately.
	mc.InvalidateCollectionIndexes(context.Background(), collID)
	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{
		CollectionID: collID,
	}).Return(&indexpb.ListIndexesResponse{
		Status: merr.Success(),
		IndexInfos: []*indexpb.IndexInfo{
			{CollectionID: collID, FieldID: 101, IndexID: 9001, IndexName: "title_inv"},
			{CollectionID: collID, FieldID: 101, IndexID: 9003, IndexName: "title_sort"},
		},
	}, nil).Once()
	refreshed := explainIndexNames(context.Background(), node, collID, []int64{9003})
	assert.Equal(t, map[int64]string{9003: "title_sort"}, refreshed)
}

func TestExplainIndexNamesBoundedTimeoutReturnsNil(t *testing.T) {
	const collID = int64(88)
	mix := mocks.NewMockMixCoordClient(t)
	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{
		CollectionID: collID,
	}).RunAndReturn(func(ctx context.Context, _ *indexpb.ListIndexesRequest, _ ...grpc.CallOption) (*indexpb.ListIndexesResponse, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}).Once()

	mc, err := metacache.NewMetaCache(mix)
	require.NoError(t, err)
	node := &fakeHintNode{mix: mix, cache: mc}

	start := time.Now()
	names := explainIndexNames(context.Background(), node, collID, []int64{5001})
	assert.Nil(t, names)
	assert.Less(t, time.Since(start), 2*time.Second)
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

	var conds []*internalpb.ExplainConditionStats
	for id, info := range leaves {
		conds = append(conds, &internalpb.ExplainConditionStats{
			ExprId:          id,
			FieldId:         info.FieldID,
			Segments:        1,
			RawScanSegments: 1,
			RowsIn:          10,
			RawScan:         10,
			RowsOut:         2,
			Reasons: []*internalpb.ExplainDeclineStats{{
				Reason:   "no_index",
				Segments: 1,
				RowsIn:   10,
				RawScan:  10,
				RowsOut:  2,
			}},
		})
	}

	seg := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       1,
			SealedSegments: 1,
			RowsTotal:      10,
			Operators: []*internalpb.ExplainOperatorStats{{
				Kind:       internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
				Segments:   1,
				RowsIn:     10,
				RowsOut:    2,
				Conditions: conds,
			}},
		},
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

	fieldNames := map[int64]string{104: "a", 105: "b"}
	cond, kind, fid, userField := describeExplainLeafWithDynamic(cmpExpr, fieldNames, nil)
	assert.Equal(t, "a lessthan b", cond)
	assert.Equal(t, "compare", kind)
	assert.Zero(t, fid)
	assert.Empty(t, userField)
}

func TestRenderExplainZeroSegmentsAndAttachZeroSegmentExplain(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{
		Name: "test_zero_seg",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", IsPrimaryKey: true, DataType: schemapb.DataType_Int64},
			{FieldID: 103, Name: "price", DataType: schemapb.DataType_Int64},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	// Unfiltered 0-segment explain emits valid summary instead of "".
	textUnfiltered, err := renderExplain(enableExplainKV(), nil, nil, schema, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "query (segments=0 [sealed=0, growing=0], rows_total=0)\n", textUnfiltered)

	jsonUnfiltered, err := renderExplain(enableJSONExplainKV(), nil, nil, schema, nil, nil)
	require.NoError(t, err)
	assert.JSONEq(t, `{"level":"summary","segments":0,"sealed_segments":0,"growing_segments":0,"rows_total":0}`, jsonUnfiltered)

	// Filtered 0-segment explain includes backfilled filter condition.
	leaves := map[int64]explain.LeafInfo{
		1: {Condition: "price in (10, +inf)", Type: "range", FieldID: 103, Field: "price"},
	}
	textFiltered, err := renderExplain(enableExplainKV(), nil, leaves, schema, nil, nil)
	require.NoError(t, err)
	assert.Contains(t, textFiltered, "query (segments=0 [sealed=0, growing=0], rows_total=0)")
	assert.Contains(t, textFiltered, "[op 0] filter (segments=0, rows: 0 -> 0)")
	assert.Contains(t, textFiltered, "expr: price in (10, +inf) (segments=0, rows: 0 -> 0, idx_scan=0, raw_scan=0)")

	// AttachZeroSegmentExplain handles disabled, invalid, and filtered early returns.
	var status *commonpb.Status
	require.NoError(t, AttachZeroSegmentExplain(&status, nil, "price > 10", nil, helper, schema))
	assert.Nil(t, status)

	assert.Error(t, ValidateExplainParams([]*commonpb.KeyValuePair{{Key: explainKey, Value: "invalid"}}))
	assert.Error(t, ValidateExplainParams([]*commonpb.KeyValuePair{{Key: explainKey, Value: "detail"}}))
	assert.Error(t, AttachZeroSegmentExplain(&status, []*commonpb.KeyValuePair{{Key: explainKey, Value: "detail"}}, "", nil, helper, schema))

	// Canceled request context skips index lookup instead of detaching via WithoutCancel.
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	mockNode := &fakeHintNode{mix: mocks.NewMockMixCoordClient(t)}
	assert.Nil(t, explainIndexNames(canceledCtx, mockNode, 1, []int64{9001}))

	require.NoError(t, AttachZeroSegmentExplain(&status, enableExplainKV(), "price > 10", nil, helper, schema))
	require.NotNil(t, status)
	assert.Contains(t, status.GetExtraInfo()[explainKey], "query (segments=0 [sealed=0, growing=0], rows_total=0)")
	assert.Contains(t, status.GetExtraInfo()[explainKey], "price in (10, +inf)")
}

func TestRenderExplainPreservesProxyASTTreeAndOrder(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Name: "ast_tree",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", IsPrimaryKey: true, DataType: schemapb.DataType_Int64},
			{FieldID: 101, Name: "title", DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "64"}}},
			{FieldID: 102, Name: "price", DataType: schemapb.DataType_Int64},
			{FieldID: 103, Name: "stock", DataType: schemapb.DataType_Int64},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	// PlanNode AST: (title == "pro" OR title like "%max%") AND NOT (stock in [1, 2])
	// Even when Segcore assigns eval_order=1 to stock and eval_order=3 to title=="pro",
	// rendering preserves Proxy's PlanNode AST tree and child order.
	plan, err := planparserv2.CreateRetrievePlan(helper, `(title == "pro" or title like "%max%") and not (stock in [1, 2])`, nil)
	require.NoError(t, err)

	var next int64
	var leaves map[int64]explain.LeafInfo
	var asts map[int]explain.PlanAST
	require.NoError(t, configureExplainPlanWithAST(plan, enableExplainKV(), false, true, -1, schema, &next, &leaves, &asts))
	require.Len(t, leaves, 3)

	seg := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:        2,
			SealedSegments:  1,
			GrowingSegments: 1,
			RowsTotal:       200,
			Operators: []*internalpb.ExplainOperatorStats{{
				Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
				Segments: 2,
				RowsIn:   200,
				RowsOut:  50,
				Conditions: []*internalpb.ExplainConditionStats{
					{
						ExprId:   1,
						FieldId:  101,
						Segments: 2,
						RowsIn:   200,
						RowsOut:  60,
						EvalOrders: []*internalpb.ExplainEvalOrderCount{{
							EvalOrder: 3,
							Segments:  2,
						}},
					},
					{
						ExprId:   2,
						FieldId:  101,
						Segments: 2,
						RowsIn:   140,
						RowsOut:  30,
						EvalOrders: []*internalpb.ExplainEvalOrderCount{{
							EvalOrder: 2,
							Segments:  2,
						}},
					},
					{
						ExprId:   3,
						FieldId:  103,
						Segments: 2,
						RowsIn:   90,
						RowsOut:  40,
						EvalOrders: []*internalpb.ExplainEvalOrderCount{{
							EvalOrder: 1,
							Segments:  2,
						}},
					},
				},
			}},
		},
	}

	textOut, err := renderExplainWithAST(enableExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, asts, schema, nil, nil)
	require.NoError(t, err)
	assert.Contains(t, textOut, "query (segments=2 [sealed=1, growing=1], rows_total=200)")
	assert.Contains(t, textOut, "    or:\n      expr: title contains \"max\"")
	assert.Contains(t, textOut, "    not:\n      expr: stock in [1, 2]")
	assert.Less(t, strings.Index(textOut, "or:\n"), strings.Index(textOut, "not:\n"))
	assert.Less(t, strings.Index(textOut, `title contains "max"`), strings.Index(textOut, `title == "pro"`))

	jsonOut, err := renderExplainWithAST(enableJSONExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, asts, schema, nil, nil)
	require.NoError(t, err)
	var summary explain.Summary
	require.NoError(t, json.Unmarshal([]byte(jsonOut), &summary))
	require.Len(t, summary.Operators, 1)
	require.Len(t, summary.Operators[0].Conditions, 2)
	assert.Equal(t, "or", summary.Operators[0].Conditions[0].Logic)
	require.Len(t, summary.Operators[0].Conditions[0].Children, 2)
	assert.Equal(t, `title contains "max"`, summary.Operators[0].Conditions[0].Children[0].Condition)
	assert.Equal(t, `title == "pro"`, summary.Operators[0].Conditions[0].Children[1].Condition)
	assert.Equal(t, "not", summary.Operators[0].Conditions[1].Logic)
	require.Len(t, summary.Operators[0].Conditions[1].Children, 1)
	assert.Equal(t, "stock in [1, 2]", summary.Operators[0].Conditions[1].Children[0].Condition)
}
