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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/explain"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func intLiteral(v int64) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: v}}
}

func stringLiteral(v string) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_StringVal{StringVal: v}}
}

func TestDescribeExplainLeaf(t *testing.T) {
	fields := map[int64]string{101: "title", 103: "price", 100: "sku_id", 105: "j"}

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

	// Templates must not echo the caller's value.
	templated := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo:           &planpb.ColumnInfo{FieldId: 103},
		Op:                   planpb.OpType_Equal,
		Value:                intLiteral(42),
		TemplateVariableName: "v",
	}}}
	condition, _, _ = describeExplainLeaf(templated, fields)
	assert.Equal(t, `price == <template "v">`, condition)

	like := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 101},
		Op:         planpb.OpType_InnerMatch,
		Value:      stringLiteral("%降噪%"),
	}}}
	condition, typ, _ = describeExplainLeaf(like, fields)
	assert.Equal(t, `title contains "%降噪%"`, condition)
	assert.Equal(t, "contains", typ)

	jsonPath := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 105, NestedPath: []string{"a", "b"}},
		Op:         planpb.OpType_GreaterThan,
		Value:      intLiteral(1),
	}}}
	condition, _, _ = describeExplainLeaf(jsonPath, fields)
	assert.Equal(t, `j["a"]["b"] in (1, +inf)`, condition)
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
	assert.Equal(t, "field_100 in [0, 1, 2, 3, 4, 5, 6, 7, ... <12 values>]", condition)
}

func TestAssignExplainIDsTagsLeavesOnly(t *testing.T) {
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "sku_id"},
		{FieldID: 103, Name: "price"},
	}}
	lower := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 103},
		Op:         planpb.OpType_GreaterThan,
		Value:      intLiteral(100),
	}}}
	sku := &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{
		ColumnInfo: &planpb.ColumnInfo{FieldId: 100},
		Values:     []*planpb.GenericValue{intLiteral(1), intLiteral(2)},
	}}}
	root := &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{
		Op:    planpb.BinaryExpr_LogicalAnd,
		Left:  lower,
		Right: sku,
	}}}
	plan := &planpb.PlanNode{Node: &planpb.PlanNode_VectorAnns{VectorAnns: &planpb.VectorANNS{
		Predicates: root,
	}}}

	var next int64
	var leaves map[int64]explain.LeafInfo
	assignExplainIDs(plan, schema, &next, &leaves)

	assert.Equal(t, int64(2), next, "only leaves get ids")
	assert.Equal(t, int64(0), root.GetExprId(), "logical nodes stay untagged")
	require.Len(t, leaves, 2)
	assert.Equal(t, "price in (100, +inf)", leaves[1].Condition)
	assert.Equal(t, "range", leaves[1].Type)
	assert.Equal(t, int64(103), leaves[1].FieldID)
	assert.Equal(t, "sku_id in [1, 2]", leaves[2].Condition)
	assert.Equal(t, "in", leaves[2].Type)
	assert.Equal(t, int64(100), leaves[2].FieldID)
}

func TestExplainParams(t *testing.T) {
	kv := func(key, value string) *commonpb.KeyValuePair {
		return &commonpb.KeyValuePair{Key: key, Value: value}
	}

	format, level, err := explainParams(nil)
	require.NoError(t, err)
	assert.Equal(t, "text", format)
	assert.Equal(t, "summary", level)

	format, level, err = explainParams([]*commonpb.KeyValuePair{kv(explainFormatKey, "JSON"), kv(explainLevelKey, "summary")})
	require.NoError(t, err)
	assert.Equal(t, "json", format)
	assert.Equal(t, "summary", level)

	format, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainFormatKey, "none")})
	require.NoError(t, err)
	assert.Equal(t, "none", format)

	_, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainFormatKey, "yaml")})
	assert.Error(t, err)

	_, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainLevelKey, "verbose")})
	assert.Error(t, err)
}

func explainSegment() *internalpb.ExecutionExplain {
	return &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "459123456789012346",
		Counters: map[string]int64{"rows_total": 50000},
		Children: []*internalpb.ExecutionExplain{{
			Kind: explain.KindFilterGroup,
			Attributes: map[string]string{
				"expr_id": "3", "field_id": "101", "family": "ngram",
				"index_id": "459123456789100001", "nested": "false",
			},
			Counters: map[string]int64{"leaves": 1, "rows_matched": 31, "rows_scanned": 0},
		}},
	}
}

func TestRenderExplainLevelsAndFormats(t *testing.T) {
	leaves := map[int64]explain.LeafInfo{3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101}}

	rendered, err := renderExplain(nil, []*internalpb.ExecutionExplain{explainSegment()}, leaves)
	require.NoError(t, err)
	assert.Contains(t, rendered, `expr title contains "%降噪%"`)
	assert.Contains(t, rendered, "index 459123456789100001 ngram")

	rendered, err = renderExplain([]*commonpb.KeyValuePair{{Key: explainFormatKey, Value: "json"}}, []*internalpb.ExecutionExplain{explainSegment()}, leaves)
	require.NoError(t, err)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(rendered), &decoded))
	assert.Equal(t, "summary", decoded["level"])

	rendered, err = renderExplain([]*commonpb.KeyValuePair{{Key: explainFormatKey, Value: "none"}}, []*internalpb.ExecutionExplain{explainSegment()}, leaves)
	require.NoError(t, err)
	assert.Empty(t, rendered)

	_, err = renderExplain([]*commonpb.KeyValuePair{{Key: explainLevelKey, Value: "detail"}}, []*internalpb.ExecutionExplain{explainSegment()}, leaves)
	assert.Error(t, err)

	rendered, err = renderExplain(nil, nil, leaves)
	require.NoError(t, err)
	assert.Empty(t, rendered)
}
