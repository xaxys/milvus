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

	// Synchronous explain renders bound template values verbatim for the caller.
	templated := &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{
		ColumnInfo:           &planpb.ColumnInfo{FieldId: 103},
		Op:                   planpb.OpType_Equal,
		Value:                intLiteral(42),
		TemplateVariableName: "v",
	}}}
	condition, _, _ = describeExplainLeaf(templated, fields)
	assert.Equal(t, "price == 42", condition)

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

	tsFields := map[int64]string{106: "ts"}
	tsAdd := &planpb.Expr{Expr: &planpb.Expr_TimestamptzArithCompareExpr{TimestamptzArithCompareExpr: &planpb.TimestamptzArithCompareExpr{
		TimestamptzColumn: &planpb.ColumnInfo{FieldId: 106},
		ArithOp:           planpb.ArithOpType_Add,
		Interval:          &planpb.Interval{Days: 1},
		CompareOp:         planpb.OpType_GreaterThan,
		CompareValue:      stringLiteral("2026-01-01T00:00:00Z"),
	}}}
	condition, typ, fieldID = describeExplainLeaf(tsAdd, tsFields)
	assert.Equal(t, `ts add interval(P1D) greaterthan "2026-01-01T00:00:00Z"`, condition)
	assert.Equal(t, "timestamptz", typ)
	assert.Equal(t, int64(106), fieldID)

	tsSub := &planpb.Expr{Expr: &planpb.Expr_TimestamptzArithCompareExpr{TimestamptzArithCompareExpr: &planpb.TimestamptzArithCompareExpr{
		TimestamptzColumn: &planpb.ColumnInfo{FieldId: 106},
		ArithOp:           planpb.ArithOpType_Sub,
		Interval:          &planpb.Interval{Hours: 2},
		CompareOp:         planpb.OpType_LessThan,
		CompareValue:      stringLiteral("2026-01-01T00:00:00Z"),
	}}}
	condition, typ, _ = describeExplainLeaf(tsSub, tsFields)
	assert.Equal(t, `ts sub interval(PT2H) lessthan "2026-01-01T00:00:00Z"`, condition)
	assert.Equal(t, "timestamptz", typ)
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

	enabled, format, level, err = explainParams([]*commonpb.KeyValuePair{kv(explainKey, "FALSE")})
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

	_, _, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainFormatKey, "none")})
	assert.Error(t, err)

	_, _, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainFormatKey, "yaml")})
	assert.Error(t, err)

	_, _, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainKey, "off")})
	assert.Error(t, err)

	_, _, _, err = explainParams([]*commonpb.KeyValuePair{kv(explainKey, "verbose")})
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
			Counters: map[string]int64{"rows_in": 50000, "idx_scan": 50000, "raw_scan": 0, "rows_out": 31},
		}},
	}
}

func TestRenderExplainLevelsAndFormats(t *testing.T) {
	leaves := map[int64]explain.LeafInfo{3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101}}

	// Default (explain omitted) produces empty output.
	rendered, err := renderExplain(nil, []*internalpb.ExecutionExplain{explainSegment()}, leaves, nil)
	require.NoError(t, err)
	assert.Empty(t, rendered)

	rendered, err = renderExplain([]*commonpb.KeyValuePair{{Key: explainKey, Value: "false"}}, []*internalpb.ExecutionExplain{explainSegment()}, leaves, nil)
	require.NoError(t, err)
	assert.Empty(t, rendered)

	rendered, err = renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{explainSegment()}, leaves, nil)
	require.NoError(t, err)
	assert.Contains(t, rendered, `expr title contains "%降噪%"`)
	assert.Contains(t, rendered, "index 459123456789100001 ngram")
	assert.Contains(t, rendered, "rows_out=31")

	rendered, err = renderExplain(enableJSONExplainKV(), []*internalpb.ExecutionExplain{explainSegment()}, leaves, nil)
	require.NoError(t, err)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal([]byte(rendered), &decoded))
	assert.Equal(t, "summary", decoded["level"])

	_, err = renderExplain([]*commonpb.KeyValuePair{{Key: explainFormatKey, Value: "none"}}, []*internalpb.ExecutionExplain{explainSegment()}, leaves, nil)
	assert.Error(t, err)

	_, err = renderExplain([]*commonpb.KeyValuePair{{Key: explainKey, Value: "detail"}}, []*internalpb.ExecutionExplain{explainSegment()}, leaves, nil)
	assert.Error(t, err)

	rendered, err = renderExplain(enableExplainKV(), nil, leaves, nil)
	require.NoError(t, err)
	assert.Empty(t, rendered)
}

func TestRenderExplainUsesIndexNames(t *testing.T) {
	leaves := map[int64]explain.LeafInfo{3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101}}
	names := map[int64]string{459123456789100001: "title_ngram"}
	segments := []*internalpb.ExecutionExplain{explainSegment()}

	rendered, err := renderExplain(enableExplainKV(), segments, leaves, func([]int64) map[int64]string { return names })
	require.NoError(t, err)
	assert.Contains(t, rendered, "index title_ngram ngram")

	rendered, err = renderExplain(
		enableJSONExplainKV(),
		segments, leaves, func([]int64) map[int64]string { return names })
	require.NoError(t, err)
	assert.Contains(t, rendered, `"index_name":"title_ngram"`)
	assert.Contains(t, rendered, `"index_id":459123456789100001`)
}

func TestRenderExplainSkipsIndexLookupForPKOnly(t *testing.T) {
	leaves := map[int64]explain.LeafInfo{1: {Condition: "id == 5", Type: "equal", FieldID: 100}}
	pkSegment := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 100},
		Children: []*internalpb.ExecutionExplain{{
			Kind: explain.KindFilterGroup,
			Attributes: map[string]string{
				"expr_id": "1", "field_id": "100", "family": "pk",
				"index_id": "0", "nested": "false",
			},
			Counters: map[string]int64{"rows_in": 100, "idx_scan": 100, "raw_scan": 0, "rows_out": 1},
		}},
	}
	called := false
	rendered, err := renderExplain(
		enableExplainKV(),
		[]*internalpb.ExecutionExplain{pkSegment},
		leaves,
		func([]int64) map[int64]string {
			called = true
			return nil
		})
	require.NoError(t, err)
	assert.False(t, called, "PK lookup (index_id=0) must not invoke coordinator index lookup")
	assert.Contains(t, rendered, "index pk pk")
}

func TestRenderExplainCallExprPreservesFieldAndCounts(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, Name: "address"}},
	}
	callNode := &planpb.Expr{
		Expr: &planpb.Expr_CallExpr{
			CallExpr: &planpb.CallExpr{
				FunctionName: "empty",
				FunctionParameters: []*planpb.Expr{{
					Expr: &planpb.Expr_ColumnExpr{
						ColumnExpr: &planpb.ColumnExpr{
							Info: &planpb.ColumnInfo{FieldId: 101},
						},
					},
				}},
			},
		},
	}
	plan := &planpb.PlanNode{
		Node: &planpb.PlanNode_Query{
			Query: &planpb.QueryPlanNode{Predicates: callNode},
		},
	}

	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
	assert.Equal(t, int64(1), callNode.GetExprId())
	require.Contains(t, leaves, int64(1))
	assert.Equal(t, int64(101), leaves[1].FieldID)
	assert.Equal(t, "empty(...)", leaves[1].Condition)
	assert.Equal(t, "call", leaves[1].Type)

	seg := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 10},
		Children: []*internalpb.ExecutionExplain{{
			Kind: explain.KindFilterGroup,
			Attributes: map[string]string{
				"expr_id":        "1",
				"field_id":       "101",
				"nested":         "false",
				"decline_reason": "no_index_on_field",
			},
			Counters: map[string]int64{"rows_in": 10, "idx_scan": 0, "raw_scan": 10, "rows_out": 2},
		}},
	}
	rendered, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, nil)
	require.NoError(t, err)
	assert.Contains(t, rendered, "expr empty(...) [idx_scan=0, raw_scan=10, rows_in=10, rows_out=2, segments=1] {expr_id=1, field_id=101, type=call}")
	assert.Contains(t, rendered, "raw [idx_scan=0, raw_scan=10, rows_in=10, rows_out=2, segments=1] {decline_reason=no_index_on_field}")
}

func TestConfigureExplainPlan(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, Name: "title"}},
	}
	newPlan := func() *planpb.PlanNode {
		return &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_UnaryRangeExpr{
							UnaryRangeExpr: &planpb.UnaryRangeExpr{
								ColumnInfo: &planpb.ColumnInfo{FieldId: 101},
								Op:         planpb.OpType_Equal,
								Value:      stringLiteral("a"),
							},
						},
					},
				},
			},
		}
	}

	// Default (omitted explain) keeps EnableExplain=false and skips leaf tagging.
	var next int64
	var leaves map[int64]explain.LeafInfo
	plan := newPlan()
	require.NoError(t, configureExplainPlan(plan, nil, false, schema, &next, &leaves))
	assert.False(t, plan.GetPlanOptions().GetEnableExplain())
	assert.Equal(t, int64(0), plan.GetQuery().GetPredicates().GetExprId())
	assert.Empty(t, leaves)

	// explain=false keeps EnableExplain=false and skips leaf tagging.
	plan = newPlan()
	err := configureExplainPlan(plan, []*commonpb.KeyValuePair{{Key: explainKey, Value: "false"}}, false, schema, &next, &leaves)
	require.NoError(t, err)
	assert.False(t, plan.GetPlanOptions().GetEnableExplain())
	assert.Equal(t, int64(0), plan.GetQuery().GetPredicates().GetExprId())
	assert.Empty(t, leaves)

	// explain=default / summary sets EnableExplain=true and tags leaves.
	next = 0
	leaves = nil
	plan = newPlan()
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
	assert.True(t, plan.GetPlanOptions().GetEnableExplain())
	assert.Equal(t, int64(1), plan.GetQuery().GetPredicates().GetExprId())
	assert.Len(t, leaves, 1)

	// Requery / forced disableExplain clears EnableExplain even on a reused plan.
	err = configureExplainPlan(plan, enableJSONExplainKV(), true, schema, &next, &leaves)
	require.NoError(t, err)
	assert.False(t, plan.GetPlanOptions().GetEnableExplain())

	// Unsupported explain=detail fails fast.
	plan = newPlan()
	err = configureExplainPlan(plan, []*commonpb.KeyValuePair{{Key: explainKey, Value: "detail"}}, false, schema, &next, &leaves)
	assert.Error(t, err)

	// Invalid explain_format=none fails fast.
	plan = newPlan()
	err = configureExplainPlan(plan, []*commonpb.KeyValuePair{{Key: explainFormatKey, Value: "none"}}, false, schema, &next, &leaves)
	assert.Error(t, err)
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

	// Repeat queries referencing already-cached index IDs must not issue a
	// second ListIndexes RPC.
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

		// Unknown id 9999 is suppressed during cooldown -> zero extra RPCs.
		names = cache.resolve(context.Background(), mix, collID, []int64{1001, 9999})
		assert.Equal(t, map[int64]string{1001: "idx_v1"}, names)

		// A newly created index (1002) is not in unresolved -> refreshes immediately.
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

		// During failure cooldown, subsequent lookups skip MixCoord immediately.
		assert.Nil(t, cache.resolve(context.Background(), mix, collID, []int64{5001}))
	})
}

func TestDescribeExplainLeafPreservesTextMatchExtraValues(t *testing.T) {
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
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	describe := func(exprStr string) string {
		parsed, err := planparserv2.ParseExpr(helper, exprStr, nil)
		require.NoError(t, err)
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{Predicates: parsed},
			},
		}
		var next int64
		var leaves map[int64]explain.LeafInfo
		require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
		require.Contains(t, leaves, int64(1))
		return leaves[1].Condition
	}

	slop0 := describe(`phrase_match(body, "red fox", 0)`)
	slop1 := describe(`phrase_match(body, "red fox", 1)`)
	assert.Equal(t, `phrase_match(body, "red fox", slop=0)`, slop0)
	assert.Equal(t, `phrase_match(body, "red fox", slop=1)`, slop1)
	assert.NotEqual(t, slop0, slop1)

	assert.Equal(t,
		`text_match(body, "red fox", minimum_should_match=2)`,
		describe(`text_match(body, "red fox", minimum_should_match=2)`))
	assert.Equal(t,
		`text_match_fuzzy(body, "query", max_edit_distance=1)`,
		describe(`text_match_fuzzy(body, "query", max_edit_distance=1)`))
}

func TestTemplatedGISExprRendersBoundWKT(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "geo", DataType: schemapb.DataType_Geometry},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	const boundPoint = "POINT (12.345 67.89)"
	parsed, err := planparserv2.ParseExpr(helper, `st_contains(geo, {wkt})`, map[string]*schemapb.TemplateValue{
		"wkt": {
			Val: &schemapb.TemplateValue_StringVal{StringVal: boundPoint},
		},
	})
	require.NoError(t, err)
	require.True(t, parsed.GetIsTemplate())

	plan := &planpb.PlanNode{
		Node: &planpb.PlanNode_Query{
			Query: &planpb.QueryPlanNode{Predicates: parsed},
		},
	}
	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
	require.Contains(t, leaves, int64(1))
	assert.Equal(t, `geo contains "POINT (12.345 67.89)"`, leaves[1].Condition)

	seg := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 10},
		Children: []*internalpb.ExecutionExplain{{
			Kind: explain.KindFilterGroup,
			Attributes: map[string]string{
				"expr_id":        "1",
				"field_id":       "101",
				"nested":         "false",
				"decline_reason": "no_index_on_field",
			},
			Counters: map[string]int64{"rows_in": 10, "idx_scan": 0, "raw_scan": 10, "rows_out": 1},
		}},
	}

	textOut, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, nil)
	require.NoError(t, err)
	assert.Contains(t, textOut, `expr geo contains "POINT (12.345 67.89)"`)

	jsonOut, err := renderExplain(
		enableJSONExplainKV(),
		[]*internalpb.ExecutionExplain{seg},
		leaves,
		nil,
	)
	require.NoError(t, err)
	assert.Contains(t, jsonOut, `"condition":"geo contains \"POINT (12.345 67.89)\""`)
}

func TestSTDWithinDistanceRendering(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "geo", DataType: schemapb.DataType_Geometry},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	nearExpr, err := planparserv2.ParseExpr(helper, `st_dwithin(geo, "POINT (0 0)", 10) and st_dwithin(geo, "POINT (0 0)", 50.5)`, nil)
	require.NoError(t, err)
	plan := &planpb.PlanNode{
		Node: &planpb.PlanNode_Query{
			Query: &planpb.QueryPlanNode{Predicates: nearExpr},
		},
	}
	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
	require.Len(t, leaves, 2)
	assert.Equal(t, `geo dwithin "POINT (0 0)" distance=10`, leaves[1].Condition)
	assert.Equal(t, `geo dwithin "POINT (0 0)" distance=50.5`, leaves[2].Condition)
	assert.NotEqual(t, leaves[1].Condition, leaves[2].Condition)

	const boundPoint = "POINT (12.34 56.78)"
	templated, err := planparserv2.ParseExpr(helper, `st_dwithin(geo, {pt}, 25)`, map[string]*schemapb.TemplateValue{
		"pt": {Val: &schemapb.TemplateValue_StringVal{StringVal: boundPoint}},
	})
	require.NoError(t, err)
	tPlan := &planpb.PlanNode{
		Node: &planpb.PlanNode_Query{
			Query: &planpb.QueryPlanNode{Predicates: templated},
		},
	}
	var tNext int64
	var tLeaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(tPlan, enableExplainKV(), false, schema, &tNext, &tLeaves))
	require.Contains(t, tLeaves, int64(1))
	assert.Equal(t, `geo dwithin "POINT (12.34 56.78)" distance=25`, tLeaves[1].Condition)
}

func TestRenderExplainBackfillsUnobservedPlanLeaves(t *testing.T) {
	emptySeg := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg_cached",
		Counters: map[string]int64{"rows_total": 100},
	}

	// Unfiltered search (no plan leaves, no filter_groups) returns empty string.
	unfiltered, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{emptySeg}, nil, nil)
	require.NoError(t, err)
	assert.Empty(t, unfiltered)

	// Filtered query whose segments emitted no filter_groups (e.g. whole-filter
	// expression cache hit) still lists every plan leaf with 0 counters and
	// skipped=false.
	leaves := map[int64]explain.LeafInfo{
		1: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103},
	}
	textOut, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{emptySeg}, leaves, nil)
	require.NoError(t, err)
	assert.Contains(t, textOut, "expr price in (100, +inf) [idx_scan=0, raw_scan=0, rows_in=0, rows_out=0, segments=0] {expr_id=1, field_id=103, type=range}")
	assert.NotContains(t, textOut, "skipped=true")
}

func TestMatchAnyStructSubfieldResolvesName(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
		StructArrayFields: []*schemapb.StructArrayFieldSchema{{
			FieldID: 200,
			Name:    "struct_array",
			Fields: []*schemapb.FieldSchema{{
				FieldID:     201,
				Name:        "struct_array[sub_int]",
				DataType:    schemapb.DataType_Array,
				ElementType: schemapb.DataType_Int32,
			}},
		}},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	parsed, err := planparserv2.ParseExpr(helper, `MATCH_ANY(struct_array, $[sub_int] > 1)`, nil)
	require.NoError(t, err)

	plan := &planpb.PlanNode{
		Node: &planpb.PlanNode_Query{
			Query: &planpb.QueryPlanNode{Predicates: parsed},
		},
	}
	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
	require.Contains(t, leaves, int64(1))
	assert.Equal(t, "struct_array[sub_int] in (1, +inf)", leaves[1].Condition)
	assert.NotContains(t, leaves[1].Condition, "field_201")

	seg := &internalpb.ExecutionExplain{
		Kind:     explain.KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 10},
		Children: []*internalpb.ExecutionExplain{{
			Kind: explain.KindFilterGroup,
			Attributes: map[string]string{
				"expr_id":        "1",
				"field_id":       "201",
				"nested":         "true",
				"decline_reason": "no_index_on_field",
			},
			Counters: map[string]int64{"rows_in": 20, "idx_scan": 0, "raw_scan": 20, "rows_out": 5},
		}},
	}

	textOut, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, nil)
	require.NoError(t, err)
	assert.Contains(t, textOut, "expr struct_array[sub_int] in (1, +inf)")
	assert.NotContains(t, textOut, "field_201")

	jsonOut, err := renderExplain(
		enableJSONExplainKV(),
		[]*internalpb.ExecutionExplain{seg},
		leaves,
		nil,
	)
	require.NoError(t, err)
	assert.Contains(t, jsonOut, `"condition":"struct_array[sub_int] in (1, +inf)"`)
	assert.NotContains(t, jsonOut, "field_201")
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
			require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
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

	preset := &commonpb.Status{
		ExtraInfo: map[string]string{reportValueKey: "42"},
	}
	attachExplain(&preset, "explain_json")
	assert.Equal(t, "42", preset.ExtraInfo[reportValueKey])
	assert.Equal(t, "explain_json", preset.ExtraInfo[explainKey])
}

func TestRewrittenTemplatedPredicatesRenderBoundValuesInExplain(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{
		Name: "test_rewritten_templates",
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      100,
				Name:         "pk",
				IsPrimaryKey: true,
				DataType:     schemapb.DataType_Int64,
			},
			{
				FieldID:    101,
				Name:       "email",
				DataType:   schemapb.DataType_VarChar,
				TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "128"}},
			},
			{
				FieldID:  102,
				Name:     "age",
				DataType: schemapb.DataType_Int64,
			},
			{
				FieldID:     103,
				Name:        "tags",
				DataType:    schemapb.DataType_Array,
				ElementType: schemapb.DataType_VarChar,
				TypeParams: []*commonpb.KeyValuePair{
					{Key: "max_capacity", Value: "16"},
					{Key: "max_length", Value: "64"},
				},
			},
		},
	}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)

	const (
		boundEmailA = "alice@private.example"
		boundEmailB = "bob@private.example"
		boundAgeLo  = int64(31337)
		boundAgeHi  = int64(99991)
		boundTagA   = "tag_alpha_bound"
		boundTagB   = "tag_beta_bound"
	)

	templates := map[string]*schemapb.TemplateValue{
		"ea": {Val: &schemapb.TemplateValue_StringVal{StringVal: boundEmailA}},
		"eb": {Val: &schemapb.TemplateValue_StringVal{StringVal: boundEmailB}},
		"lo": {Val: &schemapb.TemplateValue_Int64Val{Int64Val: boundAgeLo}},
		"hi": {Val: &schemapb.TemplateValue_Int64Val{Int64Val: boundAgeHi}},
		"ta": {Val: &schemapb.TemplateValue_StringVal{StringVal: boundTagA}},
		"tb": {Val: &schemapb.TemplateValue_StringVal{StringVal: boundTagB}},
	}

	exprStr := `(email == {ea} or email == {eb}) and (age > {lo} and age < {hi}) and (array_contains(tags, {ta}) or array_contains(tags, {tb}))`
	plan, err := planparserv2.CreateRetrievePlan(helper, exprStr, templates)
	require.NoError(t, err)

	var next int64
	var leaves map[int64]explain.LeafInfo
	require.NoError(t, configureExplainPlan(plan, enableExplainKV(), false, schema, &next, &leaves))
	require.Len(t, leaves, 3)

	var children []*internalpb.ExecutionExplain
	for id, info := range leaves {
		assert.NotContains(t, info.Condition, "<template")
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
		Children: children,
	}
	textOut, err := renderExplain(enableExplainKV(), []*internalpb.ExecutionExplain{seg}, leaves, nil)
	require.NoError(t, err)
	jsonOut, err := renderExplain(
		enableJSONExplainKV(),
		[]*internalpb.ExecutionExplain{seg},
		leaves,
		nil,
	)
	require.NoError(t, err)

	// Synchronous explain responses return bound literals verbatim to the caller.
	for _, out := range []string{textOut, jsonOut} {
		assert.NotContains(t, out, "<template")
		assert.Contains(t, out, boundEmailA)
		assert.Contains(t, out, boundEmailB)
		assert.Contains(t, out, "31337")
		assert.Contains(t, out, "99991")
		assert.Contains(t, out, boundTagA)
		assert.Contains(t, out, boundTagB)
	}

	// Server-side plan log redaction remains intact.
	redactedLog := planparserv2.RedactPlanForLog(plan).String()
	assert.NotContains(t, redactedLog, boundEmailA)
	assert.NotContains(t, redactedLog, boundEmailB)
	assert.NotContains(t, redactedLog, "31337")
	assert.NotContains(t, redactedLog, "99991")
	assert.NotContains(t, redactedLog, boundTagA)
	assert.NotContains(t, redactedLog, boundTagB)
}
