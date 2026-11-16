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

// Explain leaf table: the proxy tags each predicate leaf of the plan it sends
// with a request-unique expr_id and remembers how to describe that leaf. Segcore
// reports the id in the explain tree, which lets the proxy fold the per-segment
// decisions back into one condition.

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/internal/util/explain"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const (
	explainFormatKey = "explain_format"
	explainLevelKey  = "explain_level"

	explainFormatText = "text"
	explainFormatJSON = "json"
	explainFormatNone = "none"

	explainLevelSummary = "summary"
	explainLevelDetail  = "detail"
)

// explainParamDefaults are what a request that does not mention explain gets.
func explainParams(params []*commonpb.KeyValuePair) (format string, level string, err error) {
	format, level = explainFormatText, explainLevelSummary
	if raw, err := funcutil.GetAttrByKeyFromRepeatedKV(explainFormatKey, params); err == nil {
		switch value := strings.ToLower(strings.TrimSpace(raw)); value {
		case "":
		case explainFormatText, explainFormatJSON, explainFormatNone:
			format = value
		default:
			return "", "", merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected %q, %q or %q",
				explainFormatKey, raw, explainFormatText, explainFormatJSON, explainFormatNone)
		}
	}
	if raw, err := funcutil.GetAttrByKeyFromRepeatedKV(explainLevelKey, params); err == nil {
		switch value := strings.ToLower(strings.TrimSpace(raw)); value {
		case "":
		case explainLevelSummary, explainLevelDetail:
			level = value
		default:
			return "", "", merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected %q or %q",
				explainLevelKey, raw, explainLevelSummary, explainLevelDetail)
		}
	}
	return format, level, nil
}

// renderExplain turns the shard-reported trees into the client-visible payload.
// It returns "" when the request asked for none, or when no shard recorded
// anything (the caller then leaves extra_info alone).
func renderExplain(
	params []*commonpb.KeyValuePair,
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]explain.LeafInfo,
	indexNames map[int64]string,
) (string, error) {
	format, level, err := explainParams(params)
	if err != nil || format == explainFormatNone {
		return "", err
	}
	if level != explainLevelSummary {
		return "", merr.WrapErrParameterInvalidMsg(
			"%s %q is not supported yet", explainLevelKey, level)
	}

	segments := explain.Segments(explains...)
	if len(segments) == 0 {
		return "", nil
	}
	summary := explain.Summarize(segments, leaves)
	if len(summary.Conditions) == 0 {
		return "", nil
	}
	explain.NameIndexUses(summary, indexNames)
	if format == explainFormatJSON {
		return explain.RenderSummaryJSON(summary)
	}
	return explain.RenderSummaryText(summary), nil
}

// explainIndexNames resolves the indexes this collection carries, so the
// summary can name the one each segment used instead of only its id. A failure
// is not worth failing the query over: the rendered view keeps the ids.
func explainIndexNames(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
) map[int64]string {
	if node == nil || node.MixCoord() == nil {
		return nil
	}
	resp, err := node.MixCoord().DescribeIndex(ctx, &indexpb.DescribeIndexRequest{
		CollectionID: collectionID,
	})
	if err == nil {
		err = merr.Error(resp.GetStatus())
	}
	if err != nil {
		mlog.Warn(ctx, "index names unavailable for execution explain",
			mlog.Int64("collectionID", collectionID), mlog.Err(err))
		return nil
	}
	names := make(map[int64]string, len(resp.GetIndexInfos()))
	for _, info := range resp.GetIndexInfos() {
		names[info.GetIndexID()] = info.GetIndexName()
	}
	return names
}

// explainIndexNamesIfNeeded skips the coordinator lookup when the request asked
// for no explain payload, so a normal search pays nothing for this view.
func explainIndexNamesIfNeeded(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
	params []*commonpb.KeyValuePair,
) map[int64]string {
	format, _, err := explainParams(params)
	if err != nil || format == explainFormatNone {
		return nil
	}
	return explainIndexNames(ctx, node, collectionID)
}

// assignExplainIDs tags every predicate leaf of the plan and records how to
// describe it. next keeps ids unique across the sub-plans of one request, so a
// hybrid search does not merge two sub-searches into one condition.
func assignExplainIDs(
	plan *planpb.PlanNode,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
) {
	if plan == nil {
		return
	}
	var root *planpb.Expr
	if anns := plan.GetVectorAnns(); anns != nil {
		root = anns.GetPredicates()
	} else if query := plan.GetQuery(); query != nil {
		root = query.GetPredicates()
	}
	if root == nil {
		return
	}
	fieldNames := make(map[int64]string, len(schema.GetFields()))
	for _, field := range schema.GetFields() {
		fieldNames[field.GetFieldID()] = field.GetName()
	}
	if *leaves == nil {
		*leaves = make(map[int64]explain.LeafInfo, 4)
	}
	walkExplainLeaves(root, fieldNames, next, *leaves)
}

func walkExplainLeaves(
	node *planpb.Expr,
	fieldNames map[int64]string,
	next *int64,
	leaves map[int64]explain.LeafInfo,
) {
	if node == nil {
		return
	}
	// Logical nodes have no decision of their own; only their children are
	// reported by segcore.
	switch expr := node.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		walkExplainLeaves(expr.BinaryExpr.GetLeft(), fieldNames, next, leaves)
		walkExplainLeaves(expr.BinaryExpr.GetRight(), fieldNames, next, leaves)
		return
	case *planpb.Expr_UnaryExpr:
		walkExplainLeaves(expr.UnaryExpr.GetChild(), fieldNames, next, leaves)
		return
	case *planpb.Expr_RandomSampleExpr:
		walkExplainLeaves(expr.RandomSampleExpr.GetPredicate(), fieldNames, next, leaves)
		return
	case *planpb.Expr_MatchExpr:
		walkExplainLeaves(expr.MatchExpr.GetPredicate(), fieldNames, next, leaves)
		return
	case *planpb.Expr_ElementFilterExpr:
		walkExplainLeaves(expr.ElementFilterExpr.GetElementExpr(), fieldNames, next, leaves)
		walkExplainLeaves(expr.ElementFilterExpr.GetPredicate(), fieldNames, next, leaves)
		return
	}

	*next++
	node.ExprId = *next
	condition, typ, fieldID := describeExplainLeaf(node, fieldNames)
	leaves[*next] = explain.LeafInfo{Condition: condition, Type: typ, FieldID: fieldID}
}

// describeExplainLeaf renders one predicate in a compact, index-independent
// notation. Values that came from expression templates are elided: the explain
// payload must not echo caller-supplied out-of-band values.
func describeExplainLeaf(node *planpb.Expr, fieldNames map[int64]string) (string, string, int64) {
	switch expr := node.GetExpr().(type) {
	case *planpb.Expr_UnaryRangeExpr:
		e := expr.UnaryRangeExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		value := explainValue(e.GetValue(), e.GetTemplateVariableName())
		switch e.GetOp() {
		case planpb.OpType_GreaterThan:
			return fmt.Sprintf("%s in (%s, +inf)", field, value), "range", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_GreaterEqual:
			return fmt.Sprintf("%s in [%s, +inf)", field, value), "range", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_LessThan:
			return fmt.Sprintf("%s in (-inf, %s)", field, value), "range", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_LessEqual:
			return fmt.Sprintf("%s in (-inf, %s]", field, value), "range", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_Equal:
			return fmt.Sprintf("%s == %s", field, value), "equal", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_NotEqual:
			return fmt.Sprintf("%s != %s", field, value), "not_equal", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_PrefixMatch:
			return fmt.Sprintf("%s starts_with %s", field, value), "prefix", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_PostfixMatch:
			return fmt.Sprintf("%s ends_with %s", field, value), "postfix", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_InnerMatch:
			return fmt.Sprintf("%s contains %s", field, value), "contains", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_Match:
			return fmt.Sprintf("%s match %s", field, value), "match", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_RegexMatch:
			return fmt.Sprintf("%s regex %s", field, value), "regex", e.GetColumnInfo().GetFieldId()
		}
		op := strings.ToLower(e.GetOp().String())
		return fmt.Sprintf("%s %s %s", field, op, value), "unary_range", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_BinaryRangeExpr:
		e := expr.BinaryRangeExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		lower, upper := "[", "]"
		if !e.GetLowerInclusive() {
			lower = "("
		}
		if !e.GetUpperInclusive() {
			upper = ")"
		}
		return fmt.Sprintf("%s in %s%s, %s%s",
				field,
				lower,
				explainValue(e.GetLowerValue(), e.GetLowerTemplateVariableName()),
				explainValue(e.GetUpperValue(), e.GetUpperTemplateVariableName()),
				upper),
			"range", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_TermExpr:
		e := expr.TermExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		return fmt.Sprintf("%s in %s", field, explainValues(e.GetValues(), e.GetTemplateVariableName())),
			"in", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_BinaryArithOpEvalRangeExpr:
		e := expr.BinaryArithOpEvalRangeExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		return fmt.Sprintf("%s %s %s %s %s",
				field,
				strings.ToLower(e.GetArithOp().String()),
				explainValue(e.GetRightOperand(), e.GetOperandTemplateVariableName()),
				strings.ToLower(e.GetOp().String()),
				explainValue(e.GetValue(), e.GetValueTemplateVariableName())),
			"arith_range", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_CompareExpr:
		e := expr.CompareExpr
		return fmt.Sprintf("%s %s %s",
				explainField(e.GetLeftColumnInfo(), fieldNames),
				strings.ToLower(e.GetOp().String()),
				explainField(e.GetRightColumnInfo(), fieldNames)),
			"compare", e.GetLeftColumnInfo().GetFieldId()
	case *planpb.Expr_JsonContainsExpr:
		e := expr.JsonContainsExpr
		return fmt.Sprintf("%s json_contains %s",
				explainField(e.GetColumnInfo(), fieldNames),
				explainValues(e.GetElements(), e.GetTemplateVariableName())),
			"json_contains", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_ExistsExpr:
		e := expr.ExistsExpr
		return fmt.Sprintf("exists(%s)", explainField(e.GetInfo(), fieldNames)),
			"exists", e.GetInfo().GetFieldId()
	case *planpb.Expr_NullExpr:
		e := expr.NullExpr
		return fmt.Sprintf("%s is null", explainField(e.GetColumnInfo(), fieldNames)),
			"null", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_CallExpr:
		e := expr.CallExpr
		return fmt.Sprintf("%s(...)", e.GetFunctionName()), "call", 0
	case *planpb.Expr_GisfunctionFilterExpr:
		e := expr.GisfunctionFilterExpr
		return fmt.Sprintf("%s %s %q",
				explainField(e.GetColumnInfo(), fieldNames),
				strings.ToLower(e.GetOp().String()),
				e.GetWktString()),
			"gis", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_TimestamptzArithCompareExpr:
		e := expr.TimestamptzArithCompareExpr
		return fmt.Sprintf("%s %s %s",
				explainField(e.GetTimestamptzColumn(), fieldNames),
				strings.ToLower(e.GetCompareOp().String()),
				explainValue(e.GetCompareValue(), "")),
			"timestamptz", e.GetTimestamptzColumn().GetFieldId()
	case *planpb.Expr_BloomFilterExpr:
		e := expr.BloomFilterExpr
		return fmt.Sprintf("%s bloom_filter(%d bytes)",
				explainField(e.GetColumnInfo(), fieldNames), len(e.GetFilterBlob())),
			"bloom", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_RoaringFilterExpr:
		e := expr.RoaringFilterExpr
		return fmt.Sprintf("%s roaring_filter", explainField(e.GetColumnInfo(), fieldNames)),
			"roaring", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_MatchExpr:
		e := expr.MatchExpr
		return fmt.Sprintf("match(%s, %d)", e.GetStructName(), e.GetCount()),
			"match", 0
	case *planpb.Expr_ElementFilterExpr:
		return "element_filter", "element_filter", 0
	case *planpb.Expr_AlwaysTrueExpr:
		return "true", "always_true", 0
	}
	return "expr", "unknown", 0
}

// explainField renders a column reference: field name plus any JSON path.
func explainField(column *planpb.ColumnInfo, fieldNames map[int64]string) string {
	if column == nil {
		return "field_0"
	}
	name, ok := fieldNames[column.GetFieldId()]
	if !ok || name == "" {
		name = fmt.Sprintf("field_%d", column.GetFieldId())
	}
	var b strings.Builder
	b.WriteString(name)
	for _, key := range column.GetNestedPath() {
		b.WriteString(`["`)
		b.WriteString(key)
		b.WriteString(`"]`)
	}
	return b.String()
}

// explainValue renders one literal; a template-substituted value is elided
// because the client already knows it and logs must not carry it.
func explainValue(value *planpb.GenericValue, template string) string {
	if template != "" {
		return fmt.Sprintf("<template %q>", template)
	}
	if value == nil {
		return "null"
	}
	switch typed := value.GetVal().(type) {
	case *planpb.GenericValue_BoolVal:
		return strconv.FormatBool(typed.BoolVal)
	case *planpb.GenericValue_Int64Val:
		return strconv.FormatInt(typed.Int64Val, 10)
	case *planpb.GenericValue_FloatVal:
		return strconv.FormatFloat(float64(typed.FloatVal), 'g', -1, 32)
	case *planpb.GenericValue_StringVal:
		return strconv.Quote(typed.StringVal)
	case *planpb.GenericValue_ArrayVal:
		return explainValues(typed.ArrayVal.GetArray(), "")
	}
	return "<value>"
}

// explainValues renders a membership list, keeping the payload bounded.
func explainValues(values []*planpb.GenericValue, template string) string {
	if template != "" {
		return fmt.Sprintf("<template %q>", template)
	}
	const shown = 8
	var b strings.Builder
	b.WriteString("[")
	for i, value := range values {
		if i == shown {
			fmt.Fprintf(&b, ", ... <%d values>", len(values))
			break
		}
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(explainValue(value, ""))
	}
	b.WriteString("]")
	return b.String()
}
