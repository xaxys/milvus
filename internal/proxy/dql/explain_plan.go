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

// Explain leaf table: the proxy tags each user predicate leaf of the plan it
// sends with a request-unique expr_id and remembers how to describe that leaf.
// Segcore reports the operator pipeline and expr_id filter decisions, which
// the proxy folds across segments and resolves to user-visible field and index
// names before rendering.

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/tidwall/gjson"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/internal/util/explain"
	"github.com/milvus-io/milvus/internal/util/grpcclient"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	explainFormatKey = "explain_format"

	explainValueFalse   = "false"
	explainValueTrue    = "true"
	explainValueDefault = "default"
	explainLevelSummary = explain.LevelSummary
	explainLevelDetail  = explain.LevelDetail

	explainFormatText = "text"
	explainFormatJSON = "json"
)

// ExplainConfig holds the validated user-facing explain switch, verbosity
// level, and output format parsed once per request.
type ExplainConfig struct {
	Enabled bool
	Level   string
	Format  string
}

func lookupExplainSwitchParam(params []*commonpb.KeyValuePair) (string, bool, error) {
	if topVal, err := funcutil.GetAttrByKeyFromRepeatedKV(explainKey, params); err == nil {
		return topVal, true, nil
	}
	if rawParams, err := funcutil.GetAttrByKeyFromRepeatedKV(ParamsKey, params); err == nil && strings.TrimSpace(rawParams) != "" {
		res := gjson.Get(rawParams, explainKey)
		if res.Exists() && res.Type != gjson.Null {
			switch res.Type {
			case gjson.String:
				return res.String(), true, nil
			case gjson.True:
				return explainValueDefault, true, nil
			case gjson.False:
				return explainValueFalse, true, nil
			default:
				return "", false, merr.WrapErrParameterInvalidMsg("%s in %s must be a string or boolean", explainKey, ParamsKey)
			}
		}
	}
	return "", false, nil
}

// ParseExplainConfig parses and validates the user-facing explain switch
// ("explain") and output format ("explain_format"). Explain defaults to disabled.
func ParseExplainConfig(params []*commonpb.KeyValuePair) (ExplainConfig, error) {
	cfg := ExplainConfig{Format: explainFormatText}
	rawSwitch, hasSwitch, err := lookupExplainSwitchParam(params)
	if err != nil {
		return ExplainConfig{}, err
	}
	if hasSwitch {
		switch value := strings.ToLower(strings.TrimSpace(rawSwitch)); value {
		case "", explainValueFalse:
			cfg.Enabled = false
			cfg.Level = ""
		case explainValueTrue, explainValueDefault, explainLevelSummary:
			cfg.Enabled = true
			cfg.Level = explainLevelSummary
		case explainLevelDetail:
			cfg.Enabled = true
			cfg.Level = explainLevelDetail
		default:
			return ExplainConfig{}, merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected %q, %q, %q or %q",
				explainKey, rawSwitch, explainValueFalse, explainValueDefault, explainLevelSummary, explainLevelDetail)
		}
	}
	rawFormat, hasFormat, err := lookupStringParamWithNestedFallback(explainFormatKey, params)
	if err != nil {
		return ExplainConfig{}, err
	}
	if hasFormat {
		switch value := strings.ToLower(strings.TrimSpace(rawFormat)); value {
		case "":
		case explainFormatText, explainFormatJSON:
			cfg.Format = value
		default:
			return ExplainConfig{}, merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected %q or %q",
				explainFormatKey, rawFormat, explainFormatText, explainFormatJSON)
		}
	}
	return cfg, nil
}

// IsExplainFormatJSON reports whether params explicitly request JSON explain output.
func IsExplainFormatJSON(params []*commonpb.KeyValuePair) bool {
	cfg, err := ParseExplainConfig(params)
	return err == nil && cfg.Format == explainFormatJSON
}

func schemaFieldTable(schema *schemapb.CollectionSchema) (map[int64]string, map[int64]bool) {
	if schema == nil {
		return nil, nil
	}
	allFields := typeutil.GetAllFieldSchemas(schema)
	fieldNames := make(map[int64]string, len(allFields))
	dynamicFields := make(map[int64]bool)
	for _, field := range allFields {
		if field.GetIsDynamic() {
			dynamicFields[field.GetFieldID()] = true
			continue
		}
		fieldNames[field.GetFieldID()] = field.GetName()
	}
	return fieldNames, dynamicFields
}

// ValidateExplainParams validates user-facing explain parameters up front.
func ValidateExplainParams(params []*commonpb.KeyValuePair) error {
	_, err := ParseExplainConfig(params)
	return err
}

// AttachZeroSegmentExplain renders and attaches a 0-segment explain summary for
// early-return read paths that skip shard execution (e.g., SearchByPrimaryKeys
// when every resolved vector is null).
func AttachZeroSegmentExplain(
	status **commonpb.Status,
	params []*commonpb.KeyValuePair,
	dsl string,
	exprTemplates map[string]*schemapb.TemplateValue,
	schemaHelper *typeutil.SchemaHelper,
	schema *schemapb.CollectionSchema,
) error {
	cfg, err := ParseExplainConfig(params)
	if err != nil || !cfg.Enabled {
		return err
	}
	var leaves map[int64]explain.LeafInfo
	var asts map[int]explain.PlanAST
	if strings.TrimSpace(dsl) != "" && schemaHelper != nil {
		if plan, err := planparserv2.CreateRetrievePlan(schemaHelper, dsl, exprTemplates); err == nil {
			var next int64
			assignExplainIDsWithAST(plan, -1, schema, &next, &leaves, &asts)
		}
	}
	rendered, err := renderExplainWithConfig(cfg, nil, leaves, asts, schema, nil, nil)
	if err != nil {
		return err
	}
	attachExplain(status, rendered)
	return nil
}

// renderExplainWithAST turns the shard-reported trees into the client-visible
// payload. It returns "" when explain is disabled. When explain is enabled and
// 0 segments executed, it still emits a valid segments=0 summary.
func renderExplainWithAST(
	params []*commonpb.KeyValuePair,
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]explain.LeafInfo,
	asts map[int]explain.PlanAST,
	schema *schemapb.CollectionSchema,
	subSearches []explain.SubSearchMeta,
	indexNames func([]int64) map[int64]string,
) (string, error) {
	cfg, err := ParseExplainConfig(params)
	if err != nil {
		return "", err
	}
	return renderExplainWithConfig(cfg, explains, leaves, asts, schema, subSearches, indexNames)
}

func renderExplainWithConfig(
	cfg ExplainConfig,
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]explain.LeafInfo,
	asts map[int]explain.PlanAST,
	schema *schemapb.CollectionSchema,
	subSearches []explain.SubSearchMeta,
	indexNames func([]int64) map[int64]string,
) (string, error) {
	if !cfg.Enabled {
		return "", nil
	}

	level := cfg.Level
	if level == "" {
		level = explainLevelSummary
	}
	summary := explain.SummarizeWithSubSearchesAndLevel(level, explains, leaves, subSearches, asts)

	neededIDs := summary.NeededIndexIDs()
	var resolvedIndexes map[int64]string
	if len(neededIDs) > 0 && indexNames != nil {
		resolvedIndexes = indexNames(neededIDs)
	}
	fieldNames, dynamicFields := schemaFieldTable(schema)
	for fid := range dynamicFields {
		if _, exists := fieldNames[fid]; !exists {
			fieldNames[fid] = "dynamic"
		}
	}
	explain.FinalizeSummary(summary, resolvedIndexes, fieldNames)

	if cfg.Format == explainFormatJSON {
		return explain.RenderSummaryJSON(summary)
	}
	return explain.RenderSummaryText(summary), nil
}

const explainIndexLookupTimeout = 500 * time.Millisecond

func explainIndexNames(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
	neededIDs []int64,
) map[int64]string {
	if len(neededIDs) == 0 || node == nil || ctx.Err() != nil || (node.GetMetaCache() == nil && node.MixCoord() == nil) {
		return nil
	}
	lookupCtx, cancel := context.WithTimeout(ctx, explainIndexLookupTimeout)
	defer cancel()
	if node.GetMetaCache() == nil {
		lookupCtx = grpcclient.WithFailFastOnConnectionError(retry.WithMaxAttemptsContext(lookupCtx, 1))
	}

	colIndexes, err := resolveCollectionIndexes(lookupCtx, node, collectionID)
	if err != nil {
		mlog.Warn(ctx, "index names unavailable for execution explain",
			mlog.Int64("collectionID", collectionID), mlog.Err(err))
		return nil
	}
	if colIndexes == nil || len(colIndexes.ByID) == 0 {
		return nil
	}
	out := make(map[int64]string, len(neededIDs))
	for _, id := range neededIDs {
		if info := colIndexes.ByID[id]; info != nil {
			out[id] = info.GetIndexName()
		}
	}
	return out
}

// configureExplainPlanWithAST validates explain parameters, enables segcore
// explain collection only when requested on non-internal passes, and tags user
// predicate leaves so segcore counters map back to query predicates.
func configureExplainPlanWithAST(
	plan *planpb.PlanNode,
	params []*commonpb.KeyValuePair,
	disableExplain bool,
	hasUserExpr bool,
	subReqIndex int,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
	asts *map[int]explain.PlanAST,
) error {
	if plan == nil {
		return nil
	}
	var cfg ExplainConfig
	if !disableExplain {
		var err error
		cfg, err = ParseExplainConfig(params)
		if err != nil {
			return err
		}
	}
	configureExplainPlanWithConfig(plan, cfg, disableExplain, hasUserExpr, subReqIndex, schema, next, leaves, asts)
	return nil
}

func configureExplainPlanWithConfig(
	plan *planpb.PlanNode,
	cfg ExplainConfig,
	disableExplain bool,
	hasUserExpr bool,
	subReqIndex int,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
	asts *map[int]explain.PlanAST,
) {
	if plan == nil {
		return
	}
	if disableExplain || !cfg.Enabled {
		if plan.PlanOptions != nil {
			plan.PlanOptions.EnableExplain = false
			plan.PlanOptions.ExplainDetail = false
		}
		return
	}
	if plan.PlanOptions == nil {
		plan.PlanOptions = &planpb.PlanOption{}
	}
	plan.PlanOptions.EnableExplain = true
	plan.PlanOptions.ExplainDetail = cfg.Level == explainLevelDetail
	if hasUserExpr {
		assignExplainIDsWithAST(plan, subReqIndex, schema, next, leaves, asts)
	}
}

func assignExplainIDsWithAST(
	plan *planpb.PlanNode,
	subReqIndex int,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
	asts *map[int]explain.PlanAST,
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
	fieldNames, dynamicFields := schemaFieldTable(schema)
	if *leaves == nil {
		*leaves = make(map[int64]explain.LeafInfo, 4)
	}
	var elemTerms []*explain.ExprAST
	docAST := walkExplainLeaves(root, fieldNames, dynamicFields, subReqIndex, false, next, *leaves, &elemTerms)
	if asts != nil {
		if *asts == nil {
			*asts = make(map[int]explain.PlanAST, 2)
		}
		var elemAST *explain.ExprAST
		if len(elemTerms) == 1 {
			elemAST = elemTerms[0]
		} else if len(elemTerms) > 1 {
			elemAST = &explain.ExprAST{Logic: "and", Children: elemTerms}
		}
		(*asts)[subReqIndex] = explain.PlanAST{
			DocAST:     docAST,
			ElementAST: elemAST,
		}
	}
}

func matchTypeLabel(mt planpb.MatchType) (string, bool) {
	switch mt {
	case planpb.MatchType_MatchAny:
		return "match_any", false
	case planpb.MatchType_MatchAll:
		return "match_all", false
	case planpb.MatchType_MatchLeast:
		return "match_least", true
	case planpb.MatchType_MatchMost:
		return "match_most", true
	case planpb.MatchType_MatchExact:
		return "match_exact", true
	default:
		return "match", false
	}
}

func walkExplainLeaves(
	node *planpb.Expr,
	fieldNames map[int64]string,
	dynamicFields map[int64]bool,
	subReqIndex int,
	inElementFilter bool,
	next *int64,
	leaves map[int64]explain.LeafInfo,
	elemTerms *[]*explain.ExprAST,
) *explain.ExprAST {
	if node == nil {
		return nil
	}
	switch expr := node.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		left := walkExplainLeaves(expr.BinaryExpr.GetLeft(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves, elemTerms)
		right := walkExplainLeaves(expr.BinaryExpr.GetRight(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves, elemTerms)
		logic := "and"
		if expr.BinaryExpr.GetOp() == planpb.BinaryExpr_LogicalOr {
			logic = "or"
		}
		var children []*explain.ExprAST
		for _, side := range []*explain.ExprAST{left, right} {
			if side == nil {
				continue
			}
			if side.Logic == logic {
				children = append(children, side.Children...)
			} else {
				children = append(children, side)
			}
		}
		if len(children) == 0 {
			return nil
		}
		if len(children) == 1 {
			return children[0]
		}
		return &explain.ExprAST{Logic: logic, Children: children}
	case *planpb.Expr_UnaryExpr:
		child := walkExplainLeaves(expr.UnaryExpr.GetChild(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves, elemTerms)
		if child == nil {
			return nil
		}
		if expr.UnaryExpr.GetOp() == planpb.UnaryExpr_Not {
			return &explain.ExprAST{Logic: "not", Children: []*explain.ExprAST{child}}
		}
		return child
	case *planpb.Expr_RandomSampleExpr:
		return walkExplainLeaves(expr.RandomSampleExpr.GetPredicate(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves, elemTerms)
	case *planpb.Expr_MatchExpr:
		child := walkExplainLeaves(expr.MatchExpr.GetPredicate(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves, elemTerms)
		if child == nil {
			return nil
		}
		logic, withCount := matchTypeLabel(expr.MatchExpr.GetMatchType())
		var cnt int64
		if withCount {
			cnt = expr.MatchExpr.GetCount()
		}
		return &explain.ExprAST{
			Logic:    logic,
			Field:    expr.MatchExpr.GetStructName(),
			Count:    cnt,
			HasCount: withCount,
			Children: []*explain.ExprAST{child},
		}
	case *planpb.Expr_ElementFilterExpr:
		elemAST := walkExplainLeaves(expr.ElementFilterExpr.GetElementExpr(), fieldNames, dynamicFields, subReqIndex, true, next, leaves, elemTerms)
		if elemAST != nil && elemTerms != nil {
			*elemTerms = append(*elemTerms, elemAST)
		}
		return walkExplainLeaves(expr.ElementFilterExpr.GetPredicate(), fieldNames, dynamicFields, subReqIndex, false, next, leaves, elemTerms)
	}

	*next++
	node.ExprId = *next
	condition, typ, fieldID, userField := describeExplainLeafWithDynamic(node, fieldNames, dynamicFields)
	info := explain.LeafInfo{
		Condition:     condition,
		Type:          typ,
		FieldID:       fieldID,
		Field:         userField,
		ElementFilter: inElementFilter,
	}
	if subReqIndex >= 0 {
		info.HasSubReq = true
		info.SubReqIndex = subReqIndex
	}
	leaves[*next] = info
	return &explain.ExprAST{ExprID: *next}
}

func describeExplainLeafWithDynamic(
	node *planpb.Expr,
	fieldNames map[int64]string,
	dynamicFields map[int64]bool,
) (string, string, int64, string) {
	switch expr := node.GetExpr().(type) {
	case *planpb.Expr_UnaryRangeExpr:
		e := expr.UnaryRangeExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		fid := e.GetColumnInfo().GetFieldId()
		value := explainValue(e.GetValue())
		switch e.GetOp() {
		case planpb.OpType_GreaterThan:
			return fmt.Sprintf("%s in (%s, +inf)", field, value), "range", fid, userField
		case planpb.OpType_GreaterEqual:
			return fmt.Sprintf("%s in [%s, +inf)", field, value), "range", fid, userField
		case planpb.OpType_LessThan:
			return fmt.Sprintf("%s in (-inf, %s)", field, value), "range", fid, userField
		case planpb.OpType_LessEqual:
			return fmt.Sprintf("%s in (-inf, %s]", field, value), "range", fid, userField
		case planpb.OpType_Equal:
			return fmt.Sprintf("%s == %s", field, value), "equal", fid, userField
		case planpb.OpType_NotEqual:
			return fmt.Sprintf("%s != %s", field, value), "not_equal", fid, userField
		case planpb.OpType_PrefixMatch:
			return fmt.Sprintf("%s starts_with %s", field, value), "prefix", fid, userField
		case planpb.OpType_PostfixMatch:
			return fmt.Sprintf("%s ends_with %s", field, value), "postfix", fid, userField
		case planpb.OpType_InnerMatch:
			return fmt.Sprintf("%s contains %s", field, value), "contains", fid, userField
		case planpb.OpType_Match:
			return fmt.Sprintf("%s match %s", field, value), "match", fid, userField
		case planpb.OpType_RegexMatch:
			return fmt.Sprintf("%s regex %s", field, value), "regex", fid, userField
		case planpb.OpType_TextMatch:
			if extra := e.GetExtraValues(); len(extra) > 0 {
				return fmt.Sprintf("text_match(%s, %s, minimum_should_match=%s)", field, value, explainValue(extra[0])),
					"text_match", fid, userField
			}
			return fmt.Sprintf("text_match(%s, %s)", field, value), "text_match", fid, userField
		case planpb.OpType_PhraseMatch:
			if extra := e.GetExtraValues(); len(extra) > 0 {
				return fmt.Sprintf("phrase_match(%s, %s, slop=%s)", field, value, explainValue(extra[0])),
					"phrase_match", fid, userField
			}
			return fmt.Sprintf("phrase_match(%s, %s)", field, value), "phrase_match", fid, userField
		case planpb.OpType_TextMatchFuzzy:
			if extra := e.GetExtraValues(); len(extra) > 0 {
				return fmt.Sprintf("text_match_fuzzy(%s, %s, max_edit_distance=%s)", field, value, explainValue(extra[0])),
					"text_match_fuzzy", fid, userField
			}
			return fmt.Sprintf("text_match_fuzzy(%s, %s)", field, value), "text_match_fuzzy", fid, userField
		}
		op := strings.ToLower(e.GetOp().String())
		return fmt.Sprintf("%s %s %s", field, op, value), "unary_range", fid, userField
	case *planpb.Expr_BinaryRangeExpr:
		e := expr.BinaryRangeExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
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
				explainValue(e.GetLowerValue()),
				explainValue(e.GetUpperValue()),
				upper),
			"range", e.GetColumnInfo().GetFieldId(), userField
	case *planpb.Expr_TermExpr:
		e := expr.TermExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		fid := e.GetColumnInfo().GetFieldId()
		if e.GetIsInField() {
			if vals := e.GetValues(); len(vals) == 1 {
				return fmt.Sprintf("%s in %s", explainValue(vals[0]), field), "in", fid, userField
			}
			return fmt.Sprintf("%s in %s", explainValues(e.GetValues()), field), "in", fid, userField
		}
		return fmt.Sprintf("%s in %s", field, explainValues(e.GetValues())), "in", fid, userField
	case *planpb.Expr_BinaryArithOpEvalRangeExpr:
		e := expr.BinaryArithOpEvalRangeExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		fid := e.GetColumnInfo().GetFieldId()
		if e.GetArithOp() == planpb.ArithOpType_ArrayLength {
			return fmt.Sprintf("array_length(%s) %s %s",
					field,
					compareOpSymbol(e.GetOp()),
					explainValue(e.GetValue())),
				"array_length", fid, userField
		}
		return fmt.Sprintf("%s %s %s %s %s",
				field,
				strings.ToLower(e.GetArithOp().String()),
				explainValue(e.GetRightOperand()),
				strings.ToLower(e.GetOp().String()),
				explainValue(e.GetValue())),
			"arith_range", fid, userField
	case *planpb.Expr_BinaryArithExpr:
		e := expr.BinaryArithExpr
		leftCond, _, leftFid, leftUser := describeExplainLeafWithDynamic(e.GetLeft(), fieldNames, dynamicFields)
		if e.GetOp() == planpb.ArithOpType_ArrayLength {
			return fmt.Sprintf("array_length(%s)", leftCond), "array_length", leftFid, leftUser
		}
		rightCond, _, _, _ := describeExplainLeafWithDynamic(e.GetRight(), fieldNames, dynamicFields)
		return fmt.Sprintf("%s %s %s", leftCond, strings.ToLower(e.GetOp().String()), rightCond), "arith", leftFid, leftUser
	case *planpb.Expr_ColumnExpr:
		col := expr.ColumnExpr.GetInfo()
		field, userField := explainFieldAndUserName(col, fieldNames, dynamicFields)
		return field, "column", col.GetFieldId(), userField
	case *planpb.Expr_CompareExpr:
		e := expr.CompareExpr
		leftFid := e.GetLeftColumnInfo().GetFieldId()
		rightFid := e.GetRightColumnInfo().GetFieldId()
		leftRef, leftUser := explainFieldAndUserName(e.GetLeftColumnInfo(), fieldNames, dynamicFields)
		rightRef, rightUser := explainFieldAndUserName(e.GetRightColumnInfo(), fieldNames, dynamicFields)
		ownerFid, ownerUser := int64(0), ""
		if leftFid == rightFid && leftUser == rightUser {
			ownerFid, ownerUser = leftFid, leftUser
		}
		return fmt.Sprintf("%s %s %s",
				leftRef,
				strings.ToLower(e.GetOp().String()),
				rightRef),
			"compare", ownerFid, ownerUser
	case *planpb.Expr_JsonContainsExpr:
		e := expr.JsonContainsExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		op, label := "json_contains", "json_contains"
		switch e.GetOp() {
		case planpb.JSONContainsExpr_ContainsAll:
			op, label = "json_contains_all", "json_contains_all"
		case planpb.JSONContainsExpr_ContainsAny:
			op, label = "json_contains_any", "json_contains_any"
		}
		return fmt.Sprintf("%s %s %s", field, op, explainValues(e.GetElements())),
			label, e.GetColumnInfo().GetFieldId(), userField
	case *planpb.Expr_ExistsExpr:
		e := expr.ExistsExpr
		field, userField := explainFieldAndUserName(e.GetInfo(), fieldNames, dynamicFields)
		return fmt.Sprintf("exists(%s)", field), "exists", e.GetInfo().GetFieldId(), userField
	case *planpb.Expr_NullExpr:
		e := expr.NullExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		op := "is null"
		if e.GetOp() == planpb.NullExpr_IsNotNull {
			op = "is not null"
		}
		return fmt.Sprintf("%s %s", field, op), "null", e.GetColumnInfo().GetFieldId(), userField
	case *planpb.Expr_CallExpr:
		e := expr.CallExpr
		col := firstCallExprColumn(e)
		var fieldID int64
		var userField string
		if col != nil {
			fieldID = col.GetFieldId()
			_, userField = explainFieldAndUserName(col, fieldNames, dynamicFields)
		}
		return fmt.Sprintf("%s(...)", e.GetFunctionName()), "call", fieldID, userField
	case *planpb.Expr_GisfunctionFilterExpr:
		e := expr.GisfunctionFilterExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		fid := e.GetColumnInfo().GetFieldId()
		wkt := strconv.Quote(e.GetWktString())
		if e.GetOp() == planpb.GISFunctionFilterExpr_DWithin {
			return fmt.Sprintf("%s %s %s distance=%s",
					field,
					strings.ToLower(e.GetOp().String()),
					wkt,
					strconv.FormatFloat(e.GetDistance(), 'g', -1, 64)),
				"gis", fid, userField
		}
		return fmt.Sprintf("%s %s %s",
				field,
				strings.ToLower(e.GetOp().String()),
				wkt),
			"gis", fid, userField
	case *planpb.Expr_TimestamptzArithCompareExpr:
		e := expr.TimestamptzArithCompareExpr
		field, userField := explainFieldAndUserName(e.GetTimestamptzColumn(), fieldNames, dynamicFields)
		fid := e.GetTimestamptzColumn().GetFieldId()
		if e.GetInterval() != nil && e.GetArithOp() != planpb.ArithOpType_Unknown {
			return fmt.Sprintf("%s %s interval(%s) %s %s",
					field,
					strings.ToLower(e.GetArithOp().String()),
					planparserv2.FormatIntervalISO(e.GetInterval()),
					strings.ToLower(e.GetCompareOp().String()),
					explainValue(e.GetCompareValue())),
				"timestamptz", fid, userField
		}
		return fmt.Sprintf("%s %s %s",
				field,
				strings.ToLower(e.GetCompareOp().String()),
				explainValue(e.GetCompareValue())),
			"timestamptz", fid, userField
	case *planpb.Expr_BloomFilterExpr:
		e := expr.BloomFilterExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		return fmt.Sprintf("%s membership_filter", field), "membership", e.GetColumnInfo().GetFieldId(), userField
	case *planpb.Expr_RoaringFilterExpr:
		e := expr.RoaringFilterExpr
		field, userField := explainFieldAndUserName(e.GetColumnInfo(), fieldNames, dynamicFields)
		return fmt.Sprintf("%s membership_filter", field), "membership", e.GetColumnInfo().GetFieldId(), userField
	case *planpb.Expr_AlwaysTrueExpr:
		return "true", "always_true", 0, ""
	}
	return "expr", "unknown", 0, ""
}

func firstCallExprColumn(call *planpb.CallExpr) *planpb.ColumnInfo {
	if call == nil {
		return nil
	}
	for _, param := range call.GetFunctionParameters() {
		if col := param.GetColumnExpr(); col != nil {
			return col.GetInfo()
		}
		if inner := param.GetCallExpr(); inner != nil {
			if found := firstCallExprColumn(inner); found != nil {
				return found
			}
		}
	}
	return nil
}

func compareOpSymbol(op planpb.OpType) string {
	switch op {
	case planpb.OpType_Equal:
		return "=="
	case planpb.OpType_NotEqual:
		return "!="
	case planpb.OpType_GreaterThan:
		return ">"
	case planpb.OpType_GreaterEqual:
		return ">="
	case planpb.OpType_LessThan:
		return "<"
	case planpb.OpType_LessEqual:
		return "<="
	default:
		return strings.ToLower(op.String())
	}
}

func isSimpleIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r == '_' || unicode.IsLetter(r) || (i > 0 && unicode.IsDigit(r)) {
			continue
		}
		return false
	}
	return true
}

func explainFieldAndUserName(
	column *planpb.ColumnInfo,
	fieldNames map[int64]string,
	dynamicFields map[int64]bool,
) (string, string) {
	if column == nil {
		return "unknown", ""
	}
	fid := column.GetFieldId()
	nested := column.GetNestedPath()
	name, hasName := fieldNames[fid]
	if dynamicFields[fid] || name == "$meta" {
		if len(nested) == 0 {
			return "dynamic", ""
		}
		userField := nested[0]
		var b strings.Builder
		if isSimpleIdentifier(userField) {
			b.WriteString(userField)
		} else {
			b.WriteString(strconv.Quote(userField))
		}
		for _, key := range nested[1:] {
			b.WriteByte('[')
			b.WriteString(strconv.Quote(key))
			b.WriteByte(']')
		}
		return b.String(), userField
	}

	var userField string
	if hasName && name != "" {
		userField = name
	} else {
		name = "unknown"
	}
	var b strings.Builder
	b.WriteString(name)
	for _, key := range nested {
		b.WriteByte('[')
		b.WriteString(strconv.Quote(key))
		b.WriteByte(']')
	}
	return b.String(), userField
}

func explainValue(value *planpb.GenericValue) string {
	if value == nil {
		return "null"
	}
	switch typed := value.GetVal().(type) {
	case *planpb.GenericValue_BoolVal:
		return strconv.FormatBool(typed.BoolVal)
	case *planpb.GenericValue_Int64Val:
		return strconv.FormatInt(typed.Int64Val, 10)
	case *planpb.GenericValue_FloatVal:
		return strconv.FormatFloat(typed.FloatVal, 'g', -1, 64)
	case *planpb.GenericValue_StringVal:
		return strconv.Quote(typed.StringVal)
	case *planpb.GenericValue_ArrayVal:
		return explainValues(typed.ArrayVal.GetArray())
	}
	return "<value>"
}

func explainValues(values []*planpb.GenericValue) string {
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
		b.WriteString(explainValue(value))
	}
	b.WriteString("]")
	return b.String()
}
