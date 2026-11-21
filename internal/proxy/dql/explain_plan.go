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
	"sync"
	"time"
	"unicode"

	lru "github.com/hashicorp/golang-lru/v2"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/explain"
	"github.com/milvus-io/milvus/internal/util/grpcclient"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const (
	explainFormatKey = "explain_format"

	explainValueFalse   = "false"
	explainValueDefault = "default"
	explainLevelSummary = "summary"
	explainLevelDetail  = "detail"

	explainFormatText = "text"
	explainFormatJSON = "json"
)

// explainParams parses the user-facing explain switch/level ("explain") and
// format ("explain_format"). Explain defaults to disabled ("false").
func explainParams(params []*commonpb.KeyValuePair) (enabled bool, format string, level string, err error) {
	format = explainFormatText
	if raw, err := funcutil.GetAttrByKeyFromRepeatedKV(explainKey, params); err == nil {
		switch value := strings.ToLower(strings.TrimSpace(raw)); value {
		case "", explainValueFalse:
			enabled = false
		case explainValueDefault, explainLevelSummary:
			enabled = true
			level = explainLevelSummary
		case explainLevelDetail:
			enabled = true
			level = explainLevelDetail
		default:
			return false, "", "", merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected %q, %q, %q or %q",
				explainKey, raw, explainValueFalse, explainValueDefault, explainLevelSummary, explainLevelDetail)
		}
	}
	if raw, err := funcutil.GetAttrByKeyFromRepeatedKV(explainFormatKey, params); err == nil {
		switch value := strings.ToLower(strings.TrimSpace(raw)); value {
		case "":
		case explainFormatText, explainFormatJSON:
			format = value
		default:
			return false, "", "", merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected %q or %q",
				explainFormatKey, raw, explainFormatText, explainFormatJSON)
		}
	}
	return enabled, format, level, nil
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

// renderExplain turns the shard-reported trees into the client-visible payload.
// It returns "" when explain is disabled or when no shard recorded anything
// (the caller then leaves extra_info alone).
func renderExplain(
	params []*commonpb.KeyValuePair,
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]explain.LeafInfo,
	schema *schemapb.CollectionSchema,
	subSearches []explain.SubSearchMeta,
	indexNames func([]int64) map[int64]string,
) (string, error) {
	enabled, format, level, err := explainParams(params)
	if err != nil || !enabled {
		return "", err
	}
	if level != explainLevelSummary {
		return "", merr.WrapErrParameterInvalidMsg(
			"%s %q is not supported yet", explainKey, level)
	}

	tagged := explain.TaggedSegments(explains...)
	if len(tagged) == 0 {
		return "", nil
	}
	summary := explain.SummarizeWithSubSearches(tagged, leaves, subSearches)
	if len(summary.Operators) == 0 && len(summary.SubSearches) == 0 {
		return "", nil
	}

	var neededIDs []int64
	for _, condition := range summary.AllConditions() {
		for _, used := range condition.Used {
			if used.IndexID != 0 {
				neededIDs = append(neededIDs, used.IndexID)
			}
		}
	}
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

	if format == explainFormatJSON {
		return explain.RenderSummaryJSON(summary)
	}
	return explain.RenderSummaryText(summary), nil
}

const (
	explainIndexCacheCapacity   = 4096
	explainIndexLookupTimeout   = 500 * time.Millisecond
	explainIndexFailureCooldown = 5 * time.Second
)

type explainIndexCacheEntry struct {
	names       map[int64]string
	unresolved  map[int64]time.Time
	failedUntil time.Time
}

// explainIndexCache memoizes immutable IndexID -> IndexName mappings per
// collection so explain resolves names from memory on the read path.
type explainIndexCache struct {
	mu      sync.Mutex
	entries *lru.Cache[int64, *explainIndexCacheEntry]
	sf      conc.Singleflight[map[int64]string]
	now     func() time.Time
	timeout time.Duration
}

func newExplainIndexCache(capacity int, timeout time.Duration, now func() time.Time) *explainIndexCache {
	entries, _ := lru.New[int64, *explainIndexCacheEntry](capacity)
	return &explainIndexCache{
		entries: entries,
		now:     now,
		timeout: timeout,
	}
}

var defaultExplainIndexCache = newExplainIndexCache(
	explainIndexCacheCapacity,
	explainIndexLookupTimeout,
	time.Now,
)

func (c *explainIndexCache) purge() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries.Purge()
}

func (c *explainIndexCache) satisfied(collectionID int64, neededIDs []int64, now time.Time) (map[int64]string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries.Get(collectionID)
	if !ok || entry == nil {
		return nil, false
	}
	if !entry.failedUntil.IsZero() && now.Before(entry.failedUntil) {
		return entry.names, true
	}
	if len(neededIDs) == 0 && entry.names != nil {
		return entry.names, true
	}
	for _, id := range neededIDs {
		if id == 0 {
			continue
		}
		if _, found := entry.names[id]; found {
			continue
		}
		if until, suppressed := entry.unresolved[id]; suppressed && now.Before(until) {
			continue
		}
		return entry.names, false
	}
	return entry.names, true
}

func (c *explainIndexCache) resolve(
	ctx context.Context,
	coord types.MixCoordClient,
	collectionID int64,
	neededIDs []int64,
) map[int64]string {
	if coord == nil {
		return nil
	}
	cached, ok := c.satisfied(collectionID, neededIDs, c.now())
	if ok || ctx.Err() != nil {
		return cached
	}

	key := strconv.FormatInt(collectionID, 10)
	ch := c.sf.DoChan(key, func() (map[int64]string, error) {
		if hit, hitOK := c.satisfied(collectionID, neededIDs, c.now()); hitOK {
			return hit, nil
		}
		lookupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), c.timeout)
		defer cancel()
		lookupCtx = grpcclient.WithFailFastOnConnectionError(retry.WithMaxAttemptsContext(lookupCtx, 1))

		resp, err := coord.ListIndexes(lookupCtx, &indexpb.ListIndexesRequest{
			CollectionID: collectionID,
		})
		if err == nil {
			err = merr.Error(resp.GetStatus())
		}
		now := c.now()

		c.mu.Lock()
		defer c.mu.Unlock()
		prev, _ := c.entries.Get(collectionID)
		if err != nil {
			mlog.Warn(ctx, "index names unavailable for execution explain",
				mlog.Int64("collectionID", collectionID), mlog.Err(err))
			var prevNames map[int64]string
			var prevUnresolved map[int64]time.Time
			if prev != nil {
				prevNames = prev.names
				prevUnresolved = prev.unresolved
			}
			c.entries.Add(collectionID, &explainIndexCacheEntry{
				names:       prevNames,
				unresolved:  prevUnresolved,
				failedUntil: now.Add(explainIndexFailureCooldown),
			})
			return prevNames, nil
		}

		mergedCap := len(resp.GetIndexInfos())
		if prev != nil {
			mergedCap += len(prev.names)
		}
		merged := make(map[int64]string, mergedCap)
		if prev != nil {
			for id, name := range prev.names {
				merged[id] = name
			}
		}
		for _, info := range resp.GetIndexInfos() {
			merged[info.GetIndexID()] = info.GetIndexName()
		}

		var unresolved map[int64]time.Time
		if prev != nil {
			for id, until := range prev.unresolved {
				if _, resolved := merged[id]; !resolved && now.Before(until) {
					if unresolved == nil {
						unresolved = make(map[int64]time.Time)
					}
					unresolved[id] = until
				}
			}
		}
		for _, id := range neededIDs {
			if id != 0 {
				if _, resolved := merged[id]; !resolved {
					if unresolved == nil {
						unresolved = make(map[int64]time.Time)
					}
					unresolved[id] = now.Add(explainIndexFailureCooldown)
				}
			}
		}
		c.entries.Add(collectionID, &explainIndexCacheEntry{
			names:      merged,
			unresolved: unresolved,
		})
		return merged, nil
	})

	select {
	case res := <-ch:
		return res.Val
	case <-ctx.Done():
		return cached
	}
}

func explainIndexNames(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
	neededIDs []int64,
) map[int64]string {
	if node == nil || node.MixCoord() == nil {
		return nil
	}
	return defaultExplainIndexCache.resolve(ctx, node.MixCoord(), collectionID, neededIDs)
}

// configureExplainPlan validates explain parameters, enables segcore explain
// collection only when requested on non-internal passes, and tags user
// predicate leaves so segcore counters map back to query predicates.
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
	if plan == nil {
		return nil
	}
	enabled := false
	if !disableExplain {
		var level string
		var err error
		enabled, _, level, err = explainParams(params)
		if err != nil {
			return err
		}
		if enabled && level != explainLevelSummary {
			return merr.WrapErrParameterInvalidMsg(
				"%s %q is not supported yet", explainKey, level)
		}
	}
	if disableExplain || !enabled {
		if plan.PlanOptions != nil {
			plan.PlanOptions.EnableExplain = false
		}
		return nil
	}
	if plan.PlanOptions == nil {
		plan.PlanOptions = &planpb.PlanOption{}
	}
	plan.PlanOptions.EnableExplain = true
	if hasUserExpr {
		assignExplainIDsWithSubReq(plan, subReqIndex, schema, next, leaves)
	}
	return nil
}

// assignExplainIDs tags every predicate leaf of a single-path plan.
func assignExplainIDs(
	plan *planpb.PlanNode,
	schema *schemapb.CollectionSchema,
	next *int64,
	leaves *map[int64]explain.LeafInfo,
) {
	assignExplainIDsWithSubReq(plan, -1, schema, next, leaves)
}

func assignExplainIDsWithSubReq(
	plan *planpb.PlanNode,
	subReqIndex int,
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
	fieldNames, dynamicFields := schemaFieldTable(schema)
	if *leaves == nil {
		*leaves = make(map[int64]explain.LeafInfo, 4)
	}
	walkExplainLeaves(root, fieldNames, dynamicFields, subReqIndex, false, next, *leaves)
}

func walkExplainLeaves(
	node *planpb.Expr,
	fieldNames map[int64]string,
	dynamicFields map[int64]bool,
	subReqIndex int,
	inElementFilter bool,
	next *int64,
	leaves map[int64]explain.LeafInfo,
) {
	if node == nil {
		return
	}
	switch expr := node.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		walkExplainLeaves(expr.BinaryExpr.GetLeft(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves)
		walkExplainLeaves(expr.BinaryExpr.GetRight(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves)
		return
	case *planpb.Expr_UnaryExpr:
		walkExplainLeaves(expr.UnaryExpr.GetChild(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves)
		return
	case *planpb.Expr_RandomSampleExpr:
		walkExplainLeaves(expr.RandomSampleExpr.GetPredicate(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves)
		return
	case *planpb.Expr_MatchExpr:
		walkExplainLeaves(expr.MatchExpr.GetPredicate(), fieldNames, dynamicFields, subReqIndex, inElementFilter, next, leaves)
		return
	case *planpb.Expr_ElementFilterExpr:
		walkExplainLeaves(expr.ElementFilterExpr.GetElementExpr(), fieldNames, dynamicFields, subReqIndex, true, next, leaves)
		walkExplainLeaves(expr.ElementFilterExpr.GetPredicate(), fieldNames, dynamicFields, subReqIndex, false, next, leaves)
		return
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
}

// describeExplainLeaf renders one predicate in a compact, index-independent
// notation without dynamic-field schema context.
func describeExplainLeaf(node *planpb.Expr, fieldNames map[int64]string) (string, string, int64) {
	cond, typ, fid, _ := describeExplainLeafWithDynamic(node, fieldNames, nil)
	return cond, typ, fid
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
	case *planpb.Expr_CompareExpr:
		e := expr.CompareExpr
		leftRef, leftUser := explainFieldAndUserName(e.GetLeftColumnInfo(), fieldNames, dynamicFields)
		rightRef, _ := explainFieldAndUserName(e.GetRightColumnInfo(), fieldNames, dynamicFields)
		return fmt.Sprintf("%s %s %s",
				leftRef,
				strings.ToLower(e.GetOp().String()),
				rightRef),
			"compare", e.GetLeftColumnInfo().GetFieldId(), leftUser
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
					explainInterval(e.GetInterval()),
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
			b.WriteString(`["`)
			b.WriteString(key)
			b.WriteString(`"]`)
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
		b.WriteString(`["`)
		b.WriteString(key)
		b.WriteString(`"]`)
	}
	return b.String(), userField
}

func explainField(column *planpb.ColumnInfo, fieldNames map[int64]string) string {
	ref, _ := explainFieldAndUserName(column, fieldNames, nil)
	return ref
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

func explainInterval(iv *planpb.Interval) string {
	var b strings.Builder
	b.WriteByte('P')
	if y := iv.GetYears(); y != 0 {
		fmt.Fprintf(&b, "%dY", y)
	}
	if m := iv.GetMonths(); m != 0 {
		fmt.Fprintf(&b, "%dM", m)
	}
	if d := iv.GetDays(); d != 0 {
		fmt.Fprintf(&b, "%dD", d)
	}
	h, min, s := iv.GetHours(), iv.GetMinutes(), iv.GetSeconds()
	if h != 0 || min != 0 || s != 0 {
		b.WriteByte('T')
		if h != 0 {
			fmt.Fprintf(&b, "%dH", h)
		}
		if min != 0 {
			fmt.Fprintf(&b, "%dM", min)
		}
		if s != 0 {
			fmt.Fprintf(&b, "%dS", s)
		}
	}
	if b.Len() == 1 {
		b.WriteString("T0S")
	}
	return b.String()
}
