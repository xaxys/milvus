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
	"sync"
	"time"

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

// renderExplain turns the shard-reported trees into the client-visible payload.
// It returns "" when explain is disabled or when no shard recorded anything
// (the caller then leaves extra_info alone).
func renderExplain(
	params []*commonpb.KeyValuePair,
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]explain.LeafInfo,
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

	segments := explain.Segments(explains...)
	if len(segments) == 0 {
		return "", nil
	}
	summary := explain.Summarize(segments, leaves)
	explain.BackfillConditions(summary, leaves)
	if len(summary.Conditions) == 0 {
		return "", nil
	}
	// Only a condition that names an index needs the coordinator lookup; an
	// unfiltered search, raw-only filter, or pk lookup (index_id=0) pays nothing.
	var neededIDs []int64
	for _, condition := range summary.Conditions {
		for _, used := range condition.Used {
			if used.IndexID != 0 {
				neededIDs = append(neededIDs, used.IndexID)
			}
		}
	}
	if len(neededIDs) > 0 && indexNames != nil {
		explain.NameIndexUses(summary, indexNames(neededIDs))
	}
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

// explainIndexNames resolves the indexes this collection carries from the
// proxy-side cache, falling back to a bounded singleflight ListIndexes lookup
// only when a reported index_id is not cached yet.
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
// collection only when requested on non-internal passes, and tags predicate
// leaves so segcore counters map back to query predicates.
func configureExplainPlan(
	plan *planpb.PlanNode,
	params []*commonpb.KeyValuePair,
	disableExplain bool,
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
	assignExplainIDs(plan, schema, next, leaves)
	return nil
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
	var allFields []*schemapb.FieldSchema
	if schema != nil {
		allFields = typeutil.GetAllFieldSchemas(schema)
	}
	fieldNames := make(map[int64]string, len(allFields))
	for _, field := range allFields {
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
// notation, preserving bound literals verbatim for the caller.
func describeExplainLeaf(node *planpb.Expr, fieldNames map[int64]string) (string, string, int64) {
	switch expr := node.GetExpr().(type) {
	case *planpb.Expr_UnaryRangeExpr:
		e := expr.UnaryRangeExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		value := explainValue(e.GetValue())
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
		case planpb.OpType_TextMatch:
			if extra := e.GetExtraValues(); len(extra) > 0 {
				return fmt.Sprintf("text_match(%s, %s, minimum_should_match=%s)", field, value, explainValue(extra[0])),
					"text_match", e.GetColumnInfo().GetFieldId()
			}
			return fmt.Sprintf("text_match(%s, %s)", field, value), "text_match", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_PhraseMatch:
			if extra := e.GetExtraValues(); len(extra) > 0 {
				return fmt.Sprintf("phrase_match(%s, %s, slop=%s)", field, value, explainValue(extra[0])),
					"phrase_match", e.GetColumnInfo().GetFieldId()
			}
			return fmt.Sprintf("phrase_match(%s, %s)", field, value), "phrase_match", e.GetColumnInfo().GetFieldId()
		case planpb.OpType_TextMatchFuzzy:
			if extra := e.GetExtraValues(); len(extra) > 0 {
				return fmt.Sprintf("text_match_fuzzy(%s, %s, max_edit_distance=%s)", field, value, explainValue(extra[0])),
					"text_match_fuzzy", e.GetColumnInfo().GetFieldId()
			}
			return fmt.Sprintf("text_match_fuzzy(%s, %s)", field, value), "text_match_fuzzy", e.GetColumnInfo().GetFieldId()
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
				explainValue(e.GetLowerValue()),
				explainValue(e.GetUpperValue()),
				upper),
			"range", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_TermExpr:
		e := expr.TermExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		return fmt.Sprintf("%s in %s", field, explainValues(e.GetValues())),
			"in", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_BinaryArithOpEvalRangeExpr:
		e := expr.BinaryArithOpEvalRangeExpr
		field := explainField(e.GetColumnInfo(), fieldNames)
		return fmt.Sprintf("%s %s %s %s %s",
				field,
				strings.ToLower(e.GetArithOp().String()),
				explainValue(e.GetRightOperand()),
				strings.ToLower(e.GetOp().String()),
				explainValue(e.GetValue())),
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
		op, label := "json_contains", "json_contains"
		switch e.GetOp() {
		case planpb.JSONContainsExpr_ContainsAll:
			op, label = "json_contains_all", "json_contains_all"
		case planpb.JSONContainsExpr_ContainsAny:
			op, label = "json_contains_any", "json_contains_any"
		}
		return fmt.Sprintf("%s %s %s",
				explainField(e.GetColumnInfo(), fieldNames),
				op,
				explainValues(e.GetElements())),
			label, e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_ExistsExpr:
		e := expr.ExistsExpr
		return fmt.Sprintf("exists(%s)", explainField(e.GetInfo(), fieldNames)),
			"exists", e.GetInfo().GetFieldId()
	case *planpb.Expr_NullExpr:
		e := expr.NullExpr
		op := "is null"
		if e.GetOp() == planpb.NullExpr_IsNotNull {
			op = "is not null"
		}
		return fmt.Sprintf("%s %s", explainField(e.GetColumnInfo(), fieldNames), op),
			"null", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_CallExpr:
		e := expr.CallExpr
		var fieldID int64
		for _, param := range e.GetFunctionParameters() {
			if col := param.GetColumnExpr(); col != nil {
				fieldID = col.GetInfo().GetFieldId()
				break
			}
		}
		return fmt.Sprintf("%s(...)", e.GetFunctionName()), "call", fieldID
	case *planpb.Expr_GisfunctionFilterExpr:
		e := expr.GisfunctionFilterExpr
		wkt := strconv.Quote(e.GetWktString())
		if e.GetOp() == planpb.GISFunctionFilterExpr_DWithin {
			return fmt.Sprintf("%s %s %s distance=%s",
					explainField(e.GetColumnInfo(), fieldNames),
					strings.ToLower(e.GetOp().String()),
					wkt,
					strconv.FormatFloat(e.GetDistance(), 'g', -1, 64)),
				"gis", e.GetColumnInfo().GetFieldId()
		}
		return fmt.Sprintf("%s %s %s",
				explainField(e.GetColumnInfo(), fieldNames),
				strings.ToLower(e.GetOp().String()),
				wkt),
			"gis", e.GetColumnInfo().GetFieldId()
	case *planpb.Expr_TimestamptzArithCompareExpr:
		e := expr.TimestamptzArithCompareExpr
		if e.GetInterval() != nil && e.GetArithOp() != planpb.ArithOpType_Unknown {
			return fmt.Sprintf("%s %s interval(%s) %s %s",
					explainField(e.GetTimestamptzColumn(), fieldNames),
					strings.ToLower(e.GetArithOp().String()),
					explainInterval(e.GetInterval()),
					strings.ToLower(e.GetCompareOp().String()),
					explainValue(e.GetCompareValue())),
				"timestamptz", e.GetTimestamptzColumn().GetFieldId()
		}
		return fmt.Sprintf("%s %s %s",
				explainField(e.GetTimestamptzColumn(), fieldNames),
				strings.ToLower(e.GetCompareOp().String()),
				explainValue(e.GetCompareValue())),
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

// explainValue renders one literal value.
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

// explainValues renders a membership list, keeping the payload bounded.
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
