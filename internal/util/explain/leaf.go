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

package explain

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

const (
	summaryLevel    = "summary"
	scalarIndexKind = "scalar_index"
	primaryKeyName  = "primary_key"
	unknownReason   = "unknown"
	cachedReason    = "cached"
	unknownType     = "unknown"
)

// LeafInfo is the plan side of one user predicate leaf tagged with expr_id.
type LeafInfo struct {
	Condition     string
	Type          string
	FieldID       int64
	Field         string
	ElementFilter bool
	HasSubReq     bool
	SubReqIndex   int
}

// ExprAST mirrors the Proxy-side planpb.Expr logical structure so rendering
// preserves AND/OR/NOT/MATCH nesting and original AST child order rather than
// reordering by runtime eval_order.
type ExprAST struct {
	// Logic is "" for a leaf (identified by ExprID), or "and" | "or" | "not" |
	// "match_any" | "match_all" | "match_least" | "match_most" | "match_exact".
	Logic    string
	Field    string
	Count    int64
	HasCount bool
	ExprID   int64
	Children []*ExprAST
}

// PlanAST holds the document-domain and element-domain predicate ASTs of one
// plan or hybrid sub-plan.
type PlanAST struct {
	DocAST     *ExprAST
	ElementAST *ExprAST
}

// SubSearchMeta describes one sub-request of a hybrid search on the Proxy side.
type SubSearchMeta struct {
	SubReqIndex int
	AnnsField   string
}

// OperatorSummary is the cross-segment summary of one physical pipeline operator.
type OperatorSummary struct {
	Operator       string      `json:"operator"`
	Step           int         `json:"step"`
	Domain         string      `json:"domain,omitempty"`
	Segments       int         `json:"segments"`
	RowsIn         int64       `json:"rows_in"`
	RowsOut        int64       `json:"rows_out"`
	CachedSegments int         `json:"cached_segments,omitempty"`
	Conditions     []Condition `json:"conditions,omitempty"`

	kind internalpb.ExplainOperatorKind
}

// SubSearchSummary is the per-sub-request operator pipeline summary for HybridSearch.
type SubSearchSummary struct {
	SubReqIndex     int               `json:"sub_req_index"`
	AnnsField       string            `json:"anns_field"`
	Segments        int               `json:"segments"`
	SealedSegments  int               `json:"sealed_segments"`
	GrowingSegments int               `json:"growing_segments"`
	RowsTotal       int64             `json:"rows_total"`
	Operators       []OperatorSummary `json:"operators"`
}

// Summary is the client-visible execution summary of one Search, HybridSearch,
// or Query request.
type Summary struct {
	Level           string             `json:"level"`
	Segments        int                `json:"segments"`
	SealedSegments  int                `json:"sealed_segments"`
	GrowingSegments int                `json:"growing_segments"`
	RowsTotal       int64              `json:"rows_total"`
	Operators       []OperatorSummary  `json:"operators,omitempty"`
	SubSearches     []SubSearchSummary `json:"sub_searches,omitempty"`
}

func appendLeafConditions(dst []Condition, conds []Condition) []Condition {
	for _, c := range conds {
		if c.Logic != "" {
			dst = appendLeafConditions(dst, c.Children)
			continue
		}
		dst = append(dst, c)
	}
	return dst
}

// AllConditions returns a flattened slice of all leaf conditions across
// Operators and SubSearches in pipeline and AST order.
func (s *Summary) AllConditions() []Condition {
	if s == nil {
		return nil
	}
	var out []Condition
	for _, op := range s.Operators {
		out = appendLeafConditions(out, op.Conditions)
	}
	for _, sub := range s.SubSearches {
		for _, op := range sub.Operators {
			out = appendLeafConditions(out, op.Conditions)
		}
	}
	return out
}

// EvalOrderCount counts how many segments evaluated a condition at one
// 1-based per-segment execution order.
type EvalOrderCount struct {
	EvalOrder int `json:"eval_order"`
	Segments  int `json:"segments"`
}

// Condition represents either a logical combinator node (Logic != "", Children)
// or a predicate leaf node (Logic == "") in an operator's condition tree.
type Condition struct {
	// Logical combinator fields (serialized when Logic != "").
	Logic    string      `json:"logic,omitempty"`
	Count    int64       `json:"count,omitempty"`
	HasCount bool        `json:"-"`
	Children []Condition `json:"children,omitempty"`

	// Leaf predicate fields (serialized when Logic == "").
	ExprID              int64            `json:"-"`
	Condition           string           `json:"condition"`
	Type                string           `json:"type"`
	FieldID             int64            `json:"-"`
	Field               string           `json:"field,omitempty"`
	Segments            int              `json:"segments"`
	EvalOrder           int              `json:"eval_order,omitempty"`
	EvalOrders          []EvalOrderCount `json:"eval_orders,omitempty"`
	Skipped             bool             `json:"skipped,omitempty"`
	IdxScan             int64            `json:"idx_scan"`
	RawScan             int64            `json:"raw_scan"`
	RowsIn              int64            `json:"rows_in"`
	RowsOut             int64            `json:"rows_out"`
	ElementHits         int64            `json:"element_hits,omitempty"`
	Used                []IndexUse       `json:"used"`
	RawScanSegments     int              `json:"raw_scan_segments"`
	Reasons             []ReasonCount    `json:"reasons"`
	unresolvedCondition bool
}

type conditionLogicJSON struct {
	Logic    string      `json:"logic"`
	Field    string      `json:"field,omitempty"`
	Count    *int64      `json:"count,omitempty"`
	Children []Condition `json:"children"`
}

type conditionLeafJSON struct {
	Condition       string           `json:"condition"`
	Type            string           `json:"type"`
	Field           string           `json:"field,omitempty"`
	Segments        int              `json:"segments"`
	EvalOrder       int              `json:"eval_order,omitempty"`
	EvalOrders      []EvalOrderCount `json:"eval_orders,omitempty"`
	Skipped         bool             `json:"skipped,omitempty"`
	IdxScan         int64            `json:"idx_scan"`
	RawScan         int64            `json:"raw_scan"`
	RowsIn          int64            `json:"rows_in"`
	RowsOut         int64            `json:"rows_out"`
	ElementHits     int64            `json:"element_hits,omitempty"`
	Used            []IndexUse       `json:"used"`
	RawScanSegments int              `json:"raw_scan_segments"`
	Reasons         []ReasonCount    `json:"reasons"`
}

// MarshalJSON serializes logical combinator nodes without leaf-only counter
// noise and leaf predicate nodes with their full four-axis counter schema.
func (c Condition) MarshalJSON() ([]byte, error) {
	if c.Logic != "" {
		children := c.Children
		if children == nil {
			children = []Condition{}
		}
		var countPtr *int64
		if c.HasCount || c.Count > 0 {
			cnt := c.Count
			countPtr = &cnt
		}
		return json.Marshal(conditionLogicJSON{
			Logic:    c.Logic,
			Field:    c.Field,
			Count:    countPtr,
			Children: children,
		})
	}
	used := c.Used
	if used == nil {
		used = []IndexUse{}
	}
	reasons := c.Reasons
	if reasons == nil {
		reasons = []ReasonCount{}
	}
	return json.Marshal(conditionLeafJSON{
		Condition:       c.Condition,
		Type:            c.Type,
		Field:           c.Field,
		Segments:        c.Segments,
		EvalOrder:       c.EvalOrder,
		EvalOrders:      c.EvalOrders,
		Skipped:         c.Skipped,
		IdxScan:         c.IdxScan,
		RawScan:         c.RawScan,
		RowsIn:          c.RowsIn,
		RowsOut:         c.RowsOut,
		ElementHits:     c.ElementHits,
		Used:            used,
		RawScanSegments: c.RawScanSegments,
		Reasons:         reasons,
	})
}

// UnmarshalJSON restores either a logical combinator node or a leaf Condition.
func (c *Condition) UnmarshalJSON(data []byte) error {
	var probe struct {
		Logic string `json:"logic"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	if probe.Logic != "" {
		var lg conditionLogicJSON
		if err := json.Unmarshal(data, &lg); err != nil {
			return err
		}
		var count int64
		hasCount := lg.Count != nil
		if hasCount {
			count = *lg.Count
		}
		*c = Condition{
			Logic:    lg.Logic,
			Field:    lg.Field,
			Count:    count,
			HasCount: hasCount,
			Children: lg.Children,
		}
		return nil
	}
	var lf conditionLeafJSON
	if err := json.Unmarshal(data, &lf); err != nil {
		return err
	}
	*c = Condition{
		Condition:       lf.Condition,
		Type:            lf.Type,
		Field:           lf.Field,
		Segments:        lf.Segments,
		EvalOrder:       lf.EvalOrder,
		EvalOrders:      lf.EvalOrders,
		Skipped:         lf.Skipped,
		IdxScan:         lf.IdxScan,
		RawScan:         lf.RawScan,
		RowsIn:          lf.RowsIn,
		RowsOut:         lf.RowsOut,
		ElementHits:     lf.ElementHits,
		Used:            lf.Used,
		RawScanSegments: lf.RawScanSegments,
		Reasons:         lf.Reasons,
	}
	return nil
}

// IndexUse is how many segments answered one condition with one index family.
type IndexUse struct {
	Kind        string `json:"kind"`
	IndexID     int64  `json:"-"`
	IndexName   string `json:"index_name"`
	Family      string `json:"family"`
	Segments    int    `json:"segments"`
	IdxScan     int64  `json:"idx_scan"`
	RawScan     int64  `json:"raw_scan"`
	RowsIn      int64  `json:"rows_in"`
	RowsOut     int64  `json:"rows_out"`
	ElementHits int64  `json:"element_hits,omitempty"`
}

// ReasonCount is what the non-index leaves of a condition did for one fallback
// reason.
type ReasonCount struct {
	Reason   string `json:"reason"`
	Segments int    `json:"segments"`
	IdxScan  int64  `json:"idx_scan"`
	RawScan  int64  `json:"raw_scan"`
	RowsIn   int64  `json:"rows_in"`
	RowsOut  int64  `json:"rows_out"`
}

func finalizeIndexUse(used *IndexUse, indexNames map[int64]string) {
	if used.IndexName != "" {
		return
	}
	if used.IndexID != 0 && len(indexNames) > 0 {
		if name := indexNames[used.IndexID]; name != "" {
			used.IndexName = name
			return
		}
	}
	if used.IndexID == 0 && used.Family == "pk" {
		used.IndexName = primaryKeyName
		return
	}
	if used.Family != "" {
		used.IndexName = used.Family
		return
	}
	used.IndexName = scalarIndexKind
}

func finalizeConditionsRecursive(conds []Condition, indexNames map[int64]string, fieldNames map[int64]string) {
	for i := range conds {
		if conds[i].Logic != "" {
			finalizeConditionsRecursive(conds[i].Children, indexNames, fieldNames)
			continue
		}
		knownSchemaName := ""
		if conds[i].FieldID != 0 && len(fieldNames) > 0 {
			knownSchemaName = fieldNames[conds[i].FieldID]
		}
		if conds[i].Field == "" && knownSchemaName != "" {
			conds[i].Field = knownSchemaName
		}
		if conds[i].unresolvedCondition && conds[i].Field != "" {
			conds[i].Condition = conds[i].Field
			conds[i].unresolvedCondition = false
		}
		if conds[i].Condition == "" {
			conds[i].Condition = unknownReason
		}
		for j := range conds[i].Used {
			finalizeIndexUse(&conds[i].Used[j], indexNames)
		}
		sort.SliceStable(conds[i].Used, func(a, b int) bool {
			if conds[i].Used[a].IndexID != conds[i].Used[b].IndexID {
				return conds[i].Used[a].IndexID < conds[i].Used[b].IndexID
			}
			if conds[i].Used[a].IndexName != conds[i].Used[b].IndexName {
				return conds[i].Used[a].IndexName < conds[i].Used[b].IndexName
			}
			return conds[i].Used[a].Family < conds[i].Used[b].Family
		})
	}
}

// FinalizeSummary resolves internal index_id and field_id references into
// user-visible names across the entire summary before rendering.
func FinalizeSummary(summary *Summary, indexNames map[int64]string, fieldNames map[int64]string) {
	if summary == nil {
		return
	}
	for i := range summary.Operators {
		finalizeConditionsRecursive(summary.Operators[i].Conditions, indexNames, fieldNames)
	}
	for i := range summary.SubSearches {
		for j := range summary.SubSearches[i].Operators {
			finalizeConditionsRecursive(summary.SubSearches[i].Operators[j].Conditions, indexNames, fieldNames)
		}
	}
}

func leavesForSubReq(leaves map[int64]LeafInfo, subReqIndex int) map[int64]LeafInfo {
	if len(leaves) == 0 {
		return nil
	}
	out := make(map[int64]LeafInfo, len(leaves))
	for exprID, info := range leaves {
		if !info.HasSubReq || info.SubReqIndex == subReqIndex {
			out[exprID] = info
		}
	}
	return out
}

// SummarizeWithSubSearches builds the final Summary for Search, Query, or
// HybridSearch, organizing operator condition trees according to Proxy's ASTs.
func SummarizeWithSubSearches(
	explains []*internalpb.ExecutionExplain,
	leaves map[int64]LeafInfo,
	subSearches []SubSearchMeta,
	asts map[int]PlanAST,
) *Summary {
	summary := &Summary{
		Level: summaryLevel,
	}

	merged := AssembleHybridExplain(explains...)
	if len(subSearches) > 0 || (merged != nil && len(merged.GetSubSearches()) > 0) {
		if merged != nil && merged.GetGroup() != nil {
			g := merged.GetGroup()
			summary.Segments = int(g.GetSegments())
			summary.SealedSegments = int(g.GetSealedSegments())
			summary.GrowingSegments = int(g.GetGrowingSegments())
			summary.RowsTotal = g.GetRowsTotal()
		}

		byIdx := make(map[int]*internalpb.ExplainSegmentGroup)
		var orderedIdxs []int
		seenIdx := make(map[int]struct{})
		for _, meta := range subSearches {
			if _, ok := seenIdx[meta.SubReqIndex]; !ok {
				seenIdx[meta.SubReqIndex] = struct{}{}
				orderedIdxs = append(orderedIdxs, meta.SubReqIndex)
			}
		}
		if merged != nil {
			for _, sub := range merged.GetSubSearches() {
				idx := int(sub.GetSubReqIndex())
				byIdx[idx] = sub.GetGroup()
				if _, ok := seenIdx[idx]; !ok {
					seenIdx[idx] = struct{}{}
					orderedIdxs = append(orderedIdxs, idx)
				}
			}
		}
		metaAnns := make(map[int]string, len(subSearches))
		for _, meta := range subSearches {
			metaAnns[meta.SubReqIndex] = meta.AnnsField
		}

		summary.SubSearches = make([]SubSearchSummary, 0, len(orderedIdxs))
		for _, idx := range orderedIdxs {
			g := byIdx[idx]
			var segs, sealed, growing int
			var rowsTotal int64
			if g != nil {
				segs = int(g.GetSegments())
				sealed = int(g.GetSealedSegments())
				growing = int(g.GetGrowingSegments())
				rowsTotal = g.GetRowsTotal()
			}
			scopedLeaves := leavesForSubReq(leaves, idx)
			var planAST PlanAST
			if len(asts) > 0 {
				planAST = asts[idx]
			}
			ops := buildOperatorsFromGroup(g, segs, rowsTotal, scopedLeaves, planAST)
			if ops == nil {
				ops = []OperatorSummary{}
			}
			summary.SubSearches = append(summary.SubSearches, SubSearchSummary{
				SubReqIndex:     idx,
				AnnsField:       metaAnns[idx],
				Segments:        segs,
				SealedSegments:  sealed,
				GrowingSegments: growing,
				RowsTotal:       rowsTotal,
				Operators:       ops,
			})
		}
		return summary
	}

	var g *internalpb.ExplainSegmentGroup
	if merged != nil {
		g = merged.GetGroup()
	}
	if g != nil {
		summary.Segments = int(g.GetSegments())
		summary.SealedSegments = int(g.GetSealedSegments())
		summary.GrowingSegments = int(g.GetGrowingSegments())
		summary.RowsTotal = g.GetRowsTotal()
	}
	var planAST PlanAST
	if len(asts) > 0 {
		planAST = asts[-1]
	}
	summary.Operators = buildOperatorsFromGroup(g, summary.Segments, summary.RowsTotal, leaves, planAST)
	return summary
}

func conditionFromProto(c *internalpb.ExplainConditionStats, leaves map[int64]LeafInfo) Condition {
	condText, typ, fieldID, fieldName, unresolved := leafOf(leaves, c.GetExprId(), c.GetFieldId())
	segs := int(c.GetSegments())
	skipped := segs > 0 && int(c.GetSkippedSegments()) == segs

	var evalOrder int
	var evalOrders []EvalOrderCount
	if len(c.GetEvalOrders()) > 0 {
		evalOrders = make([]EvalOrderCount, 0, len(c.GetEvalOrders()))
		for _, eo := range c.GetEvalOrders() {
			evalOrders = append(evalOrders, EvalOrderCount{
				EvalOrder: int(eo.GetEvalOrder()),
				Segments:  int(eo.GetSegments()),
			})
		}
		if len(evalOrders) == 1 && evalOrders[0].Segments == segs {
			evalOrder = evalOrders[0].EvalOrder
		}
	}

	used := make([]IndexUse, 0, len(c.GetUsed()))
	for _, u := range c.GetUsed() {
		name := ""
		if u.GetIndexId() == 0 && u.GetFamily() == "pk" {
			name = primaryKeyName
		}
		used = append(used, IndexUse{
			Kind:        scalarIndexKind,
			IndexID:     u.GetIndexId(),
			IndexName:   name,
			Family:      u.GetFamily(),
			Segments:    int(u.GetSegments()),
			IdxScan:     u.GetIdxScan(),
			RawScan:     u.GetRawScan(),
			RowsIn:      u.GetRowsIn(),
			RowsOut:     u.GetRowsOut(),
			ElementHits: u.GetElementHits(),
		})
	}

	reasons := make([]ReasonCount, 0, len(c.GetReasons()))
	for _, r := range c.GetReasons() {
		reasons = append(reasons, ReasonCount{
			Reason:   r.GetReason(),
			Segments: int(r.GetSegments()),
			IdxScan:  r.GetIdxScan(),
			RawScan:  r.GetRawScan(),
			RowsIn:   r.GetRowsIn(),
			RowsOut:  r.GetRowsOut(),
		})
	}
	sort.Slice(reasons, func(i, j int) bool {
		return reasons[i].Reason < reasons[j].Reason
	})

	return Condition{
		ExprID:              c.GetExprId(),
		Condition:           condText,
		Type:                typ,
		FieldID:             fieldID,
		Field:               fieldName,
		Segments:            segs,
		EvalOrder:           evalOrder,
		EvalOrders:          evalOrders,
		Skipped:             skipped,
		IdxScan:             c.GetIdxScan(),
		RawScan:             c.GetRawScan(),
		RowsIn:              c.GetRowsIn(),
		RowsOut:             c.GetRowsOut(),
		ElementHits:         c.GetElementHits(),
		Used:                used,
		RawScanSegments:     int(c.GetRawScanSegments()),
		Reasons:             reasons,
		unresolvedCondition: unresolved,
	}
}

func zeroLeafCondition(exprID int64, leaves map[int64]LeafInfo) Condition {
	condText, typ, fieldID, fieldName, unresolved := leafOf(leaves, exprID, 0)
	return Condition{
		ExprID:              exprID,
		Condition:           condText,
		Type:                typ,
		FieldID:             fieldID,
		Field:               fieldName,
		Used:                []IndexUse{},
		Reasons:             []ReasonCount{},
		unresolvedCondition: unresolved,
	}
}

func isElementFilterKind(kind internalpb.ExplainOperatorKind) bool {
	return kind == internalpb.ExplainOperatorKind_EXPLAIN_OP_ELEMENT_FILTER ||
		kind == internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_ELEMENT_FILTER
}

func isDocFilterKind(kind internalpb.ExplainOperatorKind) bool {
	return kind == internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER ||
		kind == internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_FILTER
}

func buildASTConditionNode(node *ExprAST, byExprID map[int64]Condition, leaves map[int64]LeafInfo, visited map[int64]struct{}) (Condition, bool) {
	if node == nil {
		return Condition{}, false
	}
	if node.Logic == "" {
		if node.ExprID == 0 {
			return Condition{}, false
		}
		visited[node.ExprID] = struct{}{}
		if cond, ok := byExprID[node.ExprID]; ok {
			return cond, true
		}
		return zeroLeafCondition(node.ExprID, leaves), true
	}

	children := make([]Condition, 0, len(node.Children))
	for _, ch := range node.Children {
		if ch == nil {
			continue
		}
		// Flatten associative AND/OR chains and unwrap redundant immediate AND
		// inside MATCH_* so each logical level reflects user intent cleanly.
		if (node.Logic == "and" && ch.Logic == "and") ||
			(node.Logic == "or" && ch.Logic == "or") ||
			(strings.HasPrefix(node.Logic, "match_") && ch.Logic == "and") {
			for _, grand := range ch.Children {
				if built, ok := buildASTConditionNode(grand, byExprID, leaves, visited); ok {
					children = append(children, built)
				}
			}
			continue
		}
		if built, ok := buildASTConditionNode(ch, byExprID, leaves, visited); ok {
			children = append(children, built)
		}
	}
	if len(children) == 0 {
		return Condition{}, false
	}
	return Condition{
		Logic:    node.Logic,
		Field:    node.Field,
		Count:    node.Count,
		HasCount: node.HasCount || node.Count > 0,
		Children: children,
	}, true
}

func arrangeOperatorConditions(
	rawConds []Condition,
	ast *ExprAST,
	scopedLeaves map[int64]LeafInfo,
	elementDomain bool,
	backfillMissing bool,
) []Condition {
	byExprID := make(map[int64]Condition, len(rawConds)+len(scopedLeaves))
	var untagged []Condition
	for _, c := range rawConds {
		if c.ExprID != 0 {
			byExprID[c.ExprID] = c
		} else {
			untagged = append(untagged, c)
		}
	}
	if backfillMissing {
		for exprID, info := range scopedLeaves {
			if exprID == 0 || info.ElementFilter != elementDomain {
				continue
			}
			if _, ok := byExprID[exprID]; !ok {
				byExprID[exprID] = zeroLeafCondition(exprID, scopedLeaves)
			}
		}
	}

	var out []Condition
	visited := make(map[int64]struct{}, len(byExprID))
	if ast != nil {
		if ast.Logic == "and" {
			for _, ch := range ast.Children {
				if built, ok := buildASTConditionNode(ch, byExprID, scopedLeaves, visited); ok {
					out = append(out, built)
				}
			}
		} else if built, ok := buildASTConditionNode(ast, byExprID, scopedLeaves, visited); ok {
			out = append(out, built)
		}
	}

	var remaining []Condition
	for exprID, cond := range byExprID {
		if _, ok := visited[exprID]; !ok {
			remaining = append(remaining, cond)
		}
	}
	remaining = append(remaining, untagged...)
	if len(remaining) > 1 {
		sort.SliceStable(remaining, func(i, j int) bool {
			return lessCondition(remaining[i], remaining[j])
		})
	}
	out = append(out, remaining...)
	return out
}

func buildOperatorsFromGroup(
	g *internalpb.ExplainSegmentGroup,
	scopeSegments int,
	scopeRowsTotal int64,
	scopedLeaves map[int64]LeafInfo,
	planAST PlanAST,
) []OperatorSummary {
	type opRaw struct {
		summary OperatorSummary
		conds   []Condition
	}
	var built []opRaw
	hasDocFilter := false
	hasElemFilter := false

	if g != nil {
		built = make([]opRaw, 0, len(g.GetOperators()))
		for _, opPb := range g.GetOperators() {
			if opPb == nil || opPb.GetKind() == internalpb.ExplainOperatorKind_EXPLAIN_OP_UNKNOWN {
				continue
			}
			kind := opPb.GetKind()
			if isDocFilterKind(kind) {
				hasDocFilter = true
			}
			if isElementFilterKind(kind) {
				hasElemFilter = true
			}
			conds := make([]Condition, 0, len(opPb.GetConditions()))
			for _, cPb := range opPb.GetConditions() {
				if cPb == nil {
					continue
				}
				conds = append(conds, conditionFromProto(cPb, scopedLeaves))
			}
			built = append(built, opRaw{
				summary: OperatorSummary{
					Operator:       OperatorKindName(kind),
					Domain:         opPb.GetDomain(),
					Segments:       int(opPb.GetSegments()),
					RowsIn:         opPb.GetRowsIn(),
					RowsOut:        opPb.GetRowsOut(),
					CachedSegments: int(opPb.GetCachedSegments()),
					kind:           kind,
				},
				conds: conds,
			})
		}
	}

	needsDocBackfill := false
	needsElemBackfill := false
	for exprID, info := range scopedLeaves {
		if exprID == 0 {
			continue
		}
		if info.ElementFilter {
			needsElemBackfill = true
		} else {
			needsDocBackfill = true
		}
	}

	if needsDocBackfill && !hasDocFilter {
		built = append(built, opRaw{
			summary: OperatorSummary{
				Operator: OperatorKindName(internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER),
				Segments: scopeSegments,
				RowsIn:   scopeRowsTotal,
				kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
			},
		})
	}
	if needsElemBackfill && !hasElemFilter {
		built = append(built, opRaw{
			summary: OperatorSummary{
				Operator: OperatorKindName(internalpb.ExplainOperatorKind_EXPLAIN_OP_ELEMENT_FILTER),
				Domain:   "element",
				Segments: scopeSegments,
				RowsIn:   scopeRowsTotal,
				kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_ELEMENT_FILTER,
			},
		})
	}

	if len(built) == 0 {
		return nil
	}

	sort.Slice(built, func(i, j int) bool {
		return built[i].summary.kind < built[j].summary.kind
	})

	docFilterCount := 0
	elemFilterCount := 0
	for _, b := range built {
		if isDocFilterKind(b.summary.kind) {
			docFilterCount++
		}
		if isElementFilterKind(b.summary.kind) {
			elemFilterCount++
		}
	}

	ops := make([]OperatorSummary, len(built))
	for i := range built {
		kind := built[i].summary.kind
		var conds []Condition
		switch {
		case isDocFilterKind(kind):
			// When both pre-filter and iterative_filter co-exist across heterogeneous
			// shards, only backfill missing zero-count leaves if an AST is present or
			// only one doc filter operator exists.
			backfill := planAST.DocAST != nil || docFilterCount == 1 || len(built[i].conds) == 0
			conds = arrangeOperatorConditions(built[i].conds, planAST.DocAST, scopedLeaves, false, backfill)
		case isElementFilterKind(kind):
			backfill := planAST.ElementAST != nil || elemFilterCount == 1 || len(built[i].conds) == 0
			conds = arrangeOperatorConditions(built[i].conds, planAST.ElementAST, scopedLeaves, true, backfill)
		default:
			conds = built[i].conds
			if len(conds) > 1 {
				sort.SliceStable(conds, func(a, b int) bool {
					return lessCondition(conds[a], conds[b])
				})
			}
		}
		built[i].summary.Step = i
		built[i].summary.Conditions = conds
		ops[i] = built[i].summary
	}

	return ops
}

// RenderSummaryText formats a summary as indented text, two spaces per level.
func RenderSummaryText(s *Summary) string {
	if s == nil {
		return ""
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s (segments=%d [sealed=%d, growing=%d], rows_total=%d)\n",
		KindQuery, s.Segments, s.SealedSegments, s.GrowingSegments, s.RowsTotal)

	if len(s.SubSearches) > 0 {
		for _, sub := range s.SubSearches {
			writeIndent(&b, 1)
			if sub.AnnsField != "" {
				fmt.Fprintf(&b, "[sub %d] %s %s (segments=%d [sealed=%d, growing=%d], rows_total=%d)\n",
					sub.SubReqIndex, KindSubSearch, sub.AnnsField,
					sub.Segments, sub.SealedSegments, sub.GrowingSegments, sub.RowsTotal)
			} else {
				fmt.Fprintf(&b, "[sub %d] %s (segments=%d [sealed=%d, growing=%d], rows_total=%d)\n",
					sub.SubReqIndex, KindSubSearch,
					sub.Segments, sub.SealedSegments, sub.GrowingSegments, sub.RowsTotal)
			}
			for _, op := range sub.Operators {
				writeOperator(&b, 2, op)
			}
		}
		return b.String()
	}

	for _, op := range s.Operators {
		writeOperator(&b, 1, op)
	}
	return b.String()
}

// RenderSummaryJSON formats a summary as JSON.
func RenderSummaryJSON(s *Summary) (string, error) {
	encoded, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func writeOperator(b *strings.Builder, depth int, op OperatorSummary) {
	writeIndent(b, depth)
	parts := make([]string, 0, 4)
	parts = append(parts, fmt.Sprintf("segments=%d", op.Segments))
	if op.CachedSegments > 0 {
		parts = append(parts, fmt.Sprintf("cached_segments=%d", op.CachedSegments))
	}
	if op.Domain != "" {
		parts = append(parts, "domain="+op.Domain)
	}
	parts = append(parts, fmt.Sprintf("rows: %d -> %d", op.RowsIn, op.RowsOut))
	fmt.Fprintf(b, "[op %d] %s (%s)\n", op.Step, op.Operator, strings.Join(parts, ", "))

	for _, condition := range op.Conditions {
		writeCondition(b, depth+1, condition)
	}
}

func writeCondition(b *strings.Builder, depth int, condition Condition) {
	if condition.Logic != "" {
		writeIndent(b, depth)
		switch {
		case condition.Field != "" && (condition.HasCount || condition.Count > 0):
			fmt.Fprintf(b, "%s (field=%s, count=%d):\n", condition.Logic, condition.Field, condition.Count)
		case condition.Field != "":
			fmt.Fprintf(b, "%s (field=%s):\n", condition.Logic, condition.Field)
		default:
			fmt.Fprintf(b, "%s:\n", condition.Logic)
		}
		for _, child := range condition.Children {
			writeCondition(b, depth+1, child)
		}
		return
	}

	writeIndent(b, depth)
	parts := make([]string, 0, 6)
	parts = append(parts, fmt.Sprintf("segments=%d", condition.Segments))
	if condition.Skipped {
		parts = append(parts, "skipped=true")
	} else {
		parts = append(parts, fmt.Sprintf("rows: %d -> %d", condition.RowsIn, condition.RowsOut))
		parts = append(parts, fmt.Sprintf("idx_scan=%d", condition.IdxScan))
		parts = append(parts, fmt.Sprintf("raw_scan=%d", condition.RawScan))
		if condition.ElementHits > 0 {
			parts = append(parts, fmt.Sprintf("element_hits=%d", condition.ElementHits))
		}
	}
	if orderAttr := formatEvalOrderAttr(condition); orderAttr != "" {
		parts = append(parts, "eval_order="+orderAttr)
	}
	fmt.Fprintf(b, "expr: %s (%s)\n", condition.Condition, strings.Join(parts, ", "))

	for _, used := range condition.Used {
		name := used.IndexName
		if name == "" {
			if used.IndexID == 0 && used.Family == "pk" {
				name = primaryKeyName
			} else if used.Family != "" {
				name = used.Family
			} else {
				name = scalarIndexKind
			}
		}
		writeIndent(b, depth+1)
		header := "index " + name
		if used.Family != "" {
			header += " (" + used.Family + ")"
		}
		usedParts := make([]string, 0, 5)
		usedParts = append(usedParts, fmt.Sprintf("segments=%d", used.Segments))
		if used.RowsIn > 0 || used.RowsOut > 0 || (used.IdxScan == 0 && used.RawScan == 0) {
			usedParts = append(usedParts, fmt.Sprintf("rows: %d -> %d", used.RowsIn, used.RowsOut))
		}
		usedParts = append(usedParts, fmt.Sprintf("idx_scan=%d", used.IdxScan))
		usedParts = append(usedParts, fmt.Sprintf("raw_scan=%d", used.RawScan))
		if used.ElementHits > 0 {
			usedParts = append(usedParts, fmt.Sprintf("element_hits=%d", used.ElementHits))
		}
		fmt.Fprintf(b, "%s: %s\n", header, strings.Join(usedParts, ", "))
	}

	for _, reason := range condition.Reasons {
		writeIndent(b, depth+1)
		reasonParts := make([]string, 0, 4)
		reasonParts = append(reasonParts, fmt.Sprintf("segments=%d", reason.Segments))
		if reason.RowsIn > 0 || reason.RowsOut > 0 || (reason.IdxScan == 0 && reason.RawScan == 0) {
			reasonParts = append(reasonParts, fmt.Sprintf("rows: %d -> %d", reason.RowsIn, reason.RowsOut))
		}
		reasonParts = append(reasonParts, fmt.Sprintf("idx_scan=%d", reason.IdxScan))
		reasonParts = append(reasonParts, fmt.Sprintf("raw_scan=%d", reason.RawScan))
		if reason.Reason == cachedReason {
			fmt.Fprintf(b, "%s: %s\n", cachedReason, strings.Join(reasonParts, ", "))
		} else {
			fmt.Fprintf(b, "raw (%s): %s\n", reason.Reason, strings.Join(reasonParts, ", "))
		}
	}
}

func writeIndent(b *strings.Builder, depth int) {
	b.WriteString(strings.Repeat("  ", depth))
}

func leafOf(leaves map[int64]LeafInfo, exprID, fieldID int64) (condition, typ string, id int64, fieldName string, unresolved bool) {
	leaf, ok := leaves[exprID]
	if !ok {
		return unknownReason, unknownType, fieldID, "", true
	}
	condition, typ, id, fieldName = leaf.Condition, leaf.Type, leaf.FieldID, leaf.Field
	if id == 0 {
		id = fieldID
	}
	if condition == "" {
		if fieldName != "" {
			condition = fieldName
		} else {
			condition = unknownReason
			unresolved = true
		}
	}
	if typ == "" {
		typ = unknownType
	}
	return condition, typ, id, fieldName, unresolved
}

func formatEvalOrderAttr(condition Condition) string {
	if len(condition.EvalOrders) == 0 {
		if condition.EvalOrder > 0 {
			return strconv.Itoa(condition.EvalOrder)
		}
		return ""
	}
	if len(condition.EvalOrders) == 1 && condition.EvalOrders[0].Segments == condition.Segments {
		return strconv.Itoa(condition.EvalOrders[0].EvalOrder)
	}
	parts := make([]string, 0, len(condition.EvalOrders))
	for _, entry := range condition.EvalOrders {
		parts = append(parts, fmt.Sprintf("%d(%d)", entry.EvalOrder, entry.Segments))
	}
	return strings.Join(parts, "|")
}

func lessCondition(a, b Condition) bool {
	aHasID, bHasID := a.ExprID != 0, b.ExprID != 0
	if aHasID != bHasID {
		return aHasID
	}
	if a.ExprID != b.ExprID {
		return a.ExprID < b.ExprID
	}
	return a.FieldID < b.FieldID
}
