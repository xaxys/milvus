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

// Labels of the execution explain summary view.
const (
	summaryLevel        = "summary"
	scalarIndexKind     = "scalar_index"
	primaryKeyName      = "primary_key"
	unknownReason       = "unknown"
	constantTrueReason  = "constant_true"
	constantFalseReason = "constant_false"
	notNullableReason   = "not_nullable"
	cachedReason        = "cached"
	unknownType         = "unknown"
)

// LeafInfo is the plan side of one expr: the rendered predicate and the field it
// reads, so that a summary can name a condition without exposing internal IDs.
type LeafInfo struct {
	// Condition is the normalized predicate text, e.g. `title contains "%降噪%"`.
	Condition string
	// Type is the predicate kind, e.g. range, equal or contains.
	Type string
	// FieldID is the internal schema field ID the predicate reads.
	FieldID int64
	// Field is the user-visible field name (dynamic key unwrapped when $meta).
	Field string
	// ElementFilter marks leaves inside ElementFilterExpr.ElementExpr so empty
	// segment backfills land on element_filter / iterative_element_filter.
	ElementFilter bool
	// HasSubReq is true when the leaf belongs to a specific hybrid sub-request.
	HasSubReq bool
	// SubReqIndex is the 0-based hybrid sub-request index when HasSubReq is true.
	SubReqIndex int
}

// SubSearchMeta describes one sub-request of a hybrid search on the Proxy side.
type SubSearchMeta struct {
	SubReqIndex int
	AnnsField   string
}

// TimeStats is the per-segment latency distribution in microseconds across
// participating segments.
type TimeStats struct {
	SumUs int64 `json:"sum_us"`
	MinUs int64 `json:"min_us"`
	AvgUs int64 `json:"avg_us"`
	MaxUs int64 `json:"max_us"`
}

// OperatorSummary is the cross-segment summary of one physical pipeline operator.
type OperatorSummary struct {
	Operator       string      `json:"operator"`
	Step           int         `json:"step"`
	Domain         string      `json:"domain,omitempty"`
	Segments       int         `json:"segments"`
	RowsIn         int64       `json:"rows_in"`
	RowsOut        int64       `json:"rows_out"`
	TimeUs         *TimeStats  `json:"time_us,omitempty"`
	CachedSegments int         `json:"cached_segments,omitempty"`
	Conditions     []Condition `json:"conditions,omitempty"`

	foldedFilter   *OperatorSummary `json:"-"`
	foldedTimeUs   *TimeStats       `json:"-"`
	origMvccRows   int64            `json:"-"`
	origMvccTimeUs *TimeStats       `json:"-"`
}

// SubSearchSummary is the per-sub-request operator pipeline summary for HybridSearch.
type SubSearchSummary struct {
	SubReqIndex int               `json:"sub_req_index"`
	AnnsField   string            `json:"anns_field"`
	Segments    int               `json:"segments"`
	RowsTotal   int64             `json:"rows_total"`
	Operators   []OperatorSummary `json:"operators"`
}

// Summary is the client-visible execution summary of one Search, HybridSearch,
// or Query request.
type Summary struct {
	Level       string             `json:"level"`
	Segments    int                `json:"segments"`
	RowsTotal   int64              `json:"rows_total"`
	Operators   []OperatorSummary  `json:"operators,omitempty"`
	SubSearches []SubSearchSummary `json:"sub_searches,omitempty"`

	// Internal aggregated condition view kept out of external JSON.
	RowsIn     int64       `json:"-"`
	RowsOut    int64       `json:"-"`
	Conditions []Condition `json:"-"`
}

// AllConditions returns a flattened slice of all conditions across Operators and
// SubSearches in pipeline order.
func (s *Summary) AllConditions() []Condition {
	if s == nil {
		return nil
	}
	var out []Condition
	for _, op := range s.Operators {
		out = append(out, op.Conditions...)
	}
	for _, sub := range s.SubSearches {
		for _, op := range sub.Operators {
			out = append(out, op.Conditions...)
		}
	}
	if len(out) == 0 && len(s.Conditions) > 0 {
		out = append(out, s.Conditions...)
	}
	return out
}

// EvalOrderCount counts how many segments evaluated a condition at one
// 1-based per-segment execution order.
type EvalOrderCount struct {
	EvalOrder int `json:"eval_order"`
	Segments  int `json:"segments"`
}

// Condition is one user filter expression together with every decision its
// leaves made across segments. Internal ExprID and FieldID are used during
// cross-segment reduction and omitted from external serialization.
type Condition struct {
	ExprID              int64            `json:"-"`
	Condition           string           `json:"condition"`
	Type                string           `json:"type"`
	FieldID             int64            `json:"-"`
	Field               string           `json:"field,omitempty"`
	Segments            int              `json:"segments"`
	SkippedSegments     int              `json:"skipped_segments,omitempty"`
	EvalOrder           int              `json:"eval_order,omitempty"`
	EvalOrders          []EvalOrderCount `json:"eval_orders,omitempty"`
	Skipped             bool             `json:"skipped,omitempty"`
	IdxScan             int64            `json:"idx_scan"`
	RawScan             int64            `json:"raw_scan"`
	RowsIn              int64            `json:"rows_in"`
	RowsOut             int64            `json:"rows_out"`
	ElementHits         int64            `json:"element_hits,omitempty"`
	EvalTimeUs          *TimeStats       `json:"eval_time_us,omitempty"`
	PrefetchTimeUs      *TimeStats       `json:"prefetch_time_us,omitempty"`
	Used                []IndexUse       `json:"used"`
	RawScanSegments     int              `json:"raw_scan_segments"`
	Reasons             []ReasonCount    `json:"reasons"`
	unresolvedCondition bool
}

// IndexUse is how many segments answered one condition with one index family.
// Internal IndexID is resolved to IndexName before external serialization.
type IndexUse struct {
	Kind           string     `json:"kind"`
	IndexID        int64      `json:"-"`
	IndexName      string     `json:"index_name"`
	Family         string     `json:"family"`
	Segments       int        `json:"segments"`
	IdxScan        int64      `json:"idx_scan"`
	RawScan        int64      `json:"raw_scan"`
	RowsIn         int64      `json:"rows_in"`
	RowsOut        int64      `json:"rows_out"`
	ElementHits    int64      `json:"element_hits,omitempty"`
	EvalTimeUs     *TimeStats `json:"eval_time_us,omitempty"`
	PrefetchTimeUs *TimeStats `json:"prefetch_time_us,omitempty"`
}

// ReasonCount is what the non-index leaves of a condition did for one fallback
// reason.
type ReasonCount struct {
	Reason         string     `json:"reason"`
	Segments       int        `json:"segments"`
	IdxScan        int64      `json:"idx_scan"`
	RawScan        int64      `json:"raw_scan"`
	RowsIn         int64      `json:"rows_in"`
	RowsOut        int64      `json:"rows_out"`
	EvalTimeUs     *TimeStats `json:"eval_time_us,omitempty"`
	PrefetchTimeUs *TimeStats `json:"prefetch_time_us,omitempty"`
}

// NameIndexUses fills in human-readable index names across all operators,
// sub-searches, and conditions. Built-in PK lookups use "primary_key", and any
// unresolved dropped index falls back to its index family name.
func NameIndexUses(summary *Summary, names map[int64]string) {
	FinalizeSummary(summary, names, nil)
}

// FinalizeSummary resolves internal index_id and field_id references into
// user-visible names across the entire summary before rendering.
func FinalizeSummary(summary *Summary, indexNames map[int64]string, fieldNames map[int64]string) {
	if summary == nil {
		return
	}
	finalizeConditions := func(conds []Condition) {
		for i := range conds {
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
		}
	}
	for i := range summary.Operators {
		finalizeConditions(summary.Operators[i].Conditions)
	}
	for i := range summary.SubSearches {
		for j := range summary.SubSearches[i].Operators {
			finalizeConditions(summary.SubSearches[i].Operators[j].Conditions)
		}
	}
	finalizeConditions(summary.Conditions)
}

func finalizeIndexUse(used *IndexUse, indexNames map[int64]string) {
	if used == nil {
		return
	}
	if len(indexNames) > 0 {
		if name := indexNames[used.IndexID]; name != "" {
			used.IndexName = name
			return
		}
	}
	if used.IndexName != "" {
		return
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

func normalizeDeclineReason(raw string) (reason string, skip bool) {
	switch raw {
	case constantTrueReason, constantFalseReason:
		return "", true
	case "no_index_on_field", "no_index":
		return "no_index", false
	case "index_unusable", "candidate_needs_refine", "offset_input_with_ngram", "flat_json_required", "unusable":
		return "unusable", false
	case "schema_not_nullable", notNullableReason:
		return notNullableReason, false
	case cachedReason:
		return cachedReason, false
	case "value_type_mismatch", "type_mismatch":
		return "type_mismatch", false
	case "json_path_not_covered", "path_uncovered":
		return "path_uncovered", false
	case "capability_mismatch", "unsupported_op":
		return "unsupported_op", false
	case "literal_not_supported", "unsupported_literal":
		return "unsupported_literal", false
	case "hinted_index_not_selected", "hint_mismatch":
		return "hint_mismatch", false
	case "", "none":
		return unknownReason, false
	default:
		return unknownReason, false
	}
}

type groupKey struct {
	fieldID       int64
	exprID        int64
	used          bool
	indexID       int64
	family        string
	path          string
	nested        string
	declineReason string
	secondary     bool
	skipped       bool
}

type decisionKey struct {
	segmentID string
	key       groupKey
}

type conditionKey struct {
	exprID  int64
	fieldID int64
}

type usedKey struct {
	indexID int64
	family  string
	path    string
}

type axisCounters struct {
	idxScan     int64
	rawScan     int64
	rowsIn      int64
	rowsOut     int64
	elementHits int64
}

func countersOf(filterGroup *internalpb.ExecutionExplain, used bool) axisCounters {
	raw := filterGroup.GetCounters()
	if _, ok := raw["rows_in"]; ok {
		return axisCounters{
			idxScan:     raw["idx_scan"],
			rawScan:     raw["raw_scan"],
			rowsIn:      raw["rows_in"],
			rowsOut:     raw["rows_out"],
			elementHits: raw["element_hits"],
		}
	}
	legacy := axisCounters{
		rawScan:     raw["rows_scanned"],
		rowsOut:     raw["rows_matched"],
		elementHits: raw["element_hits"],
	}
	if used {
		legacy.rowsIn = legacy.rowsOut
		legacy.idxScan = legacy.rowsOut
	} else {
		legacy.rowsIn = legacy.rawScan
	}
	return legacy
}

type segCounterSet struct {
	primary   map[string]axisCounters
	secondary map[string]axisCounters
}

func newSegCounterSet() segCounterSet {
	return segCounterSet{
		primary:   make(map[string]axisCounters, 4),
		secondary: make(map[string]axisCounters, 2),
	}
}

func (s *segCounterSet) add(segCounters map[string]axisCounters, secondary bool) {
	if !secondary {
		for segID, c := range segCounters {
			cur := s.primary[segID]
			addCounters(&cur, c)
			s.primary[segID] = cur
		}
		return
	}
	for segID, c := range segCounters {
		cur := s.secondary[segID]
		maxCounters(&cur, c)
		s.secondary[segID] = cur
	}
}

func (s segCounterSet) totals() axisCounters {
	var out axisCounters
	for _, c := range s.primary {
		addCounters(&out, c)
	}
	for segID, sec := range s.secondary {
		pri, hasPri := s.primary[segID]
		if !hasPri {
			addCounters(&out, sec)
			continue
		}
		if sec.idxScan > pri.idxScan {
			out.idxScan += sec.idxScan - pri.idxScan
		}
		if sec.rawScan > pri.rawScan {
			out.rawScan += sec.rawScan - pri.rawScan
		}
		out.elementHits += sec.elementHits
	}
	return out
}

type groupAgg struct {
	key                     groupKey
	fieldID                 int64
	segments                map[string]struct{}
	skippedSegments         map[string]struct{}
	nonSkippedSegments      map[string]struct{}
	unskippedActiveSegments map[string]struct{}
	orders                  map[int]map[string]struct{}
	segCounters             map[string]axisCounters
	segEvalUs               map[string]int64
	segPrefetchUs           map[string]int64
}

type usedAgg struct {
	key           usedKey
	segments      map[string]struct{}
	counters      segCounterSet
	evalBySeg     map[string]int64
	prefetchBySeg map[string]int64
}

type conditionAgg struct {
	key                     conditionKey
	firstFieldID            int64
	segments                map[string]struct{}
	skippedSegments         map[string]struct{}
	nonSkippedSegments      map[string]struct{}
	unskippedActiveSegments map[string]struct{}
	orders                  map[int]map[string]struct{}
	counters                segCounterSet
	evalBySeg               map[string]int64
	prefetchBySeg           map[string]int64
	used                    map[usedKey]*usedAgg
	reasons                 map[string]*reasonAgg
}

type reasonAgg struct {
	segments      map[string]struct{}
	counters      segCounterSet
	evalBySeg     map[string]int64
	prefetchBySeg map[string]int64
}

type opKey struct {
	step int
	name string
}

type filterGroupCollector struct {
	groups     []*groupAgg
	groupIndex map[groupKey]int
	counted    map[decisionKey]struct{}
}

func newFilterGroupCollector(capHint int) *filterGroupCollector {
	return &filterGroupCollector{
		groups:     make([]*groupAgg, 0, capHint),
		groupIndex: make(map[groupKey]int, capHint),
		counted:    make(map[decisionKey]struct{}, capHint),
	}
}

func (c *filterGroupCollector) add(segmentID string, segRowsTotal int64, filterGroup *internalpb.ExecutionExplain) {
	if filterGroup == nil || filterGroup.GetKind() != KindFilterGroup {
		return
	}
	key, fieldID := groupKeyOf(filterGroup)
	index, ok := c.groupIndex[key]
	if !ok {
		index = len(c.groups)
		c.groupIndex[key] = index
		c.groups = append(c.groups, &groupAgg{
			key:                     key,
			fieldID:                 fieldID,
			segments:                make(map[string]struct{}, 4),
			skippedSegments:         make(map[string]struct{}, 2),
			nonSkippedSegments:      make(map[string]struct{}, 4),
			unskippedActiveSegments: make(map[string]struct{}, 4),
			orders:                  make(map[int]map[string]struct{}, 1),
			segCounters:             make(map[string]axisCounters, 4),
			segEvalUs:               make(map[string]int64, 4),
			segPrefetchUs:           make(map[string]int64, 2),
		})
	}
	agg := c.groups[index]
	if agg.fieldID == 0 && fieldID != 0 {
		agg.fieldID = fieldID
	}
	agg.segments[segmentID] = struct{}{}
	if key.skipped {
		agg.skippedSegments[segmentID] = struct{}{}
	} else {
		agg.nonSkippedSegments[segmentID] = struct{}{}
		if segRowsTotal > 0 {
			agg.unskippedActiveSegments[segmentID] = struct{}{}
		}
	}
	if order := int(int64Attribute(filterGroup.GetAttributes(), "eval_order")); order > 0 {
		segs := agg.orders[order]
		if segs == nil {
			segs = make(map[string]struct{}, 4)
			agg.orders[order] = segs
		}
		segs[segmentID] = struct{}{}
	}
	decision := decisionKey{segmentID: segmentID, key: key}
	if _, ok := c.counted[decision]; ok {
		return
	}
	c.counted[decision] = struct{}{}
	ctrs := countersOf(filterGroup, key.used)
	cur := agg.segCounters[segmentID]
	addCounters(&cur, ctrs)
	agg.segCounters[segmentID] = cur
	rawCtrs := filterGroup.GetCounters()
	if !key.skipped {
		if evalUs, ok := rawCtrs["eval_us"]; ok {
			mergeMaxInt64(agg.segEvalUs, segmentID, evalUs)
		}
	}
	if pfUs, ok := rawCtrs["prefetch_us"]; ok && pfUs > 0 {
		mergeMaxInt64(agg.segPrefetchUs, segmentID, pfUs)
	}
}

type operatorAgg struct {
	key            opKey
	domain         string
	segments       map[string]struct{}
	cachedSegments map[string]struct{}
	countedSegs    map[string]struct{}
	segTimes       map[string]int64
	rowsIn         int64
	rowsOut        int64
	fg             *filterGroupCollector
}

// FoldSegments pre-aggregates raw per-segment (KindSegment) explain trees on a
// worker into a single KindSegmentGroup combiner node. Returns nil when no
// segment recorded anything.
func FoldSegments(segments []*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	return FoldSegmentsWithName("", segments)
}

// FoldSegmentsWithName is FoldSegments with an explicit worker-slice name (e.g.
// channel/nodeID/scope) so HybridSearch top-level segment deduplication stays
// exact even when multiple vchannels share a QueryNode.
func FoldSegmentsWithName(name string, segments []*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	partial := collectRawScope(segments, nil)
	if partial.segments == 0 && len(partial.ops) == 0 {
		return nil
	}
	return partial.toProto(name)
}

// Summarize folds segment-level or pre-aggregated segment_group explain nodes
// into a single-scope Summary.
func Summarize(segments []*internalpb.ExecutionExplain, leaves map[int64]LeafInfo) *Summary {
	tagged := make([]TaggedSegment, 0, len(segments))
	for _, seg := range segments {
		if seg != nil {
			tagged = append(tagged, TaggedSegment{SubReqIndex: -1, Segment: seg})
		}
	}
	return SummarizeWithSubSearches(tagged, leaves, nil)
}

// SummarizeWithSubSearches folds tagged segment / segment_group nodes into
// either a single operator pipeline summary (Search / Query) or a
// per-sub-search summary (HybridSearch).
func SummarizeWithSubSearches(
	tagged []TaggedSegment,
	leaves map[int64]LeafInfo,
	subSearches []SubSearchMeta,
) *Summary {
	summary := &Summary{
		Level:      summaryLevel,
		Conditions: []Condition{},
	}

	isHybrid := len(subSearches) > 0
	if !isHybrid {
		for _, entry := range tagged {
			if entry.SubReqIndex >= 0 {
				isHybrid = true
				break
			}
		}
	}

	if !isHybrid {
		units := make([]*internalpb.ExecutionExplain, 0, len(tagged))
		for _, entry := range tagged {
			if entry.Segment != nil {
				units = append(units, entry.Segment)
			}
		}
		segCount, rowsTotal, ops := summarizeScope(units, leaves)
		summary.Segments = segCount
		summary.RowsTotal = rowsTotal
		summary.Operators = ops
		populateSummaryConditionTotals(summary)
		return summary
	}

	// HybridSearch: union distinct segments / worker-slice groups at the top
	// level and summarize each sub-search independently in 0..N-1 order.
	type groupTotals struct {
		segments  int
		rowsTotal int64
	}
	unionSegRows := make(map[string]int64, len(tagged))
	perSubKeyedGroups := make(map[int]map[string]groupTotals)
	perSubUnnamedGroups := make(map[int]groupTotals)

	bySubReq := make(map[int][]*internalpb.ExecutionExplain, max(1, len(subSearches)))
	metaByIdx := make(map[int]SubSearchMeta, len(subSearches))
	order := make([]int, 0, len(subSearches))
	for _, meta := range subSearches {
		if _, exists := metaByIdx[meta.SubReqIndex]; !exists {
			metaByIdx[meta.SubReqIndex] = meta
			order = append(order, meta.SubReqIndex)
		}
	}
	for _, entry := range tagged {
		if entry.Segment == nil {
			continue
		}
		subIdx := entry.SubReqIndex
		if subIdx < 0 {
			subIdx = 0
		}
		if entry.Segment.GetKind() == KindSegmentGroup {
			segs := int(entry.Segment.GetCounters()["segments"])
			rows := entry.Segment.GetCounters()["rows_total"]
			key := entry.Segment.GetName()
			if key == "" {
				key = entry.ScopeKey
			}
			if key != "" {
				subMap := perSubKeyedGroups[subIdx]
				if subMap == nil {
					subMap = make(map[string]groupTotals, 4)
					perSubKeyedGroups[subIdx] = subMap
				}
				cur := subMap[key]
				cur.segments += segs
				cur.rowsTotal += rows
				subMap[key] = cur
			} else {
				cur := perSubUnnamedGroups[subIdx]
				cur.segments += segs
				cur.rowsTotal += rows
				perSubUnnamedGroups[subIdx] = cur
			}
		} else {
			segID := entry.Segment.GetName()
			rowsTotal := entry.Segment.GetCounters()["rows_total"]
			if prev, seen := unionSegRows[segID]; !seen || rowsTotal > prev {
				unionSegRows[segID] = rowsTotal
			}
		}
		if _, exists := metaByIdx[subIdx]; !exists {
			metaByIdx[subIdx] = SubSearchMeta{SubReqIndex: subIdx}
			order = append(order, subIdx)
		}
		bySubReq[subIdx] = append(bySubReq[subIdx], entry.Segment)
	}
	sort.Ints(order)

	summary.Segments = len(unionSegRows)
	for _, rows := range unionSegRows {
		summary.RowsTotal += rows
	}
	if len(perSubKeyedGroups) > 0 {
		maxByKey := make(map[string]groupTotals, 8)
		for _, subMap := range perSubKeyedGroups {
			for key, cur := range subMap {
				prev := maxByKey[key]
				if cur.segments > prev.segments {
					prev.segments = cur.segments
				}
				if cur.rowsTotal > prev.rowsTotal {
					prev.rowsTotal = cur.rowsTotal
				}
				maxByKey[key] = prev
			}
		}
		for _, tot := range maxByKey {
			summary.Segments += tot.segments
			summary.RowsTotal += tot.rowsTotal
		}
	}
	if len(perSubUnnamedGroups) > 0 {
		var maxUnnamed groupTotals
		for _, cur := range perSubUnnamedGroups {
			if cur.segments > maxUnnamed.segments {
				maxUnnamed.segments = cur.segments
			}
			if cur.rowsTotal > maxUnnamed.rowsTotal {
				maxUnnamed.rowsTotal = cur.rowsTotal
			}
		}
		summary.Segments += maxUnnamed.segments
		summary.RowsTotal += maxUnnamed.rowsTotal
	}
	summary.SubSearches = make([]SubSearchSummary, 0, len(order))
	for _, subIdx := range order {
		meta := metaByIdx[subIdx]
		scopedLeaves := leavesForSubReq(leaves, subIdx)
		segCount, rowsTotal, ops := summarizeScope(bySubReq[subIdx], scopedLeaves)
		if ops == nil {
			ops = []OperatorSummary{}
		}
		summary.SubSearches = append(summary.SubSearches, SubSearchSummary{
			SubReqIndex: subIdx,
			AnnsField:   meta.AnnsField,
			Segments:    segCount,
			RowsTotal:   rowsTotal,
			Operators:   ops,
		})
	}
	populateSummaryConditionTotals(summary)
	return summary
}

func leavesForSubReq(leaves map[int64]LeafInfo, subIdx int) map[int64]LeafInfo {
	if len(leaves) == 0 {
		return nil
	}
	var out map[int64]LeafInfo
	for exprID, info := range leaves {
		if info.HasSubReq && info.SubReqIndex != subIdx {
			continue
		}
		if out == nil {
			out = make(map[int64]LeafInfo, len(leaves))
		}
		out[exprID] = info
	}
	return out
}

type timeAccum struct {
	sum     int64
	min     int64
	max     int64
	count   int64
	untimed bool
}

func primaryTimeAccum(reported map[string]int64, denomSegs, timedSegs map[string]struct{}) timeAccum {
	if len(denomSegs) == 0 {
		return timeAccum{}
	}
	if len(reported) == 0 || !allSegmentsTimed(denomSegs, timedSegs) {
		return timeAccum{untimed: true}
	}
	var sum, minVal, maxVal, count int64
	for segID := range denomSegs {
		val, ok := reported[segID]
		if !ok {
			return timeAccum{untimed: true}
		}
		if val < 0 {
			val = 0
		}
		sum += val
		if count == 0 || val < minVal {
			minVal = val
		}
		if count == 0 || val > maxVal {
			maxVal = val
		}
		count++
	}
	return timeAccum{sum: sum, min: minVal, max: maxVal, count: count}
}

func optionalTimeAccum(reported map[string]int64, denomSegs, timedSegs map[string]struct{}) timeAccum {
	if len(denomSegs) == 0 {
		return timeAccum{}
	}
	if !allSegmentsTimed(denomSegs, timedSegs) {
		return timeAccum{untimed: true}
	}
	var sum, minVal, maxVal, count int64
	for segID := range denomSegs {
		val := reported[segID]
		if val < 0 {
			val = 0
		}
		sum += val
		if count == 0 || val < minVal {
			minVal = val
		}
		if count == 0 || val > maxVal {
			maxVal = val
		}
		count++
	}
	return timeAccum{sum: sum, min: minVal, max: maxVal, count: count}
}

func mergeTimeAccum(dst *timeAccum, src timeAccum) {
	if src.untimed {
		dst.untimed = true
		return
	}
	if dst.untimed || src.count <= 0 {
		return
	}
	if dst.count == 0 {
		*dst = src
		return
	}
	dst.sum += src.sum
	if src.min < dst.min {
		dst.min = src.min
	}
	if src.max > dst.max {
		dst.max = src.max
	}
	dst.count += src.count
}

func (a timeAccum) primaryStats() *TimeStats {
	if a.untimed || a.count <= 0 {
		return nil
	}
	return &TimeStats{
		SumUs: a.sum,
		MinUs: a.min,
		AvgUs: a.sum / a.count,
		MaxUs: a.max,
	}
}

func (a timeAccum) optionalStats() *TimeStats {
	if a.untimed || a.count <= 0 || a.sum <= 0 {
		return nil
	}
	return &TimeStats{
		SumUs: a.sum,
		MinUs: a.min,
		AvgUs: a.sum / a.count,
		MaxUs: a.max,
	}
}

func writeTimeAccum(dst map[string]int64, prefix string, a timeAccum) {
	if a.untimed {
		dst[prefix+"_untimed"] = 1
		return
	}
	if a.count <= 0 {
		return
	}
	dst[prefix+"_count_us"] = a.count
	if a.sum != 0 {
		dst[prefix+"_sum_us"] = a.sum
	}
	if a.min != 0 {
		dst[prefix+"_min_us"] = a.min
	}
	if a.max != 0 {
		dst[prefix+"_max_us"] = a.max
	}
}

func readTimeAccum(src map[string]int64, prefix string, expectedDenom int) timeAccum {
	if expectedDenom <= 0 {
		return timeAccum{}
	}
	if src[prefix+"_untimed"] > 0 {
		return timeAccum{untimed: true}
	}
	cnt := src[prefix+"_count_us"]
	if cnt < int64(expectedDenom) {
		return timeAccum{untimed: true}
	}
	return timeAccum{
		sum:   src[prefix+"_sum_us"],
		min:   src[prefix+"_min_us"],
		max:   src[prefix+"_max_us"],
		count: cnt,
	}
}

type partialUsed struct {
	key          usedKey
	segments     int
	totals       axisCounters
	evalTime     timeAccum
	prefetchTime timeAccum
}

type partialReason struct {
	reason       string
	segments     int
	totals       axisCounters
	evalTime     timeAccum
	prefetchTime timeAccum
}

type partialCondition struct {
	key                     conditionKey
	firstFieldID            int64
	segments                int
	pureSkippedSegments     int
	skippedSegmentsAny      int
	unskippedActiveSegments int
	rawScanSegments         int
	orders                  map[int]int
	totals                  axisCounters
	evalTime                timeAccum
	prefetchTime            timeAccum
	used                    map[usedKey]*partialUsed
	reasons                 map[string]*partialReason
}

type partialOperator struct {
	key            opKey
	domain         string
	segments       int
	cachedSegments int
	rowsIn         int64
	rowsOut        int64
	time           timeAccum
	foldedTime     timeAccum
	conds          map[conditionKey]*partialCondition
}

type partialScope struct {
	segments                int
	rowsTotal               int64
	disallowEmptyFilterFold bool
	ops                     map[opKey]*partialOperator
}

func newPartialScope() *partialScope {
	return &partialScope{
		ops: make(map[opKey]*partialOperator, 4),
	}
}

func summarizeScope(
	units []*internalpb.ExecutionExplain,
	scopedLeaves map[int64]LeafInfo,
) (int, int64, []OperatorSummary) {
	acc := newPartialScope()
	var rawSegs []*internalpb.ExecutionExplain
	for _, unit := range units {
		if unit == nil {
			continue
		}
		if unit.GetKind() == KindSegmentGroup {
			acc.merge(partialScopeFromGroup(unit))
			continue
		}
		rawSegs = append(rawSegs, unit)
	}
	if len(rawSegs) > 0 {
		acc.merge(collectRawScope(rawSegs, scopedLeaves))
	}
	return acc.finalize(scopedLeaves)
}

func collectRawScope(
	segments []*internalpb.ExecutionExplain,
	scopedLeaves map[int64]LeafInfo,
) *partialScope {
	seenSegRows := make(map[string]int64, len(segments))
	timedSegs := make(map[string]struct{}, len(segments))
	opMap := make(map[opKey]*operatorAgg, 4)

	getOrCreateOp := func(step int, name string) *operatorAgg {
		k := opKey{step: step, name: name}
		agg := opMap[k]
		if agg == nil {
			agg = &operatorAgg{
				key:            k,
				segments:       make(map[string]struct{}, max(1, len(segments))),
				cachedSegments: make(map[string]struct{}),
				countedSegs:    make(map[string]struct{}, max(1, len(segments))),
				segTimes:       make(map[string]int64, max(1, len(segments))),
				fg:             newFilterGroupCollector(max(1, len(segments))),
			}
			opMap[k] = agg
		}
		return agg
	}

	for _, segment := range segments {
		if segment == nil {
			continue
		}
		segID := segment.GetName()
		segRowsTotal := segment.GetCounters()["rows_total"]
		if prev, seen := seenSegRows[segID]; !seen || segRowsTotal > prev {
			seenSegRows[segID] = segRowsTotal
		}
		if segmentHasTimingCounters(segment) {
			timedSegs[segID] = struct{}{}
		}

		if hasOperatorChildren(segment) {
			for _, child := range segment.GetChildren() {
				if child == nil || child.GetKind() != KindOperator {
					continue
				}
				attrs := child.GetAttributes()
				name := attrs["name"]
				if name == "" {
					name = child.GetName()
				}
				if name == "" {
					continue
				}
				step := int(int64Attribute(attrs, "step"))
				agg := getOrCreateOp(step, name)
				if agg.domain == "" && attrs["domain"] != "" {
					agg.domain = attrs["domain"]
				}
				agg.segments[segID] = struct{}{}
				if attrs["cached"] == "true" {
					agg.cachedSegments[segID] = struct{}{}
				}
				if _, counted := agg.countedSegs[segID]; !counted {
					agg.countedSegs[segID] = struct{}{}
					ctrs := child.GetCounters()
					agg.rowsIn += ctrs["rows_in"]
					agg.rowsOut += ctrs["rows_out"]
					if tUs, hasTime := ctrs["time_us"]; hasTime {
						agg.segTimes[segID] = tUs
					}
				}
				for _, fg := range child.GetChildren() {
					agg.fg.add(segID, segRowsTotal, fg)
				}
			}
			continue
		}

		// Legacy flat segment -> filter_group tree (rolling upgrade or unit test):
		// synthesize step=0 "filter" when filter_groups or planned leaves exist.
		var filterGroups []*internalpb.ExecutionExplain
		for _, child := range segment.GetChildren() {
			if child != nil && child.GetKind() == KindFilterGroup {
				filterGroups = append(filterGroups, child)
			}
		}
		if len(filterGroups) == 0 && len(scopedLeaves) == 0 {
			continue
		}
		agg := getOrCreateOp(0, "filter")
		agg.segments[segID] = struct{}{}
		for _, fg := range filterGroups {
			agg.fg.add(segID, segRowsTotal, fg)
		}
		if _, counted := agg.countedSegs[segID]; !counted {
			agg.countedSegs[segID] = struct{}{}
			segIn := segRowsTotal
			var segOut int64
			if rawOut, ok := segment.GetCounters()["rows_out"]; ok {
				segOut = rawOut
			} else {
				for _, fg := range filterGroups {
					if fg.GetAttributes()["secondary_operand"] == "true" {
						continue
					}
					c := countersOf(fg, fg.GetAttributes()["index_id"] != "")
					if segRowsTotal == 0 && c.rowsIn > segIn {
						segIn = c.rowsIn
					}
					segOut += c.rowsOut
				}
				if segIn > 0 && segOut > segIn {
					segOut = segIn
				}
			}
			agg.rowsIn += segIn
			agg.rowsOut += segOut
		}
	}

	out := newPartialScope()
	out.segments = len(seenSegRows)
	for _, rows := range seenSegRows {
		out.rowsTotal += rows
	}
	op0 := opMap[opKey{step: 0, name: "filter"}]
	op1 := opMap[opKey{step: 1, name: "mvcc"}]
	if op0 != nil || op1 != nil {
		var s0, s1 map[string]struct{}
		if op0 != nil {
			s0 = op0.segments
		}
		if op1 != nil {
			s1 = op1.segments
		}
		if !sameSegmentSet(s0, s1) {
			out.disallowEmptyFilterFold = true
		}
	}

	var foldedFilterMvccTime timeAccum
	if !out.disallowEmptyFilterFold && op0 != nil && op1 != nil && len(op0.cachedSegments) == 0 {
		mergedTimes := make(map[string]int64, min(len(op0.segTimes), len(op1.segTimes)))
		for segID, t0 := range op0.segTimes {
			if t1, ok := op1.segTimes[segID]; ok {
				mergedTimes[segID] = t0 + t1
			}
		}
		foldedFilterMvccTime = primaryTimeAccum(mergedTimes, op1.segments, timedSegs)
	}

	for k, agg := range opMap {
		condAggs := buildConditionAggs(agg.fg.groups)
		condMap := make(map[conditionKey]*partialCondition, len(condAggs))
		for _, cAgg := range condAggs {
			pc := cAgg.toPartial(timedSegs)
			condMap[pc.key] = pc
		}
		po := &partialOperator{
			key:            k,
			domain:         agg.domain,
			segments:       len(agg.segments),
			cachedSegments: len(agg.cachedSegments),
			rowsIn:         agg.rowsIn,
			rowsOut:        agg.rowsOut,
			time:           primaryTimeAccum(agg.segTimes, agg.segments, timedSegs),
			conds:          condMap,
		}
		if k.step == 1 && k.name == "mvcc" && !out.disallowEmptyFilterFold && op0 != nil {
			po.foldedTime = foldedFilterMvccTime
		}
		out.ops[k] = po
	}
	return out
}

func (dst *partialScope) merge(src *partialScope) {
	if src == nil {
		return
	}
	dst.segments += src.segments
	dst.rowsTotal += src.rowsTotal
	if src.disallowEmptyFilterFold {
		dst.disallowEmptyFilterFold = true
	}
	for k, srcOp := range src.ops {
		dstOp := dst.ops[k]
		if dstOp == nil {
			dstOp = &partialOperator{
				key:    k,
				domain: srcOp.domain,
				conds:  make(map[conditionKey]*partialCondition, len(srcOp.conds)),
			}
			dst.ops[k] = dstOp
		} else if dstOp.domain == "" && srcOp.domain != "" {
			dstOp.domain = srcOp.domain
		}
		dstOp.segments += srcOp.segments
		dstOp.cachedSegments += srcOp.cachedSegments
		dstOp.rowsIn += srcOp.rowsIn
		dstOp.rowsOut += srcOp.rowsOut
		mergeTimeAccum(&dstOp.time, srcOp.time)
		mergeTimeAccum(&dstOp.foldedTime, srcOp.foldedTime)
		for ck, srcCond := range srcOp.conds {
			dstCond := dstOp.conds[ck]
			if dstCond == nil {
				dstCond = &partialCondition{
					key:          ck,
					firstFieldID: srcCond.firstFieldID,
					used:         make(map[usedKey]*partialUsed, len(srcCond.used)),
					reasons:      make(map[string]*partialReason, len(srcCond.reasons)),
				}
				dstOp.conds[ck] = dstCond
			} else if dstCond.firstFieldID == 0 && srcCond.firstFieldID != 0 {
				dstCond.firstFieldID = srcCond.firstFieldID
			}
			dstCond.segments += srcCond.segments
			dstCond.pureSkippedSegments += srcCond.pureSkippedSegments
			dstCond.skippedSegmentsAny += srcCond.skippedSegmentsAny
			dstCond.unskippedActiveSegments += srcCond.unskippedActiveSegments
			dstCond.rawScanSegments += srcCond.rawScanSegments
			addCounters(&dstCond.totals, srcCond.totals)
			mergeTimeAccum(&dstCond.evalTime, srcCond.evalTime)
			mergeTimeAccum(&dstCond.prefetchTime, srcCond.prefetchTime)
			for order, cnt := range srcCond.orders {
				if dstCond.orders == nil {
					dstCond.orders = make(map[int]int, len(srcCond.orders))
				}
				dstCond.orders[order] += cnt
			}
			for uk, srcUsed := range srcCond.used {
				dstUsed := dstCond.used[uk]
				if dstUsed == nil {
					dstUsed = &partialUsed{key: uk}
					dstCond.used[uk] = dstUsed
				}
				dstUsed.segments += srcUsed.segments
				addCounters(&dstUsed.totals, srcUsed.totals)
				mergeTimeAccum(&dstUsed.evalTime, srcUsed.evalTime)
				mergeTimeAccum(&dstUsed.prefetchTime, srcUsed.prefetchTime)
			}
			for reason, srcReason := range srcCond.reasons {
				dstReason := dstCond.reasons[reason]
				if dstReason == nil {
					dstReason = &partialReason{reason: reason}
					dstCond.reasons[reason] = dstReason
				}
				dstReason.segments += srcReason.segments
				addCounters(&dstReason.totals, srcReason.totals)
				mergeTimeAccum(&dstReason.evalTime, srcReason.evalTime)
				mergeTimeAccum(&dstReason.prefetchTime, srcReason.prefetchTime)
			}
		}
	}
}

func (p *partialScope) finalize(scopedLeaves map[int64]LeafInfo) (int, int64, []OperatorSummary) {
	keys := make([]opKey, 0, len(p.ops))
	for k := range p.ops {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].step != keys[j].step {
			return keys[i].step < keys[j].step
		}
		return keys[i].name < keys[j].name
	})

	ops := make([]OperatorSummary, 0, len(keys))
	for _, k := range keys {
		op := p.ops[k]
		var outConds []Condition
		if len(op.conds) > 0 {
			outConds = make([]Condition, 0, len(op.conds))
			for _, pc := range op.conds {
				outConds = append(outConds, pc.toCondition(scopedLeaves))
			}
			sort.Slice(outConds, func(i, j int) bool { return lessCondition(outConds[i], outConds[j]) })
		}
		ops = append(ops, OperatorSummary{
			Operator:       k.name,
			Step:           k.step,
			Domain:         op.domain,
			Segments:       op.segments,
			RowsIn:         op.rowsIn,
			RowsOut:        op.rowsOut,
			TimeUs:         op.time.primaryStats(),
			CachedSegments: op.cachedSegments,
			Conditions:     outConds,
			foldedTimeUs:   op.foldedTime.primaryStats(),
		})
	}

	ops = backfillOperatorConditions(ops, p.segments, p.rowsTotal, scopedLeaves)
	ops = foldInternalEmptyFilterIntoMvcc(ops, p.disallowEmptyFilterFold)
	return p.segments, p.rowsTotal, ops
}

func encodeEvalOrders(orders map[int]int) string {
	if len(orders) == 0 {
		return ""
	}
	keys := make([]int, 0, len(orders))
	for order := range orders {
		keys = append(keys, order)
	}
	sort.Ints(keys)
	var b strings.Builder
	for i, order := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%d:%d", order, orders[order])
	}
	return b.String()
}

func decodeEvalOrders(raw string) map[int]int {
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make(map[int]int, len(parts))
	for _, part := range parts {
		idx := strings.IndexByte(part, ':')
		if idx <= 0 {
			continue
		}
		order, err1 := strconv.Atoi(part[:idx])
		cnt, err2 := strconv.Atoi(part[idx+1:])
		if err1 == nil && err2 == nil && order > 0 && cnt > 0 {
			out[order] += cnt
		}
	}
	return out
}

func axisCountersToMap(dst map[string]int64, c axisCounters) {
	if c.idxScan != 0 {
		dst["idx_scan"] = c.idxScan
	}
	if c.rawScan != 0 {
		dst["raw_scan"] = c.rawScan
	}
	dst["rows_in"] = c.rowsIn
	dst["rows_out"] = c.rowsOut
	if c.elementHits != 0 {
		dst["element_hits"] = c.elementHits
	}
}

func (p *partialScope) toProto(name string) *internalpb.ExecutionExplain {
	keys := make([]opKey, 0, len(p.ops))
	for k := range p.ops {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].step != keys[j].step {
			return keys[i].step < keys[j].step
		}
		return keys[i].name < keys[j].name
	})

	opNodes := make([]*internalpb.ExecutionExplain, 0, len(keys))
	for _, k := range keys {
		op := p.ops[k]
		condKeys := make([]conditionKey, 0, len(op.conds))
		for ck := range op.conds {
			condKeys = append(condKeys, ck)
		}
		sort.Slice(condKeys, func(i, j int) bool {
			if condKeys[i].exprID != condKeys[j].exprID {
				return condKeys[i].exprID < condKeys[j].exprID
			}
			return condKeys[i].fieldID < condKeys[j].fieldID
		})

		condNodes := make([]*internalpb.ExecutionExplain, 0, len(condKeys))
		for _, ck := range condKeys {
			pc := op.conds[ck]
			usedSlice := make([]*partialUsed, 0, len(pc.used))
			for _, u := range pc.used {
				usedSlice = append(usedSlice, u)
			}
			sort.Slice(usedSlice, func(i, j int) bool { return usedSlice[i].key.less(usedSlice[j].key) })

			reasonKeys := make([]string, 0, len(pc.reasons))
			for r := range pc.reasons {
				reasonKeys = append(reasonKeys, r)
			}
			sort.Strings(reasonKeys)

			fgNodes := make([]*internalpb.ExecutionExplain, 0, len(usedSlice)+len(reasonKeys))
			for _, u := range usedSlice {
				uAttrs := map[string]string{
					"bucket":   "used",
					"index_id": strconv.FormatInt(u.key.indexID, 10),
				}
				if u.key.family != "" {
					uAttrs["family"] = u.key.family
				}
				if u.key.path != "" {
					uAttrs["path"] = u.key.path
				}
				uCtrs := map[string]int64{"segments": int64(u.segments)}
				axisCountersToMap(uCtrs, u.totals)
				writeTimeAccum(uCtrs, "eval", u.evalTime)
				writeTimeAccum(uCtrs, "prefetch", u.prefetchTime)
				fgNodes = append(fgNodes, &internalpb.ExecutionExplain{
					Kind:       KindFilterGroup,
					Version:    1,
					Attributes: uAttrs,
					Counters:   uCtrs,
				})
			}
			for _, r := range reasonKeys {
				pr := pc.reasons[r]
				rAttrs := map[string]string{
					"bucket":         "reason",
					"decline_reason": pr.reason,
				}
				rCtrs := map[string]int64{"segments": int64(pr.segments)}
				axisCountersToMap(rCtrs, pr.totals)
				writeTimeAccum(rCtrs, "eval", pr.evalTime)
				writeTimeAccum(rCtrs, "prefetch", pr.prefetchTime)
				fgNodes = append(fgNodes, &internalpb.ExecutionExplain{
					Kind:       KindFilterGroup,
					Version:    1,
					Attributes: rAttrs,
					Counters:   rCtrs,
				})
			}

			cAttrs := map[string]string{
				"expr_id": strconv.FormatInt(pc.key.exprID, 10),
			}
			if pc.key.fieldID != 0 {
				cAttrs["cond_field_id"] = strconv.FormatInt(pc.key.fieldID, 10)
			}
			if pc.firstFieldID != 0 {
				cAttrs["first_field_id"] = strconv.FormatInt(pc.firstFieldID, 10)
			}
			if ordersStr := encodeEvalOrders(pc.orders); ordersStr != "" {
				cAttrs["eval_orders"] = ordersStr
			}
			cCtrs := map[string]int64{
				"segments": int64(pc.segments),
			}
			if pc.pureSkippedSegments > 0 {
				cCtrs["skipped_segments"] = int64(pc.pureSkippedSegments)
			}
			if pc.skippedSegmentsAny > 0 {
				cCtrs["skipped_segments_any"] = int64(pc.skippedSegmentsAny)
			}
			if pc.unskippedActiveSegments > 0 {
				cCtrs["unskipped_active_segments"] = int64(pc.unskippedActiveSegments)
			}
			if pc.rawScanSegments > 0 {
				cCtrs["raw_scan_segments"] = int64(pc.rawScanSegments)
			}
			axisCountersToMap(cCtrs, pc.totals)
			writeTimeAccum(cCtrs, "eval", pc.evalTime)
			writeTimeAccum(cCtrs, "prefetch", pc.prefetchTime)

			condNodes = append(condNodes, &internalpb.ExecutionExplain{
				Kind:       KindCondition,
				Version:    1,
				Attributes: cAttrs,
				Counters:   cCtrs,
				Children:   fgNodes,
			})
		}

		opAttrs := map[string]string{
			"step": strconv.Itoa(k.step),
			"name": k.name,
		}
		if op.domain != "" {
			opAttrs["domain"] = op.domain
		}
		opCtrs := map[string]int64{
			"segments": int64(op.segments),
			"rows_in":  op.rowsIn,
			"rows_out": op.rowsOut,
		}
		if op.cachedSegments > 0 {
			opCtrs["cached_segments"] = int64(op.cachedSegments)
		}
		writeTimeAccum(opCtrs, "time", op.time)
		writeTimeAccum(opCtrs, "folded_time", op.foldedTime)
		opNodes = append(opNodes, &internalpb.ExecutionExplain{
			Kind:       KindOperator,
			Name:       k.name,
			Version:    1,
			Attributes: opAttrs,
			Counters:   opCtrs,
			Children:   condNodes,
		})
	}

	rootCtrs := map[string]int64{
		"segments":   int64(p.segments),
		"rows_total": p.rowsTotal,
	}
	if p.disallowEmptyFilterFold {
		rootCtrs["disallow_empty_filter_fold"] = 1
	}
	return &internalpb.ExecutionExplain{
		Kind:     KindSegmentGroup,
		Name:     name,
		Version:  1,
		Counters: rootCtrs,
		Children: opNodes,
	}
}

func partialScopeFromGroup(group *internalpb.ExecutionExplain) *partialScope {
	if group == nil || group.GetKind() != KindSegmentGroup {
		return nil
	}
	out := newPartialScope()
	gCtrs := group.GetCounters()
	out.segments = int(gCtrs["segments"])
	out.rowsTotal = gCtrs["rows_total"]
	out.disallowEmptyFilterFold = gCtrs["disallow_empty_filter_fold"] > 0

	var hasFilterOp bool
	for _, opNode := range group.GetChildren() {
		if opNode == nil || opNode.GetKind() != KindOperator {
			continue
		}
		attrs := opNode.GetAttributes()
		name := attrs["name"]
		if name == "" {
			name = opNode.GetName()
		}
		if int(int64Attribute(attrs, "step")) == 0 && name == "filter" {
			hasFilterOp = true
			break
		}
	}

	for _, opNode := range group.GetChildren() {
		if opNode == nil || opNode.GetKind() != KindOperator {
			continue
		}
		attrs := opNode.GetAttributes()
		name := attrs["name"]
		if name == "" {
			name = opNode.GetName()
		}
		if name == "" {
			continue
		}
		k := opKey{step: int(int64Attribute(attrs, "step")), name: name}
		opCtrs := opNode.GetCounters()
		opSegs := int(opCtrs["segments"])
		po := out.ops[k]
		if po == nil {
			po = &partialOperator{
				key:    k,
				domain: attrs["domain"],
				conds:  make(map[conditionKey]*partialCondition, len(opNode.GetChildren())),
			}
			out.ops[k] = po
		} else if po.domain == "" && attrs["domain"] != "" {
			po.domain = attrs["domain"]
		}
		po.segments += opSegs
		po.cachedSegments += int(opCtrs["cached_segments"])
		po.rowsIn += opCtrs["rows_in"]
		po.rowsOut += opCtrs["rows_out"]
		mergeTimeAccum(&po.time, readTimeAccum(opCtrs, "time", opSegs))
		if k.step == 1 && k.name == "mvcc" && !out.disallowEmptyFilterFold && hasFilterOp {
			mergeTimeAccum(&po.foldedTime, readTimeAccum(opCtrs, "folded_time", opSegs))
		}

		for _, condNode := range opNode.GetChildren() {
			if condNode == nil || condNode.GetKind() != KindCondition {
				continue
			}
			cAttrs := condNode.GetAttributes()
			cCtrs := condNode.GetCounters()
			ck := conditionKey{
				exprID:  int64Attribute(cAttrs, "expr_id"),
				fieldID: int64Attribute(cAttrs, "cond_field_id"),
			}
			condSegs := int(cCtrs["segments"])
			pureSkipped := int(cCtrs["skipped_segments"])
			evalDenom := max(0, condSegs-pureSkipped)
			pc := &partialCondition{
				key:                     ck,
				firstFieldID:            int64Attribute(cAttrs, "first_field_id"),
				segments:                condSegs,
				pureSkippedSegments:     pureSkipped,
				skippedSegmentsAny:      int(cCtrs["skipped_segments_any"]),
				unskippedActiveSegments: int(cCtrs["unskipped_active_segments"]),
				rawScanSegments:         int(cCtrs["raw_scan_segments"]),
				orders:                  decodeEvalOrders(cAttrs["eval_orders"]),
				totals:                  countersOf(condNode, false),
				evalTime:                readTimeAccum(cCtrs, "eval", evalDenom),
				prefetchTime:            readTimeAccum(cCtrs, "prefetch", condSegs),
				used:                    make(map[usedKey]*partialUsed, 2),
				reasons:                 make(map[string]*partialReason, 2),
			}
			for _, fgNode := range condNode.GetChildren() {
				if fgNode == nil || fgNode.GetKind() != KindFilterGroup {
					continue
				}
				fgAttrs := fgNode.GetAttributes()
				fgCtrs := fgNode.GetCounters()
				fgSegs := int(fgCtrs["segments"])
				if fgAttrs["bucket"] == "used" || (fgAttrs["bucket"] == "" && fgAttrs["index_id"] != "") {
					uk := usedKey{
						indexID: int64Attribute(fgAttrs, "index_id"),
						family:  fgAttrs["family"],
						path:    fgAttrs["path"],
					}
					pu := pc.used[uk]
					if pu == nil {
						pu = &partialUsed{key: uk}
						pc.used[uk] = pu
					}
					pu.segments += fgSegs
					addCounters(&pu.totals, countersOf(fgNode, true))
					mergeTimeAccum(&pu.evalTime, readTimeAccum(fgCtrs, "eval", fgSegs))
					mergeTimeAccum(&pu.prefetchTime, readTimeAccum(fgCtrs, "prefetch", fgSegs))
				} else {
					reason := fgAttrs["decline_reason"]
					if reason == "" {
						reason = unknownReason
					}
					pr := pc.reasons[reason]
					if pr == nil {
						pr = &partialReason{reason: reason}
						pc.reasons[reason] = pr
					}
					pr.segments += fgSegs
					addCounters(&pr.totals, countersOf(fgNode, false))
					mergeTimeAccum(&pr.evalTime, readTimeAccum(fgCtrs, "eval", fgSegs))
					mergeTimeAccum(&pr.prefetchTime, readTimeAccum(fgCtrs, "prefetch", fgSegs))
				}
			}
			if existing := po.conds[ck]; existing != nil {
				tmp := newPartialScope()
				tmp.ops[k] = &partialOperator{key: k, conds: map[conditionKey]*partialCondition{ck: pc}}
				dstTmp := newPartialScope()
				dstTmp.ops[k] = po
				dstTmp.merge(tmp)
			} else {
				po.conds[ck] = pc
			}
		}
	}
	return out
}

func backfillOperatorConditions(
	ops []OperatorSummary,
	scopeSegments int,
	scopeRowsTotal int64,
	scopedLeaves map[int64]LeafInfo,
) []OperatorSummary {
	if len(scopedLeaves) == 0 {
		return ops
	}
	present := make(map[int64]struct{}, len(scopedLeaves))
	for _, op := range ops {
		for _, cond := range op.Conditions {
			if cond.ExprID != 0 {
				present[cond.ExprID] = struct{}{}
			}
		}
	}
	missingIDs := make([]int64, 0, len(scopedLeaves))
	for exprID := range scopedLeaves {
		if exprID == 0 {
			continue
		}
		if _, ok := present[exprID]; !ok {
			missingIDs = append(missingIDs, exprID)
		}
	}
	if len(missingIDs) == 0 {
		return ops
	}
	sort.Slice(missingIDs, func(i, j int) bool { return missingIDs[i] < missingIDs[j] })

	if len(ops) > 0 && ops[0].Operator == "mvcc" && ops[0].foldedFilter != nil {
		needsDocFilter := false
		for _, exprID := range missingIDs {
			if !scopedLeaves[exprID].ElementFilter {
				needsDocFilter = true
				break
			}
		}
		if needsDocFilter {
			restoredFilter := *ops[0].foldedFilter
			ops[0].RowsIn = ops[0].origMvccRows
			ops[0].TimeUs = ops[0].origMvccTimeUs
			ops[0].foldedFilter = nil
			ops[0].origMvccRows = 0
			ops[0].origMvccTimeUs = nil
			ops = append([]OperatorSummary{restoredFilter}, ops...)
			for i := range ops {
				ops[i].Step = i
			}
		}
	}

	insertOpAt := func(at int, created OperatorSummary) int {
		ops = append(ops, OperatorSummary{})
		copy(ops[at+1:], ops[at:])
		ops[at] = created
		for i := range ops {
			ops[i].Step = i
		}
		return at
	}

	findTarget := func(elementFilter bool) int {
		if elementFilter {
			for i := range ops {
				if ops[i].Operator == "element_filter" {
					return i
				}
			}
			for i := range ops {
				if ops[i].Operator == "iterative_element_filter" {
					return i
				}
			}
			insertAt := 0
			for insertAt < len(ops) {
				switch ops[insertAt].Operator {
				case "filter", "random_sample", "mvcc":
					insertAt++
					continue
				}
				break
			}
			return insertOpAt(insertAt, OperatorSummary{
				Operator: "element_filter",
				Domain:   "element",
				Segments: scopeSegments,
				RowsIn:   scopeRowsTotal,
			})
		}
		for i := range ops {
			if ops[i].Operator == "filter" {
				return i
			}
		}
		for i := range ops {
			if ops[i].Operator == "iterative_filter" {
				return i
			}
		}
		return insertOpAt(0, OperatorSummary{
			Operator: "filter",
			Segments: scopeSegments,
			RowsIn:   scopeRowsTotal,
		})
	}

	for _, exprID := range missingIDs {
		info := scopedLeaves[exprID]
		idx := findTarget(info.ElementFilter)
		condText, typ, fieldID, fieldName, unresolved := leafOf(scopedLeaves, exprID, 0)
		ops[idx].Conditions = append(ops[idx].Conditions, Condition{
			ExprID:              exprID,
			Condition:           condText,
			Type:                typ,
			FieldID:             fieldID,
			Field:               fieldName,
			Used:                []IndexUse{},
			Reasons:             []ReasonCount{},
			unresolvedCondition: unresolved,
		})
	}
	for i := range ops {
		if len(ops[i].Conditions) > 1 {
			sort.Slice(ops[i].Conditions, func(a, b int) bool {
				return lessCondition(ops[i].Conditions[a], ops[i].Conditions[b])
			})
		}
	}
	return ops
}

func sameSegmentSet(a, b map[string]struct{}) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if _, ok := b[k]; !ok {
			return false
		}
	}
	return true
}

// foldInternalEmptyFilterIntoMvcc hides an internal step-0 filter operator
// injected solely for Entity TTL, Namespace, or an unfiltered Retrieve plan
// when it contains zero user filter conditions, is uncached, covers the same
// segments as mvcc, and immediately precedes mvcc.
func foldInternalEmptyFilterIntoMvcc(ops []OperatorSummary, disallowFold bool) []OperatorSummary {
	if disallowFold || len(ops) < 2 {
		return ops
	}
	if ops[0].Step != 0 || ops[0].Operator != "filter" || len(ops[0].Conditions) > 0 || ops[0].CachedSegments > 0 || ops[1].Step != 1 || ops[1].Operator != "mvcc" {
		return ops
	}
	savedFilter := ops[0]
	ops[1].foldedFilter = &savedFilter
	ops[1].origMvccRows = ops[1].RowsIn
	ops[1].origMvccTimeUs = ops[1].TimeUs
	ops[1].RowsIn = ops[0].RowsIn
	if ops[0].Segments > ops[1].Segments {
		ops[1].Segments = ops[0].Segments
	}
	ops[1].TimeUs = ops[1].foldedTimeUs
	folded := ops[1:]
	for i := range folded {
		folded[i].Step = i
	}
	return folded
}

func populateSummaryConditionTotals(summary *Summary) {
	if summary == nil {
		return
	}
	summary.Conditions = []Condition{}
	summary.RowsIn = 0
	summary.RowsOut = 0
	for _, op := range summary.Operators {
		for _, cond := range op.Conditions {
			summary.Conditions = append(summary.Conditions, cond)
			summary.RowsIn += cond.RowsIn
			summary.RowsOut += cond.RowsOut
		}
	}
	for _, sub := range summary.SubSearches {
		for _, op := range sub.Operators {
			for _, cond := range op.Conditions {
				summary.Conditions = append(summary.Conditions, cond)
				summary.RowsIn += cond.RowsIn
				summary.RowsOut += cond.RowsOut
			}
		}
	}
}

// BackfillConditions ensures every predicate leaf from leaves appears in
// summary's filter operators (and refreshes summary.Conditions).
func BackfillConditions(summary *Summary, leaves map[int64]LeafInfo) {
	if summary == nil || len(leaves) == 0 {
		return
	}
	if len(summary.SubSearches) > 0 {
		for i := range summary.SubSearches {
			sub := &summary.SubSearches[i]
			scoped := leavesForSubReq(leaves, sub.SubReqIndex)
			sub.Operators = backfillOperatorConditions(sub.Operators, sub.Segments, sub.RowsTotal, scoped)
		}
	} else {
		summary.Operators = backfillOperatorConditions(summary.Operators, summary.Segments, summary.RowsTotal, leaves)
	}
	populateSummaryConditionTotals(summary)
}

// RenderSummaryText formats a summary as indented text, two spaces per level.
func RenderSummaryText(s *Summary) string {
	if s == nil {
		return ""
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s (segments=%d, rows_total=%d)\n", KindQuery, s.Segments, s.RowsTotal)

	if len(s.SubSearches) > 0 {
		for _, sub := range s.SubSearches {
			writeIndent(&b, 1)
			if sub.AnnsField != "" {
				fmt.Fprintf(&b, "[sub %d] %s %s (segments=%d, rows_total=%d)\n",
					sub.SubReqIndex, KindSubSearch, sub.AnnsField, sub.Segments, sub.RowsTotal)
			} else {
				fmt.Fprintf(&b, "[sub %d] %s (segments=%d, rows_total=%d)\n",
					sub.SubReqIndex, KindSubSearch, sub.Segments, sub.RowsTotal)
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

func formatTimeStatsText(label string, ts *TimeStats, segments int) string {
	if ts == nil {
		return ""
	}
	if segments <= 1 {
		return fmt.Sprintf("%s=%d", label, ts.SumUs)
	}
	return fmt.Sprintf("%s: avg=%d, max=%d, min=%d, sum=%d", label, ts.AvgUs, ts.MaxUs, ts.MinUs, ts.SumUs)
}

func writeOperator(b *strings.Builder, depth int, op OperatorSummary) {
	writeIndent(b, depth)
	parts := make([]string, 0, 5)
	parts = append(parts, fmt.Sprintf("segments=%d", op.Segments))
	if op.CachedSegments > 0 {
		parts = append(parts, fmt.Sprintf("cached_segments=%d", op.CachedSegments))
	}
	if op.Domain != "" {
		parts = append(parts, "domain="+op.Domain)
	}
	parts = append(parts, fmt.Sprintf("rows: %d -> %d", op.RowsIn, op.RowsOut))
	if timePart := formatTimeStatsText("time_us", op.TimeUs, op.Segments); timePart != "" {
		parts = append(parts, timePart)
	}
	fmt.Fprintf(b, "[op %d] %s (%s)\n", op.Step, op.Operator, strings.Join(parts, ", "))

	for _, condition := range sortedConditions(op.Conditions) {
		writeCondition(b, depth+1, condition)
	}
}

func writeCondition(b *strings.Builder, depth int, condition Condition) {
	writeIndent(b, depth)
	parts := make([]string, 0, 8)
	parts = append(parts, fmt.Sprintf("segments=%d", condition.Segments))
	if condition.SkippedSegments > 0 {
		parts = append(parts, fmt.Sprintf("skipped_segments=%d", condition.SkippedSegments))
	}
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
	evalSegments := condition.Segments - condition.SkippedSegments
	if evalPart := formatTimeStatsText("eval_us", condition.EvalTimeUs, evalSegments); evalPart != "" {
		parts = append(parts, evalPart)
	}
	if pfPart := formatTimeStatsText("prefetch_us", condition.PrefetchTimeUs, condition.Segments); pfPart != "" {
		parts = append(parts, pfPart)
	}
	fmt.Fprintf(b, "expr: %s (%s)\n", condition.Condition, strings.Join(parts, ", "))

	for _, used := range sortedUsed(condition.Used) {
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
		usedParts := make([]string, 0, 6)
		usedParts = append(usedParts, fmt.Sprintf("segments=%d", used.Segments))
		usedParts = append(usedParts, fmt.Sprintf("rows: %d -> %d", used.RowsIn, used.RowsOut))
		usedParts = append(usedParts, fmt.Sprintf("idx_scan=%d", used.IdxScan))
		usedParts = append(usedParts, fmt.Sprintf("raw_scan=%d", used.RawScan))
		if used.ElementHits > 0 {
			usedParts = append(usedParts, fmt.Sprintf("element_hits=%d", used.ElementHits))
		}
		if evalPart := formatTimeStatsText("eval_us", used.EvalTimeUs, used.Segments); evalPart != "" {
			usedParts = append(usedParts, evalPart)
		}
		if pfPart := formatTimeStatsText("prefetch_us", used.PrefetchTimeUs, used.Segments); pfPart != "" {
			usedParts = append(usedParts, pfPart)
		}
		fmt.Fprintf(b, "%s: %s\n", header, strings.Join(usedParts, ", "))
	}

	for _, reason := range sortedReasons(condition.Reasons) {
		writeIndent(b, depth+1)
		reasonParts := make([]string, 0, 5)
		reasonParts = append(reasonParts, fmt.Sprintf("segments=%d", reason.Segments))
		reasonParts = append(reasonParts, fmt.Sprintf("rows: %d -> %d", reason.RowsIn, reason.RowsOut))
		reasonParts = append(reasonParts, fmt.Sprintf("idx_scan=%d", reason.IdxScan))
		reasonParts = append(reasonParts, fmt.Sprintf("raw_scan=%d", reason.RawScan))
		if evalPart := formatTimeStatsText("eval_us", reason.EvalTimeUs, reason.Segments); evalPart != "" {
			reasonParts = append(reasonParts, evalPart)
		}
		if pfPart := formatTimeStatsText("prefetch_us", reason.PrefetchTimeUs, reason.Segments); pfPart != "" {
			reasonParts = append(reasonParts, pfPart)
		}
		fmt.Fprintf(b, "raw (%s): %s\n", reason.Reason, strings.Join(reasonParts, ", "))
	}
}

func writeIndent(b *strings.Builder, depth int) {
	b.WriteString(strings.Repeat("  ", depth))
}

func buildConditionAggs(groups []*groupAgg) []*conditionAgg {
	aggregates := make([]*conditionAgg, 0, len(groups))
	index := make(map[conditionKey]int, len(groups))
	for _, group := range groups {
		condFieldID := int64(0)
		if group.key.exprID == 0 {
			condFieldID = group.fieldID
		}
		key := conditionKey{exprID: group.key.exprID, fieldID: condFieldID}
		at, ok := index[key]
		if !ok {
			at = len(aggregates)
			index[key] = at
			aggregates = append(aggregates, &conditionAgg{
				key:                     key,
				firstFieldID:            group.fieldID,
				segments:                make(map[string]struct{}, 4),
				skippedSegments:         make(map[string]struct{}, 2),
				nonSkippedSegments:      make(map[string]struct{}, 4),
				unskippedActiveSegments: make(map[string]struct{}, 4),
				orders:                  make(map[int]map[string]struct{}, 1),
				counters:                newSegCounterSet(),
				evalBySeg:               make(map[string]int64, 4),
				prefetchBySeg:           make(map[string]int64, 2),
				used:                    make(map[usedKey]*usedAgg, 2),
				reasons:                 make(map[string]*reasonAgg, 2),
			})
		}
		aggregate := aggregates[at]
		if aggregate.firstFieldID == 0 && group.fieldID != 0 {
			aggregate.firstFieldID = group.fieldID
		}
		mergeSegments(aggregate.segments, group.segments)
		mergeSegments(aggregate.skippedSegments, group.skippedSegments)
		mergeSegments(aggregate.nonSkippedSegments, group.nonSkippedSegments)
		mergeSegments(aggregate.unskippedActiveSegments, group.unskippedActiveSegments)
		mergeOrderSegments(aggregate.orders, group.orders)
		aggregate.counters.add(group.segCounters, group.key.secondary)
		mergeMaxMap(aggregate.evalBySeg, group.segEvalUs)
		mergeMaxMap(aggregate.prefetchBySeg, group.segPrefetchUs)

		// Boolean-short-circuited leaves (skipped=true) never executed on those
		// segments even if asynchronous prefetch pre-determined an index or
		// fallback path before short-circuiting.
		if group.key.skipped {
			continue
		}

		if !group.key.used {
			reason, skipReason := normalizeDeclineReason(group.key.declineReason)
			if skipReason {
				continue
			}
			entry := aggregate.reasons[reason]
			if entry == nil {
				entry = &reasonAgg{
					segments:      make(map[string]struct{}, 4),
					counters:      newSegCounterSet(),
					evalBySeg:     make(map[string]int64, 4),
					prefetchBySeg: make(map[string]int64, 2),
				}
				aggregate.reasons[reason] = entry
			}
			mergeSegments(entry.segments, group.segments)
			entry.counters.add(group.segCounters, group.key.secondary)
			mergeMaxMap(entry.evalBySeg, group.segEvalUs)
			mergeMaxMap(entry.prefetchBySeg, group.segPrefetchUs)
			continue
		}

		useKey := usedKey{indexID: group.key.indexID, family: group.key.family, path: group.key.path}
		used := aggregate.used[useKey]
		if used == nil {
			used = &usedAgg{
				key:           useKey,
				segments:      make(map[string]struct{}, 4),
				counters:      newSegCounterSet(),
				evalBySeg:     make(map[string]int64, 4),
				prefetchBySeg: make(map[string]int64, 2),
			}
			aggregate.used[useKey] = used
		}
		mergeSegments(used.segments, group.segments)
		used.counters.add(group.segCounters, group.key.secondary)
		mergeMaxMap(used.evalBySeg, group.segEvalUs)
		mergeMaxMap(used.prefetchBySeg, group.segPrefetchUs)
	}
	return aggregates
}

func (agg *conditionAgg) toPartial(timedSegs map[string]struct{}) *partialCondition {
	var orders map[int]int
	if len(agg.orders) > 0 {
		orders = make(map[int]int, len(agg.orders))
		for order, segs := range agg.orders {
			orders[order] = len(segs)
		}
	}
	used := make(map[usedKey]*partialUsed, len(agg.used))
	for uk, entry := range agg.used {
		used[uk] = &partialUsed{
			key:          uk,
			segments:     len(entry.segments),
			totals:       entry.counters.totals(),
			evalTime:     primaryTimeAccum(entry.evalBySeg, entry.segments, timedSegs),
			prefetchTime: optionalTimeAccum(entry.prefetchBySeg, entry.segments, timedSegs),
		}
	}
	rawSegments := make(map[string]struct{}, len(agg.reasons))
	reasons := make(map[string]*partialReason, len(agg.reasons))
	for reason, entry := range agg.reasons {
		if reason != notNullableReason && reason != cachedReason {
			mergeSegments(rawSegments, entry.segments)
		}
		reasons[reason] = &partialReason{
			reason:       reason,
			segments:     len(entry.segments),
			totals:       entry.counters.totals(),
			evalTime:     primaryTimeAccum(entry.evalBySeg, entry.segments, timedSegs),
			prefetchTime: optionalTimeAccum(entry.prefetchBySeg, entry.segments, timedSegs),
		}
	}
	pureSkippedCount := max(0, len(agg.segments)-len(agg.nonSkippedSegments))
	return &partialCondition{
		key:                     agg.key,
		firstFieldID:            agg.firstFieldID,
		segments:                len(agg.segments),
		pureSkippedSegments:     pureSkippedCount,
		skippedSegmentsAny:      len(agg.skippedSegments),
		unskippedActiveSegments: len(agg.unskippedActiveSegments),
		rawScanSegments:         len(rawSegments),
		orders:                  orders,
		totals:                  agg.counters.totals(),
		evalTime:                primaryTimeAccum(agg.evalBySeg, agg.nonSkippedSegments, timedSegs),
		prefetchTime:            optionalTimeAccum(agg.prefetchBySeg, agg.segments, timedSegs),
		used:                    used,
		reasons:                 reasons,
	}
}

func (pc *partialCondition) toCondition(leaves map[int64]LeafInfo) Condition {
	totals := pc.totals
	skipped := pc.skippedSegmentsAny > 0 && totals.rowsIn == 0 && pc.unskippedActiveSegments == 0
	var condEvalTime *TimeStats
	if !skipped {
		condEvalTime = pc.evalTime.primaryStats()
	}
	condition := Condition{
		ExprID:          pc.key.exprID,
		Segments:        pc.segments,
		SkippedSegments: pc.pureSkippedSegments,
		Skipped:         skipped,
		IdxScan:         totals.idxScan,
		RawScan:         totals.rawScan,
		RowsIn:          totals.rowsIn,
		RowsOut:         totals.rowsOut,
		ElementHits:     totals.elementHits,
		EvalTimeUs:      condEvalTime,
		PrefetchTimeUs:  pc.prefetchTime.optionalStats(),
		Used:            make([]IndexUse, 0, len(pc.used)),
		Reasons:         make([]ReasonCount, 0, len(pc.reasons)),
	}
	condition.Condition, condition.Type, condition.FieldID, condition.Field, condition.unresolvedCondition = leafOf(leaves, pc.key.exprID, pc.firstFieldID)

	if len(pc.orders) > 0 {
		orders := make([]int, 0, len(pc.orders))
		for order := range pc.orders {
			orders = append(orders, order)
		}
		sort.Ints(orders)
		condition.EvalOrders = make([]EvalOrderCount, 0, len(orders))
		for _, order := range orders {
			condition.EvalOrders = append(condition.EvalOrders, EvalOrderCount{
				EvalOrder: order,
				Segments:  pc.orders[order],
			})
		}
		if len(condition.EvalOrders) == 1 && condition.EvalOrders[0].Segments == condition.Segments {
			condition.EvalOrder = condition.EvalOrders[0].EvalOrder
		}
	}
	if condition.Skipped {
		return condition
	}

	used := make([]*partialUsed, 0, len(pc.used))
	for _, entry := range pc.used {
		used = append(used, entry)
	}
	sort.Slice(used, func(i, j int) bool { return used[i].key.less(used[j].key) })
	for _, entry := range used {
		usedTotals := entry.totals
		var defaultName string
		if entry.key.indexID == 0 && entry.key.family == "pk" {
			defaultName = primaryKeyName
		}
		condition.Used = append(condition.Used, IndexUse{
			Kind:           scalarIndexKind,
			IndexID:        entry.key.indexID,
			IndexName:      defaultName,
			Family:         entry.key.family,
			Segments:       entry.segments,
			IdxScan:        usedTotals.idxScan,
			RawScan:        usedTotals.rawScan,
			RowsIn:         usedTotals.rowsIn,
			RowsOut:        usedTotals.rowsOut,
			ElementHits:    usedTotals.elementHits,
			EvalTimeUs:     entry.evalTime.primaryStats(),
			PrefetchTimeUs: entry.prefetchTime.optionalStats(),
		})
	}

	for reason, entry := range pc.reasons {
		reasonTotals := entry.totals
		condition.Reasons = append(condition.Reasons, ReasonCount{
			Reason:         reason,
			Segments:       entry.segments,
			IdxScan:        reasonTotals.idxScan,
			RawScan:        reasonTotals.rawScan,
			RowsIn:         reasonTotals.rowsIn,
			RowsOut:        reasonTotals.rowsOut,
			EvalTimeUs:     entry.evalTime.primaryStats(),
			PrefetchTimeUs: entry.prefetchTime.optionalStats(),
		})
	}
	condition.RawScanSegments = pc.rawScanSegments
	sort.Slice(condition.Reasons, func(i, j int) bool {
		return condition.Reasons[i].Reason < condition.Reasons[j].Reason
	})
	return condition
}

func addCounters(dst *axisCounters, src axisCounters) {
	dst.idxScan += src.idxScan
	dst.rawScan += src.rawScan
	dst.rowsIn += src.rowsIn
	dst.rowsOut += src.rowsOut
	dst.elementHits += src.elementHits
}

func maxCounters(dst *axisCounters, src axisCounters) {
	if src.idxScan > dst.idxScan {
		dst.idxScan = src.idxScan
	}
	if src.rawScan > dst.rawScan {
		dst.rawScan = src.rawScan
	}
	if src.rowsIn > dst.rowsIn {
		dst.rowsIn = src.rowsIn
	}
	if src.rowsOut > dst.rowsOut {
		dst.rowsOut = src.rowsOut
	}
	if src.elementHits > dst.elementHits {
		dst.elementHits = src.elementHits
	}
}

func groupKeyOf(filterGroup *internalpb.ExecutionExplain) (groupKey, int64) {
	attributes := filterGroup.GetAttributes()
	fieldID := int64Attribute(attributes, "field_id")
	key := groupKey{
		fieldID:       fieldID,
		exprID:        int64Attribute(attributes, "expr_id"),
		family:        attributes["family"],
		path:          attributes["path"],
		nested:        attributes["nested"],
		declineReason: attributes["decline_reason"],
		secondary:     attributes["secondary_operand"] == "true",
		skipped:       attributes["skipped"] == "true",
	}
	if raw, ok := attributes["index_id"]; ok {
		key.used = true
		key.indexID = parseInt64(raw)
	}
	return key, fieldID
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

func int64Attribute(attributes map[string]string, key string) int64 {
	raw, ok := attributes[key]
	if !ok {
		return 0
	}
	return parseInt64(raw)
}

func parseInt64(raw string) int64 {
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0
	}
	return value
}

func mergeSegments(dst, src map[string]struct{}) {
	for segment := range src {
		dst[segment] = struct{}{}
	}
}

func mergeMaxInt64(dst map[string]int64, key string, val int64) {
	if val < 0 {
		val = 0
	}
	if cur, ok := dst[key]; !ok || val > cur {
		dst[key] = val
	}
}

func mergeMaxMap(dst, src map[string]int64) {
	for k, v := range src {
		mergeMaxInt64(dst, k, v)
	}
}

func segmentHasTimingCounters(node *internalpb.ExecutionExplain) bool {
	if node == nil {
		return false
	}
	if ctrs := node.GetCounters(); len(ctrs) > 0 {
		if _, ok := ctrs["time_us"]; ok {
			return true
		}
		if _, ok := ctrs["eval_us"]; ok {
			return true
		}
		if _, ok := ctrs["prefetch_us"]; ok {
			return true
		}
	}
	for _, ch := range node.GetChildren() {
		if segmentHasTimingCounters(ch) {
			return true
		}
	}
	return false
}

func allSegmentsTimed(denomSegs, timedSegs map[string]struct{}) bool {
	if len(denomSegs) == 0 {
		return false
	}
	for segID := range denomSegs {
		if _, ok := timedSegs[segID]; !ok {
			return false
		}
	}
	return true
}

func mergeOrderSegments(dst, src map[int]map[string]struct{}) {
	for order, segs := range src {
		entry := dst[order]
		if entry == nil {
			entry = make(map[string]struct{}, len(segs))
			dst[order] = entry
		}
		mergeSegments(entry, segs)
	}
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

func (k usedKey) less(other usedKey) bool {
	if k.indexID != other.indexID {
		return k.indexID < other.indexID
	}
	if k.family != other.family {
		return k.family < other.family
	}
	return k.path < other.path
}

func sortedConditions(conditions []Condition) []Condition {
	sorted := make([]Condition, len(conditions))
	copy(sorted, conditions)
	sort.SliceStable(sorted, func(i, j int) bool { return lessCondition(sorted[i], sorted[j]) })
	return sorted
}

func sortedUsed(used []IndexUse) []IndexUse {
	sorted := make([]IndexUse, len(used))
	copy(sorted, used)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].IndexID != sorted[j].IndexID {
			return sorted[i].IndexID < sorted[j].IndexID
		}
		if sorted[i].IndexName != sorted[j].IndexName {
			return sorted[i].IndexName < sorted[j].IndexName
		}
		return sorted[i].Family < sorted[j].Family
	})
	return sorted
}

func sortedReasons(reasons []ReasonCount) []ReasonCount {
	sorted := make([]ReasonCount, len(reasons))
	copy(sorted, reasons)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Reason < sorted[j].Reason })
	return sorted
}
