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

// Labels of the leaf view.
const (
	// summaryLevel is the level Summarize produces: counters per condition, with
	// the decisions folded into segment counts.
	summaryLevel = "summary"
	// scalarIndexKind labels an index use. The leaf view reports which family
	// answered a predicate, not how the engine wrapped it.
	scalarIndexKind = "scalar_index"
	// unknownReason stands for a raw fallback that did not say why it happened.
	unknownReason = "unknown"
	// constantTrueReason marks a constant-true leaf (AlwaysTrueExpr), which
	// neither uses an index nor performs a raw column scan.
	constantTrueReason = "constant_true"
	// unknownType stands for a condition the plan table does not describe.
	unknownType = "unknown"
)

// LeafInfo is the plan side of one expr: the rendered predicate and the field it
// reads, so that a summary can name a condition instead of printing its id.
type LeafInfo struct {
	// Condition is the normalized predicate text, e.g. `title contains "%降噪%"`.
	Condition string
	// Type is the predicate kind, e.g. range, equal or contains.
	Type string
	// FieldID is the field the predicate reads.
	FieldID int64
}

// Summary is the leaf view of one execution: what every filter condition did,
// counted over segments instead of listed per segment. It is the "summary"
// explain level; "detail" stays reserved.
type Summary struct {
	// Level is always "summary".
	Level string `json:"level"`
	// Segments is the number of distinct segments the query ran on.
	Segments int `json:"segments"`
	// RowsIn sums the input rows of all conditions: the rows the leaves of the
	// plan received, which is not a distinct row count of the collection.
	RowsIn int64 `json:"rows_in"`
	// RowsOut sums the rows all conditions matched. Not distinct either.
	RowsOut int64 `json:"rows_out"`
	// Conditions holds one entry per filter expr of the plan, ordered by expr_id
	// with the exprs that carry no id last.
	Conditions []Condition `json:"conditions"`
}

// EvalOrderCount counts how many segments evaluated a condition at one
// 1-based per-segment execution order.
type EvalOrderCount struct {
	EvalOrder int `json:"eval_order"`
	Segments  int `json:"segments"`
}

// Condition is one expr of the plan together with every decision its leaves
// made. The four counters are independent axes, not shares of one total:
// IdxScan is the input the index path covered, RawScan the rows read back from
// the columns, RowsIn what the leaves received and RowsOut what they matched.
type Condition struct {
	ExprID          int64            `json:"expr_id"`
	Condition       string           `json:"condition"`
	Type            string           `json:"type"`
	FieldID         int64            `json:"field_id"`
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

// IndexUse is how many segments answered one condition with one index family on
// one path. Its counters are the condition's axes restricted to the segments
// that used this index.
type IndexUse struct {
	Kind        string `json:"kind"`
	IndexID     int64  `json:"index_id"`
	IndexName   string `json:"index_name,omitempty"`
	Family      string `json:"family"`
	Segments    int    `json:"segments"`
	IdxScan     int64  `json:"idx_scan"`
	RawScan     int64  `json:"raw_scan"`
	RowsIn      int64  `json:"rows_in"`
	RowsOut     int64  `json:"rows_out"`
	ElementHits int64  `json:"element_hits,omitempty"`
}

// NameIndexUses fills in the human-readable index names the proxy resolved from
// the coordinator. A name that cannot be resolved leaves the id in place.
func NameIndexUses(summary *Summary, names map[int64]string) {
	if summary == nil || len(names) == 0 {
		return
	}
	for i := range summary.Conditions {
		for j := range summary.Conditions[i].Used {
			if name := names[summary.Conditions[i].Used[j].IndexID]; name != "" {
				summary.Conditions[i].Used[j].IndexName = name
			}
		}
	}
}

// ReasonCount is what the raw leaves of a condition did for one fallback
// reason: its own counters, not the condition remainder.
type ReasonCount struct {
	Reason   string `json:"reason"`
	Segments int    `json:"segments"`
	IdxScan  int64  `json:"idx_scan"`
	RawScan  int64  `json:"raw_scan"`
	RowsIn   int64  `json:"rows_in"`
	RowsOut  int64  `json:"rows_out"`
}

// groupKey identifies one leaf decision: equal keys are the same expr making the
// same choice, on any segment. This is the cross-segment aggregation key.
type groupKey struct {
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

// decisionKey is one decision as one segment made it. A segment that reports
// twice (replica, or one report per reduce pass) repeats its decisions, and the
// counters of a repeated decision must not be added twice.
type decisionKey struct {
	segmentID string
	key       groupKey
}

// conditionKey identifies one condition. An expr the proxy assigned an id to is
// one condition; an expr without an id (older query nodes) is kept per field so
// that two fields do not collapse into one line.
type conditionKey struct {
	exprID  int64
	fieldID int64
}

// usedKey is how one index answered a condition: the same index family on the
// same path is one entry, however the leaves nested it.
type usedKey struct {
	indexID int64
	family  string
	path    string
}

// axisCounters is one filter_group counted on the four axes of the leaf view.
// They do not add up into a single workload: an index leaf covers its input
// (idxScan) without reading a column, a raw leaf reads what it received
// (rawScan), and either way rowsOut is what the predicate matched.
type axisCounters struct {
	idxScan     int64
	rawScan     int64
	rowsIn      int64
	rowsOut     int64
	elementHits int64
}

// countersOf reads one filter_group's counters. A query node that predates the
// four axes reports rows_matched / rows_scanned only; those map onto the new
// axes before aggregation, so a rolling upgrade never shows empty counters. An
// index leaf of the old contract did not know its input rows, so rows_matched
// stands in for both the input and the rows the index covered.
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

// segCounterSet accumulates per-segment counters while keeping primary and
// secondary operand groups of multi-column leaves (CompareExpr / CallExpr)
// from double-counting rows_in, rows_out, or shared fallback reasons on the
// same segment.
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

// groupAgg is the cross-segment aggregate of one decision.
type groupAgg struct {
	key                     groupKey
	fieldID                 int64
	segments                map[string]struct{}
	skippedSegments         map[string]struct{}
	unskippedActiveSegments map[string]struct{}
	orders                  map[int]map[string]struct{}
	segCounters             map[string]axisCounters
}

// usedAgg is the cross-segment aggregate of one index use.
type usedAgg struct {
	key      usedKey
	segments map[string]struct{}
	counters segCounterSet
}

// conditionAgg is one condition while it is being assembled.
type conditionAgg struct {
	key                     conditionKey
	firstFieldID            int64
	segments                map[string]struct{}
	skippedSegments         map[string]struct{}
	unskippedActiveSegments map[string]struct{}
	orders                  map[int]map[string]struct{}
	counters                segCounterSet
	used                    map[usedKey]*usedAgg
	reasons                 map[string]*reasonAgg
}

// reasonAgg is one fallback reason of a condition: the segments it covers and
// the counters of the raw leaves that fell back for it.
type reasonAgg struct {
	segments map[string]struct{}
	counters segCounterSet
}

// Summarize folds the segment-level explain tree into the leaf view: per filter
// condition, how many rows its leaves received, how many of them an index
// covered, how many were read from the columns, how many matched, and how many
// segments fell back to a raw scan. segments are the KindSegment nodes of the
// tree, already flattened by the caller; leaves is the plan's expr_id ->
// predicate table.
//
// A segment may be reported more than once, so every count is taken over
// distinct segment IDs and the counters of a (segment, decision) pair are summed
// once. The result is never nil.
func Summarize(segments []*internalpb.ExecutionExplain, leaves map[int64]LeafInfo) *Summary {
	summary := &Summary{
		Level:      summaryLevel,
		Conditions: []Condition{},
	}

	// Distinct segments: the root counts each of them once.
	seen := make(map[string]struct{}, len(segments))
	// One entry per decision seen anywhere, in the order the tree reports them.
	groups := make([]*groupAgg, 0, len(segments))
	groupIndex := make(map[groupKey]int, len(segments))
	// The counters of a decision are taken on its first report only.
	counted := make(map[decisionKey]struct{}, len(segments))

	for _, segment := range segments {
		if segment == nil {
			continue
		}
		segmentID := segment.GetName()
		seen[segmentID] = struct{}{}
		segRowsTotal := segment.GetCounters()["rows_total"]
		for _, filterGroup := range segment.GetChildren() {
			if filterGroup == nil || filterGroup.GetKind() != KindFilterGroup {
				continue
			}
			key, fieldID := groupKeyOf(filterGroup)
			index, ok := groupIndex[key]
			if !ok {
				index = len(groups)
				groupIndex[key] = index
				groups = append(groups, &groupAgg{
					key:                     key,
					fieldID:                 fieldID,
					segments:                make(map[string]struct{}, 4),
					skippedSegments:         make(map[string]struct{}, 2),
					unskippedActiveSegments: make(map[string]struct{}, 4),
					orders:                  make(map[int]map[string]struct{}, 1),
					segCounters:             make(map[string]axisCounters, 4),
				})
			}
			agg := groups[index]
			if agg.fieldID == 0 && fieldID != 0 {
				agg.fieldID = fieldID
			}
			agg.segments[segmentID] = struct{}{}
			if key.skipped {
				agg.skippedSegments[segmentID] = struct{}{}
			} else if segRowsTotal > 0 {
				agg.unskippedActiveSegments[segmentID] = struct{}{}
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
			if _, ok := counted[decision]; ok {
				continue
			}
			counted[decision] = struct{}{}
			c := countersOf(filterGroup, key.used)
			cur := agg.segCounters[segmentID]
			addCounters(&cur, c)
			agg.segCounters[segmentID] = cur
		}
	}

	summary.Segments = len(seen)
	summary.Conditions = buildConditions(groups, leaves)
	for _, condition := range summary.Conditions {
		summary.RowsIn += condition.RowsIn
		summary.RowsOut += condition.RowsOut
	}
	return summary
}

// BackfillConditions ensures every predicate leaf from the plan appears in
// summary.Conditions even when no segment emitted a filter_group for it (for
// example, a whole-predicate expression cache hit, segments with zero active
// rows, or a pruned hybrid sub-search). Those leaves were not boolean
// short-circuited during expression evaluation, so Skipped stays false.
func BackfillConditions(summary *Summary, leaves map[int64]LeafInfo) {
	if summary == nil || len(leaves) == 0 {
		return
	}
	present := make(map[int64]struct{}, len(summary.Conditions))
	for _, cond := range summary.Conditions {
		if cond.ExprID != 0 {
			present[cond.ExprID] = struct{}{}
		}
	}
	added := false
	for exprID := range leaves {
		if exprID == 0 {
			continue
		}
		if _, ok := present[exprID]; ok {
			continue
		}
		condText, typ, fieldID := leafOf(leaves, exprID, 0)
		summary.Conditions = append(summary.Conditions, Condition{
			ExprID:    exprID,
			Condition: condText,
			Type:      typ,
			FieldID:   fieldID,
			Used:      []IndexUse{},
			Reasons:   []ReasonCount{},
		})
		added = true
	}
	if added {
		sort.Slice(summary.Conditions, func(i, j int) bool {
			return lessCondition(summary.Conditions[i], summary.Conditions[j])
		})
	}
}

// RenderSummaryText formats a summary as indented text, two spaces per level:
//
//	query [rows_in=..., rows_out=..., segments=...]
//	  expr <condition> [idx_scan=..., raw_scan=..., rows_in=..., rows_out=..., segments=...] {expr_id=..., field_id=..., type=...}
//	    index <index_id> <family> [idx_scan=..., raw_scan=..., rows_in=..., rows_out=..., segments=...]
//	    raw [idx_scan=..., raw_scan=..., rows_in=..., rows_out=..., segments=...] {decline_reason=...}
//
// Every index a condition used comes before its raw fallbacks. Counters and
// attributes are sorted by key, so the same summary always renders the same
// text. The raw lines carry what the raw leaves of the condition processed: the
// reasons only split the segment count, so they all share that remainder.
func RenderSummaryText(s *Summary) string {
	if s == nil {
		return ""
	}

	var b strings.Builder
	writeNodeHeader(&b, 0, KindQuery)
	writePairs(&b, " [", "]", map[string]int64{
		"rows_in":  s.RowsIn,
		"rows_out": s.RowsOut,
		"segments": int64(s.Segments),
	})
	b.WriteByte('\n')

	for _, condition := range sortedConditions(s.Conditions) {
		writeCondition(&b, condition)
	}
	return b.String()
}

// RenderSummaryJSON formats a summary as JSON. The shape is Summary itself, with
// used and reasons always present: an empty list, never null.
func RenderSummaryJSON(s *Summary) (string, error) {
	encoded, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func writeCondition(b *strings.Builder, condition Condition) {
	writeNodeHeader(b, 1, "expr", condition.Condition)
	writePairs(b, " [", "]", counterPairs(condition.axisCounters(), condition.Segments))
	attrs := map[string]string{
		"expr_id":  strconv.FormatInt(condition.ExprID, 10),
		"field_id": strconv.FormatInt(condition.FieldID, 10),
		"type":     condition.Type,
	}
	if orderAttr := formatEvalOrderAttr(condition); orderAttr != "" {
		attrs["eval_order"] = orderAttr
	}
	if condition.Skipped {
		attrs["skipped"] = "true"
	}
	writePairs(b, " {", "}", attrs)
	b.WriteByte('\n')

	for _, used := range sortedUsed(condition.Used) {
		name := used.IndexName
		if name == "" {
			// A primary-key lookup has no scalar index id; its family names it.
			if used.IndexID == 0 && used.Family == "pk" {
				name = "pk"
			} else {
				name = strconv.FormatInt(used.IndexID, 10)
			}
		}
		writeNodeHeader(b, 2, "index", name, used.Family)
		writePairs(b, " [", "]", counterPairs(axisCounters{
			idxScan:     used.IdxScan,
			rawScan:     used.RawScan,
			rowsIn:      used.RowsIn,
			rowsOut:     used.RowsOut,
			elementHits: used.ElementHits,
		}, used.Segments))
		b.WriteByte('\n')
	}

	for _, reason := range sortedReasons(condition.Reasons) {
		writeNodeHeader(b, 2, "raw")
		writePairs(b, " [", "]", counterPairs(axisCounters{
			idxScan: reason.IdxScan,
			rawScan: reason.RawScan,
			rowsIn:  reason.RowsIn,
			rowsOut: reason.RowsOut,
		}, reason.Segments))
		writePairs(b, " {", "}", map[string]string{"decline_reason": reason.Reason})
		b.WriteByte('\n')
	}
}

// counterPairs renders the four axes of one node under the keys the text format
// uses, plus element_hits when the node hit elements.
func counterPairs(c axisCounters, segments int) map[string]int64 {
	pairs := map[string]int64{
		"idx_scan": c.idxScan,
		"raw_scan": c.rawScan,
		"rows_in":  c.rowsIn,
		"rows_out": c.rowsOut,
		"segments": int64(segments),
	}
	if c.elementHits > 0 {
		pairs["element_hits"] = c.elementHits
	}
	return pairs
}

// writeNodeHeader writes "<indent><kind> <names...>", skipping empty names.
func writeNodeHeader(b *strings.Builder, depth int, kind string, names ...string) {
	b.WriteString(strings.Repeat("  ", depth))
	b.WriteString(kind)
	for _, name := range names {
		if name == "" {
			continue
		}
		b.WriteByte(' ')
		b.WriteString(name)
	}
}

func nonNegative(value int64) int64 {
	if value < 0 {
		return 0
	}
	return value
}

func buildConditions(groups []*groupAgg, leaves map[int64]LeafInfo) []Condition {
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
				unskippedActiveSegments: make(map[string]struct{}, 4),
				orders:                  make(map[int]map[string]struct{}, 1),
				counters:                newSegCounterSet(),
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
		mergeSegments(aggregate.unskippedActiveSegments, group.unskippedActiveSegments)
		mergeOrderSegments(aggregate.orders, group.orders)
		aggregate.counters.add(group.segCounters, group.key.secondary)

		if !group.key.used {
			if (group.key.skipped && group.key.declineReason == "") || group.key.declineReason == constantTrueReason {
				continue
			}
			reason := group.key.declineReason
			if reason == "" {
				reason = unknownReason
			}
			entry := aggregate.reasons[reason]
			if entry == nil {
				entry = &reasonAgg{
					segments: make(map[string]struct{}, 4),
					counters: newSegCounterSet(),
				}
				aggregate.reasons[reason] = entry
			}
			mergeSegments(entry.segments, group.segments)
			entry.counters.add(group.segCounters, group.key.secondary)
			continue
		}

		useKey := usedKey{indexID: group.key.indexID, family: group.key.family, path: group.key.path}
		used := aggregate.used[useKey]
		if used == nil {
			used = &usedAgg{
				key:      useKey,
				segments: make(map[string]struct{}, 4),
				counters: newSegCounterSet(),
			}
			aggregate.used[useKey] = used
		}
		mergeSegments(used.segments, group.segments)
		used.counters.add(group.segCounters, group.key.secondary)
	}

	conditions := make([]Condition, 0, len(aggregates))
	for _, aggregate := range aggregates {
		conditions = append(conditions, aggregate.condition(leaves))
	}
	sort.Slice(conditions, func(i, j int) bool { return lessCondition(conditions[i], conditions[j]) })
	return conditions
}

// condition turns one aggregate into the rendered condition.
func (agg *conditionAgg) condition(leaves map[int64]LeafInfo) Condition {
	totals := agg.counters.totals()
	condition := Condition{
		ExprID:      agg.key.exprID,
		Segments:    len(agg.segments),
		Skipped:     len(agg.skippedSegments) > 0 && totals.rowsIn == 0 && len(agg.unskippedActiveSegments) == 0,
		IdxScan:     totals.idxScan,
		RawScan:     totals.rawScan,
		RowsIn:      totals.rowsIn,
		RowsOut:     totals.rowsOut,
		ElementHits: totals.elementHits,
		Used:        make([]IndexUse, 0, len(agg.used)),
		Reasons:     make([]ReasonCount, 0, len(agg.reasons)),
	}
	condition.Condition, condition.Type, condition.FieldID = leafOf(leaves, agg.key.exprID, agg.firstFieldID)

	if len(agg.orders) > 0 {
		orders := make([]int, 0, len(agg.orders))
		for order := range agg.orders {
			orders = append(orders, order)
		}
		sort.Ints(orders)
		condition.EvalOrders = make([]EvalOrderCount, 0, len(orders))
		for _, order := range orders {
			condition.EvalOrders = append(condition.EvalOrders, EvalOrderCount{
				EvalOrder: order,
				Segments:  len(agg.orders[order]),
			})
		}
		if len(condition.EvalOrders) == 1 && condition.EvalOrders[0].Segments == condition.Segments {
			condition.EvalOrder = condition.EvalOrders[0].EvalOrder
		}
	}

	used := make([]*usedAgg, 0, len(agg.used))
	for _, entry := range agg.used {
		used = append(used, entry)
	}
	sort.Slice(used, func(i, j int) bool { return used[i].key.less(used[j].key) })
	for _, entry := range used {
		usedTotals := entry.counters.totals()
		condition.Used = append(condition.Used, IndexUse{
			Kind:        scalarIndexKind,
			IndexID:     entry.key.indexID,
			Family:      entry.key.family,
			Segments:    len(entry.segments),
			IdxScan:     usedTotals.idxScan,
			RawScan:     usedTotals.rawScan,
			RowsIn:      usedTotals.rowsIn,
			RowsOut:     usedTotals.rowsOut,
			ElementHits: usedTotals.elementHits,
		})
	}

	// A segment may fall back for more than one reason, so the raw segments are
	// the union of the reasons instead of their sum.
	rawSegments := make(map[string]struct{}, len(agg.reasons))
	for reason, entry := range agg.reasons {
		mergeSegments(rawSegments, entry.segments)
		reasonTotals := entry.counters.totals()
		condition.Reasons = append(condition.Reasons, ReasonCount{
			Reason:   reason,
			Segments: len(entry.segments),
			IdxScan:  reasonTotals.idxScan,
			RawScan:  reasonTotals.rawScan,
			RowsIn:   reasonTotals.rowsIn,
			RowsOut:  reasonTotals.rowsOut,
		})
	}
	condition.RawScanSegments = len(rawSegments)
	sort.Slice(condition.Reasons, func(i, j int) bool {
		return condition.Reasons[i].Reason < condition.Reasons[j].Reason
	})
	return condition
}

// axisCounters is the counter part of a rendered condition or index use.
func (condition Condition) axisCounters() axisCounters {
	return axisCounters{
		idxScan:     condition.IdxScan,
		rawScan:     condition.RawScan,
		rowsIn:      condition.RowsIn,
		rowsOut:     condition.RowsOut,
		elementHits: condition.ElementHits,
	}
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

// groupKeyOf reads one filter_group: its decision key and the field it reads.
func groupKeyOf(filterGroup *internalpb.ExecutionExplain) (groupKey, int64) {
	attributes := filterGroup.GetAttributes()
	key := groupKey{
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
	return key, int64Attribute(attributes, "field_id")
}

// leafOf describes one expr. The plan table is the source of the rendered
// condition; a plan it does not know, or does not name, falls back to the field
// the leaves read.
func leafOf(leaves map[int64]LeafInfo, exprID, fieldID int64) (condition, typ string, id int64) {
	leaf, ok := leaves[exprID]
	if !ok {
		return fmt.Sprintf("field_%d", fieldID), unknownType, fieldID
	}
	condition, typ, id = leaf.Condition, leaf.Type, leaf.FieldID
	if id == 0 {
		id = fieldID
	}
	if condition == "" {
		condition = fmt.Sprintf("field_%d", id)
	}
	if typ == "" {
		typ = unknownType
	}
	return condition, typ, id
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

// lessCondition orders conditions by expr_id, with the exprs that carry no id
// last: they come from query nodes that predate the ids.
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
