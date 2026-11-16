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
	// RowsTotal sums rows_total over those segments, each segment counted once.
	RowsTotal int64 `json:"rows_total"`
	// Conditions holds one entry per filter expr of the plan, ordered by expr_id
	// with the exprs that carry no id last.
	Conditions []Condition `json:"conditions"`
}

// Condition is one expr of the plan together with every decision its leaves
// made. RowsMatched is the rows an index returned: a raw leaf hits no index, so
// it stays 0 and this counter is not the final match count of the condition.
type Condition struct {
	ExprID          int64         `json:"expr_id"`
	Condition       string        `json:"condition"`
	Type            string        `json:"type"`
	FieldID         int64         `json:"field_id"`
	Segments        int           `json:"segments"`
	RowsMatched     int64         `json:"rows_matched"`
	RowsScanned     int64         `json:"rows_scanned"`
	ElementHits     int64         `json:"element_hits,omitempty"`
	Used            []IndexUse    `json:"used"`
	RawScanSegments int           `json:"raw_scan_segments"`
	Reasons         []ReasonCount `json:"reasons"`
}

// IndexUse is how many segments answered one condition with one index family on
// one path.
type IndexUse struct {
	Kind        string `json:"kind"`
	IndexID     int64  `json:"index_id"`
	Family      string `json:"family"`
	Segments    int    `json:"segments"`
	RowsMatched int64  `json:"rows_matched"`
	RowsScanned int64  `json:"rows_scanned"`
	ElementHits int64  `json:"element_hits,omitempty"`
}

// ReasonCount is how many segments fell back to a raw scan for one reason.
type ReasonCount struct {
	Reason   string `json:"reason"`
	Segments int    `json:"segments"`
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

// groupAgg is the cross-segment aggregate of one decision.
type groupAgg struct {
	key         groupKey
	fieldID     int64
	segments    map[string]struct{}
	rowsMatched int64
	rowsScanned int64
	elementHits int64
}

// usedAgg is the cross-segment aggregate of one index use.
type usedAgg struct {
	key         usedKey
	segments    map[string]struct{}
	rowsMatched int64
	rowsScanned int64
	elementHits int64
}

// conditionAgg is one condition while it is being assembled.
type conditionAgg struct {
	key         conditionKey
	segments    map[string]struct{}
	rowsMatched int64
	rowsScanned int64
	elementHits int64
	used        map[usedKey]*usedAgg
	reasons     map[string]map[string]struct{}
}

// Summarize folds the segment-level explain tree into the leaf view: per filter
// condition, how many segments used which index and how many fell back to a raw
// scan. segments are the KindSegment nodes of the tree, already flattened by the
// caller; leaves is the plan's expr_id -> predicate table.
//
// A segment may be reported more than once, so every count is taken over
// distinct segment IDs and the counters of a (segment, decision) pair are summed
// once. The result is never nil.
func Summarize(segments []*internalpb.ExecutionExplain, leaves map[int64]LeafInfo) *Summary {
	summary := &Summary{
		Level:      summaryLevel,
		Conditions: []Condition{},
	}

	// Rows per distinct segment: the root counter sums each segment once.
	rowsTotal := make(map[string]int64, len(segments))
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
		if _, ok := rowsTotal[segmentID]; !ok {
			rowsTotal[segmentID] = segment.GetCounters()["rows_total"]
		}
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
					key:      key,
					fieldID:  fieldID,
					segments: make(map[string]struct{}, 4),
				})
			}
			agg := groups[index]
			agg.segments[segmentID] = struct{}{}
			decision := decisionKey{segmentID: segmentID, key: key}
			if _, ok := counted[decision]; ok {
				continue
			}
			counted[decision] = struct{}{}
			counters := filterGroup.GetCounters()
			agg.rowsMatched += counters["rows_matched"]
			agg.rowsScanned += counters["rows_scanned"]
			agg.elementHits += counters["element_hits"]
		}
	}

	summary.Segments = len(rowsTotal)
	for _, rows := range rowsTotal {
		summary.RowsTotal += rows
	}
	summary.Conditions = buildConditions(groups, leaves)
	return summary
}

// RenderSummaryText formats a summary as indented text, two spaces per level:
//
//	query [rows_total=..., segments=...]
//	  expr <condition> [rows_matched=..., rows_scanned=..., segments=...] {expr_id=..., field_id=..., type=...}
//	    index <index_id> <family> [rows_matched=..., rows_scanned=..., segments=...]
//	    raw [rows_matched=..., rows_scanned=..., segments=...] {decline_reason=...}
//
// Every index a condition used comes before its raw fallbacks. Counters and
// attributes are sorted by key, so the same summary always renders the same
// text. The raw line carries what the raw leaves of the condition scanned: the
// reasons only split the segment count, so they all share that remainder.
func RenderSummaryText(s *Summary) string {
	if s == nil {
		return ""
	}

	var b strings.Builder
	writeNodeHeader(&b, 0, KindQuery)
	writePairs(&b, " [", "]", map[string]int64{
		"rows_total": s.RowsTotal,
		"segments":   int64(s.Segments),
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
	counters := map[string]int64{
		"rows_matched": condition.RowsMatched,
		"rows_scanned": condition.RowsScanned,
		"segments":     int64(condition.Segments),
	}
	if condition.ElementHits > 0 {
		counters["element_hits"] = condition.ElementHits
	}
	writePairs(b, " [", "]", counters)
	writePairs(b, " {", "}", map[string]string{
		"expr_id":  strconv.FormatInt(condition.ExprID, 10),
		"field_id": strconv.FormatInt(condition.FieldID, 10),
		"type":     condition.Type,
	})
	b.WriteByte('\n')

	for _, used := range sortedUsed(condition.Used) {
		writeNodeHeader(b, 2, "index", strconv.FormatInt(used.IndexID, 10), used.Family)
		counters := map[string]int64{
			"rows_matched": used.RowsMatched,
			"rows_scanned": used.RowsScanned,
			"segments":     int64(used.Segments),
		}
		if used.ElementHits > 0 {
			counters["element_hits"] = used.ElementHits
		}
		writePairs(b, " [", "]", counters)
		b.WriteByte('\n')
	}

	rawMatched, rawScanned := rawRemainder(condition)
	for _, reason := range sortedReasons(condition.Reasons) {
		writeNodeHeader(b, 2, "raw")
		writePairs(b, " [", "]", map[string]int64{
			"rows_matched": rawMatched,
			"rows_scanned": rawScanned,
			"segments":     int64(reason.Segments),
		})
		writePairs(b, " {", "}", map[string]string{"decline_reason": reason.Reason})
		b.WriteByte('\n')
	}
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

// rawRemainder is what the raw leaves of a condition processed: the condition
// totals minus what its indexes reported.
func rawRemainder(condition Condition) (matched, scanned int64) {
	matched, scanned = condition.RowsMatched, condition.RowsScanned
	for _, used := range condition.Used {
		matched -= used.RowsMatched
		scanned -= used.RowsScanned
	}
	if matched < 0 {
		matched = 0
	}
	if scanned < 0 {
		scanned = 0
	}
	return matched, scanned
}

func buildConditions(groups []*groupAgg, leaves map[int64]LeafInfo) []Condition {
	aggregates := make([]*conditionAgg, 0, len(groups))
	index := make(map[conditionKey]int, len(groups))
	for _, group := range groups {
		key := conditionKey{exprID: group.key.exprID, fieldID: group.fieldID}
		at, ok := index[key]
		if !ok {
			at = len(aggregates)
			index[key] = at
			aggregates = append(aggregates, &conditionAgg{
				key:      key,
				segments: make(map[string]struct{}, 4),
				used:     make(map[usedKey]*usedAgg, 2),
				reasons:  make(map[string]map[string]struct{}, 2),
			})
		}
		aggregate := aggregates[at]
		mergeSegments(aggregate.segments, group.segments)
		aggregate.rowsMatched += group.rowsMatched
		aggregate.rowsScanned += group.rowsScanned
		aggregate.elementHits += group.elementHits

		if !group.key.used {
			reason := group.key.declineReason
			if reason == "" {
				reason = unknownReason
			}
			segments := aggregate.reasons[reason]
			if segments == nil {
				segments = make(map[string]struct{}, 4)
				aggregate.reasons[reason] = segments
			}
			mergeSegments(segments, group.segments)
			continue
		}

		useKey := usedKey{indexID: group.key.indexID, family: group.key.family, path: group.key.path}
		used := aggregate.used[useKey]
		if used == nil {
			used = &usedAgg{key: useKey, segments: make(map[string]struct{}, 4)}
			aggregate.used[useKey] = used
		}
		mergeSegments(used.segments, group.segments)
		used.rowsMatched += group.rowsMatched
		used.rowsScanned += group.rowsScanned
		used.elementHits += group.elementHits
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
	condition := Condition{
		ExprID:      agg.key.exprID,
		Segments:    len(agg.segments),
		RowsMatched: agg.rowsMatched,
		RowsScanned: agg.rowsScanned,
		ElementHits: agg.elementHits,
		Used:        make([]IndexUse, 0, len(agg.used)),
		Reasons:     make([]ReasonCount, 0, len(agg.reasons)),
	}
	condition.Condition, condition.Type, condition.FieldID = leafOf(leaves, agg.key.exprID, agg.key.fieldID)

	used := make([]*usedAgg, 0, len(agg.used))
	for _, entry := range agg.used {
		used = append(used, entry)
	}
	sort.Slice(used, func(i, j int) bool { return used[i].key.less(used[j].key) })
	for _, entry := range used {
		condition.Used = append(condition.Used, IndexUse{
			Kind:        scalarIndexKind,
			IndexID:     entry.key.indexID,
			Family:      entry.key.family,
			Segments:    len(entry.segments),
			RowsMatched: entry.rowsMatched,
			RowsScanned: entry.rowsScanned,
			ElementHits: entry.elementHits,
		})
	}

	// A segment may fall back for more than one reason, so the raw segments are
	// the union of the reasons instead of their sum.
	rawSegments := make(map[string]struct{}, len(agg.reasons))
	for reason, segments := range agg.reasons {
		mergeSegments(rawSegments, segments)
		condition.Reasons = append(condition.Reasons, ReasonCount{
			Reason:   reason,
			Segments: len(segments),
		})
	}
	condition.RawScanSegments = len(rawSegments)
	sort.Slice(condition.Reasons, func(i, j int) bool {
		return condition.Reasons[i].Reason < condition.Reasons[j].Reason
	})
	return condition
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
