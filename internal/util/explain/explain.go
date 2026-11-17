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

// Package explain merges strongly-typed execution explain payloads across
// segments, workers, shards, and hybrid sub-searches, and renders the
// client-facing text and JSON views on the Proxy.
package explain

import (
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

const (
	KindQuery     = "query"
	KindSubSearch = "sub_search"
)

// OperatorKindName returns the canonical snake_case operator label for kind.
func OperatorKindName(kind internalpb.ExplainOperatorKind) string {
	switch kind {
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER:
		return "filter"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_RANDOM_SAMPLE:
		return "random_sample"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC:
		return "mvcc"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_ELEMENT_FILTER:
		return "element_filter"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH:
		return "vector_search"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_ELEMENT_FILTER:
		return "iterative_element_filter"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_FILTER:
		return "iterative_filter"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_GROUP_BY:
		return "group_by"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_RESCORE:
		return "rescore"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_PROJECT:
		return "project"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_AGGREGATION:
		return "aggregation"
	case internalpb.ExplainOperatorKind_EXPLAIN_OP_ORDER_BY:
		return "order_by"
	default:
		return "unknown"
	}
}

type condMergeKey struct {
	exprID  int64
	fieldID int64
}

func condKeyFor(c *internalpb.ExplainConditionStats) condMergeKey {
	if c.GetExprId() != 0 {
		return condMergeKey{exprID: c.GetExprId()}
	}
	return condMergeKey{fieldID: c.GetFieldId()}
}

func lessCondProto(a, b *internalpb.ExplainConditionStats) bool {
	aHas, bHas := a.GetExprId() != 0, b.GetExprId() != 0
	if aHas != bHas {
		return aHas
	}
	if a.GetExprId() != b.GetExprId() {
		return a.GetExprId() < b.GetExprId()
	}
	return a.GetFieldId() < b.GetFieldId()
}

type usedMergeKey struct {
	indexID int64
	family  string
	path    string
}

func mergeConditionStats(dst, src *internalpb.ExplainConditionStats) {
	if dst.GetFieldId() == 0 && src.GetFieldId() != 0 {
		dst.FieldId = src.GetFieldId()
	}
	dst.Segments += src.GetSegments()
	dst.SkippedSegments += src.GetSkippedSegments()
	dst.RawScanSegments += src.GetRawScanSegments()
	dst.RowsIn += src.GetRowsIn()
	dst.RowsOut += src.GetRowsOut()
	dst.IdxScan += src.GetIdxScan()
	dst.RawScan += src.GetRawScan()
	dst.ElementHits += src.GetElementHits()

	if len(src.GetEvalOrders()) > 0 {
		byOrder := make(map[int32]*internalpb.ExplainEvalOrderCount, len(dst.EvalOrders)+len(src.EvalOrders))
		for _, eo := range dst.EvalOrders {
			byOrder[eo.GetEvalOrder()] = eo
		}
		for _, eo := range src.GetEvalOrders() {
			if existing := byOrder[eo.GetEvalOrder()]; existing != nil {
				existing.Segments += eo.GetSegments()
			} else {
				cloned := &internalpb.ExplainEvalOrderCount{
					EvalOrder: eo.GetEvalOrder(),
					Segments:  eo.GetSegments(),
				}
				dst.EvalOrders = append(dst.EvalOrders, cloned)
				byOrder[eo.GetEvalOrder()] = cloned
			}
		}
		sort.Slice(dst.EvalOrders, func(i, j int) bool {
			return dst.EvalOrders[i].GetEvalOrder() < dst.EvalOrders[j].GetEvalOrder()
		})
	}

	if len(src.GetUsed()) > 0 {
		byUsed := make(map[usedMergeKey]*internalpb.ExplainIndexUseStats, len(dst.Used)+len(src.Used))
		for _, u := range dst.Used {
			byUsed[usedMergeKey{indexID: u.GetIndexId(), family: u.GetFamily(), path: u.GetPath()}] = u
		}
		for _, u := range src.GetUsed() {
			k := usedMergeKey{indexID: u.GetIndexId(), family: u.GetFamily(), path: u.GetPath()}
			if existing := byUsed[k]; existing != nil {
				existing.Segments += u.GetSegments()
				existing.RowsIn += u.GetRowsIn()
				existing.RowsOut += u.GetRowsOut()
				existing.IdxScan += u.GetIdxScan()
				existing.RawScan += u.GetRawScan()
				existing.ElementHits += u.GetElementHits()
			} else {
				cloned := proto.Clone(u).(*internalpb.ExplainIndexUseStats)
				dst.Used = append(dst.Used, cloned)
				byUsed[k] = cloned
			}
		}
		sort.Slice(dst.Used, func(i, j int) bool {
			a, b := dst.Used[i], dst.Used[j]
			if a.GetIndexId() != b.GetIndexId() {
				return a.GetIndexId() < b.GetIndexId()
			}
			if a.GetFamily() != b.GetFamily() {
				return a.GetFamily() < b.GetFamily()
			}
			return a.GetPath() < b.GetPath()
		})
	}

	if len(src.GetReasons()) > 0 {
		byReason := make(map[string]*internalpb.ExplainDeclineStats, len(dst.Reasons)+len(src.Reasons))
		for _, r := range dst.Reasons {
			byReason[r.GetReason()] = r
		}
		for _, r := range src.GetReasons() {
			if existing := byReason[r.GetReason()]; existing != nil {
				existing.Segments += r.GetSegments()
				existing.RowsIn += r.GetRowsIn()
				existing.RowsOut += r.GetRowsOut()
				existing.IdxScan += r.GetIdxScan()
				existing.RawScan += r.GetRawScan()
			} else {
				cloned := proto.Clone(r).(*internalpb.ExplainDeclineStats)
				dst.Reasons = append(dst.Reasons, cloned)
				byReason[r.GetReason()] = cloned
			}
		}
		sort.Slice(dst.Reasons, func(i, j int) bool {
			return dst.Reasons[i].GetReason() < dst.Reasons[j].GetReason()
		})
	}
}

func mergeOperatorStats(dst, src *internalpb.ExplainOperatorStats) {
	if dst.GetDomain() == "" && src.GetDomain() != "" {
		dst.Domain = src.GetDomain()
	}
	dst.Segments += src.GetSegments()
	dst.CachedSegments += src.GetCachedSegments()
	dst.RowsIn += src.GetRowsIn()
	dst.RowsOut += src.GetRowsOut()

	if len(src.GetConditions()) == 0 {
		return
	}
	byCond := make(map[condMergeKey]*internalpb.ExplainConditionStats, len(dst.Conditions)+len(src.Conditions))
	for _, c := range dst.Conditions {
		byCond[condKeyFor(c)] = c
	}
	for _, c := range src.GetConditions() {
		if c == nil {
			continue
		}
		k := condKeyFor(c)
		if existing := byCond[k]; existing != nil {
			mergeConditionStats(existing, c)
		} else {
			cloned := proto.Clone(c).(*internalpb.ExplainConditionStats)
			dst.Conditions = append(dst.Conditions, cloned)
			byCond[k] = cloned
		}
	}
	sort.Slice(dst.Conditions, func(i, j int) bool {
		return lessCondProto(dst.Conditions[i], dst.Conditions[j])
	})
}

// MergeGroup merges src into dst in place, aligning physical operators by their
// global topological ExplainOperatorKind enum.
func MergeGroup(dst, src *internalpb.ExplainSegmentGroup) {
	if dst == nil || src == nil {
		return
	}
	dst.Segments += src.GetSegments()
	dst.SealedSegments += src.GetSealedSegments()
	dst.GrowingSegments += src.GetGrowingSegments()
	dst.RowsTotal += src.GetRowsTotal()

	if len(src.GetSegmentRefs()) > 0 {
		seen := make(map[int64]struct{}, len(dst.SegmentRefs)+len(src.SegmentRefs))
		for _, ref := range dst.SegmentRefs {
			seen[ref.GetSegmentId()] = struct{}{}
		}
		for _, ref := range src.GetSegmentRefs() {
			if ref == nil {
				continue
			}
			if _, dup := seen[ref.GetSegmentId()]; dup {
				continue
			}
			seen[ref.GetSegmentId()] = struct{}{}
			dst.SegmentRefs = append(dst.SegmentRefs, proto.Clone(ref).(*internalpb.ExplainSegmentRef))
		}
	}

	if len(src.GetOperators()) == 0 {
		return
	}
	byOp := make(map[internalpb.ExplainOperatorKind]*internalpb.ExplainOperatorStats, len(dst.Operators)+len(src.Operators))
	for _, op := range dst.Operators {
		if op != nil {
			byOp[op.GetKind()] = op
		}
	}
	for _, op := range src.GetOperators() {
		if op == nil || op.GetKind() == internalpb.ExplainOperatorKind_EXPLAIN_OP_UNKNOWN {
			continue
		}
		if existing := byOp[op.GetKind()]; existing != nil {
			mergeOperatorStats(existing, op)
		} else {
			cloned := proto.Clone(op).(*internalpb.ExplainOperatorStats)
			dst.Operators = append(dst.Operators, cloned)
			byOp[op.GetKind()] = cloned
		}
	}
	sort.Slice(dst.Operators, func(i, j int) bool {
		return dst.Operators[i].GetKind() < dst.Operators[j].GetKind()
	})
}

// Merge combines multiple ExecutionExplain payloads into one ExecutionExplain.
// It is associative and side-effect-free on its inputs.
func Merge(nodes ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	var out *internalpb.ExecutionExplain
	bySub := make(map[int32]*internalpb.ExplainSubSearchGroup)

	for _, node := range nodes {
		if node == nil {
			continue
		}
		if out == nil {
			out = &internalpb.ExecutionExplain{Version: 1}
		}
		if g := node.GetGroup(); g != nil {
			if out.Group == nil {
				out.Group = &internalpb.ExplainSegmentGroup{}
			}
			MergeGroup(out.Group, g)
		}
		for _, sub := range node.GetSubSearches() {
			if sub == nil || sub.GetGroup() == nil {
				continue
			}
			if existing := bySub[sub.GetSubReqIndex()]; existing != nil {
				MergeGroup(existing.Group, sub.GetGroup())
			} else {
				cloned := &internalpb.ExplainSubSearchGroup{
					SubReqIndex: sub.GetSubReqIndex(),
					Group:       &internalpb.ExplainSegmentGroup{},
				}
				MergeGroup(cloned.Group, sub.GetGroup())
				out.SubSearches = append(out.SubSearches, cloned)
				bySub[sub.GetSubReqIndex()] = cloned
			}
		}
	}
	if out != nil && len(out.SubSearches) > 1 {
		sort.Slice(out.SubSearches, func(i, j int) bool {
			return out.SubSearches[i].GetSubReqIndex() < out.SubSearches[j].GetSubReqIndex()
		})
	}
	return out
}

// WrapSubSearch aggregates nodes into a single ExplainSubSearchGroup tagged with
// its 0-based hybrid sub-request index.
func WrapSubSearch(subReqIndex int, nodes []*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	var group *internalpb.ExplainSegmentGroup
	for _, node := range nodes {
		if node == nil {
			continue
		}
		if g := node.GetGroup(); g != nil {
			if group == nil {
				group = &internalpb.ExplainSegmentGroup{}
			}
			MergeGroup(group, g)
		}
		for _, sub := range node.GetSubSearches() {
			if sub != nil && int(sub.GetSubReqIndex()) == subReqIndex && sub.GetGroup() != nil {
				if group == nil {
					group = &internalpb.ExplainSegmentGroup{}
				}
				MergeGroup(group, sub.GetGroup())
			}
		}
	}
	if group == nil {
		return nil
	}
	return &internalpb.ExecutionExplain{
		Version: 1,
		SubSearches: []*internalpb.ExplainSubSearchGroup{{
			SubReqIndex: int32(subReqIndex),
			Group:       group,
		}},
	}
}

func deriveGroupFromSubSearches(subs []*internalpb.ExplainSubSearchGroup) *internalpb.ExplainSegmentGroup {
	seen := make(map[int64]*internalpb.ExplainSegmentRef)
	var maxUSegs, maxUSealed, maxUGrowing, maxURows int64

	for _, sub := range subs {
		g := sub.GetGroup()
		if g == nil {
			continue
		}
		var subTrackedSegs, subTrackedSealed, subTrackedGrowing, subTrackedRows int64
		for _, ref := range g.GetSegmentRefs() {
			if ref == nil {
				continue
			}
			subTrackedSegs++
			if ref.GetIsGrowing() {
				subTrackedGrowing++
			} else {
				subTrackedSealed++
			}
			subTrackedRows += ref.GetRowsTotal()
			if _, exists := seen[ref.GetSegmentId()]; !exists {
				seen[ref.GetSegmentId()] = ref
			}
		}
		if u := g.GetSegments() - subTrackedSegs; u > maxUSegs {
			maxUSegs = u
		}
		if u := g.GetSealedSegments() - subTrackedSealed; u > maxUSealed {
			maxUSealed = u
		}
		if u := g.GetGrowingSegments() - subTrackedGrowing; u > maxUGrowing {
			maxUGrowing = u
		}
		if u := g.GetRowsTotal() - subTrackedRows; u > maxURows {
			maxURows = u
		}
	}

	var trackedSegs, trackedSealed, trackedGrowing, trackedRows int64
	for _, ref := range seen {
		trackedSegs++
		if ref.GetIsGrowing() {
			trackedGrowing++
		} else {
			trackedSealed++
		}
		trackedRows += ref.GetRowsTotal()
	}
	untrackedSegs := max(maxUSegs, maxUSealed+maxUGrowing)
	return &internalpb.ExplainSegmentGroup{
		Segments:        untrackedSegs + trackedSegs,
		SealedSegments:  maxUSealed + trackedSealed,
		GrowingSegments: maxUGrowing + trackedGrowing,
		RowsTotal:       maxURows + trackedRows,
	}
}

// AssembleHybridExplain merges sub-search explains, computes the deduplicated
// top-level segment union when not yet populated, and strips per-segment refs.
func AssembleHybridExplain(nodes ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	var grouped []*internalpb.ExecutionExplain
	var ungroupedSubs []*internalpb.ExecutionExplain
	for _, n := range nodes {
		if n == nil {
			continue
		}
		if n.GetGroup() == nil && len(n.GetSubSearches()) > 0 {
			ungroupedSubs = append(ungroupedSubs, n)
		} else {
			grouped = append(grouped, n)
		}
	}

	var merged *internalpb.ExecutionExplain
	if len(ungroupedSubs) > 0 {
		uMerged := Merge(ungroupedSubs...)
		if uMerged != nil && len(uMerged.GetSubSearches()) > 0 {
			uMerged.Group = deriveGroupFromSubSearches(uMerged.GetSubSearches())
		}
		if len(grouped) == 0 {
			merged = uMerged
		} else {
			all := make([]*internalpb.ExecutionExplain, 0, len(grouped)+1)
			all = append(all, grouped...)
			all = append(all, uMerged)
			merged = Merge(all...)
		}
	} else {
		merged = Merge(grouped...)
	}
	if merged == nil {
		return nil
	}
	if merged.Group != nil {
		merged.Group.SegmentRefs = nil
	}
	for _, sub := range merged.GetSubSearches() {
		if sub.GetGroup() != nil {
			sub.Group.SegmentRefs = nil
		}
	}
	return merged
}

func findOperator(g *internalpb.ExplainSegmentGroup, kind internalpb.ExplainOperatorKind) *internalpb.ExplainOperatorStats {
	if g == nil {
		return nil
	}
	for _, op := range g.GetOperators() {
		if op != nil && op.GetKind() == kind {
			return op
		}
	}
	return nil
}

// MergeStage1Explain statelessly merges Stage 1 sealed filter statistics into a
// cloned copy of Stage 2's sealed ExecutionExplain on the coordinator:
//   - When Stage 2 switched from pre-filter to iterative_filter (no EXPLAIN_OP_FILTER
//     in Stage 2), Stage 1's filter operator is inserted in topological order.
//   - When Stage 2 kept pre-filter on sealed segments, Stage 2's filter operator
//     (which reused Stage 1's cached filter bitmap) is replaced directly by Stage 1's
//     actual filter operator statistics.
func MergeStage1Explain(stage2 *internalpb.ExecutionExplain, stage1 ...*internalpb.ExecutionExplain) (*internalpb.ExecutionExplain, bool) {
	s1 := Merge(stage1...)
	if stage2 == nil || stage2.GetGroup() == nil || s1 == nil || s1.GetGroup() == nil {
		return stage2, false
	}
	cloned, ok := proto.Clone(stage2).(*internalpb.ExecutionExplain)
	if !ok || cloned == nil || cloned.GetGroup() == nil {
		return stage2, false
	}

	s1Group := s1.GetGroup()
	s2Group := cloned.GetGroup()
	if s1Group.GetSealedSegments() != s2Group.GetSealedSegments() ||
		(s1Group.GetSealedSegments() == 0 && s1Group.GetSegments() != s2Group.GetSegments()) {
		return stage2, false
	}
	mutated := false

	for _, kind := range []internalpb.ExplainOperatorKind{
		internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		internalpb.ExplainOperatorKind_EXPLAIN_OP_ELEMENT_FILTER,
	} {
		s1Op := findOperator(s1Group, kind)
		if s1Op == nil {
			continue
		}
		s2Op := findOperator(s2Group, kind)
		if s2Op == nil {
			s2Group.Operators = append(s2Group.Operators, proto.Clone(s1Op).(*internalpb.ExplainOperatorStats))
			sort.Slice(s2Group.Operators, func(i, j int) bool {
				return s2Group.Operators[i].GetKind() < s2Group.Operators[j].GetKind()
			})
			mutated = true
			continue
		}
		proto.Reset(s2Op)
		proto.Merge(s2Op, s1Op)
		mutated = true
	}

	if !mutated {
		return stage2, false
	}
	return cloned, true
}
