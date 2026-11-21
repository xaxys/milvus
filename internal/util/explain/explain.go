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

// Package explain builds and renders the execution explain tree: what one
// query did, grouped by decision. Segcore produces the segment and
// filter_group nodes; Go wraps them with the levels it owns (querynode,
// delegator, query) and sums their counters.
package explain

import (
	"strconv"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

// Node kinds. The tree is query -> delegator -> [sub_search ->] querynode ->
// (segment | segment_group) -> [operator -> [condition ->]] filter_group.
const (
	KindSegment      = "segment"
	KindSegmentGroup = "segment_group"
	KindOperator     = "operator"
	KindCondition    = "condition"
	KindFilterGroup  = "filter_group"
	KindQueryNode    = "querynode"
	KindDelegator    = "delegator"
	KindSubSearch    = "sub_search"
	KindQuery        = "query"
	// KindReduced is what one reduce pass produces before the next level wraps
	// it. It carries no meaning of its own: Splice replaces it with its children
	// so that a request reduced twice does not grow a level per pass.
	KindReduced = "reduced"
)

// TaggedSegment pairs a leaf execution unit (KindSegment or KindSegmentGroup)
// with the hybrid sub-request index it belongs to (-1 for non-hybrid requests)
// and its enclosing worker scope path.
type TaggedSegment struct {
	SubReqIndex int
	ScopeKey    string
	Segment     *internalpb.ExecutionExplain
}

// Segments returns the raw per-segment (KindSegment) nodes of the given trees.
// Reduced nodes are flattened on the way, and pre-aggregated segment_group
// nodes as well as wrapper levels (query / delegator / querynode / sub_search)
// are skipped.
func Segments(nodes ...*internalpb.ExecutionExplain) []*internalpb.ExecutionExplain {
	tagged := TaggedSegments(nodes...)
	segments := make([]*internalpb.ExecutionExplain, 0, len(tagged))
	for _, entry := range tagged {
		if entry.Segment != nil && entry.Segment.GetKind() == KindSegment {
			segments = append(segments, entry.Segment)
		}
	}
	return segments
}

// TaggedSegments returns all leaf execution units (KindSegment and
// KindSegmentGroup) together with the enclosing KindSubSearch index (-1 when
// outside a hybrid sub-search).
func TaggedSegments(nodes ...*internalpb.ExecutionExplain) []TaggedSegment {
	var out []TaggedSegment
	var walk func(node *internalpb.ExecutionExplain, subReq int, scopeKey string)
	walk = func(node *internalpb.ExecutionExplain, subReq int, scopeKey string) {
		if node == nil {
			return
		}
		switch node.GetKind() {
		case KindSubSearch:
			raw := node.GetAttributes()["sub_req_index"]
			if raw == "" {
				raw = node.GetName()
			}
			if parsed, err := strconv.Atoi(raw); err == nil && parsed >= 0 {
				subReq = parsed
			}
		case KindDelegator, KindQueryNode:
			if name := node.GetName(); name != "" {
				if scopeKey == "" {
					scopeKey = node.GetKind() + ":" + name
				} else {
					scopeKey = scopeKey + "/" + node.GetKind() + ":" + name
				}
			}
		case KindSegment, KindSegmentGroup:
			out = append(out, TaggedSegment{
				SubReqIndex: subReq,
				ScopeKey:    scopeKey,
				Segment:     node,
			})
			return
		}
		for _, child := range node.GetChildren() {
			walk(child, subReq, scopeKey)
		}
	}
	for _, node := range Splice(nil, nodes...) {
		walk(node, -1, "")
	}
	return out
}

// WrapSubSearch tags the explain trees of one hybrid sub-search with its
// 0-based sub-request index so Proxy reduction can attribute segments per
// sub-search.
func WrapSubSearch(subReqIndex int, children []*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	spliced := Splice(nil, children...)
	if len(spliced) == 0 {
		return nil
	}
	idxStr := strconv.Itoa(subReqIndex)
	node := Wrap(KindSubSearch, idxStr, spliced)
	node.Attributes = map[string]string{"sub_req_index": idxStr}
	return node
}

func nodeHasEvaluatedRows(node *internalpb.ExecutionExplain) bool {
	if node == nil {
		return false
	}
	if node.GetKind() == KindFilterGroup && node.GetCounters()["rows_in"] > 0 {
		return true
	}
	for _, ch := range node.GetChildren() {
		if nodeHasEvaluatedRows(ch) {
			return true
		}
	}
	return false
}

func hasOperatorChildren(segment *internalpb.ExecutionExplain) bool {
	for _, ch := range segment.GetChildren() {
		if ch != nil && ch.GetKind() == KindOperator {
			return true
		}
	}
	return false
}

func stripTimingCounters(node *internalpb.ExecutionExplain) bool {
	if node == nil {
		return false
	}
	stripped := false
	if ctrs := node.GetCounters(); len(ctrs) > 0 {
		for _, key := range []string{"time_us", "eval_us", "prefetch_us"} {
			if _, ok := ctrs[key]; ok {
				delete(ctrs, key)
				stripped = true
			}
		}
	}
	for _, ch := range node.GetChildren() {
		if stripTimingCounters(ch) {
			stripped = true
		}
	}
	return stripped
}

func fgTimingKey(attrs map[string]string) string {
	if exprID := attrs["expr_id"]; exprID != "" {
		return "e:" + exprID
	}
	return "f:" + attrs["field_id"] + ":" + attrs["secondary_operand"]
}

func accumulateFilterGroupTimings(dst []*internalpb.ExecutionExplain, src []*internalpb.ExecutionExplain) ([]*internalpb.ExecutionExplain, bool) {
	type fgTiming struct {
		evalUs       int64
		prefetchUs   int64
		evalGroups   []*internalpb.ExecutionExplain
		skippedGroup *internalpb.ExecutionExplain
	}
	s1Timings := make(map[string]*fgTiming, len(src))
	keyOrder := make([]string, 0, len(src))
	for _, fg := range src {
		if fg == nil || fg.GetKind() != KindFilterGroup {
			continue
		}
		attrs := fg.GetAttributes()
		skipped := attrs["skipped"] == "true"
		ctrs := fg.GetCounters()
		var evalUs int64
		if !skipped {
			evalUs = ctrs["eval_us"]
		}
		prefetchUs := ctrs["prefetch_us"]
		if skipped && prefetchUs <= 0 {
			continue
		}
		k := fgTimingKey(attrs)
		cur := s1Timings[k]
		if cur == nil {
			cur = &fgTiming{}
			s1Timings[k] = cur
			keyOrder = append(keyOrder, k)
		}
		if !skipped {
			cur.evalGroups = append(cur.evalGroups, fg)
			if evalUs > cur.evalUs {
				cur.evalUs = evalUs
			}
		} else if cur.skippedGroup == nil && prefetchUs > 0 {
			cur.skippedGroup = fg
		}
		if prefetchUs > cur.prefetchUs {
			cur.prefetchUs = prefetchUs
		}
	}
	if len(s1Timings) == 0 {
		return dst, false
	}
	changed := false
	out := make([]*internalpb.ExecutionExplain, 0, len(dst)+len(keyOrder))
	replacedKeys := make(map[string]struct{}, len(s1Timings))
	matchedKeys := make(map[string]struct{}, len(s1Timings))
	for _, fg := range dst {
		if fg == nil || fg.GetKind() != KindFilterGroup {
			out = append(out, fg)
			continue
		}
		attrs := fg.GetAttributes()
		k := fgTimingKey(attrs)
		t := s1Timings[k]
		if t == nil {
			out = append(out, fg)
			continue
		}
		matchedKeys[k] = struct{}{}
		if attrs["skipped"] == "true" {
			if len(t.evalGroups) > 0 {
				if _, done := replacedKeys[k]; !done {
					replacedKeys[k] = struct{}{}
					s2Prefetch := fg.GetCounters()["prefetch_us"]
					for _, srcGroup := range t.evalGroups {
						clonedGroup, _ := proto.Clone(srcGroup).(*internalpb.ExecutionExplain)
						if clonedGroup != nil && s2Prefetch > 0 {
							if clonedGroup.Counters == nil {
								clonedGroup.Counters = make(map[string]int64, 6)
							}
							clonedGroup.Counters["prefetch_us"] += s2Prefetch
						}
						out = append(out, clonedGroup)
					}
					changed = true
				}
				continue
			}
			if t.prefetchUs > 0 {
				if fg.Counters == nil {
					fg.Counters = make(map[string]int64, 5)
				}
				fg.Counters["prefetch_us"] += t.prefetchUs
				changed = true
			}
			out = append(out, fg)
			continue
		}
		if t.evalUs > 0 || t.prefetchUs > 0 {
			if fg.Counters == nil {
				fg.Counters = make(map[string]int64, 6)
			}
			if t.evalUs > 0 {
				fg.Counters["eval_us"] += t.evalUs
				changed = true
			}
			if t.prefetchUs > 0 {
				fg.Counters["prefetch_us"] += t.prefetchUs
				changed = true
			}
		}
		out = append(out, fg)
	}
	for _, k := range keyOrder {
		if _, ok := matchedKeys[k]; ok {
			continue
		}
		t := s1Timings[k]
		if len(t.evalGroups) > 0 {
			for _, srcGroup := range t.evalGroups {
				if clonedGroup, _ := proto.Clone(srcGroup).(*internalpb.ExecutionExplain); clonedGroup != nil {
					out = append(out, clonedGroup)
					changed = true
				}
			}
		} else if t.skippedGroup != nil && t.prefetchUs > 0 {
			if clonedGroup, _ := proto.Clone(t.skippedGroup).(*internalpb.ExecutionExplain); clonedGroup != nil {
				out = append(out, clonedGroup)
				changed = true
			}
		}
	}
	return out, changed
}

func mergeZeroRowStage2Timings(s1Copied []*internalpb.ExecutionExplain, s2Children []*internalpb.ExecutionExplain) []*internalpb.ExecutionExplain {
	type zeroRowTiming struct {
		evalUs      int64
		prefetchUs  int64
		sampleGroup *internalpb.ExecutionExplain
	}
	s2Timings := make(map[string]*zeroRowTiming, len(s2Children))
	keyOrder := make([]string, 0, len(s2Children))
	for _, fg := range s2Children {
		if fg == nil || fg.GetKind() != KindFilterGroup {
			continue
		}
		attrs := fg.GetAttributes()
		skipped := attrs["skipped"] == "true"
		ctrs := fg.GetCounters()
		var evalUs int64
		if !skipped {
			evalUs = ctrs["eval_us"]
		}
		prefetchUs := ctrs["prefetch_us"]
		if evalUs <= 0 && prefetchUs <= 0 {
			continue
		}
		k := fgTimingKey(attrs)
		cur := s2Timings[k]
		if cur == nil {
			cur = &zeroRowTiming{sampleGroup: fg}
			s2Timings[k] = cur
			keyOrder = append(keyOrder, k)
		}
		if evalUs > cur.evalUs {
			cur.evalUs = evalUs
		}
		if prefetchUs > cur.prefetchUs {
			cur.prefetchUs = prefetchUs
		}
	}
	if len(s2Timings) == 0 {
		return s1Copied
	}
	matchedKeys := make(map[string]struct{}, len(s2Timings))
	for _, fg := range s1Copied {
		if fg == nil || fg.GetKind() != KindFilterGroup {
			continue
		}
		attrs := fg.GetAttributes()
		k := fgTimingKey(attrs)
		t := s2Timings[k]
		if t == nil {
			continue
		}
		matchedKeys[k] = struct{}{}
		if attrs["skipped"] != "true" && t.evalUs > 0 {
			if fg.Counters == nil {
				fg.Counters = make(map[string]int64, 6)
			}
			fg.Counters["eval_us"] += t.evalUs
		}
		if t.prefetchUs > 0 {
			if fg.Counters == nil {
				fg.Counters = make(map[string]int64, 6)
			}
			fg.Counters["prefetch_us"] += t.prefetchUs
		}
	}
	for _, k := range keyOrder {
		if _, ok := matchedKeys[k]; ok {
			continue
		}
		t := s2Timings[k]
		if t.prefetchUs <= 0 || t.sampleGroup == nil {
			continue
		}
		srcAttrs := t.sampleGroup.GetAttributes()
		newAttrs := map[string]string{"skipped": "true"}
		for _, attrKey := range []string{"expr_id", "field_id", "eval_order", "secondary_operand"} {
			if v := srcAttrs[attrKey]; v != "" {
				newAttrs[attrKey] = v
			}
		}
		s1Copied = append(s1Copied, &internalpb.ExecutionExplain{
			Kind:       KindFilterGroup,
			Version:    1,
			Attributes: newAttrs,
			Counters: map[string]int64{
				"rows_in":     0,
				"idx_scan":    0,
				"raw_scan":    0,
				"rows_out":    0,
				"prefetch_us": t.prefetchUs,
			},
		})
	}
	return s1Copied
}

// MergeStage1Explain merges filter decisions recorded during Stage 1 of a
// completed two-stage search into a cloned copy of stage2. On sealed segments
// where Stage 2 reused Stage 1's cached filter bitmap, Stage 1's evaluated
// filter_group children replace the empty Stage 2 filter operator children
// (clearing the intra-request cache marker when Stage 1 itself evaluated the
// predicate from cold data rather than hitting a prior query's cache).
func MergeStage1Explain(stage2 *internalpb.ExecutionExplain, stage1 ...*internalpb.ExecutionExplain) (*internalpb.ExecutionExplain, bool) {
	if stage2 == nil || len(stage1) == 0 {
		return stage2, false
	}
	bySegment := make(map[string]*internalpb.ExecutionExplain)
	for _, node := range Segments(stage1...) {
		bySegment[node.GetName()] = node
	}
	if len(bySegment) == 0 {
		return stage2, false
	}
	cloned, ok := proto.Clone(stage2).(*internalpb.ExecutionExplain)
	if !ok || cloned == nil {
		return stage2, false
	}
	mutated := false
	for _, segment := range Segments(cloned) {
		recorded := bySegment[segment.GetName()]
		if recorded == nil {
			continue
		}
		s1Timed := segmentHasTimingCounters(recorded)
		s2Timed := segmentHasTimingCounters(segment)
		bothTimed := s1Timed && s2Timed
		mixedTiming := s1Timed != s2Timed

		if len(recorded.GetChildren()) == 0 {
			if mixedTiming && stripTimingCounters(segment) {
				mutated = true
			}
			continue
		}
		if hasOperatorChildren(segment) && hasOperatorChildren(recorded) {
			type opAlignKey struct {
				step string
				name string
			}
			stage1Ops := make(map[opAlignKey]*internalpb.ExecutionExplain, len(recorded.GetChildren()))
			for _, ch := range recorded.GetChildren() {
				if ch != nil && ch.GetKind() == KindOperator {
					stage1Ops[opAlignKey{
						step: ch.GetAttributes()["step"],
						name: ch.GetAttributes()["name"],
					}] = ch
				}
			}
			segMutated := false
			for _, ch := range segment.GetChildren() {
				if ch == nil || ch.GetKind() != KindOperator {
					continue
				}
				s1Op := stage1Ops[opAlignKey{
					step: ch.GetAttributes()["step"],
					name: ch.GetAttributes()["name"],
				}]
				if s1Op == nil {
					continue
				}
				if bothTimed {
					if s1Time, hasS1Time := s1Op.GetCounters()["time_us"]; hasS1Time {
						if ch.Counters == nil {
							ch.Counters = make(map[string]int64, 3)
						}
						if _, hasS2Time := ch.Counters["time_us"]; !hasS2Time || s1Time > 0 {
							ch.Counters["time_us"] += s1Time
							segMutated = true
						}
					}
				}
				if !nodeHasEvaluatedRows(ch) && len(s1Op.GetChildren()) > 0 {
					copied := make([]*internalpb.ExecutionExplain, len(s1Op.GetChildren()))
					for i, fg := range s1Op.GetChildren() {
						copied[i], _ = proto.Clone(fg).(*internalpb.ExecutionExplain)
					}
					if bothTimed && len(ch.GetChildren()) > 0 {
						copied = mergeZeroRowStage2Timings(copied, ch.GetChildren())
					}
					ch.Children = copied
					if ch.Attributes != nil && s1Op.GetAttributes()["cached"] != "true" {
						delete(ch.Attributes, "cached")
					}
					segMutated = true
				} else if bothTimed && nodeHasEvaluatedRows(ch) && len(s1Op.GetChildren()) > 0 {
					if mergedChildren, changed := accumulateFilterGroupTimings(ch.GetChildren(), s1Op.GetChildren()); changed {
						ch.Children = mergedChildren
						segMutated = true
					}
				}
			}
			if mixedTiming && stripTimingCounters(segment) {
				segMutated = true
			}
			if segMutated {
				if segment.GetCounters()["rows_in"] == 0 {
					if rc := recorded.GetCounters(); len(rc) > 0 {
						if segment.Counters == nil {
							segment.Counters = make(map[string]int64, len(rc))
						}
						for k, v := range rc {
							segment.Counters[k] = v
						}
					}
				}
				mutated = true
			}
			continue
		}

		// Legacy flat segment -> filter_group fallback (rolling upgrade).
		if segment.GetCounters()["rows_in"] > 0 || nodeHasEvaluatedRows(segment) {
			if mixedTiming && stripTimingCounters(segment) {
				mutated = true
			}
			continue
		}
		if len(segment.GetChildren()) > 0 && recorded.GetCounters()["rows_in"] == 0 && !nodeHasEvaluatedRows(recorded) {
			if mixedTiming && stripTimingCounters(segment) {
				mutated = true
			}
			continue
		}
		children := make([]*internalpb.ExecutionExplain, len(recorded.GetChildren()))
		for i, ch := range recorded.GetChildren() {
			children[i], _ = proto.Clone(ch).(*internalpb.ExecutionExplain)
		}
		segment.Children = children
		if rc := recorded.GetCounters(); len(rc) > 0 {
			if segment.Counters == nil {
				segment.Counters = make(map[string]int64, len(rc))
			}
			for k, v := range rc {
				segment.Counters[k] = v
			}
		}
		if mixedTiming {
			stripTimingCounters(segment)
		}
		mutated = true
	}
	if !mutated {
		return stage2, false
	}
	return cloned, true
}

// Wrap builds a node that owns children and sums their counters.
func Wrap(kind, name string, children []*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	node := &internalpb.ExecutionExplain{
		Kind:     kind,
		Name:     name,
		Version:  1,
		Counters: make(map[string]int64, 4),
	}
	for _, child := range children {
		if child == nil {
			continue
		}
		node.Children = append(node.Children, child)
		for key, value := range child.GetCounters() {
			node.Counters[key] += value
		}
	}
	return node
}

// Splice appends nodes to dst, replacing a KindReduced node with its children.
// A node of a meaningful kind is kept as a child even when it has children of
// its own.
func Splice(dst []*internalpb.ExecutionExplain, nodes ...*internalpb.ExecutionExplain) []*internalpb.ExecutionExplain {
	for _, node := range nodes {
		if node == nil {
			continue
		}
		if node.GetKind() == KindReduced {
			dst = Splice(dst, node.GetChildren()...)
			continue
		}
		dst = append(dst, node)
	}
	return dst
}

// MergeReduced joins the explains of one reduce pass into a single node that
// the next level can splice.
func MergeReduced(explains ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	return Wrap(KindReduced, "", Splice(nil, explains...))
}
