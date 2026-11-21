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
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

func segmentNode(id string, rowsTotal int64, children ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	return &internalpb.ExecutionExplain{
		Kind:     KindSegment,
		Name:     id,
		Counters: map[string]int64{"rows_total": rowsTotal},
		Children: children,
	}
}

func operatorNode(step int, name, domain string, cached bool, rowsIn, rowsOut int64, groups ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	attrs := map[string]string{
		"name": name,
		"step": strconv.Itoa(step),
	}
	if domain != "" {
		attrs["domain"] = domain
	}
	if cached {
		attrs["cached"] = "true"
	}
	return &internalpb.ExecutionExplain{
		Kind:       KindOperator,
		Attributes: attrs,
		Counters:   map[string]int64{"rows_in": rowsIn, "rows_out": rowsOut},
		Children:   groups,
	}
}

func indexCounters(rowsIn, rowsOut, rawScan int64) map[string]int64 {
	return map[string]int64{
		"leaves":   1,
		"rows_in":  rowsIn,
		"idx_scan": rowsIn,
		"raw_scan": rawScan,
		"rows_out": rowsOut,
	}
}

func rawCounters(rowsIn, rowsOut int64) map[string]int64 {
	return map[string]int64{
		"leaves":   1,
		"rows_in":  rowsIn,
		"idx_scan": 0,
		"raw_scan": rowsIn,
		"rows_out": rowsOut,
	}
}

func elementHits(counters map[string]int64, hits int64) map[string]int64 {
	counters["element_hits"] = hits
	return counters
}

func indexGroup(fieldID, exprID, indexID int64, family string, counters map[string]int64) *internalpb.ExecutionExplain {
	return &internalpb.ExecutionExplain{
		Kind: KindFilterGroup,
		Attributes: map[string]string{
			"field_id": strconv.FormatInt(fieldID, 10),
			"expr_id":  strconv.FormatInt(exprID, 10),
			"index_id": strconv.FormatInt(indexID, 10),
			"family":   family,
			"nested":   "false",
		},
		Counters: counters,
	}
}

func rawGroup(fieldID, exprID int64, reason string, counters map[string]int64) *internalpb.ExecutionExplain {
	attributes := map[string]string{
		"field_id": strconv.FormatInt(fieldID, 10),
		"expr_id":  strconv.FormatInt(exprID, 10),
		"nested":   "false",
	}
	if reason != "" {
		attributes["decline_reason"] = reason
	}
	return &internalpb.ExecutionExplain{
		Kind:       KindFilterGroup,
		Attributes: attributes,
		Counters:   counters,
	}
}

func legacyIndexGroup(fieldID, exprID, indexID int64, family string, rowsMatched int64) *internalpb.ExecutionExplain {
	return indexGroup(fieldID, exprID, indexID, family, map[string]int64{
		"leaves":       1,
		"rows_matched": rowsMatched,
		"rows_scanned": 0,
	})
}

func legacyRawGroup(fieldID, exprID int64, reason string, rowsScanned int64) *internalpb.ExecutionExplain {
	return rawGroup(fieldID, exprID, reason, map[string]int64{
		"leaves":       1,
		"rows_matched": 0,
		"rows_scanned": rowsScanned,
	})
}

func threeSegments() []*internalpb.ExecutionExplain {
	return []*internalpb.ExecutionExplain{
		segmentNode("1", 50000,
			operatorNode(0, "filter", "", false, 50000, 1000,
				indexGroup(101, 3, 459123456789100001, "ngram", indexCounters(50000, 30, 0)),
				indexGroup(102, 6, 459123456789100002, "bitmap", indexCounters(50000, 100, 0)),
				rawGroup(103, 4, "no_index_on_field", rawCounters(50000, 5842)),
			),
			operatorNode(1, "mvcc", "", false, 1000, 990),
			operatorNode(2, "vector_search", "", false, 990, 10),
		),
		segmentNode("2", 50000,
			operatorNode(0, "filter", "", false, 50000, 1000,
				indexGroup(101, 3, 459123456789100001, "ngram", indexCounters(50000, 30, 0)),
				indexGroup(102, 6, 459123456789100002, "bitmap", indexCounters(50000, 100, 0)),
				rawGroup(103, 4, "no_index_on_field", rawCounters(50000, 5842)),
			),
			operatorNode(1, "mvcc", "", false, 1000, 995),
			operatorNode(2, "vector_search", "", false, 995, 10),
		),
		segmentNode("3", 50000,
			operatorNode(0, "filter", "", false, 50000, 1000,
				indexGroup(101, 3, 459123456789100001, "ngram", indexCounters(50000, 30, 0)),
				rawGroup(102, 6, "index_unusable", rawCounters(50000, 37)),
				rawGroup(103, 4, "no_index_on_field", rawCounters(50000, 5842)),
			),
			operatorNode(1, "mvcc", "", false, 1000, 995),
			operatorNode(2, "vector_search", "", false, 995, 10),
		),
	}
}

func threeLeaves() map[int64]LeafInfo {
	return map[int64]LeafInfo{
		3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101, Field: "title"},
		4: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103, Field: "price"},
		6: {Condition: `brand == "Acme"`, Type: "equal", FieldID: 102, Field: "brand"},
	}
}

func TestSummarizeMixedDecisions(t *testing.T) {
	summary := Summarize(threeSegments(), threeLeaves())
	NameIndexUses(summary, map[int64]string{
		459123456789100001: "title_ngram",
		459123456789100002: "brand_bitmap",
	})

	assert.Equal(t, "summary", summary.Level)
	assert.Equal(t, 3, summary.Segments)
	assert.Equal(t, int64(150000), summary.RowsTotal)
	require.Len(t, summary.Operators, 3)

	assert.Equal(t, "filter", summary.Operators[0].Operator)
	assert.Equal(t, 0, summary.Operators[0].Step)
	assert.Equal(t, int64(150000), summary.Operators[0].RowsIn)
	assert.Equal(t, int64(3000), summary.Operators[0].RowsOut)

	assert.Equal(t, "mvcc", summary.Operators[1].Operator)
	assert.Equal(t, 1, summary.Operators[1].Step)
	assert.Equal(t, int64(3000), summary.Operators[1].RowsIn)
	assert.Equal(t, int64(2980), summary.Operators[1].RowsOut)

	assert.Equal(t, "vector_search", summary.Operators[2].Operator)
	assert.Equal(t, 2, summary.Operators[2].Step)
	assert.Equal(t, int64(2980), summary.Operators[2].RowsIn)
	assert.Equal(t, int64(30), summary.Operators[2].RowsOut)

	require.Equal(t, []Condition{
		{
			ExprID:    3,
			Condition: `title contains "%降噪%"`,
			Type:      "contains",
			FieldID:   101,
			Field:     "title",
			Segments:  3,
			IdxScan:   150000,
			RawScan:   0,
			RowsIn:    150000,
			RowsOut:   90,
			Used: []IndexUse{{
				Kind:      "scalar_index",
				IndexID:   459123456789100001,
				IndexName: "title_ngram",
				Family:    "ngram",
				Segments:  3,
				IdxScan:   150000,
				RawScan:   0,
				RowsIn:    150000,
				RowsOut:   90,
			}},
			Reasons: []ReasonCount{},
		},
		{
			ExprID:          4,
			Condition:       "price in (100, +inf)",
			Type:            "range",
			FieldID:         103,
			Field:           "price",
			Segments:        3,
			IdxScan:         0,
			RawScan:         150000,
			RowsIn:          150000,
			RowsOut:         17526,
			Used:            []IndexUse{},
			RawScanSegments: 3,
			Reasons: []ReasonCount{{
				Reason:   "no_index",
				Segments: 3,
				RawScan:  150000,
				RowsIn:   150000,
				RowsOut:  17526,
			}},
		},
		{
			ExprID:    6,
			Condition: `brand == "Acme"`,
			Type:      "equal",
			FieldID:   102,
			Field:     "brand",
			Segments:  3,
			IdxScan:   100000,
			RawScan:   50000,
			RowsIn:    150000,
			RowsOut:   237,
			Used: []IndexUse{{
				Kind:      "scalar_index",
				IndexID:   459123456789100002,
				IndexName: "brand_bitmap",
				Family:    "bitmap",
				Segments:  2,
				IdxScan:   100000,
				RawScan:   0,
				RowsIn:    100000,
				RowsOut:   200,
			}},
			RawScanSegments: 1,
			Reasons: []ReasonCount{{
				Reason:   "unusable",
				Segments: 1,
				RawScan:  50000,
				RowsIn:   50000,
				RowsOut:  37,
			}},
		},
	}, summary.Operators[0].Conditions)
}

func TestSummarizeCountsSegmentOnce(t *testing.T) {
	report := func() *internalpb.ExecutionExplain {
		return segmentNode("7", 1000,
			operatorNode(0, "filter", "", false, 1000, 12,
				indexGroup(102, 6, 11, "bitmap", indexCounters(1000, 40, 0)),
				indexGroup(102, 6, 11, "bitmap", indexCounters(1000, 40, 0)),
				rawGroup(103, 4, "no_index_on_field", rawCounters(1000, 12)),
			),
		)
	}

	leaves := map[int64]LeafInfo{
		4: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103, Field: "price"},
		6: {Condition: `brand == "Acme"`, Type: "equal", FieldID: 102, Field: "brand"},
	}
	summary := Summarize([]*internalpb.ExecutionExplain{report(), report()}, leaves)

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(1000), summary.RowsTotal)
	require.Len(t, summary.Operators, 1)
	assert.Equal(t, int64(1000), summary.Operators[0].RowsIn)
	assert.Equal(t, int64(12), summary.Operators[0].RowsOut)
	require.Len(t, summary.Operators[0].Conditions, 2)
}

func TestSummarizeLegacyFlatSegmentSynthesizesFilterOperator(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 1000,
			legacyIndexGroup(102, 6, 11, "bitmap", 40),
			legacyRawGroup(103, 4, "no_index_on_field", 900),
		),
	}, map[int64]LeafInfo{
		4: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103, Field: "price"},
		6: {Condition: `brand == "Acme"`, Type: "equal", FieldID: 102, Field: "brand"},
	})

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(1000), summary.RowsTotal)
	require.Len(t, summary.Operators, 1)
	assert.Equal(t, "filter", summary.Operators[0].Operator)
	assert.Equal(t, int64(1000), summary.Operators[0].RowsIn)
	require.Len(t, summary.Operators[0].Conditions, 2)
	assert.Equal(t, int64(900), summary.Operators[0].Conditions[0].RawScan)
	assert.Equal(t, int64(40), summary.Operators[0].Conditions[1].IdxScan)
}

func TestRenderSummaryTextAndJSONHidesInternalIDs(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 50000,
			operatorNode(0, "filter", "", true, 50000, 1000),
			operatorNode(1, "mvcc", "", false, 1000, 990),
			operatorNode(2, "vector_search", "", false, 990, 10),
		),
		segmentNode("2", 100000,
			operatorNode(0, "filter", "", false, 100000, 2000,
				withEvalOrder(indexGroup(103, 1, 501, "sort", indexCounters(100000, 2000, 0)), 1),
			),
			operatorNode(1, "mvcc", "", false, 2000, 1990),
			operatorNode(2, "vector_search", "", false, 1990, 20),
		),
	}, map[int64]LeafInfo{
		1: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103, Field: "price"},
	})
	FinalizeSummary(summary, map[int64]string{501: "price_sort"}, nil)

	expectedText := "" +
		"query (segments=2, rows_total=150000)\n" +
		"  [op 0] filter (segments=2, cached_segments=1, rows: 150000 -> 3000)\n" +
		"    expr: price in (100, +inf) (segments=1, rows: 100000 -> 2000, idx_scan=100000, raw_scan=0, eval_order=1)\n" +
		"      index price_sort (sort): segments=1, rows: 100000 -> 2000, idx_scan=100000, raw_scan=0\n" +
		"  [op 1] mvcc (segments=2, rows: 3000 -> 2980)\n" +
		"  [op 2] vector_search (segments=2, rows: 2980 -> 30)\n"
	assert.Equal(t, expectedText, RenderSummaryText(summary))

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	var rawJSON map[string]any
	require.NoError(t, json.Unmarshal([]byte(encoded), &rawJSON))
	assert.Equal(t, "summary", rawJSON["level"])
	assert.NotContains(t, encoded, "expr_id")
	assert.NotContains(t, encoded, "field_id")
	assert.NotContains(t, encoded, "index_id")
	assert.NotContains(t, encoded, "decline_reason")
	assert.Contains(t, encoded, `"rows_total":150000`)
	assert.Contains(t, encoded, `"cached_segments":1`)
	assert.Contains(t, encoded, `"field":"price"`)
	assert.Contains(t, encoded, `"index_name":"price_sort"`)
}

func TestFoldInternalEmptyFilterIntoMvcc(t *testing.T) {
	// Unfiltered search with Entity TTL injects step=0 filter (no user leaves)
	// before step=1 mvcc. Reduction folds filter into mvcc and renumbers steps.
	unfilteredTTL := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 1000,
			operatorNode(0, "filter", "", false, 1000, 900),
			operatorNode(1, "mvcc", "", false, 900, 880),
			operatorNode(2, "vector_search", "", false, 880, 10),
		),
	}, nil)

	require.Len(t, unfilteredTTL.Operators, 2)
	assert.Equal(t, "mvcc", unfilteredTTL.Operators[0].Operator)
	assert.Equal(t, 0, unfilteredTTL.Operators[0].Step)
	assert.Equal(t, int64(1000), unfilteredTTL.Operators[0].RowsIn)
	assert.Equal(t, int64(880), unfilteredTTL.Operators[0].RowsOut)
	assert.Equal(t, "vector_search", unfilteredTTL.Operators[1].Operator)
	assert.Equal(t, 1, unfilteredTTL.Operators[1].Step)

	// User-written AlwaysTrue ("1 == 1") has a planned leaf so filter is preserved.
	explicitTrue := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 1000,
			operatorNode(0, "filter", "", true, 1000, 1000),
			operatorNode(1, "mvcc", "", false, 1000, 980),
			operatorNode(2, "vector_search", "", false, 980, 10),
		),
	}, map[int64]LeafInfo{
		1: {Condition: "true", Type: "always_true"},
	})
	require.Len(t, explicitTrue.Operators, 3)
	assert.Equal(t, "filter", explicitTrue.Operators[0].Operator)
	require.Len(t, explicitTrue.Operators[0].Conditions, 1)
}

func TestSkippedLeafDoesNotPolluteUsedOrReasons(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: `brand == "none"`, Type: "equal", FieldID: 102, Field: "brand"},
		2: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103, Field: "price"},
	}

	// Segment 1 prefetched expr_id=2's index before boolean short-circuiting it.
	skippedWithIndex := withSkipped(withEvalOrder(indexGroup(103, 2, 777, "sort", map[string]int64{
		"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0,
	}), 2))
	skippedWithFallback := withSkipped(withEvalOrder(rawGroup(103, 2, "no_index_on_field", map[string]int64{
		"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0,
	}), 2))

	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, operatorNode(0, "filter", "", false, 100, 0,
			withEvalOrder(rawGroup(102, 1, "no_index_on_field", rawCounters(100, 0)), 1),
			skippedWithIndex,
		)),
		segmentNode("2", 100, operatorNode(0, "filter", "", false, 100, 0,
			withEvalOrder(rawGroup(102, 1, "no_index_on_field", rawCounters(100, 0)), 1),
			skippedWithFallback,
		)),
	}, leaves)

	require.Len(t, summary.Operators[0].Conditions, 2)
	skippedCond := summary.Operators[0].Conditions[1]
	assert.True(t, skippedCond.Skipped)
	assert.Empty(t, skippedCond.Used)
	assert.Empty(t, skippedCond.Reasons)
	assert.Equal(t, 0, skippedCond.RawScanSegments)
}

func TestNotNullableReasonExcludesRawScanSegments(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "age is not null", Type: "null", FieldID: 102, Field: "age"},
	}
	notNullGroup := rawGroup(102, 1, "schema_not_nullable", map[string]int64{
		"rows_in": 100, "idx_scan": 0, "raw_scan": 0, "rows_out": 100,
	})
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, operatorNode(0, "filter", "", false, 100, 100, notNullGroup)),
	}, leaves)

	cond := summary.Operators[0].Conditions[0]
	assert.Equal(t, 0, cond.RawScanSegments)
	require.Len(t, cond.Reasons, 1)
	assert.Equal(t, "not_nullable", cond.Reasons[0].Reason)
}

func TestSummarizeHybridSearchSubSearchesAndPrunedSubReq(t *testing.T) {
	seg1Sub0 := segmentNode("s1", 50000,
		operatorNode(0, "filter", "", false, 50000, 1000,
			indexGroup(101, 1, 9001, "sort", indexCounters(50000, 1000, 0)),
		),
		operatorNode(1, "mvcc", "", false, 1000, 1000),
		operatorNode(2, "vector_search", "", false, 1000, 10),
	)
	seg2Sub0 := segmentNode("s2", 30000,
		operatorNode(0, "filter", "", false, 30000, 500,
			indexGroup(101, 1, 9001, "sort", indexCounters(30000, 500, 0)),
		),
		operatorNode(1, "mvcc", "", false, 500, 500),
		operatorNode(2, "vector_search", "", false, 500, 10),
	)

	tagged := []TaggedSegment{
		{SubReqIndex: 0, Segment: seg1Sub0},
		{SubReqIndex: 0, Segment: seg2Sub0},
		// SubReq 1 was partition-key pruned to 0 segments!
	}
	leaves := map[int64]LeafInfo{
		1: {Condition: "price in (10, +inf)", Type: "range", FieldID: 101, Field: "price", HasSubReq: true, SubReqIndex: 0},
		2: {Condition: `tag == "vip"`, Type: "equal", FieldID: 102, Field: "tag", HasSubReq: true, SubReqIndex: 1},
	}
	subMeta := []SubSearchMeta{
		{SubReqIndex: 0, AnnsField: "dense_vec"},
		{SubReqIndex: 1, AnnsField: "sparse_vec"},
	}

	summary := SummarizeWithSubSearches(tagged, leaves, subMeta)
	FinalizeSummary(summary, map[int64]string{9001: "price_sort"}, nil)

	assert.Equal(t, 2, summary.Segments)
	assert.Equal(t, int64(80000), summary.RowsTotal)
	assert.Empty(t, summary.Operators)
	require.Len(t, summary.SubSearches, 2)

	assert.Equal(t, "dense_vec", summary.SubSearches[0].AnnsField)
	assert.Equal(t, 2, summary.SubSearches[0].Segments)
	assert.Equal(t, int64(80000), summary.SubSearches[0].RowsTotal)
	require.Len(t, summary.SubSearches[0].Operators, 3)

	assert.Equal(t, "sparse_vec", summary.SubSearches[1].AnnsField)
	assert.Equal(t, 0, summary.SubSearches[1].Segments)
	assert.Equal(t, int64(0), summary.SubSearches[1].RowsTotal)
	require.Len(t, summary.SubSearches[1].Operators, 1)
	assert.Equal(t, `tag == "vip"`, summary.SubSearches[1].Operators[0].Conditions[0].Condition)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "[sub 0] sub_search dense_vec (segments=2, rows_total=80000)")
	assert.Contains(t, text, "[sub 1] sub_search sparse_vec (segments=0, rows_total=0)")
}

func TestSummarizeMultiOperandConditionWithoutDoubleCounting(t *testing.T) {
	leaves := map[int64]LeafInfo{
		7: {Condition: "price lessthan budget", Type: "compare", FieldID: 103, Field: "price"},
	}

	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, operatorNode(0, "filter", "", false, 100, 40,
			indexGroup(103, 7, 5001, "sort", indexCounters(100, 40, 0)),
			withSecondaryOperand(rawGroup(104, 7, "no_index_on_field", rawCounters(100, 40))),
		)),
	}, leaves)

	cond := summary.Operators[0].Conditions[0]
	assert.Equal(t, int64(100), cond.IdxScan)
	assert.Equal(t, int64(100), cond.RawScan)
	assert.Equal(t, int64(100), cond.RowsIn)
	assert.Equal(t, int64(40), cond.RowsOut)
	assert.Equal(t, 1, cond.RawScanSegments)
	require.Len(t, cond.Reasons, 1)
	assert.Equal(t, "no_index", cond.Reasons[0].Reason)
}

func withEvalOrder(node *internalpb.ExecutionExplain, order int) *internalpb.ExecutionExplain {
	node.Attributes["eval_order"] = strconv.Itoa(order)
	return node
}

func withSecondaryOperand(node *internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	node.Attributes["secondary_operand"] = "true"
	return node
}

func withSkipped(node *internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	node.Attributes["skipped"] = "true"
	return node
}

func TestBackfillSeparatesDocAndElementDomainsDeterministically(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "sku_id == 1", Type: "equal", FieldID: 100, Field: "sku_id", HasSubReq: true, SubReqIndex: 0},
		2: {Condition: "price > 10", Type: "range", FieldID: 103, Field: "price", HasSubReq: true, SubReqIndex: 1, ElementFilter: false},
		3: {Condition: "items[score] > 5", Type: "range", FieldID: 201, Field: "items[score]", HasSubReq: true, SubReqIndex: 1, ElementFilter: true},
	}
	sub0 := WrapSubSearch(0, []*internalpb.ExecutionExplain{
		segmentNode("s1", 100,
			operatorNode(0, "filter", "", false, 100, 10, indexGroup(100, 1, 500, "inverted", indexCounters(100, 10, 0))),
			operatorNode(1, "vector_search", "", false, 10, 5),
		),
	})
	summary := SummarizeWithSubSearches(
		TaggedSegments(sub0),
		leaves,
		[]SubSearchMeta{{SubReqIndex: 0, AnnsField: "vec_a"}, {SubReqIndex: 1, AnnsField: "vec_b"}},
	)
	require.Len(t, summary.SubSearches, 2)
	zeroSegOps := summary.SubSearches[1].Operators
	require.Len(t, zeroSegOps, 2)
	assert.Equal(t, "filter", zeroSegOps[0].Operator)
	assert.Equal(t, 0, zeroSegOps[0].Step)
	assert.Equal(t, "", zeroSegOps[0].Domain)
	require.Len(t, zeroSegOps[0].Conditions, 1)
	assert.Equal(t, "price > 10", zeroSegOps[0].Conditions[0].Condition)

	assert.Equal(t, "element_filter", zeroSegOps[1].Operator)
	assert.Equal(t, 1, zeroSegOps[1].Step)
	assert.Equal(t, "element", zeroSegOps[1].Domain)
	require.Len(t, zeroSegOps[1].Conditions, 1)
	assert.Equal(t, "items[score] > 5", zeroSegOps[1].Conditions[0].Condition)
}

func TestFinalizeSummarySanitizesFieldPlaceholdersAndUnknownDeclineReason(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 50,
			operatorNode(0, "filter", "", false, 50, 5,
				rawGroup(199, 0, "future_internal_enum_name", rawCounters(50, 5)),
				rawGroup(0, 0, "no_index_on_field", rawCounters(50, 5)),
			),
		),
	}, nil)
	FinalizeSummary(summary, nil, nil)
	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.NotContains(t, encoded, "field_199")
	assert.NotContains(t, encoded, "field_0")
	assert.NotContains(t, encoded, "future_internal_enum_name")
	assert.Contains(t, encoded, `"reason":"unknown"`)
}

func TestCachedReasonExcludesRawScanSegments(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "price > 10", Type: "range", FieldID: 103, Field: "price"},
	}
	cachedGroup := rawGroup(103, 1, "cached", map[string]int64{
		"rows_in": 100, "idx_scan": 0, "raw_scan": 0, "rows_out": 25,
	})
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 100, operatorNode(0, "filter", "", false, 100, 25, cachedGroup)),
	}, leaves)

	cond := summary.Operators[0].Conditions[0]
	assert.Equal(t, 0, cond.RawScanSegments)
	require.Len(t, cond.Reasons, 1)
	assert.Equal(t, "cached", cond.Reasons[0].Reason)
	assert.Equal(t, int64(100), cond.Reasons[0].RowsIn)
	assert.Equal(t, int64(0), cond.Reasons[0].RawScan)
	assert.Equal(t, int64(25), cond.Reasons[0].RowsOut)
}

func TestSkippedConditionAcrossShortCircuitedAndEmptySegmentHasNoSubBuckets(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "a == 1", Type: "equal", FieldID: 101, Field: "a"},
		2: {Condition: "b == 2", Type: "equal", FieldID: 102, Field: "b"},
	}
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 100, operatorNode(0, "filter", "", false, 100, 0,
			rawGroup(101, 1, "no_index_on_field", rawCounters(100, 0)),
			withSkipped(rawGroup(102, 2, "no_index_on_field", map[string]int64{
				"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0,
			})),
		)),
		segmentNode("s2", 0, operatorNode(0, "filter", "", false, 0, 0,
			rawGroup(101, 1, "no_index_on_field", map[string]int64{
				"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0,
			}),
			rawGroup(102, 2, "no_index_on_field", map[string]int64{
				"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0,
			}),
		)),
	}, leaves)

	require.Len(t, summary.Operators[0].Conditions, 2)
	cond2 := summary.Operators[0].Conditions[1]
	assert.True(t, cond2.Skipped)
	assert.Empty(t, cond2.Used)
	assert.Empty(t, cond2.Reasons)
	assert.Equal(t, 0, cond2.RawScanSegments)
}

func TestFoldInternalEmptyFilterPreservesCachedAndUnfoldsOnBackfill(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "price > 10", Type: "range", FieldID: 103, Field: "price"},
	}
	// Whole-filter cache hit: CachedSegments=1 must not be folded into mvcc.
	cachedSummary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 100,
			operatorNode(0, "filter", "", true, 100, 30),
			operatorNode(1, "mvcc", "", false, 30, 30),
		),
	}, nil)
	require.Len(t, cachedSummary.Operators, 2)
	assert.Equal(t, "filter", cachedSummary.Operators[0].Operator)
	assert.Equal(t, 1, cachedSummary.Operators[0].CachedSegments)
	BackfillConditions(cachedSummary, leaves)
	require.Len(t, cachedSummary.Operators, 2)
	assert.Equal(t, 1, cachedSummary.Operators[0].CachedSegments)
	assert.Equal(t, int64(30), cachedSummary.Operators[0].RowsOut)
	require.Len(t, cachedSummary.Operators[0].Conditions, 1)

	// Uncached empty filter folded on Summarize(segs, nil) unfolds cleanly on BackfillConditions.
	emptySegSummary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 0,
			operatorNode(0, "filter", "", false, 0, 0),
			operatorNode(1, "mvcc", "", false, 0, 0),
		),
	}, nil)
	require.Len(t, emptySegSummary.Operators, 1)
	assert.Equal(t, "mvcc", emptySegSummary.Operators[0].Operator)
	BackfillConditions(emptySegSummary, leaves)
	require.Len(t, emptySegSummary.Operators, 2)
	assert.Equal(t, "filter", emptySegSummary.Operators[0].Operator)
	assert.Equal(t, "mvcc", emptySegSummary.Operators[1].Operator)

	// Asymmetric segment sets between filter and mvcc must not fold and corrupt mvcc.RowsIn.
	asymmetricSummary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 100,
			operatorNode(0, "filter", "", false, 100, 80),
			operatorNode(1, "mvcc", "", false, 80, 75),
		),
		segmentNode("s2", 200,
			operatorNode(1, "mvcc", "", false, 200, 190),
		),
	}, nil)
	require.Len(t, asymmetricSummary.Operators, 2)
	assert.Equal(t, int64(280), asymmetricSummary.Operators[1].RowsIn)
}

func TestFinalizeSummaryPreservesUserField1AndStructArrayNames(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "field_1 == 7", Type: "equal", FieldID: 900, Field: "field_1"},
		2: {Condition: "field_1[x] == 9", Type: "equal", FieldID: 201, Field: "field_1[x]"},
	}
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("s1", 50, operatorNode(0, "filter", "", false, 50, 5,
			rawGroup(900, 1, "no_index_on_field", rawCounters(50, 5)),
			rawGroup(201, 2, "no_index_on_field", rawCounters(50, 5)),
		)),
	}, leaves)
	FinalizeSummary(summary, nil, map[int64]string{
		900: "dynamic",
		201: "field_1[x]",
	})
	conds := summary.Operators[0].Conditions
	require.Len(t, conds, 2)
	assert.Equal(t, "field_1", conds[0].Field)
	assert.Equal(t, "field_1 == 7", conds[0].Condition)
	assert.Equal(t, "field_1[x]", conds[1].Field)
	assert.Equal(t, "field_1[x] == 9", conds[1].Condition)
}

func TestFoldSegmentsParityWithRawSegments(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "a < b", Type: "range", FieldID: 101, Field: "a"},
		2: {Condition: "tag is not null", Type: "is_not_null", FieldID: 102, Field: "tag"},
		3: {Condition: "x == 1", Type: "equal", FieldID: 103, Field: "x"},
		4: {Condition: "never_reached > 0", Type: "range", FieldID: 104, Field: "never_reached"},
	}

	secondaryRaw := func(fieldID, exprID int64, reason string, rawScan, rowsIn, rowsOut int64) *internalpb.ExecutionExplain {
		fg := rawGroup(fieldID, exprID, reason, map[string]int64{
			"idx_scan": 0, "raw_scan": rawScan, "rows_in": rowsIn, "rows_out": rowsOut,
		})
		fg.Attributes["secondary_operand"] = "true"
		return fg
	}
	withOrder := func(fg *internalpb.ExecutionExplain, order string) *internalpb.ExecutionExplain {
		fg.Attributes["eval_order"] = order
		return fg
	}
	skippedFg := func(fieldID, exprID int64) *internalpb.ExecutionExplain {
		fg := indexGroup(fieldID, exprID, 701, "sort", map[string]int64{
			"idx_scan": 0, "raw_scan": 0, "rows_in": 0, "rows_out": 0,
		})
		fg.Attributes["skipped"] = "true"
		return fg
	}

	qn1Segs := []*internalpb.ExecutionExplain{
		// Multi-operand expression: primary hits index, secondary scans raw on same segment.
		segmentNode("s1", 100,
			operatorNode(0, "filter", "", false, 100, 20,
				withOrder(indexGroup(101, 1, 501, "sort", indexCounters(100, 40, 0)), "1"),
				secondaryRaw(105, 1, "no_index_on_field", 100, 100, 40),
				withOrder(rawGroup(102, 2, "schema_not_nullable", map[string]int64{"idx_scan": 0, "raw_scan": 0, "rows_in": 40, "rows_out": 40}), "2"),
				withOrder(rawGroup(103, 3, "cached", map[string]int64{"idx_scan": 0, "raw_scan": 0, "rows_in": 40, "rows_out": 20}), "3"),
				skippedFg(104, 4),
			),
			operatorNode(1, "mvcc", "", false, 20, 18),
			operatorNode(2, "vector_search", "", false, 18, 10),
		),
		segmentNode("s2", 200,
			operatorNode(0, "filter", "", true, 200, 50),
			operatorNode(1, "mvcc", "", false, 50, 45),
			operatorNode(2, "vector_search", "", false, 45, 10),
		),
	}
	qn2Segs := []*internalpb.ExecutionExplain{
		segmentNode("s3", 150,
			operatorNode(0, "filter", "", false, 150, 0,
				withOrder(rawGroup(101, 1, "index_unusable", rawCounters(150, 30)), "2"),
				withOrder(indexGroup(103, 3, 601, "inverted", map[string]int64{
					"idx_scan": 150, "raw_scan": 0, "rows_in": 150, "rows_out": 0, "element_hits": 12,
				}), "1"),
				skippedFg(104, 4),
			),
			operatorNode(1, "mvcc", "", false, 0, 0),
			operatorNode(2, "vector_search", "", false, 0, 0),
		),
	}
	rawRollingSeg := segmentNode("s4", 80,
		operatorNode(0, "filter", "", false, 80, 10,
			withOrder(indexGroup(101, 1, 501, "sort", indexCounters(80, 10, 0)), "1"),
			skippedFg(104, 4),
		),
		operatorNode(1, "mvcc", "", false, 10, 10),
		operatorNode(2, "vector_search", "", false, 10, 5),
	)

	allRaw := append(append([]*internalpb.ExecutionExplain{}, qn1Segs...), qn2Segs...)
	allRaw = append(allRaw, rawRollingSeg)
	want := Summarize(allRaw, leaves)
	FinalizeSummary(want, map[int64]string{501: "a_sort", 601: "x_inv"}, nil)

	g1 := FoldSegmentsWithName("vch0/1/Historical", qn1Segs)
	g2 := FoldSegmentsWithName("vch1/2/Historical", qn2Segs)
	require.NotNil(t, g1)
	require.NotNil(t, g2)
	assert.Equal(t, KindSegmentGroup, g1.GetKind())
	assert.Empty(t, Segments(g1, g2))

	tree := MergeReduced(
		Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{g1}),
		Wrap(KindQueryNode, "2", []*internalpb.ExecutionExplain{g2}),
		Wrap(KindQueryNode, "3", []*internalpb.ExecutionExplain{rawRollingSeg}),
	)
	got := SummarizeWithSubSearches(TaggedSegments(tree), leaves, nil)
	FinalizeSummary(got, map[int64]string{501: "a_sort", 601: "x_inv"}, nil)

	assert.Equal(t, want, got)
	assert.Equal(t, RenderSummaryText(want), RenderSummaryText(got))
	wantJSON, err := RenderSummaryJSON(want)
	require.NoError(t, err)
	gotJSON, err := RenderSummaryJSON(got)
	require.NoError(t, err)
	assert.Equal(t, wantJSON, gotJSON)
}

func TestFoldSegmentsHybridAndEmptyFilterFoldParity(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "a == 1", Type: "equal", FieldID: 101, Field: "a", HasSubReq: true, SubReqIndex: 0},
		2: {Condition: "b == 2", Type: "equal", FieldID: 102, Field: "b", HasSubReq: true, SubReqIndex: 1},
	}
	subMeta := []SubSearchMeta{
		{SubReqIndex: 0, AnnsField: "vec_a"},
		{SubReqIndex: 1, AnnsField: "vec_b"},
	}

	// Two vchannels sharing QN 1; sub-search 0 prunes s2 on vch0 while sub-search 1 scans both s1 and s2.
	s1 := segmentNode("s1", 100, operatorNode(0, "filter", "", false, 100, 10, indexGroup(101, 1, 501, "sort", indexCounters(100, 10, 0))))
	s1Sub1 := segmentNode("s1", 100, operatorNode(0, "filter", "", false, 100, 25, rawGroup(102, 2, "no_index_on_field", rawCounters(100, 25))))
	s2Sub1 := segmentNode("s2", 200, operatorNode(0, "filter", "", false, 200, 30, rawGroup(102, 2, "no_index_on_field", rawCounters(200, 30))))
	s3Sub0 := segmentNode("s3", 150, operatorNode(0, "filter", "", false, 150, 15, indexGroup(101, 1, 501, "sort", indexCounters(150, 15, 0))))
	s3Sub1 := segmentNode("s3", 150, operatorNode(0, "filter", "", false, 150, 20, rawGroup(102, 2, "no_index_on_field", rawCounters(150, 20))))

	rawTree := MergeReduced(
		WrapSubSearch(0, []*internalpb.ExecutionExplain{
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{s1}),
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{s3Sub0}),
		}),
		WrapSubSearch(1, []*internalpb.ExecutionExplain{
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{s1Sub1, s2Sub1}),
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{s3Sub1}),
		}),
	)
	groupedTree := MergeReduced(
		WrapSubSearch(0, []*internalpb.ExecutionExplain{
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{FoldSegmentsWithName("vch0/1/Historical", []*internalpb.ExecutionExplain{s1})}),
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{FoldSegmentsWithName("vch1/1/Historical", []*internalpb.ExecutionExplain{s3Sub0})}),
		}),
		WrapSubSearch(1, []*internalpb.ExecutionExplain{
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{FoldSegmentsWithName("vch0/1/Historical", []*internalpb.ExecutionExplain{s1Sub1, s2Sub1})}),
			Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{FoldSegmentsWithName("vch1/1/Historical", []*internalpb.ExecutionExplain{s3Sub1})}),
		}),
	)

	wantHybrid := SummarizeWithSubSearches(TaggedSegments(rawTree), leaves, subMeta)
	gotHybrid := SummarizeWithSubSearches(TaggedSegments(groupedTree), leaves, subMeta)
	assert.Equal(t, wantHybrid, gotHybrid)
	assert.Equal(t, 3, gotHybrid.Segments)
	assert.Equal(t, int64(450), gotHybrid.RowsTotal)

	// Empty internal filter -> mvcc fold & BackfillConditions unfold parity, plus asymmetric non-fold parity.
	foldedEmpty := Summarize([]*internalpb.ExecutionExplain{
		FoldSegments([]*internalpb.ExecutionExplain{
			segmentNode("s1", 100,
				operatorNode(0, "filter", "", false, 100, 90),
				operatorNode(1, "mvcc", "", false, 90, 85),
			),
		}),
	}, nil)
	require.Len(t, foldedEmpty.Operators, 1)
	assert.Equal(t, "mvcc", foldedEmpty.Operators[0].Operator)
	assert.Equal(t, int64(100), foldedEmpty.Operators[0].RowsIn)
	BackfillConditions(foldedEmpty, map[int64]LeafInfo{9: {Condition: "k == 1", Type: "equal", FieldID: 109, Field: "k"}})
	require.Len(t, foldedEmpty.Operators, 2)
	assert.Equal(t, "filter", foldedEmpty.Operators[0].Operator)
	assert.Equal(t, int64(90), foldedEmpty.Operators[1].RowsIn)

	asymmetricGrouped := Summarize([]*internalpb.ExecutionExplain{
		FoldSegments([]*internalpb.ExecutionExplain{
			segmentNode("s1", 100,
				operatorNode(0, "filter", "", false, 100, 80),
				operatorNode(1, "mvcc", "", false, 80, 75),
			),
		}),
		FoldSegments([]*internalpb.ExecutionExplain{
			segmentNode("s2", 200,
				operatorNode(1, "mvcc", "", false, 200, 190),
			),
		}),
	}, nil)
	require.Len(t, asymmetricGrouped.Operators, 2)
	assert.Equal(t, int64(280), asymmetricGrouped.Operators[1].RowsIn)
}
