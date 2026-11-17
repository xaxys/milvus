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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

func Summarize(explains []*internalpb.ExecutionExplain, leaves map[int64]LeafInfo) *Summary {
	return SummarizeWithSubSearches(explains, leaves, nil, nil)
}

func NameIndexUses(summary *Summary, names map[int64]string) {
	FinalizeSummary(summary, names, nil)
}

func singleSegmentExplain(isGrowing bool, rowsTotal int64, ops ...*internalpb.ExplainOperatorStats) *internalpb.ExecutionExplain {
	var sealed, growing int64 = 1, 0
	if isGrowing {
		sealed, growing = 0, 1
	}
	return &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:        1,
			SealedSegments:  sealed,
			GrowingSegments: growing,
			RowsTotal:       rowsTotal,
			Operators:       ops,
		},
	}
}

func TestSummarizeFlatSegmentBreakdownAndGrowingReason(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101, Field: "title"},
		2: {Condition: "price in (100, 500]", Type: "range", FieldID: 102, Field: "price"},
	}

	seg1 := singleSegmentExplain(false, 50000, &internalpb.ExplainOperatorStats{
		Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		Segments: 1,
		RowsIn:   50000,
		RowsOut:  800,
		Conditions: []*internalpb.ExplainConditionStats{
			{
				ExprId:   1,
				FieldId:  101,
				Segments: 1,
				RowsIn:   50000,
				IdxScan:  50000,
				RawScan:  1500,
				RowsOut:  1200,
				Used: []*internalpb.ExplainIndexUseStats{{
					IndexId:  1001,
					Family:   "ngram",
					Segments: 1,
					RowsIn:   50000,
					IdxScan:  50000,
					RawScan:  1500,
					RowsOut:  1200,
				}},
			},
			{
				ExprId:   2,
				FieldId:  102,
				Segments: 1,
				RowsIn:   1200,
				IdxScan:  1200,
				RowsOut:  800,
				Used: []*internalpb.ExplainIndexUseStats{{
					IndexId:  1002,
					Family:   "sort",
					Segments: 1,
					RowsIn:   1200,
					IdxScan:  1200,
					RowsOut:  800,
				}},
			},
		},
	})

	seg2SealedNoIdx := singleSegmentExplain(false, 50000, &internalpb.ExplainOperatorStats{
		Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		Segments: 1,
		RowsIn:   50000,
		RowsOut:  900,
		Conditions: []*internalpb.ExplainConditionStats{
			{
				ExprId:          1,
				FieldId:         101,
				Segments:        1,
				RawScanSegments: 1,
				RowsIn:          50000,
				RawScan:         50000,
				RowsOut:         1400,
				Reasons: []*internalpb.ExplainDeclineStats{{
					Reason:   "no_index",
					Segments: 1,
					RowsIn:   50000,
					RawScan:  50000,
					RowsOut:  1400,
				}},
			},
			{
				ExprId:   2,
				FieldId:  102,
				Segments: 1,
				RowsIn:   1400,
				IdxScan:  1400,
				RowsOut:  900,
				Used: []*internalpb.ExplainIndexUseStats{{
					IndexId:  1002,
					Family:   "sort",
					Segments: 1,
					RowsIn:   1400,
					IdxScan:  1400,
					RowsOut:  900,
				}},
			},
		},
	})

	seg3Growing := singleSegmentExplain(true, 50000, &internalpb.ExplainOperatorStats{
		Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		Segments: 1,
		RowsIn:   50000,
		RowsOut:  600,
		Conditions: []*internalpb.ExplainConditionStats{
			{
				ExprId:          1,
				FieldId:         101,
				Segments:        1,
				RawScanSegments: 1,
				RowsIn:          50000,
				RawScan:         50000,
				RowsOut:         1000,
				Reasons: []*internalpb.ExplainDeclineStats{{
					Reason:   "growing",
					Segments: 1,
					RowsIn:   50000,
					RawScan:  50000,
					RowsOut:  1000,
				}},
			},
			{
				ExprId:          2,
				FieldId:         102,
				Segments:        1,
				RawScanSegments: 1,
				RowsIn:          1000,
				RawScan:         1000,
				RowsOut:         600,
				Reasons: []*internalpb.ExplainDeclineStats{{
					Reason:   "growing",
					Segments: 1,
					RowsIn:   1000,
					RawScan:  1000,
					RowsOut:  600,
				}},
			},
		},
	})

	summary := Summarize([]*internalpb.ExecutionExplain{seg1, seg2SealedNoIdx, seg3Growing}, leaves)
	NameIndexUses(summary, map[int64]string{1001: "title_ngram", 1002: "price_sort"})

	assert.Equal(t, 3, summary.Segments)
	assert.Equal(t, 2, summary.SealedSegments)
	assert.Equal(t, 1, summary.GrowingSegments)
	assert.Equal(t, int64(150000), summary.RowsTotal)

	require.Len(t, summary.Operators, 1)
	require.Len(t, summary.Operators[0].Conditions, 2)
	titleCond := summary.Operators[0].Conditions[0]
	assert.Equal(t, 2, titleCond.RawScanSegments, "ngram refinement raw_scan is not counted as raw_scan_segments")
	require.Len(t, titleCond.Reasons, 2)
	assert.Equal(t, "growing", titleCond.Reasons[0].Reason)
	assert.Equal(t, "no_index", titleCond.Reasons[1].Reason)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "query (segments=3 [sealed=2, growing=1], rows_total=150000)")
	assert.Contains(t, text, "raw (growing): segments=1")
	assert.Contains(t, text, "raw (no_index): segments=1")

	rawJSON, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.Contains(t, rawJSON, `"segments":3`)
	assert.Contains(t, rawJSON, `"sealed_segments":2`)
	assert.Contains(t, rawJSON, `"growing_segments":1`)
	assert.Contains(t, rawJSON, `"rows_total":150000`)

	var roundTrip Summary
	require.NoError(t, json.Unmarshal([]byte(rawJSON), &roundTrip))
	assert.Equal(t, 2, roundTrip.SealedSegments)
	assert.Equal(t, 1, roundTrip.GrowingSegments)
}

func TestRecursiveASTConditionTreeAndEvalOrderPreservation(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "a in (10, +inf)", Type: "range", FieldID: 101, Field: "a"},
		2: {Condition: "b in (20, +inf)", Type: "range", FieldID: 102, Field: "b"},
		3: {Condition: "items[score] in [90, +inf)", Type: "range", FieldID: 103, Field: "items[score]"},
	}
	// Query AST: (a > 10 OR b > 20) AND match_least(items, score >= 90, 2)
	asts := map[int]PlanAST{
		-1: {
			DocAST: &ExprAST{
				Logic: "and",
				Children: []*ExprAST{
					{
						Logic: "or",
						Children: []*ExprAST{
							{ExprID: 1},
							{ExprID: 2},
						},
					},
					{
						Logic:    "match_least",
						Field:    "items",
						Count:    2,
						Children: []*ExprAST{{ExprID: 3}},
					},
				},
			},
		},
	}

	seg := singleSegmentExplain(false, 100, &internalpb.ExplainOperatorStats{
		Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		Segments: 1,
		RowsIn:   100,
		RowsOut:  12,
		Conditions: []*internalpb.ExplainConditionStats{
			{
				ExprId:     1,
				FieldId:    101,
				Segments:   1,
				RowsIn:     100,
				RowsOut:    30,
				EvalOrders: []*internalpb.ExplainEvalOrderCount{{EvalOrder: 2, Segments: 1}},
			},
			{
				ExprId:     2,
				FieldId:    102,
				Segments:   1,
				RowsIn:     70,
				RowsOut:    20,
				EvalOrders: []*internalpb.ExplainEvalOrderCount{{EvalOrder: 3, Segments: 1}},
			},
			{
				ExprId:      3,
				FieldId:     103,
				Segments:    1,
				RowsIn:      50,
				RowsOut:     12,
				ElementHits: 28,
				EvalOrders:  []*internalpb.ExplainEvalOrderCount{{EvalOrder: 1, Segments: 1}},
			},
		},
	})

	summary := SummarizeWithSubSearches([]*internalpb.ExecutionExplain{seg}, leaves, nil, asts)
	FinalizeSummary(summary, nil, nil)

	require.Len(t, summary.Operators, 1)
	require.Len(t, summary.Operators[0].Conditions, 2)
	assert.Equal(t, "or", summary.Operators[0].Conditions[0].Logic)
	assert.Equal(t, "match_least", summary.Operators[0].Conditions[1].Logic)
	assert.Equal(t, "items", summary.Operators[0].Conditions[1].Field)
	assert.Equal(t, int64(2), summary.Operators[0].Conditions[1].Count)

	// AllConditions flattens all 3 leaves recursively in AST order (1, 2, 3), NOT eval_order (3, 1, 2).
	allLeaves := summary.AllConditions()
	require.Len(t, allLeaves, 3)
	assert.Equal(t, "a in (10, +inf)", allLeaves[0].Condition)
	assert.Equal(t, "b in (20, +inf)", allLeaves[1].Condition)
	assert.Equal(t, "items[score] in [90, +inf)", allLeaves[2].Condition)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "    or:\n      expr: a in (10, +inf)")
	assert.Contains(t, text, "    match_least (field=items, count=2):\n      expr: items[score] in [90, +inf)")
	assert.Less(t, strings.Index(text, "or:\n"), strings.Index(text, "match_least"))

	rawJSON, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.NotContains(t, rawJSON, `"logic":"or","condition"`)

	var decoded Summary
	require.NoError(t, json.Unmarshal([]byte(rawJSON), &decoded))
	assert.Equal(t, "or", decoded.Operators[0].Conditions[0].Logic)
	require.Len(t, decoded.Operators[0].Conditions[0].Children, 2)
	assert.Equal(t, 2, decoded.Operators[0].Conditions[0].Children[0].EvalOrder)
}

func TestWholeFilterUnattributedCacheOnlyAttributesBareSingleLeaf(t *testing.T) {
	t.Run("bare single leaf receives whole-operator cache attribution and clears skipped", func(t *testing.T) {
		leaves := map[int64]LeafInfo{
			1: {Condition: "a == 1", Type: "equal", FieldID: 101, Field: "a"},
		}
		asts := map[int]PlanAST{
			-1: {DocAST: &ExprAST{ExprID: 1}},
		}
		// 1 skipped cold segment + 1 warm whole-operator cached segment (emitted directly on bare leaf).
		mixed := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      100,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:       2,
					CachedSegments: 1,
					RowsIn:         0,
					RowsOut:        0,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:          1,
						FieldId:         101,
						Segments:        2,
						SkippedSegments: 1,
						Reasons: []*internalpb.ExplainDeclineStats{{
							Reason:   "cached",
							Segments: 1,
						}},
					}},
				}},
			},
		}
		summary := SummarizeWithSubSearches([]*internalpb.ExecutionExplain{mixed}, leaves, nil, asts)
		all := summary.AllConditions()
		require.Len(t, all, 1)
		assert.Equal(t, 2, all[0].Segments)
		assert.False(t, all[0].Skipped)
		require.Len(t, all[0].Reasons, 1)
		assert.Equal(t, "cached", all[0].Reasons[0].Reason)
		assert.Equal(t, 1, all[0].Reasons[0].Segments)
	})

	t.Run("compound OR/NOT/MATCH AST keeps operator cached_segments without fabricating per-leaf rows", func(t *testing.T) {
		leaves := map[int64]LeafInfo{
			1: {Condition: "a == 1", Type: "equal", FieldID: 101, Field: "a"},
			2: {Condition: "b == 2", Type: "equal", FieldID: 102, Field: "b"},
		}
		asts := map[int]PlanAST{
			-1: {
				DocAST: &ExprAST{
					Logic:    "or",
					Children: []*ExprAST{{ExprID: 1}, {ExprID: 2}},
				},
			},
		}
		warmWholeFilter := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      200,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:       2,
					CachedSegments: 2,
					RowsIn:         200,
					RowsOut:        45,
				}},
			},
		}

		summary := SummarizeWithSubSearches([]*internalpb.ExecutionExplain{warmWholeFilter}, leaves, nil, asts)
		require.Len(t, summary.Operators, 1)
		assert.Equal(t, 2, summary.Operators[0].CachedSegments)
		all := summary.AllConditions()
		require.Len(t, all, 2)
		for _, leaf := range all {
			assert.Zero(t, leaf.Segments)
			assert.Zero(t, leaf.RowsIn)
			assert.Zero(t, leaf.RowsOut)
			assert.Empty(t, leaf.Reasons)
		}
	})
}

func TestMatchZeroCountAndMultiOperandSubItemRendering(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "items[score] in (90, +inf)", Type: "range", FieldID: 101, Field: "items[score]"},
		2: {Condition: "a lt b", Type: "compare", FieldID: 102, Field: "a"},
	}
	asts := map[int]PlanAST{
		-1: {
			DocAST: &ExprAST{
				Logic: "and",
				Children: []*ExprAST{
					{
						Logic:    "match_most",
						Field:    "items",
						Count:    0,
						HasCount: true,
						Children: []*ExprAST{{ExprID: 1}},
					},
					{ExprID: 2},
				},
			},
		},
	}
	seg := singleSegmentExplain(false, 100, &internalpb.ExplainOperatorStats{
		Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		Segments: 1,
		RowsIn:   100,
		RowsOut:  20,
		Conditions: []*internalpb.ExplainConditionStats{
			{
				ExprId:   1,
				FieldId:  101,
				Segments: 1,
				RowsIn:   250,
				RowsOut:  5,
			},
			{
				ExprId:          2,
				FieldId:         102,
				Segments:        1,
				RawScanSegments: 1,
				RowsIn:          100,
				IdxScan:         100,
				RawScan:         100,
				RowsOut:         20,
				Used: []*internalpb.ExplainIndexUseStats{{
					IndexId:  501,
					Family:   "sort",
					Segments: 1,
					IdxScan:  100,
					RawScan:  0,
				}},
				Reasons: []*internalpb.ExplainDeclineStats{{
					Reason:   "no_index",
					Segments: 1,
					IdxScan:  0,
					RawScan:  100,
				}},
			},
		},
	})

	summary := SummarizeWithSubSearches([]*internalpb.ExecutionExplain{seg}, leaves, nil, asts)
	FinalizeSummary(summary, map[int64]string{501: "a_sort"}, nil)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "match_most (field=items, count=0):")
	assert.Contains(t, text, "index a_sort (sort): segments=1, idx_scan=100, raw_scan=0")
	assert.Contains(t, text, "raw (no_index): segments=1, idx_scan=0, raw_scan=100")

	rawJSON, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.Contains(t, rawJSON, `"logic":"match_most","field":"items","count":0`)

	var decoded Summary
	require.NoError(t, json.Unmarshal([]byte(rawJSON), &decoded))
	assert.Equal(t, int64(0), decoded.Operators[0].Conditions[0].Count)
	assert.True(t, decoded.Operators[0].Conditions[0].HasCount)
}
