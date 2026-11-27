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

	t.Run("bare single leaf merges cold index timing and whole-operator cached timing cleanly", func(t *testing.T) {
		leaves := map[int64]LeafInfo{
			1: {Condition: "a == 1", Type: "equal", FieldID: 101, Field: "a"},
		}
		asts := map[int]PlanAST{
			-1: {DocAST: &ExprAST{ExprID: 1}},
		}
		mixedTimed := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       3,
				SealedSegments: 3,
				RowsTotal:      300,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:       3,
					CachedSegments: 2,
					RowsIn:         300,
					RowsOut:        30,
					Time:           &internalpb.ExplainTimeAccum{Count: 3, SumUs: 520, MinUs: 8, MaxUs: 500},
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:       1,
						FieldId:      101,
						Segments:     3,
						RowsIn:       300,
						IdxScan:      100,
						RowsOut:      30,
						EvalTime:     &internalpb.ExplainTimeAccum{Count: 3, SumUs: 500, MinUs: 8, MaxUs: 480},
						PrefetchTime: &internalpb.ExplainTimeAccum{Count: 3, SumUs: 150, MinUs: 0, MaxUs: 150},
						Used: []*internalpb.ExplainIndexUseStats{{
							IndexId:  501,
							Family:   "sort",
							Segments: 1,
							RowsIn:   100,
							IdxScan:  100,
							RowsOut:  10,
							EvalTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 480, MinUs: 480, MaxUs: 480},
						}},
						Reasons: []*internalpb.ExplainDeclineStats{{
							Reason:       "cached",
							Segments:     2,
							RowsIn:       200,
							RowsOut:      20,
							EvalTime:     &internalpb.ExplainTimeAccum{Count: 2, SumUs: 20, MinUs: 8, MaxUs: 12},
							PrefetchTime: &internalpb.ExplainTimeAccum{Count: 2, SumUs: 0, MinUs: 0, MaxUs: 0},
						}},
					}},
				}},
			},
		}
		summary := SummarizeWithSubSearches([]*internalpb.ExecutionExplain{mixedTimed}, leaves, nil, asts)
		require.Len(t, summary.Operators, 1)
		assert.Equal(t, &TimeStats{SumUs: 520, MinUs: 8, AvgUs: 173, MaxUs: 500}, summary.Operators[0].TimeUs)
		all := summary.AllConditions()
		require.Len(t, all, 1)
		assert.Equal(t, 3, all[0].Segments)
		assert.Equal(t, &TimeStats{SumUs: 500, MinUs: 8, AvgUs: 166, MaxUs: 480}, all[0].EvalTimeUs)
		assert.Equal(t, &TimeStats{SumUs: 150, MinUs: 0, AvgUs: 50, MaxUs: 150}, all[0].PrefetchTimeUs)
		require.Len(t, all[0].Used, 1)
		assert.Equal(t, &TimeStats{SumUs: 480, MinUs: 480, AvgUs: 480, MaxUs: 480}, all[0].Used[0].EvalTimeUs)
		require.Len(t, all[0].Reasons, 1)
		assert.Equal(t, &TimeStats{SumUs: 20, MinUs: 8, AvgUs: 10, MaxUs: 12}, all[0].Reasons[0].EvalTimeUs)
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

func TestSummarizeTimingStatsPrefetchWaitAndTextJSONFormatting(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101, Field: "title"},
		2: {Condition: "stock > 0", Type: "range", FieldID: 102, Field: "stock"},
	}

	// Segment 1: evaluated both leaves; filter blocked 79ms waiting on 80ms async prefetch.
	seg1 := singleSegmentExplain(false, 50000,
		&internalpb.ExplainOperatorStats{
			Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
			Segments:         1,
			RowsIn:           50000,
			RowsOut:          30,
			Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 81000, MinUs: 81000, MaxUs: 81000},
			PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 79000, MinUs: 79000, MaxUs: 79000},
			Conditions: []*internalpb.ExplainConditionStats{
				{
					ExprId:       1,
					FieldId:      101,
					Segments:     1,
					RowsIn:       50000,
					IdxScan:      50000,
					RawScan:      30,
					RowsOut:      30,
					EvalOrders:   []*internalpb.ExplainEvalOrderCount{{EvalOrder: 1, Segments: 1}},
					EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
					PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 80000, MinUs: 80000, MaxUs: 80000},
					Used: []*internalpb.ExplainIndexUseStats{{
						IndexId:      501,
						Family:       "ngram",
						Segments:     1,
						RowsIn:       50000,
						IdxScan:      50000,
						RawScan:      30,
						RowsOut:      30,
						EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
						PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 80000, MinUs: 80000, MaxUs: 80000},
					}},
				},
				{
					ExprId:       2,
					FieldId:      102,
					Segments:     1,
					RowsIn:       30,
					IdxScan:      30,
					RowsOut:      30,
					EvalOrders:   []*internalpb.ExplainEvalOrderCount{{EvalOrder: 2, Segments: 1}},
					EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 200, MinUs: 200, MaxUs: 200},
					PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
				},
			},
		},
		&internalpb.ExplainOperatorStats{
			Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
			Segments:         1,
			RowsIn:           30,
			RowsOut:          10,
			Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 4000, MinUs: 4000, MaxUs: 4000},
			PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
		},
	)

	// Segment 2: leaf 1 returned 0 rows so leaf 2 was short-circuited (skipped_segments=1),
	// though leaf 2's background prefetch still ran for 400us.
	seg2 := singleSegmentExplain(true, 10000,
		&internalpb.ExplainOperatorStats{
			Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
			Segments:         1,
			RowsIn:           10000,
			RowsOut:          0,
			Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1000, MinUs: 1000, MaxUs: 1000},
			PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
			Conditions: []*internalpb.ExplainConditionStats{
				{
					ExprId:          1,
					FieldId:         101,
					Segments:        1,
					RawScanSegments: 1,
					RowsIn:          10000,
					RawScan:         10000,
					RowsOut:         0,
					EvalOrders:      []*internalpb.ExplainEvalOrderCount{{EvalOrder: 1, Segments: 1}},
					EvalTime:        &internalpb.ExplainTimeAccum{Count: 1, SumUs: 900, MinUs: 900, MaxUs: 900},
					PrefetchTime:    &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
					Reasons: []*internalpb.ExplainDeclineStats{{
						Reason:       "growing",
						Segments:     1,
						RowsIn:       10000,
						RawScan:      10000,
						RowsOut:      0,
						EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 900, MinUs: 900, MaxUs: 900},
						PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
					}},
				},
				{
					ExprId:          2,
					FieldId:         102,
					Segments:        1,
					SkippedSegments: 1,
					EvalOrders:      []*internalpb.ExplainEvalOrderCount{{EvalOrder: 2, Segments: 1}},
					PrefetchTime:    &internalpb.ExplainTimeAccum{Count: 1, SumUs: 400, MinUs: 400, MaxUs: 400},
				},
			},
		},
		&internalpb.ExplainOperatorStats{
			Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
			Segments:         1,
			RowsIn:           0,
			RowsOut:          0,
			Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 100, MinUs: 100, MaxUs: 100},
			PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
		},
	)

	summary := Summarize([]*internalpb.ExecutionExplain{seg1, seg2}, leaves)
	NameIndexUses(summary, map[int64]string{501: "title_ngram"})

	require.Len(t, summary.Operators, 2)
	fOp := summary.Operators[0]
	assert.Equal(t, &TimeStats{SumUs: 82000, MinUs: 1000, AvgUs: 41000, MaxUs: 81000}, fOp.TimeUs)
	assert.Equal(t, &TimeStats{SumUs: 79000, MinUs: 0, AvgUs: 39500, MaxUs: 79000}, fOp.PrefetchWaitTimeUs)

	// vector_search had prefetch_wait_time sum=0, so PrefetchWaitTimeUs is omitted (nil).
	vsOp := summary.Operators[1]
	assert.Equal(t, &TimeStats{SumUs: 4100, MinUs: 100, AvgUs: 2050, MaxUs: 4000}, vsOp.TimeUs)
	assert.Nil(t, vsOp.PrefetchWaitTimeUs)

	// Leaf 2 has 1 evaluated segment (200us) and 1 skipped segment (prefetch 400us).
	leaf2 := fOp.Conditions[1]
	assert.Equal(t, 1, leaf2.SkippedSegments)
	assert.Equal(t, &TimeStats{SumUs: 200, MinUs: 200, AvgUs: 200, MaxUs: 200}, leaf2.EvalTimeUs)
	assert.Equal(t, &TimeStats{SumUs: 400, MinUs: 0, AvgUs: 200, MaxUs: 400}, leaf2.PrefetchTimeUs)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "[op 0] filter (segments=2, rows: 60000 -> 30, time_us: avg=41000, max=81000, min=1000, sum=82000, prefetch_wait_us: avg=39500, max=79000, min=0, sum=79000)")
	assert.Contains(t, text, "[op 1] vector_search (segments=2, rows: 30 -> 10, time_us: avg=2050, max=4000, min=100, sum=4100)")
	assert.NotContains(t, text, "[op 1] vector_search (segments=2, rows: 30 -> 10, time_us: avg=2050, max=4000, min=100, sum=4100, prefetch_wait_us")
	assert.Contains(t, text, "expr: stock > 0 (segments=2, skipped_segments=1, rows: 30 -> 30, idx_scan=30, raw_scan=0, eval_order=2, eval_us=200, prefetch_us: avg=200, max=400, min=0, sum=400)")

	rawJSON, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.Contains(t, rawJSON, `"prefetch_wait_time_us":{"sum_us":79000,"min_us":0,"avg_us":39500,"max_us":79000}`)

	var roundTrip Summary
	require.NoError(t, json.Unmarshal([]byte(rawJSON), &roundTrip))
	assert.Equal(t, summary.Operators[0].PrefetchWaitTimeUs, roundTrip.Operators[0].PrefetchWaitTimeUs)
	assert.Equal(t, leaf2.EvalTimeUs, roundTrip.Operators[0].Conditions[1].EvalTimeUs)
	assert.Equal(t, leaf2.PrefetchTimeUs, roundTrip.Operators[0].Conditions[1].PrefetchTimeUs)

	// Mixed timed + untimed segments (rolling upgrade) suppress incomplete operator timing.
	untimedSeg := singleSegmentExplain(false, 500, &internalpb.ExplainOperatorStats{
		Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
		Segments: 1,
		RowsIn:   500,
		RowsOut:  10,
	})
	mixed := Summarize([]*internalpb.ExecutionExplain{seg1, untimedSeg}, leaves)
	assert.Nil(t, mixed.Operators[0].TimeUs)
	assert.Nil(t, mixed.Operators[0].PrefetchWaitTimeUs)
}

func TestSummarizeDetailModeSortingFinalizationAndHybrid(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101, Field: "title"},
	}

	// Fast sealed segment (total op time = 1200us).
	fastSealed := singleSegmentExplain(false, 20000,
		&internalpb.ExplainOperatorStats{
			Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
			Segments: 1,
			RowsIn:   20000,
			RowsOut:  15,
			Time:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 800, MinUs: 800, MaxUs: 800},
			Conditions: []*internalpb.ExplainConditionStats{{
				ExprId:   1,
				FieldId:  101,
				Segments: 1,
				RowsIn:   20000,
				IdxScan:  20000,
				RowsOut:  15,
				EvalTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 750, MinUs: 750, MaxUs: 750},
				Used: []*internalpb.ExplainIndexUseStats{{
					IndexId:  901,
					Family:   "ngram",
					Segments: 1,
					RowsIn:   20000,
					IdxScan:  20000,
					RowsOut:  15,
					EvalTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 750, MinUs: 750, MaxUs: 750},
				}},
			}},
		},
		&internalpb.ExplainOperatorStats{
			Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
			Segments: 1,
			RowsIn:   15,
			RowsOut:  5,
			Time:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 400, MinUs: 400, MaxUs: 400},
		},
	)
	AttachSegmentDetail(fastSealed, 449101, false)

	// Slow cold sealed segment (total op time = 85000us).
	slowSealed := singleSegmentExplain(false, 50000,
		&internalpb.ExplainOperatorStats{
			Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
			Segments:         1,
			RowsIn:           50000,
			RowsOut:          30,
			Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 81000, MinUs: 81000, MaxUs: 81000},
			PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 79000, MinUs: 79000, MaxUs: 79000},
			Conditions: []*internalpb.ExplainConditionStats{{
				ExprId:       1,
				FieldId:      101,
				Segments:     1,
				RowsIn:       50000,
				IdxScan:      50000,
				RawScan:      30,
				RowsOut:      30,
				EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
				PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 80000, MinUs: 80000, MaxUs: 80000},
				Used: []*internalpb.ExplainIndexUseStats{{
					IndexId:      902,
					Family:       "ngram",
					Segments:     1,
					RowsIn:       50000,
					IdxScan:      50000,
					RawScan:      30,
					RowsOut:      30,
					EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
					PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 80000, MinUs: 80000, MaxUs: 80000},
				}},
			}},
		},
		&internalpb.ExplainOperatorStats{
			Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
			Segments: 1,
			RowsIn:   30,
			RowsOut:  10,
			Time:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 4000, MinUs: 4000, MaxUs: 4000},
		},
	)
	AttachSegmentDetail(slowSealed, 449102, false)

	// Growing segment (total op time = 2000us).
	growingSeg := singleSegmentExplain(true, 10000,
		&internalpb.ExplainOperatorStats{
			Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
			Segments: 1,
			RowsIn:   10000,
			RowsOut:  8,
			Time:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
			Conditions: []*internalpb.ExplainConditionStats{{
				ExprId:          1,
				FieldId:         101,
				Segments:        1,
				RawScanSegments: 1,
				RowsIn:          10000,
				RawScan:         10000,
				RowsOut:         8,
				EvalTime:        &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
				Reasons: []*internalpb.ExplainDeclineStats{{
					Reason:   "growing",
					Segments: 1,
					RowsIn:   10000,
					RawScan:  10000,
					RowsOut:  8,
					EvalTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
				}},
			}},
		},
		&internalpb.ExplainOperatorStats{
			Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
			Segments: 1,
			RowsIn:   8,
			RowsOut:  4,
			Time:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 500, MinUs: 500, MaxUs: 500},
		},
	)
	AttachSegmentDetail(growingSeg, 449103, true)

	// Default summary mode hides SegmentDetails even if present on proto.
	summaryOnly := SummarizeWithSubSearches([]*internalpb.ExecutionExplain{fastSealed, slowSealed, growingSeg}, leaves, nil, nil)
	assert.Equal(t, LevelSummary, summaryOnly.Level)
	assert.Empty(t, summaryOnly.SegmentDetails)

	// Detail mode emits slowest-first SegmentDetails alongside aggregated Operators.
	detail := SummarizeWithSubSearchesAndLevel(LevelDetail, []*internalpb.ExecutionExplain{fastSealed, slowSealed, growingSeg}, leaves, nil, nil)
	assert.Equal(t, LevelDetail, detail.Level)
	assert.Zero(t, detail.OmittedSegments)
	assert.ElementsMatch(t, []int64{901, 902}, detail.NeededIndexIDs())
	assert.Len(t, detail.AllConditions(), 1, "AllConditions must not duplicate leaf conditions across N segments")

	FinalizeSummary(detail, map[int64]string{901: "title_ngram_a", 902: "title_ngram_b"}, map[int64]string{101: "title"})

	require.Len(t, detail.SegmentDetails, 3)
	assert.Equal(t, int64(449102), detail.SegmentDetails[0].SegmentID)
	assert.Equal(t, SegmentStateSealed, detail.SegmentDetails[0].State)
	assert.Equal(t, int64(50000), detail.SegmentDetails[0].RowsTotal)
	assert.Equal(t, "title_ngram_b", detail.SegmentDetails[0].Operators[0].Conditions[0].Used[0].IndexName)

	assert.Equal(t, int64(449103), detail.SegmentDetails[1].SegmentID)
	assert.Equal(t, SegmentStateGrowing, detail.SegmentDetails[1].State)

	assert.Equal(t, int64(449101), detail.SegmentDetails[2].SegmentID)
	assert.Equal(t, SegmentStateSealed, detail.SegmentDetails[2].State)
	assert.Equal(t, "title_ngram_a", detail.SegmentDetails[2].Operators[0].Conditions[0].Used[0].IndexName)

	text := RenderSummaryText(detail)
	assert.Contains(t, text, "  segments_detail (count=3):\n")
	assert.Contains(t, text, "    [segment 449102] sealed (rows_total=50000)\n")
	assert.Contains(t, text, "      [op 0] filter (segments=1, rows: 50000 -> 30, time_us=81000, prefetch_wait_us=79000)\n")
	assert.Contains(t, text, "    [segment 449103] growing (rows_total=10000)\n")
	assert.Less(t, strings.Index(text, "[segment 449102]"), strings.Index(text, "[segment 449103]"))
	assert.Less(t, strings.Index(text, "[segment 449103]"), strings.Index(text, "[segment 449101]"))

	rawJSON, err := RenderSummaryJSON(detail)
	require.NoError(t, err)
	assert.Contains(t, rawJSON, `"level":"detail"`)
	assert.NotContains(t, rawJSON, `"omitted_segments"`)
	assert.Contains(t, rawJSON, `"segments_detail":[`)

	var decoded Summary
	require.NoError(t, json.Unmarshal([]byte(rawJSON), &decoded))
	require.Len(t, decoded.SegmentDetails, 3)
	assert.Equal(t, int64(449102), decoded.SegmentDetails[0].SegmentID)

	// HybridSearch detail mode renders per-sub-search segments_detail and omitted_segments when non-zero.
	hSub0 := WrapSubSearch(0, []*internalpb.ExecutionExplain{fastSealed, slowSealed})
	hSub0.SubSearches[0].Group.OmittedSegmentDetails = 7
	hSub1 := WrapSubSearch(1, []*internalpb.ExecutionExplain{growingSeg})
	hybridDetail := SummarizeWithSubSearchesAndLevel(
		LevelDetail,
		[]*internalpb.ExecutionExplain{hSub0, hSub1},
		leaves,
		[]SubSearchMeta{{SubReqIndex: 0, AnnsField: "dense_vec"}, {SubReqIndex: 1, AnnsField: "sparse_vec"}},
		nil,
	)
	FinalizeSummary(hybridDetail, map[int64]string{901: "title_ngram_a", 902: "title_ngram_b"}, map[int64]string{101: "title"})
	require.Len(t, hybridDetail.SubSearches, 2)
	assert.Equal(t, 7, hybridDetail.SubSearches[0].OmittedSegments)
	require.Len(t, hybridDetail.SubSearches[0].SegmentDetails, 2)
	assert.Equal(t, int64(449102), hybridDetail.SubSearches[0].SegmentDetails[0].SegmentID)
	require.Len(t, hybridDetail.SubSearches[1].SegmentDetails, 1)
	assert.Equal(t, SegmentStateGrowing, hybridDetail.SubSearches[1].SegmentDetails[0].State)

	hybridText := RenderSummaryText(hybridDetail)
	assert.Contains(t, hybridText, "    segments_detail (count=2, omitted_segments=7):\n")
	assert.Contains(t, hybridText, "      [segment 449102] sealed (rows_total=50000)\n")
	assert.Contains(t, hybridText, "    segments_detail (count=1):\n")

	hybridJSON, err := RenderSummaryJSON(hybridDetail)
	require.NoError(t, err)
	assert.Contains(t, hybridJSON, `"omitted_segments":7`)
}
