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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

func TestWrapSumsChildCounters(t *testing.T) {
	leaf := func(matched int64) *internalpb.ExecutionExplain {
		return &internalpb.ExecutionExplain{
			Kind:     KindSegment,
			Counters: map[string]int64{"rows_matched": matched, "rows_scanned": 10},
		}
	}

	node := Wrap(KindQueryNode, "7", []*internalpb.ExecutionExplain{leaf(1), nil, leaf(2)})

	assert.Equal(t, KindQueryNode, node.GetKind())
	assert.Equal(t, "7", node.GetName())
	assert.Equal(t, uint32(1), node.GetVersion())
	assert.Equal(t, []int64{3, 20}, []int64{
		node.GetCounters()["rows_matched"],
		node.GetCounters()["rows_scanned"],
	})
	assert.Len(t, node.GetChildren(), 2)
}

func TestSpliceFlattensReducedOnly(t *testing.T) {
	segment := &internalpb.ExecutionExplain{Kind: KindSegment, Name: "s"}
	reduced := MergeReduced(segment, nil)
	assert.Equal(t, KindReduced, reduced.GetKind())

	assert.Equal(t, []*internalpb.ExecutionExplain{segment}, Splice(nil, reduced))

	querynode := Wrap(KindQueryNode, "1", nil)
	assert.Equal(t, []*internalpb.ExecutionExplain{querynode}, Splice(nil, querynode),
		"a meaningful node is kept even when it has children")
}

func TestWrapSubSearchAndTaggedSegments(t *testing.T) {
	segA := &internalpb.ExecutionExplain{Kind: KindSegment, Name: "s1", Counters: map[string]int64{"rows_total": 100}}
	segB := &internalpb.ExecutionExplain{Kind: KindSegment, Name: "s2", Counters: map[string]int64{"rows_total": 200}}
	sub0 := WrapSubSearch(0, []*internalpb.ExecutionExplain{Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{segA})})
	sub1 := WrapSubSearch(1, []*internalpb.ExecutionExplain{Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{segB})})
	merged := MergeReduced(sub0, sub1)

	tagged := TaggedSegments(merged)
	require.Len(t, tagged, 2)
	assert.Equal(t, 0, tagged[0].SubReqIndex)
	assert.Equal(t, "s1", tagged[0].Segment.GetName())
	assert.Equal(t, 1, tagged[1].SubReqIndex)
	assert.Equal(t, "s2", tagged[1].Segment.GetName())
}

func TestMergeStage1ExplainOperatorPipeline(t *testing.T) {
	stage2 := Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 1000},
		Children: []*internalpb.ExecutionExplain{
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0", "cached": "true"},
				Counters:   map[string]int64{"rows_in": 1000, "rows_out": 80},
			},
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "mvcc", "step": "1"},
				Counters:   map[string]int64{"rows_in": 80, "rows_out": 80},
			},
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "vector_search", "step": "2"},
				Counters:   map[string]int64{"rows_in": 80, "rows_out": 10},
			},
		},
	}})

	stage1Cold := Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "seg1",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "idx_scan": 1000, "rows_out": 80},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       KindOperator,
			Attributes: map[string]string{"name": "filter", "step": "0"},
			Counters:   map[string]int64{"rows_in": 1000, "rows_out": 80},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindFilterGroup,
				Attributes: map[string]string{"expr_id": "1", "field_id": "101", "index_id": "5001", "family": "sort"},
				Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 1000, "raw_scan": 0, "rows_out": 80},
			}},
		}},
	}})

	merged, ok := MergeStage1Explain(stage2, stage1Cold)
	require.True(t, ok)
	seg := Segments(merged)[0]
	filterOp := seg.GetChildren()[0]
	require.Len(t, filterOp.GetChildren(), 1)
	assert.Equal(t, int64(80), filterOp.GetChildren()[0].GetCounters()["rows_out"])
	assert.Empty(t, filterOp.GetAttributes()["cached"],
		"intra-request Stage 1 -> Stage 2 handoff clears Stage 2's cache hit marker when Stage 1 evaluated cold")
}

func TestMergeStage1ExplainAccumulatesOperatorAndLeafTimings(t *testing.T) {
	stage2 := Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{
		{
			Kind:     KindSegment,
			Name:     "sealed_seg",
			Counters: map[string]int64{"rows_total": 1000},
			Children: []*internalpb.ExecutionExplain{
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "filter", "step": "0", "cached": "true"},
					Counters:   map[string]int64{"rows_in": 1000, "rows_out": 80, "time_us": 20},
				},
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "mvcc", "step": "1"},
					Counters:   map[string]int64{"rows_in": 80, "rows_out": 80, "time_us": 80},
				},
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "vector_search", "step": "2"},
					Counters:   map[string]int64{"rows_in": 80, "rows_out": 10, "time_us": 4500},
				},
			},
		},
		{
			Kind:     KindSegment,
			Name:     "growing_seg",
			Counters: map[string]int64{"rows_total": 500, "rows_in": 500, "raw_scan": 500, "rows_out": 40},
			Children: []*internalpb.ExecutionExplain{
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "filter", "step": "0"},
					Counters:   map[string]int64{"rows_in": 500, "rows_out": 40, "time_us": 700},
					Children: []*internalpb.ExecutionExplain{{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
						Counters:   map[string]int64{"rows_in": 500, "idx_scan": 0, "raw_scan": 500, "rows_out": 40, "eval_us": 600},
					}},
				},
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "mvcc", "step": "1"},
					Counters:   map[string]int64{"rows_in": 40, "rows_out": 40, "time_us": 50},
				},
			},
		},
	})

	stage1 := Wrap(KindQueryNode, "1", []*internalpb.ExecutionExplain{
		{
			Kind:     KindSegment,
			Name:     "sealed_seg",
			Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "idx_scan": 1000, "rows_out": 80},
			Children: []*internalpb.ExecutionExplain{
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "filter", "step": "0"},
					Counters:   map[string]int64{"rows_in": 1000, "rows_out": 80, "time_us": 1200},
					Children: []*internalpb.ExecutionExplain{{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "1", "field_id": "101", "index_id": "5001", "family": "sort"},
						Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 1000, "raw_scan": 0, "rows_out": 80, "eval_us": 1100, "prefetch_us": 8000},
					}},
				},
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "mvcc", "step": "1"},
					Counters:   map[string]int64{"rows_in": 80, "rows_out": 80, "time_us": 100},
				},
			},
		},
		{
			Kind:     KindSegment,
			Name:     "growing_seg",
			Counters: map[string]int64{"rows_total": 500, "rows_in": 500, "raw_scan": 500, "rows_out": 40},
			Children: []*internalpb.ExecutionExplain{
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "filter", "step": "0"},
					Counters:   map[string]int64{"rows_in": 500, "rows_out": 40, "time_us": 650},
					Children: []*internalpb.ExecutionExplain{{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
						Counters:   map[string]int64{"rows_in": 500, "idx_scan": 0, "raw_scan": 500, "rows_out": 40, "eval_us": 550, "prefetch_us": 2000},
					}},
				},
				{
					Kind:       KindOperator,
					Attributes: map[string]string{"name": "mvcc", "step": "1"},
					Counters:   map[string]int64{"rows_in": 40, "rows_out": 40, "time_us": 40},
				},
			},
		},
	})

	merged, ok := MergeStage1Explain(stage2, stage1)
	require.True(t, ok)
	segs := Segments(merged)
	require.Len(t, segs, 2)

	sealed := segs[0]
	assert.Equal(t, int64(1220), sealed.GetChildren()[0].GetCounters()["time_us"])
	assert.Equal(t, int64(180), sealed.GetChildren()[1].GetCounters()["time_us"])
	assert.Equal(t, int64(4500), sealed.GetChildren()[2].GetCounters()["time_us"])
	assert.Equal(t, int64(1000), sealed.GetChildren()[0].GetCounters()["rows_in"])
	assert.Equal(t, int64(1100), sealed.GetChildren()[0].GetChildren()[0].GetCounters()["eval_us"])
	assert.Equal(t, int64(8000), sealed.GetChildren()[0].GetChildren()[0].GetCounters()["prefetch_us"])

	growing := segs[1]
	assert.Equal(t, int64(1350), growing.GetChildren()[0].GetCounters()["time_us"])
	assert.Equal(t, int64(90), growing.GetChildren()[1].GetCounters()["time_us"])
	assert.Equal(t, int64(500), growing.GetChildren()[0].GetChildren()[0].GetCounters()["rows_in"],
		"re-evaluated Stage 2 segment must not double-count row counters")
	assert.Equal(t, int64(1150), growing.GetChildren()[0].GetChildren()[0].GetCounters()["eval_us"])
	assert.Equal(t, int64(2000), growing.GetChildren()[0].GetChildren()[0].GetCounters()["prefetch_us"])
}

func TestMergeStage1ExplainPreservesZeroRowStage2LeafTimingsAndAlignsByExprID(t *testing.T) {
	stage2 := Wrap(KindQueryNode, "qn1", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "sealed_seg",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 0, "rows_out": 0},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       KindOperator,
			Attributes: map[string]string{"name": "element_filter", "step": "0", "domain": "element"},
			Counters:   map[string]int64{"rows_in": 0, "rows_out": 0, "time_us": 400},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindFilterGroup,
				Attributes: map[string]string{"expr_id": "7", "skipped": "true"},
				Counters:   map[string]int64{"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0, "prefetch_us": 300},
			}},
		}},
	}})
	stage1 := Wrap(KindQueryNode, "qn1", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "sealed_seg",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "idx_scan": 1000, "rows_out": 20},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       KindOperator,
			Attributes: map[string]string{"name": "element_filter", "step": "0", "domain": "element"},
			Counters:   map[string]int64{"rows_in": 1000, "rows_out": 20, "time_us": 900},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindFilterGroup,
				Attributes: map[string]string{"expr_id": "7", "field_id": "201", "index_id": "601", "family": "inverted"},
				Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 1000, "raw_scan": 0, "rows_out": 20, "eval_us": 850, "prefetch_us": 5000},
			}},
		}},
	}})

	merged, ok := MergeStage1Explain(stage2, stage1)
	require.True(t, ok)
	seg := Segments(merged)[0]
	op := seg.GetChildren()[0]
	assert.Equal(t, int64(1300), op.GetCounters()["time_us"])
	require.Len(t, op.GetChildren(), 1)
	fg := op.GetChildren()[0]
	assert.Empty(t, fg.GetAttributes()["skipped"])
	assert.Equal(t, int64(850), fg.GetCounters()["eval_us"])
	assert.Equal(t, int64(5300), fg.GetCounters()["prefetch_us"])
}

func TestMergeStage1ExplainStripsHalfStageTimingOnMixedVersionUpgrade(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "price > 10", Type: "range", FieldID: 101, Field: "price"},
	}

	// Case 1: New Stage 1 (timed) + Old Stage 2 (untimed) must degrade cleanly to nil TimeStats.
	oldStage2 := Wrap(KindQueryNode, "old_qn", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "seg_a",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 0, "rows_out": 0},
		Children: []*internalpb.ExecutionExplain{
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0", "cached": "true"},
				Counters:   map[string]int64{"rows_in": 1000, "rows_out": 50},
			},
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "vector_search", "step": "1"},
				Counters:   map[string]int64{"rows_in": 50, "rows_out": 10},
			},
		},
	}})
	newStage1 := Wrap(KindQueryNode, "new_qn", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "seg_a",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "idx_scan": 1000, "rows_out": 50},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       KindOperator,
			Attributes: map[string]string{"name": "filter", "step": "0"},
			Counters:   map[string]int64{"rows_in": 1000, "rows_out": 50, "time_us": 900},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindFilterGroup,
				Attributes: map[string]string{"expr_id": "1", "field_id": "101", "index_id": "501", "family": "sort"},
				Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 1000, "raw_scan": 0, "rows_out": 50, "eval_us": 800, "prefetch_us": 3000},
			}},
		}},
	}})

	merged1, ok1 := MergeStage1Explain(oldStage2, newStage1)
	require.True(t, ok1)
	sum1 := Summarize(Segments(merged1), leaves)
	require.Len(t, sum1.Operators, 2)
	assert.Nil(t, sum1.Operators[0].TimeUs)
	assert.Nil(t, sum1.Operators[1].TimeUs)
	assert.Nil(t, sum1.Operators[0].Conditions[0].EvalTimeUs)
	assert.Nil(t, sum1.Operators[0].Conditions[0].PrefetchTimeUs)

	// Case 2: Old Stage 1 (untimed) + New Stage 2 (timed) must also strip Stage 2 partial timing.
	newStage2 := Wrap(KindQueryNode, "new_qn", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "seg_a",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "raw_scan": 1000, "rows_out": 50},
		Children: []*internalpb.ExecutionExplain{
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0"},
				Counters:   map[string]int64{"rows_in": 1000, "rows_out": 50, "time_us": 600},
				Children: []*internalpb.ExecutionExplain{{
					Kind:       KindFilterGroup,
					Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
					Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 0, "raw_scan": 1000, "rows_out": 50, "eval_us": 500},
				}},
			},
			{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "vector_search", "step": "1"},
				Counters:   map[string]int64{"rows_in": 50, "rows_out": 10, "time_us": 2000},
			},
		},
	}})
	oldStage1 := Wrap(KindQueryNode, "old_qn", []*internalpb.ExecutionExplain{{
		Kind:     KindSegment,
		Name:     "seg_a",
		Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "raw_scan": 1000, "rows_out": 50},
		Children: []*internalpb.ExecutionExplain{{
			Kind:       KindOperator,
			Attributes: map[string]string{"name": "filter", "step": "0"},
			Counters:   map[string]int64{"rows_in": 1000, "rows_out": 50},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindFilterGroup,
				Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
				Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 0, "raw_scan": 1000, "rows_out": 50},
			}},
		}},
	}})

	merged2, ok2 := MergeStage1Explain(newStage2, oldStage1)
	require.True(t, ok2)
	sum2 := Summarize(Segments(merged2), leaves)
	require.Len(t, sum2.Operators, 2)
	assert.Nil(t, sum2.Operators[0].TimeUs)
	assert.Nil(t, sum2.Operators[1].TimeUs)
	assert.Nil(t, sum2.Operators[0].Conditions[0].EvalTimeUs)
}

func TestMergeStage1ExplainPreservesSkippedAndSrcUniqueKeys(t *testing.T) {
	// Zero-row non-skipped Stage 2 group must not overwrite Stage 1's skipped="true",
	// and Stage 1 unique keys on re-evaluated Stage 2 segments must be retained.
	stage2 := Wrap(KindQueryNode, "qn1", []*internalpb.ExecutionExplain{
		{
			Kind:     KindSegment,
			Name:     "zero_row_s2",
			Counters: map[string]int64{"rows_total": 1000, "rows_in": 0, "rows_out": 0},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0"},
				Counters:   map[string]int64{"rows_in": 0, "rows_out": 0, "time_us": 50},
				Children: []*internalpb.ExecutionExplain{{
					Kind:       KindFilterGroup,
					Attributes: map[string]string{"expr_id": "2", "field_id": "102", "decline_reason": "no_index_on_field"},
					Counters:   map[string]int64{"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0, "eval_us": 0, "prefetch_us": 700},
				}},
			}},
		},
		{
			Kind:     KindSegment,
			Name:     "reeval_s2",
			Counters: map[string]int64{"rows_total": 500, "rows_in": 500, "raw_scan": 500, "rows_out": 0},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0"},
				Counters:   map[string]int64{"rows_in": 500, "rows_out": 0, "time_us": 400},
				Children: []*internalpb.ExecutionExplain{{
					Kind:       KindFilterGroup,
					Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
					Counters:   map[string]int64{"rows_in": 500, "idx_scan": 0, "raw_scan": 500, "rows_out": 0, "eval_us": 350},
				}},
			}},
		},
	})
	stage1 := Wrap(KindQueryNode, "qn1", []*internalpb.ExecutionExplain{
		{
			Kind:     KindSegment,
			Name:     "zero_row_s2",
			Counters: map[string]int64{"rows_total": 1000, "rows_in": 1000, "raw_scan": 1000, "rows_out": 0},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0"},
				Counters:   map[string]int64{"rows_in": 1000, "rows_out": 0, "time_us": 300},
				Children: []*internalpb.ExecutionExplain{
					{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
						Counters:   map[string]int64{"rows_in": 1000, "idx_scan": 0, "raw_scan": 1000, "rows_out": 0, "eval_us": 250},
					},
					{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "2", "field_id": "102", "skipped": "true"},
						Counters:   map[string]int64{"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0, "prefetch_us": 400},
					},
				},
			}},
		},
		{
			Kind:     KindSegment,
			Name:     "reeval_s2",
			Counters: map[string]int64{"rows_total": 500, "rows_in": 500, "raw_scan": 500, "rows_out": 20},
			Children: []*internalpb.ExecutionExplain{{
				Kind:       KindOperator,
				Attributes: map[string]string{"name": "filter", "step": "0"},
				Counters:   map[string]int64{"rows_in": 500, "rows_out": 20, "time_us": 600},
				Children: []*internalpb.ExecutionExplain{
					{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "1", "field_id": "101", "decline_reason": "no_index_on_field"},
						Counters:   map[string]int64{"rows_in": 500, "idx_scan": 0, "raw_scan": 500, "rows_out": 20, "eval_us": 300},
					},
					{
						Kind:       KindFilterGroup,
						Attributes: map[string]string{"expr_id": "9", "field_id": "109", "index_id": "901", "family": "sort"},
						Counters:   map[string]int64{"rows_in": 20, "idx_scan": 20, "raw_scan": 0, "rows_out": 5, "eval_us": 120},
					},
				},
			}},
		},
	})

	merged, ok := MergeStage1Explain(stage2, stage1)
	require.True(t, ok)
	segs := Segments(merged)
	require.Len(t, segs, 2)

	zChildren := segs[0].GetChildren()[0].GetChildren()
	require.Len(t, zChildren, 2)
	assert.Equal(t, "true", zChildren[1].GetAttributes()["skipped"])
	_, hasEvalUs := zChildren[1].GetCounters()["eval_us"]
	assert.False(t, hasEvalUs)
	assert.Equal(t, int64(1100), zChildren[1].GetCounters()["prefetch_us"])

	rChildren := segs[1].GetChildren()[0].GetChildren()
	require.Len(t, rChildren, 2)
	assert.Equal(t, "9", rChildren[1].GetAttributes()["expr_id"])
	assert.Equal(t, int64(120), rChildren[1].GetCounters()["eval_us"])
}
