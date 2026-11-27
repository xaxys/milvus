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

func TestMergeAlignsOperatorsByKindAcrossHeterogeneousShards(t *testing.T) {
	// Shard A runs pre-filter: filter -> mvcc -> vector_search
	shardA := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       2,
			SealedSegments: 2,
			RowsTotal:      1000,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments: 2,
					RowsIn:   1000,
					RowsOut:  80,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:   1,
						FieldId:  101,
						Segments: 2,
						RowsIn:   1000,
						IdxScan:  1000,
						RowsOut:  80,
						Used: []*internalpb.ExplainIndexUseStats{{
							IndexId:  501,
							Family:   "sort",
							Segments: 2,
							RowsIn:   1000,
							IdxScan:  1000,
							RowsOut:  80,
						}},
					}},
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
					Segments: 2,
					RowsIn:   80,
					RowsOut:  80,
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
					Segments: 2,
					RowsIn:   80,
					RowsOut:  20,
				},
			},
		},
	}

	// Shard B runs post-filter on 1 growing segment: mvcc -> vector_search -> iterative_filter
	shardB := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:        1,
			GrowingSegments: 1,
			RowsTotal:       500,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
					Segments: 1,
					RowsIn:   500,
					RowsOut:  490,
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
					Segments: 1,
					RowsIn:   490,
					RowsOut:  50,
				},
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_FILTER,
					Segments: 1,
					RowsIn:   50,
					RowsOut:  10,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:          1,
						FieldId:         101,
						Segments:        1,
						RawScanSegments: 1,
						RowsIn:          50,
						RawScan:         50,
						RowsOut:         10,
						Reasons: []*internalpb.ExplainDeclineStats{{
							Reason:   "growing",
							Segments: 1,
							RowsIn:   50,
							RawScan:  50,
							RowsOut:  10,
						}},
					}},
				},
			},
		},
	}

	merged := Merge(shardA, nil, shardB)
	require.NotNil(t, merged)
	g := merged.GetGroup()
	assert.Equal(t, int64(3), g.GetSegments())
	assert.Equal(t, int64(2), g.GetSealedSegments())
	assert.Equal(t, int64(1), g.GetGrowingSegments())
	assert.Equal(t, int64(1500), g.GetRowsTotal())

	// Global topological DAG order: filter(1) -> mvcc(3) -> vector_search(5) -> iterative_filter(7)
	// No duplicate mvcc or vector_search rows!
	require.Len(t, g.GetOperators(), 4)
	assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER, g.GetOperators()[0].GetKind())
	assert.Equal(t, int64(2), g.GetOperators()[0].GetSegments())
	assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC, g.GetOperators()[1].GetKind())
	assert.Equal(t, int64(3), g.GetOperators()[1].GetSegments())
	assert.Equal(t, int64(580), g.GetOperators()[1].GetRowsIn())
	assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH, g.GetOperators()[2].GetKind())
	assert.Equal(t, int64(3), g.GetOperators()[2].GetSegments())
	assert.Equal(t, int64(570), g.GetOperators()[2].GetRowsIn())
	assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_FILTER, g.GetOperators()[3].GetKind())
	assert.Equal(t, int64(1), g.GetOperators()[3].GetSegments())
}

func TestWrapSubSearchAndAssembleHybridExplainDeduplicatesAndStripsRefs(t *testing.T) {
	// Worker 1 is homogeneous (untracked: 2 sealed segments, 200 rows)
	w1 := &internalpb.ExecutionExplain{
		Version: 1,
		Group:   &internalpb.ExplainSegmentGroup{Segments: 2, SealedSegments: 2, RowsTotal: 200},
	}
	// Worker 2 has divergent sub-selections (tracked via SegmentRefs)
	w2Sub0 := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       1,
			SealedSegments: 1,
			RowsTotal:      100,
			SegmentRefs: []*internalpb.ExplainSegmentRef{
				{SegmentId: 10, RowsTotal: 100, IsGrowing: false},
			},
		},
	}
	w2Sub1 := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:        2,
			SealedSegments:  1,
			GrowingSegments: 1,
			RowsTotal:       150,
			SegmentRefs: []*internalpb.ExplainSegmentRef{
				{SegmentId: 10, RowsTotal: 100, IsGrowing: false},
				{SegmentId: 11, RowsTotal: 50, IsGrowing: true},
			},
		},
	}

	sub0 := WrapSubSearch(0, []*internalpb.ExecutionExplain{w1, w2Sub0})
	sub1 := WrapSubSearch(1, []*internalpb.ExecutionExplain{w1, w2Sub1})
	assembled := AssembleHybridExplain(sub0, sub1)
	require.NotNil(t, assembled)
	// Exact union: 2 untracked sealed (200) + {seg10 sealed (100), seg11 growing (50)} = 4 segs (3 sealed, 1 growing, 350 rows)
	assert.Equal(t, int64(4), assembled.GetGroup().GetSegments())
	assert.Equal(t, int64(3), assembled.GetGroup().GetSealedSegments())
	assert.Equal(t, int64(1), assembled.GetGroup().GetGrowingSegments())
	assert.Equal(t, int64(350), assembled.GetGroup().GetRowsTotal())
	assert.Empty(t, assembled.GetGroup().GetSegmentRefs())
	for _, sub := range assembled.GetSubSearches() {
		assert.Empty(t, sub.GetGroup().GetSegmentRefs())
	}

	// Divergent untracked sub-searches (e.g., ignoreGrowing=true on sub0 vs false on sub1)
	// preserve Segments >= SealedSegments + GrowingSegments.
	uSub0 := WrapSubSearch(0, []*internalpb.ExecutionExplain{{
		Version: 1,
		Group:   &internalpb.ExplainSegmentGroup{Segments: 2, SealedSegments: 2, RowsTotal: 200},
	}})
	uSub1 := WrapSubSearch(1, []*internalpb.ExecutionExplain{{
		Version: 1,
		Group:   &internalpb.ExplainSegmentGroup{Segments: 2, SealedSegments: 1, GrowingSegments: 1, RowsTotal: 150},
	}})
	uAssembled := AssembleHybridExplain(uSub0, uSub1)
	assert.Equal(t, int64(3), uAssembled.GetGroup().GetSegments())
	assert.Equal(t, int64(2), uAssembled.GetGroup().GetSealedSegments())
	assert.Equal(t, int64(1), uAssembled.GetGroup().GetGrowingSegments())

	// Mixed pre-assembled node (Group != nil) + ungrouped WrapSubSearch nodes (Group == nil)
	// sums both shards into the top-level Group without dropping the ungrouped shard.
	mixed := AssembleHybridExplain(assembled, uSub0, uSub1)
	require.NotNil(t, mixed)
	assert.Equal(t, int64(7), mixed.GetGroup().GetSegments())
	assert.Equal(t, int64(5), mixed.GetGroup().GetSealedSegments())
	assert.Equal(t, int64(2), mixed.GetGroup().GetGrowingSegments())
	assert.Equal(t, int64(550), mixed.GetGroup().GetRowsTotal())
}

func TestMergeStage1ExplainStatelessModes(t *testing.T) {
	stage1Cold := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       2,
			SealedSegments: 2,
			RowsTotal:      1000,
			Operators: []*internalpb.ExplainOperatorStats{{
				Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
				Segments: 2,
				RowsIn:   1000,
				RowsOut:  80,
				Conditions: []*internalpb.ExplainConditionStats{{
					ExprId:   1,
					FieldId:  101,
					Segments: 2,
					RowsIn:   1000,
					IdxScan:  1000,
					RowsOut:  80,
					Used: []*internalpb.ExplainIndexUseStats{{
						IndexId:  5001,
						Family:   "sort",
						Segments: 2,
						RowsIn:   1000,
						IdxScan:  1000,
						RowsOut:  80,
					}},
				}},
			}},
		},
	}

	t.Run("pre-filter Stage 2 with intra-query sealed cache + cold growing segment", func(t *testing.T) {
		stage2Sealed := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      1000,
				Operators: []*internalpb.ExplainOperatorStats{
					{
						Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
						Segments:       2,
						CachedSegments: 2,
						RowsIn:         1000,
						RowsOut:        80,
					},
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
						Segments: 2,
						RowsIn:   80,
						RowsOut:  80,
					},
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
						Segments: 2,
						RowsIn:   80,
						RowsOut:  8,
					},
				},
			},
		}
		growingStage2 := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:        1,
				GrowingSegments: 1,
				RowsTotal:       200,
				Operators: []*internalpb.ExplainOperatorStats{
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
						Segments: 1,
						RowsIn:   200,
						RowsOut:  15,
						Conditions: []*internalpb.ExplainConditionStats{{
							ExprId:          1,
							FieldId:         101,
							Segments:        1,
							RawScanSegments: 1,
							RowsIn:          200,
							RawScan:         200,
							RowsOut:         15,
							Reasons: []*internalpb.ExplainDeclineStats{{
								Reason:   "growing",
								Segments: 1,
								RowsIn:   200,
								RawScan:  200,
								RowsOut:  15,
							}},
						}},
					},
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
						Segments: 1,
						RowsIn:   15,
						RowsOut:  15,
					},
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
						Segments: 1,
						RowsIn:   15,
						RowsOut:  2,
					},
				},
			},
		}

		mergedSealed, ok := MergeStage1Explain(stage2Sealed, stage1Cold)
		require.True(t, ok)
		merged := Merge(mergedSealed, growingStage2)
		filterOp := merged.GetGroup().GetOperators()[0]
		assert.Equal(t, int64(3), filterOp.GetSegments())
		assert.Zero(t, filterOp.GetCachedSegments())
		assert.Equal(t, int64(1200), filterOp.GetRowsIn())
		assert.Equal(t, int64(95), filterOp.GetRowsOut())
		require.Len(t, filterOp.GetConditions(), 1)
		cond := filterOp.GetConditions()[0]
		assert.Equal(t, int64(3), cond.GetSegments())
		assert.Equal(t, int64(1000), cond.GetIdxScan())
		assert.Equal(t, int64(200), cond.GetRawScan())
		require.Len(t, cond.GetUsed(), 1)
		require.Len(t, cond.GetReasons(), 1)
		assert.Equal(t, "growing", cond.GetReasons()[0].GetReason())
	})

	t.Run("post-filter Stage 2 switched to iterative_filter inserts Stage 1 filter in topological order", func(t *testing.T) {
		stage2Iter := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      1000,
				Operators: []*internalpb.ExplainOperatorStats{
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC,
						Segments: 2,
						RowsIn:   1000,
						RowsOut:  1000,
					},
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
						Segments: 2,
						RowsIn:   1000,
						RowsOut:  40,
					},
					{
						Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_FILTER,
						Segments: 2,
						RowsIn:   40,
						RowsOut:  10,
					},
				},
			},
		}

		merged, ok := MergeStage1Explain(stage2Iter, stage1Cold)
		require.True(t, ok)
		ops := merged.GetGroup().GetOperators()
		require.Len(t, ops, 4)
		assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER, ops[0].GetKind())
		assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_MVCC, ops[1].GetKind())
		assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH, ops[2].GetKind())
		assert.Equal(t, internalpb.ExplainOperatorKind_EXPLAIN_OP_ITERATIVE_FILTER, ops[3].GetKind())
	})

	t.Run("partial Stage 2 failure with fewer sealed segments skips full Stage 1 overwrite", func(t *testing.T) {
		partialStage2 := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       1,
				SealedSegments: 1,
				RowsTotal:      500,
				Operators: []*internalpb.ExplainOperatorStats{
					{
						Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
						Segments:       1,
						CachedSegments: 1,
						RowsIn:         500,
						RowsOut:        40,
					},
				},
			},
		}
		_, ok := MergeStage1Explain(partialStage2, stage1Cold)
		assert.False(t, ok)
	})

	t.Run("0-row sealed segment or partial cache eviction replaces sealed filter cleanly without double-counting", func(t *testing.T) {
		// Stage 1: 2 sealed segments (1 active segment with 500 rows, 1 empty 0-row segment -> Conditions.Segments=1).
		s1WithEmpty := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      500,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments: 2,
					RowsIn:   500,
					RowsOut:  40,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:   1,
						FieldId:  101,
						Segments: 1,
						RowsIn:   500,
						IdxScan:  500,
						RowsOut:  40,
					}},
				}},
			},
		}
		s2SealedWithEmpty := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      500,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:       2,
					CachedSegments: 1,
					RowsIn:         500,
					RowsOut:        40,
				}},
			},
		}
		growing := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:        1,
				GrowingSegments: 1,
				RowsTotal:       100,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments: 1,
					RowsIn:   100,
					RowsOut:  10,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:          1,
						FieldId:         101,
						Segments:        1,
						RawScanSegments: 1,
						RowsIn:          100,
						RawScan:         100,
						RowsOut:         10,
					}},
				}},
			},
		}

		mergedSealed, ok := MergeStage1Explain(s2SealedWithEmpty, s1WithEmpty)
		require.True(t, ok)
		merged := Merge(mergedSealed, growing)
		filterOp := merged.GetGroup().GetOperators()[0]
		assert.Equal(t, int64(3), filterOp.GetSegments())
		assert.Equal(t, int64(600), filterOp.GetRowsIn())
		assert.Equal(t, int64(50), filterOp.GetRowsOut())
		require.Len(t, filterOp.GetConditions(), 1)
		assert.Equal(t, int64(2), filterOp.GetConditions()[0].GetSegments())

		// Sealed-only QueryNode partial cache eviction: replaces with Stage 1's
		// complete sealed stats without double-counting the evicted segment.
		s2SealedPartialEvict := &internalpb.ExecutionExplain{
			Version: 1,
			Group: &internalpb.ExplainSegmentGroup{
				Segments:       2,
				SealedSegments: 2,
				RowsTotal:      1000,
				Operators: []*internalpb.ExplainOperatorStats{{
					Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:       2,
					CachedSegments: 1,
					RowsIn:         1000,
					RowsOut:        80,
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:   1,
						FieldId:  101,
						Segments: 1,
						RowsIn:   500,
						IdxScan:  500,
						RowsOut:  40,
					}},
				}},
			},
		}
		mergedEvict, ok := MergeStage1Explain(s2SealedPartialEvict, stage1Cold)
		require.True(t, ok)
		evictOp := mergedEvict.GetGroup().GetOperators()[0]
		assert.Equal(t, int64(2), evictOp.GetSegments())
		assert.Equal(t, int64(1000), evictOp.GetRowsIn())
		assert.Equal(t, int64(80), evictOp.GetRowsOut())
		assert.Equal(t, int64(2), evictOp.GetConditions()[0].GetSegments())
	})
}

func TestMergeStage1ExplainTimingAndPrefetchWait(t *testing.T) {
	stage1 := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       2,
			SealedSegments: 2,
			RowsTotal:      1000,
			Operators: []*internalpb.ExplainOperatorStats{{
				Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
				Segments:         2,
				RowsIn:           1000,
				RowsOut:          80,
				Time:             &internalpb.ExplainTimeAccum{Count: 2, SumUs: 3000, MinUs: 1000, MaxUs: 2000},
				PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 2, SumUs: 1200, MinUs: 400, MaxUs: 800},
				Conditions: []*internalpb.ExplainConditionStats{{
					ExprId:       1,
					FieldId:      101,
					Segments:     2,
					RowsIn:       1000,
					IdxScan:      1000,
					RowsOut:      80,
					EvalTime:     &internalpb.ExplainTimeAccum{Count: 2, SumUs: 1600, MinUs: 600, MaxUs: 1000},
					PrefetchTime: &internalpb.ExplainTimeAccum{Count: 2, SumUs: 5000, MinUs: 2000, MaxUs: 3000},
					Used: []*internalpb.ExplainIndexUseStats{{
						IndexId:      5001,
						Family:       "sort",
						Segments:     2,
						RowsIn:       1000,
						IdxScan:      1000,
						RowsOut:      80,
						EvalTime:     &internalpb.ExplainTimeAccum{Count: 2, SumUs: 1600, MinUs: 600, MaxUs: 1000},
						PrefetchTime: &internalpb.ExplainTimeAccum{Count: 2, SumUs: 5000, MinUs: 2000, MaxUs: 3000},
					}},
				}},
			}},
		},
	}

	// Stage 2 ran pre-filter on 2 sealed segments (reused Stage 1 whole-filter cache)
	// plus 1 cold growing segment merged separately after Stage 1 sealed replacement.
	stage2SealedPre := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       2,
			SealedSegments: 2,
			RowsTotal:      1000,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:         2,
					CachedSegments:   2,
					RowsIn:           1000,
					RowsOut:          80,
					Time:             &internalpb.ExplainTimeAccum{Count: 2, SumUs: 22, MinUs: 10, MaxUs: 12},
					PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 2, SumUs: 0, MinUs: 0, MaxUs: 0},
				},
				{
					Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
					Segments:         2,
					RowsIn:           80,
					RowsOut:          10,
					Time:             &internalpb.ExplainTimeAccum{Count: 2, SumUs: 3000, MinUs: 1200, MaxUs: 1800},
					PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 2, SumUs: 300, MinUs: 0, MaxUs: 300},
				},
			},
		},
	}
	stage2Growing := &internalpb.ExecutionExplain{
		Version: 1,
		Group: &internalpb.ExplainSegmentGroup{
			Segments:        1,
			GrowingSegments: 1,
			RowsTotal:       500,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:         1,
					RowsIn:           500,
					RowsOut:          40,
					Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 600, MinUs: 600, MaxUs: 600},
					PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
					Conditions: []*internalpb.ExplainConditionStats{{
						ExprId:          1,
						FieldId:         101,
						Segments:        1,
						RawScanSegments: 1,
						RowsIn:          500,
						RawScan:         500,
						RowsOut:         40,
						EvalTime:        &internalpb.ExplainTimeAccum{Count: 1, SumUs: 500, MinUs: 500, MaxUs: 500},
						PrefetchTime:    &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
						Reasons: []*internalpb.ExplainDeclineStats{{
							Reason:       "growing",
							Segments:     1,
							RowsIn:       500,
							RawScan:      500,
							RowsOut:      40,
							EvalTime:     &internalpb.ExplainTimeAccum{Count: 1, SumUs: 500, MinUs: 500, MaxUs: 500},
							PrefetchTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
						}},
					}},
				},
				{
					Kind:             internalpb.ExplainOperatorKind_EXPLAIN_OP_VECTOR_SEARCH,
					Segments:         1,
					RowsIn:           40,
					RowsOut:          10,
					Time:             &internalpb.ExplainTimeAccum{Count: 1, SumUs: 1500, MinUs: 1500, MaxUs: 1500},
					PrefetchWaitTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 0, MinUs: 0, MaxUs: 0},
				},
			},
		},
	}

	mergedSealed, ok := MergeStage1Explain(stage2SealedPre, stage1)
	require.True(t, ok)
	merged := Merge(mergedSealed, stage2Growing)

	summary := Summarize([]*internalpb.ExecutionExplain{merged}, map[int64]LeafInfo{
		1: {Condition: "price > 10", Type: "range", FieldID: 101, Field: "price"},
	})
	require.Len(t, summary.Operators, 2)
	fOp := summary.Operators[0]
	assert.Equal(t, &TimeStats{SumUs: 3600, MinUs: 600, AvgUs: 1200, MaxUs: 2000}, fOp.TimeUs)
	assert.Equal(t, &TimeStats{SumUs: 1200, MinUs: 0, AvgUs: 400, MaxUs: 800}, fOp.PrefetchWaitTimeUs)
	require.Len(t, fOp.Conditions, 1)
	cond := fOp.Conditions[0]
	assert.Equal(t, &TimeStats{SumUs: 2100, MinUs: 500, AvgUs: 700, MaxUs: 1000}, cond.EvalTimeUs)
	assert.Equal(t, &TimeStats{SumUs: 5000, MinUs: 0, AvgUs: 1666, MaxUs: 3000}, cond.PrefetchTimeUs)

	// Over-counted accumulator (Count > expectedDenom) is suppressed rather than emitted
	// with a mismatched denominator.
	assert.Nil(t, primaryTimeStats(&internalpb.ExplainTimeAccum{Count: 3, SumUs: 300, MinUs: 10, MaxUs: 250}, 2))

	// Zero-row sealed segments bypass ExprResCacheManager in both stages and record into
	// op.Time; MergeStage1Explain replaces s2Op directly with s1Op without double-counting.
	s1WithZeroRow := &internalpb.ExecutionExplain{
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       2,
			SealedSegments: 2,
			RowsTotal:      100,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:     internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments: 2,
					RowsIn:   100,
					RowsOut:  20,
					Time:     &internalpb.ExplainTimeAccum{Count: 2, SumUs: 502, MinUs: 2, MaxUs: 500},
					Conditions: []*internalpb.ExplainConditionStats{
						{
							ExprId:   1,
							FieldId:  101,
							Segments: 1,
							RowsIn:   100,
							RowsOut:  20,
							IdxScan:  100,
							EvalTime: &internalpb.ExplainTimeAccum{Count: 1, SumUs: 480, MinUs: 480, MaxUs: 480},
						},
					},
				},
			},
		},
	}
	s2WithZeroRow := &internalpb.ExecutionExplain{
		Group: &internalpb.ExplainSegmentGroup{
			Segments:       2,
			SealedSegments: 2,
			RowsTotal:      100,
			Operators: []*internalpb.ExplainOperatorStats{
				{
					Kind:           internalpb.ExplainOperatorKind_EXPLAIN_OP_FILTER,
					Segments:       2,
					CachedSegments: 1,
					RowsIn:         100,
					RowsOut:        20,
					Time:           &internalpb.ExplainTimeAccum{Count: 2, SumUs: 6, MinUs: 1, MaxUs: 5},
				},
			},
		},
	}
	mergedZeroRow, ok := MergeStage1Explain(s2WithZeroRow, s1WithZeroRow)
	require.True(t, ok)
	zeroRowSummary := Summarize([]*internalpb.ExecutionExplain{mergedZeroRow}, map[int64]LeafInfo{
		1: {Condition: "price > 10", Type: "range", FieldID: 101, Field: "price"},
	})
	require.Len(t, zeroRowSummary.Operators, 1)
	assert.Equal(t, 2, zeroRowSummary.Operators[0].Segments)
	assert.Equal(t, &TimeStats{SumUs: 502, MinUs: 2, AvgUs: 251, MaxUs: 500}, zeroRowSummary.Operators[0].TimeUs)
}
