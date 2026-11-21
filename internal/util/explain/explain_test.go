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
