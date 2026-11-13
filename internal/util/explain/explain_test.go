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

func TestRender(t *testing.T) {
	root := Wrap(KindQuery, "", []*internalpb.ExecutionExplain{
		Wrap(KindDelegator, "ch-1", []*internalpb.ExecutionExplain{
			{
				Kind:       KindFilterGroup,
				Counters:   map[string]int64{"leaves": 2, "rows_matched": 3},
				Attributes: map[string]string{"index_id": "5", "nested": "false"},
			},
		}),
	})

	assert.Equal(t, ""+
		"query [leaves=2, rows_matched=3]\n"+
		"  delegator ch-1 [leaves=2, rows_matched=3]\n"+
		"    filter_group [leaves=2, rows_matched=3] {index_id=5, nested=false}\n",
		Render(root))
	assert.Equal(t, "", Render(nil))
}
