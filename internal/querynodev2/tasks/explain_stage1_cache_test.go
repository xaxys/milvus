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

package tasks

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/util/explain"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

func TestStage1ExplainCacheMultiVChannelTTLAndEviction(t *testing.T) {
	now := time.Unix(1000, 0)
	cache := newStage1ExplainCache(60*time.Second, 2, func() time.Time { return now })

	seg := func(name string) *internalpb.ExecutionExplain {
		return &internalpb.ExecutionExplain{Kind: explain.KindSegment, Name: name}
	}

	// Zero msgID is ignored.
	cache.Put(0, map[int64]*internalpb.ExecutionExplain{1: seg("1")})
	assert.Nil(t, cache.PopSegments(0, []int64{1}))

	// Concurrent vchannels sharing the same msgID merge disjoint segment maps
	// and pop only their own segments.
	cache.Put(100, map[int64]*internalpb.ExecutionExplain{10: seg("10"), 11: seg("11")})
	cache.Put(100, map[int64]*internalpb.ExecutionExplain{20: seg("20")})

	// Streaming sub-task with non-overlapping growing segment IDs must not evict sealed entries.
	assert.Empty(t, cache.PopSegments(100, []int64{999}))

	gotVch0 := cache.PopSegments(100, []int64{10, 11})
	require.Len(t, gotVch0, 2)
	assert.Empty(t, cache.PopSegments(100, []int64{10}))

	gotVch1 := cache.PopSegments(100, []int64{20})
	require.Len(t, gotVch1, 1)
	assert.Equal(t, "20", gotVch1[0].GetName())

	// Capacity cap evicts oldest msgID.
	cache.Put(201, map[int64]*internalpb.ExecutionExplain{1: seg("1")})
	now = now.Add(time.Second)
	cache.Put(202, map[int64]*internalpb.ExecutionExplain{2: seg("2")})
	now = now.Add(time.Second)
	cache.Put(203, map[int64]*internalpb.ExecutionExplain{3: seg("3")})
	assert.Nil(t, cache.PopSegments(201, []int64{1}))
	require.Len(t, cache.PopSegments(202, []int64{2}), 1)

	// TTL expiry cleans stale entries.
	now = now.Add(61 * time.Second)
	assert.Nil(t, cache.PopSegments(203, []int64{3}))

	assert.Equal(t, "a,b/7/Historical", explainGroupName([]string{"b", "a"}, 7, querypb.DataScope_Historical))
}
