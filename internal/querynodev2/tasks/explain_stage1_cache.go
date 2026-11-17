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
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

const (
	stage1ExplainCacheTTL        = 60 * time.Second
	stage1ExplainCacheMaxEntries = 512
)

type stage1ExplainEntry struct {
	updatedAt time.Time
	segments  map[int64]*internalpb.ExecutionExplain
}

// stage1ExplainCache temporarily retains per-segment Stage-1 filter explains on
// the worker QueryNode so Stage 2 of a two-stage search can pair Stage 1 and
// Stage 2 per segment locally and return a single pre-aggregated segment_group
// over the wire.
type stage1ExplainCache struct {
	mu         sync.Mutex
	ttl        time.Duration
	maxEntries int
	now        func() time.Time
	entries    map[int64]*stage1ExplainEntry
}

func newStage1ExplainCache(ttl time.Duration, maxEntries int, now func() time.Time) *stage1ExplainCache {
	return &stage1ExplainCache{
		ttl:        ttl,
		maxEntries: maxEntries,
		now:        now,
		entries:    make(map[int64]*stage1ExplainEntry),
	}
}

var defaultStage1ExplainCache = newStage1ExplainCache(
	stage1ExplainCacheTTL,
	stage1ExplainCacheMaxEntries,
	time.Now,
)

func (c *stage1ExplainCache) sweepLocked(now time.Time) {
	for msgID, entry := range c.entries {
		if now.Sub(entry.updatedAt) >= c.ttl {
			delete(c.entries, msgID)
		}
	}
}

func (c *stage1ExplainCache) Put(msgID int64, bySeg map[int64]*internalpb.ExecutionExplain) {
	if msgID == 0 || len(bySeg) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	now := c.now()
	c.sweepLocked(now)

	entry := c.entries[msgID]
	if entry == nil {
		for len(c.entries) >= c.maxEntries {
			var oldestID int64
			var oldestTime time.Time
			first := true
			for id, e := range c.entries {
				if first || e.updatedAt.Before(oldestTime) {
					oldestID = id
					oldestTime = e.updatedAt
					first = false
				}
			}
			if first {
				break
			}
			delete(c.entries, oldestID)
		}
		entry = &stage1ExplainEntry{
			segments: make(map[int64]*internalpb.ExecutionExplain, len(bySeg)),
		}
		c.entries[msgID] = entry
	}
	entry.updatedAt = now
	for segID, node := range bySeg {
		if node != nil {
			entry.segments[segID] = node
		}
	}
}

func (c *stage1ExplainCache) PopSegments(msgID int64, segIDs []int64) []*internalpb.ExecutionExplain {
	if msgID == 0 || len(segIDs) == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	now := c.now()
	c.sweepLocked(now)

	entry := c.entries[msgID]
	if entry == nil {
		return nil
	}
	out := make([]*internalpb.ExecutionExplain, 0, len(segIDs))
	for _, segID := range segIDs {
		if node := entry.segments[segID]; node != nil {
			out = append(out, node)
			delete(entry.segments, segID)
		}
	}
	if len(entry.segments) == 0 {
		delete(c.entries, msgID)
	}
	return out
}

func explainGroupName(channels []string, nodeID int64, scope querypb.DataScope) string {
	var ch string
	switch len(channels) {
	case 0:
	case 1:
		ch = channels[0]
	default:
		sorted := append([]string(nil), channels...)
		sort.Strings(sorted)
		ch = strings.Join(sorted, ",")
	}
	return fmt.Sprintf("%s/%d/%s", ch, nodeID, scope.String())
}
