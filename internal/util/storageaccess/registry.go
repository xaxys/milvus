// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package storageaccess

import (
	"sort"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

const (
	defaultRegistryCapacity = 4096
	defaultRegistryTTL      = 30 * time.Minute
)

type registryKey struct {
	taskType string
	taskID   int64
}

type registryEntry struct {
	collector *Collector
	createdAt time.Time
}

// Registry retains a bounded, short-lived view of task collectors for GetMetrics.
type Registry struct {
	mu       sync.Mutex
	entries  map[registryKey]registryEntry
	capacity int
	ttl      time.Duration
}

// DefaultRegistry is the process-local DataNode storage-access registry.
var DefaultRegistry = NewRegistry(defaultRegistryCapacity, defaultRegistryTTL)

// NewRegistry creates a bounded registry.
func NewRegistry(capacity int, ttl time.Duration) *Registry {
	return &Registry{entries: make(map[registryKey]registryEntry), capacity: capacity, ttl: ttl}
}

// Register adds or replaces the collector for a task.
func (r *Registry) Register(collector *Collector) {
	if r == nil || collector == nil || collector.taskType == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	r.pruneLocked(now)
	key := registryKey{taskType: collector.taskType, taskID: collector.taskID}
	if _, exists := r.entries[key]; !exists && r.capacity > 0 && len(r.entries) >= r.capacity {
		r.removeOldestLocked()
	}
	r.entries[key] = registryEntry{
		collector: collector,
		createdAt: now,
	}
}

// Snapshots returns current non-empty snapshots, optionally filtered by task.
func (r *Registry) Snapshots(taskType string, taskID int64) []*metricsinfo.StorageAccessStats {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	r.pruneLocked(time.Now())
	entries := make([]registryEntry, 0, len(r.entries))
	for key, entry := range r.entries {
		if taskType != "" && key.taskType != taskType {
			continue
		}
		if taskID != 0 && key.taskID != taskID {
			continue
		}
		entries = append(entries, entry)
	}
	r.mu.Unlock()

	result := make([]*metricsinfo.StorageAccessStats, 0, len(entries))
	for _, entry := range entries {
		if snapshot := entry.collector.Snapshot(); snapshot != nil {
			result = append(result, snapshot)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].TaskType == result[j].TaskType {
			return result[i].TaskID < result[j].TaskID
		}
		return result[i].TaskType < result[j].TaskType
	})
	return result
}

func (r *Registry) pruneLocked(now time.Time) {
	if r.ttl <= 0 {
		return
	}
	for key, entry := range r.entries {
		updatedAt := entry.collector.lastUpdated()
		if updatedAt.IsZero() {
			updatedAt = entry.createdAt
		}
		if now.Sub(updatedAt) >= r.ttl {
			delete(r.entries, key)
		}
	}
}

func (r *Registry) removeOldestLocked() {
	var oldestKey registryKey
	var oldestTime time.Time
	for key, entry := range r.entries {
		if oldestTime.IsZero() || entry.createdAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.createdAt
		}
	}
	if !oldestTime.IsZero() {
		delete(r.entries, oldestKey)
	}
}
