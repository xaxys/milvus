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
	"context"
	"sort"
	"sync"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

const (
	OpRead   = "read"
	OpWrite  = "write"
	OpStat   = "stat"
	OpList   = "list"
	OpDelete = "delete"
	OpCopy   = "copy"

	TaskTypeFlush           = "flush"
	TaskTypeStats           = "stats"
	TaskTypeCompaction      = "compaction"
	TaskTypeImport          = "import"
	TaskTypePreImport       = "preimport"
	TaskTypeCopySegment     = "copy_segment"
	TaskTypeRefreshExternal = "refresh_external_collection"
	TaskTypeIndex           = "index"
	TaskTypeAnalyze         = "analyze"
)

var latencyBuckets = buildLatencyBuckets()

// buildLatencyBuckets combines adaptive exponential separation with classic
// latency boundaries that are commonly used in dashboards and alerts.
func buildLatencyBuckets() []float64 {
	classic := []float64{1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 30000, 60000, 120000}
	seen := make(map[float64]struct{}, len(classic)+20)
	for _, bucket := range classic {
		seen[bucket] = struct{}{}
	}
	for bucket := 0.25; bucket <= 131072; bucket *= 2 {
		seen[bucket] = struct{}{}
	}

	buckets := make([]float64, 0, len(seen))
	for bucket := range seen {
		buckets = append(buckets, bucket)
	}
	sort.Float64s(buckets)
	return buckets
}

// LatencyBuckets returns a copy of the bucket boundaries used in snapshots.
func LatencyBuckets() []float64 {
	return append([]float64(nil), latencyBuckets...)
}

type contextKey struct{}

type opKey struct {
	opType string
	status string
}

type opStats struct {
	requestCount uint64
	bytes        uint64
	latencySum   float64
	latencyMax   float64
	buckets      []uint64
}

// Collector records task-local storage access statistics without retaining
// raw latency samples or writing a second Prometheus metric family.
type Collector struct {
	mu        sync.Mutex
	stats     map[opKey]*opStats
	taskType  string
	taskID    int64
	updatedAt time.Time
}

// NewCollector creates a concurrency-safe storage access collector.
func NewCollector() *Collector {
	return &Collector{}
}

// NewTaskCollector creates and registers a task-scoped collector so its
// snapshot can be retrieved through the DataNode GetMetrics endpoint.
func NewTaskCollector(taskType string, taskID int64) *Collector {
	collector := &Collector{taskType: taskType, taskID: taskID, updatedAt: time.Now()}
	DefaultRegistry.Register(collector)
	return collector
}

// WithCollector returns a context carrying collector.
func WithCollector(ctx context.Context, collector *Collector) context.Context {
	if collector == nil {
		return ctx
	}
	return context.WithValue(ctx, contextKey{}, collector)
}

// FromContext returns the collector stored in ctx, if any.
func FromContext(ctx context.Context) *Collector {
	if ctx == nil {
		return nil
	}
	collector, _ := ctx.Value(contextKey{}).(*Collector)
	return collector
}

func statusFromError(err error) string {
	if err == nil {
		return metrics.SuccessLabel
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return metrics.CancelLabel
	}
	return metrics.FailLabel
}

// RecordAccess records one observation in the task-local collector carried by
// ctx. Prometheus accounting remains in the existing PersistentData metrics.
func RecordAccess(ctx context.Context, opType string, bytes int64, err error, elapsed time.Duration) {
	collector := FromContext(ctx)
	if collector == nil || opType == "" {
		return
	}
	latencyMs := float64(elapsed.Microseconds()) / 1000
	if latencyMs < 0 {
		latencyMs = 0
	}
	var byteCount uint64
	if bytes > 0 {
		byteCount = uint64(bytes)
	}
	collector.Record(opType, statusFromError(err), byteCount, latencyMs)
}

func (c *Collector) getOrCreateLocked(key opKey) *opStats {
	if c.stats == nil {
		c.stats = make(map[opKey]*opStats)
	}
	stats, ok := c.stats[key]
	if ok {
		return stats
	}
	stats = &opStats{buckets: make([]uint64, len(latencyBuckets))}
	c.stats[key] = stats
	return stats
}

// Record adds one operation observation.
func (c *Collector) Record(opType string, status string, bytes uint64, latencyMs float64) {
	if c == nil || opType == "" || status == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	stats := c.getOrCreateLocked(opKey{opType: opType, status: status})
	c.updatedAt = time.Now()
	stats.requestCount++
	stats.bytes += bytes
	stats.latencySum += latencyMs
	if latencyMs > stats.latencyMax {
		stats.latencyMax = latencyMs
	}
	idx := sort.SearchFloat64s(latencyBuckets, latencyMs)
	if idx >= len(stats.buckets) {
		idx = len(stats.buckets) - 1
	}
	if idx >= 0 {
		stats.buckets[idx]++
	}
}

func (c *Collector) lastUpdated() time.Time {
	if c == nil {
		return time.Time{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.updatedAt
}

// Snapshot returns cumulative bucket counts suitable for GetMetrics JSON.
func (c *Collector) Snapshot() *metricsinfo.StorageAccessStats {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.stats) == 0 {
		return nil
	}

	result := &metricsinfo.StorageAccessStats{TaskType: c.taskType, TaskID: c.taskID}
	keys := make([]opKey, 0, len(c.stats))
	for key := range c.stats {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].opType == keys[j].opType {
			return keys[i].status < keys[j].status
		}
		return keys[i].opType < keys[j].opType
	})

	for _, key := range keys {
		stats := c.stats[key]
		op := metricsinfo.StorageAccessOpStats{
			OpType:       key.opType,
			Status:       key.status,
			RequestCount: stats.requestCount,
			Bytes:        stats.bytes,
			LatencySumMs: stats.latencySum,
			LatencyMaxMs: stats.latencyMax,
		}
		var cumulative uint64
		for idx, count := range stats.buckets {
			cumulative += count
			op.LatencyBuckets = append(op.LatencyBuckets, metricsinfo.StorageAccessLatencyBucket{
				UpperBoundMs:    latencyBuckets[idx],
				CumulativeCount: cumulative,
			})
		}
		result.OpStats = append(result.OpStats, op)
		result.RequestCount += stats.requestCount
		result.Bytes += stats.bytes
		switch key.status {
		case metrics.FailLabel:
			result.FailedCount += stats.requestCount
		case metrics.CancelLabel:
			result.CanceledCount += stats.requestCount
		}
	}
	return result
}

// Quantile returns a Prometheus-style histogram quantile from cumulative buckets.
func Quantile(q float64, buckets []metricsinfo.StorageAccessLatencyBucket) float64 {
	if q < 0 || len(buckets) == 0 {
		return 0
	}
	if q > 1 {
		q = 1
	}
	total := buckets[len(buckets)-1].CumulativeCount
	if total == 0 {
		return 0
	}
	rank := q * float64(total)
	var prevCount uint64
	var prevBound float64
	for _, bucket := range buckets {
		if float64(bucket.CumulativeCount) >= rank {
			if bucket.CumulativeCount == prevCount {
				return bucket.UpperBoundMs
			}
			return prevBound + (bucket.UpperBoundMs-prevBound)*(rank-float64(prevCount))/float64(bucket.CumulativeCount-prevCount)
		}
		prevCount = bucket.CumulativeCount
		prevBound = bucket.UpperBoundMs
	}
	return buckets[len(buckets)-1].UpperBoundMs
}
