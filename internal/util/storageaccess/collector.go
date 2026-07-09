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
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

const (
	OpRead       = "read"
	OpWrite      = "write"
	OpStat       = "stat"
	OpList       = "list"
	OpDelete     = "delete"
	OpCopy       = "copy"
	OpCreateDir  = "create_dir"
	OpDeleteDir  = "delete_dir"
	OpDeleteFile = "delete_file"
	OpMove       = "move"
)

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

// Collector records task-local storage access stats without retaining raw samples.
type Collector struct {
	mu    sync.Mutex
	stats map[opKey]*opStats
}

// NewCollector creates a concurrency-safe storage access collector.
func NewCollector() *Collector {
	return &Collector{
		stats: make(map[opKey]*opStats),
	}
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

// RecordAccess records one storage access observation to Prometheus and to the
// task-local collector stored in ctx, if present.
func RecordAccess(ctx context.Context, opType string, bytes int64, err error, start time.Time) {
	if opType == "" {
		return
	}
	status := statusFromError(err)
	latencyMs := float64(time.Since(start).Microseconds()) / 1000
	if latencyMs < 0 {
		latencyMs = 0
	}
	var byteCount uint64
	if bytes > 0 {
		byteCount = uint64(bytes)
	}
	metrics.StorageAccessOpCount.WithLabelValues(opType, status).Inc()
	metrics.StorageAccessRequestLatency.WithLabelValues(opType, status).Observe(latencyMs)
	if byteCount > 0 {
		metrics.StorageAccessBytes.WithLabelValues(opType, status).Add(float64(byteCount))
	}
	if collector := FromContext(ctx); collector != nil {
		collector.Record(opType, status, byteCount, latencyMs)
	}
}

func (c *Collector) getOrCreateLocked(key opKey) *opStats {
	stats, ok := c.stats[key]
	if ok {
		return stats
	}
	stats = &opStats{
		buckets: make([]uint64, len(metrics.StorageAccessLatencyBucketsMs)),
	}
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
	stats.requestCount++
	stats.bytes += bytes
	stats.latencySum += latencyMs
	if latencyMs > stats.latencyMax {
		stats.latencyMax = latencyMs
	}
	if idx := sort.SearchFloat64s(metrics.StorageAccessLatencyBucketsMs, latencyMs); idx < len(stats.buckets) {
		stats.buckets[idx]++
	} else if len(stats.buckets) > 0 {
		stats.buckets[len(stats.buckets)-1]++
	}
}

// Snapshot returns cumulative bucket counts suitable for wire transport.
func (c *Collector) Snapshot() *datapb.StorageAccessStats {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.stats) == 0 {
		return nil
	}

	result := &datapb.StorageAccessStats{}
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
		op := &datapb.StorageAccessOpStats{
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
			op.LatencyBuckets = append(op.LatencyBuckets, &datapb.StorageAccessLatencyBucket{
				UpperBoundMs:    metrics.StorageAccessLatencyBucketsMs[idx],
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

// Merge records a wire snapshot into the collector.
func (c *Collector) Merge(stats *datapb.StorageAccessStats) {
	if c == nil || stats == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, incoming := range stats.GetOpStats() {
		if incoming.GetOpType() == "" || incoming.GetStatus() == "" {
			continue
		}
		current := c.getOrCreateLocked(opKey{opType: incoming.GetOpType(), status: incoming.GetStatus()})
		current.requestCount += incoming.GetRequestCount()
		current.bytes += incoming.GetBytes()
		current.latencySum += incoming.GetLatencySumMs()
		if incoming.GetLatencyMaxMs() > current.latencyMax {
			current.latencyMax = incoming.GetLatencyMaxMs()
		}

		var previous uint64
		for idx, bucket := range incoming.GetLatencyBuckets() {
			if idx >= len(current.buckets) {
				break
			}
			cumulative := bucket.GetCumulativeCount()
			if cumulative < previous {
				previous = cumulative
				continue
			}
			current.buckets[idx] += cumulative - previous
			previous = cumulative
		}
	}
}

// Quantile returns a Prometheus-style histogram quantile from cumulative buckets.
func Quantile(q float64, buckets []*datapb.StorageAccessLatencyBucket) float64 {
	if q < 0 || len(buckets) == 0 {
		return 0
	}
	if q > 1 {
		q = 1
	}
	total := buckets[len(buckets)-1].GetCumulativeCount()
	if total == 0 {
		return 0
	}
	rank := q * float64(total)
	var prevCount uint64
	var prevBound float64
	for _, bucket := range buckets {
		count := bucket.GetCumulativeCount()
		if float64(count) >= rank {
			if count == prevCount {
				return bucket.GetUpperBoundMs()
			}
			return prevBound + (bucket.GetUpperBoundMs()-prevBound)*(rank-float64(prevCount))/float64(count-prevCount)
		}
		prevCount = count
		prevBound = bucket.GetUpperBoundMs()
	}
	return buckets[len(buckets)-1].GetUpperBoundMs()
}
