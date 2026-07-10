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

package datacoord

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/internal/util/storageaccess"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

const (
	storageAccessTaskFlush           = "flush"
	storageAccessTaskStats           = "stats"
	storageAccessTaskCompaction      = "compaction"
	storageAccessTaskImport          = "import"
	storageAccessTaskPreImport       = "preimport"
	storageAccessTaskCopySegment     = "copy_segment"
	storageAccessTaskRefreshExternal = "refresh_external_collection"
	storageAccessTaskIndex           = "index"
	storageAccessTaskAnalyze         = "analyze"

	maxStorageAccessFlushFingerprints = 100000
)

func reportStorageAccessStats(ctx context.Context, taskType string, taskID int64, stats *datapb.StorageAccessStats) {
	if stats == nil || taskType == "" {
		return
	}
	if ctx == nil {
		ctx = context.TODO()
	}
	correlatedStats := stats
	if taskID != 0 && stats.GetTaskId() == 0 {
		copy := *stats
		copy.TaskId = taskID
		correlatedStats = &copy
	}
	for _, opStats := range correlatedStats.GetOpStats() {
		if opStats.GetOpType() == "" || opStats.GetStatus() == "" || opStats.GetRequestCount() == 0 {
			continue
		}
		metrics.DataCoordTaskStorageAccessOpCount.WithLabelValues(taskType, opStats.GetOpType(), opStats.GetStatus()).
			Add(float64(opStats.GetRequestCount()))
		if opStats.GetBytes() > 0 {
			metrics.DataCoordTaskStorageAccessBytes.WithLabelValues(taskType, opStats.GetOpType(), opStats.GetStatus()).
				Add(float64(opStats.GetBytes()))
		}

		var previous uint64
		for _, bucket := range opStats.GetLatencyBuckets() {
			cumulative := bucket.GetCumulativeCount()
			if cumulative < previous {
				previous = cumulative
				continue
			}
			for i := uint64(0); i < cumulative-previous; i++ {
				metrics.DataCoordTaskStorageAccessLatency.WithLabelValues(taskType, opStats.GetOpType(), opStats.GetStatus()).
					Observe(bucket.GetUpperBoundMs())
			}
			previous = cumulative
		}

		_ = summarizeStorageAccessOpStats(opStats)
	}
	if correlatedStats.GetRequestCount() == 0 {
		return
	}
	fields := []mlog.Field{mlog.String("taskType", taskType)}
	fields = append(fields, storageaccess.LogFields(correlatedStats)...)
	mlog.Info(ctx, "task storage access stats coordinated", fields...)
	storageaccess.AddTraceEvent(ctx, "storage_access.task.coordinated", correlatedStats)
}

func summarizeStorageAccessOpStats(opStats *datapb.StorageAccessOpStats) storageAccessSummary {
	if opStats == nil || opStats.GetRequestCount() == 0 {
		return storageAccessSummary{}
	}
	return storageAccessSummary{
		avgLatencyMs: opStats.GetLatencySumMs() / float64(opStats.GetRequestCount()),
		maxLatencyMs: opStats.GetLatencyMaxMs(),
		p95Ms:        storageaccess.Quantile(0.95, opStats.GetLatencyBuckets()),
		p99Ms:        storageaccess.Quantile(0.99, opStats.GetLatencyBuckets()),
	}
}

type storageAccessSummary struct {
	avgLatencyMs float64
	maxLatencyMs float64
	p95Ms        float64
	p99Ms        float64
}

type storageAccessFingerprintCache struct {
	mu   sync.Mutex
	seen map[string]struct{}
}

func newStorageAccessFingerprintCache() *storageAccessFingerprintCache {
	return &storageAccessFingerprintCache{
		seen: make(map[string]struct{}),
	}
}

func (c *storageAccessFingerprintCache) mark(fingerprint string) bool {
	if c == nil || fingerprint == "" {
		return true
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, ok := c.seen[fingerprint]; ok {
		return false
	}
	if len(c.seen) >= maxStorageAccessFlushFingerprints {
		c.seen = make(map[string]struct{})
	}
	c.seen[fingerprint] = struct{}{}
	return true
}

func (s *Server) reportSaveBinlogPathsStorageAccessStats(ctx context.Context, req *datapb.SaveBinlogPathsRequest) {
	if req.GetStorageAccessStats() == nil {
		return
	}
	if s.storageAccessFlushFingerprints == nil {
		s.storageAccessFlushFingerprints = newStorageAccessFingerprintCache()
	}
	if !s.storageAccessFlushFingerprints.mark(saveBinlogPathsStorageAccessFingerprint(req)) {
		return
	}
	reportStorageAccessStats(ctx, storageAccessTaskFlush, req.GetStorageAccessStats().GetTaskId(), req.GetStorageAccessStats())
}

func saveBinlogPathsStorageAccessFingerprint(req *datapb.SaveBinlogPathsRequest) string {
	if req == nil {
		return ""
	}

	var b strings.Builder
	appendInt := func(v int64) {
		b.WriteByte(':')
		b.WriteString(strconv.FormatInt(v, 10))
	}
	appendString := func(v string) {
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(len(v)))
		b.WriteByte(':')
		b.WriteString(v)
	}

	appendInt(req.GetCollectionID())
	appendInt(req.GetPartitionID())
	appendInt(req.GetSegmentID())
	appendString(req.GetChannel())
	appendInt(req.GetStorageVersion())
	appendInt(int64(req.GetSegLevel()))
	appendString(req.GetManifestPath())
	appendInt(boolToInt64(req.GetFlushed()))
	appendInt(boolToInt64(req.GetDropped()))
	appendFieldBinlogIDs(&b, "insert", req.GetField2BinlogPaths())
	appendFieldBinlogIDs(&b, "stats", req.GetField2StatslogPaths())
	appendFieldBinlogIDs(&b, "delta", req.GetDeltalogs())
	appendFieldBinlogIDs(&b, "bm25", req.GetField2Bm25LogPaths())
	return b.String()
}

func appendFieldBinlogIDs(b *strings.Builder, name string, fieldBinlogs []*datapb.FieldBinlog) {
	b.WriteByte('|')
	b.WriteString(name)

	ids := make([]int64, 0)
	for _, fieldBinlog := range fieldBinlogs {
		for _, binlog := range fieldBinlog.GetBinlogs() {
			ids = append(ids, binlog.GetLogID())
		}
	}
	sort.Slice(ids, func(i, j int) bool {
		return ids[i] < ids[j]
	})
	for _, id := range ids {
		b.WriteByte(':')
		b.WriteString(strconv.FormatInt(id, 10))
	}
}

func boolToInt64(v bool) int64 {
	if v {
		return 1
	}
	return 0
}
