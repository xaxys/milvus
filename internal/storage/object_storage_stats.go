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

package storage

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

const (
	ObjectStorageType = "object_storage"

	ObjectStorageOpGet    = "get"
	ObjectStorageOpPut    = "put"
	ObjectStorageOpStat   = "stat"
	ObjectStorageOpWalk   = "walk"
	ObjectStorageOpRemove = "remove"
	ObjectStorageOpCopy   = "copy"
)

var objectStorageLatencyBucketBoundsMs = []float64{
	1, 2, 4, 8, 16, 32, 64, 128, 256,
	512, 1024, 2048, 4096, 8192, 16384,
	32768, 65536, 131072,
}

type objectStorageStatsKey struct {
	storageType   string
	cloudProvider string
	operation     string
}

type objectStorageStatsCollector struct {
	mu  sync.Mutex
	ops map[objectStorageStatsKey]*metricsinfo.ObjectStorageOpMetrics
}

var globalObjectStorageStatsCollector = &objectStorageStatsCollector{
	ops: make(map[objectStorageStatsKey]*metricsinfo.ObjectStorageOpMetrics),
}

func recordObjectStorageRequest(cloudProvider, operation string, latency time.Duration, bytesRead, bytesWritten int64, err error) {
	globalObjectStorageStatsCollector.recordRequest(cloudProvider, operation, latency, bytesRead, bytesWritten, err)
}

func recordObjectStorageBytes(cloudProvider, operation string, bytesRead, bytesWritten int64) {
	globalObjectStorageStatsCollector.recordBytes(cloudProvider, operation, bytesRead, bytesWritten)
}

func GetObjectStorageStatsSnapshot() *metricsinfo.ObjectStorageMetrics {
	return globalObjectStorageStatsCollector.snapshot()
}

func resetObjectStorageStatsForTest() {
	globalObjectStorageStatsCollector.mu.Lock()
	defer globalObjectStorageStatsCollector.mu.Unlock()
	globalObjectStorageStatsCollector.ops = make(map[objectStorageStatsKey]*metricsinfo.ObjectStorageOpMetrics)
}

func (c *objectStorageStatsCollector) recordRequest(cloudProvider, operation string, latency time.Duration, bytesRead, bytesWritten int64, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	op := c.getOrCreateLocked(cloudProvider, operation)
	op.TotalRequests++
	switch {
	case err == nil:
		op.SuccessRequests++
	case errors.Is(err, context.Canceled):
		op.CanceledRequests++
	default:
		op.FailedRequests++
	}

	op.BytesRead += normalizeObjectStorageBytes(bytesRead)
	op.BytesWritten += normalizeObjectStorageBytes(bytesWritten)
	latencyMs := latency.Milliseconds()
	if latencyMs < 0 {
		latencyMs = 0
	}
	op.TotalLatencyMs += latencyMs
	bucketed := false
	for i, bound := range objectStorageLatencyBucketBoundsMs {
		if float64(latencyMs) <= bound {
			op.LatencyBucketCounts[i]++
			bucketed = true
		}
	}
	if !bucketed && len(op.LatencyBucketCounts) > 0 {
		op.LatencyBucketCounts[len(op.LatencyBucketCounts)-1]++
	}
}

func (c *objectStorageStatsCollector) recordBytes(cloudProvider, operation string, bytesRead, bytesWritten int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	op := c.getOrCreateLocked(cloudProvider, operation)
	op.BytesRead += normalizeObjectStorageBytes(bytesRead)
	op.BytesWritten += normalizeObjectStorageBytes(bytesWritten)
}

func (c *objectStorageStatsCollector) snapshot() *metricsinfo.ObjectStorageMetrics {
	c.mu.Lock()
	defer c.mu.Unlock()

	ops := make([]metricsinfo.ObjectStorageOpMetrics, 0, len(c.ops))
	for _, op := range c.ops {
		copied := *op
		copied.LatencyBucketCounts = append([]int64(nil), op.LatencyBucketCounts...)
		ops = append(ops, copied)
	}
	sort.Slice(ops, func(i, j int) bool {
		if ops[i].CloudProvider != ops[j].CloudProvider {
			return ops[i].CloudProvider < ops[j].CloudProvider
		}
		return ops[i].Operation < ops[j].Operation
	})

	return &metricsinfo.ObjectStorageMetrics{
		LatencyBucketBoundsMs: append([]float64(nil), objectStorageLatencyBucketBoundsMs...),
		Ops:                   ops,
	}
}

func (c *objectStorageStatsCollector) getOrCreateLocked(cloudProvider, operation string) *metricsinfo.ObjectStorageOpMetrics {
	key := objectStorageStatsKey{
		storageType:   ObjectStorageType,
		cloudProvider: normalizeObjectStorageLabel(cloudProvider),
		operation:     operation,
	}
	op, ok := c.ops[key]
	if ok {
		return op
	}

	op = &metricsinfo.ObjectStorageOpMetrics{
		StorageType:         key.storageType,
		CloudProvider:       key.cloudProvider,
		Operation:           key.operation,
		LatencyBucketCounts: make([]int64, len(objectStorageLatencyBucketBoundsMs)),
	}
	c.ops[key] = op
	return op
}

func normalizeObjectStorageLabel(value string) string {
	if value == "" {
		return "unknown"
	}
	return value
}

func normalizeObjectStorageBytes(value int64) int64 {
	if value < 0 {
		return 0
	}
	return value
}

type objectStorageStatsReader struct {
	FileReader
	cloudProvider string
}

func (r *objectStorageStatsReader) Read(p []byte) (int, error) {
	n, err := r.FileReader.Read(p)
	if n > 0 {
		recordObjectStorageBytes(r.cloudProvider, ObjectStorageOpGet, int64(n), 0)
	}
	return n, err
}

func (r *objectStorageStatsReader) ReadAt(p []byte, off int64) (int, error) {
	n, err := r.FileReader.ReadAt(p, off)
	if n > 0 {
		recordObjectStorageBytes(r.cloudProvider, ObjectStorageOpGet, int64(n), 0)
	}
	return n, err
}
