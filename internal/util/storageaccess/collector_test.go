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
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
)

func TestCollectorSnapshotAndQuantile(t *testing.T) {
	collector := NewTaskCollector(TaskTypeIndex, 101)
	collector.Record(OpRead, metrics.SuccessLabel, 10, 1)
	collector.Record(OpRead, metrics.SuccessLabel, 20, 5)
	collector.Record(OpWrite, metrics.FailLabel, 30, 125000)
	collector.Record(OpWrite, metrics.CancelLabel, 0, 10)

	stats := collector.Snapshot()
	require.NotNil(t, stats)
	require.Equal(t, TaskTypeIndex, stats.TaskType)
	require.EqualValues(t, 101, stats.TaskID)
	require.EqualValues(t, 4, stats.RequestCount)
	require.EqualValues(t, 1, stats.FailedCount)
	require.EqualValues(t, 1, stats.CanceledCount)
	require.EqualValues(t, 60, stats.Bytes)

	var readStatsIndex int
	for idx, opStats := range stats.OpStats {
		if opStats.OpType == OpRead && opStats.Status == metrics.SuccessLabel {
			readStatsIndex = idx
			break
		}
	}
	readStats := stats.OpStats[readStatsIndex]
	require.EqualValues(t, 2, readStats.RequestCount)
	require.EqualValues(t, 30, readStats.Bytes)
	require.Equal(t, 6.0, readStats.LatencySumMs)
	require.Equal(t, 5.0, readStats.LatencyMaxMs)
	require.Greater(t, Quantile(0.95, readStats.LatencyBuckets), 0.0)
}

func TestLatencyBucketsCombineDynamicAndClassicBounds(t *testing.T) {
	buckets := LatencyBuckets()
	require.Contains(t, buckets, 0.25)
	require.Contains(t, buckets, 4.0)
	require.Contains(t, buckets, 5.0)
	require.Contains(t, buckets, 30000.0)
	require.IsIncreasing(t, buckets)
}

func TestCollectorContextAndConcurrentRecord(t *testing.T) {
	ctx := context.Background()
	require.Nil(t, FromContext(ctx))

	collector := NewCollector()
	ctx = WithCollector(ctx, collector)
	require.Same(t, collector, FromContext(ctx))

	const goroutines = 8
	const recordsPerGoroutine = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < recordsPerGoroutine; j++ {
				RecordAccess(ctx, OpCopy, 1, nil, 2*time.Millisecond)
			}
		}()
	}
	wg.Wait()

	stats := collector.Snapshot()
	require.NotNil(t, stats)
	require.EqualValues(t, goroutines*recordsPerGoroutine, stats.RequestCount)
	require.EqualValues(t, goroutines*recordsPerGoroutine, stats.Bytes)
}

func TestRecordAccessClassifiesFailureModes(t *testing.T) {
	collector := NewCollector()
	ctx := WithCollector(context.Background(), collector)
	RecordAccess(ctx, OpRead, 0, context.Canceled, time.Millisecond)
	RecordAccess(ctx, OpRead, 0, context.DeadlineExceeded, time.Millisecond)
	RecordAccess(ctx, OpRead, 0, errors.New("storage failure"), 250*time.Second)

	stats := collector.Snapshot()
	require.NotNil(t, stats)
	require.EqualValues(t, 3, stats.RequestCount)
	require.EqualValues(t, 1, stats.FailedCount)
	require.EqualValues(t, 2, stats.CanceledCount)

	for _, opStats := range stats.OpStats {
		if opStats.Status == metrics.FailLabel {
			require.EqualValues(t, 1, opStats.LatencyBuckets[len(opStats.LatencyBuckets)-1].CumulativeCount)
		}
	}
}

func TestRegistryFiltersAndExpiresSnapshots(t *testing.T) {
	registry := NewRegistry(2, time.Minute)
	first := &Collector{taskType: TaskTypeIndex, taskID: 1}
	second := &Collector{taskType: TaskTypeImport, taskID: 2}
	first.Record(OpRead, metrics.SuccessLabel, 1, 1)
	second.Record(OpWrite, metrics.SuccessLabel, 2, 2)
	registry.Register(first)
	registry.Register(second)

	require.Len(t, registry.Snapshots("", 0), 2)
	filtered := registry.Snapshots(TaskTypeImport, 2)
	require.Len(t, filtered, 1)
	require.Equal(t, TaskTypeImport, filtered[0].TaskType)

	expired := NewRegistry(1, time.Nanosecond)
	expired.Register(first)
	time.Sleep(time.Millisecond)
	require.Empty(t, expired.Snapshots("", 0))
}
