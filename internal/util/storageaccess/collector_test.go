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
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestCollectorSnapshotAndQuantile(t *testing.T) {
	collector := NewCollector(WithTaskID(101), WithRequestID("request-1"))
	collector.Record(OpRead, metrics.SuccessLabel, 10, 1)
	collector.Record(OpRead, metrics.SuccessLabel, 20, 5)
	collector.Record(OpWrite, metrics.FailLabel, 30, 125000)
	collector.Record(OpWrite, metrics.CancelLabel, 0, 10)

	stats := collector.Snapshot()
	require.NotNil(t, stats)
	require.EqualValues(t, 4, stats.GetRequestCount())
	require.EqualValues(t, 1, stats.GetFailedCount())
	require.EqualValues(t, 1, stats.GetCanceledCount())
	require.EqualValues(t, 60, stats.GetBytes())
	require.EqualValues(t, 101, stats.GetTaskId())
	require.Equal(t, "request-1", stats.GetRequestId())

	var readStats, failedWriteStats *datapb.StorageAccessOpStats
	for _, opStats := range stats.GetOpStats() {
		switch {
		case opStats.GetOpType() == OpRead && opStats.GetStatus() == metrics.SuccessLabel:
			readStats = opStats
		case opStats.GetOpType() == OpWrite && opStats.GetStatus() == metrics.FailLabel:
			failedWriteStats = opStats
		}
	}
	require.NotNil(t, readStats)
	require.EqualValues(t, 2, readStats.GetRequestCount())
	require.EqualValues(t, 30, readStats.GetBytes())
	require.Equal(t, 6.0, readStats.GetLatencySumMs())
	require.Equal(t, 5.0, readStats.GetLatencyMaxMs())
	require.EqualValues(t, 2, readStats.GetLatencyBuckets()[2].GetCumulativeCount())
	require.Greater(t, Quantile(0.95, readStats.GetLatencyBuckets()), 0.0)

	require.NotNil(t, failedWriteStats)
	require.EqualValues(t, 1, failedWriteStats.GetLatencyBuckets()[len(failedWriteStats.GetLatencyBuckets())-1].GetCumulativeCount())
}

func TestCollectorMergeAndContext(t *testing.T) {
	ctx := context.Background()
	require.Nil(t, FromContext(ctx))

	collector := NewCollector()
	ctx = WithCollector(ctx, collector)
	require.Same(t, collector, FromContext(ctx))
	require.Same(t, ctx, WithCollector(ctx, nil))

	collector.Record(OpRead, metrics.SuccessLabel, 10, 1)
	snapshot := collector.Snapshot()

	merged := NewCollector()
	merged.Merge(snapshot)
	merged.Merge(nil)

	stats := merged.Snapshot()
	require.NotNil(t, stats)
	require.EqualValues(t, 1, stats.GetRequestCount())
	require.EqualValues(t, 10, stats.GetBytes())
	require.Len(t, stats.GetOpStats(), 1)
	require.Equal(t, OpRead, stats.GetOpStats()[0].GetOpType())
	require.Equal(t, metrics.SuccessLabel, stats.GetOpStats()[0].GetStatus())
	require.EqualValues(t, snapshot.GetTaskId(), stats.GetTaskId())
	require.Equal(t, snapshot.GetRequestId(), stats.GetRequestId())
}

func TestCollectorConcurrentRecord(t *testing.T) {
	collector := NewCollector()
	const goroutines = 8
	const recordsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < recordsPerGoroutine; j++ {
				collector.Record(OpCopy, metrics.SuccessLabel, 1, 2)
			}
		}()
	}
	wg.Wait()

	stats := collector.Snapshot()
	require.NotNil(t, stats)
	require.EqualValues(t, goroutines*recordsPerGoroutine, stats.GetRequestCount())
	require.EqualValues(t, goroutines*recordsPerGoroutine, stats.GetBytes())
}

func TestNilCollector(t *testing.T) {
	var collector *Collector
	require.Nil(t, collector.Snapshot())
	require.NotPanics(t, func() {
		collector.Record(OpRead, metrics.SuccessLabel, 1, 1)
		collector.Merge(nil)
	})
}

func TestRequestIDFromContext(t *testing.T) {
	spanRecorder := tracetest.NewSpanRecorder()
	provider := trace.NewTracerProvider(trace.WithSpanProcessor(spanRecorder))
	ctx, span := provider.Tracer("storage-access-test").Start(context.Background(), "request")
	requestID := RequestIDFromContext(ctx)
	span.End()

	require.NotEmpty(t, requestID)
	require.Equal(t, spanRecorder.Ended()[0].SpanContext().TraceID().String(), requestID)
	require.Empty(t, RequestIDFromContext(context.Background()))
}

func TestStorageAccessObservability(t *testing.T) {
	collector := NewCollector(WithTaskID(10), WithRequestID("request-10"))
	collector.Record(OpRead, metrics.SuccessLabel, 128, 2.5)
	stats := collector.Snapshot()

	fields := LogFields(stats)
	require.Len(t, fields, 7)
	require.Contains(t, formatOperations(stats), "read/success:count=1,bytes=128")

	spanRecorder := tracetest.NewSpanRecorder()
	provider := trace.NewTracerProvider(trace.WithSpanProcessor(spanRecorder))
	ctx, span := provider.Tracer("storage-access-test").Start(context.Background(), "task")
	AddTraceEvent(ctx, "storage_access.task.finished", stats)
	span.End()

	ended := spanRecorder.Ended()
	require.Len(t, ended, 1)
	require.Len(t, ended[0].Events(), 2)
	require.Equal(t, "storage_access.task.finished", ended[0].Events()[0].Name)
	require.Equal(t, "storage_access.task.finished.operation", ended[0].Events()[1].Name)
}
