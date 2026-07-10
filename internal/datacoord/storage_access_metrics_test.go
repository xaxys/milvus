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
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/milvus-io/milvus/internal/util/storageaccess"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestReportStorageAccessStatsCorrelatesTaskWithoutMetricLabel(t *testing.T) {
	metric := metrics.DataCoordTaskStorageAccessOpCount.WithLabelValues(
		storageAccessTaskIndex,
		storageaccess.OpRead,
		metrics.SuccessLabel,
	)
	before := testutil.ToFloat64(metric)

	spanRecorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder))
	ctx, span := provider.Tracer("datacoord-storage-access-test").Start(context.Background(), "task")
	reportStorageAccessStats(ctx, storageAccessTaskIndex, 1001, &datapb.StorageAccessStats{
		RequestCount: 1,
		Bytes:        256,
		OpStats: []*datapb.StorageAccessOpStats{
			{
				OpType:       storageaccess.OpRead,
				Status:       metrics.SuccessLabel,
				RequestCount: 1,
				Bytes:        256,
				LatencySumMs: 2,
				LatencyMaxMs: 2,
				LatencyBuckets: []*datapb.StorageAccessLatencyBucket{
					{UpperBoundMs: 5, CumulativeCount: 1},
				},
			},
		},
	})
	span.End()

	require.Equal(t, before+1, testutil.ToFloat64(metric))
	ended := spanRecorder.Ended()
	require.Len(t, ended, 1)
	require.Len(t, ended[0].Events(), 2)
	require.Equal(t, "storage_access.task.coordinated", ended[0].Events()[0].Name)
	requireInt64Attribute(t, ended[0].Events()[0].Attributes, "milvus.storage_access.task_id", 1001)
}

func requireInt64Attribute(t *testing.T, attributes []attribute.KeyValue, key string, value int64) {
	t.Helper()
	for _, attr := range attributes {
		if string(attr.Key) == key {
			require.Equal(t, value, attr.Value.AsInt64())
			return
		}
	}
	require.Fail(t, "attribute not found", key)
}
