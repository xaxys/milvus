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
	"fmt"
	"sort"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

const (
	traceTaskIDAttribute        = "milvus.storage_access.task_id"
	traceRequestIDAttribute     = "milvus.storage_access.request_id"
	traceRequestCountAttribute  = "milvus.storage_access.request_count"
	traceFailedCountAttribute   = "milvus.storage_access.failed_count"
	traceCanceledCountAttribute = "milvus.storage_access.canceled_count"
	traceBytesAttribute         = "milvus.storage_access.bytes"
	traceOpTypeAttribute        = "milvus.storage_access.op_type"
	traceStatusAttribute        = "milvus.storage_access.status"
	traceLatencySumAttribute    = "milvus.storage_access.latency_sum_ms"
	traceLatencyMaxAttribute    = "milvus.storage_access.latency_max_ms"
)

// LogFields returns compact fields for one boundary log entry.
func LogFields(stats *datapb.StorageAccessStats) []mlog.Field {
	if stats == nil {
		return nil
	}
	fields := make([]mlog.Field, 0, 7)
	if stats.GetTaskId() != 0 {
		fields = append(fields, mlog.FieldTaskID(stats.GetTaskId()))
	}
	if stats.GetRequestId() != "" {
		fields = append(fields, mlog.String("requestID", stats.GetRequestId()))
	}
	fields = append(fields,
		mlog.Uint64("storageRequestCount", stats.GetRequestCount()),
		mlog.Uint64("storageFailedCount", stats.GetFailedCount()),
		mlog.Uint64("storageCanceledCount", stats.GetCanceledCount()),
		mlog.Uint64("storageBytes", stats.GetBytes()),
		mlog.String("storageOperations", formatOperations(stats)),
	)
	return fields
}

func formatOperations(stats *datapb.StorageAccessStats) string {
	if stats == nil {
		return ""
	}
	operations := make([]string, 0, len(stats.GetOpStats()))
	for _, opStats := range stats.GetOpStats() {
		if opStats.GetOpType() == "" || opStats.GetStatus() == "" || opStats.GetRequestCount() == 0 {
			continue
		}
		operations = append(operations, fmt.Sprintf(
			"%s/%s:count=%d,bytes=%d,latency_sum_ms=%.3f,latency_max_ms=%.3f",
			opStats.GetOpType(),
			opStats.GetStatus(),
			opStats.GetRequestCount(),
			opStats.GetBytes(),
			opStats.GetLatencySumMs(),
			opStats.GetLatencyMaxMs(),
		))
	}
	sort.Strings(operations)
	return strings.Join(operations, ";")
}

// AddTraceEvent records one aggregate event and one event per operation on the
// current sampled span. IDs remain trace attributes and never become metrics.
func AddTraceEvent(ctx context.Context, eventName string, stats *datapb.StorageAccessStats) {
	if ctx == nil || stats == nil || stats.GetRequestCount() == 0 || eventName == "" {
		return
	}
	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return
	}

	attributes := make([]attribute.KeyValue, 0, 6)
	if stats.GetTaskId() != 0 {
		attributes = append(attributes, attribute.Int64(traceTaskIDAttribute, stats.GetTaskId()))
	}
	if stats.GetRequestId() != "" {
		attributes = append(attributes, attribute.String(traceRequestIDAttribute, stats.GetRequestId()))
	}
	attributes = append(attributes,
		attribute.Int64(traceRequestCountAttribute, int64(stats.GetRequestCount())),
		attribute.Int64(traceFailedCountAttribute, int64(stats.GetFailedCount())),
		attribute.Int64(traceCanceledCountAttribute, int64(stats.GetCanceledCount())),
		attribute.Int64(traceBytesAttribute, int64(stats.GetBytes())),
	)
	span.AddEvent(eventName, trace.WithAttributes(attributes...))

	for _, opStats := range stats.GetOpStats() {
		if opStats.GetOpType() == "" || opStats.GetStatus() == "" || opStats.GetRequestCount() == 0 {
			continue
		}
		span.AddEvent(eventName+".operation", trace.WithAttributes(
			attribute.String(traceOpTypeAttribute, opStats.GetOpType()),
			attribute.String(traceStatusAttribute, opStats.GetStatus()),
			attribute.Int64(traceRequestCountAttribute, int64(opStats.GetRequestCount())),
			attribute.Int64(traceBytesAttribute, int64(opStats.GetBytes())),
			attribute.Float64(traceLatencySumAttribute, opStats.GetLatencySumMs()),
			attribute.Float64(traceLatencyMaxAttribute, opStats.GetLatencyMaxMs()),
		))
	}
}
