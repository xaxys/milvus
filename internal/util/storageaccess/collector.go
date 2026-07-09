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
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
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

func statusFromError(err error) string {
	if err == nil {
		return metrics.SuccessLabel
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return metrics.CancelLabel
	}
	return metrics.FailLabel
}

// RecordAccess records one storage access observation to Prometheus.
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
}
