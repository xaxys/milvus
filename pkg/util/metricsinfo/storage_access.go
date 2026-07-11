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

package metricsinfo

const (
	// StorageAccessMetrics requests task-local storage access snapshots.
	StorageAccessMetrics = "storage_access"

	StorageAccessTaskTypeKey = "task_type"
	StorageAccessTaskIDKey   = "task_id"
)

type StorageAccessLatencyBucket struct {
	UpperBoundMs    float64 `json:"upper_bound_ms"`
	CumulativeCount uint64  `json:"cumulative_count"`
}

type StorageAccessOpStats struct {
	OpType         string                       `json:"op_type"`
	Status         string                       `json:"status"`
	RequestCount   uint64                       `json:"request_count"`
	Bytes          uint64                       `json:"bytes"`
	LatencySumMs   float64                      `json:"latency_sum_ms"`
	LatencyMaxMs   float64                      `json:"latency_max_ms"`
	LatencyBuckets []StorageAccessLatencyBucket `json:"latency_buckets"`
}

type StorageAccessStats struct {
	NodeID        int64                  `json:"node_id,omitempty,string"`
	TaskType      string                 `json:"task_type"`
	TaskID        int64                  `json:"task_id,omitempty,string"`
	OpStats       []StorageAccessOpStats `json:"op_stats"`
	RequestCount  uint64                 `json:"request_count"`
	FailedCount   uint64                 `json:"failed_count"`
	CanceledCount uint64                 `json:"canceled_count"`
	Bytes         uint64                 `json:"bytes"`
}
