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
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestObjectStorageStatsCollector(t *testing.T) {
	resetObjectStorageStatsForTest()
	defer resetObjectStorageStatsForTest()

	recordObjectStorageRequest("aws", ObjectStorageOpGet, 3*time.Millisecond, 10, 0, nil)
	recordObjectStorageRequest("aws", ObjectStorageOpGet, 5*time.Millisecond, 0, 0, errors.New("boom"))
	recordObjectStorageRequest("aws", ObjectStorageOpGet, 7*time.Millisecond, 0, 0, context.Canceled)
	recordObjectStorageBytes("aws", ObjectStorageOpGet, 4, 0)

	snapshot := GetObjectStorageStatsSnapshot()
	require.Len(t, snapshot.Ops, 1)
	op := snapshot.Ops[0]
	assert.Equal(t, ObjectStorageType, op.StorageType)
	assert.Equal(t, "aws", op.CloudProvider)
	assert.Equal(t, ObjectStorageOpGet, op.Operation)
	assert.EqualValues(t, 3, op.TotalRequests)
	assert.EqualValues(t, 1, op.SuccessRequests)
	assert.EqualValues(t, 1, op.FailedRequests)
	assert.EqualValues(t, 1, op.CanceledRequests)
	assert.EqualValues(t, 14, op.BytesRead)
	assert.EqualValues(t, 15, op.TotalLatencyMs)
	require.Len(t, op.LatencyBucketCounts, len(snapshot.LatencyBucketBoundsMs))
	assert.EqualValues(t, 0, op.LatencyBucketCounts[0])
	assert.EqualValues(t, 1, op.LatencyBucketCounts[2])
	assert.EqualValues(t, 3, op.LatencyBucketCounts[len(op.LatencyBucketCounts)-1])
}

func TestObjectStorageStatsSnapshotDoesNotReset(t *testing.T) {
	resetObjectStorageStatsForTest()
	defer resetObjectStorageStatsForTest()

	recordObjectStorageRequest("aws", ObjectStorageOpPut, time.Millisecond, 0, 100, nil)
	first := GetObjectStorageStatsSnapshot()
	second := GetObjectStorageStatsSnapshot()

	require.Len(t, first.Ops, 1)
	require.Len(t, second.Ops, 1)
	assert.Equal(t, first.Ops[0].TotalRequests, second.Ops[0].TotalRequests)
	assert.Equal(t, first.Ops[0].BytesWritten, second.Ops[0].BytesWritten)
}
