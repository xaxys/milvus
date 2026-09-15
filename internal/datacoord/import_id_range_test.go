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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// allocBlock is one [begin, end) handed out by a single allocN call.
type allocBlock struct{ begin, end int64 }

// recordingAlloc records what each allocN call asked for and leaves a gap between
// blocks, so a range that straddles two calls is observable: a real allocator
// gives no guarantee that consecutive AllocN results are adjacent.
func recordingAlloc(calls *[]int64, blocks *[]allocBlock) func(int64) (int64, int64, error) {
	next := int64(1000)
	return func(n int64) (int64, int64, error) {
		*calls = append(*calls, n)
		begin := next
		next += n + 1_000_000
		if blocks != nil {
			*blocks = append(*blocks, allocBlock{begin, begin + n})
		}
		return begin, begin + n, nil
	}
}

func forbiddenAlloc(t *testing.T) func(int64) (int64, int64, error) {
	return func(int64) (int64, int64, error) {
		t.Fatal("allocN must not be called")
		return 0, 0, nil
	}
}

func rangeWidths(ranges []*commonpb.IDRange) []int64 {
	out := make([]int64, len(ranges))
	for i, r := range ranges {
		out[i] = r.GetEnd() - r.GetBegin()
	}
	return out
}

func TestAssignExactPKRanges_ExactTotals(t *testing.T) {
	var calls []int64
	ranges, err := assignExactPKRanges([]int64{10, 20, 30}, recordingAlloc(&calls, nil), 0)
	require.NoError(t, err)
	assert.Equal(t, []int64{60}, calls, "Σ reserved == Σ rows: no expansion, no over-allocation")
	assert.Equal(t, []int64{10, 20, 30}, rangeWidths(ranges))
	// Contiguous in file order within the single batch.
	assert.Equal(t, ranges[0].GetEnd(), ranges[1].GetBegin())
	assert.Equal(t, ranges[1].GetEnd(), ranges[2].GetBegin())
}

func TestAssignExactPKRanges_BatchesNeverStraddle(t *testing.T) {
	var calls []int64
	var blocks []allocBlock
	half := maxIDsPerAllocBatch / 2
	// clusterID 0 so the recorded blocks are directly comparable: a non-zero
	// clusterID ORs its bits into every id, which shifts the ranges out of the
	// raw blocks the allocator handed back. The cluster bits themselves are
	// pinned by TestAssignExactPKRanges_ClusterIDBits.
	ranges, err := assignExactPKRanges([]int64{half, half, half}, recordingAlloc(&calls, &blocks), 0)
	require.NoError(t, err)
	// half+half fills one batch exactly; the third opens a new one.
	assert.Equal(t, []int64{2 * half, half}, calls)
	assert.Equal(t, []int64{half, half, half}, rangeWidths(ranges))
	for _, c := range calls {
		assert.LessOrEqual(t, c, maxIDsPerAllocBatch, "no allocN call exceeds the per-batch ceiling")
	}
	for i, r := range ranges {
		inOneBlock := false
		for _, b := range blocks {
			if r.GetBegin() >= b.begin && r.GetEnd() <= b.end {
				inOneBlock = true
			}
		}
		assert.True(t, inOneBlock, "range %d must sit inside a single batch", i)
	}
}

func TestAssignExactPKRanges_FileAtCeiling(t *testing.T) {
	var calls []int64
	ranges, err := assignExactPKRanges(
		[]int64{maxIDsPerAllocBatch, 1}, recordingAlloc(&calls, nil), 0)
	require.NoError(t, err)
	assert.Equal(t, []int64{maxIDsPerAllocBatch, 1}, calls, "a full batch forces the next file into its own")
	assert.Equal(t, []int64{maxIDsPerAllocBatch, 1}, rangeWidths(ranges))
}

func TestAssignExactPKRanges_ZeroRowFileGetsEmptyRange(t *testing.T) {
	var calls []int64
	ranges, err := assignExactPKRanges([]int64{5, 0, 7}, recordingAlloc(&calls, nil), 0)
	require.NoError(t, err)
	assert.Equal(t, []int64{12}, calls, "a zero-row file consumes no ids")
	assert.Equal(t, []int64{5, 0, 7}, rangeWidths(ranges))
	assert.Equal(t, ranges[1].GetBegin(), ranges[1].GetEnd())
	// The empty range sits at the packing position, keeping the batch contiguous.
	assert.Equal(t, ranges[0].GetEnd(), ranges[1].GetBegin())
	assert.Equal(t, ranges[1].GetEnd(), ranges[2].GetBegin())
}

func TestAssignExactPKRanges_AllZeroAllocatesNothing(t *testing.T) {
	ranges, err := assignExactPKRanges([]int64{0, 0}, forbiddenAlloc(t), 0)
	require.NoError(t, err)
	require.Len(t, ranges, 2)
	for _, r := range ranges {
		assert.Equal(t, r.GetBegin(), r.GetEnd())
	}
}

func TestAssignExactPKRanges_NoFiles(t *testing.T) {
	ranges, err := assignExactPKRanges(nil, forbiddenAlloc(t), 0)
	require.NoError(t, err)
	assert.Empty(t, ranges)
}

func TestAssignExactPKRanges_RejectsFileOverOneBatch(t *testing.T) {
	var calls []int64
	_, err := assignExactPKRanges([]int64{10, maxIDsPerAllocBatch + 1}, recordingAlloc(&calls, nil), 0)
	require.Error(t, err)
	assert.ErrorIs(t, err, merr.ErrParameterInvalid)
	assert.Contains(t, err.Error(), "split the file")
	assert.Empty(t, calls, "nothing is allocated once the request is rejected")
}

func TestAssignExactPKRanges_RejectsNegativeRows(t *testing.T) {
	_, err := assignExactPKRanges([]int64{10, -1}, forbiddenAlloc(t), 0)
	require.Error(t, err)
	assert.ErrorIs(t, err, merr.ErrImportSysFailed)
}

func TestAssignExactPKRanges_AllocErrorPropagates(t *testing.T) {
	_, err := assignExactPKRanges([]int64{10}, func(int64) (int64, int64, error) {
		return 0, 0, merr.WrapErrServiceUnavailableMsg("rootcoord unavailable")
	}, 0)
	require.Error(t, err)
	assert.ErrorIs(t, err, merr.ErrServiceUnavailable)
}

func TestAssignExactPKRanges_ClusterIDBits(t *testing.T) {
	const clusterID = uint64(0b1011)
	fileRows := []int64{4096, 100}

	var calls []int64
	ranges, err := assignExactPKRanges(fileRows, recordingAlloc(&calls, nil), clusterID)
	require.NoError(t, err)
	require.Equal(t, []int64{4196}, calls)

	// The batch begin must equal what common.AllocAutoIDN produces for the same
	// allocation and clusterID: the cluster bits ride in the high bits of every id.
	wantBegin, _, err := common.AllocAutoIDN(func(int64) (int64, int64, error) { return 1000, 1000 + 4196, nil }, 4196, clusterID)
	require.NoError(t, err)
	assert.Equal(t, wantBegin, ranges[0].GetBegin())
	assert.Equal(t, []int64{4096, 100}, rangeWidths(ranges), "ORing the cluster bits preserves the exact widths")
	assert.Equal(t, ranges[0].GetEnd(), ranges[1].GetBegin())
	assert.NotZero(t, ranges[0].GetBegin(), "clusterID 0b1011 must embed its bits")
}
