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

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestValidateImportResults(t *testing.T) {
	results := []*datapb.SegmentResult{{
		Rows: 10, Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
	}}
	require.NoError(t, validateImportResults(results, 1))
	results[0].Statistics.TimestampFrom = 101
	require.Error(t, validateImportResults(results, 1))
}

func TestApplyImportResultToPreallocatedSegment(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)
	_, err = addImportSegment(ctx, meta, 100, 1, 10, 2, 3, "v0", datapb.SegmentLevel_L1, 1, 4)
	require.NoError(t, err)

	results := []*datapb.SegmentResult{{
		Rows: 10, Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
	}}
	require.NoError(t, applyImportResults(ctx, meta, 2, 4, []int64{100}, true, false, results))

	segment := meta.GetSegment(ctx, 100)
	require.Equal(t, commonpb.SegmentState_Flushed, segment.GetState())
	require.True(t, segment.GetIsImporting())
	require.False(t, segment.GetIsInvisible())
	require.Equal(t, int64(10), segment.GetNumOfRows())
}

func TestValidateReshardManifest(t *testing.T) {
	manifest := &datapb.ReshardManifest{Fragments: []*datapb.FragmentDescriptor{{
		Path: "fragment.parquet", Rows: 10, LogicalBytes: 100,
	}}}
	require.NoError(t, validateReshardManifest(manifest))
	manifest.Fragments = append(manifest.Fragments, manifest.Fragments[0])
	require.Error(t, validateReshardManifest(manifest))

}
