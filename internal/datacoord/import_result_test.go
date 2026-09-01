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
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestValidateImportResults(t *testing.T) {
	results := []*datapb.SegmentResult{{
		Rows: 10, Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
		ManifestPath: `{"ver":1,"base_path":"root/insert_log/2/3/100"}`,
	}}
	require.NoError(t, validateImportResults(results, 1))
	results[0].Statistics.TimestampFrom = 101
	require.Error(t, validateImportResults(results, 1))
	results[0].Statistics.TimestampFrom = 1
	results[0].ManifestPath = ""
	require.Error(t, validateImportResults(results, 1))
}

func v3AcceptanceTask() *datapb.ImportTaskV3 {
	return &datapb.ImportTaskV3{
		JobId: 1, TaskId: 10, CollectionId: 2, SegmentId: 100,
		Vchannel: "v0", PartitionId: 3,
	}
}

func TestApplyImportResultsCreatesSegmentAtAcceptance(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)

	require.Nil(t, meta.GetSegment(ctx, 100))
	results := []*datapb.SegmentResult{{
		Rows: 10, Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
		ManifestPath: `{"ver":1,"base_path":"root/insert_log/2/3/100"}`,
	}}
	require.NoError(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, results))

	segment := meta.GetSegment(ctx, 100)
	require.NotNil(t, segment)
	require.Equal(t, commonpb.SegmentState_Flushed, segment.GetState())
	require.True(t, segment.GetIsImporting())
	require.False(t, segment.GetIsInvisible())
	require.Equal(t, int64(10), segment.GetNumOfRows())
	require.Equal(t, int64(2), segment.GetCollectionID())
	require.Equal(t, int64(3), segment.GetPartitionID())
	require.Equal(t, "v0", segment.GetInsertChannel())
	require.Equal(t, int32(4), segment.GetSchemaVersion())
	require.Equal(t, results[0].GetManifestPath(), segment.GetManifestPath())
	require.Equal(t, uint64(1), segment.GetStartPosition().GetTimestamp())
	require.Equal(t, uint64(100), segment.GetDmlPosition().GetTimestamp())
	require.True(t, segment.GetIsSorted())
	require.False(t, segment.GetIsSortedByNamespace())
}

func TestApplyImportResultsZeroRowCreatesNoSegment(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)

	require.NoError(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, []*datapb.SegmentResult{{}}))
	require.Nil(t, meta.GetSegment(ctx, 100))
}

// TestApplyImportResultsReplayIsIdempotent pins the marker-last crash recovery:
// a crash between segment registration and the task Completed marker replays the
// same worker result, and the replay must be a no-op.
func TestApplyImportResultsReplayIsIdempotent(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)

	newResults := func() []*datapb.SegmentResult {
		return []*datapb.SegmentResult{{
			Rows:         10,
			Statistics:   &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
			ManifestPath: `{"ver":1,"base_path":"root/insert_log/2/3/100"}`,
		}}
	}
	require.NoError(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, newResults()))
	require.NoError(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, newResults()))
	require.Equal(t, int64(10), meta.GetSegment(ctx, 100).GetNumOfRows())
}

func TestApplyImportResultsRejectsConflictingReplay(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)

	first := []*datapb.SegmentResult{{
		Rows: 10, Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
		ManifestPath: `{"ver":1,"base_path":"root/insert_log/2/3/100"}`,
	}}
	require.NoError(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, first))

	conflicting := []*datapb.SegmentResult{{
		Rows: 11, Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
		ManifestPath: `{"ver":1,"base_path":"root/insert_log/2/3/100"}`,
	}}
	require.Error(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, conflicting))
}

// TestApplyImportResultsCompressesStorageV2Binlogs pins that StorageV2 writer
// results (LogPath set, LogID zero) are compressed before persistence; the
// catalog rejects zero LogIDs and stored LogPaths, so without compression the
// created segment could never be persisted.
func TestApplyImportResultsCompressesStorageV2Binlogs(t *testing.T) {
	ctx := context.Background()
	meta, err := newMemoryMeta(t)
	require.NoError(t, err)

	results := []*datapb.SegmentResult{{
		Rows:       10,
		Statistics: &datapb.Statistics{TimestampFrom: 1, TimestampTo: 100},
		InsertLogs: []*datapb.FieldBinlog{{FieldID: 0, Binlogs: []*datapb.Binlog{{
			LogPath: "files/insert_log/2/3/100/0/555", EntriesNum: 10,
		}}}},
		PkLog: &datapb.FieldBinlog{FieldID: 100, Binlogs: []*datapb.Binlog{{
			LogPath: "files/stats_log/2/3/100/100/666", EntriesNum: 10,
		}}},
	}}
	require.NoError(t, applyImportResults(ctx, meta, 2, 4, v3AcceptanceTask(), true, false, results))

	segment := meta.GetSegment(ctx, 100)
	require.Equal(t, commonpb.SegmentState_Flushed, segment.GetState())
	require.Equal(t, storage.StorageV2, segment.GetStorageVersion())
	require.Len(t, segment.GetBinlogs(), 1)
	require.Equal(t, int64(555), segment.GetBinlogs()[0].GetBinlogs()[0].GetLogID())
	require.Empty(t, segment.GetBinlogs()[0].GetBinlogs()[0].GetLogPath())
	require.Len(t, segment.GetStatslogs(), 1)
	require.Equal(t, int64(666), segment.GetStatslogs()[0].GetBinlogs()[0].GetLogID())
	require.Empty(t, segment.GetStatslogs()[0].GetBinlogs()[0].GetLogPath())
}

func TestValidateReshardManifest(t *testing.T) {
	manifest := &datapb.ReshardManifest{Fragments: []*datapb.FragmentDescriptor{{
		Path: "fragment.parquet", Rows: 10, LogicalBytes: 100,
	}}}
	require.NoError(t, validateReshardManifest(manifest))
	manifest.Fragments = append(manifest.Fragments, manifest.Fragments[0])
	require.Error(t, validateReshardManifest(manifest))

}
