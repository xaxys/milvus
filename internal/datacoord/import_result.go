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
	"math"
	"path"
	"slices"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
)

func loadImportV3Object(ctx context.Context, cm storage.ChunkManager, objectPath string) ([]byte, error) {
	fullPath := path.Join(cm.RootPath(), objectPath)
	data, err := cm.Read(ctx, fullPath)
	if err != nil {
		return nil, merr.Wrap(err, "read import v3 object")
	}
	return data, nil
}

func loadReshardResultManifest(ctx context.Context, cm storage.ChunkManager, jobID, taskID, runID int64) (*datapb.ReshardManifest, error) {
	data, err := loadImportV3Object(ctx, cm, metautil.BuildImportReshardResultPath(jobID, taskID, runID))
	if err != nil {
		return nil, err
	}
	manifest := &datapb.ReshardManifest{}
	if err := proto.Unmarshal(data, manifest); err != nil {
		return nil, merr.WrapErrDataIntegrity(err, "unmarshal reshard result manifest")
	}
	return manifest, nil
}

func validateReshardManifest(manifest *datapb.ReshardManifest) error {
	if manifest == nil {
		return merr.WrapErrDataIntegrityMsg("reshard manifest is nil")
	}
	seen := make(map[string]struct{}, len(manifest.GetFragments()))
	for _, fragment := range manifest.GetFragments() {
		if fragment == nil || fragment.GetPath() == "" || fragment.GetRows() < 0 || fragment.GetLogicalBytes() < 0 || fragment.GetChannelIndex() < 0 || fragment.GetPartitionIndex() < 0 {
			return merr.WrapErrDataIntegrityMsg("invalid reshard fragment descriptor")
		}
		if _, ok := seen[fragment.GetPath()]; ok {
			return merr.WrapErrDataIntegrityMsg("duplicate reshard fragment path: %s", fragment.GetPath())
		}
		seen[fragment.GetPath()] = struct{}{}
	}
	return nil
}

// validateImportResults validates the worker result contract
// before any SegmentInfo is made Flushed or the task marker is persisted.
// Ordering is defined by the task's fixed output segment slice. Timestamp
// checks use the durable Statistics object so the same fence survives a
// DataCoord restart.
func validateImportResults(results []*datapb.SegmentResult, segmentCount int) error {
	if len(results) != segmentCount {
		return merr.WrapErrDataIntegrityMsg("import result segment count mismatch: got=%d want=%d", len(results), segmentCount)
	}
	for index, result := range results {
		if result == nil || result.GetRows() < 0 {
			return merr.WrapErrDataIntegrityMsg("invalid import segment result at index %d", index)
		}
		if result.GetRows() == 0 {
			if result.GetStatistics() != nil ||
				len(result.GetInsertLogs()) > 0 || result.GetPkLog() != nil ||
				len(result.GetBm25Logs()) > 0 || result.GetManifestPath() != "" ||
				len(result.GetExpirationQuantiles()) > 0 {
				return merr.WrapErrDataIntegrityMsg("zero-row import segment result at index %d has output", index)
			}
		} else {
			if result.GetStatistics() == nil || result.GetStatistics().GetTimestampFrom() > result.GetStatistics().GetTimestampTo() {
				return merr.WrapErrDataIntegrityMsg("non-empty import segment result at index %d has invalid statistics", index)
			}
			if result.GetManifestPath() == "" && len(result.GetInsertLogs()) == 0 {
				return merr.WrapErrDataIntegrityMsg("non-empty import segment result at index %d has no output", index)
			}
		}
	}
	return nil
}

// applyImportResults registers the task's output segment at acceptance. No
// segment exists before this point: the accepted SegmentInfo is built
// deterministically from the frozen task record and the worker result. A
// replayed result (a crash between this write and the task Completed marker)
// must match the already-registered segment exactly. The caller persists the
// task Completed marker only after this function succeeds, which keeps the
// marker-last recovery shape: a crash between the two writes simply replays
// the same worker result.
func applyImportResults(ctx context.Context, meta *meta, collectionID int64, schemaVersion int32, task *datapb.ImportTaskV3, sorted, namespaceSorted bool, results []*datapb.SegmentResult) error {
	for _, result := range results {
		if result == nil || result.GetManifestPath() != "" {
			continue
		}
		// The StorageV2 writer reports FieldBinlogs by LogPath with no LogID,
		// while the catalog only persists LogID-only entries. Compress before
		// validation so the replay check compares the same shape that is
		// persisted (mirroring the V2 import path).
		statslogs := []*datapb.FieldBinlog(nil)
		if result.GetPkLog() != nil {
			statslogs = []*datapb.FieldBinlog{result.GetPkLog()}
		}
		if err := binlog.CompressBinLogs(result.GetInsertLogs(), statslogs, result.GetBm25Logs()); err != nil {
			return err
		}
	}
	for _, result := range results {
		if result.GetRows() == 0 {
			// A zero-row result produces no segment; there is nothing to load,
			// index, or clean up.
			continue
		}
		segmentID := task.GetSegmentId()
		existing := meta.GetSegment(ctx, segmentID)
		if existing != nil {
			if existing.GetCollectionID() != collectionID || existing.GetState() != commonpb.SegmentState_Flushed ||
				!existing.GetIsImporting() || !acceptedImportSegmentMatches(existing, result, sorted, namespaceSorted) {
				return merr.WrapErrDataIntegrityMsg("replayed import v3 result conflicts with segment %d", segmentID)
			}
			continue
		}
		if err := meta.AddSegment(ctx, buildImportV3Segment(collectionID, schemaVersion, task, sorted, namespaceSorted, result)); err != nil {
			return err
		}
		mlog.Info(ctx, "import v3 segment accepted",
			mlog.Int64("segmentID", segmentID),
			mlog.Int64("rows", result.GetRows()),
			mlog.String("vchannel", task.GetVchannel()))
	}
	return nil
}

// buildImportV3Segment constructs the accepted SegmentInfo deterministically
// from the frozen task record and the worker result. The segment is born
// Flushed and keeps IsImporting=true until the commit fence clears it, the
// same visibility contract as the V2 import path.
func buildImportV3Segment(collectionID int64, schemaVersion int32, task *datapb.ImportTaskV3, sorted, namespaceSorted bool, result *datapb.SegmentResult) *SegmentInfo {
	stats := result.GetStatistics()
	info := &datapb.SegmentInfo{
		ID:                  task.GetSegmentId(),
		CollectionID:        collectionID,
		PartitionID:         task.GetPartitionId(),
		InsertChannel:       task.GetVchannel(),
		NumOfRows:           result.GetRows(),
		MaxRowNum:           result.GetRows(),
		State:               commonpb.SegmentState_Flushed,
		LastExpireTime:      math.MaxUint64,
		Level:               datapb.SegmentLevel_L1,
		IsImporting:         true,
		StorageVersion:      storage.StorageV3,
		SchemaVersion:       schemaVersion,
		ManifestPath:        result.GetManifestPath(),
		Stats:               stats,
		StartPosition:       &msgpb.MsgPosition{ChannelName: task.GetVchannel(), Timestamp: stats.GetTimestampFrom()},
		DmlPosition:         &msgpb.MsgPosition{ChannelName: task.GetVchannel(), Timestamp: stats.GetTimestampTo()},
		IsSorted:            sorted,
		IsSortedByNamespace: namespaceSorted,
		ExpirQuantiles:      append([]int64(nil), result.GetExpirationQuantiles()...),
	}
	if result.GetManifestPath() == "" {
		// StorageV2 writer output: file lists cross the RPC boundary instead of
		// a manifest.
		info.StorageVersion = storage.StorageV2
		info.Binlogs = result.GetInsertLogs()
		if result.GetPkLog() != nil {
			info.Statslogs = []*datapb.FieldBinlog{result.GetPkLog()}
		}
		info.Bm25Statslogs = result.GetBm25Logs()
	}
	return NewSegmentInfo(info)
}

func updateImportV3SchemaVersion(segmentID int64, schemaVersion int32) UpdateOperator {
	return func(pack *updateSegmentPack) bool {
		segment := pack.Get(segmentID)
		if segment == nil {
			return pack.fail(merr.WrapErrDataIntegrityMsg("import v3 segment %d is missing", segmentID))
		}
		segment.SchemaVersion = schemaVersion
		return true
	}
}

func acceptedImportSegmentMatches(segment *SegmentInfo, result *datapb.SegmentResult, sorted, namespaceSorted bool) bool {
	if segment.GetNumOfRows() != result.GetRows() || segment.GetManifestPath() != result.GetManifestPath() ||
		segment.GetStartPosition().GetTimestamp() != result.GetStatistics().GetTimestampFrom() || segment.GetDmlPosition().GetTimestamp() != result.GetStatistics().GetTimestampTo() ||
		segment.GetIsSorted() != sorted || segment.GetIsSortedByNamespace() != namespaceSorted ||
		!slices.Equal(segment.GetExpirQuantiles(), result.GetExpirationQuantiles()) ||
		!proto.Equal(segment.GetStats(), result.GetStatistics()) {
		return false
	}
	// Manifest-backed segments intentionally do not persist FieldBinlog arrays.
	// For V2, the arrays remain part of the durable accepted projection.
	if result.GetManifestPath() == "" {
		statslogs := []*datapb.FieldBinlog(nil)
		if result.GetPkLog() != nil {
			statslogs = []*datapb.FieldBinlog{result.GetPkLog()}
		}
		return proto.Equal(&datapb.SegmentInfo{Binlogs: segment.GetBinlogs(), Statslogs: segment.GetStatslogs(), Bm25Statslogs: segment.GetBm25Statslogs()},
			&datapb.SegmentInfo{Binlogs: result.GetInsertLogs(), Statslogs: statslogs, Bm25Statslogs: result.GetBm25Logs()})
	}
	return true
}

func dropImportV3Segments(segmentIDs []int64) UpdateOperator {
	return func(pack *updateSegmentPack) bool {
		updated := false
		for _, segmentID := range segmentIDs {
			segment := pack.Get(segmentID)
			if segment == nil || !segment.GetIsImporting() ||
				segment.GetState() == commonpb.SegmentState_Dropped {
				continue
			}
			updateSegStateAndPrepareMetrics(segment, commonpb.SegmentState_Dropped, pack.metricMutation)
			segment.DroppedAt = uint64(time.Now().UnixNano())
			updated = true
		}
		return updated
	}
}
