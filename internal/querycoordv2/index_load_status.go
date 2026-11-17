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

package querycoordv2

import (
	"context"

	"github.com/milvus-io/milvus/internal/views/coord/coordview"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// GetIndexLoadStatus is an internal Coordinator RPC that answers, per index,
// whether the QueryView replicas serving the collection actually carry it on
// the segments they serve.
func (s *Server) GetIndexLoadStatus(
	ctx context.Context,
	req *querypb.GetIndexLoadStatusRequest,
) (*querypb.GetIndexLoadStatusResponse, error) {
	resp := &querypb.GetIndexLoadStatusResponse{Status: merr.Success()}
	if err := merr.CheckHealthy(s.State()); err != nil {
		resp.Status = merr.Status(err)
		return resp, nil
	}

	collectionID := req.GetCollectionID()
	if collectionID <= 0 {
		resp.Status = merr.Status(merr.WrapErrParameterInvalidMsg("collectionID must be positive"))
		return resp, nil
	}
	indexInfos, err := s.broker.ListIndexes(ctx, collectionID)
	if err != nil {
		mlog.Warn(ctx, "failed to list indexes for load status",
			mlog.Int64("collectionID", collectionID), mlog.Err(err))
		resp.Status = merr.Status(err)
		return resp, nil
	}

	wanted := typeutil.NewSet[int64](req.GetIndexIDs()...)
	ev, err := s.indexLoadEvidence(ctx, collectionID)
	if err != nil {
		mlog.Warn(ctx, "failed to collect index load evidence",
			mlog.Int64("collectionID", collectionID), mlog.Err(err))
		resp.Status = merr.Status(err)
		return resp, nil
	}

	placementsByShard := make(map[qviews.ShardID][]indexPlacement, len(ev.expectedChannels)*len(ev.replicaIDs))
	for _, placement := range ev.placements {
		if _, isUp := ev.upShardRequired[placement.viewShard]; isUp {
			placementsByShard[placement.viewShard] = append(placementsByShard[placement.viewShard], placement)
		}
	}
	totalShards := int64(len(ev.replicaIDs) * len(ev.expectedChannels))

	for _, info := range indexInfos {
		if wanted.Len() > 0 && !wanted.Contain(info.GetIndexID()) {
			continue
		}

		indexID := info.GetIndexID()
		status := &querypb.IndexLoadStatus{
			IndexID:          indexID,
			FieldID:          info.GetFieldID(),
			IndexName:        info.GetIndexName(),
			ExpectedReplicas: int64(len(ev.replicaIDs)),
			ExpectedSegments: ev.expectedSegments,
		}

		for _, placement := range ev.placements {
			if placement.carries(indexID) {
				status.ReadySegments++
			}
		}

		var readyShards int64
		for _, replicaID := range ev.replicaIDs {
			if len(ev.expectedChannels) == 0 {
				continue
			}
			replicaReady := true
			for _, channel := range ev.expectedChannels {
				shard := qviews.ShardID{ReplicaID: replicaID, VChannel: channel}
				reqSet, isUp := ev.upShardRequired[shard]
				if !isUp {
					replicaReady = false
					continue
				}
				shardPlacements := placementsByShard[shard]
				shardReady := true
				if len(shardPlacements) == 0 {
					_, shardReady = reqSet[indexID]
				} else {
					for _, placement := range shardPlacements {
						if !placement.carries(indexID) {
							shardReady = false
							break
						}
					}
				}
				if shardReady {
					readyShards++
				} else {
					replicaReady = false
				}
			}
			if replicaReady {
				status.ReadyReplicas++
			}
		}

		invisible := !common.IsIndexInfoVisible(info.GetIndexParams(), info.GetUserIndexParams())
		status.State, status.Reason = classifyIndexLoad(stateEvidence{
			replicas:    int64(len(ev.replicaIDs)),
			readyReps:   status.ReadyReplicas,
			segments:    status.ExpectedSegments,
			readySegs:   status.ReadySegments,
			totalShards: totalShards,
			readyShards: readyShards,
			invisible:   invisible,
		})
		resp.Statuses = append(resp.Statuses, status)
	}
	return resp, nil
}

type indexPlacement struct {
	node      int64
	viewShard qviews.ShardID
	indexes   map[int64]struct{}
}

func (p indexPlacement) carries(indexID int64) bool {
	_, ok := p.indexes[indexID]
	return ok
}

type indexCollectionEvidence struct {
	replicaIDs       []int64
	expectedChannels []string
	placements       []indexPlacement
	upShardRequired  map[qviews.ShardID]map[int64]struct{}
	expectedSegments int64
}

func (s *Server) indexLoadEvidence(ctx context.Context, collectionID int64) (indexCollectionEvidence, error) {
	var ev indexCollectionEvidence
	runtime := s.qviewsRuntime
	if runtime == nil || runtime.loadConfigStore == nil || runtime.shardViewRegistry == nil ||
		runtime.resources == nil || runtime.resources.VChannels == nil {
		return ev, merr.WrapErrServiceInternalMsg("query view load metadata is not initialized")
	}
	cfg := runtime.loadConfigStore.GetConfig(collectionID)
	if cfg == nil {
		return ev, nil
	}
	ev.replicaIDs = make([]int64, 0, len(cfg.Replicas))
	for _, r := range cfg.Replicas {
		ev.replicaIDs = append(ev.replicaIDs, r.ReplicaID)
	}

	channels, err := runtime.resources.VChannels(ctx, collectionID)
	if err != nil {
		return ev, err
	}
	ev.expectedChannels = channels

	allStats := runtime.shardViewRegistry.SnapshotForCollection(collectionID).StatsMap()
	expectedShards := make(map[qviews.ShardID]struct{}, len(ev.replicaIDs)*len(ev.expectedChannels))
	for _, replicaID := range ev.replicaIDs {
		for _, channel := range ev.expectedChannels {
			expectedShards[qviews.ShardID{ReplicaID: replicaID, VChannel: channel}] = struct{}{}
		}
	}
	stats := make(map[qviews.ShardID]*coordview.ShardStats, len(expectedShards))
	for shard, stat := range allStats {
		if _, ok := expectedShards[shard]; ok {
			stats[shard] = stat
		}
	}

	requiredFn := func(version uint64) (map[int64]struct{}, bool) {
		return runtime.loadConfigStore.RequiredIndexIDs(collectionID, version)
	}
	ev.placements = qviewsPlacements(stats, requiredFn)

	ev.upShardRequired = make(map[qviews.ShardID]map[int64]struct{}, len(stats))
	segmentsPerChannel := make(map[string]int, len(ev.expectedChannels))
	hasChannelPlacement := make(map[string]bool, len(ev.expectedChannels))
	for shard, stat := range stats {
		if stat == nil {
			continue
		}
		if stat.PreparingPlacement != nil {
			hasChannelPlacement[shard.VChannel] = true
			segmentsPerChannel[shard.VChannel] = max(segmentsPerChannel[shard.VChannel], len(stat.PreparingPlacement.Assignments))
		}
		if stat.UpPlacement != nil {
			hasChannelPlacement[shard.VChannel] = true
			segmentsPerChannel[shard.VChannel] = max(segmentsPerChannel[shard.VChannel], len(stat.UpPlacement.Assignments))
		}
		if stat.UpVersion != nil {
			required, ok := requiredFn(stat.UpLoadInfoVersion)
			if !ok {
				required = make(map[int64]struct{})
			}
			ev.upShardRequired[shard] = required
		}
	}

	if runtime.balanceCache != nil {
		if entry := runtime.balanceCache.GetCollection(collectionID); entry != nil && entry.DataView() != nil {
			dv := entry.DataView()
			var partSet map[int64]struct{}
			if len(cfg.PartitionIDs) > 0 {
				partSet = make(map[int64]struct{}, len(cfg.PartitionIDs))
				for _, pid := range cfg.PartitionIDs {
					partSet[pid] = struct{}{}
				}
			}
			for _, channel := range ev.expectedChannels {
				if hasChannelPlacement[channel] {
					continue
				}
				dvShard := dv.Shard(channel)
				if dvShard == nil {
					continue
				}
				if partSet == nil {
					segmentsPerChannel[channel] = dvShard.SegmentCount
					continue
				}
				var count int
				for _, part := range dvShard.Partitions {
					if part != nil {
						if _, ok := partSet[part.PartitionID]; ok {
							count += len(part.Segments)
						}
					}
				}
				segmentsPerChannel[channel] = count
			}
		}
	}

	for _, replicaID := range ev.replicaIDs {
		for _, channel := range ev.expectedChannels {
			shard := qviews.ShardID{ReplicaID: replicaID, VChannel: channel}
			if stat := stats[shard]; stat != nil && stat.UpVersion != nil && stat.UpPlacement != nil {
				ev.expectedSegments += int64(len(stat.UpPlacement.Assignments))
			} else {
				ev.expectedSegments += int64(segmentsPerChannel[channel])
			}
		}
	}
	return ev, nil
}

func qviewsPlacements(
	stats map[qviews.ShardID]*coordview.ShardStats,
	requiredIndexes func(version uint64) (map[int64]struct{}, bool),
) []indexPlacement {
	var placements []indexPlacement
	for shard, stat := range stats {
		if stat == nil || stat.UpVersion == nil || stat.UpPlacement == nil {
			continue
		}
		required, _ := requiredIndexes(stat.UpLoadInfoVersion)
		for segmentID, node := range stat.UpPlacement.Assignments {
			indexes := required
			if loaded, ok := stat.UpPlacement.SegmentIndexes[segmentID]; ok {
				indexes = loaded
			}
			placements = append(placements, indexPlacement{
				node:      node,
				viewShard: shard,
				indexes:   indexes,
			})
		}
	}
	return placements
}

type stateEvidence struct {
	replicas    int64
	readyReps   int64
	segments    int64
	readySegs   int64
	totalShards int64
	readyShards int64
	invisible   bool
}

func notLoadedReason(e stateEvidence) string {
	if e.invisible {
		return "index is not visible; no loaded segment carries the index"
	}
	return "no loaded segment carries the index yet"
}

func classifyIndexLoad(e stateEvidence) (querypb.IndexLoadState, string) {
	if e.replicas == 0 {
		return querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, "collection has no replica"
	}
	if e.segments == 0 {
		switch {
		case e.readyReps == e.replicas && e.totalShards > 0 && e.readyShards == e.totalShards:
			return querypb.IndexLoadState_INDEX_LOAD_LOADED, ""
		case e.readyShards > 0:
			return querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, "some replicas do not carry the index on every shard"
		default:
			return querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, notLoadedReason(e)
		}
	}
	switch {
	case e.readySegs == 0:
		return querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, notLoadedReason(e)
	case e.readyReps == e.replicas && e.readySegs == e.segments:
		return querypb.IndexLoadState_INDEX_LOAD_LOADED, ""
	case e.readySegs == e.segments:
		return querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, "some replicas do not carry the index on every shard"
	default:
		return querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, "some replicas do not carry the index on every segment"
	}
}
