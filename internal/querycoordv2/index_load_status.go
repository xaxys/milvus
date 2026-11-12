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

	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// GetIndexLoadStatus answers, per index, whether the replicas that serve the
// collection actually carry it on the segments they serve.
//
// The answer is deliberately index-shaped, not distribution-shaped: a caller learns
// "this index is usable / partially usable / not there, and roughly why", which is
// what decides whether a query will use the index or fall back to scanning the raw
// column. The segment and replica counts are carried along only as evidence.
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
	indexInfos, err := s.broker.ListIndexes(ctx, collectionID)
	if err != nil {
		mlog.Warn(ctx, "failed to list indexes for load status",
			mlog.Int64("collectionID", collectionID), mlog.Err(err))
		resp.Status = merr.Status(err)
		return resp, nil
	}

	wanted := typeutil.NewSet[int64](req.GetIndexIDs()...)
	replicas := s.meta.GetByCollection(ctx, collectionID)
	segments := s.dist.SegmentDistManager.GetByFilter(meta.WithCollectionID(collectionID))

	for _, info := range indexInfos {
		if wanted.Len() > 0 && !wanted.Contain(info.GetIndexID()) {
			continue
		}

		status := &querypb.IndexLoadStatus{
			IndexID:          info.GetIndexID(),
			FieldID:          info.GetFieldID(),
			IndexName:        info.GetIndexName(),
			ExpectedReplicas: int64(len(replicas)),
			ExpectedSegments: int64(len(segments)),
		}

		// A segment counts as ready when the node that loaded it reported the index
		// as enabled. The same segment may be present once per replica.
		for _, segment := range segments {
			if indexReadyOnSegment(segment, info.GetIndexID()) {
				status.ReadySegments++
			}
		}

		// A replica counts as ready only when every segment it serves has the index:
		// a replica that is missing it on one shard still cannot answer from the index.
		for _, replica := range replicas {
			nodes := typeutil.NewSet[int64](replica.GetNodes()...)
			ready := true
			for _, segment := range segments {
				if !nodes.Contain(segment.Node) {
					continue
				}
				if !indexReadyOnSegment(segment, info.GetIndexID()) {
					ready = false
					break
				}
			}
			if ready {
				status.ReadyReplicas++
			}
		}

		status.State, status.Reason = classifyIndexLoad(stateEvidence{
			replicas:  int64(len(replicas)),
			readyReps: status.ReadyReplicas,
			segments:  status.ExpectedSegments,
			readySegs: status.ReadySegments,
		})
		resp.Statuses = append(resp.Statuses, status)
	}
	return resp, nil
}

func indexReadyOnSegment(segment *meta.Segment, indexID int64) bool {
	info, ok := segment.IndexInfo[indexID]
	return ok && info.GetEnableIndex()
}

type stateEvidence struct {
	replicas  int64
	readyReps int64
	segments  int64
	readySegs int64
}

// classifyIndexLoad turns the counts into the three states a caller can act on.
// "Partially loaded" is the interesting one: the index exists somewhere, so queries
// may use it on some shards and scan on others -- worth surfacing rather than
// reporting a single happy/not-happy bit.
func classifyIndexLoad(e stateEvidence) (querypb.IndexLoadState, string) {
	switch {
	case e.replicas == 0:
		return querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, "collection has no replica"
	case e.readySegs == 0:
		return querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, "no loaded segment carries the index yet"
	case e.readyReps == e.replicas && e.readySegs == e.segments:
		return querypb.IndexLoadState_INDEX_LOAD_LOADED, ""
	case e.readyReps == e.replicas:
		// Every replica covers the index, but the distribution still holds segments
		// that do not (for example segments being replaced during a handoff).
		return querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, "some loaded segments do not carry the index yet"
	default:
		return querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, "some replicas do not carry the index on every segment"
	}
}
