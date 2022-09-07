package querynode

import (
	"context"
	"errors"
	"fmt"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"go.uber.org/zap"
	"google.golang.org/protobuf/types/known/structpb"
)

// not implement any task
type statistics struct {
	id                 int64
	ts                 Timestamp
	ctx                context.Context
	qs                 *queryShard
	scope              querypb.DataScope
	guaranteeTimestamp Timestamp
	timeoutTimestamp   Timestamp
	tr                 *timerecord.TimeRecorder
	iReq               *internalpb.GetStatisticsRequest
	req                *querypb.GetStatisticsRequest
	Ret                *internalpb.GetStatisticsResponse
	waitCanDoFunc      func(ctx context.Context) error
	reducer            funcutil.MapReducer
}

func (s *statistics) statisticOnStreaming() error {
	// check ctx timeout
	ctx := s.ctx
	if !funcutil.CheckCtxValid(ctx) {
		return errors.New("get statistics context timeout")
	}

	// check if collection has been released, check streaming since it's released first
	_, err := s.qs.metaReplica.getCollectionByID(s.iReq.GetCollectionID())
	if err != nil {
		return err
	}

	s.qs.collection.RLock() // locks the collectionPtr
	defer s.qs.collection.RUnlock()
	if _, released := s.qs.collection.getReleaseTime(); released {
		log.Ctx(ctx).Warn("collection release before do statistics", zap.Int64("msgID", s.id),
			zap.Int64("collectionID", s.iReq.GetCollectionID()))
		return fmt.Errorf("statistic failed, collection has been released, collectionID = %d", s.iReq.GetCollectionID())
	}

	results, _, _, err := statisticStreaming(ctx, s.qs.metaReplica, s.iReq.GetCollectionID(),
		s.iReq.GetPartitionIDs(), s.req.GetDmlChannels()[0])
	if err != nil {
		log.Ctx(ctx).Warn("failed to statistic on streaming data", zap.Int64("msgID", s.id),
			zap.Int64("collectionID", s.iReq.GetCollectionID()), zap.Error(err))
		return err
	}
	return s.reduceResults(results)
}

func (s *statistics) statisticOnHistorical() error {
	// check ctx timeout
	ctx := s.ctx
	if !funcutil.CheckCtxValid(ctx) {
		return errors.New("get statistics context timeout")
	}

	// check if collection has been released, check streaming since it's released first
	_, err := s.qs.metaReplica.getCollectionByID(s.iReq.GetCollectionID())
	if err != nil {
		return err
	}

	s.qs.collection.RLock() // locks the collectionPtr
	defer s.qs.collection.RUnlock()
	if _, released := s.qs.collection.getReleaseTime(); released {
		log.Ctx(ctx).Debug("collection release before do statistics", zap.Int64("msgID", s.id),
			zap.Int64("collectionID", s.iReq.GetCollectionID()))
		return fmt.Errorf("statistic failed, collection has been released, collectionID = %d", s.iReq.GetCollectionID())
	}

	segmentIDs := s.req.GetSegmentIDs()
	results, _, _, err := statisticHistorical(ctx, s.qs.metaReplica, s.iReq.GetCollectionID(), s.iReq.GetPartitionIDs(), segmentIDs)
	if err != nil {
		return err
	}
	return s.reduceResults(results)
}

func (s *statistics) Execute(ctx context.Context) error {
	if err := s.waitCanDoFunc(ctx); err != nil {
		return err
	}
	if s.scope == querypb.DataScope_Streaming {
		return s.statisticOnStreaming()
	} else if s.scope == querypb.DataScope_Historical {
		return s.statisticOnHistorical()
	}
	return fmt.Errorf("statistics do not implement do statistic on all data scope")
}

func (s *statistics) reduceResults(results []map[string]interface{}) error {
	var stats []*structpb.Struct
	for _, result := range results {
		stat, err := structpb.NewStruct(result)
		if err != nil {
			return err
		}
		stats = append(stats, stat)
	}

	result, err := s.reducer.Reduce(stats)
	if err != nil {
		return err
	}

	s.Ret = &internalpb.GetStatisticsResponse{
		Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Stats:  result,
	}
	return nil
}

func newStatistics(ctx context.Context, src *querypb.GetStatisticsRequest, scope querypb.DataScope, qs *queryShard, waitCanDo func(ctx context.Context) error) *statistics {
	target := &statistics{
		ctx:                ctx,
		id:                 src.Req.Base.GetMsgID(),
		ts:                 src.Req.Base.GetTimestamp(),
		scope:              scope,
		qs:                 qs,
		guaranteeTimestamp: src.Req.GetGuaranteeTimestamp(),
		timeoutTimestamp:   src.Req.GetTimeoutTimestamp(),
		tr:                 timerecord.NewTimeRecorder("statistics"),
		iReq:               src.Req,
		req:                src,
		waitCanDoFunc:      waitCanDo,
		reducer:            funcutil.NewMapReducer(funcutil.KeyValuePair2Map(src.Req.GetSchema())),
	}
	return target
}
