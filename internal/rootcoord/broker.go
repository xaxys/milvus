package rootcoord

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/milvus-io/milvus/internal/proto/indexpb"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	ms "github.com/milvus-io/milvus/internal/mq/msgstream"

	"github.com/milvus-io/milvus/internal/proto/datapb"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"go.uber.org/zap"
)

type watchInfo struct {
	ts           Timestamp
	collectionID UniqueID
	partitionID  UniqueID
	schema       []byte
	vChannels    []string
	pChannels    []string
}

func genCreateCollectionMsg(ctx context.Context, info *watchInfo) *ms.MsgPack {
	msgPack := ms.MsgPack{}
	baseMsg := ms.BaseMsg{
		Ctx:            ctx,
		BeginTimestamp: info.ts,
		EndTimestamp:   info.ts,
		HashValues:     []uint32{0},
	}
	msg := &ms.CreateCollectionMsg{
		BaseMsg: baseMsg,
		CreateCollectionRequest: internalpb.CreateCollectionRequest{
			Base:                 &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection, Timestamp: info.ts},
			CollectionID:         info.collectionID,
			PartitionID:          info.partitionID,
			Schema:               info.schema,
			VirtualChannelNames:  info.vChannels,
			PhysicalChannelNames: info.pChannels,
		},
	}
	msgPack.Msgs = append(msgPack.Msgs, msg)
	return &msgPack
}

// Broker communicates with other components.
type Broker interface {
	ReleaseCollection(ctx context.Context, collectionID UniqueID) error
	GetQuerySegmentInfo(ctx context.Context, collectionID int64, segIDs []int64) (retResp *querypb.GetSegmentInfoResponse, retErr error)

	WatchChannels(ctx context.Context, info *watchInfo) error
	UnwatchChannels(ctx context.Context, info *watchInfo) error
	AddSegRefLock(ctx context.Context, taskID int64, segIDs []int64) error
	ReleaseSegRefLock(ctx context.Context, taskID int64, segIDs []int64) error
	Flush(ctx context.Context, cID int64, segIDs []int64) error
	Import(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error)

	GetIndexStates(ctx context.Context, IndexBuildIDs []int64) (idxInfo []*indexpb.IndexInfo, retErr error)
}

type ServerBroker struct {
	s *RootCoord
}

func newServerBroker(s *RootCoord) *ServerBroker {
	return &ServerBroker{s: s}
}

func (b *ServerBroker) ReleaseCollection(ctx context.Context, collectionID UniqueID) error {
	log.Info("releasing collection", zap.Int64("collection", collectionID))

	if err := funcutil.WaitForComponentHealthy(ctx, b.s.queryCoord, "QueryCoord", 100, time.Millisecond*200); err != nil {
		log.Error("failed to release collection, querycoord not healthy", zap.Error(err), zap.Int64("collection", collectionID))
		return err
	}

	resp, err := b.s.queryCoord.ReleaseCollection(ctx, &querypb.ReleaseCollectionRequest{
		Base:         &commonpb.MsgBase{MsgType: commonpb.MsgType_ReleaseCollection},
		CollectionID: collectionID,
		NodeID:       b.s.session.ServerID,
	})
	if err != nil {
		return err
	}

	if resp.GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("failed to release collection, code: %s, reason: %s", resp.GetErrorCode(), resp.GetReason())
	}

	log.Info("done to release collection", zap.Int64("collection", collectionID))
	return nil
}

func (b *ServerBroker) GetQuerySegmentInfo(ctx context.Context, collectionID int64, segIDs []int64) (retResp *querypb.GetSegmentInfoResponse, retErr error) {
	resp, err := b.s.queryCoord.GetSegmentInfo(ctx, &querypb.GetSegmentInfoRequest{
		Base: &commonpb.MsgBase{
			MsgType:  commonpb.MsgType_GetSegmentState,
			SourceID: b.s.session.ServerID,
		},
		CollectionID: collectionID,
		SegmentIDs:   segIDs,
	})
	return resp, err
}

func toKeyDataPairs(m map[string][]byte) []*commonpb.KeyDataPair {
	ret := make([]*commonpb.KeyDataPair, 0, len(m))
	for k, data := range m {
		ret = append(ret, &commonpb.KeyDataPair{
			Key:  k,
			Data: data,
		})
	}
	return ret
}

func (b *ServerBroker) WatchChannels(ctx context.Context, info *watchInfo) error {
	log.Info("watching channels", zap.Uint64("ts", info.ts), zap.Int64("collection", info.collectionID), zap.Strings("vChannels", info.vChannels), zap.Strings("pChannels", info.pChannels))

	if err := funcutil.WaitForComponentHealthy(ctx, b.s.dataCoord, "DataCoord", 100, time.Millisecond*200); err != nil {
		return err
	}

	// TODO: It seems that we can send a null message into it. Check whether datanodes or querynodes will panic if null message was received.
	msg := genCreateCollectionMsg(ctx, info)
	startPositions, err := b.s.chanTimeTick.broadcastMarkDmlChannels(info.pChannels, msg)
	if err != nil {
		return fmt.Errorf("failed to get latest positions, err: %s, vChannels: %v, pChannels: %v", err.Error(), info.vChannels, info.pChannels)
	}

	resp, err := b.s.dataCoord.WatchChannels(ctx, &datapb.WatchChannelsRequest{
		CollectionID:   info.collectionID,
		ChannelNames:   info.vChannels,
		StartPositions: toKeyDataPairs(startPositions),
	})
	if err != nil {
		return err
	}

	if resp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("failed to watch channels, code: %s, reason: %s", resp.GetStatus().GetErrorCode(), resp.GetStatus().GetReason())
	}

	log.Info("done to watch channels", zap.Uint64("ts", info.ts), zap.Int64("collection", info.collectionID), zap.Strings("vChannels", info.vChannels), zap.Strings("pChannels", info.pChannels))
	return nil
}

func (b *ServerBroker) UnwatchChannels(ctx context.Context, info *watchInfo) error {
	// TODO: release flowgraph on datanodes.
	return nil
}

func (b *ServerBroker) AddSegRefLock(ctx context.Context, taskID int64, segIDs []int64) error {
	log.Info("acquiring seg lock",
		zap.Int64s("segment IDs", segIDs),
		zap.Int64("node ID", b.s.session.ServerID))
	resp, err := b.s.dataCoord.AcquireSegmentLock(ctx, &datapb.AcquireSegmentLockRequest{
		SegmentIDs: segIDs,
		NodeID:     b.s.session.ServerID,
		TaskID:     taskID,
	})
	if err != nil {
		return err
	}
	if resp.GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("failed to acquire segment lock %s", resp.GetReason())
	}
	log.Info("acquire seg lock succeed",
		zap.Int64s("segment IDs", segIDs),
		zap.Int64("node ID", b.s.session.ServerID))
	return nil
}

func (b *ServerBroker) ReleaseSegRefLock(ctx context.Context, taskID int64, segIDs []int64) error {
	log.Info("releasing seg lock",
		zap.Int64s("segment IDs", segIDs),
		zap.Int64("node ID", b.s.session.ServerID))
	resp, err := b.s.dataCoord.ReleaseSegmentLock(ctx, &datapb.ReleaseSegmentLockRequest{
		SegmentIDs: segIDs,
		NodeID:     b.s.session.ServerID,
		TaskID:     taskID,
	})
	if err != nil {
		return err
	}
	if resp.GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("failed to release segment lock %s", resp.GetReason())
	}
	log.Info("release seg lock succeed",
		zap.Int64s("segment IDs", segIDs),
		zap.Int64("node ID", b.s.session.ServerID))
	return nil
}

func (b *ServerBroker) Flush(ctx context.Context, cID int64, segIDs []int64) error {
	resp, err := b.s.dataCoord.Flush(ctx, &datapb.FlushRequest{
		Base: &commonpb.MsgBase{
			MsgType:  commonpb.MsgType_Flush,
			SourceID: b.s.session.ServerID,
		},
		DbID:         0,
		SegmentIDs:   segIDs,
		CollectionID: cID,
	})
	if err != nil {
		return errors.New("failed to call flush to data coordinator: " + err.Error())
	}
	if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
		return errors.New(resp.Status.Reason)
	}
	log.Info("flush on collection succeed", zap.Int64("collection ID", cID))
	return nil
}

func (b *ServerBroker) Import(ctx context.Context, req *datapb.ImportTaskRequest) (*datapb.ImportTaskResponse, error) {
	return b.s.dataCoord.Import(ctx, req)
}

func (b *ServerBroker) GetIndexStates(ctx context.Context, IndexBuildIDs []int64) (idxInfo []*indexpb.IndexInfo, retErr error) {
	res, err := b.s.indexCoord.GetIndexStates(ctx, &indexpb.GetIndexStatesRequest{
		IndexBuildIDs: IndexBuildIDs,
	})
	if err != nil {
		log.Error("RootCoord failed to get index states from IndexCoord.", zap.Error(err))
		return nil, err
	}
	log.Debug("got index states", zap.String("get index state result", res.String()))
	if res.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		log.Error("Get index states failed.",
			zap.String("error_code", res.GetStatus().GetErrorCode().String()),
			zap.String("reason", res.GetStatus().GetReason()))
		return nil, fmt.Errorf(res.GetStatus().GetErrorCode().String())
	}
	return res.GetStates(), nil
}
