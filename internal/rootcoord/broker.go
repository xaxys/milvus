package rootcoord

import (
	"context"
	"fmt"
	"time"

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

	WatchChannels(ctx context.Context, info *watchInfo) error
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
