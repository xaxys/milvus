package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/util/funcutil"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/metastore/model"
)

type Step interface {
	Execute(ctx context.Context) error
}

type baseStep struct {
	core *RootCoord
}

type AddCollectionMetaStep struct {
	baseStep
	coll *model.Collection
}

func (s *AddCollectionMetaStep) Execute(ctx context.Context) error {
	return s.core.meta.AddCollection(ctx, s.coll)
}

type DeleteCollectionMetaStep struct {
	baseStep
	collectionId UniqueID
	ts           Timestamp
}

func (s *DeleteCollectionMetaStep) Execute(ctx context.Context) error {
	return s.core.meta.RemoveCollection(ctx, s.collectionId, s.ts)
}

type AddDmlChannelsStep struct {
	baseStep
	pchannels []string
}

func (s *AddDmlChannelsStep) Execute(ctx context.Context) error {
	s.core.chanTimeTick.addDmlChannels(s.pchannels...)
	return nil
}

type RemoveDmlChannelsStep struct {
	baseStep
	pchannels []string
}

func (s *RemoveDmlChannelsStep) Execute(ctx context.Context) error {
	s.core.chanTimeTick.removeDmlChannels(s.pchannels...)
	return nil
}

type AddDeltaChannelsStep struct {
	baseStep
	dmlPChannels []string
}

func (s *AddDeltaChannelsStep) Execute(ctx context.Context) error {
	var err error
	deltaChanNames := make([]string, len(s.dmlPChannels))
	for i, chanName := range s.dmlPChannels {
		if deltaChanNames[i], err = funcutil.ConvertChannelName(chanName, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta); err != nil {
			return err
		}
	}
	s.core.chanTimeTick.addDeltaChannels(deltaChanNames...)
	return nil
}

type RemoveDeltaChannelsStep struct {
	baseStep
	dmlPChannels []string
}

func (s *RemoveDeltaChannelsStep) Execute(ctx context.Context) error {
	var err error
	deltaChanNames := make([]string, len(s.dmlPChannels))
	for i, chanName := range s.dmlPChannels {
		if deltaChanNames[i], err = funcutil.ConvertChannelName(chanName, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta); err != nil {
			return err
		}
	}
	s.core.chanTimeTick.removeDeltaChannels(deltaChanNames...)
	return nil
}

type WatchChannelsStep struct {
	baseStep
	collectionId UniqueID
	channels     collectionChannels
}

func (s *WatchChannelsStep) Execute(ctx context.Context) error {
	return s.core.watchChannels(ctx, s.collectionId, s.channels.virtualChannels)
}

type UnwatchChannelsStep struct {
	baseStep
	collectionId UniqueID
	channels     collectionChannels
}

func (s *UnwatchChannelsStep) Execute(ctx context.Context) error {
	return s.core.unwatchChannels(ctx, s.collectionId, s.channels.virtualChannels)
}

type ChangeCollectionStateStep struct {
	baseStep
	collectionId UniqueID
	state        pb.CollectionState
	ts           Timestamp
}

func (s *ChangeCollectionStateStep) Execute(ctx context.Context) error {
	return s.core.meta.ChangeCollectionState(ctx, s.collectionId, s.state, s.ts)
}

type ExpireCacheStep struct {
	baseStep
	collectionNames []string
	collectionId    UniqueID
	ts              Timestamp
}

func (s *ExpireCacheStep) Execute(ctx context.Context) error {
	return s.core.ExpireMetaCache(ctx, s.collectionNames, s.collectionId, s.ts)
}

type DeleteCollectionDataStep struct {
	baseStep
	coll *model.Collection
	ts   Timestamp
}

func (s *DeleteCollectionDataStep) Execute(ctx context.Context) error {
	return s.core.notifyDataGC(ctx, s.coll, s.ts)
}

type ReleaseCollectionStep struct {
	baseStep
	collectionId UniqueID
}

func (s *ReleaseCollectionStep) Execute(ctx context.Context) error {
	return s.core.releaseCollection(ctx, s.collectionId)
}

type AddPartitionMetaStep struct {
	baseStep
	partition *model.Partition
}

func (s *AddPartitionMetaStep) Execute(ctx context.Context) error {
	return s.core.meta.AddPartition(ctx, s.partition)
}

type ChangePartitionStateStep struct {
	baseStep
	collectionId UniqueID
	partitionId  UniqueID
	state        pb.PartitionState
	ts           Timestamp
}

func (s *ChangePartitionStateStep) Execute(ctx context.Context) error {
	return s.core.meta.ChangePartitionState(ctx, s.collectionId, s.partitionId, s.state, s.ts)
}

type RemovePartitionMetaStep struct {
	baseStep
	collectionId UniqueID
	partitionId  UniqueID
	ts           Timestamp
}

func (s *RemovePartitionMetaStep) Execute(ctx context.Context) error {
	return s.core.meta.RemovePartition(ctx, s.collectionId, s.partitionId, s.ts)
}

type NullStep struct {
}

func (s *NullStep) Execute(ctx context.Context) error {
	return nil
}
