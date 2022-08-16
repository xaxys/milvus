package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/model"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type createPartitionTask struct {
	baseTaskV2
	Req    *milvuspb.CreatePartitionRequest
	collID typeutil.UniqueID
	partID typeutil.UniqueID
}

func (t *createPartitionTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_CreatePartition); err != nil {
		return err
	}
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.GetTs())
	if err != nil {
		return err
	}
	t.collID = collMeta.CollectionID

	for _, partition := range collMeta.Partitions {
		if partition.Equal(model.Partition{PartitionName: t.Req.GetPartitionName()}) {
			log.Warn("add duplicate partition", zap.String("collection", t.Req.GetCollectionName()), zap.String("partition", t.Req.GetPartitionName()))
			return nil
		}
	}

	t.partID, err = t.core.idAllocator.AllocOne()
	if err != nil {
		return err
	}

	//t.AddStep(&ExpireCollectionCacheStep{
	//	baseStep: baseStep{core: t.core},
	//	ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{
	//		CollectionId: collMeta.CollectionID,
	//		Timestamp:    t.GetTs(),
	//	},
	//}, &NullStep{})
	//
	//t.AddStep(&AddPartitionMetaStep{
	//	baseStep: baseStep{core: t.core},
	//	AddPartitionMetaStep: rootcoordpb.AddPartitionMetaStep{
	//		CollectionId: collMeta.CollectionID,
	//		PartInfo: &pb.PartitionInfo{
	//			PartitionID:               partID,
	//			PartitionName:             t.Req.GetPartitionName(),
	//			PartitionCreatedTimestamp: t.GetTs(),
	//			CollectionId:              collMeta.CollectionID,
	//			State:                     pb.PartitionState_PartitionCreating,
	//		},
	//		Timestamp: t.GetTs(),
	//	},
	//}, &DeletePartitionMetaStep{
	//	baseStep: baseStep{core: t.core},
	//	DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{
	//		CollectionId: collMeta.CollectionID,
	//		PartitionId:  partID,
	//		Timestamp:    t.GetTs(),
	//	},
	//})
	//
	//t.AddStep(&EnablePartitionMetaStep{
	//	baseStep: baseStep{core: t.core},
	//	EnablePartitionMetaStep: rootcoordpb.EnablePartitionMetaStep{
	//		CollectionId: collMeta.CollectionID,
	//		PartitionId:  partID,
	//		Timestamp:    t.GetTs(),
	//	},
	//}, &DisablePartitionMetaStep{
	//	baseStep: baseStep{core: t.core},
	//	DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{
	//		CollectionId: collMeta.CollectionID,
	//		PartitionId:  partID,
	//		Timestamp:    t.GetTs(),
	//	},
	//})

	return nil
}

func (t *createPartitionTask) Execute(ctx context.Context) error {
	t.core.ExpireMetaCache(ctx, []string{t.Req.CollectionName}, t.collID, t.GetTs())
	if err := t.core.meta.AddPartition(ctx, &model.Partition{
		PartitionID:               t.partID,
		PartitionName:             t.Req.PartitionName,
		PartitionCreatedTimestamp: t.GetTs(),
		CollectionID:              t.collID,
		State:                     pb.PartitionState_PartitionCreated,
	}); err != nil {
		return err
	}

	//if err := t.core.meta.ChangePartitionState(ctx, t.collID, t.partID, pb.PartitionState_PartitionCreated, t.GetTs()); err != nil {
	//	return err
	//}

	return nil
}
