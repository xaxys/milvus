package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type dropPartitionTask struct {
	baseTaskV2
	Req    *milvuspb.DropPartitionRequest
	collID typeutil.UniqueID
	partID typeutil.UniqueID
}

func (t *dropPartitionTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_DropPartition); err != nil {
		return err
	}
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.GetTs())
	if err != nil {
		// Is this idempotent?
		return err
	}

	t.partID = common.InvalidPartitionID
	for _, partition := range collMeta.Partitions {
		if partition.PartitionName == t.Req.GetPartitionName() {
			t.partID = partition.PartitionID
			break
		}
	}
	if t.partID == common.InvalidPartitionID {
		return nil
	}

	//t.AddSyncStep(&ExpireCollectionCacheStep{
	//	baseStep:                  baseStep{core: t.core},
	//	ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{CollectionId: collMeta.CollectionID, Timestamp: t.GetTs()},
	//})
	//t.AddSyncStep(&DisablePartitionMetaStep{
	//	baseStep:                 baseStep{core: t.core},
	//	DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{CollectionId: collMeta.CollectionID, PartitionId: partID, Timestamp: t.GetTs()},
	//})
	//t.AddAsyncStep(&DeletePartitionDataStep{
	//	baseStep:                baseStep{core: t.core},
	//	DeletePartitionDataStep: rootcoordpb.DeletePartitionDataStep{CollectionId: collMeta.CollectionID, PartitionId: partID},
	//})
	//t.AddAsyncStep(&DeletePartitionMetaStep{
	//	baseStep:                baseStep{core: t.core},
	//	DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{CollectionId: collMeta.CollectionID, PartitionId: partID, Timestamp: t.GetTs()},
	//})

	return nil
}

func (t *dropPartitionTask) Execute(ctx context.Context) error {
	if err := t.core.ExpireMetaCache(ctx, []string{t.Req.GetCollectionName()}, t.collID, t.GetTs()); err != nil {
		return err
	}
	if err := t.core.meta.ChangePartitionState(ctx, t.collID, t.partID, pb.PartitionState_PartitionDropping, t.GetTs()); err != nil {
		return err
	}
	return nil
}

func (t *dropPartitionTask) PostExecute(ctx context.Context) error {
	// TODO: delete partition data

	if err := t.core.meta.RemovePartition(ctx, t.collID, t.partID, t.GetTs()); err != nil {
		return err
	}
	return nil
}
