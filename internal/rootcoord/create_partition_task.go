package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/model"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type createPartitionTask struct {
	baseUndoTask
	Req *milvuspb.CreatePartitionRequest
}

func (t *createPartitionTask) Prepare(ctx context.Context) error {
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.GetTs())
	if err != nil {
		return err
	}

	for _, partition := range collMeta.Partitions {
		if partition.Equal(model.Partition{PartitionName: t.Req.GetPartitionName()}) {
			log.Warn("add duplicate partition", zap.String("collection", t.Req.GetCollectionName()), zap.String("partition", t.Req.GetPartitionName()))
			return nil
		}
	}

	partID, err := t.core.idAllocator.AllocOne()
	if err != nil {
		return err
	}

	t.prepareLogger()

	t.AddStep(&ExpireCollectionCacheStep{
		baseStep: baseStep{core: t.core},
		ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{
			CollectionId: collMeta.CollectionID,
			Timestamp:    t.GetTs(),
		},
	}, &NullStep{})

	t.AddStep(&AddPartitionMetaStep{
		baseStep: baseStep{core: t.core},
		AddPartitionMetaStep: rootcoordpb.AddPartitionMetaStep{
			CollectionId: collMeta.CollectionID,
			PartInfo: &pb.PartitionInfo{
				PartitionID:               partID,
				PartitionName:             t.Req.GetPartitionName(),
				PartitionCreatedTimestamp: t.GetTs(),
				CollectionId:              collMeta.CollectionID,
				State:                     pb.PartitionState_PartitionCreating,
			},
			Timestamp: t.GetTs(),
		},
	}, &DeletePartitionMetaStep{
		baseStep: baseStep{core: t.core},
		DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{
			CollectionId: collMeta.CollectionID,
			PartitionId:  partID,
			Timestamp:    t.GetTs(),
		},
	})

	t.AddStep(&EnablePartitionMetaStep{
		baseStep: baseStep{core: t.core},
		EnablePartitionMetaStep: rootcoordpb.EnablePartitionMetaStep{
			CollectionId: collMeta.CollectionID,
			PartitionId:  partID,
			Timestamp:    t.GetTs(),
		},
	}, &DisablePartitionMetaStep{
		baseStep: baseStep{core: t.core},
		DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{
			CollectionId: collMeta.CollectionID,
			PartitionId:  partID,
			Timestamp:    t.GetTs(),
		},
	})

	return nil
}
