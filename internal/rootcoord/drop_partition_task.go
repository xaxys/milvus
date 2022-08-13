package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type dropPartitionTask struct {
	baseRedoTask
	Req *milvuspb.DropPartitionRequest
}

func (t *dropPartitionTask) Prepare(ctx context.Context) error {
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.GetTs())
	if err != nil {
		// Is this idempotent?
		return err
	}

	var partID = common.InvalidPartitionID
	for _, partition := range collMeta.Partitions {
		if partition.PartitionName == t.Req.GetPartitionName() {
			partID = partition.PartitionID
			break
		}
	}
	if partID == common.InvalidPartitionID {
		return nil
	}

	t.prepareLogger()
	t.AddSyncStep(&ExpireCollectionCacheStep{
		baseStep:                  baseStep{core: t.core},
		ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{CollectionId: collMeta.CollectionID, Timestamp: t.GetTs()},
	})
	t.AddSyncStep(&DisablePartitionMetaStep{
		baseStep:                 baseStep{core: t.core},
		DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{CollectionId: collMeta.CollectionID, PartitionId: partID, Timestamp: t.GetTs()},
	})
	t.AddAsyncStep(&DeletePartitionDataStep{
		baseStep:                baseStep{core: t.core},
		DeletePartitionDataStep: rootcoordpb.DeletePartitionDataStep{CollectionId: collMeta.CollectionID, PartitionId: partID},
	})
	t.AddAsyncStep(&DeletePartitionMetaStep{
		baseStep:                baseStep{core: t.core},
		DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{CollectionId: collMeta.CollectionID, PartitionId: partID, Timestamp: t.GetTs()},
	})

	return nil
}
