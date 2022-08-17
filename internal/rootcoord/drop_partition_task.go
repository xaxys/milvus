package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/common"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type dropPartitionTask struct {
	baseTaskV2
	Req      *milvuspb.DropPartitionRequest
	collMeta *model.Collection
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
	t.collMeta = collMeta
	return nil
}

func (t *dropPartitionTask) Execute(ctx context.Context) error {
	partID := common.InvalidPartitionID
	for _, partition := range t.collMeta.Partitions {
		if partition.PartitionName == t.Req.GetPartitionName() {
			partID = partition.PartitionID
			break
		}
	}
	if partID == common.InvalidPartitionID {
		// make dropping partition idempotent.
		return nil
	}

	redoTask := newBaseRedoTask()
	redoTask.AddSyncStep(&ExpireCacheStep{
		baseStep:        baseStep{core: t.core},
		collectionNames: []string{t.Req.GetCollectionName()},
		collectionId:    t.collMeta.CollectionID,
		ts:              t.GetTs(),
	})
	// TODO: corner case, once expiring cache is done and a read(describe) request entered before you mark collection deleted.
	redoTask.AddSyncStep(&ChangePartitionStateStep{
		baseStep:     baseStep{core: t.core},
		collectionId: t.collMeta.CollectionID,
		partitionId:  partID,
		state:        pb.PartitionState_PartitionDropping,
		ts:           t.GetTs(),
	})

	// TODO: release partition when query coord is ready.
	// TODO: notify datacoord to gc partition data when it's ready.
	redoTask.AddAsyncStep(&RemovePartitionMetaStep{
		baseStep:     baseStep{core: t.core},
		collectionId: t.collMeta.CollectionID,
		partitionId:  partID,
		ts:           t.GetTs(),
	})

	return redoTask.Execute(ctx)
}
