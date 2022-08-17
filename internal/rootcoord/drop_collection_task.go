package rootcoord

import (
	"context"
	"fmt"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type dropCollectionTask struct {
	baseTaskV2
	Req *milvuspb.DropCollectionRequest
}

func (t *dropCollectionTask) validate() error {
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_DropCollection); err != nil {
		return err
	}
	if t.core.meta.IsAlias(t.Req.GetCollectionName()) {
		return fmt.Errorf("cannot drop the collection via alias = %s", t.Req.CollectionName)
	}
	return nil
}

func (t *dropCollectionTask) Prepare(ctx context.Context) error {
	return t.validate()
}

func (t *dropCollectionTask) Execute(ctx context.Context) error {
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), typeutil.MaxTimestamp)
	if err != nil {
		// make dropping collection idempotent.
		return nil
	}

	ts := t.GetTs()

	redoTask := newBaseRedoTask()

	redoTask.AddSyncStep(&ExpireCacheStep{
		baseStep:        baseStep{core: t.core},
		collectionNames: []string{collMeta.Name},
		collectionId:    collMeta.CollectionID,
		ts:              ts,
	})
	// TODO: corner case, once expiring cache is done and a read(describe) request entered before you mark collection deleted.
	redoTask.AddSyncStep(&ChangeCollectionStateStep{
		baseStep:     baseStep{core: t.core},
		collectionId: collMeta.CollectionID,
		state:        pb.CollectionState_CollectionDropping,
		ts:           ts,
	})

	redoTask.AddAsyncStep(&ReleaseCollectionStep{
		baseStep:     baseStep{core: t.core},
		collectionId: collMeta.CollectionID,
	})
	// TODO: remove index
	redoTask.AddAsyncStep(&DeleteCollectionDataStep{
		baseStep: baseStep{core: t.core},
		coll:     collMeta,
		ts:       ts,
	})
	redoTask.AddAsyncStep(&RemoveDmlChannelsStep{
		baseStep:  baseStep{core: t.core},
		pchannels: collMeta.PhysicalChannelNames,
	})
	redoTask.AddAsyncStep(&RemoveDeltaChannelsStep{
		baseStep:     baseStep{core: t.core},
		dmlPChannels: collMeta.PhysicalChannelNames,
	})
	redoTask.AddAsyncStep(&DeleteCollectionMetaStep{
		baseStep:     baseStep{core: t.core},
		collectionId: collMeta.CollectionID,
		ts:           ts,
	})

	return redoTask.Execute(ctx)
}
