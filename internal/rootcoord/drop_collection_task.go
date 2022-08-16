package rootcoord

import (
	"context"
	"fmt"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/util/funcutil"

	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type dropCollectionTask struct {
	baseTaskV2
	Req      *milvuspb.DropCollectionRequest
	collMeta *model.Collection
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

func (t *dropCollectionTask) prepareMeta(ctx context.Context) error {
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), typeutil.MaxTimestamp)
	if err != nil {
		// make dropping collection idempotent.
		return nil
	}
	t.collMeta = collMeta
	return nil
}

func (t *dropCollectionTask) Execute(ctx context.Context) error {
	collMeta := t.collMeta.Clone()
	fmt.Println("coll meta: ", collMeta)
	ts := t.GetTs()
	//ddReq := internalpb.DropCollectionRequest{
	//	Base:           t.Req.Base,
	//	DbName:         t.Req.GetDbName(),
	//	CollectionName: t.Req.GetCollectionName(),
	//	CollectionID:   collMeta.CollectionID,
	//}

	err := t.core.ExpireMetaCache(ctx, []string{t.collMeta.Name}, t.collMeta.CollectionID, ts)
	if err != nil {
		return err
	}

	err = t.core.meta.ChangeCollectionState(ctx, collMeta.CollectionID, pb.CollectionState_CollectionDropping, ts)
	if err != nil {
		return err
	}

	return nil
}

func (t *dropCollectionTask) Prepare(ctx context.Context) error {
	if err := t.validate(); err != nil {
		return err
	}
	if err := t.prepareMeta(ctx); err != nil {
		return err
	}
	return nil
}

func (t *dropCollectionTask) PostExecute(ctx context.Context) error {
	err := t.core.releaseCollection(ctx, t.collMeta.CollectionID)
	if err != nil {
		return err
	}

	// TODO: remove index

	// TODO: delete collection data

	chanNames := t.collMeta.PhysicalChannelNames
	t.core.chanTimeTick.removeDmlChannels(chanNames...)

	// remove delta channels
	deltaChanNames := make([]string, len(chanNames))
	for i, chanName := range chanNames {
		if deltaChanNames[i], err = funcutil.ConvertChannelName(chanName, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta); err != nil {
			return err
		}
	}
	t.core.chanTimeTick.removeDeltaChannels(deltaChanNames...)

	err = t.core.meta.RemoveCollection(ctx, t.collMeta.CollectionID, t.GetTs())
	if err != nil {
		return err
	}

	return nil
}
