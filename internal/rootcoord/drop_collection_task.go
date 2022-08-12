package rootcoord

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type dropCollectionTask struct {
	baseRedoTask
	Req      *milvuspb.DropCollectionRequest
	collMeta *model.Collection
}

func (t *dropCollectionTask) validate() error {
	if t.Req.GetBase().GetMsgType() != commonpb.MsgType_DropCollection {
		return fmt.Errorf("drop collection, msg type = %s", t.Req.GetBase().GetMsgType())
	}
	if t.core.meta.IsAlias(t.Req.GetCollectionName()) {
		return fmt.Errorf("cannot drop the collection via alias = %s", t.Req.CollectionName)
	}
	return nil
}

func (t *dropCollectionTask) prepareMeta(ctx context.Context) error {
	collMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	t.collMeta = collMeta
	return nil
}

func (t *dropCollectionTask) prepareStep() error {
	t.prepareLogger()

	collMeta := t.collMeta.Clone()
	fmt.Println("coll meta: ", collMeta)
	ts := t.GetTs()
	ddReq := internalpb.DropCollectionRequest{
		Base:           t.Req.Base,
		DbName:         t.Req.GetDbName(),
		CollectionName: t.Req.GetCollectionName(),
		CollectionID:   collMeta.CollectionID,
	}

	t.AddSyncStep(&ExpireCollectionCacheStep{
		baseStep: baseStep{
			core: t.core,
		},
		ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{
			CollectionId: collMeta.CollectionID,
			Timestamp:    ts,
		},
	})

	t.AddSyncStep(&DisableCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DisableCollectionMetaStep: rootcoordpb.DisableCollectionMetaStep{
			CollectionId: collMeta.CollectionID,
			Timestamp:    ts,
		},
	})

	t.AddAsyncStep(&ReleaseCollectionStep{
		baseStep: baseStep{
			core: t.core,
		},
		ReleaseCollectionStep: rootcoordpb.ReleaseCollectionStep{
			CollectionId: collMeta.CollectionID,
			Timestamp:    ts,
		},
	})

	t.AddAsyncStep(&RemoveIndexStep{
		baseStep: baseStep{
			core: t.core,
		},
		RemoveIndexStep: rootcoordpb.RemoveIndexStep{
			//IndexId: indexIDs,
		},
	})

	t.AddAsyncStep(&DeleteCollectionDataStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeleteCollectionDataStep: rootcoordpb.DeleteCollectionDataStep{
			CollectionId: collMeta.CollectionID,
			Request:      &ddReq,
			Pchannels:    collMeta.PhysicalChannelNames,
		},
	})

	t.AddAsyncStep(&RemoveChannelStep{
		baseStep: baseStep{
			core: t.core,
		},
		RemoveChannelStep: rootcoordpb.RemoveChannelStep{
			Pchannels: collMeta.PhysicalChannelNames,
		},
	})

	for _, partition := range collMeta.Partitions {
		t.AddAsyncStep(&DeletePartitionMetaStep{
			baseStep: baseStep{
				core: t.core,
			},
			DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{
				CollectionId: collMeta.CollectionID,
				PartitionId:  partition.PartitionID,
				Timestamp:    ts,
			},
		})
	}

	for i := 0; i < len(collMeta.Fields); i = i + maxTxnNum {
		end := min(i+maxTxnNum, len(collMeta.Fields))
		fieldIds := make([]UniqueID, 0, end-i)
		for j := i; j < end; j++ {
			fieldIds = append(fieldIds, collMeta.Fields[j].FieldID)
		}
		t.AddAsyncStep(&RemoveFieldsMetaStep{
			baseStep: baseStep{
				core: t.core,
			},
			RemoveFieldsMetaStep: rootcoordpb.RemoveFieldsMetaStep{
				CollectionId: collMeta.CollectionID,
				FieldIds:     fieldIds,
				Timestamp:    ts,
			},
		})
	}

	t.AddAsyncStep(&DeleteCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeleteCollectionMetaStep: rootcoordpb.DeleteCollectionMetaStep{
			CollectionId: collMeta.CollectionID,
			Timestamp:    ts,
		},
	})

	return nil
}

func (t *dropCollectionTask) Prepare(ctx context.Context) error {
	if err := t.validate(); err != nil {
		return err
	}
	if err := t.prepareMeta(ctx); err != nil {
		return err
	}
	return t.prepareStep()
}
