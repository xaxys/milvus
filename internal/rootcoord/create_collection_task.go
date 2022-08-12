package rootcoord

import (
	"context"
	"errors"
	"fmt"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"go.uber.org/zap"

	"github.com/golang/protobuf/proto"

	"github.com/milvus-io/milvus/internal/proto/schemapb"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type collectionChannels struct {
	virtualChannels  []string
	physicalChannels []string
	deltaChannels    []string
}

type createCollectionTask struct {
	baseUndoTask
	Req      *milvuspb.CreateCollectionRequest
	schema   *schemapb.CollectionSchema
	collID   UniqueID
	partID   UniqueID
	channels collectionChannels
}

func (t *createCollectionTask) validate() error {
	if t.Req == nil {
		return errors.New("empty requests")
	}
	if t.Req.GetBase().GetMsgType() != commonpb.MsgType_CreateCollection {
		return fmt.Errorf("create collection, msg type = %s", t.Req.GetBase().GetMsgType())
	}
	return nil
}

func (t *createCollectionTask) validateSchema(schema *schemapb.CollectionSchema) error {
	if t.Req.GetCollectionName() != schema.GetName() {
		return fmt.Errorf("collection name = %s, schema.Name=%s", t.Req.GetCollectionName(), schema.Name)
	}
	if hasSystemFields(schema, []string{RowIDFieldName, TimeStampFieldName}) {
		return fmt.Errorf("schema contains system field: %s, %s", RowIDFieldName, TimeStampFieldName)
	}
	return nil
}

func (t *createCollectionTask) assignFieldId(schema *schemapb.CollectionSchema) {
	for idx := range schema.GetFields() {
		schema.Fields[idx].FieldID = int64(idx + StartOfUserFieldID)
	}
}

func (t *createCollectionTask) appendSysFields(schema *schemapb.CollectionSchema) {
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{
		FieldID:      int64(RowIDField),
		Name:         RowIDFieldName,
		IsPrimaryKey: false,
		Description:  "row id",
		DataType:     schemapb.DataType_Int64,
	})
	schema.Fields = append(schema.Fields, &schemapb.FieldSchema{
		FieldID:      int64(TimeStampField),
		Name:         TimeStampFieldName,
		IsPrimaryKey: false,
		Description:  "time stamp",
		DataType:     schemapb.DataType_Int64,
	})
}

func (t *createCollectionTask) prepareSchema() error {
	var schema schemapb.CollectionSchema
	if err := proto.Unmarshal(t.Req.GetSchema(), &schema); err != nil {
		return err
	}
	if err := t.validateSchema(&schema); err != nil {
		return err
	}
	t.assignFieldId(&schema)
	t.appendSysFields(&schema)
	t.schema = &schema
	return nil
}

func (t *createCollectionTask) assignShardsNum() {
	if t.Req.GetShardsNum() <= 0 {
		t.Req.ShardsNum = 2
	}
}

func (t *createCollectionTask) assignCollectionId() error {
	var err error
	t.collID, err = t.core.idAllocator.AllocOne()
	return err
}

func (t *createCollectionTask) assignPartitionId() error {
	var err error
	t.partID, err = t.core.idAllocator.AllocOne()
	return err
}

func (t *createCollectionTask) assignChannels() error {
	vchanNames := make([]string, t.Req.ShardsNum)
	chanNames := make([]string, t.Req.ShardsNum)
	deltaChanNames := make([]string, t.Req.ShardsNum)
	for i := int32(0); i < t.Req.ShardsNum; i++ {
		vchanNames[i] = fmt.Sprintf("%s_%dv%d", t.core.chanTimeTick.getDmlChannelName(), t.collID, i)
		chanNames[i] = funcutil.ToPhysicalChannel(vchanNames[i])

		deltaChanNames[i] = t.core.chanTimeTick.getDeltaChannelName()
		deltaChanName, err1 := funcutil.ConvertChannelName(chanNames[i], Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta)
		if err1 != nil || deltaChanName != deltaChanNames[i] {
			err1Msg := ""
			if err1 != nil {
				err1Msg = err1.Error()
			}
			log.Debug("dmlChanName deltaChanName mismatch detail", zap.Int32("i", i),
				zap.String("vchanName", vchanNames[i]),
				zap.String("phsicalChanName", chanNames[i]),
				zap.String("deltaChanName", deltaChanNames[i]),
				zap.String("converted_deltaChanName", deltaChanName),
				zap.String("err", err1Msg))
			return fmt.Errorf("dmlChanName %s and deltaChanName %s mis-match", chanNames[i], deltaChanNames[i])
		}
	}
	t.channels = collectionChannels{
		virtualChannels:  vchanNames,
		physicalChannels: chanNames,
		deltaChannels:    deltaChanNames,
	}
	return nil
}

func (t *createCollectionTask) prepareStep() error {
	t.prepareLogger()

	collID := t.collID
	partID := t.partID
	ts := t.GetTs()

	vchanNames := t.channels.virtualChannels
	chanNames := t.channels.physicalChannels

	collInfo := model.Collection{
		CollectionID:         collID,
		Name:                 t.schema.Name,
		Description:          t.schema.Description,
		AutoID:               t.schema.AutoID,
		Fields:               model.UnmarshalFieldModels(t.schema.Fields),
		VirtualChannelNames:  vchanNames,
		PhysicalChannelNames: chanNames,
		ShardsNum:            t.Req.ShardsNum,
		ConsistencyLevel:     t.Req.ConsistencyLevel,
		CreateTime:           ts,
		Partitions: []*model.Partition{
			{
				PartitionID:               partID,
				PartitionName:             Params.CommonCfg.DefaultPartitionName,
				PartitionCreatedTimestamp: ts,
				State:                     pb.PartitionState_PartitionCreating,
			},
		},
		State: pb.CollectionState_CollectionCreating,
	}

	t.AddStep(&AddCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		coll: &collInfo,
	}, &DeleteCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeleteCollectionMetaStep: rootcoordpb.DeleteCollectionMetaStep{
			CollectionId: collID,
			Timestamp:    ts,
		},
	})

	t.AddStep(&CreateChannelStep{
		baseStep: baseStep{
			core: t.core,
		},
		CreateChannelStep: rootcoordpb.CreateChannelStep{
			Pchannels: chanNames,
		},
	}, &RemoveChannelStep{
		baseStep: baseStep{
			core: t.core,
		},
		RemoveChannelStep: rootcoordpb.RemoveChannelStep{
			Pchannels: chanNames,
		},
	})

	t.AddStep(&WatchChannelStep{
		baseStep: baseStep{
			core: t.core,
		},
		WatchChannelStep: rootcoordpb.WatchChannelStep{
			Vchannels: vchanNames,
		},
	}, &UnwatchChannelStep{
		baseStep: baseStep{
			core: t.core,
		},
		UnwatchChannelStep: rootcoordpb.UnwatchChannelStep{
			Vchannels: vchanNames,
		},
	})

	t.AddStep(&EnablePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		EnablePartitionMetaStep: rootcoordpb.EnablePartitionMetaStep{
			CollectionId: collID,
			PartitionId:  partID,
			Timestamp:    ts,
		},
	}, &DisablePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{
			CollectionId: collID,
			PartitionId:  partID,
			Timestamp:    ts,
		},
	})

	for i := 0; i < len(collInfo.Fields); i = i + maxTxnNum {
		end := min(i+maxTxnNum, len(collInfo.Fields))
		fieldIds := make([]UniqueID, 0, end-i)
		for j := i; j < end; j++ {
			fieldIds = append(fieldIds, collInfo.Fields[j].FieldID)
		}
		t.AddStep(&AddFieldsMetaStep{
			baseStep: baseStep{
				core: t.core,
			},
			AddFieldsMetaStep: rootcoordpb.AddFieldsMetaStep{
				CollectionId: collID,
				FieldIds:     fieldIds,
				Timestamp:    ts,
			},
		}, &RemoveFieldsMetaStep{
			baseStep: baseStep{
				core: t.core,
			},
			RemoveFieldsMetaStep: rootcoordpb.RemoveFieldsMetaStep{
				CollectionId: collID,
				FieldIds:     fieldIds,
				Timestamp:    ts,
			},
		})
	}

	t.AddStep(&EnableCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		EnableCollectionMetaStep: rootcoordpb.EnableCollectionMetaStep{
			CollectionId: collID,
			Timestamp:    ts,
		},
	}, &DisableCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DisableCollectionMetaStep: rootcoordpb.DisableCollectionMetaStep{
			CollectionId: collID,
			Timestamp:    ts,
		},
	})

	return nil
}

func (t *createCollectionTask) Prepare(ctx context.Context) error {
	if err := t.validate(); err != nil {
		return err
	}

	if err := t.prepareSchema(); err != nil {
		return err
	}

	t.assignShardsNum()

	if err := t.assignCollectionId(); err != nil {
		return err
	}

	if err := t.assignPartitionId(); err != nil {
		return err
	}

	if err := t.assignChannels(); err != nil {
		return err
	}

	return t.prepareStep()
}
