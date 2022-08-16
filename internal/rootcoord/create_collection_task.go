package rootcoord

import (
	"context"
	"errors"
	"fmt"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"

	"github.com/golang/protobuf/proto"

	"github.com/milvus-io/milvus/internal/proto/schemapb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type collectionChannels struct {
	virtualChannels  []string
	physicalChannels []string
	deltaChannels    []string
}

type createCollectionTask struct {
	baseTaskV2
	Req      *milvuspb.CreateCollectionRequest
	schema   *schemapb.CollectionSchema
	collID   UniqueID
	partID   UniqueID
	channels collectionChannels

	addDmlChannel   bool
	addDeltaChannel bool

	success bool
}

func (t *createCollectionTask) validate() error {
	if t.Req == nil {
		return errors.New("empty requests")
	}
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_CreateCollection); err != nil {
		return err
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
		deltaChanName, err := funcutil.ConvertChannelName(chanNames[i], Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta)
		if err != nil || deltaChanName != deltaChanNames[i] {
			errMsg := ""
			if err != nil {
				errMsg = err.Error()
			}
			log.Debug("dmlChanName deltaChanName mismatch detail", zap.Int32("i", i),
				zap.String("vchanName", vchanNames[i]),
				zap.String("phsicalChanName", chanNames[i]),
				zap.String("deltaChanName", deltaChanNames[i]),
				zap.String("converted_deltaChanName", deltaChanName),
				zap.String("err", errMsg))
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

	return nil
}

func (t *createCollectionTask) Execute(ctx context.Context) error {
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
		State:                pb.CollectionState_CollectionCreating,
		Partitions: []*model.Partition{
			{
				PartitionID:               partID,
				PartitionName:             Params.CommonCfg.DefaultPartitionName,
				PartitionCreatedTimestamp: ts,
				CollectionID:              collID,
				State:                     pb.PartitionState_PartitionCreated,
			},
		},
	}

	clonedCollInfoWithDefaultPartition := collInfo.Clone()
	clonedCollInfoWithDefaultPartition.Partitions = []*model.Partition{{PartitionName: Params.CommonCfg.DefaultPartitionName}}
	// need double check in meta table if we can't promise the sequence execution.
	existedCollInfo, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), typeutil.MaxTimestamp)
	if err == nil {
		equal := existedCollInfo.Equal(*clonedCollInfoWithDefaultPartition)
		if !equal {
			fmt.Println(model.MarshalCollectionModel(existedCollInfo))
			fmt.Println(model.MarshalCollectionModel(clonedCollInfoWithDefaultPartition))
			return fmt.Errorf("create collection with different parameters, collection: %s", t.Req.GetCollectionName())
		}
		// make creating collection idempotent.
		log.Warn("add duplicate collection", zap.String("collection", t.Req.GetCollectionName()))
		t.success = true
		return nil
	}

	// do not change the order of the following operations.
	// add channels must be before add collection.
	t.core.chanTimeTick.addDmlChannels(chanNames...)

	deltaChanNames := make([]string, len(chanNames))
	for i, chanName := range chanNames {
		if deltaChanNames[i], err = funcutil.ConvertChannelName(chanName, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta); err != nil {
			return err
		}
	}
	t.core.chanTimeTick.addDeltaChannels(deltaChanNames...)

	err = t.core.meta.AddCollection(ctx, &collInfo)
	if err != nil {
		return err
	}

	err = t.core.watchChannels(ctx, collID, vchanNames)
	if err != nil {
		return err
	}

	err = t.core.meta.ChangeCollectionState(ctx, collID, pb.CollectionState_CollectionCreated, ts)
	if err != nil {
		return err
	}
	t.success = true
	return nil
}

func (t *createCollectionTask) PostExecute(ctx context.Context) error {
	if t.success {
		return nil
	}

	//vchanNames := t.channels.virtualChannels
	chanNames := t.channels.physicalChannels

	if err := t.core.meta.ChangeCollectionState(ctx, t.collID, pb.CollectionState_CollectionDropping, t.ts); err != nil {
		return err
	}

	if err := t.core.meta.ChangePartitionState(ctx, t.collID, t.partID, pb.PartitionState_PartitionDropping, t.ts); err != nil {
		return err
	}

	// TODO: unwatch channels

	t.core.chanTimeTick.removeDmlChannels(chanNames...)

	// remove delta channels
	var err error
	deltaChanNames := make([]string, len(chanNames))
	for i, chanName := range chanNames {
		if deltaChanNames[i], err = funcutil.ConvertChannelName(chanName, Params.CommonCfg.RootCoordDml, Params.CommonCfg.RootCoordDelta); err != nil {
			return err
		}
	}
	t.core.chanTimeTick.removeDeltaChannels(deltaChanNames...)
	return nil

	if err := t.core.meta.RemoveCollection(ctx, t.collID, t.ts); err != nil {
		return err
	}
	return nil
}
