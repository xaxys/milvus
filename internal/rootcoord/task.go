// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rootcoord

import (
	"context"
	"fmt"
	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	model "github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"go.uber.org/zap"
)

type reqTask interface {
	Ctx() context.Context
	Type() commonpb.MsgType
	Prepare(ctx context.Context) (bool, error) // if we need to execute; if failed in preparation
	Execute(ctx context.Context) error
	Core() *Core
	WaitToFinish() error
	NotifyDone(err error)
}

type baseReqTask struct {
	ctx  context.Context
	core *Core
	done chan error
}

func newBaseReqTask(ctx context.Context, core *Core) baseReqTask {
	return baseReqTask{
		ctx:  ctx,
		core: core,
		done: make(chan error, 1),
	}
}

func (b *baseReqTask) Core() *Core {
	return b.core
}

func (b *baseReqTask) Ctx() context.Context {
	return b.ctx
}

func (b *baseReqTask) NotifyDone(err error) {
	b.done <- err
}

func (b *baseReqTask) WaitToFinish() error {
	select {
	case err := <-b.done:
		return err
	case <-b.ctx.Done():
		return fmt.Errorf("task context done")
	}
}

type baseRedoTask struct {
	baseReqTask
	logs          stepLogger // logs of steps
	syncTodoStep  []Step     // steps to execute synchronously
	asyncTodoStep []Step     // steps to execute asynchronously
}

func (b *baseRedoTask) AddSyncStep(step Step) {
	b.syncTodoStep = append(b.syncTodoStep, step)
}

func (b *baseRedoTask) AddAsyncStep(step Step) {
	b.asyncTodoStep = append(b.asyncTodoStep, step)
}

func (b *baseRedoTask) Execute(ctx context.Context) error {
	for i := len(b.asyncTodoStep) - 1; i >= 0; i-- {
		step := b.asyncTodoStep[i]
		b.logs.AddStep(step)
	}
	for i := len(b.syncTodoStep) - 1; i >= 0; i-- {
		step := b.syncTodoStep[i]
		b.logs.AddStep(step)
	}
	for _, step := range b.syncTodoStep {
		err := step.Execute(ctx)
		if err != nil {
			log.Warn("execute task failed", zap.Error(err))
			b.logs.Clear()
			return err
		}
		b.logs.PopStep()
	}
	go b.logs.Rollback(ctx)
	return nil
}

type baseUndoTask struct {
	baseReqTask
	logs     stepLogger // logs of steps
	todoStep []Step     // steps to execute
	undoStep []Step     // steps to undo
}

func (b *baseUndoTask) AddStep(todoStep, undoStep Step) {
	b.todoStep = append(b.todoStep, todoStep)
	b.undoStep = append(b.undoStep, undoStep)
}

func (b *baseUndoTask) Execute(ctx context.Context) error {
	if len(b.todoStep) != len(b.undoStep) {
		return fmt.Errorf("todo step and undo step length not equal")
	}
	for i := 0; i < len(b.todoStep); i++ {
		todoStep := b.todoStep[i]
		undoStep := b.undoStep[i]
		b.logs.AddStep(undoStep)
		err := todoStep.Execute(ctx)
		if err != nil {
			log.Warn("execute task failed", zap.Error(err))
			b.logs.Rollback(ctx)
			return err
		}
	}
	b.logs.Clear()
	return nil
}

// CreateCollectionReqTask create collection request task
type CreateCollectionReqTask struct {
	baseUndoTask
	Req *milvuspb.CreateCollectionRequest
}

// Type return msg type
func (t *CreateCollectionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func hasSystemFields(schema *schemapb.CollectionSchema, systemFields []string) bool {
	for _, f := range schema.GetFields() {
		if funcutil.SliceContain(systemFields, f.GetName()) {
			return true
		}
	}
	return false
}

// Prepare task execution
func (t *CreateCollectionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_CreateCollection {
		return false, fmt.Errorf("create collection, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}

	t.logs.info.Type = t.Type()
	t.logs.info.RequestId = t.Req.Base.MsgID
	var schema schemapb.CollectionSchema
	err := proto.Unmarshal(t.Req.Schema, &schema)
	if err != nil {
		return false, fmt.Errorf("unmarshal schema error= %w", err)
	}

	if t.Req.CollectionName != schema.Name {
		return false, fmt.Errorf("collection name = %s, schema.Name=%s", t.Req.CollectionName, schema.Name)
	}
	if t.Req.ShardsNum <= 0 {
		t.Req.ShardsNum = common.DefaultShardsNum
	}
	log.Debug("CreateCollectionReqTask Execute", zap.Any("CollectionName", t.Req.CollectionName),
		zap.Int32("ShardsNum", t.Req.ShardsNum),
		zap.String("ConsistencyLevel", t.Req.ConsistencyLevel.String()))

	if hasSystemFields(&schema, []string{RowIDFieldName, TimeStampFieldName}) {
		log.Error("failed to create collection, user schema contain system field")
		return false, fmt.Errorf("schema contains system field: %s, %s", RowIDFieldName, TimeStampFieldName)
	}

	for idx, field := range schema.Fields {
		field.FieldID = int64(idx + StartOfUserFieldID)
	}
	rowIDField := &schemapb.FieldSchema{
		FieldID:      int64(RowIDField),
		Name:         RowIDFieldName,
		IsPrimaryKey: false,
		Description:  "row id",
		DataType:     schemapb.DataType_Int64,
	}
	timeStampField := &schemapb.FieldSchema{
		FieldID:      int64(TimeStampField),
		Name:         TimeStampFieldName,
		IsPrimaryKey: false,
		Description:  "time stamp",
		DataType:     schemapb.DataType_Int64,
	}
	schema.Fields = append(schema.Fields, rowIDField, timeStampField)

	collID, _, err := t.core.IDAllocator(1)
	if err != nil {
		return false, fmt.Errorf("alloc collection id error = %w", err)
	}
	partID, _, err := t.core.IDAllocator(1)
	if err != nil {
		return false, fmt.Errorf("alloc partition id error = %w", err)
	}

	log.Debug("collection name -> id",
		zap.String("collection name", t.Req.CollectionName),
		zap.Int64("collection_id", collID),
		zap.Int64("default partition id", partID))

	vchanNames := make([]string, t.Req.ShardsNum)
	chanNames := make([]string, t.Req.ShardsNum)
	deltaChanNames := make([]string, t.Req.ShardsNum)
	for i := int32(0); i < t.Req.ShardsNum; i++ {
		vchanNames[i] = fmt.Sprintf("%s_%dv%d", t.core.chanTimeTick.getDmlChannelName(), collID, i)
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
			return false, fmt.Errorf("dmlChanName %s and deltaChanName %s mis-match", chanNames[i], deltaChanNames[i])
		}
	}

	// schema is modified (add RowIDField and TimestampField),
	// so need Marshal again
	schemaBytes, err := proto.Marshal(&schema)
	if err != nil {
		return false, fmt.Errorf("marshal schema error = %w", err)
	}

	ddCollReq := internalpb.CreateCollectionRequest{
		Base:                 t.Req.Base,
		DbName:               t.Req.DbName,
		CollectionName:       t.Req.CollectionName,
		PartitionName:        Params.CommonCfg.DefaultPartitionName,
		DbID:                 0, //TODO,not used
		CollectionID:         collID,
		PartitionID:          partID,
		Schema:               schemaBytes,
		VirtualChannelNames:  vchanNames,
		PhysicalChannelNames: chanNames,
	}

	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return false, fmt.Errorf("tso alloc fail, error = %w", err)
	}

	// build DdOperation and save it into etcd, when ddmsg send fail,
	// system can restore ddmsg from etcd and re-send
	ddCollReq.Base.Timestamp = ts

	collInfo := model.Collection{
		CollectionID:         collID,
		Name:                 schema.Name,
		Description:          schema.Description,
		AutoID:               schema.AutoID,
		Fields:               model.UnmarshalFieldModels(schema.Fields),
		VirtualChannelNames:  vchanNames,
		PhysicalChannelNames: chanNames,
		ShardsNum:            t.Req.ShardsNum,
		ConsistencyLevel:     t.Req.ConsistencyLevel,
		//FieldIDToIndexID:     make([]common.Int64Tuple, 0, 16),
		CreateTime: ts,
		Partitions: []*model.Partition{
			{
				PartitionID:               partID,
				PartitionName:             Params.CommonCfg.DefaultPartitionName,
				PartitionCreatedTimestamp: ts,
				Available:                 true,
			},
		},
	}

	curColl, _ := t.core.MetaTable.GetCollectionByName(t.Req.GetCollectionName(), 0)
	if curColl != nil {
		if curColl.Equal(&collInfo) {
			return false, nil
		}
		return false, fmt.Errorf("collection %s with different params exist", curColl.Name)
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

	t.AddStep(&EnableCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		EnableCollectionMetaStep: rootcoordpb.EnableCollectionMetaStep{
			CollectionId: collID,
		},
	}, &DisableCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DisableCollectionMetaStep: rootcoordpb.DisableCollectionMetaStep{
			CollectionId: collID,
		},
	})

	return true, nil
}

// DropCollectionReqTask drop collection request task
type DropCollectionReqTask struct {
	baseRedoTask
	Req *milvuspb.DropCollectionRequest
}

// Type return msg type
func (t *DropCollectionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

// Prepare task execution
func (t *DropCollectionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_DropCollection {
		return false, fmt.Errorf("drop collection, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	if t.core.MetaTable.IsAlias(t.Req.CollectionName) {
		return false, fmt.Errorf("cannot drop the collection via alias = %s", t.Req.CollectionName)
	}
	t.logs.info.Type = t.Type()
	t.logs.info.RequestId = t.Req.Base.MsgID

	collMeta, err := t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, 0)
	if collMeta == nil {
		return false, nil
	}

	ddReq := internalpb.DropCollectionRequest{
		Base:           t.Req.Base,
		DbName:         t.Req.DbName,
		CollectionName: t.Req.CollectionName,
		DbID:           0, //not used
		CollectionID:   collMeta.CollectionID,
	}

	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return false, fmt.Errorf("TSO alloc fail, error = %w", err)
	}

	//indexIDs := make([]int64, 0, len(collMeta.FieldIDToIndexID))
	//for _, tuple := range collMeta.FieldIDToIndexID {
	//	indexIDs = append(indexIDs, tuple.Value)
	//}

	// Allocate a new ts to make sure the channel timetick is consistent.
	ts, err = t.core.TSOAllocator(1)
	if err != nil {
		return false, fmt.Errorf("TSO alloc fail, error = %w", err)
	}

	// build DdOperation and save it into etcd, when ddmsg send fail,
	// system can restore ddmsg from etcd and re-send
	ddReq.Base.Timestamp = ts

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

	t.AddAsyncStep(&DeleteCollectionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeleteCollectionMetaStep: rootcoordpb.DeleteCollectionMetaStep{
			CollectionId: collMeta.CollectionID,
			Timestamp:    ts,
		},
	})

	return true, nil
}

// HasCollectionReqTask has collection request task
type HasCollectionReqTask struct {
	baseReqTask
	Req           *milvuspb.HasCollectionRequest
	HasCollection bool
}

// Type return msg type
func (t *HasCollectionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *HasCollectionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_HasCollection {
		return false, fmt.Errorf("has collection, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *HasCollectionReqTask) Execute(ctx context.Context) error {
	_, err := t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, t.Req.TimeStamp)
	if err == nil {
		t.HasCollection = true
	} else {
		t.HasCollection = false
	}
	return nil
}

// DescribeCollectionReqTask describe collection request task
type DescribeCollectionReqTask struct {
	baseReqTask
	Req *milvuspb.DescribeCollectionRequest
	Rsp *milvuspb.DescribeCollectionResponse
}

// Type return msg type
func (t *DescribeCollectionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *DescribeCollectionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_DescribeCollection {
		return false, fmt.Errorf("describe collection, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *DescribeCollectionReqTask) Execute(ctx context.Context) error {
	var collInfo *model.Collection
	var err error

	if t.Req.CollectionName != "" {
		collInfo, err = t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, t.Req.TimeStamp)
		if err != nil {
			return err
		}
	} else {
		collInfo, err = t.core.MetaTable.GetCollectionByID(t.Req.CollectionID, t.Req.TimeStamp)
		if err != nil {
			return err
		}
	}

	t.Rsp.Schema = &schemapb.CollectionSchema{
		Name:        collInfo.Name,
		Description: collInfo.Description,
		AutoID:      collInfo.AutoID,
		Fields:      model.MarshalFieldModels(collInfo.Fields),
	}
	t.Rsp.CollectionID = collInfo.CollectionID
	t.Rsp.VirtualChannelNames = collInfo.VirtualChannelNames
	t.Rsp.PhysicalChannelNames = collInfo.PhysicalChannelNames
	if collInfo.ShardsNum == 0 {
		collInfo.ShardsNum = int32(len(collInfo.VirtualChannelNames))
	}
	t.Rsp.ShardsNum = collInfo.ShardsNum
	t.Rsp.ConsistencyLevel = collInfo.ConsistencyLevel

	t.Rsp.CreatedTimestamp = collInfo.CreateTime
	createdPhysicalTime, _ := tsoutil.ParseHybridTs(collInfo.CreateTime)
	t.Rsp.CreatedUtcTimestamp = uint64(createdPhysicalTime)
	t.Rsp.Aliases = t.core.MetaTable.ListAliases(collInfo.CollectionID)
	t.Rsp.StartPositions = collInfo.StartPositions
	t.Rsp.CollectionName = t.Rsp.Schema.Name
	return nil
}

// ShowCollectionReqTask show collection request task
type ShowCollectionReqTask struct {
	baseReqTask
	Req *milvuspb.ShowCollectionsRequest
	Rsp *milvuspb.ShowCollectionsResponse
}

// Type return msg type
func (t *ShowCollectionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *ShowCollectionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_ShowCollections {
		return false, fmt.Errorf("show collection, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *ShowCollectionReqTask) Execute(ctx context.Context) error {
	coll, err := t.core.MetaTable.ListCollections(t.Req.TimeStamp)
	if err != nil {
		return err
	}
	for name, meta := range coll {
		t.Rsp.CollectionNames = append(t.Rsp.CollectionNames, name)
		t.Rsp.CollectionIds = append(t.Rsp.CollectionIds, meta.CollectionID)
		t.Rsp.CreatedTimestamps = append(t.Rsp.CreatedTimestamps, meta.CreateTime)
		physical, _ := tsoutil.ParseHybridTs(meta.CreateTime)
		t.Rsp.CreatedUtcTimestamps = append(t.Rsp.CreatedUtcTimestamps, uint64(physical))
	}
	return nil
}

// CreatePartitionReqTask create partition request task
type CreatePartitionReqTask struct {
	baseUndoTask
	Req *milvuspb.CreatePartitionRequest
}

// Type return msg type
func (t *CreatePartitionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

// Prepare task execution
func (t *CreatePartitionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_CreatePartition {
		return false, fmt.Errorf("create partition, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	t.logs.info.Type = t.Type()
	t.logs.info.RequestId = t.Req.Base.MsgID
	collMeta, err := t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, 0)
	if err != nil {
		return false, err
	}
	partID, _, err := t.core.IDAllocator(1)
	if err != nil {
		return false, err
	}

	curPartID, _ := t.core.MetaTable.GetPartitionByName(collMeta.CollectionID, t.Req.PartitionName, 0)
	if curPartID != 0 {
		return false, nil
	}

	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return false, fmt.Errorf("TSO alloc fail, error = %w", err)
	}

	t.AddStep(&AddPartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		AddPartitionMetaStep: rootcoordpb.AddPartitionMetaStep{
			DbId:         0, // todo, not used
			CollectionId: collMeta.CollectionID,
			PartInfo: &etcdpb.PartitionInfo{
				PartitionID:               partID,
				PartitionName:             t.Req.PartitionName,
				PartitionCreatedTimestamp: ts,
			},
			Timestamp: ts,
		},
	}, &DeletePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{
			DbId:         0, // todo, not used
			CollectionId: collMeta.CollectionID,
			PartitionId:  partID,
			Timestamp:    ts,
		},
	})

	t.AddStep(&EnablePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		EnablePartitionMetaStep: rootcoordpb.EnablePartitionMetaStep{
			DbId:         0, // todo, not used
			CollectionId: collMeta.CollectionID,
			PartitionId:  partID,
		},
	}, &DisablePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{
			DbId:         0, // todo, not used
			CollectionId: collMeta.CollectionID,
			PartitionId:  partID,
		},
	})

	t.AddStep(&ExpireCollectionCacheStep{
		baseStep: baseStep{
			core: t.core,
		},
		ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{
			DbId:         0, // todo, not used
			CollectionId: collMeta.CollectionID,
			Timestamp:    ts,
		},
	}, &NullStep{})

	return true, nil
}

// DropPartitionReqTask drop partition request task
type DropPartitionReqTask struct {
	baseRedoTask
	Req *milvuspb.DropPartitionRequest
}

// Type return msg type
func (t *DropPartitionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

// Prepare task execution
func (t *DropPartitionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_DropPartition {
		return false, fmt.Errorf("drop partition, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	t.logs.info.Type = t.Type()
	t.logs.info.RequestId = t.Req.Base.MsgID
	if t.Req.PartitionName == Params.CommonCfg.DefaultPartitionName {
		return false, fmt.Errorf("default partition cannot be deleted")
	}
	collInfo, err := t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, 0)
	if err != nil {
		return false, err
	}
	partID, err := t.core.MetaTable.GetPartitionByName(collInfo.CollectionID, t.Req.PartitionName, 0)
	if err != nil {
		return false, nil
	}

	ddReq := internalpb.DropPartitionRequest{
		Base:           t.Req.Base,
		DbName:         t.Req.DbName,
		CollectionName: t.Req.CollectionName,
		PartitionName:  t.Req.PartitionName,
		DbID:           0, // todo,not used
		CollectionID:   collInfo.CollectionID,
		PartitionID:    partID,
	}

	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return false, fmt.Errorf("TSO alloc fail, error = %w", err)
	}

	ddReq.Base.Timestamp = ts

	t.AddSyncStep(&ExpireCollectionCacheStep{
		baseStep: baseStep{
			core: t.core,
		},
		ExpireCollectionCacheStep: rootcoordpb.ExpireCollectionCacheStep{
			DbId:         0, // todo, not used
			CollectionId: collInfo.CollectionID,
			Timestamp:    ts,
		},
	})

	t.AddSyncStep(&DisablePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DisablePartitionMetaStep: rootcoordpb.DisablePartitionMetaStep{
			DbId:         0, // todo, not used
			CollectionId: collInfo.CollectionID,
			PartitionId:  partID,
		},
	})

	t.AddAsyncStep(&ReleasePartitionStep{
		baseStep: baseStep{
			core: t.core,
		},
		ReleasePartitionStep: rootcoordpb.ReleasePartitionStep{
			DbId:         0, // todo, not used
			CollectionId: collInfo.CollectionID,
			PartitionId:  partID,
			Timestamp:    ts,
		},
	})

	t.AddAsyncStep(&DeletePartitionDataStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeletePartitionDataStep: rootcoordpb.DeletePartitionDataStep{
			DbId:         0, // todo, not used
			CollectionId: collInfo.CollectionID,
			PartitionId:  partID,
			Request:      &ddReq,
			Pchannels:    collInfo.PhysicalChannelNames,
		},
	})

	t.AddAsyncStep(&DeletePartitionMetaStep{
		baseStep: baseStep{
			core: t.core,
		},
		DeletePartitionMetaStep: rootcoordpb.DeletePartitionMetaStep{
			DbId:         0, // todo, not used
			CollectionId: collInfo.CollectionID,
			PartitionId:  partID,
			Timestamp:    ts,
		},
	})

	return true, nil
}

// HasPartitionReqTask has partition request task
type HasPartitionReqTask struct {
	baseReqTask
	Req          *milvuspb.HasPartitionRequest
	HasPartition bool
}

// Type return msg type
func (t *HasPartitionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *HasPartitionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_HasPartition {
		return false, fmt.Errorf("has partition, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *HasPartitionReqTask) Execute(ctx context.Context) error {
	if t.Type() != commonpb.MsgType_HasPartition {
		return fmt.Errorf("has partition, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	coll, err := t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, 0)
	if err != nil {
		return err
	}
	t.HasPartition = t.core.MetaTable.HasPartition(coll.CollectionID, t.Req.PartitionName, 0)
	return nil
}

// ShowPartitionReqTask show partition request task
type ShowPartitionReqTask struct {
	baseReqTask
	Req *milvuspb.ShowPartitionsRequest
	Rsp *milvuspb.ShowPartitionsResponse
}

// Type return msg type
func (t *ShowPartitionReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *ShowPartitionReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_ShowPartitions {
		return false, fmt.Errorf("show partition, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *ShowPartitionReqTask) Execute(ctx context.Context) error {
	var coll *model.Collection
	var err error
	if t.Req.CollectionName == "" {
		coll, err = t.core.MetaTable.GetCollectionByID(t.Req.CollectionID, 0)
	} else {
		coll, err = t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, 0)
	}
	if err != nil {
		return err
	}

	for _, part := range coll.Partitions {
		t.Rsp.PartitionIDs = append(t.Rsp.PartitionIDs, part.PartitionID)
		t.Rsp.PartitionNames = append(t.Rsp.PartitionNames, part.PartitionName)
		t.Rsp.CreatedTimestamps = append(t.Rsp.CreatedTimestamps, part.PartitionCreatedTimestamp)

		physical, _ := tsoutil.ParseHybridTs(part.PartitionCreatedTimestamp)
		t.Rsp.CreatedUtcTimestamps = append(t.Rsp.CreatedUtcTimestamps, uint64(physical))
	}

	return nil
}

//// DescribeSegmentReqTask describe segment request task
//type DescribeSegmentReqTask struct {
//	baseReqTask
//	Req *milvuspb.DescribeSegmentRequest
//	Rsp *milvuspb.DescribeSegmentResponse //TODO,return repeated segment id in the future
//}
//
//// Type return msg type
//func (t *DescribeSegmentReqTask) Type() commonpb.MsgType {
//	return t.Req.Base.MsgType
//}
//
//func (t *DescribeSegmentReqTask) Prepare(ctx context.Context) (bool, error) {
//	if t.Type() != commonpb.MsgType_DescribeSegment {
//		return fmt.Errorf("describe segment, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
//	}
//	return nil
//}
//
//// Execute task execution
//func (t *DescribeSegmentReqTask) Execute(ctx context.Context) error {
//	coll, err := t.core.MetaTable.GetCollectionByID(t.Req.CollectionID, 0)
//	if err != nil {
//		return err
//	}
//
//	segIDs, err := t.core.CallGetFlushedSegmentsService(ctx, t.Req.CollectionID, -1)
//	if err != nil {
//		log.Debug("Get flushed segment from data coord failed", zap.String("collection_name", coll.Name), zap.Error(err))
//		return err
//	}
//
//	// check if segment id exists
//	exist := false
//	for _, id := range segIDs {
//		if id == t.Req.SegmentID {
//			exist = true
//			break
//		}
//	}
//	if !exist {
//		return fmt.Errorf("segment id %d not belong to collection id %d", t.Req.SegmentID, t.Req.CollectionID)
//	}
//	//TODO, get filed_id and index_name from request
//	index, err := t.core.MetaTable.GetSegmentIndexInfoByID(t.Req.SegmentID, -1, "")
//	log.Debug("RootCoord DescribeSegmentReqTask, MetaTable.GetSegmentIndexInfoByID", zap.Any("SegmentID", t.Req.SegmentID),
//		zap.Any("index", index), zap.Error(err))
//	if err != nil {
//		return err
//	}
//	t.Rsp.IndexID = index.IndexID
//	t.Rsp.BuildID = index.SegmentIndexes[t.Req.SegmentID].BuildID
//	t.Rsp.EnableIndex = index.SegmentIndexes[t.Req.SegmentID].EnableIndex
//	t.Rsp.FieldID = index.FieldID
//	return nil
//}
//
//// ShowSegmentReqTask show segment request task
//type ShowSegmentReqTask struct {
//	baseReqTask
//	Req *milvuspb.ShowSegmentsRequest
//	Rsp *milvuspb.ShowSegmentsResponse
//}
//
//// Type return msg type
//func (t *ShowSegmentReqTask) Type() commonpb.MsgType {
//	return t.Req.Base.MsgType
//}
//
//func (t *ShowSegmentReqTask) Prepare(ctx context.Context) (bool, error) {
//	if t.Type() != commonpb.MsgType_ShowSegments {
//		return fmt.Errorf("show segment, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
//	}
//	return nil
//}
//
//// Execute task execution
//func (t *ShowSegmentReqTask) Execute(ctx context.Context) error {
//	coll, err := t.core.MetaTable.GetCollectionByID(t.Req.CollectionID, 0)
//	if err != nil {
//		return err
//	}
//	exist := false
//	for _, partition := range coll.Partitions {
//		if partition.PartitionID == t.Req.PartitionID {
//			exist = true
//			break
//		}
//	}
//	if !exist {
//		return fmt.Errorf("partition id = %d not belong to collection id = %d", t.Req.PartitionID, t.Req.CollectionID)
//	}
//	segIDs, err := t.core.CallGetFlushedSegmentsService(ctx, t.Req.CollectionID, t.Req.PartitionID)
//	if err != nil {
//		log.Debug("Get flushed segments from data coord failed", zap.String("collection name", coll.Name), zap.Int64("partition id", t.Req.PartitionID), zap.Error(err))
//		return err
//	}
//
//	t.Rsp.SegmentIDs = append(t.Rsp.SegmentIDs, segIDs...)
//	return nil
//}
//
//type DescribeSegmentsReqTask struct {
//	baseReqTask
//	Req *rootcoordpb.DescribeSegmentsRequest
//	Rsp *rootcoordpb.DescribeSegmentsResponse
//}
//
//func (t *DescribeSegmentsReqTask) Type() commonpb.MsgType {
//	return t.Req.GetBase().GetMsgType()
//}
//
//func (t *DescribeSegmentsReqTask) Prepare(ctx context.Context) (bool, error) {
//	if t.Type() != commonpb.MsgType_DescribeSegments {
//		return fmt.Errorf("describe segments, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
//	}
//	return nil
//}
//
//func (t *DescribeSegmentsReqTask) Execute(ctx context.Context) error {
//	collectionID := t.Req.GetCollectionID()
//	segIDs, err := t.core.CallGetFlushedSegmentsService(ctx, collectionID, -1)
//	if err != nil {
//		log.Error("failed to get flushed segments",
//			zap.Error(err),
//			zap.Int64("collection", collectionID))
//		return err
//	}
//
//	t.Rsp.CollectionID = collectionID
//	t.Rsp.SegmentInfos = make(map[typeutil.UniqueID]*rootcoordpb.SegmentInfos)
//
//	segIDsMap := make(map[typeutil.UniqueID]struct{})
//	for _, segID := range segIDs {
//		segIDsMap[segID] = struct{}{}
//	}
//
//	for _, segID := range t.Req.SegmentIDs {
//		if _, ok := segIDsMap[segID]; !ok {
//			log.Warn("requested segment not found",
//				zap.Int64("collection", collectionID),
//				zap.Int64("segment", segID))
//			return fmt.Errorf("segment not found, collection: %d, segment: %d",
//				collectionID, segID)
//		}
//
//		if _, ok := t.Rsp.SegmentInfos[segID]; !ok {
//			t.Rsp.SegmentInfos[segID] = &rootcoordpb.SegmentInfos{
//				BaseInfo: &rootcoordpb.SegmentBaseInfo{
//					CollectionID: collectionID,
//					PartitionID:  0, // TODO: change this after MetaTable.partID2IndexedSegID been fixed.
//					SegmentID:    segID,
//				},
//				IndexInfos:      nil,
//				ExtraIndexInfos: make(map[typeutil.UniqueID]*etcdpb.IndexInfo),
//			}
//		}
//
//		index, err := t.core.MetaTable.GetSegmentIndexInfos(segID)
//		if err != nil {
//			continue
//		}
//
//		segIdxMeta, ok := index.SegmentIndexes[segID]
//		if !ok {
//			log.Error("requested segment index not found",
//				zap.Int64("collection", collectionID),
//				zap.Int64("indexID", index.IndexID),
//				zap.Int64("segment", segID))
//			return fmt.Errorf("segment index not found, collection: %d, segment: %d", collectionID, segID)
//		}
//
//		t.Rsp.SegmentInfos[segID].IndexInfos = append(
//			t.Rsp.SegmentInfos[segID].IndexInfos,
//			&etcdpb.SegmentIndexInfo{
//				CollectionID: index.CollectionID,
//				PartitionID:  segIdxMeta.Segment.PartitionID,
//				SegmentID:    segIdxMeta.Segment.SegmentID,
//				FieldID:      index.FieldID,
//				IndexID:      index.IndexID,
//				BuildID:      segIdxMeta.BuildID,
//				EnableIndex:  segIdxMeta.EnableIndex,
//			})
//
//		t.Rsp.SegmentInfos[segID].ExtraIndexInfos[index.IndexID] = model.MarshalIndexModel(&index)
//	}
//
//	return nil
//}
//
//// CreateIndexReqTask create index request task
//type CreateIndexReqTask struct {
//	baseUndoTask
//	Req *milvuspb.CreateIndexRequest
//}
//
//// Type return msg type
//func (t *CreateIndexReqTask) Type() commonpb.MsgType {
//	return t.Req.Base.MsgType
//}
//
//// Prepare task execution
//func (t *CreateIndexReqTask) Prepare(ctx context.Context) (bool, error) {
//	if t.Type() != commonpb.MsgType_CreateIndex {
//		return fmt.Errorf("create index, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
//	}
//	indexName := t.Req.GetIndexName()
//	if len(indexName) <= 0 {
//		indexName = Params.CommonCfg.DefaultIndexName //TODO, get name from request
//	}
//	indexID, _, err := t.core.IDAllocator(1)
//	log.Debug("RootCoord CreateIndexReqTask", zap.Any("indexID", indexID), zap.Error(err))
//	if err != nil {
//		return err
//	}
//
//	createTS, err := t.core.TSOAllocator(1)
//	if err != nil {
//		return err
//	}
//
//	idxInfo := &model.Index{
//		IndexName:   indexName,
//		IndexID:     indexID,
//		IndexParams: t.Req.ExtraParams,
//		CreateTime:  createTS,
//	}
//	log.Info("create index for collection",
//		zap.String("collection", t.Req.GetCollectionName()),
//		zap.String("field", t.Req.GetFieldName()),
//		zap.String("index", indexName),
//		zap.Int64("index_id", indexID),
//		zap.Any("params", t.Req.GetExtraParams()))
//	collMeta, err := t.core.MetaTable.GetCollectionByName(t.Req.CollectionName, 0)
//	if err != nil {
//		return err
//	}
//	segID2PartID, segID2Binlog, err := t.core.getSegments(ctx, collMeta.CollectionID)
//	flushedSegs := make([]typeutil.UniqueID, 0, len(segID2PartID))
//	for k := range segID2PartID {
//		flushedSegs = append(flushedSegs, k)
//	}
//	if err != nil {
//		log.Debug("get flushed segments from data coord failed", zap.String("collection_name", collMeta.Name), zap.Error(err))
//		return err
//	}
//
//	alreadyExists, err := t.core.MetaTable.AddIndex(t.Req.CollectionName, t.Req.FieldName, idxInfo, flushedSegs)
//	if err != nil {
//		log.Debug("add index into metastore failed", zap.Int64("collection_id", collMeta.CollectionID), zap.Int64("index_id", idxInfo.IndexID), zap.Error(err))
//		return err
//	}
//	// backward compatible with support create the same index
//	if alreadyExists {
//		return nil
//	}
//
//	segIDs, field, err := t.core.MetaTable.GetNotIndexedSegments(t.Req.CollectionName, t.Req.FieldName, idxInfo, flushedSegs)
//	if err != nil {
//		log.Debug("get not indexed segments failed", zap.Int64("collection_id", collMeta.CollectionID), zap.Error(err))
//		return err
//	}
//
//	for _, segID := range segIDs {
//		segmentIndex := model.SegmentIndex{
//			Segment: model.Segment{
//				SegmentID:   segID,
//				PartitionID: segID2PartID[segID],
//			},
//			EnableIndex: false,
//			CreateTime:  createTS,
//		}
//
//		segmentIndex.BuildID, err = t.core.BuildIndex(ctx, segID, segID2Binlog[segID].GetNumOfRows(), segID2Binlog[segID].GetFieldBinlogs(), &field, idxInfo, false)
//		if err != nil {
//			return err
//		}
//		if segmentIndex.BuildID != 0 {
//			segmentIndex.EnableIndex = true
//		}
//
//		index := &model.Index{
//			CollectionID:   collMeta.CollectionID,
//			FieldID:        field.FieldID,
//			IndexID:        idxInfo.IndexID,
//			SegmentIndexes: map[int64]model.SegmentIndex{segID: segmentIndex},
//		}
//
//		if err := t.core.MetaTable.AlterIndex(index); err != nil {
//			log.Error("alter index into meta table failed", zap.Int64("collection_id", collMeta.CollectionID), zap.Int64("index_id", index.IndexID), zap.Int64("build_id", segmentIndex.BuildID), zap.Error(err))
//			return err
//		}
//	}
//
//	return nil
//}
//
//// DescribeIndexReqTask describe index request task
//type DescribeIndexReqTask struct {
//	baseReqTask
//	Req *milvuspb.DescribeIndexRequest
//	Rsp *milvuspb.DescribeIndexResponse
//}
//
//// Type return msg type
//func (t *DescribeIndexReqTask) Type() commonpb.MsgType {
//	return t.Req.Base.MsgType
//}
//
//func (t *DescribeIndexReqTask) Prepare(ctx context.Context) (bool, error) {
//	if t.Type() != commonpb.MsgType_DescribeIndex {
//		return fmt.Errorf("describe index, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
//	}
//	return nil
//}
//
//// Execute task execution
//func (t *DescribeIndexReqTask) Execute(ctx context.Context) error {
//	coll, idx, err := t.core.MetaTable.GetIndexByName(t.Req.CollectionName, t.Req.IndexName)
//	if err != nil {
//		return err
//	}
//	for _, i := range idx {
//		f, err := GetFieldSchemaByIndexID(&coll, typeutil.UniqueID(i.IndexID))
//		if err != nil {
//			log.Warn("Get field schema by index id failed", zap.String("collection name", t.Req.CollectionName), zap.String("index name", t.Req.IndexName), zap.Error(err))
//			continue
//		}
//		desc := &milvuspb.IndexDescription{
//			IndexName: i.IndexName,
//			Params:    i.IndexParams,
//			IndexID:   i.IndexID,
//			FieldName: f.Name,
//		}
//		t.Rsp.IndexDescriptions = append(t.Rsp.IndexDescriptions, desc)
//	}
//	return nil
//}
//
//// DropIndexReqTask drop index request task
//type DropIndexReqTask struct {
//	baseReqTask
//	Req *milvuspb.DropIndexRequest
//}
//
//// Type return msg type
//func (t *DropIndexReqTask) Type() commonpb.MsgType {
//	return t.Req.Base.MsgType
//}
//
//func (t *DropIndexReqTask) Prepare(ctx context.Context) (bool, error) {
//	if t.Type() != commonpb.MsgType_DropIndex {
//		return fmt.Errorf("drop index, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
//	}
//	return nil
//}
//
//// Execute task execution
//func (t *DropIndexReqTask) Execute(ctx context.Context) error {
//	if err := t.core.MetaTable.MarkIndexDeleted(t.Req.CollectionName, t.Req.FieldName, t.Req.IndexName); err != nil {
//		return err
//	}
//	return nil
//}

// CreateAliasReqTask create alias request task
type CreateAliasReqTask struct {
	baseReqTask
	Req *milvuspb.CreateAliasRequest
}

// Type return msg type
func (t *CreateAliasReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *CreateAliasReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_CreateAlias {
		return false, fmt.Errorf("create alias, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *CreateAliasReqTask) Execute(ctx context.Context) error {
	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return fmt.Errorf("TSO alloc fail, error = %w", err)
	}
	err = t.core.MetaTable.AddAlias(t.Req.Alias, t.Req.CollectionName, ts)
	if err != nil {
		return fmt.Errorf("meta table add alias failed, error = %w", err)
	}

	return nil
}

// DropAliasReqTask drop alias request task
type DropAliasReqTask struct {
	baseReqTask
	Req *milvuspb.DropAliasRequest
}

// Type return msg type
func (t *DropAliasReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *DropAliasReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_DropAlias {
		return false, fmt.Errorf("drop alias, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *DropAliasReqTask) Execute(ctx context.Context) error {
	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return fmt.Errorf("TSO alloc fail, error = %w", err)
	}
	err = t.core.MetaTable.DropAlias(t.Req.Alias, ts)
	if err != nil {
		return fmt.Errorf("meta table drop alias failed, error = %w", err)
	}

	return t.core.ExpireMetaCache(ctx, []string{t.Req.Alias}, InvalidCollectionID, ts)
}

// AlterAliasReqTask alter alias request task
type AlterAliasReqTask struct {
	baseReqTask
	Req *milvuspb.AlterAliasRequest
}

// Type return msg type
func (t *AlterAliasReqTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *AlterAliasReqTask) Prepare(ctx context.Context) (bool, error) {
	if t.Type() != commonpb.MsgType_AlterAlias {
		return false, fmt.Errorf("alter alias, msg type = %s", commonpb.MsgType_name[int32(t.Type())])
	}
	return true, nil
}

// Execute task execution
func (t *AlterAliasReqTask) Execute(ctx context.Context) error {
	ts, err := t.core.TSOAllocator(1)
	if err != nil {
		return fmt.Errorf("TSO alloc fail, error = %w", err)
	}
	err = t.core.MetaTable.AlterAlias(t.Req.Alias, t.Req.CollectionName, ts)
	if err != nil {
		return fmt.Errorf("meta table alter alias failed, error = %w", err)
	}

	return t.core.ExpireMetaCache(ctx, []string{t.Req.Alias}, InvalidCollectionID, ts)
}
