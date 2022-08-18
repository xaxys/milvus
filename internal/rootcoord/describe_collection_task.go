package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
)

// describeCollectionTask describe collection request task
type describeCollectionTask struct {
	baseTaskV2
	Req *milvuspb.DescribeCollectionRequest
	Rsp *milvuspb.DescribeCollectionResponse
}

// Type return msg type
func (t *describeCollectionTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *describeCollectionTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.Base.MsgType, commonpb.MsgType_DescribeCollection); err != nil {
		return err
	}
	return nil
}

// Execute task execution
func (t *describeCollectionTask) Execute(ctx context.Context) (err error) {
	var collInfo *model.Collection
	t.Rsp.Status = succStatus()

	if t.Req.GetCollectionName() != "" {
		collInfo, err = t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.Req.GetTimeStamp())
		if err != nil {
			t.Rsp.Status = failStatus(commonpb.ErrorCode_CollectionNotExists, err.Error())
			return nil
		}
	} else {
		collInfo, err = t.core.meta.GetCollectionByID(ctx, t.Req.GetCollectionID(), t.Req.GetTimeStamp())
		if err != nil {
			t.Rsp.Status = failStatus(commonpb.ErrorCode_CollectionNotExists, err.Error())
			return nil
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
	t.Rsp.Aliases = nil // TODO: add aliases
	t.Rsp.StartPositions = collInfo.StartPositions
	t.Rsp.CollectionName = t.Rsp.Schema.Name
	return nil
}
