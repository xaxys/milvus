package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

// hasPartitionTask has partition request task
type hasPartitionTask struct {
	baseTaskV2
	Req *milvuspb.HasPartitionRequest
	Rsp *milvuspb.BoolResponse
}

// Type return msg type
func (t *hasPartitionTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *hasPartitionTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.Base.MsgType, commonpb.MsgType_HasPartition); err != nil {
		return err
	}
	return nil
}

// Execute task execution
func (t *hasPartitionTask) Execute(ctx context.Context) error {
	t.Rsp.Status = succStatus()
	coll, err := t.core.meta.GetCollectionByName(ctx, t.Req.CollectionName, typeutil.MaxTimestamp)
	if err != nil {
		t.Rsp.Status = failStatus(commonpb.ErrorCode_CollectionNotExists, err.Error())
		return nil
	}
	for _, part := range coll.Partitions {
		if part.PartitionName == t.Req.PartitionName {
			t.Rsp.Value = true
			break
		}
	}
	return nil
}
