package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

// hasCollectionTask has collection request task
type hasCollectionTask struct {
	baseTaskV2
	Req *milvuspb.HasCollectionRequest
	Rsp *milvuspb.BoolResponse
}

// Type return msg type
func (t *hasCollectionTask) Type() commonpb.MsgType {
	return t.Req.Base.MsgType
}

func (t *hasCollectionTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.Base.MsgType, commonpb.MsgType_HasCollection); err != nil {
		return err
	}
	return nil
}

// Execute task execution
func (t *hasCollectionTask) Execute(ctx context.Context) error {
	t.Rsp.Status = succStatus()
	_, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetCollectionName(), t.Req.GetTimeStamp())
	t.Rsp.Value = err == nil
	return nil
}
