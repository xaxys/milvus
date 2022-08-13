package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

type createAliasTask struct {
	baseUndoTask
	Req *milvuspb.CreateAliasRequest
}

func (t *createAliasTask) Prepare(ctx context.Context) error {
	return nil
}

func (t *createAliasTask) Execute(ctx context.Context) error {
	// create alias is atomic enough.
	return t.core.meta.CreateAlias(ctx, t.Req.GetAlias(), t.Req.GetCollectionName(), t.GetTs())
}
