package rootcoord

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
)

func TestRootCoord_CreateCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withAbnormalCode())
		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to add task", func(t *testing.T) {
		c := newTestCore(withHealthyCode(),
			withInvalidScheduler())

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("failed to execute", func(t *testing.T) {
		sched := newMockScheduler()
		sched.AddTaskFunc = func(t taskV2) error {
			// inject error here, task will accept error.
			t.(*createCollectionTask).NotifyDone(errors.New("error mock Execute"))
			return nil
		}

		c := newTestCore(withHealthyCode(),
			withScheduler(sched))

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.NotEqual(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})

	t.Run("normal case, everything is ok", func(t *testing.T) {
		sched := newMockScheduler()
		sched.AddTaskFunc = func(t taskV2) error {
			// notify done.
			t.(*createCollectionTask).NotifyDone(nil)
			return nil
		}

		c := newTestCore(withHealthyCode(),
			withScheduler(sched))

		ctx := context.Background()
		resp, err := c.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetErrorCode())
	})
}
