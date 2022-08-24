package rootcoord

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/stretchr/testify/assert"
)

const (
	CollID = 1
	TaskID = 1
)

func TestServerBroker_ReleaseCollection(t *testing.T) {
	t.Run("not healthy", func(t *testing.T) {
		c := newTestCore(withUnhealthyQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseCollection(ctx, CollID)
		assert.Error(t, err)
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseCollection(ctx, CollID)
		assert.Error(t, err)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseCollection(ctx, CollID)
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseCollection(ctx, CollID)
		assert.NoError(t, err)
	})
}

func TestServerBroker_GetSegmentInfo(t *testing.T) {
	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		_, err := b.GetQuerySegmentInfo(ctx, CollID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		resp, err := b.GetQuerySegmentInfo(ctx, CollID, []int64{1, 2})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.GetStatus().GetErrorCode())
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidQueryCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		resp, err := b.GetQuerySegmentInfo(ctx, CollID, []int64{1, 2})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestServerBroker_WatchChannels(t *testing.T) {
	t.Run("unhealthy", func(t *testing.T) {
		c := newTestCore(withUnhealthyDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.WatchChannels(ctx, &watchInfo{})
		assert.Error(t, err)
	})

	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidDataCoord(), withRocksMqTtSynchronizer())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.WatchChannels(ctx, &watchInfo{})
		assert.Error(t, err)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedDataCoord(), withRocksMqTtSynchronizer())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.WatchChannels(ctx, &watchInfo{})
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidDataCoord(), withRocksMqTtSynchronizer())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.WatchChannels(ctx, &watchInfo{})
		assert.NoError(t, err)
	})
}

func TestServerBroker_UnwatchChannels(t *testing.T) {
	// TODO: implement
	b := newServerBroker(newTestCore())
	ctx := context.Background()
	b.UnwatchChannels(ctx, &watchInfo{})
}

func TestServerBroker_AddSegRefLock(t *testing.T) {
	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.AddSegRefLock(ctx, TaskID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.AddSegRefLock(ctx, TaskID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.AddSegRefLock(ctx, TaskID, []int64{1, 2})
		assert.NoError(t, err)
	})
}

func TestServerBroker_ReleaseSegRefLock(t *testing.T) {
	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseSegRefLock(ctx, TaskID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseSegRefLock(ctx, TaskID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.ReleaseSegRefLock(ctx, TaskID, []int64{1, 2})
		assert.NoError(t, err)
	})
}

func TestServerBroker_Flush(t *testing.T) {
	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.Flush(ctx, CollID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.Flush(ctx, CollID, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		err := b.Flush(ctx, CollID, []int64{1, 2})
		assert.NoError(t, err)
	})
}

func TestServerBroker_Import(t *testing.T) {
	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		resp, err := b.Import(ctx, &datapb.ImportTaskRequest{})
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		resp, err := b.Import(ctx, &datapb.ImportTaskRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_UnexpectedError, resp.GetStatus().GetErrorCode())
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidDataCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		resp, err := b.Import(ctx, &datapb.ImportTaskRequest{})
		assert.NoError(t, err)
		assert.Equal(t, commonpb.ErrorCode_Success, resp.GetStatus().GetErrorCode())
	})
}

func TestServerBroker_GetIndexStates(t *testing.T) {
	t.Run("failed to execute", func(t *testing.T) {
		c := newTestCore(withInvalidIndexCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		resp, err := b.GetIndexStates(ctx, []int64{1, 2})
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("non success error code on execute", func(t *testing.T) {
		c := newTestCore(withFailedIndexCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		_, err := b.GetIndexStates(ctx, []int64{1, 2})
		assert.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		c := newTestCore(withValidIndexCoord())
		b := newServerBroker(c)
		ctx := context.Background()
		_, err := b.GetIndexStates(ctx, []int64{1, 2})
		assert.NoError(t, err)
	})
}
