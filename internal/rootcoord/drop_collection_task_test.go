package rootcoord

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/proto/querypb"

	"github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/util/funcutil"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/stretchr/testify/assert"
)

func Test_dropCollectionTask_Prepare(t *testing.T) {
	t.Run("invalid msg type", func(t *testing.T) {
		task := &dropCollectionTask{
			Req: &milvuspb.DropCollectionRequest{
				Base: &commonpb.MsgBase{MsgType: commonpb.MsgType_DescribeCollection},
			},
		}
		err := task.Prepare(context.Background())
		assert.Error(t, err)
	})

	t.Run("drop via alias", func(t *testing.T) {
		collectionName := funcutil.GenRandomStr()
		meta := newMockMetaTable()
		meta.IsAliasFunc = func(name string) bool {
			return true
		}
		core := newTestCore(withMeta(meta))
		task := &dropCollectionTask{
			baseTaskV2: baseTaskV2{core: core},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: collectionName,
			},
		}
		err := task.Prepare(context.Background())
		assert.Error(t, err)
	})

	t.Run("normal case", func(t *testing.T) {
		collectionName := funcutil.GenRandomStr()
		meta := newMockMetaTable()
		meta.IsAliasFunc = func(name string) bool {
			return false
		}
		core := newTestCore(withMeta(meta))
		task := &dropCollectionTask{
			baseTaskV2: baseTaskV2{core: core},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: collectionName,
			},
		}
		err := task.Prepare(context.Background())
		assert.NoError(t, err)
	})
}

func Test_dropCollectionTask_Execute(t *testing.T) {
	t.Run("drop non-existent collection", func(t *testing.T) {
		collectionName := funcutil.GenRandomStr()
		core := newTestCore(withInvalidMeta())
		task := &dropCollectionTask{
			baseTaskV2: baseTaskV2{core: core},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: collectionName,
			},
		}
		err := task.Execute(context.Background())
		assert.NoError(t, err)
	})

	t.Run("failed to expire cache", func(t *testing.T) {
		collectionName := funcutil.GenRandomStr()
		coll := &model.Collection{Name: collectionName}
		meta := newMockMetaTable()
		meta.GetCollectionByNameFunc = func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
			return coll.Clone(), nil
		}
		core := newTestCore(withInvalidProxyManager(), withMeta(meta))
		task := &dropCollectionTask{
			baseTaskV2: baseTaskV2{core: core},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: collectionName,
			},
		}
		err := task.Execute(context.Background())
		assert.Error(t, err)
	})

	t.Run("failed to change collection state", func(t *testing.T) {
		collectionName := funcutil.GenRandomStr()
		coll := &model.Collection{Name: collectionName}
		meta := newMockMetaTable()
		meta.GetCollectionByNameFunc = func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
			return coll.Clone(), nil
		}
		meta.ChangeCollectionStateFunc = func(ctx context.Context, collectionID UniqueID, state etcdpb.CollectionState, ts Timestamp) error {
			return errors.New("error mock ChangeCollectionState")
		}
		core := newTestCore(withValidProxyManager(), withMeta(meta))
		task := &dropCollectionTask{
			baseTaskV2: baseTaskV2{core: core},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: collectionName,
			},
		}
		err := task.Execute(context.Background())
		assert.Error(t, err)
	})

	t.Run("normal case, redo", func(t *testing.T) {
		defer cleanTestEnv()

		collectionName := funcutil.GenRandomStr()
		shardNum := 2

		ticker := newRocksMqTtSynchronizer()
		var pchans []string
		var deltaChans []string
		for i := 0; i < shardNum; i++ {
			pchans = append(pchans, ticker.getDmlChannelName())
			deltaChans = append(deltaChans, ticker.getDeltaChannelName())
		}
		ticker.addDmlChannels(pchans...)
		ticker.addDeltaChannels(deltaChans...)

		coll := &model.Collection{Name: collectionName, ShardsNum: int32(shardNum), PhysicalChannelNames: pchans}
		meta := newMockMetaTable()
		meta.GetCollectionByNameFunc = func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
			return coll.Clone(), nil
		}
		meta.ChangeCollectionStateFunc = func(ctx context.Context, collectionID UniqueID, state etcdpb.CollectionState, ts Timestamp) error {
			return nil
		}
		removeCollectionMetaCalled := false
		removeCollectionMetaChan := make(chan struct{}, 1)
		meta.RemoveCollectionFunc = func(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
			removeCollectionMetaCalled = true
			removeCollectionMetaChan <- struct{}{}
			return nil
		}

		qc := newMockQueryCoord()
		releaseCollectionCalled := false
		releaseCollectionChan := make(chan struct{}, 1)
		qc.ReleaseCollectionFunc = func(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
			releaseCollectionChan <- struct{}{}
			releaseCollectionCalled = true
			return succStatus(), nil
		}
		qc.GetComponentStatesFunc = func(ctx context.Context) (*internalpb.ComponentStates, error) {
			return &internalpb.ComponentStates{
				State: &internalpb.ComponentInfo{
					NodeID:    TestRootCoordID,
					StateCode: internalpb.StateCode_Healthy,
				},
				SubcomponentStates: nil,
				Status:             succStatus(),
			}, nil
		}

		core := newTestCore(
			withValidProxyManager(),
			withMeta(meta),
			withQueryCoord(qc),
			withTtSynchronizer(ticker))

		task := &dropCollectionTask{
			baseTaskV2: baseTaskV2{core: core},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: collectionName,
			},
		}
		err := task.Execute(context.Background())
		assert.NoError(t, err)

		// check if redo worked.

		<-releaseCollectionChan
		assert.True(t, releaseCollectionCalled)

		<-removeCollectionMetaChan
		assert.True(t, removeCollectionMetaCalled)
	})
}
