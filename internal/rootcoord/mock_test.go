package rootcoord

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus/internal/util/sessionutil"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/tso"

	"github.com/milvus-io/milvus/internal/proto/datapb"

	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/types"
)

type mockMetaTable struct {
	IMetaTableV2
	AddCollectionFunc func(ctx context.Context, coll *model.Collection) error
	CreateAliasFunc   func(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	AlterAliasFunc    func(ctx context.Context, alias string, collectionName string, ts Timestamp) error
}

func (m mockMetaTable) AddCollection(ctx context.Context, coll *model.Collection) error {
	return m.AddCollectionFunc(ctx, coll)
}

func (m mockMetaTable) CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	return m.CreateAliasFunc(ctx, alias, collectionName, ts)
}

func (m mockMetaTable) AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	return m.AlterAliasFunc(ctx, alias, collectionName, ts)
}

func newMockMetaTable() *mockMetaTable {
	return &mockMetaTable{}
}

type mockDataCoord struct {
	types.DataCoord
	WatchChannelsF func(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error)
}

func (m mockDataCoord) WatchChannels(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error) {
	return m.WatchChannelsF(ctx, req)
}

func newMockDataCoord() *mockDataCoord {
	return &mockDataCoord{}
}

func newIDAllocator() *allocator.MockGIDAllocator {
	r := allocator.NewMockGIDAllocator()
	r.AllocF = func(count uint32) (allocator.UniqueID, allocator.UniqueID, error) {
		return 0, 0, nil
	}
	r.AllocOneF = func() (allocator.UniqueID, error) {
		return 0, nil
	}
	return r
}

func newTsoAllocator() *tso.MockAllocator {
	r := tso.NewMockAllocator()
	r.GenerateTSOF = func(count uint32) (uint64, error) {
		return 0, nil
	}
	return r
}

func newTxnKV() *kv.TxnKVMock {
	r := kv.NewMockTxnKV()
	r.SaveF = func(key, value string) error {
		return nil
	}
	r.RemoveF = func(key string) error {
		return nil
	}
	return r
}

func newMeta() *mockMetaTable {
	r := newMockMetaTable()
	r.AddCollectionFunc = func(ctx context.Context, coll *model.Collection) error {
		return nil
	}
	return r
}

func newDC() types.DataCoord {
	r := newMockDataCoord()
	r.WatchChannelsF = func(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error) {
		return &datapb.WatchChannelsResponse{Status: succStatus()}, nil
	}
	return r
}

type mockProxy struct {
	types.Proxy
	InvalidateCollectionMetaCacheFunc func(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error)
}

func (m mockProxy) InvalidateCollectionMetaCache(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
	return m.InvalidateCollectionMetaCacheFunc(ctx, request)
}

func newMockProxy() *mockProxy {
	r := &mockProxy{}
	r.InvalidateCollectionMetaCacheFunc = func(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
		return succStatus(), nil
	}
	return r
}

func newTestCore(opts ...Opt) *RootCoord {
	c := &RootCoord{
		session: &sessionutil.Session{ServerID: TestRootCoordID},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func withValidProxyManager() Opt {
	return func(c *RootCoord) {
		c.proxyClientManager = &proxyClientManager{
			proxyClient: make(map[UniqueID]types.Proxy),
		}
		p := newMockProxy()
		p.InvalidateCollectionMetaCacheFunc = func(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
			return succStatus(), nil
		}
	}
}

func withInvalidProxyManager() Opt {
	return func(c *RootCoord) {
		c.proxyClientManager = &proxyClientManager{
			proxyClient: make(map[UniqueID]types.Proxy),
		}
		p := newMockProxy()
		p.InvalidateCollectionMetaCacheFunc = func(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
			return succStatus(), errors.New("mock")
		}
		c.proxyClientManager.proxyClient[TestProxyID] = p
	}
}

func withInvalidMeta() Opt {
	return func(c *RootCoord) {
		meta := newMockMetaTable()
		meta.CreateAliasFunc = func(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
			return errors.New("mock")
		}
		meta.AlterAliasFunc = func(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
			return errors.New("mock")
		}
		c.meta = meta
	}
}
