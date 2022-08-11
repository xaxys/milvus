package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/tso"

	"github.com/milvus-io/milvus/internal/proto/datapb"

	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/types"
)

type mockMetaTable struct {
	IMetaTableV2
	AddCollectionF    func(ctx context.Context, coll *model.Collection) error
	EnableCollectionF func(ctx context.Context, collectionID UniqueID) error
	DeleteCollectionF func(ctx context.Context, collectionID UniqueID, ts Timestamp) error
}

func (m mockMetaTable) AddCollection(ctx context.Context, coll *model.Collection) error {
	return m.AddCollectionF(ctx, coll)
}

func (m mockMetaTable) GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
	panic("implement me")
}

func (m mockMetaTable) DeleteCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
	return m.DeleteCollectionF(ctx, collectionID, ts)
}

func (m mockMetaTable) EnableCollection(ctx context.Context, collectionID UniqueID) error {
	return m.EnableCollectionF(ctx, collectionID)
}

func (m mockMetaTable) DisableCollection(ctx context.Context, collectionID UniqueID) error {
	panic("implement me")
}

func (m mockMetaTable) IsAlias(name string) bool {
	panic("implement me")
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
	r.AddCollectionF = func(ctx context.Context, coll *model.Collection) error {
		return nil
	}
	r.EnableCollectionF = func(ctx context.Context, collectionID UniqueID) error {
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
