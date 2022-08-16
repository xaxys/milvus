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
	AddCreatingCollectionF func(ctx context.Context, coll *model.Collection) error
}

func (m mockMetaTable) AddCollection(ctx context.Context, coll *model.Collection) error {
	return m.AddCreatingCollectionF(ctx, coll)
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
	r.AddCreatingCollectionF = func(ctx context.Context, coll *model.Collection) error {
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
