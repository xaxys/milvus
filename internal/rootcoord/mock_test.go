package rootcoord

import (
	"context"
	"errors"
	"math/rand"
	"os"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/util/dependency"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/querypb"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

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

const (
	TestProxyID     = 100
	TestRootCoordID = 200
)

type mockMetaTable struct {
	IMetaTableV2
	AddCollectionFunc         func(ctx context.Context, coll *model.Collection) error
	GetCollectionByNameFunc   func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error)
	GetCollectionByIDFunc     func(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error)
	ChangeCollectionStateFunc func(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error
	RemoveCollectionFunc      func(ctx context.Context, collectionID UniqueID, ts Timestamp) error
	AddPartitionFunc          func(ctx context.Context, partition *model.Partition) error
	ChangePartitionStateFunc  func(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error
	RemovePartitionFunc       func(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error
	CreateAliasFunc           func(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	AlterAliasFunc            func(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	DropAliasFunc             func(ctx context.Context, alias string, ts Timestamp) error
	IsAliasFunc               func(name string) bool
}

func (m mockMetaTable) AddCollection(ctx context.Context, coll *model.Collection) error {
	return m.AddCollectionFunc(ctx, coll)
}

func (m mockMetaTable) GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
	return m.GetCollectionByNameFunc(ctx, collectionName, ts)
}

func (m mockMetaTable) GetCollectionByID(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
	return m.GetCollectionByIDFunc(ctx, collectionID, ts)
}

func (m mockMetaTable) ChangeCollectionState(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error {
	return m.ChangeCollectionStateFunc(ctx, collectionID, state, ts)
}

func (m mockMetaTable) RemoveCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
	return m.RemoveCollectionFunc(ctx, collectionID, ts)
}

func (m mockMetaTable) AddPartition(ctx context.Context, partition *model.Partition) error {
	return m.AddPartitionFunc(ctx, partition)
}

func (m mockMetaTable) ChangePartitionState(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error {
	return m.ChangePartitionStateFunc(ctx, collectionID, partitionID, state, ts)
}

func (m mockMetaTable) RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error {
	return m.RemovePartitionFunc(ctx, collectionID, partitionID, ts)
}

func (m mockMetaTable) CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	return m.CreateAliasFunc(ctx, alias, collectionName, ts)
}

func (m mockMetaTable) AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	return m.AlterAliasFunc(ctx, alias, collectionName, ts)
}

func (m mockMetaTable) DropAlias(ctx context.Context, alias string, ts Timestamp) error {
	return m.DropAliasFunc(ctx, alias, ts)
}

func (m mockMetaTable) IsAlias(name string) bool {
	return m.IsAliasFunc(name)
}

func newMockMetaTable() *mockMetaTable {
	return &mockMetaTable{}
}

type mockDataCoord struct {
	types.DataCoord
	GetComponentStatesFunc func(ctx context.Context) (*internalpb.ComponentStates, error)
	WatchChannelsF         func(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error)
}

func newMockDataCoord() *mockDataCoord {
	return &mockDataCoord{}
}

func (m mockDataCoord) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	return m.GetComponentStatesFunc(ctx)
}

func (m mockDataCoord) WatchChannels(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error) {
	return m.WatchChannelsF(ctx, req)
}

type mockQueryCoord struct {
	types.QueryCoord
	GetComponentStatesFunc func(ctx context.Context) (*internalpb.ComponentStates, error)
	ReleaseCollectionFunc  func(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error)
}

func (m mockQueryCoord) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	return m.GetComponentStatesFunc(ctx)
}

func (m mockQueryCoord) ReleaseCollection(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	return m.ReleaseCollectionFunc(ctx, req)
}

func newMockQueryCoord() *mockQueryCoord {
	return &mockQueryCoord{}
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
			return succStatus(), errors.New("error mock InvalidateCollectionMetaCache")
		}
		c.proxyClientManager.proxyClient[TestProxyID] = p
	}
}

func withMeta(meta IMetaTableV2) Opt {
	return func(c *RootCoord) {
		c.meta = meta
	}
}

func withInvalidMeta() Opt {
	meta := newMockMetaTable()
	meta.GetCollectionByNameFunc = func(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
		return nil, errors.New("error mock GetCollectionByName")
	}
	meta.AddPartitionFunc = func(ctx context.Context, partition *model.Partition) error {
		return errors.New("error mock AddPartition")
	}
	meta.ChangePartitionStateFunc = func(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error {
		return errors.New("error mock ChangePartitionState")
	}
	meta.CreateAliasFunc = func(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
		return errors.New("error mock CreateAlias")
	}
	meta.AlterAliasFunc = func(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
		return errors.New("error mock AlterAlias")
	}
	meta.DropAliasFunc = func(ctx context.Context, alias string, ts Timestamp) error {
		return errors.New("error mock DropAlias")
	}
	return withMeta(meta)
}

func withIdAllocator(idAllocator allocator.GIDAllocator) Opt {
	return func(c *RootCoord) {
		c.idAllocator = idAllocator
	}
}

func withValidIdAllocator() Opt {
	idAllocator := newIDAllocator()
	idAllocator.AllocOneF = func() (allocator.UniqueID, error) {
		return rand.Int63(), nil
	}
	return withIdAllocator(idAllocator)
}

func withInvalidIdAllocator() Opt {
	idAllocator := newIDAllocator()
	idAllocator.AllocOneF = func() (allocator.UniqueID, error) {
		return -1, errors.New("error mock AllocOne")
	}
	return withIdAllocator(idAllocator)
}

func withQueryCoord(qc types.QueryCoord) Opt {
	return func(c *RootCoord) {
		c.queryCoord = qc
	}
}

func withInvalidQueryCoord() Opt {
	qc := newMockQueryCoord()
	qc.ReleaseCollectionFunc = func(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
		return nil, errors.New("error mock ReleaseCollection")
	}
	return withQueryCoord(qc)
}

func withValidQueryCoord() Opt {
	qc := newMockQueryCoord()
	qc.ReleaseCollectionFunc = func(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
		return succStatus(), nil
	}
	return withQueryCoord(qc)
}

// cleanTestEnv clean test environment, for example, files generated by rocksmq.
func cleanTestEnv() {
	path := "/tmp/milvus"
	if err := os.RemoveAll(path); err != nil {
		log.Warn("failed to clean test directories", zap.Error(err), zap.String("path", path))
	}
	log.Debug("clean test environment", zap.String("path", path))
}

func withTtSynchronizer(ticker *timetickSync) Opt {
	return func(c *RootCoord) {
		c.chanTimeTick = ticker
	}
}

func newRocksMqTtSynchronizer() *timetickSync {
	Params.InitOnce()
	Params.RootCoordCfg.DmlChannelNum = 4
	ctx := context.Background()
	factory := dependency.NewDefaultFactory(true)
	chans := map[UniqueID][]string{}
	ticker := newTimeTickSync(ctx, TestRootCoordID, factory, chans)
	return ticker
}

// cleanTestEnv should be called if tested with this option.
func withRocksMqTtSynchronizer() Opt {
	ticker := newRocksMqTtSynchronizer()
	return withTtSynchronizer(ticker)
}

func withDataCoord(dc types.DataCoord) Opt {
	return func(c *RootCoord) {
		c.dataCoord = dc
	}
}
