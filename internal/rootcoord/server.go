package rootcoord

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/milvus-io/milvus/internal/util/crypto"

	"github.com/milvus-io/milvus/internal/metastore"

	"github.com/milvus-io/milvus/internal/util"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/util/timerecord"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/common"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metastore/db/dao"
	"github.com/milvus-io/milvus/internal/metastore/db/dbcore"
	rootcoord2 "github.com/milvus-io/milvus/internal/metastore/db/rootcoord"
	"github.com/milvus-io/milvus/internal/metastore/kv/rootcoord"
	kvmetestore "github.com/milvus-io/milvus/internal/metastore/kv/rootcoord"

	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/metrics"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/sessionutil"

	"github.com/milvus-io/milvus/internal/util/metricsinfo"

	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/milvus-io/milvus/internal/types"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/tso"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
)

type metaKVCreator func(root string) (kv.MetaKv, error)

type Opt func(*RootCoord)

func defaultMetaKVCreator(etcdCli *clientv3.Client) metaKVCreator {
	return func(root string) (kv.MetaKv, error) {
		return etcdkv.NewEtcdKV(etcdCli, root), nil
	}
}

type RootCoord struct {
	types.RootCoord // TODO: remove me after everything is ready.

	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	etcdCli          *clientv3.Client
	meta             IMetaTableV2
	scheduler        IScheduler
	broker           Broker
	garbageCollector GarbageCollector

	metaKVCreator metaKVCreator

	proxyCreator       proxyCreator
	proxyManager       *proxyManager
	proxyClientManager *proxyClientManager

	metricsCacheManager *metricsinfo.MetricsCacheManager

	chanTimeTick *timetickSync

	idAllocator  allocator.GIDAllocator
	tsoAllocator tso.Allocator

	dataCoord  types.DataCoord
	queryCoord types.QueryCoord
	indexCoord types.IndexCoord

	stateCode atomic.Value
	initOnce  sync.Once
	startOnce sync.Once
	session   *sessionutil.Session

	factory dependency.Factory

	importManager *importManager
}

func NewRootCoord(ctx context.Context, factory dependency.Factory, opts ...Opt) (*RootCoord, error) {
	ctx1, cancel := context.WithCancel(ctx)
	rand.Seed(time.Now().UnixNano())
	core := &RootCoord{
		ctx:     ctx1,
		cancel:  cancel,
		factory: factory,
	}
	core.UpdateStateCode(internalpb.StateCode_Abnormal)

	for _, opt := range opts {
		opt(core)
	}

	return core, nil
}

func (c *RootCoord) UpdateStateCode(code internalpb.StateCode) {
	c.stateCode.Store(code)
}

func (c *RootCoord) checkHealthy() (internalpb.StateCode, bool) {
	code := c.stateCode.Load().(internalpb.StateCode)
	ok := code == internalpb.StateCode_Healthy
	return code, ok
}

func (c *RootCoord) tsLoop() {
	defer c.wg.Done()
	tsoTicker := time.NewTicker(tso.UpdateTimestampStep)
	defer tsoTicker.Stop()
	ctx, cancel := context.WithCancel(c.ctx)
	defer cancel()
	for {
		select {
		case <-tsoTicker.C:
			if err := c.tsoAllocator.UpdateTSO(); err != nil {
				log.Warn("failed to update timestamp: ", zap.Error(err))
				continue
			}
			ts := c.tsoAllocator.GetLastSavedTime()
			metrics.RootCoordTimestampSaved.Set(float64(ts.Unix()))
			if err := c.tsoAllocator.UpdateTSO(); err != nil {
				log.Warn("failed to update id: ", zap.Error(err))
				continue
			}
		case <-ctx.Done():
			return
		}
	}
}

func (c *RootCoord) sendTimeTick(t Timestamp, reason string) error {
	pc := c.chanTimeTick.listDmlChannels()
	pt := make([]uint64, len(pc))
	for i := 0; i < len(pt); i++ {
		pt[i] = t
	}
	ttMsg := internalpb.ChannelTimeTickMsg{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_TimeTick,
			Timestamp: t,
			SourceID:  c.session.ServerID,
		},
		ChannelNames:     pc,
		Timestamps:       pt,
		DefaultTimestamp: t,
	}
	return c.chanTimeTick.updateTimeTick(&ttMsg, reason)
}

func (c *RootCoord) startTimeTickLoop() {
	defer c.wg.Done()
	ticker := time.NewTicker(Params.ProxyCfg.TimeTickInterval)
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			if ts, err := c.tsoAllocator.GenerateTSO(1); err == nil {
				err := c.sendTimeTick(ts, "timetick loop")
				if err != nil {
					log.Warn("failed to send timetick", zap.Error(err))
				}
			}
		}
	}
}

func (c *RootCoord) SetNewProxyClient(f func(sess *sessionutil.Session) (types.Proxy, error)) {
	c.proxyCreator = f
}

func (c *RootCoord) SetDataCoord(ctx context.Context, s types.DataCoord) error {
	if err := s.Init(); err != nil {
		return err
	}
	if err := s.Start(); err != nil {
		return err
	}
	c.dataCoord = s
	return nil
}

func (c *RootCoord) SetIndexCoord(s types.IndexCoord) error {
	if err := s.Init(); err != nil {
		return err
	}
	if err := s.Start(); err != nil {
		return err
	}
	c.indexCoord = s
	return nil
}

func (c *RootCoord) SetQueryCoord(s types.QueryCoord) error {
	if err := s.Init(); err != nil {
		return err
	}
	if err := s.Start(); err != nil {
		return err
	}
	c.queryCoord = s
	return nil
}

func (c *RootCoord) ExpireMetaCache(ctx context.Context, collNames []string, collectionID UniqueID, ts typeutil.Timestamp) error {
	// if collectionID is specified, invalidate all the collection meta cache with the specified collectionID and return
	if collectionID != InvalidCollectionID {
		req := proxypb.InvalidateCollMetaCacheRequest{
			Base: &commonpb.MsgBase{
				Timestamp: ts,
				SourceID:  c.session.ServerID,
			},
			CollectionID: collectionID,
		}
		return c.proxyClientManager.InvalidateCollectionMetaCache(ctx, &req)
	}

	// if only collNames are specified, invalidate the collection meta cache with the specified collectionName
	for _, collName := range collNames {
		req := proxypb.InvalidateCollMetaCacheRequest{
			Base: &commonpb.MsgBase{
				MsgType:   0, //TODO, msg type
				MsgID:     0, //TODO, msg id
				Timestamp: ts,
				SourceID:  c.session.ServerID,
			},
			CollectionName: collName,
		}
		err := c.proxyClientManager.InvalidateCollectionMetaCache(ctx, &req)
		if err != nil {
			// TODO: try to expire all or directly return err?
			return err
		}
	}
	return nil
}

func (c *RootCoord) Register() error {
	c.session.Register()
	go c.session.LivenessCheck(c.ctx, func() {
		log.Error("Root Coord disconnected from etcd, process will exit", zap.Int64("Server Id", c.session.ServerID))
		if err := c.Stop(); err != nil {
			log.Fatal("failed to stop server", zap.Error(err))
		}
		// manually send signal to starter goroutine
		if c.session.TriggerKill {
			if p, err := os.FindProcess(os.Getpid()); err == nil {
				p.Signal(syscall.SIGINT)
			}
		}
	})

	c.UpdateStateCode(internalpb.StateCode_Healthy)
	return nil
}

func (c *RootCoord) SetEtcdClient(etcdClient *clientv3.Client) {
	c.etcdCli = etcdClient
}

func (c *RootCoord) initSession() error {
	c.session = sessionutil.NewSession(c.ctx, Params.EtcdCfg.MetaRootPath, c.etcdCli)
	if c.session == nil {
		return fmt.Errorf("session is nil, the etcd client connection may have failed")
	}
	c.session.Init(typeutil.RootCoordRole, Params.RootCoordCfg.Address, true, true)
	Params.SetLogger(c.session.ServerID)
	return nil
}

func (c *RootCoord) initKVCreator() {
	if c.metaKVCreator == nil {
		c.metaKVCreator = defaultMetaKVCreator(c.etcdCli)
	}
}

func (c *RootCoord) initMetaTable() error {
	fn := func() error {
		var catalog metastore.RootCoordCatalog
		var err error

		switch Params.MetaStoreCfg.MetaStoreType {
		case util.MetaStoreTypeEtcd:
			var metaKV kv.MetaKv
			var ss *kvmetestore.SuffixSnapshot
			var err error

			if metaKV, err = c.metaKVCreator(Params.EtcdCfg.KvRootPath); err != nil {
				return err
			}

			if ss, err = kvmetestore.NewSuffixSnapshot(metaKV, snapshotsSep, Params.EtcdCfg.MetaRootPath, snapshotPrefix); err != nil {
				return err
			}

			catalog = &rootcoord.Catalog{Txn: metaKV, Snapshot: ss}
		case util.MetaStoreTypeMysql:
			// connect to database
			err := dbcore.Connect(&Params.DBCfg)
			if err != nil {
				return err
			}

			catalog = rootcoord2.NewTableCatalog(dbcore.NewTxImpl(), dao.NewMetaDomain())
		default:
			return fmt.Errorf("not supported meta store: %s", Params.MetaStoreCfg.MetaStoreType)
		}

		if c.meta, err = newMetaTableV2(c.ctx, catalog); err != nil {
			return err
		}

		return nil
	}

	return retry.Do(c.ctx, fn, retry.Attempts(10))
}

func (c *RootCoord) initIDAllocator() error {
	tsoKV := tsoutil.NewTSOKVBase(c.etcdCli, Params.EtcdCfg.KvRootPath, globalIDAllocatorSubPath)
	idAllocator := allocator.NewGlobalIDAllocator(globalIDAllocatorKey, tsoKV)
	if err := idAllocator.Initialize(); err != nil {
		return err
	}
	c.idAllocator = idAllocator
	return nil
}

func (c *RootCoord) initTSOAllocator() error {
	tsoKV := tsoutil.NewTSOKVBase(c.etcdCli, Params.EtcdCfg.KvRootPath, globalTSOAllocatorSubPath)
	tsoAllocator := tso.NewGlobalTSOAllocator(globalTSOAllocatorKey, tsoKV)
	if err := tsoAllocator.Initialize(); err != nil {
		return err
	}
	c.tsoAllocator = tsoAllocator

	return nil
}

func (c *RootCoord) createRootCredential() error {
	encryptedRootPassword, err := crypto.PasswordEncrypt(util.DefaultRootPassword)
	if err != nil {
		return err
	}
	return c.meta.AddCredential(&internalpb.CredentialInfo{Username: util.UserRoot, EncryptedPassword: encryptedRootPassword})
}

func (c *RootCoord) initCredentials() error {
	// error ignored if no root credential found, we'll then create default root credential.
	credInfo, _ := c.meta.GetCredential(util.UserRoot)

	if credInfo == nil {
		return c.createRootCredential()
	}

	return nil
}

func (c *RootCoord) initRbac() (initError error) {
	// create default roles, including admin, public
	if initError = c.meta.CreateRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: util.RoleAdmin}); initError != nil {
		return
	}
	if initError = c.meta.CreateRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: util.RolePublic}); initError != nil {
		return
	}

	// grant privileges for the public role
	globalPrivileges := []string{
		commonpb.ObjectPrivilege_PrivilegeDescribeCollection.String(),
		commonpb.ObjectPrivilege_PrivilegeShowCollections.String(),
	}
	collectionPrivileges := []string{
		commonpb.ObjectPrivilege_PrivilegeIndexDetail.String(),
	}

	for _, globalPrivilege := range globalPrivileges {
		if initError = c.meta.OperatePrivilege(util.DefaultTenant, &milvuspb.GrantEntity{
			Role:       &milvuspb.RoleEntity{Name: util.RolePublic},
			Object:     &milvuspb.ObjectEntity{Name: commonpb.ObjectType_Global.String()},
			ObjectName: util.AnyWord,
			Grantor: &milvuspb.GrantorEntity{
				User:      &milvuspb.UserEntity{Name: util.RoleAdmin},
				Privilege: &milvuspb.PrivilegeEntity{Name: globalPrivilege},
			},
		}, milvuspb.OperatePrivilegeType_Grant); initError != nil {
			return
		}
	}
	for _, collectionPrivilege := range collectionPrivileges {
		if initError = c.meta.OperatePrivilege(util.DefaultTenant, &milvuspb.GrantEntity{
			Role:       &milvuspb.RoleEntity{Name: util.RolePublic},
			Object:     &milvuspb.ObjectEntity{Name: commonpb.ObjectType_Collection.String()},
			ObjectName: util.AnyWord,
			Grantor: &milvuspb.GrantorEntity{
				User:      &milvuspb.UserEntity{Name: util.RoleAdmin},
				Privilege: &milvuspb.PrivilegeEntity{Name: collectionPrivilege},
			},
		}, milvuspb.OperatePrivilegeType_Grant); initError != nil {
			return
		}
	}
	return nil
}

func (c *RootCoord) initImportManager() error {
	impTaskKv, err := c.metaKVCreator(Params.EtcdCfg.KvRootPath)
	if err != nil {
		return err
	}

	f := NewImportFactory(c)
	c.importManager = newImportManager(
		c.ctx,
		impTaskKv,
		f.NewIdAllocator(),
		f.NewImportFunc(),
		f.NewGetCollectionNameFunc(),
	)
	c.importManager.init(c.ctx)

	return nil
}

func (c *RootCoord) initInternal() error {
	if err := c.initSession(); err != nil {
		return err
	}

	c.initKVCreator()

	if err := c.initMetaTable(); err != nil {
		return err
	}

	if err := c.initIDAllocator(); err != nil {
		return err
	}

	if err := c.initTSOAllocator(); err != nil {
		return err
	}

	c.scheduler = newScheduler(c.ctx, c.idAllocator, c.tsoAllocator)

	c.factory.Init(&Params)

	chanMap := c.meta.ListCollectionPhysicalChannels()
	c.chanTimeTick = newTimeTickSync(c.ctx, c.session.ServerID, c.factory, chanMap)
	c.chanTimeTick.addSession(c.session)
	c.proxyClientManager = newProxyClientManager(c.proxyCreator)

	c.broker = newServerBroker(c)
	c.garbageCollector = newGarbageCollectorCtx(c)

	c.proxyManager = newProxyManager(
		c.ctx,
		c.etcdCli,
		c.chanTimeTick.initSessions,
		c.proxyClientManager.GetProxyClients,
	)
	c.proxyManager.AddSessionFunc(c.chanTimeTick.addSession, c.proxyClientManager.AddProxyClient)
	c.proxyManager.DelSessionFunc(c.chanTimeTick.delSession, c.proxyClientManager.DelProxyClient)

	c.metricsCacheManager = metricsinfo.NewMetricsCacheManager()

	if err := c.initImportManager(); err != nil {
		return err
	}

	if err := c.initCredentials(); err != nil {
		return err
	}

	if err := c.initRbac(); err != nil {
		return err
	}

	return nil
}

func (c *RootCoord) Init() error {
	var initError error
	c.initOnce.Do(func() {
		initError = c.initInternal()
	})
	return initError
}

func (c *RootCoord) restore(ctx context.Context) error {
	colls, err := c.meta.ListAbnormalCollections(ctx, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}

	for _, coll := range colls {
		ts, err := c.tsoAllocator.GenerateTSO(1)
		if err != nil {
			return err
		}

		switch coll.State {
		case pb.CollectionState_CollectionDropping:
			go c.garbageCollector.ReDropCollection(coll.Clone(), ts)
		case pb.CollectionState_CollectionCreating:
			go c.garbageCollector.RemoveCreatingCollection(coll.Clone())
		default:
		}
	}

	colls, err = c.meta.ListCollections(ctx, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	for _, coll := range colls {
		for _, part := range coll.Partitions {
			ts, err := c.tsoAllocator.GenerateTSO(1)
			if err != nil {
				return err
			}

			switch part.State {
			case pb.PartitionState_PartitionDropping:
				go c.garbageCollector.ReDropPartition(coll.PhysicalChannelNames, part.Clone(), ts)
			default:
			}
		}
	}
	return nil
}

func (c *RootCoord) startInternal() error {
	if err := c.proxyManager.WatchProxy(); err != nil {
		log.Fatal("rootcoord failed to watch proxy", zap.Error(err))
		// you can not just stuck here,
		panic(err)
	}

	if err := c.restore(c.ctx); err != nil {
		panic(err)
	}

	c.wg.Add(5)
	go c.tsLoop()
	go c.startTimeTickLoop()
	go c.chanTimeTick.startWatch(&c.wg)
	go c.importManager.expireOldTasksLoop(&c.wg, c.broker.ReleaseSegRefLock)
	go c.importManager.sendOutTasksLoop(&c.wg)

	c.scheduler.Start()

	Params.RootCoordCfg.CreatedTime = time.Now()
	Params.RootCoordCfg.UpdatedTime = time.Now()

	return nil
}

// Start starts RootCoord.
func (c *RootCoord) Start() error {
	var err error
	c.startOnce.Do(func() {
		err = c.startInternal()
	})
	return err
}

// Stop stops rootCoord.
func (c *RootCoord) Stop() error {
	c.UpdateStateCode(internalpb.StateCode_Abnormal)
	c.scheduler.Stop()
	c.cancel()
	c.wg.Wait()
	// wait at most one second to revoke
	c.session.Revoke(time.Second)
	return nil
}

// GetComponentStates get states of components
func (c *RootCoord) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	code := c.stateCode.Load().(internalpb.StateCode)

	nodeID := common.NotRegisteredID
	if c.session != nil && c.session.Registered() {
		nodeID = c.session.ServerID
	}

	return &internalpb.ComponentStates{
		State: &internalpb.ComponentInfo{
			// NodeID:    c.session.ServerID, // will race with Core.Register()
			NodeID:    nodeID,
			Role:      typeutil.RootCoordRole,
			StateCode: code,
			ExtraInfo: nil,
		},
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		SubcomponentStates: []*internalpb.ComponentInfo{
			{
				NodeID:    nodeID,
				Role:      typeutil.RootCoordRole,
				StateCode: code,
				ExtraInfo: nil,
			},
		},
	}, nil
}

// GetTimeTickChannel get timetick channel name
func (c *RootCoord) GetTimeTickChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Value: Params.CommonCfg.RootCoordTimeTick,
	}, nil
}

// GetStatisticsChannel get statistics channel name
func (c *RootCoord) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	return &milvuspb.StringResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
			Reason:    "",
		},
		Value: Params.CommonCfg.RootCoordStatistics,
	}, nil
}

func (c *RootCoord) CreateCollection(ctx context.Context, in *milvuspb.CreateCollectionRequest) (*commonpb.Status, error) {
	t := &createCollectionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) DropCollection(ctx context.Context, in *milvuspb.DropCollectionRequest) (*commonpb.Status, error) {
	t := &dropCollectionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) DescribeCollection(ctx context.Context, in *milvuspb.DescribeCollectionRequest) (*milvuspb.DescribeCollectionResponse, error) {
	metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.TotalLabel).Inc()
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.DescribeCollectionResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode"+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	tr := timerecord.NewTimeRecorder("DescribeCollection")

	log.Debug("DescribeCollection", zap.String("role", typeutil.RootCoordRole),
		zap.String("collection name", in.CollectionName), zap.Int64("id", in.CollectionID), zap.Int64("msgID", in.Base.MsgID))
	t := &describeCollectionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
		Rsp: &milvuspb.DescribeCollectionResponse{},
	}
	if err := c.scheduler.AddTask(t); err != nil {
		log.Error("DescribeCollection task enqueue failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("id", in.CollectionID), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.FailLabel).Inc()
		return &milvuspb.DescribeCollectionResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "DescribeCollection failed: "+err.Error()),
		}, nil
	}
	if err := t.WaitToFinish(); err != nil {
		log.Error("DescribeCollection failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("id", in.CollectionID), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.FailLabel).Inc()
		return &milvuspb.DescribeCollectionResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "DescribeCollection failed: "+err.Error()),
		}, nil
	}

	metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues("DescribeCollection").Observe(float64(tr.ElapseSpan().Milliseconds()))
	return t.Rsp, nil
}

func (c *RootCoord) HasCollection(ctx context.Context, in *milvuspb.HasCollectionRequest) (*milvuspb.BoolResponse, error) {
	metrics.RootCoordDDLReqCounter.WithLabelValues("HasCollection", metrics.TotalLabel).Inc()
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.BoolResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
			Value:  false,
		}, nil
	}
	tr := timerecord.NewTimeRecorder("HasCollection")

	log.Debug("HasCollection", zap.String("role", typeutil.RootCoordRole),
		zap.String("collection name", in.CollectionName), zap.Int64("msgID", in.Base.MsgID))
	t := &hasCollectionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
		Rsp: &milvuspb.BoolResponse{},
	}
	if err := c.scheduler.AddTask(t); err != nil {
		log.Error("HasCollection task enqueue failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("HasCollection", metrics.FailLabel).Inc()
		return &milvuspb.BoolResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "HasCollection failed: "+err.Error()),
			Value:  false,
		}, nil
	}
	if err := t.WaitToFinish(); err != nil {
		log.Error("HasCollection failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("HasCollection", metrics.FailLabel).Inc()
		return &milvuspb.BoolResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "HasCollection failed: "+err.Error()),
			Value:  false,
		}, nil
	}

	metrics.RootCoordDDLReqCounter.WithLabelValues("HasCollection", metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues("HasCollection").Observe(float64(tr.ElapseSpan().Milliseconds()))
	return t.Rsp, nil
}

func (c *RootCoord) ShowCollections(ctx context.Context, in *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error) {
	metrics.RootCoordDDLReqCounter.WithLabelValues("ShowCollections", metrics.TotalLabel).Inc()
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.ShowCollectionsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	tr := timerecord.NewTimeRecorder("ShowCollections")

	log.Debug("ShowCollections", zap.String("role", typeutil.RootCoordRole),
		zap.String("dbname", in.DbName), zap.Int64("msgID", in.Base.MsgID))
	t := &showCollectionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
		Rsp: &milvuspb.ShowCollectionsResponse{},
	}
	if err := c.scheduler.AddTask(t); err != nil {
		log.Error("ShowCollections task enqueue failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("dbname", in.DbName), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("ShowCollections", metrics.FailLabel).Inc()
		return &milvuspb.ShowCollectionsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "ShowCollections failed: "+err.Error()),
		}, nil
	}
	if err := t.WaitToFinish(); err != nil {
		log.Error("ShowCollections failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("dbname", in.DbName), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("ShowCollections", metrics.FailLabel).Inc()
		return &milvuspb.ShowCollectionsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "ShowCollections failed: "+err.Error()),
		}, nil
	}

	metrics.RootCoordDDLReqCounter.WithLabelValues("ShowCollections", metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues("ShowCollections").Observe(float64(tr.ElapseSpan().Milliseconds()))
	return t.Rsp, nil
}

func (c *RootCoord) CreatePartition(ctx context.Context, in *milvuspb.CreatePartitionRequest) (*commonpb.Status, error) {
	t := &createPartitionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) DropPartition(ctx context.Context, in *milvuspb.DropPartitionRequest) (*commonpb.Status, error) {
	t := &dropPartitionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) HasPartition(ctx context.Context, in *milvuspb.HasPartitionRequest) (*milvuspb.BoolResponse, error) {
	metrics.RootCoordDDLReqCounter.WithLabelValues("HasPartition", metrics.TotalLabel).Inc()
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.BoolResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
			Value:  false,
		}, nil
	}
	tr := timerecord.NewTimeRecorder("HasPartition")

	log.Debug("HasPartition", zap.String("role", typeutil.RootCoordRole),
		zap.String("collection name", in.CollectionName), zap.String("partition name", in.PartitionName),
		zap.Int64("msgID", in.Base.MsgID))
	t := &hasPartitionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
		Rsp: &milvuspb.BoolResponse{},
	}
	if err := c.scheduler.AddTask(t); err != nil {
		log.Error("HasPartition task enqueue failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.String("partition name", in.PartitionName),
			zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("HasPartition", metrics.FailLabel).Inc()
		return &milvuspb.BoolResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "HasPartition failed: "+err.Error()),
			Value:  false,
		}, nil
	}
	if err := t.WaitToFinish(); err != nil {
		log.Error("HasPartition failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.String("partition name", in.PartitionName),
			zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("HasPartition", metrics.FailLabel).Inc()
		return &milvuspb.BoolResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "HasPartition failed: "+err.Error()),
			Value:  false,
		}, nil
	}

	metrics.RootCoordDDLReqCounter.WithLabelValues("HasPartition", metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues("HasPartition").Observe(float64(tr.ElapseSpan().Milliseconds()))
	return t.Rsp, nil
}

func (c *RootCoord) ShowPartitions(ctx context.Context, in *milvuspb.ShowPartitionsRequest) (*milvuspb.ShowPartitionsResponse, error) {
	metrics.RootCoordDDLReqCounter.WithLabelValues("ShowPartitions", metrics.TotalLabel).Inc()
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.ShowPartitionsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	tr := timerecord.NewTimeRecorder("ShowPartitions")

	log.Debug("ShowPartitions", zap.String("role", typeutil.RootCoordRole),
		zap.String("collection name", in.CollectionName), zap.Int64("msgID", in.Base.MsgID))
	t := &showPartitionTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
		Rsp: &milvuspb.ShowPartitionsResponse{},
	}
	if err := c.scheduler.AddTask(t); err != nil {
		log.Error("ShowPartitions task enqueue failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("ShowPartitions", metrics.FailLabel).Inc()
		return &milvuspb.ShowPartitionsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "ShowPartitions failed: "+err.Error()),
		}, nil
	}
	if err := t.WaitToFinish(); err != nil {
		log.Error("ShowPartitions failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("ShowPartitions", metrics.FailLabel).Inc()
		return &milvuspb.ShowPartitionsResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "ShowPartitions failed: "+err.Error()),
		}, nil
	}
	metrics.RootCoordDDLReqCounter.WithLabelValues("ShowPartitions", metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues("ShowPartitions").Observe(float64(tr.ElapseSpan().Milliseconds()))
	return t.Rsp, nil
}

func (c *RootCoord) CreateAlias(ctx context.Context, in *milvuspb.CreateAliasRequest) (*commonpb.Status, error) {
	t := &createAliasTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) DropAlias(ctx context.Context, in *milvuspb.DropAliasRequest) (*commonpb.Status, error) {
	t := &dropAliasTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) AlterAlias(ctx context.Context, in *milvuspb.AlterAliasRequest) (*commonpb.Status, error) {
	t := &alterAliasTask{
		baseTaskV2: baseTaskV2{
			core: c,
			done: make(chan error, 1),
		},
		Req: in,
	}
	if err := c.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) AllocTimestamp(ctx context.Context, in *rootcoordpb.AllocTimestampRequest) (*rootcoordpb.AllocTimestampResponse, error) {
	if code, ok := c.checkHealthy(); !ok {
		return &rootcoordpb.AllocTimestampResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	ts, err := c.tsoAllocator.GenerateTSO(in.GetCount())
	if err != nil {
		log.Error("AllocTimestamp failed", zap.String("role", typeutil.RootCoordRole),
			zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		return &rootcoordpb.AllocTimestampResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "AllocTimestamp failed: "+err.Error()),
		}, nil
	}

	//return first available  time stamp
	ts = ts - uint64(in.Count) + 1
	metrics.RootCoordTimestamp.Set(float64(ts))
	return &rootcoordpb.AllocTimestampResponse{
		Status:    succStatus(),
		Timestamp: ts,
		Count:     in.Count,
	}, nil
}

func (c *RootCoord) AllocID(ctx context.Context, in *rootcoordpb.AllocIDRequest) (*rootcoordpb.AllocIDResponse, error) {
	if code, ok := c.checkHealthy(); !ok {
		return &rootcoordpb.AllocIDResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	start, _, err := c.idAllocator.Alloc(in.Count)
	if err != nil {
		log.Error("AllocID failed", zap.String("role", typeutil.RootCoordRole),
			zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		return &rootcoordpb.AllocIDResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "AllocID failed: "+err.Error()),
			Count:  in.Count,
		}, nil
	}
	metrics.RootCoordIDAllocCounter.Add(float64(in.Count))
	return &rootcoordpb.AllocIDResponse{
		Status: succStatus(),
		ID:     start,
		Count:  in.Count,
	}, nil
}

func (c *RootCoord) UpdateChannelTimeTick(ctx context.Context, in *internalpb.ChannelTimeTickMsg) (*commonpb.Status, error) {
	if code, ok := c.checkHealthy(); !ok {
		log.Warn("failed to updateTimeTick because rootcoord is not healthy", zap.Any("state", code))
		return failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]), nil
	}
	if in.Base.MsgType != commonpb.MsgType_TimeTick {
		log.Warn("failed to updateTimeTick because base messasge is not timetick, state", zap.Any("base message type", in.Base.MsgType))
		msgTypeName := commonpb.MsgType_name[int32(in.Base.GetMsgType())]
		return failStatus(commonpb.ErrorCode_UnexpectedError, "invalid message type "+msgTypeName), nil
	}
	err := c.chanTimeTick.updateTimeTick(in, "gRPC")
	if err != nil {
		log.Warn("failed to updateTimeTick", zap.String("role", typeutil.RootCoordRole),
			zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		return failStatus(commonpb.ErrorCode_UnexpectedError, "UpdateTimeTick failed: "+err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) ReleaseDQLMessageStream(ctx context.Context, in *proxypb.ReleaseDQLMessageStreamRequest) (*commonpb.Status, error) {
	if code, ok := c.checkHealthy(); !ok {
		return failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]), nil
	}
	err := c.proxyClientManager.ReleaseDQLMessageStream(ctx, in)
	if err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (c *RootCoord) InvalidateCollectionMetaCache(ctx context.Context, in *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
	if code, ok := c.checkHealthy(); !ok {
		return failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]), nil
	}
	err := c.proxyClientManager.InvalidateCollectionMetaCache(ctx, in)
	if err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}
