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

	ms "github.com/milvus-io/milvus/internal/mq/msgstream"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/proto/querypb"

	"github.com/milvus-io/milvus/internal/util/funcutil"

	"github.com/milvus-io/milvus/internal/proto/schemapb"

	"github.com/milvus-io/milvus/internal/metastore/model"

	"github.com/milvus-io/milvus/internal/util/timerecord"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/common"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	kvmetestore "github.com/milvus-io/milvus/internal/metastore/kv"

	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/metrics"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/sessionutil"

	"github.com/milvus-io/milvus/internal/util/metricsinfo"

	"github.com/milvus-io/milvus/internal/proto/datapb"
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

type RootCoord struct {
	types.RootCoord // TODO: remove me after everything is ready.

	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	etcdCli   *clientv3.Client
	meta      IMetaTableV2
	scheduler IScheduler

	txn          kv.TxnKV
	kvBaseCreate func(root string) (kv.TxnKV, error)
	metaKVCreate func(root string) (kv.MetaKv, error)

	NewProxyClient      func(sess *sessionutil.Session) (types.Proxy, error)
	proxyManager        *proxyManager
	proxyClientManager  *proxyClientManager
	metricsCacheManager *metricsinfo.MetricsCacheManager

	chanTimeTick *timetickSync

	idAllocator  allocator.GIDAllocator
	tsoAllocator tso.Allocator

	dataCoord           types.DataCoord
	CallUnwatchChannels func(ctx context.Context, collectionID UniqueID, vChannels []string) error
	queryCoord          types.QueryCoord
	indexCoord          types.IndexCoord

	stateCode atomic.Value
	initOnce  sync.Once
	startOnce sync.Once
	session   *sessionutil.Session
	factory   dependency.Factory
}

func NewRootCoord(ctx context.Context, factory dependency.Factory) (*RootCoord, error) {
	ctx1, cancel := context.WithCancel(ctx)
	rand.Seed(time.Now().UnixNano())
	core := &RootCoord{
		ctx:     ctx1,
		cancel:  cancel,
		factory: factory,
	}
	core.UpdateStateCode(internalpb.StateCode_Abnormal)
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
			// Server is closed and it should return nil.
			log.Debug("tsLoop is closed")
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
			log.Debug("rootcoord context closed", zap.Error(c.ctx.Err()))
			return
		case <-ticker.C:
			if ts, err := c.tsoAllocator.GenerateTSO(1); err == nil {
				err := c.sendTimeTick(ts, "timetick loop")
				if err != nil {
					log.Warn("Failed to send timetick", zap.Error(err))
				}
			}
		}
	}
}

func (c *RootCoord) SetNewProxyClient(f func(sess *sessionutil.Session) (types.Proxy, error)) {
	c.NewProxyClient = f
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

func (c *RootCoord) Init() error {
	var initError error
	if c.kvBaseCreate == nil {
		c.kvBaseCreate = func(root string) (kv.TxnKV, error) {
			return etcdkv.NewEtcdKV(c.etcdCli, root), nil
		}
	}
	if c.metaKVCreate == nil {
		c.metaKVCreate = func(root string) (kv.MetaKv, error) {
			return etcdkv.NewEtcdKV(c.etcdCli, root), nil
		}
	}
	c.initOnce.Do(func() {
		if err := c.initSession(); err != nil {
			initError = err
			log.Error("RootCoord init session failed", zap.Error(err))
			return
		}
		connectEtcdFn := func() error {
			if c.txn, initError = c.kvBaseCreate(Params.EtcdCfg.KvRootPath); initError != nil {
				log.Error("RootCoord failed to new EtcdKV for kvBase", zap.Any("reason", initError))
				return initError
			}
			var metaKV kv.TxnKV
			metaKV, initError = c.kvBaseCreate(Params.EtcdCfg.MetaRootPath)
			if initError != nil {
				log.Error("RootCoord failed to new EtcdKV", zap.Any("reason", initError))
				return initError
			}

			var ss *kvmetestore.SuffixSnapshot
			if ss, initError = kvmetestore.NewSuffixSnapshot(metaKV, "_ts", Params.EtcdCfg.MetaRootPath, "snapshots"); initError != nil {
				log.Error("RootCoord failed to new suffixSnapshot", zap.Error(initError))
				return initError
			}
			if c.meta, initError = newMetaTableV2(c.ctx, metaKV, ss); initError != nil {
				log.Error("RootCoord failed to new MetaTable", zap.Any("reason", initError))
				return initError
			}

			return nil
		}
		log.Debug("RootCoord, Connecting to Etcd", zap.String("kv root", Params.EtcdCfg.KvRootPath), zap.String("meta root", Params.EtcdCfg.MetaRootPath))
		err := retry.Do(c.ctx, connectEtcdFn, retry.Attempts(100))
		if err != nil {
			return
		}

		log.Debug("RootCoord, Setting TSO and ID Allocator")
		kv := tsoutil.NewTSOKVBase(c.etcdCli, Params.EtcdCfg.KvRootPath, "gid")
		idAllocator := allocator.NewGlobalIDAllocator("idTimestamp", kv)
		if initError = idAllocator.Initialize(); initError != nil {
			return
		}
		c.idAllocator = idAllocator

		kv = tsoutil.NewTSOKVBase(c.etcdCli, Params.EtcdCfg.KvRootPath, "tso")
		tsoAllocator := tso.NewGlobalTSOAllocator("timestamp", kv)
		if initError = tsoAllocator.Initialize(); initError != nil {
			return
		}
		c.tsoAllocator = tsoAllocator

		c.scheduler = newScheduler(c.ctx, c.idAllocator, c.tsoAllocator)

		c.factory.Init(&Params)

		// TODO
		chanMap := map[UniqueID][]string{}
		c.chanTimeTick = newTimeTickSync(c.ctx, c.session.ServerID, c.factory, chanMap)
		c.chanTimeTick.addSession(c.session)
		c.proxyClientManager = newProxyClientManager(c.NewProxyClient)

		log.Debug("RootCoord, set proxy manager")
		c.proxyManager = newProxyManager(
			c.ctx,
			c.etcdCli,
			c.chanTimeTick.initSessions,
			c.proxyClientManager.GetProxyClients,
		)
		c.proxyManager.AddSessionFunc(c.chanTimeTick.addSession, c.proxyClientManager.AddProxyClient)
		c.proxyManager.DelSessionFunc(c.chanTimeTick.delSession, c.proxyClientManager.DelProxyClient)

		c.metricsCacheManager = metricsinfo.NewMetricsCacheManager()

		// init data
		initError = c.initData()
		if initError != nil {
			return
		}

		if initError = c.initRbac(); initError != nil {
			return
		}
		log.Debug("RootCoord init user root done")
	})
	if initError != nil {
		log.Debug("RootCoord init error", zap.Error(initError))
	}
	log.Debug("RootCoord init done")
	return initError
}

func (c *RootCoord) initData() error {
	// TODO: implement me.
	return nil
}

func (c *RootCoord) initRbac() (initError error) {
	// TODO: implement me.
	return nil
}

func (c *RootCoord) reDropCollection(collMeta *model.Collection, ts Timestamp) {
	// TODO: remove this after data gc can be notified by rpc.
	c.chanTimeTick.addDmlChannels(collMeta.PhysicalChannelNames...)
	defer c.chanTimeTick.removeDmlChannels(collMeta.PhysicalChannelNames...)
	// no need to add delta channels, since this collection should be dropped, so it's ok to let tSafe not updated.

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	if err := c.releaseCollection(ctx, collMeta.CollectionID); err != nil {
		log.Error("failed to release collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
		return
	}

	if err := c.notifyDataGC(ctx, collMeta, ts); err != nil {
		log.Error("failed to notify datacoord to gc collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
		return
	}

	if err := c.meta.RemoveCollection(ctx, collMeta.CollectionID, ts); err != nil {
		log.Error("failed to remove collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
	}
}

func (c *RootCoord) removeCreatingCollection(collMeta *model.Collection) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	if err := c.unwatchChannels(ctx, collMeta.CollectionID, collMeta.VirtualChannelNames); err != nil {
		log.Error("failed to unwatch channels when recovery",
			zap.Error(err),
			zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID),
			zap.Strings("vchans", collMeta.VirtualChannelNames), zap.Strings("pchans", collMeta.PhysicalChannelNames))
		return
	}

	if err := c.meta.RemoveCollection(ctx, collMeta.CollectionID, collMeta.CreateTime); err != nil {
		log.Error("failed to remove collection when recovery", zap.Error(err), zap.String("collection", collMeta.Name), zap.Int64("collection id", collMeta.CollectionID))
	}
}

func (c *RootCoord) reDropPartition(partition *model.Partition, ts Timestamp) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	// TODO: release partition when query coord is ready.
	// TODO: notify datacoord to gc partition data when it's ready.
	if err := c.meta.RemovePartition(ctx, partition.CollectionID, partition.PartitionID, ts); err != nil {
		log.Error("failed to remove partition when recovery", zap.Error(err))
	}
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
			go c.reDropCollection(coll.Clone(), ts)
		case pb.CollectionState_CollectionCreating:
			go c.removeCreatingCollection(coll.Clone())
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
				go c.reDropPartition(part.Clone(), ts)
			default:
			}
		}
	}
	return nil
}

// Start starts RootCoord.
func (c *RootCoord) Start() error {
	log.Debug("starting service",
		zap.String("service role", typeutil.RootCoordRole),
		zap.Int64("node id", c.session.ServerID))

	c.startOnce.Do(func() {
		if err := c.proxyManager.WatchProxy(); err != nil {
			log.Fatal("RootCoord Start WatchProxy failed", zap.Error(err))
			// you can not just stuck here,
			panic(err)
		}
		if err := c.restore(c.ctx); err != nil {
			panic(err)
		}
		c.wg.Add(3)
		go c.tsLoop()
		go c.startTimeTickLoop()
		go c.chanTimeTick.startWatch(&c.wg)
		c.scheduler.Start()
		Params.RootCoordCfg.CreatedTime = time.Now()
		Params.RootCoordCfg.UpdatedTime = time.Now()
	})

	return nil
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
	log.Debug("GetComponentStates", zap.String("State Code", internalpb.StateCode_name[int32(code)]))

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

func (c *RootCoord) watchChannels(ctx context.Context, collectionID UniqueID, vChannels []string) error {
	if err := funcutil.WaitForComponentHealthy(ctx, c.dataCoord, "DataCoord", 100, time.Millisecond*200); err != nil {
		return err
	}
	resp, err := c.dataCoord.WatchChannels(ctx, &datapb.WatchChannelsRequest{
		CollectionID: collectionID,
		ChannelNames: vChannels,
	})
	if err != nil {
		return err
	}
	if resp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("failed to watch channels, code: %s, reason: %s", resp.GetStatus().GetErrorCode(), resp.GetStatus().GetReason())
	}
	return nil
}

func (c *RootCoord) unwatchChannels(ctx context.Context, collectionID UniqueID, vChannels []string) error {
	// TODO: unwatching channels to release flow graph on datanodes.
	if c.CallUnwatchChannels != nil {
		return c.CallUnwatchChannels(ctx, collectionID, vChannels)
	}
	return nil
}

func (c *RootCoord) notifyDataGC(ctx context.Context, coll *model.Collection, ts typeutil.Timestamp) error {
	msgPack := ms.MsgPack{}
	baseMsg := ms.BaseMsg{
		Ctx:            ctx,
		BeginTimestamp: ts,
		EndTimestamp:   ts,
		HashValues:     []uint32{0},
	}
	msg := &ms.DropCollectionMsg{
		BaseMsg: baseMsg,
		DropCollectionRequest: internalpb.DropCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_DropCollection,
				Timestamp: ts,
				SourceID:  c.session.ServerID,
			},
			CollectionName: coll.Name,
			CollectionID:   coll.CollectionID,
		},
	}
	msgPack.Msgs = append(msgPack.Msgs, msg)
	return c.chanTimeTick.broadcastDmlChannels(coll.PhysicalChannelNames, &msgPack)
}

func (c *RootCoord) releaseCollection(ctx context.Context, collectionID UniqueID) error {
	if err := funcutil.WaitForComponentHealthy(ctx, c.queryCoord, "QueryCoord", 100, time.Millisecond*200); err != nil {
		return err
	}
	resp, err := c.queryCoord.ReleaseCollection(ctx, &querypb.ReleaseCollectionRequest{
		Base:         &commonpb.MsgBase{MsgType: commonpb.MsgType_ReleaseCollection},
		CollectionID: collectionID,
		NodeID:       c.session.ServerID,
	})
	if err != nil {
		return err
	}
	if resp.GetErrorCode() != commonpb.ErrorCode_Success {
		return fmt.Errorf("failed to release collection, code: %s, reason: %s", resp.GetErrorCode(), resp.GetReason())
	}
	return nil
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

func (c *RootCoord) describeCollection(ctx context.Context, in *milvuspb.DescribeCollectionRequest) (*milvuspb.DescribeCollectionResponse, error) {
	var collInfo *model.Collection
	var err error
	if in.GetCollectionName() != "" {
		collInfo, err = c.meta.GetCollectionByName(ctx, in.GetCollectionName(), in.GetTimeStamp())
	} else {
		collInfo, err = c.meta.GetCollectionByID(ctx, in.GetCollectionID(), in.GetTimeStamp())
	}
	if err != nil {
		return nil, err
	}
	ret := &milvuspb.DescribeCollectionResponse{}
	ret.Schema = &schemapb.CollectionSchema{
		Name:        collInfo.Name,
		Description: collInfo.Description,
		AutoID:      collInfo.AutoID,
		Fields:      model.MarshalFieldModels(collInfo.Fields),
	}
	ret.CollectionID = collInfo.CollectionID
	ret.VirtualChannelNames = collInfo.VirtualChannelNames
	ret.PhysicalChannelNames = collInfo.PhysicalChannelNames
	if collInfo.ShardsNum == 0 {
		collInfo.ShardsNum = int32(len(collInfo.VirtualChannelNames))
	}
	ret.ShardsNum = collInfo.ShardsNum
	ret.ConsistencyLevel = collInfo.ConsistencyLevel

	ret.CreatedTimestamp = collInfo.CreateTime
	createdPhysicalTime, _ := tsoutil.ParseHybridTs(collInfo.CreateTime)
	ret.CreatedUtcTimestamp = uint64(createdPhysicalTime)
	ret.Aliases = nil // TODO: not sure if this is reasonable.
	ret.StartPositions = collInfo.StartPositions
	ret.CollectionName = collInfo.Name
	return ret, nil
}

func (c *RootCoord) DescribeCollection(ctx context.Context, in *milvuspb.DescribeCollectionRequest) (*milvuspb.DescribeCollectionResponse, error) {
	metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.TotalLabel).Inc()
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.DescribeCollectionResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode"+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	tr := timerecord.NewTimeRecorder("DescribeCollection")

	rsp := &milvuspb.DescribeCollectionResponse{}
	rsp, err := c.describeCollection(ctx, in)
	if err != nil {
		log.Error("DescribeCollection failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("collection name", in.CollectionName), zap.Int64("id", in.CollectionID), zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.FailLabel).Inc()
		return &milvuspb.DescribeCollectionResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "DescribeCollection failed: "+err.Error()),
		}, nil
	}

	metrics.RootCoordDDLReqCounter.WithLabelValues("DescribeCollection", metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues("DescribeCollection").Observe(float64(tr.ElapseSpan().Milliseconds()))
	rsp.Status = succStatus()
	return rsp, nil
}

func (c *RootCoord) HasCollection(ctx context.Context, in *milvuspb.HasCollectionRequest) (*milvuspb.BoolResponse, error) {
	resp := &milvuspb.BoolResponse{Status: succStatus(), Value: true}
	_, err := c.meta.GetCollectionByName(ctx, in.GetCollectionName(), in.GetTimeStamp())
	if err != nil {
		resp.Value = false
	}
	return resp, nil
}

func (c *RootCoord) ShowCollections(ctx context.Context, in *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error) {
	resp := &milvuspb.ShowCollectionsResponse{Status: succStatus()}
	ts := in.GetTimeStamp()
	if ts == 0 {
		ts = typeutil.MaxTimestamp
	}
	colls, err := c.meta.ListCollections(ctx, ts)
	if err != nil {
		resp.Status = failStatus(commonpb.ErrorCode_UnexpectedError, err.Error())
	}
	for _, coll := range colls {
		resp.CollectionNames = append(resp.CollectionNames, coll.Name)
		resp.CollectionIds = append(resp.CollectionIds, coll.CollectionID)
		resp.CreatedTimestamps = append(resp.CreatedTimestamps, coll.CreateTime)
		physical, _ := tsoutil.ParseHybridTs(coll.CreateTime)
		resp.CreatedUtcTimestamps = append(resp.CreatedUtcTimestamps, uint64(physical))
	}
	return resp, nil
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
	resp := &milvuspb.BoolResponse{Status: succStatus(), Value: true}
	coll, err := c.meta.GetCollectionByName(ctx, in.GetCollectionName(), typeutil.MaxTimestamp)
	if err != nil {
		resp.Status = failStatus(commonpb.ErrorCode_CollectionNotExists, err.Error())
		resp.Value = false
	} else {
		resp.Value = false
		for _, partition := range coll.Partitions {
			if partition.PartitionName == in.GetPartitionName() {
				resp.Value = true
				break
			}
		}
	}
	return resp, nil
}

func (c *RootCoord) ShowPartitions(ctx context.Context, in *milvuspb.ShowPartitionsRequest) (*milvuspb.ShowPartitionsResponse, error) {
	resp := &milvuspb.ShowPartitionsResponse{Status: succStatus()}
	var coll *model.Collection
	var err error
	if in.GetCollectionName() != "" {
		coll, err = c.meta.GetCollectionByName(ctx, in.GetCollectionName(), typeutil.MaxTimestamp)
	} else {
		coll, err = c.meta.GetCollectionByID(ctx, in.GetCollectionID(), typeutil.MaxTimestamp)
	}
	if err != nil {
		resp.Status = failStatus(commonpb.ErrorCode_CollectionNotExists, err.Error())
	} else {
		for _, partition := range coll.Partitions {
			if !partition.Available() {
				continue
			}
			resp.PartitionIDs = append(resp.PartitionIDs, partition.PartitionID)
			resp.PartitionNames = append(resp.PartitionNames, partition.PartitionName)
			resp.CreatedTimestamps = append(resp.CreatedTimestamps, partition.PartitionCreatedTimestamp)
			physical, _ := tsoutil.ParseHybridTs(partition.PartitionCreatedTimestamp)
			resp.CreatedUtcTimestamps = append(resp.CreatedUtcTimestamps, uint64(physical))
		}
	}
	return resp, nil
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

func (c *RootCoord) ListPolicy(ctx context.Context, in *internalpb.ListPolicyRequest) (*internalpb.ListPolicyResponse, error) {
	return &internalpb.ListPolicyResponse{
		Status:      succStatus(),
		PolicyInfos: nil,
		UserRoles:   nil,
	}, nil
}
