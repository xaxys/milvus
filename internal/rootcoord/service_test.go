package rootcoord

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/tso"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/dependency"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/stretchr/testify/assert"
	clientv3 "go.etcd.io/etcd/client/v3"
)

func TestRootCoord_ShowCollections(t *testing.T) {
	type fields struct {
		RootCoord           types.RootCoord
		ctx                 context.Context
		cancel              context.CancelFunc
		wg                  sync.WaitGroup
		etcdCli             *clientv3.Client
		meta                IMetaTableV2
		scheduler           IScheduler
		txn                 kv.TxnKV
		kvBaseCreate        func(root string) (kv.TxnKV, error)
		metaKVCreate        func(root string) (kv.MetaKv, error)
		NewProxyClient      func(sess *sessionutil.Session) (types.Proxy, error)
		proxyManager        *proxyManager
		proxyClientManager  *proxyClientManager
		metricsCacheManager *metricsinfo.MetricsCacheManager
		chanTimeTick        *timetickSync
		idAllocator         allocator.GIDAllocator
		tsoAllocator        tso.Allocator
		dataCoord           types.DataCoord
		CallUnwatchChannels func(ctx context.Context, collectionID UniqueID, vChannels []string) error
		queryCoord          types.QueryCoord
		indexCoord          types.IndexCoord
		stateCode           atomic.Value
		initOnce            sync.Once
		startOnce           sync.Once
		session             *sessionutil.Session
		factory             dependency.Factory
	}
	type args struct {
		ctx context.Context
		in  *milvuspb.ShowCollectionsRequest
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *milvuspb.ShowCollectionsResponse
		wantErr assert.ErrorAssertionFunc
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &RootCoord{
				RootCoord:           tt.fields.RootCoord,
				ctx:                 tt.fields.ctx,
				cancel:              tt.fields.cancel,
				wg:                  tt.fields.wg,
				etcdCli:             tt.fields.etcdCli,
				meta:                tt.fields.meta,
				scheduler:           tt.fields.scheduler,
				txn:                 tt.fields.txn,
				kvBaseCreate:        tt.fields.kvBaseCreate,
				metaKVCreate:        tt.fields.metaKVCreate,
				NewProxyClient:      tt.fields.NewProxyClient,
				proxyManager:        tt.fields.proxyManager,
				proxyClientManager:  tt.fields.proxyClientManager,
				metricsCacheManager: tt.fields.metricsCacheManager,
				chanTimeTick:        tt.fields.chanTimeTick,
				idAllocator:         tt.fields.idAllocator,
				tsoAllocator:        tt.fields.tsoAllocator,
				dataCoord:           tt.fields.dataCoord,
				CallUnwatchChannels: tt.fields.CallUnwatchChannels,
				queryCoord:          tt.fields.queryCoord,
				indexCoord:          tt.fields.indexCoord,
				stateCode:           tt.fields.stateCode,
				initOnce:            tt.fields.initOnce,
				startOnce:           tt.fields.startOnce,
				session:             tt.fields.session,
				factory:             tt.fields.factory,
			}
			got, err := c.ShowCollections(tt.args.ctx, tt.args.in)
			if !tt.wantErr(t, err, fmt.Sprintf("ShowCollections(%v, %v)", tt.args.ctx, tt.args.in)) {
				return
			}
			assert.Equalf(t, tt.want, got, "ShowCollections(%v, %v)", tt.args.ctx, tt.args.in)
		})
	}
}
