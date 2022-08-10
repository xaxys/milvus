package rootcoord

import (
	"context"
	"sync"

	kvmetestore "github.com/milvus-io/milvus/internal/metastore/kv"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	// DDLLogPrefix prefix for DDL log
	DDLLogPrefix = kvmetestore.ComponentPrefix + "/ddl-log"
)

type IMetaTableV2 interface {
	AddCollection(ctx context.Context, coll *model.Collection) error
	GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error)
	DeleteCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error
	IsAlias(name string) bool
}

type MetaTableV2 struct {
	txn      kv.TxnKV      // client of a reliable txnkv service, i.e. etcd client
	snapshot kv.SnapShotKV // client of a reliable snapshotkv service, i.e. etcd client
	catalog  metastore.Catalog

	collID2Meta  map[typeutil.UniqueID]model.Collection // collection id -> collection meta
	collName2ID  map[string]typeutil.UniqueID           // collection name to collection id
	collAlias2ID map[string]typeutil.UniqueID           // collection alias to collection id

	ddLock sync.RWMutex
}

func (m *MetaTableV2) AddCollection(ctx context.Context, coll *model.Collection) error {
	return nil
}

func (m *MetaTableV2) GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
	return nil, nil
}

func (m *MetaTableV2) DeleteCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
	return nil
}

func (m *MetaTableV2) IsAlias(name string) bool {
	return false
}
