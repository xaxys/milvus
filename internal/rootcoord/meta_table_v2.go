package rootcoord

import (
	"context"
	"fmt"
	"sync"

	"github.com/milvus-io/milvus/internal/util/funcutil"

	"github.com/milvus-io/milvus/internal/proto/schemapb"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	kvmetestore "github.com/milvus-io/milvus/internal/metastore/kv"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	// DDLLogPrefix prefix for DDL log
	DDLLogPrefix = kvmetestore.ComponentPrefix + "/ddl-log"
	maxTxnNum    = 64
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type IMetaTableV2 interface {
	AddCreatingCollection(ctx context.Context, coll *model.Collection) error
	SaveCollectionOnly(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error
	RemoveCollectionOnly(ctx context.Context, collectionID UniqueID, ts Timestamp) error
	GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error)
	GetCollectionByID(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error)
	ListCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error)
	AddCreatingPartition(ctx context.Context, partition *model.Partition) error
	SavePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error
	RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error
	SaveFields(ctx context.Context, collectionID UniqueID, fieldIds []UniqueID, state schemapb.FieldState, ts Timestamp) error
	RemoveFields(ctx context.Context, collectionID UniqueID, fieldIds []UniqueID, ts Timestamp) error
	CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	DropAlias(ctx context.Context, alias string, ts Timestamp) error
	AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	IsAlias(name string) bool
}

type MetaTableV2 struct {
	ctx      context.Context
	txn      kv.TxnKV      // client of a reliable txnkv service, i.e. etcd client
	snapshot kv.SnapShotKV // client of a reliable snapshotkv service, i.e. etcd client
	catalog  metastore.Catalog

	collID2Meta  map[typeutil.UniqueID]*model.Collection // collection id -> collection meta
	collName2ID  map[string]typeutil.UniqueID            // collection name to collection id
	collAlias2ID map[string]typeutil.UniqueID            // collection alias to collection id

	ddLock sync.RWMutex
}

func newMetaTableV2(ctx context.Context, txn kv.TxnKV, snapshot kv.SnapShotKV) (*MetaTableV2, error) {
	m := &MetaTableV2{
		ctx:      ctx,
		txn:      txn,
		snapshot: snapshot,
		catalog:  &kvmetestore.Catalog{Txn: txn, Snapshot: snapshot},
	}
	if err := m.reload(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *MetaTableV2) reload() error {
	m.collID2Meta = make(map[UniqueID]*model.Collection)
	m.collName2ID = make(map[string]UniqueID)
	m.collAlias2ID = make(map[string]UniqueID)

	// max ts means listing latest resources.
	collections, err := m.catalog.ListCollections(m.ctx, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	for name, collection := range collections {
		m.collID2Meta[collection.CollectionID] = collection
		m.collName2ID[name] = collection.CollectionID
	}

	aliases, err := m.catalog.ListAliases(m.ctx, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	for _, alias := range aliases {
		m.collAlias2ID[alias.Name] = alias.CollectionID
	}

	return nil
}

func (m *MetaTableV2) AddCreatingCollection(ctx context.Context, coll *model.Collection) error {
	if coll.State != pb.CollectionState_CollectionCreating {
		return fmt.Errorf("collection state should be creating, collection name: %s, collection id: %d, state: %s", coll.Name, coll.CollectionID, coll.State)
	}
	m.collID2Meta[coll.CollectionID] = coll.Clone()
	m.collName2ID[coll.Name] = coll.CollectionID
	return nil
}

func (m *MetaTableV2) SaveCollectionOnly(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error {
	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	clone := coll.Clone()
	clone.State = state
	if err := m.catalog.CreateCollection(ctx, clone, ts); err != nil {
		return err
	}
	m.collID2Meta[collectionID] = clone
	return nil
}

func (m *MetaTableV2) RemoveCollectionOnly(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
	if err := m.catalog.DropCollectionOnly(ctx, collectionID, ts); err != nil {
		return err
	}
	delete(m.collID2Meta, collectionID)
	return nil
}

func (m *MetaTableV2) GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
	var collectionID UniqueID
	collectionID, ok := m.collName2ID[collectionName]
	if ok {
		return m.GetCollectionByID(ctx, collectionID, ts)
	}
	collectionID, ok = m.collAlias2ID[collectionName]
	if ok {
		return m.GetCollectionByID(ctx, collectionID, ts)
	}
	return nil, fmt.Errorf("collection not exist: %s", collectionName)
}

func (m *MetaTableV2) GetCollectionByID(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
	coll, ok := m.collID2Meta[collectionID]
	if !ok || !coll.Available() {
		return nil, fmt.Errorf("collection not exist: %d", collectionID)
	}
	clone := coll.Clone()
	// remove not available resources.
	toRemovedPartitionIndexes := make([]int, 0, len(clone.Partitions))
	for i := len(clone.Partitions) - 1; i >= 0; i-- {
		if !clone.Partitions[i].Available() {
			toRemovedPartitionIndexes = append(toRemovedPartitionIndexes, i)
		}
	}
	for _, loc := range toRemovedPartitionIndexes {
		coll.Partitions = append(coll.Partitions[:loc], coll.Partitions[loc+1:]...)
	}
	return clone, nil
}

func (m *MetaTableV2) ListCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error) {
	colls, err := m.catalog.ListCollections(ctx, ts)
	if err != nil {
		return nil, err
	}
	onlineCollections := make([]*model.Collection, 0, len(colls))
	for _, coll := range colls {
		if coll.Available() {
			onlineCollections = append(onlineCollections, coll)
		}
	}
	return onlineCollections, nil
}

func (m *MetaTableV2) AddCreatingPartition(ctx context.Context, partition *model.Partition) error {
	if partition.State != pb.PartitionState_PartitionCreating {
		return fmt.Errorf("partition state is not creating, collection: %d, partition: %d, state: %s", partition.CollectionID, partition.PartitionID, partition.State)
	}
	_, ok := m.collID2Meta[partition.CollectionID]
	if !ok {
		return fmt.Errorf("collection not exists: %d", partition.CollectionID)
	}
	m.collID2Meta[partition.CollectionID].Partitions = append(m.collID2Meta[partition.CollectionID].Partitions, partition.Clone())
	return nil
}

func (m *MetaTableV2) SavePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error {
	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	for idx, part := range coll.Partitions {
		if part.PartitionID == partitionID {
			clone := part.Clone()
			clone.State = state
			if err := m.catalog.CreatePartition(ctx, clone, ts); err != nil {
				return err
			}
			coll.Partitions[idx] = clone
			return nil
		}
	}
	return fmt.Errorf("partition not exist, collection: %d, partition: %d", collectionID, partitionID)
}

func (m *MetaTableV2) RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error {
	if err := m.catalog.DropPartition(ctx, collectionID, partitionID, ts); err != nil {
		return err
	}
	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	var loc = -1
	for idx, part := range coll.Partitions {
		if part.PartitionID == partitionID {
			loc = idx
			break
		}
	}
	if loc != -1 {
		coll.Partitions = append(coll.Partitions[:loc], coll.Partitions[loc+1:]...)
	}
	return nil
}

func (m *MetaTableV2) SaveFields(ctx context.Context, collectionID UniqueID, fieldIds []UniqueID, state schemapb.FieldState, ts Timestamp) error {
	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return fmt.Errorf("collection not exist: %d", collectionID)
	}
	toSavedFieldIndexes := make([]int, 0, len(fieldIds))
	toSavedFields := make([]*model.Field, 0, len(fieldIds))
	for idx, field := range coll.Fields {
		if funcutil.SliceContain(fieldIds, field.FieldID) {
			toSavedFieldIndexes = append(toSavedFieldIndexes, idx)
			clone := field.Clone()
			clone.State = state
			toSavedFields = append(toSavedFields, clone)
		}
	}
	if err := m.catalog.AddFields(ctx, collectionID, toSavedFields, ts); err != nil {
		return err
	}
	for idx, fieldIndex := range toSavedFieldIndexes {
		coll.Fields[fieldIndex] = toSavedFields[idx]
	}
	return nil
}

func (m *MetaTableV2) RemoveFields(ctx context.Context, collectionID UniqueID, fieldIds []UniqueID, ts Timestamp) error {
	if err := m.catalog.RemoveFields(ctx, collectionID, fieldIds, ts); err != nil {
		return err
	}
	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	toSavedFieldIndexes := make([]int, 0, len(fieldIds))
	for idx := len(coll.Fields) - 1; idx >= 0; idx-- {
		if funcutil.SliceContain(fieldIds, coll.Fields[idx].FieldID) {
			toSavedFieldIndexes = append(toSavedFieldIndexes, idx)
		}
	}
	for _, fieldIndex := range toSavedFieldIndexes {
		coll.Fields = append(coll.Fields[:fieldIndex], coll.Fields[fieldIndex+1:]...)
	}
	return nil
}

func (m *MetaTableV2) CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	collectionID, ok := m.collName2ID[collectionName]
	if !ok {
		return fmt.Errorf("collection not exists: %s", collectionName)
	}
	if err := m.catalog.CreateAlias(ctx, &model.Alias{
		Name:         alias,
		CollectionID: collectionID,
		CreatedTime:  ts,
		State:        pb.AliasState_AliasCreated,
	}, ts); err != nil {
		return err
	}
	m.collAlias2ID[alias] = collectionID
	return nil
}

func (m *MetaTableV2) DropAlias(ctx context.Context, alias string, ts Timestamp) error {
	if err := m.catalog.DropAlias(ctx, alias, ts); err != nil {
		return err
	}
	delete(m.collAlias2ID, alias)
	return nil
}

func (m *MetaTableV2) AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	collectionID, ok := m.collName2ID[collectionName]
	if !ok {
		return fmt.Errorf("collection not exists: %s", collectionName)
	}
	if err := m.catalog.AlterAlias(ctx, &model.Alias{
		Name:         alias,
		CollectionID: collectionID,
		CreatedTime:  ts,
		State:        pb.AliasState_AliasCreated,
	}, ts); err != nil {
		return err
	}
	m.collAlias2ID[alias] = collectionID
	return nil
}

func (m *MetaTableV2) IsAlias(name string) bool {
	_, ok := m.collAlias2ID[name]
	return ok
}
