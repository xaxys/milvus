package rootcoord

import (
	"context"
	"fmt"
	"sync"

	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"

	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

	"github.com/milvus-io/milvus/internal/metastore"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	maxTxnNum = 64
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type IMetaTableV2 interface {
	AddCollection(ctx context.Context, coll *model.Collection) error
	ChangeCollectionState(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error
	RemoveCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error
	GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error)
	GetCollectionByID(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error)
	ListCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error)
	ListAbnormalCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error)
	AddPartition(ctx context.Context, partition *model.Partition) error
	ChangePartitionState(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error
	RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error
	CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	DropAlias(ctx context.Context, alias string, ts Timestamp) error
	AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	IsAlias(name string) bool
	GetCollectionNameByID(collID UniqueID) (string, error)
	GetPartitionNameByID(collID UniqueID, partitionID UniqueID, ts Timestamp) (string, error)
}

type MetaTableV2 struct {
	ctx     context.Context
	catalog metastore.RootCoordCatalog

	collID2Meta  map[typeutil.UniqueID]*model.Collection // collection id -> collection meta
	collName2ID  map[string]typeutil.UniqueID            // collection name to collection id
	collAlias2ID map[string]typeutil.UniqueID            // collection alias to collection id

	ddLock sync.RWMutex
}

func newMetaTableV2(ctx context.Context, catalog metastore.RootCoordCatalog) (*MetaTableV2, error) {
	m := &MetaTableV2{
		ctx:     ctx,
		catalog: catalog,
	}
	if err := m.reload(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *MetaTableV2) reload() error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

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

func (m *MetaTableV2) AddCollection(ctx context.Context, coll *model.Collection) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

	if coll.State != pb.CollectionState_CollectionCreating {
		return fmt.Errorf("collection state should be creating, collection name: %s, collection id: %d, state: %s", coll.Name, coll.CollectionID, coll.State)
	}
	if err := m.catalog.CreateCollection(ctx, coll, coll.CreateTime); err != nil {
		return err
	}
	m.collName2ID[coll.Name] = coll.CollectionID
	m.collID2Meta[coll.CollectionID] = coll
	log.Info("add collection to meta table", zap.String("collection", coll.Name), zap.Int64("id", coll.CollectionID), zap.Uint64("ts", coll.CreateTime))
	return nil
}

func (m *MetaTableV2) ChangeCollectionState(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	clone := coll.Clone()
	clone.State = state
	if err := m.catalog.AlterCollection(ctx, coll, clone, metastore.MODIFY, ts); err != nil {
		return err
	}
	m.collID2Meta[collectionID] = clone
	log.Info("change collection state", zap.Int64("collection", collectionID), zap.String("state", state.String()), zap.Uint64("ts", ts))

	return nil
}

func (m *MetaTableV2) RemoveCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

	if err := m.catalog.DropCollection(ctx, &model.Collection{CollectionID: collectionID}, ts); err != nil {
		return err
	}
	delete(m.collID2Meta, collectionID)
	return nil
}

func (m *MetaTableV2) GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
	m.ddLock.RLock()
	defer m.ddLock.RUnlock()

	var collectionID UniqueID
	collectionID, ok := m.collAlias2ID[collectionName]
	if ok {
		return m.GetCollectionByID(ctx, collectionID, ts)
	}
	collectionID, ok = m.collName2ID[collectionName]
	if ok {
		return m.GetCollectionByID(ctx, collectionID, ts)
	}
	// travel meta information from catalog.
	return m.catalog.GetCollectionByName(ctx, collectionName, ts)
}

func (m *MetaTableV2) GetCollectionByID(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
	m.ddLock.RLock()
	defer m.ddLock.RUnlock()

	coll, ok := m.collID2Meta[collectionID]
	if !ok || !coll.Available() || coll.CreateTime > ts {
		// travel meta information from catalog.
		return m.catalog.GetCollectionByID(ctx, collectionID, ts)
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
	m.ddLock.RLock()
	defer m.ddLock.RUnlock()

	// list collections should always be loaded from catalog.
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

func (m *MetaTableV2) ListAbnormalCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error) {
	m.ddLock.RLock()
	defer m.ddLock.RUnlock()

	// list collections should always be loaded from catalog.
	colls, err := m.catalog.ListCollections(ctx, ts)
	if err != nil {
		return nil, err
	}
	abnormalCollections := make([]*model.Collection, 0, len(colls))
	for _, coll := range colls {
		if !coll.Available() {
			abnormalCollections = append(abnormalCollections, coll)
		}
	}
	return abnormalCollections, nil
}

func (m *MetaTableV2) AddPartition(ctx context.Context, partition *model.Partition) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

	_, ok := m.collID2Meta[partition.CollectionID]
	if !ok {
		return fmt.Errorf("collection not exists: %d", partition.CollectionID)
	}
	if partition.State != pb.PartitionState_PartitionCreated {
		return fmt.Errorf("partition state is not created, collection: %d, partition: %d, state: %s", partition.CollectionID, partition.PartitionID, partition.State)
	}
	m.collID2Meta[partition.CollectionID].Partitions = append(m.collID2Meta[partition.CollectionID].Partitions, partition.Clone())
	return nil
}

func (m *MetaTableV2) ChangePartitionState(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

	coll, ok := m.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	for idx, part := range coll.Partitions {
		if part.PartitionID == partitionID {
			clone := part.Clone()
			clone.State = state
			if err := m.catalog.AlterPartition(ctx, part, clone, metastore.MODIFY, ts); err != nil {
				return err
			}
			coll.Partitions[idx] = clone
			return nil
		}
	}
	return fmt.Errorf("partition not exist, collection: %d, partition: %d", collectionID, partitionID)
}

func (m *MetaTableV2) RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

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

func (m *MetaTableV2) CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

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
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

	if err := m.catalog.DropAlias(ctx, alias, ts); err != nil {
		return err
	}
	delete(m.collAlias2ID, alias)
	return nil
}

func (m *MetaTableV2) AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	m.ddLock.Lock()
	defer m.ddLock.Unlock()

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
	m.ddLock.RLock()
	defer m.ddLock.RUnlock()

	_, ok := m.collAlias2ID[name]
	return ok
}

func (m *MetaTableV2) GetCollectionNameByID(collID UniqueID) (string, error) {
	panic("implement me")
}

func (m *MetaTableV2) GetPartitionNameByID(collID UniqueID, partitionID UniqueID, ts Timestamp) (string, error) {
	panic("implement me")
}
