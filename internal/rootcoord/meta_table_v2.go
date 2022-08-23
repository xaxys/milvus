package rootcoord

import (
	"context"
	"fmt"
	"sync"

	"github.com/milvus-io/milvus/internal/proto/internalpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/util/funcutil"

	"github.com/milvus-io/milvus/internal/util/contextutil"

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
	ListCollectionPhysicalChannels() map[typeutil.UniqueID][]string
	AddPartition(ctx context.Context, partition *model.Partition) error
	ChangePartitionState(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error
	RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error
	CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	DropAlias(ctx context.Context, alias string, ts Timestamp) error
	AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error
	IsAlias(name string) bool
	GetCollectionNameByID(collID UniqueID) (string, error)
	GetPartitionNameByID(collID UniqueID, partitionID UniqueID, ts Timestamp) (string, error)

	// TODO: better to accept ctx.
	AddCredential(credInfo *internalpb.CredentialInfo) error
	GetCredential(username string) (*internalpb.CredentialInfo, error)
	DeleteCredential(username string) error
	ListCredentialUsernames() (*milvuspb.ListCredUsersResponse, error)

	// TODO: better to accept ctx.
	CreateRole(tenant string, entity *milvuspb.RoleEntity) error
	DropRole(tenant string, roleName string) error
	OperateUserRole(tenant string, userEntity *milvuspb.UserEntity, roleEntity *milvuspb.RoleEntity, operateType milvuspb.OperateUserRoleType) error
	SelectRole(tenant string, entity *milvuspb.RoleEntity, includeUserInfo bool) ([]*milvuspb.RoleResult, error)
	SelectUser(tenant string, entity *milvuspb.UserEntity, includeRoleInfo bool) ([]*milvuspb.UserResult, error)
	OperatePrivilege(tenant string, entity *milvuspb.GrantEntity, operateType milvuspb.OperatePrivilegeType) error
	SelectGrant(tenant string, entity *milvuspb.GrantEntity) ([]*milvuspb.GrantEntity, error)
	ListPolicy(tenant string) ([]string, error)
	ListUserRole(tenant string) ([]string, error)
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
		ctx:     contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName),
		catalog: catalog,
	}
	if err := m.reload(); err != nil {
		return nil, err
	}
	return m, nil
}

func (mt *MetaTableV2) reload() error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	mt.collID2Meta = make(map[UniqueID]*model.Collection)
	mt.collName2ID = make(map[string]UniqueID)
	mt.collAlias2ID = make(map[string]UniqueID)

	// max ts means listing latest resources.
	collections, err := mt.catalog.ListCollections(mt.ctx, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	for name, collection := range collections {
		mt.collID2Meta[collection.CollectionID] = collection
		mt.collName2ID[name] = collection.CollectionID
	}

	aliases, err := mt.catalog.ListAliases(mt.ctx, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	for _, alias := range aliases {
		mt.collAlias2ID[alias.Name] = alias.CollectionID
	}

	return nil
}

func (mt *MetaTableV2) AddCollection(ctx context.Context, coll *model.Collection) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	if coll.State != pb.CollectionState_CollectionCreating {
		return fmt.Errorf("collection state should be creating, collection name: %s, collection id: %d, state: %s", coll.Name, coll.CollectionID, coll.State)
	}
	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.CreateCollection(ctx1, coll, coll.CreateTime); err != nil {
		return err
	}
	mt.collName2ID[coll.Name] = coll.CollectionID
	mt.collID2Meta[coll.CollectionID] = coll
	log.Info("add collection to meta table", zap.String("collection", coll.Name), zap.Int64("id", coll.CollectionID), zap.Uint64("ts", coll.CreateTime))
	return nil
}

func (mt *MetaTableV2) ChangeCollectionState(ctx context.Context, collectionID UniqueID, state pb.CollectionState, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	coll, ok := mt.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	clone := coll.Clone()
	clone.State = state
	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.AlterCollection(ctx1, coll, clone, metastore.MODIFY, ts); err != nil {
		return err
	}
	mt.collID2Meta[collectionID] = clone
	log.Info("change collection state", zap.Int64("collection", collectionID), zap.String("state", state.String()), zap.Uint64("ts", ts))

	return nil
}

func (mt *MetaTableV2) RemoveCollection(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.DropCollection(ctx1, &model.Collection{CollectionID: collectionID}, ts); err != nil {
		return err
	}
	delete(mt.collID2Meta, collectionID)
	return nil
}

func (mt *MetaTableV2) GetCollectionByName(ctx context.Context, collectionName string, ts Timestamp) (*model.Collection, error) {
	mt.ddLock.RLock()
	defer mt.ddLock.RUnlock()

	var collectionID UniqueID
	collectionID, ok := mt.collAlias2ID[collectionName]
	if ok {
		return mt.GetCollectionByID(ctx, collectionID, ts)
	}
	collectionID, ok = mt.collName2ID[collectionName]
	if ok {
		return mt.GetCollectionByID(ctx, collectionID, ts)
	}
	// travel meta information from catalog.
	return mt.catalog.GetCollectionByName(ctx, collectionName, ts)
}

func (mt *MetaTableV2) GetCollectionByID(ctx context.Context, collectionID UniqueID, ts Timestamp) (*model.Collection, error) {
	mt.ddLock.RLock()
	defer mt.ddLock.RUnlock()

	coll, ok := mt.collID2Meta[collectionID]
	if !ok || !coll.Available() || coll.CreateTime > ts {
		// travel meta information from catalog.
		ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
		return mt.catalog.GetCollectionByID(ctx1, collectionID, ts)
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

func (mt *MetaTableV2) ListCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error) {
	mt.ddLock.RLock()
	defer mt.ddLock.RUnlock()

	// list collections should always be loaded from catalog.
	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	colls, err := mt.catalog.ListCollections(ctx1, ts)
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

func (mt *MetaTableV2) ListAbnormalCollections(ctx context.Context, ts Timestamp) ([]*model.Collection, error) {
	mt.ddLock.RLock()
	defer mt.ddLock.RUnlock()

	// list collections should always be loaded from catalog.
	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	colls, err := mt.catalog.ListCollections(ctx1, ts)
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

func (mt *MetaTableV2) ListCollectionPhysicalChannels() map[typeutil.UniqueID][]string {
	mt.ddLock.RLock()
	defer mt.ddLock.RUnlock()

	chanMap := make(map[UniqueID][]string)

	for id, collInfo := range mt.collID2Meta {
		chanMap[id] = collInfo.PhysicalChannelNames
	}

	return chanMap
}

func (mt *MetaTableV2) AddPartition(ctx context.Context, partition *model.Partition) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	_, ok := mt.collID2Meta[partition.CollectionID]
	if !ok {
		return fmt.Errorf("collection not exists: %d", partition.CollectionID)
	}
	if partition.State != pb.PartitionState_PartitionCreated {
		return fmt.Errorf("partition state is not created, collection: %d, partition: %d, state: %s", partition.CollectionID, partition.PartitionID, partition.State)
	}
	mt.collID2Meta[partition.CollectionID].Partitions = append(mt.collID2Meta[partition.CollectionID].Partitions, partition.Clone())
	return nil
}

func (mt *MetaTableV2) ChangePartitionState(ctx context.Context, collectionID UniqueID, partitionID UniqueID, state pb.PartitionState, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	coll, ok := mt.collID2Meta[collectionID]
	if !ok {
		return nil
	}
	for idx, part := range coll.Partitions {
		if part.PartitionID == partitionID {
			clone := part.Clone()
			clone.State = state
			ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
			if err := mt.catalog.AlterPartition(ctx1, part, clone, metastore.MODIFY, ts); err != nil {
				return err
			}
			coll.Partitions[idx] = clone
			return nil
		}
	}
	return fmt.Errorf("partition not exist, collection: %d, partition: %d", collectionID, partitionID)
}

func (mt *MetaTableV2) RemovePartition(ctx context.Context, collectionID UniqueID, partitionID UniqueID, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.DropPartition(ctx1, collectionID, partitionID, ts); err != nil {
		return err
	}
	coll, ok := mt.collID2Meta[collectionID]
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

func (mt *MetaTableV2) CreateAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	collectionID, ok := mt.collName2ID[collectionName]
	if !ok {
		return fmt.Errorf("collection not exists: %s", collectionName)
	}
	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.CreateAlias(ctx1, &model.Alias{
		Name:         alias,
		CollectionID: collectionID,
		CreatedTime:  ts,
		State:        pb.AliasState_AliasCreated,
	}, ts); err != nil {
		return err
	}
	mt.collAlias2ID[alias] = collectionID
	return nil
}

func (mt *MetaTableV2) DropAlias(ctx context.Context, alias string, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.DropAlias(ctx1, alias, ts); err != nil {
		return err
	}
	delete(mt.collAlias2ID, alias)
	return nil
}

func (mt *MetaTableV2) AlterAlias(ctx context.Context, alias string, collectionName string, ts Timestamp) error {
	mt.ddLock.Lock()
	defer mt.ddLock.Unlock()

	collectionID, ok := mt.collName2ID[collectionName]
	if !ok {
		return fmt.Errorf("collection not exists: %s", collectionName)
	}
	ctx1 := contextutil.WithTenantID(ctx, Params.CommonCfg.ClusterName)
	if err := mt.catalog.AlterAlias(ctx1, &model.Alias{
		Name:         alias,
		CollectionID: collectionID,
		CreatedTime:  ts,
		State:        pb.AliasState_AliasCreated,
	}, ts); err != nil {
		return err
	}
	mt.collAlias2ID[alias] = collectionID
	return nil
}

func (mt *MetaTableV2) IsAlias(name string) bool {
	mt.ddLock.RLock()
	defer mt.ddLock.RUnlock()

	_, ok := mt.collAlias2ID[name]
	return ok
}

func (mt *MetaTableV2) GetCollectionNameByID(collID UniqueID) (string, error) {
	panic("implement me")
}

func (mt *MetaTableV2) GetPartitionNameByID(collID UniqueID, partitionID UniqueID, ts Timestamp) (string, error) {
	panic("implement me")
}

// AddCredential add credential
func (mt *MetaTableV2) AddCredential(credInfo *internalpb.CredentialInfo) error {
	if credInfo.Username == "" {
		return fmt.Errorf("username is empty")
	}

	credential := &model.Credential{
		Username:          credInfo.Username,
		EncryptedPassword: credInfo.EncryptedPassword,
	}
	return mt.catalog.CreateCredential(mt.ctx, credential)
}

// GetCredential get credential by username
func (mt *MetaTableV2) GetCredential(username string) (*internalpb.CredentialInfo, error) {
	credential, err := mt.catalog.GetCredential(mt.ctx, username)
	return model.MarshalCredentialModel(credential), err
}

// DeleteCredential delete credential
func (mt *MetaTableV2) DeleteCredential(username string) error {
	return mt.catalog.DropCredential(mt.ctx, username)
}

// ListCredentialUsernames list credential usernames
func (mt *MetaTableV2) ListCredentialUsernames() (*milvuspb.ListCredUsersResponse, error) {
	usernames, err := mt.catalog.ListCredentials(mt.ctx)
	if err != nil {
		return nil, fmt.Errorf("list credential usernames err:%w", err)
	}
	return &milvuspb.ListCredUsersResponse{Usernames: usernames}, nil
}

// CreateRole create role
func (mt *MetaTableV2) CreateRole(tenant string, entity *milvuspb.RoleEntity) error {
	if funcutil.IsEmptyString(entity.Name) {
		return fmt.Errorf("the role name in the role info is empty")
	}
	return mt.catalog.CreateRole(mt.ctx, tenant, entity)
}

// DropRole drop role info
func (mt *MetaTableV2) DropRole(tenant string, roleName string) error {
	return mt.catalog.DropRole(mt.ctx, tenant, roleName)
}

// OperateUserRole operate the relationship between a user and a role, including adding a user to a role and removing a user from a role
func (mt *MetaTableV2) OperateUserRole(tenant string, userEntity *milvuspb.UserEntity, roleEntity *milvuspb.RoleEntity, operateType milvuspb.OperateUserRoleType) error {
	if funcutil.IsEmptyString(userEntity.Name) {
		return fmt.Errorf("username in the user entity is empty")
	}
	if funcutil.IsEmptyString(roleEntity.Name) {
		return fmt.Errorf("role name in the role entity is empty")
	}

	return mt.catalog.AlterUserRole(mt.ctx, tenant, userEntity, roleEntity, operateType)
}

// SelectRole select role.
// Enter the role condition by the entity param. And this param is nil, which means selecting all roles.
// Get all users that are added to the role by setting the includeUserInfo param to true.
func (mt *MetaTableV2) SelectRole(tenant string, entity *milvuspb.RoleEntity, includeUserInfo bool) ([]*milvuspb.RoleResult, error) {
	return mt.catalog.ListRole(mt.ctx, tenant, entity, includeUserInfo)
}

// SelectUser select user.
// Enter the user condition by the entity param. And this param is nil, which means selecting all users.
// Get all roles that are added the user to by setting the includeRoleInfo param to true.
func (mt *MetaTableV2) SelectUser(tenant string, entity *milvuspb.UserEntity, includeRoleInfo bool) ([]*milvuspb.UserResult, error) {
	return mt.catalog.ListUser(mt.ctx, tenant, entity, includeRoleInfo)
}

// OperatePrivilege grant or revoke privilege by setting the operateType param
func (mt *MetaTableV2) OperatePrivilege(tenant string, entity *milvuspb.GrantEntity, operateType milvuspb.OperatePrivilegeType) error {
	if funcutil.IsEmptyString(entity.ObjectName) {
		return fmt.Errorf("the object name in the grant entity is empty")
	}
	if entity.Object == nil || funcutil.IsEmptyString(entity.Object.Name) {
		return fmt.Errorf("the object entity in the grant entity is invalid")
	}
	if entity.Role == nil || funcutil.IsEmptyString(entity.Role.Name) {
		return fmt.Errorf("the role entity in the grant entity is invalid")
	}
	if entity.Grantor == nil {
		return fmt.Errorf("the grantor in the grant entity is empty")
	}
	if entity.Grantor.Privilege == nil || funcutil.IsEmptyString(entity.Grantor.Privilege.Name) {
		return fmt.Errorf("the privilege name in the grant entity is empty")
	}
	if entity.Grantor.User == nil || funcutil.IsEmptyString(entity.Grantor.User.Name) {
		return fmt.Errorf("the grantor name in the grant entity is empty")
	}
	if !funcutil.IsRevoke(operateType) && !funcutil.IsGrant(operateType) {
		return fmt.Errorf("the operate type in the grant entity is invalid")
	}

	return mt.catalog.AlterGrant(mt.ctx, tenant, entity, operateType)
}

// SelectGrant select grant
// The principal entity MUST be not empty in the grant entity
// The resource entity and the resource name are optional, and the two params should be not empty together when you select some grants about the resource kind.
func (mt *MetaTableV2) SelectGrant(tenant string, entity *milvuspb.GrantEntity) ([]*milvuspb.GrantEntity, error) {
	var entities []*milvuspb.GrantEntity
	if entity.Role == nil || funcutil.IsEmptyString(entity.Role.Name) {
		return entities, fmt.Errorf("the role entity in the grant entity is invalid")
	}
	return mt.catalog.ListGrant(mt.ctx, tenant, entity)
}

func (mt *MetaTableV2) ListPolicy(tenant string) ([]string, error) {
	return mt.catalog.ListPolicy(mt.ctx, tenant)
}

func (mt *MetaTableV2) ListUserRole(tenant string) ([]string, error) {
	return mt.catalog.ListUserRole(mt.ctx, tenant)
}
