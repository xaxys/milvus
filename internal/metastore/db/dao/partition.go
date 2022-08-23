package dao

import (
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/db/dbmodel"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type partitionDb struct {
	db *gorm.DB
}

func (s *partitionDb) GetByCollectionID(tenantID string, collectionID typeutil.UniqueID, ts typeutil.Timestamp) ([]*dbmodel.Partition, error) {
	var r []*dbmodel.Partition

	err := s.db.Model(&dbmodel.Partition{}).Where("tenant_id = ? AND collection_id = ? AND ts = ? AND is_deleted = false", tenantID, collectionID, ts).Find(&r).Error
	if err != nil {
		log.Error("get partitions by collection_id and ts failed", zap.String("tenant", tenantID), zap.Int64("collID", collectionID), zap.Uint64("ts", ts), zap.Error(err))
		return nil, err
	}

	return r, nil
}

func (s *partitionDb) Insert(in []*dbmodel.Partition) error {
	err := s.db.CreateInBatches(in, 100).Error
	if err != nil {
		log.Error("insert partition failed", zap.Error(err))
		return err
	}

	return nil
}

func (s *partitionDb) Upsert(in *dbmodel.Partition) error {
	err := s.db.Clauses(clause.OnConflict{
		// constraint UNIQUE (tenant_id, collection_id, ts)
		DoUpdates: clause.AssignmentColumns([]string{"tenant_id", "collection_id", "partition_id", "ts"}),
	}).Create(in).Error

	if err != nil {
		log.Error("upsert partition failed", zap.String("tenant", in.TenantID), zap.Int64("collID", in.CollectionID), zap.Int64("partID", in.PartitionID), zap.Uint64("ts", in.Ts), zap.Error(err))
		return err
	}

	return nil
}
