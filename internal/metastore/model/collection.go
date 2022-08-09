package model

import (
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
)

type Collection struct {
	TenantID             string
	CollectionID         int64
	Partitions           []*Partition
	Name                 string
	Description          string
	AutoID               bool
	Fields               []*Field
	VirtualChannelNames  []string
	PhysicalChannelNames []string
	ShardsNum            int32
	StartPositions       []*commonpb.KeyDataPair
	CreateTime           uint64
	ConsistencyLevel     commonpb.ConsistencyLevel
	Aliases              []string
	Available            bool
	Extra                map[string]string // extra kvs
}

func (c Collection) Clone() *Collection {
	return &Collection{
		TenantID:             c.TenantID,
		CollectionID:         c.CollectionID,
		Name:                 c.Name,
		Description:          c.Description,
		AutoID:               c.AutoID,
		Fields:               c.Fields,
		Partitions:           c.Partitions,
		VirtualChannelNames:  c.VirtualChannelNames,
		PhysicalChannelNames: c.PhysicalChannelNames,
		ShardsNum:            c.ShardsNum,
		ConsistencyLevel:     c.ConsistencyLevel,
		CreateTime:           c.CreateTime,
		StartPositions:       c.StartPositions,
		Aliases:              c.Aliases,
		Available:            c.Available,
		Extra:                c.Extra,
	}
}

func (c *Collection) Equal(other *Collection) bool {
	if c == other {
		return true
	}
	if other == nil {
		return false
	}
	if !funcutil.SliceSetEqualCmp(c.Fields, other.Fields, func(a, b *Field) bool {
		return a.Equal(b)
	}) {
		return false
	}
	filterFn := func(part *Partition) bool {
		return part.Available
	}
	parts := funcutil.SliceFilter(c.Partitions, filterFn)
	otherParts := funcutil.SliceFilter(other.Partitions, filterFn)
	if !funcutil.SliceSetEqualCmp(parts, otherParts, func(a, b *Partition) bool {
		return a.Equal(b)
	}) {
		return false
	}
	//if !funcutil.SliceSetEqual(c.VirtualChannelNames, other.VirtualChannelNames) {
	//	return false
	//}
	//if !funcutil.SliceSetEqual(c.PhysicalChannelNames, other.PhysicalChannelNames) {
	//	return false
	//}
	//if !funcutil.SliceSetEqual(c.Aliases, other.Aliases) {
	//	return false
	//}
	return c.TenantID == other.TenantID &&
		//c.CollectionID == other.CollectionID &&
		c.Name == other.Name &&
		c.Description == other.Description &&
		c.AutoID == other.AutoID &&
		c.ShardsNum == other.ShardsNum &&
		c.ConsistencyLevel == other.ConsistencyLevel
	//c.CreateTime == other.CreateTime &&
	//c.StartPositions == other.StartPositions &&
}

func UnmarshalCollectionModel(coll *pb.CollectionInfo) *Collection {
	if coll == nil {
		return nil
	}

	// backward compatible for deprecated fields
	var partitions []*Partition
	if len(coll.Partitions) != 0 {
		partitions = make([]*Partition, len(coll.Partitions))
		for idx, partition := range coll.Partitions {
			partitions[idx] = &Partition{
				PartitionID:               partition.GetPartitionID(),
				PartitionName:             partition.GetPartitionName(),
				PartitionCreatedTimestamp: partition.GetPartitionCreatedTimestamp(),
				Available:                 true,
			}
		}
	} else {
		partitions = make([]*Partition, len(coll.PartitionIDs))
		for idx := range coll.PartitionIDs {
			partitions[idx] = &Partition{
				PartitionID:               coll.PartitionIDs[idx],
				PartitionName:             coll.PartitionNames[idx],
				PartitionCreatedTimestamp: coll.PartitionCreatedTimestamps[idx],
				Available:                 true,
			}
		}
	}

	//filedIDToIndexIDs := make([]common.Int64Tuple, len(coll.FieldIndexes))
	//for idx, fieldIndexInfo := range coll.FieldIndexes {
	//	filedIDToIndexIDs[idx] = common.Int64Tuple{
	//		Key:   fieldIndexInfo.FiledID,
	//		Value: fieldIndexInfo.IndexID,
	//	}
	//}

	return &Collection{
		CollectionID:         coll.ID,
		Name:                 coll.Schema.Name,
		Description:          coll.Schema.Description,
		AutoID:               coll.Schema.AutoID,
		Fields:               UnmarshalFieldModels(coll.Schema.Fields),
		Partitions:           partitions,
		VirtualChannelNames:  coll.VirtualChannelNames,
		PhysicalChannelNames: coll.PhysicalChannelNames,
		ShardsNum:            coll.ShardsNum,
		ConsistencyLevel:     coll.ConsistencyLevel,
		CreateTime:           coll.CreateTime,
		StartPositions:       coll.StartPositions,
		Available:            true,
	}
}

func MarshalCollectionModel(coll *Collection) *pb.CollectionInfo {
	if coll == nil {
		return nil
	}

	fields := make([]*schemapb.FieldSchema, len(coll.Fields))
	for idx, field := range coll.Fields {
		fields[idx] = &schemapb.FieldSchema{
			FieldID:      field.FieldID,
			Name:         field.Name,
			IsPrimaryKey: field.IsPrimaryKey,
			Description:  field.Description,
			DataType:     field.DataType,
			TypeParams:   field.TypeParams,
			IndexParams:  field.IndexParams,
			AutoID:       field.AutoID,
		}
	}
	collSchema := &schemapb.CollectionSchema{
		Name:        coll.Name,
		Description: coll.Description,
		AutoID:      coll.AutoID,
		Fields:      fields,
	}

	partitions := make([]*pb.PartitionInfo, len(coll.Partitions))
	for idx, partition := range coll.Partitions {
		partitions[idx] = &pb.PartitionInfo{
			PartitionID:               partition.PartitionID,
			PartitionName:             partition.PartitionName,
			PartitionCreatedTimestamp: partition.PartitionCreatedTimestamp,
		}
	}

	return &pb.CollectionInfo{
		ID:                   coll.CollectionID,
		Schema:               collSchema,
		Partitions:           partitions,
		CreateTime:           coll.CreateTime,
		VirtualChannelNames:  coll.VirtualChannelNames,
		PhysicalChannelNames: coll.PhysicalChannelNames,
		ShardsNum:            coll.ShardsNum,
		ConsistencyLevel:     coll.ConsistencyLevel,
		StartPositions:       coll.StartPositions,
	}
}
