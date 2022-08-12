package model

import pb "github.com/milvus-io/milvus/internal/proto/etcdpb"

type Partition struct {
	PartitionID               int64
	PartitionName             string
	PartitionCreatedTimestamp uint64
	Extra                     map[string]string
	CollectionID              int64
	State                     pb.PartitionState
}

func (p Partition) Available() bool {
	return p.State == pb.PartitionState_PartitionCreated
}

func (p Partition) Clone() *Partition {
	return &Partition{
		PartitionID:               p.PartitionID,
		PartitionName:             p.PartitionName,
		PartitionCreatedTimestamp: p.PartitionCreatedTimestamp,
		Extra:                     p.Extra,
		CollectionID:              p.CollectionID,
		State:                     p.State,
	}
}

func MarshalPartitionModel(partition *Partition) *pb.PartitionInfo {
	return &pb.PartitionInfo{
		PartitionID:               partition.PartitionID,
		PartitionName:             partition.PartitionName,
		PartitionCreatedTimestamp: partition.PartitionCreatedTimestamp,
		CollectionId:              partition.CollectionID,
		State:                     partition.State,
	}
}

func UnmarshalPartitionModel(info *pb.PartitionInfo) *Partition {
	return &Partition{
		PartitionID:               info.GetPartitionID(),
		PartitionName:             info.GetPartitionName(),
		PartitionCreatedTimestamp: info.GetPartitionCreatedTimestamp(),
		CollectionID:              info.GetCollectionId(),
		State:                     info.GetState(),
	}
}
