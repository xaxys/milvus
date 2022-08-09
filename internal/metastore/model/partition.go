package model

type Partition struct {
	PartitionID               int64
	PartitionName             string
	PartitionCreatedTimestamp uint64
	Available                 bool
	Extra                     map[string]string
}

func (p *Partition) Equal(other *Partition) bool {
	if other == nil {
		return false
	}
	if p == other {
		return true
	}
	if len(p.Extra) != len(other.Extra) {
		return false
	}
	for k, v := range p.Extra {
		if other.Extra[k] != v {
			return false
		}
	}
	return p.PartitionName == other.PartitionName
	//p.PartitionCreatedTimestamp == other.PartitionCreatedTimestamp &&
	//p.Available == other.Available
}
