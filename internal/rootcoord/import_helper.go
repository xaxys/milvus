package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"go.uber.org/zap"
)

type GetCollectionNameFunc func(collID, partitionID UniqueID) (string, string, error)
type IdAllocator func(count uint32) (UniqueID, UniqueID, error)
type ImportFunc func(ctx context.Context, req *datapb.ImportTaskRequest) *datapb.ImportTaskResponse

type ImportFactory interface {
	NewGetCollectionNameFunc() GetCollectionNameFunc
	NewIdAllocator() IdAllocator
	NewImportFunc() ImportFunc
}

type ImportFactoryImpl struct {
	c *RootCoord
}

func (f ImportFactoryImpl) NewGetCollectionNameFunc() GetCollectionNameFunc {
	return GetCollectionNameWithRootCoord(f.c)
}

func (f ImportFactoryImpl) NewIdAllocator() IdAllocator {
	return IdAllocatorWithRootCoord(f.c)
}

func (f ImportFactoryImpl) NewImportFunc() ImportFunc {
	return ImportFuncWithRootCoord(f.c)
}

func NewImportFactory(c *RootCoord) ImportFactory {
	return &ImportFactoryImpl{c: c}
}

func GetCollectionNameWithRootCoord(c *RootCoord) GetCollectionNameFunc {
	return func(collID, partitionID UniqueID) (string, string, error) {
		colName, err := c.meta.GetCollectionNameByID(collID)
		if err != nil {
			log.Error("RootCoord failed to get collection name by id", zap.Int64("ID", collID), zap.Error(err))
			return "", "", err
		}

		partName, err := c.meta.GetPartitionNameByID(collID, partitionID, 0)
		if err != nil {
			log.Error("RootCoord failed to get partition name by id", zap.Int64("ID", partitionID), zap.Error(err))
			return colName, "", err
		}

		return colName, partName, nil
	}
}

func IdAllocatorWithRootCoord(c *RootCoord) IdAllocator {
	return func(count uint32) (UniqueID, UniqueID, error) {
		return c.idAllocator.Alloc(count)
	}
}

func ImportFuncWithRootCoord(c *RootCoord) ImportFunc {
	return func(ctx context.Context, req *datapb.ImportTaskRequest) *datapb.ImportTaskResponse {
		// TODO: better to handle error here.
		resp, _ := c.broker.Import(ctx, req)
		return resp
	}
}
