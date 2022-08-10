package rootcoord

import (
	"context"
	"strings"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/tso"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/kv"
	"github.com/milvus-io/milvus/internal/log"
)

type RootCoord struct {
	ctx          context.Context
	meta         IMetaTableV2
	scheduler    IScheduler
	txn          kv.TxnKV
	chanTimeTick *timetickSync
	idAllocator  allocator.GIDAllocator
	tsoAllocator tso.Allocator
}

func (s *RootCoord) start() error {
	if err := s.recover(s.ctx); err != nil {
		return err
	}
	s.scheduler.Start()
	return nil
}

func (s *RootCoord) recover(ctx context.Context) error {
	paths, tasks, err := s.txn.LoadWithPrefix(DDLLogPrefix)
	if err != nil {
		if strings.Contains(err.Error(), "there is no value on key") {
			log.Info("skip restore with no ddl-log key", zap.Error(err))
			return nil
		}
	}

	for i, task := range tasks {
		info := &rootcoordpb.TaskInfo{}
		err := proto.Unmarshal([]byte(task), info)
		if err != nil {
			log.Error("unmarshal task recover info failed", zap.Error(err))
			continue
		}
		logs, err := UnmarshalTaskInfo(info, s)
		if err != nil {
			log.Error("extract task recover info failed", zap.Int64("msgID", info.RequestId), zap.Error(err))
			continue
		}
		logs.writeFunc = func(data []byte) error {
			if len(data) == 0 {
				return s.txn.Remove(paths[i])
			}
			return s.txn.Save(paths[i], string(data))
		}
		logs.Rollback(ctx)
	}

	return nil
}

func (s *RootCoord) CreateCollection(ctx context.Context, in *milvuspb.CreateCollectionRequest) (*commonpb.Status, error) {
	t := &createCollectionTask{
		baseUndoTask: baseUndoTask{},
		Req:          in,
	}
	if err := s.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}

func (s *RootCoord) DropCollection(ctx context.Context, in *milvuspb.DropCollectionRequest) (*commonpb.Status, error) {
	t := &dropCollectionTask{
		baseRedoTask: baseRedoTask{},
		Req:          in,
	}
	if err := s.scheduler.AddTask(t); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	if err := t.WaitToFinish(); err != nil {
		return failStatus(commonpb.ErrorCode_UnexpectedError, err.Error()), nil
	}
	return succStatus(), nil
}
