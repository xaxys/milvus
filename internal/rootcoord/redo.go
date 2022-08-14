package rootcoord

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"
)

type baseRedoTask struct {
	baseTaskV2
	logs          stepLogger // logs of steps
	syncTodoStep  []Step     // steps to execute synchronously
	asyncTodoStep []Step     // steps to execute asynchronously
}

func (b *baseRedoTask) AddSyncStep(step Step) {
	b.syncTodoStep = append(b.syncTodoStep, step)
}

func (b *baseRedoTask) AddAsyncStep(step Step) {
	b.asyncTodoStep = append(b.asyncTodoStep, step)
}

func (b *baseRedoTask) Execute(ctx context.Context) error {
	if len(b.syncTodoStep) <= 0 && len(b.asyncTodoStep) <= 0 {
		return nil
	}
	for i := len(b.asyncTodoStep) - 1; i >= 0; i-- {
		step := b.asyncTodoStep[i]
		b.logs.AddStep(step)
	}
	for i := len(b.syncTodoStep) - 1; i >= 0; i-- {
		step := b.syncTodoStep[i]
		b.logs.AddStep(step)
	}
	for _, step := range b.syncTodoStep {
		err := step.Execute(ctx)
		if err != nil {
			log.Warn("execute task failed", zap.Error(err))
			b.logs.Clear()
			return err
		}
		b.logs.PopStep()
	}
	go b.logs.Rollback(ctx)
	return nil
}

func (b *baseRedoTask) prepareLogger() {
	writeFunc := func(data []byte) error {
		k := fmt.Sprintf("%s/%d", DDLLogPrefix, b.GetID())
		if len(data) == 0 {
			return b.core.txn.Remove(k)
		}
		return b.core.txn.Save(k, string(data))
	}
	b.logs = stepLogger{
		steps:     make([]Step, 0),
		info:      rootcoordpb.TaskInfo{},
		writeFunc: writeFunc,
	}
}
