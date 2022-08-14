package rootcoord

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"

	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"
)

type baseUndoTask struct {
	baseTaskV2
	logs     stepLogger // logs of steps
	todoStep []Step     // steps to execute
	undoStep []Step     // steps to undo
}

func (b *baseUndoTask) AddStep(todoStep, undoStep Step) {
	b.todoStep = append(b.todoStep, todoStep)
	b.undoStep = append(b.undoStep, undoStep)
}

func (b *baseUndoTask) Execute(ctx context.Context) error {
	if len(b.todoStep) != len(b.undoStep) {
		return fmt.Errorf("todo step and undo step length not equal")
	}
	if len(b.todoStep) <= 0 || len(b.undoStep) <= 0 {
		return nil
	}
	for i := 0; i < len(b.todoStep); i++ {
		todoStep := b.todoStep[i]
		undoStep := b.undoStep[i]
		b.logs.AddStep(undoStep)
		err := todoStep.Execute(ctx)
		if err != nil {
			log.Warn("execute task failed", zap.Error(err))
			b.logs.Rollback(ctx)
			return err
		}
	}
	b.logs.Clear()
	return nil
}

func (b *baseUndoTask) prepareLogger() {
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
