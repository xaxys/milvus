package rootcoord

import (
	"context"
	"sync"
)

type taskScheduler struct {
	ctx    context.Context
	cancel context.CancelFunc

	taskChan chan reqTask
	wg       sync.WaitGroup
}

func newTaskScheduler(ctx context.Context) *taskScheduler {
	ctx1, cancel := context.WithCancel(ctx)
	s := &taskScheduler{
		ctx:      ctx1,
		cancel:   cancel,
		taskChan: make(chan reqTask, Params.RootCoordCfg.MaxTaskChanSize),
	}
	return s
}

func (s *taskScheduler) Start() {
	s.wg.Add(1)
	go s.taskLoop()
}

func (s *taskScheduler) Stop() {
	s.cancel()
	s.wg.Wait()
}

func (s *taskScheduler) taskLoop() {
	defer s.wg.Done()
	for {
		select {
		case <-s.ctx.Done():
			return
		case task := <-s.taskChan:
			if ok, err := task.Prepare(s.ctx); !ok || err != nil {
				task.NotifyDone(err)
				continue
			}
			err := task.Execute(s.ctx)
			task.NotifyDone(err)
		}
	}
}

func (s *taskScheduler) AddTask(task reqTask) {
	s.taskChan <- task
}
