package rootcoord

import (
	"context"
	"sync"

	"github.com/milvus-io/milvus/internal/tso"

	"github.com/milvus-io/milvus/internal/allocator"
)

type IScheduler interface {
	Start()
	Stop()
	AddTask(t taskV2) error
}

type scheduler struct {
	ctx    context.Context
	cancel context.CancelFunc

	idAllocator  allocator.GIDAllocator
	tsoAllocator tso.Allocator

	taskChan chan taskV2
	wg       sync.WaitGroup
}

func (s *scheduler) Start() {
	s.wg.Add(1)
	go s.taskLoop()
}

func (s *scheduler) Stop() {
	s.cancel()
	s.wg.Wait()
}

func (s *scheduler) taskLoop() {
	defer s.wg.Done()
	for {
		select {
		case <-s.ctx.Done():
			return
		case task := <-s.taskChan:
			if err := task.Prepare(s.ctx); err != nil {
				task.NotifyDone(err)
				continue
			}
			err := task.Execute(s.ctx)
			task.NotifyDone(err)
		}
	}
}

func (s *scheduler) setID(task taskV2) error {
	id, err := s.idAllocator.AllocOne()
	if err != nil {
		return err
	}
	task.SetID(id)
	return nil
}

func (s *scheduler) setTs(task taskV2) error {
	ts, err := s.tsoAllocator.GenerateTSO(1)
	if err != nil {
		return err
	}
	task.SetTs(ts)
	return nil
}

func (s *scheduler) enqueue(task taskV2) {
	s.taskChan <- task
}

func (s *scheduler) AddTask(task taskV2) error {
	if err := s.setID(task); err != nil {
		return err
	}
	if err := s.setTs(task); err != nil {
		return err
	}
	s.enqueue(task)
	return nil
}
