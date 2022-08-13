package rootcoord

import (
	"context"
	"sync"
)

type logManager struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	logs   chan *stepLogger
}

func newLogManager(ctx context.Context) *logManager {
	ctx1, cancel := context.WithCancel(ctx)
	return &logManager{
		ctx:    ctx1,
		cancel: cancel,
		logs:   make(chan *stepLogger, 1024*10),
	}
}

func (l *logManager) rollbackLoop() {
	l.wg.Add(1)
	defer l.wg.Done()
	for {
		select {
		case <-l.ctx.Done():
			return
		case log, ok := <-l.logs:
			if !ok {
				return
			}
			log.Rollback(l.ctx)
		}
	}
}

func (l *logManager) Start() {
	go l.rollbackLoop()
}

func (l *logManager) Stop() {
	l.cancel()
	l.wg.Wait()
}

func (l *logManager) AddLog(log *stepLogger) {
	l.logs <- log
}
