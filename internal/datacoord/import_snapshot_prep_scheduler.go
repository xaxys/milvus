// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datacoord

// Legacy Import V2 snapshot preparation scheduler. Import V3 prepares
// synchronously in its SnapshotPreparing stage; this bounded scheduler exists
// only for in-flight V2 snapshot jobs and is deleted together with Import V2.

import (
	"context"
	"errors"
	"sync"

	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

// snapshotPrepScheduler owns the bounded V2-only source preparation. V2
// prepares in Pending; the scheduler keeps the source I/O off the checker loop
// and bounds it to a small fixed number of jobs.
type snapshotPrepScheduler struct {
	ctx        context.Context
	meta       *meta
	alloc      allocator.Allocator
	importMeta ImportMeta

	mu        sync.Mutex
	preparing map[int64]context.CancelFunc
	wg        sync.WaitGroup
	closed    bool
}

func newSnapshotPrepScheduler(ctx context.Context, meta *meta, alloc allocator.Allocator, importMeta ImportMeta) *snapshotPrepScheduler {
	return &snapshotPrepScheduler{ctx: ctx, meta: meta, alloc: alloc, importMeta: importMeta}
}

// cancelInactive is called each V2 checker tick, including for jobs whose
// channels are not ready. A terminal job must not keep doing I/O.
func (p *snapshotPrepScheduler) cancelInactive() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for id, cancel := range p.preparing {
		job := p.importMeta.GetJob(p.ctx, id)
		if job == nil || job.GetState() != internalpb.ImportJobState_Pending || !importutilv2.IsSnapshotPreparation(job.GetFiles()) {
			cancel()
		}
	}
}

func (p *snapshotPrepScheduler) schedule(job ImportJob) {
	p.mu.Lock()
	// A fixed small bound prevents a backlog from creating one goroutine and
	// source client per pending job. The next checker tick admits queued work.
	if p.closed || p.preparing[job.GetJobID()] != nil || len(p.preparing) >= 4 {
		p.mu.Unlock()
		return
	}
	ctx, cancel := snapshotExpansionContext(p.ctx, job)
	if p.preparing == nil {
		p.preparing = make(map[int64]context.CancelFunc)
	}
	p.preparing[job.GetJobID()] = cancel
	p.wg.Add(1)
	p.mu.Unlock()
	go func() {
		defer p.wg.Done()
		defer func() {
			cancel()
			p.mu.Lock()
			delete(p.preparing, job.GetJobID())
			p.mu.Unlock()
		}()
		p.expandAndPublish(ctx, job)
	}()
}

// expandAndPublish runs one expansion and persists its result. V2 owns the
// failure transition: any expansion error marks the Pending job Failed.
func (p *snapshotPrepScheduler) expandAndPublish(ctx context.Context, job ImportJob) {
	files, options, sources, err := expandSnapshotJob(ctx, p.meta, p.alloc, job)
	if ctx.Err() != nil {
		return // Cancellation/leadership loss never publishes a partial plan.
	}
	if err != nil {
		if errors.Is(err, errSnapshotExpansionRetry) {
			mlog.Warn(ctx, "failed to allocate snapshot import file IDs", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
			return // Retry on the next tick; nothing was published.
		}
		if saveErr := p.importMeta.UpdateJob(ctx, job.GetJobID(), func(current ImportJob) {
			if current.GetState() != internalpb.ImportJobState_Pending || !importutilv2.IsSnapshotPreparation(current.GetFiles()) {
				return
			}
			UpdateJobState(internalpb.ImportJobState_Failed)(current)
			UpdateJobReason(err.Error())(current)
		}); saveErr != nil {
			mlog.Warn(ctx, "failed to persist snapshot preparation failure", mlog.FieldJobID(job.GetJobID()), mlog.Err(saveErr))
		}
		return
	}
	err = p.importMeta.UpdateJob(ctx, job.GetJobID(), func(current ImportJob) {
		// UpdateJob holds the metadata lock and refuses terminal updates. Keep
		// the prepared plan atomic with the files and inventories; replay after
		// a failed save is safe and never creates duplicate tasks.
		if current.GetState() != internalpb.ImportJobState_Pending || !importutilv2.IsSnapshotPreparation(current.GetFiles()) {
			return
		}
		j := current.(*importJob)
		j.Files, j.Options, j.SnapshotL0Sources = files, options, sources
	})
	if err != nil {
		mlog.Warn(ctx, "failed to persist prepared snapshot import", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
	}
}

func (p *snapshotPrepScheduler) close() {
	p.mu.Lock()
	p.closed = true
	for _, cancel := range p.preparing {
		cancel()
	}
	p.mu.Unlock()
	p.wg.Wait()
}
