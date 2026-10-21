// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bulkimport

import (
	"context"
	"errors"
	"sync"

	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/meta"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	importcommon "github.com/milvus-io/milvus/internal/util/importutilv2/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// snapshotExpansionConcurrency bounds the source I/O a backlog of
// SnapshotPreparing jobs can start at once. The next checker tick admits
// queued work.
const snapshotExpansionConcurrency = 4

// snapshotExpander runs the SnapshotPreparing work off the shared V3 checker
// goroutine. The stage still owns the state transition; this type only bounds
// and cancels the source I/O so one slow snapshot cannot stall unrelated jobs.
type snapshotExpander struct {
	ctx        context.Context
	meta       *meta.Meta
	alloc      allocator.Allocator
	importMeta ImportMeta

	mu       sync.Mutex
	inflight map[int64]context.CancelFunc
	wg       sync.WaitGroup
	closed   bool
}

func newSnapshotExpander(ctx context.Context, mt *meta.Meta, alloc allocator.Allocator, importMeta ImportMeta) *snapshotExpander {
	return &snapshotExpander{
		ctx:        ctx,
		meta:       mt,
		alloc:      alloc,
		importMeta: importMeta,
	}
}

// schedule starts one bounded background expansion for a job. The handler only
// schedules; the worker persists the result and the next tick either creates
// PreImport tasks or observes the failure.
func (e *snapshotExpander) schedule(job ImportJob) {
	e.mu.Lock()
	if e.closed || e.inflight[job.GetJobID()] != nil || len(e.inflight) >= snapshotExpansionConcurrency {
		e.mu.Unlock()
		return
	}
	ctx, cancel, capped := snapshotExpansionContext(e.ctx, job)
	if e.inflight == nil {
		e.inflight = make(map[int64]context.CancelFunc)
	}
	e.inflight[job.GetJobID()] = cancel
	e.wg.Add(1)
	e.mu.Unlock()

	go func() {
		defer e.wg.Done()
		defer func() {
			cancel()
			e.mu.Lock()
			delete(e.inflight, job.GetJobID())
			e.mu.Unlock()
		}()
		e.run(ctx, job, capped)
	}()
}

// cancelInactive stops work for jobs that left SnapshotPreparing or no longer
// carry the preparation marker.
func (e *snapshotExpander) cancelInactive() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for id, cancel := range e.inflight {
		job := e.importMeta.GetJob(e.ctx, id)
		if job == nil || job.GetState() != internalpb.ImportJobState_SnapshotPreparing || !importutilv2.IsSnapshotPreparation(job.GetFiles()) {
			cancel()
		}
	}
}

// close stops every attempt and waits for the workers so a leadership loss
// does not leave source I/O running against a retired checker.
func (e *snapshotExpander) close() {
	e.mu.Lock()
	e.closed = true
	for _, cancel := range e.inflight {
		cancel()
	}
	e.mu.Unlock()
	e.wg.Wait()
}

func (e *snapshotExpander) wait() {
	e.wg.Wait()
}

func (e *snapshotExpander) run(ctx context.Context, job ImportJob, capped bool) {
	files, options, sources, err := expandSnapshotJob(ctx, e.meta, e.alloc, job)
	if ctx.Err() != nil {
		if capped && errors.Is(ctx.Err(), context.DeadlineExceeded) {
			e.fail(job, merr.WrapErrImportFailedMsg("snapshot expansion exceeded %s", snapshotExpansionTimeout).Error())
		}
		return // Leadership loss or job timeout: the job GC owns the transition.
	}
	if err != nil {
		if errors.Is(err, errSnapshotExpansionRetry) {
			mlog.Warn(ctx, "failed to allocate snapshot import file IDs", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
			return // Retry on the next tick; nothing was published.
		}
		if importcommon.IsTerminalImportV3Err(err) {
			e.fail(job, err.Error())
			return
		}
		mlog.Warn(ctx, "snapshot expansion failed", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
		return
	}
	if err := e.importMeta.UpdateJob(e.ctx, job.GetJobID(), func(current ImportJob) {
		// UpdateJob holds the metadata lock and refuses terminal updates. The
		// state/marker check keeps the prepared plan atomic with the job; replay
		// after a failed save retries preparation, which is idempotent.
		if current.GetState() != internalpb.ImportJobState_SnapshotPreparing || !importutilv2.IsSnapshotPreparation(current.GetFiles()) {
			return
		}
		j := current.(*Job)
		j.Files, j.Options, j.SnapshotL0Sources = files, options, sources
	}); err != nil {
		mlog.Warn(ctx, "failed to persist expanded snapshot import", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
	}
}

func (e *snapshotExpander) fail(job ImportJob, reason string) {
	if err := e.importMeta.UpdateJob(e.ctx, job.GetJobID(), func(current ImportJob) {
		if current.GetState() != internalpb.ImportJobState_SnapshotPreparing || !importutilv2.IsSnapshotPreparation(current.GetFiles()) {
			return
		}
		// Route through the shared action so a terminal transition strips the
		// source credentials and schedules GC; assigning State directly leaves
		// CleanupTs at its sentinel and keeps the job forever.
		UpdateJobState(internalpb.ImportJobState_Failed)(current)
		UpdateJobReason(reason)(current)
	}); err != nil {
		mlog.Warn(e.ctx, "failed to persist snapshot expansion failure", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
	}
}
