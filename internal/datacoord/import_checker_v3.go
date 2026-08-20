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

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/samber/lo"

	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/broker"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
)

// importCheckerV3 is the ImportTaskV3 state machine. It only owns V3 jobs
// (ImportJob.version == ImportJobVersionV3) and never touches the legacy
// PreImportTask/ImportTaskV2 path, which remains fully owned by importChecker.
type importCheckerV3 struct {
	ctx        context.Context
	meta       *meta
	broker     broker.Broker
	alloc      allocator.Allocator
	importMeta ImportMeta
	ci         CompactionInspector
	handler    Handler

	hooks importCheckerHooks

	closeOnce sync.Once
	closeChan chan struct{}
}

func NewImportCheckerV3(ctx context.Context,
	meta *meta,
	broker broker.Broker,
	alloc allocator.Allocator,
	importMeta ImportMeta,
	ci CompactionInspector,
	handler Handler,
	hooks importCheckerHooks,
) ImportChecker {
	return &importCheckerV3{
		ctx:        ctx,
		meta:       meta,
		broker:     broker,
		alloc:      alloc,
		importMeta: importMeta,
		ci:         ci,
		handler:    handler,
		hooks:      hooks,
		closeChan:  make(chan struct{}),
	}
}

// Start runs the checker loops until Close. The state-machine loop and the
// timeout/GC loop deliberately run on separate goroutines: checkGC's rollback
// broadcast can park on the ctx-insensitive resource-key lock (see checkGC), and
// isolating it guarantees the state machine keeps making progress no matter how
// long GC blocks. All state shared by the two loops lives behind importMeta's
// mutex (which already serves concurrent RPC and ack-callback goroutines), and
// UpdateJob refuses transitions out of Completed/Failed, so the loops cannot
// resurrect or regress each other's terminal states.
func (c *importCheckerV3) Start() {
	mlog.Info(c.ctx, "start import checker v3")
	go c.runGCLoop()
	c.runStateMachineLoop()
}

func (c *importCheckerV3) runStateMachineLoop() {
	ticker := time.NewTicker(Params.DataCoordCfg.ImportCheckIntervalHigh.GetAsDuration(time.Second)) // 2s
	defer ticker.Stop()
	for {
		select {
		case <-c.closeChan:
			mlog.Info(c.ctx, "import checker v3 state-machine loop exited")
			return
		case <-ticker.C:
			jobs := c.importMeta.GetJobBy(c.ctx)
			for _, job := range jobs {
				if job.GetVersion() != datapb.ImportJobVersion_ImportJobVersionV3 {
					continue
				}
				if !funcutil.SliceSetEqual[string](job.GetVchannels(), job.GetReadyVchannels()) {
					// wait for all channels to send signals
					mlog.Info(c.ctx, "waiting for all channels to send signals",
						mlog.Strings("vchannels", job.GetVchannels()),
						mlog.Strings("readyVchannels", job.GetReadyVchannels()),
						mlog.FieldJobID(job.GetJobID()))
					continue
				}
				switch job.GetState() {
				case internalpb.ImportJobState_Pending:
					c.checkPendingJob(job)
				case internalpb.ImportJobState_Resharding:
					c.checkReshardJob(job)
				case internalpb.ImportJobState_Planning:
					c.checkPlanningJob(job)
				case internalpb.ImportJobState_Importing:
					c.checkImportingJob(job)
				case internalpb.ImportJobState_IndexBuilding:
					c.checkIndexBuildingJob(job)
				case internalpb.ImportJobState_Uncommitted:
					c.checkUncommittedJob(job)
				case internalpb.ImportJobState_Committing:
					c.checkCommittingJob(job)
				case internalpb.ImportJobState_Failed:
					c.checkFailedJob(job)
				}
			}
		}
	}
}

func (c *importCheckerV3) runGCLoop() {
	ticker := time.NewTicker(Params.DataCoordCfg.ImportCheckIntervalLow.GetAsDuration(time.Second)) // 2min
	defer ticker.Stop()
	for {
		select {
		case <-c.closeChan:
			mlog.Info(c.ctx, "import checker v3 gc loop exited")
			return
		case <-ticker.C:
			jobs := c.importMeta.GetJobBy(c.ctx)
			for _, job := range jobs {
				if job.GetVersion() != datapb.ImportJobVersion_ImportJobVersionV3 {
					continue
				}
				c.tryTimeoutJob(job)
				c.checkGC(job)
			}
			jobsByColl := lo.GroupBy(lo.Filter(jobs, func(job ImportJob, _ int) bool {
				return job.GetVersion() == datapb.ImportJobVersion_ImportJobVersionV3
			}), func(job ImportJob) int64 {
				return job.GetCollectionID()
			})
			for collID, collJobs := range jobsByColl {
				c.checkCollection(collID, collJobs)
			}
			c.LogJobStats(jobs)
			c.LogTaskStats()
		}
	}
}

func (c *importCheckerV3) Close() {
	c.closeOnce.Do(func() {
		close(c.closeChan)
	})
}

func (c *importCheckerV3) LogJobStats(jobs []ImportJob) {
	jobs = lo.Filter(jobs, func(job ImportJob, _ int) bool {
		return job.GetVersion() == datapb.ImportJobVersion_ImportJobVersionV3
	})
	byState := lo.GroupBy(jobs, func(job ImportJob) string {
		return job.GetState().String()
	})
	stateNum := make(map[string]int)
	for state := range internalpb.ImportJobState_value {
		if state == internalpb.ImportJobState_None.String() {
			continue
		}
		num := len(byState[state])
		stateNum[state] = num
		metrics.ImportJobs.WithLabelValues(state).Set(float64(num))
	}
	mlog.Info(c.ctx, "import job stats", mlog.Any("stateNum", stateNum))
}

func (c *importCheckerV3) LogTaskStats() {
	logFunc := func(tasks []ImportTask, taskType TaskType) {
		byState := lo.GroupBy(tasks, func(t ImportTask) datapb.ImportTaskStateV2 {
			return t.GetState()
		})
		pending := len(byState[datapb.ImportTaskStateV2_Pending])
		inProgress := len(byState[datapb.ImportTaskStateV2_InProgress])
		completed := len(byState[datapb.ImportTaskStateV2_Completed])
		failed := len(byState[datapb.ImportTaskStateV2_Failed])
		mlog.Info(c.ctx, "import task stats", mlog.String("type", taskType.String()),
			mlog.Int("pending", pending), mlog.Int("inProgress", inProgress),
			mlog.Int("completed", completed), mlog.Int("failed", failed))
		metrics.ImportTasks.WithLabelValues(taskType.String(), datapb.ImportTaskStateV2_Pending.String()).Set(float64(pending))
		metrics.ImportTasks.WithLabelValues(taskType.String(), datapb.ImportTaskStateV2_InProgress.String()).Set(float64(inProgress))
		metrics.ImportTasks.WithLabelValues(taskType.String(), datapb.ImportTaskStateV2_Completed.String()).Set(float64(completed))
		metrics.ImportTasks.WithLabelValues(taskType.String(), datapb.ImportTaskStateV2_Failed.String()).Set(float64(failed))
	}
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ReshardTaskType))
	logFunc(tasks, ReshardTaskType)
	tasks = c.importMeta.GetTaskBy(c.ctx, WithType(ImportTaskV3Type))
	logFunc(tasks, ImportTaskV3Type)
}

// checkPendingJob is the first V3 state. It restores existing ReshardTasks,
// fills missing source bins, and advances the job once every bin has a task.
func (c *importCheckerV3) checkPendingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	if err := c.createV3ReshardTasks(job); err != nil {
		log.Warn(c.ctx, "create import v3 reshard tasks failed", mlog.Err(err))
		if !merr.IsRetryableErr(err) {
			_ = c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(err.Error()))
		}
	}
}

// checkReshardJob waits for every ReshardTask of the job to reach the
// catalog-verified Completed marker, then advances to Planning. Empty input
// jobs (zero total rows) shortcut to Uncommitted/Completed exactly like the
// legacy import path does for empty preimports.
func (c *importCheckerV3) checkReshardJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	completed, totalRows, err := c.summarizeV3ReshardResults(job)
	if err != nil {
		log.Warn(c.ctx, "validate import v3 reshard results failed", mlog.Err(err))
		if !merr.IsRetryableErr(err) {
			_ = c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(err.Error()))
		}
		return
	}
	if !completed {
		return
	}
	if totalRows == 0 {
		state := internalpb.ImportJobState_Uncommitted
		if job.GetAutoCommit() {
			state = internalpb.ImportJobState_Completed
		}
		if err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(state)); err != nil {
			log.Warn(c.ctx, "finish empty import v3 job failed", mlog.Err(err))
		}
		return
	}
	if err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Planning)); err != nil {
		log.Warn(c.ctx, "advance import v3 job to Planning failed", mlog.Err(err))
	}
}

func (c *importCheckerV3) checkPlanningJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	if err := c.planV3Job(job); err != nil {
		log.Warn(c.ctx, "plan import v3 job failed", mlog.Err(err))
		if !merr.IsRetryableErr(err) {
			_ = c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(err.Error()))
		}
	}
}

func (c *importCheckerV3) checkImportingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	currentSchemaVersion, err := validateImportV3Schema(c.meta, job.GetCollectionID(), job.GetSchema())
	if err != nil {
		log.Warn(c.ctx, "import v3 schema changed incompatibly before index build", mlog.Err(err))
		_ = c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(err.Error()))
		return
	}
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ImportTaskV3Type), WithJob(job.GetJobID()))
	if len(tasks) == 0 {
		return
	}
	for _, task := range tasks {
		if task.GetState() != datapb.ImportTaskStateV2_Completed {
			return
		}
	}
	if err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_IndexBuilding)); err != nil {
		log.Warn(c.ctx, "advance import v3 job to IndexBuilding failed", mlog.Err(err))
		return
	}
	for _, task := range tasks {
		for _, segmentID := range task.(*importTaskV3).task.Load().GetSegments() {
			segment := c.meta.GetSegment(c.ctx, segmentID)
			if segment != nil && segment.GetNumOfRows() > 0 && segment.GetSchemaVersion() != currentSchemaVersion {
				if err := c.meta.UpdateSegmentsInfo(c.ctx, updateImportV3SchemaVersion(segmentID, currentSchemaVersion)); err != nil {
					log.Warn(c.ctx, "update import v3 segment schema version failed", mlog.Err(err))
					return
				}
			}
		}
	}
}

func (c *importCheckerV3) checkIndexBuildingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(ImportTaskV3Type), WithJob(job.GetJobID()))
	segmentIDs := make([]int64, 0)
	for _, task := range tasks {
		for _, segmentID := range task.(*importTaskV3).task.Load().GetSegments() {
			segment := c.meta.GetHealthySegment(c.ctx, segmentID)
			if segment == nil || segment.GetNumOfRows() == 0 {
				continue
			}
			segmentIDs = append(segmentIDs, segmentID)
		}
	}
	healthySegments := c.meta.GetSegments(segmentIDs, isSegmentHealthy)
	unindexed := c.meta.indexMeta.GetUnindexedSegments(job.GetCollectionID(), healthySegments)
	if Params.DataCoordCfg.WaitForIndex.GetAsBool() && len(unindexed) > 0 {
		for _, segmentID := range unindexed {
			select {
			case getBuildIndexChSingleton() <- segmentID:
			default:
			}
		}
		return
	}
	if err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Uncommitted)); err != nil {
		log.Warn(c.ctx, "advance import v3 job to Uncommitted failed", mlog.Err(err))
	}
}

// checkUncommittedJob handles jobs in the Uncommitted state.
// If auto_commit=true, it triggers a commit via broadcastCommitImportMessage.
// If auto_commit=false, it waits for an explicit CommitImport RPC from the platform.
func (c *importCheckerV3) checkUncommittedJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	if !job.GetAutoCommit() {
		// Wait for explicit CommitImport from the replication platform.
		return
	}
	// auto_commit=true: trigger commit by broadcasting the WAL message.
	// Repeated invocations across ticks are safe: the broadcaster's exclusive
	// collection-level resource-key lock serializes overlapping broadcasts, the
	// ack callback only transitions when the job is still Uncommitted, and
	// HandleCommitVchannel is idempotent on committed_vchannels.
	if c.hooks.commitImport == nil {
		log.Error(c.ctx, "commit hook is nil but auto_commit=true; this is a programming error")
		return
	}
	if err := c.hooks.commitImport(c.ctx, job); err != nil {
		log.Warn(c.ctx, "auto-commit broadcast failed, will retry on next tick", mlog.Err(err))
	}
}

// checkCommittingJob handles jobs in the Committing state.
// Once all vchannels have acknowledged the commit fence, the job transitions to Completed.
func (c *importCheckerV3) checkCommittingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	// When Vchannels is empty, len == len is trivially true. This handles the degenerate
	// case of a zero-channel import (e.g., empty collection); proceed to Completed immediately.
	if len(job.GetCommittedVchannels()) < len(job.GetVchannels()) {
		return // still waiting for remaining vchannels
	}
	completeTime := time.Now().Format("2006-01-02T15:04:05Z07:00")
	if err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(),
		UpdateJobState(internalpb.ImportJobState_Completed),
		UpdateJobCompleteTime(completeTime),
	); err != nil {
		log.Warn(c.ctx, "failed to transition Committing to Completed", mlog.Err(err))
		return
	}
	totalDuration := job.GetTR().ElapseSpan()
	metrics.ImportJobLatency.WithLabelValues(metrics.TotalLabel).Observe(float64(totalDuration.Milliseconds()))
	log.Info(c.ctx, "import job Committing done, all vchannels committed",
		mlog.Duration("jobTimeCost/total", totalDuration))
}

func (c *importCheckerV3) checkFailedJob(job ImportJob) {
	c.tryFailingTasks(job)
}

func (c *importCheckerV3) tryFailingTasks(job ImportJob) {
	tasks := c.importMeta.GetTaskBy(c.ctx, WithJob(job.GetJobID()), WithStates(datapb.ImportTaskStateV2_None, datapb.ImportTaskStateV2_Pending,
		datapb.ImportTaskStateV2_InProgress, datapb.ImportTaskStateV2_Completed, datapb.ImportTaskStateV2_Retry))
	if len(tasks) == 0 {
		return
	}
	mlog.Warn(c.ctx, "Import job has failed, all tasks with the same jobID will be marked as failed",
		mlog.FieldJobID(job.GetJobID()), mlog.String("reason", job.GetReason()))
	for _, task := range tasks {
		err := c.importMeta.UpdateTask(c.ctx, task.GetTaskID(), UpdateState(datapb.ImportTaskStateV2_Failed),
			UpdateReason(job.GetReason()))
		if err != nil {
			mlog.Warn(c.ctx, "failed to update import task state to failed", WrapTaskLog(task, mlog.Err(err))...)
			continue
		}
	}
}

func (c *importCheckerV3) tryTimeoutJob(job ImportJob) {
	if job.GetState() == internalpb.ImportJobState_Failed ||
		job.GetState() == internalpb.ImportJobState_Completed ||
		job.GetState() == internalpb.ImportJobState_Committing {
		return
	}
	timeoutTime := tsoutil.PhysicalTime(job.GetTimeoutTs())
	if time.Now().After(timeoutTime) {
		mlog.Warn(c.ctx, "Import timeout, expired the specified time limit",
			mlog.FieldJobID(job.GetJobID()), mlog.Time("timeoutTime", timeoutTime))
		err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed),
			UpdateJobReason("import timeout"))
		if err != nil {
			mlog.Warn(c.ctx, "failed to update job state to Failed", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
		}
	}
}

func (c *importCheckerV3) checkCollection(collectionID int64, jobs []ImportJob) {
	if len(jobs) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(c.ctx, 10*time.Second)
	defer cancel()
	has, err := c.broker.HasCollection(ctx, collectionID)
	if err != nil {
		mlog.Warn(c.ctx, "verify existence of collection failed", mlog.Int64("collection", collectionID), mlog.Err(err))
		return
	}
	if !has {
		jobs = lo.Filter(jobs, func(job ImportJob, _ int) bool {
			return job.GetState() != internalpb.ImportJobState_Failed &&
				job.GetState() != internalpb.ImportJobState_Completed &&
				job.GetState() != internalpb.ImportJobState_Committing
		})
		for _, job := range jobs {
			err = c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Failed),
				UpdateJobReason(fmt.Sprintf("collection %d dropped", collectionID)))
			if err != nil {
				mlog.Warn(c.ctx, "failed to update job state to Failed", mlog.FieldJobID(job.GetJobID()), mlog.Err(err))
			}
		}
	}
}

func (c *importCheckerV3) checkGC(job ImportJob) {
	if job.GetState() != internalpb.ImportJobState_Completed &&
		job.GetState() != internalpb.ImportJobState_Failed {
		return
	}
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))

	// A terminal job reached before the GC record existed (or a crash between the
	// terminal job save and the record save) is back-filled idempotently here.
	record, err := c.importMeta.GetImportJobGCRecord(c.ctx, job.GetJobID())
	if err != nil {
		log.Warn(c.ctx, "get import job GC record failed", mlog.Err(err))
		return
	}
	if record == nil {
		if err := c.importMeta.EnterTerminalAndInitGC(c.ctx, job.GetJobID()); err != nil {
			log.Warn(c.ctx, "initialize import job GC record failed", mlog.Err(err))
			return
		}
		record = &datapb.ImportJobGCRecord{JobId: job.GetJobID(), State: datapb.ImportJobGCState_ImportJobGCStateQuiesce}
	}

	switch record.GetState() {
	case datapb.ImportJobGCState_ImportJobGCStateQuiesce:
		c.quiesceImportJob(job, log)
	case datapb.ImportJobGCState_ImportJobGCStateRetain:
		if time.Now().After(tsoutil.PhysicalTime(job.GetCleanupTs())) {
			if err := c.importMeta.AdvanceImportJobGCState(c.ctx, job.GetJobID(),
				datapb.ImportJobGCState_ImportJobGCStateRetain, datapb.ImportJobGCState_ImportJobGCStateDelete); err != nil {
				log.Warn(c.ctx, "advance import job GC state to Delete failed", mlog.Err(err))
			}
		}
	case datapb.ImportJobGCState_ImportJobGCStateDelete:
		c.deleteImportJob(job, log)
	}
}

// quiesceImportJob releases the CDC peer (for a failed 2PC import) and removes
// tasks that are no longer pinned to a node. Only when every task is removed is
// the GC record advanced to Retain, which starts the retention window.
func (c *importCheckerV3) quiesceImportJob(job ImportJob, log *mlog.Logger) {
	tasks := c.importMeta.GetTaskBy(c.ctx, WithJob(job.GetJobID()))
	shouldRetain := true
	for _, task := range tasks {
		if task.GetNodeID() != NullNodeID {
			shouldRetain = false
			continue
		}
		if job.GetState() == internalpb.ImportJobState_Failed && task.GetType() == ImportTaskV3Type {
			segmentIDs := task.(*importTaskV3).task.Load().GetSegments()
			if len(segmentIDs) > 0 {
				if err := c.meta.UpdateSegmentsInfo(c.ctx, dropImportV3Segments(segmentIDs, false)); err != nil {
					log.Warn(c.ctx, "drop import v3 segments during failed job GC", WrapTaskLog(task, mlog.Err(err))...)
					shouldRetain = false
					continue
				}
			}
		}
		if err := c.importMeta.RemoveTask(c.ctx, task.GetTaskID()); err != nil {
			log.Warn(c.ctx, "remove task failed during GC", WrapTaskLog(task, mlog.Err(err))...)
			shouldRetain = false
			continue
		}
		log.Info(c.ctx, "task removed during GC quiesce", WrapTaskLog(task)...)
	}
	if !shouldRetain {
		return
	}
	// In a CDC replicating cluster, a failed 2PC source import must release the
	// peer cluster's replicated Uncommitted job before we drop it — otherwise the
	// peer is stranded with invisible imported segments and no recovery path, since
	// source GC never touches the peer. Removal of the job is itself the idempotency
	// guard: once gone we never re-broadcast. Auto-commit jobs have no 2PC peer to
	// release, so they skip the gate entirely.
	if c.hooks.rollbackImport != nil && c.hooks.isReplicatingCluster != nil &&
		job.GetState() == internalpb.ImportJobState_Failed && !job.GetAutoCommit() {
		// The check reaches the streaming balancer future, which blocks until the
		// balancer is registered — under the server-lifetime c.ctx that would park
		// the GC loop during the window before streamingcoord registers
		// it (e.g. a restart recovering a job already past retention). Bound it like
		// checkCollection does; a timeout is just another indeterminate status.
		replicateCheckCtx, cancel := context.WithTimeout(c.ctx, 10*time.Second)
		replicating, err := c.hooks.isReplicatingCluster(replicateCheckCtx)
		cancel()
		switch {
		case err != nil:
			log.Warn(c.ctx, "cannot determine replication status before GC of failed import job, will retry", mlog.Err(err))
			return
		case replicating:
			rollbackCtx, rollbackCancel := context.WithTimeout(c.ctx, 10*time.Second)
			err := c.hooks.rollbackImport(rollbackCtx, job)
			rollbackCancel()
			if err != nil && !isPermanentRollbackErr(err) {
				log.Warn(c.ctx, "failed to broadcast rollback before GC of failed replicate import job, will retry", mlog.Err(err))
				return
			}
			log.Info(c.ctx, "proceeding with GC of failed replicate import job after rollback attempt")
		}
	}
	if err := c.importMeta.AdvanceImportJobGCState(c.ctx, job.GetJobID(),
		datapb.ImportJobGCState_ImportJobGCStateQuiesce, datapb.ImportJobGCState_ImportJobGCStateRetain); err != nil {
		log.Warn(c.ctx, "advance import job GC state to Retain failed", mlog.Err(err))
	}
}

// deleteImportJob removes the job's temporary OSS prefix, then the remaining
// task catalog entries, the job, and finally the GC record. Each step is
// idempotent so a crash in between resumes from the same Delete state.
func (c *importCheckerV3) deleteImportJob(job ImportJob, log *mlog.Logger) {
	prefix := metautil.BuildImportV3JobPath(job.GetJobID()) + "/"
	if err := c.meta.chunkManager.RemoveWithPrefix(c.ctx, prefix); err != nil {
		log.Warn(c.ctx, "remove import job temporary objects failed", mlog.Err(err))
		return
	}
	tasks := c.importMeta.GetTaskBy(c.ctx, WithJob(job.GetJobID()))
	for _, task := range tasks {
		if err := c.importMeta.RemoveTask(c.ctx, task.GetTaskID()); err != nil {
			log.Warn(c.ctx, "remove task failed during GC delete", WrapTaskLog(task, mlog.Err(err))...)
			return
		}
	}
	if err := c.importMeta.RemoveJob(c.ctx, job.GetJobID()); err != nil {
		log.Warn(c.ctx, "remove import job failed", mlog.Err(err))
		return
	}
	if err := c.importMeta.DropImportJobGCRecord(c.ctx, job.GetJobID()); err != nil {
		log.Warn(c.ctx, "drop import job GC record failed", mlog.Err(err))
	}
	log.Info(c.ctx, "import job removed")
}
