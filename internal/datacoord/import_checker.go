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

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/broker"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/replicateutil"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type ImportChecker interface {
	Start()
	Close()
}

// importCheckerHooks bundles the coordinator callbacks the import checker invokes,
// injected as one named unit (instead of a growing positional-arg list) so the checker
// does not depend on *Server. A nil callback disables the corresponding behavior; tests
// inject only the hooks they exercise.
type importCheckerHooks struct {
	// commitImport broadcasts a CommitImport WAL message. Required in production; a nil
	// value is a programming error only when reached on the auto_commit=true path.
	commitImport func(ctx context.Context, job ImportJob) error
	// rollbackImport broadcasts a RollbackImport WAL message. nil disables GC self-heal.
	rollbackImport func(ctx context.Context, job ImportJob) error
	// isReplicatingCluster reports whether this cluster is currently replicating. A
	// non-nil error means the status is indeterminate (e.g. a transient balancer error
	// during shutdown) and the caller must not make an irreversible GC decision. nil hook
	// is treated as "not replicating" (GC self-heal disabled).
	isReplicatingCluster func(ctx context.Context) (bool, error)
	// replicationRole reports this cluster's live replication role and whether it is
	// replicating at all. A non-nil error means the role is indeterminate and the caller
	// must NOT allocate (a secondary that allocated would diverge from the primary's
	// authoritative range). nil hook is treated as "not replicating → primary" so tests
	// that inject only assignImportIDRange proceed to allocate/broadcast.
	replicationRole func(ctx context.Context) (role replicateutil.Role, replicating bool, err error)
	// assignImportIDRange allocates the exact per-file PK ranges and broadcasts the
	// ImportIDRange WAL message. Required in production for autoID (non-backup, non-L0)
	// imports; a nil value disables the two-phase range assignment and is only valid in
	// tests that do not exercise the PreImporting range gate.
	assignImportIDRange func(ctx context.Context, job ImportJob, fileRows []int64) error
}

type importChecker struct {
	ctx        context.Context
	meta       *meta
	broker     broker.Broker
	alloc      allocator.Allocator
	importMeta ImportMeta
	ci         CompactionInspector
	handler    Handler

	hooks importCheckerHooks

	// pkRangeWaitLogged throttles the "waiting for ImportIDRange" logs so a stalled
	// secondary does not emit one every state-machine tick (~2s). Keyed by jobID; only
	// ever touched from the single state-machine goroutine (checkAssigningIDRangeJob
	// via ensurePKRanges), so no lock is needed.
	pkRangeWaitLogged map[int64]time.Time

	closeOnce sync.Once
	closeChan chan struct{}
}

func NewImportChecker(ctx context.Context,
	meta *meta,
	broker broker.Broker,
	alloc allocator.Allocator,
	importMeta ImportMeta,
	ci CompactionInspector,
	handler Handler,
	hooks importCheckerHooks,
) ImportChecker {
	return &importChecker{
		ctx:        ctx,
		meta:       meta,
		broker:     broker,
		alloc:      alloc,
		importMeta: importMeta,
		ci:         ci,
		handler:    handler,
		hooks:      hooks,
		closeChan:  make(chan struct{}),

		pkRangeWaitLogged: make(map[int64]time.Time),
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
func (c *importChecker) Start() {
	mlog.Info(c.ctx, "start import checker")
	go c.runGCLoop()
	c.runStateMachineLoop()
}

func (c *importChecker) runStateMachineLoop() {
	ticker := time.NewTicker(Params.DataCoordCfg.ImportCheckIntervalHigh.GetAsDuration(time.Second)) // 2s
	defer ticker.Stop()
	for {
		select {
		case <-c.closeChan:
			mlog.Info(c.ctx, "import checker state-machine loop exited")
			return
		case <-ticker.C:
			jobs := c.importMeta.GetJobBy(c.ctx)
			for _, job := range jobs {
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
				case internalpb.ImportJobState_PreImporting:
					c.checkPreImportingJob(job)
				case internalpb.ImportJobState_AssigningIDRange:
					c.checkAssigningIDRangeJob(job)
				case internalpb.ImportJobState_Importing:
					c.checkImportingJob(job)
				case internalpb.ImportJobState_Sorting:
					c.checkSortingJob(job)
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

func (c *importChecker) runGCLoop() {
	ticker := time.NewTicker(Params.DataCoordCfg.ImportCheckIntervalLow.GetAsDuration(time.Second)) // 2min
	defer ticker.Stop()
	for {
		select {
		case <-c.closeChan:
			mlog.Info(c.ctx, "import checker gc loop exited")
			return
		case <-ticker.C:
			jobs := c.importMeta.GetJobBy(c.ctx)
			for _, job := range jobs {
				c.tryTimeoutJob(job)
				c.checkGC(job)
			}
			jobsByColl := lo.GroupBy(jobs, func(job ImportJob) int64 {
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

func (c *importChecker) Close() {
	c.closeOnce.Do(func() {
		close(c.closeChan)
	})
}

func (c *importChecker) LogJobStats(jobs []ImportJob) {
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

func (c *importChecker) LogTaskStats() {
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
	tasks := c.importMeta.GetTaskBy(c.ctx, WithType(PreImportTaskType))
	logFunc(tasks, PreImportTaskType)
	tasks = c.importMeta.GetTaskBy(c.ctx, WithType(ImportTaskType))
	logFunc(tasks, ImportTaskType)
}

func (c *importChecker) getLackFilesForPreImports(job ImportJob) []*internalpb.ImportFile {
	lacks := lo.KeyBy(job.GetFiles(), func(file *internalpb.ImportFile) int64 {
		return file.GetId()
	})
	exists := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(PreImportTaskType))
	for _, task := range exists {
		for _, file := range task.GetFileStats() {
			delete(lacks, file.GetImportFile().GetId())
		}
	}
	return lo.Values(lacks)
}

func (c *importChecker) getLackFilesForImports(job ImportJob) []*datapb.ImportFileStats {
	preimports := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(PreImportTaskType))
	lacks := make(map[int64]*datapb.ImportFileStats, 0)
	for _, t := range preimports {
		for _, stat := range t.GetFileStats() {
			lacks[stat.GetImportFile().GetId()] = stat
		}
	}
	exists := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(ImportTaskType))
	for _, task := range exists {
		for _, file := range task.GetFileStats() {
			delete(lacks, file.GetImportFile().GetId())
		}
	}
	return lo.Values(lacks)
}

func (c *importChecker) checkPendingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	lacks := c.getLackFilesForPreImports(job)
	if len(lacks) == 0 {
		return
	}
	fileGroups := lo.Chunk(lacks, Params.DataCoordCfg.FilesPerPreImportTask.GetAsInt())

	newTasks, err := NewPreImportTasks(fileGroups, job, c.alloc, c.importMeta)
	if err != nil {
		log.Warn(c.ctx, "new preimport tasks failed", mlog.Err(err))
		return
	}
	for _, t := range newTasks {
		err = c.importMeta.AddTask(c.ctx, t)
		if err != nil {
			log.Warn(c.ctx, "add preimport task failed", WrapTaskLog(t, mlog.Err(err))...)
			return
		}
		log.Info(c.ctx, "add new preimport task", WrapTaskLog(t, mlog.Any("fileStats", t.GetFileStats()))...)
	}

	err = c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_PreImporting))
	if err != nil {
		log.Warn(c.ctx, "failed to update job state to PreImporting", mlog.Err(err))
		return
	}
	pendingDuration := job.GetTR().RecordSpan()
	metrics.ImportJobLatency.WithLabelValues(metrics.ImportStagePending).Observe(float64(pendingDuration.Milliseconds()))
	log.Info(c.ctx, "import job start to execute", mlog.Duration("jobTimeCost/pending", pendingDuration))
}

func (c *importChecker) checkPreImportingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))

	preimports := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(PreImportTaskType))
	totalRows := int64(0)
	for _, t := range preimports {
		if t.GetState() != datapb.ImportTaskStateV2_Completed {
			// Preimport tasks are not fully completed, thus generating imports should not be triggered.
			return
		}
		totalRows += lo.SumBy(t.GetFileStats(), func(stat *datapb.ImportFileStats) int64 {
			return stat.GetTotalRows()
		})
	}

	updateJobState := func(state internalpb.ImportJobState, actions ...UpdateJobAction) {
		actions = append(actions, UpdateJobState(state))
		err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), actions...)
		if err != nil {
			log.Warn(c.ctx, "failed to update job state to Importing", mlog.Err(err))
			return
		}
		preImportDuration := job.GetTR().RecordSpan()
		metrics.ImportJobLatency.WithLabelValues(metrics.ImportStagePreImport).Observe(float64(preImportDuration.Milliseconds()))
		log.Info(c.ctx, "import job preimport done", mlog.String("state", state.String()), mlog.Duration("jobTimeCost/preimport", preImportDuration))
	}
	// The PreImport stage span ends here: every exit below records it, including the
	// PreImporting → AssigningIDRange transition. The wait inside AssigningIDRange is
	// intentionally not given its own ImportStage metric label; the AssigningIDRange →
	// Importing transition only advances the TimeRecorder (see checkAssigningIDRangeJob)
	// so the wait is attributed to neither stage.

	if totalRows == 0 {
		if job.GetAutoCommit() {
			// auto-commit: no data to import, skip Uncommitted directly to Completed
			log.Info(c.ctx, "no data to import, auto_commit=true, transitioning directly to Completed")
			updateJobState(internalpb.ImportJobState_Completed)
		} else {
			// replication cluster: surface Uncommitted so platform can observe and commit
			log.Info(c.ctx, "no data to import, auto_commit=false, transitioning to Uncommitted")
			updateJobState(internalpb.ImportJobState_Uncommitted)
		}
		return
	}

	// Two-phase autoID PK-range gate: an autoID import (not backup, not L0) must carry the
	// exact per-file PK ranges in its job meta before any Import task is created. If
	// preimport is done but the ranges are not present yet, move to AssigningIDRange —
	// the primary broadcasts (and the secondary waits for) the ImportIDRange message
	// there. The gate is symmetric across clusters, so the arrival order of "all
	// preimport completed" and "ranges applied" does not matter. Jobs whose ranges are
	// not needed or already set (legacy ImportMsg-carried ranges, non-autoID, backup,
	// L0) never enter AssigningIDRange and fall through to the tail below.
	if c.needsPKRanges(job) && !c.pkRangesSet(job) {
		updateJobState(internalpb.ImportJobState_AssigningIDRange)
		return
	}

	c.finishRangedPreimport(job, preimports, updateJobState)
}

// checkAssigningIDRangeJob drives a job waiting for its exact per-file autoID PK ranges
// (entered from checkPreImportingJob when preimport completed before the ranges were
// applied). Restart recovery is state-agnostic: a job persisted in AssigningIDRange is
// picked up here by the state-machine switch. The job stays in AssigningIDRange,
// bounded by its timeoutTs (see tryTimeoutJob), until the ImportIDRange ack callback
// applies the ranges.
func (c *importChecker) checkAssigningIDRangeJob(job ImportJob) {
	preimports := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(PreImportTaskType))
	for _, t := range preimports {
		if t.GetState() != datapb.ImportTaskStateV2_Completed {
			// Preimport must be fully completed before Import tasks are created;
			// without this, the tail below would regroup only a subset of files.
			return
		}
	}

	if !c.pkRangesSet(job) {
		c.ensurePKRanges(job, preimports)
		return
	}

	// Ranges applied: run the standard tail. The transition only advances the
	// TimeRecorder past the wait — it must NOT emit another PreImport stage metric,
	// that span already ended at the PreImporting → AssigningIDRange transition.
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	updateJobState := func(state internalpb.ImportJobState, actions ...UpdateJobAction) {
		actions = append(actions, UpdateJobState(state))
		err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), actions...)
		if err != nil {
			log.Warn(c.ctx, "failed to update job state to Importing", mlog.Err(err))
			return
		}
		waitDuration := job.GetTR().RecordSpan()
		log.Info(c.ctx, "import job id range assigned", mlog.String("state", state.String()), mlog.Duration("jobTimeCost/assigningIDRange", waitDuration))
	}
	c.finishRangedPreimport(job, preimports, updateJobState)
}

// finishRangedPreimport runs the post-gate tail shared by checkPreImportingJob (jobs whose
// ranges were already set without the new state: legacy ImportMsg-carried ranges,
// non-autoID, backup, L0) and checkAssigningIDRangeJob (exact-range jobs whose
// ImportIDRange broadcast has been applied): exact-marker divergence validation first,
// then lacks/disk-quota/regroup/NewImportTasks → Importing. The caller supplies
// updateJobState so each entry state records its own stage span.
func (c *importChecker) finishRangedPreimport(job ImportJob, preimports []ImportTask, updateJobState func(state internalpb.ImportJobState, actions ...UpdateJobAction)) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))

	// Post-gate validation for exact-range jobs: the local exact row count must equal the
	// reserved range size for every file, checked before any segment is written. A mismatch
	// means the file content differs between the cluster that allocated the range and this
	// one — the precondition CDC import already requires operators to guarantee — so fail
	// loudly on the diverging side instead of silently assigning divergent PKs.
	if job.GetPkRangesExact() {
		if fileID, localRows, reserved, ok := c.validateExactPKRanges(job, preimports); !ok {
			log.Warn(c.ctx, "ImportIDRange divergence: local row count differs from the reserved exact PK range; failing import — files must be identical across clusters",
				mlog.Int64("fileID", fileID), mlog.Int64("localRows", localRows), mlog.Int64("reserved", reserved))
			updateJobState(internalpb.ImportJobState_Failed, UpdateJobReason(fmt.Sprintf(
				"import file %d row count %d does not match the exactly reserved PK range size %d (cross-cluster file divergence)",
				fileID, localRows, reserved)))
			return
		}
	}

	lacks := c.getLackFilesForImports(job)
	if len(lacks) == 0 {
		return
	}

	// Stamp the authoritative per-file ranges onto the task fileStats about to be regrouped
	// into Import tasks, so they persist inside ImportTaskV2 meta and flow through
	// AssembleImportRequest into the datanode's pkCursor. Harmless no-op when already equal
	// (legacy jobs carried the range on the ImportMsg).
	c.stampPKRangesOntoStats(job, lacks)

	requestSize, err := CheckDiskQuota(c.ctx, job, c.meta, c.importMeta)
	if err != nil {
		log.Warn(c.ctx, "import failed, disk quota exceeded", mlog.Err(err))
		updateJobState(internalpb.ImportJobState_Failed, UpdateJobReason(err.Error()))
		return
	}

	segmentMaxSize := GetSegmentMaxSize(job, c.meta)
	groups := RegroupImportFiles(job, lacks, segmentMaxSize)
	newTasks, err := NewImportTasks(groups, job, c.alloc, c.meta, c.importMeta, segmentMaxSize)
	if err != nil {
		log.Warn(c.ctx, "new import tasks failed", mlog.Err(err))
		return
	}
	for _, t := range newTasks {
		err = c.importMeta.AddTask(c.ctx, t)
		if err != nil {
			log.Warn(c.ctx, "add new import task failed", WrapTaskLog(t, mlog.Err(err))...)
			updateJobState(internalpb.ImportJobState_Failed, UpdateJobReason(err.Error()))
			return
		}
		log.Info(c.ctx, "add new import task", WrapTaskLog(t, mlog.Any("fileStats", t.GetFileStats()))...)
	}

	updateJobState(internalpb.ImportJobState_Importing, UpdateRequestedDiskSize(requestSize))
}

// needsPKRanges reports whether the job requires per-file autoID PK ranges: its primary
// key is autoID and it is neither a backup (which keeps embedded PKs) nor an L0 import
// (which carries no autoID PK). This is the same predicate the removed broadcast-time
// sizing path used, evaluated here from the persisted job meta. A schema without a
// resolvable primary key is left to normal validation (no ranges).
func (c *importChecker) needsPKRanges(job ImportJob) bool {
	pkField, err := typeutil.GetPrimaryFieldSchema(job.GetSchema())
	if err != nil {
		return false
	}
	return pkField.GetAutoID() &&
		!importutilv2.IsBackup(job.GetOptions()) &&
		!importutilv2.IsL0Import(job.GetOptions())
}

// pkRangesSet reports whether every job file already carries a reserved PK range.
func (c *importChecker) pkRangesSet(job ImportJob) bool {
	return jobPKRangesSet(job)
}

// jobPKRangesSet reports whether every job file has a non-nil reserved PK range. It is a
// nil check, NOT End>Begin: a zero-row file legitimately carries an empty (Begin==End)
// range that still counts as set.
func jobPKRangesSet(job ImportJob) bool {
	for _, f := range job.GetFiles() {
		if f.GetPreAllocatedAutoIds() == nil {
			return false
		}
	}
	return true
}

// ensurePKRanges triggers (primary / non-replicating cluster) or waits for (secondary) the
// ImportIDRange broadcast that populates the job's per-file PK ranges. Called from the
// AssigningIDRange state when the ranges are unset; the job stays in AssigningIDRange
// either way, bounded by its timeoutTs.
func (c *importChecker) ensurePKRanges(job ImportJob, preimports []ImportTask) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))

	// Bound the whole ensure step. The broadcast blocks on the ctx-insensitive resource-key
	// lock and on per-vchannel WAL appends; under the server-lifetime c.ctx a stalled WAL or
	// unavailable streamingnode would park the state-machine loop. Same pattern as checkGC:
	// a timeout is just another transient status → retry next tick.
	ctx, cancel := context.WithTimeout(c.ctx, 10*time.Second)
	defer cancel()

	// Decide the replication role. A nil hook means the feature is disabled (tests): treat
	// as a non-replicating primary and proceed to allocate/broadcast. An indeterminate
	// (error) role must NOT allocate — a secondary that allocated would diverge from the
	// primary's authoritative range — so wait and retry next tick.
	var role replicateutil.Role
	replicating := false
	if c.hooks.replicationRole != nil {
		r, rep, err := c.hooks.replicationRole(ctx)
		if err != nil {
			log.Warn(ctx, "cannot determine replication role before ImportIDRange broadcast, will retry next tick", mlog.Err(err))
			return
		}
		role, replicating = r, rep
	}

	if replicating && role == replicateutil.RoleSecondary {
		// Secondary: the primary broadcasts ImportIDRange and it is replicated here; that
		// ack callback applies the ranges. Do NOT allocate locally (it would diverge from
		// the primary's authoritative range). Rate-limit the log so a stalled secondary
		// does not spam every ~2s tick.
		c.logPKRangeWaitThrottled(job, "waiting for replicated ImportIDRange")
		return
	}

	if c.hooks.assignImportIDRange == nil {
		log.Error(ctx, "assignImportIDRange hook is nil but autoID import requires PK ranges; this is a programming error")
		return
	}

	// Build the exact per-file row counts, aligned with job.GetFiles() order, from the
	// completed preimport stats. A missing stat is an internal error (preimport is fully
	// completed before the gate); retry next tick.
	fileRows, ok := c.exactFileRows(job, preimports)
	if !ok {
		return
	}

	log.Info(ctx, "triggering ImportIDRange broadcast",
		mlog.Int("fileCount", len(job.GetFiles())), mlog.Int64("totalRows", lo.Sum(fileRows)))

	if err := c.hooks.assignImportIDRange(ctx, job, fileRows); err != nil {
		if errors.Is(err, broadcaster.ErrNotPrimary) {
			// Stale role during switchover: this cluster thought it was primary but the
			// broadcaster rejected the append. Treat as "wait" — the real primary broadcasts
			// the authoritative range and it is replicated here. Retry next tick.
			log.Info(ctx, "role flipped to standby while importing, waiting for replicated ImportIDRange")
		} else {
			log.Warn(ctx, "ImportIDRange broadcast failed, will retry next tick", mlog.Err(err))
		}
		return
	}
}

// exactFileRows returns the exact per-file row counts aligned with job.GetFiles() order,
// summed from the completed preimport task stats by fileID. A file with no stat is an
// internal error — preimport is fully completed before the gate — and ok=false tells the
// caller to retry next tick.
func (c *importChecker) exactFileRows(job ImportJob, preimports []ImportTask) ([]int64, bool) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	rowsByFile := make(map[int64]int64)
	for _, t := range preimports {
		for _, stat := range t.GetFileStats() {
			rowsByFile[stat.GetImportFile().GetId()] += stat.GetTotalRows()
		}
	}
	files := job.GetFiles()
	fileRows := make([]int64, len(files))
	for i, f := range files {
		rows, ok := rowsByFile[f.GetId()]
		if !ok {
			log.Warn(c.ctx, "preimport stats missing for an import file, will retry next tick", mlog.Int64("fileID", f.GetId()))
			return nil, false
		}
		fileRows[i] = rows
	}
	return fileRows, true
}

// validateExactPKRanges checks each file's local exact row count against the reserved PK
// range size for an exact-range job. It returns the first divergence (fileID, localRows,
// reserved) with ok=false; ok=true means every file matches its reservation exactly.
func (c *importChecker) validateExactPKRanges(job ImportJob, preimports []ImportTask) (fileID, localRows, reserved int64, ok bool) {
	reservedByFile := make(map[int64]*commonpb.IDRange, len(job.GetFiles()))
	for _, f := range job.GetFiles() {
		reservedByFile[f.GetId()] = f.GetPreAllocatedAutoIds()
	}
	for _, t := range preimports {
		for _, stat := range t.GetFileStats() {
			id := stat.GetImportFile().GetId()
			r := reservedByFile[id]
			size := r.GetEnd() - r.GetBegin()
			if stat.GetTotalRows() != size {
				return id, stat.GetTotalRows(), size, false
			}
		}
	}
	return 0, 0, 0, true
}

// stampPKRangesOntoStats copies the job's authoritative per-file PK ranges onto the task
// fileStats (by fileID) so they persist into ImportTaskV2 meta and flow through
// AssembleImportRequest. An already-equal range is left untouched (idempotent), so this is
// a harmless no-op for legacy jobs that carried the range on the ImportMsg.
func (c *importChecker) stampPKRangesOntoStats(job ImportJob, lacks []*datapb.ImportFileStats) {
	rangeByFile := make(map[int64]*commonpb.IDRange, len(job.GetFiles()))
	for _, f := range job.GetFiles() {
		if r := f.GetPreAllocatedAutoIds(); r != nil {
			rangeByFile[f.GetId()] = r
		}
	}
	for _, stat := range lacks {
		importFile := stat.GetImportFile()
		if importFile == nil {
			continue
		}
		r, ok := rangeByFile[importFile.GetId()]
		if !ok {
			continue
		}
		cur := importFile.GetPreAllocatedAutoIds()
		if cur.GetBegin() == r.GetBegin() && cur.GetEnd() == r.GetEnd() {
			continue
		}
		importFile.PreAllocatedAutoIds = r
	}
}

// importIDRangeWaitLogInterval throttles repeated "waiting for ImportIDRange" logs.
const importIDRangeWaitLogInterval = 30 * time.Second

// logPKRangeWaitThrottled logs a waiting-for-range state at Info at most once per
// importIDRangeWaitLogInterval per job, and at Debug in between, so a stalled secondary
// does not emit an Info line on every ~2s state-machine tick. Called only from the single
// state-machine goroutine, so the throttle map needs no lock.
func (c *importChecker) logPKRangeWaitThrottled(job ImportJob, msg string) {
	now := time.Now()
	if last, seen := c.pkRangeWaitLogged[job.GetJobID()]; seen && now.Sub(last) < importIDRangeWaitLogInterval {
		mlog.Debug(c.ctx, msg, mlog.FieldJobID(job.GetJobID()))
		return
	}
	c.pkRangeWaitLogged[job.GetJobID()] = now
	mlog.Info(c.ctx, msg, mlog.FieldJobID(job.GetJobID()))
}

func (c *importChecker) checkImportingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	tasks := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(ImportTaskType), WithRequestSource())
	for _, t := range tasks {
		if t.GetState() != datapb.ImportTaskStateV2_Completed {
			return
		}
	}
	err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Sorting))
	if err != nil {
		log.Warn(c.ctx, "failed to update job state to Stats", mlog.Err(err))
		return
	}
	importDuration := job.GetTR().RecordSpan()
	metrics.ImportJobLatency.WithLabelValues(metrics.ImportStageImport).Observe(float64(importDuration.Milliseconds()))
	log.Info(c.ctx, "import job import done", mlog.Duration("jobTimeCost/import", importDuration))
}

func (c *importChecker) checkSortingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	updateJobState := func(state internalpb.ImportJobState, reason string) {
		err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(state), UpdateJobReason(reason))
		if err != nil {
			log.Warn(c.ctx, "failed to update job state", mlog.Err(err))
			return
		}
		statsDuration := job.GetTR().RecordSpan()
		metrics.ImportJobLatency.WithLabelValues(metrics.ImportStageStats).Observe(float64(statsDuration.Milliseconds()))
		log.Info(c.ctx, "import job stats done", mlog.String("state", state.String()), mlog.Duration("jobTimeCost/stats", statsDuration))
	}

	// Skip stats stage if not enable stats or is l0 import.
	if !enableSortCompaction() ||
		importutilv2.IsL0Import(job.GetOptions()) {
		updateJobState(internalpb.ImportJobState_IndexBuilding, "")
		return
	}

	// Check and trigger stats tasks.
	var (
		taskCnt = 0
		doneCnt = 0
	)
	tasks := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(ImportTaskType))
	for _, task := range tasks {
		originSegmentIDs := task.(*importTask).GetSegmentIDs()
		sortSegmentIDs := task.(*importTask).GetSortedSegmentIDs()
		taskCnt += len(originSegmentIDs)
		for i, originSegmentID := range originSegmentIDs {
			logger := mlog.With(WrapTaskLog(task, mlog.Int64("origin", originSegmentID), mlog.Int64("target", sortSegmentIDs[i]))...)
			originSegment := c.meta.GetHealthySegment(c.ctx, originSegmentID)
			targetSegment := c.meta.GetHealthySegment(c.ctx, sortSegmentIDs[i])
			if originSegment == nil {
				// import zero num rows segment
				doneCnt++
				continue
			}
			if targetSegment != nil {
				// sort compaction is already done
				doneCnt++
				continue
			}
			// if not compacting, trigger sort compaction task
			isCompacting := c.meta.IsSegmentCompacting(originSegmentID)
			if !isCompacting {
				compactionTask, err := createSortCompactionTask(c.ctx, task, originSegment, sortSegmentIDs[i], c.meta, c.handler, c.alloc)
				if err != nil {
					logger.Warn(c.ctx, "create sort compaction task failed", mlog.Err(err))
					continue
				}
				if compactionTask == nil {
					logger.Info(c.ctx, "maybe it no need to create sort compaction task")
					doneCnt++
					continue
				}
				err = c.ci.enqueueCompaction(compactionTask)
				if err != nil {
					logger.Warn(c.ctx, "sort compaction task enqueue failed", mlog.Err(err))
					continue
				}
				logger.Info(c.ctx, "create sort compaction task and enqueue success")
			}
		}
	}

	// All segments are stats-ed. Update job state to `IndexBuilding`.
	if taskCnt == doneCnt {
		updateJobState(internalpb.ImportJobState_IndexBuilding, "")
	}
}

func (c *importChecker) checkIndexBuildingJob(job ImportJob) {
	log := mlog.With(mlog.FieldJobID(job.GetJobID()))
	tasks := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithType(ImportTaskType))
	originSegmentIDs := lo.FlatMap(tasks, func(t ImportTask, _ int) []int64 {
		return t.(*importTask).GetSegmentIDs()
	})
	statsSegmentIDs := lo.FlatMap(tasks, func(t ImportTask, _ int) []int64 {
		return t.(*importTask).GetSortedSegmentIDs()
	})

	targetSegmentIDs := statsSegmentIDs
	if !enableSortCompaction() {
		targetSegmentIDs = originSegmentIDs
	}

	healthySegments := c.meta.GetSegments(targetSegmentIDs, isSegmentHealthy)
	unindexed := c.meta.indexMeta.GetUnindexedSegments(job.GetCollectionID(), healthySegments)
	if Params.DataCoordCfg.WaitForIndex.GetAsBool() && len(unindexed) > 0 && !importutilv2.IsL0Import(job.GetOptions()) {
		for _, segmentID := range unindexed {
			select {
			case getBuildIndexChSingleton() <- segmentID: // accelerate index building:
			default:
			}
		}
		log.Debug(c.ctx, "waiting for import segments building index...", mlog.Int64s("unindexed", unindexed))
		return
	}
	buildIndexDuration := job.GetTR().RecordSpan()
	metrics.ImportJobLatency.WithLabelValues(metrics.ImportStageBuildIndex).Observe(float64(buildIndexDuration.Milliseconds()))
	log.Info(c.ctx, "import job build index done", mlog.Duration("jobTimeCost/buildIndex", buildIndexDuration))

	// 2PC: hand off to Uncommitted regardless of auto_commit. Segment visibility
	// (is_importing=false) is cleared only by HandleCommitVchannel after the WAL
	// commit fence is processed per vchannel; auto_commit=true jobs are then
	// driven through the commit broadcast by checkUncommittedJob.
	err := c.importMeta.UpdateJob(c.ctx, job.GetJobID(), UpdateJobState(internalpb.ImportJobState_Uncommitted))
	if err != nil {
		log.Warn(c.ctx, "failed to update job state to Uncommitted", mlog.Err(err))
		return
	}
	LogResultSegmentsInfo(job.GetJobID(), c.meta, targetSegmentIDs)
	log.Info(c.ctx, "import job indexes built, transitioned to Uncommitted",
		mlog.Bool("autoCommit", job.GetAutoCommit()))
}

// checkUncommittedJob handles jobs in the Uncommitted state.
// If auto_commit=true, it triggers a commit via broadcastCommitImportMessage.
// If auto_commit=false, it waits for an explicit CommitImport RPC from the platform.
func (c *importChecker) checkUncommittedJob(job ImportJob) {
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
func (c *importChecker) checkCommittingJob(job ImportJob) {
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

func (c *importChecker) checkFailedJob(job ImportJob) {
	c.tryFailingTasks(job)
}

func (c *importChecker) tryFailingTasks(job ImportJob) {
	tasks := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID(), WithStates(datapb.ImportTaskStateV2_Pending,
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

func (c *importChecker) tryTimeoutJob(job ImportJob) {
	switch job.GetState() {
	case internalpb.ImportJobState_Failed, internalpb.ImportJobState_Completed,
		internalpb.ImportJobState_Committing:
		// Fast path on this tick's snapshot only; UpdateJobState enforces the
		// rule against the current state under importMeta's lock.
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

func (c *importChecker) checkCollection(collectionID int64, jobs []ImportJob) {
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
			return job.GetState() != internalpb.ImportJobState_Failed && job.GetState() != internalpb.ImportJobState_Completed
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

func (c *importChecker) checkGC(job ImportJob) {
	if job.GetState() != internalpb.ImportJobState_Completed &&
		job.GetState() != internalpb.ImportJobState_Failed {
		return
	}
	cleanupTime := tsoutil.PhysicalTime(job.GetCleanupTs())
	if time.Now().After(cleanupTime) {
		log := mlog.With(mlog.FieldJobID(job.GetJobID()))
		GCRetention := Params.DataCoordCfg.ImportTaskRetention.GetAsDuration(time.Second)
		log.Info(c.ctx, "job has reached the GC retention",
			mlog.Time("cleanupTime", cleanupTime), mlog.Duration("GCRetention", GCRetention))
		tasks := c.importMeta.GetTaskByJob(c.ctx, job.GetJobID())
		shouldRemoveJob := true
		for _, task := range tasks {
			if job.GetState() == internalpb.ImportJobState_Failed && task.GetType() == ImportTaskType {
				if len(task.(*importTask).GetSegmentIDs()) != 0 || len(task.(*importTask).GetSortedSegmentIDs()) != 0 {
					shouldRemoveJob = false
					continue
				}
			}
			if task.GetNodeID() != NullNodeID {
				shouldRemoveJob = false
				continue
			}
			err := c.importMeta.RemoveTask(c.ctx, task.GetTaskID())
			if err != nil {
				log.Warn(c.ctx, "remove task failed during GC", WrapTaskLog(task, mlog.Err(err))...)
				shouldRemoveJob = false
				continue
			}
			log.Info(c.ctx, "reached GC retention, task removed", WrapTaskLog(task)...)
		}
		if !shouldRemoveJob {
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
				// Indeterminate replication status (e.g. a transient balancer error during
				// shutdown, when streamingcoord stops before datacoord). Removing the job now
				// could strand a replicating peer's Uncommitted job with no recovery path,
				// which is irreversible — a false "not replicating" costs nothing but a retry,
				// so keep the job and re-evaluate on the next GC tick.
				log.Warn(c.ctx, "cannot determine replication status before GC of failed import job, will retry", mlog.Err(err))
				return
			case replicating:
				// Broadcast the RollbackImport to release the peer. A transient error keeps
				// the job to retry next tick; a permanent error (standby ErrNotPrimary, or the
				// collection was dropped — itself a replicated DDL, so the peer fails its own
				// job independently) falls through to GC, since retrying it forever would leak
				// the job's metadata.
				//
				// Bound the broadcast like the replication check above: it blocks in
				// BlockUntilDone until every vchannel append succeeds, and under the
				// server-lifetime c.ctx an unavailable streamingnode would park this
				// loop until shutdown. A timeout is just another transient status —
				// keep the job and retry on the next GC tick. The resource-key lock on
				// the broadcast path is still ctx-insensitive (making it fail-fast is a
				// follow-up), which is one reason this GC loop runs on its own
				// goroutine (see Start): even an unbounded park here can only delay
				// GC, never the import state machine.
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
		err := c.importMeta.RemoveJob(c.ctx, job.GetJobID())
		if err != nil {
			log.Warn(c.ctx, "remove import job failed", mlog.Err(err))
			return
		}
		log.Info(c.ctx, "import job removed")
	}
}

// isPermanentRollbackErr reports whether a RollbackImport broadcast error is permanent,
// i.e. retrying it can never succeed, so the failed job should still be GC'd rather than
// retried forever (which would leak its metadata). Everything else is treated as transient
// and retried on the next GC tick — misclassifying a transient error as permanent would
// drop a replicating job without releasing the peer, which is irreversible.
func isPermanentRollbackErr(err error) bool {
	// ErrNotPrimary: this cluster is a replication standby, not the primary that owns the
	// broadcast; its own failed job is independent and safe to drop.
	// ErrCollectionNotFound: the collection was dropped. DropCollection is itself a
	// replicated DDL, so the peer marks its own import job Failed independently — there is
	// no peer left to release, and the broadcast can never succeed.
	// errRollbackImportNoVchannels: the job carries no vchannels (fixed at creation), so
	// the broadcast has no peer to address and can never succeed.
	return errors.Is(err, broadcaster.ErrNotPrimary) || errors.Is(err, merr.ErrCollectionNotFound) ||
		errors.Is(err, errRollbackImportNoVchannels)
}
