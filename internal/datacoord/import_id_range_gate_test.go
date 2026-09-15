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

// Tests for the two-phase autoID PK-range gate in checkPreImportingJob
// (needsPKRanges / pkRangesSet / ensurePKRanges / exactFileRows /
// validateExactPKRanges / stampPKRangesOntoStats) and the UpdateJobPKRanges
// action. The exact allocation itself (assignExactPKRanges) is covered by
// import_id_range_test.go; the ImportIDRange ack callback by
// ddl_callbacks_import_test.go.

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/replicateutil"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

// ---------------------------------------------------------------------------
// Helpers (newCapturingImportMeta is shared with ddl_callbacks_import_test.go)
// ---------------------------------------------------------------------------

// newCapturingImportMeta builds an importMeta whose catalog records every proto
// handed to SaveImportJob, so tests can assert what was actually persisted.
func newCapturingImportMeta(t *testing.T, saved *[]*datapb.ImportJob) ImportMeta {
	catalog := mocks.NewDataCoordCatalog(t)
	catalog.EXPECT().ListImportJobs(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListPreImportTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().ListImportTasks(mock.Anything).Return(nil, nil)
	catalog.EXPECT().SaveImportJob(mock.Anything, mock.Anything).RunAndReturn(
		func(_ context.Context, job *datapb.ImportJob) error {
			*saved = append(*saved, job)
			return nil
		}).Maybe()
	alloc := allocator.NewMockAllocator(t)
	importMeta, err := NewImportMeta(context.Background(), catalog, alloc, nil)
	require.NoError(t, err)
	return importMeta
}

// enableGateMocks sets the recurring catalog/allocator expectations the gate
// tests need: preimport task saves (pending -> completed) and open-ended AllocN.
func (s *ImportCheckerSuite) enableGateMocks() {
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	catalog.EXPECT().SavePreImportTask(mock.Anything, mock.Anything).Return(nil)
	s.alloc.EXPECT().AllocN(mock.Anything).RunAndReturn(func(n int64) (int64, int64, error) {
		id := rand.Int63()
		return id, id + n, nil
	}).Maybe() // tests that add tasks manually never reach the allocator
}

// driveToPreImportDone runs checkPendingJob for the given job (creating one
// preimport task per file group) and completes every task with exact per-file
// row counts, reproducing the state the gate observes in production: all
// PreImport tasks completed, TotalRows present in each ImportFileStats.
func (s *ImportCheckerSuite) driveToPreImportDone(jobID int64, rowsByFile map[int64]int64) {
	ctx := context.TODO()
	s.checker.checkPendingJob(s.importMeta.GetJob(ctx, jobID))
	s.Equal(internalpb.ImportJobState_PreImporting, s.importMeta.GetJob(ctx, jobID).GetState())

	tasks := s.importMeta.GetTaskByJob(ctx, jobID, WithType(PreImportTaskType))
	s.NotEmpty(tasks)
	for _, t := range tasks {
		// Replace the task's fileStats with fresh protos carrying the exact counts,
		// like the datanode report does (also un-aliasing them from job.GetFiles()).
		stats := make([]*datapb.ImportFileStats, 0, len(t.GetFileStats()))
		for _, st := range t.GetFileStats() {
			f := st.GetImportFile()
			stats = append(stats, &datapb.ImportFileStats{
				ImportFile: &internalpb.ImportFile{Id: f.GetId(), Paths: f.GetPaths()},
				TotalRows:  rowsByFile[f.GetId()],
			})
		}
		s.NoError(s.importMeta.UpdateTask(ctx, t.GetTaskID(),
			UpdateState(datapb.ImportTaskStateV2_Completed), UpdateFileStats(stats)))
	}
}

// setupAutoIDPreImportDone flips the suite job's PK to autoID and drives it to
// "all preimport tasks completed" with the given exact per-file row counts
// (keyed by the suite job's fileIDs 1, 2, 3).
func (s *ImportCheckerSuite) setupAutoIDPreImportDone(rowsByFile map[int64]int64) {
	s.manuallyUpdateJob(s.jobID, func(job ImportJob) {
		job.(*importJob).Schema.Fields[0].AutoID = true
	})
	s.driveToPreImportDone(s.jobID, rowsByFile)
}

// gateTestSchema builds a PK-only schema; autoID toggles the gate predicate.
func gateTestSchema(autoID bool) *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      100,
				Name:         "pk",
				DataType:     schemapb.DataType_Int64,
				IsPrimaryKey: true,
				AutoID:       autoID,
			},
		},
	}
}

// ---------------------------------------------------------------------------
// Gate: predicate helpers
// ---------------------------------------------------------------------------

func (s *ImportCheckerSuite) TestNeedsPKRanges() {
	cases := []struct {
		name    string
		schema  *schemapb.CollectionSchema
		options []*commonpb.KeyValuePair
		want    bool
	}{
		{"autoID", gateTestSchema(true), nil, true},
		{"non-autoID", gateTestSchema(false), nil, false},
		{"autoID backup", gateTestSchema(true),
			[]*commonpb.KeyValuePair{{Key: importutilv2.BackupFlag, Value: "true"}}, false},
		{"autoID backup=false", gateTestSchema(true),
			[]*commonpb.KeyValuePair{{Key: importutilv2.BackupFlag, Value: "false"}}, true},
		{"autoID L0", gateTestSchema(true),
			[]*commonpb.KeyValuePair{{Key: importutilv2.L0Import, Value: "true"}}, false},
		{"no primary key", &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "vec", DataType: schemapb.DataType_FloatVector},
		}}, nil, false},
	}
	for _, tc := range cases {
		job := &importJob{ImportJob: &datapb.ImportJob{Schema: tc.schema, Options: tc.options}}
		s.Equal(tc.want, s.checker.needsPKRanges(job), tc.name)
	}
}

// pkRangesSet is a nil check, not End>Begin: a zero-row file legitimately
// carries an empty (Begin==End) range that still counts as set.
func TestJobPKRangesSet_NilCheckNotEmptyCheck(t *testing.T) {
	job := &importJob{ImportJob: &datapb.ImportJob{Files: []*internalpb.ImportFile{
		{Id: 1, PreAllocatedAutoIds: &commonpb.IDRange{Begin: 5, End: 5}}, // empty range counts as set
		{Id: 2},
	}}}
	assert.False(t, jobPKRangesSet(job))
	job.GetFiles()[1].PreAllocatedAutoIds = &commonpb.IDRange{Begin: 5, End: 9}
	assert.True(t, jobPKRangesSet(job))
}

// ---------------------------------------------------------------------------
// Gate: primary path (Test Plan item 1)
// ---------------------------------------------------------------------------

// Preimport done, no ranges, primary role (or non-replicating, or role hook
// disabled): the first tick moves the job PreImporting → AssigningIDRange without
// broadcasting; later ticks in AssigningIDRange invoke assignImportIDRange with the
// exact per-file row counts aligned to job.GetFiles() order -- including a zero-row
// file -- under a bounded ctx, and the job stays in AssigningIDRange until the ranges
// are applied.
func (s *ImportCheckerSuite) TestPKRangeGate_PrimaryAssignsWithAlignedFileRows() {
	s.enableGateMocks()
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	calls := 0
	var gotJobID int64
	var gotFileRows []int64
	var deadlineOK bool
	s.checker.hooks.assignImportIDRange = func(ctx context.Context, job ImportJob, fileRows []int64) error {
		calls++
		gotJobID = job.GetJobID()
		gotFileRows = fileRows
		deadline, hasDeadline := ctx.Deadline()
		deadlineOK = hasDeadline && time.Until(deadline) > 0 && time.Until(deadline) <= 10*time.Second
		return nil
	}

	// Tick 1: the transition itself never consults the role nor broadcasts.
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		s.Fail("PreImporting → AssigningIDRange must not consult the replication role")
		return replicateutil.RolePrimary, false, nil
	}
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(0, calls, "the transition tick must not broadcast")
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())
	s.Empty(s.importMeta.GetJob(context.TODO(), s.jobID).GetReason())

	// A nil replicationRole hook means "not replicating -> primary" (the
	// feature-disabled default); an explicit primary role -- replicating or not
	// -- takes the same path.
	roles := []struct {
		name string
		set  func()
	}{
		{"nil role hook", func() { s.checker.hooks.replicationRole = nil }},
		{"not replicating", func() {
			s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
				return replicateutil.RolePrimary, false, nil
			}
		}},
		{"replicating primary", func() {
			s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
				return replicateutil.RolePrimary, true, nil
			}
		}},
	}
	for i, r := range roles {
		r.set()
		s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
		s.Equal(i+1, calls, r.name+": each tick without applied ranges retries the broadcast")
		s.Equal(s.jobID, gotJobID)
		// Aligned with job.GetFiles() order (fileIDs 1,2,3), zero-row file included.
		s.Equal([]int64{100, 0, 200}, gotFileRows, r.name)
		s.True(deadlineOK, r.name+": hook ctx must be bounded (~10s)")

		job := s.importMeta.GetJob(context.TODO(), s.jobID)
		s.Equal(internalpb.ImportJobState_AssigningIDRange, job.GetState(), r.name)
		s.Empty(job.GetReason(), r.name)
		s.Equal(0, len(s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType))),
			r.name+": no Import task may be created before the ranges are applied")
	}
}

// ---------------------------------------------------------------------------
// Gate: secondary path (Test Plan item 2)
// ---------------------------------------------------------------------------

// A secondary never allocates: it waits for the replicated ImportIDRange. Once
// the ack callback (simulated here via UpdateJobPKRanges) applies the primary's
// ranges, the next tick passes the gate: Import tasks are created and the
// ranges are stamped onto the task fileStats (ImportTaskV2 meta carries
// PreAllocatedAutoIds; the job carries the exact marker).
func (s *ImportCheckerSuite) TestPKRangeGate_SecondaryWaitsThenProceeds() {
	s.enableGateMocks()
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil)
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	roleCalls, assignCalls := 0, 0
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		roleCalls++
		return replicateutil.RoleSecondary, true, nil
	}
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	// Tick 1: PreImporting → AssigningIDRange, without consulting the role.
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(0, roleCalls, "the transition tick must not consult the replication role")
	s.Equal(0, assignCalls)
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	// Tick 2: secondary waits -- no allocation, no broadcast, stays AssigningIDRange.
	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(1, roleCalls)
	s.Equal(0, assignCalls, "a secondary must never allocate")
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	// Tick 3: still waiting (throttled-log path); nothing changes.
	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(0, assignCalls)
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())
	s.Equal(0, len(s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType))))

	// The replicated ImportIDRange ack applies the primary's exact ranges
	// (what importIDRangeAckCallback does via UpdateJobPKRanges).
	ranges := []*commonpb.IDRange{
		{Begin: 5000, End: 5100}, // file 1: 100 rows
		{Begin: 5100, End: 5100}, // file 2: zero-row file, empty range
		{Begin: 5100, End: 5300}, // file 3: 200 rows
	}
	s.NoError(s.importMeta.UpdateJob(context.TODO(), s.jobID, UpdateJobPKRanges(ranges)))

	// Tick 3: gate passes -> Import tasks created, ranges stamped onto task fileStats.
	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(0, assignCalls, "ranges arrived via replication; no local allocation ever")
	s.Equal(2, roleCalls, "gate satisfied -> role is no longer consulted")

	job := s.importMeta.GetJob(context.TODO(), s.jobID)
	s.Equal(internalpb.ImportJobState_Importing, job.GetState())
	s.True(job.GetPkRangesExact())

	importTasks := s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType))
	s.NotEmpty(importTasks)
	wantByFileID := map[int64]*commonpb.IDRange{1: ranges[0], 2: ranges[1], 3: ranges[2]}
	seen := make(map[int64]bool)
	for _, t := range importTasks {
		for _, stat := range t.GetFileStats() {
			f := stat.GetImportFile()
			want := wantByFileID[f.GetId()]
			s.NotNil(want, "task file %d must belong to the job", f.GetId())
			got := f.GetPreAllocatedAutoIds()
			s.NotNil(got, "range must be stamped onto ImportTaskV2 meta for file %d", f.GetId())
			s.Equal(want.GetBegin(), got.GetBegin(), "file %d", f.GetId())
			s.Equal(want.GetEnd(), got.GetEnd(), "file %d", f.GetId())
			seen[f.GetId()] = true
		}
	}
	for fileID := range wantByFileID {
		s.True(seen[fileID], "every job file must appear in some Import task")
	}
}

// ---------------------------------------------------------------------------
// Gate: indeterminate role (Test Plan item 3)
// ---------------------------------------------------------------------------

// An indeterminate replication role must NOT allocate -- a secondary that
// allocated would diverge from the primary's authoritative range. No broadcast,
// job waits in AssigningIDRange (reached from PreImporting without consulting the
// role at all).
func (s *ImportCheckerSuite) TestPKRangeGate_IndeterminateRoleWaits() {
	s.enableGateMocks()
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	assignCalls := 0
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		return replicateutil.RolePrimary, false, errors.New("balancer not ready")
	}
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(0, assignCalls, "an indeterminate role must never reach allocation")
	job := s.importMeta.GetJob(context.TODO(), s.jobID)
	s.Equal(internalpb.ImportJobState_AssigningIDRange, job.GetState())
	s.Empty(job.GetReason())
}

// ---------------------------------------------------------------------------
// Gate: assign hook errors are transient (Test Plan item 4)
// ---------------------------------------------------------------------------

// ErrNotPrimary (stale role during switchover) is tolerated as "wait"; any
// other error is logged and retried next tick. Neither fails the job nor
// creates Import tasks.
func (s *ImportCheckerSuite) TestPKRangeGate_AssignErrorsAreTransient() {
	s.enableGateMocks()
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	calls := 0
	hookErr := error(nil)
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		calls++
		return hookErr
	}

	// Reach the waiting state first; the errors below all happen on AssigningIDRange ticks.
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	errCases := []struct {
		name string
		err  error
	}{
		{"ErrNotPrimary tolerated as wait", broadcaster.ErrNotPrimary},
		{"wrapped ErrNotPrimary", errors.Wrap(broadcaster.ErrNotPrimary, "append rejected")},
		{"generic error retried", errors.New("wal unavailable")},
	}
	for _, tc := range errCases {
		hookErr = tc.err
		s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
		job := s.importMeta.GetJob(context.TODO(), s.jobID)
		s.Equal(internalpb.ImportJobState_AssigningIDRange, job.GetState(), tc.name)
		s.Empty(job.GetReason(), tc.name+": a transient broadcast error must not fail the job")
		s.Equal(0, len(s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType))), tc.name)
	}
	s.Equal(len(errCases), calls, "every tick must retry the broadcast")
}

// ---------------------------------------------------------------------------
// Gate: legacy job with pre-set ranges (Test Plan item 5)
// ---------------------------------------------------------------------------

// A job created from an old-format ImportMsg already carries (upper-bound)
// ranges on its files and no exact marker: the gate is satisfied immediately,
// the role hook is never consulted, nothing is broadcast, and the job proceeds
// to Importing with the legacy ranges untouched (rows < reserved is legal).
func (s *ImportCheckerSuite) TestPKRangeGate_LegacyRangedJobSkipsBroadcast() {
	s.enableGateMocks()
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil)
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	// Legacy upper-bound ranges (oversized, expansion factor baked in), no marker.
	s.manuallyUpdateJob(s.jobID, func(job ImportJob) {
		files := job.(*importJob).Files
		files[0].PreAllocatedAutoIds = &commonpb.IDRange{Begin: 1000, End: 1400} // 100 rows, 400 reserved
		files[1].PreAllocatedAutoIds = &commonpb.IDRange{Begin: 1400, End: 1400} // zero-row file
		files[2].PreAllocatedAutoIds = &commonpb.IDRange{Begin: 1400, End: 2200} // 200 rows, 800 reserved
	})

	roleCalls, assignCalls := 0, 0
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		roleCalls++
		return replicateutil.RolePrimary, false, nil
	}
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(0, assignCalls, "a legacy ranged job must never broadcast")
	s.Equal(0, roleCalls, "gate satisfied -> role never consulted")

	job := s.importMeta.GetJob(context.TODO(), s.jobID)
	s.Equal(internalpb.ImportJobState_Importing, job.GetState())
	s.False(job.GetPkRangesExact(), "legacy jobs keep the > guard: no exact marker")

	// The oversized legacy ranges are stamped unchanged onto the task fileStats.
	importTasks := s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType))
	s.NotEmpty(importTasks)
	wantByFileID := map[int64]*commonpb.IDRange{
		1: {Begin: 1000, End: 1400},
		2: {Begin: 1400, End: 1400},
		3: {Begin: 1400, End: 2200},
	}
	for _, t := range importTasks {
		for _, stat := range t.GetFileStats() {
			f := stat.GetImportFile()
			got := f.GetPreAllocatedAutoIds()
			s.NotNil(got, "file %d", f.GetId())
			s.Equal(wantByFileID[f.GetId()].GetBegin(), got.GetBegin(), "file %d", f.GetId())
			s.Equal(wantByFileID[f.GetId()].GetEnd(), got.GetEnd(), "file %d", f.GetId())
		}
	}
}

// ---------------------------------------------------------------------------
// Gate: excluded jobs proceed without the hook (Test Plan item 6)
// ---------------------------------------------------------------------------

// Non-autoID, backup, and L0 jobs do not need PK ranges: each proceeds
// straight to Importing without the role or assign hooks ever firing.
func (s *ImportCheckerSuite) TestPKRangeGate_ExcludedJobsProceedWithoutHook() {
	s.enableGateMocks()
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil)

	roleCalls, assignCalls := 0, 0
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		roleCalls++
		return replicateutil.RolePrimary, false, nil
	}
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	variants := []struct {
		name    string
		autoID  bool
		options []*commonpb.KeyValuePair
	}{
		{"non-autoID", false, nil},
		{"backup", true, []*commonpb.KeyValuePair{{Key: importutilv2.BackupFlag, Value: "true"}}},
		{"L0", true, []*commonpb.KeyValuePair{{Key: importutilv2.L0Import, Value: "true"}}},
	}
	for i, v := range variants {
		jobID := int64(900 + i)
		job := &importJob{
			ImportJob: &datapb.ImportJob{
				JobID:        jobID,
				CollectionID: 1,
				PartitionIDs: []int64{2},
				Vchannels:    []string{"ch0"},
				State:        internalpb.ImportJobState_Pending,
				TimeoutTs:    1000,
				Schema:       gateTestSchema(v.autoID),
				Options:      v.options,
				Files: []*internalpb.ImportFile{
					{Id: 1, Paths: []string{"a.json"}},
					{Id: 2, Paths: []string{"b.json"}},
				},
			},
			tr: timerecord.NewTimeRecorder("import job"),
		}
		s.NoError(s.importMeta.AddJob(context.TODO(), job))
		s.driveToPreImportDone(jobID, map[int64]int64{1: 10, 2: 20})

		s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), jobID))
		s.Equal(internalpb.ImportJobState_Importing,
			s.importMeta.GetJob(context.TODO(), jobID).GetState(), v.name)
	}
	s.Equal(0, assignCalls, "excluded jobs must never allocate/broadcast ranges")
	s.Equal(0, roleCalls, "excluded jobs must never consult the replication role")
}

// ---------------------------------------------------------------------------
// Gate: totalRows == 0 (Test Plan item 7)
// ---------------------------------------------------------------------------

// A zero-row autoID job takes the existing empty-import branch before the gate:
// no range is needed and nothing is broadcast, regardless of auto_commit.
func (s *ImportCheckerSuite) TestPKRangeGate_ZeroRowsSkipsRangeAssignment() {
	s.enableGateMocks()
	roleCalls, assignCalls := 0, 0
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		roleCalls++
		return replicateutil.RolePrimary, false, nil
	}
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	// auto_commit=false -> Uncommitted (2PC surface), no broadcast.
	s.manuallyUpdateJob(s.jobID, func(job ImportJob) {
		job.(*importJob).Schema.Fields[0].AutoID = true
		job.(*importJob).AutoCommit = false
	})
	s.driveToPreImportDone(s.jobID, map[int64]int64{1: 0, 2: 0, 3: 0})
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_Uncommitted, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	// auto_commit=true -> Completed, no broadcast.
	jobID := int64(950)
	job := &importJob{
		ImportJob: &datapb.ImportJob{
			JobID:        jobID,
			CollectionID: 1,
			PartitionIDs: []int64{2},
			Vchannels:    []string{"ch0"},
			State:        internalpb.ImportJobState_Pending,
			TimeoutTs:    1000,
			AutoCommit:   true,
			Schema:       gateTestSchema(true),
			Files: []*internalpb.ImportFile{
				{Id: 1, Paths: []string{"a.json"}},
			},
		},
		tr: timerecord.NewTimeRecorder("import job"),
	}
	s.NoError(s.importMeta.AddJob(context.TODO(), job))
	s.driveToPreImportDone(jobID, map[int64]int64{1: 0})
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), jobID))
	s.Equal(internalpb.ImportJobState_Completed, s.importMeta.GetJob(context.TODO(), jobID).GetState())

	s.Equal(0, assignCalls, "a zero-row job needs no range and must never broadcast")
	s.Equal(0, roleCalls)
}

// ---------------------------------------------------------------------------
// Gate: post-gate divergence check (Test Plan item 8)
// ---------------------------------------------------------------------------

// An exact-range job whose local row count differs from the reserved range size
// (cross-cluster file divergence) fails loudly before any Import task exists,
// with both numbers in the reason. The validation now fires from
// checkAssigningIDRangeJob: the job first waits in AssigningIDRange, then fails
// once the divergent ranges are applied.
func (s *ImportCheckerSuite) TestPKRangeGate_ExactDivergenceFailsJob() {
	s.enableGateMocks()
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	// The job must fail before any Import task exists, so this must NOT fire.
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil).Maybe()
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	// Tick 1: preimport done, ranges unset → wait in AssigningIDRange.
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	// Exact ranges applied, but file 3 reserved 150 ids while preimport counted 200 rows.
	s.NoError(s.importMeta.UpdateJob(context.TODO(), s.jobID, UpdateJobPKRanges([]*commonpb.IDRange{
		{Begin: 5000, End: 5100}, // file 1: 100 == 100, ok
		{Begin: 5100, End: 5100}, // file 2: 0 == 0, ok
		{Begin: 5100, End: 5250}, // file 3: 150 != 200, divergence
	})))

	assignCalls := 0
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	// Tick 2: the divergence check fires from the AssigningIDRange state.
	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))

	job := s.importMeta.GetJob(context.TODO(), s.jobID)
	s.Equal(internalpb.ImportJobState_Failed, job.GetState())
	s.Contains(job.GetReason(), "import file 3")
	s.Contains(job.GetReason(), "row count 200")
	s.Contains(job.GetReason(), "range size 150")
	s.Equal(0, assignCalls, "ranges were applied; the divergence is terminal, not a re-broadcast")
	s.Equal(0, len(s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType))),
		"the job must fail before any Import task is created")
}

// ---------------------------------------------------------------------------
// Gate: missing preimport stat (Test Plan item 9)
// ---------------------------------------------------------------------------

// A job file without a completed preimport stat at ensurePKRanges time is an
// internal inconsistency: warn and retry next tick -- no panic, no allocation.
// The transition itself only checks task states, so the gap surfaces on the
// AssigningIDRange tick that builds the file-row picture.
func (s *ImportCheckerSuite) TestPKRangeGate_MissingStatWaits() {
	s.enableGateMocks()
	s.manuallyUpdateJob(s.jobID, func(job ImportJob) {
		job.(*importJob).Schema.Fields[0].AutoID = true
		job.(*importJob).State = internalpb.ImportJobState_PreImporting
	})

	// One completed preimport task covering only files 1 and 2; the job has file 3 too.
	task := &preImportTask{tr: timerecord.NewTimeRecorder("preimport task")}
	task.task.Store(&datapb.PreImportTask{
		JobID:  s.jobID,
		TaskID: 555,
		State:  datapb.ImportTaskStateV2_Completed,
		FileStats: []*datapb.ImportFileStats{
			{ImportFile: &internalpb.ImportFile{Id: 1, Paths: []string{"a.json"}}, TotalRows: 100},
			{ImportFile: &internalpb.ImportFile{Id: 2, Paths: []string{"b.json"}}, TotalRows: 50},
		},
	})
	s.NoError(s.importMeta.AddTask(context.TODO(), task))

	assignCalls := 0
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	// The lone task is completed, so the transition fires despite the coverage gap.
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())
	s.Equal(0, assignCalls, "the transition tick must not allocate")

	s.NotPanics(func() {
		s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	})
	s.Equal(0, assignCalls, "no allocation with an incomplete file-row picture")
	job := s.importMeta.GetJob(context.TODO(), s.jobID)
	s.Equal(internalpb.ImportJobState_AssigningIDRange, job.GetState())
	s.Empty(job.GetReason())
}

// ---------------------------------------------------------------------------
// Gate: full primary transition PreImporting → AssigningIDRange → Importing
// ---------------------------------------------------------------------------

// End-to-end on the primary side: the transition tick broadcasts nothing, the
// waiting tick broadcasts once, and once the ack callback's ranges are applied
// (simulated via UpdateJobPKRanges) the job reaches Importing with stamped tasks.
func (s *ImportCheckerSuite) TestPKRangeGate_PrimaryFullTransition() {
	s.enableGateMocks()
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil)
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})

	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		return replicateutil.RolePrimary, false, nil
	}
	assignCalls := 0
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}

	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())
	s.Equal(0, assignCalls, "the transition tick must not broadcast")

	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(1, assignCalls, "the waiting tick broadcasts")
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	s.NoError(s.importMeta.UpdateJob(context.TODO(), s.jobID, UpdateJobPKRanges([]*commonpb.IDRange{
		{Begin: 5000, End: 5100},
		{Begin: 5100, End: 5100},
		{Begin: 5100, End: 5300},
	})))
	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(1, assignCalls, "ranges applied; no further broadcast")

	job := s.importMeta.GetJob(context.TODO(), s.jobID)
	s.Equal(internalpb.ImportJobState_Importing, job.GetState())
	s.True(job.GetPkRangesExact())
	s.NotEmpty(s.importMeta.GetTaskByJob(context.TODO(), s.jobID, WithType(ImportTaskType)))
}

// ---------------------------------------------------------------------------
// Gate: restart recovery from the persisted AssigningIDRange state
// ---------------------------------------------------------------------------

// importMeta reload is state-agnostic, so a job persisted in AssigningIDRange is
// picked up by the new state-machine case: it proceeds when ranges are set and
// waits (secondary) when they are not. Constructing the job directly in
// AssigningIDRange reproduces the post-restart snapshot.
func (s *ImportCheckerSuite) TestPKRangeGate_RestartRecovery() {
	s.enableGateMocks()
	catalog := s.importMeta.(*importMeta).catalog.(*mocks.DataCoordCatalog)
	catalog.EXPECT().SaveImportTask(mock.Anything, mock.Anything).Return(nil).Maybe()

	// Recovered with ranges already applied → proceeds to Importing.
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})
	s.NoError(s.importMeta.UpdateJob(context.TODO(), s.jobID, UpdateJobPKRanges([]*commonpb.IDRange{
		{Begin: 5000, End: 5100},
		{Begin: 5100, End: 5100},
		{Begin: 5100, End: 5300},
	})))
	s.manuallyUpdateJob(s.jobID, UpdateJobState(internalpb.ImportJobState_AssigningIDRange))
	s.Equal(internalpb.ImportJobState_AssigningIDRange,
		s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_Importing,
		s.importMeta.GetJob(context.TODO(), s.jobID).GetState(), "a recovered ranged job must proceed")

	// Recovered without ranges (secondary) → keeps waiting, never allocates.
	jobID := int64(960)
	job := &importJob{
		ImportJob: &datapb.ImportJob{
			JobID:        jobID,
			CollectionID: 1,
			PartitionIDs: []int64{2},
			Vchannels:    []string{"ch0"},
			State:        internalpb.ImportJobState_Pending,
			TimeoutTs:    1000,
			Schema:       gateTestSchema(true),
			Files: []*internalpb.ImportFile{
				{Id: 1, Paths: []string{"a.json"}},
				{Id: 2, Paths: []string{"b.json"}},
			},
		},
		tr: timerecord.NewTimeRecorder("import job"),
	}
	s.NoError(s.importMeta.AddJob(context.TODO(), job))
	s.driveToPreImportDone(jobID, map[int64]int64{1: 10, 2: 20})
	s.manuallyUpdateJob(jobID, UpdateJobState(internalpb.ImportJobState_AssigningIDRange))

	assignCalls := 0
	s.checker.hooks.replicationRole = func(context.Context) (replicateutil.Role, bool, error) {
		return replicateutil.RoleSecondary, true, nil
	}
	s.checker.hooks.assignImportIDRange = func(context.Context, ImportJob, []int64) error {
		assignCalls++
		return nil
	}
	s.checker.checkAssigningIDRangeJob(s.importMeta.GetJob(context.TODO(), jobID))
	s.Equal(0, assignCalls, "a recovered secondary must never allocate")
	s.Equal(internalpb.ImportJobState_AssigningIDRange,
		s.importMeta.GetJob(context.TODO(), jobID).GetState())
}

// ---------------------------------------------------------------------------
// Gate: progress bucket for AssigningIDRange
// ---------------------------------------------------------------------------

// AssigningIDRange reports the end-of-preimport bucket (40), coalesced to the
// Importing display state like PreImporting, so GetImportProgress distinguishes
// "preimport running" (< 40) from "waiting for ID range broadcast/replication".
func (s *ImportCheckerSuite) TestPKRangeGate_AssigningIDRangeProgress() {
	s.enableGateMocks()
	s.setupAutoIDPreImportDone(map[int64]int64{1: 100, 2: 0, 3: 200})
	s.checker.checkPreImportingJob(s.importMeta.GetJob(context.TODO(), s.jobID))
	s.Equal(internalpb.ImportJobState_AssigningIDRange, s.importMeta.GetJob(context.TODO(), s.jobID).GetState())

	progress, state, _, _, reason := GetJobProgress(context.TODO(), s.jobID, s.importMeta, s.checker.meta)
	s.Equal(int64(40), progress)
	s.Equal(internalpb.ImportJobState_Importing, state)
	s.Empty(reason)
}

// ---------------------------------------------------------------------------
// UpdateJobPKRanges (Test Plan item 16)
// ---------------------------------------------------------------------------

func TestUpdateJobPKRanges_SetsRangesAndMarkerAndPersists(t *testing.T) {
	ctx := context.Background()
	var saved []*datapb.ImportJob
	importMeta := newCapturingImportMeta(t, &saved)

	job := &importJob{
		ImportJob: &datapb.ImportJob{
			JobID: 1,
			State: internalpb.ImportJobState_PreImporting,
			Files: []*internalpb.ImportFile{{Id: 10}, {Id: 11}},
		},
		tr: timerecord.NewTimeRecorder("test"),
	}
	require.NoError(t, importMeta.AddJob(ctx, job))

	ranges := []*commonpb.IDRange{
		{Begin: 100, End: 110},
		{Begin: 110, End: 110}, // zero-row file keeps its empty range
	}
	require.NoError(t, importMeta.UpdateJob(ctx, 1, UpdateJobPKRanges(ranges)))

	// Survives re-GetJob: ranges by position + exact marker.
	got := importMeta.GetJob(ctx, 1)
	require.NotNil(t, got)
	assert.True(t, got.GetPkRangesExact())
	assert.EqualValues(t, 100, got.GetFiles()[0].GetPreAllocatedAutoIds().GetBegin())
	assert.EqualValues(t, 110, got.GetFiles()[0].GetPreAllocatedAutoIds().GetEnd())
	assert.EqualValues(t, 110, got.GetFiles()[1].GetPreAllocatedAutoIds().GetBegin())
	assert.EqualValues(t, 110, got.GetFiles()[1].GetPreAllocatedAutoIds().GetEnd())

	// Persisted: the proto handed to the catalog carries the ranges and the marker.
	require.NotEmpty(t, saved)
	last := saved[len(saved)-1]
	assert.True(t, last.GetPkRangesExact())
	assert.EqualValues(t, 100, last.GetFiles()[0].GetPreAllocatedAutoIds().GetBegin())
	assert.EqualValues(t, 110, last.GetFiles()[0].GetPreAllocatedAutoIds().GetEnd())
	assert.EqualValues(t, 110, last.GetFiles()[1].GetPreAllocatedAutoIds().GetBegin())
	assert.EqualValues(t, 110, last.GetFiles()[1].GetPreAllocatedAutoIds().GetEnd())
}
