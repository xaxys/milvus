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
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/balancer/balance"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster/registry"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/replicateutil"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
)

// importV1AckCallback handles the ack callback for import messages.
func (c *DDLCallbacks) importV1AckCallback(ctx context.Context, result message.BroadcastResultImportMessageV1) error {
	body := result.Message.MustBody()

	// Ensure Schema.DbName is populated from the broadcast message's DbName,
	// matching the behavior in master where this was set before calling ImportV2.
	if body.Schema != nil {
		body.Schema.DbName = body.DbName
	}

	// Process each vchannel with its own TimeTick (not deprecated MsgBase)
	// Each vchannel gets its own import job with the corresponding TimeTick
	vchannels := make([]string, 0, len(result.Results))
	for vchannel := range result.Results {
		if funcutil.IsControlChannel(vchannel) {
			continue
		}
		vchannels = append(vchannels, vchannel)
	}

	// Call createImportJobFromAck directly instead of ImportV2
	// ImportV2 is only for proxy broadcast, not for ack callback
	importResp, err := c.createImportJobFromAck(ctx, &internalpb.ImportRequestInternal{
		DbID:           0, // already deprecated.
		CollectionID:   body.GetCollectionID(),
		CollectionName: body.GetCollectionName(),
		PartitionIDs:   body.GetPartitionIDs(),
		ChannelNames:   vchannels,
		Schema:         body.GetSchema(),
		Files: lo.Map(body.GetFiles(), func(file *msgpb.ImportFile, _ int) *internalpb.ImportFile {
			// PreAllocatedAutoIds is legacy: only an old-format ImportMsg carries
			// broadcast-time ranges, and in-flight jobs created from one must keep
			// them. New jobs receive their ranges via the ImportIDRange broadcast
			// after preimport.
			return &internalpb.ImportFile{
				Id:                  file.GetId(),
				Paths:               file.GetPaths(),
				PreAllocatedAutoIds: file.GetPreAllocatedAutoIds(),
			}
		}),
		Options:       funcutil.Map2KeyValuePair(body.GetOptions()),
		DataTimestamp: result.GetMaxTimeTick(), // TODO: use per-vchannel TimeTick in future, must be supported for CDC.
		JobID:         body.GetJobID(),
	})

	err = merr.CheckRPCCall(importResp, err)
	if errors.Is(err, merr.ErrCollectionNotFound) {
		mlog.Warn(ctx, "import job creation failed because of collection not found, skip it",
			mlog.Strings("vchannels", vchannels),
			mlog.String("job_id", importResp.GetJobID()), mlog.Err(err))
		return nil
	}
	return err
}

// validateImportRequest validates the import request before broadcasting.
// This includes all validation logic previously done in CheckCallback and Proxy.
//
// All of this runs before the broadcaster's idempotency lookup, which cannot happen
// until the resource keys are held inside Broadcast. A retry therefore has to pass
// these checks again before it can resolve to its original jobID, and not all of them
// are a pure function of the request: ValidateMaxImportJobExceed counts in-flight jobs,
// ValidateBinlogImportRequest lists the backup files in object storage, and
// validateImportReplication reads the replication topology. A retry sent while the job
// limit is saturated -- by the original request among others -- or after the backup
// files or the replication topology changed is rejected here rather than returning the
// original jobID. Retrying the same key once the limit frees up resolves normally;
// minting a fresh key instead is what would import the data twice.
func (s *Server) validateImportRequest(ctx context.Context, files []*msgpb.ImportFile, options []*commonpb.KeyValuePair) error {
	// Validate timeout
	_, err := importutilv2.GetTimeoutTs(options)
	if err != nil {
		return err
	}

	// Validate binlog import files if it's a backup
	if importutilv2.IsBackup(options) {
		err = ValidateBinlogImportRequest(ctx, s.meta.chunkManager, files, options)
		if err != nil {
			return err
		}
	}

	// Validate max import job count
	err = ValidateMaxImportJobExceed(ctx, s.importMeta)
	if err != nil {
		return err
	}

	if err := s.validateImportReplication(ctx, options); err != nil {
		return err
	}

	return nil
}

func (s *Server) validateImportReplication(ctx context.Context, options []*commonpb.KeyValuePair) error {
	balancer, err := balance.GetWithContext(ctx)
	if err != nil {
		return err
	}
	assignment, err := balancer.GetLatestChannelAssignment()
	if err != nil {
		return err
	}
	if assignment == nil {
		return nil
	}
	if !isReplicatingCluster(assignment.ReplicateConfiguration) {
		return nil
	}

	if !paramtable.Get().DataCoordCfg.ImportInReplicatingCluster.GetAsBool() {
		return merr.WrapErrOperationNotSupportedMsg("import in replicating cluster is not supported yet")
	}
	if importutilv2.IsAutoCommit(options) {
		return merr.WrapErrOperationNotSupportedMsg("auto_commit=true import in replicating cluster is not supported")
	}
	return nil
}

func isReplicatingCluster(cfg *commonpb.ReplicateConfiguration) bool {
	return cfg != nil && (len(cfg.GetCrossClusterTopology()) > 0 || len(cfg.GetClusters()) > 1)
}

// isReplicatingClusterNow reports whether this cluster is currently part of a CDC
// replication topology. A non-nil error means the status could not be determined (e.g. a
// transient balancer error, or OnShutdownError while streamingcoord is stopping before
// datacoord); the caller must treat that as indeterminate rather than "not replicating",
// because at GC time a false "not replicating" would irreversibly drop a replicating job
// without releasing the peer. A nil assignment is an unambiguous "not replicating".
func (s *Server) isReplicatingClusterNow(ctx context.Context) (bool, error) {
	balancer, err := balance.GetWithContext(ctx)
	if err != nil {
		return false, err
	}
	assignment, err := balancer.GetLatestChannelAssignment()
	if err != nil {
		return false, err
	}
	if assignment == nil {
		return false, nil
	}
	return isReplicatingCluster(assignment.ReplicateConfiguration), nil
}

// replicationRole reports this cluster's live replication role (primary/secondary) and
// whether it is part of a CDC replication topology at all. It is modeled on
// isReplicatingClusterNow and reads the same balancer channel assignment. A non-nil
// error means the role could not be determined (a transient balancer error, or
// OnShutdownError while streamingcoord stops before datacoord); the caller must treat
// that as indeterminate and NOT allocate, because a secondary that allocated its own
// range would diverge from the primary's authoritative range. A nil assignment or a
// non-replicating configuration is an unambiguous "not replicating" → (RolePrimary,
// false, nil), so a standalone cluster proceeds to allocate and broadcast.
func (s *Server) replicationRole(ctx context.Context) (replicateutil.Role, bool, error) {
	balancer, err := balance.GetWithContext(ctx)
	if err != nil {
		return replicateutil.RolePrimary, false, err
	}
	assignment, err := balancer.GetLatestChannelAssignment()
	if err != nil {
		return replicateutil.RolePrimary, false, err
	}
	if assignment == nil {
		return replicateutil.RolePrimary, false, nil
	}
	cfg := assignment.ReplicateConfiguration
	if !isReplicatingCluster(cfg) {
		return replicateutil.RolePrimary, false, nil
	}
	helper, err := replicateutil.NewConfigHelper(Params.CommonCfg.ClusterPrefix.GetValue(), cfg)
	if err != nil {
		return replicateutil.RolePrimary, false, err
	}
	return helper.GetCurrentCluster().Role(), true, nil
}

// jobIDFromDuplicatedBroadcast recovers the original import jobID from the broadcast
// message the broadcaster returned on an idempotency hit. The broadcaster does not
// know about import-specific structures, so the decode happens here.
//
// The request payload is deliberately NOT compared against the original: keeping the
// key unique per logical request is the client's contract, and enforcing it
// server-side would mean inventing an equality predicate over file lists whose false
// mismatches would reject legitimate retries -- pushing the caller to mint a new key
// and import the data twice, the very outcome this feature exists to prevent.
//
// The collectionID comparison is not such a predicate and is not a semantic guard: the
// idempotency key is scoped to this collection's ID, so a hit already means both
// broadcasts targeted it. It is checked as an invariant, to fail loudly on an encoding
// or scoping bug rather than hand back a jobID for another collection's import.
func jobIDFromDuplicatedBroadcast(msg message.BroadcastMutableMessage, collectionID int64) (int64, error) {
	importMsg, err := message.AsBroadcastImportMessageV1(msg)
	if err != nil {
		return 0, merr.Wrap(err, "malformed duplicated import broadcast message")
	}
	body, err := importMsg.Body()
	if err != nil {
		return 0, merr.Wrap(err, "malformed duplicated import broadcast message body")
	}
	if body.GetCollectionID() != collectionID {
		return 0, merr.WrapErrServiceInternalMsg(
			"idempotency scope resolved to an import into collection %d, not %d",
			body.GetCollectionID(), collectionID)
	}
	return body.GetJobID(), nil
}

// broadcastImport broadcasts the import message to all vchannels.
// This method is called from the new ImportV2 flow where proxy calls DataCoord directly.
func (s *Server) broadcastImport(ctx context.Context,
	collectionName string,
	collectionID int64,
	partitionIDs []int64,
	files []*internalpb.ImportFile,
	options []*commonpb.KeyValuePair,
	schema *schemapb.CollectionSchema,
	jobID int64,
	vchannels []string,
	idempotencyKey string,
) (duplicatedJobID int64, duplicated bool, err error) {
	// Convert files to msgpb format for validation
	msgFiles := lo.Map(files, func(file *internalpb.ImportFile, _ int) *msgpb.ImportFile {
		return &msgpb.ImportFile{
			Id:    file.GetId(),
			Paths: file.GetPaths(),
		}
	})

	// Validate the request before broadcasting
	if err := s.validateImportRequest(ctx, msgFiles, options); err != nil {
		return 0, false, merr.Wrap(err, "failed to validate import request")
	}

	// Get database name from collection metadata via broker
	// This is safer than extracting from schema which may be stale
	broadcaster, err := s.startBroadcastWithCollectionID(ctx, collectionID)
	if err != nil {
		return 0, false, merr.Wrap(err, "failed to start broadcast with collection id")
	}
	defer broadcaster.Close()

	// Re-check the replication state now that the broadcast holds the shared-cluster
	// resource key. AlterReplicateConfig takes the exclusive-cluster key, so it cannot
	// change the replication topology while this lock is held. The pre-lock check in
	// validateImportRequest can go stale before the lock is acquired: if CDC was
	// enabled in that window, an auto_commit / non-enableInReplicatingCluster import
	// would otherwise be broadcast into a replicating topology and diverge.
	if err := s.validateImportReplication(ctx, options); err != nil {
		return 0, false, merr.Wrap(err, "failed to re-validate import replication under broadcast lock")
	}

	coll, err := s.broker.DescribeCollectionInternal(ctx, collectionID)
	if err := merr.CheckRPCCall(coll, err); err != nil {
		return 0, false, err
	}
	// Build import message without deprecated MsgBase
	msg := message.NewImportMessageBuilderV1().
		WithHeader(&message.ImportMessageHeader{}).
		WithBody(&msgpb.ImportMsg{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_Import,
				Timestamp: 0,
			},
			DbName:         coll.DbName,
			CollectionName: collectionName,
			CollectionID:   collectionID,
			PartitionIDs:   partitionIDs,
			Options:        funcutil.KeyValuePair2Map(options),
			Files:          msgFiles,
			Schema:         schema, // TODO: should we use the schema from the collection?
			JobID:          jobID,
		}).
		// Scoped to the collection by ID, so the same client key stays a distinct
		// operation against another collection, and a rename does not move the key off
		// the collection it was bound to: a retry naming the renamed collection still
		// resolves to its original job. A retry still naming the OLD collection never
		// reaches here -- the proxy resolves the name first -- so it fails rather than
		// importing twice. The broadcaster adds the message type; everything else about
		// the dedup identity is this scope.
		WithIdempotencyKey(message.NewCollectionScopedIdempotencyKey(collectionID, idempotencyKey)).
		WithBroadcast(vchannels).
		MustBuildBroadcast()

	// Broadcast the message
	result, err := broadcaster.Broadcast(ctx, msg)
	if err != nil {
		return 0, false, err
	}
	if result.Duplicated == nil {
		return 0, false, nil
	}
	// The broadcaster resolved this idempotency key to an earlier broadcast, so no
	// new job was created; recover what that broadcast carried.
	originalJobID, err := jobIDFromDuplicatedBroadcast(result.Duplicated, collectionID)
	if err != nil {
		return 0, false, err
	}
	// Never log the raw key: it is client-controlled and may carry sensitive data.
	keyFingerprint := mlog.String("idempotencyKeyFingerprint", message.IdempotencyKeyFingerprint(idempotencyKey))

	mlog.Info(ctx, "import broadcast deduplicated by idempotency key",
		mlog.FieldCollectionID(collectionID),
		mlog.FieldJobID(originalJobID),
		keyFingerprint)
	return originalJobID, true, nil
}

func (c *DDLCallbacks) registerImportCallbacks() {
	registry.RegisterImportV1AckCallback(c.importV1AckCallback)
	registry.RegisterCommitImportV2AckCallback(c.commitImportV2AckCallback)
	registry.RegisterRollbackImportV2AckCallback(c.rollbackImportV2AckCallback)
	registry.RegisterImportIDRangeV2AckCallback(c.importIDRangeAckCallback)
}

// commitImportV2AckCallback handles the ack callback for CommitImport WAL message.
// It transitions the import job from Uncommitted → Committing state.
// Concurrency safety is guaranteed by the broadcaster framework's resource key lock
// (exclusive collection-level lock), so no CAS is needed here.
func (c *DDLCallbacks) commitImportV2AckCallback(ctx context.Context, result message.BroadcastResultCommitImportMessageV2) error {
	header := result.Message.Header()
	jobID := header.GetJobId()
	mlog.Info(ctx, "CommitImport broadcast ack received", mlog.FieldJobID(jobID))

	job := c.importMeta.GetJob(ctx, jobID)
	if job == nil {
		mlog.Info(ctx, "CommitImport: job not found, retry later", mlog.FieldJobID(jobID))
		return merr.WrapErrImportSysFailedMsg("job %d not found, waiting for import job creation", jobID)
	}
	switch job.GetState() {
	case internalpb.ImportJobState_Uncommitted:
		// proceed
	case internalpb.ImportJobState_Committing, internalpb.ImportJobState_Completed:
		mlog.Info(ctx, "CommitImport: job already committing or completed, no-op",
			mlog.FieldJobID(jobID), mlog.String("state", job.GetState().String()))
		return nil
	case internalpb.ImportJobState_Failed:
		// Divergence signal: the source committed but this replica already failed, so
		// this replica will NOT make the data visible. Left as a no-op here; surfaced
		// at WARN for alerting.
		mlog.Warn(ctx, "CommitImport ack landed on a Failed import job; this replica will NOT commit while the source commits — potential primary/standby divergence",
			mlog.FieldJobID(jobID), mlog.String("reason", job.GetReason()))
		return nil
	default:
		// CommitImport may be replicated before the local import task reaches
		// Uncommitted. Returning an error keeps the broadcast task alive so the
		// callback can retry after the import task finishes writing local meta.
		mlog.Info(ctx, "CommitImport: job is not ready, retry later",
			mlog.FieldJobID(jobID), mlog.String("state", job.GetState().String()))
		return merr.WrapErrImportSysFailedMsg("job %d is in state %s, waiting for Uncommitted", jobID, job.GetState())
	}

	if err := c.importMeta.UpdateJob(ctx, jobID,
		UpdateJobState(internalpb.ImportJobState_Committing),
	); err != nil {
		return err
	}

	uncommittedDuration := job.GetTR().RecordSpan()
	mlog.Info(ctx, "import job uncommitted stage done",
		mlog.FieldJobID(jobID),
		mlog.Duration("jobTimeCost/uncommitted", uncommittedDuration))
	return nil
}

// rollbackImportV2AckCallback handles the ack callback for RollbackImport WAL message.
// It transitions the import job to Failed state and records that the failure
// was user-initiated so AbortImport retries can be idempotent.
// Concurrency safety is guaranteed by the broadcaster framework's resource key lock
// (exclusive collection-level lock), so no CAS is needed here.
// Segment cleanup is handled by the import inspector (processFailed), not here.
func (c *DDLCallbacks) rollbackImportV2AckCallback(ctx context.Context, result message.BroadcastResultRollbackImportMessageV2) error {
	header := result.Message.Header()
	jobID := header.GetJobId()
	mlog.Info(ctx, "RollbackImport broadcast ack received", mlog.FieldJobID(jobID))

	job := c.importMeta.GetJob(ctx, jobID)
	if job == nil {
		mlog.Warn(ctx, "RollbackImport: job not found, skipping", mlog.FieldJobID(jobID))
		return nil
	}
	state := job.GetState()
	if state == internalpb.ImportJobState_Committing ||
		state == internalpb.ImportJobState_Completed ||
		state == internalpb.ImportJobState_Failed {
		mlog.Info(ctx, "RollbackImport: job already in terminal/committed state, no-op",
			mlog.FieldJobID(jobID), mlog.String("state", state.String()))
		return nil
	}

	return c.importMeta.UpdateJob(ctx, jobID,
		UpdateJobState(internalpb.ImportJobState_Failed),
		UpdateJobReason(importJobReasonAbortedByUser),
	)
}

// importIDRangeAckJobNotFoundRetryWindow bounds how long the ImportIDRange ack callback
// retries a "job not found" before giving up. The ack holds the collection's exclusive
// resource-key lock until it returns success, so an unbounded retry (what CommitImport
// does) would pin that lock and block later collection DDL. See importIDRangeAckCallback.
const importIDRangeAckJobNotFoundRetryWindow = 10 * time.Minute

// importIDRangeAckCallback handles the ack callback for the ImportIDRange WAL message.
// It runs on BOTH clusters (primary: from its own broadcast; secondary: from the
// REPLICATED broadcast task rebuilt by the secondary's broadcast manager) and applies
// the primary-allocated per-file autoID PK ranges to the local import job meta, so every
// cluster derives identical autoID primary keys (and RowIDs). Concurrency safety is
// guaranteed by the broadcaster framework's resource-key lock (exclusive collection-level
// lock), so no CAS is needed here.
func (c *DDLCallbacks) importIDRangeAckCallback(ctx context.Context, result message.BroadcastResultImportIDRangeMessageV2) error {
	header := result.Message.Header()
	jobID := header.GetJobId()
	body := result.Message.MustBody()

	job := c.importMeta.GetJob(ctx, jobID)
	if job == nil {
		// Normally impossible: the ImportIDRange broadcast targets the same data vchannels
		// as the ImportMsg, and per-PChannel WAL order guarantees the ImportMsg's
		// job-creating ack callback ran first. It is permanent when it does happen (e.g. a
		// secondary that joined mid-import and never received the ImportMsg, or a dropped
		// collection whose job creation was skipped — itself replicated, so the peer fails
		// its job independently). Retry only for a bounded window measured from the
		// broadcast's TimeTick, then WARN and release the collection lock rather than pin it.
		maxTick := result.GetMaxTimeTick()
		if maxTick == 0 {
			// No tick to bound the window; treat as retryable.
			mlog.Info(ctx, "ImportIDRange: job not found and no time tick, retry later", mlog.FieldJobID(jobID))
			return merr.WrapErrImportSysFailedMsg("job %d not found, waiting for import job creation", jobID)
		}
		broadcastTime := tsoutil.PhysicalTime(maxTick)
		if time.Since(broadcastTime) > importIDRangeAckJobNotFoundRetryWindow {
			mlog.Warn(ctx, "ImportIDRange ack found no local job after bounded retries; releasing",
				mlog.FieldJobID(jobID), mlog.Time("broadcastTime", broadcastTime))
			return nil
		}
		mlog.Info(ctx, "ImportIDRange: job not found, retry later", mlog.FieldJobID(jobID))
		return merr.WrapErrImportSysFailedMsg("job %d not found, waiting for import job creation", jobID)
	}

	// The gate holds an autoID import in PreImporting/AssigningIDRange until the ranges
	// are applied, so an in-flight job can only be Pending, PreImporting or
	// AssigningIDRange when this lands. Anything at or past Uncommitted means a race
	// we must not disturb (the job already advanced, failed, or committed) — no-op
	// success.
	switch job.GetState() {
	case internalpb.ImportJobState_Failed, internalpb.ImportJobState_Completed,
		internalpb.ImportJobState_Committing, internalpb.ImportJobState_Uncommitted:
		mlog.Info(ctx, "ImportIDRange: job already past the range gate, no-op",
			mlog.FieldJobID(jobID), mlog.String("state", job.GetState().String()))
		return nil
	}

	// Protocol invariant: exactly one exact-sized range per job file, keyed by position.
	// A violation means the two clusters disagree on the job's shape — fail loudly rather
	// than apply a partial or misaligned range.
	fileRanges := body.GetFileRanges()
	files := job.GetFiles()
	if len(fileRanges) != len(files) {
		reason := fmt.Sprintf("ImportIDRange carries %d file ranges but the job has %d files", len(fileRanges), len(files))
		mlog.Warn(ctx, "ImportIDRange file count does not match the job; failing import",
			mlog.FieldJobID(jobID), mlog.Int("fileRanges", len(fileRanges)), mlog.Int("jobFiles", len(files)))
		return c.importMeta.UpdateJob(ctx, jobID,
			UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(reason))
	}
	seenFileIndex := make(map[int64]struct{}, len(fileRanges))
	for _, fr := range fileRanges {
		idx := fr.GetFileIndex()
		idRange := fr.GetIdRange()
		if idx < 0 || idx >= int64(len(files)) {
			reason := fmt.Sprintf("ImportIDRange file index %d out of range [0,%d)", idx, len(files))
			mlog.Warn(ctx, "ImportIDRange file index out of range; failing import",
				mlog.FieldJobID(jobID), mlog.Int64("fileIndex", idx), mlog.Int("jobFiles", len(files)))
			return c.importMeta.UpdateJob(ctx, jobID,
				UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(reason))
		}
		if _, dup := seenFileIndex[idx]; dup {
			// Without this, a duplicate would leave another file's range nil while
			// PkRangesExact=true, wedging the job in AssigningIDRange until timeout.
			reason := fmt.Sprintf("ImportIDRange carries duplicate file index %d", idx)
			mlog.Warn(ctx, "ImportIDRange carries duplicate file index; failing import",
				mlog.FieldJobID(jobID), mlog.Int64("fileIndex", idx))
			return c.importMeta.UpdateJob(ctx, jobID,
				UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(reason))
		}
		seenFileIndex[idx] = struct{}{}
		if fr.GetRowCount() != idRange.GetEnd()-idRange.GetBegin() {
			reason := fmt.Sprintf("ImportIDRange file %d row count %d does not match its reserved range size %d",
				idx, fr.GetRowCount(), idRange.GetEnd()-idRange.GetBegin())
			mlog.Warn(ctx, "ImportIDRange row count does not match the reserved range size; failing import",
				mlog.FieldJobID(jobID), mlog.Int64("fileIndex", idx), mlog.Int64("rowCount", fr.GetRowCount()),
				mlog.Int64("rangeSize", idRange.GetEnd()-idRange.GetBegin()))
			return c.importMeta.UpdateJob(ctx, jobID,
				UpdateJobState(internalpb.ImportJobState_Failed), UpdateJobReason(reason))
		}
	}

	// Idempotency / first-range-wins. If the ranges are already applied (the exact marker
	// is set, or every file already carries a range), compare per file. An equal range is
	// an at-least-once redelivery — a clean no-op. A different range must never happen:
	// authority comes from the persisted WAL message and the first applied range wins, so
	// surface it as a critical WARN and keep the existing range rather than overwrite it.
	if job.GetPkRangesExact() || jobPKRangesSet(job) {
		for _, fr := range fileRanges {
			existing := files[fr.GetFileIndex()].GetPreAllocatedAutoIds()
			applied := fr.GetIdRange()
			if existing.GetBegin() != applied.GetBegin() || existing.GetEnd() != applied.GetEnd() {
				mlog.Warn(ctx, "ImportIDRange conflicts with an already-applied range; ignoring it, first applied range wins",
					mlog.FieldJobID(jobID), mlog.Int64("fileIndex", fr.GetFileIndex()),
					mlog.Int64("existingBegin", existing.GetBegin()), mlog.Int64("existingEnd", existing.GetEnd()),
					mlog.Int64("incomingBegin", applied.GetBegin()), mlog.Int64("incomingEnd", applied.GetEnd()))
				return nil
			}
		}
		mlog.Info(ctx, "ImportIDRange already applied, no-op (at-least-once redelivery)", mlog.FieldJobID(jobID))
		return nil
	}

	// Apply: index each range by file position and mark the job as carrying exact ranges.
	ranges := make([]*commonpb.IDRange, len(fileRanges))
	var totalReserved int64
	for _, fr := range fileRanges {
		idRange := fr.GetIdRange()
		ranges[fr.GetFileIndex()] = idRange
		totalReserved += idRange.GetEnd() - idRange.GetBegin()
	}
	if err := c.importMeta.UpdateJob(ctx, jobID, UpdateJobPKRanges(ranges)); err != nil {
		// Transient persistence failure → return the error so the scheduler retries.
		return err
	}
	mlog.Info(ctx, "ImportIDRange applied to import job",
		mlog.FieldJobID(jobID), mlog.Int("fileCount", len(ranges)), mlog.Int64("totalReservedIDs", totalReserved))
	return nil
}
