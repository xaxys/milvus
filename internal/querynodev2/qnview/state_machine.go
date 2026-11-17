package qnview

import (
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
)

// QNQueryViewStateMachine manages the lifecycle state machine of a single
// query view on a QueryNode.
//
// The QN is a follower: it responds to Coord pushes and local events.
// The state machine is purely in-memory and non-blocking.
// I/O (reporting to Coord) is signaled through pending proto consumed via ConsumeReport.
//
// The QN only stores its own portion of the view: QueryViewMeta + QueryViewOfQueryNode.
// It does not have access to the full QueryViewOfShard.
//
// State flow:
//
//	Normal:  Preparing → Ready → Dropping → Dropped
//	Error:   Preparing → Unrecoverable → Dropping → Dropped
//
// The Preparing → Ready transition is automatic: when OnSegmentsReady reports
// all segments across all partitions as ready, the SM transitions to Ready.
//
// QN has no Up/Down states. Once Ready, it serves queries until Dropped.
//
// Thread-safety: NOT thread-safe. The caller must serialize access.
type QNQueryViewStateMachine struct {
	state  qviews.QueryViewState
	meta   *viewpb.QueryViewMeta
	qnView *viewpb.QueryViewOfQueryNode
	// loadedIndexes returns the index ids a segment actually loaded; nil when the
	// node cannot tell (the report then carries no per-segment index evidence).
	loadedIndexes func(segmentID int64) []int64

	// Per-partition ready segment set, updated incrementally by OnSegmentsReady.
	readySegments map[int64]map[int64]struct{}
	// Per-partition assigned segment set used for Ready transition counting.
	assignedSegments map[int64]map[int64]struct{}

	// Counters for O(1) completion check.
	totalSegments int
	readyCount    int

	pendingReport  *viewpb.QueryViewOfShard
	pendingRelease bool
}

// NewQNQueryViewStateMachine creates a state machine when the QN receives
// a Preparing push from Coord.
//
// After construction:
//   - State is Preparing. No pendingReport (subsequent OnSegmentsReady /
//     OnUnrecoverable will drive progress and generate reports).
func NewQNQueryViewStateMachine(
	meta *viewpb.QueryViewMeta,
	qnView *viewpb.QueryViewOfQueryNode,
	loadedIndexes func(segmentID int64) []int64,
) *QNQueryViewStateMachine {
	readySegments := make(map[int64]map[int64]struct{}, len(qnView.Partitions))
	for _, p := range qnView.Partitions {
		readySegments[p.PartitionId] = make(map[int64]struct{})
	}
	assignedSegments, total := buildAssignedSegmentSet(qnView)
	return &QNQueryViewStateMachine{
		loadedIndexes:    loadedIndexes,
		state:            qviews.QueryViewStatePreparing,
		meta:             meta,
		qnView:           qnView,
		readySegments:    readySegments,
		assignedSegments: assignedSegments,
		totalSegments:    total,
	}
}

// State returns the current in-memory state of the query view.
func (sm *QNQueryViewStateMachine) State() qviews.QueryViewState {
	return sm.state
}

// Meta returns the query view meta.
func (sm *QNQueryViewStateMachine) Meta() *viewpb.QueryViewMeta {
	return sm.meta
}

// QNView returns the original QueryViewOfQueryNode.
func (sm *QNQueryViewStateMachine) QNView() *viewpb.QueryViewOfQueryNode {
	return sm.qnView
}

// OnCoordStateDelivered handles a state push from the Coordinator.
//
// In a distributed state machine, any Coord push must produce a response
// so that Coord can learn the node's current state and fast-forward.
//
// Coord pushes handled:
//   - Preparing: if QN has advanced past Preparing, re-report current state
//     for fast-forward. If still Preparing, no re-report needed (local events
//     will eventually drive the transition).
//   - Dropped: transition to Dropped from any state.
func (sm *QNQueryViewStateMachine) OnCoordStateDelivered(pushedState qviews.QueryViewState) {
	switch pushedState {
	case qviews.QueryViewStatePreparing:
		sm.handleCoordPreparing()
	case qviews.QueryViewStateDropped:
		sm.handleCoordDropped()
	}
}

// OnSegmentsReady reports incremental segment loading progress.
// readySegmentIDs maps partition ID to the newly loaded segment IDs (delta).
// Duplicate segment IDs are deduplicated internally.
//
// When all assigned segments across all partitions are ready, the SM automatically
// transitions to Ready state.
//
// Valid in Preparing and Ready state; ignored in other states.
func (sm *QNQueryViewStateMachine) OnSegmentsReady(readySegmentIDs map[int64][]int64) {
	if sm.state != qviews.QueryViewStatePreparing && sm.state != qviews.QueryViewStateReady {
		return
	}

	changed := false
	for partitionID, segIDs := range readySegmentIDs {
		pSet := sm.readySegments[partitionID]
		if pSet == nil {
			continue
		}
		assignedSet := sm.assignedSegments[partitionID]
		for _, segID := range segIDs {
			if _, exists := pSet[segID]; !exists {
				pSet[segID] = struct{}{}
				changed = true
				if _, assigned := assignedSet[segID]; assigned {
					sm.readyCount++
				}
			}
		}
	}

	if sm.state == qviews.QueryViewStateReady {
		if !changed {
			return
		}
	} else if sm.readyCount >= sm.totalSegments {
		sm.state = qviews.QueryViewStateReady
	}
	sm.pendingReport = sm.buildReport()
}

func buildAssignedSegmentSet(qnView *viewpb.QueryViewOfQueryNode) (map[int64]map[int64]struct{}, int) {
	assignedSegments := make(map[int64]map[int64]struct{}, len(qnView.GetPartitions()))
	total := 0
	for _, partition := range qnView.GetPartitions() {
		segments := make(map[int64]struct{}, len(partition.GetSegmentIds()))
		for _, segmentID := range partition.GetSegmentIds() {
			segments[segmentID] = struct{}{}
		}
		assignedSegments[partition.GetPartitionId()] = segments
		total += len(segments)
	}
	return assignedSegments, total
}

// OnUnrecoverable reports a fatal error (e.g., OOM during segment loading).
// Transitions from Preparing to Unrecoverable.
// Only valid in Preparing state; ignored in other states.
func (sm *QNQueryViewStateMachine) OnUnrecoverable() {
	if sm.state != qviews.QueryViewStatePreparing {
		return
	}
	sm.state = qviews.QueryViewStateUnrecoverable
	sm.pendingReport = sm.buildReport()
}

// ConsumeReport returns the view to report to the Coordinator and clears the flag.
// Returns nil if no report is needed.
func (sm *QNQueryViewStateMachine) ConsumeReport() *viewpb.QueryViewOfShard {
	v := sm.pendingReport
	sm.pendingReport = nil
	return v
}

// ConsumeRelease returns true if the SM has a pending Release operation
// (i.e., entered Dropping state) and clears the flag.
func (sm *QNQueryViewStateMachine) ConsumeRelease() bool {
	v := sm.pendingRelease
	sm.pendingRelease = false
	return v
}

// OnDropped is called by the SegmentManager Release callback when segment
// release completes. Transitions Dropping → Dropped.
// Only valid in Dropping state; ignored in other states.
func (sm *QNQueryViewStateMachine) OnDropped() {
	if sm.state != qviews.QueryViewStateDropping {
		return
	}
	sm.state = qviews.QueryViewStateDropped
	sm.pendingReport = sm.buildReport()
}

// --- Coord push handlers ---

func (sm *QNQueryViewStateMachine) handleCoordPreparing() {
	if sm.state == qviews.QueryViewStatePreparing {
		// Still Preparing: local events will drive progress. No re-report needed.
		return
	}
	// Node has advanced past Preparing: re-report current state so Coord
	// can fast-forward (e.g., Ready, Unrecoverable, Dropped).
	sm.pendingReport = sm.buildReport()
}

func (sm *QNQueryViewStateMachine) handleCoordDropped() {
	switch sm.state {
	case qviews.QueryViewStateDropping:
		// Already releasing segments, wait for OnDropped callback.
		return
	case qviews.QueryViewStateDropped:
		// Terminal state: re-report for Coord fast-forward.
		sm.pendingReport = sm.buildReport()
		return
	default:
		// Transition to Dropping: signal that Release should be called.
		// Clear any stale pending report (e.g., from prior OnSegmentsReady).
		sm.state = qviews.QueryViewStateDropping
		sm.pendingReport = nil
		sm.pendingRelease = true
	}
}

// --- Helpers ---

// readySegmentSlice returns the ready segment IDs for a partition as a slice.
func (sm *QNQueryViewStateMachine) readySegmentSlice(partitionID int64) []int64 {
	pSet := sm.readySegments[partitionID]
	if len(pSet) == 0 {
		return nil
	}
	segs := make([]int64, 0, len(pSet))
	for segID := range pSet {
		segs = append(segs, segID)
	}
	return segs
}

// loadedIndexesOf reports, for every segment that finished loading, the indexes
// it can answer from. A node that cannot tell leaves the list empty, which the
// coordinator reads as "no evidence" rather than "loaded nothing".
func (sm *QNQueryViewStateMachine) loadedIndexesOf(segmentIDs []int64) []*viewpb.SegmentLoadedIndexes {
	if sm.loadedIndexes == nil || len(segmentIDs) == 0 {
		return nil
	}
	loaded := make([]*viewpb.SegmentLoadedIndexes, 0, len(segmentIDs))
	for _, segmentID := range segmentIDs {
		indexes := sm.loadedIndexes(segmentID)
		if len(indexes) == 0 {
			continue
		}
		ids := make([]int64, 0, len(indexes))
		seen := make(map[int64]struct{}, len(indexes))
		for _, id := range indexes {
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
		loaded = append(loaded, &viewpb.SegmentLoadedIndexes{
			SegmentId: segmentID,
			IndexIds:  ids,
		})
	}
	return loaded
}

// buildReport constructs a QueryViewOfShard report from the QN's current state.
func (sm *QNQueryViewStateMachine) buildReport() *viewpb.QueryViewOfShard {
	meta := proto.Clone(sm.meta).(*viewpb.QueryViewMeta)
	meta.State = viewpb.QueryViewState(sm.state)

	qnView := proto.Clone(sm.qnView).(*viewpb.QueryViewOfQueryNode)
	// Populate ReadySegmentIds and the indexes each ready segment actually loaded.
	for _, p := range qnView.Partitions {
		p.ReadySegmentIds = sm.readySegmentSlice(p.PartitionId)
		p.SegmentLoadedIndexes = sm.loadedIndexesOf(p.ReadySegmentIds)
	}

	return &viewpb.QueryViewOfShard{
		Meta:      meta,
		QueryNode: []*viewpb.QueryViewOfQueryNode{qnView},
	}
}
