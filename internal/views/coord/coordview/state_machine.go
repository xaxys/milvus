package coordview

import (
	"cmp"
	"maps"
	"slices"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
)

// CoordQueryViewStateMachine manages the lifecycle state machine of a single
// query view on the Coordinator.
//
// The state machine is purely in-memory and non-blocking.
// All I/O (ETCD persistence, node sync) is represented by the latest pending
// external effect and atomically drained by the shared flush scheduler through
// ShardViewManager.
//
// State flow:
//
//	Normal:  Preparing → Ready → Up → Down → Dropping → Dropped
//	Error:   Preparing/Ready/Up → Unrecoverable → Dropping → Dropped
//
// Unrecoverable is a stable state. The Manager decides when to advance to
// Dropping (typically after generating a replacement view) via EnterDropping.
//
// Thread-safety: NOT thread-safe. The caller (Manager) must serialize access.
type CoordQueryViewStateMachine struct {
	preparation *Preparation // Exact-version completion, protected by the manager lock.
	state       qviews.QueryViewState
	view        *viewpb.QueryViewOfShard

	// Per-node reported states.
	snState  qviews.QueryViewState
	qnStates map[int64]qviews.QueryViewState

	// Per-QN ready segment IDs reported during Preparing.
	// Used by Balancer/Manager for progress tracking and decision-making.
	qnReadySegments map[int64][]int64

	// Per-segment loaded index IDs reported by QNs during Preparing.
	// When hasExpectedIndexes is true, segments whose loaded index set matches
	// expectedIndexes are omitted so steady-state memory and ETCD cost stay zero.
	// Kept out of sm.view so Sync payloads and placement stay immutable.
	qnSegmentIndexes   map[int64]map[int64]struct{}
	expectedIndexes    map[int64]struct{}
	hasExpectedIndexes bool

	// Pending external effects, atomically drained through ShardViewManager by
	// the Coordinator flush scheduler.
	pending queryViewFlush

	// ref is the acquired DataView reference backing this view's row stats.
	// It is bound at construction (New/Recover) by the Manager, which
	// acquires it from the DataViewRefProvider, and released exactly once at
	// durable removal (finalizeRemoval). Nil when the version no longer
	// exists (already GC'd) or the caller passes no ref.
	ref             qviews.DataViewRef
	releaseLoadInfo func()
}

// queryViewFlush is the latest unflushed external effect of a state machine.
// Persist and Sync are replaced independently so multiple transitions that
// have not been externalized can fast-forward to their latest desired state.
type queryViewFlush struct {
	Persist *viewpb.QueryViewOfShard
	Sync    []qviews.QueryViewAtWorkNode
}

func (f queryViewFlush) Empty() bool {
	return f.Persist == nil && len(f.Sync) == 0
}

// NewCoordQueryViewStateMachine creates a state machine for a freshly
// generated query view.
//
// ref is the DataView reference acquired by the Manager for this view's
// version (nil when the version does not exist); the state machine holds it
// (lifetime(QueryView) < lifetime(DataView)) until ReleaseRef at durable
// removal.
//
// The caller transfers ownership of view and its placement payload to the state
// machine and must not mutate them after construction.
//
// After construction, the pending flush contains the Preparing view for
// write-ahead persistence and the Preparing targets for all work nodes.
func NewCoordQueryViewStateMachineWithRef(view *viewpb.QueryViewOfShard, ref qviews.DataViewRef) *CoordQueryViewStateMachine {
	sm := &CoordQueryViewStateMachine{
		state:            qviews.QueryViewStatePreparing,
		view:             view,
		ref:              ref,
		snState:          qviews.QueryViewStateNil,
		qnStates:         make(map[int64]qviews.QueryViewState, len(view.QueryNode)),
		qnReadySegments:  make(map[int64][]int64, len(view.QueryNode)),
		qnSegmentIndexes: make(map[int64]map[int64]struct{}),
	}
	for _, qn := range view.QueryNode {
		sm.qnStates[qn.NodeId] = qviews.QueryViewStateNil
	}
	sm.pending.Persist = sm.viewWithState(qviews.QueryViewStatePreparing)
	sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStatePreparing)
	return sm
}

// RecoverCoordQueryViewStateMachine reconstructs a state machine from a view
// loaded from ETCD during Coordinator crash recovery.
//
// ref is the DataView reference the Manager re-acquired for the persisted
// version (nil when the version no longer exists, e.g. already GC'd); the
// caller then decides terminal recovery based on that.
//
// The caller transfers ownership of view and must not mutate it afterwards.
//
// Recovery behavior by persisted state:
//   - Preparing:     re-push Preparing to all nodes.
//   - Up:            no pending (wait for events).
//   - Down:          re-push Down to SN.
//   - Unrecoverable: stays Unrecoverable, waits for Manager to call EnterDropping.
func RecoverCoordQueryViewStateMachineWithRef(view *viewpb.QueryViewOfShard, ref qviews.DataViewRef) *CoordQueryViewStateMachine {
	recoveredState := qviews.QueryViewState(view.Meta.State)
	qnSegmentIndexes := make(map[int64]map[int64]struct{})
	hasPersistedSegmentIndexes := false
	for _, qn := range view.GetQueryNode() {
		for _, part := range qn.GetPartitions() {
			if len(part.GetSegmentLoadedIndexes()) > 0 {
				hasPersistedSegmentIndexes = true
			}
			for _, item := range part.GetSegmentLoadedIndexes() {
				ids := make(map[int64]struct{}, len(item.GetIndexIds()))
				for _, id := range item.GetIndexIds() {
					ids[id] = struct{}{}
				}
				qnSegmentIndexes[item.GetSegmentId()] = ids
			}
		}
	}
	if hasPersistedSegmentIndexes {
		cleaned := &viewpb.QueryViewOfShard{
			Meta:          view.Meta,
			QueryNode:     make([]*viewpb.QueryViewOfQueryNode, len(view.QueryNode)),
			StreamingNode: view.StreamingNode,
		}
		cleaned.ProtoReflect().SetUnknown(view.ProtoReflect().GetUnknown())
		for i, qn := range view.QueryNode {
			qnCopy := proto.Clone(qn).(*viewpb.QueryViewOfQueryNode)
			for _, part := range qnCopy.GetPartitions() {
				part.SegmentLoadedIndexes = nil
			}
			cleaned.QueryNode[i] = qnCopy
		}
		view = cleaned
	}
	sm := &CoordQueryViewStateMachine{
		state:            recoveredState,
		view:             view,
		ref:              ref,
		snState:          qviews.QueryViewStateNil,
		qnStates:         make(map[int64]qviews.QueryViewState, len(view.QueryNode)),
		qnReadySegments:  make(map[int64][]int64, len(view.QueryNode)),
		qnSegmentIndexes: qnSegmentIndexes,
	}
	for _, qn := range view.QueryNode {
		sm.qnStates[qn.NodeId] = qviews.QueryViewStateNil
	}

	switch recoveredState {
	case qviews.QueryViewStatePreparing:
		// Already persisted; re-push to all nodes.
		sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStatePreparing)
	case qviews.QueryViewStateUp:
		// Active view; no re-push needed. Up is persisted precisely to
		// avoid unnecessary Coord↔node communication on recovery.
	case qviews.QueryViewStateDown:
		// Re-push Down to SN.
		sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateDown)
	case qviews.QueryViewStateUnrecoverable:
		// Stable state; wait for Manager to call EnterDropping.
	default:
		panic("coordview: invalid recovered state: " + recoveredState.String())
	}
	return sm
}

// State returns the current in-memory state of the query view.
func (sm *CoordQueryViewStateMachine) State() qviews.QueryViewState {
	return sm.state
}

// View returns the immutable query view definition. Callers must not mutate it.
func (sm *CoordQueryViewStateMachine) View() *viewpb.QueryViewOfShard {
	return sm.view
}

// Version returns the parsed QueryViewVersion of this view.
func (sm *CoordQueryViewStateMachine) Version() qviews.QueryViewVersion {
	return qviews.FromProtoQueryViewVersion(sm.view.Meta.Version)
}

// ReleaseRef releases the DataView reference bound at construction.
// Idempotent: each acquirer must release exactly once, so a second call is a
// no-op.
func (sm *CoordQueryViewStateMachine) ReleaseRef() {
	if release := sm.takeLoadInfoRef(); release != nil {
		release()
	}
	if ref := sm.takeRef(); ref != nil {
		ref.Deref()
	}
}

// takeRef transfers ownership to the caller. Managers detach under their lock
// and Deref after unlocking: the provider may wait on a collection mutation.
func (sm *CoordQueryViewStateMachine) takeRef() qviews.DataViewRef {
	ref := sm.ref
	sm.ref = nil
	return ref
}

func (sm *CoordQueryViewStateMachine) takeLoadInfoRef() func() {
	release := sm.releaseLoadInfo
	sm.releaseLoadInfo = nil
	return release
}

// Ref returns the attached DataView reference, or nil when the DataView
// layer is not wired or the version no longer exists.
func (sm *CoordQueryViewStateMachine) Ref() qviews.DataViewRef {
	return sm.ref
}

// QNReadySegments returns the ready segment IDs reported by each QN.
// The map is keyed by QN node ID. Used by Balancer/Manager for decisions.
func (sm *CoordQueryViewStateMachine) QNReadySegments() map[int64][]int64 {
	return sm.qnReadySegments
}

// QNSegmentIndexes returns the per-segment loaded index IDs reported by QNs
// (or only segments deviating from expectedIndexes when SetExpectedIndexes is configured).
func (sm *CoordQueryViewStateMachine) QNSegmentIndexes() map[int64]map[int64]struct{} {
	return sm.qnSegmentIndexes
}

// SetExpectedIndexes binds the view's required index set from LoadConfig.
// Segments whose loaded index set equals expected are omitted from in-memory
// tracking and Up persistence because consumers already default unreported
// segments to the view's required index set.
func (sm *CoordQueryViewStateMachine) SetExpectedIndexes(expected map[int64]struct{}) {
	copied := make(map[int64]struct{}, len(expected))
	maps.Copy(copied, expected)
	sm.expectedIndexes = copied
	sm.hasExpectedIndexes = true
	for segID, ids := range sm.qnSegmentIndexes {
		if equalInt64Sets(ids, sm.expectedIndexes) {
			delete(sm.qnSegmentIndexes, segID)
		}
	}
}

// reportMakesProgress compares a report with this node's last accepted report,
// before OnNodeStateReported replaces it. Reports carry cumulative ready IDs.
// State transitions of the whole view are tracked separately by the observer.
func (sm *CoordQueryViewStateMachine) reportMakesProgress(report qviews.QueryViewAtWorkNode) bool {
	previousState := sm.snState
	readyIncreased := false
	if qn, ok := report.(*qviews.QueryViewAtQueryNode); ok {
		previousState = sm.qnStates[qn.NodeID()]
		if sm.state == qviews.QueryViewStatePreparing {
			readyCount := 0
			for _, partition := range qn.ViewOfQueryNode().Partitions {
				readyCount += len(partition.ReadySegmentIds)
			}
			readyIncreased = readyCount > len(sm.qnReadySegments[qn.NodeID()])
		}
	}
	switch sm.state {
	case qviews.QueryViewStatePreparing:
		return readyIncreased || (previousState != report.State() &&
			(report.State() == qviews.QueryViewStateReady || report.State() == qviews.QueryViewStateUp))
	case qviews.QueryViewStateDropping:
		return previousState != qviews.QueryViewStateDropped && report.State() == qviews.QueryViewStateDropped
	default:
		return false
	}
}

// OnNodeStateReported is called when a work node (SN or QN) reports its
// current state for this view via SyncQueryView response.
func (sm *CoordQueryViewStateMachine) OnNodeStateReported(report qviews.QueryViewAtWorkNode) {
	sm.updateNodeState(report)

	switch sm.state {
	case qviews.QueryViewStatePreparing:
		sm.handlePreparing(report)
	case qviews.QueryViewStateReady:
		sm.handleReady(report)
	case qviews.QueryViewStateUp:
		sm.handleUp(report)
	case qviews.QueryViewStateDown:
		sm.handleDown(report)
	case qviews.QueryViewStateDropping:
		sm.handleDropping(report)
	}
}

// EnterUnrecoverable is called by the Manager to force this view into
// Unrecoverable state. Used for preemption and RequestRelease.
// Valid from Preparing, Ready, or Up. No-op in other states.
func (sm *CoordQueryViewStateMachine) EnterUnrecoverable() {
	switch sm.state {
	case qviews.QueryViewStatePreparing, qviews.QueryViewStateReady, qviews.QueryViewStateUp:
		sm.transitionToUnrecoverable()
	}
}

// OnQueryNodeLost is called by the Manager when QueryNode service discovery
// removes a QueryNode targeted by this view.
//
// StreamingNode loss is intentionally not modeled here: SN availability is
// handled by the channel assignment layer, not by the per-view state machine.
func (sm *CoordQueryViewStateMachine) OnQueryNodeLost(node qviews.QueryNode) {
	if _, ok := sm.qnStates[node.ID]; !ok {
		return
	}

	switch sm.state {
	case qviews.QueryViewStatePreparing:
		sm.transitionToUnrecoverable()
	case qviews.QueryViewStateDropping:
		sm.qnStates[node.ID] = qviews.QueryViewStateDropped
		if sm.allNodesDropped() {
			sm.state = qviews.QueryViewStateDropped
			sm.pending.Persist = sm.viewWithState(qviews.QueryViewStateDropped)
		}
	}
}

// EnterDown starts ordinary retirement with serving-lease protection.
func (sm *CoordQueryViewStateMachine) EnterDown() {
	sm.EnterDownWithControl(nil)
}

// EnterDownWithControl can strengthen an already-published Down command. A
// control-only update must be persisted and resent while SN still reports Up.
// It reports whether the state or control changed, preserving pending effects on a no-op.
func (sm *CoordQueryViewStateMachine) EnterDownWithControl(control *viewpb.QueryViewControl) bool {
	if sm.state != qviews.QueryViewStateUp && sm.state != qviews.QueryViewStateDown {
		return false
	}
	disableLease := control.GetPreemptServingLease() && !sm.view.Meta.GetControl().GetPreemptServingLease()
	if sm.state == qviews.QueryViewStateDown && !disableLease {
		return false
	}
	if disableLease {
		meta := proto.Clone(sm.view.Meta).(*viewpb.QueryViewMeta)
		if meta.Control == nil {
			meta.Control = &viewpb.QueryViewControl{}
		}
		meta.Control.PreemptServingLease = true
		next := &viewpb.QueryViewOfShard{
			Meta:          meta,
			QueryNode:     sm.view.QueryNode,
			StreamingNode: sm.view.StreamingNode,
		}
		next.ProtoReflect().SetUnknown(sm.view.ProtoReflect().GetUnknown())
		sm.view = next
	}
	sm.state = qviews.QueryViewStateDown
	sm.pending.Persist = sm.viewWithState(qviews.QueryViewStateDown)
	sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateDown)
	return true
}

// EnterDropping is called by the Manager to transition this view from
// Unrecoverable to Dropping. The Manager typically calls this after
// generating a replacement view, so both views can be pushed atomically.
// No-op if not in Unrecoverable state.
func (sm *CoordQueryViewStateMachine) EnterDropping() {
	if sm.state != qviews.QueryViewStateUnrecoverable {
		return
	}
	sm.state = qviews.QueryViewStateDropping
	sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateDropped)
}

// ConsumeFlush atomically drains the latest pending persist and sync effects.
func (sm *CoordQueryViewStateMachine) ConsumeFlush() queryViewFlush {
	flush := sm.pending
	sm.pending = queryViewFlush{}
	return flush
}

// --- State handlers ---

// Preparing: wait for all nodes to report Ready.
//   - Any Unrecoverable → Unrecoverable (persist, wait for Manager).
//   - All ready, SN=Ready → Ready (push Up to SN).
//   - All ready, SN=Up (recovery fast-forward) → Up (persist Up).
func (sm *CoordQueryViewStateMachine) handlePreparing(report qviews.QueryViewAtWorkNode) {
	if report.State() == qviews.QueryViewStateUnrecoverable {
		sm.transitionToUnrecoverable()
		return
	}
	if !sm.allNodesReady() {
		return
	}
	if sm.snState == qviews.QueryViewStateUp {
		// Fast-forward: SN already Up from recovery.
		sm.state = qviews.QueryViewStateUp
		sm.pending.Persist = sm.viewWithState(qviews.QueryViewStateUp)
	} else {
		// Normal flow: all Ready → Ready, push Up to SN.
		sm.state = qviews.QueryViewStateReady
		sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateUp)
	}
}

// Ready: wait for SN to confirm Up.
//   - Any Unrecoverable → Unrecoverable.
//   - SN reports Up → Up (persist Up).
//   - SN not Up → re-push Up.
func (sm *CoordQueryViewStateMachine) handleReady(report qviews.QueryViewAtWorkNode) {
	if report.State() == qviews.QueryViewStateUnrecoverable {
		sm.transitionToUnrecoverable()
		return
	}
	if !sm.isSNReport(report) {
		return
	}
	if report.State() == qviews.QueryViewStateUp {
		sm.state = qviews.QueryViewStateUp
		sm.pending.Persist = sm.viewWithState(qviews.QueryViewStateUp)
		return
	}
	// SN not Up yet (e.g., still Ready) → re-push Up.
	sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateUp)
}

// Up: active view serving queries.
//   - Any Unrecoverable → Unrecoverable.
//   - Up → Down is handled by EnterDown, not node reports.
func (sm *CoordQueryViewStateMachine) handleUp(report qviews.QueryViewAtWorkNode) {
	if report.State() == qviews.QueryViewStateUnrecoverable {
		sm.transitionToUnrecoverable()
	}
}

// Down: wait for SN to confirm Down.
//   - Any Unrecoverable → Unrecoverable (persist, wait for Manager).
//   - SN reports Down or Dropped → Dropping (push Dropped to all).
//     Dropped means the SN already advanced past Down (e.g., Coord crash
//     recovery regressed from Dropping to Down). Treat it the same as Down.
//   - SN not Down → re-push Down.
func (sm *CoordQueryViewStateMachine) handleDown(report qviews.QueryViewAtWorkNode) {
	if report.State() == qviews.QueryViewStateUnrecoverable {
		sm.transitionToUnrecoverable()
		return
	}
	if !sm.isSNReport(report) {
		return
	}
	if report.State() == qviews.QueryViewStateDown || report.State() == qviews.QueryViewStateDropped {
		sm.state = qviews.QueryViewStateDropping
		sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateDropped)
		return
	}
	// SN not Down yet → re-push Down.
	sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateDown)
}

// Dropping: wait for all nodes to confirm Dropped.
//   - All Dropped → Dropped (delete from ETCD).
//   - Node not Dropped → re-push Dropped.
func (sm *CoordQueryViewStateMachine) handleDropping(report qviews.QueryViewAtWorkNode) {
	if report.State() == qviews.QueryViewStateDropped {
		if sm.allNodesDropped() {
			sm.state = qviews.QueryViewStateDropped
			sm.pending.Persist = sm.viewWithState(qviews.QueryViewStateDropped)
		}
		return
	}
	// Node not Dropped → re-push Dropped.
	sm.pending.Sync = sm.syncViewsForState(qviews.QueryViewStateDropped)
}

// transitionToUnrecoverable persists Unrecoverable and stays.
// The Manager must call EnterDropping to advance to Dropping.
func (sm *CoordQueryViewStateMachine) transitionToUnrecoverable() {
	sm.state = qviews.QueryViewStateUnrecoverable
	sm.pending.Persist = sm.viewWithState(qviews.QueryViewStateUnrecoverable)
	sm.pending.Sync = nil
}

// --- Helpers ---

// syncViewsForState builds per-node QueryViewAtWorkNode slices for the given state.
// SN is always included. QNs are included for Preparing and Dropped (states that
// require all nodes to acknowledge), excluded for Up and Down (SN-only transitions).
func (sm *CoordQueryViewStateMachine) syncViewsForState(state qviews.QueryViewState) []qviews.QueryViewAtWorkNode {
	includeQN := state != qviews.QueryViewStateUp && state != qviews.QueryViewStateDown

	meta := proto.Clone(sm.view.Meta).(*viewpb.QueryViewMeta)
	meta.State = viewpb.QueryViewState(state)

	cap := 1
	if includeQN {
		cap += len(sm.view.QueryNode)
	}
	views := make([]qviews.QueryViewAtWorkNode, 0, cap)
	views = append(views, qviews.NewFullQueryViewAtStreamingNode(meta, sm.view.StreamingNode, sm.view.QueryNode))
	if includeQN {
		for _, qn := range sm.view.QueryNode {
			views = append(views, qviews.NewQueryViewAtQueryNode(meta, qn))
		}
	}
	return views
}

func (sm *CoordQueryViewStateMachine) updateNodeState(report qviews.QueryViewAtWorkNode) {
	switch n := report.WorkNode().(type) {
	case qviews.QueryNode:
		sm.qnStates[n.ID] = report.State()
		sm.updateQNReadySegments(n.ID, report)
	case qviews.StreamingNode:
		sm.snState = report.State()
	}
}

func (sm *CoordQueryViewStateMachine) updateQNReadySegments(nodeID int64, report qviews.QueryViewAtWorkNode) {
	qnReport, ok := report.(*qviews.QueryViewAtQueryNode)
	if !ok {
		return
	}
	var readySegs []int64
	recordIndexes := sm.state == qviews.QueryViewStatePreparing
	for _, p := range qnReport.ViewOfQueryNode().Partitions {
		readySegs = append(readySegs, p.ReadySegmentIds...)
		if recordIndexes {
			for _, item := range p.GetSegmentLoadedIndexes() {
				ids := make(map[int64]struct{}, len(item.GetIndexIds()))
				for _, id := range item.GetIndexIds() {
					ids[id] = struct{}{}
				}
				if sm.hasExpectedIndexes && equalInt64Sets(ids, sm.expectedIndexes) {
					delete(sm.qnSegmentIndexes, item.GetSegmentId())
					continue
				}
				sm.qnSegmentIndexes[item.GetSegmentId()] = ids
			}
		}
	}
	sm.qnReadySegments[nodeID] = readySegs
}

func (sm *CoordQueryViewStateMachine) isSNReport(report qviews.QueryViewAtWorkNode) bool {
	_, ok := report.WorkNode().(qviews.StreamingNode)
	return ok
}

// allNodesReady returns true when SN is Ready or Up (recovery) and all QNs are Ready.
func (sm *CoordQueryViewStateMachine) allNodesReady() bool {
	if sm.snState != qviews.QueryViewStateReady && sm.snState != qviews.QueryViewStateUp {
		return false
	}
	for _, state := range sm.qnStates {
		if state != qviews.QueryViewStateReady {
			return false
		}
	}
	return true
}

func (sm *CoordQueryViewStateMachine) allNodesDropped() bool {
	if sm.snState != qviews.QueryViewStateDropped {
		return false
	}
	for _, state := range sm.qnStates {
		if state != qviews.QueryViewStateDropped {
			return false
		}
	}
	return true
}

// viewWithState owns the envelope and metadata while sharing immutable placement.
// State-machine inputs transfer ownership; neither transitions nor persistence
// mutate placement. Sync constructors still detach their mutable transport data.
func (sm *CoordQueryViewStateMachine) viewWithState(state qviews.QueryViewState) *viewpb.QueryViewOfShard {
	queryNodes := sm.view.QueryNode
	if state == qviews.QueryViewStateUp && len(sm.qnSegmentIndexes) > 0 {
		var copiedNodes []*viewpb.QueryViewOfQueryNode
		hasAny := false
		for i, qn := range sm.view.QueryNode {
			qnHas := false
			for _, part := range qn.GetPartitions() {
				for _, segID := range part.GetSegmentIds() {
					idsSet, has := sm.qnSegmentIndexes[segID]
					if has && (!sm.hasExpectedIndexes || !equalInt64Sets(idsSet, sm.expectedIndexes)) {
						qnHas = true
						break
					}
				}
				if qnHas {
					break
				}
			}
			if !qnHas {
				continue
			}
			if copiedNodes == nil {
				copiedNodes = slices.Clone(sm.view.QueryNode)
			}
			hasAny = true
			qnCopy := proto.Clone(qn).(*viewpb.QueryViewOfQueryNode)
			for _, part := range qnCopy.GetPartitions() {
				var loaded []*viewpb.SegmentLoadedIndexes
				for _, segID := range part.GetSegmentIds() {
					idsSet, has := sm.qnSegmentIndexes[segID]
					if !has || (sm.hasExpectedIndexes && equalInt64Sets(idsSet, sm.expectedIndexes)) {
						continue
					}
					var ids []int64
					if len(idsSet) > 0 {
						ids = make([]int64, 0, len(idsSet))
						for id := range idsSet {
							ids = append(ids, id)
						}
						slices.Sort(ids)
					}
					loaded = append(loaded, &viewpb.SegmentLoadedIndexes{
						SegmentId: segID,
						IndexIds:  ids,
					})
				}
				slices.SortFunc(loaded, func(a, b *viewpb.SegmentLoadedIndexes) int {
					return cmp.Compare(a.GetSegmentId(), b.GetSegmentId())
				})
				part.SegmentLoadedIndexes = loaded
			}
			copiedNodes[i] = qnCopy
		}
		if hasAny {
			queryNodes = copiedNodes
		}
	}
	v := &viewpb.QueryViewOfShard{
		Meta:          proto.Clone(sm.view.Meta).(*viewpb.QueryViewMeta),
		QueryNode:     queryNodes,
		StreamingNode: sm.view.StreamingNode,
	}
	// Preserve unknown wire fields without copying protobuf's internal mutex.
	v.ProtoReflect().SetUnknown(sm.view.ProtoReflect().GetUnknown())
	v.Meta.State = viewpb.QueryViewState(state)
	return v
}

func equalInt64Sets(a, b map[int64]struct{}) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if _, ok := b[k]; !ok {
			return false
		}
	}
	return true
}

// NewCoordQueryViewStateMachine preserves the standalone state-machine API.
// Reference-owning managers use NewCoordQueryViewStateMachineWithRef.
func NewCoordQueryViewStateMachine(view *viewpb.QueryViewOfShard) *CoordQueryViewStateMachine {
	return NewCoordQueryViewStateMachineWithRef(view, nil)
}

// RecoverCoordQueryViewStateMachine preserves standalone recovery without a ref.
func RecoverCoordQueryViewStateMachine(view *viewpb.QueryViewOfShard) *CoordQueryViewStateMachine {
	return RecoverCoordQueryViewStateMachineWithRef(view, nil)
}
