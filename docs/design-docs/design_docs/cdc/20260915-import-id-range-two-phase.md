# Two-Phase Import ID Range Assignment (ImportIDRange Message)

- Date: 2026-09-15
- Status: Proposal
- Related: #51667 / #51825 (broadcast-time PK range pre-allocation, being replaced), #48524 (Import 2PC), #51029 (CDC 2PC import), #52544 (idempotent broadcast)

## Summary

Today, for every autoID bulk import, DataCoord synchronously sizes all import
files (parquet footer reads, npy header reads, or byte-derived upper bounds for
JSON/CSV) inside the `ImportV2` RPC path, then allocates per-file PK ranges and
ships them on the `ImportMsg` broadcast so that a CDC secondary derives identical
autoID primary keys.

This design replaces that with a **two-phase** flow:

1. **Phase 1 (job creation)** — `ImportMsg` is broadcast **without** any file
   sizing or ID allocation. The `ImportV2` RPC returns in milliseconds regardless
   of file count.
2. **Phase 2 (post-preimport)** — after PreImport tasks have produced the
   **exact** per-file row counts, the primary allocates per-file ID ranges sized
   exactly (`Σ row_count`, no expansion factor, never over-estimated) and
   broadcasts a new WAL message, **`ImportIDRange`**, to the job's data
   vchannels. The message is replicated to secondaries; each cluster's ack
   callback applies the ranges to its local job meta. The `PreImporting →
   Importing` transition is gated on the ranges being present.

Both clusters then run the existing datanode `pkCursor` path unchanged, so every
imported row gets the same autoID PK (and RowID) on the primary and on every
secondary.

## Motivation

The broadcast-time sizing path (`assignPKRangesToFiles`,
`internal/datacoord/import_row_bound.go`) has three problems:

1. **Latency on the RPC critical path.** An import request may carry up to
   `dataCoord.maxFilesPerImportReq` files (hundreds to thousands). Sizing does
   one object-store HEAD/footer read per file inside `broadcastImport`, turning a
   millisecond RPC into a minutes-long one that can time out, while the work is
   pure waste: PreImport reads every file completely anyway and produces exact
   counts shortly after.
2. **Gross over-allocation for text formats.** JSON/CSV/JSONL have no cheap
   exact count, so the bound is `file_size / min_row_bytes`, which collapses to
   ~1 byte/row for wide-empty schemas. A single CSV file can reserve up to
   `MaxUint32` ids (~4.3e9) for a few million real rows. Exact counts make the
   expansion factor and the byte heuristics unnecessary.
3. **Duplicated I/O.** The coordinator footer pass and the datanode preimport
   pass both read the same files over the network.

Deferring allocation to post-preimport removes all three: no file I/O at
creation, exact-size reservations, and the sizing pass disappears entirely.

## Goals

- Remove all file-content I/O (footer/header/size reads) from the `ImportV2` /
  broadcast path.
- Keep the #51825 guarantee: for autoID imports, every row gets an identical PK
  (and RowID) on the primary and all CDC secondaries.
- Reserve ID ranges of **exact** size (`row_count` per file), never
  over-estimated.
- Crash-safe and retry-safe on both clusters (WAL remains the source of truth).
- Survive failover: a secondary promoted before the range message is persisted
  can complete the import by allocating and broadcasting the range itself.
- Fail loudly on cross-cluster file divergence (row-count mismatch).

## Non-Goals

- **Ts determinism** — import timestamps remain per-cluster (unchanged scope
  from #51825).
- **Segment IDs / binlog logIDs / file IDs** — remain locally allocated per
  cluster (internal identities; binlog paths are already cluster-local).
- **Non-autoID imports** — user-provided PKs need no reservation; RowIDs stay
  per-task-local as today.
- **Backup and L0 imports** — keep embedded PKs / carry no autoID PKs; excluded
  from the mechanism exactly as today.

## Background: Current Flow (master @ ec40d94e19)

```
proxy ImportV2
  → datacoord ImportV2 (alloc jobID)
    → broadcastImport:
        validateImportRequest
        assignPKRangesToFiles            ← sizing + AllocAutoIDN (REMOVED by this design)
        Broadcast ImportMsg (data vchannels, files carry PreAllocatedAutoIds,
                             idempotency key from client)
    → ack callback importV1AckCallback → createImportJobFromAck
        (both clusters: same jobID, same file order, local fileIDs,
         DataTs = broadcast MaxTimeTick)
  → importChecker state machine (2s tick):
      Pending → PreImporting   (PreImport tasks on datanodes: full file read,
                                exact TotalRows + HashedStats per file → etcd)
      PreImporting → Importing (RegroupImportFiles + NewImportTasks;
                                AssembleImportRequest: ErrPKRangeTooSmall guard,
                                per-task logID ID_range allocation)
      Importing → Sorting → IndexBuilding → Uncommitted
      Uncommitted → Committing (auto_commit: broadcastCommitImportMessage;
                                2PC: user CommitImport → CommitImport broadcast)
      Committing → Completed   (per-vchannel HandleCommitVchannel fence)
```

CDC: replicating clusters require `dataCoord.import.enableInReplicatingCluster`
and `auto_commit=false` (2PC). The secondary receives replicated WAL messages,
rebuilds broadcast tasks (`REPLICATED` state), runs the same ack callbacks, and
re-executes the whole import pipeline reading the same files from its own object
storage.

Datanode: per file, `pkCursor` consumes `ImportFile.PreAllocatedAutoIds`
sequentially; PK == RowID for autoID; exceeding the range fails the task loudly.

## Design

### Phase overview

```
PRIMARY                                   SECONDARY
-------                                   ---------
ImportV2 RPC (ms, no file I/O)
Broadcast ImportMsg (no ranges)  ──CDC──► ImportMsg ack → create job (Pending)
ack → create job (Pending)                PreImport tasks (exact row counts)
PreImport tasks (exact row counts)
checker tick: preimport done,             checker tick: preimport done,
  ranges needed & unset                     ranges needed & unset
  → AssigningIDRange                        → AssigningIDRange
checker tick (AssigningIDRange):            checker tick (AssigningIDRange):
  role=primary                                role=secondary
  → AllocAutoIDN(exact Σ rows)                → wait (no broadcast)
  → Broadcast ImportIDRange      ──CDC──► ImportIDRange ack callback:
ack callback: apply ranges to               validate + apply ranges to job meta
  job meta (idempotent)
checker tick: gate passes                 checker tick: gate passes
  → validate rows == range sizes            → validate rows == range sizes
  → stamp ranges on task fileStats          → stamp ranges on task fileStats
  → NewImportTasks → Importing              → NewImportTasks → Importing
datanode pkCursor (unchanged)             datanode pkCursor (unchanged)
```

The gate is symmetric: neither cluster creates Import tasks until
(a) all PreImport tasks completed, and (b) the per-file ranges are present in
job meta. Arrival order of (a) and (b) does not matter.

### New WAL message: ImportIDRange

| Property | Value |
|---|---|
| MessageType | `MessageTypeImportIDRange` (`ExclusiveRequired: false`, not SelfControlled, replicable — no `_ur`) |
| Header | `messagespb.ImportIDRangeMessageHeader { int64 collection_id; int64 job_id; }` |
| Body | `messagespb.ImportIDRangeMessageBody { repeated FileIDRange file_ranges; }` |
| `FileIDRange` | `{ int64 file_index; int64 row_count; commonpb.IDRange id_range; }` |
| Dispatch | Broadcast to the job's **data vchannels** (same routing as ImportMsg / CommitImport / RollbackImport) |
| ResourceKeys | `SharedDBName + ExclusiveCollectionName` (via `startBroadcastWithCollectionID`) |
| Idempotency key | `NewCollectionScopedIdempotencyKey(collectionID, "import-id-range:<jobID>")` |
| Ack callback | `RegisterImportIDRangeV2AckCallback` → datacoord applies ranges |

Payload is defined in this repo's `pkg/proto/messages.proto` (like
Commit/RollbackImport bodies) — no milvus-proto (go-api) change required.

**Why data vchannels, not CChannel-only:** ImportMsg is broadcast to the same
data vchannels. Per-PChannel WAL order then guarantees every ImportIDRange copy
is appended after the ImportMsg copy on the same PChannel, so on a secondary the
ImportMsg broadcast task (smaller broadcastID) is acked — and its job-creating
ack callback executed — before the ImportIDRange callback runs. Conflicting
ResourceKey callbacks execute in broadcast order, so the "job not found" retry
path stays a rare safety net instead of the normal path. (CChannel-only would
cross PChannels and lose this ordering.)

**file_index, not fileID:** each cluster assigns its own fileIDs in
`createImportJobFromAck`; the file *order* is identical on both clusters because
both jobs are created from the same ImportMsg. Ranges are therefore keyed by
position in `job.Files`.

**row_count** is the primary's exact preimport count for that file. It makes the
message self-describing and enables the secondary-side divergence check below.
For exact-size allocation `row_count == id_range.End - id_range.Begin` (including
zero-row files, which get an empty range).

### ID allocation (exact, post-preimport)

Runs on the cluster that triggers the broadcast (normally the primary), inside
the checker tick that observes preimport completion:

- Input: per-file exact `TotalRows` from completed PreImport task stats, in
  `job.Files` order.
- Reuse the existing greedy batching (`reserveRanges` in
  `import_row_bound.go`): one `common.AllocAutoIDN` per ≤ `MaxUint32` batch,
  cluster-ID high bits via `AllocAutoID*`, each file's range contiguous and
  never straddling batches.
- **Expansion factor = 1.** `sizeReservations`'s upper-bound logic and
  `dataCoord.import.preAllocateIDExpansionFactor` no longer apply to PK ranges
  (the parameter remains for the per-task logID `ID_range` in
  `AssembleImportRequest`, unchanged).
- A single file with more than `MaxUint32` rows is rejected (existing
  semantics: one contiguous range cannot cover it; the user must split the
  file). This is now detected at preimport-completion time instead of at
  submission time.
- Zero-row files get an empty range (`Begin == End`). Unlike the old flow there
  is no "at least one id" hack: the datanode treats an empty range as no cursor,
  but a zero-row file consumes no ids, so no divergence is possible. (The
  datanode's fallback WARN fires for such files — cosmetic log noise, accepted.)
- `totalRows == 0` for the whole job: unchanged — the job skips Importing
  entirely (Uncommitted/Completed branch); **no ImportIDRange message is
  broadcast**. Both clusters see the same zero total from the same files.
- Allocation failure (rootcoord unavailable): transient — retry next tick.
  If the process crashes after `AllocAutoIDN` but before the broadcast is
  persisted, the ids leak harmlessly (id space is TSO-derived); the retry
  allocates a fresh range. Exactly one range ever becomes authoritative because
  authority comes from the persisted WAL message, not from local memory.

### State-machine gate (`checkPreImportingJob` + `checkAssigningIDRangeJob`)

After the existing "all preimport tasks completed" check and the
`totalRows == 0` branch:

```
# checkPreImportingJob:
if needsPKRanges(job) && !pkRangesSet(job):
    → UpdateJobState(AssigningIDRange)   # PreImport stage span ends here
    return
# gate passed (ranges not needed, or legacy ImportMsg-carried ranges):
... existing tail: validate → lacks → disk quota → regroup → NewImportTasks → Importing

# checkAssigningIDRangeJob (new state):
if !pkRangesSet(job):
    role := cluster replication role (live, from balancer assignment
             ReplicateConfiguration + replicateutil.ConfigHelper)
    if not replicating || role == primary:
        allocate exact ranges; broadcast ImportIDRange (bounded ctx, ~10s)
        ErrNotPrimary → treat as "wait" (role may be stale during switchover)
        other errors  → log, retry next tick (idempotency key dedups)
    else:  # secondary
        wait for the replicated message (log at Info, rate-limited)
    return  # stay in AssigningIDRange
# gate passed:
validate per-file: local exact TotalRows == range size (and == msg row_count)
    mismatch → WARN divergence + job Failed (reason recorded)
stamp ranges onto each ImportFileStats.ImportFile (by fileID lookup in job.Files)
RegroupImportFiles → NewImportTasks → Importing   (existing code)
```

- `needsPKRanges(job)` = PK field autoID && !backup && !L0 — the same predicate
  `broadcastImport` uses today, evaluated from persisted job meta.
- `pkRangesSet(job)` = every `job.Files[i].PreAllocatedAutoIds` non-nil (or the
  job carries the exact-range marker, see below). Jobs created from an
  **old-format ImportMsg that already carries ranges** (rolling upgrade of a
  single cluster, in-flight jobs) satisfy the gate immediately and never
  broadcast — the migration is data-driven, no version flag needed.
- New `ImportJobState.AssigningIDRange`: preimport is complete and the job is
  waiting for the exact per-file autoID PK ranges to be broadcast (primary) or
  replicated (secondary) and applied by the ImportIDRange ack callback.
  Transitions: `PreImporting → AssigningIDRange` (ranges needed & unset) →
  `Importing`; legacy-ranged / non-autoID jobs go `PreImporting → Importing`
  directly and never enter the new state. Rationale = observability:
  `GetImportProgress` distinguishes "preimport running" (< 40) from "waiting for
  ID range broadcast/replication" (== 40) — the latter stalling points at WAL/CDC
  link problems rather than at datanode preimport work. The new state is a
  non-terminal, abortable, timeout-able in-progress state like `PreImporting`
  (abort/rollback → `Failed`; `tryTimeoutJob` → `Failed`; never in
  `UnfailableJobStates`).
- The wait is bounded by the job's `timeoutTs` (existing `tryTimeoutJob` →
  Failed; for 2PC jobs the existing GC RollbackImport self-heal releases the
  peer). Typical duration: primary ~1–4s (transition tick + one broadcast + one
  checker tick); secondary ≈ replication lag, and zero when the message arrives
  before local preimport completes.
- The broadcast uses a bounded context so a stalled WAL cannot park the
  state-machine loop (same pattern as `checkGC`'s rollback broadcast);
  timeout is just another transient status.

### Ack callback (`importIDRangeAckCallback`)

Runs on **both** clusters (primary: from its own broadcast; secondary: from the
`REPLICATED` broadcast task rebuilt by the secondary's broadcast manager).

```
job := importMeta.GetJob(header.JobId)
if job == nil:
    return error → bounded retry (backoff), give up after a bounded window
    (e.g. 10 min) with a divergence-style WARN, then no-op success.
if job.State ∈ {Failed, Completed, Committing}: no-op success
if len(body.file_ranges) != len(job.Files): WARN + fail job (protocol divergence)
if any file_index out of range, duplicated, or row_count != range size:
    WARN + fail job (a duplicate would leave one file's range nil while
    PkRangesExact=true, wedging the job in AssigningIDRange until timeout)
if ranges already set on job:
    equal → no-op success (at-least-once redelivery)
    different → CRITICAL WARN + no-op (must not happen; first applied range wins)
apply: job.Files[file_index].PreAllocatedAutoIds = id_range; set exact-range marker
persist via importMeta.UpdateJob (new UpdateJobAction)
```

Retry policy rationale: ack callbacks hold the collection's exclusive
ResourceKey lock until they succeed, and an unbounded "job not found" retry
would pin that lock and block later collection DDL. The job-not-found case is
normally impossible (see dispatch ordering above) and permanent when it does
occur (e.g. the ImportMsg callback skipped job creation because the collection
was dropped — itself replicated, so the peer fails its job independently).
Bounded retry + WARN + release is strictly safer than the unbounded retry
CommitImport uses.

### Divergence validation (strong)

The gate (both clusters) fails the job when a local exact row count differs
from the reserved range size / message `row_count`. Same file names with
different content on the two clusters — the precondition the CDC import doc
already requires operators to guarantee — becomes a loud, simultaneous failure
on both sides instead of silently different PKs. Failure propagates through the
existing machinery (2PC: source GC broadcasts RollbackImport; user can
AbortImport).

`AssembleImportRequest`'s existing guard is tightened from
`TotalRows > reserved` to `TotalRows != reserved` **for exact-range jobs only**
(marker set by the ack callback). Legacy upper-bound jobs keep the `>` check
(their ranges carry the old expansion factor).

### Range → import tasks → datanode (unchanged contract)

- Ranges are stamped onto `ImportFileStats.ImportFile` (by fileID) in
  `checkPreImportingJob` after the gate passes, so they persist inside
  `ImportTaskV2` meta and flow through `AssembleImportRequest` into
  `ImportRequest.Files` exactly as today.
- Datanode `pkCursor` consumption, the `ErrPKRangeTooSmall`-style loud failure
  on overrun, PK==RowID assignment, and the per-task logID `ID_range` local
  allocator are **untouched**.
- The datanode's warn-and-fallback for a missing range stays as-is (dead code
  under the new gate; kept for rolling-upgrade windows where an older datacoord
  dispatches to a newer datanode).

### Removals

Deleted outright (old producer path):

- `assignPKRangesToFiles`, `computeFileRowUpperBounds`, `sizeReservations`
  (`internal/datacoord/import_row_bound.go`) — `reserveRanges` is kept and
  refactored to take exact counts.
- `importutilv2.RowCountUpperBound` and the coordinator-side parquet footer
  sizing (`parquet.NumRows` sizing pass, `common.NewSizingReaderAt` usage there)
  and `dataCoord.import.parquetFooterMaxSize` — after auditing remaining
  callers; anything still used elsewhere stays.
- The sizing block in `broadcastImport` (`ddl_callbacks_import.go:253-266`) and
  its TOCTOU re-check comment adjustments.

Kept:

- `internalpb.ImportFile.pre_allocated_auto_ids` / `msgpb.ImportFile` field —
  old WAL messages must still decode; `createImportJobFromAck` keeps copying it
  (in-flight legacy jobs).
- `ErrPKRangeTooSmall` guard (tightened for exact-range jobs).
- `dataCoord.import.preAllocateIDExpansionFactor` — still used for logID
  pre-allocation in `AssembleImportRequest`.

### Component touch list

| Component | Change |
|---|---|
| `pkg/proto/messages.proto` | `ImportIDRangeMessageHeader/Body`, `FileIDRange` |
| `pkg/streaming/util/message` | `MessageTypeImportIDRange` + codegen entry (`codegen/reflect_info.json`, generated builders/converters) |
| streamingcoord broadcaster registry | `RegisterImportIDRangeV2AckCallback` + future map entry |
| datacoord | `broadcastImportIDRange` (services.go), ack callback (ddl_callbacks_import.go), gate + validation + stamping (import_checker.go), new `importCheckerHooks` entry wired in server.go, exact-range marker + `UpdateJobAction` (import_job.go / data_coord.proto), allocation refactor (import_row_bound.go) |
| streamingnode flusher | no-op case for `MessageTypeImportIDRange` on data vchannels (mirrors RollbackImport) |
| streamingnode recovery storage | none (unknown types fall through; `Broadcast().Ack` is generic) |
| CDC replicator / replicate interceptor | none (type-agnostic forwarding) |
| datanode | none |
| docs | `docs/agent_guides/streaming-system/message/message-semantic-collection.md` table row; CDC 2PC import user doc note (same-version requirement) |

## Failure & Edge-Case Analysis

| # | Scenario | Outcome |
|---|---|---|
| 1 | Primary crashes after AllocN, before broadcast persist | Ids leak; checker re-allocates + broadcasts next tick; WAL message is the only authority |
| 2 | Primary crashes after broadcast persist, before ack callback | Broadcast task recovered (PENDING), callback re-executed (at-least-once, idempotent); checker retries broadcast → idempotency-key dedup returns the original → same ranges |
| 3 | Ack callback redelivered | Ranges equal → no-op |
| 4 | Secondary preimport finishes before message arrives | Job waits in AssigningIDRange; message applies ranges; gate passes on a later tick |
| 5 | Message arrives before secondary preimport finishes | Callback applies ranges early; PreImporting → AssigningIDRange transition sees ranges set and runs the tail when preimport completes |
| 6 | Secondary joins topology between ImportMsg and ImportIDRange | ImportMsg never replicated → no local job; IDRange callback hits bounded job-not-found retry → WARN + release lock. The import is invisible on that secondary (inherent to mid-import join, same as CommitImport today) |
| 7 | Collection dropped mid-import | DropCollection is exclusive-serialized with the import broadcasts; ImportMsg ack skips job creation on collection-not-found; IDRange callback bounded-retries then no-ops; segments/GC handled by existing paths |
| 8 | Failover: secondary promoted before IDRange persisted | Promoted cluster's gate sees role=primary → allocates from its own rootcoord (its own cluster bits) → broadcasts → both survive consistently. The dead primary's range never existed in any WAL, so there is nothing to diverge from |
| 9 | Failover after IDRange persisted, before primary's Import tasks | Promoted secondary already received/applied the replicated ranges (or applies them as part of catch-up); its gate proceeds with the primary-allocated ranges — PKs stay identical to what the primary would have used |
| 10 | File content differs across clusters | Row-count equality check fails the job on the diverging side (loud WARN, 2PC rollback releases the peer) |
| 11 | Single file > MaxUint32 rows | Rejected at allocation with an explicit "split the file" error → job Failed (detected later than today, but before any segment is written) |
| 12 | WAL unavailable at broadcast | Bounded-ctx broadcast fails → transient → retried each tick; job eventually bounded by `timeoutTs` |
| 13 | Two concurrent autoID jobs on one collection | Both gate independently; ExclusiveCollectionName serializes their broadcasts; 10s bounded ctx prevents loop parking |
| 14 | Stale role during switchover (thinks primary, gets ErrNotPrimary) | Treated as "wait", retried next tick |

## Compatibility

- **CDC pairs:** `ImportIDRange` is a new message type. An old secondary would
  silently ignore it (unregistered ack-callback type → no-op) and its datanode
  would fall back to local id allocation — silent PK divergence. Per product
  decision, the CDC import feature (`enableInReplicatingCluster`) has not been
  enabled in the field and replication peers are upgraded in lockstep; the
  user doc will state that CDC import requires both clusters to run a version
  containing this change.
- **Old primary + new secondary:** old-format ImportMsg carries ranges; the new
  gate sees ranges already set and never broadcasts. Fully compatible.
- **Single-cluster rolling upgrade:** in-flight jobs created before the upgrade
  carry old-format (upper-bound) ranges in job/task meta; the exact-range marker
  is absent, so the `>` guard and the datanode contract apply unchanged.
- No client-facing API change except one additive job state: request/response
  shapes are untouched; `GetImportProgress` coalesces `AssigningIDRange` into
  the end-of-preimport bucket (progress 40, display state `Importing`), so the
  v1 API keeps showing `ImportStarted`.

## Observability

- Gate waiting logs (Info, rate-limited): reason = "waiting for ID range
  broadcast ack" (primary) / "waiting for replicated ID range" (secondary).
- Broadcast/apply logs with jobID, file count, total reserved ids, elapsed.
- Divergence WARNs: row-count mismatch, range mismatch on redelivery,
  bounded-retry give-up.
- `ImportJobLatency{stage=PreImport}` ends at the `PreImporting →
  AssigningIDRange` transition; no new metric is introduced for the wait (the
  `AssigningIDRange → Importing` transition only advances the TimeRecorder so
  the wait is attributed to neither stage).

## Test Plan

Unit (datacoord):
- Exact allocation: batching across `MaxUint32`, zero-row files, >MaxUint32
  single file rejection, factor-free sizing.
- Gate: primary broadcasts once, dedup on retry, secondary waits, stale-role
  ErrNotPrimary tolerance, totalRows==0 skip, legacy-ranged job skips broadcast.
- Ack callback: apply, idempotent redelivery, job-not-found bounded retry then
  release, terminal-state no-op, file-count mismatch → fail.
- Validation: row-count mismatch fails job; `!=` guard for exact-range jobs vs
  `>` for legacy.

Integration (single cluster): autoID import end-to-end with range applied
post-preimport; PK values equal `range.Begin + row ordinal` per file; zero-row
file; job timeout while waiting (WAL stopped).

CDC e2e (primary + secondary):
- Happy path: identical PKs both clusters (query-level diff).
- Secondary preimport slower/faster than replication (both arrival orders).
- Primary kill between preimport and broadcast → promotion or restart completes
  with a single authoritative range.
- Divergent file content injected on secondary → both sides fail loudly, peer
  released via rollback.

Fault matrix (per verification gate G2): each row of the failure table above is
either traced by hand or injected in the CDC e2e.

## Follow-Ups (out of scope)

- Bounded retry for `commitImportV2AckCallback`'s job-not-found path (same
  lock-pinning hazard; pre-existing).
- Per-vchannel `DataTs` for CDC import (existing TODO in
  `importV1AckCallback`).
- Ts cross-cluster determinism, if ever required.
