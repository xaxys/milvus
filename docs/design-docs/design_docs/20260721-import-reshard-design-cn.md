# Milvus Import Re-shard 详细设计与一次性实施计划

> 状态：实施计划。
>
> 本文以 `/home/zilliz/Downloads/Import Re-shard：把数据重组前移到第一次读取.md` 为产品与架构权威方案，以当前分支源代码提交 `c04e16fbc0967bcabe0da5426d22f8691dae80f8`（本文 amend 前的父提交）为实现事实。两者冲突时，本文按用户确认采用当前源码事实。本文描述的是一次完整架构改造，不是分阶段临时方案。

## 1. 结论先行

当前 Import 的主要问题不是某个 buffer 太小，而是阶段边界放错了：PreImport 已经完整解析源文件，却只留下统计；正式 Import 再读一次，并且每个按文件分组的 task 独立分桶、独立写 segment。同一个 `(vchannel, partition)` bucket 因 task 边界被反复切开，最终形成更多小 segment、小 binlog 和大的 `hashed_stats` 元数据。

新链路把数据重组前移到第一次读取：

```text
Pending
  → Resharding
  → Planning
  → Importing
  → IndexBuilding
  → Uncommitted
  → Committing
  → Completed

任意未提交阶段 → Failed
```

各阶段职责固定如下：

1. DataCoord 在广播 Import WAL 前固定 job、ImportTask 版本、稳定 file ID，以及普通格式每文件唯一的字面 `IDRange`。
2. `Resharding` 状态由 DataNode 的 `ReshardTask` 第一次且唯一一次解析普通源文件；完成字段规范化、PK/RowID 物化和 bucket 路由，并把每批数据按统一 `SortSpec` 排序后写成临时 fragment。
3. Planning 在全部 ReshardTask 的胜出 manifest 到齐后，跨文件 task 汇总同一 bucket，按完整 fragment 顺序规划正式 segment。
4. Importing 对每个 segment plan 做单头游标 k-way merge；`dataCoord.import.fragmentMergeFanIn` 是可调运行参数，默认 16，只做基础 sanity 校验（`2 <= fan-in <= 1024`），实际内存和并发由 task working-set slot 估算与 DataNode slot admission 约束；输入超过该值时在同一 run 内做分层归并；最后执行过滤、函数、TEXT/LOB、统计和正式 Storage V2/V3 writer。
5. 新 ImportTaskV3 产出的正式 segment 已按当前 sort compaction 规则有序，直接进入 IndexBuilding，不再运行后置 sort compaction。

本方案的固定决策：

- 历史 ImportTaskV2 job 继续按旧 Import 路径恢复；新普通格式和 backup/binlog job 固定创建 ImportTaskV3，L0 job 固定使用 ImportTaskV2。版本在 Import WAL 和 job/catalog 中持久化，创建后不能因配置、重启、节点变化而重新选择，也不能静默降级。这里把版本放进 WAL 是为了允许滚动升级时旧广播 task 与新广播 task 并存，并让 CDC、ACK callback 和重启恢复逐条识别原来选定的执行链路；它不是用户数据语义，也不是用户可见 API。
- fragment 预排序是 ImportTaskV3 的协议要求，不提供开关。
- ReshardTask 以完整 `ImportFile` 为原子单元，按估算大小降序后使用尽量简单的 BFD（Best-Fit Decreasing）装箱：每个文件放入“放入后剩余空间最小、且仍能容纳它”的已有 task；不做回溯、不做多维优化、不切文件。每个 V3 task 的 Reshard reader/sort/write pipeline 首期串行，多个 task 是否并行由 DataNode slot admission 决定。
- Planning 按规范序顺序装箱，不使用 BFD，不切 fragment。
- Importing 的 direct merge fan-in 是可调运行参数，默认 16。只做基础 sanity 校验 `[2,1024]`：1 会导致分层归并不收敛，过大的值由 task working-set slot 估算、reader 数和对象存储压力约束，不把参数校验伪装成完整资源证明。
- TTL 只冻结 collection 的 TTL 时长，不保存、不冻结、不比较任何 `filter_reference_time`。Importing 在最终过滤每个 batch 时直接读取当时的 `time.Now()`，已经过期的行不写入正式 segment；同一行在重试、换节点或进程重启后得到不同结果是允许的。已经写入正式 segment 后变过期的行沿用现有查询 TTL 过滤和 compaction 物理删除；不为这个过渡增加新的保留时间或新阶段。用户 API 不应看到已过期行，内部何时完成物理回收只要最终一致即可。
- 当前 Import V2 没有 `{root}/import/v1/{job_id}` 临时目录，正式对象直接使用 `insert_log/{collection}/{partition}/{segment}`。旧 `PreImportTask`/`ImportTaskV2` 路径暂时完整保留，方便未来大版本整体删除。V3 临时对象统一从 `{root}/import_v3/{job_id}/` 开始，采用独立实现和简短层级，不使用带字段名和等号的键值式目录。
- 当前普通格式和 backup/binlog 请求都从 `Resharding` 状态的 ReshardTask 开始；L0 继续现有 ImportTaskV2。ImportTaskV3 的 Planning/Importing 只依赖统一 fragment 引用，不依赖 ReshardTask 控制记录。

旧链路行为简述：旧 Import 仍按 `PreImportTask → ImportTaskV2 → Sorting` 执行。PreImport 读取文件并把统计写入 task meta；ImportTaskV2 再读源文件，按 task 自己看到的 bucket 分桶，使用预分配的 segment 和 allocator range 写正式 segment；完成后 Sorting 再读这些 segment 做 sort compaction，之后才进入 IndexBuilding。重试沿用同一 task ID 和旧 task 的 segment/重置 imported rows 语义，DataCoord 内存里的 `retryTimes` 只用于 task version；DataNode task map 也只按 task ID。V3 保留这些对用户可见的 job、CommitImport/RollbackImport、auto-commit 和可见性行为，但把内部执行拆成独立 ReshardTask/ImportTaskV3，并用 task 内 `run_id` 做迟到结果隔离。

## 2. 目标、非目标和明确假设

### 2.1 目标

- 源文件只完整解析一次，第一次读取结果可被后续 Importing 直接消费。
- bucket 归属在 task 边界之前物化为临时数据，使 DataCoord 可以跨源文件 task 统一规划 segment。
- 删除新链路中的 `hashed_stats` 大表，精确统计随 fragment manifest 放在对象存储，不进入 etcd。
- Importing 任意时刻只顺序构建一个正式 segment，归并 reader、heap、writer 和函数 batch 受 task working-set slot 估算和 DataNode slot admission 约束；fan-in 参数只做基础范围校验。
- 正式 segment 在 Importing 中一次生成最终有序形态，避免再执行一次 sort compaction 读写。
- 普通格式、backup/binlog、AutoID、显式 PK、partition key、namespace、函数、TEXT/LOB、BM25、Storage V2/V3 和现有 2PC Import 都有完整实现落点和测试计划。
- task 重试保留 task 内 `run_id`、独立输出文件名、新物理 segment ID 和结果校验，用于拒绝迟到结果、避免不同 run 覆盖同一对象；首期不把这些机制描述成旧 writer 的强停写证明。
- DataCoord/DataNode 重启、取消、超时、滚动升级和二进制回滚有明确恢复边界。

### 2.2 非目标

- L0 Import。L0 数据只按 vchannel 组织，继续使用现有 L0 ImportTaskV2 语义。
- Importing 完成后直接进入 IndexBuilding；本次不增加任何位于 Importing 之后、Uncommitted/Committing 之前的新状态、字段、manifest、task、RPC、hook 或测试分支。未来若要在这里做去重或 unique PK 支持，届时在第 7.3 节的单一状态推进点插入即可；本次不为它命名、不预留 proto、不增加空实现。
- 本次不增加用户可见 option、公开 API、内部 RPC endpoint 或函数入口来选择外部 fragment 输入。
- 旧 ImportTaskV2/PreImportTask 路径暂时完整保留。V3 使用独立的新 task、request、checker 分支，方便下一次大版本在旧 job 全部结束后删除 V2 路径；本次不把 V3 实现硬塞进旧 V2 文件。
- 本次不让外部 Spark 分配 Milvus task ID、run ID、segment ID 或生成内部 Planning 结果。
- 本次不保证全 collection 的 PK range 不重叠；本方案只保证每个 fragment 和每个正式 segment 内按 `SortSpec` 有序。
- 本次不支持部分成功。任一必须处理的源文件、fragment 或正式 segment 失败，整个 job 失败或重试。
- 本次不新建通用 DAG、插件框架、通用多维资源调度协议或新的 QuerySlot 资源向量。
- 本次不重新设计 Storage V2/V3 正式文件格式、函数框架、TEXT LOB 格式或索引流程。

### 2.3 用户明确给出的假设

以下边界被明确排除，本文不增加对应防御协议：

- 源 Import 文件由受管对象存储管理，Import 期间用户不会修改、覆盖或删除文件。
- CDC 两端使用的 Import 文件内容一致。
- 不记录或比较源对象 version ID、ETag、source identity。
- 不做读取前后 `SOURCE_CHANGED` 检查。
- 不增加 CDC 两端源文件校验握手。
- 不增加逐行 SHA sum/xor 守恒协议或下载整对象后再读取的 verified cache。

临时内部对象仍需要 manifest-last 发布、准确路径、行数、字节数和排序边界，因为这些是 task 完成、Planning 和 run 裁决所需的内部契约，不是对用户源文件可变性的防御。

## 3. 当前源码事实和根因

### 3.1 当前调用链

当前源码中的入口是：

```text
Proxy.ImportV2
→ internalpb.ImportRequestInternal
→ DataCoord.ImportV2
→ broadcastImport
→ msgpb.ImportMsg WAL 广播
→ importV1AckCallback
→ createImportJobFromAck
→ datapb.ImportJob
→ ImportChecker
→ PreImportTask
→ ImportTaskV2
→ Sorting
→ IndexBuilding
→ Uncommitted / Committing / Completed
```

关键文件：

- `internal/proxy/task_import.go`
- `internal/datacoord/services.go`
- `internal/datacoord/ddl_callbacks_import.go`
- `internal/datacoord/import_checker.go`
- `internal/datacoord/import_meta.go`
- `internal/datacoord/import_task_preimport.go`
- `internal/datacoord/import_task_import.go`
- `internal/datanode/importv2/`（旧 V2，仅作为当前源码事实和保留路径）

当前广播实际使用 `startBroadcastWithCollectionID`，Import 消息带 `SharedDBName`、`ExclusiveCollectionName`，Broadcaster 再追加 `SharedCluster`。Streaming 指南中“Import 没有 ResourceKey”的描述与源码不一致；本文按当前源码设计，并把更新 `docs/agent_guides/streaming-system/message/message-semantic-collection.md` 的 ResourceKey 表格和说明列入 pending documentation todo，避免实现完成后 agent guide继续冲突。

该 Streaming 表不只 Import 一处过期。源码审计还发现：表中没有列出 `CommitImport`/`RollbackImport`；`DropSnapshot` 实际使用 `SharedDBName + ExclusiveCollectionName + ExclusiveSnapshotName`，不是只有 `ExclusiveSnapshotName`；`RefreshExternalCollection` 实际通过 `startBroadcastWithCollectionID` 使用 `SharedDBName + ExclusiveCollectionName`，不是无 ResourceKey。本文不顺带直接修改 agent guide，但把这四类偏差与 Import 一起记录为 pending documentation todo；实施功能时必须以当前生产者构造的 Broadcast ResourceKeys 为事实统一校准表格、消息说明和对应测试，不能只修 Import 一行。

### 3.2 当前两次读取

当前 PreImport：

- 使用 `importutilv2.NewReader` 完整读取文件；
- 做行对齐、StructArray 一致性检查；
- 调用 `GetRowsStats` 计算 `hashed_stats`；
- 只保存文件大小、总行数、总内存大小和 bucket 统计；
- 丢弃已经解析的行。

当前 Import：

- 再次调用相同 reader；
- 执行 `AppendSystemFieldsData`、nullable/default、dynamic、function；
- 调用 `HashData`；
- 把每批 bucket 数据 sync 到 DataCoord 预分配的 segment。

### 3.3 task 边界造成碎片

设 collection 有 `V` 个 vchannel，本次导入覆盖 `P` 个 partition，则 routing bucket 数量 `B = V × P`。每个旧 ImportTask 只看自己的文件组，却可能同时触达全部 `B` 个 bucket。同一 bucket 的数据会先被文件组切开，再由各 task 写进各自预分配的 segment，任务完成后无法跨 task 共同填充。

例如 `V=16、P=1024` 时有 16384 个 bucket。`T × B` 不是 segment 数等式，但它是“task 与 bucket 组合”的上界，说明 task 边界会反复制造 bucket-local 尾段。结果是：

- buffer 被大量 bucket 分摊，频繁 flush；
- 同一 bucket 出现更多未填满 segment；
- Storage V2 下小 binlog 数还会被字段数放大；
- PreImport 的 `hashed_stats` 随 `V × P` 增长并进入 etcd；
- AutoID 场景的现有平均分配估算可能与真实 hash 路由不一致。

### 3.4 当前恢复与重试缺口

当前 `ImportJob`、`PreImportTask`、`ImportTaskV2` 都持久化；DataCoord 重启后 `NewImportMeta` 恢复元数据，`importInspector.reloadFromMeta` 把 InProgress task 放回 scheduler。但当前重试不具备持久化 run fencing：

- `retryTimes` 只在 DataCoord 内存中；
- DataNode `TaskManager` 只按 `task_id` 管理；
- 重复 Add 只打日志，Create RPC 仍可能被视为成功；
- Query、Drop、异步完成回调都没有 run ID；
- Import 正式结果更新 SegmentInfo 和 task Completed 是分开的持久化操作。

Reshard 会产生不可变临时对象和多个正式 segment，必须在新链路中一次解决这些边界。

### 3.5 当前 Import 无法正确处理执行中的 schema change

这里说的不是用户修改 Import 文件，而是 Import 运行期间 collection schema 被升级。当前旧 Import 保存 job 创建时的 schema，但后续 checker 不比较 frozen/current schema；index inspector却读取当前 schema，schema-bump policy又跳过 `IsImporting` segment，因此可能形成无界等待：

```text
T0：Import job 创建，保存 schema version 1
    例如只有原字段，或只有旧 function 定义

T1：Import 仍在 Resharding/Importing/Sorting
    管理员把 collection 升级为 schema version 2
    例如新增 function/output field，或修改需要物化新输出的 schema

T2：Import 继续按 job 中冻结的 version 1 写 segment
    因为 task/retry 使用 job schema，不会在中途切 version 2

T3：job 进入 IndexBuilding
    Import checker只检查 segment是否还有未完成索引，不检查 schema变化
    index inspector读取当前 collection version 2
    如果当前 schema含 functions且 segment仍是 version 1，它拒绝建索引

T4：schema-bump policy看到该 segment仍是 IsImporting=true
    当前规则只处理 Flushed && !IsImporting 的 segment，因此跳过

结果：Import checker等索引
    → index inspector等 schema-bump先补齐新输出
    → schema-bump等 Import commit后清除 IsImporting
    → Import又必须先等索引完成
```

当前源码没有专门错误终止该循环；`dataCoord.import.waitForIndex=true` 又是默认值，因此 job可能一直停在 IndexBuilding，直到外部 timeout或人工处理。这是旧 Import 已存在的缺陷，不是 Reshard 新造的问题。

ImportTaskV3 不对任何 version变化一刀切失败，而是比较 Import-relevant projection：物理字段、PK/partition key/namespace、TEXT物理规则、functions及其 input/output、目标 index字段。projection等价表示变化无 Import副作用，例如只改 description；job继续并安全提升预创建 SegmentInfo 的 schema version。projection不等价表示 frozen segment会缺物理列、function output或 index输入；job明确 Failed、quiesce并回收，不能再进入上述等待环。

## 4. 名词和不变量

| 名词 | 定义 |
| --- | --- |
| `ImportFile` | 请求中的逻辑文件。JSON/CSV/Parquet 通常含一个路径；NumPy 多列路径合起来仍是一个原子 ImportFile；backup 展开后的一个源 segment 也是一个 ImportFile。 |
| bucket | 一行数据的完整目标 `(vchannel, partition)`；一个正式 L1 segment 只能属于一个 bucket。 |
| `SortSpec` | fragment 和 Importing 共同使用的排序字段及顺序。 |
| fragment | 一个 bucket 的一批临时、单 column group、packed Parquet 数据，内部按 `SortSpec` 有序。 |
| spill chunk | DataNode 本地 Arrow IPC stream 文件，只属于一个 ReshardTask run；关闭后不可变。 |
| segment plan | DataCoord 为一个 bucket 选定的一段连续 fragment 规范序列。 |
| ReshardTask | 读取一组完整 ImportFile，输出多个 bucket 的 fragment。 |
| Importing task | 通常负责一个 vchannel，顺序执行多个 segment plan。 |
| run | 同一个 task 的一次执行代次，由 task 内单调递增的 `run_id` 标识；不是独立 proto、catalog 对象或目录层级。 |
| task plan | 一个 ImportTaskV3 的唯一持久化 Planning 结果；包含执行所需的冻结 schema、WriterSpec、segment plans 和资源参数。 |

全链路必须保持以下不变量：

1. 每一源行只属于一个 bucket。
2. 每个 fragment 只属于一个 bucket，内部按统一 `SortSpec` 单调不降。
3. 每个胜出 fragment 恰好进入一个 segment plan，不能切开，也不能被两个 plan 引用。
4. 一个 segment plan 只属于一个 bucket，输入是规范序中的连续区间。
5. 一个 reader 在 merge heap 中最多一个 head；heap 大小 `O(fan_in)`。
6. 重复 PK 全部保留；相同 key 的相对顺序不是契约。
7. 函数、BM25、TEXT LOB 和正式统计只在最终轮执行一次。
8. 任何正式 segment 在 CommitImport vchannel fence 前都保持 `is_importing=true`。
9. DataCoord 只接受当前 task `run_id` 的完整结果；旧 run 的结果和 Drop 不能修改当前 task。网络分区下旧 writer 仍可能继续写旧 run 的隔离路径，因此首期只保证“旧结果不被接受、对象路径不互相覆盖”，不保证旧进程已停止。
10. ImportTask 版本和普通格式逐文件唯一 IDRange 在 WAL/job 中固定，重试、恢复和 CDC 不重新选择。

这些执行模式与业界成熟机制一致，但不声称算法完全相同：

- Spark `ExternalSorter`：内存排序、immutable spill run、最终 merge、task 结束清理；
- Spark merge sort：每 reader 一个 buffered head，heap 为 `O(fan-in)`；
- Hadoop `Merger`：有限 fan-in 的多轮归并；
- Iceberg：immutable output、manifest-last、精确引用提交，提交状态未裁决时不先删输出。

Planning 和 hierarchical merge 使用“按规范顺序连续分组”是 Milvus 为了恢复确定性做的选择，不是 Hadoop 优先合并小 run 以降低 I/O 的策略。

## 5. ImportJob/ImportTask 版本和 fragment 输入边界

新普通格式和 backup/binlog Import 使用 `ImportTaskV3`；历史 job 和 L0 继续按现有 `ImportTaskV2` 执行。`ImportMsg.version` 是 `int64` 执行兼容标记，只版本化 WAL/worker 执行协议；`ImportJobVersion` 只版本化 DataCoord job 状态机。两者是独立版本序列，不能直接复制或比较。新普通格式和 backup/binlog 请求固定写 Task V3，L0 固定写 Task V2，历史 WAL 缺失字段时按 Task V2。`ImportV2` 在广播前用常量和简单比较检查 Task version；ACK callback 是唯一把 Task V2→Job V1、Task V3→Job V3 的映射结果写入 ImportJob 的边界，之后 Job version 不可变。

ACK 遇到未知 Task version 时，如果消息已有 JobID，则保存只包含查询、GC 和 CDC rollback 所需字段的 Failed Job V1 后 ACK，不分配 file ID，也不读取 collection metadata。如果未知版本消息没有 JobID，则消息本身已损坏，返回 `DataIntegrity` 且不 ACK，保留 broadcaster 记录供人工处理。

DataNode 的第一次读取任务命名为 `ReshardTask`，替代当前只做统计的 `PreImportTask`；正式写 segment 的任务命名为 `ImportTaskV3`，作为 `ImportTaskV2` 的升级。两者是 ImportTaskV3 的内部任务类型，不是两条可选链路。

这里保留一个很窄的版本选择接缝：DataCoord 在请求归一化/广播边界决定 Task version，`ImportMsg.version` 负责把决定带过 WAL/CDC；ACK callback 再通过简单的常量比较生成 Job version，checker 只按已持久化的 Job version 派发。当前所有新普通格式、backup/binlog 请求都无条件写 Task V3，L0 和历史缺失值走 Task V2；本次不增加用户参数、开关或新的 RPC。未来增加 Task V4 或 Job V4 时必须显式增加常量和映射分支，不能依赖数值相同，也不能在任务执行中临时切换。这两个字段都不暴露给 Proxy 用户 API。

为什么版本必须进入 `msg.proto`：Streaming/WAL 的消息会被 broadcaster catalog 持久化、被 CDC 原样复制，也可能在 DataCoord/StreamingCoord 重启后重新执行 ACK callback。滚动升级期间，旧 Import broadcaster task 不能要求全部完成后才能开 gate；它们可以继续按“字段缺失即 V2”恢复，同时新消息带 V3。若只把版本放在 DataCoord 内存或 job catalog，尚未创建 job 的 pending/replicated broadcaster task 就没有可靠版本事实，重新 ACK 时只能猜测；猜错会把 V3 当 V2（或反过来），导致错误 task、错误路径和无法恢复。因此这是消息兼容字段，不是把执行机制暴露给用户。

对应的控制面字段只表达版本：

```protobuf
message ImportMsg {
  int64 version = 11;
  // 0: 未指定或历史字段缺失，按 ImportTaskV2 解释。
  // 2: ImportTaskV2。
  // 3: ReshardTask + ImportTaskV3。
}

enum ImportJobVersion {
  ImportJobVersionUnspecified = 0;
  ImportJobVersionV1 = 1; // PreImportTask + ImportTaskV2
  ImportJobVersionV3 = 3; // ReshardTask + ImportTaskV3
}
```

这个改动保留字段号 `11` 和 protobuf varint wire type；旧 enum 的 `0/2/3` 与新 `int64` 的 `0/2/3` 在线上编码相同，因此滚动升级时新旧二进制可以互相读取这些已定义值。变化只删除生成代码中的 enum 类型约束，未知整数仍由 DataCoord 的常量比较和错误分支处理。

旧 Catalog 中缺失 Job version 的历史记录解释为 Job V1。Job version 2 没有定义，也不需要 `reserved 2`；该 enum 是本分支内部协议，没有需要保护的已发布字段。显式写入但当前不支持的 Job version 属于持久化状态不兼容，DataCoord 拒绝恢复，不猜测状态机。

`ReshardTaskPlan`、`ReshardManifest` 和 `ImportTaskPlan` 是 job 生命周期内的内部临时 protobuf，不再各自增加 `format_version`。向后兼容的字段变化直接使用 protobuf 演进；V3 正式发布后，任何不兼容的执行协议变化必须提升 `ImportMsg.version` 和 `ImportJobVersion`，创建新的 job/task workflow，不能让同一个 V3 reader 假装支持多个对象版本。本分支尚未发布，因此删除 result object、改为 Completed Query 直接返回结果是对首版 V3 的直接替换，不额外制造 V4。V3 临时对象只在当前 job 和 retention 窗口内存在，活动 V3 job 未清空时禁止回滚到不认识该协议的二进制。

Planning 和 Importing 使用内部 `FragmentRef`，而不是直接引用 `ReshardTask`。每个 ref 精确引用一个完整、已接受的内部 fragment；本次不支持切片或外部 fragment 输入，也不为它们预留字段、入口、API、RPC、鉴权或对象所有权协议。

## 6. Import WAL、file ID 和确定性 RowID/AutoID

### 6.1 为什么必须在广播前完成

当前源码在 `createImportJobFromAck` 中给文件分配 ID。CDC 两端会分别执行 ACK callback，因此会得到不同 file ID；当前正式 Import 又在每个本地 task 中分配 IDRange。新方案要求同一普通源文件在所有重试、节点和 CDC 集群上生成相同 RowID/AutoID PK，因此这些决定必须进入 WAL，不能留给 ACK 后本地决定。

普通 Import 的 `DataTs` 仍以 Import WAL 广播的最大 vchannel timetick 为准，沿用当前 ACK callback 的 `result.GetMaxTimeTick()`，不在广播前另分配 timestamp。它不影响 Reshard 路由，ACK 后写入 ImportJob，并在所有 Importing run 中固定复用。逐 vchannel `DataTs` 仍是当前源码已有 TODO，本方案不顺带改变该语义。

旧 Import 的行为简述：DataNode 收到 task 后用 task request 中的一段本地 allocator range，同时分配 RowID/AutoID 和本 task 的 log ID；普通 AutoID PK 为该 allocator 生成的 `int64`，VarChar PK 为其十进制字符串，RowID 与生成 ID 相同，timestamp 使用 request 的 `Ts`。这里的 `MaxUint32` 不是生成的 RowID 数值上限；生成 ID 是 `int64`，还包含 cluster bits。它限制的是一次 allocator 请求的数量。backup 会先关闭 AutoID，保留源文件中的 PK/RowID/timestamp。PR #51825 把普通 Import 改成“DataCoord 广播前按文件预留字面 PK range，DataNode 只消费这个 range”，从而让 CDC 主备生成相同 PK；它明确拒绝单文件的行数/分配宽度上界超过 `MaxUint32`，要求拆分文件。ImportTaskV3 沿用这个确定性分配语义，不再在 DataNode 为普通 Import 重新取号。

ImportTaskV3 在每个集群内使用本地 collection metadata 的 vchannel 顺序做 PK hash 路由，保证导入数据与该集群后续 insert/delete/upsert 使用同一 shard 规则。CDC 复制 CreateCollection 时按原数组位置替换 pchannel 前缀，保留 vchannel shard 后缀和数组顺序；后续 DML 也按同一 pchannel 映射，所以两端的本地 metadata 仍表达同一 shard ordinal。CDC 只要求最终用户数据一致，不要求两端生成相同的 fragment、segment、task ID 或 channel ordinal。ACK callback 从 `result.Results` 收集已完成 channel，`ReadyVchannels` 只作为集合参与 readiness 判断；不能把该 map 的遍历顺序当作 job 路由顺序。`createImportJobFromAck` 继续使用 `importCollectionInfo.VChannelNames` 保存 `ImportJob.Vchannels`。

partition 顺序同样是协议，不是集合。当前 `HashData` 用 `partitionIDs` slice 下标作为 partition ordinal，`hash % P` 的结果直接索引该 slice。因此 ImportTaskV3 以 `ImportMsg.partitionIDs` 的字面顺序作为规范 partition 顺序；Proxy、DataCoord 广播、ACK、老 Proxy 转发、CDC 消息体、ImportJob、ReshardTaskPlan 和 ImportTaskPlan 全部原样保留，不能排序、去重或从 collection metadata 重新生成。

新的顺序：

```text
DataCoord.ImportV2
→ 校验请求
→ backup 时调用现有 ListBinlogImportRequestFiles，展开为按源 segment 组织的规范 ImportFile
→ `jobID != 0` 时复用请求中的 job ID；仅 `jobID == 0` 时分配 job ID
→ 按稳定 file key 规范化 ImportFile 顺序并分配稳定 file ID
→ 普通格式计算每文件行数上界并分配每文件唯一的字面 IDRange
→ 固定 `version`
→ 构造完整 msgpb.ImportMsg
→ WAL 广播
→ ACK callback 原样创建 ImportJob
```

backup 展开必须在广播前完成，否则主备 ACK callback 各自 LIST 后不仅 file ID 不同，规范文件顺序也可能不同。普通与 backup 都使用同一 stable file key：`format + 按语义顺序保留的 length-prefixed paths`。不同 key 按字节序排序；完全相同的 key 只能用原请求中的 occurrence ordinal 区分，因为相同路径没有其他可推导身份。这一处明确使用原请求 occurrence ordinal，其余不同 key 不依赖用户清单顺序、并发 LIST 返回顺序或完成顺序。规范顺序固化后再分配 file ID 和每文件唯一的 pre-allocated AutoID range。ACK callback 只有处理老 WAL 时才保留当前本地展开和 file-ID 分配兼容分支；不新增“拒绝重复路径”防御规则。

继续复用当前真实的 `ImportMessageV1 + msgpb.ImportMsg`，只追加 `int64 version` 和逐文件范围，改动小于新建一套 WAL message、builder、registry 和 adaptor。需要修改外部 `milvus-proto/proto/msg.proto` 的 `ImportMsg`/`ImportFile`，再更新本仓 go-api 依赖。不得手改生成文件。这里的 `ImportMessageV1` 是现有 Streaming message payload 版本名，不是新增 Import 链路名称。`ImportMsg.version` 只用于 WAL/worker 协议选择；Job 状态机使用独立 `ImportJobVersion`。WAL 仍然同时保存真正影响数据结果的文件、路径、选项、schema、channel/partition 顺序、timestamp 和 `pre_allocated_auto_ids`。

### 6.2 普通格式逐文件范围

```protobuf
message ImportFile {
  int64 id = 1;
  repeated string paths = 2;
  common.IDRange pre_allocated_auto_ids = 3; // 普通格式每文件唯一的连续 range；backup/L0 为空
}
```

以 PR #51825 的行为为准，直接复用它提供的单个 `common.IDRange`/`ImportFile.pre_allocated_auto_ids`，不新增 `ImportIDRange`，也不使用 repeated ranges。本次当前 checkout 的 go-api 依赖仍可能生成旧字段名（例如 `PreAllocatedAutoIds`）；实施时必须升级到 PR #51825 的最终 proto，并全仓统一使用最终字段名，不能同时保留两个同义字段。若合入时字段名发生调整，以 PR #51825 最终 proto 为准，但语义固定为“一个文件一个连续 range”。本仓 `internal.ImportFile`、ACK、StreamingCoord 转发、ImportJob 和 task plan 都原样复制这个单值 range；`internal.ImportRequestInternal.version` 和 `msg.ImportMsg.version` 都使用相同的 `int64` 字面值，`datapb.ImportJob.version` 使用独立 `ImportJobVersion`。`ImportV2` 广播前只做常量比较，只有 ACK job 创建边界会把映射结果持久化。实现前运行 proto dependency/lint 检查，不能复制同义 range 类型或形成 import 循环。

对每个普通 `ImportFile` 分配一个互不相交的字面 `[begin, end)`。广播端按 PR #51825 的批处理方式调用当前 `common.AllocAutoID(..., paramtable.CommonCfg.ClusterID)`；单文件范围不能跨 allocation batch。写入 WAL 的 begin/end 已经物化主集群当前数字 `common.clusterID` 的 bits；这里不是字符串 `clusterPrefix`，也不从 CDC `ReplicateHeader.ClusterID` 推导。DataNode 和 CDC 备集群不得再用本地 cluster ID 做第二次变换：

```text
range = file.pre_allocated_auto_ids
generated_id = range.begin + row_offset
RowID        = generated_id
```

如果 collection 使用系统生成 AutoID，并且按当前 `allow_insert_auto_id` 规则需要生成 PK，则：

```text
Int64 PK   = generated_id
VarChar PK = decimal(generated_id)
```

如果源文件提供并允许保留 PK，则保留源 PK，但 RowID 仍取该文件范围。这个规则与当前 `AppendSystemFieldsData` 的语义一致，只把 allocator 结果变成按文件固定的字面范围。

范围宽度：

- Parquet footer、NumPy header 可得精确行数，直接使用精确值作为上界。
- JSON/JSON Lines/CSV 使用文件物理字节数作为硬上界，因为每条有效记录至少占一个字节。
- 单文件上界为 0 时 range 为空，不调用 allocator；实际读到任何行都会因范围不足失败。
- 单个文件的行数/分配宽度上界超过 `MaxUint32` 时直接拒绝请求，错误明确要求拆分文件；这不是说生成的 PK 数值超过 `MaxUint32`，而是一个文件不能需要超过 `MaxUint32` 个连续分配位置。不通过多次 allocator 请求拼接单文件 range，不本地续号，也不退回 ImportTaskV2。整个 job 仍可包含多个文件、分别拥有各自 range。
- `row_offset` 只在该文件唯一的 `[begin,end)` range 内定位。实际行数超过 range 宽度时立即失败，不能本地续号、不能重新申请、不能退回 ImportTaskV2。

每个文件只保存一个字面 range；多个文件的 range 总量可以自然超过 `MaxUint32`。备集群只消费字面值，不调用本地 allocator，也不按自己的 `common.clusterID` 重写。

`pre_allocated_auto_ids` 的宽度就是该文件的行数上界，不再存第二个 `row_count_upper_bound` 字段，避免两个真相源不一致。正式 log ID 与这些 RowID/AutoID ranges 独立分配，不能从文件 range 剩余部分中切 log ID。

文件内 `row_offset` 从 0 开始，按现有 reader 的逻辑记录顺序：

- Parquet：row group 和行的读取顺序；
- NumPy：对齐后的数组下标；
- CSV：表头后数据记录顺序；
- JSON：逻辑记录顺序。

坏记录让整个 task 失败，不跳过后继续编号。

### 6.3 backup/binlog 差异

backup 同样进入 Resharding，但不分配普通格式的 `pre_allocated_auto_ids`：

- 保留源 PK、RowID、timestamp；
- 临时 fragment 包含系统字段；
- 继续使用当前 binlog reader 的 schema 验证、time range 和 deltalog 过滤；
- Reshard 得到的是已经按 backup 语义过滤后的行，Importing 不得再次执行同一 backup time-range/deltalog 过滤；
- 不重新运行源中已存在的 function output；保持当前 backup 规则。

### 6.4 CDC 不变量

主、备必须相同：

- job ID、ImportTask version；
- 稳定 file ID；
- 普通格式每文件唯一的连续字面 `pre_allocated_auto_ids`；
- 源文件清单和影响 PK/RowID 的不可变选项；
- 生成的 RowID、AutoID PK 和行路由结果。

主、备允许不同：

- ReshardTask ID、task 内 `run_id`、DataNode；
- fragment 切分和对象路径；
- ImportTaskV3 task ID、正式 segment ID、索引 ID。

现有 CommitImport/HandleCommitVchannel 是 per-vchannel fence，不是整个 job 同时可见。备集群尚未到 `Uncommitted` 时，CommitImport callback 保持重试；这里的组件和动作与旧 Import 机制完全对齐：仍由 Streaming durable broadcaster/WAL callback 重试，仍由 `HandleCommitVchannel` 逐 vchannel 推进，不增加 job-wide 可见 fence。

## 7. 作业状态机和推进条件

### 7.1 新状态

在 `internal.ImportJobState` 尾部追加，不重编号现有值：

```text
Planning = 10
Resharding = 11
```

`Importing` 复用现有值 3。完整状态机：

```text
Pending
  → Resharding
  → Planning
  → Importing
  → IndexBuilding
  → Uncommitted
  → Committing
  → Completed
```

旧 job 继续使用：

```text
Pending → PreImporting → Importing → Sorting → IndexBuilding → ...
```

`GetImportProgress` 对外仍可把 `Resharding`、`Planning`、`Importing`、`IndexBuilding` 映射为外部 `Importing`，但内部指标和日志必须保留真实阶段。`Resharding` 是 V3 新增的 job state（enum 值 11），V3 在该阶段调度新的 `ReshardTask`，不再使用旧 `PreImportTask` 的“只统计不落 fragment”逻辑。

### 7.2 各状态的提交点

| 状态 | 进入条件 | 离开条件 |
| --- | --- | --- |
| Pending | ACK callback 已保存 job | ImportTask version 已解释；V3 的 ReshardTask plan/task 已完整保存 |
| Resharding | 全部 ReshardTask 已成为可调度状态 | 每个 task 都有一个 catalog 裁决并验证的 ReshardManifest |
| Planning | manifest barrier 已满足 | 全部自包含 ImportTaskPlan、预分配的 SegmentInfo 和 ImportTaskV3 已保存，最后将 job 状态写为 Importing |
| Importing | task 集合完整可见 | 每个 Importing task result 已组合提交到 SegmentInfo 和 task meta |
| IndexBuilding | ImportTaskV3 各 task 的 committed result 已固定非零 segment | 只按 committed result 中的正式 segment 沿用当前 wait-for-index 规则；Import checker/index inspector 组件和“重复检查直到索引完成”动作与旧 Import 对齐 |
| Uncommitted | 索引条件满足，segment 仍不可见 | auto commit 广播或显式 CommitImport；Import checker/durable broadcaster 和 manual commit 行为与旧 Import 对齐 |
| Committing | 至少一个 commit fence 开始处理 | 所有 vchannel 已记录 committed；Streaming WAL replay/`HandleCommitVchannel` 的逐 vchannel 幂等恢复与旧 Import 对齐 |
| Completed | 所有 vchannel commit 完成 | 终态保留与 GC；用户可见结果与旧 Import 对齐，V3 只额外回收新临时前缀/task catalog |

空数据 job：DataCoord 对所有 ReshardManifest fragment rows 求和为 0 后不创建 Importing segment。`auto_commit=true` 可沿用当前空导入直接 Completed；`auto_commit=false` 进入 Uncommitted 等待显式 commit。

### 7.3 Importing 到 IndexBuilding 的边界

ImportTaskV3 的全部 task result 完成后，checker 把 job 推进到 `IndexBuilding`。与 V2 一样，每个非零 SegmentInfo 在其完整 result 被 DataCoord 应用后直接成为 `Flushed`，index/stats inspector 可以提前消费；`IsImporting=true` 继续阻止 CommitImport 前的查询可见性。task `Completed` 仍是 DataCoord 接受完整 task result 的最后 marker，但不再额外增加 V3 专用发布 gate。

## 8. ReshardTask 分组和不可变计划

### 8.1 原子单元

完整 `ImportFile` 是不可切分原子单元：

- NumPy 的多字段路径不能拆开；
- backup 展开后的一个源 segment 的 insert/delta 路径组合不能拆开；
- 同一文件只属于一个 ReshardTask；
- task 分组一旦持久化，重试和恢复不得重新分组。

### 8.2 大小估算

分组发生在第一次完整读取前，只能使用低成本元数据：

- JSON/CSV：物理文件字节数；
- Parquet：优先使用 footer uncompressed size，无法获得时用物理字节数；
- NumPy：header 的 shape × dtype width；
- backup：现有 `ListBinlogImportRequestFiles` 只返回每个源 segment 的 insert/delta prefix，不带 size。展开并稳定排序后，DataCoord 只做一次低成本路径展开：沿用 binlog reader 的 `listInsertLogs`/delta walk 规则得到该源 segment 实际要读的对象清单，对每个对象调用 `ChunkManager.Size` 并累加；同时把这份准确对象清单或 reader 可直接消费的展开结果写入 `SourceFileSpec`。ReshardTask 不再对 prefix 重新 LIST，避免同一 job 再次展开。本方案不用这份清单做源文件变化检查。

估算只用于均衡 task，不参与数据正确性。完成后 fragment manifest 提供准确规范化行数和字节数。

### 8.3 稳定 BFD（Best-Fit Decreasing）

使用以下确定性算法：

```text
files = sort(files, estimated_bytes DESC, file_id ASC)
tasks = []

for file in files:
    best = none
    for task in tasks: // task_id ASC，已创建顺序
        remaining = task_target_bytes - task.estimated_bytes
        if file.estimated_bytes <= remaining:
            // 选择放入后 remaining 最小的 task；相同则 task_id 更小
            if best == none || remaining < best.remaining ||
               (remaining == best.remaining && task.id < best.task.id):
                best = (task, remaining)
    if best != none:
        best.task.add(file)
    else:
        tasks.append(new task with file)
```

规则：

- 只做一维 `estimated_bytes` BFD；不做回溯、交换、二次重排或 CPU/内存/文件数多维求解。
- `task_target_bytes` 复用 `dataCoord.import.maxSizeInMBPerImportTask` 作为软目标。
- 单文件超过目标时独占一个 task，不因为超过软目标而拒绝。
- 文件按 `estimated_bytes DESC, file_id ASC` 排序；候选 task 按当前 `task_id ASC` 扫描，剩余空间相同取 task ID 更小者，保证重启、重试和配置不变时分组字面稳定。
- BFD 的复杂度为 `O(file_count × task_count)`；当前 Import 请求文件数已有上限，首期不引入平衡器、优先队列或近似回溯。
- 单文件是否超过现有 `dataNode.import.maxImportFileSizeInGB` 仍按当前用户输入校验处理。
- 不再增加独立 `filesPerReshardTask` 配置；请求已有 `maxImportFileNumPerReq`，task 由字节目标控制，减少无必要配置。

旧链路的大小检查位于将被 ImportTaskV3 替换的 `PreImportTask.readFileStat`，所以不能只在配置表中写“保持不变”。ImportTaskV3 对每个 `SourceFileSpec` 创建完整逻辑 `importutilv2.Reader` 后，必须在该 source 第一次 `Read()` 前调用一次 `Reader.Size()`，并与执行时的 `dataNode.import.maxImportFileSizeInGB` 比较：JSON/CSV/Parquet 是单对象大小，NumPy 是同一 ImportFile 全部字段文件大小之和，backup 是显式展开后全部 insert 对象之和，严格沿用各现有 reader 的 `Size()` 语义。超过上限继续返回当前 ParameterInvalid/InputError；`Size()` 的对象存储错误保留其 SystemError/重试语义。检查失败时关闭 reader、终止 run，不读取该 source，也不发布当前 run 的 manifest；同 task 此前写出的临时对象由文件名中的 run ID 隔离并等待失败清理。DataCoord 的 source `estimated_bytes` 只用于 ReshardTask 分组，不写入 plan，也不能代替这项 DataNode 校验。

### 8.4 plan 内容和存储

每个 task 有不可变 `ReshardTaskPlan`：

```protobuf
message ReshardTaskPlan {
  int64 collection_id = 1;
  schema.CollectionSchema schema = 2;
  schema.CollectionSchema temp_schema = 3;
  repeated string vchannels = 4;
  repeated int64 partitions = 5;
  repeated SourceFileSpec sources = 6; // file_id ASC
  SortSpec sort = 7;
  int64 fragment_size = 8;
}

enum ImportSourceFormat {
  IMPORT_SOURCE_FORMAT_UNSPECIFIED = 0;
  IMPORT_SOURCE_FORMAT_JSON = 1;
  IMPORT_SOURCE_FORMAT_JSON_LINES = 2;
  IMPORT_SOURCE_FORMAT_CSV = 3;
  IMPORT_SOURCE_FORMAT_PARQUET = 4;
  IMPORT_SOURCE_FORMAT_NUMPY = 5;
  IMPORT_SOURCE_FORMAT_BACKUP_BINLOG = 6;
}

message SourceFileSpec {
  int32 ordinal = 1;             // 规范文件顺序
  internal.ImportFile file = 2;         // id/paths/pre_allocated_auto_ids 的唯一真相源
  ImportSourceFormat format = 3;
  bool backup = 4;
  repeated ExpandedBinlogField insert_fields = 5; // backup only，field/group ID 升序
  repeated string delta_paths = 6;               // backup only，path 升序
  ReaderOptions options = 7;
}

message ExpandedBinlogField {
  int64 id = 1;
  repeated string paths = 2;             // path 升序
}

message ReaderOptions {
  string separator = 1;          // 已规范化后的单个 rune；非 CSV 为空
  string null_key = 2;           // 非 CSV 为空
  uint64 start_ts = 3;           // 非 backup 为 0
  uint64 end_ts = 4;             // 非 backup 为 MaxUint64
  int64 storage_version = 5;     // backup only
}
```

`ReshardTaskPlan` 是跨重试不变的逻辑计划。plan 和输出位置都由 `job_id + task_id` 固定推导，不在 task、request 或 plan 中重复保存 path。`SourceFileSpec.file` 直接复用已扩展的 `internal.ImportFile`，不再定义第三份 range 结构。backup 的 insert 对象不能只存一个扁平 path 列表，必须保留当前 reader 需要的 `field/group ID → ordered paths` 关系；Storage V2/V3 的 column group ID 也使用同一结构。BFD 使用的文件估算大小在分组完成后已经失去执行语义，不写入 plan。

当前 binlog reader 只接受一到两个 segment/delta prefix，并在 `init` 中再次 LIST/Walk。ImportTaskV3 必须给 `internal/util/importutilv2/binlog` 增加一个显式对象入口：直接接收 `field/group ID → ordered insert objects` 和 `ordered delta objects`，复用现有 `verify`、`createFieldBinlogList`、`storage.NewBinlogRecordReader`、delete reader 和 time/delete filter。该入口不允许再 LIST；旧 prefix 入口原样保留，只服务 ImportTaskV2。对象顺序已经写入 task plan，重试不能按新的 LIST 结果改变输入。

plan 必须保存会真实改变 reader 语义、且不含凭证的已解析 typed option，不能只写一句“保存 option”却没有字段。首期 allowlist 就是当前源码实际使用的 CSV `sep/nullkey`、backup `start_ts/end_ts` 和源 `storage_version`；别名 `startTs/endTs` 在广播前统一解析为 typed timestamp，不原样保留两套 key。`timeout`、`auto_commit`、`skip_disk_quota_check` 属于 job 控制语义，不进入 `ReaderOptions`；backup/L0 由现有 option 解析和 ImportTask version 共同确定；`ezk` 永远不进入 plan。若以后 reader 新增会影响解析结果的非敏感 option，必须先把 typed 字段加入 plan，再允许 ImportTaskV3 使用，不能执行时临时读取变化后的 job options。

plugin context 可包含密钥，不能写入 OSS plan/manifest/result、ImportTask/GC catalog、日志、metric 或 trace。当前 `ImportJob.options` 本来就持久 backup `ezk`，本方案不扩大、复制或重新设计这个已有凭证边界。DataCoord 每次创建 V3 worker task 时，用已持久的 job options/schema 重新调用现有 `GetReadPluginContext(job.options)` 和 `WrapPluginContext(collectionID, schema.properties, request)`，只把读/写 context 放在 CreateTask RPC payload 内存中。`WrapPluginContext` 的 type switch 增加 V3 request 类型，不引入新的密钥版本协议。

每个 Reshard plan 先写固定的 `plans/reshard/{task_id}/plan.pb`，再按稳定 task ID 写入 catalog；task 一旦以 Pending 保存就是完整、自包含且可独立调度的。恢复按 task ID 推导 plan path，读取已覆盖的 source file ID，只对剩余 source 继续做 stable BFD并创建缺少的 task，不删除或重新分组已有 Pending/Running/Completed task。全部 source 都有 task 后推进 job 到 Resharding；即使本轮没有缺少的 task，也必须重试该状态更新。job 状态只驱动 checker 阶段，不是 scheduler 发布 marker。task catalog 不保存 plan/output path，也不增加 task-set index、checksum 或 generation。

## 9. SortSpec、临时 schema 和行语义

### 9.1 SortSpec

完全沿用当前 DataNode sort compaction 的排序字段：

```protobuf
enum SortKeyType {
  SORT_KEY_TYPE_UNSPECIFIED = 0;
  SORT_KEY_TYPE_INT64 = 1;
  SORT_KEY_TYPE_STRING = 2;
}

message SortFieldSpec {
  int64 field_id = 1;
  SortKeyType key_type = 2;
}

message SortSpec {
  repeated SortFieldSpec fields = 1; // 比较优先级顺序
}

message SortKeyComponent {
  SortKeyType key_type = 1;
  sint64 int64_value = 2; // INT64 时使用
  bytes string_value = 3; // STRING 时使用原始 UTF-8 bytes
}

message SortKey {
  repeated SortKeyComponent components = 1;
}
```

```text
普通 collection：
  [PK]

namespace-enabled collection：
  [partition key, PK]
```

比较规则：

- Int64 数值升序；
- VarChar 使用当前 Go 字符串/字节字典序升序；
- 不增加 RowID、file ID、源行号等隐藏 tie-breaker；
- 不提供升降序、collation 或 null ordering 配置；
- 重复 key 全部保留，相对顺序不作为 API 契约。

`SortSpec` 写入 ReshardTaskPlan 和自包含 ImportTaskPlan。ReshardManifest 不再回显 SortSpec；manifest 只有在 DataNode 按 plan 完成全部 sort/write 后才发布，因此重复保存再比较同一个输入没有增加提交保证。fragment 的 first/last key 当前不参与 Planning 或 Importing，也不再逐行计算和持久化。

### 9.2 普通格式规范化边界

Reshard 对普通格式按当前 Import 顺序执行：

```text
reader.Read
→ CheckRowsEqual
→ CheckStructArrayConsistency
→ AppendNullableDefaultFieldsData
→ FillDynamicData
→ 按每文件唯一的连续 `pre_allocated_auto_ids` 物化 RowID/AutoID PK
→ HashData 的相同 hash 规则
→ fragment sort
```

临时 fragment：

- 包含全部原始用户字段、materialized PK、RowID；
- 不写 timestamp 列，普通数据的时间语义继续由 `ImportJob.DataTs` 表示；
- 不运行 embedding、BM25、MinHash；
- 不生成 bloom/BM25 stats、text index 或 TEXT LOB；
- function output 字段若本应由 Milvus 生成，临时 schema 不要求其存在；最终轮统一生成。

这避免被 TTL/delete 最终过滤掉的行提前调用外部 embedding，也避免中间多轮归并重复执行函数。

### 9.3 backup 临时行

backup reader 已返回保留系统字段并经过 backup time range/deltalog 过滤的行。其 fragment：

- 保留 PK、RowID、timestamp；
- 保留当前 reader 判定有效的 function output；
- 对目标 schema 缺失的 nullable/default/dynamic 字段使用当前 Import 规则补齐；
- 后续 Importing 不再次执行 backup source filter。

### 9.4 TEXT temporary schema adapter

当前源码中逻辑 TEXT 是 UTF-8 string，但 `storage.NewRecordBuilder` 对内部 manifest TEXT 使用 binary LOB reference，普通 packed writer也会拒绝 TEXT schema。因此临时数据不能直接用正式 TEXT 物理 schema。

最小实现：

1. 深拷贝 collection schema 得到 `temp_schema`。
2. 保持 TEXT 的 field ID、名称、nullable、字段顺序和逻辑用途不变，但给临时 packed reader/writer 使用的 Arrow schema 显式映射为 UTF-8 string。
3. Reshard fragment 和 intermediate merge run 都保存原始 UTF-8，不创建 LOB。
4. final transform 把 UTF-8 TEXT 交给现有 TEXT-aware正式 writer；函数在 LOB 编码前读取原始字符串。
5. 当前不支持的 TEXT + storage version 组合继续由现有校验拒绝，本方案不扩大用户能力范围。

不要通过把 `DataType_Text` 永久改成 VarChar 来欺骗业务 schema。应在 storage 层增加明确的 temporary raw-TEXT Arrow schema/builder 入口，使逻辑 schema和临时物理 schema分离。

## 10. Reshard 执行：内存、spill、排序和 fragment

### 10.1 执行流水线

```text
完整 ImportFile 队列
→ 单 reader 顺序读取
→ parse / normalize / materialize IDs
→ route to BucketKey
→ 有界 bucket 数据
→ storage.Sort
→ 单对象 packed Parquet
→ FragmentDescriptor
→ manifest-last
```

首期 V3 task 内 fragment sort concurrency 固定为 1，Reshard source reader concurrency 也固定为 1；多个 V3 task 通过 DataNode slot admission 并行，而不是在一个 task 内无限增加 reader。路由队列必须有界，不能把满 bucket 无界堆积。

#### 10.1.1 task 级总内存契约

不能只限制单个 fragment，否则高 `V × P` 时每个活跃 bucket 各留一个尾 batch，task 驻留内存仍会随 bucket 数线性增长。每个 V3 task 的本地 working-set 上限固定为：

```text
task_memory_budget = slot * dataCoord.import.memoryLimitPerSlot
```

DataNode 在 Create V3 task 时先校验 `slot > 0`，再以该字面值计算本 task 的固定 buffer、batch、spill 和 writer 工作集；不从当前配置重算。V3 不调用旧 Import 的全局 `MemoryAllocator`，也不在任何全局 `Acquire` 上阻塞等待。一个 batch 经解析、nullable/default、dynamic、RowID/PK 物化和路由后，按 `InsertData.GetMemorySize()` 和 routing slice 容器开销计算 resident bytes；达到本 task 的本地上限时立即 spill 或完成当前 fragment，不能继续积累。

```text
更新 task-local resident_bytes
→ 持有数据
→ spill/sort/writer 消费并释放
→ 减少 task-local resident_bytes
```

reader 固定 buffer、sorter 临时 row index、packed reader/native state、packed writer buffer 和 final transform buffer 都必须在 task slot 的 peak estimate 中预留；不能把这些内存重复计入可增长的 bucket resident data。这里的 `task_memory_budget` 是实现和测试使用的本 task 预算，不是一个跨 task 的阻塞 allocator。

单个 bucket 的累计 logical bytes 达到 fragment 目标（§10.2）时立即完成该 fragment；否则若所有 bucket 的 resident bytes 之和超过 task 预算，不阻塞等待，而是按确定规则把最大的 bucket tail spill 到本地 IPC：

1. 选 `in_memory bytes` 最大的 bucket；
2. 相同时选 `(channel_index, partition_index)` 最小者；
3. 把该 bucket 当前内存 batch 写成一个 immutable IPC chunk；
4. writer close 成功后立即释放对应 batch 并减少 task-local resident bytes；
5. 失败则 run 失败，不丢弃该内存数据后继续。

EOF 时按 bucket ordinal 升序把每个仍有数据的 bucket（spill chunks + 内存尾批）统一 replay/sort 成 fragment，不保留另一条“直接用内存尾批”分支。

### 10.2 fragment 目标

`dataCoord.import.fragmentSizeInMB=128` 表示规范化后、未压缩逻辑字节的软目标，不是：

- Parquet 对象物理大小；
- row group 硬大小；
- 每 bucket 预留；
- task 内存硬要求。

实际批次上限：

```text
effective_fragment_input = min(
    configured_fragment_target,
    本 task 当前可安全交给 storage.Sort 的输入额度
)
```

`storage.Sort` 会物化该 fragment 的全部输入 record 和 row index，所以内存不足时应提前切小 fragment，不能把 128 MiB 当作必须达到的大小。IPC replay 通过 Arrow record `[start,end)` slice 严格切到已保留的 sort 输入预算内；slice 在 `storage.Sort` 结束前 retain，之后 release，不引用已切换的 borrowed record。单行超过 fragment 软目标可以形成单行 fragment；单行的实际计费内存超过 task-local working-set 上限时，不临时超额、不增加额外单行上限或复杂保护，当前 run 直接以资源错误失败。slot peak-memory 估算是这里的主要保证；设计按绝大多数实际记录可被 slot 正确覆盖处理，极端估算偏差由现有 Prometheus 告警和 SRE 监控发现，不在 Import 内再建第二套防线。

### 10.3 不预创建 `V × P` 个 writer

只为实际出现数据的 bucket 建轻量状态：

```text
BucketState {
  in_memory_batches
  logical_bytes
  spill_chunks[]
  next_seq
}
```

长期驻留的只是小 map 和计数，不是每 bucket 一个 packed writer 或 5 MiB multipart buffer。任意时刻最多一个 sorter 和一个 packed writer 工作。

### 10.4 Arrow IPC spill

本地 spill 使用 Apache Arrow Go 已有 IPC stream writer/reader，不自研行格式、frame CRC 或 reopen-append 协议。

路径：

```text
{local_root}/import_v3/{job_id}/{task_id}/
  {run_id}_{channel_index}_{partition_id}_{chunk_seq}.arrow
```

规则：

- 一个 chunk 只属于一个 bucket；
- writer 关闭后 chunk 不可变；
- 同一 bucket 后续继续 spill 时创建新的数字 `chunk_seq`；
- EOF 后按 `(channel_index, partition_index, chunk_seq)` 顺序回放；
- 每次只累计到 `effective_fragment_input`，再调用 `storage.Sort`；
- IPC reader 产生的每批数据也必须先 `Acquire`，sort/writer 消费后立即 `Release`；
- spill 文件不上传、不进 manifest；
- task 终态清理本 task 的本地目录；DataNode 启动时清扫整个本地 `import_v3` 根目录。

本地盘不足、只读或 IPC 写失败时 run 失败并可换节点重试；不退回无界内存。

### 10.5 fragment writer

每个 fragment：

- 一个 bucket；
- 一个 column group；
- 一个 packed Parquet 对象；
- 使用当前 packed writer 的默认编码与压缩，不增加新的 codec 开关；
- `seq` 在同一个 ReshardTask manifest 内单调递增；
- 写入完成、writer close 成功后才记录 descriptor。

当前 `FFIPackedWriter.Close()` 返回的 `Groups` 偏向 manifest transaction，无法直接安全导出物理文件描述。实现需要增加一个只读、拥有清晰生命周期的 Go descriptor，或给旧 `PackedWriter` 增加明确单文件输出 helper。不要通过 OSS LIST 猜 writer 写了哪个对象。

### 10.6 排序实现

对每个有界 fragment 输入调用现有 `storage.Sort`：

- Int64 单键复用 radix sort；
- VarChar/多键复用比较排序；
- 输出 batch size 只控制 writer batch，不改变 fragment 输入上限；
- 写完检查第一/最后 sort key；测试和 debug build 可用单调性 wrapper 全量检查。

预排序固定开启，没有运行时开关。

### 10.7 权威方案开放问题裁决

权威方案最后列出的开放问题在本次实现中全部直接裁决，不把选择留给下一阶段：

| 问题 | 本次选择 | 原因和实现边界 |
| --- | --- | --- |
| ReshardTask 如何分组源文件 | 完整 `ImportFile` 是原子单元，按估算大小降序做一维 BFD | 只按“放入后剩余空间最小”选择已有 task；相同剩余空间按 task ID 稳定打破；不回溯、不切文件、不做多维资源优化。这样 task 数量可控，同时保持计划可重放。 |
| 临时文件格式和 fragment 大小 | fragment 使用现有 packed writer 写单 column group Parquet；默认 128 MiB 逻辑字节软目标；本地 spill 使用 Arrow IPC stream | Parquet fragment 便于 Importing 有界顺序读取和复用 Storage V2/V3 reader；Arrow IPC 适合逐批追加和回放。128 MiB 只影响切分和对象开销，不是必须达到的硬值，单行过大仍可独占 fragment。 |
| spill 内存和磁盘边界 | 使用现有 task slot、`memoryLimitPerSlot`、task-local resident accounting、read buffer 和 localStorage；达到 task-local working-set 上限时同步 spill/flush | 不增加新的多维资源协议，也不使用旧 V2 的全局 Import MemoryAllocator。单行超过 task-local working-set 上限时当前 run 按资源错误终止，主要依赖 slot 估算和 Prometheus/SRE 监控。 |
| Re-shard 预排序是否为可选开关 | 首期固定开启；每个 fragment 在上传前按 `SortSpec` 排序，Importing 使用 strict one-head MergeSort | 预排序让 Importing 直接做归并，heap 只保存每个 reader 的 head，降低内存和后续排序成本。它的额外 CPU/内存开销已经纳入 Reshard task slot；小文件收益有限，但不再引入两条语义不同的执行路径或可变开关。 |

这里的“固定开启”不等于把 `dataCoord.import.fragmentMergeFanIn` 固定为 16：fan-in 仍是运行参数，默认 16、基础范围 `[2,1024]`，实际资源由 task working-set slot 估算约束。

## 11. ReshardManifest 和 run 发布

### 11.1 manifest 内容

每个成功 Reshard task run 发布一份 protobuf `ReshardManifest`：

```protobuf
message ReshardManifest {
  repeated FragmentDescriptor fragments = 1;
}

message FragmentDescriptor {
  int32 channel_index = 1;
  int32 partition_index = 2;
  int64 seq = 3;
  string path = 4;
  int64 rows = 5;
  int64 logical_bytes = 6;
}

message FragmentRef {
  string path = 1;
  int64 row_count = 2;
}
```

manifest 只保存下一阶段真正需要的 fragment 位置、行数、装箱字节和 bucket ordinal。vchannel、partition ID 从 ImportJob 的规范数组按 ordinal 取得；fragment 格式固定为 Parquet，不再逐项回显。source coverage、SortSpec、first/last key、列统计、physical bytes 和 totals 都不被当前 Planning/Importing 使用，因此不计算、不持久化。DataNode 只有完整读完所有 plan sources并成功 close 全部 fragment writer 后才写 manifest，EOF 已由 manifest-last隐含证明。

当前不为 fragment 增加对象内容 SHA-256，也不为校验目的额外执行完整远端 GET。最小流程：

```text
packed writer Write
→ Close 成功
→ 从 writer output 取得唯一对象路径、rows 和逻辑字节
→ 成功后才构造 FragmentDescriptor
```

控制面不计算 manifest checksum。对象存储负责已写对象的内容完整性；`job_id + task_id + run_id` 已经进入确定性 manifest path，DataCoord 只读取当前 catalog task 对应的准确路径。

### 11.2 manifest-last

DataNode 顺序：

```text
写完全部 fragment
→ close 全部 writer
→ 汇总最小 descriptor
→ 写固定的 `reshard/{task_id}/manifests/{run_id}.pb`
→ worker 状态变为 Completed
```

没有 manifest 的 task run 永远不算完成。每个 run 使用固定 manifest path；同一个 `run_id` 的并发执行必须由调度和 task ownership 避免，checksum 不承担 fencing 职责。

### 11.3 Reshard result 接受和裁决

worker Query 只返回当前 run 的状态和原因，不返回 result path。DataCoord 不通过 LIST 找 manifest，接受流程固定为：

1. 在 import-meta task-key 锁下读取当前序列化的 job 和 task，保留两者准确旧值。
2. 要求 job 仍在 `Resharding`、task 的 `run_id` 等于 Query run，且 task 仍绑定相同 node。更大 run 已经成为 current，或 timeout/Abort 已经提交终态时，迟到结果直接视为 stale。
3. 根据 task 的 `job_id + task_id + run_id` 推导准确 manifest path并执行 GET。
4. 校验 descriptor 非空 path、非负 rows/bytes、非负 bucket ordinal和 path 不重复。bucket 上界在 Planning 使用 ImportJob 数组前检查。
5. 用现有 task catalog update把 task 改为 `Completed`。对象已经由固定路径和当前 run 唯一选中，不再保存同义的 result path。
6. catalog 成功后才更新内存 meta；重复接受同一 Completed task按现有幂等状态处理。

`ReshardManifest` 被 task 当前 `run_id` 精确选中后，`Resharding` 才能进入 Planning。manifest 对象存在、DataNode 返回 Completed 或旧 run 曾经 Running 都不能单独作为胜出证据。

## 12. 任务内 run fencing、任务模型和 RPC

### 12.1 当前 Import 如何处理重试

当前 Import 没有独立 attempt/run proto，也没有独立 catalog key：

- DataCoord 只持久化 `PreImportTask` 或 `ImportTaskV2`；
- `retryTimes` 只存在 DataCoord 内存 task 对象中，`GetTaskVersion()` 返回该值；DataCoord 重启后从 0 重新开始；
- Create、Query、Drop 和 DataNode `TaskManager` 都只使用 `task_id`；
- Query RPC 失败或 worker 返回 `Retry` 时，DataCoord 把同一个 task 重置为 `Pending`，再把同一 task ID 派到节点；
- DataNode 的 task map 也只按 task ID 保存，重复 Add 只记录日志；
- 当前正式 Import 重试继续使用同一批预分配 segment，DataCoord 先清零已记录的 imported rows，再重新执行；
- `ErrNodeNotFound` 按当前语义视为 Drop 成功并解绑，不额外证明旧进程已经停止。

V3 与这个模型对齐：不增加 `ReshardRun`、`ImportRunV3`、RunSpec、独立 run state、独立 run catalog 前缀或独立 run 目录。只在 task 自身保存一个单调递增的 `run_id`，相当于把当前仅在内存中的 `retryTimes` 持久化并传到 worker，用于拒绝迟到结果。

### 12.2 V3 task proto

V3 使用独立的 `ReshardTask` 和 `ImportTaskV3`，旧 `PreImportTask`/`ImportTaskV2` proto、catalog API 和执行文件保持不变。状态只保留任务真正需要的单词，不为结果分批提交或停止过程增加状态：

```protobuf
message ReshardTask {
  enum State {
    None = 0;
    Pending = 1;
    Running = 2;
    Retry = 3;
    Completed = 4;
    Failed = 5;
  }

  int64 job_id = 1;
  int64 task_id = 2;
  int64 collection_id = 3;
  State state = 4;
  string reason = 5;
  int64 run_id = 6;
  int64 node_id = 7;
  int64 slot = 8;
}

message ImportTaskV3 {
  enum State {
    None = 0;
    Pending = 1;
    Running = 2;
    Retry = 3;
    Completed = 4;
    Failed = 5;
  }

  int64 job_id = 1;
  int64 task_id = 2;
  int64 collection_id = 3;
  State state = 4;
  string reason = 5;
  int64 run_id = 6;
  int64 node_id = 7;
  repeated int64 segments = 8;
  IDRange log_range = 9;
  int64 slot = 10;
  int64 rows = 11; // 进度统计，不读取 OSS plan
}
```

V3 worker Query response 和持久化 task/job 都不保存独立的 task failure merr code；Query response 的通用 `common.Status` 仍只表达 Query RPC 本身是否成功。DataNode 在执行结束时使用 `merr.IsRetryableErr` 把结果固定为 `Retry` 或 `Failed`，DataCoord 信任该 state：`Retry` 创建新的 `Pending` run，`Failed` 终止 job；重启后同样只按持久化 state 恢复。reason 只用于人读，不参与调度。现有 `GetImportProgress` 形态不变：异步 job 仍按旧 Import 机制返回成功 Status，并用 `state=Failed + reason` 展示失败。同步入口在广播前发现 `ErrImportFailed` 时仍直接返回 InputError Status。

每次重新派发时，DataCoord 在同一个 task 记录内执行：

```text
run_id++
node_id = selected node
state = Running
ImportTaskV3 额外写入新的 segments 和 log_range
清空上一个 run 的 reason
```

该 task 更新和预分配的正式 SegmentInfo 在 CreateTask 前持久化。若需要分批写 SegmentInfo，task 的 `Running + run_id` 是最后 commit marker；marker 未落盘不能发 RPC。

V3 的结果控制流程与当前 ImportTaskV2 对齐：DataNode 在 `Running` 时只通过 Query 返回进度；全部 segment 完成后，`QueryImportTaskV3Response` 以相同顺序返回完整 `SegmentResult` 数组并标记 `Completed`。DataCoord 校验当前 `job_id + task_id + run_id` 后幂等补齐该 task 的全部 SegmentInfo，最后才持久化 task `Completed` marker。不保存 result object/path，也不增加 `Saving`、`Validated`、`Committing` 等结果状态。

`slot` 存在同一 task 记录和 Create request 中，不是独立 reservation/allocator 对象。DataCoord 恢复或重派使用已持久的字面 slot；DataNode 只校验 request 与 task 记录一致。这样 `QuerySlot.importUsed` 能在 DataCoord 重启或 task 重试时继续准确统计 V3 用量，不需要为一次执行再创建 attempt 配额层。

catalog 只新增两个 task 前缀：

```text
datacoord-meta/reshard-task/{task_id}
datacoord-meta/import-task-v3/{task_id}
```

### 12.3 直接 worker request

新增 `taskcommon.Reshard = "Reshard"` 和 `taskcommon.ImportV3 = "ImportV3"`。V3 仍使用现有通用 CreateTask/QueryTask/DropTask 通道，不增加新的 RPC endpoint；`run_id` 作为当前 task 的执行版本，复用通用 property `task_version`。Create payload 直接是 task request，不再套一层 RunSpec：

```protobuf
message ReshardTaskRequest {
  int64 job_id = 1;
  int64 task_id = 2;
  int64 run_id = 3;
  int64 slot = 4;
  index.StorageConfig storage_config = 5;
  repeated common.KeyValuePair plugin_context = 6; // RPC-only
}

message ImportTaskV3Request {
  int64 job_id = 1;
  int64 task_id = 2;
  int64 run_id = 3;
  repeated int64 segments = 4;
  IDRange log_range = 5;
  int64 slot = 6;
  index.StorageConfig storage_config = 7;
  repeated common.KeyValuePair plugin_context = 8; // RPC-only
}
```

Query 和 Drop 继续通过通用 task properties 传 `(task_id, task_version=run_id)`。DataNode `TaskManager` 仍以 `task_id` 为 map key，task value 内保存当前 `run_id`：

| 请求 | 行为 |
| --- | --- |
| 首次 Create | 添加 task，启动一次执行 |
| 相同 run ID | 幂等成功，不启动第二个 goroutine |
| 更小 run ID | stale，no-op 或明确返回 stale |
| 更大 run ID | 取消并替换旧 task value，再启动新 run；取消是进程内 best effort，不构成跨网络停写证明 |
| 旧 run Query/result | 因 task version 不匹配被拒绝 |
| 旧 run Drop | 幂等成功，不影响当前 run |
| 当前 run Drop | 与 V2 控制流一致，cancel/remove 后返回；清理本地 spill，不承诺等待远端或旧 goroutine已经停止 |

`StorageConfig/plugin_context` 只在 CreateTask RPC 内存中传递，不写 OSS、catalog、日志或 metric。Importing 所需的非凭证依赖继续全部进入第 14.5 节的自包含 ImportTaskPlan。TTL 只在实际最终过滤 batch 时读取当前时钟，不创建、不传递、不持久化任何 TTL 参考时间。

### 12.4 重试顺序和旧输出

V3 重试沿用当前 Import 的控制流程，只把重做输入换成不可变 fragment，并用独立对象名降低旧输出污染：

1. task 先进入 `Retry`；
2. 对当前 `(task_id, run_id)` 执行 version-aware Drop；`ErrNodeNotFound` 仍按当前 Import 语义视为解绑成功。Drop 成功只表示本次控制请求已处理，不证明网络分区中的旧 writer 已永久停止；
3. 清理仍处于 `Importing` 的本 run SegmentInfo 进度；若 V3 为对象隔离使用新的正式 segment ID，则旧 `segments` 保持不可见并按标准 Dropped/GC 路径回收；
4. 分配新的 segment IDs 和 log ID range；ReshardTask 不需要正式 segment；
5. 在 task 内增加 `run_id`，保存新 node，再进入 `Running`；
6. 发送新的 CreateTask。

临时对象不增加 run 目录。对象文件名包含 `run_id`，所以迟到 writer 不会覆盖新 run 文件；失败 run 的临时对象允许留到 job 前缀 GC。正式 segment 每个 run 使用新的 segment ID 和正式路径，旧 segment 永远保持不可见并走现有 Dropped/segment GC。该方案不保存无界 run history，也不增加独立清理状态。

首期明确接受与 V2 同类的所有权风险：Query 超时、session 丢失或网络分区后，旧 DataNode worker 可能仍继续运行；`run_id`、独立文件名/segment ID 和结果校验只能阻止旧结果成为当前结果，不能证明旧 writer 已停止，也不能杜绝孤儿对象。恢复正确性依赖“只接受当前 run 的完整结果”和最终 GC，不声称已经实现 writer epoch、lease 或强 fencing。writer epoch/tombstone、带 run 的 `DropAndWait`、worker lease/session fencing 和更强 result-acceptance CAS 都只列为未来增强，不是首期阻塞条件。

`log_range` 是正式 writer 的 ID 预算，不是第 6 节按源文件分配的 RowID/AutoID PK range。当前旧 Import 已经在解决同一个“DataNode 写文件前要先拿到一段 ID”的问题，只是用更粗的估算：

```text
DataCoord AssembleImportRequest
→ totalRows = task 内源文件统计行数之和
→ fieldsNum = user fields + timestamp + RowID
→ binlogNum = fieldsNum + stats + BM25 stats
→ preAllocIDNum = (totalRows + 1) * binlogNum * expansionFactor
→ common.AllocAutoID(... uint32(preAllocIDNum) ...)
→ 把一个 IDRange 随 ImportRequest 发给 DataNode

DataNode
→ 用同一个 LocalAllocator 先按行分配 RowID/AutoID
→ writer 再从剩余范围分配 insert log、PK stats、BM25 stats ID
→ 范围用完则返回 ID exhausted，当前 run/task失败或重试
```

所以“正式 log ID需要 range”不是 ImportTaskV3 新问题。旧方案之所以没有单独讨论 log range，是因为它把 RowID/AutoID 和 log ID 混在同一个大范围，并按 `行数 × 字段数 × expansion factor` 大量预留。这里旧代码实际限制的是一次 task 预留多少个位置，不是最终 RowID 的数值能否超过 `uint32`。它也不是完整的 `MaxUint32` 解决方案：当前 `preAllocIDNum` 先按 `int64` 计算，随后直接转成 `uint32` 传给 `common.AllocAutoID`，没有在转换前拒绝、拆 task 或用多个 range 续号；如果理论值超过 `MaxUint32`，转换会截断高位（若截断后为 0，底层 allocator 仍会按至少 1 个 ID 处理），之后 DataNode 的 local allocator 可能很快以 `ID exhausted` 暴露。DataNode 的 Import 执行路径会把这个错误写成 `ImportTaskStateV2_Failed`；DataCoord 查询到 Failed 后把整个 ImportJob 置为 Failed，不会自动重新估算、更换 allocator 或拆分 task。

ImportTaskV3 按 PR #51825 把 RowID/AutoID 改成每文件固定 range后，正式 log ID不能再从同一大池里拿；同时一个 ImportTaskV3 Importing task可顺序写多个 segment，所以必须把 log ID预算单独表达。首期固定使用一个连续 range，并在 Planning 阶段显式检查 `MaxUint32`，把旧路径中隐含且未闭环的 allocator batch 边界变成可规划、可测试的行为。

每个 Importing task 固定一个 `log_range`，不使用 repeated ranges，也不为该极端边界自动拆 task。该选择延续现有 `LocalAllocator` 的 `[Begin,End)` 接口，避免扩大 allocator、writer 和恢复协议。Planning 必须先证明总预算；超过 `MaxUint32` 时显式拒绝该 task plan，不能像旧路径一样静默转换成 `uint32`。它仍是预算上界，不是“实际精确消耗”。相比旧公式，ImportTaskV3 Planning 已经固定 segment plan、Storage V2 column groups、PK stats 和 BM25 output，能直接按真实 writer 分支计算更紧的上界：

旧路径的数量级可以直接从公式估算。默认 `expansionFactor=10`，令 schema 原始字段数为 `F`，则 `binlogNum=F+4`，接近上界时：

```text
max totalRows per old ImportTask
≈ floor(MaxUint32 / (10 × (F + 4))) - 1
```

示例：

| 原始字段数 `F` | 旧公式每行预留 ID 数 | 接近 `MaxUint32` 的 task 行数 |
| ---: | ---: | ---: |
| 1 | 50 | 约 8,590 万行 |
| 5 | 90 | 约 4,772 万行 |
| 10 | 140 | 约 3,067 万行 |
| 50 | 540 | 约 795 万行 |
| 100 | 1,040 | 约 413 万行 |

当前 `maxSizeInMBPerImportTask=16384`，即约 16 GiB 源文件物理大小软目标。若平均每行 256 B，一个 task 约 6,711 万行；平均每行 1 KiB约 1,678 万行。因此窄行、宽 schema 或高压缩 Parquet 都可能让旧公式越过 `MaxUint32`，不能只用 16 GiB 文件大小推断安全。相反，ImportTaskV3 的正式 log ID 按输出文件数量而非行数计算：

```text
V2: 每个非空 segment = column group 数 + 1 个 PK stats + BM25 output 数
V3: 每个非空 segment = 1 个 PK stats + BM25 output 数
```

例如 V2 每 segment 8 个 column groups、1 个 PK stats、2 个 BM25 stats，共 11 个 ID；要让一个 Importing task 的正式 log ID预算超过 `MaxUint32`，需要约 `4,294,967,295 / 11 ≈ 3.90 亿` 个非空 segment。即使每 segment需要 100 个 ID，也要约 4,295 万个 segment。因此在正常 segment 数和 writer spec下，这个边界远比旧的“按行数放大”难触发；它仍需明确行为，是因为 allocator wire 的单次 `Count` 确实是 `uint32`，不是因为预计生产会接近该规模。

结合当前公开配置上限再估算一次：单请求最多 1024 个 ImportFile，每个最大 16 GiB，源对象总物理大小约 16 TiB；默认普通 segment target约 1 GiB。先忽略压缩/规范化膨胀，整个 job约 16,384 个 segment。即使把 `maxFieldNum=256` 当成极端上界，Storage V2保守按每字段一个 column group，再按最多同量 BM25 output计算，每 segment约 `256 + 1 + 256 = 513` 个正式 log ID，整个 job约 840 万个，只是 `MaxUint32` 的约 0.20%；常见每 segment 11 个 ID时约 18 万个。要在 513 IDs/segment下超过 `MaxUint32`，单个 Importing task需要约 837 万个非空 segment。即使压缩源在规范化后膨胀 100 倍，按上述极端字段数估算也约 8.4 亿 IDs，仍未到上限；而 task plan、SegmentInfo、索引任务和对象数量会更早成为不可接受的系统规模。

所以 ImportTaskV3 “处理超 `MaxUint32`”的含义是：在调用 allocator前做显式检查，不能像旧 Import一样发生 `int64 → uint32` 静默截断；它不代表首期需要真正支持一次 task产生数百万到数亿 segment。首期固定为单 range + 显式失败，不增加 repeated ranges，也不只为该极端值自动拆 task。

- Storage V2 在第一个真实 RecordBatch 到达后才动态拆 column groups，每个 group 一个 insert log ID；close 时 PK stats 一个 ID，每个 BM25 output 一个 ID。
- Storage V3 的 insert parquet 路径不消耗 local log allocator；close 时 bloom/PK stats 一个 ID，每个 BM25 output 一个 ID。TEXT-aware V3 writer 也使用同一 stats 分配路径。
- 零行 segment 的 writer 可能完全不初始化，实际消耗为 0。

因此 Planning 先物化下文的不可变 `WriterSpec`，再按每个已计划非空 segment 计算确定性上界：

```text
V2 required IDs per segment
  = final_column_group_count + 1 PK stats + bm25_output_field_count

V3 required IDs per segment
  = 1 PK stats + bm25_output_field_count
```

DataCoord 在 Planning 时按所有已计划 segment 计算 log ID 预算上界，但不把这个可从 `segments + writer` 推导的中间值再写入 `ImportTaskPlan`。final predicate 可能让某个计划为非空的 segment 最终零行；final adapter 只在第一批保留行到达时创建正式 writer，零行时实际消耗为 0。DataCoord 要求计算值位于 `(0, MaxUint32]`，单次 `AllocN`；超过上限时直接拒绝 Planning 结果。DataNode 用已分配的 `[Begin,End)` 构造 local allocator，范围耗尽由现有 allocator/writer 返回错误，不另建一套预算协议。

当前 V2 BM25 stats collector 和 V3 `appendV3Stats` 都遍历 Go map，field 到 log ID 的映射不稳定。ImportTaskV3 final adapter 必须按 `bm25_fields ASC` 分配 ID并写 stats；PK stats 始终先于 BM25。这不改变预算公式，只保证同一输入的元数据和测试稳定。

DataCoord 生成 plan 和 DataNode 读取 plan 时都必须校验 `2 <= fan_in <= 1024`。Create request 不再重复携带 fan-in；后续所有分组只使用已通过基础校验的 plan 字面值，也不在 DataNode 按当前配置重算。这个范围只是基础 sanity 校验，不是资源证明；真实峰值由 task working-set slot 估算、reader 数和现有外部监控约束。非法 fan-in 是内部协议 SystemError，不能归咎用户输入，也不能用新 run 重试同一份非法 plan。

### 12.5 派发顺序

```text
选择有足够 slot 的 DataNode
→ 分配更大 run ID
→ 为 IMPORT run 分配新的正式 segment ID 集合和 log ID range
→ 构造 V3 task request 的不可变字段
→ 预分配 IMPORT 正式 SegmentInfo
→ 在一个 guarded composite update 中递增 task.run_id、保存新 node 和全部 SegmentInfo
→ 以当前 task 的 state=Running 作为最后 commit marker
→ 发送 CreateTask
→ worker 接受后保持 state=Running
```

task 记录、预分配的 IMPORT SegmentInfo 和 task 指针必须先持久化，再发 RPC。RPC 超时或结果不确定时，下一次必须使用更大的 `run_id`，不能重用同一 run 但换规格。DataCoord 对旧 run 发送带 run 的 best-effort Drop；节点失联或 `ErrNodeNotFound` 时按旧 Import 语义解绑并继续重派，不等待强停写证明。旧 run 使用独立对象名和新的物理 segment ID，因此即使迟到写也不能覆盖新 run；孤儿对象由 retention/GC 最终回收。

V3 scheduler 在持有 task-key 锁、已选定 node 之后调用 `PrepareRun(nodeID)`。该方法读取并保存当前序列化的 job 和 task；catalog update 以 job/task 的准确旧值做 `ValueEqual` fence，并一次完成：递增 task 的 `run_id`、保存新 node、IMPORT 的全部预分配 SegmentInfo 和 task 的 `Running` marker。无旧 task 时使用 task key 不存在比较；当前 predicate API 只有 `ValueEqual`，所以 catalog 只做最小扩展支持 key-not-exists compare，不能用普通 Save 覆盖并发创建。任何 compare 失败都 reload，不发送 RPC。pending/running queue 不持有可被再次派发的旧 request；Create 超时后回到 Retry 必须重新 `PrepareRun`。DataNode 在实际执行最终过滤每个 batch 时直接取当前 `time.Now()` 作为临时值，不保存或传递 `filter_reference_time`；同一 run 的不同 batch 可以使用不同时刻。

IMPORT 直接沿用当前 `AllocImportSegment` 的预创建生命周期：Planning 在 DataNode 写正式路径前分配物理 segment ID 并保存普通 SegmentInfo，而不是等 result 才创建。每个 SegmentInfo 固定：

- `ID/CollectionID/PartitionID/InsertChannel/Level/StorageVersion/SchemaVersion`；
- `State=Importing`、`IsImporting=true`、`NumOfRows=0`、`LastExpireTime=MaxUint64`；不设置 Import V3 专用 `IsInvisible`，与 V2 一样由 `IsImporting` 负责 CommitImport 前的查询可见性；
- `ManifestPath` 和 binlog/stats 引用在预创建时都为空，等完整 result 返回后按 Storage V2/V3 的现有投影补齐。

这个 SegmentInfo 引用让当前 segment GC、index/handler 视图和正式路径所有权与现有 Import 一致。无需给 SegmentInfo 增加 run 字段：task 当前的 `segments` 给出准确归属。若同一 guarded update 因 txn op 上限走 marker-last fallback，SegmentInfo 可以先落盘，但 task `Running` marker 未落前不能发 RPC；恢复时要么补完同一 PrepareRun，要么把 task 切到 Retry 并按准确 output IDs 标准化标 Dropped。

Import run 的逻辑 segment plan 不变，但每次重试使用新的物理 segment ID 和正式路径。DataCoord 完成 best-effort Drop，或按当前语义得到 `ErrNodeNotFound` 并解绑后，再把落败 segment 标为 Dropped。这里不声称旧 worker 已停止，安全性来自新路径和结果拒绝。

### 12.6 DataNode TaskManager 语义

DataCoord scheduler 和 DataNode TaskManager 的 map key 都继续使用稳定 `task_id`，不改成以 `run_id` 为 key；task value 内保存当前 `run_id`。`PrepareRun` 在重新派发前递增这个字段：

| 请求 | 行为 |
| --- | --- |
| 首次 Create | 添加 task，启动一次执行 |
| 相同 run | 幂等成功，不启动第二个 goroutine |
| 较小 run Create/Update/result | stale，no-op 或明确返回 stale |
| 较大 run Create | 取消并替换旧 run，启动新 run；不把进程内 cancel 描述成跨节点停写证明 |
| 较小 run Drop | 幂等成功，不影响当前 run |
| 当前 run Drop | cancel/remove、释放已知本地资源；控制 RPC 与 V2 一样不承诺旧执行已经停止 |

所有 scheduler 完成回调捕获 run ID；旧 goroutine 的 Update 只能在 task 当前 `run_id` 仍相等时生效。Query/result 对较小 run 返回 stale，不能把它重新当成可提交结果。Create、Query、Drop 和 result acceptance 都在 DataCoord 的 task-key lock 下串行化，沿用 V2 的本地调度顺序；这只约束 DataCoord 本地并发，不等价于旧 DataNode worker 已停止。不增加 `Retired`、`Quiesced` 等状态。

## 13. OSS 与本地路径设计

### 13.1 统一根目录

遵循 Milvus 现有 `insert_log/{collection}/{partition}/{segment}`、`analyze_stats/{task}/{version}` 和 `metautil.JoinIDPath` 风格：

```text
R = {storage_root}/import_v3/{job_id}/
```

job ID 全局唯一，不再重复放 collection ID。禁止使用键值式目录名。

### 13.2 完整布局

```text
{root}/import_v3/{job_id}/
├── plans/
│   ├── reshard/{task_id}/plan.pb
│   └── import/{task_id}/plan.pb
├── reshard/{task_id}/
│   ├── fragments/{channel_index}/{partition_id}/{run_id}_{seq}.parquet
│   └── manifests/{run_id}.pb
└── import/{task_id}/
    └── merge/{run_id}_{segment_ordinal}_{round}_{group}_{seq}.parquet
```

约束：

- 所有 ID 是纯十进制字符串；
- vchannel 路径使用 job 有序列表中的 ordinal，manifest 同时保存原始名称；
- 不增加 run 目录；run ID 只出现在文件名，因此任务 ID 仍是主要目录层级，目录层级接近旧 Import；
- 同一 task 的不同 run 通过文件名中的 run ID 隔离，旧 writer 不能覆盖新 run 的对象；
- plan path 由 `job_id + task_id` 和固定目录直接推导，task catalog 不重复保存；Import result 只在 Completed Query RPC 中返回；
- 正式 segment 永远写现有 `insert_log`/stats/manifest 路径，不放在 `R` 下；
- 业务代码通过新增 `pkg/util/metautil/import_v3.go` 统一使用 `path.Join`、`strconv.FormatInt` 和 `metautil.JoinIDPath`，不散落字符串拼接。不得引入 `jobs/j=...`、`tasks/t=...`、`attempts/a=...`、`collections/c=...` 这类键值式路径。

所有层级只使用上述纯十进制 ID/ordinal 和固定的语义目录名，不增加键值式 ID 前缀。

### 13.3 LIST 的使用边界

- 正常 Barrier、Planning、Importing 和恢复只按 ID 推导的准确 path 读取，不用 LIST 选择“最新”对象。
- LIST/Walk 只用于 GC 和诊断。
- 不能用对象 mtime 决定胜出 run。

## 14. Barrier 和 Planning

### 14.1 Barrier

进入 Planning 前，DataCoord 对每个 ReshardTask 使用 task 当前 `run_id` 推导准确 manifest path：

1. manifest 存在且 protobuf 可解析；
2. fragment path 非空且不重复；
3. rows、logical bytes 非负；
4. fragment bucket ordinal 在 job vchannel/partition 数组范围内。

manifest 是当前 run 完成后才发布的输出，不再回显 job/task/run、SortSpec、source coverage 或 totals。这些值要么已由路径选中，要么可从 plan/descriptor 求和，重复保存只会引入两份真相。

Barrier 不为每个 fragment 做额外完整 GET。Importing packed reader 在真实消费时继续检查对象缺失、长度、解码、实际行数和单调性；本次不实现 checksum 字段或校验逻辑。

### 14.2 fragment 规范序

同一个 bucket 内：

```text
ORDER BY reshard_task_id ASC, seq ASC
```

路径、run ID、对象创建时间不参与排序。每个 FragmentRef 只从当前 task `run_id` 推导的 manifest 构造；ImportTaskV3 进入 Pending 后，后续恢复直接使用已经持久化的自包含 ImportTaskPlan。

### 14.3 顺序 segment 装箱

对每个 bucket 的规范序 fragment 使用顺序 Next-Fit：

```text
current = empty
for fragment in canonical_order:
    if current is empty:
        current.add(fragment)
    else if current.logical_bytes + fragment.logical_bytes <= segment_target:
        current.add(fragment)
    else:
        seal(current)
        current = [fragment]
seal(current)
```

规则：

- 不使用 BFD；
- 不切 fragment；
- 单 fragment 超过 segment target 时独占 plan；
- plan 输入是规范序连续区间；
- segment target 复用当前 `getExpectedSegmentSize`，包括磁盘索引 collection 的现有目标；
- `logical_bytes` 只影响装箱，最终 writer 的实际大小允许有偏差；
- 相同 manifest 集合、task ID 和 allocator 结果必须产生字节一致的 ImportTaskPlan。

Reshard 已在写 fragment 时统计整个 batch 的逻辑字节，Planning 直接使用该值。不再保存每列统计、推测 function output 大小或建立物理压缩估算模型。这是装箱的软估算；最终 segment 超过 target 不拆写、不重规划。

### 14.4 Importing task 粒度

Importing task 主要按 vchannel 切分：

- 默认一个 task 负责一个 vchannel 的全部 segment plan；
- 单 vchannel 数据过大时，只能按连续 partition ordinal 区间拆 task；
- 一个 task 内按 `(partition_index, segment_ordinal)` 顺序执行多个 plan；
- 一个 plan 不跨 task；
- 任意时刻只构建一个正式 segment。

这使 task 数主要约等于 `V`，而不是源文件数或 segment 数。

每个 ImportTaskV3 task 的 `slot` 在生成 task catalog/request 时固定写入，不能由 DataNode 重算。V3 不再沿用旧 Import 的“文件数并行 CPU 项”和 `baseBuffer × V × P` 公式；ReshardTask 首期单 reader、单 sort/write pipeline，slot 使用以下固定估算：

```text
reshard_working_set =
    3 * read_buffer
  + 2 * fragment_target
  + packed_writer_buffer

reshard_slot =
  max(1, ceil(reshard_working_set / memory_limit_per_slot))
```

`3 * read_buffer` 分别覆盖 source reader、normalized batch 和 routing 后 batch；`2 * fragment_target` 覆盖 fragment 数据本身及 `storage.Sort` 的 record/row-index 临时空间；`packed_writer_buffer` 使用现有 `packed.DefaultWriteBufferSize`。公式不乘 `V×P`，因为 bucket 尾部由 task-local budget 触发 spill。Importing task 使用第 15.5 节的 `fan_in` working-set 公式。两类 slot 都在 Planning/派发前持久化，DataNode 只校验为正数并使用字面值。多个 task 的 slot 在节点级相加，因此同一 DataNode 可并行多个 V3 task。

所有组成项直接使用已有配置或常量：`read_buffer=dataNode.import.readBufferSizeInMB`、`fragment_target=dataCoord.import.fragmentSizeInMB`、`packed_writer_buffer=packed.DefaultWriteBufferSize`、`memory_limit_per_slot=dataCoord.import.memoryLimitPerSlot`。除这些值外不增加调度配置。默认值下 working set 为 `3*16 + 2*128 + 32 = 336 MiB`，向上取整为 3 slots。

对比旧 `CalculateTaskSlot`，V3 slot 预计明显更小：旧 Import 取“文件数 / `fileNumPerSlot`”和 `16 MiB × V × P / 160 MiB` 的较大值，高 partition 场景会把单 task slot 放大到整个节点都无法同时容纳第二个 Import。V3 不再按文件数或 bucket 数收费，只为真正同时存在的 reader/batch/sort/writer 工作集收费。但不把 slot 固定为 1：当 fragment sort 或配置的 merge fan-in 确实需要超过一个 `memoryLimitPerSlot` 时，仍按上式向上取整。这使小/默认工作集可以大概率只占少量 slot、允许节点并行多个 task，同时不会在高 fan-in 时故意低报内存。

### 14.5 自包含 ImportTaskPlan

Planning 不持久化 job-wide snapshot，也不创建独立 plan index。每个 ImportTaskV3 只有一个内容完整的 `ImportTaskPlan`，其 path 由 job/task ID 推导。当前正式 writer 会从执行时 schema、首个 RecordBatch 和当前配置决定 storage version、column groups、writer format 与 TEXT 行为。ImportTaskV3 不能在每次派发时重算这些值，因此 Planning 把该 task 执行所需的非敏感信息写进同一个 plan：

```protobuf
message WriterSpec {
  int64 storage_version = 1;
  int64 schema_version = 2;
  repeated ColumnGroupSpec groups = 3;
  string format = 4;
  repeated TextColumnWriteSpec text = 5;
  V2PackedIOConfig v2 = 6;
  int64 ttl_nanos = 7;
  int64 pk_capacity = 8;
  repeated int64 bm25_fields = 9;
  string bloom_type = 10;
  double bloom_fpp = 11;
}

message ColumnGroupSpec {
  int64 id = 1;
  repeated int64 fields = 2; // 按 target schema 顺序
  string format = 3;            // 已解析的最终格式
}

message TextColumnWriteSpec {
  int64 field_id = 1;
  int64 inline_limit = 2;
  int64 lob_limit = 3;
  int64 flush_limit = 4;
}

message V2PackedIOConfig {
  int64 buffer_size = 1;
  int64 multipart_size = 2;
}

message SegmentPlan {
  string vchannel = 1;
  int64 partition_id = 2;
  repeated FragmentRef fragments = 3; // 规范序连续区间
  int64 rows = 4;
}

message ImportTaskPlan {
  SortSpec sort = 1;
  repeated SegmentPlan segments = 2;
  int32 fan_in = 3;
  WriterSpec writer = 4;
  schema.CollectionSchema schema = 5;
  schema.CollectionSchema temp_schema = 6;
  uint64 data_ts = 7;
  int64 collection_id = 8;
  string cluster_id = 9;
}
```

一个 plan 必须能脱离 ImportJob 和其他 Planning 对象独立执行。DataNode 由 job/task ID 推导 path，只 GET 一次 plan，再开始 merge。plan 不回显 job/task ID，因为对象已由路径唯一选中；DataNode 只验证真正的执行前提，例如 collection、schema、WriterSpec、segment 和 fan-in。每个 ImportTaskV3 先以 `None` 保存 task catalog 和 segment ID，再预创建 SegmentInfo，全部成功后改为 `Pending`。Planning 恢复只清理未准备完成的 `None` task 及其 SegmentInfo，按 vchannel 保留已有 Pending/Running/Completed task，并只创建缺少的 vchannel task。

字段语义固定如下：

- `storage_version` 在 Planning 一次解析。TEXT collection 继续遵守当前“至少 Storage V3”的安全网；不允许 DataNode 根据当前 `UseLoonFFI` 重算。
- `schema_version` 对应同一 ImportTaskPlan 中的完整 target schema。该 schema 来自广播时的 `ImportMsg/ImportJob.schema`，包含 parser timezone、`allow_insert_auto_id`、functions、namespace 和 file-resource IDs 等 properties；Reshard、Planning 和 Importing 都不能改读更新后的 collection schema。
- collection 当前 schema version 与 frozen version 不同本身不自动失败。DataCoord 在 Planning 发布前、每次 Importing result commit前，以及进入 /IndexBuilding前执行最小影响检查：
  1. 从 frozen WriterSpec/schema 和当前 collection schema分别构造 **Import-relevant projection**，只包含会影响正式 segment或索引输入的事实：字段 ID/type、nullable/default/dynamic、PK/partition key/namespace、TEXT物理规则与 analyzer/file resources、functions的 type/input/output/params，以及当前 target index引用的 field ID/type。
  2. 两个 projection 完全相等时，schema变化被视为无副作用，例如只改 description或不参与写入/函数/索引的 property。job继续按 frozen WriterSpec写数据；DataCoord 在提交 SegmentInfo时把 `SchemaVersion` 元数据提升为已验证等价的当前 version，避免当前 index inspector仅因“version落后且 collection有 functions”把本来等价的 segment挡住。数据格式和 WriterSpec 不改写。
  3. projection 不相等时，说明旧 schema写出的 segment与当前 collection在物理列、function output或 index输入上不同。ImportTaskV3 job按 non-retriable schema-changed SystemError进入 Failed/quiesce/回收，错误记录 frozen/current version和首个不同项。首期不依赖 Commit后的 schema-bump来掩盖差异，避免在补写前短暂暴露缺列数据，也避免重新引入另一条异步等待链。
  4. 若 job执行中 schema再次变化，每个边界都重新比较；已经安全提升的不可见 segment若随后遇到有影响的新版本，仍随失败 job统一 Dropped/回收。一个已发布 task plan 的 worker 输入始终是 frozen WriterSpec，不切换中途 schema。

实现只增加一个小的 Import-specific `CompareImportSchemaProjection(frozen,current,targetIndexes)` helper，返回 `equivalent/firstDifference`。不做通用全 schema diff框架，不比较纯展示元数据，也不依赖错误字符串。新增会影响物化或 index可建性的 schema能力时，必须同步扩展 projection和真值表测试。
- `groups` 为显式 `id/fields/format`。Planning 使用当前 `storagecommon.SplitColumns(..., DefaultPolicies()...)` 为整个 task 物化一份 WriterSpec；当前实现没有使用 per-segment 列统计，因此不增加 WriterSpec 表和 `writer_index`。DataNode 按 target schema 顺序从 `fields` 重建 column indices。
- `format` 和每组 `format` 使用当前 `FillColumnGroupFormats/ColumnGroupFormats` 的最终结果，不让 writer 回退读实时 `dataNode.storage.format`；传给 writer 的 schema-based formats 直接按规范 group 顺序从 `ColumnGroupSpec.format` 派生。
- `text` 显式保存 field ID、inline/max-LOB/flush threshold；LOB base path 从 storage root、collection、partition、segment 推导，不持久化。Importing 输入是 raw UTF-8 TEXT，固定使用非 rewrite 语义；不从当前 paramtable 重新生成阈值。当前 collection 创建约束已要求 TEXT 使用 Storage V3，因此 ImportTaskV3 不设计 V2 inline-only TEXT 分支。
- `v2` 只在 Storage V2 保存当前 packed writer 实际使用的 buffer/multipart 参数；V3 FFI writer不靠这两个值决定输出，不写伪字段。
- `ttl_nanos` 只从冻结 schema properties 解析，不记录 TTL 参考时间。ImportTaskV3 在实际最终过滤每个 batch 时直接读取 `time.Now()`，不写入 task、run、ImportTaskPlan 或 WriterSpec。过期行应在本次过滤时尽早剔除；已写入的过期行沿用现有查询 TTL 过滤和 compaction 物理回收，最终保证用户 API 不返回过期行，而不追求跨 run 的物理结果一致。
- `pk_capacity` 固定传给 `NewBinlogRecordWriter(maxRowNum)` 的值；它会影响 bloom filter 容量和内容，不能仅从执行时实际行数临时选择。每个 segment plan 可使用自己的 planned rows/capacity。
- `bloom_type` 和 `bloom_fpp` 也必须在 Planning 时从当前配置解析并写入 spec。它们不能在 DataNode 重试时再次读取 refreshable paramtable；否则同一个已发布 task plan 的重试可能生成不同的 PK stats。正式 writer/stat writer 必须增加显式 option，把这两个值传入 `NewPrimaryKeyStats`、`GenerateByData` 及其调用链；缺失或非法值按内部协议错误处理，不回退到 worker 当前配置。
- `bm25_fields` 按 field ID 升序固定，同时作为 log ID 预算输入。

DataNode 创建正式 writer 时必须显式传入 WriterSpec 中的 column groups、writer format、TEXT config、storage version、V2 I/O 参数、PK stats capacity、Bloom filter type 和 Bloom 误判率；不允许留空以触发当前 writer 的“首 batch 动态分组/读取 paramtable”分支。raw TEXT 由 Storage V3 writer直接生成 inline/LOB。TEXT 索引继续由现有后台 inspector 在 segment 发布后创建；本次 V3 不在 plan、worker 或 result 中预留 inline text-index 协议。当前源码的普通 V3 writer 已有 plugin context→CMEK writer properties 的构造逻辑，但 TEXT 路径的 `rw.go → NewPackedTextManifestRecordWriter → NewPackedTextBatchWriter → NewFFISegmentWriter` 尚未传递该 context。ImportTaskV3 实施必须沿整条构造链补上 `StoragePluginContext` 参数，并复用 `packed_writer_ffi.go` 中的加密属性生成逻辑，确保 TEXT inline 文件和 LOB 文件与普通 V3 文件使用相同的 CMEK 配置；不能只在最外层保存 context。加密 TEXT、fragment、intermediate run 和正式 segment 都要做恢复读写测试。当前 `appendV3Stats → packed.WriteFile` 不接收 plugin context，本方案保持这项现有边界，不顺带把 PK/BM25 stats 扩展为 CMEK 写入；只要求这些 stats 按当前方式可恢复读取。DataNode 直接使用 plan 内的 WriterSpec，并校验 schema version、column group、字段和 writer 参数等结构约束。

完整 WriterSpec、target/temporary schema、SortSpec、data timestamp、collection/cluster identity、segment plans 和 fan-in 只存在这一份自包含 plan 中，不复制到 job-wide snapshot、独立 index 或 Create request。`slot`、segment IDs 和 log ID range 属于调度 run，保存在 task catalog/Create request，不进入不可变 plan。Planning 恢复保留已进入 Pending 的 frozen plan；已有 plan 与本轮 Planning 只要求 vchannel 和 fragment coverage 一致。

## 15. Importing：严格单头 k-way merge 和分层归并

### 15.1 当前源码依赖

当前源码基线 `c04e16fbc0967bcabe0da5426d22f8691dae80f8` 已包含提交 `fb960cf7c2 enhance: rewrite storage.MergeSort as a proper k-way merge (#51998)`。因此不需要再移植或重写 MergeSort；ImportTaskV3 直接复用当前实现及其契约：

- 每 reader 一个 head，heap 大小 `O(K)`；
- typed value heap，避免每行 allocation；
- RecordBatch 切换时缓存 Int64/VarChar key 列；
- predicate 每行恰好调用一次；
- 空 reader、空 batch、全部过滤 batch；
- 遵守 `RecordReader.Next()` borrowed record 生命周期；
- Int64、VarChar、多字段键；
- 输出单调性检查，无序输入返回 `merr.ErrDataIntegrity` 并带 reader/record/row 坐标。

实施工作只是把 V3 fragment/intermediate reader 接入这个现有 MergeSort，并补齐 Import V3 特有的 context cancel、predicate、temporary TEXT schema、reader 行数/单调性检查和资源关闭测试；不复制第二份归并实现。

### 15.2 direct fan-in

配置：

```text
dataCoord.import.fragmentMergeFanIn = 16
```

- 缺省 16；
- 只做基础 sanity 校验 `[2,1024]`，配置越界时 DataCoord 启动校验失败，不静默 clamp；fan-in=1 会让 hierarchical merge 不收敛。这个范围不是资源承诺，真实可运行值仍由 task working-set slot 估算、reader 内存和 FD 约束；
- 不使用 `dataNode.compaction.maxSegmentMergeSort=30`；
- Planning 保存实际 cap，恢复不因配置变化重算。

### 15.3 分层归并

对一个 segment plan 的有序输入 `I0`：

```text
inputs = I0
round = 0

while len(inputs) > plan.fan_in:
    groups = split_contiguously(inputs, max=plan.fan_in)
    next = []
    for group_index, group in groups:
        if len(group) == 1:
            next.append(group[0])
        else:
            run = k_way_merge(group)
            next.append(run)
    inputs = next
    round++

final_merge(inputs)
```

`split_contiguously` 只产生大小为 `[1, fan_in]` 的连续组。singleton group 不创建新 run，直接把原 `FragmentRef/RunRef` 转发到下一轮；这既避免无意义 I/O，也要求清理器按 ref 计算存活集合，不能按“上一轮目录”整段删除。

示例：

```text
17 路：
  round 0: [0..15] → run 0；[16] 原样沿用
  final: run 0 + input 16

256 路：
  round 0: 16 组 → 16 runs
  final: 16 runs
```

上述伪代码中的 16 是缺省示例；真实实现全部使用 DataNode 已校验的 `plan.fan_in`。每轮：

- 实际 fan-in `<= plan.fan_in`；
- 只按当前输入顺序切连续组，不按 key range、大小或运行时完成顺序重新排列；
- intermediate run 使用 temporary schema、相同 SortSpec、单 column group packed Parquet；
- `output_rows = sum(input_rows)`；
- 检查输出单调性；
- 不执行 TTL/delete、timestamp、函数、BM25、stats、TEXT LOB；
- 不去重。

每个 intermediate run 执行 `Close → 取得 path/rows → 交给下一轮`，不做额外完整 GET、对象内容 SHA-256 或独立 run manifest。下一轮只读已成功 close 的输出。

一轮的全部非 singleton 输出 ref 都成功发布后，立即计算“旧输入 refs - 下一轮仍引用的 refs”，精确删除不再被引用的上一轮 run；singleton passthrough 的 ref 仍在下一轮存活集合中，不能误删。这样远端中间空间峰值是相邻两轮，而不是所有历史轮次之和。活跃进程内按内存中的准确 run-ref set 清理；不通过 LIST 推断哪些 run 属于当前轮。节点丢失或进程崩溃后不恢复内部 round，也不假设 catalog/plan 持久了 run refs，由 job 前缀 GC 在 task worker 停止后兜底。删除失败同样留给该 GC 重试，不影响已经发布的下一轮引用。

intermediate run 只属于当前 Import run。节点丢失后新 run 从原始 Reshard fragments 重做，不恢复内部 round。

### 15.4 fragment reader

复用 `packed.NewFFIPackedReaderWithFragments` 的准确 fragment 能力，在 storage 层增加公开 helper：

```text
NewImportFragmentRecordReader(FragmentRef, temporarySchema, ...)
```

reader 打开时校验：

- format/version；
- path 和完整对象行数；
- schema field ID、Arrow type；
- manifest rows 与实际读取行数；
- 每个 reader 自身按 SortSpec 单调。

不先下载到本地 verified cache。packed reader 直接从受管存储读取；错误沿现有 storage/merr 语义上报。

### 15.5 内存和 FD 上界

这个变化只删除“执行时再向全局 allocator 申请/等待”的第二层控制，不删除内存估算。严格区分：

- 常量上界是相对 `V×P`、文件数和 task 内 segment 数而言；
- 单 task 峰值仍与实际 `fan_in` 和固定 reader/writer/function buffer 成正比；
- 多 task 并发峰值是各 task slot 之和，由节点 `available_slots` admission 控制；
- 若 slot 估算偏低，多个 task 会一起放大误差，因此测试和监控必须验证估算，而不能把“没有 allocator”理解成“内存不需要管理”。

Importing 单 task 峰值：

```text
fan_in × (packed reader buffer + 当前 Arrow batch + native reader state)
+ O(fan_in) heap/key cache
+ output RecordBuilder
+ final function batch
+ 一个 intermediate 或正式 writer
```

intermediate 和 final 串行执行，预算取两者最大值，不相加。FD 上界约为 `fan_in` 个 reader + 1 个 writer + 固定运行库开销；默认配置下即约 16 + 1。

不增加新的多维 QuerySlot 协议。V3 继续复用当前 task slot 和 `dataCoord.import.memoryLimitPerSlot`，但不复用旧 Import 的全局 `MemoryAllocator`；V2 仍保留自己的 allocator。V3 task 的 peak estimate 由 slot admission 直接控制，V3 executor 只负责运行已经通过 DataCoord slot admission 的 task，不再承担跨 task 内存排队。`dataNode.import.memoryLimitPercentage` 继续作为 DataNode Import 总内存监控/准入参考，不作为 V3 task 间的二次阻塞队列；packed reader/writer 已有 buffer 配置仍计入 peak estimate。DataNode `QuerySlot` 的 `importUsed` 必须汇总 V2 和 V3 task 的 slot，不能让两条执行路径各自认为节点空闲。

Import V3 不使用 global scheduler 的 worker version constraint，也不追加 Import 专用 capability 字段。DataNode 兼容性由发布功能开关保证：开关只允许在所有 DataNode 都完成 V3 二进制升级后开启。`QuerySlot` 在运行时只继续提供 slot，不为 Import V3 增加新的资源向量或版本 reservation 协议。

V3 Importing slot 不用 segment 逻辑大小代替峰值内存，而是按实际同时打开的 reader 数估算：

```text
import_working_set =
    fan_in * 2 * read_buffer
  + 3 * read_buffer
  + packed_writer_buffer

import_slot =
  max(1, ceil(import_working_set / memory_limit_per_slot))
```

`fan_in * 2 * read_buffer` 覆盖每个 packed reader 的读取 buffer 和当前 Arrow batch；固定的 `3 * read_buffer` 覆盖 merge output、final transform 和 function batch；writer 使用现有 packed writer buffer。heap、key cache 和 native reader 固定状态由这些 buffer 预算共同吸收，不另造不可配置的经验常量。hierarchical round 和 final round 串行，不叠加两轮 working set。默认 `fan_in=16` 时 working set 为 `16*2*16 + 3*16 + 32 = 592 MiB`，向上取整为 4 slots。

Planning 保存实际 fan-in 字面值；如果配置并选择 fan-in=1024，就按 1024 个同时打开的 reader 估算。该 task 可能长期 Pending，直到出现足够 slot 的节点。功能开关开启后，所有可调度 DataNode 都按发布前提视为支持 Import V3；scheduler 只按 slot 选择节点，不再按版本过滤，也不为 V3 单独实现 exact admission——节点剩余 slot 不足时沿用与旧 Import 相同的 best-effort fallback。V3 的内存安全由 task 内局部 spill（§10）、slot 估算和现有节点监控共同约束。

**OOM 风险边界**（best-effort 调度下，slot 估算只是近似峰值上限，超出部分先由 §10 的 task 内 spill 吸收，spill 也兜不住时由外部监控发现，Import 内不再加第二套硬防线）：

- 单行内存远大于估算假设：按 §10.2 单行超 task-local working-set 直接资源失败，不临时超额；
- 多个 task 的 `ceil(peak/memoryLimitPerSlot)` 取整偏差在同一节点叠加；
- packed reader、Arrow 或函数实现的实际驻留内存超过上述 buffer 估算；
- `InsertData → Arrow Record` 转换与 packed writer 的临时 buffer 未在 slot 公式中逐一精确预留。

## 16. Final merge：过滤、函数和正式 writer

### 16.1 普通 Import 最终顺序

```text
有序 fragment readers
→ k-way MergeSort predicate
    → TTL/TTL field
→ 已排序且保留的 RecordBatch
→ final transform
    → 物化 timestamp = ImportJob.DataTs
    → Record → InsertData
    → embedding.RunAll
       TextEmbedding → BM25 → MinHash
    → InsertData → Record
→ 正式 TEXT-aware 或普通 writer
→ bloom/BM25 stats、manifest/binlog
→ SegmentResult
```

普通 Import V3 首期不在 final merge 处理 collection delete。它不读取 collection delete snapshot，不增加 deltalog ref、delete watermark 或 delete read timestamp，也不把当前 collection delete map 带进 final predicate。普通 Import 只写入 Import 文件中的普通数据；之后发生或已经进入当前 DML 链路的 delete 继续写入 deltalog，由 Query 按当前可见性规则隐藏，并由后续 sort/mix compaction 使用现有 deltalog 组合和 `EntityFilter` 物理清理。这与当前普通 Import reader/ImportTaskV2 本身不读取 collection delete 的边界一致；区别是旧链路紧接着有一次 Sorting，可能顺带消费已经附着的 deltalog，而 V3 跳过该阶段，所以这里只保证与当前 Query + 后续 compaction 的最终结果一致，不声称 V3 final merge 已经处理 collection delete。

普通 fragment 不含 timestamp，predicate 使用固定 `ImportJob.DataTs` 作为 row timestamp，无需先构造整列。正式 writer 前再物化该列。

TTL/TTL field 与 collection delete 分开处理。V3 在 MergeSort predicate 首次处理一个新的 RecordBatch 时用当时的 `time.Now()` 创建本 batch 临时 TTL filter，批次结束即释放；可复用 `compaction.EntityFilter` 的 TTL 判断时必须传空 delete map，也可以抽取现有 TTL helper，但不能复制一套不同算法。这不是逐行读时钟，不会在大数据量下产生额外时间开销；也不保存任何可恢复的 reference time。已经进入系统的 delete 仍由 Query/后续 compaction 处理，本次裁决不改变已确认的 TTL 行为。

### 16.2 backup 最终顺序

backup fragment 含源 timestamp，并已经过 source backup time range/deltalog 过滤：

```text
有序 fragment readers
→ merge（不重复 source filter）
→ 当前 backup schema/default 语义
→ 不重复运行已有 function output
→ 正式 writer
```

若目标 collection 的现有规则要求对缺失 function output 报错或补齐，继续使用当前 `binlog.verify`/Import 行为，不在新链路创造不同语义。

这里的 backup deltalog 是源 backup/binlog 文件的一部分，只用于恢复 source time range 内本来应保留的行；它与普通 Import 的 collection DML delete 是两件事。Reshard 已按当前 backup reader 语义过滤一次，Importing 不重复执行，也不把普通 collection delete 混入该步骤。

### 16.3 正式 writer 复用

最终输出必须复用当前 Import/compaction 正式路径：

- `storage.NewBinlogRecordWriter`；
- Storage V2 packed writer；
- Storage V3 manifest writer；
- TEXT-aware segment writer 和 LOB 路径；
- bloom、BM25、expiration quantile、Statistics；
- `ImportSegmentInfo`/manifest path 组装。

不要另写一套 segment 文件格式。需要抽取的是一个“接受已排序 Record stream，并执行 final transform 后写单 segment”的 adapter。

### 16.4 sorted 标志

- 非 namespace：`sorted=true`，`namespace_sorted=false`；
- namespace：`sorted=false`，`namespace_sorted=true`。

ImportTaskV3 的 Importing 完成后直接进入 IndexBuilding；历史 ImportTaskV2 job 继续进入现有 Sorting。

即使一个 segment plan 最终只有一个 fragment/run，也不能把该对象直接改名或作为正式 segment 引用。它仍经过 strict streaming merge/adaptor：单 reader head、predicate、输入单调性检查、final transform 和正式 writer 全部执行。这样 TTL/delete、timestamp、函数、TEXT/LOB、stats、Storage V2/V3 manifest 和正式路径语义不会因为输入数为 1 而被跳过。

### 16.5 一个 task 多个 segment

```text
for segment_plan in task.segments:
    打开该 plan 输入
    完成所有 intermediate rounds
    完成 final merge + writer
    close/release 全部 reader、builder、writer
```

task result 包含多个 `SegmentResult`。只有全部 segment plan 成功才由 Completed Query 返回完整结果；任一失败，task 只返回失败状态和原因，同时保留本次待回收的 `segments`。DataCoord 完成 best-effort Drop，或 Drop 按当前旧 Import 语义返回 `ErrNodeNotFound` 并解绑后，再把这些预创建 SegmentInfo/正式输出用标准 Dropped 更新回收。重试为所有 plan 分配新的物理 segment ID 集合。

## 17. Import result 提交和可见性

### 17.1 QueryImportTaskV3Response

```protobuf
message SegmentResult {
  int64 rows = 1;
  repeated data.FieldBinlog insert_logs = 2; // Storage V2
  data.FieldBinlog pk_log = 3;               // Storage V2
  repeated data.FieldBinlog bm25_logs = 4;  // Storage V2
  string manifest_path = 5;                  // Storage V3
  data.Statistics statistics = 6;
  repeated int64 expiration_quantiles = 7;
}

message QueryImportTaskV3Response {
  common.Status status = 1;
  ImportTaskStateV2 state = 2;
  string reason = 3;
  repeated SegmentResult segments = 4;
}
```

DataNode 的进程内 TaskManager 保存当前成功 run 的完整 `SegmentResult` 数组；只有 Completed Query 携带结果，Running/Retry/Failed 不携带。结果按 task 已持久化的 segment ID slice 对齐，不回显 ordinal、segment ID、bucket、storage/schema version、sorted flags 或 totals。DataCoord 从 task、job schema 和数组下标取得这些已知信息。`Statistics.TimestampFrom/TimestampTo` 是时间范围的唯一来源，不再平行保存 `min_ts/max_ts`。

V3 继续保持“一条 vchannel 一个 ImportTaskV3”。当前默认 RPC send/receive 上限为 512 MiB。Storage V3 每个结果只含 manifest path、固定 Statistics、rows 和最多 5 个 expiration quantile；即使按当前单请求理论上限约 16 TiB、约 17K 个 segment，并保守按 1 KiB/segment 估算，完整结果也约 17 MiB。Storage V2 极宽 schema 会更大，但首期明确接受这个极端风险，不为它增加对象存储协议或 task 拆分。

DataNode 在组装 Completed response 时计算内层 `QueryImportTaskV3Response` 的 protobuf 大小；达到本地 `dataNode.grpc.serverMaxSendSize` 与 `dataNode.grpc.clientMaxRecvSize` 较小值的 50% 时记录 warning，包含 task/run、segment 数、结果字节数和限制值。warning 提前到 50% 是因为通用 `workerpb.QueryTaskResponse` 还会把内层 bytes 再包装一次，TaskManager 对象图、内层 marshal bytes 和外层 gRPC marshal buffer 也可能同时存在。该检查只用于诊断，不拒绝任务、不改变调度，也不增加 fallback。

直接 RPC 的运行边界固定如下：

- gRPC 最终对外层 response 执行 send/receive size 检查；本地 warning 不是接受证明。若外层消息超过任一端真实限制，gRPC 返回 `ResourceExhausted`。当前 V3 把这个 transport error 当作不可重试错误并让 job Failed，不切换 `ImportResultManifest`、不拆 task、不自动调大配置。
- DataCoord 的 Query 使用 `dataCoord.requestTimeoutSeconds`，默认 600 秒。结果序列化或网络传输超过该时间时，当前 transport error 同样会终止 job；首期不因超时重新执行已经完成的超大 task。正常 Storage V3 结果远小于该边界。
- Completed result 在 DataNode TaskManager 中只读保存，直到 DataCoord 接受结果并发送 Drop；Query marshal 不得修改它。DataCoord 暂时不可用时，节点只会保留当时已派发的有限 task result，不继续接收来自该 DataCoord 的新 task。DataNode 重启会丢失这些进程内结果，DataCoord 按既定 node-lost 语义用新物理 segment ID 重做 task。
- DataCoord 接受结果后立即沿用 global scheduler 的 best-effort Drop。Drop 成功或得到 `ErrNodeNotFound` 时释放 TaskManager result；若接受完成后恰好只有 Drop RPC 发生暂时错误，现有 scheduler 不提供独立 Drop 重试队列，DataNode 可能把该进程内结果保留到节点重启，task 的 node 解绑/终态 GC 也可能延后。这是 V2 已有的 best-effort Drop 边界，本次不为极端大结果新增 cleanup scheduler；运维上可由 result-size warning、DataNode 内存和终态 job retention 联合发现。
- result 可能包含 Storage V2 binlog path，只通过集群内部 worker RPC 传输，不写日志；大小 warning 只记录计数和字节数。内部 TLS、认证和现有 V2 Query 的安全边界保持不变。
- `segments` 是 Completed response 的必需字段。旧 DataNode 不返回它，新 DataNode 也不再写 result object，因此旧/新 Import V3 二进制不能混跑同一个 active V3 job。发布系统必须先把所有 DataCoord/DataNode 升级到同一协议，再开启 V3 gate；本次是尚未发布协议的直接替换，不增加兼容 fallback 或新的 job version。

`ImportResultManifest` 只保留为未来增强设想，当前不定义 proto、不写对象、不保存 path、不增加 GC。这个增强**不推荐预先实现**。只有线上证据表明单 task 结果持续逼近 RPC 上限，例如接近 16 TiB 的单 vchannel Storage V2 import 同时使用极宽 schema、产生约 16K segment，或未来把单请求上限提高到数百 TiB/PiB，才应重新评估把结果写入确定性对象路径并由 DataCoord GET。在达到这种极端条件前，直接 Query 返回结果更简单，也与 V2 一致。

`SegmentResult` 数量必须与 task 的 segment ID 数量相同。非零行结果必须有合法 `Statistics`；Storage V2 返回 binlog/stats 引用，Storage V3 返回最终 manifest path。

final predicate 可能把非空 segment plan 全部过滤为零行。此时仍在对应下标返回 `SegmentResult{rows=0}`，不得带 `Statistics`、insert/stats/BM25 logs、V3 manifest 或 expiration quantiles，也不得把预创建 SegmentInfo 提升为 `Flushed`。task marker 提交后，DataCoord 在 Drop/解绑阶段把零行 SegmentInfo 标为 Dropped。

`DataType_Text` collection 的 target storage version 固定为 V3；当前源码没有正式的 V2 TEXT/LOB writer，本方案不新增 V2 inline-only TEXT 分支。V3 final writer 只生成 TEXT inline/LOB 数据；text index 沿用 segment 发布后的现有后台 inspector 流程，不修改 Query result 或 SegmentInfo 的 text-index 元数据。Storage V2 只覆盖不含 `DataType_Text` 的 schema；若普通 VarChar field 另有 match 能力，沿用其当前支持边界，不把它与 TEXT/LOB 混为一谈。

### 17.2 完整结果拉取与 marker-last 接受

当前 ImportTaskV2 的控制顺序是：DataNode Query 返回 Completed 完整结果，DataCoord 更新 SegmentInfo，最后把 task 标 Completed。V3 保持这个顺序，不引入独立的 result acceptance 状态：

1. DataNode 在 Running 时只返回进度；全部 segment writer 成功 close 后，Query 才返回 `Completed + segments`。DataCoord 在同一个 DataCoord import-meta task-key lock 下校验：
   - job 仍存在且不是 `Failed/Completed`；
   - job 仍处于 `Importing`；
   - task `run_id` 等于 result run；
   - task 仍为 `Running`，job/task/node 和具体 task 类型都一致，Query Completed 对应 task 的当前 run；
   - result 的 segment 数量与 task 已分配 segment ID 数量一致；
   - 每个非零结果有合法 Statistics 和对应 storage writer 输出；零行结果不带任何输出。
2. DataCoord 不持久化 result 或 result path。task/job 已变化时把这次 Completed 视为 stale/no-op；同一 run 重复 Query 时幂等处理。
3. result 不创建新 SegmentInfo，只补齐 task 已预创建的 SegmentInfo。逐下标使用 task 中对应的 segment ID；非零 segment 补 rows、binlogs/statslogs/bm25logs 或正式 manifest、position、Statistics、expiration quantiles 和从 job schema 推导的 sorted flags，然后提升为 `Flushed`，仍保持 `IsImporting=true`；零行 SegmentInfo 保持 `Importing`，等待后续 Dropped/GC。
4. SegmentInfo 可以按 catalog txn 上限分批补齐。相同 SegmentInfo 若已部分/全部补齐，所有已有非零字段必须与 result 一致；完全一致才幂等，不一致返回 `ErrDataIntegrity`，不覆盖。非零 segment 成为 `Flushed` 后，与 V2 一样可以被 index/stats inspector 消费，不增加 V3 专用 `IsInvisible` gate。落败、零行或失败 segment 在 task 重试/终态清理时标 Dropped，提前创建的 index/stats meta 和对象沿用现有 segment ID 清理路径。
5. 所有 SegmentInfo 都完成后，重新读取 job/task；只有 job 仍为 `Importing`、task 仍是同一 `Running + run_id` 时，才把 task 写为 `Completed`。
6. timeout/Abort、较大 run 或并发恢复只要先改变任一旧值，最终 CAS 就失败。前置补齐的 SegmentInfo 仍保持 `IsImporting=true`，因此不能被查询；但已成为 `Flushed` 的 segment 可能被 inspector 提前消费，这是与 V2 对齐后明确接受的行为。若 DataCoord 在 SegmentInfo 已更新、task Completed marker 尚未保存时崩溃，恢复先 Query 原节点获取同一个 Completed result；若节点仍可用，重新读取同一完整 result 并幂等重放。若节点已丢失或 task-not-found，则把 task 回到 Pending/Retry、清理本次 Importing 进度并从 immutable fragment 重做整个 task，不依赖一个已持久化的中间 result state。
7. marker 成功后再更新内存 meta。checker 只根据 ImportTaskV3 的 `Completed` marker 判断全部 task 是否完成，完成后把 ImportJob 推进到 `IndexBuilding`，不再执行额外 segment publish。

只有全部 op 数不超过 `TxnKV.MaxTxnOps()` 时，SegmentInfo 补齐和最终 marker 才可放在同一原子事务。超限时复用现有 caller-ordered chunked fallback：SegmentInfo 先分批落盘，task Completed 最后落地；这是 marker-last 可恢复协议，不是原子事务。当前 `txn.Commit` 的 atomic/fallback 路径都不接收 predicates，实现需做最小扩展：让 `txn.Commit` 可选接收只应用在“原子全量 txn”或“fallback 最终 commit txn”的 KV predicates；前置分批子项不带虚假的全局 compare。最终 predicate 失败时不直接删除 SegmentInfo key，而是恢复补完，或在终态清理中使用 `UpdateStatusOperator(..., Dropped)` 等价的 composite 操作写入 `DroppedAt` 后交给现有 segment GC。

恢复时，只有 task `Completed` marker 允许 checker 把该 task 计入 Importing 完成条件。正常恢复对 Running task Query 原节点；拿到相同 Completed result 就幂等补写，节点/session 丢失或 task-not-found 则回到 Pending/Retry，从 immutable fragment 重做。若 job 已被 timeout/Abort 率先提交为终态，迟到 result 是 stale/no-op，对应预创建 SegmentInfo 按标准 Dropped/GC 回收。不需要给 `SegmentInfo` 增加 import-run 字段。

### 17.3 可见性

正式 segment 从预创建到 IndexBuilding、Uncommitted 都保持 `is_importing=true`，由 CommitImport 清除。沿用当前：

- auto_commit=true：checker 在 Uncommitted 广播 CommitImport；
- auto_commit=false：等待平台显式 CommitImport；
- `HandleCommitVchannel` 只从每个 Completed ImportTaskV3 的 task segment ID slice 枚举该 vchannel 的非零 segment；严禁把 loser/零行 SegmentInfo 加入集合；
- 每个 vchannel flusher 先 Flush 到 CommitImport WAL message 的 TimeTick，再调用 `HandleCommitVchannel`；失败持续重试，成功后才 observe 该 WAL message。这个组件和动作与当前 Import 完全对齐；
- `HandleCommitVchannel` 对这些 segment 先校验 `commit_timestamp >= SegmentInfo.Stats.TimestampTo`，成功后再幂等设置 `commit_timestamp` 并清 `is_importing`；
- job 在全部 vchannel committed 后才 Completed。

这里必须修复当前源码已有的 V3 fence 缺口：现有 `UpdateCommitTimestamp` 只读取 `SegmentInfo.Binlogs[].TimestampTo`，而 manifest-backed segment 在 catalog 中不持久化这些数组，DataCoord 重启后该上界会错误退化成 0。ImportTaskV3 禁止依赖 `maxBinlogTimestampTo` 作为唯一来源；它直接读取 result acceptance 已持久化的 `SegmentInfo.Stats.TimestampTo`，并要求：

```text
commit_timestamp
>= persisted SegmentInfo.Stats.TimestampTo
```

为兼容旧 Storage V2 segment，公共 helper 可以取 `max(Stats.TimestampTo, max(Binlogs.TimestampTo))`；但 V3 的协议真相仍是上述 accepted result 与 Stats 投影等式。低于上界时返回当前 `ErrImportSysFailed` 并让 flusher 保持重试/阻塞，不能清 `is_importing`、不能写 committed-vchannel marker。

当前 `UpdateSegmentsInfo → catalog.AlterSegments → SaveByBatch` 在 segment 数超过事务上限时会分批保存，因此这里不声称一个 vchannel 的全部 SegmentInfo 在存储层原子修改。协议采用 marker-last：先幂等更新该 vchannel final segment，全部成功后再把 vchannel 追加到 `ImportJob.committed_vchannels`，它是该 fence 的完成 marker。若 segment 分批更新中崩溃，或 segment 已完成 commit 但 job marker 保存失败，因为 vchannel 尚未记录，callback 会在重试中重新执行全部 segment 更新，再补 marker。本文只声称 per-vchannel 可恢复、幂等和 marker-last，不声称 visibility 与 job marker 原子，也不声称 job-wide 同时可见。与 V2 一样，`is_importing=true` 只保护 CommitImport 前的查询可见性，不阻止后台 index/stats 提前处理已经 `Flushed` 的 segment。

`Committing` 是 point-of-no-return：第一个 vchannel 开始 commit 后，timeout 和 Abort 都不能把 job 降为 Failed；已在 `committed_vchannels` 中的 vchannel 及其 segment 不得进入 Failed cleanup。其他 vchannel 继续依赖 durable broadcaster/WAL replay 和 `HandleCommitVchannel` 重试，直到全部 marker 到齐后 job Completed。本次不新增 `PartialCommitted` 状态，也不提供 job-wide 同时可见：Committing 期间允许短暂的部分 vchannel 可见。“不支持部分成功”仅指执行阶段不能接受部分 task/segment 作为成功结果，不表示 per-vchannel commit 在任意瞬间都 job-wide 原子。

## 18. 失败、重试、取消和恢复

### 18.1 错误分类原则

实现必须使用 `merr`：

- `ErrImportFailed`（code **2100**，InputError，`retriable=false`）：用户文件格式/schema/字段内容错误，以及**广播前**发现单文件请求的预分配上界超过 `MaxUint32`。这些错误显示为用户输入错误；当前 run 立即终止，不重试同一内容，也不创建新 run。
- `ErrDataIntegrity`（code **1009**，SystemError，`retriable=false`）：组件间持久化协议不一致，包括 fragment/intermediate 宣称有序但实际下降、WriterSpec 与 plan/schema 不一致、result segment 数量与 task segment 数量不一致或零行结果带输出。
- `ErrImportSysFailed`（code **2101**，SystemError，`retriable=false`）：内部编排或配置协议错误，包括 invalid `fan_in`、无法按确定性路径读取 task plan、未知 V3 状态/字段组合、缺失必须的 schema 或 writer spec。
- **schema projection mismatch** 也使用 `ErrImportSysFailed(2101)`，不可重试：它是 frozen Import schema 与运行中 collection schema 的物理/函数/index 输入不兼容，是合法 schema alter 与 Import 之间的 TOCTOU/内部协议冲突，不使用 InputError 的 `ErrCollectionSchemaMismatch(109)`，也不进入旧 schema-bump 等待环。
- DataNode 在运行时发现实际 `row_offset` 已耗尽已持久化的 `pre_allocated_auto_ids` 时，使用 `ErrDataIntegrity(1009)`，而不是 `ErrImportFailed`：range 是 DataCoord 已生成并写进 WAL/plan 的内部事实，耗尽说明 source coverage 或上界估算契约被破坏。该当前 run 直接 Failed，不申请新号段、不换 run 重做同一错误输入。
- 暂时对象存储/节点故障必须保留底层可重试 merr code：`ErrServiceNotReady`(1)、`ErrServiceUnavailable`(2)、`ErrIoUnexpectEOF`(1002)、`ErrIoTooManyRequests`(1003) 等。只在 storage client 的当前 operation retry 用尽后返回；DataCoord 才创建新 run。不要用 `merr.WrapErrImportSysFailedErr` 把这些 code 改成 2101。
- 资源暂不可用沿用现有 `ErrServiceResourceInsufficient`（12，retriable）或节点 slot Pending 语义；本地 ENOSPC 只允许在任务失败后换节点/新 run，不能在同一节点无界重试。若 job timeout 已到，统一终止为 Failed。
- `stale result` 不是失败：run ID 小于当前 run、task 已被取消/终态，或相同 run 已提交时，返回幂等 no-op，不生成 merr、不重试 scheduler、不改变 job 状态。
- DataNode `task-not-found` 是节点重启/任务所有权丢失的特殊控制信号，与 Query RPC 失败走同一 Pending/Retry 重派路径。当前底层可能包装为 `ErrImportSysFailed(2101)`，V3 必须在 direct Query error 的 terminal 分类前按 Query 语境识别，不能把它当作 worker `Failed` 并终止 job；同一个 2101 出现在 task plan/fan-in 等内部协议边界时仍不可重试。
- 增加上下文使用 `merr.Wrap/Wrapf`，不掩盖底层错误码；`WrapErrXxxErr` 只用于明确改变分类的边界。

异步 V3 worker Query 的 Running/Retry/Failed result 只返回 `state + reason`，Completed 额外返回完整 `segments`；外层 `common.Status` 继续只表示 Query RPC 本身是否成功。DataNode 是 worker执行结果 Retry/Failed 分类的唯一边界，DataCoord 不做第二次 code 分类；收到 `Retry` 就创建新 run，收到 `Failed` 就终止 job。task/job 只持久化 state/reason，重启恢复只看 state。当前 `GetImportProgress` 与旧 Import 对齐：RPC 本身成功，失败通过 `state=Failed + reason` 返回，因此不新增用户 API 字段；广播前同步拒绝仍通过 `common.Status` 暴露 2100 的 InputError 标记。

首期必须实现以下最小错误契约，不能只靠文字 reason：

| 场景 | merr code / 类型 | 当前 run | 新 run | job/scheduler 动作 |
| --- | --- | --- | --- | --- |
| result segment 数量与 task 不一致，或零行结果带输出 | `ErrDataIntegrity` / 1009 / System / non-retriable | 立即终止 | 禁止 | job/task 持久化 Failed + reason，进入 quiesce |
| stale result | 无错误 | no-op | 不创建 | 不改 job，不重排，不计失败指标 |
| schema projection mismatch | `ErrImportSysFailed` / 2101 / System / non-retriable | 立即终止 | 禁止 | job Failed + quiesce，记录首个 projection difference |
| PK range exhausted（运行时） | `ErrDataIntegrity` / 1009 / System / non-retriable | 立即终止 | 禁止 | job Failed，不续号、不换 run |
| 单文件 range 上界超过 `MaxUint32`（广播前） | `ErrImportFailed` / 2100 / Input / non-retriable | 尚未创建 | 不创建 | RPC 直接返回 InputError，要求拆文件 |
| WriterSpec/schema/segment plan 结构不一致 | `ErrDataIntegrity` / 1009 / System / non-retriable | 立即终止 | 禁止 | job Failed + quiesce |
| invalid fan-in（配置或持久 plan） | 配置启动校验失败；持久 plan 为 `ErrImportSysFailed` / 2101 / System / non-retriable | 不启动非法 run | 禁止 | 配置错误阻止 DataCoord 启动；已落盘非法 plan 的 job Failed，不进入 Retry |
| Query task-not-found | ownership-lost 控制分支 | 当前 run 放弃 | 允许 | 清理 Importing 进度，Pending/Retry 后从 immutable fragment 重做 |
| commit timestamp 低于 accepted `Stats.TimestampTo` | `ErrImportSysFailed` / 2101 / System；但 job 已在 Committing point-of-no-return | 当前 vchannel fence 不通过 | 不创建 Import run | flusher持续重试/阻塞；不清 `is_importing`、不写 committed marker、不把 job Failed |

worker response 和持久化记录中的 `state` 是调度与恢复的唯一机器字段，`reason` 只用于人读。DataCoord 不能用 reason 文本推断是否可重试，也不能在恢复时重新解释已经持久化的 Pending/Failed 决策。

重试策略固定为两层：

1. **当前 run 内重试**：仅对上述可重试 storage/node error 做现有 client/backoff 的有限 operation retry；不改变 run ID、输出前缀或 segment ID。
2. **新 run 重试**：当前 run 因节点丢失、可重试错误耗尽或节点资源失败退出后，DataCoord 才递增 run ID、分配新正式 segment ID，再由 scheduler 重派。新 run 的临时输出路径由 run ID 自然隔离。

所有 `InputError`、`ErrDataIntegrity`、`ErrImportSysFailed` 和 stale no-op 都禁止 scheduler 无限重试。一般错误只有 `merr.IsRetryableErr(err)==true` 才能回到 Pending/Retry；`ErrNodeNotFound`/session lost 是旧 Import 已有的“任务所有权丢失”特殊分支，scheduler 可以按节点失效语义重派一次新 run，但不能把它当成普通可重试 merr 或在同一节点忙等。任何未知/未分类错误在 V3 外层必须先归类为 `ErrImportSysFailed`，不能落到 `Unexpected(65535)` 或默认重试分支。

### 18.2 失败矩阵

| 失败 | 行为 | 与旧 Import 的关系 |
| --- | --- | --- |
| reader/parse/字段一致性失败 | `ErrImportFailed(2100)`；当前 run 失败，不发布 ReshardManifest 或返回 Completed result，不创建新 run，job Failed | 输入错误分类和 job 终止语义与旧 Import 对齐；旧、新都不重试同一份无法解析的输入。 |
| 单文件 `pre_allocated_auto_ids` 广播前上界超过 `MaxUint32` | `ErrImportFailed(2100)`；直接拒绝并要求拆文件 | 按 PR #51825 和旧 Import 的输入边界对齐。 |
| 运行时 `pre_allocated_auto_ids` exhausted | `ErrDataIntegrity(1009)`；当前 run/job 直接 Failed，不续号、不换 run、不调用本地 allocator | 是已发布 range 与实际 source coverage 不一致的内部协议故障，不归咎用户。 |
| S3 暂时错误/节点暂时不可用 | 保留 `ErrServiceUnavailable(2)`、`ErrIoTooManyRequests(1003)` 等 retryable code；当前 run 先做 operation retry，耗尽后新 run/换节点。`ErrIoFailed(1001)` 本身不可重试，除非 storage mapper 已把具体 throttle/EOF 映射为 retryable code | 重试分类、有界 operation retry 和换节点行为与旧 Import 对齐；V3 额外使用新 run 及输出前缀隔离。 |
| 本地 ENOSPC | 归一化为 `ErrServiceResourceInsufficient(12)`；当前 run 失败并清理本地目录，允许换节点创建新 run；同一节点不无界重试 | 资源错误导致任务失败并重派的旧 Import 行为保持；V3 清理的是新增的 local spill。 |
| fragment 对象因系统 GC/生命周期竞态缺失 | 保留底层对象存储错误，job 直接 Failed；不能通过 LIST 找替代对象 | task plan 已保存准确 path，缺对象表示系统组件生命周期契约被破坏。 |
| fragment 解码/序列化失败 | 固定 fragment 对象已读到但不符合 manifest/schema 时用 `ErrDataIntegrity(1009)`，不可重试，job 直接 Failed；只有不属于已发布对象内容的纯内存转换错误才用 `ErrSerializationFailed(1004)`；若底层是明确 retryable S3 error，则先按底层 code 做当前 run operation retry | 读到了固定对象但内容不能按 manifest/schema 解码，不创建新 run 掩盖错误。 |
| fragment/intermediate run 无序 | `ErrDataIntegrity(1009)`，不可重试，终止 job；不回退 full sort 掩盖问题 | 这是 V3 预排序输入的内部协议故障。 |
| writer spec mismatch、result segment 数量不一致或零行带输出 | `ErrDataIntegrity(1009)`，不可重试，终止 job；保留正式输出供诊断，按 Failed GC 回收 | 这是 V3 不可变、自包含 task plan 契约的新内部故障。 |
| fan-in 越界/非法或缺字段的内部 task plan | `ErrImportSysFailed(2101)`，不可重试，终止 job；不按用户输入报错 | 是 V3 plan 协议的新内部边界；旧 Import 没有 direct merge fan-in/自包含 plan 字段。 |
| stale result（旧 run、已终态、同 run 重复） | 无错误、幂等 no-op；不重试、不改 job | 这是 V3 run fencing 新增的正常竞态，不应污染错误指标。 |
| writer close/结果构造失败 | 保留底层 code：可重试 S3/节点/网络错误先在当前 run 做有限 operation retry，耗尽后新 run；`ErrSerializationFailed(1004)`、`ErrDataIntegrity(1009)` 或 `ErrImportSysFailed(2101)` 直接 Failed；best-effort Drop/解绑后把本次预创建 SegmentInfo 标 Dropped | task 失败、重试和正式输出清理的总体行为与旧 Import 对齐；V3 使用新 segment/path 和迟到结果拒绝替代旧的同 segment 重置。 |
| DataCoord 在派发前崩溃 | 根据 task 的 `Running + run_id` 恢复并精确 Query；不确定则递增 run 重派 | DataCoord 重启后恢复 task 并重派失联节点任务的总体语义与旧 Import 对齐；持久 `run_id` 、CAS 和旧输出隔离是 V3 新增。 |
| DataCoord 在 result 子项提交中崩溃 | 读取 task commit marker；未完成则补提交或回收未裁决 segment | 旧 Import 也会在重启后根据已持久元数据补齐进度；V3 将该行为收敛为 result marker-last，并增加准确的 run/spec 校验。 |
| DataNode 丢失或 task-not-found | `ErrNodeNotFound(901)`/session unavailable/task-not-found 都按任务所有权丢失处理：当前 task 回到 Pending/Retry，清理本 run Importing 进度，重新选 capable/有 slot 的 DataNode，从不可变 fragment 重做整个 task | 节点丢失后重派任务与旧 Import 控制流对齐；V3 不同之处是从已写好的 fragment 重做，而旧 Import 从源文件重读。task-not-found 当前底层可能包装为 2101，但 V3 外层必须在 terminal 分类前识别为 ownership-lost 特殊分支。 |
| 旧 writer 迟到 | 只能写旧 run 独立文件名/segment 路径；result 因 run 落后被拒绝，孤儿对象由 GC 回收 | V3 新增迟到结果隔离，但不声称已经停止旧 writer；网络分区风险与 V2 同类。 |
| job timeout | 未进入 `Committing` 时标 Failed并停止创建新 run；`Committing` 后忽略 timeout，继续完成剩余 vchannel commit | 未提交阶段超时与旧 Import 用户行为对齐；排除 Committing 是首期必须修复的 point-of-no-return 保护。 |
| collection 删除 | 未进入 `Committing` 时沿用当前 checker 标 Failed，所有输出保持不可见并 best-effort Drop 后标 Dropped；Committing 后不回退 | collection 删除时未提交 job Failed 与旧 Import 对齐；V3 不增加强停写保证。 |
| collection schema 在 ImportTaskV3 job 执行中升级 | projection 等价则继续并安全提升预创建 SegmentInfo 的 schema version；projection 不等价则以 `ErrImportSysFailed(2101)` 标记 SystemError，job Failed、quiesce并回收；不能进入旧 Import 的索引/schema-bump等待环 | 这是 V3 针对旧 Import 无法正确处理 schema change 的新改善；不与旧机制声称对齐。 |

不增加源文件变更、ETag 或 CDC 文件不一致错误分支。

### 18.3 取消与 quiesce

当前失败 job 只改 task meta，不能保证 DataNode 停写。ImportTaskV3 进入任一未提交终态时，第一步必须通过 `EnterTerminalAndInitGC` 原子 fence job、创建 GC record并禁止新 run；RollbackImport WAL 的 ACK 只决定 CDC/job 语义，不证明本地 worker 已停。`Committing` 是 point-of-no-return，不进入 Failed/quiesce 路径。完整顺序固定为：

```text
原子持久化 terminal job + GCRecord，禁止新 run
→ 从 ImportMeta 按 job 枚举全部 V3 task ID，逐个调用 GlobalScheduler.AbortAndRemoveTask(task_id)
→ 对每个 task 当前 `(task_id, run_id)` 发送 version-aware best-effort Drop；节点失联则按 ErrNodeNotFound 解绑
→ DataNode cancel context，reader/sorter/merge/writer/upload 响应 task context，不再等待全局 MemoryAllocator
→ 清理本地 spill
→ 释放 task slot；V3 不存在需要释放的全局 MemoryAllocator 配额（V2 task 仍按旧路径释放）
→ Completed 仅保留各 ImportTaskV3 当前 Completed result 中非零 segment；其余预创建 SegmentInfo 标准化标 Dropped；Failed 把 task 当前 SegmentInfo 标 Dropped
→ 进入 retention/GC
```

DataCoord 必须先完成 Drop 请求或按现有 `ErrNodeNotFound` 语义解绑，再把 loser/failure/zero-row segment 用当前 `UpdateStatusOperator(segmentID, Dropped)` 或等价 composite 更新并写入 `DroppedAt`。首期明确接受：这不是旧 writer 已停止的证明，网络分区中的旧 writer 仍可能产生迟到对象；新 run 使用新的物理 segment ID/路径，result acceptance 拒绝旧 run，现有 GC missing-tolerance 和 job retention 窗口负责最终回收。已 committed vchannel 的 segment 不得进入该清理。

当前 scheduler 只有 task-ID 级 `AbortAndRemoveTask`。V3 保持这个粒度：从 pending/running/backoff/key-lock 视图移除稳定 task，并对 task 当前 `run_id` 发起 best-effort Drop。不枚举独立 run catalog，因为不存在该层。

旧 V2 `MemoryAllocator.BlockingAllocate` 当前不接收 context，V2 继续保留现状；V3 不调用它，因此不为 V3 增加可取消 allocator 等待。DataNode `TaskManager.Remove` 当前只 `Cancel()` 后从 map 删除，`DropImport` 会立即返回；V3 增加 run 校验、cancel/remove 和进程内 done 追踪只用于本地资源释放与避免旧回调更新当前 task，不把它定义成跨节点强 `DropAndWait` 协议。DataNode `Stop()` 顺序固定为：先停止接收新的 ImportTaskV3，cancel 并等待本进程可见的 V3 worker退出，再关闭 sync manager、writer、storage/plugin 等底层资源；这只是单进程 shutdown 正确性，不解决网络分区下旧节点的停写证明。

当前旧 Import 的节点失效行为是：scheduler Query task 失败，或 DataNode 返回 `Retry` 时，把 task 重置为 `Pending`，之后可重派；DataNode 进程重启后的 task-not-found 也会被 Query error 推回 Pending；`DropImportTask` 遇到 `ErrNodeNotFound` 会把 Drop 当作成功并解绑 node，而不是等待严格的旧进程停止证明。ImportTaskV3 与旧行为对齐：节点可达时发送带 task version 的 best-effort Drop；节点/session 已不可用且返回 `ErrNodeNotFound` 时按当前语义完成解绑并继续回收。task-not-found 必须在 V3 错误分类外层识别为 ownership-lost 并重派，不能因为其当前底层包装为 `ErrImportSysFailed` 就直接把 job 终止。不新增静默期、凭证 revoke、反复扫描或受管 session 证明。

这意味着 ImportTaskV3 不声称 node-not-found 在理论上证明旧进程绝不再写。风险边界与当前 Import 相同；ImportTaskV3 额外依赖 task 内 `run_id`、文件名中的 run ID、Query run 校验和 loser segment 清理来拒绝迟到结果、隔离旧输出。新 run 使用更大的 run ID 和新的正式 segment ID，不能覆盖旧对象或复用旧正式 segment。

用户取消 API 仍严格沿用当前 AbortImport/rollback 入口，本次不增加新取消 endpoint：

- 手动 Commit/Abort 只服务 `auto_commit=false`；
- Abort 在 `Committing/Completed` 拒绝；
- 其他非提交状态和真实错误的 `Failed` 都可以广播 `RollbackImport`；
- 已经由用户 Abort 的 `Failed` 幂等成功；
- `auto_commit=true` 不扩展手动取消能力。

### 18.4 DataCoord 恢复表

| catalog 状态 | 恢复动作 | 与旧 Import 机制的对齐关系（组件 + 动作） |
| --- | --- | --- |
| 老 job / `version` 缺失或为 V2 | 固定 ImportTaskV2 | **完全沿用旧 Import 机制**：`NewImportMeta`/旧 `importInspector.reloadFromMeta`、旧 `PreImportTask`/`ImportTaskV2`、旧 DataNode scheduler 和旧 catalog 都不变；恢复动作也不变。 |
| Pending，存在部分 ReshardTask | 从已有 plan 扣除已覆盖 source file ID，只对剩余 source 做 stable BFD 并创建缺少的 task；已有 Pending/Running/Completed task 继续恢复；全部 source 被覆盖后重试推进 Resharding | **与旧 Import 的增量恢复方向一致**：task 保存后即可独立调度，checker 按文件覆盖补缺失 task；V3 额外使用自包含 plan 固定每个已有 task 的输入。 |
| ReshardTask Pending/Running/Retry | 精确 Query 当前 run；节点失联、session 丢失或 task-not-found 时清理当前进度，持久化更大 run 再派发，从 immutable source fragment 重做 | **主控组件和恢复动作与旧 Import 对齐**：仍由 DataCoord global scheduler/Import task 在 Query 失败或 worker `Retry` 后回到 Pending，再选 DataNode 并 CreateTask；`ErrNodeNotFound` 仍当作已解绑。**V3 只增加**持久 `run_id`、带 run 的 Query/Drop、结果校验、新路径隔离和 task 内 spill（内存安全）；节点二进制兼容性由发布功能开关保证，不在 scheduler 重复判断。 |
| ReshardTask Completed | 按 job/task/run ID 推导 path 读取 manifest | **V3 新机制**：旧 `PreImportTask` 从 task meta 读统计，没有独立 fragment manifest；V3 manifest 在受管 OSS 固定路径下。 |
| Planning，存在部分 ImportTaskV3/SegmentInfo | `None` task 表示单 task 尚未准备完成，按其准确 segment ID 清理后重建；Pending/Running/Completed task 按 vchannel 和 fragment coverage 保留，只创建缺少的 vchannel task；全部 active vchannel 被覆盖后重试推进 Importing | **与旧 Import 的增量补 task 方向一致**；V3 额外用 `None→Pending` 保护 task catalog 与预创建 SegmentInfo 的准备窗口，并用自包含 plan 恢复已有 task。 |
| Importing run 丢失 | 新 segment ID 集合，从原始 Reshard fragments 重做 | **任务重派组件和总体动作与旧 Import 对齐**：DataCoord scheduler 仍把丢失的 worker task 回到 Pending 并重派。**V3 的执行输入不同**：从已裁决 fragment 重做并换新 segment ID；旧 Import 从源文件重读并复用/重置旧 task segment 语义。 |
| ImportTaskV3 Running | Query 原节点；若返回 Completed + segments，DataCoord 校验结果、补齐 SegmentInfo 并最后写 task Completed；若节点失联/task-not-found，则 Pending/Retry 后从 immutable fragment 重做 | **控制流与旧 Import 对齐**：旧 Import 也是 Query 完整结果后更新 SegmentInfo、再写 task Completed；V3 不增加 result object/path。 |
| 全部 ImportTaskV3 result 已完成 | 推进 `IndexBuilding`，不增加额外 segment publish 步骤 | **与旧 Import 的 segment 生命周期对齐**：非零 result 已对应 `Flushed + IsImporting` segment；V3 只因正式 segment 已排序而跳过旧 Import 的 `Sorting` 阶段。 |
| IndexBuilding | 仅针对 Completed task 中的正式非零 segment，重新执行现有 wait-for-index 检查；未完成则继续等待 | **组件和恢复动作与旧 Import 对齐**：仍使用 Import checker、现有 index inspector/index meta 和 `dataCoord.import.waitForIndex`；重启后仍是重新检查索引是否完成，不重建已完成索引。 |
| Uncommitted + auto commit | 重试 CommitImport 广播 | **组件和恢复动作与旧 Import 完全对齐**：仍由 Import checker 调用现有 durable broadcaster 重试 `CommitImport`，不新建 commit API 或 fence。 |
| Uncommitted + manual commit | 保持 Uncommitted，等待平台显式 CommitImport | **组件和恢复动作与旧 Import 完全对齐**：`auto_commit=false` 仍不由 checker 自动推进，仍等待同一 CommitImport 广播入口。 |
| Committing | 依赖现有 durable broadcaster catalog/WAL replay 继续原 commit fence，等待剩余 vchannel；不新建 fence，不重做 segment；timeout/Abort 不得降为 Failed | **组件和恢复动作与旧 Import 完全对齐**：仍由 Streaming durable broadcaster catalog/WAL replay、`HandleCommitVchannel` 和 Import checker 逐 vchannel 幂等补齐，已完成 vchannel 不重做。V3 只从 accepted result 准确枚举 segment，并增加持久 Stats.TimestampTo fence。 |
| Completed | 保持终态和已提交 segment，恢复后只继续 retention/GC | **用户可见终态与旧 Import 对齐**：不重做 Import 或 Commit，已提交 segment 继续可见。**GC 组件不完全相同**：V3 额外使用 `ImportJobGCRecord` 回收新临时前缀和 V3 task catalog。 |

上表中写“与旧 Import 对齐”时，必须以第三列列出的具体组件和恢复动作为准，不表示 V3 的 task proto、run fencing、fragment 输入或 slot admission 也与 V2 相同。正常恢复不通过 OSS LIST 猜测进度。

## 19. GC 和空间配额

### 19.1 临时对象生命周期

所有内部 plan、fragment、intermediate run、manifest/result 都位于：

```text
{root}/import_v3/{job_id}/
```

正式 segment 不在此目录。

最小正确 GC：

1. ImportTaskV3 不再调用普通 `UpdateJob(UpdateJobState(terminal))` 单独进终态，而是新增一个最小 `EnterTerminalAndInitGC`：在同一个 catalog update 中保存终态 ImportJob 的 state/reason，并创建独立 `ImportJobGCRecord{job_id,state=Quiesce,retention_deadline}`，然后同步更新内存 meta。当前 `ImportMeta.UpdateJob` 对已终态 job 直接 no-op，因此终态后的 GC 进度不能通过它回写 ImportJob。这是 V3 新增的终态与 GC marker；恢复时若发现历史/中间状态的 ImportTaskV3 终态 job 缺 GC record，以 job 现有 cleanup deadline 幂等补建。`Committing` 不调用这个失败入口，必须继续 commit/replay 到 Completed。
2. 如果 `Failed && auto_commit=false`，先保留当前 CDC 语义：查询复制状态；无法确定或暂时广播失败时保留 job 下次重试；处于复制拓扑时必须先成功广播 `RollbackImport`，或得到现有语义中可永久放弃的裁决（非 primary、collection 已删除、job 无 vchannel），才继续清理。这一行明确沿用旧 Import 的 CDC 复制状态查询、`RollbackImport` 重试和可永久放弃裁决组件；V3 只在成功回收前多一步本进程 worker cancel/cleanup，不声称远端旧 writer 已 quiesce。
3. 按 job 枚举全部 V3 task，不能只看 task 当前 run；历史 run 没有独立 catalog 记录，旧临时对象通过 job 前缀统一回收。
4. 对每个 task 先读取当前 `run_id`，再调用 version-aware best-effort Drop；明确 ACK，或按当前旧 Import 语义得到 `ErrNodeNotFound` 并完成解绑后，才允许处置该 task 的输出。不要引入 `Retired`、`Quiesced` 或其它停止状态。
5. 处置正式 segment 前，先完成 Drop/解绑：`Completed` job 只保留各 task 当前 `Completed` result 中的非零 segment；其余失败 task、旧 output segment 和零行 SegmentInfo 用标准 Dropped 更新写入 `DroppedAt`。`Failed` job 只处置未 committed segment；已 committed vchannel 的 segment 永不标 Dropped。首期接受网络分区旧 writer 仍可能迟到，依赖新路径、missing-tolerance 和 retention 降低风险。
6. 全部 task 都完成 best-effort Drop/解绑后，通过专用 catalog 方法将 GC record 单调推进为 `Retain`；此前禁止删除 job 前缀，避免把仍可能迟到的旧 run 对象过早删除。
7. 沿用 `dataCoord.import.taskRetention` 保留诊断窗口；保留期配置和运维组件与旧 Import 对齐，V3 只为新前缀使用独立 GC record。
8. 到期后先将 GC record 单调推进为 `Delete`，再对带结尾 `/` 的完整 `{job_id}/` 前缀执行幂等 `RemoveWithPrefix`。
9. 删除成功后删除 V3 task catalog，最后删除 job meta 和 GC record。

专用 GC catalog 方法只允许 `Quiesce → Retain → Delete` 的单调转移与幂等重放，不开放一般 ImportJob 字段绕过终态 fence。不需要 `Verify` 状态，但 `Delete` 必须持久化，保证“对象已经删除、catalog 还没删”时重启可以继续。孤儿扫描只删除“无任何 catalog 引用且超过现有 `dataCoord.gc.missingTolerance`”的纯数字 job 目录。

### 19.2 正式 segment GC

- PrepareRun 已为每个 IMPORT output segment 预创建 SegmentInfo；胜出 segment 后续补齐同一记录，不受临时根目录 GC 影响。
- 落败或失败的 segment 在当前 task worker 退出或完成 `ErrNodeNotFound` 解绑后标 Dropped 并写 `DroppedAt`，走现有 segment GC。
- 空输出 segment 不进入索引和 commit 集合；对应当前 task worker 退出后再标 Dropped。

当前源码仍有两个与“writer 尚未发布完整 SegmentInfo 引用”有关的既有 GC 窗口，首期固定沿用，不增加 active-run pin、不暂停 GC、不调整窗口：

- Storage V2 的 `dataCoord.gc.missingTolerance`：segment GC 发现 binlog/stats/delta 对象尚无 SegmentInfo 引用时，不会立刻当 orphan 删除，而是等待这个安全窗口。ImportTaskV3 writer close 到 SegmentInfo result commit 之间主要依赖该窗口保护。
- TEXT LOB safety window：LOB 已写出但正式 manifest/SegmentInfo 尚不可读时，LOB 暂时看起来像 orphan；现有 safety window 避免刚写完就被清理。ImportTaskV3 直接沿用这一保护。

TTL 过期行的物理删除不新建 Import 专用 GC 机制：query 层先按现有 TTL 条件隔离它，现有 compaction 按既有调度和回收路径最终物理删除它。ImportJob 临时对象的 retention GC 只清理 V3 的 plan/fragment/result 文件，不会为已过期行增加新的保留窗口或清理延迟。

这是明确接受的现有风险边界：对象在正式元数据发布前主要由时间窗口保护，不再新增一套 run 活跃引用协议。

### 19.3 配额

远端磁盘峰值至少包括：

```text
源数据（用户/managed，不计临时所有权）
+ Reshard fragments ≈ 规范化数据量
+ 当前 Import run intermediate runs（最坏额外若干轮）
+ 正式 segment（现有 import disk quota）
```

`CheckDiskQuota` 需要把 ImportTaskV3 的临时峰值估算计入 `RequestedDiskSize`，终态后像当前一样释放请求值。不要为此增加全新远端 reservation RPC；DataCoord 使用已有 quota 检查和保守估算。

本地 spill 使用 `localStorage.path`；执行前检查可用空间，写入过程中以实际错误为边界。首期不增加独立本地磁盘百分比配置。

## 20. 滚动升级、发布切换和二进制回滚

### 20.1 功能开关负责 DataNode 兼容性

ImportTaskV3 对旧 DataNode 不兼容，但 Import V3 不使用 global scheduler 的 worker version constraint。ReshardTask 和 ImportTaskV3 不实现 `MinimumWorkerVersion()`；scheduler 不读取 `QuerySlotResponse.version` 判断 Import capability，也不增加 `max_version` 或 Import 专用 capability proto。

DataNode 兼容性由发布功能开关和固定升级顺序保证：

```text
功能开关关闭
→ 新普通 Import 仍选择 V2，V3 链路视为不存在
→ 完成所有 DataNode 的 V3 二进制升级
→ 完成所有可能 active 的 DataCoord/CDC 控制面升级
→ 打开功能开关
→ 此后的新普通格式和 backup/binlog Import 固定选择 V3
```

打开开关意味着部署系统已经保证所有可注册、可调度的 DataNode 都支持完整 ReshardTask/ImportTaskV3 worker 协议。因此 global scheduler 只负责 slot admission，不再对每个 Import task重复做二进制版本过滤。PR #50978 的 `WorkerVersionConstraint` 机制继续保留给 external snapshot 等其它 task 使用，本方案不删除或改变该通用能力，只是不让 Import V3 实现这个接口。

新 DataNode 同时支持 ImportTaskV2 task 和 ImportTaskV3 task。新 DataCoord：

- `version` 缺失或为 V2 的历史 job 按 ImportTaskV2；
- 新普通格式和 backup/binlog job 固定 ImportTaskV3；L0 固定 ImportTaskV2；
- ImportTaskV3 job 一旦创建就按 V3 完成；没有节点或没有足够 slot时与旧 Import 一样保持 Pending并受现有 job timeout 控制。这里的组件是 DataCoord checker/global scheduler，恢复动作是继续轮询并在有资源时派发；不因 slot admission 改成同步 Failed。

### 20.2 发布顺序

Milvus global scheduler 和 QuerySlot 不增加 ImportTaskV3 版本开关；新旧链路选择由发布功能开关控制。安全发布顺序固定为：

1. 先逐台升级所有 DataNode。新 DataNode 同时执行 V2 和 V3；此时旧 DataCoord 仍只创建当前 V2 job，因此集群行为保持不变。
2. CDC 场景先在所有参与集群完成 DataNode 升级，由部署系统确认旧 DataNode 副本数为 0，且至少一个节点的 `available_slots` 能承载最小 V3 task。这里不依赖 scheduler 的逐 task version filter。
3. 再升级所有可能成为 active 的 DataCoord。混部期间不能让已经升级的 DataCoord 对外接收新 Import，也不能让它成为 active 后又故障切回旧 DataCoord；发布系统需使用已有流量摘除、leader/active 控制或维护窗口形成运维 fence。本文不为此新增 Import 配置开关。
4. 确认所有可能成为 active 的 DataCoord 都已升级、所有 CDC 参与集群都能解释 V3 WAL/job/catalog 后，解除运维 fence。此后所有新普通格式和 backup/binlog 请求固定 V3，L0 继续 V2。

解除 gate 前**不要求排空旧 Import broadcaster task**。gate 前已经持久化、尚未完成 ACK callback 的旧消息没有 `version`，新 DataCoord 按缺失值 V2 创建旧链路 job；gate 后新广播显式写 V3。两类 `PENDING`/`REPLICATED` broadcaster task 可以并存并各自按消息中的版本完成。发布 gate 检查的是“所有可能处理消息的 DataCoord 和 CDC 集群都理解 V3”，不是“旧 broadcaster pending 数为 0”。这也是必须把版本写进 `ImportMsg` 的直接原因。

不能在旧 DataCoord 仍可能成为 active 时接受新的 V3 Import：旧 DataCoord 会把未知 `version` 当缺失字段处理，并且不认识 Planning、ReshardTask 和 V3 task catalog，无法安全恢复或推进新 job。若部署系统不能保证上述 fence，就必须用 Import 维护窗口完成 DataCoord 切换，而不是在代码中偷偷回退 V2。

### 20.3 运行时节点变化

- 功能开关开启后，所有注册 DataNode 都必须支持 Import V3；scheduler 只要求至少一个节点满足 `available_slots >= slot`。
- 节点暂时无 slot时 task 保持 Pending，沿用现有 job timeout。DataCoord checker/global scheduler 的 Pending 轮询、timeout 判定和重新派发动作与旧 Import 对齐；V3 与旧 Import 一样使用 best-effort 超售 fallback，不新增 exact admission 分支，节点内存压力由 task 内 spill（§10）和外部监控兜底（OOM 风险见 §15）。
- 集群没有 DataNode 或全部节点没有足够 slot时同样保持 Pending、不退 ImportTaskV2。配置了 job timeout 时，由 checker 超时后标 Failed；未设置 timeout 时当前 `GetTimeoutTs` 返回 `MaxUint64`，可以无限 Pending。ImportTaskV3 不暗中增加新 timeout或同步失败分支。
- 一个 V3 Import 一旦进入 broadcaster catalog/WAL，ACK callback、CDC `REPLICATED` 消息和后续恢复都必须按持久化的 `version=V3` 创建并完成 job，不能根据当前节点集合改回 V2。
- DataNode 单节点下线或同版本替换前先 drain 该节点 ImportTaskV3 tasks；仍有其他节点时可重派。Drain 必须按 task slot 停止新 V3 task，并等待每个 task 的 context/future 退出；不需要等待或清空 V2 全局 MemoryAllocator 之外的 V3 配额。
- 功能开关开启后，部署系统禁止把不支持 Import V3 的旧 DataNode重新注册进集群。违反该发布不变量时，scheduler 不提供第二层保护，旧节点可能收到未知 task type；这是发布错误，不在运行时自动回退 V2。

### 20.4 旧二进制回滚

存在以下任一条件时禁止回滚到不认识 ImportTaskV3 的 DataCoord：

- 有尚未完成 ACK callback 的 ImportTaskV3 Import 广播；
- 有非终态 ImportTaskV3 job；
- 有活跃 ImportTaskV3 task run；
- 有 Planning、result commit 或 Committing 中的 ImportTaskV3 job；
- ImportTaskV3 临时对象或 V3 task catalog 尚未清理。

其中“尚未完成 ACK callback 的 ImportTaskV3 Import 广播”没有 ImportJob 可供扫描。回滚检查必须直接读 Streaming broadcaster catalog，解析 Import message body，找出 `version=V3` 且处于 `PENDING` 或 `REPLICATED` 的 task；不能只查 DataCoord ImportMeta。`TOMBSTONE` 表示 ACK callback 已完成且仍为幂等保留，不是未完成 blocker；已完成 task随后会从 catalog 删除，不能使用模糊的“记录仍存在”判断。

回滚流程：先用现有流量层或维护窗口停止新的 Import 请求 → drain/完成或显式失败所有 ImportTaskV3 job → 等待当前 task worker 全部退出 → 等待所有终态 job 的临时 GC 完成 → 确认 V3 broadcaster task、job、task 和 GC record 数为 0 → 再回滚 DataCoord。不手工绕过 GC 删 meta，不删除 Completed job 已正常引用的 SegmentInfo。回滚后新请求会由旧二进制按 V2 创建；这只是二进制回滚结果，不是 V3 运行时的自动降级能力。

回滚验证还必须包含 downgrade 测试：旧 DataCoord 能忽略 Completed ImportJob 的未知新字段，并正常识别现有 SegmentInfo。未终态 ImportTaskV3 job 必须为 0；否则旧 DataCoord 无法推进新状态或清理新前缀。

功能开关开启后，单个 DataNode 不能直接回滚到不支持 Import V3 的版本并重新加入集群。若必须做 DataNode 二进制回滚，先关闭功能开关停止创建新 V3 job，再完成或终止全部非终态 V3 broadcaster/job/task，确认集群中不存在可能重派的 V3 task，最后才允许旧 DataNode注册；仅 drain 单个节点不足以证明安全。

## 21. 配置设计

### 21.1 新增配置：只保留必要项

| 配置 | 缺省 | 作用 |
| --- | ---: | --- |
| `dataCoord.import.fragmentSizeInMB` | `128` | 规范化后未压缩逻辑字节软目标 |
| `dataCoord.import.fragmentMergeFanIn` | `16` | direct/hierarchical 每组最大 reader；只接受基础范围 `[2,1024]`，越界启动失败 |

不新增 ImportTaskV3 开关配置，也不新增 files-per-reshard-task、sort concurrency、source reader concurrency、packed reader/writer buffer、manifest 数量硬限制、多维资源 token 等配置。发布切换使用第 20 节的运维 fence；运行时按持久化的 ImportTask version 执行。

### 21.2 复用现有配置

| 配置 | 当前作用 | ImportTaskV3 作用 |
| --- | --- | --- |
| `dataCoord.import.maxImportFileNumPerReq` | 请求文件上限 | 不变 |
| `dataNode.import.maxImportFileSizeInGB` | 单 ImportFile 大小上限 | 不变 |
| `dataCoord.import.maxSizeInMBPerImportTask` | 旧 task 分组目标 | Reshard BFD 一维软目标 |
| `dataNode.import.readBufferSizeInMB` | reader buffer | source reader 基础 buffer |
| `dataNode.import.concurrencyPerCPUCore` | V2 Import execution pool | V2 保留；V3 使用独立 slot-aware executor，不按文件并发 |
| `dataCoord.import.memoryLimitPerSlot` | slot 内存换算 | V3 task working-set 换算 |
| `dataNode.import.memoryLimitPercentage` | Import 总内存上限 | V2 继续复用；V3 用于监控/准入参考，不建立第二个阻塞 allocator |
| `dataCoord.import.taskRetention` | job/task 保留 | ImportTaskV3 临时对象 retention |
| `dataCoord.gc.missingTolerance` | 孤儿安全窗口 | ImportTaskV3 orphan job 目录扫描 |
| `dataCoord.segment.maxSize`/disk target | segment 目标 | Planning 装箱目标 |
| `dataCoord.import.waitForIndex` | 等索引 | 不变 |

旧 `filesPerPreImportTask`、`fileNumPerSlot`、全局 Import MemoryAllocator 和旧 `baseBuffer × V × P` 计算只服务 ImportTaskV2；V3 使用第 8、14、15 节的固定 task plan、task-local working set 和节点 slot admission。

`dataNode.import.maxImportFileSizeInGB` 的 ImportTaskV3 检查落点是第 8.3 节规定的 Reshard reader 创建后、首次 `Read()` 前的 `Reader.Size()`；不能因为删除 PreImportTask 就删除这项现有输入校验。

## 22. 可观测性

遵循现有 `mlog`、Prometheus 和 OpenTelemetry 规范，不记录用户行内容、PK 原值或对象凭证。

### 22.1 日志字段

所有 ImportTaskV3 job/task 日志至少带：

```text
job_id
task_id
run_id
task_type
job_state
channel_index
partition_id
ordinal
segment_id
```

错误日志保留源 file ID、路径和逻辑 row offset，但不打印整行。

### 22.2 指标

- job 各阶段数量和时长；
- Resharding 读取行/字节、fragment 行/逻辑/物理字节；
- bucket 数、fragment 数、spill bytes/chunks；
- sort rows、sort working set、sort latency；
- manifest size、Planning bucket/segment/task 数；
- merge fan-in、round 数、intermediate bytes、heap size；
- final filter deleted/expired rows；
- function、TEXT LOB、writer、stats 时长；
- run retry/stale result/spec conflict；
- pending 无 capable node、slot 等待和 task-local working-set 超限；V3 不记录全局 memory allocator 等待；
- GC 删除对象/字节和 orphan 目录。

高基数 ID 只能作为日志/trace 字段，不能成为 Prometheus label。

### 22.3 tracing

span 层级：

```text
ImportJob
├── ReshardTask (task_id, run_id)
│   ├── source read
│   ├── spill replay
│   └── sort/write fragment
├── Planning
└── ImportTaskV3 (task_id, run_id)
    ├── segment/{ordinal}/merge round
    └── segment/{ordinal}/final writer
```

跨 WAL、DataCoord worker RPC 和 storage I/O 传播现有 trace context；不要为每行建 span。

## 23. 代码和协议实施清单

### 23.1 外部 proto 和本仓 proto

- 外部 `milvus-proto/proto/msg.proto`
  - `ImportMsg.version` 使用 `int64`，不定义 enum；字段注释固定 `0=未指定/历史 V2、2=ImportTaskV2、3=ReshardTask+ImportTaskV3`；
  - 直接复用 PR #51825 的 `common.IDRange` 和唯一的 `msg.ImportFile.pre_allocated_auto_ids`；
  - 不增加另一份 Task version 类型。
- `ImportMsg.version` 虽定义在外部 proto 仓库，但仍是 Milvus 内部消息兼容字段：原因只是 `ImportMsg` 本身定义在那里，并且该字段必须随 broadcaster catalog、WAL 和 CDC 保存。它不加入用户 RPC request，不允许 Proxy/用户选择，也不能与 Streaming `ImportMessageV1` 的 payload format version 混为一谈。旧消息的 `int64` 零值/缺失值解释为 V2；新消息必须显式写 `2` 或 `3`。
- `pkg/proto/internal.proto`
  - 在 `ImportJobState` 尾部追加 `Planning` 和 `Resharding`，复用现有 `Importing`；
  - `internal.ImportFile.pre_allocated_auto_ids` 直接使用单个 `common.IDRange`；
  - `ImportRequestInternal.version` 是内部选择/搬运字段：用户 Proxy 新请求不设置版本，由 DataCoord 在广播边界选择；老 Proxy/StreamingCoord 转发一个已经带版本的 ImportMsg 时必须原样复制，不能重新选择；ACK callback 也必须把 WAL body 中的 Task version 原样带到 job 创建边界，再显式映射为 `ImportJobVersion`。
- 固定新增 `pkg/proto/import_v3.proto`，只放不依赖 `data_coord.proto` 的纯 OSS/control-contract 类型：
  - SortSpec/SortFieldSpec、ImportSourceFormat、ReaderOptions、SourceFileSpec、ExpandedBinlogField；
  - ReshardTaskPlan/ReshardManifest、FragmentDescriptor/FragmentRef；
  - ColumnGroupSpec、TextColumnWriteSpec、V2PackedIOConfig、WriterSpec；
  - SegmentPlan 和自包含 ImportTaskPlan；
  - 文件头固定为 `package milvus.proto.data`，`go_package = "github.com/milvus-io/milvus/pkg/v3/proto/datapb"`，与 `data_coord.proto` 共用 datapb package；生成目标是 `pkg/proto/datapb/import_v3.pb.go`。
- `pkg/proto/data_coord.proto`
  - 定义独立 `ImportJobVersion`；`ImportJob.version` 只保存 DataCoord 状态机版本：Job V1 对应旧 PreImport/ImportTaskV2，Job V3 对应 Reshard/ImportTaskV3。它不直接保存 `ImportMsg.version`；ImportJob 不保存 Planning snapshot/index/generation。
  - `ImportJob` 不保存 task error code；异步 GetImportProgress 保持旧的成功 Status + Failed state/reason 行为。
  - 定义 `ReshardTask`、`ReshardTaskRequest` 及 query/drop；`run_id` 直接放在 task/request 中，不定义独立 Run 或 RunSpec；
  - 定义依赖现有 `IDRange`、`FieldBinlog`、`Statistics` 的 `ImportTaskV3`、`ImportTaskV3Request`、`SegmentResult` 及 query/drop；Completed Query 直接携带 repeated SegmentResult，同样不定义独立 Run proto；
  - 不修改 PR #50978 已有的 `QuerySlotResponse.version`，不增加 Import 专用 capability 字段；Import V3 不消费该字段做 task 调度。

该拆分是为了保持 proto import DAG 单向：`import_v3.proto` 只 import `internal.proto`、`schema.proto` 等现有下游依赖；`data_coord.proto` import `import_v3.proto` 并复用其中的 plan/manifest 类型；后者绝不 import `data_coord.proto`。凡字段需要 `IDRange`、`FieldBinlog` 或 `Statistics`，消息必须留在 `data_coord.proto`；不能形成互相 import 的循环。当前 `internal.proto` 不反向 import `import_v3.proto`。

同一 `ImportMessageV1` body 只是 protobuf 追加字段，不需要更改 `pkg/streaming/util/message/reflect_info` 中的 message type/header/body 类型定义；需要更新 go-api/proto 依赖，并增加新老字段序列化、适配、未知字段兼容测试。

`scripts/generate_proto.sh` 当前逐文件显式调用 protoc，不会因为 `data_coord.proto` import 新文件就自动生成后者的 Go 代码。实施必须在 datapb 生成段新增独立的 `import_v3.proto` protoc 输入（与 `data_coord.proto` 相同的 `--go_out=.../datapb`；文件无 service 时不要求产生 grpc 文件），并让现有 backup/restore-generated-files 逻辑覆盖其输出。之后执行 `make generated-proto-without-cpp`，确认 `import_v3.pb.go` 被生成且 `data_coord.pb.go` 能引用同 package 类型，再按模块生成 mocks；不手改 `*.pb.go` 和 `internal/mocks/*`。

### 23.2 Proxy、WAL 和 DataCoord 入口

- `internal/proxy/task_import.go`
  - 保持用户 API 不变；不在 Proxy 决定 ImportTask version。
- `internal/datacoord/services.go`
  - 普通格式和 backup/binlog 新请求在广播前固定 `version=V3`；L0 固定 V2；不读取 enable 开关，不按 `V×P`、文件大小或节点瞬时状态选择版本；
  - 用户 Proxy 进入 `ImportV2` 且未携带版本时，由 DataCoord 按 gate 规则归一化为 V3（L0 为 V2）；老 Proxy/StreamingCoord 转发已带版本的消息时沿用该版本，只有真正没有版本的旧用户消息才在该入口按“新请求”规则选择；ACK callback 使用 WAL 中已固定的版本创建 job；
  - backup 展开前移，并按稳定 paths 键排序；
  - 保留旧 Proxy 传入的非零 job ID，只有零值时分配；再分配 file ID 和普通格式每文件唯一的连续 `pre_allocated_auto_ids`；
  - 历史 WAL 缺失 version 时按 V2 创建；新消息原样创建对应版本 job。
- `internal/datacoord/ddl_callbacks_import.go`
  - `broadcastImport` 写完整 WAL contract；
  - ACK 从 `result.Results` 收集 ready channels；`ReadyVchannels` 只按集合比较，不承载路由顺序；
  - `createImportJobFromAck` 使用本地 collection info 的 `VChannelNames` 保存 `ImportJob.Vchannels`，保证与本集群正常 DML 路由一致；
  - ACK 显式复制 `version` 和每个 file 的单个 `pre_allocated_auto_ids`；
  - 保持当前 ResourceKey 和 2PC callback。
- `docs/agent_guides/streaming-system/message/message-semantic-collection.md`
  - 按当前 `startBroadcastWithCollectionID` 源码把 Import 的 ResourceKey 从“无”改为 `SharedDBName + ExclusiveCollectionName`，并注明 Broadcaster 自动附加 `SharedCluster`；
  - 说明旧 Proxy 的 Import broadcast 会先转发到 `DataCoord.ImportV2`，由 DataCoord 重新按上述 ResourceKey 发起本地广播；
  - 同一 pending todo 必须补齐表中遗漏的 `CommitImport`/`RollbackImport`；按当前源码将 `DropSnapshot` 修正为 `SharedDBName + ExclusiveCollectionName + ExclusiveSnapshotName`，将 `RefreshExternalCollection` 修正为 `SharedDBName + ExclusiveCollectionName`；
  - 对每一项从实际 producer 的 `StartBroadcastWithResourceKeys` 调用和 message builder 反查 ResourceKey，补充 callback/replication 语义测试；
  - 这是 pending documentation todo：随功能实现放在同一提交/PR完成；当前设计文档只记录差异，不直接改 agent guide，后续不能继续保留与源码冲突的表格。
- `internal/streamingcoord/server/service/broadcast.go`
  - 老 Proxy 转发 `msgpb.ImportFile → internal.ImportFile` 时显式复制 `pre_allocated_auto_ids`，并在 `ImportMsg.version` 非零时原样复制 task version；字段缺失只表示该消息是旧格式，DataCoord 结合入口语义处理：旧 broadcaster ACK 按 V2 恢复，旧 Proxy 发起的新请求按当前 gate 规则选择 V3；
  - 追加字段透明序列化，保持现有 `ImportMessageV1` message type。

### 23.3 DataCoord meta、checker、scheduler

- `internal/datacoord/import_job.go`
  - ImportTask version；GC 进度使用独立 record，不经终态 `UpdateJob`。
- `internal/datacoord/import_task_v3.go`（新增）
  - V3 task interface、task 内 `run_id`、Create/Query/Drop；状态只使用 `None/Pending/Running/Retry/Completed/Failed`；worker response 和持久化 task 都只用 state/reason，DataCoord 信任 Retry/Failed，恢复只按持久化 state 执行。
- `internal/datacoord/import_meta.go`
  - 恢复 V3 task；
  - composite job/task/segment 更新；
  - 以 job/task 两方准确旧值做 CAS，task 内 `run_id` 提供 fencing；
  - Reshard result 使用 job/task 两方 CAS，按 job/task/run ID 推导 manifest path，把 task 改为 `Completed`；
  - Import result 按 V2 式“Query Completed 完整结果 → SegmentInfo 更新 → task Completed”执行；结果按 task segment ID slice 下标对齐，时间范围直接使用 `Statistics`；
  - PrepareRun 使用 job/task 两方 CAS，IMPORT 在 Create RPC 前按 V2 生命周期预创建正式 SegmentInfo；
  - Planning 按 vchannel 恢复已有 ready task、补齐缺少 task；`None` task 按准确 segment ID 清理后重建，ready task 不删除；
  - 零行 placeholder 不提升 Flushed、不进入索引或 commit 集合，当前 task worker 退出后标 Dropped；
  - `EnterTerminalAndInitGC`：同步保存终态 job state/reason 与初始 GC record；终态后只能通过专用 GC 方法单调推进。
- `internal/datacoord/import_checker.go`
  - 仅 ImportTaskV2；从 state-machine/GC 两个 loop 过滤掉 `version==V3` 的 job；
- `internal/datacoord/import_checker_v3.go`（新增）
  - 仅 ImportTaskV3；独立 checker 实例由 DataCoord server 启动/停止；
  - V3 固定执行 Reshard barrier、Planning、Importing，再直接进入 IndexBuilding；不增加独立 result publish 阶段；
  - Planning、Import result commit、进入 IndexBuilding 前调用 `CompareImportSchemaProjection`；projection相同则继续并把预创建 SegmentInfo 的 schema version提升为当前等价版本，projection不同才让 ImportTaskV3 job Failed；禁止进入无限等待；
  - `tryTimeoutJob` 明确排除 `Committing`：一旦 `HandleCommitVchannel` 把 job 推到 Committing，timeout/Abort 不再写 Failed；`checkCommittingJob` 只等待剩余 vchannel marker，已 committed vchannel 永不进入 Failed cleanup；
  - V3 `HandleCommitVchannel` 只枚举 Completed result 非零 segment，并在清 `is_importing` 前校验持久 `Stats.TimestampTo` fence；
  - quiesce 和 ImportTaskV3 GC。
- `internal/datacoord/import_inspector.go`
  - Pending/InProgress task 只按持久化 task state 调度或恢复，不检查 job stage；`None` task 不进入 scheduler；
  - Reshard/Planning checker 负责按 source/vchannel 补齐缺少 task，并在覆盖完整后推进 job state。
- direct Create/Query error 在调用点直接用 `!merr.IsRetryableErr(err)` 判断终态；Query/Drop 先用 `errors.Is(err, merr.ErrNodeNotFound)` 识别 worker ownership 丢失；worker执行错误由 DataNode 使用 `merr.IsRetryableErr` 映射为 Retry/Failed，DataCoord不从 reason 字符串重新分类。
- `internal/datacoord/index_inspector.go`、`internal/datacoord/stats_inspector.go`
  - 不增加 Import V3 特判；与 V2 一样按 `Flushed`、sorted、level 等既有条件消费 segment。
- `internal/datacoord/import_util.go`
  - stable one-dimensional BFD；
  - per-file ordered range list；
  - Planning sequential pack；
  - V3 slot working-set；
  - fragment logical-bytes 顺序装箱和 task 级 WriterSpec 物化；
  - 每次 CreateTask 时复用 `GetReadPluginContext`/`WrapPluginContext` 填充 RPC-only plugin context，不写 OSS/catalog；
  - ImportTaskV3 不预分 sorted segment ID、不进入 Sorting。
- `internal/datacoord/task/global_scheduler.go`
  - PR #50978 的 `WorkerVersionConstraint`、`MinimumWorkerVersion()` 和 SemVer 节点过滤保持原样服务其它 task；ReshardTask/ImportTaskV3 不实现该接口，不增加 Import version constraint；
  - V3 不增加 exact admission，slot 不足时沿用旧 Import 的 best-effort fallback（节点剩余 slot 记为 0 后照常派发）；节点内存压力由 task 内 spill（§10）和外部监控兜底；
  - 同一 scheduling round 每成功放置一个 V3 task都从 heap 扣除其 slot，因此可以安全地把多个 V3 task放到同一节点；
  - 保留 `AbortAndRemoveTask(taskID)`；终态时由 ImportMeta 按 job 枚举 V3 task ID 逐个调用，并对 task 当前 run 发送 version-aware best-effort Drop；Committing 不进入终态失败清理；
  - pending/running/backoff/key lock 继续只用稳定 task ID；每个 task 只有一个 current run，所有 worker RPC/回调/result 额外校验 run ID。
- `internal/datacoord/session/{session.go,node_manager.go,cluster.go}`
  - `cluster.QuerySlot → WorkerSlots.Version` 保持 PR #50978 原样，不增加 Import capability 字段；Import V3 调度只使用 `available_slots`；
  - Reshard/V3 Create/Query/Drop 带 run。
- `pkg/taskcommon/{type.go,properties.go}`
  - 增加 `ImportV3` 字符串类型和完整 allowlist/parser 支持；旧节点对未知类型返回协议错误。

### 23.4 Catalog

- `internal/metastore/datacoord_catalog.go`
- `internal/metastore/update_action.go`
- `internal/metastore/kv/datacoord/{constant.go,util.go,kv_catalog.go,update.go}`
  - ImportTaskV3 prefix；
  - ImportJobGCRecord prefix 和终态后专用单调 GC 状态更新；
  - ImportJob/Task V3 composite entry；
  - `predicates.ValueEqual` job/task CAS，另最小增加 key-not-exists compare 供首次 task 创建；
  - `txn.Builder.CommitSave` 用于 PrepareRun/result 的最终 marker；
  - 最小扩展 `txn.Commit` 使 predicates 只作用于原子全量 txn 或 fallback 最终 commit txn；
  - SegmentInfo + task result 组合提交；
  - tests 覆盖 txn-limit fallback 和故障中间态。

### 23.5 DataNode V3 独立执行路径

- `internal/datanode/importv3_services.go`（新增，Reshard/Importing 两个 executor 入口）
  - Reshard：reader 创建后首次 Read 前执行现有 ImportFile Size 校验；随后规范化、逐文件 ID、路由、bounded bucket 累积、spill、sort、fragment、manifest；
  - Importing：multi-segment 顺序执行、分层 merge、final transform（timestamp/function），完整结果保存在进程内 TaskManager 并由 Completed Query 返回；
  - task-local resident 记账和 spill/flush 直接内联在 Reshard executor，不单独立全局 allocator 或 `task_memory.go` helper。
- `internal/util/importutilv2/binlog/{reader.go,util.go}`
  - 增加从显式 `field/group → ordered insert objects` 和 ordered delta objects 构造 reader 的入口；
  - 复用现有 verify/record reader/delete filter，ImportTaskV3 入口禁止再 LIST，旧 prefix 入口保留 ImportTaskV2。
- `internal/datanode/importv3/fragment.go`
  - fragment reader、SortFields、TTL-only predicate。
- `internal/datanode/importv3/merge.go`
  - 严格单头 `MergeExecutor` 和分层归并。
- `internal/datanode/importv3/reshard_spill.go`（新增）
  - Arrow IPC immutable chunks 和 replay。
- `internal/datanode/importv3/task_manager.go`
  - map key 仍是 task ID；task value 保存 `run_id`，Add/Update/Get/Drop 按 version 判断 stale/equal/newer，并追踪本进程 worker done；不把本地等待定义成跨节点强 DropAndWait 协议。
- 旧 `internal/datanode/importv2/`、现有 `PreImportTask`/`ImportTaskV2` 和旧 catalog API 原样保留；V3 不继承或改写旧 task 主体，只共享已经存在且语义稳定的 reader/hash/writer/storage helper。下一大版本删除 V2 时，可以直接删除旧 checker/task/proto/catalog 适配，不影响 V3 主体。
- `internal/datanode/services.go`、`data_node.go`
  - V3 task dispatch/query/drop；
  - V2 继续使用现有 importv2 scheduler/MemoryAllocator；V3 使用独立 scheduler/executor，按 task slot admission 允许同一 DataNode 并行多个 ReshardTask/ImportTaskV3；
  - `QuerySlot.importUsed = V2 slots + V3 slots`；V3 executor 不重复做第二套全局内存准入，只校验 task 字面 slot/local budget；
  - shutdown 顺序：停止新 ImportTaskV3 task → cancel 并等待本进程可见的 V3 worker → 关闭 sync/writer/storage 底层资源；
  - `QuerySlotResponse.version` 沿用 PR #50978 的现有实现但不用于 Import V3 调度；不硬编码返回 ImportTask capability 数字。

### 23.6 storage、merge 和正式写入

- `internal/storage/sort.go`、`sort_test.go`
  - 直接复用当前已有 strict one-head `MergeSort`，不再移植或复制实现；只补 Import V3 reader/predicate/context 组合测试。
- `internal/datanode/compactor/{merge_sort.go,mix_compactor.go}` 及现有测试
  - 复用当前 reader-index 诊断、close-on-error 和 `canMergeSort`/namespace 契约；不把 Import V3 改成对 compactor 主体的重写。
- `internal/storage/record_reader.go`、`rw.go`
  - 公开准确 fragment reader；
  - temporary raw-TEXT schema；
  - reader 行数/单调性 wrapper。
- `internal/storage/arrow_util.go`、`schema.go`
  - 临时 UTF-8 TEXT builder，不改变正式 LOB builder。
- `internal/storage/record_writer.go`
  - 可写单临时 packed fragment 的最小 adapter；
  - final transform writer 复用正式 writer；
  - ImportTaskV3 显式接收 per-segment column groups/formats、V2 I/O、TEXT config、PK stats capacity、Bloom filter type 和误判率，不走首 batch/paramtable 动态分支；
  - TEXT writer 构造链显式接收 plugin context，并复用 packed FFI 的 CMEK properties 生成逻辑；
  - V2/V3 close 输出提供可核对的 physical file size descriptor。
- `internal/storage/stats_collector.go`、`internal/storage/binlog_record_writer.go`
  - V2/V3 BM25 stats 按 output field ID 升序分配 log ID，PK stats 固定先分配；
  - PK stats 构造不再读取 refreshable Bloom 配置，改用 WriterSpec 显式值；
  - 对 local allocator 使用预算上界，允许 final filter 零行造成 unused tail，但禁止越界。
- `internal/storagev2/packed/packed_writer_ffi.go`
  - 安全导出 writer 生成的唯一物理 file descriptor，供 path/rows/physical size 建立 descriptor；
  - 保持 C 资源生命周期。
- `internal/datanode/importv3/task_import.go`（新增）
  - multi-segment 顺序执行、分层 merge、final transform/result。
- `internal/compaction/entity_filter.go`
  - 直接复用，不复制 TTL/delete 逻辑。
- `internal/util/function/embedding/runner.go`
  - 复用 `RunAll`，确保只在最终保留行执行一次。
- `DataType_Text` 只走 Storage V3；Importing 只写 TEXT inline/LOB，text index 继续由现有后台 inspector 创建。

### 23.7 路径、配置、指标

- `pkg/util/metautil/import_v3.go`：统一纯数字路径 builder。
- `pkg/util/paramtable/component_param.go`、`configs/milvus.yaml`：两个必要配置。
- `pkg/metrics` 和 DataCoord/DataNode metrics：阶段、fragment、merge、run、GC。
- 本文不依赖图片；审计以当前源码、权威方案和文字路径契约为准。设计文档不把图片作为 agent 的实现输入。

## 24. 测试、故障注入和 benchmark

所有 Go 测试必须带：

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 ...
```

### 24.1 底层排序与归并

- `storage.Sort`：Int64 radix、VarChar、多字段、nullable 非 key 字段、TEXT temporary schema。
- strict MergeSort：
  - Int64、VarChar、多字段；
  - 0/1/16 readers；
  - 首 reader 为空、连续空 batch、全部 predicate=false；
  - 跨 RecordBatch；
  - predicate 每行恰好一次；
  - duplicate PK；
  - 无序输入报告 reader/record/row；
  - borrowed record 生命周期；
  - reader/writer error、context cancel。
- benchmark：1/4/16 readers，不同 batch size、Int64/VarChar、宽 schema，记录 heap、allocs、PSS、吞吐。

### 24.2 Reshard

- JSON、JSON Lines、CSV、Parquet、NumPy；
- backup Storage V2/V3，以及 ImportTaskV2/ImportTaskV3；
- 显式 PK、AutoID、`allow_insert_auto_id`；
- file range 重试/CDC 字面一致；
- partition key、namespace SortSpec；
- 0 行、单行大记录、超目标单文件；
- memory 路径与强制 spill 路径逐行等价；
- spill 切换点、EOF 尾批、多 bucket、取消、ENOSPC；
- 多个 V3 task 在同一 DataNode 并行时的 slot admission、task-local resident bytes 不互相阻塞、一个 task 取消不影响其它 task；
- 现有读/写 storage plugin context 透传，加密 collection 的 fragment、intermediate run 和正式 segment 可恢复读取；
- fragment 单 bucket、单 group、行数和逻辑字节；
- Int64/VarChar/多字段 sort key 的排序与 merge；
- JSON/CSV/Parquet/NumPy/backup 在 reader 首次 Read 前执行现有单 ImportFile Size 上限，边界值和超限错误分类不变；
- fragment close 后记录 path/rows/logical_bytes，不额外 GET；
- manifest-last，以及固定 path 下的恢复。

### 24.3 Planning

- task 分组 BFD 的稳定性、best-fit 选择、相同剩余空间 tie-break 和超大单文件独占 task；
- fragment 规范序；
- 顺序 segment 装箱；
- 单大 fragment；
- 一个 vchannel 多 partition 和连续 partition range task；
- Reshard/Importing slot 公式、ceil 边界和高 `V×P` 不再乘 bucket 数；
- 使用 fragment `logical_bytes` 进行顺序装箱；
- 自包含 ImportTaskPlan 恢复不受配置变化影响；
- worker 由 job/task ID 推导固定 path，只读取一个自包含 plan，不从 job、目录 LIST 或第二个 Planning 对象猜输入；
- WriterSpec 固定 schema、task 级 column groups/formats、V2 I/O、TEXT V3 config、PK stats capacity、Bloom filter type/误判率；TTL 只固定 collection TTL duration，每个过滤 batch 使用当时的系统时钟。
- 相同 fragment 统计在不同配置/节点/重启下仍复用 task plan 字面 spec；DataNode 留空触发动态分组必须失败；
- Planning 后动态刷新 Bloom 配置，再重试同一个 task plan，仍使用 plan 中冻结的 Bloom 配置；
- V2/V3 log ID 预算公式、`[begin,end)`、final filter 产生零行时允许 unused tail、越界失败；
- 多 BM25 output 按 field ID 升序绑定 log ID；
- 每 fragment 恰好被一个 plan 引用。

### 24.4 分层 merge

- 输入数 `1/16/17/255/256/257`；
- 每组实际 fan-in `<= plan.fan_in`；plan fan-in 只允许 `[2,1024]`，测试覆盖 2/16/1024/1025 的基础边界，不把 1024 测试写成重型 reader 压测；
- 连续分组和 deterministic path；
- intermediate 每轮行数守恒、单调；
- intermediate run close 后取得 path/rows，不额外 GET；只有 close 成功后才交给下一轮；
- hierarchical 结果与一次 full sort 的行 multiset 和排序键一致；
- intermediate 不执行函数、TTL、timestamp、stats、LOB。

### 24.5 最终语义等价

用同一输入对照 ImportTaskV2 `Import + sort compaction`：

- 普通 DataTs、commit timestamp；
- TTL、TTL field；
- 普通 Import final merge 不读取 collection delete snapshot、不带 deltalog ref、不调用 delete predicate；Import 后普通数据正常写入，之后的 DML delete 能被 Query 隐藏，后续 compaction 能物理清理，最终结果与 V2 Import + Query/compaction 一致；
- backup time range/deltalog；
- TextEmbedding、BM25、MinHash；
- function output 已提供/缺失规则；
- raw TEXT、inline/LOB；
- CMEK collection 的普通 V3 与 TEXT V3：fragment、intermediate、正式 data、inline/LOB 使用同一 plugin context，重启换节点后可读；PK/BM25 stats 保持当前无 plugin-context 写入边界并验证可读；
- `DataType_Text` 强制 Storage V3，验证不存在 V2 TEXT/LOB 分支；
- dynamic、nullable/default、StructArray、ArrayOfVector；
- Storage V2/V3 binlog/manifest/stats；
- sorted flags、segment positions、Statistics；
- duplicate PK 全部保留。

对照的是逐行和元数据语义，不要求 segment ID、文件名或相同 key 的物理顺序相同。

### 24.6 run、恢复和 2PC 故障注入

- DataCoord 在 task `run_id` 持久化前/后、Create RPC 前/后崩溃；
- Reshard result 的 task marker 前后崩溃；Import Query 返回 Completed 后 DataCoord 分批补齐 SegmentInfo、最后写 task Completed marker，崩溃后重新 Query/拉取同一完整结果并幂等重放，节点失联/task-not-found则重派整个 task；不增加 `RESULT_VALIDATED`、`RESULT_COMMITTING` 或其它结果状态；
- Pending→Resharding 创建中崩溃，已有 task 继续运行，只对未覆盖 source 创建稳定 task ID，不重新分组已有 task；task 已齐但 job state 未更新时只重试状态推进；
- 旧 Proxy 传入非零 job ID 保持不变，零值才分配；
- DataNode 双 writer、旧 result 晚到、旧 Drop 晚到；
- result 子 SegmentInfo 写完但 task marker 未写；
- ImportTaskV3 Running/Retry/Failed Query 不带 segments，Completed Query 按 task segment slice 顺序返回完整结果；TaskManager 在成功 run 完成后保持结果只读，Drop 后释放；
- Completed Query 内层 protobuf 达到本地 gRPC result limit 的 50% 时记录 size warning，日志不包含 binlog/manifest path；超过真实外层 gRPC 限制时得到 `ResourceExhausted` 并按当前边界让 job Failed，不触发对象 fallback；
- DataCoord 重启但原 DataNode 存活时重复 Query 同一结果并幂等补写；DataNode 重启丢失进程内结果时使用新物理 segment ID 重做 task；旧/新 V3 result 协议不得混跑 active job；
- final filter 产生零行 placeholder：无 writer/log/manifest、不 Flushed、不进入索引或 commit 集合，当前 task worker 退出后 Dropped；
- 非零 result 在 task marker 前已成为 `Flushed + IsImporting` 时，允许 index/stats 沿用 V2 行为提前创建 task/meta；若该 run 后续失败，验证旧 segment 及其后台产物沿现有 segment ID 清理路径回收；零行 placeholder 保持 Importing，当前 task worker 退出后再 Dropped；
- 错误矩阵逐项验证 Input/System、`Retriable`、worker Retry/Failed state 和 scheduler 动作：fragment order/WriterSpec/result count mismatch=`ErrDataIntegrity(1009)`；invalid fan-in/自包含 task plan 缺字段=`ErrImportSysFailed(2101)`；广播前单文件上界超限=`ErrImportFailed(2100)`；运行时 PK range exhausted=`ErrDataIntegrity(1009)`；stale result=no-op；可重试 S3 error 在 DataNode 映射为 Retry；
- final output path、output promotion、IndexBuilding marker 各崩溃点恢复；
- 部分 ImportTaskV3 已 Pending/Running、部分仍为 None 或缺失时，清理 None、保留 ready task、补齐缺少 vchannel；task 已齐但 job state 未更新时只重试状态推进；
- Importing round 中节点丢失；
- DataNode shutdown 时多个 V3 task 正在执行或等待 slot；V3 不等待全局 MemoryAllocator，V2 旧 task 仍按旧路径停止；
- V3 task 在 local working-set 达到上限时同步 spill/flush，不使用全局 MemoryAllocator；
- job timeout/Abort 与 result 竞争；
- CommitImport 在本地尚未 Uncommitted 时到达；
- CDC 主备普通 AutoID 逐行相同；
- per-vchannel segment 分批 visibility 更新中重启；commit timestamp 低于 accepted `Stats.TimestampTo` 时 flusher持续重试且不写 marker；Committing timeout/Abort 不降 Failed，已 committed vchannel 不进入 Failed cleanup；
- 全部 segment 已更新但 `committed_vchannels` marker 保存失败，重试幂等补 marker。

### 24.7 benchmark 和上线门槛

场景至少覆盖：

- 小 `V × P`，确认额外临时 I/O 的代价；
- `V=16,P=1024` 的高 bucket 场景；
- 多小文件、少量超大文件；
- 无 spill、强制 spill；
- 16/17/256 fragment fan-in；
- TEXT + embedding；
- backup shard 拓扑变化。

验收门槛：

- 无丢行、重行或错误 bucket；
- heap head 数不超过 fan-in；
- task 峰值内存不随 bucket 数或 task 内 segment 数线性增长；
- 新链路正式 segment 数和 binlog 数显著接近按 bucket 总数据量规划的理论值；
- etcd 不再保存 ImportTaskV3 `hashed_stats`；
- 所有故障注入后没有可见的落败 segment、没有旧 run 覆盖；
- 等价矩阵通过后，按第 20 节运维 fence 完成发布切换。

## 25. 实施顺序和完成定义

这是一次完整实现，不把核心架构留到“后续阶段”。同一个 feature 合入必须包含：

1. WAL `version`、file ID/每文件唯一 `pre_allocated_auto_ids` 前移和 CDC 测试；覆盖“旧无字段 broadcaster task 与新 V3 task 并存”的滚动升级场景。
2. V3 job/task proto、task 内 run fencing、catalog、scheduler 和恢复。
3. Reshard reader、规范化、spill、固定预排序、fragment/manifest。
4. Planning barrier、规范序装箱、multi-segment Importing plans。
5. 复用当前已有 `fb960cf7c2` strict MergeSort，实现可配置 fan-in 的 hierarchical merge（基础范围 `[2,1024]`）。
6. final filter/function/TEXT/Storage V2/V3 writer 全语义。
7. Importing 完成后直接进入 IndexBuilding；只保留一个清晰的状态推进调用点。
8. 组合 result commit、2PC、取消、GC；不增加 Importing 后的额外处理阶段。
9. 基于发布功能开关的 DataNode 全量先升级、再开放 V3 的滚动升级/回滚保护；ReshardTask/ImportTaskV3 不实现 `MinimumWorkerVersion()`。测试覆盖开关关闭时只创建 V2、全量 DataNode 升级后开关开启创建 V3，以及开关开启后禁止旧 DataNode重新注册的发布约束。
10. 完成 pending Streaming System agent guide ResourceKey todo：Import、CommitImport/RollbackImport、DropSnapshot、RefreshExternalCollection 一起校准，不只修 Import。
11. 指标、trace、全部测试和 benchmark。

只有以下条件同时满足才能声称完成：

- 新普通格式和 backup/binlog 请求固定走 ImportTaskV3；L0 和历史/缺失 version job 继续走 ImportTaskV2；已有 job 按持久化 `version` 继续；
- backup 已在新链路；
- fragment 必排序；
- Importing 是严格 one-head merge，超过自包含 task plan 中已固定的 `fan_in` 会分层；fan-in 默认 16、基础范围 `[2,1024]`，实际资源由 task working-set slot 估算和 DataNode slot admission 负责；
- 正式 writer 全语义通过；
- run fencing 和 composite commit 经过故障注入；
- 滚动升级/回滚 runbook 验证；
- 发布切换前等价与资源验收通过；
- V2/V3 的创建、恢复和滚动升级/回滚行为都有对应测试。
- 第 9、10、11 项已确认并统一写回正文；不存在未决的首期执行边界。

## 26. 审计结果和边界

本文已经按权威方案、当前源码和用户补充要求反向审计，确认已处理：

- 删除 V×P 阈值回退；新普通格式和 backup/binlog job 固定 V3，L0 和历史/缺失 version job固定 V2；
- backup 纳入，L0 保持 ImportTaskV2；
- 普通格式每文件唯一的连续 `pre_allocated_auto_ids` 前移到 WAL，CDC 使用相同字面范围；单文件上界超过 `MaxUint32` 时按 PR #51825 拒绝并要求拆文件；
- 保留旧 Proxy 传入的非零 job ID，零值才分配；
- fragment 预排序固定开启；
- V3 删除全局 `MemoryAllocator` 依赖，使用 task-local working-set + DataNode slot admission；同一 DataNode 允许多个 V3 task 并行，V2 allocator 和 V2 scheduler 保持不变；
- task 改为稳定的一维 BFD，segment 保持规范序顺序装箱；
- 一 Importing task 顺序执行多个 segment；
- Importing 完成后直接进入 IndexBuilding，并收敛到单一状态推进调用点；
- Planning/Importing 依赖 `FragmentRef`，但未增加外部 fragment 入口、字段、枚举或对象所有权协议；
- OSS 路径全部使用裸数字 ID；
- 删除源 ETag/version/source identity、verified cache、行 hash 守恒；
- 删除重型多维资源 RPC和大量新硬限制配置；
- 复用基线已有 strict one-head MergeSort，并增加 TEXT temporary adapter、rollback/rolling upgrade/quiesce/marker-last result commit。
- 增加不可变 WriterSpec，固结 storage/schema/column groups/writer format/TEXT 和 TTL duration 语义；TTL 过滤使用实时时钟，不保存 reference time，只要最终一致即可；
- 补齐 TEXT+CMEK 构造链、Bloom 参数冻结和单文件 Size 校验迁移；
- 补齐最小 plan/manifest proto、proto 单向依赖、零行 SegmentResult 和 task 增量恢复；
- 修正 log ID 为可证明的预算上界，不再声称与实际消耗精确相等；
- backup ImportTaskV3 使用显式对象 reader 入口，不在 DataNode 再 LIST；
- 精确限定 per-vchannel marker-last 语义；
- 后台消费只处理当前正式非零 segment。

本轮附件评审的逐条裁决如下。这里区分“首版必须改的协议”和“源码事实下接受的风险”，不把尚未实现的新 proto 当成当前源码缺陷：

| 评论 | 结论 | 本文裁决和源码依据 |
| --- | --- | --- |
| A1：不要把 result path 或额外结果状态当成接受协议，按完整结果更新 SegmentInfo、最后 task Completed | **成立，首版修改** | V3 Query 直接返回 Completed + segments，DataCoord 校验当前 job/task/run 后幂等更新预创建 SegmentInfo，最后写 task Completed。网络分区旧 worker 仍可能继续写，`run_id` 和新物理 segment 路径只阻止迟到结果成为当前结果，不是强 fencing。 |
| A2：DataNode 丢失从 immutable fragment 重做，与 V2 控制流对齐，不做本地 checkpoint | **成立，首版修改** | 当前 scheduler 在 Query error 时把 task 回 Pending；DataNode task 只在内存，进程重启后 task-not-found 也走 Query error。V3 清理当前 Importing 进度并重派整 task，从 immutable fragment 重做，不恢复内部 round。`task-not-found` 在 V3 外层按 ownership-lost 特殊分支处理，不能被同名 2101 terminal code 截断。 |
| A3：CommitImport 是 per-vchannel、marker-last，不是 job-wide 原子；Committing 应为 point-of-no-return | **成立，首版修改** | 当前 `HandleCommitVchannel` 先 callback 更新 segment，再追加 `committed_vchannels`；Streaming flusher 先 flush 到 WAL TimeTick，成功回调后才 observe，天然允许部分 vchannel 可见。V3 继续复用这些组件和动作，不新增 PartialCommitted/job-wide 事务；补充 `tryTimeoutJob`/collection-drop 对 Committing 的保护，已 committed vchannel 不进入 Failed cleanup。 |
| A4：普通 Import final merge 不处理 collection delete；backup/binlog 的 source deltalog 语义独立 | **成立，首版修改** | 当前普通 reader 和 `ImportTaskV2` 不读 collection delete snapshot/deltalog，也不调用 `EntityFilter`；旧链路后续 SortCompaction 才处理已附着 deltalog。V3 普通 final merge 只处理 TTL/TTL field，DML delete 继续由 Query/后续 compaction 处理；backup source time-range/deltalog 只在 Reshard 过滤一次，不重复执行。 |
| A6：commit timestamp fence 必须持久化，使用 `Stats.TimestampTo` | **完全成立，首版修改** | V3 result 直接持久 `Statistics`；CommitImport 读取 `SegmentInfo.Stats.TimestampTo`（兼容 V2 时取 Stats 与 binlog 上界最大值），低于上界则 fence 不通过、flusher 重试，不清 `is_importing`/不写 marker。 |

未采纳的隐含更强主张：A1/A2 文字中的 `DropAndWait`、writer epoch、lease、旧 writer 已停止证明不属于当前 V2 行为，也不是本轮已授权新增的强协议；本文只保留未来方向说明。A4 也不把“V3 final merge 不处理 collection delete”误写成“系统永远不处理 delete”：后续 Query/compaction 仍按当前 deltalog 机制处理。

以下记录审计后的执行边界。所有首期执行边界已经确认并写回正文；它们都不检查用户源文件是否变化，也不比较 CDC 两端文件内容。

### 26.1 已确认的首期执行边界

1. **没有 DataNode 或没有足够 slot**：与旧 Import 对齐。scheduler 没节点或没有足够 slot时 task 保持 Pending；配置 timeout 时 checker 超时后 Failed；未配置时 `GetTimeoutTs=MaxUint64`，可无限 Pending。ImportTaskV3 不回退 ImportTaskV2。功能开关开启后，所有注册 DataNode 都由发布系统保证支持 V3；一个节点可以并行接收多个 V3 task，只要每个 task 的 `slot` 都能从该节点 `available_slots` 中扣除。
2. **DataNode session/node 失效**：与旧 Import 对齐。Query 失败、session 丢失、进程重启后 task-not-found或 worker 返回 Retry 时，task 回到 Pending/Retry，清理当前 Importing 进度，重新选择有 slot 的 DataNode，并从不可变 fragment 重做整个 task；DataNode task 状态只保存在内存，不增加本地 checkpoint恢复。Drop 遇到 `ErrNodeNotFound` 当作成功并解绑。ImportTaskV3 保留 run ID、独立文件名/segment ID和结果校验隔离迟到结果，但不新增 session 停写证明；task-not-found 是 ownership-lost 特殊重派分支，不按 2101 terminal错误处理。V3 task 被取消时只取消自己的 context 和本地 working set，不影响同节点其它 V3 task。
3. **内部对象校验**：不做 fragment/intermediate 或 task plan 内容 checksum，不为校验额外执行完整 GET。plan 对象由 job/task ID 推导的固定 path 选中，不在对象内回显 identity。Import result 不写对象，Completed Query 直接返回完整结果。
4. **单行超过 task-local working-set 上限**：当前 run 按资源失败终止，不临时超额、不增加复杂防御。主要信任 task slot 的 peak-memory 估算，极端偏差交给现有 Prometheus/SRE 监控。
5. **单文件 ID 上界超过 `MaxUint32`**：以 PR #51825 为权威，直接拒绝并要求拆文件。一个文件只有一个连续 `pre_allocated_auto_ids`，不能由多个 range 拼接；整个 job 可用多个文件、多个 allocator batch，因此总量可超过 `MaxUint32`。
6. **新请求的版本**：新普通格式和 backup/binlog 请求在广播前固定 `version=V3`；L0 固定 V2。历史消息缺失字段时按 V2恢复；已经进入 WAL/ACK/CDC、已经创建或正在执行的 job 始终按消息/Job 中已持久化的 version 继续，不能按节点瞬时能力回退。开 gate 前不要求旧 broadcaster task 排空，旧无字段消息与新显式 V3 消息可以并存；但解除 gate 后旧 DataCoord 不能再成为 active，否则它会把 V3 当缺失字段按 V2 误处理。
7. **Storage V2 orphan GC 窗口**：沿用 `dataCoord.gc.missingTolerance`。它是在删除无 SegmentInfo 引用的旧 binlog/stats/delta 前等待的安全窗口；ImportTaskV3 不增加 active-run pin 或暂停 GC。由于首期不提供强停写证明，旧 run 的迟到对象仍依赖该窗口和 job retention 兜底。
8. **TEXT LOB GC 窗口**：沿用现有 LOB safety window。LOB 在正式 manifest/SegmentInfo 可读前暂时像 orphan，该窗口避免刚写完就被清理；ImportTaskV3 不增加 active-run pin 或暂停 GC。
9. **正式 log ID range**：每个 Importing task 一个连续 `[begin,end)`；Planning 从 `segments + writer` 计算预算并检查 `<= MaxUint32`，但不把可推导的 `log_count` 重复写入 plan。超限直接以 `ErrImportSysFailed(2101)` 拒绝，不使用 repeated ranges、不自动拆 task。
10. **TTL 时钟**：只保存 collection TTL duration，不保存 `filter_reference_time`。ImportTaskV3 在实际最终过滤每个 batch 时直接使用当时 `time.Now()`，已过期行尽早剔除并不写入正式 segment。已写入的行由现有查询 TTL 过滤和 compaction 最终物理回收，同一行在不同 batch/run 的处理时间不一致不影响协议；用户 API 不应返回已过期行，内部只要最终一致即可。该 TTL 规则与普通 collection delete 分开，不读取 collection delete snapshot，也不在 Import final merge 构造 delete map。
11. **Import 期间 collection schema 升级**：比较 frozen/current 的 Import-relevant projection。projection等价时继续并把预创建 SegmentInfo 的 schema version安全提升到当前等价版本；projection不等价且影响物理字段、function output或 target index输入时，以 `ErrImportSysFailed(2101)`（SystemError、不可重试）让 job Failed、quiesce并回收，不能进入旧 Import 的索引/schema-bump等待环；不使用 `ErrCollectionSchemaMismatch(109)`，因为这不是用户 Import 请求格式错误。只改描述等无副作用变化继续。
12. **V3 错误分类和重试**：`ErrImportFailed(2100)` 只用于广播前用户输入错误；`ErrDataIntegrity(1009)` 用于 fragment/manifest/Query result/spec/range 契约破坏；`ErrImportSysFailed(2101)` 用于 fan-in、自包含 task plan 缺字段、状态等内部协议错误；这三类都不可重试。S3/节点暂时错误保留底层 retriable 语义，当前 run 做有限 operation retry，耗尽后 DataNode 返回 Retry；stale result 是 no-op。worker task result、V3 task 和 ImportJob 都不保存独立 task failure code，只保存 state/reason；DataCoord信任 Retry/Failed state，重启恢复只按持久化 state 执行。
13. **Flushed segment 的后台消费**：源码中的 index/stats inspector 主要按 `Flushed`、level 和 sorted flags 选择 segment，不以 `is_importing` 作为后台消费 gate。V3 明确与 V2 对齐：完整的单 segment result 被应用并成为 `Flushed` 后即可提前创建 index/stats task；task `Completed` 只作为 checker 接受整个 task result 的 marker，`IsImporting` 只保护 CommitImport 前的查询可见性。若 run 后续失败，旧物理 segment ID 及其后台产物走现有 Dropped/GC 路径。这里不引入 V3 专用 `IsInvisible` 或额外 publish/recovery 阶段。
14. **ImportTaskPlan 的 WriterSpec**：每个 ImportTaskPlan 直接内嵌该 task 使用的一份 WriterSpec、target/temporary schema 和其它冻结执行参数；不再存在 WriterSpec 表、`writer_index`、PlanningSnapshot、ImportPlanIndex、planning generation 或 checksum。worker 由 job/task ID 推导 path，只读取一次 plan。

### 附录 A：未来真正开放功能时再裁决

- **用户取消范围**：本次严格保持当前 `auto_commit=false` AbortImport 范围，不给 `auto_commit=true` 增加取消；若以后要改，作为独立 API/语义设计。
