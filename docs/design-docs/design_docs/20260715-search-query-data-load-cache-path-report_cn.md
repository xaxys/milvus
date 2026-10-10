# Milvus Search / Query 路径与数据加载、缓存、Lazy Load、Warmup 详解

- 日期：2026-07-15
- 代码基线：`4f730d9985`
- 分支：`feat.metrics_storage_access`
- 范围：Proxy、QueryCoord、QueryNode v2、Delegator、Segcore、Tiered Storage caching layer
- 重点：Search / Query 请求在什么阶段访问什么数据；数据何时从对象存储冷读、何时命中本地缓存；lazy load、warmup、mmap、eviction 的真实行为

## 1. 执行摘要

Milvus 的查询路径可以分成两条相互独立但最终汇合的链：

1. **控制与路由链**：Proxy 解析请求和计划，选择每个 channel 的 Delegator；Delegator 等待 tSafe、选择 sealed/growing segment，并把子任务分发给真正持有 segment 的 QueryNode worker。
2. **数据访问链**：worker 进入 segcore 执行表达式、向量搜索和结果字段回填。只有这一层真正触碰字段数据、标量索引、向量索引，并可能触发对象存储冷读。

核心结论：

- Proxy 的 MetaCache、shard leader cache，以及 Delegator 的 distribution snapshot 都属于**元数据/路由缓存**，不会读取用户字段或索引字节。
- sealed segment 加载时，当前代码不是简单地“把所有文件全部下载进内存”，而是先建立 segment、index/column 的 `CacheSlot` 和 translator。translator 知道数据源、cell 划分、mmap 策略和 warmup 策略。
- `warmup=sync`：创建 `CacheSlot` 时同步 pin 所有 cell，segment load 完成前就会完成数据或索引加载。
- `warmup=async`：segment 可以先进入可服务状态，后台预取；若请求先到，请求会等待同一个 cell 的加载结果。
- `warmup=disable`：只建立元数据和 lazy handle；第一次 Search/Query 调用 `PinCells` 时才加载相关 cell，请求线程会承担冷读延迟。
- 当前默认值是：scalar field `sync`、scalar index `sync`、vector index `sync`、vector raw field `disable`。因此默认配置下，标量过滤和普通索引搜索通常已被预热，而原始向量字段更可能在结果回填或无索引 brute-force 时首次冷读。
- 默认 `queryNode.mmap.vectorField=true`。原始向量的第一次冷读通常经历“对象存储读取 → 解码 → 写入本地 mmap 文件 → 建立映射”；之后由 Milvus cache cell 和 OS page cache 共同服务。
- tiered storage eviction 默认关闭。关闭时，已加载的 cell 通常一直保留到 segment release；开启后，未被 pin 的 cell 可被淘汰，后续访问会再次冷读。
- Search 和 Query 不一定加载同样的数据：
  - Search 先过滤，再访问向量索引/向量原始数据，最后只对 TopK 结果做 PK 和 output field 的 late materialization。
  - Query 先过滤，再按 limit/order/group/aggregation 处理；普通有限 limit 查询可启用两阶段 retrieve，先只取 PK/offset，跨 segment 归并后再按最终 offset 拉取 output fields。
- `scanned_remote_bytes` 在当前实现中实际来自 caching layer 的 `scanned_cold_bytes`。它表示“本次访问涉及的未缓存 cell 的持久化字节估计”，并不总等于 provider 实际 GET response bytes；只有开启 `storageUsageTrackingEnabled` 时才会被累计。

## 2. 总体架构和请求时序

```text
SDK
  |
  | Search / Query
  v
Proxy
  |-- MetaCache: collection/schema/partition 元数据
  |-- 解析 DSL/expr，生成 serialized plan
  |-- ShardClientMgr/LB: channel -> Delegator
  |
  +----------------------- 每个 channel 一次 RPC ----------------------+
                                                                      |
                                                                      v
                                                              QueryNode Leader
                                                              Shard Delegator
                                                               |-- wait tSafe
                                                               |-- pin distribution
                                                               |-- segment prune
                                                               |-- sealed 按 worker 分组
                                                               |-- growing 留在 delegator
                                                               |
                              +--------------------------------+------------------+
                              |                                                   |
                              v                                                   v
                       QueryNode Worker                                    Delegator 本机
                       sealed segments                                     growing segments
                              |                                                   |
                              +------------------- scheduler ---------------------+
                                                  |
                                                  v
                                               Segcore
                           predicate -> vector/retrieve -> materialize fields
                                                  |
                               CacheSlot hit -----+----- CacheSlot miss
                                                  |             |
                                                  |             v
                                                  |      translator.get_cells()
                                                  |             |
                                                  |      object storage / manifest
                                                  |             |
                                                  |      memory or local mmap file
                                                  v
                                          per-segment result
                                                  |
                           worker reduce -> delegator reduce -> proxy reduce
```

关键代码入口：

- Proxy Search：[`internal/proxy/task_search.go`](../../../internal/proxy/task_search.go)，`PreExecute`、`Execute`、`searchShard`、`PostExecute`。
- Proxy Query：[`internal/proxy/task_query.go`](../../../internal/proxy/task_query.go)，`PreExecute`、`Execute`、`queryShard`、`PostExecute`。
- QueryNode leader：[`internal/querynodev2/handlers.go`](../../../internal/querynodev2/handlers.go) 的 `searchChannel` / `queryChannel`。
- Delegator：[`internal/querynodev2/delegator/delegator.go`](../../../internal/querynodev2/delegator/delegator.go) 的 `Search` / `Query` / `organizeSubTask` / `executeSubTasks`。
- Worker task：[`internal/querynodev2/tasks/search_task.go`](../../../internal/querynodev2/tasks/search_task.go) 和 [`internal/querynodev2/tasks/query_task.go`](../../../internal/querynodev2/tasks/query_task.go)。
- Segment 访问：[`internal/querynodev2/segments/search.go`](../../../internal/querynodev2/segments/search.go)、[`retrieve.go`](../../../internal/querynodev2/segments/retrieve.go)、[`segment.go`](../../../internal/querynodev2/segments/segment.go)。

### 2.1 本文中的关键术语和边界

下表只解释容易影响查询路径判断的 Milvus 特有概念。

| 术语 | 在本文中的准确含义 | 容易混淆的边界 |
|---|---|---|
| vchannel / shard | collection 写入流的逻辑分片，也是 Proxy 分发 Search/Query 的基本路由单位 | 一个请求通常会访问多个 channel；channel 不是 segment，也不等同于 partition |
| Delegator | 每个 replica 内某个 channel 的查询入口，维护该 shard 的可读 segment 分布、tSafe 和 delete 状态 | Delegator 决定“去哪些 worker 查”，通常不持有全部 sealed field/index bytes |
| Worker | 真正持有 sealed segment 并执行 segcore Search/Retrieve 的 QueryNode | 同一个 QueryNode 进程可以同时承担 Delegator 和 Worker 角色 |
| Target | QueryCoord 期望系统最终拥有的 channel/segment 集合 | 是 desired state，不代表数据已经加载完成 |
| Distribution | QueryNode 当前实际上报告的 segment 所在节点和版本 | 是 observed state；Target 与 Distribution 的差异驱动 load/release/reopen |
| Sealed segment | 已停止接收普通 insert、通常已有持久化 binlog/manifest/index 的 segment | 查询时可能只加载了 metadata 和 lazy handles，并不保证所有 bytes 都已 resident |
| Growing segment | 正在接收 DML 的 segment，数据主要由 WAL/DML pipeline 写入 QueryNode | 通常不存在 Search/Query 首次从对象存储加载整列的问题 |
| Chunk | segcore 对一列连续行范围的访问单元 | 是执行侧概念；在 Storage V1 中经常对应一个 field binlog |
| Cache cell | caching layer 的加载、pin、命中统计和 eviction 单元 | cell 可能包含一个 chunk、多个 row groups、一个 column group chunk 或整个 index object |
| Column group | Storage V2/V3 把多个列组织在一起的持久化/读取单元 | 一个 group 可被投影成只含某个 lazy field 的独立 cache entry |
| Translator | `CacheSlot` 的数据源适配器，负责估算资源并把 cell id 翻译成实际读取操作 | translator 不是 cache；它只在 warmup 或 cache miss 时执行读取和解码 |
| Pin | 为请求取得 cell 的稳定引用；pin 期间 cell 不能被淘汰 | pin 不等于预热。cache hit 和 miss 都会 pin，只是 miss 还要先 load |
| Warmup | 在正常业务请求访问前主动触发 `PinCells`/`PinAllCells` | warmup 描述“何时主动加载”；mmap 描述“最终数据放在哪里”，两者正交 |
| Lazy load | load segment 时只注册可访问对象，在第一次真正访问时 materialize cell | collection/segment 可以显示 loaded，但第一条请求仍可能承担 cold load |
| Ready bit / loaded state | segcore 已注册该 field/index 的一致访问路径，QueryCoord 已观察到 segment 可路由 | 对 lazy column 而言表示“可按需取得”，不一定表示 cell bytes 已在 memory/disk cache |
| Cold bytes | 本次访问中需要执行加载的 cache-cell 持久化字节估计 | 不严格等于对象存储实际传输字节，也不包含所有 index 内部 lazy I/O |
| Reopen | segment 已存在时，根据新 `SegmentLoadInfo` 增量替换 index、manifest 或 field state | 不是 release 后 full reload；当前实现通过 `LoadDiff` 和 COW 发布新状态 |
| tSafe | Delegator 已完整消费并应用该 channel 数据的时间边界 | 请求的 guarantee timestamp 超过 tSafe 时必须等待，否则会漏掉已承诺可见的写入 |
| MVCC timestamp | segcore 实际用于 timestamp/delete visibility mask 的查询时间点 | guarantee timestamp 用于等待数据追上；MVCC timestamp 用于执行时判断可见性 |
| Historical / Streaming scope | sealed segment 子任务 / growing segment 子任务 | 这是 QueryNode 内部数据范围，不是用户 API 的 stream/non-stream 模式 |

### 2.2 流程背后的设计约束

后续步骤主要由以下约束决定：

1. **一致性先于执行**：Delegator 必须先等待 tSafe，并固定一份 readable distribution snapshot，避免查询一半时 segment 路由或数据可见边界变化。
2. **期望状态与实际状态分离**：QueryCoord 使用 Target/Distribution 收敛模型，才能统一处理首次加载、balance、compaction、node failure 和 reopen。
3. **可服务不等于全量 resident**：segment readiness 表示执行所需对象已经注册且一致性条件满足；lazy field/index 可以在请求时 materialize，以缩短 load blocking 并控制资源占用。
4. **资源必须在 I/O 前预留**：cache miss 可能同时产生最终内存/磁盘占用和解码临时峰值，所以 `RunLoad` 在读对象存储前做 reservation，而不是下载后再检查。
5. **只读取结果真正需要的宽字段**：Search late materialization 和 Query 两阶段 retrieve 都先缩小 row offsets，再读取 output fields，避免每个 segment 提前展开大量最终会被 reduce 丢弃的数据。
6. **在线读不能看到半更新 segment**：LoadDiff/Reopen 先在 staged runtime 中完成昂贵 I/O，再通过 copy-on-write 发布，避免查询线程观察到 index 已替换但 raw field 尚未恢复等中间状态。

## 3. Collection / Segment 在查询前如何进入 QueryNode

### 3.1 LoadCollection 的控制面流程

当前主流程及其设计原因如下：

| 步骤 | 做什么 | 为什么这样设计 | 对数据读取的影响 |
|---|---|---|---|
| 1 | Proxy 将 `LoadCollection` 放入 DDL task queue | collection load 会改变集群级元数据，需要与同 collection 的 load/release/alter 操作有序执行 | 只排队，不读取 field/index bytes |
| 2 | QueryCoord 广播 load config control message | load intent 需要持久、可重放且幂等；节点恢复后仍能从同一控制记录重建期望状态 | 只传播配置，不执行 segment I/O |
| 3 | load job 建立 collection、partition、replica metadata | 先明确副本数、resource group 和加载字段，后续 checker 才知道每份数据应出现在哪些节点 | 形成 Target 的约束条件 |
| 4 | `TargetObserver` 从 DataCoord 获取 segment/binlog/index/manifest metadata | QueryCoord 不保存数据文件真相；DataCoord 是持久化 segment 布局和 index 完成状态的来源 | 读取 metadata/path/size，不读取文件主体 |
| 5 | checker 比较 Target 与 Distribution，生成 grow/load/reopen/reduce action | 同一套差异收敛机制可覆盖首次 load、balance、compaction、index build 完成和节点故障恢复 | 决定哪些 worker 需要真正加载或释放 |
| 6 | executor 构造 `SegmentLoadInfo` 和 `LoadSegmentsRequest` | worker 必须拿到自包含的 schema、文件路径、index 参数、warmup/mmap 策略和版本，避免执行时再拼接不一致信息 | 固化本次加载的数据源和策略 |
| 7 | Delegator 将请求转给目标 worker，worker 开始加载 | Delegator 掌握 shard 路由和 delete 补偿状态；实际 bytes 应由最终持有 segment 的 worker 读取 | 从此处开始可能发生对象存储 I/O |

相关代码：

- [`internal/proxy/impl.go`](../../../internal/proxy/impl.go) 的 `Proxy.LoadCollection`。
- [`internal/querycoordv2/services.go`](../../../internal/querycoordv2/services.go) 的 `Server.LoadCollection`。
- [`internal/querycoordv2/job/job_load.go`](../../../internal/querycoordv2/job/job_load.go) 的 `LoadCollectionJob.Execute`。
- [`internal/querycoordv2/task/executor.go`](../../../internal/querycoordv2/task/executor.go) 的 `getLoadInfo`。
- [`internal/querycoordv2/task/utils.go`](../../../internal/querycoordv2/task/utils.go) 的 `packLoadSegmentRequest`。

这一阶段主要处理元数据和调度。真正的 index/field bytes 只在 QueryNode worker 的 segment loader 和 segcore 内开始读取。

### 3.2 QueryNode `LoadSegments` 流程

[`internal/querynodev2/services.go`](../../../internal/querynodev2/services.go) 的 `QueryNode.LoadSegments` 执行：

| 步骤 | 做什么 | 为什么需要 | 数据/cache 影响 |
|---|---|---|---|
| 1 | 检查 QueryNode 健康状态和 index metadata | 避免把新资源提交给退出中的节点，也避免创建缺少 collection index contract 的不可查询 segment | 尚未读数据 |
| 2 | 缺失 `MemorySize` 时以 `LogSize` 回填 | 旧版本 metadata 可能没有解压后大小；资源估算必须至少有保守输入，不能完全跳过保护 | 只影响 reservation 估算 |
| 3 | `NeedTransfer=true` 时交给 Delegator 转发 | QueryCoord 可以只面向 shard leader 发请求，由 leader 根据 shard ownership 和 L0 规则修正目标 worker | 目标 worker 确定后才读数据 |
| 4 | worker 注册/引用 collection 和 schema | segment 的字段解释、plan 校验、mmap/warmup 决策都依赖同一 schema snapshot | 建立执行上下文，不 materialize field |
| 5 | 按 `LoadScope` 选择 full/delta/index/stats/reopen | index build、compaction、schema/manifest 更新不应每次触发 full reload；差异化 scope 降低 I/O 和服务抖动 | 只加载发生变化的资源 |
| 6 | full load 进入 `segmentLoader.Load` | Go 层统一负责重复任务过滤、资源保护、并发控制和 segment manager 生命周期 | 开始准备实际 load |

[`internal/querynodev2/segments/segment_loader.go`](../../../internal/querynodev2/segments/segment_loader.go) 的 `Load`：

| 步骤 | 做什么 | 为什么需要 | 数据/cache 影响 |
|---|---|---|---|
| 1 | 过滤已加载或正在加载的 segment | checker 重试和并发 action 可能重复到达；重复构建相同 segment 会浪费资源并破坏版本判断 | 复用正在进行的 load 结果 |
| 2 | 估算并预留内存、磁盘、GPU 和临时加载资源 | index/Arrow 解码的峰值可能高于最终 resident size；必须在 I/O 前限流，避免多个大 segment 同时导致 OOM | reservation 失败时不开始下载 |
| 3 | `NewSegment` 创建 Go/C++ segment，并把 `SegmentLoadInfo` 交给 C++ | segment 自己需要持有完整数据源描述，才能统一执行首次 Load 和后续 Reopen diff | 先创建 metadata-only segment |
| 4 | 按资源允许的 concurrency 并行 `LoadSegment` | segment 之间没有数据依赖，可并行隐藏对象存储和解码延迟；并发又必须受资源估算约束 | 多 segment 可并发冷读 |
| 5 | sealed segment 进入 `ChunkedSegmentSealedImpl::Load` | 当前设计把 field/index 的实际编排收敛到 segcore，使首次 load 和 reopen 使用同一状态模型 | 创建并按 warmup 策略 materialize cache slots |
| 6 | Go 层加载 delta log 和 PK bloom filter | delete visibility 和 PK routing 是 shard 一致性的一部分，不能仅依赖 lazy user field；这些结构在 segment 对外可读前必须就绪 | 读取 delta/stats 辅助数据 |
| 7 | segment 放入 manager 并上报 Distribution | QueryCoord 只能依据 observed state 判断 Target 是否已满足；RPC 返回本身不足以证明 leader 已能路由到该 segment | 从此 segment 才进入稳定查询路由 |

### 3.3 Segcore 当前的 `LoadDiff` 自管理加载

当前 sealed segment 的实际加载中心是：

```text
ChunkedSegmentSealedImpl::Load
  -> SegmentLoadInfo::GetLoadDiff
  -> ApplyLoadDiff
       1. indexes_to_load
       2. column_groups_to_load / column_groups_to_lazyload
       3. binlogs_to_load
       4. text indexes / JSON stats
       5. default fields
       6. publish immutable/COW runtime state
```

关键代码：

- [`internal/core/src/segcore/SegmentLoadInfo.cpp`](../../../internal/core/src/segcore/SegmentLoadInfo.cpp) 的 `GetLoadDiff` / `ComputeDiff`。
- [`internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp`](../../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp) 的 `ApplyLoadDiff`、`LoadBatchIndexes`、`LoadBatchFieldData`、`LoadColumnGroups`、`Load`。
- 设计背景：[`segcore/20260204-loaddiff_based_segment_load.md`](segcore/20260204-loaddiff_based_segment_load.md)。

`Reopen` 使用同一套 diff 机制，只增量加载、替换或删除发生变化的 index/field/manifest，而不是总是释放并重载整个 segment。

采用 `LoadDiff + staged commit` 主要解决两个问题：

- **避免无变化数据的重复 I/O**：例如 compaction 只改变 manifest，或新 index 刚构建完成时，不需要重读全部 fields。
- **保持读路径原子性**：替换 index/field 时先把新资源完整加载到 staged runtime，再发布新 snapshot；不能先 drop 旧资源，让并发查询观察到半更新状态。

执行顺序也因此是“先准备替代资源，再删除旧资源，最后统一发布 ready bitset、runtime 和 load info”。`check_search` 看到的可用标记必须与实际可访问对象属于同一版本。

## 4. 数据源、cell 划分和实际冷读位置

### 4.1 统一触发点：`CacheSlot::Pin*`

字段和索引的 cache object 都通过 caching layer 管理。核心逻辑位于：

- [`internal/core/output/include/cachinglayer/CacheSlot.h`](../../../internal/core/output/include/cachinglayer/CacheSlot.h)
- [`internal/core/output/include/cachinglayer/Manager.h`](../../../internal/core/output/include/cachinglayer/Manager.h)
- [`internal/core/output/include/cachinglayer/Translator.h`](../../../internal/core/output/include/cachinglayer/Translator.h)

请求访问数据时会调用：

- `PinCells`：只 pin 指定 cell；适合按 chunk/offset 访问。
- `PinAllCells`：需要整列或整个对象时 pin 全部 cell。
- `PinOneCellDirect`：同步单 cell 快路径。

内部判断：

```text
cell 已 LOADED
  -> cache hit
  -> 增加 pin count
  -> 直接返回 CellAccessor

cell 未 LOADED
  -> cache miss
  -> 资源预留/必要时 eviction
  -> translator.get_cells(cids)
  -> 从持久化存储读取并解码
  -> 写入 memory 或本地 mmap file
  -> 标记 LOADED 并唤醒等待者
```

同一个 cell 的并发 miss 不会重复产生多个独立的最终加载结果：第一个请求建立 loading future，其他请求等待同一结果。这样可避免热点冷启动时出现“并发请求数 × cell 大小”的下载和解码放大。cell 在 `CellAccessor` 生命周期内被 pin，不能被 eviction，因为执行算子取得的 span/指针必须在本次计算结束前保持有效。

### 4.2 Storage V1：field binlog

[`internal/core/src/segcore/storagev1translator/ChunkTranslator.cpp`](../../../internal/core/src/segcore/storagev1translator/ChunkTranslator.cpp) 的行为：

- 一个 field 的每个 binlog 文件对应一个 cache cell。
- miss 时调用 `LoadArrowReaderFromRemote`。
- `RemoteChunkManager` 从 MinIO/S3/其他对象存储读取对象并解码为 Arrow reader。
- 非 mmap：转换成 `Chunk` 后保存在内存。
- mmap：先解码，再把 chunk 写入 QueryNode local storage 下的 mmap 文件并映射。

因此 V1 lazy load 的最小粒度通常是一个 field binlog 文件，而不是一行。

选择“一个 binlog 一个 cell”是因为 V1 metadata 已经按文件记录 row count、size 和 path，文件是天然的独立重载单元。再切得更细需要额外 row-group/offset metadata，而合并多个文件又会放大低选择性访问的冷读量。

### 4.3 Storage V2：column group / packed files

[`internal/core/src/segcore/storagev2translator/GroupChunkTranslator.cpp`](../../../internal/core/src/segcore/storagev2translator/GroupChunkTranslator.cpp) 的行为：

- 一个 column group 中包含多个 field。
- Parquet row group 会按照 `queryNode.segcore.storageV2.cellTargetSizeBytes` 合并为 cache cell，默认目标约 4 MiB。
- miss 时批量读对应 row groups，再为 column group 中的各 field 构造 `GroupChunk`。
- 若多个 field 共享同一个 eager group，访问其中一个 field 可能把该 cell 内其他 field 一起加载。

约 4 MiB 的目标 cell 是 I/O 批量效率与 cache 精度之间的折中：cell 太小会增加 object-store range/read 和调度开销，太大则会让一次少量 offsets 访问加载过多无关行。

### 4.4 Storage V3：manifest + projected column group

[`internal/core/src/segcore/storagev2translator/ManifestGroupTranslator.cpp`](../../../internal/core/src/segcore/storagev2translator/ManifestGroupTranslator.cpp) 的行为：

- segment load 先解析 manifest/column-group metadata，并创建 reader/chunk reader。
- eager field 可以共享 column-group reader 和 cache entry。
- lazy field 被 `ComputeDiffColumnGroups` 拆成“一条 lazy entry 对应一个 field”。
- lazy entry 使用单字段 projection 和带 field-id suffix 的 cache key，因此首次触碰一个 lazy field，不会自动把同组所有 lazy sibling field 一起拉入 cache。
- cell 仍按 row groups 聚合，miss 时由 chunk reader 从对象存储读取指定 row-group 范围。

拆分 lazy field 的原因是：Storage V3 的物理 column group 可以包含多个业务字段，但查询可能长期只访问其中一列。若 lazy entry 仍共享整组 projection，访问一个字段会额外加载 sibling fields。当前方案保留 row-group 批量 I/O，同时把列投影缩小到单字段。

### 4.5 Index 数据

普通 scalar/vector index 由 [`SealedIndexTranslator.cpp`](../../../internal/core/src/segcore/storagev1translator/SealedIndexTranslator.cpp) 管理：

- segment/field index 通常表现为一个 cache cell，cell id 固定为 0。
- `LoadIndexData` 先创建 translator 和 `CacheSlot`，真正 `index->Load` 在该 slot warmup 或首次 `PinCells({0})` 时发生。
- index 文件由 FileManager / default Arrow filesystem / RemoteChunkManager 读取。
- mmap index 会写入/使用 local index mmap path；非 mmap index 主要驻留内存。

普通 index 使用单 cell，是因为 index 查询依赖完整的内部结构和 metadata，不能像原始列一样简单按 row range 独立计算。支持内部 lazy load 的 index 会在 Knowhere 内再次细分其 payload，因此不要求外层 CacheSlot理解 index 私有布局。

特殊情况：若 Knowhere index type 支持内部 `LAZY_LOAD`：

- 外层 cache slot 会同步加载 index metadata，以保证 index object 可用。
- index payload 的按需加载由 Knowhere/index 内部继续管理。
- 因此即使 `vectorIndex=sync`，第一次向量搜索仍可能出现 index 内部冷读；外层 slot 的 `scanned_remote_bytes` 也可能无法完整代表该内部 I/O。

### 4.6 Delta log、Bloom filter 和其他辅助数据

这些数据不完全遵循 field/index warmup 规则：

- delta log：Go segment loader 在 segment load/reopen 流程中读取并应用，保证 MVCC delete 可见性。
- PK bloom filter/stats：Go loader 从 stats log 读取，放入 PK oracle；Delegator 用它做 delete forwarding 和 PK predicate segment filter。
- BM25/IDF：由 Delegator/IDF oracle 管理，`queryNode.idfOracle.preload=true` 时会在 load/首次 target 前预处理；关闭时可延迟到分布同步。
- expression result cache：缓存的是过滤结果 bitmap，不是源字段或索引，详见第 9 节。

## 5. Warmup 策略

### 5.1 策略优先级

field data 的有效策略：

```text
field type_params 中 warmup
  > QueryCoord 从 collection warmup.scalarField/vectorField 下推的 warmup
  > autoWarmupForNonPKIsolationCollection
  > QueryNode 全局 tieredStorage.warmup 配置
```

index 的有效策略：

```text
index params 中 warmup
  > QueryCoord 从 collection warmup.scalarIndex/vectorIndex 下推的 warmup
  > autoWarmupForNonPKIsolationCollection
  > QueryNode 全局 tieredStorage.warmup 配置
```

QueryCoord 下推逻辑位于 [`internal/querycoordv2/task/utils.go`](../../../internal/querycoordv2/task/utils.go) 的 `applyCollectionWarmupSetting` 和 `applyIndexWarmupSetting`；QueryNode 解析逻辑位于 [`internal/querynodev2/segments/utils.go`](../../../internal/querynodev2/segments/utils.go)。

之所以由 QueryCoord 把 collection 级策略下推到 field/index，而不是让每次请求在 QueryNode 临时查 collection properties，是为了让一份 `LoadSegmentsRequest` 自包含并可重放。这样 balance、recovery 和 reopen 使用同一份明确策略，不会因节点本地默认值不同而产生副本间加载差异。

若 `queryCoord.autoWarmupForNonPKIsolationCollection=true` 且 collection 未启用 partition-key isolation：

- scalar field 强制 `sync`；
- scalar index 和 vector index 强制 `sync`；
- vector raw field 不会被该开关强制预热。

### 5.2 默认策略矩阵

默认值来自 [`configs/milvus.yaml`](../../../configs/milvus.yaml)：

| 内容 | 默认 warmup | 默认 mmap | 默认 load 行为 |
|---|---:|---:|---|
| Scalar raw field | `sync` | `false` | load segment 时读入内存 |
| Scalar index / system PK index | `sync` | `false` | load segment 时建立 index |
| Vector raw field | `disable` | `true` | 首次实际访问时下载并建立 local mmap cell |
| Vector index | `sync` | `false` | load segment 时建立 index；内部 lazy index 除外 |
| JSON shredding stats | 随 scalar data/index 路径 | `true` | 取决于 stats/index 类型 |
| Growing raw data | 不使用 sealed warmup | `growingMmapEnabled=false` | 写入时进入 resident InsertRecord |

### 5.3 `sync` 的详细过程

`Manager.CreateCacheSlot` 创建 slot 后立即调用 `CacheSlot.Warmup`：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | 枚举 slot 的所有 cells | sync 的 contract 是 segment ready 前相关内容已可直接访问，不能只预热首个 chunk |
| 2 | 调用 `PinCellsDirect` | 复用正常请求完全相同的 cache/load 状态机，避免 warmup 和 query 形成两套加载实现 |
| 3 | miss cells 进入 `RunLoad` | 已加载 cell 可直接复用；reopen 或共享对象场景不应重复读取 |
| 4 | 预留最终资源和临时 loading overhead | 解码、index build、mmap 写入会产生峰值资源，必须在 I/O 前拒绝无法安全完成的加载 |
| 5 | translator 读取并生成 memory/mmap cells | translator 隔离 V1/V2/V3/index 的物理差异，CacheSlot 只管理生命周期和资源 |
| 6 | 全部完成后 warmup 返回 | QueryCoord 收到 loaded distribution 后可合理预期第一条请求不会承担外层 cache cold load |

所以 `sync` 会增加 collection load 时间，但通常降低第一条请求延迟。

### 5.4 `async` 的详细过程

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | `CreateCacheSlot` 提交后台 prefetch task | 把 segment readiness 与大对象预热解耦，缩短 load blocking 时间 |
| 2 | segment load 不等待全部 cells | 允许 collection 更早进入服务状态，适合有明确预热窗口的部署 |
| 3 | 后台仍通过 `PinCellsDirect` 加载 | 与请求共享同一状态机、资源保护和去重机制 |
| 4 | 请求先到时等待相同 cell future | 保证只有一份加载工作，同时请求不会读到未完成对象 |
| 5 | release/reopen/drop 取消 warmup | 防止已失效 segment 继续占用网络、CPU 和本地磁盘 |

async prefetch pool 的线程数由 CPU 数和 low-priority thread coefficient 初始化。若没有可用 prefetch pool，C++ 实现会回退为同步 warmup。

### 5.5 `disable` / lazy load 的详细过程

`disable` 并不表示该 field 永远不可用，而是：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | load 时建立 translator、cell metadata 和 ready state | 计划执行前必须知道字段可从哪里取得，但无需立即支付全部 I/O 和 resident 成本 |
| 2 | 不主动 materialize cells | 对低频字段、宽向量字段和只加载不用的 collection，避免无收益预热 |
| 3 | 第一次实际算子访问时调用 `PinCells` | 由真实访问的 field、chunk 或 offsets 决定加载范围，而不是由 schema 全量决定 |
| 4 | 请求等待下载、解码和 memory/mmap 建立 | 算子必须在 cell 完整可用后执行；lazy load 转移的是时机，不降低一致性要求 |
| 5 | cell 标记 loaded，后续请求命中 | 将首次访问成本摊销到后续请求；若 eviction 开启，之后仍可能重新变冷 |

如果访问模式只覆盖少量 row offsets，lazy load 可以只加载相关 cell；如果执行整列 predicate scan 或 brute-force vector search，则第一次请求仍可能加载该字段全部 cell。

## 6. Search 路径：逐阶段说明

### 6.1 Proxy：校验、计划和路由

`searchTask.PreExecute`：

| 步骤 | 动作 | 为什么在 Proxy 完成 |
|---|---|---|
| 1 | 从 MetaCache 取 schema、collection 和 partition info | 字段名到 field ID、partition name 到 ID 的转换应只做一次，避免每个 shard 重复解析 |
| 2 | 校验 output fields、NQ、TopK、params 和 partition key | 输入错误与具体 replica 无关，应在 fan-out 前失败，避免污染 QueryNode blacklist/retry |
| 3 | 把 DSL/expr 编译成 protobuf plan | 所有 QueryNode 必须执行同一份已类型检查的逻辑计划，防止各节点重复 parser 工作或产生版本差异 |
| 4 | 计算 guarantee/MVCC/TTL | 把用户 consistency 语义转换成内部时间边界，Delegator 和 segcore 才能分别执行等待与可见性过滤 |

`searchTask.Execute`：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | 从 shard leader cache 取得每个 vchannel 的 Delegator 候选 | collection 数据按 channel 独立维护一致性和 segment 分布，必须逐 shard 查询 |
| 2 | 各 channel 并行执行 `searchShard` | shards 之间无执行依赖，串行会把总延迟变成各 shard 延迟之和 |
| 3 | 优先 serviceable replica，失败时刷新并重试 | replica 提供读容错；但输入错误不重试，因为换 replica 不会改变结果 |

这一阶段没有 user data cold read。

### 6.2 Delegator：一致性、segment 选择和分发

`shardDelegator.Search`：

| 步骤 | 动作 | 为什么需要 |
|---|---|---|
| 1 | 校验 channel ownership | Proxy leader cache 可能短暂过期；错误 leader 必须显式返回可重路由错误，不能查询错误 shard |
| 2 | 等待 `tSafe >= guaranteeTimestamp` | guarantee timestamp 承诺此前写入已可见；未追上就执行会产生缺行而不是单纯旧读 |
| 3 | pin readable distribution snapshot | balance/release 可并发发生；一次请求必须基于稳定的 segment->worker 版本执行 |
| 4 | partition stats prune | 在不触碰 segment field/index 的前提下排除统计范围不相交的 segments，降低 fan-out |
| 5 | PK min/max + bloom filter prune | PK predicate 具有便宜且安全的“不可能命中”判断，可避免大量无效 worker 和 cold-cache 访问 |
| 6 | 可选 two-stage search | predicate 选择率会影响最合适的 ANN 参数；先取得 valid count 可避免统一按最坏情况搜索 |
| 7 | sealed 按 worker 分组，growing 留本机 | sealed 可在 replica 内分布；growing 由该 shard Delegator 消费维护，放本机保证最新状态 |
| 8 | 并发执行子任务 | worker/segment 之间独立，最终由 reduce 合并；并行降低 shard 内总延迟 |

Delegator 只读 distribution、partition stats、PK oracle、delete buffer 等已在内存中的辅助结构；正常 Search 不在 Delegator 层下载 field/index 文件。

### 6.3 Worker：scheduler 与 segment pin

worker `SearchSegments` 创建 `tasks.SearchTask` 并进入 read scheduler。

`SearchTask.Execute`：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | 按 scope 选择 Historical/Streaming segment 集合 | sealed 和 growing 的存储、index 与生命周期不同，不能用同一 segment selector 混查 |
| 2 | manager 校验并 pin segment | distribution snapshot 固定了路由，但本机 release 仍可能并发；pin 保护 segment C++ pointer 生命周期 |
| 3 | segment 间并行 Search | 每个 segment 产生独立候选集，天然适合并行后 TopK reduce |
| 4 | 进入 segcore plan visitor | predicate、MVCC、delete mask 和 vector operator 必须在同一 segment snapshot 内组合执行 |

### 6.4 Segcore：过滤阶段会加载什么

predicate 的数据访问取决于表达式和可用索引：

| predicate 路径 | cache 命中时 | cache miss 时 |
|---|---|---|
| Scalar index | pin index cell，直接计算 bitmap | 下载/加载整个 scalar index cell 后计算 |
| Raw scalar field scan | pin所需 column chunks | 下载对应 chunks；全 segment scan 常会覆盖全部 chunks |
| Text match index | pin text index slot | 加载 text index 文件/本地 mmap 后计算 |
| JSON stats/index | 访问 JSON index/stats | 可能加载 index/stats 或 raw JSON chunks |
| Expr result cache hit | 直接复用完整 filter bitmap | 不访问源字段；miss 后才执行上述路径 |

MVCC timestamp mask 和 delete mask 会在 filter/vector search 前后合并，保证只返回指定时间点可见且未删除的数据。

过滤优先生成 bitset 的原因是：ANN 或 brute-force 只需要在有效行集合中搜索。把 predicate、MVCC、TTL 和 delete 都统一成 mask，可以让不同 index type 共享相同的可见性约束，而不必把 Milvus 的事务语义分别实现到每一种向量索引中。

### 6.5 Segcore：向量阶段会加载什么

`ChunkedSegmentSealedImpl::vector_search` 有三条主路径：

1. **binlog/interim index**：访问 sealed segment 的临时 index。
2. **正式 vector index**：`SearchOnSealedIndex` 调用 index slot `PinCells({0})`；miss 时加载 index，hit 时直接查询。
3. **无 index brute-force**：`SearchOnSealedColumn` 访问 raw vector column；各 chunk 首次被访问时触发 cold load。

选择顺序是“正式/临时 index 优先，raw field fallback”。原因是 index 提供主要性能能力，而 raw field 同时也是兼容路径：小 segment、index 尚未完成、index 被替换或某些特殊 metric 下仍必须可查询。

正式 index 通常只需要 index，不需要同时加载 raw vector field。但以下情况可能触碰 raw vector：

- 无正式/临时 index；
- refine 需要 raw data 且 index 本身不带 raw data；
- 用户把 vector field 放入 output fields；
- nullable vector 的 validity/raw fallback；
- 某些 index 或功能链需要原始向量。

### 6.6 Search 结果字段的 late materialization

segment vector search 先产生 `seg_offsets + distances`。随后 Search export/reduce 流程才填充 PK、group-by 和 output fields：

1. 先得到每个 segment 的候选 TopK offsets。
2. `FillPrimaryKeys` / `FillTargetEntry` 按 offsets 执行 `bulk_subscript`。
3. raw column 根据 offsets 映射到 cell，只 pin 包含这些 offsets 的 cells。
4. 如果 vector index 自带 raw data，可直接从 index 取向量，避免 raw vector column cold read。
5. worker Go reduce 完成 segment 归并和字段导出。

因此 Search 的 output field cold read 通常与最终候选数有关，而不是与 segment 总行数线性相关；但 predicate scan 和 brute-force vector search 仍可能扫描大范围数据。

late materialization 不能提前到 ANN 前做，因为 ANN 最终只保留很少的 offsets。若每个 segment 在 reduce 前就展开全部候选的宽字段，网络和反序列化成本会随 `segment 数 × per-segment TopK × 字段宽度` 放大。

### 6.7 多级 reduce

1. worker：segment 级 reduce，生成 worker result。
2. Delegator：同一 channel 多 worker 结果 reduce。
3. Proxy：跨 channel/shard reduce，并处理 rerank、group-by、output field、格式转换等。

reduce 阶段通常处理已经返回的 result bytes，不再读取原 segment 数据；需要 requery 的特殊 Search 流程会再发起 Query 请求拉取字段。

分层 reduce 与数据分布边界一致：worker 最接近 segment，可先压缩局部候选；Delegator 汇总一个 shard；Proxy 才拥有完整 collection 视图。这样避免所有 segment 原始候选直接跨网络汇聚到 Proxy。

## 7. Query / Get 路径：逐阶段说明

### 7.1 Proxy

`queryTask.PreExecute`：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | 获取 schema/collection info | Query 的字段投影、类型和主键排序都依赖稳定 schema；应在 fan-out 前统一解析 |
| 2 | 解析 expr，构造 retrieve plan | worker 只负责执行 plan，不重复处理用户字符串表达式 |
| 3 | 解析 limit/iterator/group/order/aggregation | 这些参数会改变 segment pipeline 和 reduce 策略，必须在请求下发前固定 |
| 4 | 将 Get IDs 转成 `pk in [...]` | 复用 Query 的 predicate、segment filter、MVCC 和 output pipeline，避免维护独立 Get 执行器 |
| 5 | 计算 partition 和时间边界 | 缩小 shard/segment 范围，并保证与 Search 相同的一致性语义 |
| 6 | 序列化 plan，标记 PK filter | Delegator 可在不反复解析完整 plan 的情况下决定是否执行 PK segment pruning |

`queryTask.Execute` 与 Search 一样按 channel 选择 Delegator 并并行发 RPC。

### 7.2 Delegator

`shardDelegator.Query`：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | 等待 tSafe | Query/Get 返回具体实体和字段，不能在承诺时间点之前缺失数据 |
| 2 | pin readable distribution | 保证整个 Query 基于同一 segment 路由版本 |
| 3 | 使用 partition stats 和 PK bloom/min-max prune | Query 常见主键过滤，先排除 segment 可显著减少 field/index 访问 |
| 4 | sealed 按 worker 分组，growing 发本机 | 遵循 segment 实际 ownership 和 growing 最新状态来源 |
| 5 | 要求完整子任务成功 | Query 的 limit、aggregation、count 和字段集合依赖全局完整输入；缺一个 worker 会改变确定性结果，而不只是降低 recall |

### 7.3 Worker 与 segcore Retrieve

`QueryTask.Execute` 调用 `segments.Retrieve`：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | pin 目标 segments | 防止本机 release/reopen 破坏正在执行的 C++ segment 引用 |
| 2 | 每个 segment 执行 Retrieve | predicate 和 MVCC 依赖 segment 本地 index/data；远程集中执行会搬运大量原始数据 |
| 3 | 产生 offsets 或 pipeline columns | 普通 Query 适合先保留轻量 offsets；ORDER BY/aggregation 必须在 segment 内看到排序/聚合字段 |
| 4 | 按 plan 回填 output fields | 只 materialize API 真正需要的字段，并将 cold-load 范围限制到匹配 offsets 所在 cells |

数据访问模式：

- predicate field：与 Search filter 相同，可能走 scalar index、text/JSON index 或 raw field scan。
- output field：按匹配 offsets 读取相关 cells。
- `count(*)`：通常不需要加载用户 output fields，但 predicate 仍可能加载。
- ORDER BY：排序字段必须在 segment pipeline 中读取；若字段是 lazy，可能加载比普通 limit query 更多的数据。
- GROUP BY / aggregation：group/aggregate 输入字段必须参与 pipeline，通常需要扫描所有匹配行对应的数据。

### 7.4 有限 limit Query 的两阶段 retrieve

当满足条件时，`shouldEnableIgnoreNonPk` 开启两阶段读取：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | 第一阶段只返回 PK 和 segment offset | PK/offset 足以进行去重、排序和 limit 选择，宽字段此时还没有保留价值 |
| 2 | QueryNode 先跨 segment 归并 | limit 是 worker 内多个 segments 的整体约束，不能由每个 segment 独立决定最终行 |
| 3 | 只对最终 offsets 调用 `RetrieveByOffsets` | 把 output-field I/O 从“所有局部候选”缩小为“最终保留行” |
| 4 | 第二阶段读取对应 cache cells | 保持字段读取的 cell 粒度，并复用正常 lazy-load、mmap 和 cache 统计路径 |

相关代码：

- [`internal/querynodev2/segments/retrieve.go`](../../../internal/querynodev2/segments/retrieve.go) 的 `shouldEnableIgnoreNonPk`。
- [`internal/querynodev2/segments/query_pipeline.go`](../../../internal/querynodev2/segments/query_pipeline.go) 的 ignore-non-PK pipeline。
- [`internal/querynodev2/segments/ignore_non_pk_ops.go`](../../../internal/querynodev2/segments/ignore_non_pk_ops.go) 的 `RetrieveByOffsets` 调度。

这能显著减少“很多 segment 各自先返回大量宽字段，最终只保留少量行”的浪费。ORDER BY、GROUP BY 或 aggregation 需要在第一阶段读取参与计算的字段，因此不能总是使用该优化。

## 8. Growing segment 的差异

Growing segment 的主要数据来源是实时 DML pipeline，而不是 Search/Query 时从对象存储下载：

| 步骤 | 动作 | 设计原因 |
|---|---|---|
| 1 | Delegator 消费 WAL/DML | 每个 shard 需要单一有序消费者推进 tSafe，才能确定实时写入的完整边界 |
| 2 | insert 写入 segcore `InsertRecord` chunks | growing 查询必须立即看到已消费写入，不能等待 flush 后再从对象存储加载 |
| 3 | delete 进入 delete record/buffer | delete 与 insert 按 timestamp 参与 MVCC；延迟到查询时临时补读会破坏顺序保证 |
| 4 | growing 使用 streaming scope 在 Delegator 本机执行 | growing 状态由该 Delegator 维护，本机执行避免把高速变化的未封存数据复制到多个 workers |

特点：

- raw data 通常已 resident。
- 可为 growing chunks 建 interim vector index，默认 `queryNode.segcore.interimIndex.enableIndex=true`。
- `queryNode.mmap.growingMmapEnabled=false` 默认关闭；开启后 growing raw data 使用本地 mmap，减少 heap 压力但可能增加访问延迟。
- tiered field warmup 主要针对 sealed segment，不控制在线 insert 的到达时机。
- expression result cache 的 memory mode 可缓存 growing snapshot；disk mode 只支持 sealed segment。

## 9. 缓存分类：不要把几种 cache 混为一谈

| Cache/状态 | 位置 | 缓存内容 | miss 后动作 | 是否涉及用户数据 bytes |
|---|---|---|---|---|
| Proxy MetaCache | Proxy | schema、collection、partition、load 状态 | 向协调组件刷新 metadata | 否 |
| Shard leader cache | Proxy | channel -> Delegator replicas | 向 QueryCoord 刷新路由 | 否 |
| Delegator distribution | QueryNode leader | segment -> worker、版本、可读状态 | 等待/刷新 distribution | 否 |
| PK oracle / bloom filter | Delegator/QueryNode | segment PK 可能性 | stats load 时建立 | 少量 stats bytes，非字段主体 |
| Tiered `CacheSlot` | Segcore | field chunks、group chunks、index objects | translator 从持久化存储加载 | 是，主 cold-read 路径 |
| OS page cache | Linux | mmap 文件页 | local disk page fault | 是，但不等于对象存储读 |
| Knowhere index internal lazy cache | Vector index | index pages/metadata | index 内部读取 | 是，外层统计可能不完整 |
| Expression result cache | Segcore | predicate result/validity bitmap | 重算表达式 | 否，不缓存源字段 |
| Interim index | Segcore | growing/sealed unindexed vector 的临时索引 | 从 resident raw vector 构建 | 使用已有 raw data |

### 9.1 Expression result cache

配置默认关闭：

```yaml
queryNode:
  exprCache:
    enabled: false
    mode: disk
```

启用后以 `(segment_id, expression_signature, active_count)` 识别结果：

- hit：直接取得完整 filter bitmap，避免再次访问 scalar/text/JSON data/index。
- miss：正常计算表达式；满足执行耗时和频率准入后写入 cache。
- memory mode：支持 sealed 和 growing，可压缩 bitmap，Clock eviction。
- disk mode：每个 sealed segment 一个固定 slot 文件，只支持 sealed。
- segment release 时清理对应 expression cache。

它优化的是 CPU 和 predicate 数据访问，不缓存最终 Search/Query result，也不替代 field/index cache。

设计与实现入口：

- [`20260602-expression-result-cache.md`](20260602-expression-result-cache.md)
- [`internal/core/src/exec/expression/ExprCache.cpp`](../../../internal/core/src/exec/expression/ExprCache.cpp)
- [`internal/core/src/exec/operator/FilterBitsNode.cpp`](../../../internal/core/src/exec/operator/FilterBitsNode.cpp)

## 10. Mmap、Milvus cache 和 OS page cache 的关系

`mmap=true` 不是“完全不占内存”，也不是“直接把 S3 对象 mmap”。实际过程是：

```text
对象存储对象
  -> QueryNode 下载/读取
  -> 解码/转换
  -> 写 local storage mmap file
  -> mmap address space
  -> 查询访问时由 OS page cache/page fault 提供物理页
```

需要区分两个命中层次：

1. **Milvus CacheSlot hit**：cell object 和本地 mmap 文件已建立，不再执行 translator 的远端加载。
2. **OS page cache hit**：访问 mmap 地址时目标页已在内存；否则仍会发生 local disk page fault。

所以 CacheSlot hit 不保证零磁盘 I/O，但通常保证零对象存储重下载。

## 11. Eviction 和再次冷读

默认：

```yaml
queryNode.segcore.tieredStorage.evictionEnabled: false
queryNode.segcore.tieredStorage.backgroundEvictionEnabled: false
```

默认情况下，warmup 或首次请求加载的 cache cell 通常一直存在到 segment release/reopen/drop。

若开启 eviction：

1. cache manager 按内存/磁盘 high watermark 判断压力。
2. 只淘汰未被 pin 的、translator 声明可淘汰的 cell。
3. 淘汰到 low watermark 或满足资源申请。
4. background eviction 可按周期和 TTL 淘汰长期未访问 cell。
5. 后续 `PinCells` 再次 miss，重新从持久化存储加载。

注意：sync warmup 的 cell 也可被后续 eviction 淘汰；`sync` 只定义首次 load readiness，不承诺永久驻留。

## 12. 典型请求场景

### 场景 A：默认配置，有正式向量索引，Search 输出 scalar 字段

Collection load：

1. scalar fields sync warmup，读入内存。
2. scalar/vector indexes sync warmup。
3. raw vector field 不 warmup，只建立 lazy mmap slot。

第一条 Search：

1. scalar predicate 多数命中 scalar field/index cache。
2. vector search 多数命中 vector index cache。
3. PK/scalar output 多数命中已 warm scalar cells。
4. 若不输出 raw vector，通常不会触碰 raw vector mmap slot。

### 场景 B：默认配置，Search 输出原始 vector field

1. vector index 完成 ANN。
2. 对 TopK offsets 回填 vector output。
3. 若 index 含 raw data，直接从 index 取向量。
4. 否则 raw vector field 首次 `PinCells`，下载相关 cells 并创建 local mmap 文件。
5. 后续相同 cell 的输出请求命中 Milvus cache；物理页是否命中由 OS page cache 决定。

### 场景 C：无向量索引或使用 brute-force

1. predicate 生成 bitset。
2. `SearchOnSealedColumn` 遍历 raw vector chunks。
3. vector raw warmup 默认 disable，因此第一次查询可能逐 chunk 冷读，最终接近加载整列。
4. 第一条请求延迟和远端读取量显著增加；后续请求命中已加载 cells。

### 场景 D：所有 warmup 设为 `disable`

1. collection/segment 更快进入 loaded 状态。
2. 第一条访问每种 field/index 的请求承担初始化和 cold read。
3. 同一请求可能依次冷加载 predicate index、vector index、PK/output field。
4. 若请求并发到达相同 cell，一个请求负责 load，其他请求等待；不会每个请求都独立下载完整 cell。

### 场景 E：warmup 设为 `async`

1. segment load 返回后后台开始预取。
2. 若业务流量晚于预取完成，表现接近 sync。
3. 若流量早于预取完成，请求等待正在加载的 cell，尾延迟可能接近 cold load。
4. async 更适合希望缩短 load blocking 时间、又能预留一段无流量预热窗口的场景。

### 场景 F：Query 有 limit 且 output fields 很宽

1. predicate 在所有候选 segments 上执行。
2. 第一阶段只返回 PK/offset。
3. QueryNode 归并到最终 limit。
4. 第二阶段只对最终 offsets 读取宽 output fields。
5. 冷读量取决于最终 offsets 落入多少 cell，而不是所有匹配行的宽字段总量。

### 场景 G：Cache eviction 后访问

1. cell 曾被 warmup 或请求加载。
2. 内存/磁盘压力使 cell 被淘汰。
3. 下一条请求访问该 cell 时重新执行 translator。
4. memory cell 重新下载/解码；mmap cell 可能重新创建本地文件。
5. 这次请求再次计入 cold/miss bytes。

## 13. `scanned_remote_bytes`、cache hit ratio 与指标解释

### 13.1 代码语义

`CacheSlot::PinInternal` 在请求 cell 时：

- 所有访问 cell 的持久化大小累计到 `scanned_total_bytes`。
- 本次需要 load 的 miss cells 累计到 `scanned_cold_bytes`。

segcore 将 `scanned_cold_bytes` 投影为 `ScannedRemoteBytes`，并逐级通过 worker、Delegator、Proxy reduce 汇总。

Proxy 用以下公式生成 cache hit ratio：

```text
cache_hit_ratio = (scanned_total_bytes - scanned_remote_bytes)
                  / scanned_total_bytes
```

相关代码：

- [`internal/core/output/include/cachinglayer/CacheSlot.h`](../../../internal/core/output/include/cachinglayer/CacheSlot.h)
- [`internal/core/src/segcore/SegmentInterface.cpp`](../../../internal/core/src/segcore/SegmentInterface.cpp)
- [`internal/proxy/util.go`](../../../internal/proxy/util.go)

### 13.2 使用限制

这些值应理解为 cache-cell 逻辑访问指标，不是精确对象存储账单：

- 默认 `storageUsageTrackingEnabled=false`，关闭时请求级值可能为 0。
- `cells_storage_bytes` 可能来自 metadata 估算，不一定等于实际网络 response bytes。
- row-group 合并、压缩、解码和 read amplification 会造成差异。
- local mmap file 的后续 page fault 不算远端 cold bytes。
- Knowhere 内部 lazy index I/O 可能不完整进入外层 CacheSlot 统计。
- 同一 cold cell 加载可能服务多个并发请求，归因到单个请求时需谨慎解释。

### 13.3 Caching layer 原生指标

C++ caching layer 还提供按 data type 和 storage location 区分的指标：

- `internal_cache_cell_access_hit_bytes_total`
- `internal_cache_cell_access_miss_bytes_total`
- `internal_cache_loaded_bytes`
- `internal_cache_cell_loading_count`
- `internal_cache_load_latency_microseconds`

它们适合判断 scalar/vector field/index 的 cache 行为和 cold-load 延迟。

## 14. 配置影响速查

| 配置 | 默认 | 影响 |
|---|---:|---|
| `queryNode.segcore.tieredStorage.warmup.scalarField` | `sync` | scalar raw field load 时预热 |
| `queryNode.segcore.tieredStorage.warmup.scalarIndex` | `sync` | scalar/system PK index load 时预热 |
| `queryNode.segcore.tieredStorage.warmup.vectorField` | `disable` | raw vector 默认请求时加载 |
| `queryNode.segcore.tieredStorage.warmup.vectorIndex` | `sync` | vector index 默认 load 时预热 |
| `queryNode.mmap.vectorField` | `true` | raw vector cold load 后落 local mmap |
| `queryNode.mmap.vectorIndex` | `false` | vector index 默认驻内存 |
| `queryNode.mmap.scalarField` | `false` | scalar field 默认驻内存 |
| `queryNode.mmap.scalarIndex` | `false` | scalar index 默认驻内存 |
| `queryNode.segcore.tieredStorage.evictionEnabled` | `false` | 是否允许策略性 cache eviction |
| `queryNode.segcore.tieredStorage.backgroundEvictionEnabled` | `false` | 是否后台周期淘汰 |
| `queryNode.segcore.tieredStorage.cacheTtl` | `0` | 未访问 cell 的时间淘汰，0 关闭 |
| `queryNode.segcore.tieredStorage.storageUsageTrackingEnabled` | `false` | 是否生成请求级 scanned cold/total bytes |
| `queryNode.segcore.tieredStorage.loadingTimeoutMs` | `-1` | 请求 cold-load 资源等待超时，-1 不限 |
| `queryNode.segcore.tieredStorage.warmupLoadingTimeoutMs` | `0` | warmup 资源等待，0 表示无法立即预留就失败 |
| `queryNode.segcore.storageV2.cellTargetSizeBytes` | `4194304` | Storage V2/V3 cache cell 目标大小 |
| `queryNode.exprCache.enabled` | `false` | 是否缓存 predicate result bitmap |
| `queryNode.exprCache.mode` | `disk` | expression cache backend |
| `queryNode.segcore.interimIndex.enableIndex` | `true` | growing/未索引 sealed 的临时向量索引 |
| `queryNode.mmap.growingMmapEnabled` | `false` | growing raw data 是否使用 local mmap |

## 15. 实际排查一条慢 Search/Query 的建议顺序

1. 在 Proxy 确认耗时是在 route/retry、Delegator wait tSafe，还是 QueryNode execution。
2. 查看目标 channel 是否发生 replica failover 或 shard leader cache refresh。
3. 查看 Delegator prune 后实际访问了多少 sealed/growing segments。
4. 区分 predicate、vector search、result materialization 三个阶段。
5. 检查查询字段/索引的最终 warmup 策略，而不只看全局默认；collection/field/index params 可能覆盖默认值。
6. 检查是否存在正式 vector/scalar index，以及 index 是否带 raw data。
7. 查看 `internal_cache_cell_access_miss_bytes_total` 和 load latency，按 scalar/vector field/index 分类。
8. 若启用 storage usage tracking，查看 response extra info 中的 scanned remote/total bytes 和 cache hit ratio。
9. 对 mmap 数据同时看 local disk I/O/page fault；CacheSlot hit 不代表 OS page cache hit。
10. 检查 eviction 是否开启、是否刚发生 segment reopen/balance/reload。
11. 对 Query 检查是否启用了 ignore-non-PK 两阶段读取；ORDER BY/GROUP BY/aggregation 可能主动关闭或绕过该优化。
12. 对重复昂贵 predicate 检查 expression result cache 是否启用、是否通过频率与耗时准入。

## 16. 最终结论

Milvus 当前 Search/Query 的数据访问模型是“**segment 预注册 + cell 级按策略 materialize + 请求期间 pin + 可选 eviction**”。

判断某一份数据到底何时被读取，应按以下顺序回答：

1. 它是 metadata、field、index、stats、delta 还是 expression bitmap？
2. 它属于 growing 还是 sealed segment？
3. sealed 数据使用 V1 binlog、V2 column group 还是 V3 manifest？
4. 有效 warmup 是 sync、async 还是 disable？
5. 是否 mmap？mmap 只改变本地落点与页访问方式，不消除首次对象存储读取。
6. 当前请求访问的是 predicate、vector search，还是 result materialization？
7. 请求访问的是全部 chunks，还是最终 offsets 所在的少量 cells？
8. cell 是否仍在 Milvus cache；如果在，OS page 是否仍在内存？
9. index 是否还有自己的内部 lazy-load 层？

只有把这些层次分开，才能准确解释“collection 已 loaded，为什么第一条查询仍然会读对象存储”，以及“cache hit 为什么仍可能产生本地磁盘延迟”。

## 17. 主要源码索引

- Proxy Search：[`internal/proxy/task_search.go`](../../../internal/proxy/task_search.go)
- Proxy Query：[`internal/proxy/task_query.go`](../../../internal/proxy/task_query.go)
- Proxy shard LB：[`internal/proxy/shardclient/lb_policy.go`](../../../internal/proxy/shardclient/lb_policy.go)
- QueryNode RPC：[`internal/querynodev2/services.go`](../../../internal/querynodev2/services.go)
- QueryNode handler：[`internal/querynodev2/handlers.go`](../../../internal/querynodev2/handlers.go)
- Delegator：[`internal/querynodev2/delegator/delegator.go`](../../../internal/querynodev2/delegator/delegator.go)
- Search task：[`internal/querynodev2/tasks/search_task.go`](../../../internal/querynodev2/tasks/search_task.go)
- Query task：[`internal/querynodev2/tasks/query_task.go`](../../../internal/querynodev2/tasks/query_task.go)
- Segment loader：[`internal/querynodev2/segments/segment_loader.go`](../../../internal/querynodev2/segments/segment_loader.go)
- Segment wrapper：[`internal/querynodev2/segments/segment.go`](../../../internal/querynodev2/segments/segment.go)
- Query pipeline：[`internal/querynodev2/segments/query_pipeline.go`](../../../internal/querynodev2/segments/query_pipeline.go)
- Segcore load diff：[`internal/core/src/segcore/SegmentLoadInfo.cpp`](../../../internal/core/src/segcore/SegmentLoadInfo.cpp)
- Sealed segment：[`internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp`](../../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp)
- Cache slot：[`internal/core/output/include/cachinglayer/CacheSlot.h`](../../../internal/core/output/include/cachinglayer/CacheSlot.h)
- Storage V1 field translator：[`internal/core/src/segcore/storagev1translator/ChunkTranslator.cpp`](../../../internal/core/src/segcore/storagev1translator/ChunkTranslator.cpp)
- Storage V2 field translator：[`internal/core/src/segcore/storagev2translator/GroupChunkTranslator.cpp`](../../../internal/core/src/segcore/storagev2translator/GroupChunkTranslator.cpp)
- Storage V3 manifest translator：[`internal/core/src/segcore/storagev2translator/ManifestGroupTranslator.cpp`](../../../internal/core/src/segcore/storagev2translator/ManifestGroupTranslator.cpp)
- Index translator：[`internal/core/src/segcore/storagev1translator/SealedIndexTranslator.cpp`](../../../internal/core/src/segcore/storagev1translator/SealedIndexTranslator.cpp)
- Sealed vector search：[`internal/core/src/query/SearchOnSealed.cpp`](../../../internal/core/src/query/SearchOnSealed.cpp)
- Tiered storage 初始化：[`internal/util/initcore/init_core.go`](../../../internal/util/initcore/init_core.go)
- 默认配置：[`configs/milvus.yaml`](../../../configs/milvus.yaml)
