# Milvus 对象存储指标与 Storage Profile 实现说明

- **日期：** 2026-07-13
- **状态：** 已完成首期实现，默认关闭 Profile，聚合指标始终启用
- **对应设计：** [20260713-object-storage-profiling.md](20260713-object-storage-profiling.md)
- **运维与指标说明：** [20260713-object-storage-profiling-operations.md](20260713-object-storage-profiling-operations.md)
- **主要组件：** Proxy、QueryNode、DataNode、DataCoord、StreamingNode、Segcore、Storage、Metrics

## 1. 文档目的

本文不是设计提案的翻译，而是对当前代码实现的说明。它回答以下问题：

1. 最终采用了什么架构；
2. Profile、Prometheus 指标和业务归因如何协同；
3. Go、C++、FFI 和内部 RPC 之间如何传递数据；
4. Search、Query、Flush、Import、Compaction、Index、Load 等路径具体改造了什么；
5. 哪些能力已经完整实现，哪些只有部分覆盖，哪些被明确留到后续版本；
6. 当前实现经过了哪些单元测试和端到端验证。

本文以当前代码为准。若本文与原始设计中的计划项存在差异，以第 19 节“已实现、部分实现和未实现”为准。

## 2. 实现结果概览

本次实现形成了同一套语义模型下的两类输出：

1. **始终开启的低基数 Prometheus 指标**
   - 不依赖 `storage.profile.enabled`；
   - 记录 Milvus 可见的逻辑存储操作、耗时、字节、重试、错误、并发和缓存数据；
   - 只使用有限枚举作为 label，不使用请求 ID、任务 ID、对象路径、Bucket、Collection ID 等高基数信息。

2. **按请求或后台任务显式开启的 `storage_profile` 摘要**
   - 默认关闭；
   - 使用固定大小、可并发写入、可跨节点合并的数据结构；
   - 支持 count、success/failure/cancel/timeout、bytes、sum/min/max、固定桶分布、近似 p50/p95/p99、cache 和 coverage；
   - Search/Query 可以从多个 QueryNode 收集 contribution，并在 Proxy 合并；
   - 当前生产 `SummarySink` 是 noop，不写入外部系统，也不返回给公开 API。

当前实现不修改 `milvus-storage`。所有观测点都位于 Milvus 仓库自己拥有的 Go、C++ 或 FFI 调用边界。

## 3. 总体架构

```text
配置与策略
storage.profile.*
       │
       ▼
ProfileDecider ──► disabled / summary
       │
       ▼
Scope（request 或 task）
       │
       ├── Attribution：workload/component/phase/role/backend/ID
       └── Recorder：noop 或 active
                    │
业务 Context ──────┤
                    ▼
        Milvus-owned storage boundary
        ├── ChunkManager
        ├── streaming FileReader
        ├── Storage V2/V3 packed/FFI
        ├── Segcore Search/Query materialization
        └── Index/Analyze/Load/Compaction task boundary
                    │
          BeginOperation / Finish
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
Prometheus 聚合指标          Scope 内固定大小摘要
始终记录                     仅 Profile 开启时记录
                                 │
                                 ▼
                        Snapshot / Contribution
                                 │
                     Search/Query 跨节点合并
                                 │
                                 ▼
                         NoopSummarySink
```

架构的核心原则是：**聚合指标和按 Scope 的摘要共享同一个观测入口，但生命周期和存储成本彼此独立。**

`storageprofile.BeginOperation` 总会负责 Prometheus 指标；只有 Context 中绑定了 active recorder 时，操作才会同时写入请求/任务摘要。Profile 关闭时使用单例 noop recorder，不分配每个 Scope 的 histogram 数据。

## 4. 测量语义

### 4.1 观测的是 Milvus 逻辑操作

指标和 Profile 记录的是 Milvus 可见的逻辑存储调用，而不是云厂商 SDK 的精确 HTTP 请求数量。

例如，一次 `PackedReader.ReadNext` 可能在 `milvus-storage` 内部展开为多个 Range GET，也可能发生 SDK 内部重试。首期实现只记录 Milvus 边界上可见的一次逻辑操作。因此这些指标适合性能优化、工作负载比较和容量分析，但不适合云账单核对。

### 4.2 Access Layer

代码保留两个访问层枚举：

- `AccessLayerMilvus`：当前实际产生；
- `AccessLayerProvider`：为未来 `milvus-storage` 或 provider SDK 内部观测预留。

Provider access coverage 在首期实现中始终为 unavailable。

### 4.3 操作分类

当前公共操作枚举包括：

- `read`
- `range_read`
- `write`
- `stat`
- `list`
- `delete`
- `copy`
- `multipart_create`
- `multipart_write`
- `multipart_complete`
- `multipart_abort`

首期主要实际产生前七类。Multipart 枚举和数据结构已经保留，但还没有完整的 Milvus-owned 事件生产路径。

### 4.4 Outcome 与错误分类

Outcome 是有限集合：

- `success`
- `failure`
- `canceled`
- `timeout`

错误类别也是有限集合：

- `none`
- `not_found`
- `throttled`
- `permission_denied`
- `invalid_credentials`
- `bucket_not_found`
- `invalid_argument`
- `invalid_range`
- `entity_too_large`
- `unexpected_eof`
- `timeout`
- `canceled`
- `io_failed`
- `unknown`

Go 侧优先通过 `errors.Is` 和 typed `merr` 分类。`WithErrorCategory` 只附加观测类别，不改变原始 error 文本、`errors.Is` 行为、wire code 或重试策略。

Context 已取消或超时时，`Finish` 会优先把结果归入 `canceled` 或 `timeout`，避免把取消误记成普通 I/O failure。

一个重要的 E2E 后修正是：`ChunkManager.Exist` 在对象不存在时对调用者返回 `(false, nil)`，因此其逻辑结果应记录为 `success`，不能因为底层 Stat 看到了 not-found 就把公开 API 的成功语义改成 failure。

### 4.5 字节语义

Profile 区分：

- `BytesRequested`：边界上请求或准备写入的字节；
- `BytesCompleted`：实际成功读取、写入或复制的逻辑字节。

字节只使用当前调用已经知道的数据，例如 buffer 长度、Arrow record 大小、binlog/index 元数据或已有 storage usage。实现不会为了统计字节额外发起 Stat/HEAD。

若字节未知，操作 count 和 latency 仍然有效，同时 coverage 标记为 unavailable 或 partial，而不是填入伪造的零。

### 4.6 延迟与分布

每个操作记录：

- count；
- sum；
- exact min/max；
- 固定非累积 bucket；
- 可选 TTFB；
- 可选 size distribution。

Go 内部延迟桶共 20 个，覆盖 250 微秒到 300 秒及 `+Inf`；大小桶共 13 个，覆盖 1 KiB 到 4 GiB 及 `+Inf`。

Profile 的 p50/p95/p99 是按固定桶上界估算的。空分布返回 unavailable，不把“没有样本”解释成 0 延迟。

并行操作的累计存储耗时可能大于请求墙钟时间，这是预期语义；当前实现不声称能够计算精确的存储 critical path。

## 5. `internal/storageprofile` 公共实现

### 5.1 枚举与低基数约束

`internal/storageprofile/enums.go` 定义了所有 operation、outcome、error category、scope type、workload class、workload kind、subtype、phase、role、backend、coverage、profile level 和 cache 枚举。

所有进入 Prometheus label 的值都经过 `Bounded()` 或枚举 `String()` 转换。Component 也只允许有限集合，例如 `proxy`、`querynode`、`datanode`、`datacoord`、`streamingnode` 等；未知值统一降级为 `unknown`。

请求 ID、任务 ID、Trace ID、Collection ID 可以保存在 Profile 的 attribution 中，但不会进入 Prometheus label。

### 5.2 Attribution

`Attribution` 同时承载两类信息：

1. 只用于 Profile 的身份信息，例如 request/task/trace/collection/node；
2. 用于 Prometheus 的有限维度，例如 component、workload class/kind、storage role、backend kind。

`WithDefaultAttribution` 只补齐缺失字段，不覆盖上游已经设置的 attribution。这个行为使复用组件能够提供默认归因，同时允许父任务覆盖它。

例如 SyncTask 默认是 Flush，但 Import 创建的 SyncTask 已经带有 Import attribution，因此不会被内部的 Flush 默认值覆盖。

### 5.3 Recorder

Recorder 有两种实现：

- 全局单例 `noopRecorder`；
- 带互斥锁的 `activeRecorder`。

`activeRecorder` 使用固定数组记录 operation 和分布，不在热路径保留原始样本列表。单个 Profile 最多保留 64 个 `(operation, phase, storage_role)` breakdown 条目，超限后只增加 dropped 计数。

Operation recorder 使用：

- `atomic.Uint64` 累加 completed bytes；
- `atomic.Int64` 记录首次非零读取的 TTFB；
- `sync.Once` 保证同一操作只 Finish 一次。

这保证 EOF、Close、error 或并发结束路径不会重复计数。

### 5.4 Prometheus 与 Profile 的双写

`BeginOperation` 返回外层 `observedOperation`：

1. 创建时增加 `operations_inflight`；
2. `Finish` 时更新聚合 count、duration、size、bytes、retry、error；
3. 再把标准化后的结果传给 Scope 内部 recorder；
4. 最终减少 inflight gauge。

因此 aggregate metrics 不依赖 Profile 是否开启；Profile recorder 为 noop 时，只执行低基数指标路径。

### 5.5 Coverage

Profile 为以下信号分别记录 coverage：

- Go storage operation；
- C++ storage operation；
- storage bytes；
- streaming TTFB；
- tiered cache usage；
- cache wait；
- provider access。

状态包括 not applicable、instrumented、partial 和 unavailable。

合并时只要不同 contribution 的覆盖状态不一致，通常会保守地得到 partial。旧节点不返回 contribution 时，也会把现有 contribution 标记为 quantile incomplete 和 partial，而不是把遗漏解释成零操作。

### 5.6 序列化和合并

内部 contribution 当前使用 JSON envelope 编码，并通过 internal protobuf 的 `bytes storage_profile` 传递。

实现约束包括：

- `schema_version = 1`；
- `bucket_schema = 1`；
- 单个 contribution payload 最大 1 MiB；
- 只序列化非零 operation；
- contribution identity 用于去重；
- 已知相同 bucket schema 才按桶合并；
- bucket schema 未知时仍合并兼容的 count/sum/min/max/bytes，但把 quantile 标记为 incomplete。

Contribution identity 当前包含：

- cluster ID；
- node ID；
- scope ID；
- task attempt；
- execution ID。

`MergeContributions` 按完整 identity 去重，防止 RPC retry 或重复 delivery 重复计费。

## 6. 配置、策略与安全限制

### 6.1 配置项

`configs/milvus.yaml` 和 ParamTable 增加了：

```yaml
storage:
  profile:
    enabled: false
    level: summary
    request:
      allowExplicit: false
    task:
      enabled: false
      types: []
    cache:
      enabled: true
    maxActiveScopes: 1024
    maxProfiledRequestsPerSecond: 10
    maxProfiledTasks: 128
```

所有配置目前都是 refreshable。未知 task type 或非法 level 会在转换 PolicyConfig 时返回 typed internal error。

### 6.2 决策顺序

`ProfileDecider` 的实际决策顺序是：

1. 全局 `enabled`；
2. 预留的管理员 TTL rule；
3. 后台任务配置与 task type 白名单；
4. 请求是否显式提出 Profile；
5. `request.allowExplicit`；
6. 请求是否被授权；
7. active/rate limit；
8. effective level。

`detailed` 已定义但首期会被降级到 `summary`。当前 TTL rule provider 是 noop。

### 6.3 Scope 限流

实现有三类进程内限制：

- 全部 active profile scope 上限；
- 每秒新建 request profile 上限；
- active task profile 上限。

超过限制只会禁用本次 Profile 并增加 dropped/decision 指标，不会让用户请求或后台任务失败。

### 6.4 显式请求 Profile 的授权

内部预留 header 为：

```text
x-milvus-storage-profile: summary
```

当前它不是公开 SDK 能力。E2E 后的安全修正确保 Proxy 只有在 `contextutil.IsIntraClusterRequest(ctx)` 为真时才认为显式请求已授权。外部客户端即使自行注入同名 metadata，也不能直接开启诊断 Profile。

## 7. Prometheus 指标实现

### 7.1 操作指标

已实现并注册：

- `milvus_storage_operations_total`
- `milvus_storage_operation_duration_seconds`
- `milvus_storage_operation_size_bytes`
- `milvus_storage_bytes_total`
- `milvus_storage_retries_total`
- `milvus_storage_operation_errors_total`
- `milvus_storage_operations_inflight`

主要有限 label：

- component；
- workload_class；
- workload_kind；
- operation；
- outcome；
- storage_role；
- backend_kind。

`error_category` 只放在 failure-only counter 上，避免把错误分类乘到 duration 和 size histogram 的所有时间序列。

### 7.2 Cache 指标

已定义并注册：

- `milvus_storage_cache_lookups_total`
- `milvus_storage_cache_bytes_total`
- `milvus_storage_cache_wait_duration_seconds`
- `milvus_storage_cache_load_duration_seconds`

当前明确接入的生产事件主要是 QueryNode 已有 storage usage tracking 提供的：

- requested bytes；
- cold bytes；
- served bytes。

Cache wait/load/lookup 的 Recorder 和指标接口已经完成，但现有生产路径只在确实可见的位置产生事件；首期没有为所有缓存实现新增侵入式计时。

### 7.3 Profile 控制指标

已实现并注册：

- `milvus_storage_profile_decisions_total`
- `milvus_storage_profile_active_scopes`
- `milvus_storage_profile_dropped_summaries_total`
- `milvus_storage_profile_snapshot_duration_seconds`

### 7.4 旧指标兼容

原有 `PersistentData*`、filesystem 全局指标仍然注册，名称、单位和含义未改变。新指标不会覆盖或重定义旧 histogram。

新旧操作数不要求相等，因为新模型：

- 采用语义化 unary operation；
- 抑制嵌套调用；
- 覆盖更多 Milvus-owned C++/FFI 边界；
- 增加 workload 和 storage role 归因；
- 只统计 Milvus 可见重试。

## 8. Go Storage 边界实现

### 8.1 LocalChunkManager 与 RemoteChunkManager

Local 和 Remote ChunkManager 的主要 unary 方法已接入：

| API | 记录操作 |
|---|---|
| `Path`、`Size`、`Exist` | `stat` |
| `Read`、`Reader` | `read` |
| `ReadAt`、`ReaderAtOffset` | `range_read` |
| `Write` | `write` |
| `WalkWithPrefix` | `list` |
| `Remove` | `delete` |
| `Copy` | `copy` |

Remote backend 根据 cloud provider 归为 `s3_compatible`、`azure` 或 `gcp`；Local 使用 `local`。

Batch 方法复用 unary 方法，不额外记录一个虚构的 batch operation。例如 `MultiWrite` 只记录每个对象的 Write。

### 8.2 嵌套抑制

`Path -> Exist`、`Copy -> Exist`、`ReaderAtOffset -> Reader` 等路径可能发生嵌套。实现通过 `WithSuppressed(ctx)` 抑制内部辅助调用，保证一个公开逻辑操作不会被重复统计。

同样，Import retryable reader 重新打开底层 reader 时会抑制重开的 nested observation，只由外层 retrying logical read 记录完整耗时和 retry count。

### 8.3 Streaming FileReader

`instrumentedFileReader` 包装现有 FileReader：

- 在 Reader 创建/open 时开始计时；
- 首次成功返回非零字节时记录 TTFB；
- 每次只累加真实返回的 `n`；
- EOF 作为成功终止；
- 非 EOF error 作为失败；
- Close 会终止尚未完成的操作；
- `sync.Once` 防止 EOF、error、Close 重复 Finish。

`DetachProfiledFileReader` 用于把观察所有权转移给更高层的 retryable reader，避免底层每次 reopen 都单独计数。

### 8.4 Milvus 可见重试

Remote `Read`、`Size` 和 Import `retryableReader` 会记录实际由 Milvus retry loop 执行的 retry count 和最后可见 retry reason。

SDK 或 `milvus-storage` 内部隐藏的重试不在当前覆盖范围内。

## 9. Storage V2/V3 与 FFI 实现

### 9.1 Context 传播

Packed reader/writer 和 FFI wrapper 增加了可选 Profile Context，或新增 `WithContext` 版本 API：

- `NewPackedReader(..., profileContexts...)`
- `NewPackedWriter(..., profileContexts...)`
- `NewFFIPackedReader(..., profileContexts...)`
- `NewFFIPackedWriterWithContext(...)`
- `WriteFileWithContext(...)`
- `ReadFileWithContext(...)`
- `CommitManifestUpdatesWithContext(...)`

旧签名仍通过 `context.Background()` 兼容，因此调用方可以渐进迁移。

### 9.2 观测边界

当前记录：

- reader/writer 创建和 manifest 打开：`stat` 或 metadata phase；
- `ReadNext`：`read`；
- `WriteRecordBatch`：`write`；
- writer close/flush：`write`；
- manifest commit：`write_metadata`；
- FFI whole-file read/write：`read` / `write`。

Arrow record 的逻辑字节使用 column data 的 `SizeInBytes()` 累加。Packed 操作可能内部展开为多个 provider 请求，因此 coverage 和运维文档明确将其称为 Milvus 逻辑边界。

## 10. Milvus-owned C++ 与 Go/C++ Handoff

### 10.1 采用有界原始快照

C++ 没有复制完整的 Go taxonomy 和 histogram schema，而是实现了轻量的 `StorageProfileSnapshot`：

- 最多保存 64 个 read duration；
- 累加 known completed bytes；
- 超过上限后增加 dropped count；
- 支持快照合并。

这样避免了：

- 每次 C++ 操作回调 Go；
- 在线程池中使用 thread-local attribution；
- 在 Go 和 C++ 两边维护两套可演进的 histogram schema；
- 修改 `milvus-storage` API。

### 10.2 Search 路径

Segcore 在 materialize output fields 或 cold tiered-storage 数据时：

1. 读取已有 `OpContext.storage_usage`；
2. 累加 scanned total/cold bytes；
3. 当 cold bytes 大于 0 时记录一次 C++ read duration 和 cold bytes；
4. reduce 多 segment 结果时合并 `StorageProfileSnapshot`；
5. 通过 `GetSearchResultStorageProfile` C API 一次性导出；
6. Go `SearchResult.GetMetadata()` 读取最多 64 个 observation；
7. QueryNode 调用 `ObserveCppReadProfile` 折叠进统一 Go histogram 和 Prometheus 指标。

### 10.3 Query/Retrieve 路径

Retrieve 使用 `segcore.StorageProfileStats` 子消息携带：

- repeated read duration nanos；
- completed bytes；
- dropped observation count。

Go 在结果反序列化后立即调用 `ObserveCppReadProfile`。该 raw C++ 快照不会跨 QueryNode internal RPC 继续传播；跨节点传播的是已经编码好的 Go contribution。

### 10.4 Load、Index 和 Analyze C++ 边界

对于没有现成细粒度快照的 C++ 调用，Go 在 Milvus-owned 调用边界记录整体 read/write：

- segment field data load；
- index load；
- segment load/reopen；
- index build；
- index upload；
- analyze read/write；
- text/BM25/JSON stats build 和 upload。

字节优先使用 binlog/index metadata 和上传结果中的文件大小。若元数据未知，只记录 count 和 latency。

## 11. 内部 Proto 设计

### 11.1 请求控制字段使用二级子结构

为避免把 `level` 和 `scope_id` 作为两个并列字段散落在主请求中，当前实现定义：

```protobuf
message StorageProfileContext {
  int32 level = 1;
  string scope_id = 2;
}
```

主请求只持有一个可选字段：

```protobuf
message SearchRequest {
  // ...
  StorageProfileContext storage_profile = 35;
}

message RetrieveRequest {
  // ...
  StorageProfileContext storage_profile = 27;
}
```

这满足以下目标：

- Profile 控制信息作为一个整体可选；
- 老节点遗漏整个子消息时语义明确；
- 后续可在子消息内增加 schema version、权限或采样元数据，而不继续占用主请求顶层字段；
- `level` 和 `scope_id` 始终成对传播。

### 11.2 Internal result contribution

QueryNode 返回：

```protobuf
bytes SearchResults.storage_profile = 24;
bytes RetrieveResults.storage_profile = 20;
```

这里使用 bytes 是为了让 Go 内部 versioned envelope 独立演进，并允许一个结果携带多个 contribution。

### 11.3 Segcore 局部 Handoff

`segcore.proto` 定义二级子结构：

```protobuf
message StorageProfileStats {
  repeated uint64 read_duration_nanos = 1;
  uint64 read_completed_bytes = 2;
  uint64 dropped_read_observations = 3;
}

message RetrieveResults {
  // ...
  StorageProfileStats storage_profile = 10;
}
```

生成的 Go protobuf 文件通过正常 proto 生成流程更新，没有手工维护 generated descriptor。

## 12. 分布式 Search/Query Profile

### 12.1 Proxy 创建 Scope

Proxy 在 Search、Hybrid Search 和 Query 入口：

1. 读取内部显式 header；
2. 生成 attribution；
3. 使用 trace ID 作为 request/scope ID；
4. 无 trace ID 时使用 `nodeID/unixNano` fallback；
5. 调用 `NewRequestScope` 得到 effective level；
6. 把 `StorageProfileContext` 写入 internal Search/Retrieve request。

Search Scope 已移动到 search-by-PK lookup 之前创建，因此 PK lookup、转换和早返回路径都处于同一个 Profile Context 内。

### 12.2 QueryNode contribution

QueryNode 的 `SearchSegments` 和 `QuerySegments` 根据 internal request 中的 level/scope ID 创建 contribution scope。

Query workload 还根据 query label 细分为：

- query；
- upsert；
- delete。

完成时 QueryNode：

1. 合入 C++ read snapshot；
2. 如果 storage usage tracking 开启，记录 requested/cold/served cache bytes；
3. Snapshot 当前 recorder；
4. 生成 contribution identity；
5. 编码到 internal result 的 `storage_profile` bytes。

### 12.3 中间 reduce 与 Proxy 合并

QueryNode 的 Search reduce、advanced Search reduce 和 Query pipeline 会合并 payload envelope，并保留每个 contribution identity。

Proxy 收集所有 result payload 后：

1. 解码 contribution；
2. 按 identity 去重；
3. 合并 count、bytes、distribution、coverage；
4. 交给 `NoopSummarySink`。

当前合并结果不会进入公开 Search/Query response、gRPC trailer、access log 或管理 API。

## 13. 后台任务归因

### 13.1 通用生命周期

后台任务使用 `NewTaskScope(attribution)`：

1. Scope 使用 `context.Background()` 建立独立生命周期，避免长期任务继承已取消的外部 RPC Context；
2. Task Context 绑定 attribution 和 recorder；
3. 可复用组件通过 `WithDefaultAttribution` 继承父 Scope；
4. 任务成功、失败、停止、取消或从 manager 移除时 Finish；
5. Summary 交给 noop sink，active slot 被释放。

E2E 后补充了 Import TaskManager 的 Remove 路径：尚未执行或被取消的 Import task 在移除时也会调用 `FinishStorageProfile`，避免泄漏 `maxProfiledTasks` reservation。

### 13.2 Flush 与 Streaming Flush

`SyncTask` 根据来源设置：

- DataNode 普通 flush：`flush/auto`；
- 显式 flush：`flush/manual`；
- StreamingNode：`flush/streaming`；
- phase：`write_output`；
- role：`persistent`。

Growing source sync 和 WAL flusher 路径也会创建或继承 StreamingNode task scope。

如果 SyncTask 来自 Import，workload 会保持 `import/ingest`，不会被内部默认 Flush attribution 覆盖。这是避免 Import 输出被重复记为 Flush 的关键实现。

### 13.3 Import

已覆盖：

| 任务 | subtype | 读阶段 | 写阶段 |
|---|---|---|---|
| PreImport | `preimport` | source/read_source | 无持久输出 |
| Import | `ingest` | source/read_source | persistent/write_output |
| L0 PreImport | `l0_preimport` | source/read_source | 无持久输出 |
| L0 Import | `l0_ingest` | source/read_source | persistent/write_output |
| CopySegment | `copy_segment` | persistent/copy_object | persistent/copy_object |

Import task 自己持有一个 Scope。异步 future 全部完成后由 scheduler Finish；从 TaskManager Remove 时也会 Finish。创建 SyncTask 时通过 phase/role 派生新 Context，但继续共享父 Import recorder。

### 13.4 Compaction

DataNode `CompactionV2` 创建 `compaction` task scope，并根据当前枚举映射：

- level0 delete；
- mix；
- clustering；
- sort；
- bump schema version；
- 未映射类型使用 unknown subtype。

ChunkManager 使用 attributed wrapper；Compactor 再由 `profiledCompactor` 包装，确保 Compact 完成或 Stop 时 Finish。

Storage V3 writer、stats file 和 manifest commit 都继续传递同一 Context。

### 13.5 Index、Analyze 与 Stats

DataNode index service 为 task 创建独立 Scope：

- index build：`index/build`；
- analyze：`index/analyze`；
- text stats：`index/stats_text`；
- BM25 stats：`index/stats_bm25`；
- JSON key stats：`index/stats_json_key`；
- sort stats：`index/sort`。

直接进入 C++ 的 build/analyze/stats 调用由 Go 边界记录 read；上传结果记录 write。Task Reset 会 Finish Scope，入队失败则由创建方回收。

### 13.6 Load

QueryNode `LoadSegments` 创建：

- workload：`load`；
- subtype：`segment_load`；
- phase：`read_source`；
- role：`persistent`。

LocalSegment 对 field load、index load、segment load 和 reopen 增加 C++ boundary observation。已知字节来自 binlog/index metadata。

### 13.7 Recovery

StreamingNode WAL flusher 在构造恢复组件时创建：

- workload class：`recovery`；
- workload kind：`recovery`；
- phase：`read_metadata`；
- role：`persistent`。

该 Scope 只覆盖 Milvus-owned recovery build path 内实际传播 Context 的存储访问；没有使用全局 filesystem counter 前后差值。

### 13.8 Snapshot

DataCoord snapshot manager 已为以下入口建立 Scope 或继承已有 Scope：

- create；
- drop；
- drop by collection；
- describe/read metadata；
- restore；
- pin；
- unpin。

嵌套调用检测到已有 active recorder 时不会创建竞争的顶层 Scope，从而避免同一物理执行被重复统计。

### 13.9 GC

DataCoord garbage collector 的 recycle task 使用：

- workload：`gc`；
- phase：`cleanup`；
- role：`persistent`。

List/Delete 等实际操作仍由 ChunkManager 边界统计。

### 13.10 External Sync

External refresh 和 function execution 使用 `external_sync`：

- 读取外部源：source/read_source；
- 写入 Milvus 产物：persistent/write_output 或 write_metadata。

Packed writer 和 manifest commit 都使用带 Context 的版本。

### 13.11 尚未生产的 Task 类型

`replication` workload kind 和部分 subtype 已在枚举中预留，但首期没有完整的生产 Scope 接入。

## 14. Cache 语义

QueryNode 在 `storageUsageTrackingEnabled=true` 时，把已有：

- `scanned_total_bytes` 记为 requested；
- `scanned_cold_bytes` 记为 cold；
- `total - cold` 记为 served。

这是 cache-cell 使用量，不等于 provider 网络传输量。

若 tracking 关闭或某路径不提供 usage，Profile coverage 保持 unavailable；不会把不存在的观测写成 0 命中或 0 miss。

当前实现没有改造 OS page cache、mmap fault、result cache、expression cache、index metadata cache 等非直接对象存储缓存。

## 15. 兼容性和滚动升级

### 15.1 配置兼容

全部新配置默认保持旧行为：Profile 关闭、显式请求关闭、task profiling 关闭。

### 15.2 RPC 兼容

所有 internal request/result 字段都是新增可选字段。旧节点会忽略 request control 或不返回 contribution。

新节点看到缺失 contribution 时标记 coverage partial/quantile incomplete，不解释成零存储访问。

### 15.3 Proto 顶层稳定性

Profile 控制被封装在 `StorageProfileContext` 二级消息中，减少未来继续扩展主请求顶层字段的需要。

### 15.4 `milvus-storage` 兼容

没有修改 `milvus-storage` 源码或 API，也没有为了本功能引入新的外部运行时依赖。

## 16. 性能与资源控制

主要性能设计包括：

- Profile disabled 时使用 noop recorder；
- attribution 使用小型值对象并限制 label；
- histogram 使用固定数组；
- C++ raw sample 上限为 64；
- operation breakdown 上限为 64；
- contribution 最大 1 MiB；
- request rate、active scope、active task 都有限制；
- production sink 为 noop，不执行外部 I/O。

已有 microbenchmark 基线：

| 模式 | 时间 | 分配 |
|---|---:|---:|
| 空循环基线 | 1.198 ns/op | 0 B/op，0 allocs/op |
| 聚合指标，Profile 关闭 | 3.553 us/op | 320 B/op，1 alloc/op |
| Summary recorder 开启 | 4.365 us/op | 432 B/op，2 allocs/op |

这些数字只是初始微基准，不代表已经满足最终生产预算。真实 cache hit/miss、大文件、并发 Flush/Compaction/Import 仍需要在目标部署环境持续测量。

## 17. E2E 后修正

独立 E2E 验证后，当前代码补充了四类行为修正：

1. **Search Profile 提前创建**
   - Scope 在 search-by-PK lookup 之前建立；
   - PK lookup 和 early return 不再遗漏 Profile Context。

2. **显式请求 Profile 只信任内部请求**
   - 使用 `contextutil.IsIntraClusterRequest` 判断授权；
   - 外部 metadata 不能直接开启内部诊断能力。

3. **Exist not-found 按公开 API 语义记为成功**
   - `Exist` 返回 `(false, nil)` 时记录 successful stat；
   - 不产生虚假的 not_found failure 指标。

4. **Import pending task 释放 Profile slot**
   - TaskManager Remove 时 Finish Profile；
   - 避免被取消或未执行任务占用 active task quota。

这些修正均有针对性单元测试。

## 18. 验证结果

### 18.1 针对性单元测试

按仓库要求使用：

```bash
go test -tags dynamic,test -gcflags='all=-N -l' -count=1 ...
```

已通过：

- `internal/storageprofile` 的目标测试；
- `internal/storage`：streaming reader exactly-once、Local/Remote Exist not-found 语义；
- `internal/proxy`：内部请求授权、search-by-PK Scope 创建时机；
- `internal/datanode/importv2`：TaskManager Remove 释放 Profile reservation。

本地 CGo 测试需要把非默认位置的 pkg-config 和动态库目录加入环境，例如：

```bash
PKG_CONFIG_PATH=$PWD/internal/core/output/lib/pkgconfig
LD_LIBRARY_PATH=$PWD/internal/core/output/lib
CGO_LDFLAGS="-Wl,-rpath-link,$PWD/internal/core/output/lib"
```

### 18.2 独立 E2E

端到端测试在独立的 `milvus-storage-access-metrics-test` 工程中执行。测试工程只负责 compose、workload、Prometheus/Grafana 和报告生成，使用该次测试前由本仓库源码编译的 `bin/milvus`，没有把 Milvus 实现代码复制到测试仓库。

该 intensive E2E 验证的是主体实现；第 17 节列出的四项后续修正发生在该二进制构建之后，因此它们由第 18.1 节的针对性单元测试验证，尚未重新执行同规模 intensive E2E。

最新 intensive run 的主要结果：

| 项目 | 结果 |
|---|---:|
| 运行时长 | 322.8 秒 |
| 插入行数 | 80,000 |
| 高维向量维度 | 8,192 |
| 成功逻辑存储操作 | 7,130 |
| 观测逻辑字节 | 80,103,976,637 bytes，约 74.603 GiB |
| 覆盖 workload | flush、index、load、compaction |
| 非法/高基数 label | 0 |
| 指标存在性检查 | PASS |
| workload 归因检查 | PASS |

观察到的主要 attribution 包括：

- StreamingNode flush 的 stat/write；
- DataNode compaction 的 read/stat/write；
- DataNode index 的 read/write；
- QueryNode load 的 read。

Profile policy 指标同时观察到：

- 未请求时的 `disabled/not_requested`；
- task config 允许时的 `summary/allowed`。

### 18.3 验证边界

上述 E2E 是成功路径与大数据量指标验证，不等于完整故障验证：

- 没有产生 visible retry delta；
- 没有产生 error-category delta；
- 没有产生 cache byte delta；
- 没有注入 throttling、permission、credential、timeout、cancel、unexpected EOF 等失败；
- 最新 intensive run 为稳定性关闭了 mutation phase；此前 mutation-enabled run 在 growing-source flush 期间发生 standalone crash，需要单独定位，不能被本功能的指标 PASS 掩盖。

因此当前可以确认的是：主要成功路径、指标注册、有限 label、task attribution 和大字节量采集有效。不能宣称所有失败分类和 retry 行为已经通过端到端验证。

## 19. 已实现、部分实现和未实现

### 19.1 已实现

- 稳定枚举、attribution、coverage 和 histogram schema；
- noop/active recorder；
- operation、bytes、latency、retry、error、inflight Prometheus 指标；
- Profile policy、active、drop、snapshot 指标；
- Local/Remote ChunkManager instrumentation；
- streaming FileReader 和 Milvus-visible retry accounting；
- Storage V2/V3 packed/FFI Context 传播；
- C++ Search/Query cold-read 有界快照；
- Search/Query internal RPC 控制与 contribution 合并；
- Flush、Import、Compaction、Index/Analyze/Stats、Load、Recovery、Snapshot、GC、External Sync 的主要 task attribution；
- Import source/persistent role 拆分；
- rolling upgrade 缺失 contribution 的 partial/unavailable 处理；
- noop production sink；
- 主要单元测试、microbenchmark 和独立 E2E。

### 19.2 部分实现

- C++ 只覆盖当前明确接入的 Milvus-owned boundary；返回结果之前失败的 C++ 路径可能没有快照；
- cache 主要接入 existing tiered storage usage bytes，wait/load/lookup 事件生产仍不完整；
- 部分 packed read/write 是较高层逻辑操作，不能代表精确 provider RPC 数；
- task coverage 依赖 Context 是否能到达具体 owned storage boundary；
- backend kind 对直接 C++ index/analyze 边界可能是 unknown；
- retry 只覆盖 Milvus 自己执行的 retry loop。

### 19.3 明确未实现

- `milvus-storage` 内部 provider instrumentation；
- provider 精确 GET/PUT/RangeGET、内部重试、provider TTFB 和 transferred bytes；
- public SDK Profile API；
- public response、gRPC trailer、Proxy access log；
- Admin API/UI 和 task-status 展示；
- Kafka、ClickHouse、OTLP 等外部 sink；
- `detailed` 级别的慢操作样本；
- 管理员 TTL rule 的 etcd provider；
- 完整 replication workload；
- 云账单级别精确计量。

## 20. 关键代码索引

| 能力 | 主要文件 |
|---|---|
| 枚举和语义 | `internal/storageprofile/enums.go` |
| 分布与 quantile | `internal/storageprofile/distribution.go` |
| Profile、coverage、merge | `internal/storageprofile/profile.go` |
| Recorder、指标双写、错误分类 | `internal/storageprofile/recorder.go` |
| Context 传播 | `internal/storageprofile/context.go` |
| Scope 生命周期和限流 | `internal/storageprofile/scope.go` |
| Policy | `internal/storageprofile/policy.go` |
| Contribution 编码 | `internal/storageprofile/encoding.go` |
| Sink/Presenter 预留 | `internal/storageprofile/sink.go` |
| Prometheus 指标 | `pkg/metrics/persistent_store_metrics.go` |
| Config | `configs/milvus.yaml`、`pkg/util/paramtable/component_param.go` |
| Local/Remote ChunkManager | `internal/storage/local_chunk_manager.go`、`remote_chunk_manager.go` |
| Streaming reader | `internal/storage/profile_reader.go` |
| Attribution wrapper | `internal/storage/attributed_chunk_manager.go` |
| Packed/FFI | `internal/storagev2/packed/` |
| C++ snapshot | `internal/core/src/common/StorageProfile.h` |
| Segcore Search handoff | `internal/core/src/segcore/search_result_export_c.*` |
| Proxy request Scope | `internal/proxy/storage_profile.go`、`impl.go` |
| QueryNode contribution | `internal/querynodev2/storage_profile.go`、`services.go` |
| Internal Proto | `pkg/proto/internal.proto`、`pkg/proto/segcore.proto` |
| Import | `internal/datanode/importv2/` |
| Flush | `internal/flushcommon/syncmgr/` |
| Compaction | `internal/datanode/services.go`、`internal/datanode/compactor/` |
| Index/Analyze/Stats | `internal/datanode/index_services.go`、`internal/datanode/index/` |
| Snapshot/GC | `internal/datacoord/snapshot_manager.go`、`garbage_collector.go` |
| Streaming recovery | `internal/streamingnode/server/flusher/flusherimpl/wal_flusher.go` |

## 21. 结论

首期实现已经建立了统一、低基数、可合并的对象存储观测基础：

- Prometheus 用于持续聚合监控；
- Storage Profile 用于显式请求或任务的有限摘要；
- Go、C++、FFI 和内部 RPC 使用同一套最终 Go 数据模型；
- Import、Flush 等复用链路通过父 Scope 继承避免双计数；
- 老节点、未知桶和缺失字节通过 coverage 表达不确定性；
- 外部 sink、公开展示和 provider 深层观测被刻意留在后续版本。

当前最重要的后续工作不是继续增加无边界 label，而是完成故障矩阵 fault injection、补齐 cache wait/load 的实际生产事件、测量目标环境性能预算，并在确认需求后设计安全的 Profile 持久化或展示接口。
