# milvus-storage 对象存储指标能力与最小先行实现分析报告

- **日期：** 2026-07-14
- **范围：** milvus-storage、Storage V2/V3 access layer、Milvus object-storage-profiling
- **参考设计：** [20260713-object-storage-profiling.md](20260713-object-storage-profiling.md)
- **相关实现说明：** [20260713-object-storage-profiling-implementation_cn.md](20260713-object-storage-profiling-implementation_cn.md)
- **状态：** 调研与实施建议

## 1. 执行摘要

`milvus-storage` 当前已经提供一套 filesystem 级累计指标，能够观测读写次数、读写字节、部分文件系统操作、总失败次数以及 multipart upload 创建和完成次数。但是，这套指标的能力和语义距离 object-storage-profiling MEP 的要求仍有较大差距。

主要结论如下：

1. `milvus-storage` 原始 FFI 当前提供 13 个 filesystem 累计值：
   - read/write count；
   - read/write bytes；
   - get file info/create directory/delete directory/delete file/move/copy count；
   - failed count；
   - multipart upload created/finished。
2. Milvus 当前只读取并发布其中 8 项，尚未发布 create/delete/move/copy 相关的 5 项原始值。
3. 这些值是每个 cached filesystem 实例上的共享累计快照，不是 request、task、workload、Storage V2 或 Storage V3 的独立指标。
4. Local、S3-compatible 和 Azure 的计数位置并不完全相同，因此当前 `read_count`、`write_count` 不是严格统一的 provider request 语义。
5. 现有指标没有 operation latency、histogram、p50/p95/p99、retry、error category、TTFB、inflight、workload attribution 或 per-scope summary。
6. 当前 feature branch 中较完整的 profiling 实现位于 Milvus-owned logical boundary，没有修改 `milvus-storage`，因此它仍看不到隐藏在 `milvus-storage` 或 provider SDK 内部的真实请求展开和重试。
7. 如果撤销当前 feature branch，Milvus master 上已有的 legacy filesystem metrics 仍然存在；丢失的是 `internal/storageprofile`、workload attribution、request/task profile、分布式 contribution 合并及新的逻辑存储指标。
8. 如果希望先在 `milvus-storage` 做最小实现，建议第一阶段只实现 versioned provider aggregate metrics，不立即实现完整 request/task profile。

推荐的第一阶段能力是：

```text
provider operation count
+ operation outcome
+ bounded error category
+ duration histogram
+ requested/completed bytes
+ inflight
+ aggregate retry
+ multipart operations
+ versioned snapshot FFI
```

第一阶段不应尝试通过 filesystem global counter 的前后差值构造 task profile，也不应在任务开始时 reset 共享 metrics。

## 2. 当前代码和依赖状态

当前 Milvus 使用的 `milvus-storage` commit 是：

```text
bb3a97563cf59c9db7bda471f61340e622de3a17
```

依赖定义位于：

- `internal/core/thirdparty/milvus-storage/CMakeLists.txt`

当前 object-storage-profiling feature branch 在 Milvus master 上增加了统一 profile 模型、逻辑操作指标、任务归因和跨节点合并等能力。但是，已有的 filesystem metrics 并不是该 feature branch 新增的，Milvus master 已经包含：

- `internal/storagev2/filesystem_metrics.go`；
- `pkg/metrics/persistent_store_metrics.go` 中的 `filesystem_*` metrics；
- QueryNode Load、DataNode Flush、Compaction 和 Index 等路径上的 snapshot publish。

因此，如果回退当前 feature branch：

仍然保留：

- `loon_filesystem_get_metrics`；
- legacy filesystem snapshot；
- legacy Prometheus filesystem gauges；
- 部分任务完成后的 snapshot publish。

将会丢失：

- `internal/storageprofile`；
- `milvus_storage_operations_total` 等新逻辑指标；
- workload、phase、storage role attribution；
- request/task storage profile；
- Search/Query contribution 和 Proxy merge；
- Storage V2/V3 Milvus-owned boundary profiling；
- coverage、fixed histogram 和 profile policy。

## 3. milvus-storage 当前原生指标

### 3.1 原始 FFI 指标

`milvus-storage` 的 `LoonFilesystemMetricsSnapshot` 当前包含以下字段：

| 字段 | 当前含义 |
|---|---|
| `read_count` | 被插桩 read 操作的累计次数 |
| `write_count` | 被插桩 write 操作的累计次数 |
| `read_bytes` | 被插桩 read 路径累计读取的字节 |
| `write_bytes` | 被插桩 write 路径累计写入的字节 |
| `get_file_info_count` | GetFileInfo 累计次数 |
| `create_dir_count` | CreateDir 累计次数 |
| `delete_dir_count` | DeleteDir 累计次数 |
| `delete_file_count` | DeleteFile 累计次数 |
| `move_count` | Move 累计次数 |
| `copy_file_count` | CopyFile 累计次数 |
| `failed_count` | 当前已插桩路径上的总失败次数 |
| `multi_part_upload_created` | multipart upload create 累计次数 |
| `multi_part_upload_finished` | multipart upload complete/commit 累计次数 |

定义位于 `milvus-storage/ffi_filesystem_metrics_c.h`。

当前还提供：

```c
loon_filesystem_get_metrics(...)
loon_filesystem_reset_metrics(...)
```

这套接口只返回当前累计值，不返回：

- 时间分布；
- operation taxonomy；
- outcome 分类；
- error category；
- retry；
- request/task scope；
- workload 或 storage role；
- Storage V2/V3 版本；
- provider access coverage。

### 3.2 Milvus 当前只读取 8 项

Milvus 的 Go 包装类型当前是：

```go
type FilesystemMetrics struct {
    ReadCount               int64
    WriteCount              int64
    ReadBytes               int64
    WriteBytes              int64
    GetFileInfoCount        int64
    FailedCount             int64
    MultiPartUploadCreated  int64
    MultiPartUploadFinished int64
}
```

实现位于 `internal/storagev2/filesystem_metrics.go`。

因此，以下已经由 `milvus-storage` FFI 提供的字段尚未进入 Milvus Prometheus metrics：

- `create_dir_count`；
- `delete_dir_count`；
- `delete_file_count`；
- `move_count`；
- `copy_file_count`。

如果目标只是快速增强 legacy metrics，Milvus 可以直接读取并发布这 5 项，不需要先修改 `milvus-storage`。但是，这种增强仍然不能满足完整 object-storage-profiling 要求，因为它们依旧是共享 filesystem 累计值。

## 4. 当前指标语义并不跨后端统一

当前 `FilesystemMetrics` 表面上是一组统一字段，但 Local、S3-compatible 和 Azure 的插桩层次不同。因此，现有 read/write count 不能直接解释为跨后端统一的 provider HTTP request 数。

### 4.1 Local filesystem

Local filesystem 使用 `LocalFileSystemWrapper` 和 stream wrapper 记录指标。

主要语义是：

- `read_count` 在成功返回非零数据的 `Read` 或 `ReadAt` 后增加；
- `read_bytes` 是实际返回给调用者的字节；
- `write_count` 主要在成功打开 output stream 或 append stream 时增加；
- `write_bytes` 在每次成功的 stream `Write` 后增加；
- stat/create/delete/move/copy 是 Arrow filesystem 方法级计数。

例如：

```text
一次 OpenOutputStream
+ 十次 Write
```

可能得到：

```text
write_count = 1
write_bytes = 十次 Write 的总和
```

所以 Local 指标更接近 stream 或 filesystem logical operation，而不是操作系统 syscall 数。

### 4.2 S3-compatible filesystem

S3 当前只对少量 SDK 方法进行了显式 metrics override：

- `CreateMultipartUpload`；
- `UploadPart`；
- `PutObject`；
- `GetObject`；
- 特殊包装的 `CompleteMultipartUpload`。

当前大致语义是：

- `read_count`：调用一次 S3 SDK `GetObject` 增加一次；
- `write_count`：调用一次 `PutObject` 或 `UploadPart` 增加一次；
- `read_bytes`：成功 GetObject response 的 Content-Length；
- `write_bytes`：成功 PutObject/UploadPart request 的 Content-Length；
- `failed_count`：这些被 override 的调用失败时增加。

这些指标比 Local 更接近 provider SDK operation，但是仍不是精确 HTTP attempt，因为一次 SDK method call 可能在 SDK 内部发生多个 retry attempt。

以下大量 S3 操作目前没有进入对应的 operation counter：

- `HeadObject`；
- `HeadBucket`；
- `ListObjectsV2`；
- `ListBuckets`；
- `DeleteObject`；
- `DeleteObjects`；
- `CopyObject`；
- `CreateBucket`；
- `DeleteBucket`；
- `AbortMultipartUpload`。

因此对 S3-compatible backend 而言：

- stat/list/delete/copy 指标明显不完整；
- `get_file_info_count` 不能代表全部 HEAD；
- `failed_count` 不是全部 provider failure；
- `read_count` 是顶层 `GetObject` 调用数，不是 SDK retry 后的真实 HTTP GET attempt 数。

### 4.3 Azure filesystem

Azure 的指标也是混合层次：

- `GetFileInfo/CreateDir/DeleteFile/Move/CopyFile` 在 Azure filesystem 方法入口计数；
- read 通过 `MetricsInputStream` 和 `MetricsRandomAccessFile` 统计；
- write count/bytes 主要在 `StageBlock` 处统计；
- multipart created/finished 映射为 empty block blob 和 block-list commit。

因此：

- Azure `write_count` 更接近 block upload 数；
- Local `write_count` 更接近 output stream open 数；
- S3 `write_count` 是 PutObject/UploadPart SDK call 数。

这三种值不能直接横向比较。

### 4.4 对现有指标的正确解释

当前指标适合：

- 同一 backend、同一版本前后的趋势比较；
- 粗略观察读写字节；
- 粗略判断 multipart 行为；
- 发现部分 failure 增长；
- 比较 packed reader 参数调整前后的 read count；
- 观察某个部署是否出现明显的 I/O amplification。

当前指标不适合：

- 精确比较 S3 与 Azure 的 provider request 数；
- 云厂商账单核对；
- 精确统计 GET/HEAD/LIST/DELETE/COPY；
- 精确统计 SDK retry；
- 计算 TTFB 或 latency quantile；
- 把 filesystem 全局值归因到单个 request/task；
- 把 Storage V2 与 Storage V3 分开。

## 5. Storage V2/V3 的覆盖情况

### 5.1 V2/V3 共享 FilesystemCache

`milvus-storage` 的 Reader、Writer、Manifest、Transaction、Segment Reader/Writer 等路径最终都会从 `FilesystemCache` 获取 filesystem。

Filesystem cache 根据 storage properties 和 path 解析配置、生成 cache key，并复用 cached filesystem instance。

因此，现有 filesystem metrics 会自然覆盖：

- Storage V2 packed reader/writer；
- Storage V3 manifest read/write；
- V3 transaction begin/get manifest/commit；
- V3 stats、delta log 和 LOB；
- Index/Analyze/Stats 使用的 `milvus-storage` 访问；
- external filesystem 访问；
- Parquet、Vortex、Lance 等 format reader/writer 发起的 filesystem I/O。

### 5.2 当前无法区分 V2 与 V3

现有 snapshot 不携带：

```text
storage_version
reader_type
writer_type
manifest/data/stats role
workload
phase
scope_id
```

如果同一个 DataNode 同时执行：

- V2 Flush；
- V3 Compaction；
- V3 Import；
- V2 Index；

最终只能取得这些任务共享的 filesystem 累计值。

不建议通过 object path 推断 V2/V3，例如解析：

```text
_delta
_stats
manifest
insert_log
```

原因包括：

- storage layout 会演进；
- external filesystem 的路径布局不同；
- 路径解析会把高层格式知识引入 provider metrics；
- object path 具有高基数和潜在敏感性；
- path-based 推断容易在兼容路径和迁移路径上出错。

如果必须区分 Storage V2/V3，正确方式是由 Milvus 在创建 reader/writer/transaction 时显式传入 bounded `storage_version` enum。

## 6. 与 object-storage-profiling MEP 的差距

MEP 要求两类输出：

1. 始终开启的低基数 Prometheus aggregate metrics；
2. 显式启用的 request/task storage profile。

当前 `milvus-storage` 的满足情况如下：

| MEP 能力 | 当前 milvus-storage | 评价 |
|---|---|---|
| Read/write count | 有 | 部分，跨 backend 语义不一致 |
| Stat/list/delete/copy count | 部分有 | S3 覆盖明显不足 |
| Multipart count | 有 | S3/Azure 映射略有差异 |
| Logical/provider bytes | 部分有 | caller bytes 与 provider bytes 未形成统一契约 |
| Operation latency | 无 | 完全缺失 |
| avg/min/max | 无 | 完全缺失 |
| p50/p95/p99 | 无 | 完全缺失 |
| Latency histogram | 无 | 完全缺失 |
| Operation size histogram | 无 | 完全缺失 |
| Outcome | 只有总 failed count | 不足 |
| Error category | 无 | 完全缺失 |
| Timeout/canceled | 无 | 完全缺失 |
| Retry count | 无 | SDK retry 不可见 |
| Retry reason/backoff | 无 | 完全缺失 |
| Inflight | 无 | 完全缺失 |
| TTFB | 无 | 完全缺失 |
| Transfer duration | 无 | 完全缺失 |
| Provider-transferred bytes | S3 部分近似 | 未形成明确契约 |
| workload attribution | 无 | 应主要由 Milvus 提供 |
| phase/storage role | 无 | 应主要由 Milvus 提供 |
| request/task scope | 无 | 需要显式 profile handle |
| distributed merge | 无 | 应由 Milvus/Proxy 完成 |
| Cache usage | 无 | Milvus tiered cache 不在 milvus-storage |
| Coverage state | 无 | 需要新增 |

所以，现有 metrics 只能作为 legacy aggregate signal，不能替代 MEP。

## 7. 当前 Milvus legacy filesystem metrics 的限制

### 7.1 使用 Gauge snapshot

Milvus 当前将 filesystem 累计值发布为 Gauge，例如：

```text
milvus_storage_filesystem_read_count
milvus_storage_filesystem_write_count
milvus_storage_filesystem_read_bytes
milvus_storage_filesystem_write_bytes
milvus_storage_filesystem_get_file_info_count
milvus_storage_filesystem_failed_count
milvus_storage_filesystem_multi_part_upload_created
milvus_storage_filesystem_multi_part_upload_finished
```

使用 Gauge 而不是 Counter 有一定合理性，因为底层值不保证永远单调：

- `loon_filesystem_reset_metrics` 可以归零；
- filesystem cache 被清理后 collector 会重新创建；
- cache eviction 后同一 config 可能创建新的 filesystem instance；
- 进程重启会归零。

因此不能假设这些值是永久单调的 process counter。

### 7.2 当前不是周期性实时采集

当前主要在部分任务完成后发布 snapshot，例如：

- QueryNode Load 完成后；
- Sync/Flush 完成后；
- Compaction 完成后；
- Index task 完成后。

这会导致：

- 指标可能滞后；
- 未触发 publish 的路径不会及时更新；
- 失败提前返回的路径可能没有 publish；
- 纯 manifest/stat 操作可能长期没有刷新；
- Prometheus scrape 得到的是上次 publish 的 snapshot，而不是实时 collector 值。

如果只追求极小改动，可以在各 component 增加周期性 filesystem snapshot collector，但这依旧只能提供 aggregate metrics。

### 7.3 `fs` label 不适合作为新指标契约

当前 Milvus 的 filesystem key 对 remote storage 通常是：

```text
address + "/" + bucket
```

Local storage 则可能使用 root path。

这与 MEP 的低基数和隐私约束不一致。新的 Prometheus metrics 不应携带：

- bucket；
- endpoint；
- object path；
- collection/request/task ID；
- raw error string。

因此，现有 `filesystem_*{fs="endpoint/bucket"}` 应被视为 legacy metric，不建议继续作为新的 profiling metric contract 扩展。

## 8. 回退 feature branch 后的零 milvus-storage 改动方案

如果目标是尽快获得更多 aggregate 数据，可以先不修改 `milvus-storage`。

Milvus 侧可以：

1. 保留现有 `loon_filesystem_get_metrics`；
2. 接出尚未发布的 5 项字段：
   - create dir；
   - delete dir；
   - delete file；
   - move；
   - copy。
3. 增加 component-level 周期性 snapshot；
4. 保持这些指标为 legacy gauge；
5. 明确文档说明：
   - 它们是 filesystem-instance aggregate；
   - 不区分 workload；
   - 不区分 Storage V2/V3；
   - 不代表 provider HTTP billing request；
   - 不能用于 task profile。

这个方案能够快速提供：

- 读写吞吐趋势；
- read amplification 粗略趋势；
- multipart part 数变化；
- 部分 failure 增长；
- packed reader 参数优化前后对比。

但是，不能把它描述为完成 object-storage-profiling MEP。

## 9. 推荐的 milvus-storage 最小先行方案

### 9.1 不扩展旧语义，新增 versioned contract

不建议继续向当前 `FilesystemMetrics` 中直接堆叠字段。当前结构已经存在历史语义，而且不同 backend 的含义不一致。直接修改旧字段语义会让已有 dashboard 在升级后静默变化。

建议新增一套明确版本化的 provider metrics contract。

第一阶段只实现：

> always-on provider aggregate metrics。

第一阶段暂不实现：

- request profile；
- task profile；
- workload attribution；
- Proxy merge；
- public API；
- detailed slow sample；
- 完整 TTFB；
- provider object/path sample。

### 9.2 Operation taxonomy

建议与 MEP 保持一致：

```text
read
range_read
write
stat
list
delete
copy
multipart_create
multipart_write
multipart_complete
multipart_abort
```

### 9.3 Outcome taxonomy

```text
success
failure
timeout
canceled
```

### 9.4 Error category taxonomy

```text
none
not_found
throttled
permission_denied
invalid_credentials
bucket_not_found
invalid_argument
invalid_range
entity_too_large
unexpected_eof
timeout
canceled
io_failed
unknown
```

### 9.5 每个 operation 的最小统计

建议至少支持：

```text
count by outcome
requested_bytes
completed_bytes
duration_sum
duration_buckets[]
error_count by bounded category
retry_count
inflight
```

可后续增加：

```text
duration_min
duration_max
size_buckets[]
retry_backoff_duration
TTFB buckets
transfer duration buckets
```

## 10. 第一阶段优先覆盖 S3-compatible backend

优先覆盖以下 backend：

- AWS S3；
- MinIO；
- GCS S3 compatibility；
- Aliyun OSS；
- Tencent COS；
- Huawei OBS。

这些 backend 当前共享 S3 filesystem/client 路径，投入产出比最高。

建议至少覆盖：

```text
GetObject
PutObject
UploadPart
HeadObject
HeadBucket
ListObjectsV2
ListBuckets
DeleteObject
DeleteObjects
CopyObject
CreateMultipartUpload
CompleteMultipartUpload
AbortMultipartUpload
CreateBucket
DeleteBucket
```

建议映射：

| Provider call | Operation |
|---|---|
| 无 Range 的 GetObject | `read` |
| 带 Range 的 GetObject | `range_read` |
| PutObject | `write` |
| HeadObject/HeadBucket | `stat` |
| ListObjectsV2/ListBuckets | `list` |
| DeleteObject/DeleteObjects | `delete` |
| CopyObject | `copy` |
| CreateMultipartUpload | `multipart_create` |
| UploadPart | `multipart_write` |
| CompleteMultipartUpload | `multipart_complete` |
| AbortMultipartUpload | `multipart_abort` |

## 11. FFI 兼容设计

### 11.1 不要原地扩展旧 struct 和旧函数语义

现有接口是：

```c
loon_filesystem_get_metrics(
    FileSystemHandle,
    LoonFilesystemMetricsSnapshot*
)
```

不应直接让旧函数写入一个被扩大的复杂 struct。旧调用者可能仍按旧 struct size 分配内存，造成 ABI 越界。

建议新增：

```c
LoonFFIResult loon_filesystem_get_access_metrics_v2(
    FileSystemHandle handle,
    LoonAccessMetricsSnapshotV2* snapshot
);
```

新 struct 至少包含：

```c
typedef struct {
    uint32_t struct_size;
    uint32_t schema_version;
    uint32_t bucket_schema_version;
    uint32_t operation_count;
    uint32_t outcome_count;
    uint32_t error_category_count;

    uint64_t operation_counts[...];
    uint64_t completed_bytes[...];
    uint64_t duration_nanos_sum[...];
    uint64_t duration_buckets[...];
    uint64_t error_counts[...];
    uint64_t retry_counts[...];
    uint64_t inflight[...];
} LoonAccessMetricsSnapshotV2;
```

也可以使用两阶段 buffer API：

```c
loon_filesystem_access_metrics_snapshot_size(...);
loon_filesystem_get_access_metrics(..., buffer, buffer_size, written);
```

第一版使用固定数组 struct 会更容易实现和审核。

旧 API 应保持不变，用于兼容已有 Milvus 和其他客户端。

## 12. milvus-storage 内部实现建议

### 12.1 固定枚举

建议定义：

```cpp
enum class AccessOperation : uint8_t;
enum class AccessOutcome : uint8_t;
enum class AccessErrorCategory : uint8_t;
enum class BackendKind : uint8_t;
```

Recorder 不应接受动态 operation/error label，例如：

```cpp
std::string operation;
std::string error;
std::string bucket;
std::string path;
```

否则很容易将动态值带入 Prometheus 或 profile schema。

### 12.2 RAII OperationGuard

建议使用 RAII guard 统一 operation 生命周期：

```cpp
class OperationGuard {
 public:
    OperationGuard(
        AccessMetricsRecorder* recorder,
        AccessOperation operation,
        uint64_t requested_bytes,
        bool requested_bytes_known);

    void SetCompletedBytes(uint64_t bytes);
    void AddRetry(AccessErrorCategory reason, uint64_t backoff_nanos);
    void FinishSuccess();
    void FinishFailure(AccessErrorCategory category);
    void FinishTimeout();
    void FinishCanceled();

    ~OperationGuard();
};
```

S3 调用点可以写成：

```cpp
auto guard = recorder->Begin(
    AccessOperation::RangeRead,
    requested_bytes,
    true);

auto outcome = BaseS3Client::GetObject(request);

if (!outcome.IsSuccess()) {
    guard.FinishFailure(ClassifyS3Error(outcome.GetError()));
    return outcome;
}

guard.SetCompletedBytes(outcome.GetResult().GetContentLength());
guard.FinishSuccess();
return outcome;
```

这样能够统一保证：

- duration 记录；
- inflight 增减；
- exception/early return 不容易漏记；
- operation 只 finish 一次；
- 后续可以同时写 global recorder 和 scoped recorder。

### 12.3 固定 histogram bucket

建议和 MEP 使用同一 latency bucket schema，或者至少显式携带 `bucket_schema_version`。

例如：

```text
250us
500us
1ms
2ms
5ms
10ms
25ms
50ms
100ms
250ms
500ms
1s
2.5s
5s
10s
30s
60s
120s
300s
+Inf
```

`milvus-storage` 不需要自己计算 p50/p95/p99，只需输出固定 bucket count。Prometheus 或 Milvus profile merger 可以在上层计算 quantile。

这种方式具有以下优点：

- 固定内存；
- 不保存原始样本；
- 可并发累加；
- 可跨节点合并；
- 可计算近似 quantile；
- 不存在 ring buffer percentile 无法确定性合并的问题。

## 13. Retry 指标设计

### 13.1 当前限制

S3 SDK retry 通常发生在一次 SDK method call 内部。

例如：

```cpp
client->GetObject(req)
```

可能在 SDK 内部执行多个 HTTP attempt，但当前 `read_count` 只增加一次。

因此现有指标看不到：

- 实际 attempt 数；
- retry reason；
- backoff；
- throttling amplification；
- retry 对 latency tail 的贡献。

`CompleteMultipartUpload` 当前存在 `milvus-storage` 自己实现的特殊 retry loop，但没有输出统一 retry metrics。

### 13.2 第一阶段建议

第一阶段可以：

1. 完整记录 `milvus-storage` 自己显式执行的 retry；
2. 包装 AWS retry strategy，在 `ShouldRetry` 返回 true 时增加 aggregate retry counter；
3. 记录 bounded retry reason；
4. 记录 backoff duration；
5. 暂时不承诺 retry 与某个 request/task scope 精确关联。

Global retry metrics 相对容易实现。

Scoped retry 较难，因为：

- S3Client 被 filesystem cache 共享；
- retry strategy 通常是 client-level；
- 同一个 client 同时服务多个任务；
- retry callback 默认没有 Milvus profile handle；
- 不能在 shared filesystem 上维护一个可变的 current scope。

因此第一阶段可以声明：

```text
provider retry aggregate: instrumented
provider retry per scope: unavailable 或 partial
```

### 13.3 不应依赖 thread-local 完成最终归因

单个同步 SDK call 内部使用 thread-local 辅助可能暂时可行，但不应成为 request/task attribution 的最终架构，因为：

- async read/write 可能跨线程；
- completion callback 可能在其他线程执行；
- 一个 request 会并行执行多个 storage operation；
- retry policy 和 HTTP callback 的线程模型未必稳定；
- Milvus task 会跨线程池迁移。

最终 scoped profile 应使用显式 handle 传播。

## 14. 第二阶段：Scoped provider profile

第一阶段 aggregate metrics 稳定后，可以增加 scoped profile handle。

### 14.1 建议的 profile FFI

```c
typedef void* LoonStorageProfileHandle;

LoonFFIResult loon_storage_profile_create(
    const LoonStorageProfileOptions* options,
    LoonStorageProfileHandle* out);

LoonFFIResult loon_storage_profile_snapshot(
    LoonStorageProfileHandle handle,
    LoonStorageProfileSnapshot* out);

void loon_storage_profile_destroy(
    LoonStorageProfileHandle handle);
```

Reader、Writer、Transaction 增加兼容扩展 API：

```c
loon_reader_new_v2(..., const LoonExecutionContext* ctx, ...);
loon_writer_new_v2(..., const LoonExecutionContext* ctx, ...);
loon_transaction_begin_v2(..., const LoonExecutionContext* ctx, ...);
```

其中：

```c
typedef struct {
    uint32_t struct_size;
    LoonStorageProfileHandle storage_profile;
    int32_t storage_version;
} LoonExecutionContext;
```

旧 API 内部调用新 API 并传入 `ctx = NULL`。

这样可以保证：

- 旧调用者兼容；
- 新 Milvus 显式传 profile handle；
- V2/V3 由调用者显式提供；
- async task 显式捕获 ref-counted recorder；
- 不需要每个 C++ operation 回调 Go；
- snapshot 可以在任务结束时一次性返回。

### 14.2 milvus-storage 不应理解 Milvus workload 语义

`milvus-storage` 不应自己推断或维护：

```text
search
query
flush
import
compaction
index
load
recovery
snapshot
gc
```

它也不应自行判断：

```text
phase=read_source
phase=write_output
storage_role=source
storage_role=persistent
```

这些属于 Milvus 调度和业务语义。

正确分工是：

```text
Milvus:
    request/task/workload/phase/role/component attribution

milvus-storage:
    provider operation/latency/bytes/retry/error

Milvus final merger:
    将业务 attribution 与 provider snapshot 合并
```

## 15. Logical metrics 与 provider metrics 的双计数问题

MEP 的 profile model 预留了：

```text
AccessLayerMilvus
AccessLayerProvider
```

但是当前 Prometheus metric labels 没有 `access_layer`。

如果未来把 provider operation 也写入现有：

```text
milvus_storage_operations_total
```

一次 logical read 可能同时产生：

```text
1 次 Milvus logical read
N 次 provider range GET
```

如果没有 access layer 区分，它们会进入同一 series 并相加，导致指标含义失真。

建议选择以下一种方案。

### 15.1 方案 A：增加 `access_layer` label

```text
milvus_storage_operations_total{
  access_layer="milvus|provider",
  ...
}
```

如果 metric contract 尚未正式发布，这是最统一的方案。

### 15.2 方案 B：独立 provider metric family

```text
milvus_storage_provider_operations_total
milvus_storage_provider_operation_duration_seconds
milvus_storage_provider_bytes_total
milvus_storage_provider_retries_total
milvus_storage_provider_errors_total
milvus_storage_provider_operations_inflight
```

对第一阶段 aggregate provider metrics，更推荐方案 B。

原因是 filesystem-global provider snapshot 通常没有可靠的：

- workload kind；
- phase；
- storage role；
- request/task scope。

独立 metric family 不需要用大量 `unknown` 填充 logical metric labels，也能避免 dashboard 意外把 logical 与 provider 数据相加。

Profile summary 内部仍然可以使用 `AccessLayer` 统一表示和合并。

## 16. Provider 层可以进一步获得的高价值指标

### 16.1 Range amplification

建议后续记录：

```text
logical_requested_bytes
provider_requested_bytes
provider_completed_bytes
provider_range_request_count
```

可以计算：

```text
provider bytes / logical bytes
provider request count / logical read count
```

这对优化以下组件非常重要：

- Parquet prebuffer；
- row group；
- Vortex layout；
- packed reader range planning；
- cold storage access；
- small range coalescing。

### 16.2 Retry amplification

```text
provider_attempts
provider_retries
retry_backoff_seconds
throttled_retries
timeout_retries
```

可以帮助判断 latency tail 来自：

- 服务端 throttling；
- 网络错误；
- connection error；
- SDK backoff；
- Milvus 自己的并发策略。

### 16.3 List amplification

```text
list_request_count
list_page_count
listed_object_count
```

一次 Milvus list 操作可能触发多页 `ListObjectsV2`，provider instrumentation 才能看到真实分页成本。

### 16.4 Multipart 行为

```text
multipart_create_count
multipart_part_count
multipart_complete_count
multipart_abort_count
multipart_failed_count
multipart_bytes
multipart_part_size histogram
```

当前只有 create 和 finished，无法判断：

- 一个 upload 使用了多少 part；
- 是否发生 abort；
- incomplete upload 是否积累；
- part size 是否合理；
- complete retry 是否产生 latency tail。

### 16.5 HTTP/connection 层

后续可增加低基数指标：

```text
http_status_class = 2xx/4xx/5xx
connection_acquire_duration
active_http_requests
request_queue_duration
dns_duration
tls_handshake_duration
```

这些指标不建议与第一版同时实现。第一版应先稳定 operation、latency、bytes、retry 和 error semantics。

## 17. 不能仅在 milvus-storage 完成的能力

即使 `milvus-storage` 提供完整 provider metrics，下列能力仍然必须由 Milvus 配合：

1. **workload kind**
   - `milvus-storage` 不知道当前 I/O 属于 Search、Import 还是 Compaction。
2. **phase**
   - 它不知道当前是 `read_source`、`write_output` 还是 `warmup`。
3. **storage role**
   - 它不知道 source 与 persistent storage 的业务区别。
4. **request/task scope**
   - 需要 Milvus 创建 profile handle 并显式传递。
5. **distributed Search merge**
   - 需要 QueryNode contribution 和 Proxy merger。
6. **tiered-storage cache metrics**
   - 主要发生在 Milvus/segcore cache 层。
7. **critical-path latency**
   - provider operation duration 累计值可能因并行执行而大于 request wall time。
8. **public response/access log/admin UI**
   - 这些属于 Milvus 产品和展示层。

因此，修改 `milvus-storage` 是补充 provider visibility，不是替代 Milvus profiling framework。

## 18. 推荐落地顺序

### 18.1 PR 1：milvus-storage metrics contract

完成：

- operation/outcome/error enums；
- 固定 histogram schema；
- aggregate recorder；
- versioned snapshot；
- 新 FFI；
- legacy FFI 保持不变；
- 单元测试和 benchmark。

不做 scope。

### 18.2 PR 2：S3-compatible provider instrumentation

覆盖：

- GET/Range GET；
- PUT；
- HEAD；
- LIST；
- DELETE；
- COPY；
- multipart create/write/complete/abort；
- error category；
- duration；
- bytes；
- inflight；
- aggregate retry。

增加 MinIO/mock S3 fault tests。

### 18.3 PR 3：Milvus dependency bump 和 minimal exporter

Milvus 侧：

- 升级 `milvus-storage` commit；
- 周期性读取 provider snapshot；
- 发布独立 `milvus_storage_provider_*` metrics；
- labels 只使用 bounded enum；
- 不带 bucket/endpoint/path；
- 明确 workload attribution unavailable；
- 不使用 global delta 构造 task profile。

这一步已经能够提供有实际价值的 provider-level dashboard。

### 18.4 PR 4：Azure parity

把 Azure provider 调用映射到相同 taxonomy。

需要注意：

- StageBlock 不完全等价于 PutObject；
- CommitBlockList 对应 multipart complete；
- GetProperties 对应 stat；
- ListBlobs 对应 list；
- SDK retry 可能需要 Azure pipeline policy 支持。

### 18.5 PR 5：Scoped profile handle

增加：

- profile create/destroy/snapshot；
- execution context；
- reader/writer/transaction `_v2` APIs；
- async 显式 capture；
- storage version；
- provider coverage。

### 18.6 PR 6：Milvus request/task integration

最后再接入：

- Search/Query；
- Flush；
- Import；
- Compaction；
- Index/Analyze/Stats；
- Load；
- Recovery；
- Snapshot/GC；
- Proxy contribution merge。

## 19. 必须避免的实现方式

### 19.1 不要使用 global counter 前后差值

错误示例：

```text
before = filesystem metrics
run compaction
after = filesystem metrics
compaction usage = after - before
```

并发 Flush、Load、Search、Index 的 I/O 都会进入这个差值。

### 19.2 不要在 task 开始时 reset filesystem metrics

`reset` 会影响同一 filesystem 上的全部并发调用。它只能用于测试，不应成为生产 profile 生命周期的一部分。

### 19.3 不要修改 legacy counter 的既有含义

例如不能把旧的 S3：

```text
write_count = PutObject + UploadPart
```

静默改成：

```text
write_count = OpenOutputStream
```

应新增 versioned metric contract。

### 19.4 不要把动态值放进 Prometheus label

不应使用：

```text
bucket
object_path
manifest_path
raw_error_message
request_id
task_id
endpoint
```

### 19.5 不要为了 metrics 额外发 HEAD

如果 copy bytes 未知，应标记 unavailable。

不能为了统计 bytes 额外发送 `HeadObject`，否则：

- metrics 本身改变系统 I/O；
- 增加对象存储成本；
- 放大 latency；
- 产生递归统计和双计数问题。

## 20. 验证建议

### 20.1 Success matrix

- full GET；
- range GET；
- zero-length/EOF；
- PutObject；
- multipart upload；
- stat；
- list single page；
- list multiple pages；
- delete；
- batch delete；
- copy；
- Local/S3/Azure 基本语义。

### 20.2 Failure matrix

- object not found；
- bucket not found；
- invalid credentials；
- permission denied；
- 429 throttling；
- S3 SlowDown；
- 500/503；
- timeout；
- connection failure；
- invalid range；
- entity too large；
- multipart complete embedded error；
- cancel；
- unexpected EOF。

### 20.3 Retry matrix

Mock endpoint 可以依次返回：

```text
503
503
200
```

应验证：

```text
provider operation count = 1
retry count = 2
final outcome = success
duration 包含 backoff
retry/error category 正确
```

### 20.4 Concurrency

- 多个 filesystem clients；
- 多个 async reads；
- 同一个 cached filesystem；
- 多个 scoped profile handles；
- filesystem cache eviction；
- snapshot 与 operation 并发；
- profile destroy 前仍有 async task 的生命周期安全。

### 20.5 ABI compatibility

- 旧 Milvus + 新 milvus-storage；
- 新 Milvus + 新 milvus-storage；
- 旧 FFI snapshot 保持原行为；
- 新 struct size/version 校验；
- 未支持 schema 返回明确错误。

### 20.6 性能

需要 benchmark：

- disabled/noop recorder 开销；
- aggregate counter 开销；
- latency histogram 开销；
- scoped recorder 开销；
- 高并发 atomic contention；
- snapshot cost；
- profile handle 内存上限。

## 21. 最终建议

如果目标是尽快获得一些对象存储数据，可以先不修改 `milvus-storage`，使用现有 13 项 filesystem snapshot，在 Milvus 补齐遗漏字段并增加周期性采集。

但是，这些数据必须被称为：

```text
legacy filesystem aggregate metrics
```

不能称为完整 object-storage profiling，也不能用于 request/task attribution 或云账单核对。

如果目标是建立长期正确的基础，建议先在 `milvus-storage` 新增 versioned provider aggregate metrics，优先覆盖 S3-compatible backend，然后再增加 explicit scoped profile handle。

推荐的最小首期范围是：

```text
provider operation
+ outcome
+ error category
+ latency histogram
+ requested/completed bytes
+ inflight
+ aggregate retry
+ multipart
+ versioned FFI
```

首期不应承担：

```text
workload attribution
request/task profile
distributed merge
cache profiling
public presentation
```

最重要的架构边界是：

```text
Milvus logical layer
    负责：谁发起、什么 workload、什么 phase、什么 storage role

milvus-storage/provider layer
    负责：实际执行了哪些 provider operation、耗时、字节、重试和错误

最终 Profile
    由 Milvus 使用显式 scope handle 合并两层数据
```

这样即使撤销当前 object-storage-profiling feature branch，也可以先通过边界清晰、风险较低的 `milvus-storage` PR 建立 provider observability，再逐步恢复 Milvus request/task profiling。
