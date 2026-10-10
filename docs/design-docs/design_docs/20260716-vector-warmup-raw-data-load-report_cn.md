# Milvus 向量字段 Warmup、Raw Vector 加载与首次查询延迟分析

- 日期：2026-07-16
- 代码基线：`4f730d9985`
- 分支：`feat.metrics_storage_access`
- 范围：QueryCoord、QueryNode v2、Segcore、Tiered Storage caching layer
- 主题：完整向量索引场景下，如何避免独立 raw vector field 从对象存储下载；warmup 的配置层级、首次查询影响、多向量字段配置和运行时修改限制

## 1. 执行摘要

在 sealed segment 已有完整且可用的向量索引，并且 Search 不返回原始向量字段、也不需要外部 raw vector 做 refine 或 brute-force fallback 时，以下配置可以避免 QueryNode 在 Collection load 和普通 ANN Search 中主动将独立 raw-vector binlog 从 S3 下载到本地：

```text
warmup.vectorField = disable
warmup.vectorIndex = sync
```

为避免第一次 Search 因 PK、过滤字段或输出标量字段发生对象存储冷读，通常还应保持：

```text
warmup.scalarField = sync
warmup.scalarIndex = sync
```

这四项也是当前 QueryNode 的默认 warmup 组合。

需要明确以下边界：

1. Search/LoadCollection 的字段和索引加载发生在 **QueryNode**，不是 DataNode。DataNode 的写入、flush、compaction、import 等存储链路不受 QueryNode warmup 控制。
2. `vectorField=disable` 只禁止主动 warmup。QueryNode 仍保存 binlog 路径和 lazy translator；未来真正访问向量字段时仍可按需下载。
3. 该配置避免的是独立 raw-vector field binlog 的下载，不保证索引文件内部不包含全精度向量。FLAT 或其他携带 raw data 的索引仍可能接近原始向量规模。
4. `vectorIndex=sync` 把索引 I/O 放在 Collection load 阶段，从而避免把索引冷加载延迟推迟到第一次 Search。
5. Collection、Field 和 Index 级 warmup 属性不能在 Collection loaded 时修改；需要 `release -> alter -> load`。
6. QueryNode 全局 warmup 参数支持动态刷新，但不会追溯改变已创建 CacheSlot 的状态，主要影响后续新加载的 segment/cache slot。
7. 多向量字段可以使用 Field 级 raw-data warmup 和 Index 级 warmup 分别覆盖 Collection 默认值。
8. 即使最终响应不包含向量，partial upsert、尚未建好正式索引的 sealed segment、interim index 构建，以及某些直接引用向量列的表达式，仍可能触发 QueryNode 读取 raw vector。

## 2. 组件边界：DataNode 与 QueryNode

### 2.1 查询侧

Collection load、sealed segment 恢复和 ANN Search 的对象存储读取由 QueryNode 执行：

```text
Proxy
  -> QueryCoord
      -> QueryNode LoadSegments
          -> Segcore CacheSlot / Translator
              -> S3 或其他对象存储
```

QueryNode 可能加载：

- vector index；
- scalar index；
- PK、时间戳和删除信息；
- 过滤或输出涉及的 scalar field；
- 只有在需要时才加载的 raw vector field。

### 2.2 存储侧

DataNode 在以下场景仍需要处理原始向量：

- 接收写入和维护 growing segment；
- flush insert binlog；
- compaction；
- import；
- 数据恢复或 binlog 重写。

因此，QueryNode 的 warmup 配置不能实现“DataNode 永远不读取原始向量”。本报告中的“避免从 S3 下载 raw vector”专指查询侧 QueryNode。

## 3. Warmup 的真实语义

CacheSlot 创建后会根据 translator 中解析出的 warmup policy 执行一次 `Warmup()`：

| 策略 | CacheSlot 行为 | 对 Collection load 的影响 | 对首次访问的影响 |
|---|---|---|---|
| `sync` | 同步 pin 全部 cell | load 等待数据加载完成 | 首次访问通常为 cache hit |
| `async` | 后台 pin 全部 cell | load 可以先完成 | 请求先到时可能等待后台加载 |
| `disable` | 直接返回，不 pin cell | 只创建 translator/cache metadata | 第一次真正访问时按需加载 |

关键实现：

```cpp
switch (warmup_policy) {
    case CacheWarmupPolicy::CacheWarmupPolicy_Disable:
        return;
    case CacheWarmupPolicy::CacheWarmupPolicy_Async:
        // background PinCellsDirect(...)
    case CacheWarmupPolicy::CacheWarmupPolicy_Sync:
        PinCellsDirect(...);
}
```

因此，`warmup.vectorField=disable` 时，创建 raw vector column 的 translator 并不等于已经下载向量字节。此时本地主要存在：

- binlog 路径；
- row count、文件大小等元数据；
- cell 划分；
- mmap 策略；
- lazy-load handle。

只有后续 `PinCells` 命中相关向量 cell 时，才会发生对象存储读取、解码以及可能的 mmap 文件写入。

## 4. 完整索引 Search 为什么不读取 Raw Vector Field

sealed segment 搜索首先检查正式 index 是否 ready：

```text
index_ready
  -> SearchOnSealedIndex
      -> PinCells(vector index cell)
      -> vector index Search
```

只有没有可用 index 时，才进入 column/brute-force 路径：

```text
index not ready
  -> get vector column
      -> SearchOnSealedColumn
```

因此在正式索引完整可用时，ANN 阶段 pin 的是 vector index，不是独立 raw vector column。

Search 的结果字段采用 late materialization：先由索引得到 segment offset 和 distance，再只对计划中的目标字段执行 `bulk_subscript`。如果 `output_fields` 不包含向量字段，结果回填阶段也不会访问 raw vector field。

典型路径如下：

```text
LoadCollection
  |- sync load vector index
  |- sync load PK / common scalar data and indexes
  `- register raw vector lazy translator, no raw-vector cell pin

First Search
  |- predicate on warmed scalar field/index
  |- ANN on warmed vector index
  |- fill PK
  `- fill requested scalar output fields
```

在这一限定路径中，raw vector field 不参与执行。

## 5. 推荐配置与每项配置的职责

### 5.1 Collection 级推荐值

```text
warmup.vectorField = disable
warmup.vectorIndex = sync
warmup.scalarField = sync
warmup.scalarIndex = sync
```

| 配置 | 目的 |
|---|---|
| `vectorField=disable` | 不主动下载独立 raw vector field |
| `vectorIndex=sync` | Collection load 完成前准备好 ANN index |
| `scalarField=sync` | 避免 PK/输出标量字段第一次访问发生冷读 |
| `scalarIndex=sync` | 避免过滤或 PK index 第一次访问发生冷读 |

### 5.2 mmap 的作用

```text
queryNode.mmap.vectorField = true
```

mmap 决定数据真正加载后的存放和访问方式，不决定是否主动加载：

```text
warmup=disable  -> 当前不主动下载
mmap=true       -> 以后需要时倾向写入本地 mmap 并使用 OS page cache
```

因此 mmap 是内存与本地磁盘之间的放置策略，不是 remote-load 开关。

### 5.3 PreferFieldDataWhenIndexHasRawData

```text
queryNode.preferFieldDataWhenIndexHasRawData = false
```

默认值为 `false`。当 index 自身可以提供 raw data 时，QueryNode 默认跳过独立 field data 加载，避免 index raw data 与 field raw data 同时驻留。设置为 `true` 可能让两份数据同时存在，增加内存或本地存储占用。

## 6. 第一次 Search 是否会很慢

仅关闭 raw vector warmup，不会自动导致第一次 ANN Search 读取 raw vector。

第一次 Search 是否产生其他 S3 冷读，取决于它实际使用的其他数据：

### 6.1 只返回 PK 和 distance

Distance 是 vector index 搜索计算结果，不是存储列。Search 还需要 PK：

- INT64 PK 可以走压缩 PK index 快路径；
- VARCHAR PK 可能读取 PK column；
- system PK index 和 scalar field/index 默认均为 `sync` warmup。

因此，在默认完整 load 和推荐 warmup 组合下，第一次 Search 通常不会因结果回填发生 remote cold read。

### 6.2 返回标量字段

例如：

```text
output_fields = ["title", "category"]
```

QueryNode 会在 ANN 得到最终 offset 后读取这些字段。如果字段已经同步 warmup，则首次 Search 不需要去 S3；如果字段处于 lazy 状态，则可能冷读。

### 6.3 会触发首次冷读的主要条件

- `warmup.vectorIndex=disable`；
- `warmup.vectorIndex=async` 且请求早于后台 warmup 完成；
- `warmup.scalarField=disable/async`；
- `warmup.scalarIndex=disable/async`；
- partial load 没有包含过滤字段或输出字段；
- 请求返回低频 dynamic/JSON/标量字段；
- cache 已被 eviction；
- Search 使用需要独立 raw vector 的 refine/rerank；
- 正式 index 不可用，需要 column/brute-force fallback。

### 6.4 `load_fields` 的要求

如果使用 partial load，建议包含：

```text
PK
ANN vector field
所有常用 filter 字段
所有常用 output scalar 字段
```

Milvus 要求 load field list 至少包含 PK 和一个 vector field。`load_fields` 更接近加载/warmup 提示，不是严格的字段访问权限；未主动加载的字段仍可能在请求时 lazy load。

## 7. 配置层级与优先级

### 7.1 Raw field data

普通字段数据的 warmup 优先级是：

```text
Field 级 warmup
  > Collection 级 warmup.vectorField / warmup.scalarField
      > QueryNode 全局 field warmup
```

Field 级属性键为：

```text
warmup = sync | async | disable
```

### 7.2 Index

普通 vector/scalar index 的 warmup 优先级是：

```text
Index params 中的 warmup
  > Collection 级 warmup.vectorIndex / warmup.scalarIndex
      > QueryNode 全局 index warmup
```

Field data 和普通 vector index 在当前加载路径中分别传播。Field 级 `warmup` 用于字段数据；Index 级 `warmup` 用于对应 index。不要仅根据 `Schema::WarmupPolicy(..., is_index)` 的通用接口推断普通 vector index 的最终策略，应以 QueryCoord 写入 field TypeParams/index params，以及 QueryNode 实际解析路径为准。

## 8. 多向量字段的独立配置

假设 Collection 包含：

```text
image_vector
text_vector
audio_vector
```

Collection 级：

```text
warmup.vectorField = disable
warmup.vectorIndex = sync
```

会对所有 vector field/index 提供默认值，不能使用以下不存在的键：

```text
warmup.vectorField.image_vector
warmup.vectorIndex.text_vector
```

如果需要逐字段区分，应组合使用 Field 和 Index 覆盖：

```text
Collection defaults:
  warmup.vectorField = disable
  warmup.vectorIndex = sync

Field overrides:
  image_vector.warmup = sync
  text_vector.warmup  = disable
  audio_vector.warmup = async

Index overrides:
  image_vector_idx.warmup = sync
  text_vector_idx.warmup  = sync
  audio_vector_idx.warmup = async
```

结果示例：

| 向量字段 | 独立 raw field data | 对应 vector index |
|---|---|---|
| `image_vector` | load 时同步加载 | load 时同步加载 |
| `text_vector` | lazy | load 时同步加载 |
| `audio_vector` | 后台加载 | 后台加载 |

Index 创建时可以携带 `warmup` 参数；已有 index 可以通过 AlterIndexProperties 修改 `warmup`，但修改时 Collection 必须处于 released 状态。

## 9. 运行时修改与刷新限制

### 9.1 Collection 级

以下属性不能在 Collection loaded 时修改：

```text
warmup.vectorField
warmup.vectorIndex
warmup.scalarField
warmup.scalarIndex
```

修改流程：

```text
release_collection
  -> alter collection properties
      -> load_collection
```

### 9.2 Field 级

Field 的 `warmup` 也不能在 Collection loaded 时修改：

```text
release_collection
  -> alter field property warmup
      -> load_collection
```

### 9.3 Index 级

AlterIndexProperties 同样拒绝修改 loaded Collection 的 index：

```text
release_collection
  -> alter index property warmup
      -> load_collection
```

### 9.4 QueryNode 全局级

以下参数标记为 refreshable：

```text
queryNode.segcore.tieredStorage.warmup.vectorField
queryNode.segcore.tieredStorage.warmup.vectorIndex
queryNode.segcore.tieredStorage.warmup.scalarField
queryNode.segcore.tieredStorage.warmup.scalarIndex
```

运行时修改会更新 QueryNode 全局默认值，但 CacheSlot 在创建时已保存解析后的 policy，而且 `Warmup()` 只调用一次。因此：

- `sync -> disable` 不会删除已经加载的数据；
- `disable -> sync` 不会立即加载已有 lazy cell；
- 已存在的 Collection/Field/Index 显式属性继续覆盖全局值；
- 新加载的 segment/cache slot 才会稳定使用新默认值。

如果要求整个 Collection 一致应用新策略，仍应 release/reload。

### 9.5 请求级

当前 Search/Query/Get 和 LoadCollection 请求不提供 per-request warmup override。请求只通过实际访问行为触发 lazy load，例如将向量字段放入 `output_fields`。

## 10. 配置能解决与不能解决的问题

| 问题 | 仅靠 warmup 配置能否解决 |
|---|---|
| LoadCollection 主动下载独立 raw vector field | 可以，设置 `vectorField=disable` |
| 普通 indexed Search 访问 raw vector field | 在不输出、不 refine、不 fallback 的条件下可以避免 |
| 第一次 Search 冷加载 vector index | 可以，设置 `vectorIndex=sync` |
| 第一次 Search 冷读常用 scalar field/index | 可以，通过 scalar sync warmup 和正确的 `load_fields` 缓解 |
| 删除 raw-vector binlog 和 lazy access path | 不可以 |
| 阻止 DataNode compaction 等读取 raw vector | 不可以 |
| 保证 index 内部不含全精度向量 | 不可以，取决于 index type |
| 立即清理已经存在的本地 raw-vector cache | 不可以，需 release/eviction/正常清理 |
| 完全消除 load resource estimation 中的 raw-vector 预算 | 不可以，当前估算仍可能保守计入 |
| 在 loaded Collection 上原地改变 Field/Index warmup | 不可以 |

## 11. 没有输出 Raw Vector，但仍可能触发读取的场景

`output_fields` 不包含向量，只能证明“结果回填阶段不需要把向量返回给用户”，不能证明整个请求和内部工作流都不需要向量。应按照实际执行阶段判断是否会 pin raw-vector cell。

下面区分三类行为：

1. **确定会访问独立 raw-vector field 的路径**；
2. **满足条件时才访问独立 raw-vector field 的路径**；
3. **只访问 index 内部 raw/refine 数据，不访问独立 field binlog 的路径**。

### 11.1 Partial upsert 内部回读全部旧字段

这是最容易被业务侧忽略的场景。

用户可能只更新一个标量字段：

```text
partial_upsert(
    pk=100,
    category="book"
)
```

请求没有返回值，也没有显式读取 vector。但当前 Proxy 为了把未更新字段与旧记录合并，会先按照 PK 发起内部 Query，并硬编码：

```text
OutputFields = ["*"]
```

实际路径：

```text
Partial Upsert
  -> Proxy queryPreExecute
      -> retrieveByPKs
          -> internal Query output_fields=["*"]
              -> QueryNode retrieve all old fields
                  -> vector index raw data, or raw-vector field lazy load
```

如果对应 vector index 能直接提供 raw data，QueryNode 可以从 index 取向量；如果 index 不携带 raw data，则内部 retrieve 会调用 raw field column，进而触发 S3 cold read。向量数据只用于 Proxy 内部重建完整 upsert row，不会作为 partial-upsert 响应暴露给用户。

因此：

> `warmup.vectorField=disable` 可以避免主动加载，但不能阻止 partial upsert 的内部 wildcard retrieve 在需要时触发 lazy load。

从优化角度看，若业务大量使用只更新标量字段的 partial upsert，真正的解决方案应是让内部 retrieve 只读取“本次未提供且合并确实需要的字段”，而不是固定读取 `*`。

### 11.2 Collection 有 index 定义，但某些 sealed segment 尚无正式 index

“Collection 已创建 vector index”不等于“每一个被搜索的 sealed segment 都已经 `index_ready`”。以下情况可能出现未索引 segment：

- 刚 flush 的新 segment，IndexNode 尚未完成构建；
- 小 segment 未达到正式 index 构建条件；
- index build 正在排队或失败后等待恢复；
- 强一致性或较新的 guarantee timestamp 让 Search 覆盖刚生成的数据；
- rebalance/load 期间正式 index metadata 尚未进入当前 segment 状态。

对于这些 segment，QueryNode 有两种执行方式。

#### 构建 interim/binlog index

默认开启 interim index 时，QueryNode 会创建 `InterimSealedIndexTranslator`。该 translator 以 vector column 为数据源，构建过程中调用 `GetChunk()` 读取原始向量：

```text
new sealed segment without formal index
  -> create interim index CacheSlot
      -> vectorIndex warmup=sync
          -> InterimSealedIndexTranslator::get_cells
              -> vec_data_->GetChunk(...)
                  -> raw-vector cell miss
                      -> S3 download
```

这个过程的目标是建立临时 ANN index，看起来是“index 准备”，但输入仍然是 raw vector field。

#### Brute-force Search

如果 interim index 未启用、不支持或尚未 ready，sealed search 会进入：

```text
SearchOnSealedColumn
```

此时 ANN 计算直接扫描 vector column，即使用户只返回 PK 和 distance，也必须读取 raw vector。

因此，避免该路径要求“每个参与搜索的 sealed segment 都有可用正式 index”，不能只检查 Collection 层是否存在 index definition。

### 11.3 Hybrid Search 的次级向量字段没有 ready index

Hybrid Search 可以包含多个 ANN sub-search。最终只向用户返回统一排名后的 PK、distance 和标量字段，但每个 sub-search 都需要执行自己的向量检索。

例如：

```text
sub-search 1: image_vector，正式 index ready
sub-search 2: text_vector，正式 index 尚未 ready
final output: id, title
```

虽然最终输出没有任何 vector，`text_vector` 子搜索仍可能：

- 构建 interim index并读取 raw vector；
- 或通过 `SearchOnSealedColumn` brute-force 扫描 raw vector。

多向量 Collection 的验证应按“字段 × segment”检查 index readiness，不能只确认主向量字段或其中一个 index 已完成。

### 11.4 Vector field 的 Field 级 warmup 覆盖了 Collection 默认值

运维人员可能看到 Collection 配置：

```text
warmup.vectorField=disable
```

但某个 vector field 以前设置了：

```text
warmup=sync
```

Field 级优先于 Collection 级。该字段仍会在 segment load 时主动下载 raw vector，即使没有任何 Search 输出它。

类似地：

```text
warmup=async
```

会让 Collection 很快进入 loaded，但 raw vector 随后在后台下载。业务可能把后台 S3 流量误认为第一条 Search 或 rebalance 触发。

排查时必须同时查看：

- Collection properties；
- 每个 vector field 的 TypeParams/properties；
- QueryNode 全局 fallback。

### 11.5 用户使用 wildcard，但没有显式写出 vector field 名称

以下请求没有显式列出向量字段：

```text
output_fields=["*"]
```

但 Proxy 会把 `*` 展开为 schema 中的字段，包括可返回的 dense vector fields。因此它在语义上仍然属于“请求输出 raw vector”，可能触发 requery 和 raw-vector lazy load。

这和 partial upsert 的区别是：

- 普通 Search/Query 的 `*` 会把向量暴露给用户；
- partial upsert 的内部 `*` 只用于合并旧记录，不暴露给用户。

审计请求时不能只搜索具体 vector field 名称，还应检查 wildcard。

### 11.6 对 Vector Array 执行 NULL 判断

当前 `PhyNullExpr` 对 `VECTOR_ARRAY` 强制选择 RawData execution path。类似过滤：

```text
vector_array_field IS NULL
vector_array_field IS NOT NULL
```

虽然没有返回向量值，但执行器需要读取对应 column 的 validity/raw chunk 信息。如果 cell 尚未缓存，可能触发对象存储 cold read。

普通标量字段的 NULL 判断只读取对应标量列或索引；这里的风险来自表达式直接引用了 vector-array column。

### 11.7 Index 本身携带 Raw Data

`warmup.vectorField=disable` 只控制独立 field binlog。某些 index 类型本身保存全精度向量或能够通过 `GetVector/CalcDistByIDs` 返回、使用向量数据。

因此，以下行为可能下载与 raw vector 规模相近的数据，但对象存储路径属于 index files：

- 加载 FLAT 或其他携带 raw data 的 vector index；
- 加载包含 data-view/refiner 数据的 index；
- index 内部执行 exact-distance refine。

这不属于 raw-vector field lazy load，但从网络、本地磁盘和容量角度仍可能表现为“下载了原始向量”。评估压缩收益时必须查看实际 index type 和 index files，而不能只看 `vectorField=disable`。

### 11.8 Global Refine/Rerank 的边界

当前 Global Refine 路径调用：

```text
segment->CalcDistByIDs(...)
  -> vector index PinCells
      -> VectorIndex::CalcDistByIDs
```

它没有直接调用独立 raw field 的 `get_raw_data()`。因此不应笼统地写成“开启 refine 一定加载 raw-vector field”。更准确的描述是：

- 如果 index 自身携带或 lazy-load refiner/raw data，refine 会使用这些 index files；
- 如果某种执行模式明确退化到 column/brute-force，才会访问独立 raw-vector field；
- Proxy function-chain rerank 当前只支持可转换成 Arrow 的标量/Text 输入，不支持直接把 dense vector field 当作 L2 rerank输入。

### 11.9 通常不会触发独立 Raw Vector Field 的操作

在正式 index ready 且没有上述特殊条件时，以下操作本身通常不会读取独立 raw vector：

- 只返回 PK 和 distance 的普通 ANN Search；
- 只输出已 warmup scalar fields；
- 只使用 scalar filter/scalar index；
- delete by PK 或基于标量表达式查找 PK；
- Collection rebalance/restart，且 vector field warmup 为 `disable`；
- 仅进行 resource estimation。

其中 resource estimation 可能把 raw-vector 大小计入预算，但不会因此产生实际 S3 GET。

### 11.10 风险总结

| 用户动作或系统状态 | 是否可能拉取独立 raw-vector field | 用户是否看到 vector |
|---|---:|---:|
| 普通 indexed Search，只返回 PK/distance | 否，满足本文前提时 | 否 |
| Partial upsert，只更新标量字段 | 是，index 不能提供 raw data 时 | 否 |
| 新 sealed segment 尚无正式 index | 是，interim build 或 brute force | 否 |
| Hybrid Search 的某个次级 vector field 未建好 index | 是 | 否 |
| Field 级 `warmup=sync/async` 覆盖 Collection disable | 是 | 否 |
| `output_fields=["*"]` | 是或从 index 取 raw | 是 |
| `VECTOR_ARRAY IS NULL/IS NOT NULL` | 可能 | 否 |
| Global Refine，index 自带 refiner/raw data | 不一定访问独立 field；可能加载 index 内部数据 | 否 |
| 仅 resource estimation | 否 | 否 |

## 12. 资源估算与实际下载需要分开观察

当前 segment loader 在 tiered eviction 未启用时，可能仍把 vector binlog 大小计入预计内存或 mmap 磁盘需求。这可能造成：

- load concurrency 更低；
- capacity admission 更保守；
- 量化 index 很小，但调度估算仍接近 raw vector 规模。

这不等于 raw vector 已经从 S3 下载。排查时应区分：

1. load resource estimation；
2. CacheSlot metadata overhead；
3. 实际 S3 GET；
4. 本地 mmap 文件；
5. QueryNode RSS 和 OS page cache。

## 13. 建议的线上验证方法

为了验证某个部署中确实没有下载独立 raw-vector binlog，建议使用冷 QueryNode 或清洁的本地缓存环境：

1. 确认正式 vector index 已完成并可用。
2. Release Collection。
3. 设置 Collection 属性：

   ```text
   warmup.vectorField=disable
   warmup.vectorIndex=sync
   warmup.scalarField=sync
   warmup.scalarIndex=sync
   ```

4. 重新 Load Collection。
5. 观察对象存储 GET 路径：应看到 index files 和必要 system/scalar data，不应看到 ANN vector field 的 insert binlog 被字段 warmup 下载。
6. 发起只返回 PK、distance 和已 warmup scalar fields 的 Search。
7. 检查 Search storage profile、对象存储访问日志和 QueryNode 本地 cache 文件变化。
8. 再发起一次显式返回 vector field 的请求作为对照；此时若 index 不能提供 raw data，应观察到相关 vector cell 的 lazy cold read。
9. 发起一次只更新标量字段的 partial upsert，检查内部 `UpsertQueryLabel` retrieve 是否读取 vector binlog。
10. 构造一个刚 flush、正式 index 尚未 ready 的 sealed segment，分别检查 interim-index build 和 brute-force Search 的对象存储访问。

需要注意：已有本地 cache 会掩盖 cold-read 行为；只观察 Search latency 也不能证明是否发生 S3 GET。

## 14. 验证结果

本次分析同时进行了代码路径交叉检查和针对性单测验证。

通过的单测：

```text
internal/querynodev2/segments:
  TestGetFieldWarmupPolicy
  TestGetIndexWarmupPolicy

internal/querycoordv2/task:
  TestApplyCollectionWarmupSettingAutoWarmup
  TestApplyIndexWarmupSettingAutoWarmup

internal/proxy:
  TestTranslateOutputFields
```

测试使用仓库要求的参数：

```text
-tags dynamic,test
-gcflags="all=-N -l"
-count=1
```

这些测试验证：

- field TypeParams 和 QueryNode 全局 field warmup 的解析；
- index params 和 QueryNode 全局 index warmup 的解析；
- Collection 级 field warmup 向 schema field TypeParams 的传播；
- Collection 级 index warmup 向 segment index params 的传播。
- wildcard `*` 会展开为实际 schema fields，包括可返回的 dense vector fields。

另外通过源码端到端追踪确认：

- partial upsert 的 `retrieveByPKs` 使用内部 `OutputFields=["*"]`；
- `InterimSealedIndexTranslator::get_cells` 通过 vector column 的 `GetChunk()` 取得构建输入；
- 无 ready index 时 sealed search 进入 `SearchOnSealedColumn`；
- `VECTOR_ARRAY` 的 NULL expression 强制使用 RawData execution path；
- Global Refine 的当前实现调用 vector index 的 `CalcDistByIDs`，没有直接进入独立 field column 的 `get_raw_data()`。

本次没有执行真实 S3 的端到端故障注入或网络抓包，因此“没有实际 GET”仍建议按照第 13 节在目标部署上进行 cold-cache 验证。

## 15. 关键源码

- QueryNode 默认 warmup 配置：`pkg/util/paramtable/component_param.go`
- 配置示例：`configs/milvus.yaml`
- Collection/Field warmup 校验和 loaded 状态限制：`internal/proxy/task.go`
- Index warmup 校验和 loaded 状态限制：`internal/proxy/task_index.go`
- Partial upsert 内部 wildcard retrieve：`internal/proxy/task_upsert.go`
- Wildcard output field 展开：`internal/proxy/util.go`
- Collection warmup 向 field/index 的传播：`internal/querycoordv2/task/utils.go`
- QueryNode field/index warmup 解析：`internal/querynodev2/segments/utils.go`
- QueryNode index load 参数：`internal/querynodev2/segments/segment.go`
- sealed segment index/column Search 分支：`internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp`
- index Search：`internal/core/src/query/SearchOnSealed.cpp`
- Global Refine reduce 路径：`internal/core/src/segcore/reduce/Reduce.cpp`
- Vector Array NULL expression：`internal/core/src/exec/expression/NullExpr.cpp`
- field/index translator warmup 解析：`internal/core/src/segcore/Utils.cpp`
- raw field translator：`internal/core/src/segcore/storagev1translator/ChunkTranslator.cpp`
- interim index 对 raw vector column 的访问：`internal/core/src/segcore/storagev1translator/InterimSealedIndexTranslator.cpp`
- index translator：`internal/core/src/segcore/storagev1translator/SealedIndexTranslator.cpp`
- CacheSlot warmup 行为：`internal/core/output/include/cachinglayer/CacheSlot.h`
- conservative load resource estimation：`internal/querynodev2/segments/segment_loader.go`

## 16. 最终判断

对于“完整向量索引、Search 不返回 raw vector、只需要 ANN index”的业务场景，Collection 级：

```text
warmup.vectorField=disable
warmup.vectorIndex=sync
```

可以避免 QueryNode 在正常 load/search 路径中主动下载独立 raw-vector field，并避免将 vector index 冷加载推迟到第一条 Search。再配合 scalar field/index 的同步 warmup以及正确的 `load_fields`，第一次 Search 通常也不需要为了 PK、过滤字段或常用输出字段访问 S3。

该结论只覆盖普通、正式 index-ready 的 ANN Search。它不扩展到 DataNode 存储链路、partial upsert 的内部 wildcard retrieve、尚无正式 index 的 sealed segment、interim index 构建、Hybrid Search 中未索引的次级 vector field、Vector Array NULL 表达式、Field 级 warmup 覆盖、索引内部自带 raw/refiner data，以及未 warmup 的低频输出字段。
