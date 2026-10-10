# Milvus Vector Raw Data Load Optimization 调研报告

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

## 1. 摘要

多数 ANN workload 只需要 PK、distance 和标量字段，并不返回 embedding。对于量化 index 和多副本部署，独立 vector field raw data 可能重新接近原始数据规模，显著增加 QueryNode 的本地磁盘、cache、恢复时间和资源预算。

当前 Milvus 已经通过以下默认组合避免在 load 阶段主动 warmup raw vector field：

```text
warmup.vectorField = disable
warmup.vectorIndex = sync
```

这能解决一个窄但重要的问题：当每个参与搜索的 sealed segment 都有可用正式 index，且执行计划不引用 vector column 时，普通 ANN Search 不会主动读取独立 raw-vector field。

但它没有给用户一个清楚、稳定的使用规则：

1. `warmup=disable` 只是“不主动预热”，不是“不可取回”；显式 vector output 仍可能从 index 反查或触发 remote lazy read。
2. “用户没有请求 vector”不等于“系统不需要 vector”。Partial upsert、未完成 index 的 segment、interim index、brute-force、Hybrid Search 次级向量字段以及 VectorArray NULL filter 都可能读取 raw vector。
3. Collection、Field、Index 级 warmup 在 collection loaded 时均不能修改；全局参数热更新也不会追溯改变既有 CacheSlot。长期业务模式切换仍需要 release/reload。
4. 当前系统没有分开处理“字段是否允许返回”和“字段什么时候加载到本地”，也无法判断所有 segment 是否已经使用新设置。

因此，本报告建议把两个问题分开设置：

```text
raw_data.enabled = true | false   # 是否允许取回 raw vector
warmup = sync | async | disable   # 允许取回后，什么时候加载到本地
```

这同时覆盖两类业务：

- 确定不返回 vector：`enabled=false`，显式 output 报错，系统清理非搜索必需的 field raw data。
- 90% 不返回、10% 偶尔返回：`enabled=true + warmup=disable`，避免频繁 alter，但接受 cold read 和 cache 增长。

在线修改建议复用 QueryCoord `Reopen` 和 Segcore `LoadDiff`，并新增 desired/applied generation。`AlterCollectionField` 成功只表示目标设置已经保存，并已通知 QueryCoord 开始更新；它不等待所有 segment 完成实际加载或清理。

在切换新 collection 默认值之前，必须先解决两个会阻塞上线的问题：

- Partial upsert 如何保留未更新的 disabled vector；
- mixed-version 集群如何保证旧 Proxy/QueryNode 不绕过这个取回开关。

## 2. 背景与目标

### 2.1 为什么优化 field raw data

向量检索通常只返回标识、距离和业务属性。原始向量更多用于调试、导出、rerank 或少量下游计算，而不是主查询路径。

独立 field raw data 的理论规模为：

```text
raw_size = row_count × dimension × bytes_per_dimension
cluster_cost ≈ raw_size × loaded_replicas
```

例如 1 亿条、768 维 FloatVector 约为 307.2 GB；两副本约为 614.4 GB，尚未计入文件格式、mmap、cache metadata 和临时加载空间。对 PQ/SQ 等压缩 index，额外 field raw data 可能远大于 index 本身，抵消 index 压缩节省的空间。

优化目标不是删除持久化数据，而是减少 QueryNode 为在线查询维护的非必要副本：

- LoadCollection 和 QN restart 的对象存储读取；
- 本地 mmap/cache 占用；
- replica 放大的磁盘成本；
- load 时较保守的资源检查；
- failover/rebalance 的恢复时间。

### 2.2 真正需要回答的问题

本调研不讨论“如何再增加一种 cache 加载选项”，而是回答三个设计问题：

1. 当前 warmup 是否已经能节省大部分加载和存储开销？
2. 用户意图与系统实际执行在哪些场景不一致？
3. 如果需要明确并且可以在线修改的规则，最少要增加哪些开关、metadata 和更新流程？

### 2.3 三种 raw data 必须区分

| 数据 | 位置 | 能否由本方案省略 |
|---|---|---|
| Field raw data | insert binlog、Storage V2/V3 column group、QueryNode field cache/mmap | 对有可用 index 的 sealed segment 可以省略 |
| Index-embedded raw/refine data | vector index files，能力由 Knowhere `HasRawData()` 等接口决定 | 通常不能单独移除 |
| Growing raw data | QueryNode growing segment | 不能省略；写入、实时搜索和 flush 需要 |

下文的“raw-vector load optimization”主要指第一类。Index file 下载量接近原始向量，不代表独立 field raw data 被加载。

---

# Part 1：当前行为、能解决的问题和不足

## 3. 当前配置

### 3.1 配置层级

当前 QueryNode 默认值为：

```text
queryNode.segcore.tieredStorage.warmup.vectorField = disable
queryNode.segcore.tieredStorage.warmup.vectorIndex = sync
queryNode.segcore.tieredStorage.warmup.scalarField = sync
queryNode.segcore.tieredStorage.warmup.scalarIndex = sync
```

Collection 级 property：

```text
warmup.vectorField
warmup.vectorIndex
warmup.scalarField
warmup.scalarIndex
```

Field 和 Index 级 property：

```text
warmup = sync | async | disable
```

优先级为：

```text
Field/Index 显式配置
  > Collection 同类默认
      > QueryNode 全局默认
```

### 3.2 warmup 实际做什么

| Policy | Load 时行为 | 后续访问 |
|---|---|---|
| `sync` | 同步 pin 全部 cache cells | 通常命中本地数据 |
| `async` | 后台 pin | 请求可能早于 warmup 完成 |
| `disable` | 只创建 translator/cache metadata，不主动 pin | 访问时仍可 remote lazy load |

`disable` 不会删除 binlog path、manifest、translator 或 lazy access path，也不禁止从 index 返回 raw vector。

这是当前行为的关键：

> Warmup 描述数据何时进入 cache，不描述用户是否有权取回字段。

## 4. 当前方案什么时候已经够用

当以下条件同时成立时，当前默认配置已经可以避免独立 raw-vector field 的主动下载：

1. 每个参与搜索的 sealed segment 都有可用正式 vector index；
2. Search/Query plan 不引用 vector column；
3. `output_fields` 不包含 vector，也没有 wildcard 展开到 vector；
4. 没有内部 retrieve、brute-force 或 interim-index build 依赖该 column；
5. Field 级 `warmup` 没有覆盖 Collection 的 `vectorField=disable`。

典型路径为：

```text
LoadCollection
  |- sync load vector index
  |- load required scalar/system data
  `- register raw-vector lazy translator, no cell pin

Search(output_fields=["title"])
  |- SearchOnSealedIndex
  |- fill PK and distance
  `- bulk_subscript(title)
```

如果业务接受以下行为，当前机制可能已经足够：

- 平时不主动下载 raw vector；
- 偶发 vector output 允许变慢并触发 lazy load；
- 改变长期策略时允许 release/reload；
- 可以接受 resource estimate 只是保守估算，不一定完全准确。

如果产品接受这些行为，短期只需要补充用户文档和 cold-cache 验证，不必立即改内核。

## 5. 用户没有请求 Vector，系统仍可能读取 Raw Data

这是现状调研中最重要的发现。`output_fields` 不包含 vector，只能证明结果回填阶段不返回 vector，不能证明完整执行路径不需要 vector column。

### 5.1 Partial upsert 内部读取全部旧字段

用户可能只更新一个标量字段：

```text
partial_upsert(pk=100, category="book")
```

当前 Proxy 为合并未更新字段，会先按 PK 发起内部强一致 Query，并硬编码：

```text
OutputFields = ["*"]
```

执行路径为：

```text
Partial Upsert
  -> Proxy queryPreExecute
      -> retrieveByPKs
          -> internal Query output_fields=["*"]
              -> retrieve old vector
                  -> index raw data, or field raw-data lazy load
```

向量不会返回给用户，但它是 Proxy 重建完整 upsert row 的输入。如果 index 不能完整返回 raw vector，内部 Query 会访问 field column，并可能触发 S3 cold read。

因此：

> `warmup.vectorField=disable` 可以避免主动 warmup，但不能阻止 scalar-only partial upsert 读取 raw vector。

这也是新取回开关不能只在 wildcard 展开时过滤 vector 的原因；否则 partial upsert 可能丢失旧向量。

### 5.2 Collection 有 index，不代表每个 segment 都 index-ready

以下 sealed segment 可能没有可用正式 index：

- 刚 flush，index build 尚未完成；
- 小 segment 未达到构建条件；
- index build 排队、失败，或者最新 metadata 还没有传到当前节点；
- Hybrid Search 的某个次级 vector field 尚未完成 index；
- load/rebalance 期间 segment 的 index state 尚未 ready。

在 interim index 开启时，`InterimSealedIndexTranslator` 直接以 vector column 为输入，并通过 `GetChunk()` 构建临时 index：

```text
sealed segment without formal index
  -> build interim index
      -> InterimSealedIndexTranslator::get_cells
          -> vector column GetChunk
              -> raw-vector cache miss
                  -> object storage read
```

如果 interim index 不可用或尚未 ready，Search 进入：

```text
SearchOnSealedColumn
```

此时即使只返回 PK 和 distance，ANN 本身也必须扫描 raw vector。

因此，评估“不会读取 raw vector”必须按：

```text
vector field × segment × index readiness
```

不能只检查 collection 是否定义了 index。

### 5.3 Field override 和 wildcard 会改变观察结果

运维看到：

```text
warmup.vectorField=disable
```

不代表每个 vector field 都是 disable。某个字段的：

```text
warmup=sync | async
```

会覆盖 Collection 默认，并在 load 或后台主动读取 raw data。

类似地：

```text
output_fields=["*"]
```

虽然没有写出 vector 名称，但 Proxy 会展开所有当前可返回字段，其中包括普通 vector field，所以它仍然会返回 raw vector。

### 5.4 VectorArray NULL filter 使用 RawData path

当前 `PhyNullExpr` 对 `VECTOR_ARRAY` 强制选择 `RawData` execution path。因此：

```text
vector_array_field IS NULL
vector_array_field IS NOT NULL
```

即使不返回 vector，也可能读取 column validity/raw chunks，并在 cache miss 时触发 remote read。

这说明取回开关不能简单等价为“字段是否出现在 output_fields”；filter 和内部 operator 也可能为了正确执行而读取这个字段。

### 5.5 Global Refine 的边界不同

Global Refine 通过：

```text
segment->CalcDistByIDs(...)
  -> vector index PinCells
      -> VectorIndex::CalcDistByIDs
```

主要使用 index 内部 raw/refiner data，不直接调用 field `get_raw_data()`。因此不能笼统地把“refine”写成独立 raw-vector field load。

从存储和网络开销看，index files 仍可能包含接近全精度向量的数据；但这属于 index 自己的数据，不是 field raw-data 设置控制的内容。

### 5.6 风险矩阵

| 用户动作或系统状态 | 可能读取独立 field raw data | 用户看到 vector |
|---|---:|---:|
| 正式 index-ready Search，只返回 PK/distance | 否，满足本节前提时 | 否 |
| Scalar-only partial upsert | 是，index 无法返回完整 raw value 时 | 否 |
| Sealed segment 无正式 index | 是，interim build 或 brute-force | 否 |
| Hybrid Search 某个 vector field 未 index-ready | 是 | 否 |
| Field `warmup=sync/async` 覆盖 Collection disable | 是 | 否 |
| `output_fields=["*"]` | 是，或从 index 反查 | 是 |
| `VECTOR_ARRAY IS NULL` | 可能 | 否 |
| Global Refine 使用 index refiner/raw data | 通常不访问独立 field | 否 |
| 仅 resource estimation | 否 | 否 |

## 6. 在线修改的限制

### 6.1 Loaded collection 不能修改 warmup

当前以下操作都会被 Proxy 拒绝：

```text
loaded collection
  -> alter collection warmup       # rejected
  -> alter field warmup            # rejected
  -> alter index warmup            # rejected
```

必须执行：

```text
release
  -> alter
      -> load
```

这会中断读取，并重新加载与目标字段无关的 index、scalar 和 system data。

### 6.2 QueryNode 全局 refresh 只影响未来对象

全局 warmup 参数可以 refresh，但 CacheSlot 在创建时已经固定 warmup 设置，且 `Warmup()` 只执行一次：

- `disable -> sync` 不会主动加载既有 lazy cells；
- `sync -> disable` 不会删除已加载数据；
- Field/Index 显式配置仍优先；
- 新设置主要影响后续创建的 segment/cache slot。

它不能让 loaded collection 的所有现有 segment 都改用新设置。

### 6.3 请求级没有 override

Search/Query/Get 和 LoadCollection 没有 per-request warmup override。请求只能通过实际访问触发 lazy load。

这反而是合理的限制：如果每个请求都能改变加载方式，cache 会频繁加载和清理。新设计应该保存一个长期有效的字段开关，再在后台更新所有 segment，而不是增加 per-request warmup。

## 7. 现状总结

| 目标 | 现有 warmup 是否足够 |
|---|---|
| Load 阶段不主动 warmup raw vector | 是 |
| 正常运行中的 indexed Search 不读取 field raw data | 满足前面列出的条件时是 |
| 允许低频 vector output 做 lazy read | 是 |
| 声明不可取回后，显式 output 必须失败 | 否 |
| 用户未请求 vector 时，系统内部也绝不读取 | 否，而且部分执行流程必须读取才能正确运行 |
| Loaded collection 在线切换策略 | 否 |
| 能查看所有副本是否已经使用同一设置 | 否 |
| 立即释放已有 raw-data cache | 否 |
| 从 resource estimate 中准确排除 raw data | 否，当前可能保守计入 |
| 按节点差异化加载并按 output 路由 | 否 |

### 7.1 何时可以停止在 Part 1

如果需求是：

> 默认不主动下载 raw vector；少量请求需要时允许 lazy read。

则现有机制基本自洽。建议动作是：

- 明确 `disable` 不是 permission；
- 审计 Field override 和 wildcard；
- 对 partial upsert、unindexed segment 做 workload 说明；
- 用 cold-cache storage profile 验证实际 GET；
- 将 resource estimation 与实际下载分开观测。

### 7.2 何时必须进入 Part 2

如果需求包含任一项：

- disabled vector output 必须报错；
- 希望给用户一个清楚、稳定的使用规则；
- loaded collection 必须在线 enable/disable；
- disable 后需要能查看本地资源的清理进度；
- 新 collection 默认不能取回 raw vector；

则 warmup 已经不能清楚表达需求，需要增加独立的取回开关。

---

# Part 2：Raw Vector 取回开关设计

## 8. 设计原则

### 8.1 分开设置“能否取回”和“何时加载”

推荐字段属性：

```text
raw_data.enabled = true | false
```

用户看到的行为是：

> 是否允许通过 Query/Get/Search output 取回该向量字段的原始值。

现有 warmup 保留为高级性能策略：

```text
warmup = sync | async | disable
```

不建议将 `enable=true` 自动映射成 `warmup=sync`。否则打开取回开关会自动触发全量预热，再次把“允许访问”和“立即加载到本地”混在一起。

### 8.2 每个 Field 单独保存设置

多向量 collection 可能只返回其中一个 vector。建议：

```text
Field-level setting = 最终保存和执行的设置
Collection-level switch = 批量设置多个 Field 的便捷接口
```

Generation、load state、资源统计和请求检查都使用 Field 的最终设置。

### 8.3 取回开关只控制用户输出，不影响系统正常运行

`enabled=false` 不应破坏：

- ANN search；
- no-index segment fallback；
- growing segment；
- 系统内部必须读取旧值的工作流。

因此，“是否允许用户取回”和“数据是否实际存在于本地”必须分开：系统可能为了正确执行而加载字段，但仍然不能把它返回给用户。

## 9. 用户使用规则

### 9.1 取回开关与 warmup 的组合

| 取回开关 | Warmup | 用户输出 | 实际行为 |
|---|---|---|---|
| disabled | 任意 | 禁止 | 有可用 index 时清理非搜索必需 field raw data |
| enabled | sync | 允许 | Reopen 完成前同步预热 |
| enabled | async | 允许 | 后台预热，短期可能 Loading |
| enabled | disable | 允许 | 建立 lazy data source，访问时 cold read |

### 9.2 Output fields

推荐规则：

| Request | Behavior |
|---|---|
| `output_fields=["vector"]` 且 disabled | InputError，客户端不应重试 |
| `output_fields=["*"]` | 展开 all retrievable fields，过滤 disabled vector |
| `output_fields=["*", "vector"]` 且 disabled | 因显式指定 vector 返回 InputError |
| `anns_field="vector"` 且 disabled | ANN search 允许 |

Wildcard 过滤与当前 BM25 function output 的先例一致，也符合会议中“`*` 去掉不可取回列”的建议。

其代价是 `*` 不再等于 schema 全字段。Response 必须返回实际 `OutputFields`，SDK 和文档应把 `*` 定义为 all retrievable fields。

### 9.3 90/10 混合 workload

对“90% 不返回、10% 偶尔返回”的场景，不应频繁 alter：

```text
raw_data.enabled = true
warmup = disable
```

它明确表示该字段可以取回，同时避免主动预热。代价是偶发请求可能较慢，并且读取后会占用 cache。

如果用户要求给本地资源设置明确上限，则必须选择 `enabled=false`，并接受偶发 output 失败。

按少量专用 QueryNode 加载 raw data、再根据 output 选择节点，是一种可能的后续优化。但它要求不同 replica 使用不同设置，Proxy 也要根据请求选择节点，还要处理不同磁盘规格和节点故障后的容量问题，因此不属于首期。

## 10. 在线修改流程

### 10.1 API 返回成功代表什么

`AlterCollectionField` 成功表示：

```text
目标设置已保存
Proxy cache expiration 已提交
QueryCoord 已接收新的 desired generation
```

不表示所有 segment 已经完成实际更新。

### 10.2 Enable

```text
Disabled
  -> Loading
      -> Loaded
```

流程：

1. 持久化 `enabled=true` 和新 generation；
2. QueryCoord 为 generation 落后的 segment 调度 Reopen；
3. QueryNode/Segcore 加载 field data，或建立后续 lazy read 使用的数据源；
4. QN 上报 applied generation；
5. 所有目标 segment 都完成更新后，状态变为 Loaded。

Loading 阶段 ANN 继续服务。Vector output 尚未 ready 时返回可重试的系统错误，而不是 InputError。

### 10.3 Disable

```text
Loaded
  -> Disabling
      -> Disabled
```

Disable 分为两步：

1. 最新 metadata 和 cache 设置生效后，拒绝新的用户 vector output；
2. Reopen 异步清理可省略的 field column、cache cells 和 mmap files。

不应等到实际数据清理完成后才禁止 output。No-index segment 即使继续保留 raw data，也必须遵守用户取回设置。

### 10.4 用 generation 记录更新进度

Schema property 只能记录用户想要的设置，不能证明各副本已经完成更新。这里建议使用两个 generation：desired generation 表示目标版本，applied generation 表示某个 segment 已经使用的版本。它们用于处理：

- 大 collection 需要在后台逐步更新；
- QN restart/failover；
- 快速 enable/disable 反转；
- Reopen 与 compaction/index update 并发；
- mixed replica readiness；
- 查询 Field 当前是否完成加载。

第一期可使用 collection-level generation；后续再优化为 field generation 或 Field 设置的 hash。

## 11. 配置如何传到所有节点

### 11.1 Proxy and RootCoord

复用 `AlterCollectionField`，但只对 `raw_data.enabled` 放开 loaded collection 修改。

需要完成：

- 只允许向量字段设置；
- 只接受明确的 true/false；
- FieldSchema 中保存最终设置；
- Proxy output 展开和检查；
- cache expiration；
- RootCoord ACK callback 把目标设置提交给 QueryCoord。

不应复用 `field.skipLoad`。Vector field 仍需要参与 index/ANN load，需求只是控制额外 field raw data 和用户 output。

### 11.2 WAL 如何记录字段设置

当前 Field alter 使用 `FieldMaskCollectionSchema`，StreamingNode 会将其视为 data schema change，并 flush/fence growing segment。

Raw-data 取回开关不改变行布局、DML encoding 或持久格式。建议新增 runtime property mask，并拆分：

```text
HasSchemaPayloadUpdate      # 更新 schema snapshot
IsDataLayoutSchemaChange   # 是否 flush/fence growing segments
```

取回开关变化只属于前者。

下面三种 version 分别记录不同事情：

| Version | Purpose |
|---|---|
| CollectionSchema.Version | 字段定义和数据布局 |
| schema barrier | 同版本 runtime property snapshot 顺序 |
| raw-data generation | QueryCoord/QN 是否已经完成实际更新 |

当前 QueryNode 已支持在 schema version 不变时，用更新的 barrier 替换 schema snapshot；它也会为 Segcore 生成不断增加的内部 version。这部分可以复用。

消息仍建议广播到 VChannels + CChannel，以获得与 DML 的顺序、ACK callback 和 CDC 复制；但不触发 growing flush。

### 11.3 QueryCoord

扩展 load config：

```protobuf
enum RawDataPolicy {
    RAW_DATA_POLICY_UNSPECIFIED = 0;
    RAW_DATA_POLICY_ENABLED = 1;
    RAW_DATA_POLICY_DISABLED = 2;
}

message LoadFieldConfig {
    int64 field_id = 1;
    int64 index_id = 2;
    RawDataPolicy raw_data_policy = 3;
}
```

Collection load meta 增加：

```text
field_raw_data_policy
raw_data_policy_generation
```

Segment distribution 增加 applied generation。SegmentChecker 在以下条件触发 Reopen：

```text
segment.applied_generation < collection.desired_generation
```

Update job 只保存目标设置、触发 checker，然后立即返回；失败由现有 scheduler 重试。

## 12. 数据加载链路

### 12.1 首次加载如何判断

统一加载判断：

```text
need_field_raw_data(field, segment) =
    no_usable_vector_index
    OR internal_read_required
    OR raw_data.enabled
```

System field 和 growing data 继续按现有规则处理。

### 12.2 如何复用 Reopen 和 LoadDiff

当前 Reopen/LoadDiff 已经支持 load、replace、lazy-load、drop，并能一次性发布新状态。但 raw-data 设置还没有参与目标 SegmentLoadInfo 的构造。

Enable：

- Storage V1：加入 `binlogs_to_load/replace`；
- Storage V2：按 child field projection load，避免共享 column group 连带加载；
- Storage V3：加入 `column_groups_to_load/replace`，或建立单字段 lazy entry；
- Warmup 决定 sync/async/disable。

Disable：

- 取消进行中的 async warmup；
- 删除可省略 field column 和 ready state；
- 清理 memory/disk cache；
- 更新资源使用统计；
- 无可用 index 时保留 Search 需要的 fallback；
- 先确认替代搜索数据源可用，再 drop field data。

### 12.3 防止旧任务覆盖新设置

每个 Reopen task 携带 generation。Publish 前校验 generation，防止旧 enable task 在新 disable 之后提交。

必须覆盖：

- enable/disable 快速反转；
- index replace/drop；
- compaction manifest update；
- release 与 async warmup；
- QN restart 后的 replay。

### 12.4 Output 检查

Proxy 做主校验，QueryNode/Segcore 做 defense-in-depth：

- disabled 但 field data 因 fallback 已经加载时仍拒绝；
- index `HasRawData=true` 不能绕过取回开关；
- old Proxy 或内部 RPC 不能绕过 QueryNode 的检查。

如果 partial update 采用 trusted read，请求中必须带有明确的 internal purpose，用来区分系统内部读取和普通用户 output，不能让用户伪造这种内部访问。

## 13. 上线前必须解决的问题

### 13.1 No-index segment

Disabled 不代表 raw data 实际上绝不存在。没有可用 index 时，应先保证 Search 正常运行，再考虑节省资源：

- 允许 interim index 或 brute-force 使用 raw vector；
- 禁止用户输出；
- 正式 index ready 后再通过 Reopen 清理。

对外承诺应是：

> 对有可用 index 的 sealed segment，不维护非搜索必需的 field raw data。

### 13.2 Partial upsert

这是 default-disabled 上线前必须先决定的问题。

方案 A：要求 partial update 请求携带 disabled vector。

- 优点：实现简单、不会增加隐藏的 remote read，节省多少资源也更容易估算。
- 缺点：scalar-only update 失去易用性，客户端重复上传大向量。

方案 B：内部 trusted read。

- 优点：保持 API 兼容。
- 缺点：需要 internal-purpose/no-store path，并把 S3 throttle、timeout、corruption 等故障引入写路径。

无论选择哪种方案，都不能让内部 `OutputFields=["*"]` 静默继承用户 wildcard 过滤结果。

### 13.3 Complex vector types

首期建议覆盖普通顶层向量：Float、Binary、Float16、BFloat16、Int8、SparseFloat。

ArrayOfVector、StructArray vector sub-field、复杂 function output 和 External Collection 保持兼容 enabled，直到解决：

- row-level reconstruction；
- nullable validity；
- parent wildcard semantics；
- grouped column projection；
- whole-struct partial update。

### 13.4 Index-embedded raw data

加载判断必须使用 IndexFactory/Knowhere 提供的实际能力，不能按 index name 硬编码。

关闭取回开关可以禁止用户输出，并清理额外 field data，但不能移除 index 搜索或 refine 所需的数据。容量统计必须分别报告 index bytes 和 field raw-data bytes。

### 13.5 Upgrade and rollback

推荐：

```text
existing collection + missing property -> enabled
new supported vector field             -> eventually default disabled
```

不能直接把 missing 解释为 disabled，否则升级会改变现有 output、wildcard 和 partial update。

Rollout 分两阶段：

1. 发布协议、metadata、QueryNode 检查和 generation，默认保持 enabled；
2. 确认所有 Proxy/QN 支持后，再切换新 collection 默认值。

旧版本忽略未知属性时会重新加载 raw data，优化丢失但数据不丢失；因此不能在 disable 时删除对象存储 binlog。

## 14. 错误、监控和上线步骤

### 14.1 错误分类和监控

用户显式请求 disabled vector 时返回 InputError，客户端不应重试。Enabled 但 Reopen 未完成，或者发生 S3 timeout/throttle、OOM、磁盘不足时，返回可重试的系统错误。数据缺失或损坏返回 DataIntegrity/IO 错误。代码增加错误说明时，必须保留原来的错误类型和 code。

监控至少要区分目标设置、applied generation、field data 是否 ready、index `HasRawData`、field raw-data bytes、no-index fallback 和 Reopen result。Resource estimate、CacheSlot metadata 与实际 object-storage GET 不能混为一谈。

### 14.2 上线顺序

| 步骤 | 主要工作 | 完成标准 |
|---|---|---|
| Baseline | 测量 field raw-data bytes、实际 GET、vector output/partial-upsert/no-index 比例 | 确认可以节省多少资源以及主要 workload |
| User rules | Property、包含 unspecified/enabled/disabled 的 enum、Proxy/QN 检查；默认仍 enabled | mixed-version 不绕过取回开关 |
| Data loading | V1/V2/V3 initial load、resource estimate | disabled 且 index-ready 时可省略 field data |
| Online update | Runtime WAL mask、generation、Reopen/LoadDiff、load state | loaded collection 无需 release 即可完成更新 |
| Default switch | 新 collection 默认 disabled | partial upsert、complex type、upgrade gates 已关闭 |

### 14.3 验证要求

验证矩阵至少覆盖 V1/V2/V3、index raw true/false/no-index、multi-vector Hybrid Search、在线 enable/disable 快速切换、partial upsert、replica failover 和 rolling upgrade。S3 throttle/timeout、missing/corrupt object、OOM、disk full、cancel 必须从错误产生的位置一直检查到最终 Input/System 和 retriable 分类，并确认 C++ `ThrowInfo`、Arrow Status 和 cgo translation 没有丢失 typed code。

## 15. 建议

建议按以下顺序决策：

1. 业务是否接受偶发 vector output 的 lazy read？如果接受，当前 warmup 配置加文档即可解决主要目标。
2. 是否需要保证 disabled output 一定报错？如果需要，采用独立 `raw_data.enabled`，不要改变 warmup 的含义。
3. Wildcard 采用 all retrievable fields，显式 disabled field 报错。
4. Online alter 使用 desired/applied generation 在后台更新，不阻塞 DDL 等待所有 segment 完成。
5. Partial upsert 在默认切换前选择 request-complete 或 trusted-read 方案。
6. 首期只覆盖普通顶层向量，不实现按 QueryNode 差异化路由。

目标态建议为：

```text
raw_data.enabled controls whether raw vector can be returned
warmup controls when raw vector is loaded locally
QueryCoord generation tracks whether all segments are updated
Reopen/LoadDiff changes the actual loaded data
```

这不是新的 lazy-load feature。它只是把“这个字段是否允许返回原始向量”变成一个会保存、可以查看状态、并能在重启后恢复的 Field 设置。

## 16. 关键代码

| Concern | Code |
|---|---|
| warmup keys/defaults | `pkg/common/common.go`, `pkg/util/paramtable/component_param.go` |
| loaded alter restrictions | `internal/proxy/task.go`, `internal/proxy/task_index.go` |
| output expansion | `internal/proxy/util.go`, `pkg/util/typeutil/schema.go` |
| partial upsert internal wildcard | `internal/proxy/task_upsert.go` |
| field alter WAL path | `internal/rootcoord/ddl_callbacks_alter_collection_field.go` |
| Streaming schema-change gate | `pkg/streaming/util/message/messageutil/header.go` |
| Streaming flush/fence | `internal/streamingnode/server/wal/interceptors/shard/shards/shard_manager_collection.go` |
| QueryCoord load config/checker | `internal/querycoordv2/job/load_config.go`, `internal/querycoordv2/checkers/segment_checker.go` |
| QueryNode Reopen | `internal/querynodev2/segments/segment.go` |
| resource estimation | `internal/querynodev2/segments/segment_loader.go` |
| interim sealed index | `internal/core/src/segcore/storagev1translator/InterimSealedIndexTranslator.cpp` |
| vector search/output | `internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp` |
| LoadDiff | `internal/core/src/segcore/SegmentLoadInfo.cpp` |
| VectorArray NULL path | `internal/core/src/exec/expression/NullExpr.cpp` |
| CacheSlot warmup | `internal/core/output/include/cachinglayer/CacheSlot.h` |

## 17. 相关调研

- [Milvus 向量字段 Warmup、Raw Vector 加载与首次查询延迟分析](20260716-vector-warmup-raw-data-load-report_cn.md)
- [向量索引原始数据加载逻辑优化调研](20260714-vector-index-raw-data-loading-optimization.md)
