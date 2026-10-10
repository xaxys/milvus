# 向量索引原始数据加载逻辑优化调研

> 状态：调研设计稿（待产品语义确认）
>
> 日期：2026-07-14
>
> 范围：Milvus 3.x，重点覆盖 Proxy、RootCoord、Streaming WAL、QueryCoord、QueryNode/Segcore、SDK/REST、升级兼容与 CI
>
> 术语约定：本文中的“原始向量数据”主要指 sealed segment 中可用于返回向量字段值的 field/binlog/column data；它不等同于索引文件内部为搜索算法保留的 raw data。

## 1. 摘要与结论

本调研的核心结论是：当前问题不能仅通过修改 `warmup` 默认值解决。`warmup=disable` 的含义是“不主动把缓存 cell 预热到本地内存或磁盘”，并不表示“该向量字段不具备原始数据访问能力”。当前系统仍会为字段建立可按需读取的数据源，在首次查询、search requery、query/get、partial update 等路径上触发远端读取或本地缓存填充。

因此，建议把两个概念明确拆开：

1. **能力开关**：用户是否允许读取某个向量列的原始值，以及 QueryNode 是否需要为该字段维护原始数据访问能力。
2. **驻留策略**：当能力开启后，原始数据采用 `sync`、`async` 还是 `disable` warmup。

建议引入向量字段级持久属性，本文暂命名为：

```text
raw_data.enabled = true | false
```

推荐语义如下：

- `false`：不允许用户通过 query/get/search output field 返回该字段的原始向量；对已有可用向量索引的 sealed segment，不加载或缓存额外的 field raw data。
- `true`：允许返回原始向量；具体是否在 load 阶段同步/异步加载，继续由现有 `warmup` 决定。
- 该属性不影响向量索引本身的加载与 ANN 搜索。
- 该属性不删除对象存储中的 binlog，也不改变数据持久性。
- growing segment 仍必须保留原始数据完成实时搜索与 DML；能力关闭只禁止用户输出，并在 sealed/index-ready 后释放可省略的数据副本。
- 某些索引（如常见 HNSW、IVF_FLAT，具体由 Knowhere `HasRawData()` 决定）在索引文件内部已经包含 raw data。新属性可以禁止用户输出，但不能移除搜索索引自身必需或内嵌的 raw bytes，因此不同索引类型的磁盘收益会明显不同。

配置粒度方面，推荐采用“**字段级为唯一内部真相，collection 级仅作为批量设置或默认值语法糖**”的方式。这样既支持多向量字段的细粒度控制，也允许产品提供简单的表级接口。

动态变更方面，不建议恢复旧的 collection 级 lazy load，也不建议让第一次用户查询承担整列加载。推荐由 QueryCoord 持久化一个 raw-data load-config generation，SegmentChecker 发现 QueryNode 上的 generation 落后后，调度现有 `Reopen` 动作；Segcore 使用 LoadDiff 完成字段加载或释放。这样可复用现有故障恢复、任务重试、资源检查和分布收敛机制。

通配符语义暂不拍板，保留两种候选方案：

- **候选 A——过滤**：`output_fields=["*"]` 展开为“当前允许返回的全部字段”，自动过滤 `raw_data.enabled=false` 的向量字段。兼容性和易用性更好，但用户可能没有注意到返回字段集合发生变化。
- **候选 B——整体报错**：只要 wildcard 覆盖到 disabled vector，整个请求返回输入错误。契约更严格、可见性更强，但新建 collection 默认关闭后，大量希望获取“全部标量字段”的 `*` 请求也会失败。
- 无论最终选择哪一种，显式指定关闭字段，例如 `output_fields=["vector"]` 或 `["*", "vector"]`，都应在 Proxy 侧快速返回输入错误；QueryNode/Segcore 仍需提供第二道校验。

升级兼容建议采用分阶段策略：

- 存量 collection 中属性缺失时，兼容解释为 `enabled=true`。
- 新建 collection 由 RootCoord 对“已具备完整执行能力的向量类型”权威地物化 `raw_data.enabled=false`，不依赖 SDK/Proxy 是否为新版本；D5 尚未纳入的复杂类型继续物化 enabled。
- 协议层使用 `UNSPECIFIED/ENABLED/DISABLED` 三态枚举，禁止用 proto3 普通 bool 的默认 `false` 表达兼容语义。
- 第一阶段先发布协议与执行能力，仍保持兼容默认；确认所有组件升级后，再把新建 collection 中已支持向量类型的默认切为关闭。

## 2. 背景、目标与非目标

### 2.1 背景

当前业务观察表明：只有约 10% 的用户会在 query/get/search 的输出字段中取回原始向量，而多数用户仅需要 ANN 搜索结果的主键、距离和标量字段。为少数用户默认维护所有向量列的原始数据访问路径，会带来以下成本：

- QueryNode 本地磁盘或缓存占用增加。
- QueryNode 重启、故障迁移、rebalance 时，sealed segment 恢复需要处理更多字段数据与资源估算。
- 多副本场景中，原始向量副本成本按 replica 数放大。
- 量化索引本身较小，但额外 field raw data 可能重新接近完整原始数据规模，削弱索引压缩收益。

原始向量理论大小可按下式估算：

```text
raw_size = row_count × dimension × bytes_per_dimension
cluster_cost ≈ raw_size × loaded_replicas
```

例如 1 亿条、768 维、FloatVector 的原始向量约为 307.2 GB（十进制），约 286 GiB；两副本约为 572 GiB，尚未计入文件格式、缓存元数据、临时下载和 mmap 管理开销。

### 2.2 目标

- 新建 collection 中已支持的普通向量类型默认不向用户提供原始向量输出能力；复杂类型按 D5 的实施范围分阶段切换。
- ANN search、标量过滤、主键与标量字段输出不受影响。
- 有需求的用户可以按向量字段动态开启原始数据能力。
- 关闭能力后显著降低可省略的 field raw data 本地磁盘、缓存与恢复成本。
- 配置具备 WAL/元数据持久性、CDC 复制、故障恢复和幂等性。
- 多向量字段可以独立设置。
- 明确 wildcard、显式 output field、partial update、无索引小 segment 等边界行为。

### 2.3 非目标

- 不删除对象存储中的原始 binlog。
- 不尝试从所有向量索引中剥离索引内部 raw data。
- 不改变索引构建格式和 Knowhere 搜索算法。
- 不让 scalar field 使用同一默认关闭策略；本期只处理向量字段。
- 不恢复已经删除的旧 collection-level lazy-load 实现。

## 3. 术语澄清：三类容易混淆的“raw data”

| 类型 | 物理位置 | 当前用途 | 本方案是否可省略 |
|---|---|---|---|
| Field raw data | binlog、Storage V2/V3 column group、QueryNode cache/mmap | 输出字段、无索引 brute-force、部分内部回表 | 对 index-ready sealed segment 可省略 |
| Index-embedded raw data | HNSW/IVF_FLAT/DISKANN 等索引文件内部，实际由 `HasRawData()` 决定 | 搜索、refine、从索引反查向量 | 通常不可单独移除 |
| Growing raw data | QueryNode growing segment | 实时 DML、brute-force/interim index、尚未 flush 的查询 | 不可省略 |

如果不做这一区分，容易产生两个错误预期：

1. “关闭 raw data 后所有索引磁盘都会大幅下降”——对于索引自身包含 raw data 的类型不成立。
2. “warmup=disable 已经等同于关闭能力”——不成立，它通常只是把读取推迟到第一次访问。

## 4. 当前架构与实现

### 4.1 LoadCollection 入口与 LoadFields

当前 Proxy 在 `internal/proxy/task.go` 的 `loadCollectionTask.Execute()` 中：

1. 读取 collection schema。
2. 通过 `schemaInfo.GetLoadFieldIDs()` 生成 `LoadFields`。
3. 检查被加载的向量字段是否存在 collection index。
4. 向 QueryCoord 发送 `LoadCollectionRequest`。

当用户未在 load 请求中显式指定字段时，`pkg/common/common.go:GetCollectionLoadFields()` 会根据字段 TypeParams 中的 `field.skipLoad` 生成字段列表。

`field.skipLoad` 不能直接复用到本需求，原因是：

- 它表达“字段是否参加 collection load”，不是“仅禁止原始值输出”。
- `LoadFields` 同时影响字段与索引进入加载目标。
- Proxy 当前要求 load field 列表至少包含主键和一个向量字段。
- 如果把目标向量字段设为 skipLoad，通常连用于 ANN search 的索引也不再按正常加载目标处理。

本需求要求的是“索引继续加载并可搜索，field raw data 可不加载”，必须引入独立维度。

### 4.2 QueryCoord 的持久 LoadConfig

当前 QueryCoord 使用 Streaming WAL 的 `AlterLoadConfig` 消息持久化 collection 的 load 配置。关键结构位于：

- `pkg/proto/messages.proto: AlterLoadConfigMessageHeader`
- `internal/querycoordv2/job/load_config.go`
- `internal/querycoordv2/job/job_load.go`

当前每个 `LoadFieldConfig` 只有：

```protobuf
message LoadFieldConfig {
    int64 field_id = 1;
    int64 index_id = 2;
}
```

也就是说，当前 load config 只知道“加载哪个字段、使用哪个索引”，不知道“该字段是否需要原始数据能力”。

QueryCoord 把 load config 写入自身 meta，然后 TargetObserver 拉取 segment target；SegmentChecker 比较 Target 与 Distribution，生成 Grow、Reduce、Update、Reopen 等任务。QueryNode 故障后，segment 从 Distribution 消失，SegmentChecker 会以高优先级重新加载，这也是当前故障恢复的主要链路。

### 4.3 QueryNode 和 Segcore 的初始加载

QueryNode 的主要入口位于：

- `internal/querynodev2/services.go: LoadSegments()`
- `internal/querynodev2/segments/segment_loader.go`
- `internal/querynodev2/segments/segment.go`

QueryCoord 发送的 `LoadMetaInfo.LoadFields` 会在 `NewCollection()` 中传给 C++ `Schema::UpdateLoadFields()`。随后 sealed segment 的实际加载由 C++ `SegmentLoadInfo` 和 `LoadDiff` 管理：

- `internal/core/src/segcore/SegmentLoadInfo.{h,cpp}`
- `internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp`

当前关键规则是：

1. 索引先加载。
2. `IndexFactory::VecIndexLoadResource()` / Knowhere `HasRawData()` 判断索引是否可以提供 raw data。
3. 当索引已有 raw data，且 `preferFieldDataWhenIndexHasRawData=false` 时，额外 field binlog 可以跳过。
4. 当索引不含 raw data 时，field raw data 通常仍作为可读取数据源存在。
5. Storage V3 的 `ComputeDiffColumnGroups()` 已支持 eager/lazy column group；Storage V1/V2 的 binlog path 也会创建 caching-layer translator。
6. `warmup` 决定 cache cell 是同步、异步还是按需加载，不决定用户是否有权访问字段。

### 4.4 当前 retrieve/search output 数据源选择

`ChunkedSegmentSealedImpl::bulk_subscript()` 的向量路径大致为：

```text
如果索引 HasRawData 且没有强制 prefer field data
    -> 从索引反查向量
否则
    -> 从 field column / remote take 读取
```

当前还有 QueryNode 全局配置：

```text
queryNode.segcore.tieredStorage.rejectRemoteVectorOutput
```

它为 `true` 时，在向量输出未位于本地 cache 时拒绝远端读取；默认值为 `false`，并且当前是 QueryNode 全局隐藏配置，不是 collection/field 级用户能力配置。因此它可以作为底层实现参考，但不能直接满足产品需求。

### 4.5 当前 wildcard 行为

Proxy 的 `translateOutputFields()` 位于 `internal/proxy/util.go`。当前 `output_fields=["*"]` 会展开为全部 `CanRetrieveRawFieldData()` 为真的字段。现有先例是 BM25 function output field：它不会被 wildcard 展开，显式请求则返回“not allowed to retrieve raw data”。

这为新方案提供了可复用的产品语义：`*` 已经不是绝对的“所有 schema 字段”，而是“所有允许返回 raw value 的字段”。

### 4.6 当前 AlterCollectionField 链路

现有 API 已支持字段属性动态修改：

- gRPC：`AlterCollectionField`
- REST：`/v2/vectordb/collections/fields/alter_properties`
- Proxy：`internal/proxy/task.go: alterCollectionFieldTask`
- RootCoord：`internal/rootcoord/ddl_callbacks_alter_collection_field.go`

当前字段属性包括 mmap、warmup、max_length、description 等。mmap/warmup 在 collection 已加载时被 Proxy 禁止修改。

RootCoord 当前会：

1. 修改 FieldSchema.TypeParams。
2. `schema.Version + 1`。
3. 以 `FieldMaskCollectionSchema` 广播 AlterCollection 到 VChannels + CChannel。
4. StreamingNode 把该消息视为 schema change，flush/fence growing segments。
5. QueryNode DML pipeline 消费 schema message，更新 collection schema。
6. ACK callback 更新 RootCoord meta、通知 DataCoord、过期 Proxy cache。

直接把新属性加入现有流程会产生不必要的 growing segment flush，因此必须把“数据布局 schema change”和“运行时字段加载策略 change”区分开。

这里还存在第二个可复用的现有能力：`internal/querynodev2/segments/collection.go` 已允许“逻辑 `schema.Version` 不变，但 schema barrier 更新”的 properties-only snapshot 刷新，并为传入 Segcore 的 schema 单独生成单调递增的 `segcoreSchemaVersion`。因此 raw-data runtime property 没有必要伪装成数据布局变更来推进逻辑 schema version；它可以沿用同版本、新 barrier 的刷新模型，再由独立的 raw-data policy generation 表达物理加载收敛进度。

## 5. 历史方案及其问题

### 5.1 旧 collection-level lazy load

旧实现使用：

```text
lazyload.enabled
queryNode.lazyload.*
```

它在初始 load 时跳过完整资源检查和 segment 实际加载，第一次 search/retrieve 通过 DiskCache 触发整个 segment 加载，并包含资源等待、重试和 eviction。

该实现已在 2026-02 的提交 `1bd65fc1ce` 中删除，原因是已被 warmup 体系替代。旧方案的主要缺陷是：

- collection 粒度过粗，无法适应多向量字段。
- 第一次查询承担下载、解压、mmap/cache 建立和资源等待。
- 用户看到的是高首查延迟、超时或资源不足，而不是明确的能力配置。
- cold load 最终仍可能填满本地磁盘，不能保证长期节省。
- segment 被标记 loaded 与实际数据可用之间存在额外状态。

### 5.2 同步 warmup

同步 warmup 可以保证 load 完成后直接访问，但所有用户都要支付加载成本，无法利用“只有 10% 用户需要原始向量”的业务分布。

### 5.3 现有 warmup=disable

它减少初始 cache 热身，但仍保留访问能力，并把成本推迟到第一次访问，不能表达“绝大多数用户永远不需要该数据”。

### 5.4 根本问题

过去的方案把“是否需要”和“什么时候加载”混成一个配置。新设计必须拆成能力与驻留两个正交维度。

## 6. 推荐产品语义

### 6.1 字段属性

暂定属性名：

```text
raw_data.enabled
```

建议只允许设置在向量字段的 `FieldSchema.TypeParams` 中。

有效值：严格的 `true` / `false`，不接受空字符串、数字或大小写不明确的其他形式。

不建议允许 DeleteKeys 删除此属性。用户应显式设置 true/false，避免属性缺失在存量兼容、新建默认和回滚场景中产生歧义。

### 6.2 有效值优先级

推荐最终模型：

```text
字段显式配置
    > collection 级批量默认（如果未来提供）
    > collection 创建代际默认
```

内部落盘时应尽量把最终值物化到每个向量 FieldSchema，减少运行时继承逻辑。

### 6.3 与 warmup 的组合

| raw_data.enabled | warmup | 语义 |
|---|---|---|
| false | 任意 | 用户不可输出；可省略的 field raw data 不加载，warmup 对该字段 raw data 无效 |
| true | sync | load/reopen 完成前同步预热，适合严格低延迟输出 |
| true | async | 能力开启，后台预热，短时间内可能未完全 ready |
| true | disable | 能力开启但按需远端读取；保留现有 lazy/cold 行为 |

建议 SDK 在用户开启 raw data 时允许同时传 warmup，避免用户只开能力后仍遭遇首次访问延迟。

### 6.4 数据类型范围

第一期建议覆盖：

- FloatVector
- BinaryVector
- Float16Vector
- BFloat16Vector
- Int8Vector
- SparseFloatVector

需要单独评估：

- 顶层 ArrayOfVector / StructArray 内向量子字段：涉及 row reconstruction、父 struct wildcard 与部分字段重建，详见第 7.5 节。
- BM25 function output：当前永久禁止 raw retrieve，不应允许开启。
- 其他 function output vector：需确认数据是否物化，以及 partial update/函数重算语义。
- External Collection：其数据本身是远端按需读取，建议第一期不纳入，避免把普通 collection 与 external table 的访问模型混在一起。

### 6.5 search 与输出

- 作为 `anns_field` 的向量字段即使 `raw_data.enabled=false`，ANN search 仍允许。
- search 返回 ID、distance 和标量字段不受影响。
- search/query/get 显式输出关闭的向量字段时失败。
- 禁止输出是逻辑能力，不因索引恰好 `HasRawData=true` 而绕过。

### 6.6 wildcard：待决策的两套完整语义

已确定的共同规则：

| 请求 | 共同规则 |
|---|---|
| 无 output_fields | 保持当前默认，只返回 PK 或当前 API 默认字段 |
| `["vector"]` | 输入错误：该字段未开启 raw data |
| `["*", "vector"]` | `vector` 属于显式指定，因此返回输入错误 |
| 多向量字段，部分开启 | 显式请求任一 disabled vector 都报错 |

仅 `output_fields=["*"]` 的行为暂缓决定：

| 候选 | 行为 | 优点 | 代价与风险 |
|---|---|---|---|
| A：过滤 | 返回所有允许输出的字段，过滤 disabled vector | 与当前 BM25 function output 的先例一致；默认关闭后，获取全部标量字段的常见请求仍成功；返回结果已有 `OutputFields` 可描述实际字段 | `*` 不再等于 schema 全字段；应用如果不检查 `OutputFields`，可能静默少拿向量；SDK/文档必须强调 |
| B：整体报错 | wildcard 命中任何 disabled vector 时，整个请求报 InputError | 契约严格，调用者不会无感知地少字段；有利于尽早暴露旧应用依赖 | 新 collection 默认 disabled 后，几乎所有带向量 schema 的 `*` 都会失败，即使调用者只关心标量；迁移和 CI 改造面更大 |

决策前实现应把 wildcard 解析与 raw-data policy 判断封装成独立策略点，避免把候选 A 的过滤行为散落到 query/search/partial-update 多条链路。最终只保留一种对外契约，不建议长期增加 collection 级开关让两种 wildcard 语义并存，否则 SDK、缓存和跨 collection 查询行为会变得不可预测。

## 7. 必须正视的功能边界

### 7.1 无索引文件的小 segment

collection 有 index 不代表每个 sealed segment 都有可用 index file。小 segment 可能因行数不足而没有构建索引，当前依赖 field raw data 做 brute-force 或构建 interim index。

推荐行为：

- 当 segment 没有可用向量索引时，为保证 ANN search 正确性，允许内部加载该字段 raw data。
- 即使物理上加载了 raw data，用户输出仍受 `raw_data.enabled=false` 限制。
- index 后续可用后，Reopen/LoadDiff 再释放该字段的额外 raw data。

这意味着“关闭后零 raw data”不是绝对保证；准确表述应是“对有可用索引的 sealed segment 不维护非搜索必需的 field raw data”。

如果产品要求绝对不加载，则必须改为：没有 index file 的 segment 不可搜索或 collection 不可 load。该方案对现有行为破坏更大，不推荐。

### 7.2 索引内部 raw data

当前 Knowhere capability 示例：

- HNSW：通常 `HasRawData=true`。
- IVF_FLAT：`HasRawData=true`。
- IVF_SQ：`HasRawData=false`。
- HNSW_SQ/HNSW_PQ：通常为 false，取决于配置。
- DISKANN：与 metric/config 相关，不能静态假设。

实际判断必须继续使用 `IndexFactory` / Knowhere 的 capability，而不是在 Go 层硬编码索引名称。

因此，量化索引通常是本优化收益最大的场景；FLAT/HNSW 可能本来就已跳过额外 field raw data，收益主要体现为禁止输出、减少远端 fallback 和更清晰的能力语义。

### 7.3 Growing segment

Growing segment 必须保留写入的向量原始值，用于实时搜索、interim index、flush 和 DML 一致性。本方案不能把 growing raw data 从内存中移除。

但用户输出仍应遵守字段能力开关，避免 sealed 与 growing 返回行为不同。

### 7.4 Partial update

这是当前实现中最容易被遗漏的重大依赖。

Proxy 的 partial upsert 会通过内部强一致 Query，使用 `OutputFields=["*"]` 读取旧行，然后把未更新字段合并回新行。大量 CI 明确验证“只更新标量字段时原向量保持不变”。

如果用户 wildcard 最终选择过滤，或者内部查询也受到 disabled policy 限制，partial update 将无法获得旧向量，可能导致：

- 合并后的行缺失向量。
- 校验失败。
- 更严重时写入错误数据。

必须在实现前选择以下方案之一：

#### 方案 A：要求请求携带全部关闭的向量字段

- 对 existing row 的 partial update，如果某个 `raw_data.enabled=false` 的向量字段未出现在请求中，Proxy 返回输入错误。
- 错误信息明确要求用户提交该向量值或先开启 raw data。
- 为了判断一批 PK 中哪些是 existing row，Proxy 可以先做只返回 PK 的强一致查询；确认 update/insert 分组后，再检查 update 行是否具备所有 disabled vector。这样不需要为了执行校验而先读回旧向量。
- 当前 partial update 是列式请求，同一字段通常覆盖整批行。混合 insert/update batch 中，只要请求带上完整 vector column，insert 与 update 都可以继续；如果缺字段，则 existing row 明确拒绝，insert row 继续遵守普通必填、nullable 和 default 校验。
- 优点：实现简单、不会暗中触发大规模远端读取、数据安全边界清晰。
- 缺点：scalar-only partial update 的易用性下降；默认关闭 raw data 后，原本合法的“只改一个标量字段”会变成必须重复上传大向量，网络与客户端成本可能很高。
- 适用条件：第一期工期和故障面控制优先，产品能够接受明确的兼容性变化。

#### 方案 B：内部 trusted read 绕过用户能力

- 用户不能输出向量，但 partial update 内部可以从对象存储按需读取旧向量。
- 必须保证读取不形成长期本地缓存，否则会抵消磁盘优化。
- Storage V1/V2/V3 都要有一致实现，并处理 S3/OOM/timeout。
- 需要在 Query/Plan/Segcore 读取链路中携带明确的 internal purpose，不能通过伪造普通用户 output field 绕过。该 purpose 只能由可信内部调用产生，并应具备审计指标。
- 对 sealed/index-ready segment，若索引不能完整重建字段值，则走 no-store 或短生命周期的对象存储读取；对 growing segment，可使用本来就在内存中的 raw data。
- 强一致语义必须保持：读取旧行的时间戳仍是当前 partial upsert 的 `BeginTs()`，不能为了降低远端成本改成不一致的快照。
- 优点：保持 partial update 兼容。
- 缺点：复杂、延迟高，并重新引入 cold-load 风险；S3 throttle、timeout、对象损坏等会从“向量输出路径”扩散到原本只更新标量的写请求；高频小批量 partial update 可能产生随机远端读取和对象存储费用。
- 适用条件：scalar-only partial update 兼容性是硬要求，且团队愿意实现 no-store trusted read、typed error、限流和故障注入测试。

#### 方案 C：服务端存储差量 update

- 不再读取完整旧行，写入 patch，由下游合并。
- 架构改动远超本需求，不适合作为第一期依赖。

无论选择哪种方案，都不能让内部 `OutputFields=["*"]` 无条件沿用新的用户 wildcard 过滤逻辑。

当前决策状态：**未决，完整保留方案 A 与 B**。当前不指定默认选择，也不把任一方案写成首期既定实现。简单判断标准是：

- 如果更重视按期交付、磁盘收益确定性和较小故障面，选 A。
- 如果更重视默认关闭后的 API 向后兼容，尤其大量用户依赖 scalar-only partial update，选 B。
- 不建议第一期选择 C。

### 7.5 Nullable vector、ArrayOfVector 与 StructArray

部分向量索引虽然报告 `HasRawData=true`，但 nullable valid-data 或 VectorArray row-level embedding-list 重建仍可能依赖 field column。必须通过真实 capability 判断，而不是单一 bool：

```text
search self-contained
raw value retrievable
nullable validity retrievable
row-level VectorArray reconstructable
refine supported
```

第一期可以保留 `HasRawData()` 作为加载优化判断，但测试必须覆盖 nullable 与 VectorArray，避免把“索引有向量 bytes”误判为“能完整恢复用户字段值”。

这里需要把两个经常被合并讨论的对象拆开：

1. **顶层 ArrayOfVector**：一个逻辑 row 内包含数量可变的 embedding list。索引可能保存搜索所需的展开向量，但不一定保存 row 到 embedding-list 的边界、空数组、element type、dim 和完整可返回表示。当前 Go merge/index 逻辑也把 ArrayOfVector 视为 row-dense 数据，而普通 nullable vector 可能使用 compact physical index，二者不能共用一套 offset 假设。
2. **StructArray 内的向量子字段**：除上述向量数组问题外，还存在父 struct 与叶子字段的属性归属、wildcard 展开、兄弟子字段共同加载和整 struct 更新问题。当前 partial update 明确不支持只更新 struct sub-field，而是要求提交整个 struct field；如果其中一个向量子字段 disabled，旧 struct 的重建会再次依赖该向量。

如果第一期直接支持，需要先回答：

- `raw_data.enabled` 设置在父 StructArray、向量叶子字段，还是两者都允许？冲突时谁优先？
- wildcard 请求父 struct 时，disabled 向量子字段是从父结果中裁掉，还是整个父字段报错？裁掉后 SDK 能否稳定反序列化不完整 struct？
- 索引 `HasRawData=true` 时，是否真的可以还原每一行的 embedding list，而不是只有搜索需要的扁平向量？
- Storage V2/V3 column group 是否能只释放向量子字段，还是会连带加载/保留兄弟标量字段？
- partial update 提交整个 struct 时，disabled 子字段缺失应采用第 7.4 节的方案 A 还是 trusted read 方案 B？

首期范围有三种可选方式：

| 候选 | 范围 | 优点 | 代价 |
|---|---|---|---|
| V1：首期排除二者 | 只支持普通顶层向量；ArrayOfVector 和 StructArray vector 保持兼容 enabled，设置属性时报“不支持” | 风险最低，不让不完整 capability 破坏数据重建；可按期交付普通向量的主要收益 | 复杂 schema 暂时拿不到优化收益，默认关闭规则存在类型例外 |
| V2：支持顶层 ArrayOfVector，排除 StructArray | 为 ArrayOfVector 单独实现 row reconstruction capability、资源估算和测试；StructArray 保持 enabled | 覆盖 embedding-list 用户，同时避免父子字段契约 | C++/Go/SDK 测试矩阵明显扩大；仍不能仅依赖 `HasRawData` |
| V3：全部支持 | 顶层 ArrayOfVector 和 StructArray vector sub-field 均可独立配置 | 产品语义最完整 | 首期范围最大，并与 wildcard、partial update、column-group projection 三项未决设计耦合 |

当前决策状态：**待确认**。工程上仍推荐 V1；如果 ArrayOfVector 是本功能的核心目标用户，则推荐 V2，而不是直接跳到 V3。无论选择哪一种，未支持类型都应显式保持兼容 enabled，不能在没有执行能力时被新建 collection 默认物化为 disabled。

## 8. 推荐总体架构

```text
SDK / REST / gRPC
    |
    | AlterCollectionField(raw_data.enabled=true/false)
    v
Proxy
    |- 校验字段类型和值
    |- loaded collection 允许修改该属性
    v
RootCoord
    |- 更新 FieldSchema.TypeParams
    |- 广播 runtime-schema property change
    |- ACK callback 持久化 meta、刷新 Proxy cache
    |- 通知 QueryCoord desired raw-data policy generation
    v
QueryCoord
    |- collection 级 desired generation 持久化
    |- Target/SegmentChecker 比较 desired 与 QN applied generation
    |- 调度 Reopen，不影响普通 search
    v
QueryNode segmentLoader
    |- enable: 资源检查后 Reopen
    |- disable: Reopen 并释放可省略 field data/cache
    v
Segcore SegmentLoadInfo / LoadDiff
    |- false -> true: load binlog/column group
    |- true -> false: drop field data，保留搜索必需 fallback
    |- generation fencing 防止旧任务覆盖新配置
```

### 8.1 为什么推荐复用 Reopen/LoadDiff

- 已有 QueryCoord task scheduler、失败重试、节点分布收敛和故障恢复。
- Segcore 已能按 index/binlog/column group 差异增量加载和释放。
- 避免 release/reload 整个 collection。
- 避免第一次业务查询承担全量加载。
- 配置翻转与 compaction/index update 可以统一排序。

### 8.2 为什么需要 generation

仅更新 schema 属性不足以保证所有 segment 已应用新状态。需要区分：

- Desired policy：RootCoord/QueryCoord 中用户希望的状态。
- Applied policy：某个 QueryNode 上某个 segment 已实际应用的状态。

建议引入 collection load-config generation，并让 segment distribution 上报 applied generation。SegmentChecker 在以下条件触发 Reopen：

```text
segment.applied_raw_data_generation < collection.desired_raw_data_generation
```

第一期使用 collection generation 即可，修改一个字段会 reopen 全部 segment；后续可优化为 field policy hash 或 field generation。

## 9. Streaming/WAL 设计

### 9.1 当前问题

当前任何 `FieldMaskCollectionSchema` 都会被 `messageutil.IsSchemaChange()` 识别为数据 schema 变化，StreamingNode 会 flush/fence growing segment。raw-data policy 不改变行布局、DML 校验或持久数据格式，不应触发 flush。

### 9.2 推荐拆分

新增可区分的 update mask，例如：

```text
schema.runtime_properties
```

消息处理拆成两个判断：

```text
HasSchemaPayloadUpdate:
    更新 StreamingNode / QueryNode 持有的 schema snapshot

IsDataLayoutSchemaChange:
    flush/fence growing segments
```

raw-data policy 只属于前者。

版本语义建议同时拆分：

- `CollectionSchema.Version`：只在字段布局、类型、函数等真正的数据 schema 变化时递增。
- schema barrier：保证同一逻辑 schema version 下，较新的 runtime property snapshot 可以覆盖较旧 snapshot。
- raw-data policy generation：表示 QueryCoord/QN segment 对字段加载能力的 desired/applied 版本。

当前 QueryNode collection manager 已支持同逻辑版本、较新 barrier 的 schema 更新，并维护独立的 Segcore 单调 schema version；实现时应复用该机制。RootCoord `ApplyUpdates`、catalog 持久化和 recovery snapshot 也必须认识新的 runtime-property mask，不能因为它不再使用 `FieldMaskCollectionSchema` 就漏掉 FieldSchema.TypeParams 的持久化。

### 9.3 顺序与一致性

- 仍建议广播到 collection VChannels + CChannel，使配置与 DML 在每个 shard 上有明确顺序。
- QueryNode pipeline 应消费 runtime schema update 并更新 schema，但不要求 flush barrier。
- CChannel ACK callback 负责 RootCoord meta、Proxy cache 和 QueryCoord desired generation。
- Broadcast 当前会等待 ACK callback 完成后返回，因此 API 成功可保证配置已经持久化并已提交给 QueryCoord；但不应等待所有 segment 实际加载完成。

### 9.4 CDC

- AlterCollection runtime property 必须正常复制到 secondary。
- secondary 的 QueryCoord 应使用本地 load config/replica 拓扑执行相同 policy。
- 回调必须幂等，重复 replay 不得重复增加 generation 或重复调度无效任务。

## 10. QueryCoord 变更建议

### 10.1 元数据

建议在 collection load meta 中增加：

```text
field_raw_data_policy: map<field_id, RawDataPolicy>
raw_data_policy_generation: int64
```

协议必须使用三态：

```protobuf
enum RawDataPolicy {
    RAW_DATA_POLICY_UNSPECIFIED = 0;
    RAW_DATA_POLICY_ENABLED = 1;
    RAW_DATA_POLICY_DISABLED = 2;
}
```

`UNSPECIFIED` 的兼容语义为 enabled。

可选地扩展 `LoadFieldConfig`：

```protobuf
message LoadFieldConfig {
    int64 field_id = 1;
    int64 index_id = 2;
    RawDataPolicy raw_data_policy = 3;
}
```

即使最终执行从 schema TypeParams 读取，load config 中保留显式 policy 仍有以下价值：

- load config 自描述。
- WAL replay 不依赖再次查询 RootCoord 才能解释历史配置。
- `proto.Equal` 可识别 policy change。
- mixed-version 中 `UNSPECIFIED` 有明确兼容行为。

### 10.2 动态更新 Job

扩展 `UpdateLoadConfigRequest` 或新增内部 RPC，使 RootCoord ACK callback 可以提交 field policy 变化。Job 应：

1. 校验 collection 已加载；未加载时只更新持久 schema，下一次 load 自然生效。
2. 更新 desired policy 和 generation。
3. 触发 SegmentChecker。
4. 立即返回，不等待所有 segment 完成。
5. 实际失败由 checker/scheduler 重试并暴露状态。

### 10.3 SegmentChecker

当前 Reopen 主要由 DataVersion/ManifestPath 变化触发。需要增加 raw-data generation 比较。

Reopen task 应携带：

- 最新 schema。
- 最新 LoadMeta/raw policy。
- desired generation。
- 原 SegmentLoadInfo。

### 10.4 完成状态

建议提供字段级状态：

```text
Disabled
Disabling
Loading
Loaded
LoadFailed
```

可以扩展现有 `GetLoadState`，也可以新增 `GetFieldLoadState`。不建议让 `AlterCollectionField` 阻塞数分钟等待全量加载。

## 11. QueryNode/Segcore 变更建议

### 11.1 初始加载

加载决策建议为：

```text
need_field_raw_data(field, segment) =
    system_required
    OR no_usable_vector_index
    OR internal_correctness_required
    OR raw_data.enabled
```

其中 `raw_data.enabled=false` 只移除“用户输出”导致的加载需求，不能移除搜索或系统正确性所需的数据。

### 11.2 LoadDiff 转换

#### false -> true

- Storage V1：把对应 field binlog 加入 `binlogs_to_load/replace`。
- Storage V2：按 child_fields 精确加载；多字段 column group 需要投影，避免把同组其他关闭字段一起加载。
- Storage V3：把字段加入 `column_groups_to_load/replace`。
- warmup=sync/async/disable 继续决定实际 cache warmup。

#### true -> false

- 取消该字段进行中的 async warmup。
- 从 `fields_` / field-data-ready 状态移除可省略的 column。
- 清理相应 memory/disk cache cell。
- 如果没有可用向量索引，必须保留 field raw data 作为 search fallback。
- 先确认搜索数据源可用，再 drop field data，遵守现有 LoadDiff “先加载替代数据源、后删除旧数据源”的顺序。

### 11.3 资源估算

当前 Go 和 C++ 都会根据 index `HasRawData()` 决定是否从资源估算中删除 field binlog。需要加入 policy：

- disabled 且有可用 index：field raw data 不计入最终 memory/disk cost。
- enable 动态加载：按增量资源估算，而不是长期使用 Reopen 的全 segment 粗略估算。
- 第一期如果无法提供增量 C API，可先保守使用全量估算，保证安全但会降低可加载率。
- disable 动作不应要求额外资源。

### 11.4 并发与 generation fencing

必须覆盖以下竞态：

- enable 加载中再次 disable。
- disable 清理中再次 enable。
- Reopen 与 index replace/drop 并发。
- Reopen 与 compaction manifest update 并发。
- release collection/segment 与 async warmup 并发。

建议每次 Reopen 携带 generation；commit 新 SegmentLoadInfo 前检查 generation，旧任务即使完成也不能覆盖新 policy。

### 11.5 输出侧第二道防线

Proxy 会做主校验，但 C++ `bulk_subscript` / TakeAPI 仍必须检查 policy：

- 用户输出 purpose 且 disabled：返回明确错误。
- internal trusted purpose：按 partial update 最终方案决定是否允许。
- 不允许仅依赖 `HasFieldData()`，因为 disabled 字段可能因小 segment fallback 物理存在，但仍不应暴露给用户。

## 12. API 与 SDK 建议

### 12.1 复用现有 AlterCollectionField

第一期建议复用现有 API：

```text
AlterCollectionField(
    collection_name,
    field_name,
    properties={"raw_data.enabled": "true"}
)
```

优点：

- gRPC、REST、权限、alias 解析、缓存失效已有基础设施。
- SDK 只需要增加字段属性常量和便捷方法。
- 避免新增一套高度相似的 DDL RPC。

### 12.2 便捷接口

SDK 可提供：

```text
enable_raw_data(collection, field, warmup="sync")
disable_raw_data(collection, field)
get_field_load_state(collection, field)
```

### 12.3 返回语义

- `AlterCollectionField` 成功：desired policy 已持久化并已提交给 QueryCoord。
- 不表示所有 QN segment 已物理完成加载/释放。
- 用户通过字段 load state 查询进度。
- enable 未 ready 时显式输出返回 retriable 的 field-not-loaded/system error。
- policy 本身为 false 时显式输出在 Proxy 返回 non-retriable input error。

## 13. 错误模型

遵循 Milvus merr Input/System blame test：

| 场景 | 分类 | 建议 |
|---|---|---|
| 用户显式请求 disabled vector output | InputError | Proxy 返回 ParameterInvalid 或新的 FieldRawDataDisabled |
| 用户给非向量字段设置 raw_data.enabled | InputError | ParameterInvalid |
| enabled 但 QN 尚未完成加载 | System/Transient | FieldNotLoaded，保持 retriable |
| S3 throttle/timeout | System/Transient | 保留 IO typed code，不得压成 ParameterInvalid |
| OOM/磁盘不足 | System/Transient | ResourceInsufficient，checker 重试 |
| binlog 缺失或损坏 | System/Permanent | DataIntegrity/IO typed error，状态 LoadFailed |
| mixed-version 老 QN 不识别 policy | System capability issue | rollout gate，不应归咎用户 |
| partial update 缺少 disabled vector（若采用方案 A） | InputError | 明确要求携带字段或开启能力 |

边界代码只应使用 `merr.Wrap/Wrapf` 添加上下文，不能用 `WrapErrXxxErr` 意外覆盖底层 IO、resource 或 retriable code。

## 14. 升级、回滚与混合版本

### 14.1 新旧 collection 区分

推荐：

- 存量 schema 属性缺失：解释为 enabled。
- 新建 collection：RootCoord 在持久化前为每个支持的向量字段物化 disabled。
- 不依赖 Proxy 注入，因为 rolling upgrade 期间可能存在旧 Proxy。

### 14.2 为什么不能把“缺失”直接解释为 false

- 升级后所有存量 vector output 立即失败。
- partial update 立即受影响。
- wildcard 返回字段集合变化。
- 回滚旧版本无法判断用户原意。

### 14.3 协议兼容

- 使用 enum `UNSPECIFIED=0`，不能使用普通 bool。
- 新 QN 收到旧消息：UNSPECIFIED -> enabled。
- 旧 QN 收到新消息：忽略未知字段，仍可能加载 raw data；功能正确但优化不生效。
- 旧 Proxy 可能向 disabled 字段发 output 请求，因此 QN 必须有服务端校验。

### 14.4 Rollout gate

若旧 QN 会忽略 disabled 并仍返回向量，则严格的能力隔离在混合版本期间无法保证。建议分两阶段：

1. 发布协议、字段属性、QN enforcement、QueryCoord generation，但默认仍兼容 enabled。
2. 当集群确认所有 Proxy/QN 支持后，开启新建 collection 中已支持向量类型的默认 disabled。

如果 Milvus 当前没有可靠的组件 capability 汇总，应增加 feature flag，在 operator/Helm 层由管理员升级完成后显式开启。

### 14.5 回滚

- 旧版本忽略未知字段属性，回滚后会重新加载 raw data，功能优先、优化丢失，属于安全回滚。
- 新版本重新升级后可恢复 disabled desired state。
- 不得在 disable 时删除对象存储原始数据，否则回滚不可恢复。

## 15. 可观测性

当前 `HasRawData` 指标容易混淆“索引内嵌 raw data”和“field raw data 已加载”。建议拆分：

```text
milvus_querynode_vector_raw_data_policy{collection,field}=0|1
milvus_querynode_vector_field_data_ready{collection,field,segment}=0|1
milvus_querynode_vector_index_has_raw_data{collection,field,segment}=0|1
milvus_querynode_raw_data_applied_generation{collection,segment}
milvus_querynode_raw_data_disk_bytes{collection,field}
milvus_querynode_raw_data_memory_bytes{collection,field}
milvus_querynode_raw_data_reopen_total{result,reason}
milvus_proxy_raw_data_output_rejected_total{api,field_type}
```

日志至少包含：collectionID、fieldID、segmentID、desired/applied generation、index HasRawData、是否因 no-index fallback 保留 field data、资源估算和失败 typed code。

## 16. CI 与测试改造

### 16.1 现有用例拆分原则

现有大量 query/search/get/wildcard/partial-update 用例默认能返回 vector。建议分成两组：

1. **验证原行为的用例**：创建 schema 时显式设置 `raw_data.enabled=true`。
2. **验证新默认的用例**：不设置属性，确认默认 disabled。

不要通过全局 QueryNode 配置统一开启，否则无法验证字段级、多字段混合和动态变更。

### 16.2 Proxy 单测

- 创建 collection 时向量字段默认物化 disabled。
- 存量缺失属性解析为 enabled。
- 非向量字段设置属性报错。
- BM25/function output 限制。
- wildcard 候选 A：过滤 disabled vector；候选 B：wildcard 命中 disabled vector 时整体报错。决策前保留两套契约测试，拍板后删除未采用的一套。
- 显式 vector 报错。
- `["*", "vector"]` 报错。
- 多向量字段部分开启。
- alias、REST、gRPC 行为一致。
- partial update 按最终规则处理。

### 16.3 RootCoord/Streaming 单测

- FieldSchema.TypeParams 正确持久化。
- runtime property 不触发 flush/fence。
- schema snapshot 仍在 StreamingNode recovery 中更新。
- ACK callback 幂等。
- Proxy cache expiration。
- CDC replicated message 重放。
- RootCoord crash 后 callback 重试不重复增加 generation。

### 16.4 QueryCoord 单测

- policy/generation 持久化与恢复。
- checker 发现 generation 落后并创建 Reopen。
- generation 已一致时不重复调度。
- QN crash 后 distribution 消失，重新加载使用最新 policy。
- 新 segment、compaction、index build 完成后使用最新 policy。
- loaded/unloaded collection 动态修改差异。

### 16.5 QueryNode Go 单测

- disabled + usable index 的资源估算排除 field raw data。
- disabled + no index 保留 search fallback。
- enabled 动态 Reopen 请求资源。
- disable 不请求额外资源并释放计费。
- Reopen cancellation/release。
- applied generation 正确上报。

### 16.6 Segcore C++ 单测

至少覆盖以下矩阵：

| 维度 | 取值 |
|---|---|
| Storage | V1、V2 grouped binlog、V3 manifest |
| Index raw | true、false、无 index file |
| Policy transition | initial disabled、initial enabled、false->true、true->false、快速反转 |
| Vector type | dense、binary、fp16/bf16/int8、sparse、nullable；若选择 V2/V3，再加入 ArrayOfVector/StructArray matrix |
| Warmup | sync、async、disable |
| Concurrent action | query、release、index replace、manifest update |

必须验证：

- ANN 结果不因 policy 变化而改变。
- disabled 时显式用户输出被拒绝。
- wildcard 候选 A 下，Proxy 过滤后不触发 C++ raw access；候选 B 下，请求在 Proxy 被整体拒绝。
- no-index segment 搜索仍正确。
- disable 后缓存 cell 和磁盘占用实际下降。
- enable 失败不会发布半完成状态。
- 旧 generation 完成后不会覆盖新 generation。

### 16.7 E2E/CI 场景

1. 新 collection 的受支持普通向量默认 disabled：search 成功、scalar output 成功、vector output 失败；wildcard 按最终选择验证“过滤”或“整体报错”。
2. 创建时 enabled：保持原始 query/get/search output 行为。
3. loaded collection 在线 enable：无需 release，状态从 Loading 到 Loaded，随后可输出。
4. 在线 disable：输出立即禁止，后台清理完成后磁盘下降。
5. 多向量字段混合策略。
6. QN restart/failover/rebalance 后 policy 保持。
7. replica=2 时所有副本收敛。
8. index build 前后、小 segment、量化索引、HNSW/IVF_FLAT。
9. S3 throttle、object missing、corrupt binlog、OOM、disk full、cancel、timeout fault injection。
10. partial update：方案 A 验证缺少 disabled vector 时明确拒绝且不写坏旧行；方案 B 验证 trusted read 保留旧向量并覆盖远端故障。拍板后将选中的方案升级为必跑契约。
11. rolling upgrade：新旧 Proxy/QN 混合。
12. CDC primary/secondary。
13. snapshot/restore、alias、collection rename、schema evolution。

### 16.8 验证指标

除功能正确性外，应在 CI 或 nightly benchmark 中采集：

- LoadCollection wall time。
- QN restart 到 collection loaded 的恢复时间。
- 本地磁盘新增字节。
- 峰值内存和峰值临时磁盘。
- enable/disable Reopen 完成时间。
- 第一次普通 ANN search 延迟。
- enabled + warmup=disable 的第一次 vector output 延迟。

## 17. 实施阶段建议

### Phase 0：基线与数据确认

- 按 index type、storage version、vector type 测量当前 field raw data 磁盘占比。
- 区分 index-embedded raw 与额外 field raw。
- 统计 output vector、wildcard、partial update 的真实使用率。

### Phase 1：协议与能力模型

- 增加字段属性和三态枚举。
- Proxy/RootCoord 校验与持久化。
- output field gating。
- 保持新旧 collection 默认均为兼容 enabled，验证 rolling upgrade。

### Phase 2：初始加载优化

- Segcore initial LoadDiff 按 policy 跳过 field raw data。
- 资源估算与指标同步。
- 覆盖 Storage V1/V2/V3。

### Phase 3：在线 enable/disable

- QueryCoord generation。
- SegmentChecker Reopen。
- QN applied generation。
- 字段 load state API。

### Phase 4：默认切换与 CI 拆分

- 新建 collection 的已支持向量类型默认 disabled；尚未实现执行能力的 ArrayOfVector/StructArray 类型保持兼容 enabled。
- 存量保持 enabled。
- 批量迁移工具。
- 全量 SDK/REST 文档和示例更新。

### Phase 5：高级优化

- partial update trusted remote read（如果最终选择方案 B，但不要求在首期交付）。
- field-level generation，减少无关 Reopen。
- collection 级批量默认/模板。
- 更精确的 index capability 模型。

## 18. 风险清单

| 风险 | 严重度 | 缓解措施 |
|---|---|---|
| 存量 collection 升级后行为变化 | P0 | 缺失属性兼容 enabled，新建显式 disabled |
| partial update 丢失旧向量 | P0 | 实现前明确方案，禁止静默过滤内部读取 |
| no-index 小 segment 无法搜索 | P0 | 保留内部 fallback raw data |
| runtime property 触发全量 flush | P1 | 新 runtime-schema mask，拆分 schema update 类型 |
| enable 过程中 OOM/S3 失败导致半 ready | P1 | generation + 原子 commit + 状态机 |
| mixed-version 旧 QN 泄露 disabled vector | P1 | rollout gate + QN enforcement + 两阶段默认切换 |
| disable 与 index drop 并发导致无搜索数据源 | P1 | LoadDiff 顺序和 generation fencing |
| wildcard 候选 A 静默少字段引起应用误解 | P1 | 文档、Response.OutputFields、显式字段报错；若风险不可接受则选择候选 B |
| wildcard 候选 B 使常见 `*` 请求大面积失败 | P1 | 升级扫描、SDK migration guide；若兼容成本不可接受则选择候选 A |
| 未完整支持 ArrayOfVector/StructArray 却默认 disabled | P0 | 未支持类型保持兼容 enabled，按 capability 分阶段开放 |
| 误把 index raw 算作可节省数据 | P2 | 指标拆分、按 Knowhere capability 报告 |
| 多字段 column group 被整组加载 | P2 | projection、单字段 lazy/eager entry |
| DDL callback 等待物理加载导致锁长期持有 | P1 | callback 只提交 desired state，不等待收敛 |

## 19. 产品与工程决策记录

本节记录 2026-07-14 讨论结果。状态为“已确认”的事项作为后续设计约束；状态为“待确认”的事项保留所有候选、trade-off 和推荐理由，不在其他章节伪装成最终结论。

| ID | 议题 | 状态 | 当前结论 |
|---|---|---|---|
| D1 | 新建与存量默认 | 已确认 | 只对新建 collection 中已支持的向量类型默认关闭；存量属性缺失兼容 enabled |
| D2 | wildcard | 待确认 | 保留“过滤”和“整体报错”两套候选，暂缓拍板 |
| D3 | partial update | 待确认 | 完整保留“请求补齐 disabled vector”和“内部 trusted read”，暂不指定默认方案 |
| D4 | 在线 enable 完成语义 | 已确认 | 异步提交 desired state，并提供状态查询 |
| D5 | ArrayOfVector/StructArray 首期范围 | 待确认 | 在 V1/V2/V3 三档范围中选择 |
| D6 | no-index 小 segment | 已确认 | 允许内部 raw fallback，用户输出仍受 policy 限制 |
| D7 | 配置粒度 | 已确认 | 字段级是唯一内部真相，collection 级只做批量语法糖 |
| D8 | 与 warmup 的关系 | 已确认 | 能力与驻留策略分离，不自动把 enable 等同于 sync warmup |

### D1：只对新建 collection 默认关闭

决定：存量 collection 中属性缺失解释为 enabled；新建 collection 由 RootCoord 对已支持的普通向量字段显式物化 disabled。

理由：

- 避免升级后现有 vector output、wildcard 和 partial update 立即改变行为。
- 保证 rolling upgrade 和回滚时，`UNSPECIFIED` 有稳定的兼容语义。
- 新默认通过 RootCoord 物化，不依赖 SDK/Proxy 版本。

未采用的替代方案是“升级后所有存量 collection 立即关闭”。它能更快获得资源收益，但会造成无迁移窗口的行为破坏，并使旧版本回滚无法恢复用户原意。

### D2：wildcard 暂缓决定

保留第 6.6 节的两套候选：

- 候选 A 过滤：更兼容、更适合“取全部标量”的常见用法，但存在静默少字段风险。
- 候选 B 整体报错：契约更严格，但默认关闭后会让大量 `*` 请求失败。

已确定的是：显式请求 disabled vector 必须报 InputError；两种 wildcard 语义不建议长期作为可配置模式并存。

### D3：partial update 待确认

保留第 7.4 节方案 A 与 B：

- 方案 A 要求请求补齐 disabled vector：工程范围小、资源收益确定、故障面较小，但破坏 scalar-only partial update 的易用性，并增加向量重复上传成本。
- 方案 B 使用内部 trusted read：保持 API 兼容，但必须新增 no-store/internal-purpose 读取语义，并让 S3/OOM/timeout 等远端故障进入写路径。

当前不指定默认方案或首期推荐。两套方案都保留在设计、风险和测试计划中，等待真实 workload 数据或产品兼容性要求进一步明确后再决策。

决定前建议补一项真实数据：统计 partial update 请求中“未携带向量字段”的比例、批量大小、向量维度和调用频率。如果比例很低，A 的兼容成本可能可接受；如果 scalar-only partial update 是核心工作负载，B 更符合产品预期。

### D4：在线 enable 异步提交

决定：`AlterCollectionField` 成功只表示 desired policy 已持久化并提交 QueryCoord，不等待全部 QN segment ready；用户通过 field load state 查询 `Loading/Loaded/LoadFailed`。

理由：

- 大 collection 的 Reopen 可能持续数分钟，不适合占用 DDL RPC、锁或 callback。
- 异步模型可复用 QueryCoord checker 的重试、故障恢复与分布收敛。
- enable 尚未 ready 时返回 retriable System/Transient error；policy 为 false 时返回 non-retriable InputError，两者语义可区分。

### D5：ArrayOfVector/StructArray 首期范围待确认

保留第 7.5 节三档方案：V1 首期都排除、V2 只支持顶层 ArrayOfVector、V3 全部支持。决策状态保持未决，但当前工程推荐明确为 V1；如果后续确认 embedding-list 用户是本功能核心目标，再重新评估 V2。未支持类型必须保持兼容 enabled，不能仅因“新 collection 默认关闭”就物化成系统尚不能正确执行的 disabled 状态。

### D6：允许 no-index 小 segment 内部 fallback

决定：没有可用 index file 时，允许 sealed segment 内部保留或加载 raw vector 完成 brute-force/interim search，但禁止用户输出 disabled vector。

理由：绝对禁止 raw load 会让小 segment 不可搜索，改变当前 ANN 正确性和实时可见性。该决定意味着产品承诺必须表述为“省略非搜索必需的 field raw data”，而不是“关闭后物理上绝对不存在 raw data”。

### D7：字段级内部模型，collection 级为语法糖

决定：持久化、generation、状态和执行均以 field 为基本单位；未来若提供 collection 级开关，只展开成对全部支持向量字段的批量修改。

理由：多向量 collection 可能只需要返回其中一个向量。collection 级作为内部真相会重新制造旧 lazy-load 的粗粒度问题；作为批量 API 则可以兼顾易用性。

### D8：raw-data capability 与 warmup 分离

决定：`raw_data.enabled` 只控制能力，`warmup` 继续控制 sync/async/disable 驻留策略；enable 不隐式改写 warmup。SDK 可以让用户在一次便捷调用中显式同时选择 warmup。

理由：自动 sync 会让开启能力的 DDL 支付全量同步预热成本，也会混淆“可访问”和“必须立即驻留”两个维度。保持正交后，用户可以在低首访延迟、后台预热和按需读取之间自行选择。

## 20. 当前基线方案与未决门槛

综合当前架构、故障恢复、Streaming WAL、LoadDiff 和已确认决策，后续实现可以依赖以下基线：

1. 字段级 `raw_data.enabled` 作为持久能力开关。
2. `warmup` 独立负责能力开启后的驻留时机。
3. 新建 collection 的已支持向量类型显式默认 disabled；存量缺失属性兼容 enabled。
4. 显式输出 disabled vector 报错；wildcard 单独保留 D2 决策门槛。
5. QueryCoord generation + SegmentChecker Reopen 完成在线异步收敛。
6. runtime schema update 不触发 growing flush。
7. no-index segment 保留内部 raw fallback，但不向用户暴露。
8. partial update 行为由 D3 决定，实施前不得静默丢弃旧向量。
9. ArrayOfVector/StructArray 的默认和执行范围由 D5 决定；未支持类型保持 enabled。
10. 两阶段 rolling rollout，禁止在混合版本中直接翻转新建默认。

进入完整实现前必须关闭三个决策门槛：D2 wildcard、D3 partial update、D5 复杂向量类型范围。其余架构和协议工作可以按已确认基线推进。

该方案的本质不是增加另一个 lazy-load 开关，而是把“原始向量是否是 collection 的在线服务能力”提升为持久、可观测、可恢复的字段级契约，再让现有 warmup、LoadDiff 和 QueryCoord 调度体系负责其物理实现。

## 21. 关键代码索引

| 模块 | 文件/入口 |
|---|---|
| Proxy load/output | `internal/proxy/task.go`, `internal/proxy/meta_cache.go`, `internal/proxy/util.go` |
| Partial update | `internal/proxy/task_upsert.go` |
| RootCoord field alter | `internal/rootcoord/ddl_callbacks_alter_collection_field.go` |
| AlterCollection callback | `internal/rootcoord/ddl_callbacks_alter_collection_properties.go` |
| Streaming collection semantic | `docs/agent_guides/streaming-system/message/message-semantic-collection.md` |
| QueryCoord load config | `internal/querycoordv2/job/load_config.go`, `internal/querycoordv2/job/job_load.go` |
| QueryCoord checker/executor | `internal/querycoordv2/checkers/segment_checker.go`, `internal/querycoordv2/task/executor.go` |
| QueryNode load | `internal/querynodev2/segments/segment_loader.go`, `internal/querynodev2/segments/segment.go` |
| QueryNode schema freshness | `internal/querynodev2/segments/collection.go` |
| Segcore load diff | `internal/core/src/segcore/SegmentLoadInfo.cpp` |
| Segcore sealed segment | `internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp` |
| Index capability | `internal/core/src/index/IndexFactory.cpp` |
| Warmup config | `pkg/util/paramtable/component_param.go`, `configs/milvus.yaml` |
| Streaming schema-change gate | `pkg/streaming/util/message/messageutil/header.go`, `internal/streamingnode/server/wal/interceptors/shard/shards/shard_manager_collection.go` |
| 历史 lazy load 删除 | git commit `1bd65fc1ce` |
