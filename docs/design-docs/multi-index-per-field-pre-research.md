# 一列多索引预研：架构调整背景与可行路径

> 面向内核开发者。本文只梳理“如果支持一列多索引，要动哪些东西、要打破哪些既有设定”，不是实现级设计文档。主线方案是 Scalar 一列多索引同时加载、Vector 一列多索引构建但每次 Load 每字段只加载一个；真正同时加载多个 Vector 索引、查询动态选 Vector 索引等偏离主线的内容放在最后，只做简略说明。

## 1. 现状：一列一索引是怎么形成的

### 1.1 索引元数据其实已经以 indexID 为主

DataCoord 的索引元数据 `internal/datacoord/index_meta.go` 里，结构是 `collID -> indexID -> index`，段索引是 `segmentID -> indexID -> segmentIndex`。也就是说，底层存储和 SegmentIndex 模型本身没有“一个字段只能有一个索引”的硬限制，它天然能容纳同字段多个 indexID。

真正卡住多索引的地方在入口校验：`indexMeta.canCreateIndex` 里有一段显式逻辑，除了 JSON 字段不同 json_path 之外，如果同一个 `fieldID` 再来一个未删除的索引，就直接报 `creating multiple indexes on same field is not supported`。所以“同字段多个索引可以构建”不是要重写索引元数据，而是要松开这个入口校验，并把后续链路里所有 `fieldID -> indexID` 的单值假设换成 `fieldID -> []indexID`。

### 1.2 Proxy 在 Load 时把多索引压成了一个

`internal/proxy/task.go` 的 `loadCollectionTask.Execute` 和 `loadPartitionsTask.Execute` 里有同样一段：

- 从 `DescribeIndex` 拿到所有索引；
- 注释直接写着 `not support multiple indexes on one field`；
- 用一个 `map[int64]int64`（`fieldID -> indexID`）保存，循环里同一个字段后来的索引会覆盖先来的。

所以即使 DataCoord 放开了多索引创建，Proxy 也会在 Load 入口处丢失索引，QueryCoord 永远只能看到一个字段一个索引。这个覆盖逻辑是主方案必须最先拆掉的点。

### 1.3 QueryCoord 的“期望加载”以 fieldID 为键

QueryCoord 持久化的 `CollectionLoadInfo.field_indexID` 是 `map<int64, int64>`，`PartitionLoadInfo` 也一样（已标记 deprecated）。查询侧的 `meta.Collection`、`job.LoadCollectionJob` 都围绕这个 map 工作。

Load 请求通过 streaming 消息广播时，`messagespb.LoadFieldConfig` 里是 `field_id + index_id` 两个单值。QueryCoord 收到广播后，`job_load.go` 又把每个 `LoadFieldConfig` 还原成 `map[fieldID]indexID` 写进 meta。这个单值结构是 QueryCoord 侧要换掉的第二个关键点。

### 1.4 QueryNode/Segcore 的运行时结构

Go 侧其实已经比想象中更接近多索引：

- `internal/querynodev2/segments/segment.go` 里 `fieldIndexes` 以 `indexID` 为键，`GetIndex(fieldID)` 返回的是一个切片；
- `separateIndexAndBinlog` / `separateLoadInfoV2` 会遍历 `SegmentLoadInfo.index_infos`，同一个字段多个 `FieldIndexInfo` 会被分别插入；
- QueryNode 上报给 QueryCoord 的 `SegmentVersionInfo.IndexInfo` 也已经是 `map[indexID]FieldIndexInfo`。

这说明“加载多个索引并上报实际加载状态”在 Go 侧基础不差，真正的硬骨头在 C++ segcore。

C++ segcore 的 `CollectionIndexMeta` 是 `map<FieldId, FieldIndexMeta>`（`internal/core/src/common/IndexMeta.h`），它从 proto 构建时，同字段的多个 `FieldIndexMeta` 会互相覆盖。查询计划 `plan_c.cpp` 也假设 `col_index_meta->GetFieldIndexMeta(field_id)` 只拿一个，并用它决定 metric 和搜索参数。

`ChunkedSegmentSealedImpl` 里：

- vector 索引存在 `runtime.vector_indexings[field_id]`，`LoadVecIndex` 非 replace 时会断言这个字段没有索引；
- scalar 索引存在 `runtime.scalar_indexings[field_id]`，同样是单值；只有 JSON 字段是特例，按 `json_indices` 数组维护多个 path。

所以主方案里，Scalar 多索引要打破的是 C++ 的“一个字段一个 scalar index”和“一个字段一个 FieldIndexMeta”；Vector 主方案不打破 vector 单索引运行时，只要求“构建”和“Load 选择”支持多索引。

### 1.5 Import/Copy 路径的代码与 proto 描述不一致

`pkg/proto/data_coord.proto` 里 `CopySegmentResult.index_infos` 注释写成 `map<fieldID, VectorScalarIndexInfo>`，但实际实现不是：

- `internal/datanode/importv2/copy_segment_utils.go` 的 `buildIndexInfoFromSource` 以 `buildID` 为 key 填充 `IndexInfos`；
- `internal/datacoord/copy_segment_task.go` 的 `syncVectorScalarIndexes` 遍历 map 的 value，并靠 `index_name` 解析目标 `indexID`。

消费端已经忽略 key，因此这段代码天然能容纳同字段多个索引。本文以当前代码行为为准，proto 注释按过期说明看待，不列前置工作。
本文以当前代码行为为准，proto 注释按过期说明看待，不列前置工作。

### 1.6 公开问题与同类产品的佐证

通过公开 issue 可以确认几个事实，帮助校准架构判断：

- 2022 年 Milvus 曾允许同字段创建多个索引，但当时上层没有对应的选择语义，测试里同一个字段反复建不同索引后行为错乱，于是 PR #16875 / issue #16866 把入口改成了“同一字段禁止多索引”。这说明底层元数据一直能存多索引，真正缺的是本文要补齐的上层身份与选择语义。
- JSON path 索引已经是一个“同字段多索引”的特例（issue #40442）：同一个 JSON 字段不同 json_path 可以各建各的索引，DataCoord、SegmentLoadInfo 和 QueryNode 已有按 indexID 区分的基础。本文主方案本质上是把这个特例泛化到普通 Scalar 和 Vector 构建。
- 用户对 Vector 原始暴力搜索的诉求（issue #24843）说明“不摧毁已加载索引、必要时走原始向量”是一个真实需求，这支持 Scalar 索引不可用时回退原始扫描、以及 Vector 多加载扩展里缺某个索引时用原始向量的方向。
- 同类产品方面，Qdrant 采用 named vectors 表达多个向量空间，每个命名向量一套索引配置，不会在同一字段上同时维护多个索引；Elasticsearch/Lucene 的加速结构也是按字段组织，而不是同字段多个用户索引。因此“Vector 每次 Load 每字段只加载一个索引”的 A1 主线与行业惯例一致；Scalar 多索引同时加载是更靠前的探索，第一版只保证“选一个能正确执行的索引”是合理的。


## 2. 主方案：架构上改成什么样

### 2.1 把“索引身份”统一到 indexID

现在链路里同时存在 `fieldID`、`indexID`、`buildID`、`index_name` 四种身份。主方案只保留一个原则：

- `indexID` 是索引定义的唯一身份，跨 RootCoord/DataCoord/QueryCoord/QueryNode/Segcore 全部显式传递；
- `buildID` 只表示“某次对某个 segment 的构建产物”，段索引文件仍以 buildID 区分；
- `index_name` 只作为用户可见名字；
- `fieldID` 只表示索引属于哪个字段，不再作为索引身份或 map 主键。

所有 `fieldID -> indexID` 的单值结构，都要改成 `fieldID -> []indexID`，或者直接以 `indexID` 为键、`fieldID` 为属性。

### 2.2 五层状态分离

主方案把现在混在 `CollectionLoadInfo` 里的东西拆成五层：

1. **索引定义层**：RootCoord/DataCoord 的索引元数据，`collID -> indexID -> index`。已经存在，只需放开多索引创建，并保证 CreateIndex/DropIndex 的幂等与名字唯一性逻辑不退化。
2. **Collection 默认索引层**：每个字段有一个默认 `indexID`。这是 collection 级元数据，Load 不指定时从这里取。单独操作修改，不与 Load 请求绑定。Drop 了默认索引时，默认值置为“未设置”，不自动挑一个替代。
3. **当前 Load 目标层**：QueryCoord 持久化的“这个 collection/partition 应该加载哪些索引”。主方案从 `map[fieldID]indexID` 改为 `fieldID -> []indexID`。LoadCollection/LoadPartitions 可以显式指定；Vector 字段只允许一个，Scalar 字段允许多个。Release 后结束当前 Load，但保留更高修改版本的“已释放目标”和默认索引，重启恢复时不再猜测。
4. **实际加载状态层**：QueryNode 上报“每个 segment 实际加载了哪些 indexID”。QueryCoord 把它存在 dist manager 的 segment 状态里，只反映现实，不反推目标。
5. **单次查询选择层**：查询计划为每个过滤条件选一个 indexID。Scalar 自动选择；内部计划节点预留显式 indexID 能力。Vector 主方案固定使用当前 Load 的唯一索引。

这五层分开后，QueryCoord 不再从“实际加载状态”推断“应该加载什么”，也不会从“默认索引”推断“这次 Load 选了什么”。

### 2.3 组件级改动

#### RootCoord / DataCoord（索引定义）

- 放开 `indexMeta.canCreateIndex` 的“同字段多索引”限制。同名字同参数仍幂等返回已有 `indexID`；同字段不同 `index_name` 或不同参数允许共存。
- `GetIndexInfos(segmentIDs)` 已经返回该 segment 所有已完成的 `IndexFilePathInfo`，不需要改存储，只需保证同字段多个索引都返回。
- `DropIndex` 继续通过 WAL 广播 DDL。内核只检查本地 QueryCoord 引用，不负责跨集群一致性；跨集群 Drop 顺序由外部平台协调。Vector 索引在 collection loaded 时禁止 Drop 的现有保护保留。

#### Proxy（Load 与查询入口）

- `loadCollectionTask` / `loadPartitionsTask` 不再做 `map[fieldID]indexID` 覆盖。
- Load 请求新增结构化字段表达“字段 -> 多个索引名”，例如 `field_name + index_names: [A, B]`，替代从通用字符串参数里解析。Scalar 可传多个，Vector 只允许一个。
- Load 未显式指定时，Proxy 从 collection 默认索引元数据里取默认值，再把最终选择显式传给 QueryCoord。未指定时的产品行为（只加载默认还是自动加载全部）不属于架构问题，本文不展开。
- 查询侧：Scalar 过滤条件解析成内部计划后，每个可索引的条件叶子可以带一个可选 `indexID`。自动选择时由计划层决定；公开语法指定索引不在本期，只保留内部能力。Vector 搜索不增加用户指定索引。

#### QueryCoord（目标与恢复）

- `CollectionLoadInfo` / `PartitionLoadInfo` 增加“字段 -> 索引列表”的结构，替代 `field_indexID` map。QueryCoord 持久化的是“本次 Load 选择了哪些 indexID”。
- `job.LoadCollectionJob` 从 `LoadFieldConfig` 还原出 `fieldID -> []indexID`，Vector 校验只允许一个。
- `load_config.go` 的 `generateLoadFields` 改为生成 `LoadFieldConfig { field_id, index_ids[] }`。
- `messagespb.LoadFieldConfig` 从 `index_id` 单值改为 `repeated int64 index_ids`。
- `task/executor.go` 的 `getLoadInfo` 现在把 DataCoord 返回的所有段索引都塞进 `SegmentLoadInfo`。主方案要按“当前 Load 目标”过滤，只打包选中的 indexID，避免把未选择的备用索引加载进 QueryNode。
- `checkers/index_checker.go` 的 `checkSegment` 现在要求 collection 的所有索引都在 segment 上。主方案改成只检查“当前 Load 目标中的 indexID”，否则一个备用索引没建好会阻塞所有 segment 可查。
- `ReleaseCollection` / `ReleasePartitions` 后不删除目标记录，只把状态改成“已释放”并保留版本号；重新 Load 时可基于目标与默认索引继续演进，也能支持“释放最后一个 partition 即结束当前 Load”的语义。

#### QueryNode（加载与上报）

- `LoadSegments` 收到的 `index_info_list` 改为包含所有选中索引的 `IndexInfo`，`ComposeIndexMeta` 需要带上 `index_id`，不再靠 fieldID 去重。
- 实际加载：Go 侧 segment loader 已经按 `indexID` 存索引，主方案基本沿用；对 Scalar 字段，同字段多个 `FieldIndexInfo` 会依次加载。
- 数据上报：`buildSegmentVersionInfo` 已经按 `indexID` 上报 `IndexInfo`，主方案沿用，不需要改 QueryNode 上报格式。
- Vector 在线切换：QueryCoord 把目标从索引 A 改成 B 后，segment 更新任务带上 B 的 `FieldIndexInfo`，QueryNode `ReopenSegments` 走 `LoadVecIndex(is_replace=true)` 替换 A。切换过程中，一部分 segment 还是 A，一部分已变成 B；只有 A/B 查询语义兼容（metric、参数、refine、raw data 能力）时才允许渐进切换，否则走 release/reload。

#### Segcore（C++）

- `CollectionIndexMeta` 从 `map<FieldId, FieldIndexMeta>` 改成支持同字段多个 `FieldIndexMeta`，内部以 `index_id` 区分。
- Scalar 运行时：`scalar_indexings[field_id]` 从单值改成 `field_id -> 多个索引条目`（JSON path 已有的数组思路可以统一）。`LoadScalarIndex` 的非 replace 断言要改成“同 field 同 index_id 才冲突”。
- 查询执行：Scalar 条件叶子选择索引时，先看叶子上有没有显式 `indexID`，没有再在“语义兼容的已加载索引”里自动选一个；选不到就退回原始字段扫描，绝不悄悄换另一个 indexID。第一版不保证最快，只保证正确执行。
- 原始数据路径：Scalar 字段始终保留可读的原始数据路径，不依赖某个索引存在。
- Vector 运行时保持 `vector_indexings[field_id]` 单值；主方案不要求 segcore 同时驻留多个 vector 索引。

### 2.4 数据流：Scalar 查询怎么走

1. Proxy 解析表达式，生成计划节点。每个可索引的过滤条件叶子带字段信息，并预留可选 `indexID`。
2. 计划发给 QueryNode 后，segcore 对每个条件叶子：
   - 有显式 `indexID`：找到该索引；可用就用，不可用退回原始扫描；
   - 无显式：在已加载的、算子语义匹配的候选索引里自动选一个；暂时只保证能正确执行，不比较速度。
3. 原始字段数据独立可读，因此索引选不到时查询仍能完成。
4. 执行计划选中的 `indexID` 可由 QueryNode 返回给 Proxy（或通过 debug 接口），为以后做 explain/plan 展示留口，不改变执行语义。

### 2.5 数据流：Vector 索引构建与在线切换怎么走

1. 用户对同一字段创建两个索引 A/B，DataCoord 正常分配两个 `indexID`，分别对 segment 构建，产生两套段索引文件。
2. Load 时 Proxy/QueryCoord 只选择一个 `indexID`（默认或显式）。QueryCoord 按该 `indexID` 过滤段索引并打包，QueryNode 每个字段只加载一个 vector 索引。
3. 在线切换 A -> B：QueryCoord 更新当前 Load 目标为 B；IndexChecker 发现目标变化，对每个 segment 生成 reopen 任务；QueryNode 加载 B 并 replace A。只允许 A/B 语义兼容时渐进切换；否则要求 release/reload。
4. 切换中查询仍使用当前 Load 的索引；兼容场景下不同 segment 短暂处于 A/B 混合，查询语义一致；不兼容场景不进渐进切换。
5. 正在写入的 Growing 数据不强制按 `indexID` 维护多份临时索引。Sealed 数据严格按 `indexID`；Growing 使用兼容的临时索引或原始向量。

### 2.6 数据流：CDC 与多集群

CreateIndex / DropIndex 本来就是 WAL 控制通道的 DDL 消息。Load/Release 也是通过 `AlterLoadConfigMessage` / `DropLoadConfigMessage` 广播。主方案把“Load 目标中的字段 -> 索引列表”编码进 `LoadFieldConfig`，因此索引定义、默认索引规则、Load/Release DDL 会继续沿 CDC 复制。

不复制的是：

- QueryNode 实际上报的加载进度；
- 单次查询选了哪个索引。

也就是说，每个查询集群收到同样的“应该加载什么”，但“实际加载到什么程度”和“每个查询怎么执行”是本地状态。这与现在 Load/Release DDL 复制、查询请求不复制的模型一致。

### 2.7 升级与回滚

- 预研阶段不承诺向前兼容，但需要保证滚动升级和功能启用前回滚。
- Scalar 多索引和 Vector 多索引使用两个独立配置开关。升级顺序：先升级所有节点，开关保持关闭；确认集群正常后，再打开 Scalar 开关。启用前回滚只需关闭开关并退回旧代码，旧路径仍然完整。
- 新 proto 字段在开关打开后才写入；开关关闭时，各组件继续走旧的 `field_indexID` 单值路径。
- 打开 Vector 开关后，已创建的多 vector 索引才可被 Load 指定；开关前只能走旧的“按 fieldID 覆盖”路径，行为不变。

## 3. 资源成本

### 3.1 Scalar 多索引同时加载

- 内存/磁盘：每个选中并加载的 Scalar 索引都计入 QueryNode 资源估算。现有估算已经按 `SegmentLoadInfo.index_infos` 逐个索引计算，主要改动是让估算接受同字段多个索引。
- 原始数据：只要加载的索引不提供 raw data 替代，原始字段数据仍驻留；加载多个 scalar 索引不会消除原始数据。
- 查询 CPU：自动选择只选一个执行，多索引不增加单查询执行成本，只增加加载常驻成本。

### 3.2 Vector 多索引构建 + 单索引加载

- 构建成本：每个索引各自触发构建任务，占用 DataNode/IndexNode 计算与存储；QueryNode 内存不变（每个字段仍只加载一个）。
- 在线切换：切换期间，部分 segment 会在短时间内同时持有 A/B 两个 vector 索引的加载资源，峰值内存上升；切换完成后回落。
- Growing：主线不为 Growing 维护多份临时 vector 索引，插入 CPU/内存与现在基本一致。

## 4. 偏离主线的扩展

### 4.1 A2：真正同时加载多个 Vector 索引

这是最贵的扩展。要动的地方：

- QueryCoord：Load 目标允许 Vector 字段多个 `indexID`，校验从“只允许一个”变为“允许多个”。
- QueryNode：segment 从 `vector_indexings[field_id]` 单值变为按 `indexID` 维护多个已加载 vector 索引。
- Segcore：`LoadVecIndex` 从“非 replace 即断言”变为可追加；`CollectionIndexMeta` 支持同字段多个 vector 元数据；Search 请求必须携带要用的 `indexID`。
- Proxy：Search 请求增加可选 `indexID`，并下推到计划。
- 新 segment：如果允许多个 vector 索引同时加载，新 segment 上某个索引 B 还没建好时，按“缺 B 时用原始向量”处理，数据可以更早可查，但显式指定 B 不再是严格物理保证。
- Growing：多个 vector 索引若也要在 Growing 生效，需要维护多份临时索引，插入 CPU、内存、延迟都会增加。
- 资源：内存与加载索引数量成正比，这是它和主线 A1 的本质区别。

### 4.2 B2：任意组合无停机切换

主线 B1 只允许语义兼容时渐进切换。若要所有组合都无停机，必须先有 A2 的“同时驻留多个 vector 索引”，并且：

- 查询计划固定到 `indexID`，不依赖 collection 单当前索引；
- 切换时要对请求排空，避免一个 segment 在 A/B 之间被两种语义读到；
- Growing 也要按 `indexID` 处理，否则切换期间 Growing/Sealed 语义不一致。

成本约等于 A2 + 请求版本化/排空机制。

### 4.3 C2：Milvus 动态选择 Vector 索引

主线 C1 是“未指定用固定 Load 默认，允许显式指定”。C2 要 Milvus 每次查询自己选，前提是 A2 已成立，并且要引入：

- 每个索引/segment 的统计摘要（nlist、nprobe、数据量、类型、内存、延迟等）；
- 一个成本/收益模型，把召回、延迟、资源折中量化；
- 查询计划层在选择前读取统计并决策，还要处理统计过期与冷热问题。

这是独立的大模块，主线只需保证 `CollectionIndexMeta` 和计划层以后能挂上这个选择入口。

### 4.4 Scalar 执行计划展示

主线在内部计划节点预留了 `indexID`。以后可以像 MySQL 的 explain 一样，把每个过滤条件最终使用的 `indexID` 返回给用户或平台，只读不改变执行。这个增强很小，但要等 Scalar 自动选择落地后再定义公开格式。

## 5. 建议的验证路径

预研阶段可以用一条垂直链路验证所有关键假设：

1. DataCoord 放开同字段两个 Scalar 索引创建；
2. Proxy Load 传两个索引名；
3. QueryCoord 持久化 `fieldID -> []indexID`，并按目标过滤段索引；
4. QueryNode 把两个 `FieldIndexInfo` 加载进同一个 segment；
5. Segcore `CollectionIndexMeta` 支持同字段两个索引，查询条件分别命中两个索引，确认自动选择与原始扫描回退。

Vector 在线切换单独验证：同 metric 的两个 vector 索引 A/B，Load A 后把目标改成 B，观察 segment 逐个 replace 和查询语义一致性。

这两条路径分别对应 Scalar 多索引主线和 Vector 在线切换主线，不需要先实现任何扩展能力。
