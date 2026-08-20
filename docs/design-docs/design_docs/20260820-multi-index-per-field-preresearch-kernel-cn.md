# Milvus 一列多索引预研（内核视角背景梳理）

> 本文是背景梳理与预研，不是详细设计文档，也不是产品/行为决策记录。
> 目的：从内核开发者视角，看清「支持一个字段多个索引」到底要改哪些地方、要打破哪些现有设定，以及各方案的成本大致落在哪里，用来决定哪些方向要做、哪些先不做。
> 主线：Scalar 字段可同时加载多个索引并参与过滤；Vector 字段可定义/构建多个索引，但一次 Load 每字段只加载一个、可指定并可在已加载状态下替换。
> 偏离主线（真正同时加载多 Vector 索引、无停机热切换、查询动态选 Vector 索引）只在最后一节简略说明改什么，不在主线范围内。

---

## 0. 一句话结论

「一列一索引」并不是一个孤立开关，而是分散在从元数据校验、Load 状态、WAL/CDC 消息，一直到 Segcore 内存结构和查询计划里的一个默认假设。好消息是：索引定义、构建、DDL 复制、QueryCoord 的加载进度这几层其实**已经按 indexID 组织**，真正被「一列一索引」卡住的是四层：

1. DataCoord 的建索引校验（第一道闸门）；
2. Proxy 和 QueryCoord 的 Load 状态（`fieldID -> indexID` 的 1:1 map）；
3. Load 状态的 WAL 消息（`LoadFieldConfig{field_id, index_id}`）；
4. Segcore 的配置与运行时内存结构（`fieldID -> 一个索引`）。

主线方案（Scalar 多加载 + Vector 单加载可选）的主要工作量集中在「把 `fieldID -> 单个 indexID` 改成 `fieldID -> 一组 indexID`」这条贯穿链路，以及 Segcore 里把 Scalar 索引的挂载键从 fieldID 换成 indexID。真正同时加载多个 Vector 索引则要动 Segcore 的向量索引持有结构和查询路径，成本明显高一档，建议作为后续独立扩展。

---

## 1. 现状：这个假设埋在哪些地方

按一次查询/加载数据经过的路径，从定义到执行，逐层看。

### 1.1 索引定义与构建（DataCoord）

- 元数据本体**已经按 indexID 存**：`internal/datacoord/index_meta.go` 里
  `indexes = map[collectionID]map[indexID]*model.Index`。一个字段在数据结构上已经可以有多个 `Index` 对象。
- 每个 segment 的索引产物也**已经按 indexID 独立**：`model.SegmentIndex` 以 `(segmentID, indexID, buildID)` 为键，构建、文件、版本都是每个 indexID 一份。
- **真正的闸门在校验**：`internal/datacoord/index_meta.go` 的 `canCreateIndex()` 里，当请求的 `fieldID` 和已有索引相同，直接返回「creating multiple indexes on same field is not supported」。唯一例外是 JSON 字段不同 `jsonPath` 的索引被放行——这本身就是一个「同字段多索引」的既有先例。

所以 DataCoord 这一层要改的很小：放开 `canCreateIndex` 的拒绝逻辑。顺带一个现状：`index_name` 目前在 collection 内实质唯一（同名但指向不同字段会被拒），放开多索引后它天然可以继续按 collection 内唯一来管，不需要为多索引单独设计。构建链路本身不需要动。

### 1.2 Proxy 的 Load 解析

- `internal/proxy/task.go` 的 `loadCollectionTask` 里，注释直接写着「not support multiple indexes on one field」：它把 `DescribeIndex` 返回的所有索引塞进 `fieldIndexIDs := map[int64]int64`，同一字段多个索引时**后写的覆盖先写的**，最终只留一个。

这一层是把「字段 → 一组索引」拍平成「字段 → 一个索引」的第一处显式代码，主线必须改。

### 1.3 Load 状态的 proto 与持久化

- 公开协议 `query_coord.proto` 里 `LoadCollectionRequest`、`CollectionLoadInfo`、`PartitionLoadInfo` 都有一个 `field_indexID = map<int64,int64>`（fieldID → 单个 indexID）。
- 持久化模型 `internal/metastore/model/load_info.go` 的 `CollectionLoadInfo.FieldIndexID` / `PartitionLoadInfo.FieldIndexID` 同样是 `map[int64]int64`。
- QueryCoord 内存态 `internal/querycoordv2/meta/collection_manager.go` 的 `Collection` 内嵌 `CollectionLoadInfo`，用的就是这个 1:1 map。

这三处一起构成「当前 Load 加载了哪个索引」的状态模型，是主线要动的核心：把 `map[int64]int64` 升级成能表达「一个字段一组 indexID」的结构（例如 `map[int64][]int64` 或结构化列表），同时给 Vector 字段加上「至多一个」的约束。

### 1.4 Load 状态的 WAL / CDC 消息

- `internal/querycoordv2/job/load_config.go` 的 `generateLoadFields()` 把上面的 map 转成 `messages.proto` 里的 `LoadFieldConfig{field_id, index_id}`，放进 `AlterLoadConfigMessageHeader.load_fields`。
- 这条 `AlterLoadConfigMessage` 走 WAL 广播，也就是 CDC 复制的载体。它现在是 1:1（一个字段一条、一条一个 index_id）。

Load 目标沿 CDC 复制、实际加载进度本地管理这个边界本身没问题；要改的只是让这条消息里的「字段 → 索引」能带多个 indexID（Vector 仍限制一个）。

### 1.5 QueryCoord 的加载进度与索引检查器（基本不用大改）

这一层反而是**已经按 indexID 组织好的**，是主线改动里最省的部分：

- `internal/querycoordv2/checkers/index_checker.go` 的 `checkSegment()` 遍历 `ListIndexes` 返回的**所有** indexID，检查 `segment.IndexInfo[indexID]` 是否已加载；`segment.IndexInfo` 本身就是 `map[indexID]*FieldIndexInfo`。`checkRedundantIndices()` 也按 indexID 清掉不在目标集合里的索引。
- `internal/querycoordv2/meta/coordinator_broker.go` 的 `GetIndexInfo()` 返回 `map[segmentID][]*FieldIndexInfo`（列表），`SegmentLoadInfo.index_infos` 也是 repeated 列表。
- 结论：QueryCoord「实际加载了什么」已经是多索引友好的。主线只需把「期望加载的 indexID 集合」从 field→单 id 换成 field→id 集合，检查器几乎能直接复用。

### 1.6 Segcore 的配置层（C++）

- `internal/core/src/common/IndexMeta.h`：`CollectionIndexMeta.fieldMetas_ = map<FieldId, FieldIndexMeta>`，而 `FieldIndexMeta` 只放一组 `index_params`（index_type + metric 等），即「一个字段一个索引配置」。

这是 Segcore 侧的「一列一索引」第一处，需要把 `FieldId -> 一个 FieldIndexMeta` 改成 `FieldId -> 一组 FieldIndexMeta`（或改按 indexID 索引）。

### 1.7 Segcore 的运行时层（C++）

- `internal/core/src/segcore/FieldIndexing.h`：`IndexingRecord.field_indexings_ = map<FieldId, FieldIndexing>`。`ScalarFieldIndexing` 内部持有一个 `ScalarIndex`，`VectorFieldIndexing` 内部持有一个 `VectorIndex`。
- `internal/core/src/segcore/SealedIndexingRecord.h`：sealed segment 的 `field_indexings_ = map<FieldId, SealedIndexingEntry>`（向量索引每字段一个）。
- 查询计划 `internal/core/src/exec/expression/*.h`（`TermExpr`、`BinaryRangeExpr`、`BloomFilterExpr`、`JsonContainsExpr` 等）按 `field_id` 绑定一个索引，`DetermineExecPath` 决定走索引还是原始数据。
- `internal/core/src/segcore/load_index_c.cpp`：`LoadIndexInfo` 其实已经同时带 `field_id` 和 `index_id`，加载通道本身是 indexID 粒度的。

这是最深的一层，也是「Scalar 多加载」和「Vector 多加载」成本分叉的地方：
- **Scalar 多加载**：要把挂载键从 fieldID 换成 (fieldID, indexID) 或 indexID，并让每个过滤条件在多个候选索引里选一个。原始字段数据的独立可读路径本来就有（`has_raw_data` 的语义），所以「索引不可用时退回扫描」这条兜底路是现成的。
- **Vector 多加载**：要改 `VectorFieldIndexing` 让它持有多个 `VectorIndex`，search 路径要带上 indexID 选一个，还要处理 growing segment 的临时索引、warmup、资源估算。成本明显更高，主线不做。

### 1.8 索引 DDL 的 CDC 复制

- 建/删索引的 DDL 已经走 WAL：`AlterIndexMessageV2` 带 `FieldIndexes []*indexpb.FieldIndex`（列表，已按 indexID），datacoord 消费后 `AlterIndex`。
- 所以「索引定义沿 CDC 复制」这条链路已经是列表语义，不需要为多索引专门改。要改的只有第 1.4 节那条 Load 目标消息。

---

## 2. 主线方案要动什么（Scalar 多加载 + Vector 单加载可选）

主线一句话：**把「字段 → 一个索引」这条贯穿状态链改成「字段 → 一组索引（Vector 至多一个）」，并让 Scalar 的查询计划能按 indexID 选索引；Vector 只让「加载哪个」可选，不动「同时加载多个」。**

逐层改动：

1. **DataCoord**：放开 `canCreateIndex` 的「同字段拒建」校验；`index_name` 继续按 collection 内唯一，无需新规则。构建、`SegmentIndex` 不动。
2. **Proxy**：`loadCollectionTask` 不再把多索引拍平，改为构造 `fieldID -> indexID 列表`；Vector 字段校验至多一个。
3. **Load 状态模型**：`query_coord.proto` 的三个 `FieldIndexID map<int64,int64>` 改成 `map<int64, 列表>` 或等价结构；`model/load_info.go`、QueryCoord 内存态、`load_config.go` 同步改。
4. **WAL Load 消息**：`LoadFieldConfig{field_id, index_id}` 改成能带一组 indexID（Vector 一个）。CDC 复制语义不变。
5. **QueryCoord 检查器**：几乎不改，只把「期望集合」换成多 indexID；现有 `segment.IndexInfo`（按 indexID）直接可用。
6. **Segcore 配置层**：`CollectionIndexMeta` 从 `FieldId -> FieldIndexMeta` 改成 `FieldId -> 一组 FieldIndexMeta`（或按 indexID）。
7. **Segcore 运行时（Scalar）**：`IndexingRecord` 的 scalar 部分改按 indexID 挂多个索引；查询计划的每个过滤条件保留可选 indexID，执行时从候选里选一个能正确执行的（第一版不追求最优，只保证选一条能执行、语义一致的路）；索引不可用时退回原始扫描。
8. **Segcore 运行时（Vector）**：本期保持「每字段一个 VectorIndex」不变，只让「加载哪个 indexID」在 Load 时可选。已加载状态下从 A 换到 B，本质就是改 Load 目标里的 indexID，由 index checker 触发 reopen 完成替换——语义兼容时是渐进替换，不兼容时走 release/reload。

需要打破的旧设定，集中列一下：

- `index_name` 在 collection 内实质唯一（现状），多索引放开后继续按 collection 内唯一即可，不引入新的唯一性规则。
- Load 状态 `fieldID -> indexID` → `fieldID -> []indexID`。
- `LoadFieldConfig` 一个字段一条一条 index_id → 一个字段可带多个 index_id。
- Segcore 索引挂载键 `fieldID` → `indexID`（Scalar 部分）。
- 查询计划按 `fieldID` 绑定索引 → 按 indexID 绑定并做选择（Scalar 部分）。

---

## 3. 发散选项：真正同时加载多 Vector 索引 / 无停机热切换（简略）

这些是偏离主线的方向，只说明「如果要做，内核需要改什么」，用于评估要不要单独立项。

### 3.1 同一个 Vector 字段同时加载多个索引

- Segcore：`VectorFieldIndexing` 从「持有一个 `VectorIndex`」改成「持有一组 `VectorIndex`（按 indexID）」，`SealedIndexingRecord` 同理。
- 查询路径：Search 要能带 indexID 选一个，或在查询计划里按 indexID 路由到具体向量索引。
- growing segment：临时向量索引要么按 indexID 维护多份，要么对未选中的索引退化为原始向量暴力算。
- 资源层：warmup、mmap、内存估算都要按 indexID 粒度，不能再用「字段一份」的假设。
- 这是完整 V2 的成本，主线不动，只保留公共状态模型（字段 → indexID 集合）给它复用。

### 3.2 已加载状态下无停机热切换 Vector 索引

- 渐进版（语义兼容，如只换 index type 不换 metric）：可复用主线的「改 Load 目标 → index checker reopen」机制，新索引建好后再把目标切过去，不需要双加载。
- 完全无停机版（metric/参数也变）：需要双加载（新旧索引同时在场）+ 查询固定 indexID + 请求排空（旧索引上的在途请求跑完再回收）+ growing 处理，成本接近 3.1，属于完整 V2。

### 3.3 查询时动态选择 Vector 索引

- 主线是「未指定时用 Load 的固定默认，允许显式指定」。动态自动选要定义召回/延迟/资源之间的取舍，并把选择器放进查询计划，属于另一层复杂度，主线不做。

### 3.4 Scalar 自动选择的成本估算

- 第一版只保留「每个过滤条件带可选 indexID + 一个选择函数入口」，不承诺选最快。
- 真正按统计选最优索引，需要定义并保存每个索引的统计摘要、随构建/Compaction 更新，会动元数据和构建链路，作为独立增强，不进入主线。

---

## 4. 升级、回滚与开关

- **滚动升级**：新增的「字段 → indexID 集合」字段对旧节点是可选的，旧节点读不到时按旧行为处理（单索引）。两个独立开关（Scalar 多索引、Vector 多索引）控制是否允许「一列多索引」的创建，关闭时走旧的单索引校验路径。
- **回滚（功能启用前）**：关开关 + 不创建多索引即可，内核不需要保证版本一致性；是否启用由控制面/云平台管，回滚前不启用即可。已创建多索引的 collection 在回滚后旧代码会回到「拍平成单索引」或「拒建」的旧行为，属于启用后的兼容问题，不在「启用前回滚」范围内。
- **CDC**：索引定义、默认规则、Load/Release DDL 继续沿 CDC 复制（已经是列表语义或改后的列表语义）；实际加载进度和单次查询选择由各查询集群本地管理，不复制。升级顺序上建议 secondary 先升，但最终一致性由开关 + 控制面保证，内核不做强约束。

---

## 5. 成本小结（按「改哪、改什么、影响谁」展开，不做高/中/低拍脑袋）

| 层 | 当前 | 主线改动 | 影响范围 |
|---|---|---|---|
| DataCoord 校验 | 同字段拒建第二个索引 | 放开校验，index_name 保持 collection 内唯一 | 建索引入口一处；构建/产物不动 |
| Proxy Load 解析 | 拍平成 `fieldID->indexID` | 改成 `fieldID->[]indexID`，Vector 校验一个 | Load 入口一处 |
| Load 状态 proto/meta | `FieldIndexID map[int64]int64` | 升级为多 indexID 结构 | query_coord.proto、load_info.go、querycoord 内存态、load_config.go |
| WAL Load 消息 | `LoadFieldConfig{field_id,index_id}` | 改为可带多 indexID | messages.proto、CDC 消费端 |
| QueryCoord 检查器 | 已按 indexID | 基本不动 | 只换「期望集合」来源 |
| Segcore 配置 | `FieldId->FieldIndexMeta` | 改为一对多 | IndexMeta.h、加载入口 |
| Segcore 运行时 Scalar | `FieldId->一个 ScalarIndex` | 改为按 indexID 挂多个、表达式按 indexID 选 | FieldIndexing.h、exec/expression/* |
| Segcore 运行时 Vector | `FieldId->一个 VectorIndex` | 主线不动，只让加载哪个可选 | Load 目标切换 + reopen |
| 真正多 Vector 加载 / 无停机热切换 | — | 见第 3 节，V2 成本 | VectorFieldIndexing、search 路由、growing、资源层 |

一句话成本判断：主线（Scalar 多加载 + Vector 单加载可选）改动集中、链路清晰，最大工作量在 Segcore 的 Scalar 索引挂载与选择；而「真正同时加载多 Vector 索引 / 无停机热切换」的成本主要在 Segcore 向量索引持有结构与查询路由，和主线不是同一量级，应单独立项评估。
