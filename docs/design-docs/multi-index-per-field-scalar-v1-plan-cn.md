# 一列多索引（标量 v1）实现方案

> 前置文档：[multi-index-per-field-pre-research-background-cn.md](./multi-index-per-field-pre-research-background-cn.md)（背景与改造面）、[一列多索引：背景与改造面.md](./design_docs/一列多索引：背景与改造面.md)（kernel view）。
> 本文是**重新审计后**的实现方案。前置文档的结论有 5 处已经过时（第 1 节），**第 3 节列出的 8 条现有不变量是本文最重要的增量** —— 不遵守它们会静默改变执行计划和索引可见性。

## 0. 基线与状态

- **代码基线**：`upstream/master` @ `addade7294`（2026-10-09）。前置文档基于 2026-08-13 的代码（`c04e16fbc0`），中间 upstream 前进了 380 个提交。
- 前置文档中标注为"现有代码"的行号与判断，全部按新基线重新核对过。
- 标记约定：**[已核对]** = 逐行读过代码确认；**[手工追踪]** = 按代码推导成立、未实际跑过复现；**[待实现时确认]** = 动手时必须先看的点。

**当前状态**：

- **反例已跑（2026-10-10，实证）**，两个用例都通过，1.2 从 [手工追踪] 升为 **[已核对·实证]**：
  - `SegmentLoadInfoTest.GetLoadDiffWithTwoScalarIndexesOnSameField` —— 同字段两条 `index_infos` → `indexes_to_load[field].size() == 2`、`indexes_to_replace` 为空（**链条 1-2 环**）
  - `TestChunkSegment.MultiScalarIndexOnSameFieldAsserts` —— 加载第二个同字段标量索引时真的命中断言，`--gtest_filter` 两个参数化实例（pk 为 int64 / varchar）均通过（**链条 3-4 环 + 断言本体**）
  - 运行日志实测到断言文本：`Assert "!has_index" => scalar index has been exist at 100`（`ChunkedSegmentSealedImpl.cpp:807`），抛出 `SegcoreError`
  - 构建配置为 `-a OFF`（无 ASAN）→ 顺带确认 `AssertInfo` 在 Release 下**依然生效**，不是被 `NDEBUG` 编译掉的 `assert`
- 1.3（无限 Reopen 循环）仍为 **[手工追踪]**：反例只证明了断言必然触发，没走到"断言没触发时的替换循环"那条分支。修好后 `AllLoadIndexesLoaded` 之类路径由 S3 的容器改造消除，不再依赖对 1.3 的判断。
- 跨集群按 ground truth 处理，见 §2.2。

---

## 1. 与旧调研的差异（审计结论）

### 1.1 "Load 单值链路是第一个真正丢索引的地方" —— 不成立 [已核对]

`field_indexID` 这条链（Proxy → QueryCoord 持久化 → WAL `LoadFieldConfig`）**从来没有驱动 QueryNode 加载哪些索引**：

| 环节 | 现状 | 证据 |
|---|---|---|
| 公开内部请求 | 仍是 `map<int64,int64>` | `pkg/proto/query_coord.proto:245-258`（LoadCollection，map 在 `:252`）、`:275-290`（LoadPartitions，map 在 `:283`） |
| Proxy 填值 | 同字段多个索引**静默覆盖、后写胜**，且顺序随机 | `internal/proxy/task.go:3159-3163`、`:3426-3430` |
| 实际加载什么 | DataCoord 报告 Finished 的**全部**段索引，无过滤 | `internal/datacoord/index_service.go:1253-1298` |
| 集合级索引元数据 | **全部**集合索引 | `internal/querycoordv2/task/executor.go:772-797` → `ListIndexes` |
| 唯一读取者 | `meta.Collection.GetFieldIndex()` **无任何非测试调用者** | `internal/querycoordv2/meta/collection_manager.go:379` |
| WAL | `LoadFieldConfig{field_id, index_id}` 单值，但只被 QueryCoord 自己的 ack 回调消费 | `pkg/proto/messages.proto:396-399`、`internal/querycoordv2/ddl_callbacks_alter_load_info.go:32-53` |

**结论**：v1 **不需要改 proto / QueryCoord 持久化 / WAL**。列表化的意义是"表达用户意图 / 资源选择 / CDC 语义"，不是"让索引起作用"。→ 决策 D1：不做。

**但有一个真实副作用必须处理**：Proxy 的覆盖顺序来自 `DescribeIndex` 遍历 `map[UniqueID]*model.Index`（`index_meta.go:79`），**顺序随机**。两次 Load 之间若选中不同 indexID，WAL load config 就不同（`GenerateAlterLoadConfigMessage` 用 `proto.Equal` 判变，`load_config.go:197`），会重新广播并把已 `Loaded` 的 collection 打回 `Loading/0`。→ S2。

### 1.2 "C++ 只是按字段只保存一个索引" —— 实际是**断言失败** [已核对·实证]

`LoadScalarIndex` 在 `is_replace=false` 且该字段已有索引时 `AssertInfo(!has_index)`（`ChunkedSegmentSealedImpl.cpp:1047-1050`）。推导链：

1. 段加载时**整段一次性**交给 C++：`cSegmentImpl.Load` → `C.AsyncSegmentLoad(ctx, s.ptr)`（`internal/util/segcore/segment.go:329-345`），索引信息在 `CreateCSegment` 时就随 `SegmentLoadInfo` 一起进去了（`internal/querynodev2/segments/segment.go:491-512`）。所以同字段的两个索引在**同一个** `SegmentLoadInfo` 里。
2. `ComputeDiffIndexes` 把它们**同时**放进 `indexes_to_load[field]`（`SegmentLoadInfo.cpp:294-303`），是个 2 元素向量。
3. `LoadBatchIndexes`（`.cpp:9196-9240`）为每个元素提交一个 future，每个 future 在 `committer.Commit(...)` 里调 `LoadIndex(..., is_replace=false, ...)`。
4. `StagedStateCommitter::Commit` 持互斥锁（`ChunkedSegmentSealedImpl.h:1592-1598`），提交后 `NormalizePublishedState` 把 `scalar_indexings` 折进 `index_ready_bitset`（`.cpp:1647-1657`）。
5. 第二次 `LoadIndex` 读到的 `has_index == true` → **断言触发**。

**为什么 JSON 多 path 反而没事** [已核对]：`json_indices` **没有**被折进 `index_ready_bitset`（`NormalizePublishedState` 只折 `scalar_indexings` / `ngram_fields` / `ngram_indexings`），所以 JSON path 索引走同一段代码时 `has_index` 始终 false。JSON **NGRAM** 例外——它走 `ngram_indexings`，**会**被折进 bitset（`.cpp:1669-1677`），所以"同一 JSON 字段两个不同 path 的 NGRAM 索引"理论上会命中同一个断言。**[手工追踪，未复现]**

另外 JSON 的 `indexes_to_replace` 事实上等效于"追加"：`is_replace=true` 时 JSON 分支按 **path** 擦除（`EraseJsonIndexesAtPath`），而 path 是新的，擦不掉东西（`.cpp:1058-1061`）。

### 1.3 只放开创建校验还有第二个后果：无限 Reopen 循环 [手工追踪]

```
ComputeDiffIndexes: 新 indexID + 字段已有索引 → indexes_to_replace   (SegmentLoadInfo.cpp:294-303)
  → is_replace=true → cancel_and_erase_scalar_index 先删再覆盖        (.cpp:1040-1041, 1121-1123)
  → 段上只剩新索引
  → QueryNode 上报的是实际加载的索引                                  (data_distribution_report.go:134-136)
  → IndexChecker 发现另一个缺失 → 再发 Reopen                          (index_checker.go:201-215, 235-259)
  → 再替换 → 循环
```

### 1.4 `index_info_list` 是死字段；QueryNode 的 `LoadCollection` RPC 没实现 [已核对]

- `LoadCollectionRequest` 没有 `index_info_list`；`LoadPartitionsRequest.index_info_list = 10`（`query_coord.proto:287`）**全仓无人读写**。
- QueryNode 的 `LoadPartitions` 完全忽略 `req.GetIndexInfoList()`（`querynodev2/services.go:447-467`）。
- `LoadCollection` RPC 在 QueryNode 侧无 Go 实现（`internal/distributed/querynode/service.go:361,373` 只有 `LoadSegments`/`LoadPartitions`）。
- 真正生效的是 `LoadSegmentsRequest.IndexInfoList` / `WatchDmChannelsRequest.IndexInfoList` / `SyncDistributionRequest.IndexInfoList`。

→ 前置文档说"公开 proto/SDK、Proxy、QueryCoord、WAL 消息都要动"，影响面比想象小得多。

### 1.5 `HybridScalarIndex` 与 JSON 特例的机制描述不准确 [已核对]

- `HybridScalarIndex` 是**按基数在 bitmap / stlsort / marisa 之间选一个物理实现**（`HybridScalarIndex.h:257-258`），与 path 无关。
- `JsonHybridScalarIndex<T>` 只持**一个** `nested_path_`、一个 `cast_type_`（`JsonHybridScalarIndex.h:37-60`）。
- 多 path 实际是 **N 个兄弟索引对象**，存在 `std::vector<JsonIndex> json_indices`（`ChunkedSegmentSealedImpl.h:348`），按 `(field_id, nested_path)` **线性扫描**（`PinJsonIndex` `.cpp:585-635`）。
- `JsonIndex = {field_id, nested_path, cast_type, index}`（`.h:323-328`）——**不含 indexID**。

### 1.6 仍然成立的部分 [已核对]

- 存储层天然支持多索引（etcd key 按 indexID/buildID）；DataCoord 其余部分**全部 indexID 粒度**，唯一按字段唯一的是 `canCreateIndex`。
- **QueryNode Go 侧已就绪**：`fieldIndexes ConcurrentMap[int64, *IndexedFieldInfo] // indexID`（`segments/segment.go:448`）、`fieldID2IndexInfo map[int64][]*FieldIndexInfo`（`segment_loader.go:792`）、上报按 indexID（`data_distribution_report.go:134-136`）。
- C++ `LoadIndexInfo` **已经带 `index_id`**（`segcore/Types.h:45`）→ 容器改造不需要新增协议字段。
- `FieldIndexMeta.index_id` proto 字段仍然**两端都不用**（Go 侧 `segments/index_meta.go:34-42` 不写，C++ `IndexMeta.cpp:41-48` 不读，proto 在 `segcore.proto:92`）。
- 选择层现状：`num_index_chunk_ == 1` 7 处断言、`pinned_index_[0]` 11 处使用、`pinned_indexes_`（复数）**不存在**。
- **前置文档说"内部预留 indexID"是错的**：`plan.proto` 里连一个 `index` 子串都没有（0 匹配）。

### 1.7 两个月内的新变化

- **DataCoord 就绪模型已是 indexID 粒度**（`#53963`）：`completeIndexInfo` 按 indexID 判定，`newIndexCoverageChecker` 的 ancestor coverage 绑单个 indexID（`index_service.go:806-855, 866-967`）。**对多索引利好**。`SegmentIndex.IsDeleted` 现在会持久化。
- **packed scalar index / StorageV3 / 异步加载**（`#53483`、`#53048`）：与"每字段一个索引"正交；但 `SegmentLoadInfo` 的 `field_index_id_cache_`（`SegmentLoadInfo.h:1252`）**已是 indexID 粒度记账**。
- **`bound_field_indexes`**（`messages.proto:371`，`rootcoord/ddl_callbacks_alter_collection_properties.go:531-572`）：AddField 时自动物化索引，不阻塞多索引，但放开校验后仍只建一个。
- **标量索引引擎版本机制成熟**（`pkg/common/common.go:120-158`）——S0 复用它。

---

## 2. 范围与决策

**目标**：同一标量/JSON 字段可以建多个索引、**同时加载**、每个过滤条件按能力挑一个执行，挑不到退回原始扫描；原始数据始终保留。

### 2.1 决策表

| # | 决策 | 结论 |
|---|---|---|
| D1 | `field_indexID` 列表化（proto + 持久化 + WAL） | **不做**，只让 Proxy 选择确定化（S2） |
| D2 | C++ `CollectionIndexMeta` 列表化 | **不做**，只改段级容器（S3） |
| D3 | 创建判重身份键 | **只按 `index_name` 判重**，不限数量 |
| D4 | 多候选选择 | 默认 = 优先级表 + 复用 `ShouldUseOp`；策略对象可替换 |
| D5 | Vector 字段 | **保持"只能建一个"** |
| D6 | 选中后被索引拒绝算子时 | **按序尝试下一个候选**，全被拒才回 raw |
| D7 | 默认优先级表粒度 | **索引能力类 × 算子类**（不是 `is_ngram` 二分） |
| D8 | 空 `index_name` 时字段已有索引 | **自动生成唯一默认名** |
| D9 | JSON 同 path 多 cast | **放开** |
| D10 | 滚动升级门禁 | **能力协商**（复用 scalar index engine version 机制） |
| D11 | 跨集群一致性 | **不做加载侧兜底**；以"备集群先升级"为 ground truth（§2.2） |
| — | 原始数据 | 始终保留独立可读路径 |
| — | 成本选优 | 不做 |
| — | 显式指定查询索引 | 不做 |

### 2.2 跨集群：ground truth 与它的运维契约

**Ground truth（不再讨论）**：备集群保证**先完成升级**。因此 v1 不做"加载侧按能力裁剪"这类跨集群兼容方案。

**必须写进 release note 的契约**（因为内核不再保护这条路径）：

1. 升级顺序：备集群**全部节点**（DataCoord / QueryCoord / QueryNode）先升到含本特性的版本，主集群后升。
2. **回滚约束**：本特性启用并已创建"同字段多索引"之后，回滚到老版本前**必须先 drop 掉多余索引**，否则老 QueryNode 加载时会命中 1.2 的断言。
3. 跨集群 DDL 复制**绕过创建侧门禁**：`ddl_callbacks_create_index.go` 的 ack 回调直接 `indexMeta.CreateIndex`，没有任何校验。所以 S0 的门禁只保护本集群，备集群的安全性完全由第 1 条保证。

### 2.3 为什么 D6 必须"按序尝试"

索引会**主动拒绝**算子（实测代码，非推测）：

- `InvertedIndexTantivy::ShouldUseOp` 对 `RegexMatch` / `PostfixMatch` / `InnerMatch` **返回 false**（`InvertedIndexTantivy.h:310-313`）。
- `FMIndex::ShouldUseOp` 的 `default` **返回 false**，拒绝等值族（Equal/NotEqual/IN）和字典序范围（`FMIndex.h:307-308`），注释原文：`fall back to the scan / another index`。
- `ScalarIndex<T>::ShouldUseOp` 对**非 pattern 算子一律返回 true**（`ScalarIndex.h:241-253`）。

所以"选一个、被拒就回 raw"在 inverted + FMINDEX 这类常见组合下会**频繁白丢索引**。而 v1 没有"用户显式指定索引"，候选又都在同一字段上，**候选间切换不改变结果语义**。失败的那次 pin 由 `PinWrapper` 的 RAII 自动释放。

---

## 3. 必须保住的 8 条现有不变量 ★

这一节是本次完善方案的核心增量。这些语义在现网被依赖（尤其是执行计划采样和 JSON 可见性），改错不会报错、只会静默变行为。

| # | 不变量 | 现状与证据 | 新方案怎么保住 |
|---|---|---|---|
| **I1** | `index_ready_bitset[field]` 为真的条件 | 折进 bitset 的是 `scalar_indexings` / `ngram_fields` / `ngram_indexings`（`.cpp:1647-1677`）。**JSON 非 NGRAM 的 path 索引不折进** | 折进条件 = `entry.json_path.empty() \|\| entry.is_ngram` |
| **I2** | `HasJsonIndex(field)` = "JSON path 索引中**非 NGRAM** 的存在" | `HasJsonIndex` 扫 `json_indices`（`.cpp:6925-6936`），而 JSON NGRAM 走 `ngram_indexings` 不进 `json_indices` | = 存在 `!json_path.empty() && !is_ngram` 的 entry |
| **I3** | `HasIndex()` 是**执行计划采样输入**，必须保持"窄" | `Expr.cpp:617 p.has_index = segment->HasIndex(...)`；`Expr.cpp:693`、`:817` 也用它做计划决策 | 由 I1 保住（不要"顺手"把 JSON 也折进 bitset） |
| **I4** | NGRAM 索引**只服务 pattern 类叶子** | 非 JSON：`PinIndex(include_ngram=false)` 在 `ngram_fields` 命中时返回空（`.h:139-159`）→ RawData。JSON：`PinJsonIndex` 不碰 `ngram_indexings` | `SelectAndPinIndex` 的接受性检查加一条：候选 `capability == PatternIndexed` 且本叶子**不含** pattern 类算子 → 跳过该候选 |
| **I5** | `num_index_chunk_ == pinned_index_.size()`，且 `pinned_index_[0]` 是唯一索引 | `EnsurePinnedIndex()` 里 `num_index_chunk_ = pinned_index_.size()`（`Expr.h:538-554`） | pin 恰好一个；7 处断言与 11 处 `[0]` **不动** |
| **I6** | `HasCompatibleScalarIndex()` 必须**不 pin** 就能用 | 注释 `Expr.h:487-496`：short-circuit 路径和 RawData 路径不该付 `PinCells()` 的 cold fetch | `GetScalarIndexCandidates()` 只读元数据；只有 `SelectAndPinIndex` 才 pin |
| **I7** | JSON flat index 的 path 前缀匹配 + 数字下标拒绝 | `IsJsonPathCompatible()`（`Expr.h:3470-3500`）用 `GetJsonFlatIndexNestedPath()`，后者在 `cast_type == JsonCastType::DataType::JSON` 的索引里做**最长前缀匹配** | 逻辑原样保留，作为 JSON 候选的 path 预过滤；`GetJsonFlatIndexNestedPath` 改扫统一容器 |
| **I8** | 只 pin **真正会被用到**的那一个 | 同上注释 | 策略只排序；循环里第一次被接受就停止，后续候选不 pin |

---

## 4. 实现步骤

### S0. 滚动升级门禁（能力协商）——必须先做

**做法：完全复用现成的 scalar index engine version 机制**，不加新协议字段。

1. `pkg/common/common.go`：`CurrentScalarIndexEngineVersion` / `MaximumScalarIndexEngineVersion` 从 `5` 提到 `6`，并按现有格式补注释块：

```go
// Scalar index engine version 6:
// - Multiple scalar indexes may coexist on one field: they are loaded
//   together and each filter leaf picks one by capability. An older
//   QueryNode holds at most one scalar index per field and asserts while
//   loading such a segment, so creating the second index on a field is
//   gated on the whole cluster reporting >= 6
//   (see MinScalarIndexVersionForScalarMultiIndex).
// - On-disk file format is unchanged from v3.
MinScalarIndexVersionForScalarMultiIndex = int32(6)
```

2. 门禁函数放 `internal/datacoord/index_service.go`，与 `checkFMIndexEngineVersion`（`:139-149`）并列，用 `ResolveScalarIndexVersion()` 比对。

3. **QueryNode 上报代码不用动** —— `querynodev2/server.go:175-178` 直接引用这三个常量。

**为什么天然安全**：`ResolveScalarIndexVersion()` 取**所有 QueryNode session 的 `CurrentIndexVersion` 的最小值**（`index_engine_version_manager.go:256-261`），集群里有一个老 QueryNode 就是 5 → 拒绝；没有 session 时返回 **0**（`noSessionVersion`，`:344`）→ 拒绝。与 FMINDEX 门禁同模式，可对照 review。

**不会连累索引构建**：`load_index_c.cpp:88` 和 `SegmentLoadInfo.cpp:145` 只是把版本号塞进 `index_params["scalar_index_engine_version"]`，**没有任何"版本过高就拒绝加载"的逻辑**（已 grep 确认）。v6 不引入新构建行为。**上线前确认** `dataCoord.forceRebuildScalarSegmentIndex` 关闭且 `dataCoord.targetScalarIndexVersion` 未设成 6，否则会触发全量标量索引重建。

**门禁的精确适用范围**（避免误伤现有能力）：

| 场景 | 老 QueryNode 能加载吗 | 门禁 |
|---|---|---|
| 非 JSON 标量字段的第 2 个索引 | **不能**（1.2） | **需要** |
| JSON 字段的第 2 个 path | 能 | 不需要 |
| JSON **同 path** 多 cast | **不能**（`EraseJsonIndexesAtPath` 会删兄弟 cast；`PinJsonIndex` 会选错） | **需要** |

→ 实现成：**"本请求是否落入一个已存在索引的 `(isJSON, json_path)` 身份"**。不同身份的 JSON 多 path 直接放行。

---

### S1. DataCoord 放开创建校验

**位置**：`datacoord/index_meta.go:511-555` `canCreateIndex`；wrapper 同文件 `:500-509`；调用点 `index_service.go:300`。

**改动 1：把字段类型策略移到 Server（实现时的修正）**

原计划是改 `canCreateIndex` 的签名（`isJSON bool` → 字段类型）。实现时改成**保留签名**，把字段类型策略整体移到 Server：

- `indexMeta.canCreateIndex` **收缩为只管一件事**：这个名字是否已被 collection 里的另一个索引占用（同名 + 同字段 + 同参数 → 幂等忽略；其余同名 → 报错）。
- 新增 `Server.checkIndexCreationPolicy(ctx, req, isJSON, fieldType)` 统一处理两种字段类型策略：向量字段拒绝第二个索引；标量/JSON 在同一 `(field, json_path)` 身份上的第二个索引走 S0 门禁。

理由：① 策略集中在一处（Server 有 schema、也有版本管理器），`indexMeta` 保持"纯元数据/判重"；② **`CanCreateIndex` 签名不变 → 21 处测试调用点零改动**；③ 向量拒绝与标量门禁本来就需要同一次"同字段已有索引"扫描，合并后只扫一次。

**改动 2：判重逻辑（D3）**

`canCreateIndex` 里原来的两段：

- 同名分支：保持"同名 + 同字段 + 同参数 → 幂等忽略"，其余同名**改文案**为 `index name is already used by another index in this collection`（原来是 `at most one distinct index is allowed per field`，码 1100，已不准确）。
- 同字段分支：**整段删除**。放行与拒绝的判断都交给 `checkIndexCreationPolicy`；JSON 的 `jsonPath1 != jsonPath2 → continue` 特例随之消失。
- 连带地 `index_meta_test.go` 的 `"multiple indexes"` 子测试从 `assert.Error` 改为 `assert.NoError`（重命名为 `"multiple scalar indexes on one field"`），因为"同字段第二个标量索引"在元数据层现在是合法的。

**改动 3：默认索引名解析（D8）**

现在是 `index_service.go:250-275`，语义是"字段已有索引就复用它的名字"，所以第二个索引会被当成"同名不同参数"报错。改成按顺序：

```
1. 在**同字段**未删除索引里找参数完全相同的（checkParams + checkIdenticalJSON）
   → 找到就复用它的名字（后续 canCreateIndex 返回 errIndexOperationIgnored，保持幂等）
2. 否则 base = <fieldName>（非 JSON）或 <fieldName><jsonPath>（JSON，保持现状）
   base 在**整个 collection** 内未被占用 → 用它
3. 否则 base + "_" + <index_type 小写>
4. 否则 base + "_" + <index_type 小写> + "_" + <n>（n 从 2 递增，上限 100，超了报错）
```

- **第 1 步是保持现有幂等语义的关键**：同一请求重发两次仍解析到同一个名字，不会创建出第二个索引。
- 唯一性检查用 **collection 范围**（`GetIndexIDByName(collID, name)`），因为现有"同名 + 不同字段 → 报错"也是 collection 范围的。
- `index_type` 为空或 `AUTOINDEX` 时退化成第 4 步的序号后缀。

**顺带记录（不修）**：`index_service.go:293` 每次 CreateIndex 都 `_, err = s.allocator.AllocID(ctx)` 然后丢弃结果。

---

### S2. Proxy 默认索引确定化

`internal/proxy/task.go:3159-3163` 和 `:3426-3430`，把"后写胜"改成"取 indexID 最小者"：

```go
// v1 不支持用户选择索引；这里只为了让 load config 稳定（见 S2 说明），
// 不代表任何用户可见的"默认索引"语义。取 indexID 最小 = 最早创建的那个。
for _, index := range indexResponse.IndexInfos {
    if prev, ok := fieldIndexIDs[index.FieldID]; !ok || index.IndexID < prev {
        fieldIndexIDs[index.FieldID] = index.IndexID
    }
}
```

**不要**把它包装成 `DefaultIndexForField()` 之类的抽象——v1 没有"默认索引"概念。

---

### S3. C++ 段级容器统一到 indexID

目标：**字段 → indexID → 一个 entry**，把普通 scalar、JSON path/cast、JSON NGRAM 三个特例收编进同一个容器。因为 D9 放开了同 path 多 cast，`(field, path)` 不再是唯一键，**必须用 indexID**。

#### S3.1 容器结构（`ChunkedSegmentSealedImpl.h` `RuntimeResourceState`）

现在（`.h:337-352`）：

```cpp
std::unordered_map<FieldId, index::CacheIndexBasePtr> scalar_indexings;              // :337
std::unordered_set<FieldId>                           ngram_fields;                  // :341
std::unordered_map<FieldId, std::unordered_map<std::string, index::CacheIndexBasePtr>>
                                                      ngram_indexings;               // :342-345
std::vector<JsonIndex>                                json_indices;                  // :348
```

改成：

```cpp
enum class ScalarIndexCapability { Unknown, TermIndexed, PatternIndexed };

struct ScalarIndexEntry {
    int64_t index_id{0};
    std::string index_type;                    // index_params["index_type"]，注意段级覆盖
    std::string json_path;                     // 空 = 非 JSON path 索引
    JsonCastType json_cast_type{JsonCastType::UNKNOWN};
    bool is_ngram{false};                      // index_type == index::NGRAM_INDEX_TYPE ("NGRAM")
    ScalarIndexCapability capability{ScalarIndexCapability::Unknown};
    index::CacheIndexBasePtr cache_index;
};
// field -> indexID -> entry
std::unordered_map<FieldId,
    std::unordered_map<int64_t, ScalarIndexEntry>> scalar_indexings;
```

- `capability` 在加载时从 `index_type` 推导：`INVERTED` / `BITMAP` / `STL_SORT` → `TermIndexed`；`NGRAM` / `FMINDEX` → `PatternIndexed`；`HYBRID` / `AUTOINDEX` / 未知 → `Unknown`。**[可选精化]** `ResolvePackedHybridIndexType`（`HybridScalarIndex.cpp:95`）能从 packed 文件名解析出物理类型，可以把 `Unknown` 填准。
- 删除 `ngram_fields`、`ngram_indexings`、`json_indices`、`JsonIndex`（`.h:323-328`）、`SyncJsonNgramIndexState`（`.cpp:1548`，4 处调用）。
- `json_stats`（JsonKeyStats）是统计不是索引，**不动**。

**使用点统计**（`internal/core/src`，含测试）：`scalar_indexings` 20 / `ngram_fields` 11 / `ngram_indexings` 13 / `json_indices` 11。

#### S3.2 由容器派生的访问规则（对齐 §3 的 I1–I4/I7）

| 接口 | 新规则 |
|---|---|
| 折进 `index_ready_bitset` | `e.json_path.empty() \|\| e.is_ngram` —— **I1** |
| `HasJsonIndex(field)` | 存在 `!e.json_path.empty() && !e.is_ngram` —— **I2** |
| `GetJsonFlatIndexNestedPath(field, qp)` | 在 `e.json_cast_type.data_type() == JsonCastType::DataType::JSON` 的 entry 里做最长前缀匹配 —— **I7** |
| `GetNgramIndex(field)`（非 JSON） | 唯一的 `json_path.empty() && is_ngram`（多个则取 indexID 最小） |
| `GetNgramIndexForJson(field, path)` | `json_path == path && is_ngram`（多个则取 indexID 最小） |
| `GetScalarIndexCandidates(field)` | 映射整个内层 map 为 `std::vector<ScalarIndexCandidate>` |
| `PinScalarIndex(field, index_id)` | 只在 `e.json_path` 匹配时 pin（typed 精确匹配 / flat 前缀匹配）；cast 兼容由调用方的接受性检查做 |

#### S3.3 加载 / 删除

- `LoadScalarIndex`（`.cpp:980-1150`）：
  - 用 `info.index_id`（`segcore/Types.h:45` 已有）作为 key。
  - **删掉 `AssertInfo(!has_index)`**（`:1047-1050`）。改成：`(field, index_id)` 已存在时，`is_replace` 为真才 retire 旧的并覆盖，否则幂等返回。
  - 删除 JSON 专用分支（`:1052-1110`）；`json_path` / `json_cast_type` / `is_ngram` / `capability` 从 `info.index_params` 填进 entry，写入 `scalar_indexings[field_id][info.index_id]`（**不再覆盖同字段别的索引**，`:1121/:1123` 旧写法作废）。
- `cancel_and_erase_scalar_index`（`:500-509`）→ 增加 `index_id` 参数。
- `EraseJsonIndexings` / `EraseJsonNgramIndexing` / `EraseJsonIndexesAtPath`（`:1486-1538`）→ **删除**，统一按 `(field, index_id)` 删。身份键是 indexID，同 path 多 cast 是不同 indexID；`AlterIndex` 改 path 只是覆盖同一个 indexID 的 entry。
- `DropIndex(field_id)`（`:4730-4747`）→ `DropIndex(field_id, index_id)`；`DropIndexFromState`（`:1889`）同理。
- `CloneRuntimeResourceState` / `FreezeRuntimeResourceState`（`:1035-1042`、`:1373-1376`、`:1767-1770`）→ 成员变少，拷贝点收敛。

#### S3.4 LoadDiff 与 ComputeDiffIndexes

`SegmentLoadInfo.h:52-137`：

```cpp
// 保留
std::unordered_map<FieldId, std::vector<LoadIndexInfo>> indexes_to_load;
// 保留，语义收窄为"向量字段换索引"；标量不再进这里
std::unordered_map<FieldId, std::vector<LoadIndexInfo>> indexes_to_replace;
// 改：从 std::set<FieldId> 改成 (field, indexID)
std::unordered_map<FieldId, std::unordered_set<int64_t>> indexes_to_drop;
// 删：json_indexes_to_drop（被 indexes_to_drop 吸收）
```

`ComputeDiffIndexes`（`SegmentLoadInfo.cpp:251-329`）：

```cpp
for (const auto& [field_id, load_index_infos] : new_info.converted_field_index_cache_) {
    if (!new_info.HasFieldInSchema(field_id)) continue;
    const bool vector_field = IsVectorField(new_info.schema, field_id);
    for (const auto& info : load_index_infos) {
        if (current_index_ids.count(info.index_id)) continue;   // 已在，保持现状 no-op
        if (vector_field && current_indexed_fields.count(field_id)) {
            diff.indexes_to_replace[field_id].push_back(info);  // 向量：保持替换语义
        } else {
            diff.indexes_to_load[field_id].push_back(info);     // 标量：追加
        }
    }
}
// drop 改成逐个 indexID 判
```

`current_indexed_fields`（字段级）只对**向量**字段继续用作"要不要替换"的判据；标量一律按 indexID 追加 —— 这正好保住向量语义不变。`FinalizeLoadDiffForReopen`（`.cpp:8092-8130`）的 drop 循环改成按 `(field, index_id)`，"同字段有 replace/load 就跳过"的守卫改成"**按 indexID** 不冲突"。

**连带简化**：`json_index_path_cache_`（`SegmentLoadInfo.h:1254-1258`）存在的唯一理由是"按 path 删除会丢兄弟身份"，indexID 化后不再需要。**[待实现时确认它是否还有其他用途（manifest / `CompactRuntimeInfoForManifest`）后再删]**

#### S3.5 raw data

`IndexHasRawData`（`.cpp:1807`，消费者 `.cpp:6818`、`.cpp:9293`、cgo `segment_c.h:220`）是"索引自带原始数据 → 可以不加载字段列"的优化。v1 决策是原始数据始终保留，所以**保守处理：字段下有多于一个索引时返回 false**（强制保留字段列），避免 fallback 到 raw 时找不到列。**[待实现时确认]** 单索引时的现有行为不动。

---

### S4. 选择层

#### S4.1 现状与**全部 6 个 pin 通道** ★

| # | 通道 | 位置 | 选择准则 | 现状 |
|---|---|---|---|---|
| C1 | `SegmentExpr` 基类（7 个 leaf 共用） | `Expr.h:538-554` `EnsurePinnedIndex` → 免费函数 `PinIndex`（`Expr.h:85-104`） | 按算子（op 在子类 refine 段才问） | 字段唯一索引 |
| C2 | 7 个 leaf 的 refine 段 | `UnaryExpr.cpp:2075`、`BinaryRangeExpr.cpp:1190`、`TermExpr.cpp:1246`、`NullExpr.cpp:137`、`ExistsExpr.cpp:48`、`GISFunctionFilterExpr.cpp:237`、`TimestamptzArithCompareExpr.cpp:157` | `CanUseIndexForOp<T>(op)` | 同上 |
| C3 | `MembershipFilterExpr` | `MembershipFilterExpr.h:197-205` | op + **反向查找**（`IndexSupportsReverseLookup()`） | `HasCompatibleScalarIndex()` + `EnsurePinnedIndex()` |
| C4 | `PhyColumnExpr` | `ColumnExpr.h:60-64` | **要 raw data**（`use_index_data_ = is_indexed_ && segment->HasRawData(field)`） | 免费 `PinIndex` |
| C5 | `PhyCompareExpr` | `CompareExpr.h:188-195` | 同上，左右各一个 | 免费 `PinIndex` ×2 |
| C6 | `SearchGroupByOperator` | `SearchGroupByOperator.h:252` | 要 raw data（分组读列） | `segment_.PinIndex(op_ctx_, field_id_)` |

**C4/C5/C6 是原方案漏掉的**。它们的准则是"索引能不能提供原始数据"，不是算子。好消息：因为 S3.5 让 `IndexHasRawData` 在多索引时返回 false，`use_index_data_` 会自然变成 false → 退回从 `fields` 列读，**pin 到哪个都无所谓**。所以 v1 对它们的处理是：**pin 有序候选里的第一个**（确定性的 indexID 最小者），并保持 `is_indexed_` 语义不变。**[待实现时确认]** `is_indexed_` / `left_is_indexed_` 除 `use_index_data_` 外是否还有别的用途。

#### S4.2 关键洞察：`pinned_index_` 不需要变成多份

只需把"取哪个"从"字段那唯一一个"改成"字段下按策略**排序后依次尝试**"。`num_index_chunk_ == 1` 的 7 处断言和 `pinned_index_[0]` 的 11 处使用**全部保持不动**（I5）。

#### S4.3 把 `ShouldUseOp` 上提到 `IndexBase`

现在 `ShouldUseOp` 只声明在 `ScalarIndex<T>` 上（`ScalarIndex.h:241`），问一次就要 `dynamic_cast` 一次，且 `Expr.h:3584-3588` 带两处 `AssertInfo`（pin 到的不是 `ScalarIndex<T>` 就断言失败）。

改：`index/Index.h` 的 `IndexBase` 加默认实现，`ScalarIndex<T>` 的实现改成 `override`：

```cpp
// IndexBase
virtual bool ShouldUseOp(proto::plan::OpType /*op*/,
                         const std::string& /*pattern*/ = "") const {
    return true;   // 非标量索引（向量/文本）不参与标量 op 路由
}
```

好处：① 选择循环变成**非模板**、可放基类；② `CanUseIndexForOp<T>` 简化成 `pinned_index_[0]->ShouldUseOp(op, pattern)`，去掉 dynamic_cast 与两处断言；③ 不同类型候选可统一询问。

#### S4.4 新增接口

```cpp
struct ScalarIndexCandidate {
    int64_t index_id;
    std::string index_type;
    std::string json_path;
    JsonCastType json_cast_type;
    bool is_ngram;
    ScalarIndexCapability capability;
};

enum class IndexSelectionPurpose { Filter, ReadColumn };

struct LeafQueryDesc {
    IndexSelectionPurpose purpose{IndexSelectionPurpose::Filter};
    std::vector<proto::plan::OpType> ops;   // 通常 1 个；BinaryRange 是 2 个（lower/upper）
    std::string pattern;                    // pattern 类算子才有
    DataType field_type;
    // JSON 专用（对齐免费 PinIndex 的入参，Expr.h:85-92）
    std::string json_path;
    DataType value_type{DataType::NONE};
    bool allow_any_json_cast_type{false};
    bool json_is_array{false};
    bool int64_unsafe_for_double{false};    // UnaryExpr.cpp:2211-2218 的精度检查结果
};

class ScalarIndexSelector {
 public:
    virtual ~ScalarIndexSelector() = default;
    // 优先级从高到低排列的候选 indexID；空 = 无可用候选。
    virtual std::vector<int64_t> Order(
        const std::vector<ScalarIndexCandidate>& candidates,
        const LeafQueryDesc& leaf) const = 0;
};
```

返回值是**有序 indexID 列表**而非单个 id：未来换成本模型只换排序，接口形状不变。

配套的叶子描述入口（基类给默认实现，7 个 leaf + C3/C4/C5 各自 override）：

```cpp
// 只读元数据，不 pin。基类返回 {purpose}；叶子填自己的 ops / pattern / JSON 参数。
virtual LeafQueryDesc DescribeLeaf(
    IndexSelectionPurpose purpose = IndexSelectionPurpose::Filter) const;
```

**默认实现 `PriorityTableSelector`**（规则集中在这一个类，D7）：

1. **JSON 身份过滤**：`leaf.json_path` 非空时只留 `json_path == leaf.json_path` 的候选（含 `GetJsonFlatIndexNestedPath` 的最长前缀匹配 + 数字下标拒绝，I7）；为空时只留 `json_path` 为空的。同 path 多 cast 的候选都留下（cast 兼容性交给接受性检查）。
2. **按能力类分档**：
   - 含 pattern 类算子（Match/PrefixMatch/PostfixMatch/InnerMatch/RegexMatch）→ `PatternIndexed` → `Unknown` → `TermIndexed`；
   - 否则 → `TermIndexed` → `Unknown` → `PatternIndexed`。
3. 同档内 `index_id` 升序（稳定兜底）。

**分档只是排序，不是判定**：`HYBRID` 的内部实现是构建时按基数定的（`HybridScalarIndex.h:134-137` 转发给 `internal_index_`），声明不代表真实能力。**真正的判定永远是 pin 之后的 `ShouldUseOp`**（D6 的根据）。猜错只多付一次 pin。

**`SegmentInterface` 新增两个入口**：

```cpp
// 读元数据，不 pin（I6）
virtual std::vector<ScalarIndexCandidate> GetScalarIndexCandidates(FieldId field_id) const { return {}; }
// 按 (field, indexID) pin 一个
virtual std::vector<PinWrapper<const index::IndexBase*>>
PinScalarIndex(milvus::OpContext*, FieldId field_id, int64_t index_id) const { return {}; }
```

#### S4.5 执行路径改造

**基类 `SegmentExpr`**：

```cpp
// 替代 HasCompatibleScalarIndex()，仍然不 pin（I6）。purpose 默认 Filter。
bool HasScalarIndexCandidates(IndexSelectionPurpose purpose = IndexSelectionPurpose::Filter) const {
    if (CanUseNgramIndex()) return false;
    return !FilterCandidates(segment_->GetScalarIndexCandidates(field_id_),
                             DescribeLeaf(purpose)).empty();
}

// 替代 EnsurePinnedIndex()（I5：仍然只 pin 一个）
bool SelectAndPinIndex(const LeafQueryDesc& leaf) {
    if (pinned_index_initialized_) return !pinned_index_.empty();
    pinned_index_initialized_ = true;
    auto filtered = FilterCandidates(segment_->GetScalarIndexCandidates(field_id_), leaf);
    auto lookup = [&](int64_t id) -> const ScalarIndexCandidate* {
        for (const auto& c : filtered) { if (c.index_id == id) return &c; }
        return nullptr;
    };
    for (auto index_id : selector_->Order(filtered, leaf)) {
        auto pinned = segment_->PinScalarIndex(op_ctx_, field_id_, index_id);
        if (pinned.empty()) continue;                      // 冷索引/正在淘汰，试下一个
        if (!AcceptsLeaf(lookup(index_id), pinned[0].get(), leaf)) continue;
        pinned_index_ = std::move(pinned);
        num_index_chunk_ = pinned_index_.size();           // == 1（I5）
        return true;
    }
    return false;                                          // 全被拒 → 调用方回 raw
}

// DetermineExecPath（基类版）：只判断有没有候选，不 pin
virtual void DetermineExecPath() {
    if (!HasScalarIndexCandidates()) {
        exec_path_ = RawData; return;
    }
    exec_path_ = ScalarIndex;   // 试探性；SelectAndPinIndex 失败时调用方降回 RawData
}
```

`FilterCandidates(candidates, leaf)` 与 `AcceptsLeaf(candidate, index, leaf)` 的判定（**全部必须保留**）：

1. `FilterCandidates`：JSON 身份过滤（`Filter` 目的下含 `GetJsonFlatIndexNestedPath` 的最长前缀匹配 + 数字下标拒绝，I7）；`ReadColumn` 目的不做 JSON 过滤。
2. `index->ShouldUseOp(op, pattern)` 对 `leaf.ops` 里**每一个** op 都为真（`ReadColumn` 目的下 `ops` 为空，跳过）。
3. **I4**（仅 `purpose == Filter`）：候选 `capability == PatternIndexed` 且 `leaf.ops` 不含 pattern 类算子 → 拒绝。
4. JSON（仅 `purpose == Filter`）：`IsDataTypeSupported(cast_type, value_type, json_is_array)`（现有自由函数，`PinJsonIndex` 已在用）。

**7 个 leaf 的改动**（C2）：把 refine 段里的 `SegmentExpr::CanUseIndexForOp<T>(...)` 换成 `SegmentExpr::SelectAndPinIndex(DescribeLeaf(...))`，失败时沿用它们**已有的** `exec_path_ = RawData` 兜底（`UnaryExpr.cpp:2176-2250`、`BinaryRangeExpr.cpp:1233-1267`、`TermExpr.cpp:1292-1316`）。每个 leaf 新增 `DescribeLeaf()` 填自己的 ops / pattern / JSON 参数。

**`PhyBinaryRangeFilterExpr` 注意**：它同时问 `lower_op` 和 `upper_op`（`BinaryRangeExpr.cpp:1266-1267`），两个都能用才走索引 → `leaf.ops` 是 2 元素。

**C1 的免费 `PinIndex`（`Expr.h:85-104`）**：C1/C4/C5 共用。改造后它退化为"`GetScalarIndexCandidates` + `SelectAndPinIndex`（Filter 目的）"或"pin 有序候选第一个（ReadColumn 目的）"。**[待实现时确认]** 没有其他调用者（已 grep：仅 `Expr.h:546`、`ColumnExpr.h:60`、`CompareExpr.h:188-189`）。

**C3 `MembershipFilterExpr`**（`MembershipFilterExpr.h:197-205`）：`HasCompatibleScalarIndex()` + `EnsurePinnedIndex()` → `HasScalarIndexCandidates()` + `SelectAndPinIndex()`，然后照旧检查 `IndexSupportsReverseLookup()`。

**C4/C5/C6（ReadColumn 目的）**：`SelectAndPinIndex({purpose=ReadColumn})`，策略对 ReadColumn 只按 `index_id` 升序（不看算子）。因为 S3.5 让多索引字段的 `HasRawData` 为 false，`use_index_data_` 会是 false，pin 的结果是惰性的。

**NGRAM 通道收编**：`GetNgramIndex`（`.cpp:4126-4144`）/ `GetNgramIndexForJson`（`.cpp:4145-4174`）按 S3.2 的规则重写；`pinned_ngram_index_`（`UnaryExpr.h:1019-1021`）保留；`CanUseNgramIndex()`（`UnaryExpr.cpp:2387`）与 `ConjunctExpr.cpp:121` 不动。

**不动的**：`num_index_chunk_`、`pinned_index_[0]`、`PinnedJsonIndexIsFlat()`、`UseIndexCursor`、TextIndex/PkIndex/JsonStats 的 short-circuit（`UnaryExpr.cpp:2078-2103`）、`IndexHasRawData`。

---

### S5. 测试计划

**C++ UT**（`internal/core/src/segcore/`，`all_tests` 二进制）：

| 测试 | 位置 | 断言 |
|---|---|---|
| T1 同字段两索引进 `indexes_to_load` | `SegmentLoadInfoTest.cpp` 新增 case | `indexes_to_load[field].size() == 2`（1.2 的前置条件） |
| T2 同字段两索引可加载 | `ChunkedSegmentSealedImplTest.cpp` 新增 case | 不再抛 `AssertInfo`；两 entry 都在容器里；`index_ready_bitset` 置位（**这一条同时是 1.2 的正向验证**） |
| T3 drop 一个只删一个 | 同上 | 删 indexID-A 后 B 仍在且可用 |
| T4 reopen 不替换 | `SegmentLoadInfoTest.cpp` | 同字段第二个 indexID → `indexes_to_load`（不是 `indexes_to_replace`）；drop 是 `(field, indexID)` 粒度 |
| T5 **I1/I2 回归** | `ChunkedSegmentSealedImplTest.cpp` | JSON 非 NGRAM path 索引**不**置 `index_ready_bitset`；`HasJsonIndex` 为真；JSON NGRAM 置位且 `HasJsonIndex` 为假 |
| T6 **I4 回归** | 同上 | 只有 NGRAM 索引的字段 + `Equal` 叶子 → `DetermineExecPath() == RawData` |
| T7 选择器 | `PriorityTableSelector` 单测 | inverted + FMINDEX 下 `PostfixMatch` 把 FMINDEX 排前、`Equal` 把 inverted 排前；顺序与创建顺序无关 |
| T8 JSON 同 path 多 cast | 同上 | 两个 cast 都可加载；按 cast 选中；drop 只删一个 |

**Go UT**：

| 测试 | 位置 | 断言 |
|---|---|---|
| G1 放行标量第 2 个索引 / 拒绝向量第 2 个 | `datacoord/index_meta_test.go` | 错误码与文案 |
| G2 门禁 | `datacoord/index_service_test.go` | `ResolveScalarIndexVersion()` < 6 → 拒绝；≥ 6 → 放行 |
| G3 默认名生成 | `index_service_test.go` | 幂等（同请求两次 → 同名 + `errIndexOperationIgnored`）；已占用 → 加后缀；collection 级唯一 |
| G4 Proxy 确定性 | `proxy/task_test.go` | 同字段两索引时 `fieldIndexIDs[field]` 恒为最小 indexID（多次调用一致） |
| G5 IndexChecker | `querycoordv2/checkers/index_checker_test.go` | 同字段两索引时，只缺一个也发 Reopen（现有 field 粒度行为不回归） |

**回归**：`make test-go` 全量 + C++ `all_tests`（排除 `*AzureChunkManagerTest*`）+ e2e L0/L1。

**构建与资源注意**（本机内存 1GB free / swap 已满、根盘 13GB 可用）：编译必须限制并行度，先释放磁盘；**绝不 `make clean`**（`cmake_build` 有 7.5GB 热缓存）。命令见 `milvus-build` skill。

---

## 5. 明确不做

1. `field_indexID` 列表化（D1）。
2. C++ `CollectionIndexMeta` 列表化 + `FieldIndexMeta.index_id` 接通（D2）。
3. Vector 同字段多索引（D5）。`querynodev2/services.go:791-793` 的硬报错保持。
4. 显式指定查询索引（表达式语法 / SDK / 计划绑定）。
5. 成本选优。
6. 按"索引是否提供 raw data"释放字段列。
7. **加载侧按能力裁剪**（跨集群兼容兜底）—— D11。
8. `index_info_list`（`query_coord.proto:287`）死字段清理。

## 6. 遗留风险

| # | 项 | 状态 |
|---|---|---|
| 1 | 1.2 断言与 1.3 Reopen 循环 | **1.2 已实证**（反例 T1/T2 通过，日志实测断言文本）；1.3 仍为 **[手工追踪]**，且 S3 容器改造后该路径消失 |
| 2 | JSON NGRAM 多 path 是否已会命中断言（1.2 末尾） | **[手工追踪]** —— 建议顺带在 T5 里加一个两 NGRAM path 的 case |
| 3 | 多索引时 `IndexHasRawData` 取 false 是否够（S3.5） | **[待实现时确认]** |
| 4 | `json_index_path_cache_` 是否还有别的用途（S3.4） | **[待实现时确认]** |
| 5 | `is_indexed_` / `left_is_indexed_` 除 `use_index_data_` 外的用途（S4.1 C4/C5） | **[待实现时确认]** |
| 6 | 免费 `PinIndex` 是否还有别的调用者（S4.5） | 已 grep，3 处；**[实现时再确认一次]** |
| 7 | 跨集群 | 按 §2.2 的 ground truth 与运维契约处理，不再是代码风险 |
| 8 | 外部系统是否依赖 `CollectionLoadInfo.field_indexID` 单值语义 | 用户后续确认，v1 视为无风险 |
| 9 | `SegmentLoadInfo.current_index_ids` 是跨字段扁平 set（`SegmentLoadInfo.cpp:253-261`） | indexID 全局唯一，判断为安全；改 `ComputeDiffIndexes` 时注意 |

## 7. 实现顺序与完成标准

**顺序**（S0/S1/S2 是 Go，可独立提交；S3/S4 是 C++，必须一起）：

```
S0 版本门禁  →  S1 创建校验 + 默认名  →  S2 Proxy 确定化
                                          ↓
S3 容器（先保证现有一索引路径不回归、T1/T4/T5 通过）
                                          ↓
S4 选择层（C1/C2 主干 → C3 → C4/C5/C6 → NGRAM 通道）
                                          ↓
S5 T1–T8 + G1–G5 + 全量回归 + e2e
```

**完成标准**：

1. 同一 varchar 字段 inverted + NGRAM 同时加载；`Equal` 走 inverted、`LIKE` 走 NGRAM、都不支持时走 raw 且结果正确。
2. 同字段 inverted + FMINDEX 时 `PostfixMatch` 落 FMINDEX、`Equal` 落 inverted，且**与创建顺序无关**。
3. `DropIndex` 一个，另一个仍可查；compact / reopen / 重启后两个都在。
4. §3 的 I1–I4 有**显式测试**（T5/T6），不是靠"看代码应该没问题"。
5. 单字段单索引与 Vector 路径零回归（`make test-go` + C++ `all_tests` + e2e L0/L1）。
6. S0 门禁实测：挂一个低版本 session 时第二个索引建不出来。
