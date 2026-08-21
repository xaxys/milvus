# Import Re-shard V3 过度设计与实现审计

- **状态：** 审计意见，未修改任何设计与代码。
- **审计基线：** `bbffa112d2 feat: implement import reshard v3`（分支 `feat/import-reshard-v3`）。
- **审计对象：**
  - `docs/design-docs/design_docs/20260721-import-reshard-design-cn.md`
  - 该 commit 的全部 Go/proto 改动。
- **验证限制：** 当前环境是 Windows，Milvus 主模块因缺 `pkg-config`、woodpecker Windows 编译问题无法完整 `go test`；`pkg` 子模块的 `util/metautil`、`taskcommon`、`common` 空跑编译通过。结论来自源码逐段审计、符号引用统计和路径追踪，不是完整 CI 结论。

---

## 0. 一句话结论

方案主链路（Reshard → Planning → Importing → 直接 IndexBuilding）是成立的，代码整体比设计文档简单，方向没问题。但设计文档和实现都还有明显过度设计：

1. 最大的过度设计是 `import_row_bound.go`：为了“估算每文件要预留多少 AutoID/RowID”，实现了一套 665 行、带 731 行测试的行数下界、恶意文件防御、并发门控机器；设计文档本来只要求“文件字节数 / footer 行数”这种简单上界。
2. 有几处为“未来可能”预留、但当前没有任何调用方的抽象：`BuildTextIndex` 重构、TaskManager progress、fragment reader 的通用 range/format、GC record 状态机。
3. 设计文档里有若干协议复杂度没有实现，且不看代码也能判断不需要：双版本序列、未知版本信封、逐轮中间文件精确清理、广播前稳定 file ID/backup 展开。

下面按“建议删除/简化”和“实现缺口（不要和简化混淆）”两类列全。

---

## 1. 过度设计清单（按优先级）

### P0-1：`import_row_bound.go` 的整套行数上界/防恶意文件机器

**位置**

- `internal/datacoord/import_row_bound.go`：`minRowTextBytes`（L92）、`computeFileRowUpperBound`（L157）、`validateNpyHeader`（L229）、`parquetFooterMaxSize`（L322）、`validateParquetFooter`（L361）、`parquetNumRows`（L387）、`sizingReaderAt`、`assignPKRangesToFiles`（L514）、`reserveRanges`（L537）、`computeFileRowUpperBounds`（L584）、`sizeReservations`（L630）。
- 配套测试 `import_row_bound_test.go` 731 行。

**为什么是过度设计**

设计文档 §6.2 已经写死：

- Parquet footer、NumPy header 可得精确行数，直接用精确值；
- JSON/JSON Lines/CSV 用文件物理字节数作为硬上界；
- 单文件上界超 `MaxUint32` 直接拒绝、要求拆文件。

实现比设计多做了：

1. `minRowTextBytes`：按 schema 字段名、JSON 花括号、分隔符计算“每行最少字节”，用来把大稀疏文本文件的行数上界压下去。
2. `sizeReservations`：对“精确行数”再乘 `ImportPreAllocIDExpansionFactor`，与设计“精确值直接作为上界”不一致；这个因子原本是 V2 正式 log ID 的旧公式参数，被复用到 V3 预分配上。
3. `validateNpyHeader` / `validateParquetFooter`：专门防御声明超大 header/footer 的恶意 numpy/parquet 文件。
4. 全局 `parquetFooterParseSem`（4 并发）、慢等待日志、进程级 footer 解码门控。
5. 新增配置 `dataCoord.import.parquetFooterMaxSize`，设计只允许新增两个配置，且该配置没有写进 `configs/milvus.yaml`。
6. `sizingReaderAt` 自己实现 `ReadAt/Seek/Read` 和重试。

**建议简化**

保留核心需求（广播前给每个普通文件一个连续 `[begin,end)`，CDC 复用字面值），其余按设计文档砍掉：

```text
Parquet：读 footer 拿 NumRows，精确，直接作为上界。
NumPy：按 SourcePaths 读 header shape，精确，直接作为上界。
JSON/JSONLines/CSV：sum(chunkManager.Size(paths)) 作为上界。
```

然后：

- 上界 `<1` 时按 1 处理（避免空 range 被 DataNode 误判）。
- 单文件上界 `> MaxUint32` 直接 InputError 拒绝。
- 多文件跨 allocation batch 的打包可以保留：一次 `AllocAutoIDN` 多个文件，减少 allocator RPC，同时保证每个文件 range 连续；这一小段不是过度设计。
- 删除 `minRowTextBytes`、expansion factor、`parquetFooterMaxSize` 配置、并发门控、`sizingReaderAt`。
- 如果担心 DataCoord 被恶意 footer/header 打挂：把旧的解析风险留给 DataNode reader，或只保留一个**硬编码**的 footer/header 上限检查，不要再加可配置项、全局信号量和自定义 reader。这属于旧链路也存在的输入风险，按“极端 case 旧方案也有则不必为它建防线”原则，首版可以不建。

**正确性审计**

- 文件字节数除以 1 是永远安全的上界，不会少预留；少预留会以 `ErrDataIntegrity` 在 DataNode 暴露，这是必须避免的方向，当前简化不会走到这个方向。
- 精确行数不乘 expansion factor 也安全：footer/header 行数与实际读出不一致时本来就应该失败，而不是靠多预留 ID 掩盖文件损坏。
- 唯一代价：单个 `>4 GiB` 的 JSON/CSV 即使实际只有很少行也会被拒绝、要求拆文件。这是设计文档已经明确接受的边界，不是新引入的语义。
- 唯一风险转移：去掉 footer/header 防恶意解析后，DataCoord 广播路径可能受坏文件解析影响；旧 PreImport 也会完整解析同一文件，只是发生在 DataNode。若团队认为该风险必须现在处理，保留一个简单的硬编码长度检查即可，不要把整套并发门控留下。

预计 `import_row_bound.go` 可从 665 行降到约 150–200 行，测试也相应大幅减少。

---

### P0-2：`BuildTextIndex` 重构没有任何 V3 调用方

**位置**

- `internal/datanode/compactor/compactor_common.go` L58–L268：新增 `TextIndexBuildParams`、导出 `BuildTextIndex`，再把旧的 `createTextIndex` 改成薄包装。

**为什么是过度设计**

注释说“shared by compaction and Import V3”，但全仓没有任何 Import V3 调用 `BuildTextIndex`。设计文档 §14.5 也明确说 TEXT index 继续走 segment 发布后的后台 inspector，不在 Import V3 plan/worker 里建。也就是说这次重构只是把一个旧函数拆成“参数结构体 + 通用函数”，没有第二消费者。

**建议**

直接 revert `internal/datanode/compactor/compactor_common.go` 这一部分。`createTextIndex` 保持原样。

**正确性审计**

当前包装与旧逻辑等价，revert 后行为不变；如果未来真要做内联 text index，再抽公共函数，不要在 Reshard commit 里带一个没人用的重构。

---

### P0-3：`ImportJobGCRecord` 状态机没有按设计使用，且没有测试

**位置**

- `pkg/proto/data_coord.proto` L1424–L1431：`ImportJobGCState` / `ImportJobGCRecord`。
- `internal/datacoord/import_meta.go` L54–L57、L290–L360：4 个 GC 方法。
- `internal/datacoord/import_checker_v3.go` L449–L577：`checkGC/quiesceImportJob/deleteImportJob`。
- `internal/metastore/kv/datacoord/kv_catalog.go`：GC record 的 Save/Get/Drop。

**为什么是过度设计**

设计文档要求“终态保存和 GC record 初始化必须在同一个 catalog update 里完成（EnterTerminalAndInitGC）”。实现里：

- `EnterTerminalAndInitGC` 只在 `checkGC` 发现“已经终态但缺 record”时补建；
- 所有正常进入 Failed/Completed 的路径仍然只调 `UpdateJob`，没有同步建 record；
- 没有任何 `ImportJobGCRecord` / Quiesce / Retain / Delete 的测试；
- `quiesceImportJob` 也没有按设计发 version-aware Drop，只是等 `node_id == NullNodeID` 后删任务。

也就是说，这套 `Quiesce → Retain → Delete` 状态机在运行上是“延迟初始化 + 幂等重放”的壳，正确性收益很小。

**建议简化**

回到 V2 的 GC 形状，只加一个动作：

```text
job 终态且 cleanupTs 已到
→ 等所有 V3 task 的 node_id 为 NullNodeID（global scheduler 已完成 best-effort Drop/解绑）
→ RemoveWithPrefix(import_v3/{job_id}/)   // 幂等
→ 删除 V3 task catalog
→ 删除 job
```

删除 `ImportJobGCRecord`、`EnterTerminalAndInitGC`、`AdvanceImportJobGCState`、`DropImportJobGCRecord` 和对应 catalog 方法。

**正确性审计**

- `RemoveWithPrefix` 是幂等的，所以不需要 `Delete` 状态来记录“对象已删、catalog 未删”；崩溃后重跑同一批删除即可。
- “等 node_id 为空”与 V2 GC 相同；V2 已经接受“scheduler 负责把终态任务 Drop/解绑”这一边界。
- 如果担心节点失联、任务永远 `node_id != NullNodeID`，V2 有同样问题；按“旧方案已有则首版不新造防线”处理。
- 保留 CDC `Failed && !auto_commit` 的 RollbackImport 判断，这段不是过度设计，是 V2 已有语义，必须留下。

---

### P1-4：`ImportV3Catalog` 可选接口 + 到处 type assertion

**位置**

- `internal/metastore/datacoord_catalog.go` L160–L176。
- `internal/datacoord/import_meta.go` L124、L309、L323、L334、L356、L380、L389、L425、L434、L486、L494。

**为什么是过度设计**

这个接口被做成“optional”，理由是旧测试 double 不用实现。结果：

- 生产 catalog 实现它；
- `internal/metastore/mocks/mock_datacoord_catalog.go` 没有重新生成，不实现它；
- `NewImportMeta` 对不实现该接口的 catalog 静默跳过 V3 task 恢复；
- 现有测试全都没有覆盖 V3 catalog 恢复。

这等于给测试留了一个“V3 恢复代码可以不测”的暗门。

**建议**

把 ReshardTask / ImportTaskV3 / GC 相关方法直接并入 `DataCoordCatalog`，重新生成 mock，去掉所有 type assertion。若 GC record 按 P0-3 删除，需要加的方法更少。

**正确性审计**

这只是接口收口，不改变持久化格式和运行时语义；真实 catalog 本来就实现这些方法。

---

### P1-5：TaskManager 的 progress 是死代码

**位置**

- `internal/datanode/importv3/task_manager.go`：`Snapshot.Progress`（L32）、`task.progress`、`UpdateProgress`（L189）。
- 全仓只有 `task_manager_test.go` 调用 `UpdateProgress`；生产执行器从不更新 progress，Query 响应也不带 progress。

**建议**

删除 `Progress` 字段和 `UpdateProgress` 方法，完成时仍置 `StateCompleted`。`NewTaskManager()` 也只被测试用，生产用 `NewTaskManagerWithContext`，可一并收敛成一个构造器。

**正确性审计**

progress 不影响调度、fencing、结果提交，删除无行为变化。

---

### P1-6：未知 Import version 的 Failed job“信封”是为不存在的 V4 做的防御

**位置**

- `internal/datacoord/services.go` L1924–L1938：`ImportV2` 对未知 version 报错。
- `internal/datacoord/services.go` L2004–L2048：ACK 遇到未知 version 时创建一个只有查询/GC 字段的 Failed Job V1。

**为什么是过度设计**

当前协议只有 0/2/3。功能开关保证所有 active DataCoord 都认识 V3；未知 version 只有未来 V4 才会出现。现在为 V4 预建“失败信封 + 不 ACK 分支”没有任何可触发的真实场景。

**建议**

- 入口统一：`0 → 按 L0/V2、非 L0/V3`；`2 → V2`；`3 → V3`；其他直接 `ErrDataIntegrity`，ACK 不 ACK。
- 删除 `createImportJobFromAck` 的 Failed 信封分支。
- 设计文档里“未来 V4 必须显式加分支”一句话保留即可，不需要今天实现 V4 的错误路径。

**正确性审计**

V4 不存在，删除后对 V2/V3 行为零影响。将来加 V4 时本来就要显式加常量和分支。

---

### P1-7：schema projection 里的 `IndexedFieldIDs` 会误杀合法 Import

**位置**

- `internal/datacoord/import_schema_v3.go`：`IndexedFieldIDs`（L24）、收集（L87–L90）、比较（L109、L132–L135）、`validateImportV3Schema`（L145）。

**为什么是过度设计且有害**

projection 已经完整比较所有物理字段、struct 字段、functions 和关键 properties。任何会影响写入/物化的字段变化都会被这层比较抓住。`IndexedFieldIDs` 再额外比较“当前有哪些字段建了索引”，会导致：

- Import 运行中用户创建/删除一个索引，字段和 schema 都没变；
- projection 判为不等，job 直接 Failed。

但索引元数据变化不影响已经写出的 segment 内容，也不影响该 segment 之后由 index inspector 建索引；V3 不应该为此失败。

**建议**

删除 `IndexedFieldIDs` 以及 `meta.indexMeta.GetIndexesForCollection` 查询。`validateImportV3Schema` 不再需要 `meta.indexMeta`。

**正确性审计**

字段 ID/类型/physical 属性仍在 `Fields` 全量比较中；去掉索引 ID 集合不会漏掉任何会改变物理输出或函数输出的 schema 变化，反而修掉“索引 DDL 误杀 Import”的正确性问题。

---

### P1-8：fragment reader 和 intermediate factory 的“未来钩子”

**位置**

- `internal/storage/record_reader.go`：`ImportFragmentReaderSpec{Format, StartRow, EndRow, Rows}`（L28–L34）、`NewImportFragmentRecordReader`（L257）。注释明确写“future external fragment producers”和“checksum hook can be added here later”。
- `internal/datanode/importv3/merge.go`：`IntermediateWriterFactory` 的 `inputs []Source` 参数（L38）只有一个实现，且实现里参数是 `_`，注释写“give caller a precise place to add optional content validation later”。

**为什么是过度设计**

设计文档 §2.2 明确本次不支持外部 fragment 输入，也不做 checksum。当前所有调用都是“完整读一个 Parquet fragment”：

```go
StartRow: 0, EndRow: rows, Rows: rows
```

`Format` 也只有 `ImportFragmentFormatParquet` 一个值。`inputs` 参数没有消费者。

**建议**

- `ImportFragmentReaderSpec` 收敛为 `{Path string; Rows int64}`，内部固定 `StartRow=0`、`EndRow=Rows`、格式 Parquet。
- `IntermediateWriterFactory` 去掉 `inputs []Source` 参数。
- 删除“未来可插 checksum / 外部 producer / 内容校验”注释。

**正确性审计**

这是纯接口收窄；所有现调用点都满足更窄签名，数据读取路径不变。

---

### P2-9：fan-in 配置 [2,1024] 与 benchmark 自己的结论冲突

**位置**

- `configs/milvus.yaml`：`fragmentMergeFanIn: 16`。
- `pkg/util/paramtable/component_param.go`：`FragmentMergeFanIn` 和启动校验。
- `internal/datacoord/import_planner_v3.go` / `internal/datanode/importv3_services.go` / `importv3/merge.go`：读写与再校验。
- 同 commit 的 `20260721-import-reshard-sort-benchmark-cn.md` 明确最终约束是“首期 direct merge fan-in hard cap 为 16”。

**为什么是过度设计**

分层归并本身需要，但首版不需要把 fan-in 做成运行参数并允许到 1024。1024 会按公式算出巨大 slot，长期 Pending；范围校验只是“参数不为非法值”的证明，不产生任何资源保证。设计文档自己也承认这一点。

**建议**

首版固定 `fanIn=16`：

- 删除 `dataCoord.import.fragmentMergeFanIn` 配置和启动校验；
- `ImportTaskPlan.fan_in` 可以保留（plan 自包含），但固定写 16；
- `MergeExecutor.FanIn` 仍保留字段，调用处固定传 16；
- DataNode 保留一次 `fanIn == 16` 校验（或直接信任 plan，只校验 `>1`）。

**正确性审计**

分层算法对任意输入数都收敛，16 只是分组宽度；benchmark 已按 16 验证内存。删除配置不改变数据路径，只减少一个没有收益的调参面。未来有证据再恢复配置即可。

---

### P2-10：V2/V3 checker 大量复制，且指标会互相覆盖

**位置**

- `internal/datacoord/import_checker_v3.go` 的 `checkUncommittedJob`、`checkCommittingJob`、`tryTimeoutJob`、`checkCollection`、`checkGC` 与 `import_checker.go` 几乎逐行相同。
- 状态映射有两套完全相同的函数：
  - `internal/datanode/importv3_services.go` L899–L944；
  - `internal/datacoord/session/cluster.go` 的 `reshardStateFromTaskCommon` / `importV3StateFromTaskCommon`。

**额外 bug**

`metrics.ImportJobs` 被两个独立 goroutine 写：

- V2 checker 只统计 V2 job，V3 checker 只统计 V3 job；
- 两者都把“自己那部分每个 state 的数量”写进同一组 gauge；
- 最后写入者决定最终值。例如只有 V2 Completed job 时，V3 loop 可能随后把所有 state 写成 0，指标错误。

**建议**

- 把 `Uncommitted/Committing/timeout/collection-drop` 这些与 V2 完全相同的尾部逻辑抽成共享函数，或只保留一个 checker loop、按 `job.version` 走 V2/V3 分支。
- 状态映射只留一对，两个方向复用。
- `ImportJobs` gauge 从全量 job 列表一次计算，不要在 V2/V3 两个 checker 里分别写。

**正确性审计**

这是控制流合并，不改 job 状态机语义；但需要逐函数对拍 V2/V3 现有行为，属于小型重构，不建议和 P0 一起无测试地改。

---

### P2-11：设计文档自身的“协议级”过度设计

以下点写进了 2388 行设计文档，但实现没有做，而且按当前需求可以不做：

1. **广播前稳定 file ID + 稳定 file key 排序 + backup 展开前移**（§6.1、§6.4）。
   代码仍然在 ACK 里分配 file ID、在 ACK 里展开 backup。由于普通文件的可复用 ID range 已经在 WAL 里固定，file ID 只是 job 本地引用，不影响用户数据；backup 保留源 PK/RowID/timestamp，也不需要 CDC 两端 file ID 一致。建议**改设计为“file ID 是本地引用，不进 CDC 一致性契约”**，而不是实现这套稳定排序/前移协议。

2. **逐轮精确删除 intermediate run**（§15.3）。
   代码没实现，所有 intermediate 文件活到 job GC。默认 128 MiB fragment + ~1 GiB segment 时一个 plan 很少超过 16 个 fragment，分层归并本身都很少发生。建议**改设计为“intermediate 由 job 前缀 GC 兜底”**，删除“相邻两轮存活集”算法。极端情况只是临时对象多占一点，首版可接受。

3. **`ImportMsg.version` 与 `ImportJobVersion` 必须永远独立、未来 V4 分支、未知版本信封**（§5）。
   建议收敛为“一套版本常量，两个存储位置（WAL/job）各放一个字段”，只支持 2/3。

4. **`EnterTerminalAndInitGC` 与 GC record 状态机**（§19）。
   见 P0-3。

5. **结果 50% gRPC 阈值 warning**（§17.1）。
   代码有实现，但只是 warning，不改变任何行为；属于可观测性锦上添花，不阻塞。可留可删，建议不作为首版必须项。

6. **滚动升级/回滚 runbook 的完整检查**（§20）。
   作为发布文档有价值，但很多检查（回滚时扫描 broadcaster catalog 的 V3 消息）当前实现没有配套代码；如果团队实际发布流程不依赖这些检查，建议降级为运维 runbook 或删除代码级承诺。

---

## 2. 实现与设计不一致 / 正确性缺口（必须与“简化”分开处理）

这些不是过度设计，但审计中发现实现没有达到设计承诺，或者有实际 bug。建议先让用户拍板“改实现还是改设计”，不要顺手混在删代码里。

### BUG-1：backup 的 BFD 大小估算会拿 prefix 去 `Size`

`groupV3ReshardSources`（`import_planner_v3.go` L257）对所有文件执行：

```go
size, err := storage.GetFilesSize(ctx, file.GetPaths(), cm)
```

backup 文件展开后的 `paths` 是 insert/delta 目录 prefix，不是具体对象；在对象存储上对 prefix 调 `Size` 通常返回 key not found，ReshardTask 创建会失败。`buildV3SourceFileSpec` 里倒是先 `ExpandObjects` 拿到了真实对象列表，但估算发生在它之前。

**建议**

backup 源文件先 `ExpandObjects`，再按展开后的真实对象路径求和估算；或者首版对 backup 直接按文件数给常量估算（例如每个 segment 1 MiB 或源文件序号），因为估算只影响分组，不影响正确性。真实对象求和更准，代码也只需要挪一下调用顺序。

### GAP-1：广播前 file ID / backup 展开没有实现

见 P2-11.1。当前行为：

- `broadcastImport`（`ddl_callbacks_import.go` L179）只冻结 version 和 `pre_allocated_auto_ids`；
- `createImportJobFromAck`（`services.go` L2066–L2087）仍然 `ListBinlogImportRequestFiles` + 本地 `AllocN` 分配 file ID。

对普通文件，RowID/AutoID 的确定性没被破坏，因为 range 已进 WAL；对 backup，源 PK/RowID 保留，也不破坏用户数据。所以推荐**改设计**，不实现复杂的广播前排序/展开协议。

### GAP-2：Reshard sort 输入没有按 task memory budget 切小

`writeReshardFragment` 会把 bucket 的全部 spill chunks + 内存尾批一次交给 `storage.Sort`，触发条件是 `logicalBytes >= fragmentTarget`。默认配置下 `fragmentTarget=128 MiB`，而 `slot` 预算约 480 MiB，安全；但如果运维把 `memoryLimitPerSlot` 调小、或单批超过预算，sort 的物化峰值可能超过 `slot * memoryLimitPerSlot`。设计文档 §10.2 有 `effective_fragment_input`，代码没实现。

**建议**

- 最小修法：创建 plan/发送 task 前校验 `fragmentTarget` 与 task memory budget 的关系（例如 sort 输入上限 `<= budget/2`），不满足就按 job 失败或切小 fragment；
- 或者明确把该边界写为配置前置条件，文档说明“fragmentSizeInMB 必须小于等于 slot 内存预算的一半”。

不要为此恢复设计里整套 slice/replay/borrowed record 协议；一条简单预算检查足够覆盖默认和常规配置。

### GAP-3：V3 GC 没有 version-aware Drop，也没有测试

见 P0-3。如果保留 GC record 状态机，也必须补测试并把 Drop 语义实现清楚；如果按建议简化成 V2 式 GC，则此缺口自然消失。

### GAP-4：V3 catalog 恢复路径未被 mock 覆盖

见 P1-4。`mocks.DataCoordCatalog` 没有实现 `ImportV3Catalog`，`NewImportMeta` 静默跳过恢复，测试环境永远练不到 V3 恢复。合并接口后必须补一个“重启恢复 V3 job/task”的测试。

### GAP-5：go.mod 把 go-api 指向个人 fork

`go.mod` / `pkg/go.mod` 新增：

```text
github.com/milvus-io/milvus-proto/go-api/v3 => github.com/xaxys/milvus-proto/go-api/v3 ...
```

这是为了拿到 `msg.ImportFile.pre_allocated_auto_ids` / `ImportMsg.version`。合并上游前必须换成官方 tag/commit，不能带个人 fork 进主分支。

### GAP-6：ReshardManifest 校验不完整

`validateReshardManifest` 允许 `rows==0`、不校验 `seq<0`。零行 fragment 在 Planning 生成的 `FragmentRef` 会被 `SourceFromFragment` 拒绝，把错误延迟到 DataNode。建议在 manifest 校验处直接拒绝零行/负 seq，失败更早、更简单。

### GAP-7：`sameImportV3FragmentCoverage` 只比较扁平 ref

该函数把 segment plans 的 `FragmentRef` 摊平比较，忽略 segment 边界和 partition 分组。当前同 vchannel 内 partition 顺序冻结，fragment 规范序也冻结，所以大多数情况下结果一致；但若 segment target 配置在重启间变化，已有 plan 会被静默保留旧边界。设计文档说“只要求 vchannel 和 fragment coverage 一致”，因此可以接受，但建议在代码注释里显式写出这个接受，或改成同时比较 `(partition_id, segment boundary)`。

---

## 3. 建议保留、不要再砍的部分

以下看起来复杂，但要么是方案核心，要么是确定性/正确性所需：

1. `ReshardTask + ImportTaskV3` 两段式：消除 task 边界碎片的根本手段。
2. fragment 必排序 + 严格单头 k-way merge：跳过 sort compaction 的前提。
3. `run_id` fencing：没有它重试会接受旧结果。
4. 自包含 `ImportTaskPlan` / `WriterSpec`：保证跨重试、换节点输出语义一致；可讨论字段是否过多，但方向不能删。
5. 本地 Arrow IPC spill：高 `V×P` 下有界内存的必要手段；当前实现已经比设计文档简单。
6. `WrapDecodeErr` / `ReadAt` 错误分类：把对象存储瞬时故障从 InputError 里救回来，属于正确性。
7. TTL 只冻结时长、final batch 用当时时钟：比保存 reference time 简单且符合最终一致。
8. 零行 segment placeholder 与 `IsImporting=true` 可见性规则：与 V2 对齐，不能删。

---

## 4. 建议的第一批简化（若同意）

按“先删无调用/防御，再收口，最后改设计措辞”的顺序：

1. revert `compactor_common.go` 的 `BuildTextIndex` 重构。
2. 删 TaskManager progress + `NewTaskManager`。
3. 收窄 `ImportFragmentReaderSpec`、去掉 `IntermediateWriterFactory.inputs`。
4. 删 GC record 状态机，V3 GC 改成 V2 形状 + `RemoveWithPrefix`。
5. 把 `ImportV3Catalog` 并入 `DataCoordCatalog`，重新生成 mock，补 V3 恢复测试。
6. `import_row_bound.go` 按设计文档重写为简单上界；删除 `minRowTextBytes`、expansion factor、footer/npy 防御门控和 `parquetFooterMaxSize` 配置。
7. 删未知 version Failed 信封，统一版本常量。
8. 去掉 `IndexedFieldIDs`。
9. fan-in 固定 16，删配置。
10. 修正 backup BFD 估算（先展开再估算）。
11. 修 V2/V3 指标覆盖问题，顺手把尾部 checker 逻辑合并。
12. 按 P2-11 修改设计文档，把“file ID 是本地引用、intermediate 由 job GC 兜底、GC 无独立 record”写回正文。

第 1–8 项不改变任何数据正确性；第 9 项是运行参数收口；第 10–11 是必修 bug；第 12 是文档与代码对齐。

---

## 5. 需要确认的两个 tradeoff（非阻塞）

1. **接受 `>4 GiB` 的 JSON/CSV 直接拒绝。** 这是设计文档原文已经接受的边界，但实现里 `minRowTextBytes` 明显是为了绕开它。若团队不能接受这个限制，就不要按 P0-1 简化，而应保留 row floor、只删防恶意文件机器。
2. **fan-in 首版固定 16。** benchmark 报告自己下了这个结论；但如果后续想给大内存节点调参，就保留配置、只把上限从 1024 收到 32 或 64。
