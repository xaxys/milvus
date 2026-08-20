# Milvus 单字段多索引详细实施计划

本文给出单字段多索引的完整代码与架构改造计划。所有现状和改造边界以当前源码为准。

本次实现范围固定为：

- Sealed Scalar segment 可以为同一字段同时加载多个索引，并由 Segcore 对每个过滤条件自动选择一个可执行索引。
- Growing GEOMETRY 继续每字段维护一份内部 RTree；多个 indexID 只在 sealed segment 同时驻留。
- Scalar 查询计划内部保留可选 indexID，但本次不增加公开的查询指定语法。
- Vector 字段可以创建多个索引，但一次 Load 对每个 Vector 字段只加载一个索引。
- 已加载 collection 通过新的 Load 配置从 Vector 索引 A 切换到 B，不需要先 Release collection。
- B 的索引文件尚未完成时，sealed segment 强制使用原始向量；B 完成后再 Reopen 到 B。Growing segment 保持现有临时索引行为。
- Release 清理完成后，QueryNode 使用有过期时间的轻量版本记录拒绝晚到的旧请求；该记录不进入正常 distribution。
- Vector 多索引同时驻留、Search 指定 Vector indexID 和 Vector 动态选索引不进入本次实现。

## 1. 改造后的整体链路

当前代码已经在 DataCoord 中用 indexID 区分索引定义、构建任务和每个 segment 的索引文件；真正的单索引限制主要出现在后面的 Load 与查询链路。Proxy 和 QueryCoord 使用 `fieldID → indexID` 单值 map，QueryCoord 的部分检查又会重新读取 collection 的全部索引，QueryNode / Segcore 的普通 Scalar 和 Vector 运行状态也大多按 fieldID 保存一个索引。

这使系统同时存在两个问题：上游不能稳定表达“同一字段选择 A 和 B”，下游也不能准确区分“目标是 A、实际是 B”还是“字段上已经有某个索引就算完成”。因此本次不能只放开 DataCoord 的创建限制，必须把索引定义、当前 Load 目标和 segment 实际状态拆开，并让三层都使用明确 indexID。

完成后，索引定义、Load 目标和 segment 实际状态分成三份数据。

```text
DataCoord
  保存所有 indexID 的定义、构建任务和文件
          ↓
QueryCoord
  保存 collection 默认索引和本次 Load 的目标 indexID
          ↓
QueryNode / Segcore
  保存每个 segment 实际已经加载的 indexID 和字段数据
```

索引定义仍由 DataCoord 负责保存。Create Index 和 Drop Index 的回调在完成 DataCoord 元数据修改后，同时通知 QueryCoord 更新默认索引和当前 Load 对 indexID 的引用。Load 和 Reopen 只读取 QueryCoord 已保存的当前目标；QueryNode 上报实际状态后，QueryCoord 再把尚未达到目标的 segment 继续向前推进。

Scalar 和 Vector 共用控制面状态与版本控制，进入 Segcore 后采用不同的数据结构：

```text
Scalar field
  fieldID → indexID → Scalar 索引

Vector field
  fieldID → 一个当前 Vector 索引
```

## 2. 协议和数据形状

### 2.1 公开 Load 请求

公开的 LoadCollectionRequest 和 LoadPartitionsRequest 用于告诉 Proxy 要加载哪个 database、collection 和 partition。它们已有 db name、collection name、replica number、resource groups、refresh、load fields、是否跳过 dynamic field 和 load params；LoadPartitions 还带 partition names。公开消息目前完全没有字段名与索引名的对应关系，所以用户不能在 Load 时表达“这个字段选择 A、B”。

Proxy 解析名称和 schema 后，会构造发往 QueryCoord 的内部 LoadCollectionRequest 或 LoadPartitionsRequest。内部消息已有 collectionID、schema、replica number、resource groups、load fields 和 priority；LoadPartitions 还带 partitionIDs。与索引有关的字段只有 `field_indexID`，它是 `fieldID → indexID` 的单值 map。

这个 map 适合“一字段一个索引”，但同一 Scalar 字段的 A、B 无法同时写入。Proxy 当前收集多个索引时也只能反复给同一个 fieldID 赋值，最后留下哪个值取决于遍历结果。公开请求使用字段名和索引名更适合用户，内部请求继续使用 ID 更适合持久化和跨组件传递，因此在两层分别增加列表结构，而不是把默认选择逻辑放进 Proxy。

在公开的 LoadCollection 和 LoadPartitions 请求中增加结构化配置：

```text
FieldIndexLoadConfig
  field_name
  index_names[]
```

请求中只携带用户明确写出的字段。Proxy 只负责名称到 ID 的转换，不读取和补齐 collection 默认值。

Proxy 把名称解析成 ID 后，向 QueryCoord 传递内部结构：

```text
FieldIndexIDConfig
  field_id
  index_ids[]
```

Scalar 字段的 `index_ids` 可以有多个值。Vector 字段最多有一个值。字段和 indexID 在进入 QueryCoord 前去重并排序，后续持久化和消息比较都使用相同顺序。滚动升级期间保留现有单值字段；新功能开关关闭时仍走单值字段，打开后只读新的结构化字段。

### 2.2 AlterLoadConfig

AlterLoadConfig 是当前 Load / Alter Load 的完整配置消息，不只是“增加一个索引”的增量消息。它的 header 已经保存 dbID、collectionID、partitionIDs、loadFields 和 replicas；`userSpecifiedReplicaMode` 说明 replica 是否来自用户请求，`useLocalReplicaConfig` 让 Secondary 在 CDC 场景中改用本集群的 replica / resource group 配置。body 目前为空，真正的配置都在 header 中。

其中 LoadFieldConfig 只有 fieldID 和 indexID。虽然 `load_fields` 本身是 repeated，但当前消息生成只会为一个 field 输出一个值，回调又把它写回单值 map；所以仅靠 repeated 声明并没有形成多索引能力。现有无变化判断还会比较整个 header，因此必须让消息生成、回调聚合和比较三处都看到完整 indexID 集合，否则只改索引仍可能被错误地当成无操作。

现有 AlterLoadConfig 已经使用重复的 LoadFieldConfig。继续保持单条记录只保存一个 fieldID 和一个 indexID，同一个 Scalar field 通过多条记录表达：

```text
field 100, index A
field 100, index B
field 101, index C
```

生成消息时按 fieldID、indexID 排序。当前用于生成消息的单值 map 改为完整集合；回调处理时先按 fieldID 聚合，不能再写入单值 map 后覆盖前一个 indexID。

现有“Load 配置没有变化”的判断继续比较完整的 AlterLoadConfig 消息头，但消息头中的 LoadFieldConfig 必须来自完整 indexID 集合。`[A] → [B]`、`[A] → [A,B]` 和 `[A,B] → [B]` 都会得到不同的消息头，不能因为 partition、replica 和 load field 没变而被当成无操作。

### 2.3 Collection 默认索引消息

当前持久状态只有“这一次已经 Load 了什么”。CollectionLoadInfo 会在 Release 后删除，AlterLoadConfig / DropLoadConfig 也只描述当前 Load 的建立和释放；系统没有一份独立状态表达“下次 Load 未指定索引时使用什么”。如果把默认值继续塞进 CollectionLoadInfo，Release 会把默认值一起删掉；如果每次临时从 DataCoord 索引列表取第一个，重启或 map 顺序变化又可能得到不同结果。

因此默认索引需要独立持久化，同时仍要沿现有 collection DDL 顺序和 CDC 链路复制。新增完整快照消息的原因是：回调重试和 Secondary 重放时不依赖旧状态，收到同一条消息总能得到同一份 defaultIndexes。

增加一个 CChannel 消息 `AlterIndexLoadDefault`，消息中保存 collectionID 和完整的 `fieldID → indexID[]` 默认集合。

默认集合中的字段条目允许 indexID 列表为空。字段条目存在但列表为空表示“默认已经明确设置为空”；字段条目不存在才表示“尚未初始化”。Create 第一个索引只初始化后一种状态，因此删除默认索引后不会因为以后又创建了索引而自动得到新的默认值。

该消息使用与 AlterLoadConfig 相同的 collection 资源锁，走现有广播、ACK 回调和 CDC 复制链路。消息不带 Unreplicable 标记。Secondary 保留 fieldID 和 indexID，只按现有规则处理本地 replica 与 resource group。

公开协议增加读取和修改 collection 默认索引的请求，并复用 FieldIndexLoadConfig。Proxy 把 field name 和 index name 解析成 ID。QueryCoord 的内部更新入口只接收一份已经解析完成的全量 defaultIndexes；请求层先在 collection 锁内把输入转换成这份完整状态，再广播 AlterIndexLoadDefault。消息回调用消息内容整体替换 defaultIndexes 并保存 defaultVersion，读取接口把持久化 ID 重新转换为稳定排序的字段名和索引名。

消息类型、构建器和类型转换由现有 Streaming 消息生成流程生成，不手写重复代码。

### 2.4 Scalar 查询计划

ColumnInfo 是过滤表达式中“这一列是谁”的公共描述。它当前包含 fieldID、数据类型、数组元素类型、JSON nested path、nullable 和 element-level 等信息，并被 Unary、Term、Range、Exists、ColumnExpr 等多种表达式复用。它天然跟随单个过滤条件，而不是跟随整个 Search 请求。

当前执行层只根据这些字段找到该 field 上的唯一 Scalar 索引。以后若要让同一字段的两个条件分别绑定 A、B，indexID 必须跟 ColumnInfo 一起进入计划；放在请求级别会无法说明它约束哪个条件。此次只增加可选字段，不开放公开语法，但相等判断和表达式合并必须同步认识它，避免计划改写时把两个不同索引的条件合并。

在计划的 ColumnInfo 中增加可选的 Scalar indexID，并同步到 C++ 的 ColumnInfo。

本次 Proxy 不从公开请求写入该值，因此自动选择时它保持未设置。计划解析和表达式改写逻辑仍要把它加入条件相等判断和合并键，避免以后两个指定不同 indexID 的条件被错误合并。

### 2.5 Load 版本

LoadSegmentsRequest 用于把 sealed segment 的字段文件、索引文件、schema、Load 信息、replica 和加载范围发给 QueryNode；其中已有的 Version 表示这次 segment 分布更新。WatchDmChannelsRequest 用于建立 growing channel，已有 partition、channel、schema、Load 信息、replica、collection 索引定义、channel version 和 target version。SyncDistributionRequest 用 action、schema、Load 信息和 replica 更新 shard leader 看到的分布。ReleaseSegmentsRequest 和 UnsubDmChannelRequest 则分别指出要释放的 segment 或 channel。

这些消息目前没有一条共同版本表示“整份 collection Load 配置属于哪一次修改”。LoadSegmentsRequest.Version 只保护 segment 分布，WatchDmChannelsRequest.Version 只保护 channel，SyncDistribution 中 action 的 Version 也只保护对应分布；Release 和 Unsub 甚至没有同类字段。UpdateIndexRequest 当前只有 collectionID 和一次 Add/Drop action，也没有配置版本。

因此 A → B → C 连续修改时，即使 QueryCoord 已经取消旧任务，先前发出的 RPC 仍可能晚到 QueryNode，Segcore 也只知道请求到达顺序，不知道哪个目标更新。新的 loadVersion 必须与现有分布 Version 分开：前者保护 collection Load 配置，后者继续保护 segment / channel 分布，避免把两个不同含义的版本混在一起。

增加独立的 `loadVersion`。它取自本集群 CChannel 上真正改变 currentIndexes 或 currentLoaded 的 AlterLoadConfig、DropLoadConfig、DropIndex 等消息的 TimeTick，只在本集群内比较。只改变 defaultIndexes 的消息更新 defaultVersion，不无意义地推动已加载 collection 的 loadVersion。

`loadVersion` 贯穿：

```text
QueryCoord 持久状态
  → SegmentTask / ChannelTask / LeaderTask
  → LoadSegments / WatchDmChannels / SyncDistribution / ReleaseSegments / UnsubDmChannel / UpdateIndex
  → QueryNode collection 版本记录
  → LocalSegment
  → QueryNode distribution 上报
```

现有 LoadSegmentsRequest.Version 继续表示一次 segment 分布更新的版本，不改作 Load 配置版本。

## 3. QueryCoord 持久状态

### 3.1 新状态

当前 CollectionLoadInfo 主要保存 collectionID、replica number、LoadStatus、LoadType、loadFields、dbID 和用户是否明确指定 replica；其中 `field_indexID` 仍是单值 map。它代表“当前正在运行的 Load”，PartitionLoadInfo 则保存各 partition 的 LoadStatus 和恢复次数。ReleaseCollection 会删除这些运行记录，所以它们不适合保存长期默认值。

同时，CollectionLoadInfo 的单值 map 无法表达 Scalar `[A,B]`，也没有地方保存“当前已经 Release，但必须拒绝旧 Load”的版本和“Release 尚未通知完所有 QueryNode”的中间状态。新增状态不是复制整个 CollectionLoadInfo，而是只补齐索引选择、查询规则和版本这些现有结构表达不了的内容。

新增独立的 CollectionIndexLoadState：

```text
CollectionIndexLoadState
  collectionID
  defaultIndexes: fieldID → indexID[]
  currentIndexes: fieldID → indexID[]
  vectorQueryRules: fieldID → 当前 Vector 查询规则摘要
  currentLoaded
  releaseInProgress
  defaultVersion
  loadVersion
```

`defaultIndexes` 用于 collection 处于未加载状态时的新 Load。内存结构和持久结构都必须保留“字段存在但列表为空”，不能在序列化时把它和“字段不存在”合并。`currentIndexes` 表示这一次 Load 的完整目标；`vectorQueryRules` 保存从当前 Vector 定义得到的 metric 和查询参数规则，避免旧物理索引定义已经 Drop 后失去过渡期所需的查询含义。Release 开始时设置 `currentLoaded=false` 和 `releaseInProgress=true`，但暂时保留 currentIndexes 和 vectorQueryRules；节点确认新版本后再清空它们，并把 `releaseInProgress` 设为 false。

这份状态使用独立的 catalog key 和独立的内存管理器。现有 CollectionLoadInfo 继续表示 collection、partition 和 replica 的运行状态，不保存 collection 默认值。这样 Release 仍可以删除现有运行记录，而不会删除默认索引。

### 3.2 Catalog 和内存管理器

当前 QueryCoord catalog 保存 CollectionLoadInfo 后，再单独批量保存 PartitionLoadInfo；Release 也先删除 collection key，再按 key 前缀删除 partition key。Replica 又由独立 manager 保存。这个顺序在现有单索引路径中依靠 ACK callback 幂等重试收敛，但新增一份索引目标状态后，如果两份 collection 级状态分开写，崩溃会产生“运行记录还是 A、索引目标已经是 B”或“已经标记未加载、replica 却被恢复逻辑当成孤儿删除”的中间状态。

因此只把两个真正需要同时可见的 collection key 放进一次 KV 提交，并把它作为当前配置的最终发布点。Partition 和 replica 不强行塞进一个可能超出事务大小的提交，而是先幂等准备、在最终发布前对普通 observer 隐藏，失败后继续复用现有 ACK 重试。Release 则必须保留运行记录到节点确认完成，不能为了追求一次删除而丢掉恢复所需的 replica。

QueryCoordCatalog 增加以下能力：

- 保存一份 CollectionIndexLoadState。
- 批量恢复所有 CollectionIndexLoadState。
- Drop Collection 时删除该 collection 的状态。
- 用一次 KV 提交同时保存 CollectionIndexLoadState 和 CollectionLoadInfo。
- 用一次 KV 提交开始 Release，保存 `currentLoaded=false`、`releaseInProgress=true` 和仍然存在的 CollectionLoadInfo。
- 用一次 KV 提交结束 Release，保存未加载状态并删除 CollectionLoadInfo。

AlterLoadConfig 不能先写新状态、再单独写 CollectionLoadInfo。新的组合保存接口把这两个 collection 级 key 放进同一个 MultiSave；这里的原子保证只覆盖这两个 key，不宣称 partition 和 replica 也在同一个 KV 事务内。Release 分成两次明确提交：第一次只进入“释放中”，保留 CollectionLoadInfo、partition 和 replica 供恢复与节点通知使用；第二次在节点确认后用 MultiSaveAndRemove 写入最终未加载状态并删除 CollectionLoadInfo。partition key 随后按现有批量方式清理。

QueryCoord 增加 collection 级“配置应用中”标记。AlterLoadConfig 回调先准备 partition 和 replica，再用两个 collection key 作为最终发布点；最终发布前，普通 observer、checker 和任务生成都跳过这个 collection。进程重启时，从待重放的 AlterLoadConfig ACK callback 建立同样的标记；callback 到达 collection 级最终发布点后即可清除标记并启动普通收敛，不要求等待整个 Load 完成。释放中的 collection 走专用 Release 恢复，不被这个屏障阻塞。

启动恢复看到 `releaseInProgress=true` 时，不把 collection 恢复为可查询状态，而是恢复它的 CollectionLoadInfo 和 replica，仅用于继续发送 `currentLoaded=false` 和完成释放。看到最终未加载状态时，不恢复残留 partition 记录，并继续清理它们。

内存管理器也用一把 collection 级锁同时替换 CollectionIndexLoadState 和 CollectionLoadInfo。任务创建和 RPC 组装必须从同一快照中取得 current loadFields、`currentIndexes` 与 `loadVersion`，不能分多次读取后得到不同版本的数据。

### 3.3 启动恢复

当前启动恢复先从 catalog 重建 collection、partition 和 replica，再启动 target、collection observer 和 checker。它只认识“有 CollectionLoadInfo 就是已加载、没有就是未加载”，也会清理找不到 collection 的 replica。

加入“配置应用中”和“释放中”后，原顺序会把尚未完成 Release 的 replica 提前清掉，或者在待重放的 AlterLoadConfig callback 完成前让 checker 读到半套状态。因此恢复必须先读索引 Load 状态给 collection 分类，再恢复依赖记录；同时等待 QueryNode session 和第一份 distribution 到齐，避免把暂时空的 NodeManager 当成节点已经退出。

QueryCoord 启动时按下面顺序恢复：

```text
恢复 CollectionIndexLoadState
    ↓
恢复 CollectionLoadInfo、partition 和 replica，并按新状态分类
    ↓
为缺少新状态的旧 collection 执行一次初始化
    ↓
确认引用的 indexID 仍然有效
    ↓
等待 QueryNode session 恢复和第一次 distribution 同步
    ↓
启动 TargetObserver、CollectionObserver 和各类 Checker
```

旧 collection 的 current 和 default 分开初始化：

- 每个集群都用本地已加载 collection 的现有 FieldIndexID 初始化 current，并从对应 Vector 定义生成 vectorQueryRules，因为实际 QueryNode、replica 和加载进度不跨集群复制。
- 功能开始启用时，Primary 为已有 collection 生成完整 default：已加载 collection 使用现有 FieldIndexID，未加载 collection 读取 DataCoord 定义，并只在字段有唯一有效索引时写入该 indexID。
- Primary 把生成结果作为 AlterIndexLoadDefault 广播。Secondary 只应用复制来的完整 default，不独立从自己的索引列表选择默认值。
- 协议中没有 loadVersion 的旧请求会被解析为 0，因此 current 初始化记录使用保留的正数版本，0 只表示升级前请求。正常 CChannel TimeTick 大于这个初始化版本，后续 DDL 可以直接覆盖。

初始化不会删除或改写旧 CollectionLoadInfo 字段。功能开关关闭时，旧字段仍是运行路径的输入。

恢复出的 collection 分成三类：正常已加载、释放中、最终未加载。释放中的 CollectionLoadInfo 和 replica 只提供给 CollectionIndexObserver 和 Release job，用来找到仍需通知的 QueryNode；它们不进入普通 Load 恢复、资源分配、平衡或 IndexChecker。第一次 QueryNode session 恢复和 distribution 同步完成前，不能因为暂时看不到节点就判定节点已经退出，也不能最终删除这些 replica。

## 4. 索引定义、Create 和 Drop

### 4.1 DataCoord 索引元数据

当前 DataCoord 的底层元数据已经以 indexID 为主键保存 Index、SegmentIndex、buildID 和索引文件，同一个 collection 的查询接口也能返回多个 IndexInfo。真正阻止普通同字段多索引的是 CreateIndex 前的校验：普通字段和同一 JSON path 最多一个索引，JSON 不同 path 是现有例外。

这说明定义和文件层不需要再引入“索引组”或新的文件格式。继续保留现有 indexID 生命周期，移除同字段限制并修正默认 index name 的单索引假设，是比重做构建系统更小也更直接的改法；后面的复杂度主要在 Load 和查询，不在 DataCoord 文件布局。

DataCoord 保持现有的 indexID、SegmentIndex、buildID 和索引文件结构。主要改动是移除“同字段只能有一个索引”的检查，同时保留：

- collection 内 index name 唯一。
- 相同名字、相同字段、相同参数的幂等处理。
- 每个新索引分配独立 indexID。
- Drop 一个 indexID 不影响同字段其他 indexID。

JSON path 索引也按 indexID 独立保存，同一个 path 可以存在多个不同名字的索引。

索引定义和每个 segment 的索引文件继续使用现有 indexID、SegmentIndex 和 buildID 结构，不增加“索引组”，也不改变索引文件的保存格式。

### 4.2 DataCoord 与 QueryCoord 的索引变化连接

当前 CreateIndex / DropIndex 的 Streaming ACK callback 只在 DataCoord 中注册：Create 写入索引定义并通知构建检查，Drop 把对应 indexID 标记删除。QueryCoord 没有同一条可靠回调来更新 defaultIndexes 或 currentIndexes，只能在其他路径重新 ListIndexes。

多索引后，Drop A 必须精确删除 A 的引用并提高 loadVersion；依赖某个后台 checker 事后发现会缺少产生这次变化的 CChannel TimeTick，也无法处理未加载 collection 的默认值。因此索引定义写成功后应在同一个 ACK callback 中调用 QueryCoord 的通知函数，失败仍由原广播重试，保证 DataCoord 权威定义与 QueryCoord 引用按同一条 DDL 收敛。

CreateIndex 和 DropIndex 的 Streaming 回调继续由 DataCoord 注册。MixCoord 启动时给 DataCoord 安装一个内部索引变更通知函数，DataCoord 完成权威元数据更新后调用它更新 QueryCoord。

```text
CreateIndex / DropIndex CChannel 回调
    ↓
DataCoord 写入或删除权威索引定义
    ↓
调用 QueryCoord 索引变更通知函数
    ↓
QueryCoord 更新 default/current 引用和 loadVersion
```

通知函数使用 indexID 和消息的 CChannel TimeTick，必须支持重复调用。通知失败时，原 ACK 回调返回错误，沿现有广播重试流程再次执行。

Create 回调完成后，通知函数读取该字段的完整有效 indexID 集合。只有 default 尚未初始化，并且集合中唯一成员就是本次新建的 indexID 时，才用该消息的 TimeTick 写入 default 和 defaultVersion。字段已经有多个索引、default 已明确为空或已经有值时，都不自动覆盖。

Drop 时，通知函数从 default 中精确移除被删除的 indexID；集合为空时保存空集合，不能从剩余索引中补一个值。Scalar indexID 如果还在 current 中，也从 current 中移除，并写入新的 loadVersion，随后唤醒 IndexChecker 清理 QueryNode 上对应的实际索引。Vector 当前 indexID 由 Drop 请求前的检查保证不会走到这条分支。

AlterIndex 继续沿 CDC 复制，但当前公开入口会拒绝在已加载 collection 上修改索引属性。因此本次不把 AlterIndex 接入 currentIndexes 和在线 Reopen 通知链；未加载 collection 的 AlterIndex 仍只更新 DataCoord 定义，后续 Load 再读取更新后的参数。

### 4.3 删除已加载 collection 上的 Vector 索引

当前 DropIndex 只要发现 collection 已加载，就拒绝删除其中任何 Vector 索引。这个检查来自“一字段只有一个 Vector 索引”的前提：删除字段上的 Vector 索引就等于删除当前查询所需的唯一索引。

当同字段可以定义 A、B 后，这个检查过宽。删除未加载的 A 不应被 B 的运行状态阻挡，但删除当前目标仍会让 CCollection 缺少生成 Search plan 所需的 metric 和查询参数。因此检查必须从“字段是不是 Vector”收紧为“这次删除的 indexID 是否仍是当前 Load 目标”，并把实际内存清理交给已经有版本保护的 Reopen / Release 链路。

DataCoord 把现有“loaded collection 上不能删除任何 Vector 索引”的检查改成精确检查：解析本次要删除的 indexID，并向 QueryCoord 读取该 collection 的当前 Load 目标；只有要删除的 indexID 仍是已加载 collection 的当前 Vector 目标时才拒绝广播。

删除旧索引 A 的正常代码链路是：先用 AlterLoadConfig 把当前目标改为 B，或先写入 DropLoadConfig 完成 Release；随后再广播 DropIndex A。这几条消息使用同一个 collection 资源锁，因此 Primary 上的顺序明确，CDC 也按这个顺序把目标修改或 Release 放在 Drop 之前送到 Secondary。

Secondary 收到复制来的 DropIndex 后直接执行 DataCoord 回调和 QueryCoord 索引变更通知函数，不重新进入公开 DropIndex 请求，也不再执行本地 loaded 检查。此时 Secondary 的当前目标已经由前一条复制消息改为 B，或已经处于未加载状态；Drop 回调删除 A 的定义和默认引用，QueryCoord 再通过现有 Reopen / Release 收敛链路清理 QueryNode 上仍然存在的 A。

### 4.4 Drop Collection

当前 Drop Collection 会合成 DropLoadConfig 和“空 indexID 的 DropIndex”回调，分别清理 QueryCoord 运行状态和 DataCoord 的全部索引，最后再删除 collection 元数据。新增状态如果不接入这条合成链路，普通 DropIndex 测试可能正确，但 Drop Collection 会留下 defaultIndexes 或 release 状态。

因此不新增另一套 collection 删除流程，只让现有两个合成回调分别清理 current 与全部默认引用，并在 collection 最终删除时移除新的 catalog key。

Drop Collection 继续复用现有的合成 DropLoadConfig 和 DropIndex 回调：

```text
DropLoadConfig 回调
  写入“释放中”和更新版本；节点确认后再清理 current 状态

DropIndex 空 indexID 回调
  清理该 collection 的全部 default/current 索引引用

RootCoord 最后删除 collection 元数据
```

CollectionIndexLoadState 的 catalog 记录在 Drop Collection 完成时一并删除。

## 5. Load 请求进入 QueryCoord

### 5.1 Proxy 解析

当前 Proxy 在 LoadCollection 和 LoadPartitions 中读取 collection 索引信息，并构造 QueryCoord 请求里的 `field_indexID`。因为目标类型是 map，同字段的多个 IndexInfo 会互相覆盖；Proxy 还因此被迫参与“未指定时选哪个”的决定。

Proxy 最适合处理名字、schema 和权限相关的请求校验，不适合保存 collection 当前 Load 与默认值。把它收窄为“只解析用户明确写出的 field name / index name”，可以让重试、CDC 和重启都由 QueryCoord 的持久状态得到同一结果，也避免 LoadCollection 与 LoadPartitions 各自实现一套默认逻辑。

Proxy 为 LoadCollection 和 LoadPartitions 建立相同的索引解析函数：

1. 读取 collection schema 和全部索引定义。
2. 建立 index name 到 indexID、fieldID 的稳定映射。
3. 把公开请求中的 field name 和 index name 解析成 ID。
4. 只传递用户明确指定的字段。
5. 不再遍历全部索引并写入 `map[fieldID]indexID`。

LoadPartitions 不重新猜测 collection 的完整索引集合，只传递本次请求明确写出的部分。完整目标仍由 QueryCoord 生成并保存为 collection 级状态；PartitionLoadInfo 不新增独立的索引选择，因此同一个 loaded collection 的不同 partition 不会长期使用不同的索引目标。

### 5.2 QueryCoord 生成完整目标

当前 QueryCoord 收到的是一份单值 FieldIndexID 和 loadFields。初次 Load、LoadPartitions、修改 replica / resource group 等路径会分别组装 ExpectedLoadConfig；它们没有一个公共步骤按“显式请求、当前 Load、collection 默认”补齐索引，也无法区分“字段以前已经加载”与“这次新加入 load_fields”。

如果简单规定“已加载 collection 的未指定字段都沿用 current”，新加入的字段会因为 current 中没有值而失去索引；如果每条路径重新 ListIndexes，又会回到不稳定地取第一个。完整目标函数必须以最终 loadFields 为范围，一次生成 collection 级 indexID 集合，后续 replica、partition 和 target 变化只携带这份结果。

QueryCoord 使用一个公共函数生成完整目标：

```text
请求中明确写出的字段
    使用请求的 indexID 集合

collection 当前已加载、该字段已经在 current loadFields 中，且请求未写该字段
    沿用 currentIndexes

collection 当前未加载，或该字段是本次新加入的 load field，且请求未写该字段
    使用 defaultIndexes
```

这个函数同时接收当前 loadFields 和请求完成后的 loadFields，先分清“已经加载的字段”和“本次新增字段”，再补索引。完成合并后先删除不在最终 loadFields 中的字段，再用 schema 和 DataCoord 的有效索引定义检查 fieldID、indexID 和字段类型。Scalar 字段保留完整集合；每个需要加载的 Vector 字段必须得到且只得到一个 indexID。未加载 collection 或本次新增的字段如果既没有显式值也没有 default，Load 在生成 AlterLoadConfig 前返回错误，不按索引列表顺序补值。

LoadPartitions 在 collection 已加载时也使用这套合并：请求没有写索引时沿用 currentIndexes；请求写了索引时，得到的新完整集合替换 collection 级 currentIndexes，并同时用于已有 partition 和新增 partition。

完整目标用于生成 AlterLoadConfig。生成消息和“配置未变化”比较都使用完整、排序后的 indexID 集合。Replica、resource group、partition 或 load field 的后续修改也读取并携带同一份 currentIndexes，不能再次从 DataCoord 的全部索引中生成选择。

### 5.3 AlterLoadConfig 回调

当前 AlterLoadConfig ACK callback 会先准备 replica，再把 repeated LoadFieldConfig 写成 `map[fieldID]indexID`，创建 CollectionLoadInfo / PartitionLoadInfo，更新 next target 并启动 CollectionObserver。相同 fieldID 的第二条记录会覆盖第一条；保存 collection、partition 和 replica 的过程也不是一个整体事务。

多索引后，回调不仅要聚合重复 fieldID，还必须避免“索引目标已经发布，partition / replica 还没准备好”。因此沿用现有幂等准备顺序，但把 CollectionIndexLoadState 与 CollectionLoadInfo 的组合提交作为最终发布点；在这以前用“配置应用中”隔离 checker，之后再启动 target 收敛。

ACK 回调按以下顺序落地：

```text
从重复 LoadFieldConfig 聚合 fieldID → indexID[]
    ↓
标记 collection 配置应用中
    ↓
按现有幂等规则准备并持久化本次需要的 replica 和 partition；暂不发布给 observer/checker
    ↓
使用一次 KV 提交保存 currentIndexes、vectorQueryRules、currentLoaded=true、releaseInProgress=false、loadVersion 和 CollectionLoadInfo
    ↓
在同一把 collection 锁内发布内存中的 collection、partition、replica 和索引状态
    ↓
清理不再属于新配置的旧 partition / replica，并清除“配置应用中”标记
    ↓
更新 next target
    ↓
启动 collection load 收敛
```

状态保存成功后，即使 QueryCoord 重启，也会继续向同一个 indexID 集合收敛，不会从索引列表顺序重新选择。

### 5.4 Release

当前 DropLoadConfig callback 最终会删除 CollectionLoadInfo、partition、target 和 replica。对于普通单索引流程，这表示 collection 已经不再加载；但如果旧 Load RPC 仍在路上，QueryNode 没有更高的 collection 版本阻止它重新创建 CCollection 或 segment。更重要的是，若一开始就删 CollectionLoadInfo，QueryCoord 重启会把剩余 replica 当成无主数据清掉，随后无法知道还要通知哪些 QueryNode。

因此 Release 必须变成“先写不可再加载的新版本，再排空运行状态，最后删除记录”。`releaseInProgress` 不是新的用户状态，而是让 ACK 重试和 QueryCoord 重启能继续同一条 Release 的内部恢复点。

DropLoadConfig 回调先用一次 KV 提交保存更高的 loadVersion、`currentLoaded=false` 和 `releaseInProgress=true`，同时保留 CollectionLoadInfo、partition 和 replica。CollectionIndexObserver 随后向当前 replica 的在线 QueryNode 下发这个版本；只有节点上报已经接受该版本，或在 QueryCoord 完成 session 恢复后确认该节点会话已经退出、不会再服务查询，Release job 才继续删除 segment、channel、partition、target 和 replica 运行状态。通知失败时保持“释放中”并重试，不能直接越过这一步。

运行状态清理完成后，再用一次 KV 提交把 `releaseInProgress` 改为 false、清空 currentIndexes，并删除 CollectionLoadInfo。QueryCoord 在这次最终提交后才把 collection 当成普通未加载状态。

释放部分 partition 时保留 currentIndexes，只更新 partition 集合。释放最后一批 partition 时使用与 ReleaseCollection 相同的未加载状态。

## 6. QueryCoord 只处理当前 Load 的索引

### 6.1 公共过滤结果

当前 QueryCoord 已经把 FieldIndexID 保存进 CollectionLoadInfo，但任务执行时仍会分别调用 ListIndexes 和 GetIndexInfo：WatchDmChannels 会带 collection 的全部 IndexInfo，LoadSegments / Reopen 也会根据 segment 返回的全部已完成索引组装请求。不同入口没有共同的“只保留本次 Load 目标”规则。

这会让持久化的选择失去实际约束：用户选择 A，QueryNode 仍可能收到 B；Scalar 同字段多个索引时，不同入口还可能得到不同集合。新增一个公共视图函数的目的不是再加一层抽象，而是让所有下发点共享同一条交集规则，避免以后修好 LoadSegments 却漏掉 Watch 或 Reopen。

增加一个 QueryCoord 内部的索引视图构建函数，输入为：

- collection schema。
- 当前 CollectionIndexLoadState 快照。
- DataCoord 返回的全部 collection 索引定义。
- DataCoord 返回的某个 segment 的全部索引文件。

输出分为两部分：

```text
collectionIndexInfos
  当前 Vector 字段的唯一索引定义
  + 现有 growing GEOMETRY 路径需要的单值 RTree 定义

segmentIndexInfos
  当前目标中、并且该 segment 文件已经完成的 Scalar/Vector indexID
```

普通 Scalar 定义不再塞入按 fieldID 单值保存的 CollectionIndexMeta。Scalar 查询选择直接读取 segment 实际加载的索引描述。CollectionIndexMeta 继续表示“每个字段最多一份 collection 运行规则”：Vector 使用当前唯一目标；现有 growing GEOMETRY 路径从当前 Scalar 目标中按 indexID 稳定选择一份 RTree 定义；同时接通其中已有但当前没有真正使用的 indexID 字段。

GEOMETRY 的这一个 indexID 只作为 growing 内部 RTree 的规则来源和诊断身份。现有 growing segment 仍按 fieldID 只有一份内存 RTree，也不会因为 CCollection 的 IndexMeta 更新而重建；新 growing segment 使用当时最新的单值规则。按 indexID 同时驻留多个 Scalar 索引的能力只用于 sealed segment。

普通 Scalar 定义仍用于给 segmentIndexInfos 补齐加载参数和建立 Scalar 索引描述，只是不再作为 fieldID 单值定义传给 CCollection。其他 growing Scalar 继续使用原始字段或现有专用路径。

所有下发入口必须调用同一个视图构建函数：

- 初次 LoadSegments。
- Reopen。
- WatchDmChannels。
- SyncDistribution。
- QueryNode UpdateIndex。

### 6.2 CollectionIndexObserver

当前 QueryCoord 没有向 QueryNode 发送 UpdateIndex 的任务或 observer，QueryNode 的 UpdateIndex 也只是占位实现。现有 distribution manager 只汇总 segment 和 channel，因此即使新增 RPC 返回成功，QueryCoord 也无法确认某个只有 growing segment、或暂时没有 Reopen 的节点已经安装了新的 CCollection 查询定义。

Vector 查询计划在 CCollection 层读取 metric 和参数规则，必须先更新 collection 视图，再允许同版本的 segment / channel 发布。把这项工作放进独立的 collection observer，可以复用现有定时重试和任务去重，同时用 collection 级 distribution 做真实确认，而不是把 RPC 成功误当成状态已经生效。

增加一个 collection 级 observer，负责把 QueryCoord 的完整索引目标可靠地下发到每个 QueryNode。它复用现有 observer 的定时检查和任务去重方式，不把 UpdateIndex 混入 segment Reopen 任务。

observer 为每个 replica 中的 QueryNode 比较：

```text
期望状态
  collectionID + loadVersion + currentLoaded + 当前 collection 运行规则

实际状态
  QueryNode distribution 上报的同一组 collection 级信息
```

不一致时，observer 通过 QueryCoord 的 cluster 调用入口发送 UpdateIndex，并持续重试，直到 QueryNode 上报相同版本、Vector 定义和保留的 GEOMETRY 单值定义。即使一个节点只有 growing segment、暂时没有 sealed segment，或当前没有 Reopen 任务，也会收到 collection 查询定义。

SegmentTask、ChannelTask 和 LeaderTask 在发往某个节点前，先确认该节点已经接受同一个 loadVersion；未确认的任务继续等待，不先发布 segment 状态。Release 使用相反顺序：先让在线节点确认 `currentLoaded=false` 的新版本；发送失败的节点保持待重试，只有节点确认或退出当前服务会话后，才执行该节点上的 segment 和 channel 释放。

observer 在启动后先等待 QueryNode session 恢复和第一次 distribution 同步完成。这个屏障以前，空的节点列表只表示“状态还没收齐”，不能当成节点已经退出。释放中的 replica 仍参与版本通知，但不参与普通加载、平衡和资源调度。

### 6.3 IndexChecker

当前 IndexChecker 对每个已加载 collection 调用 ListIndexes，把 DataCoord 返回的全部有效索引都当成目标。它虽然在 segment distribution 中按 indexID 查看实际状态，但缺失列表和后续任务仍以 fieldID / 完整索引列表为中心；因此它既无法表达“只要 A，不要 B”，也无法在 Scalar `[A,B] → [B]` 时只删除 A。

Checker 适合比较“期望与实际”并生成物理 Reopen，不适合修改 default/current 持久状态：它拿不到导致索引变化的 CChannel TimeTick，且不会遍历未加载 collection。将期望集合改为 currentIndexes、比较单位改为 indexID，同时把持久状态修改留给 DDL callback，可以保持职责清楚并避免版本倒退。

IndexChecker 的比较单位从 fieldID 改为 indexID：

```text
desired = currentIndexes 中该字段的 indexID 集合
actual = QueryNode 上报的 segment indexID 和加载摘要
available = DataCoord 已完成文件的 indexID 和加载摘要
```

Scalar 检查按每个 indexID 独立处理。新增 indexID 或相同 indexID 的摘要变化产生 Reopen，已经不在目标中的 indexID 产生清理 Reopen。同字段另一个 indexID 的存在不能代表目标 indexID 已经完成。

Vector 检查保持每字段单值：

- 实际是目标 B：不创建任务。
- 实际是 A，B 文件已完成：Reopen 到 B。
- 实际是 A，B 文件未完成：Reopen 到原始向量状态。
- 实际上报 raw_scan_field_ids 包含该字段，且 B 文件未完成：等待文件完成。
- 实际上报 raw_scan_field_ids 包含该字段，且 B 文件已完成：Reopen 到 B。
- 实际没有物理 indexID，但未上报严格原始扫描状态：继续生成修复任务，不能当成目标已完成。

IndexChecker 只比较目标、实际状态和已经完成的索引文件，并据此创建 Reopen 任务；它不修改 defaultIndexes 或 currentIndexes。索引引用只由 Create/Drop DDL 回调、Load/Release 回调和启动恢复修改，避免后台检查绕过 DDL 顺序写出新的控制状态。

### 6.4 任务版本

当前 Grow/Reduce 类任务主要用 target / distribution version 判断动作是否完成，Update/Reopen 类 segment action 在 RPC 返回后即可结束。两类任务都不知道创建它时 collection 选择的是 A 还是 B。任务取消只能阻止尚未执行的部分，已经发出的 RPC 仍可能完成。

因此 loadVersion 要在任务创建时固定，并在执行前、发送前和完成判断时重复检查；QueryNode 的实际上报也必须带同一版本。这样版本保护覆盖的是最终发布结果，而不是只保护 QueryCoord 队列。

SegmentTask、ChannelTask 和 LeaderTask 在创建时记录 loadVersion。执行前和 RPC 发送前都比较当前版本；不一致的任务直接结束。

任务完成条件同时检查：

- segment 或 channel 的普通分布版本。
- QueryNode 上报的 loadVersion。
- 目标 indexID 集合。
- 每个 indexID 的加载摘要；同一 indexID 指向新文件时不能误判为完成。

这样连续发生 A → B → C 时，旧的 A/B 任务不能把 C 覆盖回去。

## 7. QueryNode 状态和下发接口

### 7.1 Collection Load 版本记录

当前 QueryNode 的 collection 生命周期主要由 CCollection、channel delegator 和 LocalSegment 共同体现，没有一份独立状态在 CCollection 不存在时仍保存“这个 collection 至少要拒绝版本 0 的旧 Load”。Load、Watch、Sync 和 Release 也没有共享的 collection 级进入检查。

仅把 loadVersion 加进 RPC 不够：Release 会删除 CCollection，而版本恰恰需要在删除后继续存在，才能拒绝晚到请求。因此增加一份很小、独立于 CCollection 的状态，集中管理最低版本、当前是否允许加载、待应用的 collection 运行规则和正在执行的状态修改请求。

QueryNode 增加一份独立于 CCollection 生命周期的小型状态：

```text
collectionID
  latestLoadVersion
  updatingToVersion
  currentLoaded
  currentCollectionIndexDefinitions
  正在执行的加载或释放请求
```

LoadSegments、WatchDmChannels、SyncDistribution、ReleaseSegments 和 UnsubDmChannel 进入时先登记自己的 loadVersion，并一直保留到 collection、channel 或 segment 状态发布完成。旧版本请求不能登记，也不能创建、替换或删除新版本的运行状态。

版本相同时仍按请求类型处理：`currentLoaded=false` 时只允许释放类请求继续排空，拒绝 Load、Watch 和新增分布；`currentLoaded=true` 时允许同版本的正常加载、平衡和释放任务。

更高版本的 UpdateIndex 到达时，先记录待生效的新版本，阻止新的旧版本请求进入；再取消仍在执行的旧版本请求并等待它们退出。旧请求如果已经进入最后发布窗口，就等待这次发布结束，然后再提交新版本。UpdateIndex 返回后，不会再有旧版本结果晚到发布。

Release 时，QueryCoord 先向当前 collection replica 中的全部在线 QueryNode 发送更高版本的 `currentLoaded=false`，再开始释放 segment 和 channel。即使旧 Load RPC 晚到，QueryNode 也不会重新创建已释放状态。

`currentLoaded=false` 的活动记录先保留到本地 CCollection、segment、channel 都已清理，并且没有正在执行的旧状态修改请求。随后采用有过期时间的轻量版本拦截记录：只保存 collectionID、最低可接受的 loadVersion 和过期时间，不再出现在正常 collection distribution 中。旧版本请求继续被拒绝；更高版本的新 Load 可以把它重新变成活动记录。

过期时间从现有 QueryCoord segment task timeout 和相关 QueryNode RPC deadline 中取最大值，再留一个固定安全余量；不增加由用户填写的第二套时间配置。只有确认本地没有旧请求仍在执行时才开始计算。到期后，已经发出的旧 RPC 必然失效，可以安全删除。Drop Collection 使用同一清理路径。QueryNode 重启会终止旧连接和请求，因此不恢复这份记录，只由 QueryCoord 按持久状态重新建立仍需要的活动版本记录。

### 7.2 UpdateIndex

当前 UpdateIndexRequest 只有 collectionID 和一次 AddIndex / DropIndex action；AddIndex 携带一份 IndexInfo，DropIndex 只有 indexID。QueryNode 实现仍直接返回成功，没有修改 CCollection；QueryCoord cluster 侧也没有可靠下发链路。另一方面，首次 Load 时 CCollection 还不存在，只有 WatchDmChannels 或 LoadSegments 带着 schema、load meta 和 collection index_info_list 到达后才能创建。

因此把 UpdateIndex 改成完整快照，并允许“先保存、后应用”：没有 CCollection 时先保存版本和 collection 运行规则，首个 Watch / Load 创建对象后再应用；已有 CCollection 时先替换 IndexMeta，再上报版本。完整快照比一连串 add/drop 更适合重试和重启，因为任何一次成功请求都能恢复完整目标。

复用现有但尚未实现的 UpdateIndex RPC，把它改成完整快照：

```text
UpdateIndexRequest
  collectionID
  loadVersion
  currentLoaded
  current collection index definitions[]
    每个 Vector 字段的唯一当前定义
    每个 GEOMETRY 字段保留的一份 growing RTree 定义
```

QueryNode 只接受不旧于本地版本的请求；同一 loadVersion 如果携带不同内容也要拒绝。`currentLoaded=true` 时，先把完整 collection 运行规则保存在独立的版本状态中；CCollection 已存在时同时替换它的 IndexMeta，不存在时由后续 LoadSegments 或 WatchDmChannels 建立 CCollection 时读取这份定义，并在发布首个 segment 或 channel 前完成应用。`currentLoaded=false` 时只提高本地允许执行的最低版本，已有查询使用的对象继续由正常释放流程排空。

LoadSegments 和 WatchDmChannels 仍携带相同的 collection 运行规则，因此新 QueryNode、重启后的 QueryNode 和没有提前收到 UpdateIndex 的节点都能建立完整状态。

GetDataDistributionResponse 当前用于周期性上报 segment、channel、leader view、节点资源和修改时间；full 模式给出完整 segment/channel 状态，delta 模式给出发生变化或已经删除的 segment/channel，leader view 仍按当前 delegator 生成。它没有 collection 级条目，所以只有 growing 数据或尚未加载 sealed segment 的节点，无法单独证明自己已经接受新的 CCollection 规则。

QueryNode distribution 增加 collection 级上报：collectionID、latestLoadVersion、currentLoaded，以及当前每个字段的运行规则 indexID。这里的“已上报”表示版本检查和待应用的运行规则已经保存，可以安全接收该版本的 Load / Release 任务；它不要求 CCollection 已经存在。

现有 distribution 的 full / delta 路径只跟踪 segment 和 channel，这正是只加 UpdateIndex RPC 仍无法闭环的地方。需要补上一类 collection 变化：

- full 上报包含该 QueryNode 进程内全部 collection 版本状态。
- delta tracker 增加“需要上报的 collection”和“已经删除的 collection”集合。
- UpdateIndex、首次保存待应用的 collection 运行规则和状态删除都记录 collection 变化，并更新 distribution modify time。
- Release 清理完成并把活动记录转成轻量版本拦截记录时，产生“collection 已删除”的 delta；轻量记录本身不进入 full 或 delta 上报。
- QueryCoord 增加 collection distribution manager；full 上报按 node 整体替换，delta 上报做增删，QueryNode 会话退出时清空该 node 的 collection 状态。

CollectionIndexObserver 以这份 manager 中的状态作为 UpdateIndex 已真正生效的确认，不能只把 RPC 返回成功当成完成。

### 7.3 Segment 实际状态

当前 ReopenSegments 在进入 LocalSegment.Reopen 前使用 loadingSegments 去重；同一 segment 已经 Reopen 时，后到请求会被跳过并返回成功。资源申请又在整批 segment 进入 LocalSegment 前完成，当前 LoadInfo 在运行后还会被压缩，无法让两个并发请求都基于最新状态计算差量。

这会直接破坏 A → B → C：C 请求可能被 B 的“正在加载”状态吞掉，或者 B、C 同时按 A 计算资源。串行范围必须覆盖版本复查、差量计算、资源申请、C++ 发布和 Go 状态提交，而不只是包住最后的 C++ Reopen；同时仍按 segment 分锁，避免一个大 collection 的所有 Reopen 被全局串行。

现有 ReopenSegments 会把“同一个 segment 已经在 Reopen”当成无需处理并直接返回，这会吞掉后到的新目标。Reopen 路径需要改成每个 segment 独立串行：同一个 segment 等待前一个完成，不同 segment 仍可并行；初次 Load 的去重逻辑保持不变，Reopen 不再走“正在处理就返回成功”的分支。

LocalSegment 继续以 indexID 保存 Go 侧索引信息，并增加 loadVersion。每次 Reopen 取得该 segment 的串行锁后，按固定顺序完成：再次比较版本、计算差量、申请差量资源、调用 C++ Reopen、一次提交 Go 侧状态、释放本次预留。后到请求必须在取得锁后重新读取最新 LoadedResourceState，不能在锁外基于旧状态提前估算。

Reopen 从进入 C++ 到 Go 侧状态提交期间一直登记为“正在执行的加载请求”。更高版本的 UpdateIndex 会先阻止新的旧请求，再取消并等待这次 Reopen 完整退出，因此不能在旧 C++ 状态刚发布、Go 状态尚未提交时插入新版本。ReleaseSegments 和 UnsubDmChannel 也经过同一个 collection 版本检查，旧 Release 不能删除新 Load 已经发布的 segment 或 channel。Reopen 成功后才在同一个串行区间内同时更新：

- Go 侧 indexID 集合。
- 每个 indexID 的加载摘要。
- 严格原始扫描的 Vector fieldID 集合。
- SegmentLoadInfo。
- loadVersion。
- 字段数据可读状态。

同一 segment 的 A → B → C 请求即使已经到达 QueryNode，也只能按锁内版本检查后的顺序提交；B 不能在 C 之后覆盖回来。

SegmentVersionInfo 当前已经上报 segmentID、collection、partition、channel、普通分布 version、已加载索引信息、JSON stats 和 manifest；ChannelVersionInfo 只上报 channel、collection 和普通分布 version。它们能说明物理分布，但不能说明这些内容属于 A、B 还是 C，也不能区分同一 indexID 的旧文件和新文件。

QueryNode distribution 上报增加：

- segment 的 loadVersion。
- channel 的 loadVersion。
- 全部已加载 indexID。
- 每个 indexID 的加载摘要。
- 严格原始扫描的 Vector fieldID 集合。
- 已经具备字段数据读取能力的 fieldID 集合。

QueryCoord 同时检查严格原始扫描字段和字段数据可读状态，区分“没有物理 Vector 索引且确定会原始扫描”“仍有 sealed 临时索引”和“字段数据尚未准备好”。

## 8. Scalar 多索引运行时

### 8.1 Segcore 容器

当前 sealed segment 的普通 Scalar 索引访问仍以 fieldID 为主，JSON 索引再辅以 path / cast，NGRAM 也有自己的专用状态。SegmentLoadInfo 虽然已经能看到一个 field 的多个 indexID，但最终发布到运行时后，表达式通常只能取得该字段上的一个普通索引；同字段新索引会走替换语义，而不是与旧索引并存。

多索引的所有权、删除和缓存都必须以 indexID 为准，否则 `[A,B] → [B]` 仍只能按 field 删除两者。保留 JSON path、cast 和 NGRAM 参数作为候选能力，而把最外层身份统一成 `fieldID → indexID`，可以复用现有各类索引对象，不需要把它们重写成同一种实现。

Sealed segment 的用户 Scalar 索引统一改成：

```text
fieldID
  └─ indexID
       ├─ index type
       ├─ JSON path / cast
       ├─ NGRAM 参数
       ├─ 轻量能力信息
       └─ 独立缓存项
```

普通 Scalar、JSON path/cast 和 NGRAM 都使用 indexID 作为精确身份。同一个 JSON path 下的多个 indexID 可以同时存在，删除其中一个不会删除同 path 的其他索引。

TextMatch、PhraseMatch、Primary Key 和 JSON Stats 继续使用现有专用容器，不进入普通 Scalar indexID 集合。

现有字段级“已有索引”标记只表示该字段至少有一个普通 Scalar 索引。某个具体索引是否存在和可用，始终从 indexID map 读取。

### 8.2 LoadDiff 和 Reopen

SegmentLoadInfo 是一个 sealed segment 的完整加载说明。它已有 segment、collection、partition、行数、字段文件或 manifest、统计文件和重复的 index_infos；每个 FieldIndexInfo 又用 fieldID、indexID、buildID、index version、文件列表、参数和大小描述一个已构建索引。因此现有消息已经能传递“同一字段有 A、B 两个索引”，这一层不需要重做。

真正的缺口在差量计算和运行容器。当前 ComputeDiffIndexes 虽然会收集多个 indexID，但仍用“这个 field 是否已经有索引”决定 load 还是 replace，所以 `[A] → [A,B]` 会把 B 当成替换；普通索引删除仍以 fieldID 为主，JSON 删除以 path 为主，也不能精确只删除某个 indexID。比较核心又只有 indexID，同一个 indexID 的 build、version 或文件已经改变时不会进入 replace。运行后的轻量信息还会压缩完整文件内容，Go 和 C++ 都缺少下一次稳定比较所需的身份。

因此差量结构本身必须改到 indexID 粒度，并保留稳定加载摘要。摘要不是新的业务版本，而是把现有 buildID、indexVersion、存储路径版本和文件身份压成可比较值，用于区分“没有变化”和 replace；Go 和 C++ 共用同一摘要，避免两边对同一 Reopen 得出不同结论。

SegmentLoadInfo 已能携带同字段多个 FieldIndexInfo。需要把 ComputeDiffIndexes 和 LoadDiff 改成 indexID 粒度：

```text
新增 indexID
  加入 indexes_to_add

相同 indexID 的文件或版本改变
  加入 indexes_to_replace

旧状态存在、目标不存在的 indexID
  加入 indexes_to_drop
```

仅保存 indexID 不足以识别“同一个 indexID 的文件已经重建”。QueryNode 在完整 FieldIndexInfo 到达时生成一次加载摘要，输入为现有 fieldID、indexID、buildID、indexVersion、indexStorePathVersion，以及排序后的索引文件路径或 manifest 身份。LocalSegment 和 Segcore 的 SegmentLoadInfo 保存同一个摘要，避免 Go 与 C++ 各自实现不同的比较规则。ComputeDiffIndexes 用 `fieldID + indexID + 加载摘要` 比较新旧状态：不存在是 add，同一 indexID 的摘要变化是 replace，目标中消失是 drop。

`[A] → [A,B]` 只加载 B，`[A,B] → [B]` 只删除 A。普通 Scalar、JSON 和 NGRAM 使用同一规则。

现有 Reopen 已经会把新索引、字段数据和新的 LoadInfo 放进未发布状态，全部成功后再一次发布。这里继续复用这套结构，只修改差量粒度和各容器写入方式；准备或发布失败时，旧状态保持不变。已经取得旧索引对象的查询继续执行，发布后再回收被删除索引的缓存项。

### 8.3 Scalar 原始字段数据

当前 Segcore 已经有按 fieldID 保存的字段数据列和“字段数据可读”状态，并不是缺少独立的字段容器。问题在于加载、删除和资源估算仍允许在 Scalar 索引声明 HasRawData、且配置没有要求优先字段数据时，省略对应字段数据；retrieve、ColumnExpr 和字段比较等路径也可能从当前那一个索引反查原值。这套做法建立在“一字段只有一个 Scalar 索引”上。

同字段多索引后，A 有 raw data 不代表 B 也有，也不代表 A 能执行当前过滤条件。删除 A、改选 B、原始扫描、retrieve、group-by 或 NGRAM 二次检查都可能需要字段值，而且 Storage V2/V3 的一个 column group 还可能同时承载多个字段。因此改造重点不是新建另一套字段存储，而是把已有字段数据列的生命周期与 indexID 生命周期彻底分开。

已加载的 Scalar 字段始终建立可读的字段数据列；它仍可使用现有 mmap、分层缓存和延迟打开能力，不等于必须全部常驻堆内存。Scalar 索引只增加或删除自己的资源，不能替代、删除或决定字段数据列的生命周期。

这会让部分“索引 raw data 完全代替字段数据”的旧场景增加一份基础资源，但它换来稳定的原始扫描、retrieve 和字段计算能力，也避免删除一个 indexID 时意外破坏同字段其他查询。

逻辑状态仍按 fieldID 表示“这个字段可以读取”，资源和文件生命周期则按实际加载单元管理。旧存储格式中的一组 FieldBinlog、以及新存储格式中 manifest 描述的一个 column group，都可能覆盖一个或多个字段；同一个物理单元只加载和计算一次。

这项改动必须同时接通三处，不能只在 QueryNode 保留文件列表：

1. QueryNode 拆分 SegmentLoadInfo 时，只要一个物理加载单元中包含需要读取的 Scalar 字段，就保留对应 FieldBinlog 或 column group；不再因为其中某个 Scalar 索引带 raw data 就删除这组字段数据。
2. Segcore 初次 Load 和 Reopen 的字段加载判断中，普通 Scalar 索引的 HasRawData 不再成为跳过字段 column 的条件；功能开关关闭时保留旧路径。
3. 初次 Load 和 Reopen 使用的资源估算都按物理加载单元计算 Scalar 字段数据，再分别累加各 indexID 的索引资源；多个字段共享一个 column group 时不能重复计算，估算阶段也不能因为任意一个 Scalar 索引有 HasRawData 就扣掉这组数据。

Segcore 初次 Load 和 Reopen 都建立字段 column。以下路径直接读取字段数据，不再按 fieldID 从任意索引反查：

- 原始过滤扫描。
- retrieve。
- group-by。
- ColumnExpr 和字段间比较。
- NGRAM 二次检查。

删除一个或全部 Scalar indexID 不改变字段数据可读状态，也不释放该字段的 column。

### 8.4 Scalar 索引选择

当前 Scalar 表达式先定位 fieldID / JSON path，再直接取得该位置上的唯一索引，并调用现有的操作支持、值范围、JSON 精度或 NGRAM 检查；没有“列出同字段候选，再选一个”的公共入口。不同表达式各自包含一部分索引判断。

第一步不需要建立完整成本优化器，但必须把正确性筛选集中起来，否则 Unary、Term、Range 等入口会各自选出不同规则。选择器先排除不能执行该条件的索引，再稳定选择一个；估算结果允许未知，未知时仍能得到确定结果并在不可执行时回到原始扫描。

Segcore 增加两个简单入口：

- 读取某个 field 已加载索引的轻量描述，不打开索引内容。
- 按 fieldID 和 indexID 取得一个索引对象，并保证它在本次条件执行结束前不会被回收。

过滤条件执行时走下面的链路：

```text
Text / PK / JSON Stats 专用路径先检查
    ↓
读取该字段全部 Scalar 索引描述
    ↓
按 path、cast、op、值类型和索引能力筛选
    ↓
按 indexID 稳定排序
    ↓
只取得最终选中的一个索引对象
    ↓
执行现有 ShouldUseOp、JSON 精度和 NGRAM 检查
    ↓
索引不可执行时读取原始字段扫描
```

候选筛选和候选排序写成两个独立的小函数。排序函数接收一个可选估算结果：有估算值时按值排序，没有估算值时按 indexID 保持稳定顺序。本次改造不新增统计生产链，普通候选可以返回“未知”；以后增加行数分布、索引冷热或历史耗时后，只补估算函数，不改表达式执行和正确性判断。

计划中的可选 Scalar indexID 如果被设置，选择器只检查该 indexID；本次公开请求不会设置该值。

Unary、Term、BinaryRange、Null、Exists、JSON、GIS 和 NGRAM 等普通过滤入口统一调用选择器。多个条件仍由现有表达式树完成 AND、OR 和 NOT 合并。

### 8.5 Cache key

当前普通 sealed index 的缓存项身份以 segmentID 和 fieldID 为主。同字段 A、B 同时存在时会命中同一个身份，导致 Warmup、淘汰、资源归属或取消无法区分具体索引。因此这里只需把 indexID 加进身份，不需要为多索引新建一套缓存系统。

Scalar 索引缓存 key 从 `segmentID + fieldID` 改成：

```text
segmentID + fieldID + indexID
```

所有创建普通 Scalar 索引缓存项的路径使用相同身份。Warmup、取消、淘汰、资源归属和诊断都能精确指向一个 indexID。

## 9. Vector 单索引 Load 和在线替换

当前 Vector 运行时本来就是单索引模型：QueryCoord 的旧状态按 fieldID 保存一个 indexID，CCollection 按 fieldID 保存一份查询规则，sealed segment 也按 fieldID 保存一个 Vector 索引。真正的缺口是这份“当前选择”没有约束所有下发入口，QueryCoord 仍可能把 collection 的全部定义和 segment 的全部已完成文件交给 QueryNode。

因此本次不把 Vector 容器改成多索引。主要改造放在 QueryCoord 目标过滤、collection 查询规则更新、Reopen 和版本控制上，始终保证一个 Vector field 在已发布状态中最多只有一个物理索引。

### 9.1 Collection 查询定义

CollectionIndexMeta 是 QueryNode 建立或更新 CCollection 时交给 Segcore 的 collection 级运行规则，不是 segment 文件清单。它的重要内容包括：

- maxIndexRowCount，用于 growing 或 sealed 临时索引的建立判断。
- 重复的 FieldIndexMeta；每条记录包含 fieldID、index name、type params、index params、auto index 标记、user index params，以及协议中已经存在的 index_id。

实际某个 sealed segment 加载哪些数据和索引文件，由 SegmentLoadInfo 及其中的 FieldIndexInfo 表示；FieldIndexInfo 已有 indexID、buildID、版本、参数和文件路径。前一份消息回答“这个字段按什么规则生成 Search plan 或内部临时索引”，后一份消息回答“这个 segment 当前有哪些物理内容”。

虽然 CollectionIndexMeta.index_metas 是重复字段，当前 C++ 仍把它收成 `fieldID → 一份 FieldIndexMeta`。同时，QueryNode 当前组装消息时没有填写已有的 index_id，C++ FieldIndexMeta 也没有读取它。因此把同字段 A、B 都放进去只会由顺序决定留下哪一份，而且运行规则无法与 QueryCoord 选择的 indexID 核对。

CollectionIndexMeta 继续保持“每个字段最多一份运行规则”的形状：

- Vector 字段放当前 Load 选择的唯一 indexID 定义。
- 普通 Scalar 多索引不放入这个单值视图，它们从每个 segment 的 FieldIndexInfo 建立。
- 现有 growing GEOMETRY 路径仍需要一份字段级 RTree 规则，每个 GEOMETRY 字段从当前 Scalar 目标中按稳定顺序保留一份。

接通协议中已有的 FieldIndexMeta.index_id：QueryNode 组装时填写，C++ FieldIndexMeta 保存并暴露。这样 UpdateIndex、Search plan 和诊断都能确认当前规则对应哪个 indexID，而不需要新增另一个字段。

这里不把 growing GEOMETRY 改成多索引。选中的 indexID 只是单值 RTree 规则的身份；已有 growing segment 继续使用创建时的一份 RTree，新 growing segment 使用更新后的规则。普通 Scalar 多 indexID 同时驻留只发生在 sealed segment。

Search plan 继续按 Vector fieldID 读取一份 metric 和查询参数规则，不在 Search 请求中增加 Vector indexID。

### 9.2 Vector 查询规则检查

当前 Search plan 只指定 Vector fieldID，不指定 indexID。创建 plan 时，Segcore 会从 CCollection 当前的 FieldIndexMeta 取得 metric，并补齐原始向量搜索需要的参数；真正执行时，同一份 plan 可能落到仍使用 A、已经使用 B，或暂时只有原始向量的 segment。

在单索引模型中，collection 规则和 segment 物理索引总是一起建立，这个差异不会暴露。在线切换时，CCollection 必须先更新到 B，旧 segment 才能逐个 Reopen；如果 A、B 或原始向量不能执行同一份 plan，过渡期间可能出现参数不被接受、打分含义变化，或相同请求在不同 segment 上含义不同。查询规则检查因此是在线切换的前置条件，不只是普通参数校验。

QueryCoord 在接受新的当前 Vector indexID 前，从目标索引定义生成查询规则摘要，包含：

- metric。
- index type。
- 会限制 Search 参数范围的构建参数。
- brute-force 路径需要的参数。
- growing 和 sealed 临时索引会读取的参数。
- refine 和 raw data 相关条件。

摘要先把默认值补齐，再按稳定顺序生成，不能直接比较用户参数列表的原始顺序。第一次 Load 把摘要与 currentIndexes 一起持久化。修改当前 Vector indexID 时，新摘要与持久化的当前摘要比较；固定的相等规则具有传递性，因此 A → B 尚未完成时再改到 C，只要 A 与 B 已经通过同一规则、B 与 C 也通过，仍在部分 segment 上服务的 A 就不会失去查询兼容保证。DataCoord 中旧 A 的定义已经 Drop 时，也不需要重新猜测它的参数。

QueryNode 实际上报的历史 indexID 仍用于生成 Reopen 和清理任务；如果它的定义仍可读取，可以额外核对摘要，但实际状态不再是恢复查询规则的唯一来源。检查通过后，QueryCoord 在同一次 collection 级提交中保存新的 currentIndexes、vectorQueryRules 和 loadVersion。

在线切换检查固定要求相同 index type，并要求 metric、Search 参数含义、brute-force、refine 和 raw data 路径都能执行同一份 Search plan。检查不通过时，在保存新 currentIndexes 之前返回错误；内核不自动编排 Release → Load。检查和规则生成集中在一个函数中，不把判断散落到任务和 QueryNode。

初次 Load 也复用其中的原始向量能力检查。目标索引文件未完成时，只有 Segcore 原始向量路径能够保持同一 metric、过滤和打分含义，索引视图才生成“无物理索引、保留原始向量”的 SegmentLoadInfo；不满足时不伪装成可查询状态。

### 9.3 A 直接切换到 B

当前 Segcore Reopen 已经具备底层原子替换能力：它比较新旧 SegmentLoadInfo，在未发布的新运行状态中加载新增内容，准备失败时保留旧状态，最后短暂阻止新读进入、等待旧读结束后一次发布。C++ 内部也已经按 segment 串行 Reopen。

现有缺口在外层控制链：QueryCoord 没有稳定过滤当前目标，UpdateIndex 还是空操作，QueryNode 会把同一 segment 后到的 Reopen 当成“正在加载”而直接跳过，Go 侧资源和状态更新又不在 C++ 串行范围内，也没有 loadVersion 阻止旧请求。因此这里复用现有 Segcore 发布结构，只补齐目标、顺序、版本和 Go 侧提交，不新增 Vector 多索引容器。

B 文件已经完成时：

```text
QueryCoord 保存目标 B 和新 loadVersion
    ↓
UpdateIndex 把 CCollection 查询定义更新为 B
    ↓
IndexChecker 为仍使用 A 的 segment 创建 Reopen
    ↓
SegmentLoadInfo 只包含 B 文件
    ↓
Segcore 在未发布的新状态中加载 B，A 继续服务
    ↓
发布前等待旧读结束
    ↓
一次发布 B，随后回收 A
```

Vector 运行状态仍是 `fieldID → 一个索引`，不改成多索引容器。A 和 B 只在 Reopen 准备阶段短暂同时占用资源。

### 9.4 B 文件未完成

当前 sealed segment 在没有已构建 Vector 索引、但原始向量已经加载时，可以执行原始向量搜索。不过如果全局临时索引功能打开且数据量达到阈值，Segcore 会自动从原始向量建立临时索引。因此“SegmentLoadInfo 没有 B 文件”本身不能保证用户已经选定的严格原始扫描；同时，当前 distribution 中“没有 Vector indexID”也无法区分“原始向量已经可查”和“字段数据尚未准备好”。

B 文件未完成时，QueryCoord 仍下发 B 的 collection 查询规则，但该 segment 的 SegmentLoadInfo 不包含 A 或 B 的物理索引文件，并保留原始向量 binlog 或 manifest 字段数据。QueryCoord 发给 QueryNode 的 querypb SegmentLoadInfo 已有 binlog_paths、manifest_path 和 index_infos；进入 C++ 的 segcorepb SegmentLoadInfo 是同一加载说明的副本。两份消息都增加一个很小的 `force_raw_scan_field_ids` 字段，并在 Go 到 C++ 的转换中原样复制，说明哪些 sealed Vector 字段必须使用原始扫描。

QueryCoord 的索引视图在“目标 B、B 文件未完成”时写入该 fieldID。QueryNode 资源估算看到它后不计算新的 sealed 临时索引资源。Segcore 在未发布的新状态中同时做两件事：不再建立新的临时 Vector 索引；如果该字段已经存在 sealed 临时索引，则移除这份临时索引、内部查询配置和可用标记，但保留原始向量列。只有原始向量准备成功且临时索引已经从新状态移除后，才允许发布。

该标记只影响这个 sealed segment，不使用全局开关，也不改变 growing segment 现有的临时索引路径。LocalSegment 在成功后记录该字段处于严格原始扫描状态；失败时继续保留旧物理索引或旧临时索引状态，不更新 Go 侧记录。

```text
A 继续服务
    ↓
在未发布的新状态中加载原始向量
    ↓
原始向量准备成功
    ↓
一次发布“无物理 Vector 索引、原始向量可读”状态
    ↓
随后回收 A
```

加载原始向量或移除旧临时索引失败时不发布新状态，A 或原有临时索引继续服务。发布完成后，Search plan 仍使用 B 的 metric 和查询规则，segment 严格走原始向量搜索。

B 文件完成后，IndexChecker 再创建一次 Reopen：

```text
原始向量
    ↓
加载 B
    ↓
一次发布 B
```

这次 Reopen 不再携带 force_raw_scan_field_ids，正常加载 B。SegmentVersionInfo 增加 raw_scan_field_ids；只有 C++ 成功发布“没有物理索引、没有 sealed 临时索引、原始向量可读”的状态后，QueryNode 才上报该 fieldID。Segment distribution 通过“目标 B、实际无 Vector indexID、raw_scan_field_ids 包含该字段、Vector 字段数据可读、相同 loadVersion”表达中间状态；仅仅没有物理 indexID 不能被当成原始扫描已经完成。

### 9.5 Reopen 失败

现有 C++ Reopen 已经使用未发布状态和读排空完成一次替换，本次继续沿用。新增的 loadVersion 和 Go 侧串行提交只负责保证外层不会把失败或旧版本结果写回运行状态。

Vector Reopen 继续使用现有“发布时短暂停止新查询进入”的机制和查询重试：

- 文件下载、打开或资源申请失败时不发布准备中的新状态。
- 等待旧读超时或取消时不发布。
- 发布期间暂时不能进入的新 Search/Query 在 QueryNode 内部重试。
- C++ 发布成功后，LocalSegment 才更新 Go 状态和 loadVersion。

## 10. 资源计算

### 10.1 初次 Load

当前 QueryNode 资源估算以完整 SegmentLoadInfo 为输入。它已经会遍历重复的 index_infos，并对每个索引调用现有 C++ 估算，得到内存、本地磁盘、mmap 和 GPU 等加载成本；随后再累加字段数据、统计文件和删除数据。因此多 Scalar indexID 不需要重写整套索引估算，现有单索引估算可以逐个复用。

缺口在字段数据部分。当前逻辑允许某个索引声明 HasRawData 时按 fieldID 扣掉对应 binlog，而一个 FieldBinlog 或 manifest column group 可能覆盖多个字段。同字段多个索引后，这会让“其中一个索引带 raw data”错误影响所有候选和原始扫描；如果反过来按字段逐个相加，又会重复计算共享物理单元。因此初次 Load 要明确使用两种计算单位：索引按 indexID，字段数据按实际物理加载单元。

初次 Load 的资源模型调整为：

```text
Scalar
  每个实际字段数据加载单元的一份原始数据
  + 每个目标 Scalar indexID 的资源

Vector
  当前目标 Vector indexID 的资源
  或目标文件未完成时的一份原始向量资源
```

多个 Scalar indexID 不能重复计算同一份字段数据；多个字段共享一个 column group 时也只计算一次。Scalar index 报告 HasRawData 时，不能从字段数据资源中扣除对应物理加载单元。

### 10.2 Reopen 差量

当前 Reopen 在逐 segment 进入 C++ 差量计算以前，就把新的完整 SegmentLoadInfo 交给 requestResource。节点实际占用已经包含旧 A 和所有未变化字段，再按完整新目标预留一次，会重复计算 A 或未变化数据；只删除索引时也可能做无意义申请，从而拒绝本来可以完成的 Reopen。

另一个缺口是 LocalSegment 在加载后会压缩 SegmentLoadInfo，删除大文件列表等运行时不再需要的内容。下一次 Reopen 如果只读这份压缩结果，就无法稳定判断同一个 indexID 的文件是否已经改变，也无法把共享字段数据还原成准确的物理加载单元。为此在 LocalSegment 增加一份独立的小型 `LoadedResourceState`，只保存差量计算需要的信息：

- 每个 indexID 的加载摘要和上次实际采用的资源估算。
- 每个字段数据加载单元的身份、它覆盖的 fieldID 集合、由 binlog 路径或 column group / manifest 身份生成的加载摘要，以及上次实际采用的资源估算。
- 每个 sealed Vector 字段是否存在临时索引、是否处于严格原始扫描，以及对应的上次资源估算。
- 当前 loadVersion。

初次 Load 以每个索引和每个物理字段数据单元的实际估算结果建立这份状态；Reopen 成功后与 Go 侧 indexID 状态一起替换；失败或取消时保持旧值。它不保存完整 binlog 列表，也不替代现有 SegmentLoadInfo。

LoadedResourceState 放在 Go 侧，是因为资源申请发生在调用 C++ Reopen 之前，而当前 C++ 没有提供“先返回 Reopen 差量资源、再执行”的接口。Go 与 C++ 必须使用第 8.2 节同一份加载摘要，不能各自判断文件是否变化。

ReopenSegments 不再先为整批完整 SegmentLoadInfo 统一申请资源。每个 segment 取得自己的 Reopen 串行锁并通过版本复查后，才用最新 LoadedResourceState 和新目标算出新增、替换和删除项，再把只包含新增或替换项的 LoadInfo 交给现有 C++ 资源估算入口。预留规则为：

- Scalar `[A] → [A,B]`：只预留 B。
- Scalar 替换 B：预留新的 B。
- Scalar 删除 B：不申请加载资源。
- Vector A → B：预留 B 的加载峰值。
- Vector A → 原始向量：只预留新增的原始向量。
- Vector 临时索引 → 严格原始向量：只预留缺少的原始向量数据，不为新临时索引申请资源。
- Vector 原始向量 → B：只预留 B。

索引估算继续复用现有 C++ load resource 结果，覆盖内存、本地磁盘、mmap 和 GPU。节点当前物理占用已经包含旧 A 或旧临时索引；把新增或替换差量加入当前占用后，就是发布前的实际峰值。删除项不申请新资源，只有新状态发布后才释放旧物理索引或旧临时索引占用。

不同 segment 的 Reopen 仍使用现有并发资源预留并累计差量。同一个 segment 的后续请求在前一个提交并释放预留后才重新计算；成功、失败和取消路径都释放各自的预留。

## 11. 重启、晚到任务和状态收敛

### 11.1 QueryCoord 重启

当前 QueryCoord 先从 catalog 的 CollectionLoadInfo 恢复已加载 collection 和 partition，再用这些 collectionID 恢复 replica；找不到 collection 的 replica 会被清理。随后恢复 target，并在启动 observer 和 checker 前读取一轮 QueryNode distribution。这套流程能恢复现有单索引 Load，但它只认识“有 CollectionLoadInfo 就是已加载”，没有 default、current、releaseInProgress 和 loadVersion，也不能恢复一条尚未完成的 Release。

因此重启恢复继续复用现有 catalog、target 和首次 distribution 同步，但先以新的 CollectionIndexLoadState 给 collection 分类。持久状态回答“当前目标是什么、是否正在释放”，DataCoord 回答“定义和文件是否仍存在”，QueryNode 上报回答“实际已经到哪里”；三者不能互相代替。

QueryCoord 从 catalog 恢复 default、current、currentLoaded 和 loadVersion，再读取 DataCoord 有效 indexID 和 QueryNode 实际状态。

```text
持久 current 目标
    + DataCoord 有效定义/文件
    + QueryNode 实际 indexID、字段数据和 loadVersion
    ↓
重新生成缺失的 Load、Reopen 或清理任务
```

DataCoord 暂时不可用时不删除任何索引引用。只有成功取得完整定义并确认 indexID 已不存在后，才清理持久状态。QueryNode session 和第一次 distribution 尚未收齐时，也不能把“暂时没有节点上报”当成节点已经退出。

如果恢复到 `releaseInProgress=true`，QueryCoord 不生成新的 Load 或 Reopen，而是使用仍保留的 CollectionLoadInfo 和 replica 继续完成版本通知与 Release；最终提交完成后再删除这些运行记录。

### 11.2 QueryNode 重启

当前 QueryNode 不持久化 CCollection、segment、channel 和分布状态；进程重启后从空状态开始，由 QueryCoord 按持久 Load 配置重新发送 WatchDmChannels 和 LoadSegments。这是合适的边界：QueryNode 重启会同时终止旧连接和旧请求，没有必要再把运行态复制进本地持久存储。

新增的 collection loadVersion、currentLoaded 和 Vector/GEOMETRY 运行规则也只保存在 QueryNode 内存。QueryNode 重启后，QueryCoord 使用持久 current 目标重新发送 UpdateIndex、WatchDmChannels 和 LoadSegments；当前未加载的 collection 不恢复。这样 QueryCoord 仍是恢复来源，QueryNode 不增加第二份持久真相。

### 11.3 晚到请求

当前 segment、channel 和 target version 只保护各自的一次分布更新。Reopen 不把 LoadSegmentsRequest.Version 当成 collection 索引配置版本，Update/Reopen 类型任务在 RPC 返回后也基本完成，无法确认实际 indexID 已经上报。于是 A → B → C 时，旧 RPC 即使已经从 QueryCoord 队列删除，仍可能在 QueryNode 内完成并发布。

独立 loadVersion 必须在 QueryCoord 任务、QueryNode 进入检查和最终发布三处同时生效。只在 RPC 入口检查不够，因为旧请求可能已经进入下载或 C++ Reopen；发布前还要再检查，并让任务完成以 distribution 的实际状态为准。

QueryCoord 和 QueryNode 两端都检查 loadVersion：

- QueryCoord 不发送已经过期的任务。
- QueryNode 不接受旧版本的 UpdateIndex、WatchDmChannels、LoadSegments、SyncDistribution、ReleaseSegments 或 UnsubDmChannel，并在提高版本前取消和排空仍在执行的旧状态修改请求。
- Release 先写入并下发更高版本，旧 Load 不能在释放后重新发布。
- Drop Index 改变 current 后，旧 Reopen 不能重新加载已删除 indexID。
- Reopen 和清理任务只有在上报的 loadVersion、indexID 和加载摘要都与目标一致时才完成；同 indexID 的新文件不能只靠 ID 被误判为完成。

## 12. CDC 数据流

当前 AlterLoadConfig 和 DropLoadConfig 都是 CChannel DDL 消息，没有 Unreplicable 标记。CDC 会把它们从 Primary WAL 复制到 Secondary，在 Secondary 改写 channel 后写入本地 WAL，再执行相同的 QueryCoord ACK callback。因此 Load 和 Release 配置本来就会在集群之间同步，并不是各集群各自猜一份。

AlterLoadConfig 保存 collection、完整 partition 集合、LoadFieldConfig、replica 和用户 replica 模式；复制服务可以把 use_local_replica_config 设为 true，使 Secondary 保留相同 fieldID、indexID 和 partition，但使用自己的 replica / resource group。DropLoadConfig 只用 dbID 和 collectionID 删除整份当前 Load 配置。QueryNode 实际加载到哪一步、一次查询最终使用哪个 Scalar indexID，以及本地资源情况都不属于这些 DDL，不会通过 CDC 复制。

现有缺口只在 Release 之后仍需保留的 collection 默认索引：当前没有对应消息，Secondary 无法从 DropLoadConfig 之后的状态恢复“下一次默认 Load 哪个 indexID”。因此新增 AlterIndexLoadDefault 进入同一 CChannel 和 CDC 顺序；当前 Load 仍继续由 AlterLoadConfig / DropLoadConfig 同步，不增加另一套跨集群 QueryNode 状态复制。

Primary 上的以下消息继续沿 CDC 复制：

- CreateIndex、AlterIndex 和 DropIndex。
- AlterIndexLoadDefault。
- AlterLoadConfig 和 DropLoadConfig。

数据流为：

```text
Primary CChannel DDL
    ↓ CDC
Secondary 本地 WAL
    ↓ 相同 ACK callback
Secondary DataCoord / QueryCoord 保存相同 fieldID 和 indexID 目标
    ↓
Secondary 使用自己的 QueryNode、replica 和资源完成加载
```

复制服务只按现有配置改写 AlterLoadConfig 的本地 replica 标记，不改写 field、partition 和 indexID。新默认消息不增加额外改写。复制消息进入 Secondary 本地 WAL 后取得 Secondary 自己的 TimeTick，所以 Primary 和 Secondary 各自生成并比较本集群的 loadVersion，不跨集群比较数值。

Search、Query、Scalar 最终选择的 indexID、原始扫描路径、QueryNode 实际状态和加载进度不进入 CDC。

## 13. 滚动升级和功能启用前回滚

当前单索引链路的公开 Load 请求、QueryCoord 内部请求和 CollectionLoadInfo 都使用单值 map，QueryNode 也没有 loadVersion 和可工作的 UpdateIndex。滚动升级期间，如果旧进程仍在服务就直接放行新格式，旧 Proxy 或 QueryCoord 会折叠同字段多个 Scalar indexID，旧 QueryNode 会忽略新字段且不能阻止晚到请求；旧版本也不认识新的 AlterIndexLoadDefault 消息。

因此“代码已经升级”和“功能可以使用”必须分开。所有组件先带着新代码但继续走旧路径，等两个集群的组件、默认快照和正数初始化版本都准备好后再开放新请求。Scalar 和 Vector 使用独立开关，是因为它们共用控制面数据形状，但运行时改造和资源风险彼此独立，可以分别完成核对和启用。

增加两个独立功能开关：

- Scalar 多索引。
- Vector 多定义和单索引 Reopen。

每个开关在进程内还有一个简单的“功能就绪”状态。外部开关已经打开、但默认快照、旧状态初始化和正数版本下发尚未完成时，新代码必须能解析并落盘 AlterIndexLoadDefault、带多 indexID 的 AlterLoadConfig 和初始化版本等内部准备消息；否则 Secondary 无法靠 Primary 复制来的消息完成准备。这个阶段只是不放行公开的多索引请求，也不让普通加载任务读取半套新状态。准备完成后再把查询和 Load 主路径切到新状态。它不是第三个配置项，只表示本次启用流程是否完成。

开关关闭时：

- DataCoord 保持当前 Create Index 规则：普通字段和同一 JSON path 仍限制一个索引，JSON 不同 path 的现有例外保持不变。
- Proxy 不发送新的 FieldIndexLoadConfig。
- QueryCoord 继续使用旧 FieldIndexID 单值路径。
- 不生成 AlterIndexLoadDefault。
- QueryNode 新代码继续接受旧的单索引消息。

升级与启用顺序：

```text
Secondary 全部组件升级，两个开关关闭
    ↓
Primary 全部组件升级，两个开关关闭
    ↓
两端可以预先建立本地 current 状态，但不写新的 Streaming 消息
    ↓
此处以前仍可回滚旧版本
    ↓
打开 Secondary 对应开关，结束旧版本回滚窗口
    ↓
打开 Primary 对应开关；Primary 生成 default 快照并沿 CDC 复制
    ↓
两端完成 CollectionIndexLoadState 核对，并向本集群 QueryNode 下发正数初始化版本
    ↓
取消并排空版本 0 的旧加载请求后，对外接受新的多索引请求
```

第一次打开任一集群的功能开关以前，旧 CollectionLoadInfo 和旧协议字段仍保持完整，也没有写入旧版本不认识的 Streaming 消息；此时回滚旧版本，新 catalog prefix 会被忽略。开关打开并可能写入新消息后不再保证回滚旧版本。内核不检查其他集群版本，升级和开关顺序由外部控制面保证。

## 14. 同一次实现中的落地顺序

以下顺序用于组织同一个完整实现，不作为分期交付：

1. 增加公开 Load 配置、QueryCoord 内部 ID 配置、CollectionIndexLoadState、loadVersion 和 distribution 字段，并重新生成协议代码。
2. 增加 QueryCoord catalog、内存管理器、启动迁移和两个功能开关。
3. 增加 AlterIndexLoadDefault Streaming 消息、回调和 CDC 测试。
4. 放开 DataCoord 同字段多索引定义，并连接 DataCoord index callback 与 QueryCoord 索引变更通知函数。
5. 改造 Proxy Load 解析和 QueryCoord 完整目标生成。
6. 增加 QueryCoord 公共索引视图过滤和 CollectionIndexObserver，接入 LoadSegments、WatchDmChannels、SyncDistribution、Reopen 和 IndexChecker。
7. 增加任务 loadVersion、加载与释放 RPC 的版本字段、QueryNode 最低版本记录、UpdateIndex 完整状态和 distribution 上报。
8. 完成 Segcore Scalar indexID 容器、LoadDiff、选择器、原始字段数据和 cache key 改造。
9. 完成 Vector 查询规则检查、A → B、A → 原始向量 → B 和 Reopen 差量资源计算。
10. 补齐重启、Drop、CDC、滚动升级和并发 Reopen 的完整测试。

功能开关只有在上述代码全部合入后才进入启用流程；默认快照复制、正数版本下发和旧请求排空属于启用流程本身，在完成前不接受新的多索引请求。

## 15. 测试代码改造

协议和 Streaming 测试需要覆盖新的重复 indexID、空默认字段、稳定排序、FieldIndexMeta 现有 index_id 的接通、force_raw_scan_field_ids、消息生成和 CDC 转发，确保 Primary 与 Secondary 得到相同的 fieldID / indexID 目标，同时仍使用各自的 replica 配置。

QueryCoord 测试重点放在三处。第一处是完整目标生成函数，直接构造“显式值、已有 load field、新增 load field、删除 load field、空默认”组合。第二处是 catalog 组合提交，在 KV 保存失败、删除失败和进程重启后检查 CollectionIndexLoadState 与 CollectionLoadInfo 不会出现半套状态。第三处是 CollectionIndexObserver 和任务版本，用可控的 QueryNode 上报模拟 RPC 失败、节点离开、启动时 session / distribution 尚未恢复、释放中重启，以及 A → B → C 连续修改。

QueryNode 测试需要为 collection 版本管理器和 per-segment Reopen 串行队列增加并发用例，刻意让旧 Load、旧 Release 和新 Reopen 交错，并在 C++ 发布前后注入暂停，检查最终提交的 loadVersion、indexID 和 collection 查询定义保持一致。Release 用例还要覆盖活动记录转成轻量版本拦截记录、distribution 删除上报、过期前拒绝旧 RPC、过期后清理，以及更高版本 Load 重新建立活动状态。

Segcore 测试按数据结构分开。Scalar 覆盖同字段多个普通索引、JSON path 索引和 NGRAM 索引的独立加载、选择与删除，并检查字段 column 始终存在；GEOMETRY 另测 sealed 多 indexID 与 growing 单 RTree 的边界。资源用例还要构造一个 column group 覆盖多个 Scalar 字段，确认只加载和计算一次。Vector 继续检查单索引槽的 A → B 和 A → 原始向量 → B；后一个用例要在 sealed 临时索引全局开关打开且数据量超过阈值时先建立临时索引，再确认 force_raw_scan_field_ids 会从新状态移除旧临时索引、阻止重新建立，并上报 raw_scan_field_ids。LoadDiff 与资源测试使用相同 indexID、不同加载摘要的输入，确认 replace、distribution 完成判断和差量预留都能识别文件变化。

最后增加带功能开关的集成测试：开关关闭时运行现有单索引链路并验证可以回滚；先打开 Secondary、再打开 Primary，完成默认快照复制和正数版本下发后才放行新的多索引请求；Scalar 和 Vector 两套开关分别执行这条流程。

## 16. 完成后的核心状态

实现完成后，系统保持以下关系：

```text
DataCoord 全部有效索引
    ∩
QueryCoord 当前 Load indexID
    =
允许发送给 QueryNode 的索引范围
```

Sealed Scalar segment 可以在这个范围内同时保存多个 indexID，并按条件选择一个执行；growing GEOMETRY 仍是每字段一份内部 RTree。Vector segment 始终只有一个物理索引；切换期间可以暂时只有原始向量。所有 Load、Reopen、Release 和 Drop 都受 loadVersion 保护，旧任务不能覆盖最新目标。
