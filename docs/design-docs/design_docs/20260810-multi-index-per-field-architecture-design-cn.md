# Milvus 单字段多索引架构预研方案

本文给出 Milvus 在同一个字段上支持多个索引的整体架构调整方案。

主方案包含：

1. 同一个字段可以独立定义和构建多个索引。
2. Scalar 字段可以同时加载多个索引，并在每个过滤条件执行时选择一个索引。
3. Vector 字段可以构建多个索引，但一次 Load 对每个字段只加载一个索引。
4. 已加载 collection 可以通过修改当前 Load，从 Vector 索引 A 在线替换到 B。
5. Collection 默认索引、当前 Load 选择和单次查询选择分层管理。
6. 索引定义、collection 默认规则和 Load/Release DDL 沿 CDC 复制；实际加载进度和单次查询选择由每个集群本地管理。

Vector 多索引同时加载、查询指定 Vector 索引和动态选择 Vector 索引作为独立扩展能力，不属于主方案的运行时范围。

Milvus 当前源码是本文判断现状和架构边界的权威依据。本文只讨论架构、组件职责、状态、数据流和资源边界，不讨论具体类、协议字段、锁、任务实现和文件写入细节。

## 1. 背景

Milvus 当前的索引定义、构建任务和索引文件已经能够通过 indexID 区分。

查询侧仍然建立在“一个字段只有一个可用索引”的前提上：

- Load 入口会把同字段索引压缩成一个选择。
- QueryCoord 保存了部分 Load 索引信息，但检查和下发没有完全按该选择工作。
- QueryNode 的部分状态已经按 indexID 管理，Segcore 普通 Scalar 和 Vector 索引仍主要按 fieldID 保存一个。
- Scalar 表达式直接取得字段上的唯一索引。
- Vector 查询计划主要根据字段和 metric 取得唯一索引。
- Growing Segment 每个 Vector 字段最多维护一个临时索引。

因此，解除索引定义限制本身并不能完成一列多索引。查询侧需要把索引身份、Load 目标、实际加载状态和查询选择都改为明确使用 indexID。

### 1.1 Scalar 同类系统参考

- PostgreSQL 允许同一列建立多个索引，由查询计划结合统计信息和每种索引的成本估算自动选择。它也明确建议用 `EXPLAIN` 检查计划，而不是承诺每次都选到理论最快路径。
- MySQL 同时提供自动选择和 `USE/FORCE/IGNORE INDEX`，说明“自动选择”和“用户指定”可以共存。
- ClickHouse 的 data skipping index 会自动参与过滤，也可以要求查询必须使用某类索引；它更强调索引是否能减少读取，不把“指定”解释为任意强制执行。

这些系统的共同点是：每种索引先说明自己能支持什么查询，再由公共选择层比较；较好的自动选择依赖统计信息。Milvus 第一版只做正确候选和稳定选择，成本估算作为独立增强，方向与这些系统一致。

参考：[PostgreSQL](https://www.postgresql.org/docs/current/indexes-examine.html)、[MySQL](https://dev.mysql.com/doc/refman/8.4/en/index-hints.html)、[ClickHouse](https://clickhouse.com/docs/optimize/skipping-indexes)。

### 1.2 Vector 同类系统参考

- Qdrant 和 Weaviate 常用“有名字的向量”表达不同向量目标，查询先明确选择目标。
- pgvector 让距离运算符决定 metric，再由 PostgreSQL 选择能支持该运算的物理索引。
- 没有看到成熟产品把“同一个 Vector 字段上的多个物理 ANN 索引自动选最快”作为通用能力；不同 ANN 索引还可能带来不同召回结果，不只是速度差异。

这更支持把 Vector 的默认查询和显式选择放在查询开始前完成。动态选择需要单独研究延迟、召回和资源之间的取舍，不应成为基础多索引的前提。

参考：[Qdrant](https://qdrant.tech/documentation/concepts/vectors/#named-vectors)、[Weaviate](https://docs.weaviate.io/weaviate/config-refs/collections#named-vectors)、[pgvector](https://github.com/pgvector/pgvector#querying)。

## 2. 目标

### 2.1 共同目标

- 一个字段可以拥有多个独立 indexID。
- 每个 indexID 独立创建、构建、保存、查看和删除。
- 所有跨组件的索引身份都使用明确 indexID。
- 不依赖 map、列表或元数据返回顺序选择索引。
- 索引定义、当前 Load 目标、实际加载状态和查询选择互相分开。
- QueryCoord 保存“应该加载什么”，QueryNode 上报“实际加载了什么”。
- ReleaseCollection，或释放最后一批已加载 partition，结束当前 Load，但保留更高修改版本的“当前未加载”状态和 collection 默认索引。
- 重启恢复当前 Load 目标，不重新猜测索引。

### 2.2 Scalar 目标

- 同一个 Scalar 字段可以同时加载多个索引。
- 多索引范围包括用户创建的普通 Scalar 索引、JSON path/cast 索引和 NGRAM 索引。
- 每个过滤条件独立选择一个索引。
- 查询可以自动选择，也可以为某个过滤条件指定 indexID。
- 只有查询语义一致的索引才能互为候选。
- 原始字段数据始终有独立可读路径，不依赖某个索引提供。
- 第一版只保证选择一个能正确执行的路径，不保证最快。
- 架构保留以后增加成本估算的入口。

### 2.3 Vector 主方案目标

- 同一个 Vector 字段可以定义和构建多个索引。
- 一次 Load 对每个 Vector 字段只选择一个 indexID。
- QueryNode 和 Segcore 稳定服务时，每个 Vector 字段只保存一个索引。
- 已加载 collection 可以从当前索引 A 修改为 B。
- 查询使用当前 Load 的唯一 Vector 索引。
- 主方案不让多个 Vector 索引长期同时驻留。
- 主方案不让 Search 访问已构建但未加载的其他 Vector 索引。

### 2.4 不在主方案中的扩展

- 同一个 Vector 字段同时加载多个索引。
- Search 在多个已加载 Vector 索引中显式选择。
- Milvus 根据查询和运行状态动态选择 Vector 索引。
- 不同 metric 或查询参数规则的 Vector 索引之间进行查询级无停机切换。
- Growing Segment 为同一字段长期维护多份 Vector 临时索引。

这些能力可以建立在共同状态模型上，但需要额外的 QueryNode、Segcore、查询计划和资源架构调整。

## 3. 总体架构

多索引架构分为四层：

~~~text
索引定义与构建
    DataCoord 管理每个 indexID 的定义、构建结果和文件
                    ↓
查询集群 Load 目标
    QueryCoord 管理 collection 默认和当前 Load 选择
                    ↓
运行时索引状态
    QueryNode / Segcore 管理每个 segment 实际加载的索引
                    ↓
查询选择
    Scalar 每个过滤条件选择索引
    Vector 使用当前 Load 的唯一索引
~~~

每层只管理自己的事实：

| 层 | 负责的事实 | 不负责的事实 |
| --- | --- | --- |
| 索引定义与构建 | 有哪些 indexID、如何构建、文件是否完成 | 查询集群加载哪个索引 |
| Load 目标 | 本次 Load 需要哪些 indexID、默认使用哪个 | QueryNode 当前是否已经加载完成 |
| 运行时状态 | 每个 segment 实际有哪些索引、占用多少资源 | 用户原始索引名称和持久索引定义 |
| 查询选择 | Scalar 条件最终使用哪个 indexID；Vector 使用哪个查询含义或目标 indexID | 构建和索引文件生命周期 |

## 4. 核心状态模型

### 4.1 索引定义

每个索引定义独立保存：

- collection 和 field 身份。
- indexID 和 index name。
- index type、metric 和构建参数。
- JSON path、cast 等定义参数。
- 每个 segment 的构建状态和索引文件。

同字段索引之间没有主从关系，也不依赖定义顺序。

索引定义属于数据管理状态，由 DataCoord 管理并沿 CDC 复制。

候选判断使用两类信息：

- 定义级兼容：由 index type、path、cast 和参数判断，避免明显不匹配的索引进入候选。
- 运行时能力：由 QueryNode / Segcore 针对具体 segment、indexID 和过滤条件判断，包括索引当前是否可用、是否能处理当前查询值、是否需要原始数据复查，以及实际内部索引类型。

TextMatch/PhraseMatch、Primary Key 专用索引和 JSON Stats 沿用现有 fieldID 专用路径，不变成用户可选择的多 indexID 候选。它们仍可以在普通 Scalar 多索引选择之前完成自己的判断。

### 4.2 Collection 默认 Load 规则

每个集群的 QueryCoord 持久保存每个字段的默认 Load 规则：

~~~text
fieldID
  ├─ 默认加载的 indexID 集合
  └─ 默认查询 indexID，仅 Vector 多加载扩展需要
~~~

默认规则允许处于“未设置”状态，并把这个状态明确保存下来；未设置不等于从索引列表临时取第一个。

Scalar 字段只保存默认加载集合，自动查询选择使用稳定规则，不保存一个没有执行作用的“默认查询索引”。Vector 主方案的默认集合只包含一个 indexID。Vector 多加载扩展可以包含多个 indexID，并另外保存默认查询 indexID。

默认规则由 QueryCoord 管理。Primary 上的修改作为查询配置 DDL 沿 CDC 复制，Secondary 保存相同的明确 indexID。Proxy 只解析请求中明确写出的 index name，不直接读取持久化默认规则。

系统需要提供读取和修改默认规则的入口。修改默认规则只影响以后从“当前未加载”开始的新 Load，不直接改变已经加载的 collection；修改当前 Load 使用独立入口。Primary 接收普通修改请求，Secondary 通过 CDC 应用同一条修改。

默认规则必须保存明确 indexID，不能在每次 Load 时从 map 或索引列表中临时取“第一个”。初始化分为两种情况：

- 新 collection 的某个字段第一次拥有有效索引时，把这个 indexID 保存为初始默认值并沿 CDC 复制；后来创建的索引不自动改变它。
- 已有 collection 第一次启用本能力时，如果当前已经 Load，使用当前 Load 的 indexID 初始化；如果当前没有 Load，使用升级前唯一有效的 indexID 初始化。

Primary QueryCoord 在收到索引定义变化或重启核对 DataCoord 时生成明确初始化结果，Secondary 通过 CDC 接收；默认值不依赖 collection 已经 Load。

如果旧元数据无法得到唯一结果，该 collection 在明确写入默认值前不能启用多索引。正常旧数据满足“一字段一个索引”的现有限制，因此通常不会进入这个分支。

### 4.3 当前 Load 状态与目标

当前 Load 目标描述这一次 collection Load 需要的索引和字段数据。

Scalar：

~~~text
fieldID
  └─ 一组需要加载的 Scalar indexID
~~~

Vector 主方案：

~~~text
fieldID
  └─ 一个需要加载的 Vector indexID
~~~

Vector 多加载扩展：

~~~text
fieldID
  ├─ 一组需要加载的 Vector indexID
  └─ 其中一个默认查询 indexID
~~~

Vector 多加载扩展要求为该 Vector 字段保存“原始向量必须可读”。当目标 indexID 的物理索引尚未就绪时，sealed segment 使用原始向量执行同一 metric 和打分含义。

Scalar 字段还包含“原始字段数据必须可读”的目标。该资源只按字段计算一次，不随同字段索引数量重复计算。

当前 Load 状态保存明确的修改版本，并区分“已加载目标”和“当前未加载”：

~~~text
当前 Load 状态
  ├─ 修改版本
  └─ 已加载目标，或当前未加载
~~~

运行状态只能向最新修改版本收敛，旧 Load 或 Reopen 的结果不能覆盖新状态。

当前 Load 目标属于可复制的查询配置 DDL。Primary 和 Secondary 保存相同的期望 partition、field、indexID、默认查询 indexID 和 Load/Release 状态；replica 与 resource group 继续服从现有的本地 replica 配置开关。

正在排空哪些旧索引、查询入口是否已经接受新视图、QueryNode 实际加载进度和资源占用都属于集群本地状态，不沿 CDC 复制。

生成完整 Load 目标时按以下顺序取值：

1. 请求明确指定的字段使用请求中的 indexID。
2. Collection 已经 Load 时，未指定字段沿用当前 Load 目标。
3. 只有从“当前未加载”开始的新 Load，未指定字段才使用 collection 默认规则。

如果第 3 步需要的默认规则处于“未设置”，QueryCoord 拒绝这次未明确指定 indexID 的 Load；不能临时选择索引列表中的第一个。

LoadPartitions、修改 replica 或修改其他 Load 内容都继承同一份 collection 级索引目标，不为不同 partition 建立长期不同的索引选择。

ReleaseCollection，或 ReleasePartitions 删除最后一批已加载 partition，写入更高修改版本的“当前未加载”状态，并保留 collection 默认索引。只释放部分 partition 时，保留当前索引目标，只修改 partition 集合。这个状态不能立即删除，否则晚到的旧任务可能重新发布已经释放的索引。如果现有失败策略最终放弃一次 Load，也使用同样的状态变化。只有 Drop Collection 同时删除默认规则和当前 Load 状态。

### 4.4 Segment 实际状态

QueryNode 上报每个 segment 实际加载的 indexID，以及 Load 目标要求的字段原始数据或原始向量是否已经可读。

Scalar 的每个已加载 indexID 还要有一份不打开索引内容即可读取的运行描述，至少说明定义级查询语义、当前是否可用、索引内容是否仍是冷状态，以及是否需要原始数据复查。候选筛选只读这些描述，选定后才准备具体索引内容。

QueryCoord 通过比较 Load 目标和实际状态得到：

- 哪些 indexID 尚未加载。
- 哪些 indexID 已经过期。
- 哪些 segment 需要初次 Load。
- 哪些 segment 需要 Reopen。
- 哪些索引可以释放。
- 哪些字段原始数据尚未就绪。
- 当前 Load 或索引切换是否完成。

实际状态是运行时事实，不替代持久化的 Load 目标。

Vector 多加载扩展中，QueryCoord 还从实际状态得到“可查询 indexID 集合”。只有所有目标 segment 和服务副本都能通过该物理索引或原始向量执行对应查询含义，它才能进入集合。查询入口只能选择该集合中的索引。

“可以查询”和“物理索引目标已经全部完成”是两种状态。原始向量可读时，B 可以先进入可查询集合；QueryCoord 仍继续跟踪哪些 segment 尚未加载 B，直到物理目标完全收敛。

QueryCoord 把 Vector 多加载的查询视图发布给所有查询入口：

~~~text
collection / fieldID
  ├─ 视图版本
  ├─ 默认查询 indexID
  ├─ 可查询 indexID 集合
  └─ 各 indexID 的 metric 和查询参数规则
~~~

查询入口使用这份带版本的视图固定本次 Search 的 indexID。Load 目标回答“系统正在准备什么”，查询视图回答“新查询现在允许使用什么”，两者不能合成一份状态。

复制来的期望默认可以先变为 B，但在本集群确认 B 可查询前，当前生效的查询默认仍可保持 A。这个延迟只存在于本集群查询视图，不修改或反向复制期望配置。

Vector 查询级无停机切换还需要一份每集群本地、可恢复的过渡状态：

~~~text
本集群过渡状态
  ├─ 正在应用的期望 Load 修改版本
  ├─ 当前查询视图版本
  ├─ 正在排空的旧 indexID
  └─ 各查询入口对新视图的确认
~~~

Primary 和 Secondary 根据同一份复制目标分别推进这份状态。一个集群完成 A 排空不能代表另一个集群也已经完成。

### 4.5 查询绑定

Scalar 查询绑定到单个过滤条件：

~~~text
过滤条件
  ├─ 原有字段、操作和值
  └─ 可选 indexID
~~~

未指定 indexID 时，由 Segcore 从当前已加载候选中选择一个。

Vector 主方案的稳定状态中，查询使用当前 Load 的唯一 indexID。

Vector 主方案继续由 Search 请求提供 metric 和查询参数。渐进切换期间，查询绑定 A、B 共同支持的 metric、打分含义和参数；每个 segment 使用自己当前已经发布的 A 或 B。此时一次查询不绑定单一物理 indexID。

Vector 多加载和查询级无停机切换中，查询入口从最新查询视图中固定一个目标 indexID，并把 indexID 和视图版本带到执行端；查询结束前保持不变。QueryNode 只接受仍处于可查询或排空状态的 indexID，过期请求不能重新启用已经回收的索引。

## 5. 组件职责调整

### 5.1 Proxy

Proxy 负责请求入口和索引名称解析：

- 允许同字段创建多个独立索引。
- 将 Load 中明确写出的 index name 解析为 indexID。
- 将修改 collection 默认规则的请求路由到 QueryCoord；Primary 生成可复制的配置 DDL。
- 将 Scalar 查询中指定的 index name 绑定到具体过滤条件。
- 将请求中明确指定的 indexID 传入后续 Load 或查询链路。
- 索引定义删除后刷新名称到 indexID 的本地映射。
- 在 Vector 多加载扩展中接收 QueryCoord 发布的查询视图，并在 Search 分发前固定 indexID。

Proxy 不负责：

- 判断某个 segment 是否已经加载索引。
- 比较 Scalar 索引执行成本。
- 根据 QueryNode 本地资源选择索引。

### 5.2 DataCoord

DataCoord 继续作为索引定义和构建结果的管理者：

- 按 indexID 管理同字段多个索引定义。
- 为每个 segment、每个 indexID 独立构建索引。
- 独立保存每个索引的构建结果和文件。
- 向 QueryCoord 提供有效索引定义、参数和构建完成状态。
- Drop Index 后将 indexID 从有效定义中删除。

DataCoord 不管理：

- 各查询集群的 collection 默认索引。
- 当前 Load 的索引集合。
- 单次查询最终使用的索引。

### 5.3 QueryCoord

QueryCoord 成为查询集群索引目标的管理者：

- 管理本集群每个字段的默认 Load 集合和默认查询 indexID。
- 校验并持久保存 collection 默认规则的修改。
- 在 Primary 生成 Load、Release 和默认规则 DDL，在 Secondary 应用复制来的相同配置。
- 对未明确指定的字段读取当前 Load 或 collection 默认，生成完整 Load 目标。
- 持久保存当前 Load 状态和修改版本。
- 使运行状态只能向最新修改版本收敛。
- 只检查并下发当前 Load 需要的 indexID。
- 同时过滤 segment 索引文件信息和 collection 级索引信息。
- 比较索引目标、字段数据目标与 QueryNode 实际状态。
- 汇总每个 indexID 在各 segment、replica 上的完成情况。
- 得出 Vector indexID 是否已经可以被查询统一选择。
- 向所有查询入口发布带版本的默认查询 indexID 和可查询集合。
- 在查询级无停机切换中保存本集群旧索引正在排空的阶段；该状态不复制。
- 重启后继续未完成的 Load 或索引切换。

QueryCoord 不再使用“字段存在任意索引”表示目标索引已经可用。

### 5.4 QueryNode

QueryNode 管理本节点每个 segment 的实际索引：

- 按 indexID 接收、加载、卸载和上报索引。
- 根据索引目标和字段数据目标计算统一运行资源。
- 向 Segcore 提供已加载索引、定义参数、字段数据状态和可选的用户指定 indexID；具体查询能否执行由 Segcore 判断。
- 实际状态以 Segcore 已经发布的 indexID 集合为准，不能只用 Load 信息或 QueryNode 的 Go 元数据代替。
- 在发布新 segment 状态前检查修改版本。
- 在旧查询排空后回收旧索引资源。

QueryNode 的资源模型同时考虑：

- 最终稳定占用。
- Load 过程的临时内存。
- 在线切换时新旧索引同时存在的峰值。
- 多个 segment 并行 Reopen 带来的 collection 总峰值。

### 5.5 Segcore

Segcore 是运行时多索引保存和查询选择的核心。

Scalar：

- 同一个字段按 indexID 保存多个用户索引；indexID 是所有权和删除、替换的主身份，fieldID、JSON path 和 cast 只用于查找候选。
- 普通 Scalar、JSON path/cast 和 NGRAM 可以继续使用各自适合的运行时对象，但都必须能从 indexID 找到并独立删除，加载同 path 的新索引不能覆盖另一个 indexID。
- 原始字段数据状态与索引状态分开。
- 先按定义筛选候选，再由当前 segment 上的索引判断运行时能力。
- 自动选择一个可用索引，或执行查询指定的 indexID。
- 自动模式的最终 indexID 由 Segcore 对当前过滤条件选择，QueryNode 不提前决定。
- 只准备最终选中的索引内容。
- 必要时读取原始字段数据完成扫描或二次检查。
- TextMatch/PhraseMatch、Primary Key 专用索引和 JSON Stats 保持现有专用路径，不进入这组用户 indexID 候选。

Vector 主方案：

- 每个字段保存一个当前 Vector 索引。
- Reopen 在临时状态中准备新索引。
- 新索引全部准备成功后才发布。
- 失败时继续保留旧索引。

Vector 多加载扩展：

- 同字段按 indexID 保存多个 Vector 索引。
- 每个索引独立准备、查询和删除。
- raw data、refine 和资源能力与具体 indexID 对应。
- 删除索引 A 不能影响同字段的索引 B。
- 查询必须携带明确 indexID。

### 5.6 存储层

存储层继续使用每个 indexID 独立文件的模型：

- 不新增组合索引文件。
- 不新增必须整体构建和删除的“索引组”。
- 多个索引可以共享同一份字段原始数据。
- 删除一个索引不直接删除仍被查询需要的原始字段数据。

## 6. 索引定义与构建链路

~~~text
Create Index
    ↓
DataCoord 创建独立 indexID
    ↓
按 segment 为该 indexID 创建构建任务
    ↓
生成独立索引文件
    ↓
记录每个 segment、indexID 的完成状态
    ↓
供 QueryCoord 按当前 Load 目标读取
~~~

同字段其他索引不参与该索引的构建状态和文件生命周期。

新 segment、Import 和 Compaction 产生的数据仍按每个 indexID 独立进入索引构建流程。

对 Scalar 和 Vector 主方案，具体等待哪些备用索引完成主要是使用规则：查询仍有原始字段路径，或只要求当前唯一 Vector indexID。

对 Vector 多加载和 Search 显式选择，架构采用原始向量补充路径。一个 indexID 只要仍在“可查询 indexID 集合”中，每个已经进入查询范围的 sealed segment 就必须能用该物理索引或原始向量执行它。

新 segment 可以在原始向量可读后进入查询范围，不等待所有备用 Vector 索引完成。查询指定 B 时：

- 已有 B 的 segment 使用 B。
- 尚未有 B 的 segment 使用原始向量，保持相同 metric 和打分含义。
- B 构建完成并加载后，该 segment 后续查询再使用 B。

因此显式 indexID 表示本次查询的目标索引和查询含义，不承诺每个 sealed segment 都物理使用它。Vector Load 目标、Segment 实际状态、资源核算和重启恢复都必须包含“原始向量可读”。如果以后需要严格物理保证，可以增加“等待全部索引后再放行 segment”的更强模式，但它不是本方案前提。

## 7. Scalar 多索引架构

### 7.1 Load 链路

~~~text
Load 请求
    ↓
Proxy 解析请求中明确指定的 Scalar indexID
    ↓
QueryCoord 用当前 Load 或 collection 默认补齐未指定字段
    ↓
QueryCoord 保存完整索引集合和字段原始数据目标
    ↓
QueryCoord 获取各 segment 的索引文件
    ↓
只保留目标集合中的 indexID
    ↓
QueryNode 计算索引和字段数据资源
    ↓
按 segment、indexID 加载多个 Scalar 索引，并建立字段数据可读路径
    ↓
Segcore 发布新的多索引状态
    ↓
QueryNode 上报索引状态和字段数据状态
~~~

Load 状态始终按 indexID 表达，不再把同字段多个 Scalar 索引压成一个值。

### 7.2 修改当前 Load

Scalar 已加载后可以直接调整当前索引集合，不需要先 Release collection：

~~~text
旧目标：[A]
新目标：[A,B]
    ↓
QueryCoord 保存更高修改版本
    ↓
QueryNode 保留 A 和字段原始数据，在临时状态中准备 B
    ↓
单个 segment 原子发布 [A,B]
    ↓
QueryNode 上报实际集合
~~~

删除一个已加载 Scalar 索引使用相反方向：先保存不再包含 A 的目标；单个 segment 等正在使用 A 的读结束后，发布新集合，此后该 segment 的自动选择不再使用 A，并回收 A。字段原始数据和其他 indexID 不受影响。

Scalar 索引只改变过滤执行路径，不改变过滤语义，因此不同 segment 在调整期间使用 A、B 或原始扫描仍能正确合并。准备或发布失败时保留该 segment 的旧集合，QueryCoord 继续按最新修改版本收敛。

### 7.3 自动选择链路

~~~text
过滤条件进入 Segcore
    ↓
读取已加载索引支持哪些查询
    ↓
按字段、JSON path、cast 和操作筛选定义级候选
    ↓
结合当前 segment 和查询值排除不能正确执行的索引
    ↓
    ├─ 有候选
    │    → 按稳定规则选择一个索引
    │    → 只准备选中的索引内容并执行
    │    → 选中索引若按当前查询值拒绝执行，则扫描原始字段
    │    → 必要时读取原始字段数据做二次检查
    │    → 输出 bitmap
    │
    └─ 无候选
         → 扫描原始字段
         → 输出 bitmap
~~~

Scalar 字段原始数据不可读表示该 segment 尚未达到当前 Load 目标，不能作为正常可查询 segment 发布；它不是查询执行时的普通分支。

不同过滤条件可以选择不同索引：

~~~text
age > 20       → 支持范围条件的已加载索引
age IN (...)   → 支持 IN 条件的已加载索引
name LIKE ...  → 支持该 LIKE 形式和查询值的已加载索引
~~~

各条件继续由现有表达式树完成 AND、OR 和 NOT 合并。多索引选择层不重新实现布尔表达式计算。

TextMatch/PhraseMatch、Primary Key 和 JSON Stats 继续先走现有专用路径。NGRAM 属于用户 indexID 候选，但必须检查 path、查询值和 gram 参数，并保留原始数据二次检查；只有多个 LIKE 条件选择了同一个 indexID 时，才复用该索引的批量执行。

### 7.4 指定索引链路

~~~text
过滤条件指定 index name
    ↓
Proxy 解析为 indexID
    ↓
indexID 绑定到该过滤条件
    ↓
Segcore 只检查该 indexID 与当前 segment、path、cast、操作和查询值是否匹配
    ↓
    ├─ 可用
    │    → 执行指定索引
    │    → 必要时完成原始数据二次检查
    │
    └─ 不可用
         → 扫描原始字段
         → 不尝试其他 indexID
~~~

同一查询中未指定索引的其他过滤条件继续自动选择。指定 indexID 不会静默换成另一个 indexID；该索引在某个 sealed segment 缺失、与查询值不兼容或运行时拒绝执行时，使用该字段的原始数据扫描。

因此显式 indexID 表示优先使用的加速路径，不是“所有 sealed segment 必须物理使用该索引”的保证。主方案不为 Scalar 增加全局可查询集合。

### 7.5 原始字段数据

Scalar 多索引下，原始字段数据是字段级独立能力：

~~~text
Scalar 字段
  ├─ 原始字段数据
  ├─ indexID A
  ├─ indexID B
  └─ indexID C
~~~

上层只依赖“字段当前是否有可读路径”，不依赖具体保存方式。可读路径可以来自内存、mmap、本地缓存或远端读取。

这个目标必须由字段自己的读取路径满足。某个 indexID 自带 raw data 不能替代字段原始数据目标，否则删除或切换该索引仍会破坏 retrieve、group-by、字段间比较和无索引扫描。非过滤值读取不能从同字段多个索引中任取一个反查。

原始字段数据服务于：

- 没有可用索引时的扫描。
- NGRAM 等索引的二次检查。
- 索引未覆盖的 Scalar 操作。
- 其他需要字段值的查询路径。

原始字段数据进入 Load 目标和 Segment 实际状态。QueryCoord 负责要求该字段可读，QueryNode 上报是否已经具备读取路径；重启、Reopen 和索引删除都不能破坏这条独立状态。

Scalar 资源按“字段数据一份 + 所有目标索引”计算。相对当前允许用索引 raw data 代替字段数据的路径，部分字段可能接近同时保留两份主要数据结构；这是 Scalar 多索引主方案的明确资源代价。

### 7.6 Growing Scalar

持久化 Scalar 索引主要服务 sealed segment。

- Sealed segment 从已加载索引中选择执行路径。
- Growing Segment 继续使用原始字段扫描或已有的临时路径。
- Sealed 和 Growing 的条件结果继续由现有表达式执行链合并。
- 查询指定的持久化 indexID 不要求 Growing 构造同一份物理索引。

### 7.7 成本估算扩展

第一版复用现有单索引已经提供的“能执行或拒绝执行”判断，包括部分索引按当前查询值判断是否值得执行。它不建立多个候选之间的统一成本比较。

多索引选择层先回答：

- 索引是否支持本次查询语义和操作。
- 索引是否需要原始数据。
- 索引是否已经可用。

筛选时只读取每个 indexID 的运行描述，不为比较候选而打开全部冷索引。按稳定规则选定一个索引后才准备其内容；如果这个索引随后拒绝当前查询值，直接走原始字段扫描，不继续打开其他冷候选。

以后每种索引可以增加可选估算：

- 预计命中行数，允许未知。
- 预计执行工作量，允许未知。

估算由 Segcore 对当前过滤条件发起，QueryNode 提供本地运行状态。

未来信息分为三层：

- segment、fieldID：字段行数、值分布、字符串大小等同字段共享信息。
- segment、indexID：索引大小、词项数量和索引独有摘要。
- QueryNode 运行状态：内存、mmap、缓存和远端读取状态。

要把“能正确执行”升级为“尽量选择更快的索引”，需要新增一条完整信息链：

~~~text
索引构建时生成简短统计
    ↓
统计随 segment、fieldID 或 indexID 保存
    ↓
QueryNode 在不打开全部冷索引的情况下读取统计
    ↓
Segcore 让每种候选索引返回估算结果
    ↓
用实际执行数据校准估算规则
~~~

当前 Milvus 没有覆盖所有 Scalar 索引的统一统计和成本模型，因此这不是在选择函数里增加一个排序规则就能完成。候选筛选可以先做；稳定的成本估算需要同时调整索引构建、统计保存、QueryNode 本地状态和 Segcore 选择层，适合作为独立预研。

估算结果只改变性能路径，不能改变查询正确性。没有估算结果时继续使用稳定规则。

### 7.8 查询计划查看（可选增强）

可以参考 MySQL `EXPLAIN`，预研一个可选的查询计划查看能力，但它不是多索引主方案的前置条件。

Milvus 的 Scalar 索引选择以“过滤条件 + segment”为单位，同一查询可能出现：

- 一部分 segment 使用 indexID A。
- 一部分 segment 因索引缺失或运行时拒绝而扫描原始字段。
- TextMatch、Primary Key 或 JSON Stats 走自己的专用路径。

因此不能只返回一个“本查询使用 index A”。更准确的输出分为两层：

- 执行前计划：每个过滤条件的指定 indexID、候选范围和预期选择规则。
- 实际执行观察：最终成功执行中观察到的 indexID、原始扫描、专用路径和回退原因。

自动模式的最终 indexID 由 Segcore 按具体 segment 决定，执行前无法准确知道。执行前计划只能展示定义级候选和稳定规则；如果要提前得到逐 segment 结果，需要先向各 QueryNode 做一次分布式预检查，不属于轻量能力。

~~~text
显式开启计划分析的请求进入 Segcore
    ↓
记录最终成功 attempt 中实际执行的路径
    ↓
QueryNode 与查询入口按 segmentID 去重并汇总
~~~

精确汇总还要区分条件短路、缓存命中、未执行和查询重试，不能简单累加每次观察。该记录只在用户明确开启计划分析时生效；普通查询不做逐条件记录，也不附带跨节点结果。

逻辑执行前计划改动相对较小；准确的实际执行观察需要跨 QueryNode 汇总、去重并控制返回大小，改动更大。预研只需验证信息能否正确向上汇总，不要求第一版增加新的用户命令。

同一入口以后也可以显示 Vector 的目标 indexID，以及观察到哪些 sealed segment 使用物理索引或原始向量。

参考：[MySQL EXPLAIN](https://dev.mysql.com/doc/refman/8.4/en/explain.html)。

## 8. Vector 主方案

### 8.1 初次 Load

~~~text
Proxy 解析请求中明确指定的 Vector indexID
    ↓
QueryCoord 用当前 Load 或 collection 默认补齐未指定字段
    ↓
QueryCoord 保存当前 Load 目标
    ↓
只读取和下发该 indexID 的索引信息
    ↓
QueryNode / Segcore 加载一个 Vector 索引
    ↓
QueryNode 上报实际 indexID
    ↓
QueryCoord 比较目标和实际状态
~~~

稳定状态下，所有 segment 使用当前 Load 目标对应的唯一 Vector 索引。

已构建但未加载的其他 Vector 索引只占用对象存储和构建元数据，不占用 QueryNode 稳定运行资源。

### 8.2 修改当前 Load

从索引 A 切换到 B：

~~~text
QueryCoord 读取 A、B 定义，确认 metric、打分含义和查询参数规则兼容
    ↓
保存新的目标 B 和更高修改版本
    ↓
QueryCoord 只获取 B 的定义和 segment 文件
    ↓
对仍使用 A 的 segment 下发 Reopen
    ↓
该 segment 继续用 A 服务，同时在临时状态中准备 B
    ↓
确认请求仍属于当前修改版本
    ↓
短暂阻止该 segment 接收新读；新读由 QueryNode 内部重试
    ↓
等待正在使用 A 的读完成
    ↓
原子发布 B，并恢复新读
    ↓
回收 A
    ↓
上报实际 indexID
    ↓
QueryCoord 持续补齐未完成副本
~~~

准备 B、资源检查或等待旧读失败时不发布新状态，旧索引 A 继续服务。

如果上一次替换尚未完成又收到新的目标，兼容检查必须覆盖当前仍可能被 segment 使用的全部索引，而不能只比较“旧目标”和“新目标”。新修改版本仍会取消旧任务，但过渡期可能同时看到多个历史索引，它们都必须能执行同一份 Search 请求。

这条路径能复用现有 Reopen 的单索引槽，但每个 segment 发布时会有一个很短的新读重试窗口。它避免 collection 级 Drop/Load，不等于“切换期间任何请求都不等待”。如果要求连这个短窗口也没有，需要进入 Vector 多加载扩展：先同时提供 A、B，再把新查询切到 B。

### 8.3 渐进在线替换

Vector 主方案通过逐 segment Reopen 完成在线替换。

该路径要求 A、B 能共同执行同一份查询：

- metric 和打分含义一致。
- 同一份查询参数在 A、B 上含义一致。
- refine 和 raw data 能力兼容。

主方案不按每个 segment 的实际索引类型转换查询参数。因此，最直接可支持的组合是“相同 index type、不同构建参数”。不同 index type 只有在明确证明查询参数规则相同时才能加入允许列表；否则需要让查询计划知道具体 indexID，并归入 Vector 多加载和查询选择扩展。

过渡期可能有部分 segment 使用 A、部分 segment 使用 B。该路径不需要多个 Vector 索引长期同时驻留；正在切换的 segment 只在准备阶段短暂同时占用 A、B 的资源，B 发布前 A 仍是唯一可查询索引。

### 8.4 查询路径

~~~text
Proxy 从 Search 请求生成 metric 和查询参数
    ↓
生成不绑定单一物理 indexID 的主方案查询计划
    ↓
每个 sealed segment 校验请求能用于自己当前发布的索引对象，并执行查询
    ↓
Growing 使用相同 metric 和打分含义的执行路径
    ↓
合并结果
~~~

稳定状态下，各 segment 的物理索引与当前 Load 目标相同。渐进切换期间，部分 segment 可以仍使用 A，部分已经使用 B；查询绑定的是两者共同的查询含义，而不是单一物理 indexID。

主方案的有效索引选择发生在 Load 或修改当前 Load 时。Search 不能访问其他已构建但未加载的索引。

## 9. Vector 多加载扩展

Vector 多加载是独立架构扩展，不是主方案内部的简单开关。

### 9.1 Load 多个索引

~~~text
QueryCoord 保存目标集合 [A,B]、默认查询 A 和原始向量可读目标
    ↓
按 segment、indexID 分别下发加载
    ↓
QueryNode / Segcore 准备原始向量，并独立准备 A、B
    ↓
实际状态按 indexID 上报
    ↓
QueryCoord 比较目标集合和实际集合
~~~

即使没有开放查询指定，查询也必须在开始时使用当前 Load 的默认查询 indexID，不能依赖容器顺序取得任意索引。

一个 indexID 只有在所有目标 segment 和所有服务副本上都能通过该物理索引或原始向量执行对应查询含义后，才进入“可查询 indexID 集合”。显式选择、默认查询和动态选择都只能使用该集合中的索引。

### 9.2 运行时调整

Primary 写入的期望配置按 DDL 顺序沿 CDC 复制：

~~~text
期望 1：[A]，默认 A
    ↓
期望 2：[A,B]，默认 A
    ↓
期望 3：[A,B]，默认 B
    ↓
期望 4：[B]，默认 B
~~~

Primary 和 Secondary 都接收相同的期望配置，但分别维护本地过渡状态：

~~~text
本地过渡状态
  ├─ 最新期望配置版本
  ├─ 实际已加载集合
  ├─ 当前生效的查询默认
  ├─ 可查询 indexID 集合
  └─ 排空中的旧 indexID
~~~

即使最新复制目标已经是 `[B]`，某个集群仍可以在自己的过渡状态中保留 A，直到本集群完成查询视图切换和 A 请求排空。复制目标不能直接命令 Secondary 立即回收 A。

QueryNode 和 Segcore 的加载、删除、refine、raw data、资源和恢复都必须按 indexID 工作。

### 9.3 查询级无停机切换

~~~text
本集群收到期望目标 B 和默认 B
    ↓
A 继续服务；本集群准备 B 和原始向量路径
    ↓
B 在本集群所有目标 segment 和服务副本上都能通过物理索引或原始向量执行
    ↓
发布更高版本的查询视图：B 可查询、默认 B、A 排空中
    ↓
所有查询入口确认不再产生新的 A 查询
    ↓
新查询固定使用 B；已经带着 A 的请求继续执行
    ↓
等待 A 请求从网络、队列和执行路径全部结束
    ↓
本集群实际状态收敛为 [B]，回收 A
    ↓
尚未完成的 B 物理索引继续构建和加载，完成后替代原始向量搜索
~~~

该切换不要求所有机器在同一瞬间具备 B 的物理索引，但要求一次查询固定同一个目标 indexID 和查询含义。缺少 B 的 segment 使用原始向量，不改用 A。

只有本集群所有查询入口都确认使用了新的查询视图后，才能开始判断 A 已经排空；否则旧入口仍可能继续产生新的 A 请求。Primary 和 Secondary 分别完成这一步，不互相复制确认或排空进度。

### 9.4 Growing Segment

Growing Segment 没有 sealed segment 的持久索引文件。

查询固定同一个目标 indexID 和查询含义；缺少该物理索引的 sealed segment 使用原始向量。Growing 使用能保持相同查询含义的临时索引或原始向量搜索。

原始向量搜索必须保持相同 metric、topK / range 条件、过滤条件和打分顺序。只对具体物理索引有意义的查询调优参数，在原始向量路径中不生效；它们不能改变用户可见的查询含义。

查询级无停机切换时，Growing 必须能够同时处理携带 A 和 B 查询含义的请求。

### 9.5 资源成本

Vector 多加载需要：

- 按 A+B+…核算稳定资源。
- 单独核算字段原始向量读取能力，不能假设由某个 Vector 索引提供。
- 计算 Load 临时内存峰值。
- 计算本地磁盘、mmap、缓存、GPU 和 DiskANN 资源。
- 在默认切换后继续保留旧索引，直到旧查询排空。
- 管理多个 segment 并行加载造成的 collection 总峰值。

该扩展的主要改造位于 Segcore 多索引状态、Growing 查询含义、旧查询排空和资源管理，而不是索引定义元数据。

## 10. Vector 查询选择扩展

### 10.1 默认查询与显式指定

进入 Vector 多加载查询阶段后：

- QueryCoord 先向查询入口发布带版本的默认查询 indexID 和可查询集合。
- Search 未指定索引时，查询入口使用该视图中的默认查询 indexID；默认处于“未设置”时拒绝查询。
- Search 指定索引时，查询入口只能从该视图的可查询集合中选择明确 indexID。
- 查询入口还要用查询视图中的参数规则检查本次 Search；原始向量路径必须能表达本次 metric、topK / range 和过滤条件，不能只因为 B 在 collection 级可查询集合中就无条件放行。
- 查询入口把选定的 indexID 和视图版本放入 Search 执行链路。
- 查询开始后固定同一目标 indexID；已有物理索引的 sealed segment 使用该索引，缺少索引的 segment 使用原始向量。
- Growing 使用保持相同查询含义的兼容路径。

这条查询规则不扩大 Vector 主方案。主方案每字段只加载一个索引，Search 无法选择其他未加载索引。

### 10.2 动态选择扩展

动态选择建立在 Vector 多加载和查询携带 indexID 之上。

选择链路分为两层：

~~~text
第一层：筛出能正确执行的候选
    ↓
第二层：在候选中估算更合适的索引
    ↓
    ├─ 得到结果 → 查询开始时固定一个 indexID
    │
    └─ 结果未知 → 使用当前 Load 的默认查询 indexID
                    ├─ 默认已设置 → 固定该 indexID
                    └─ 默认未设置 → 拒绝查询
~~~

第一层使用索引定义和加载状态：

- metric 和打分含义。
- 查询参数支持范围。
- segment 上的准备状态。
- refine 和 raw data 条件。
- Growing 是否能保持相同查询含义。

第二层需要额外信息：

- nq、topK 和 Search 参数。
- 提前估算的过滤比例。
- segment 数量和大小。
- 索引类型、构建参数和离线基准。
- mmap、缓存、本地文件和硬件状态快照。

查询开始前通常无法取得每个 segment 的精确过滤结果，也无法取得所有 QueryNode 的完全实时资源状态。第一版估算只能使用提前统计和有延迟的状态快照。

如果要先收集精确过滤结果和实时节点负载，再选择 indexID，需要增加“先收集、再选择、再执行”的两段查询流程。这属于独立的大型查询架构调整。

动态选择结果允许返回未知；未知时回到当前 Load 的默认查询 indexID。如果默认也处于“未设置”，则拒绝查询，不能临时选择第一个 indexID。

## 11. 状态变化与恢复

### 11.1 Reopen

Vector 主方案的单个 segment Reopen 流程：

~~~text
A 继续服务
    ↓
创建临时状态并准备 B
    ↓
确认当前修改版本仍然要求 B
    ↓
阻止该 segment 的新读；新读在 QueryNode 内部重试
    ↓
等待正在使用 A 的读完成
    ↓
原子发布 B
    ↓
恢复新读并回收 A
~~~

下载、打开、资源检查或准备失败时不进入读切换窗口，旧状态继续服务。等待旧读失败或超时时也不发布 B。

Vector 多加载扩展使用不同顺序：A、B 已经同时可查询后，先让新查询固定使用 B，旧查询继续使用 A；A 的查询全部结束后再回收 A。这个流程不能与主方案的单索引槽 Reopen 混写。

### 11.2 QueryCoord 重启

~~~text
读取 collection 默认
    ↓
读取当前 Load 状态和修改版本
    ↓
读取默认查询 indexID 和正在排空的旧索引
    ↓
与仍有效的索引定义核对
    ↓
收集 QueryNode 实际状态
    ↓
确认定义读取成功后，删除已经不存在的引用
    ↓
重新生成缺失的 Load、Reopen 或释放动作
    ↓
重新发布最新查询视图，并继续未完成的入口确认和旧索引排空
~~~

QueryCoord 重启不把当前 Load 退回 collection 默认。

如果持久状态是“当前未加载”，重启后只保留 collection 默认，不自动发起 Load。

如果 DataCoord 暂时不可用、读取超时或没有返回完整结果，QueryCoord 保留默认和当前 Load 状态，并暂停本轮收敛；只有成功取得完整有效索引集合且确认 indexID 已不存在时，才能清理引用。

### 11.3 QueryNode 重启

QueryNode 的实际索引状态丢失后，由 QueryCoord 根据当前已加载目标重新加载；“当前未加载”的 collection 不恢复索引。

- Scalar 恢复目标 indexID 集合。
- Scalar 同时恢复字段原始数据的可读路径。
- Vector 主方案恢复当前唯一 indexID。
- Vector 多加载扩展恢复目标集合、默认查询 indexID、正在排空的旧索引和原始向量读取能力。

### 11.4 Drop Index

Drop Index 删除权威索引定义。

删除后的收敛链路必须是可靠状态链，而不是只依赖一次内存通知：

~~~text
DataCoord 确认 indexID 已删除
    ↓
通知 QueryCoord 和查询入口受影响的 indexID
    ↓
QueryCoord 从默认规则、当前 Load 和可查询集合中移除引用；查询入口刷新名称映射
    ↓
QueryNode / Segcore 拒绝新的旧 indexID 使用
    ↓
QueryNode 排空已有读并回收实际索引
~~~

通知用于尽快生效。QueryCoord 在重启和定期检查时还要从 DataCoord 读取完整的有效索引集合，补齐丢失的通知。只有读取成功并确认 indexID 不存在时才能执行清理；读取失败时保留原状态并重试。

如果被删除的 indexID 位于默认加载集合中，只移除该成员；如果因此没有默认加载索引，或被删除的是 Vector 默认查询 indexID，就保存“默认未设置”。系统不能按索引列表顺序静默选择替代项。外部控制面可以在 Drop 前先在 Primary 写入新的默认规则并沿 CDC 复制，避免之后出现没有默认的状态。

本地查询状态必须向删除结果收敛：

- Collection 默认不能继续引用已删除 indexID。
- 当前 Load 目标不能继续引用已删除 indexID。
- 新查询不能再生成使用该 indexID 的计划。
- 已开始查询可以排空。
- QueryNode 随后回收实际索引。
- Scalar 索引删除不直接破坏字段原始数据的可读状态。
- 旧 Reopen 任务不能重新加载已删除索引。
- 重启时不能从旧 Load 状态恢复已删除索引。

如果删除的是当前唯一可用索引，本地可用性处理属于上层规则，不改变上述架构不变量。

## 12. CDC 架构

### 12.1 复制状态

CDC 复制：

- Create Index。
- Alter Index。
- Drop Index。
- Collection 默认 Load 规则的修改。
- LoadCollection、LoadPartitions、ReleaseCollection 和 ReleasePartitions 对应的 AlterLoadConfig / DropLoadConfig。
- Collection 数据变化。

Load/Release 是可复制的查询配置 DDL。Load 中明确选择的 indexID、partition 和 field 到达 Secondary 后保持相同；replica 与 resource group 继续服从现有本地 replica 配置开关。

CDC 不复制：

- Search / Query 请求。
- 单次查询最终选择的 Scalar indexID、Vector indexID 或原始扫描路径。
- QueryNode 实际加载到哪些节点和 segment、加载进度、缓存与资源状态。
- QueryCoord 根据本集群实际状态生成的可查询集合、当前生效默认、查询视图版本、排空中 indexID 和入口确认。

因此两端同步的是“期望加载什么”，不是“某次查询实际怎么执行”。Secondary 根据复制来的目标在自己的 QueryNode 上收敛，完成时间和实际资源状态可以与 Primary 不同。

### 12.2 Primary 与 Secondary

~~~text
Primary 索引、默认规则和 Load/Release DDL
    ↓ CDC
Secondary 接收相同的期望配置

Primary Query 集群：本地实际加载状态和本地查询
Secondary Query 集群：独立实际加载状态和本地查询
~~~

两端期望使用相同 partition、field 和 indexID，但各自完成加载、生成查询视图和执行 Search。单次查询可以在两端选择不同执行路径，因为查询请求本身不复制。

### 12.3 Replicated Drop

~~~text
Primary 产生 Drop
    ↓
Drop 到达 Secondary
    ↓
Secondary 接受权威删除
    ↓
清理本地默认和 Load 引用
    ↓
停止新查询使用该 indexID
    ↓
排空旧查询并回收索引
~~~

Secondary 不重新执行本地 loaded 校验，也不否决已经复制到本地的 Drop。

外部平台可以在 Primary Drop 前先提交新的 Load 选择，等待该 DDL 复制并让 Secondary 收敛，以降低查询影响；但这不是保证元数据正确的架构前提。

## 13. 滚动升级与启用前回滚

### 13.1 升级流程

~~~text
Secondary 部署新代码，功能关闭
    ↓
Primary 部署新代码，功能关闭
    ↓
所有组件继续只读写旧状态
    ↓
此时可以回滚旧版本
~~~

### 13.2 功能启用

功能开关分为以下几类：

- Scalar 多索引开关。
- Vector 主方案开关，只启用“同字段多定义、每次 Load 单索引和 Reopen 替换”。
- Vector 多加载与查询选择扩展开关；动态选择如进入实现，再使用独立开关。

同一能力按以下顺序启用：

~~~text
Secondary 先允许读取并应用新的多索引 DDL 状态
    ↓
Primary 再允许写入新状态
    ↓
Primary 为已有 collection 写入明确默认规则，并沿 CDC 复制
    ↓
允许创建同字段第二索引；Scalar 可写多索引 Load 状态，Vector 主方案仍只写单索引 Load 状态
~~~

已有 collection 的初始化规则见 4.2：已 Load 时沿用当前目标，未 Load 时使用升级前唯一有效索引。默认规则初始化完成前，不允许该 collection 写入同字段第二索引、Scalar 多索引 Load 状态或 Vector 多加载扩展状态。

Vector 多加载与查询选择扩展只能在 Vector 主方案稳定启用后再打开。它负责 Vector indexID 集合、查询视图和排空状态，不随 Vector 主方案开关自动启用。

开关覆盖所有可能创建或接受新多索引状态的路径：

- 普通请求入口。
- CDC 回调。
- 恢复路径。
- 导入或恢复元数据路径。
- 后台自动任务。

权威 Drop Index 及其本地清理不能被功能开关拦截。开关可以阻止创建新的多索引状态，但不能让已经删除的 indexID 继续被默认、Load 或查询引用。

### 13.3 回滚边界

- 第一次写入并复制新的默认规则、同字段多索引定义或多索引 Load 状态前，可以回滚旧代码。
- 新状态已经写入后，不保证旧版本能够继续工作。
- 内核不主动查询其他集群版本；升级顺序由 Cloud 或其他外部控制面保证。

## 14. 资源与存储影响

| 资源 | 多定义、只加载一个 | 多个同时加载 |
| --- | --- | --- |
| 对象存储 | 每个 indexID 都有独立文件 | 相同 |
| 构建资源 | 每个 indexID 独立构建 | 相同 |
| QueryNode 内存 | 稳定状态主要为当前索引；Reopen 短暂 A+B | 按 A+B+…核算 |
| 本地磁盘与缓存 | 当前索引和切换临时数据 | 随加载集合扩大 |
| Load 时间 | 只准备当前索引 | 准备整个集合 |
| 临时峰值 | 并行 Reopen 会放大峰值 | 可能高于最终 A+B |
| 原始数据 | Scalar 始终单独保留字段读取路径，不能由某个索引的 raw data 替代；Vector 主方案按当前索引能力保留 | Vector 多加载必须独立保留原始向量读取能力；多索引共享，不为每个索引复制一份 |

逻辑资源需求与实际物理占用分开。mmap、lazy load 和缓存淘汰会改变实际驻留页，但不能改变是否有足够资源完成目标 Load 的判断。

## 15. 预研验证方案

### 15.1 共同状态模型

验证：

- 同字段多个 indexID 独立创建、构建、恢复和删除。
- Collection 默认、当前 Load 目标和实际状态互相独立。
- 所有 Load、Reopen 和释放动作按 indexID 工作。
- 旧任务不能覆盖新目标。
- 已删除 indexID 不能被恢复。

输出：

- 组件状态边界。
- 需要调整的现有字段级检查清单。
- 重启和任务晚到的状态收敛证明。

### 15.2 Scalar 多加载

验证：

- 同字段多个 Scalar 索引同时驻留。
- 已加载集合可以按 [A] → [A,B] → [B] 调整，失败时旧集合继续服务。
- 每个过滤条件独立选择索引。
- 显式 indexID 不可用时回到原始扫描，并且不会换用另一个 indexID。
- JSON path、cast、NGRAM 查询值检查和二次检查语义正确。
- 候选筛选不需要先加载全部冷索引。
- 原始字段数据始终可读。
- TextMatch、Primary Key 和 JSON Stats 专用路径不被普通多索引选择改变。
- 多条件 bitmap 继续由现有表达式树合并。

输出：

- Scalar 查询选择架构。
- 各索引类型的能力描述。
- 多索引和原始字段数据的资源模型。
- 可选的查询计划与实际执行摘要原型。

### 15.3 Vector 主方案

验证：

- Load 只下发目标 Vector indexID。
- 当前 Load 从 A 修改到 B 后能够持续收敛。
- Reopen 失败时 A 继续服务。
- 旧任务不会覆盖新目标。
- 重启后继续当前 Load 选择。
- 兼容索引渐进替换期间的资源峰值和查询影响。

输出：

- Vector 单加载完整链路。
- 在线替换状态流。
- 兼容条件和资源数据。

### 15.4 Vector 多加载扩展

验证：

- 单个 sealed segment 同时保存 A、B。
- [A] → [A,B] → [B] 全部按 indexID 表达。
- 查询开始时固定 indexID。
- 查询入口接收带版本的查询视图，并在查询级无停机切换时完成版本确认。
- 查询级无停机切换期间旧查询排空。
- Growing 能保持对应查询含义。
- 新 sealed segment 先用原始向量、随后切到物理索引时的可见性和资源影响。
- “已经可以查询”和“物理索引全部完成”两种状态能够独立收敛。
- A+B 稳定资源与 Load 临时峰值。

输出：

- Vector 多加载是否值得进入正式设计。
- Segcore、Growing 和资源架构的完整影响范围。

### 15.5 Vector 动态选择扩展

验证：

- 正确候选筛选。
- 可选估算返回未知时回到默认查询 indexID。
- nq、topK、segment 大小、过滤比例和硬件对选择的影响。
- 提前统计和有延迟状态快照是否足够稳定。
- 是否需要两段查询流程。

输出：

- 能力检查与成本估算的边界。
- 可以稳定使用的规则。
- 不能在基础多索引功能中承诺的优化目标。

## 16. 组件调整汇总

| 组件 | 主方案调整 | Vector 多加载扩展 |
| --- | --- | --- |
| Proxy | 解析 Load indexID；Scalar 条件绑定 indexID | Search 解析并固定 Vector indexID |
| DataCoord | 同字段多定义和独立构建；按 indexID 提供结果 | 无新的查询职责 |
| QueryCoord | 每字段默认规则、索引和字段数据目标、修改版本与实际状态比较 | Load 目标从单值变为集合，管理默认查询 indexID、可查询集合和排空阶段 |
| QueryNode | 按 indexID 加载、上报并核算索引与字段数据资源 | 同字段多个 Vector 索引长期驻留 |
| Segcore Scalar | 同字段多个索引；按条件选择；原始字段数据独立 | 成本估算作为后续增强 |
| Segcore Vector | 每字段一个索引；Reopen 替换 | 按 indexID 保存多个；查询固定 indexID |
| Growing | 保持当前查询含义的临时索引或原始搜索 | 查询级无停机切换时同时服务 A、B 查询含义 |
| CDC | 复制索引定义、默认规则和 Load/Release 目标；不复制实际加载状态和单次查询选择 | 相同 |

## 17. 核心不变量

- indexID 是索引定义、Load 目标和运行状态的统一身份；Scalar 指定索引和 Vector 多加载查询也使用该身份。
- Vector 主方案渐进切换期间，查询绑定共同查询含义，每个 segment 可以使用自己当前发布的 A 或 B。
- Collection 默认 Load 规则和当前 Load 目标是两份独立状态。
- CDC 复制期望 Load 目标和期望默认；可查询集合、当前生效默认、查询视图和排空进度由各集群独立维护。
- 索引目标、字段数据目标和 QueryNode 实际状态分开。
- QueryCoord 只下发当前 Load 选择的 indexID。
- 运行状态只能向最新修改版本收敛。
- 已删除 indexID 不能被默认、Load、查询或恢复继续引用。
- Reopen 失败不能破坏旧可查询状态。
- Scalar 自动选择以单个过滤条件为单位。
- Scalar 原始字段数据与索引状态分开。
- Vector 主方案每个 segment 对外发布一个 Vector 索引状态。
- Vector 多加载时，只有在全部目标 segment 和服务副本都能通过物理索引或原始向量执行对应查询含义的 indexID 才能被查询选择。
- Vector 查询级无停机切换的旧索引排空阶段必须可以恢复。
- CDC 复制索引定义、默认规则和 Load/Release 目标，不复制 QueryNode 实际状态和单次查询选择。

## 18. 总结

主方案通过统一 indexID、分离索引定义与 Load 状态、让 QueryCoord 管理目标、让 QueryNode 上报实际状态，完成一列多索引的共同基础。

Scalar 在该基础上完成真正的多索引加载和按过滤条件选择。Vector 主方案保持每字段只加载一个索引，通过 Reopen 完成当前索引替换，从而避免立即改造整个 Vector 查询运行时。

Vector 多加载和动态选择都有可实现路径，但核心成本位于 Segcore 多索引状态、查询固定 indexID、Growing 查询含义、旧请求排空和长期资源管理，应作为独立扩展能力设计和验证。
