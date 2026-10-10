# 一列多索引：背景与改造面

## 1. 现状

存储和大部分 Go 代码天然支持多索引；**真正挡路的是 DataCoord 创建校验、Proxy→QueryCoord→WAL 的 Load 单值链路（Proxy 给 QueryCoord 的内部请求、QueryCoord 持久化、WAL 配置都是“字段 -> 一个 indexID”）、C++ segcore 按字段只保存一个索引的地方**。前两段好改，C++ 最难。

- **创建（用户请求 -> Proxy -> DataCoord -> WAL/CDC DDL）**：元数据按“collection -> indexID -> 索引定义”、段索引按“segment -> indexID -> 构建产物”存，**存储不用改**。唯一挡路的是**创建校验**：同字段已有未删除索引就拒绝（JSON 不同 path 例外）。**放开即可**。
- **Load（用户 Load 请求 -> Proxy -> QueryCoord -> WAL Load 配置）**：**第一个真正丢索引的地方**，丢在 **Proxy 给 QueryCoord 发内部 Load 请求这一步**。Proxy 把 DescribeIndex 返回的索引列表填进 LoadCollectionRequest/LoadPartitionsRequest 时，这两个内部请求的字段只支持一个 `fieldID` 一个 `indexID`；**同字段多个就是在这里丢的**。QueryCoord 持久化的 Load 目标同样单值；广播到 WAL 的 Load 配置每个字段也只带一个 indexID。改法：**这三处都改成“字段 -> indexID 列表”**，公开 proto/SDK、Proxy、QueryCoord、WAL 消息都要动，但只是改结构，不难。
- **QueryCoord 的 IndexChecker 和 segment 打包**：IndexChecker 循环比较“collection 全部索引定义”和“每个 segment 实际上报的索引”，缺了就发更新任务；现在的可查前提相当于**“所有索引建完”**。

  这里有一个**决策点**：
  - **A. 所有目标索引建完才加载/可见**（沿用现状，最简单，可见性被最慢索引拖住）；
  - **B. 建好一个加载一个**（要定义 segment 的部分就绪语义）；
  - **C. segment 先可见，缺的索引走 raw 全量，索引完成后再补**（查询不阻塞，但要求就绪状态和 raw 回退配合）。
  
  给 QueryNode 打 segment 加载包时**只装选中的索引**，这一点无论选哪个都要做。
- **QueryNode 的 Go 侧**：已经按 indexID 存段索引、按 indexID 上报实际加载状态，**基本沿用**；只是给 C++ 的 collection 索引元数据补上 indexID，不能再靠 fieldID 去重。这是**最好改的一层**。
- **C++ segcore（最难）**：C++ 里有**两处把一个字段一个索引写死**。一处是 collection 级的索引参数表：按字段只存一份索引参数，搜索准备和 Growing 临时索引都从这份参数取 metric/参数；另一处是 sealed segment 的运行时：一个字段只保存一个 scalar 索引和一个 vector 索引，查表达式时按字段只取那一个。Scalar 多索引要把这两处都改成**“字段 -> 多个索引”**，再加**“按条件选索引”**；JSON 现有的按 path 多索引特例可以当统一底座。**Vector 主线不要求改这两处**。
- **Import/Copy**：已按 buildID 为 key、按 index_name 解析目标 indexID，天然容纳同字段多索引；proto 上的 fieldID 注释是过期说明，**只记录，不修**。

## 2. 机制速览

- **构建与加载是两条链路**：构建侧 DataCoord 按索引定义给每个 sealed segment 建“segment + indexID”的索引文件；加载侧 QueryCoord 决定加载什么，QueryNode 把文件装进 segcore。**多索引主要改加载侧**。
- **IndexChecker 是“期望 vs 实际”对齐器**：它**不发起构建**，只把“目标里有、segment 上没有”的索引变成 segment 更新任务；QueryNode 加载完按 indexID 上报，QueryCoord 把上报存进 segment 状态。
- **QueryNode 分 Go 和 C++ 两半**：Go 负责通信、拉文件、组织 segment/索引，已经按 indexID 组织；C++ segcore 才真正打开索引供查询用。多索引在 Go 侧是**透传**，在 C++ 侧要**改数据结构和执行路径**。
- **Sealed 与 Growing 是两套索引机制**：Sealed 索引是构建好的文件；Growing 持续插入，用 segcore 自己维护的临时索引或原始数据，不加载构建文件。所以 Vector 主线的 Growing 只跟当前选中的索引语义走，**不翻倍**。
- **查询选择发生在 segcore 执行表达式时**：每个过滤条件在 segment 上选一条路径：原始数据、scalar 索引、NGRAM、JSON 索引/统计等。多 scalar 索引要求选择发生在**每个条件上，而不是字段上**，这是 C++ 改动最大的原因。
- **WAL/CDC 只同步 DDL，不同步加载进度**：Create/Drop Index 和 Load 配置走控制通道并复制到备集群；QueryNode 实际上报和单次查询选索引是本地状态。

## 3. 共同改造

**总原则**：**indexID 是唯一身份**；所有“字段 -> 一个索引”改成“字段 -> 索引列表”；**默认索引、本次 Load 目标、实际加载状态、单次查询选择分开存**，QueryCoord 存目标，QueryNode 报实际。

- **DataCoord**：**放开创建校验**；Create/Drop DDL 继续走 WAL/CDC。
- **Proxy + 公开 Load 协议 + WAL Load 配置**：这三处都从**“字段 -> 一个 indexID”改成“字段 -> indexID 列表”**；**Scalar 可多个，Vector 只允许一个**；未指定时由 Proxy 从 collection 默认索引取并显式传给 QueryCoord。
- **QueryCoord**：**Load 目标持久化列表化**，恢复按列表；释放只结束当前 Load，保留默认索引和已释放目标；IndexChecker 和 segment 打包**只认当前 Load 目标**。
- **QueryNode Go**：沿用按 indexID 的存储和上报；传给 C++ 的 collection 索引元数据带 indexID；资源估算按**实际加载的索引数量**计算。
- **C++**：collection 级索引元数据从“字段 -> 一个”改成“字段 -> 多个”，这是**两条路共用底座**；Scalar 的 segment 运行时改动在下一节。
- **CDC / 多集群**：DDL 复制 Load 目标列表；**实际加载进度和查询选择是本地状态，不复制**。
- **升级回滚**：**Scalar 和 Vector 两个独立开关**；先升级所有节点、开关关闭，再按需打开；关闭时走旧单值路径，**启用前回滚无需迁移数据**。

## 4. Scalar：同字段多索引同时加载

**现状**：**创建被入口拒绝**；Load 在 Proxy 给 QueryCoord 的内部请求处被压成“字段 -> 一个 indexID”，QueryCoord 持久化和 WAL 同样单值；C++ segment 每个字段只存一个 scalar 索引，表达式按字段取。

**改造**：

1. **C++ collection 索引元数据改列表**，与 Vector 共用底座。
2. segment 的 scalar 容器从“字段 -> 一个”改成**“字段 -> 索引列表”**，把 JSON 按 path 多索引的特例统一进去，不再开 Scalar 专用例外。
3. **每个过滤条件独立选索引**：按“算子 + 字段类型 + JSON path/cast”找语义兼容的候选。

   候选/指定索引在部分 segment 不可用时，**决策点**：
   - **A. 查询报错**：语义最强，但要求知道所有 segment 的索引就绪状态，实现最重；
   - **B. 自动换另一个 indexID**：查询不失败，但可能换到语义不同的索引，结果含义会变；
   - **C. 退回 raw 全量扫描**：最简单，但“指定索引”只是加速提示，不是物理保证。

   此前倾向 C，且不自动换索引；是否做成本选优是另一个决策，见第 7 节。
4. **原始数据是否始终保留**。

   **决策点**：
   - **A. 始终保留独立可读路径**：切换、回退和 Drop 都简单，但常驻资源更多；
   - **B. 按“已加载索引是否提供 raw data”决定释放**：资源更省，但切换、Drop 和回退都会变复杂。

   此前倾向 A。

**成本**：**难点在 C++ 三处联动**——collection 索引元数据、segment 容器、表达式选择路径。QueryNode 内存/磁盘随加载的索引数线性增长；单条查询仍只用一个索引，**查询 CPU 不翻倍**。

## 5. Vector：多索引构建，一次 Load 只加载一个

**现状**：**同字段只能建一个**；segment 里**一个字段只保存一个 vector 索引**；搜索直接用它。

**改造**：

1. 构建：共享创建放开；每个索引对每个 sealed segment 独立构建。**构建成本线性增加，QueryNode 常驻内存不变**。
2. Load：共享列表协议，但 **Vector 字段只允许一个索引名**。
3. 运行时：**每个字段仍只保存一个 vector 索引**，搜索路径不改成按 indexID 携带。
4. 切换 A -> B：QueryCoord 改当前 Load 目标；IndexChecker 对缺 B 的 segment 发更新任务；QueryNode 用已有的替换式加载换掉 A。

   **决策点**：
   - **A. 只承诺语义兼容时逐 segment 渐进**：不兼容走 release/reload，替换机制够用，但切换中有混合窗口；
   - **B. 任意 A->B 都无停机**：要双加载、查询固定 indexID、请求排空。

   此前倾向 A。切换峰值内存短暂升高。
5. Growing：只按当前索引语义，用临时索引或原始向量。

   新 segment 缺目标索引时，**决策点**：
   - **A. 等索引建完再可查**：IndexChecker 语义简单，但可见性被最慢的 segment 拖住；
   - **B. 先 raw 可查，索引就绪后补齐**：数据更早可见，但要允许“未就绪但可查”。

   此前倾向 B，需确认；它决定 IndexChecker 的就绪模型。

**偏离主线**：同一字段同时加载多个 Vector 索引，要把 segment 改成**按 indexID 保存多个 vector 索引**，Search 请求带 indexID，资源、就绪上报、Growing 全部按索引粒度管理，**等于重做 Vector 运行时**；动态自动选索引还要统计和成本模型。**主方案只铺底座，不做这些**。

---

## 6. 放到两个具体需求里看

### 6.1 Scalar：varchar 各种表达式算子尽量走索引

需求不是让一个索引支持所有算子，而是**同一字段挂多个索引，每个覆盖一部分算子**。典型组合：精确匹配、IN、范围比较走 inverted/bitmap；LIKE、前后缀、正则走 NGRAM；全文匹配仍走独立文本索引通道。

**主线能覆盖，但要补“按条件选索引”模块。** 现在执行器是“按字段拿一个索引，再问它能不能处理当前算子，不能就退回原始扫描”；多索引后倒过来：

1. **每个已加载索引带能力描述**：索引类型、参数、适用算子和类型，不能靠零散的类型判断。
2. **每个条件叶子先选索引再执行**：按“算子 + 字段类型 + 字面量形态”匹配；匹配不到退回原始扫描。
3. **匹配规则集中定义**：例如 NGRAM 只处理它声明支持的字符串模式，inverted 声明支持等值/范围；不能到处写 if。
4. **选择在 QueryNode 执行时做**：看这个 segment 实际加载了什么，而不是只看 QueryCoord 目标；单个索引缺 segment 时先原始扫描，不阻塞查询。
5. **多个候选时**：第一版按固定优先级挑一个，不比较成本；以后把优先级换成本估算。

满足程度：对“等值走 inverted、LIKE 走 NGRAM”**覆盖度很高**。没有对应索引类型的算子仍原始扫描；第一版不保证最快，只保证**“尽可能走索引”**。

### 6.2 Scalar：JSON 按路径覆盖索引

JSON 已有“同一字段按不同 path 建多个索引”的特例，运行时也有按 path 找索引的专用通道。主线把这个特例收编成普通能力。

需要补三处：

1. **决策点**：JSON 同一 path 是否允许多 cast 或多索引类型。

   - **A. 只允许多 path**：现有“按 path 判重”基本够用，改动集中在 Load 目标和查询选择；
   - **B. 允许同一 path 多 cast/多索引类型**：创建入口的判重身份要放开到 path + cast + index_type，运行时选择键也要跟着扩展。
2. **Load 目标按索引列表表达**：一个 JSON 字段加载 path A/B/C 三个索引，正好用“字段 -> 索引列表”；检查、打包、上报都按列表，不再像现在 Proxy 填内部 Load 请求时那样只留一个 indexID。
3. **查询选择键是 path + cast + 算子**：精确匹配才算候选，模糊或动态 path 一律原始扫描；带数组下标、通配的匹配规则要单独定，否则“覆盖”会变成误用索引。

满足程度：不同 path 覆盖满足度高，因为创建和运行时基础已存在，主要补 Load 目标模型和显式选择。同一 path 多 cast/多索引类型要先放开创建校验，并统一运行时容器。查询里临时出现、没建索引的 path 只能原始扫描，这是需求边界。

### 6.3 Vector：从内存 HNSW 降级到 DiskANN

这个需求落在主线内：同一字段建 HNSW 和 DiskANN，**平时 Load HNSW，内存紧张时把 Load 目标改成 DiskANN**，segment 逐个替换。**不需要多 Vector 驻留。**

降级场景特有的四点：

1. **资源类型跟着索引走**：HNSW 占内存，DiskANN 占本地磁盘和 mmap/磁盘缓存。切换时资源估算和配额按新索引算，加载完成后释放旧 HNSW 内存。
2. **切换峰值按内存 + 磁盘同时计费**：一个 segment 在替换窗口会短暂同时持有 HNSW 内存和 DiskANN 磁盘文件。
3. **查询参数按索引类型处理**：HNSW 用 ef，DiskANN 用 search_list。混合窗口里不同 segment 用不同索引，请求参数必须两边都能解释，或节点按当前 segment 实际索引取参数，否则一半 segment 报参数错。
4. **本地磁盘生命周期按 indexID 管理**：来回切换时不能漏删旧文件缓存，也不能在替换完成前删新文件；释放、失败、回滚都以 indexID 为准。

满足程度：同 metric 的 HNSW -> DiskANN 降级，**主线满足度很高**，核心是**目标切换 + 替换加载 + 资源切换**；真正要补的是第 1、3 点。

切换时某 segment 的 DiskANN 还没建完，**决策点**：

- **A. 等索引建好再切**：IndexChecker 语义更简单，但降级切换会被最慢的 segment 拖住；
- **B. 先走原始向量，不等待**：数据更早可查，但要允许“没有目标索引但可查”。

此前倾向 B。

## 7. 影响架构走向的决策边界

以下是需要领导拍板的决策点；纯用户表现差异不列。“此前倾向”是讨论时暂定的方向，不是最终结论。

1. **Vector 是单加载还是多加载。** 当前主线按每字段只加载一个。如果要求多个 Vector 索引同时可查，QueryNode/segcore 的 vector 运行时、Search 请求、就绪上报和资源模型都要按 indexID 重做。
2. **Vector 切换是逐 segment 渐进，还是全集群原子无停机。** 当前按渐进替换。原子无停机必须依赖多加载、查询固定 indexID 和请求排空，是另一套数据流。
3. **目标索引没建好的 segment 怎么处理。** 三个选项：等所有目标索引建完才加载/可见；建好一个加载一个；segment 先可见、缺索引走 raw 全量。此前倾向第三项；选择决定 IndexChecker 和 QueryNode 的就绪模型。
4. **Scalar 索引选择按字段还是按过滤条件。** 当前按每个条件叶子选择。若只按字段选择，查询计划和协议改动小很多，但同一字段的不同条件无法各用各的索引。
5. **Scalar 显式指定查询索引是否进公开协议。** 当前只做自动选择，内部预留 indexID。如果公开指定，要新增表达式语法、SDK 和计划绑定，改动比加载侧更大。
6. **Scalar 第一版是否要求成本选优。** 当前只选能正确执行的索引。如果要求最快，需要索引统计、元数据和 compaction 联动、成本模型，是一条独立数据流。
7. **Vector 多索引是否允许不同 metric。** 当前允许“metric + 索引”成对构建。若只允许同 metric，切换兼容判定和默认索引语义简单很多；允许不同 metric 时，查询计划和切换必须显式处理距离含义变化。
8. **DropIndex 的跨集群一致性。** 当前由外部平台协调顺序，内核只查本地引用。若内核要保证，Drop/CDC 必须与本地加载状态耦合，数据流完全不同。
9. **CDC 复制到什么粒度。** 当前复制索引 DDL 和 Load/Release 目标，实际加载进度与查询选择是本地状态。若要求进度也跨集群一致，QueryNode 上报要进入复制流，恢复模型重做。
10. **Scalar 缺索引或显式索引不可用时怎么兜底。** 报错、自动换另一个 indexID、raw 扫描三选一；此前倾向 raw 扫描且不自动换索引。
11. **Scalar 原始数据是否始终保留。** 始终保留最简单；按“已加载索引是否提供 raw data”决定释放更省资源，但切换、Drop 和回退会复杂。此前倾向始终保留。
12. **JSON 同一 path 是否允许多 cast/多索引类型。** 只允许多 path 时改动集中在 Load 目标和选择；允许同一 path 多 cast 时，创建判重和运行时选择键都要扩展。
