# 03 — ② 通讯原语与代价模型章：内容大纲

术语依据根 `CONTEXT.md`；结构顺序依据 `docs/adr/0001-report-structure-and-ordering.md`。

## 目标

给出 all-reduce / all-gather / reduce-scatter / all-to-all / p2p 五种通讯原语的直觉图 + 统一的 **α-β 代价模型**，作为③序列并行、④训练并行范式代价推导的共同语言。核心演示：ring all-reduce 如何把"单点瓶颈"变成"环上均摊"，配分步动画；再用同一套 α/β 数字把五种原语串成一张对比表，预告谁在哪一章会再出现（TP→all-reduce，DP/ZeRO→all-gather+reduce-scatter，EP→all-to-all，PP→p2p）。

## 记法与假设（贯穿全章）

- α-β 代价模型：单次点对点传输 `T = α + β·M`（α = 每条消息固定延迟，β = 每字节传输时间 = 1/带宽）。
- 硬件参数取真实量级：节点内 **NVLink**（A100，单向 <span class="num">300</span> GB/s，α ≈ <span class="num">5</span> μs）；跨节点 **200Gbps InfiniBand**（<span class="num">25</span> GB/s，约为 NVLink 的 <span class="num">1/12</span>）——为④讲多机训练时"拓扑感知"埋伏笔。
- 复用①的数字：经典 LLM (Llama-2-7B 级) 梯度 bf16 = <span class="num">13.4</span> GB，<span class="term">P=8</span> 卡节点内 NVLink 环。
- Ring 算法的均摊数据量公式：all-reduce 每卡搬运 `2(P-1)/P · M`；all-gather / reduce-scatter 各搬运 `(P-1)/P · M`（两者相加正好是 all-reduce）。

## Slide 1 — 五种原语：谁在搬什么

**要点：**
- 一张 5 列小图/表格（复用 `.stat-card` 或新建简表），每种原语一句话 + 一个"输入→输出"示意：
  - **all-reduce**：每卡都有一份数据，规约（如求和）后每卡都拿到相同的完整结果（"多对多、结果相同"）。
  - **all-gather**：每卡有一片数据，规约后每卡都拿到所有分片拼起来的完整集合（不做规约，只做收集）。
  - **reduce-scatter**：每卡都有一份完整数据，规约后每卡只保留自己那一片（结果分散、不重复）。
  - **all-to-all**：每卡给每张卡发不同的一小片（转置式交换），常见于路由类操作。
  - **p2p (point-to-point)**：只有两张卡之间单向或双向传一份数据，无广播无规约。
- 落点一句话：前三种（all-reduce/all-gather/reduce-scatter）常成对出现在数据/参数分片场景，all-to-all 用于路由，p2p 用于流水线相邻阶段——④会逐一对号入座。

## Slide 2 — α-β 代价模型 + Ring All-Reduce 数字例子

**要点（`.stat-row` + 一段解读）：**
- 公式行：单次传输 `T = α + β·M`；α 是"起步成本"（消息越小越吃亏），β=1/带宽 是"搬运成本"（消息越大越吃这个）。
- 硬件参数：NVLink 节点内 <span class="num">300</span> GB/s（α≈<span class="num">5</span>μs），跨节点 200Gbps IB <span class="num">25</span> GB/s。
- **朴素做法（单卡收集再广播）**：8 卡，每卡梯度 13.4GB，瓶颈卡要收 7 份再发 7 份 = `2×7×13.4GB / 300GB/s` ≈ **<span class="num">625</span> ms**，且这个时间随 P 线性变差。
- **Ring all-reduce**：每卡只搬 `2×(8-1)/8×13.4GB ≈ 23.45GB`，耗时 `23.45GB / 300GB/s` ≈ **<span class="num">78</span> ms**——比朴素做法快 **<span class="num">8</span>×**，且 P 越大这个比值越接近 P（环形算法的每卡数据量随 P 增大趋于常数 `2M`，不再随 P 线性增长）。
- 落点一句话：ring 算法的精髓是把"单点瓶颈"摊成"环上均摊"，代价从 `O(P)` 降到 `O(1)`（每卡视角）。

## Slide 3 — Ring All-Reduce 分步动画：reduce-scatter 半环 + all-gather 半环

**要点：**
- SVG 环形拓扑（P=8 节点排成圆环），CSS 关键帧分步动画分两阶段：
  - **阶段一 reduce-scatter**（P-1 步）：每步每卡把自己的一个分片发给右邻居并累加，动画高亮"当前传输中的分片"沿环移动；P-1 步后每卡恰好持有一个"完全规约好"的分片。
  - **阶段二 all-gather**（P-1 步）：把这些已规约的分片继续沿环传递（不再累加，只转发），P-1 步后每卡集齐全部分片 = 完整 all-reduce 结果。
  - 动画用 `@keyframes` 循环播放（`animation-iteration-count: infinite` 或较长 duration 分步淡入），不要求用户交互，纯观察节奏。
- 图注标出：总步数 `2(P-1)=14` 步，每步每卡收发一个分片（大小 `M/P`），验证 Slide 2 的 `2(P-1)/P·M` 数据量公式。
- 一句话：这就是 Slide 2 数字背后的真实数据流，也是为什么 ring 代价与 P 无关（大 P 时）。

## Slide 4 — 五原语代价对比 + 谁在后面章节出现

**要点（表格，`.panel` 内嵌 HTML `<table>` 或复用 stat-card 网格）：**

| 原语 | 每卡数据量（ring/最优实现） | 延迟敏感度 | 后续出场 |
|---|---|---|---|
| all-reduce | `2(P-1)/P·M` | 中（消息大时β主导） | ④ TP 每层前向/反向同步激活 |
| all-gather | `(P-1)/P·M` | 中 | ④ DP/ZeRO 收集分片参数 |
| reduce-scatter | `(P-1)/P·M` | 中 | ④ DP/ZeRO 规约并分片梯度 |
| all-to-all | `(P-1)/P·M`，但常配小消息 | **高**（α 常主导，见下） | ④ EP 路由 token 到专家卡 |
| p2p | `M`（单次） | 低（大消息时） | ④ PP 相邻 stage 传激活 |

- all-to-all 的延迟敏感度举例：MoE 路由场景，每卡给其余 7 卡各发一小片 token 激活（例如每片 <span class="num">2</span> MB），`T ≈ 7×5μs + (7/8×16MB)/300GB/s` ≈ **<span class="num">35</span> μs (α项)** + **<span class="num">0.047</span> μs (β项，可忽略)**——消息越碎，α 项越主导，这是 EP 章要处理的"通讯效率"问题的根源。
- p2p 举例：PP 相邻 stage 间传一个激活张量（<span class="num">100</span> MB），`T ≈ 5μs + 100MB/300GB/s` ≈ **<span class="num">0.34</span> ms**，量级远小于 all-reduce，因为只发一份不做规约/收集。
- 落点一句话：五种原语的代价差异，本质上是"消息大小 vs 消息数量"在 α/β 两项之间的取舍——这套语言会在③④两章反复复用。

## 图表清单

| # | 图表 | 类型 | 复用组件类 | 新增 |
|---|------|------|-----------|------|
| 1 | 五原语总览 | stat-card ×5（或 2 行网格） | `.stat-row`, `.stat-card` | 无 |
| 2 | α-β 公式 + 朴素 vs ring 对比 | stat-card ×2 + 段落 | `.stat-row`, `.stat-card` | 无 |
| 3 | Ring all-reduce 环形分步动画 | SVG 环形图 + CSS keyframes | `.chart`, `.chart-label`, `.chart-caption` | 新增环形节点/弧线动画样式（`.ring-node`, `.ring-arc`，复用 `--accent`/`--accent-2` 变量，写入 theme.css） |
| 4 | 五原语代价对比表 | HTML `<table>` | `.panel` 包裹，新增 `.data-table` 类（复用主题字体/边框变量） | `.data-table` 表格样式写入 theme.css |

## 与 spec / ADR 的一致性检查

- 覆盖 all-reduce/all-gather/reduce-scatter/all-to-all/p2p 五种原语 + α-β 代价模型 ✅（Slide 1/2/4）。
- ring all-reduce 数据流有分步动画 ✅（Slide 3）。
- 通讯代价数字可复算：625ms/78ms/8×/23.45GB 等均可由 `2(P-1)/P·M/β` 系列公式复算，量纲自洽（GB / (GB/s) = s）。
- 复用①已建立的 7B 模型梯度数字（13.4GB），保持跨章数字一致性。
- 时间预算：本章 ~7 分钟，4 张 slide，为③④打基础。

## 交付形态

- 大纲确认后 → `slide.html`，与共享主题同构、可独立打开。
- 并入 `index.html`（追加在①性能估计章之后），图表数字与本大纲一致，不在合并时二次修改数字。
