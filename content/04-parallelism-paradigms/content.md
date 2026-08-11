# 05 — ④ 训练并行范式章：内容大纲

全场最重的一章（~12 分钟）。术语依据根 `CONTEXT.md`。

## 目标

讲清 DP/ZeRO、TP、PP、EP 四种主流训练并行模式：各自切什么、用②的哪种原语、代价几何。DP/ZeRO 落到 Philoflow 真实的 16-GPU FSDP 配置并说明为何不用 TP/PP；EP 落到 minimax h3 + 经典 MoE LLM，并与 MAGI 的 modality-experts 对比，讲清 EP 只作用于 token-routed experts。

## 记法与假设

- 沿用①经典 LLM（7B，训练权重+优化器 120.6GB）、MAGI（12.6B，权重 23.4GB）数字。
- 沿用②③的 α-β 代价模型、NVLink 300GB/s、②Slide4 的 all-to-all 35μs 路由数字（本章直接复用，不重新编数字）。
- Philoflow 16-GPU 配置：训练 MAGI 级模型，FSDP=ZeRO-3。

## Slide 1 — DP 的问题 + ZeRO/FSDP 怎么解决

**要点：**
- 朴素数据并行（DP）：每卡存**完整**模型+优化器状态，仅切分 batch；①已算出 7B 训练权重+优化器 = <span class="num">120.6</span> GB，单张 80GB 卡放不下——DP 单独用不了。
- ZeRO 三级（FSDP = ZeRO-3）：
  - ZeRO-1：只分片优化器状态
  - ZeRO-2：+分片梯度
  - ZeRO-3（FSDP）：+分片参数本身——每卡只常驻 1/P 的权重/梯度/优化器状态，用前**all-gather**参数、用后**reduce-scatter**梯度（②的原语，本章直接复用公式）。
- 落点：ZeRO-3 让"单卡装不下"的模型变得能训，代价是多了 all-gather/reduce-scatter 通讯。

## Slide 2 — 落到 Philoflow：16-GPU FSDP 训练 MAGI 级模型

**要点（`.stat-row`）：**
- 配置：MAGI 级（<span class="num">12.6</span>B，①已建立），Philoflow 用 `accelerate` + **FSDP（=ZeRO-3）**，<span class="term">P=16</span> 卡。
- 训练总显存占用（同①方法论，AdamW 混合精度）：bf16权重<span class="num">23.4</span>GB × 1.75(fp32主权重+梯度) + Adam动量4×<span class="num">23.4</span>GB，按①7B同比例（120.6/13.4≈9倍）估算 ≈ <span class="num">210.6</span> GB，FSDP-3 分到 16 卡 ≈ **<span class="num">13.2</span> GB/卡**——单卡轻松放下，且留有余量给激活。
- 通讯：每层前向/反向各一次 all-gather 收权重分片，反向后一次 reduce-scatter 规约梯度，用②公式量级与 TP 相当（下一节对比）。
- **为什么不用 TP/PP**（明确交代，不硬套）：MAGI 级模型（12.6B）比需要 TP/PP 的百亿级+ LLM 小得多，FSDP 分片已经能把单卡显存压到 <20GB；引入 TP 要承受每层同步的 all-reduce 延迟（下节），引入 PP 要承受流水线气泡（下下节）——在当前模型规模和 16 卡单集群规模下，两者带来的复杂度收益比不划算，Philoflow 选择只用 FSDP 扩展。

## Slide 3 — TP（张量并行）：切开单层权重矩阵

**要点：**
- Megatron 式 TP：把单层的权重矩阵（如 attention QKV、MLP 上下投影）按列/行切到多卡，每卡算一部分，层末用 **all-reduce** 合并结果——用的是②建立的 all-reduce 公式，但发生在**每一层**内部（而非每步一次）。
- 数字例子（经典 LLM，hidden=4096, TP=8）：单层激活 all-reduce 消息 = `4096×4096×2bytes≈33.6MB`（batch=1,seq=4096 简化取单 token 维度示意），耗时 `2×(8-1)/8×33.6MB/300GB/s≈0.196ms`；每层 attn+MLP 各一次 ≈ **<span class="num">0.39</span> ms/层**，32 层 ≈ <span class="num">12.6</span> ms/step。
- 关键约束：TP 的 all-reduce 是**同步阻塞**的（下一步计算必须等它完成），因此对延迟极敏感，几乎只用节点内 NVLink，很少跨节点——这也是 Slide 2 里 Philoflow 放弃 TP 的另一个理由（16 卡若跨多机，TP 的每层同步会被拖慢）。

## Slide 4 — PP（流水线并行）：切开层，用 p2p 传激活，代价是气泡

**要点：**
- 把模型按层切成多个 stage，跨卡分布；stage 间只需 **p2p** 传激活/梯度（②的 p2p 公式：小激活张量，代价很低，回顾③ Slide3 100MB→0.34ms 量级）。
- 核心代价不是通讯量，而是**流水线气泡（bubble）**：P 个 stage、M 个 micro-batch 时，气泡占比 ≈ `(P-1)/(P-1+M)`。
- 数字例子：P=<span class="num">4</span> stage，M=<span class="num">8</span> micro-batch → 气泡 = 3/(3+8) ≈ **<span class="num">27.3</span>%** 卡处于空闲等待；M 增到 <span class="num">16</span> → 气泡降到 <span class="num">15.8</span>%——micro-batch 数越多气泡摊得越薄，但显存换取的调度复杂度也上升（需缓存更多 in-flight 激活）。
- 落点：PP 用极低通讯量换取"层维度"的可扩展性，代价是调度气泡而非带宽。

## Slide 5 — EP（专家并行）：token-routed 才能 EP，modality-experts 不行

**要点：**
- EP 场景：MoE 层每卡常驻若干专家的**全部权重**，一个 **router** 给每个 token 动态选 top-k 专家，token 需要通过 **all-to-all**（②已建立公式）跨卡送到专家所在卡，算完再 all-to-all 送回——这是"token-routed experts"。
- 数字：直接复用②Slide4 的路由数字——8 卡、每卡给其余 7 卡各发 2MB 分片，`T≈35μs`（α 主导）——EP 的通讯瓶颈本质就是②讲过的"消息碎、α 项吃紧"。
- **落点模型**：<span class="term">minimax h3</span>（Hailuo 3.0，token-routed MoE 视频，Hailuo-02 引入 MoE + Noise-aware Compute Redistribution）+ 经典 MoE LLM（如 8 专家、top-2 路由）都是 token-routed，EP 直接适用。
- **对比 MAGI modality-experts**（`NativeMoELinear`, `num_modality=3`）：不是 router 学出来的动态选择，而是**按模态（video/text/…）静态决定**用哪套投影——同一模态的 token 永远走同一分支，不需要跨卡动态路由，也就**不需要 all-to-all**，可以把每个模态的 expert 干脆放在处理该模态数据的卡上。
- 一句话区分：EP 解决"动态路由到哪张卡"的通讯问题，只对 token-routed experts 有意义；MAGI 的 modality-experts 是静态分派，压根不产生 EP 要解决的那种通讯模式。

## Slide 6 — 四种范式对比小结

**要点（`.data-table`）：**

| 范式 | 切什么 | 用②哪种原语 | 代价来源 | 本章落点 |
|---|---|---|---|---|
| DP/ZeRO(FSDP) | 参数/梯度/优化器状态 | all-gather + reduce-scatter | 通讯量随分片数增加 | Philoflow 16-GPU，13.2GB/卡 |
| TP | 单层权重矩阵 | all-reduce（层内，同步阻塞） | 延迟敏感，需高速互联 | 经典 LLM，0.39ms/层 |
| PP | 层（stage） | p2p（stage间） | 流水线气泡，非带宽 | 4 stage/8 microbatch，27.3%气泡 |
| EP | token-routed 专家 | all-to-all | 消息碎、α 主导 | minimax h3 / 经典 MoE LLM，35μs |

- 落点一句话：四种范式切的维度互不相同（参数 vs 层内权重 vs 层 vs 专家），可以叠加组合（如 TP+PP+DP 的 3D 并行），但都建立在②的同一套原语与代价语言之上——⑤会在此基础上讲显存/计算优化如何进一步压缩每一项代价。

## 图表清单

| # | 图表 | 类型 | 复用组件类 | 新增 |
|---|------|------|-----------|------|
| 1 | ZeRO 三级分片示意 | SVG 条形（权重/梯度/优化器三层，逐级变窄） | `.chart` | 无 |
| 2 | Philoflow FSDP 16卡显存对比 | stat-card | `.stat-row`, `.stat-card` | 无 |
| 3 | TP 层内切分+all-reduce示意 | SVG（矩阵竖切 + 汇聚箭头） | `.chart` | 无 |
| 4 | PP 流水线气泡示意 | SVG 甘特图风格时间轴 | `.chart` | 无 |
| 5 | EP token-routed vs modality-experts 对比 | SVG 双示意图 | `.chart` | 无 |
| 6 | 四范式对比表 | `<table class="data-table">` | `.panel`, `.data-table` | 无（复用②新增类） |

## 与 spec / ADR 的一致性检查

- DP/ZeRO 用 Philoflow FSDP 16-GPU 举例，明确说明未采用 TP/PP 的原因 ✅（Slide 2）。
- EP 用 minimax h3 + 经典 MoE LLM 讲，并与 MAGI modality-experts 对比，区分 routed vs modality experts ✅（Slide 5）。
- 数字复用①②③已建立的基准（7B/MAGI 显存、all-to-all 35μs），未产生孤立新数字，量纲自洽。
- 时间预算：本章 ~12 分钟，6 张 slide，符合 spec"全场最重"的定位。

## 交付形态

- 大纲确认后 → `slide.html`，与共享主题同构、可独立打开。
- 并入 `index.html`（追加在③序列并行章之后），图表数字与本大纲一致。
