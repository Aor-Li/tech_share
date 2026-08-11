# 06 — ⑤ 显存与计算优化章：内容大纲

对应 ticket [06-memory-compute-optimization](../../.scratch/ai-infra-deck/issues/06-memory-compute-optimization.md)。术语依据根 `CONTEXT.md`。

## 目标

把①②③④埋下的伏笔逐一兑现：①的"97GB→2GB重计算"钩子、"混合精度120.6GB分解"、①③的 FlashAttention/KV cache 伏笔。讲清激活重计算/混合精度/offload 三种典型显存优化，FlashAttention 基础 + MagiAttention flex-flash-attn 进阶，最后用 Philoflow SNL 流式 + LTX-2 vs minimax h3 收口 KV cache。

## 记法与假设

- 沿用①的经典 LLM 7B（120.6GB 训练显存分解、97GB/2GB 激活对比）、MAGI（3.125GB reservoir KV）、LTX-2（6GB sink+滑窗 KV）数字，不重新编号。
- 硬件参数：A100 HBM 带宽 <span class="num">2039</span> GB/s（①已用）、SRAM 带宽量级 <span class="num">19</span> TB/s（FlashAttention 场景新增）、PCIe4 x16 <span class="num">25</span> GB/s（offload 场景新增）。

## Slide 1 — 激活重计算 (activation checkpointing)：97GB→2GB 怎么来的

**要点：**
- 机制：反向传播需要前向的中间激活值；不重计算时全部保留 = O(L)×每层激活；重计算只保留层边界的少量张量，反向时**重新跑一遍前向**补出被丢弃的中间激活，用算力换显存。
- 兑现①的钩子：经典 LLM 7B，seq=4096，不开重计算单层 <span class="num">3.03</span>GB×32层≈<span class="num">97</span>GB；开满重计算降到单层 <span class="num">0.06</span>GB×32≈**<span class="num">2</span>GB**——只保留层边界输入，中间激活按需重算。
- 代价：多一次前向重算，理论额外计算量 ≈ **<span class="num">33</span>%**（原本 1 次前向+1 次反向(≈2倍前向算力)=3 份，加一次重算前向=4 份，4/3-1≈33%）。
- 落点：这是"存 vs 算"权衡的第一个具体例子，呼应①的 roofline 收口。

## Slide 2 — 混合精度：120.6GB 分解里每一项在干嘛

**要点：**
- 重新拆解①的 <span class="num">120.6</span>GB：bf16权重<span class="num">13.4</span>GB（前向/反向用，tensor core 快）+ fp32主权重<span class="num">26.8</span>GB（累积小梯度更新不下溢）+ fp32梯度<span class="num">26.8</span>GB + Adam一阶二阶矩<span class="num">53.6</span>GB（=2×26.8，优化器状态要 fp32 保数值稳定）。
- 为什么用 **bf16** 而不是 fp16：bf16 指数位与 fp32 相同（8 位），动态范围一致，**不需要 loss scaling**；fp16 指数位只有 5 位，训练大模型容易下溢，需要额外的 loss-scaling 机制。
- 落点：混合精度不是"权重减半"这么简单，是"计算用低精度、状态累积用高精度"的分层策略。

## Slide 3 — Offload：显存换 PCIe 带宽

**要点：**
- 机制：把不在计算关键路径上的状态（典型是优化器状态）挪到 CPU 内存，用时再搬回 GPU——本质是①roofline 里"存 vs 算"权衡的另一种形式："存 vs 搬运带宽"。
- 数字：PCIe4 x16 带宽仅 <span class="num">25</span> GB/s，远低于②的 NVLink <span class="num">300</span>GB/s（约 1/12）。若把 7B 模型 Adam 优化器状态 <span class="num">53.6</span>GB 整体搬到 CPU，单次传输 `53.6GB/25GB/s≈`**<span class="num">2.14</span> s**——比②③④任何一次通讯都慢一个数量级，因此 offload 只对"能和计算重叠、非关键路径"的状态划算（如优化器 step 与下一步前向重叠），不能天真地每步同步搬运。
- 落点：offload 把①的显存三项分解再加一维"在哪"（GPU/CPU），付出的代价用②③④建立的带宽语言可以直接估算。

## Slide 4 — FlashAttention 基础：tiling + IO-aware，不物化 N×N

**要点：**
- 朴素 attention 要显式算出 `S=QK^T` 这个 N×N 矩阵：seq=4096, heads=32, bf16，单层 `4096×4096×2bytes×32heads≈`**<span class="num">1.07</span> GB**，且要整份写回/读回 HBM（③④用到的同一个 HBM 带宽瓶颈）。
- FlashAttention 核心思想：**tiling**（把 Q/K/V 切成小块）+ **IO-aware**（尽量在 SRAM 里完成 softmax 归约，只把最终结果写回 HBM）——SRAM 带宽量级 <span class="num">19</span> TB/s，比 HBM 的 <span class="num">2039</span>GB/s 快近 <span class="num">10</span>×，但容量小得多（A100 每 SM 仅约 192KB）。
- 落点：FlashAttention 省的不是 FLOPs（数学上等价），是 HBM 读写次数——又一次"用什么带宽做什么事"的roofline式思考。

## Slide 5 — MagiAttention / flex-flash-attn：FA 的分布式 + 灵活掩码扩展

**要点：**
- 建立在 Slide 4 的 FA-3 kernel 之上，③已讲过它在**通讯**维度的表现（chunk 级负载均衡+零冗余重叠，跳过掩码外分片）；本节收口它在**kernel/掩码**维度的贡献：<span class="term">AttnSlice</span> 广义掩码让 block-causal、多模态等异构掩码也能享受 tiling/IO-aware 的好处，而不必退化成朴素稠密 attention。
- 一句话：③讲的是"分布式怎么切"，本节讲的是"单卡 kernel 怎么快"，两者叠加才是 MAGI 实际训练/推理用的完整方案。

## Slide 6 — KV Cache 收口：Philoflow SNL 流式 + LTX-2 vs minimax h3

**要点：**
- **Philoflow SNL (Same-Noise-Level)**：MAGI self-forcing DMD 自回归生成的 KV-cache 噪声机制——上下文与当前去噪 chunk 处于同一噪声级，用 block-causal 掩码，每个噪声级一份 KV-cache；`streaming` 模式逐 chunk 生成、配 first-block/first-latent **sink**，实现无限长生成。兑现①的 <span class="num">3.125</span>GB reservoir 数字：这就是 SNL 机制下的窗口大小。
- **LTX-2 对比**（①已给数字 <span class="num">6.0</span>GB）：sink+滑窗 KV 策略不同于 SNL 的"按噪声级缓存"，是按时间窗口缓存；额外差异——文本交叉注意力 K/V 依赖当前噪声级 sigma，**不缓存**，每步重算，用算力换存储（呼应①roofline）。
- **minimax h3 对比（显式标注推断边界）**：h3 公开信息描述为 flow-matching diffusion transformer 联合去噪视频+音频 latent，采用 token-routed MoE（④已讲 EP），但**未公开明确的 AR 式 KV-cache 流式生成机制**（不同于 MAGI/LTX-2 是 philoflow-core 内部实现、细节可查）；若其推理管线包含类似的上下文缓存/流式扩展，其显存策略会面临同样的"缓存窗口 vs 生成长度"权衡，但具体数字未公开、无法复算，本页仅作**定性类比**，不给虚构数字。
- 三者对比小结：经典 LLM KV 随 seq 线性无界增长（①）→ MAGI SNL 窗口恒定→ LTX-2 sink+滑窗恒定且部分K/V不存 → minimax h3 策略未公开（推断边界），四者是同一问题在不同架构假设下的不同取舍。

## 图表清单

| # | 图表 | 类型 | 复用组件类 | 新增 |
|---|------|------|-----------|------|
| 1 | 重计算前后激活对比 | stat-card + 段落 | `.stat-row`, `.stat-card` | 无 |
| 2 | 120.6GB 四项分解条形图 | SVG 堆叠条形 | `.chart` | 无 |
| 3 | Offload 显存↔PCIe示意 | stat-card | `.stat-row`, `.stat-card` | 无 |
| 4 | 朴素attention N×N物化 vs FlashAttention tiling | SVG 对比图 | `.chart` | 无 |
| 5 | （文字收口，无新图，引用③环形图） | — | — | 无 |
| 6 | 四模型 KV cache 策略对比表 | `<table class="data-table">` | `.panel`, `.data-table` | 无（复用②新增类） |

## 与 spec / ADR 的一致性检查

- 覆盖激活重计算/混合精度/offload ✅（Slide 1/2/3）。
- FlashAttention 基础 + MagiAttention flex-flash-attn 都讲到 ✅（Slide 4/5）。
- KV cache 结合 Philoflow SNL 讲流式无限长，并做 LTX-2 vs minimax h3 对比，minimax h3 内容标注为推断边界 ✅（Slide 6）。
- 数字复用①②③已建立的基准，未产生孤立新数字（除 offload/FlashAttention 新增的硬件参数，均标注来源）。
- 时间预算：本章 ~9 分钟，6 张 slide。

## 交付形态

- 大纲确认后 → `slide.html`，与共享主题同构、可独立打开。
- 并入 `index.html`（追加在④训练并行范式章之后），图表数字与本大纲一致。
