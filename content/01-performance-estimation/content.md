# 02 — ① 性能估计基础章：内容大纲

对应 ticket [02-performance-estimation](../../.scratch/ai-infra-deck/issues/02-performance-estimation.md)。术语依据根 `CONTEXT.md`；结构顺序依据 `docs/adr/0001-report-structure-and-ordering.md`。

## 目标

让听众建立"显存去哪了 / 算力受什么约束"的量级直觉，为后续②通讯、④并行范式、⑤显存优化打基础。核心是**同一套三项分解公式**（模型权重 / 激活 / KV cache）套在三个真实模型形状上，数字越滚越大、越滚越有画面感，最后用 roofline 收口到"什么时候该省显存、什么时候该省带宽"。

## 记法与假设（贯穿全章）

- 精度统一 bf16（2 bytes/elem），fp32 中间态（优化器/梯度）4 bytes/elem。
- 经典 LLM 取 **Llama-2-7B 级配置**（真实可查）：`hidden=4096, layers=32, heads=32, head_dim=128`，参数量 <span class="num">6.7</span>B。
- MAGI 取 CONTEXT.md 已核实规格：`hidden=5120, layers=40, head_dim=128` → heads=40；参数量用标准 dense block 估算 `12·h²·L`（**近似值**，非官方发布数字，标注推断边界）。
- LTX-2 仅 `layers=48` 公开；hidden/heads 未公开，示例取 `hidden=4096, head_dim=128`（heads=32，与经典 LLM 同阶，**仅用于展示 sink+滑窗 KV 的形状差异，非产品真实参数**，需在页面上标注"示例参数"）。
- 所有数字可用 `2 × layers × heads × head_dim × tokens × batch × bytes`（KV）与 Megatron 激活显存公式 `seq·batch·hidden·(34 + 5·heads·seq/hidden)`（无重计算）复算，见页面公式行。

## Slide 1 — 显存去哪了：三项分解框架

**要点：**
- 标题给出总览：任意时刻的显存 = **权重 (weights)** + **激活 (activations)** + **KV cache**，三者此消彼长（训练时前两者主导，推理/解码时权重+KV主导，几乎不产生新激活）。
- 三个 stat-card（`.stat-card`）分别标注三项的"决定因素"：
  - 权重：参数量 × 精度字节数 × (是否含优化器状态)
  - 激活：batch × seq × hidden × layers，是否开重计算（细节留给⑤）
  - KV cache：`2 × layers × heads × head_dim × 已生成 token 数 × batch × 字节数`（K和V各一份，故×2）
- 不出具体模型数字（留给后面三页 walkthrough），只建立框架，方便后三页复用同一张"公式卡片"。

## Slide 2 — 经典 LLM 走一遍：Llama-2-7B 级

**要点（`.stat-row` 三张卡 + 一段解读）：**
- 配置：`hidden=4096, layers=32, heads=32, head_dim=128`，参数量 <span class="num">6.7</span>B。
- **权重+优化器（训练，混合精度 AdamW）**：bf16 权重 <span class="num">13.4</span> GB + fp32 主权重 <span class="num">26.8</span> GB + fp32 梯度 <span class="num">26.8</span> GB + Adam 一阶二阶矩 <span class="num">53.6</span> GB = **<span class="num">120.6</span> GB**（单卡放不下 7B——这是④ ZeRO/FSDP 要解决的问题，此处埋伏笔）。
  - 若只做推理（仅 bf16 权重）：<span class="num">13.4</span> GB。
- **激活（训练前向，batch=1，seq=4096，不开重计算）**：单层 <span class="num">3.03</span> GB × 32 层 ≈ **<span class="num">97</span> GB**；若开满重计算：单层 <span class="num">0.06</span> GB × 32 ≈ **<span class="num">2</span> GB**（97→2GB 的对比，是⑤重计算章的钩子，此处只给数字不展开机制）。
- **KV cache（推理解码，batch=1，seq=4096）**：`2×32×32×128×4096×1×2 bytes` = **<span class="num">2.0</span> GB**（整数，好记）；若 <span class="num">32</span> 个并发请求同时占满上下文：**<span class="num">64</span> GB**——KV cache 是服务吞吐的隐形天花板。
- 落点一句话：三项里权重相对固定，激活和 KV cache 才是"看配置暴涨"的部分。

## Slide 3 — 落到 MAGI：更大的 hidden，更快垒起来

**要点：**
- 配置：`hidden=5120, layers=40, head_dim=128` → heads=40。参数量按 dense block 估算 `12·h²·L ≈ ` **<span class="num">12.6</span> B**（标注：近似估算，MAGI 未公开精确参数量，仅用于量级对比）。
- 权重（bf16，推理/SFT-DMD 蒸馏后常见形态）：**<span class="num">23.4</span> GB**。
- 激活（同方法论，假设单步处理 <span class="num">4096</span> 个 latent token，batch=1，不开重计算）：单层 <span class="num">3.79</span> GB × 40 层 ≈ **<span class="num">151.6</span> GB**——同一公式，hidden 和 layers 都更大，数字比经典 LLM 涨得更快，直觉上"DiT 比同 token 数的 LLM 更吃激活显存"。
- KV / 缓存（AR 化流式生成，frame-reservoir + KV eviction，见 CONTEXT.md SNL 机制）：缓存只保留 reservoir 深度内的 chunk（示例：<span class="num">4096</span> token 等效窗口），大小**不随生成总长度增长**，算得 **<span class="num">3.125</span> GB**，且恒定——这是和经典 LLM KV cache（随 seq 线性增长）的关键差异，为⑤ KV cache 章埋伏笔。
- 一句话小结：换成视频 DiT 形状，公式不变，但每一项的"账单"都变了。

## Slide 4 — 落到 LTX-2：sink + 滑窗，显存不随生成长度增长

**要点：**
- 明确标注：LTX-2 仅 `layers=48` 是公开规格，本页 `hidden=4096` 为**示例参数**（非产品真实值），只用于展示"注意力汇聚点(sink)+滑动窗口"这种 KV 策略的形状，不做权重/激活的具体数字（避免用虚构 hidden 数字误导权重估算）。
- KV cache 策略：`sink_chunks=2` + `max_chunks=6`（滑动窗口），每 chunk <span class="num">1024</span> token → 缓存总 token 数恒为 `(2+6)×1024=8192`，与生成总帧数无关。按 `2×48×32×128×8192×2 bytes` 算得 **<span class="num">6.0</span> GB**，且**这是硬上限**，无论生成多长视频都不会突破。
- 额外差异点（引用 CONTEXT.md）：文本交叉注意力的 K/V 依赖当前噪声级 sigma，**不做缓存**——每步都重新算，用算力换掉了这部分存储，是"存 vs 算"权衡的具体例子，直接呼应下一页的 roofline。
- 三模型对比小结（可用一行文字或简表）：经典 LLM KV 随 seq 线性无界增长 → MAGI reservoir 恒定但仍需存 → LTX-2 sink+滑窗恒定且部分 K/V 干脆不存，三者是同一问题的三种取舍。

## Slide 5 — Roofline 直觉：什么时候该省显存、什么时候该省带宽

**要点：**
- Roofline 核心思想一句话：某个算子的实际吞吐 = `min(峰值算力, 带宽 × 算术强度)`；算术强度 AI = `FLOPs / 搬运字节数`，AI 越低越"带宽受限"，越高越"算力受限"。
- 用 A100 80GB SXM 举例给出脊点（ridge point）：峰值 bf16 算力 <span class="num">312</span> TFLOPs，HBM 带宽 <span class="num">2039</span> GB/s → 脊点 AI ≈ <span class="num">153</span> FLOPs/byte（AI 低于此值就是带宽受限，高于则是算力受限）。
- 两个对照点：
  - **训练/prefill 的大矩阵乘**：AI 高（batch×seq 大，权重复用充分）→ 落在算力受限区，加卡/加算力有效。
  - **自回归解码 (decode)**：每步只处理 1 个新 token，却要整份读一遍 KV cache（对照 Slide 2 的 2GB/64GB 数字）→ AI 很低 → 落在带宽受限区，加算力没用，得靠更大 batch（continuous batching）或更快的显存带宽。
- roofline 图：SVG 折线图（`.chart`），x 轴算术强度（log），y 轴吞吐；一条"屋顶"折线（斜线段=带宽受限，水平段=算力受限，交点=脊点），标出 GEMM 点（右侧水平段）和 decode 点（左侧斜线段）两个散点，直接引用 Slide 2/Slide 3 算出的 KV cache 数字作为 decode 点的注解。
- 落点一句话：显存三项分解告诉你"占多少"，roofline 告诉你"快不快"——两者合起来才是①的完整答案，②通讯的代价模型会在此基础上继续展开。

## 图表清单

| # | 图表 | 类型 | 复用组件类 | 新增 |
|---|------|------|-----------|------|
| 1 | 三项分解总览卡片 | stat-card ×3 | `.panel`, `.stat-row`, `.stat-card`, `.stat-num`, `.stat-label` | 无 |
| 2 | 经典 LLM 数字 walkthrough | stat-card ×3 + 段落 | 同上 | 无 |
| 3 | MAGI 数字 walkthrough | stat-card ×3 + 段落 | 同上 | 无 |
| 4 | LTX-2 sink+滑窗示意 + KV 数字 | SVG 时间轴图（sink chunk + 滑动窗口高亮）+ stat-card | `.chart`, `.chart-label`, `.stat-card` | 1 个新 SVG 局部样式（滑窗高亮矩形，复用 `--accent`/`--accent-2` 变量，不新增 CSS 类以外的硬编码色） |
| 5 | Roofline 折线图 | SVG 折线图 + 两个散点 | `.chart`, `.chart-line`, `.chart-grid`, `.chart-label` | 无（沿用既有图表组件类） |

## 与 spec / ADR 的一致性检查

- 显存三项分解覆盖经典 LLM + MAGI + LTX-2，各给完整数字例子 ✅（Slide 2/3/4）。
- roofline 给出算力/带宽约束的直觉解释 ✅（Slide 5），并与 Slide 2/3 的 KV cache 数字联动，避免孤立公式。
- 近似/示例参数（MAGI 参数量估算、LTX-2 示例 hidden）均显式标注推断边界，符合 spec Further Notes 的要求。
- 时间预算：本章 ~9 分钟，5 张 slide，平均每页 ~1.8 分钟，与 spec 的时间预算量级吻合。

## 交付形态

- 大纲确认后 → `slide.html`，与共享主题同构、可独立打开。
- 并入 `index.html`（追加在 00 章之后），图表数字与本大纲一致，不在合并时二次修改数字。
