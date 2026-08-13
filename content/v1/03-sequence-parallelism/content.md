# 04 — ③ 序列维并行章：内容大纲

术语依据根 `CONTEXT.md`；结构顺序依据 `docs/adr/0001`。

## 目标

讲清楚为什么长序列（长视频 token）需要按序列切分（**上下文并行 Context Parallelism, CP**），并对比三种方案 —— **Ulysses**（all-to-all 换轴）、**Ring-Attention**（KV 环形流转）、**MagiAttention/flex-flash-attn**（FA-3 + 灵活掩码 + 分布式）——在②建立的通讯代价语言下的差异。落点用长视频 token 序列（MAGI 离线/非流式场景）。

## 记法与假设

- 沿用①MAGI 规格：`hidden=5120, heads=40, head_dim=128`，bf16。
- 场景：**离线/非流式**长视频序列（区别于①③MAGI streaming 用的 4096-token 单 chunk），完整序列 <span class="num">131072</span>（128K）token，跨 `P=8` 张卡做 CP，每卡本地 <span class="num">16384</span> token。
- 沿用②的硬件与代价模型：NVLink <span class="num">300</span> GB/s、α≈<span class="num">5</span>μs，all-to-all / p2p 原语代价公式。
- 为什么要切：①已展示同 seq=4096 时单层激活就有 3GB 量级；128K 序列的激活/QKV 张量线性暴涨，单卡放不下，必须把"序列"这一维切开——这是 CP 与④TP（切"隐藏维"）的关键区别。

## Slide 1 — 为什么需要按序列切：CP 与 TP 切的是不同维度

**要点：**
- 复用①的公式直觉：激活显存正比于 `batch × seq × hidden × layers`；seq 从 4096 增到 131072（32×），单卡装不下同一批次的完整序列。
- CP（本章）沿 **序列维** 切分，每卡处理一段 token；对比④要讲的 TP 沿 **隐藏/头维** 切分——两者可以叠加使用，但解决的是不同瓶颈（CP 解决"序列太长装不下"，TP 解决"单层太宽装不下"）。
- 三种 CP 方案的共同问题：Q 在本地就能算，但 attention 需要**全部** K/V 才能算对——如何跨卡拿到全部 K/V，是三种方案的分歧点。

## Slide 2 — Ulysses：all-to-all 换轴

**要点：**
- 思路：平时每卡按"序列切分、头全量"存 Q/K/V；attention 计算前做一次 **all-to-all**，换成"序列全量、头切分"（每卡拿到全部 token 但只算一部分 head），算完再 all-to-all 换回去。
- 用②的 all-to-all 公式估算：每卡 QKV 张量 `16384×5120×2bytes ≈ 160MB`，all-to-all 移动 `(P-1)/P×160MB=140MB`，耗时 `140MB/300GB/s≈0.47ms`；前后各一次（QKV 换轴 + 输出 O 换回）≈ **<span class="num">0.93</span> ms/层**。
- 限制：并行度 `P` 不能超过头数（MAGI heads=40，故 Ulysses CP 最多切 40 路），这是它相对 Ring 的硬约束。

## Slide 3 — Ring-Attention：KV 沿环流转

**要点：**
- 思路：每卡固定持有一段 Q，K/V 分片像②的 ring all-reduce 一样沿环逐步传递（p2p，`P-1` 步），每收到一个新 K/V 分片就和本地 Q 做一次局部 attention 并累加 softmax 统计量（online softmax），不需要等全部 K/V 到齐。
- 用②的 p2p 公式估算：每步传递 K+V 分片 `2×16384×40×128×2bytes≈320MB`，耗时 `320MB/300GB/s≈1.07ms`；`P-1=7` 步，原始通讯量 ≈ **<span class="num">7.47</span> ms/层**——比 Ulysses 高一个数量级，但**可与本地 attention 计算重叠**（边收边算），实际暴露的墙钟时间通常远小于这个数字。
- 优势：并行度不受头数限制，纯按 token 数切，`P` 可以很大；对超长序列（128K+）是更常见的选择。

## Slide 4 — MagiAttention / flex-flash-attn：给 Ring 加上灵活掩码与负载均衡

**要点：**
- 建立在 FlashAttention-3 kernel 之上（tiling/IO-aware，见⑤细节），核心新增两点（引用 CONTEXT.md）：
  - **AttnSlice 广义掩码**：不再假设标准因果掩码，能表达 MAGI 的 block-causal + 多模态掩码等异构 mask，并让这些不规则掩码依然可以按 chunk 切分、分布式计算。
  - **chunk 级负载均衡 + 零冗余通讯重叠**：像 Ring-Attention 一样沿环传 KV，但对被掩码整块跳过的区域**不计算也不搬运**，避免 Ring-Attention 朴素实现中"传了但被掩码浪费掉"的那部分通讯/算力。
- 定性结论（不给虚构精确基准数字，标注为定性比较）：对 MAGI 这种带 block-causal 结构化稀疏掩码的场景，FFA 的有效通讯量 **低于** Slide 3 朴素 Ring 的 7.47ms/层估算，因为掩码之外的分片直接跳过；具体加速比取决于掩码稀疏度，未公开可复算的基准数字，仅作定性说明。
- 这是 MAGI 真实使用的方案：block-causal + 多模态掩码的分布式长序列训练/推理，靠 FFA 而非朴素 Ring 或 Ulysses。

## Slide 5 — 三方案对比小结

**要点（表格 `.data-table`）：**

| 方案 | 切分依据 | 通讯原语 | 并行度上限 | 本章数字 |
|---|---|---|---|---|
| Ulysses | 序列↔头 反复换轴 | all-to-all ×2/层 | 受限于头数（≤40） | 0.93 ms/层 |
| Ring-Attention | 固定按序列切，KV 环传 | p2p ×(P-1)/层 | 不受头数限制 | 7.47 ms/层（原始，可与计算重叠） |
| MagiAttention/FFA | 同 Ring + 灵活掩码 | p2p（跳过掩码块） | 不受头数限制 | 低于 Ring（定性，视掩码稀疏度） |

- 落点一句话：三者都在回答"Q 在本地，K/V 怎么凑齐"，差异在换轴代价 vs 环传代价 vs 能否利用掩码稀疏性省掉无效传输——④训练并行范式会在此基础上叠加 DP/TP/PP/EP。

## 图表清单

| # | 图表 | 类型 | 复用组件类 | 新增 |
|---|------|------|-----------|------|
| 1 | CP vs TP 切分维度示意 | SVG 简图（张量方块沿两个轴切分对比） | `.chart`, `.chart-label` | 无 |
| 2 | Ulysses 换轴示意 | SVG（序列切分块 ↔ 头切分块，箭头表示 all-to-all） | `.chart`, `.chart-label` | 无 |
| 3 | Ring-Attention 环形传 KV | SVG 环形图，复用②的 `.ring-node`/`.ring-arc` 组件 | `.ring-node`, `.ring-arc`, `.ring-arc-flow` | 无（直接复用②新增的环形组件类） |
| 4 | MagiAttention 掩码跳过示意 | SVG 网格（灰色=跳过，色块=计算） | `.chart` | 无 |
| 5 | 三方案对比表 | `<table class="data-table">` | `.panel`, `.data-table` | 无（复用②新增的表格类） |

## 与 spec / ADR 的一致性检查

- 覆盖 Ulysses / Ring-Attention / MagiAttention 三种 CP 方案对比 ✅（Slide 2/3/4/5）。
- 通讯代价数字建立在②的 α-β 模型与硬件参数之上，可复算、量纲自洽（Slide 2/3）。
- 长视频 token 序列作为落点 ✅（128K MAGI 离线序列场景）。
- FFA 的加速未给虚构精确数字，仅定性描述并标注推断边界，符合 spec 对近似值的处理要求。
- 时间预算：本章 ~8 分钟，5 张 slide。

## 交付形态

- 大纲确认后 → `slide.html`，与共享主题同构、可独立打开。
- 并入 `index.html`（追加在②通讯原语章之后），图表数字与本大纲一致。
