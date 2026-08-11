# Spec: AI Infra 技术分享报告（单文件 HTML 幻灯）

Status: ready-for-agent
Feature slug: ai-infra-deck

> 术语见根 `CONTEXT.md`；本 spec 遵循 `docs/adr/0001–0004`。正文中文 + 英文术语（首现中英并注，之后统一英文）。

## Problem Statement

作者要做一场 ≤1 小时、以 AI Infra 为主题、面向混合工程团队的技术分享。手头素材分散、缺少统一结构与统一视觉，且很多理论（显存/算力估计、通讯、并行、优化）如果只讲公式会枯燥、听众抓不住量级直觉。作者希望产出一份**风格统一、能本地保存、可离线放映、便于逐章打磨**的报告，并能结合自己熟悉的真实系统（Philoflow 的 MAGI / LTX-2、以及 minimax h3）来讲，而不是纯抽象。

## Solution

产出一份自包含的**单文件 HTML 幻灯**（仓库根 `index.html`），Catppuccin Mocha 深色风、SVG 线性图表 + 简单分步动画，覆盖六段主线（性能估计 → 通讯 → 序列并行 → 并行范式 → 优化，首尾加全景/部署页）。每个核心概念先给直觉图 + 一个"走一遍"的量级数字，再用一行公式收口，并挂到最自然的模型上（经典 LLM 主线，MAGI/LTX-2/minimax h3 作视频落点）。开发按章推进：先出 Markdown 大纲供确认，再转成同构 HTML 片段，最后零成本并入母文件。所有中间产物随仓库进 git，支持远程/AFK 开发。

## User Stories

1. As a 分享者, I want 一份自包含单文件 `index.html`, so that 我能在任意机器上离线打开放映、无需网络或构建。
2. As a 分享者, I want 报告能存入 git 仓库, so that 我能内部保存、版本化、并在远程环境继续开发。
3. As a 分享者, I want 全场风格统一（Catppuccin Mocha + Mauve/Teal + 统一 SVG 图表）, so that 观感专业、不分散注意力。
4. As a 分享者, I want 键盘翻页与清晰的页序, so that 现场放映顺畅。
5. As a 分享者, I want 每章可独立打开的 HTML 片段, so that 我能逐章审阅、局部修改而不影响其他章。
6. As a 分享者, I want 每章先有 Markdown 大纲（要点+图表清单+具体数字）, so that 在做图前就能确认内容正确、节奏合适。
7. As a 分享者, I want 全场总时长可控在 ~50 分钟正文 + 缓冲, so that 一小时档期内讲得完并留 Q&A。
8. As a 混合背景听众, I want 每个概念先有直觉图再有一行公式, so that 不懂推导也能抓住量级与结论。
9. As a 听众, I want 关键机制有简单分步动画（ring all-reduce 数据流、TP 切分、PP 气泡、KV cache 滑窗）, so that 抽象的并行/通讯过程变得可视。
10. As a 听众, I want 开场有一页"全景地图", so that 我知道各主题如何串联、现在讲到哪。
11. As a 听众, I want 结尾有一页"组合成真实部署"（Philoflow 16-GPU / minimax 举例）, so that 我理解这些技术如何在真实系统里协同。
12. As a 分享者, I want 在"性能估计"章用经典 LLM + MAGI(5120/40) + LTX-2(48层) 估显存（模型/激活/KV 三项）, so that 听众看到真实模型的显存分解而非抽象公式。
13. As a 分享者, I want 在"性能估计"章用 roofline 直觉解释算力/时间, so that 听众理解何时受算力约束、何时受带宽约束。
14. As a 分享者, I want 在"通讯"章讲 all-reduce/all-gather/reduce-scatter/all-to-all/p2p 原语与 α-β 代价模型, so that 后续并行章的代价推导有统一基础。
15. As a 分享者, I want 在"序列并行"章对比 Ulysses / Ring-Attention / MagiAttention 三种 CP 方案, so that 听众理解长序列注意力如何跨卡切分及其通讯差异。
16. As a 分享者, I want 用长视频 token 序列作为序列并行的落点, so that 例子贴近作者真实场景。
17. As a 分享者, I want 在"并行范式"章讲 DP/ZeRO、TP、PP、EP, so that 覆盖主流训练并行模式。
18. As a 分享者, I want DP/ZeRO 用 Philoflow FSDP(=ZeRO-3)、16-GPU 配置举例, so that 听众看到真实分片训练。
19. As a 分享者, I want 明确说明 Philoflow 未采用 TP/PP 及其原因（靠 FSDP 扩展）, so that 牵强处如实交代而非硬套。
20. As a 分享者, I want EP 用 minimax h3（MoE 视频）+ 经典 MoE LLM 讲, 并对比 MAGI 的 modality-experts, so that 听众区分 token-routed experts 与 modality experts、理解 EP 只作用于前者。
21. As a 分享者, I want 在"优化"章讲激活重计算(checkpoint)、混合精度、offload, so that 覆盖典型显存优化。
22. As a 分享者, I want FlashAttention 讲基础(tiling/IO-aware/不物化 N×N) 并加 MagiAttention flex-flash-attn(FA-3+灵活掩码+分布式), so that 听众既懂原理又见到进阶工程。
23. As a 分享者, I want KV cache 章结合 Philoflow SNL 流式讲无限长生成, so that 展示 KV cache 在扩散/AR 视频里的真实用法。
24. As a 分享者, I want KV cache 章对比 LTX-2（文本交叉注意力 K/V 依赖 sigma 故不缓存、时序 K/V 缓存）与 minimax h3, so that 听众看到 KV cache 大小/策略如何随架构剧烈变化。
25. As a 分享者, I want 每个"能立刻改变部署决策"的量（显存/算力/通讯代价/KV cache）都给完整数字例子, so that 结论可落地。
26. As a 远程/AFK 开发者, I want spec 与设计决策（CONTEXT.md、ADR、chapter 大纲）都在 git 里, so that 我能在远程环境凭仓库自足地继续开发。
27. As a 维护者, I want 每节内容（`content.md` + `slide.html`）同址存放于 `content/NN-<slug>/`，ticket 仍在 `.scratch/ai-infra-deck/issues/`, so that 内容与展示可同处迭代，工作项仍可被 wayfinder / to-tickets 等技能接管。
28. As a 分享者, I want 母文件由各章 HTML 片段通过共享 CSS 变量零成本合并, so that 改主题只需改一处、合并无冲突。

## Implementation Decisions

- **结构（ADR 0001）**：六段主线 + 首尾统领页，顺序为 度量 → 通讯 → 序列并行 → 并行范式 → 优化。全景地图开场、真实部署收尾。
- **落地映射（ADR 0002）**：各取所长——经典 LLM 主线；MAGI/LTX-2/minimax h3 作视频落点；合理处扩展到 Philoflow，牵强处（TP/PP）说明原因。EP 讲清 routed-experts（minimax h3 / 经典 MoE LLM，EP 作用于此）vs modality-experts（MAGI `NativeMoELinear`）。
- **交付形态（ADR 0003）**：自包含单文件本地 `index.html`（HTML/CSS/JS，键盘翻页），关键机制用 SVG + CSS 关键帧分步动画。**不以 Artifact 为最终物**（仅开发期预览）。
- **中间产物入库（ADR 0004）**：`.scratch/` 取消忽略、随仓库进 git，支持远程开发。
- **视觉 token**：Catppuccin Mocha 深色（base `#1e1e2e`），主强调 Mauve `#cba6f7`、次强调 Teal `#94e2d5`；正文无衬线，术语/数字/代码等宽。全部图表 SVG 线性风格 + 单一强调色。主题以 CSS 变量集中定义。
- **文件布局**：最终母文件 `index.html`（根，自包含）；各节内容同址存放于 `content/NN-<slug>/`，每节含 `content.md`（Markdown 大纲）+ `slide.html`（同构、可独立打开、并入母文件）；共享 `content/theme.css` + `content/nav.js`；spec/issues 仍在 `.scratch/ai-infra-deck/`（进 git）。
- **开发流程（逐节）**：`content/NN-<slug>/content.md` 出 Markdown 大纲 → 作者确认 → 同址 `slide.html` → 作者审 → 并入 `index.html`。
- **时间预算**：正文 ~50′，④并行范式 12′ 最重，①性能估计 9′ 次之；总控 1 小时内留 Q&A。
- **数学深度**：折中——显存/算力/通讯/KV cache 给完整数字例子，其余给公式 + 一句量级直觉。
- **合并机制**：各章片段共享同一套 CSS 变量与组件类名，母文件仅做拼接 + 统一导航/页码，无逐章样式分叉。
- **语言**：中文正文 + 英文术语（首现中英并注）。

## Testing Decisions

好的"测试"只验证外部可观察行为，不绑定实现细节。本项目为演示文档，验证以**可打开、可放映、内容正确、风格一致**为准：

- **首选单一 seam：合并后的 `index.html` 在浏览器打开**——验证全场可放映：页序正确、键盘翻页可用、每章关键图表与数字都在、无横向溢出、深浅（若做兜底）主题正常。这是最高层 seam，尽量只在此验证。
- **次级 seam：单节 `content/NN-<slug>/slide.html` 独立打开**——验证片段自足渲染且套用共享主题（作者逐章审阅时用）。
- **内容正确性检查**：每章大纲里"走一遍"的数字（如 Llama-70B KV cache、MAGI/LTX-2 显存分解、通讯代价）需可复算、量纲自洽；数字以大纲为准入库，图表不得与大纲数字冲突。
- **风格一致性检查**：所有颜色/字体走 CSS 变量，不出现硬编码偏离 Mocha 调色板的值；图表统一 SVG 线性风。
- **时间预算检查**：各章幻灯数 × 预估讲述时长 ≤ 该章分钟预算，全场 ≤ ~50′ 正文。
- **Prior art**：无（全新仓库）。测试以人工/视觉审阅清单为主，可辅以一个轻量脚本校验各章片段能拼接且共享同一 CSS 变量块。

## Out of Scope

- 不做 Artifact/在线托管为最终交付（仅开发期预览可用）。
- 不做构建工具链/打包器（保持无构建、单文件）。
- 不做演讲逐字稿/讲者备注（除非后续单独提出）。
- 不深入实现级源码讲解——报告注重广度与科普，不逐行剖析 Philoflow/模型源码。
- 不单列"线性/闪电注意力"章节（已降级删除，非原始需求）。
- 不覆盖 minimax h3 的内部实现细节（其架构以公开信息推断，报告中明确标注为推断而非代码事实）。
- 不做多语言版本（中文为主 + 英文术语即可）。

## Further Notes

- 关键事实已核实并写入 `CONTEXT.md`：MAGI(`hidden 5120 / 40 层 / head_dim 128`)、LTX-2(`48 层`, sink+滑窗 KV)、minimax h3(Hailuo 3.0, flow-matching DiT + token-routed MoE 视频, 2K/15s)、MagiAttention flex-flash-attn(FA-3+AttnSlice 灵活掩码+分布式)、SNL(Same-Noise-Level 流式无限长)。
- minimax h3 相关 KV/EP 论述以公开信息为据，报告需显式标注推断边界。
- 下一步建议：`/to-tickets` 将本 spec 拆成逐章 ticket（`01` 性能估计 … 起），再按开发流程逐章推进。
- 远程开发前置：`.scratch/` 已入库；作者需在远端 `git clone` 后即可凭 `spec.md` + `CONTEXT.md` + `docs/adr/` 自足开工。
