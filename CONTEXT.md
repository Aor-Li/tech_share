# Tech Share — AI Infra 技术分享报告

本项目的"领域"就是这份报告本身：一场时长 ≤1 小时、以 AI Infra 为主题、面向混合工程团队的科普性技术分享。产出物是一份风格统一、可本地保存的单文件 HTML 幻灯。本文件是报告的**统一术语表（ubiquitous language）**，只记术语选择，不记实现细节。

## Language

**报告 (The Report / The Deck)**:
本项目唯一的交付物 —— 一份自包含、可离线打开、可存入 git 仓库的单文件 HTML 幻灯。
_Avoid_: PPT、slides 文件、artifact（Artifact 仅是开发期预览手段，不是交付物本体）

**章节稿 (Chapter Draft)**:
单章的中间产物，先以 Markdown 内容大纲形态存在（要点 + 图表清单 + 量级估算数字），确认后再转成与最终样式同构、可独立打开的 HTML 片段。
_Avoid_: 草稿、临时稿（章节稿是要逐章审阅并最终并入母文件的正式中间态）

**经典 LLM (Classic LLM)**:
报告中作为主线基准的 Llama 类 dense decoder-only Transformer。用来讲 FLOPs / KV cache / TP / PP 等通用范式。
_Avoid_: GPT（除非特指某具体模型）；"大模型"泛指

**minimax h3 (Hailuo 3.0)**:
MiniMax 于 2026-08 发布的开放权重**全模态视频生成模型**：flow-matching diffusion transformer，联合去噪视频+音频 latent，多模态文本编码器 + 视频/音频 VAE；Hailuo 谱系采用 **token-routed MoE**（Hailuo-02 引入 MoE + Noise-aware Compute Redistribution），15s / 2K / 原生立体声。是报告中 **EP（专家并行）的视频落点**，也用于显存/超长序列对比。
_Avoid_: 把它当 LLM；把它和 MiniMax-01（文本 LLM）混为一谈

**MAGI**:
`philoflow-core` 训练的视频 DiT（`hidden_size 5120 / 40 层 / head_dim 128`，text encoder dim 3584），含 SFT / DMD 蒸馏 / AR 化流式生成。用 **modality experts**（`NativeMoELinear`, `num_modality=3`，按模态分投影），区别于 token-routed MoE —— 这是 EP 章的对比点。AR 推理用 frame-reservoir + KV eviction 做流式无限长生成。
_Avoid_: 把 MAGI 的 modality experts 说成可做 EP 的 routed experts（两回事）

**LTX-2**:
`philoflow-core` 内实现的视频 DiT（`num_layers 48`），推理用 **attention-sink + sliding-window KV cache**（`sink_chunks` / `max_chunks`）做流式生成。是 KV cache 章的精确落点。
_Avoid_: 与 minimax/MAGI 混用规格数字

**MagiAttention / flex-flash-attn (FFA)**:
Sand.ai 2026.03 的分布式注意力（面向超长上下文、异构掩码训练的线性扩展）。**flex-flash-attn** 是 FlashAttention-3 的扩展：用 AttnSlice 广义掩码 + 定制 kernel 表达多样/packed 掩码并使其可分布式切分；配 chunk 级负载均衡 + 零冗余通讯重叠。既是 FlashAttention 章的落点，又是序列/上下文并行章的一种 CP 方案（与 Ulysses/Ring 并列）。MAGI 的 block-causal/多模态掩码即用它。
_Avoid_: 把 FFA 等同于普通 FlashAttention（它多了掩码灵活性 + 分布式）

**SNL (Same-Noise-Level)**:
MAGI self-forcing DMD 自回归生成的 KV-cache 噪声机制：上下文与当前去噪 chunk 处于**同一噪声级** → block-causal 掩码 → 每个噪声级一份 KV-cache。`streaming` 模式下逐 chunk AR、N 份级配 KV → **无限长生成**，配 first-block/first-latent **sink**。对 NNL(Not-same)。
_Avoid_: 把 SNL 当普通 LLM 的单份 prompt KV cache

**Philoflow**:
`~/Projects/Philoflow-monorepo`：视频扩散 / 流匹配栈。`philoflow-core` 用 accelerate + FSDP 训练上述模型（有 16-GPU 配置）；另有 inference-gateway / runtime 等推理服务子仓。用来讲 FSDP/ZeRO 显存分片、激活重计算、序列并行、流式 KV cache、推理服务侧通讯与显存。
_Avoid_: 把 Philoflow 说成 LLM 训练栈（它是视频扩散栈；TP/PP 在此未采用，需说明原因）
