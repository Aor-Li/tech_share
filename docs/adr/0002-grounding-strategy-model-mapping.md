# 落地素材映射策略：各取所长，合理处尽量扩展到 Philoflow

**重要更正**：作者手边的三个模型（MAGI、LTX-2、minimax h3）**全是视频生成模型**，唯一的 LLM 是"经典 LLM"。这重塑了映射——尤其 EP 现在有视频落点（minimax h3 是 MoE 视频模型），KV cache 是 Philoflow 流式推理的核心（不是"不存在"）。

每个概念挂到最自然的那个模型上，而非强行统一：

- **经典 LLM（Llama 类 dense decoder）** → 主线基准：FLOPs、KV cache、TP、PP。
- **minimax h3（Hailuo 3.0，flow-matching DiT + token-routed MoE 视频模型）** → **EP（专家并行）的视频落点**、显存/超长序列对比。
- **MAGI（视频 DiT，5120/40 层，modality-experts + AR 流式）** → FSDP/ZeRO、激活重计算、序列并行、AR 流式 KV cache；modality-experts vs routed-experts 的对比点。
- **LTX-2（视频 DiT，48 层，sink+sliding-window KV cache）** → KV cache 精确落点、显存估计。

**规则**：合理处尽量把概念扩展到 Philoflow（作者更熟、更有共鸣）；明显牵强时（TP/PP —— Philoflow 未采用张量/流水并行，靠 FSDP 扩展）**明确说明"为什么不适用/未采用"**，而不硬凑。

**EP 的两种"专家"要讲清**：token-routed MoE（minimax h3 / 经典 MoE LLM 如 DeepSeek/Mixtral，EP 作用于此）vs modality experts（MAGI 的 `NativeMoELinear` 按模态分投影，不是 EP 对象）。

**FlashAttention 章**：保留 FA 基础介绍（tiling / IO-aware / 不物化 N×N），并加 **MagiAttention 的 flex-flash-attn**（FA-3 + 灵活掩码 + 分布式）作为进阶落点；它同时呼应序列并行章（是与 Ulysses/Ring 并列的 CP 方案）。

**KV cache 章**：结合 Philoflow 的 **SNL 流程**讲流式无限长；并做 **LTX-2 vs minimax h3** 的 KV 对比 —— LTX-2 缓存时序自注意力 K/V（分 chunk + sink + 滑窗）与 a2v/v2a 交叉 K/V，但**文本交叉注意力 K/V 依赖 sigma(噪声级)故从不缓存、每 chunk 重算**（与 LLM 缓存一次 prompt 相反）；minimax h3 则以 omni-modal 联合 video+audio、2K/15s 的更大 token 量、MoE 不增 KV 的角度对比（h3 内部为公开信息推断，非代码）。

**为什么**：初始假设 minimax h3 是 MoE LLM、Philoflow 无 KV cache，均已被作者纠正并经仓库核实。故采用"各取所长 + 合理扩展 + 牵强处说明原因"，并以一张"概念 × 模型"对照表逐格确认。
