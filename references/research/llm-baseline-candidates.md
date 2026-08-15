# 经典 LLM 对比模型候选调研

## 调研问题

为以 MiniMax H3 为主要实例的分享选择一个经典 LLM 对比模型。这里的“更合适”不是指模型能力更强，而是更适合向主要从事大模型训练与数据、仅简单了解 Infra 的听众解释计算、显存、并行和训练效率。

评估维度包括：与 MiniMax H3 对比时的规模可比性、经典 dense decoder-only 架构的代表性、听众认知度、官方材料的可追溯性、训练 Infra 材料丰富度，以及是否会引入与主线无关的架构复杂度。

## 候选比较

| 候选 | 适合之处 | 主要问题 | 判断 |
| --- | --- | --- | --- |
| **Qwen2.5-32B** | 32B 规模对齐；官方明确归类为 dense decoder-only；国内听众熟悉；Apache 2.0；没有混合思考模式这一额外叙事 | 官方训练 Infra、集群拓扑与 MFU 资料不如 OLMo 2 透明；与 Qwen3-32B 的基础设计差异不足以抵消型号较旧的表达劣势 | 可作为更纯粹的备用基线，但不必为此替换 Qwen3-32B |
| **Qwen3-32B** | 32B dense 模型；64 层、GQA、上下文等规格清晰；国内认知度高；与 Qwen2.5-32B 的核心设计接近，主体计算框架无需改变 | 已不是 Qwen 最新系列；后训练模型支持 thinking / non-thinking，需要避免让推理行为干扰基础 Infra 主线 | **最适合担任本报告全文可见的经典 LLM 对照** |
| **OLMo 2 32B** | 规模完全对齐；代码、数据、检查点、日志和训练配方开放；官方披露 160 节点、每节点 8×H100、超过 1800 tokens/s/GPU、约 38% MFU；还详述 FSDP/HSDP、激活重计算、异步 checkpoint 和网络拓扑问题 | 中文听众认知度较低；模型主要面向英文；若把它设为全文主角，需要先花时间介绍背景 | **最适合担任 Infra 事实与案例来源**，但不一定适合取代 Qwen 成为听众侧主对照 |
| Llama 3.1 8B / 70B | 经典 dense Transformer，认知度高；官方论文披露大规模训练与基础设施经验 | 没有约 32B 的型号：8B 太小，70B 又超过两倍，显存与并行比较容易被规模差异干扰；使用自定义许可证 | 适合补充行业案例，不适合本分享的一对一主对照 |
| Gemma 3 27B | 规模接近；官方材料较完整 | 原生多模态，并采用 local/global attention 等设计，KV cache 与推理讨论会混入额外变量 | 不建议作为“经典 LLM”基线 |
| Mistral Small 3 24B | Apache 2.0；规模尚可；低延迟定位清晰 | 规模偏小，公开训练 Infra 与训练过程材料相对有限 | 没有明显优于 Qwen2.5-32B 的理由 |

## 结论

不存在一个在所有维度都优于 Qwen3-32B 的单一替代模型。结合报告的听众、讲解目标和型号新旧，最终采用以下分工：

1. **保留 Qwen3-32B 作为全文可见的经典 LLM 对照。** 它与 Qwen2.5-32B 同为 64 层、5120 隐藏维度的 32B dense decoder-only 模型，计算量、参数显存、激活显存和并行策略的主体讲解框架基本一致。使用更新的 Qwen3 型号更符合报告表达，不需要为了少量架构差异换回 Qwen2.5。
2. **不把 Qwen3-32B 表述为“Qwen 最新模型”。** Qwen 后续已有 Qwen3.5 和 Qwen3.6；但接近这一规模的新型号引入原生多模态、混合线性注意力或 MoE 等额外变量，不适合作为经典 dense LLM 基线。Qwen3-32B 的选择依据是“较新的标准 32B dense LLM”，而不是追逐最新版本。
3. **将 thinking / non-thinking 限制在相关推理内容中。** 计算与训练 Infra 主线只使用 Qwen3-32B 的基础模型结构；只有讲到 reasoning 输出、服务框架或推理预算时，才介绍混合思考模式。
4. **将 OLMo 2 32B 作为计算与 Infra 章节的事实型参考模型。** 它不成为与 MiniMax H3 并列的第三个叙事主角，只在 MFU、集群规模、并行、checkpoint、网络拓扑等页面中提供公开案例。

最终组合为：**MiniMax H3（主要实例） + Qwen3-32B（经典 LLM 对照） + OLMo 2 32B（Infra 公开案例来源）**。

## 官方来源

- [Qwen2.5 官方发布说明](https://qwenlm.github.io/blog/qwen2.5/)：模型尺寸、dense decoder-only 定位与许可证。
- [Qwen2.5-32B 官方模型配置](https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json)：层数、隐藏维度、注意力头和上下文配置。
- [Qwen3 官方发布说明](https://qwenlm.github.io/blog/qwen3/)：32B dense 模型规格及 hybrid thinking modes。
- [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)：训练与模型族技术背景。
- [Qwen3.6-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/README.md)：后续 dense 型号的多模态与混合注意力架构。
- [Qwen3.6-35B-A3B 官方发布说明](https://qwen.ai/blog?id=qwen3.6-35b-a3b)：后续同规模 MoE 型号及激活参数量。
- [OLMo 2 32B 官方发布说明](https://allenai.org/blog/olmo2-32b)：训练集群、吞吐、MFU、并行与工程问题。
- [OLMo 2 32B 官方模型卡](https://huggingface.co/allenai/OLMo-2-0325-32B)：模型规格、训练 token、开放材料和许可证。
- [OLMo-core 官方仓库](https://github.com/allenai/OLMo-core)：训练实现和官方 32B 训练脚本。
- [Llama 3.1 官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)：8B/70B/405B 型号、GQA 与训练信息。
- [Llama 3 技术报告](https://arxiv.org/abs/2407.21783)：模型架构与大规模训练基础设施。
- [Gemma 3 官方模型卡](https://ai.google.dev/gemma/docs/core/model_card_3)：多模态能力及模型规格。
- [Mistral Small 3 官方模型说明](https://docs.mistral.ai/models/model-cards/mistral-small-3-0-25-01)：24B 模型定位与许可信息。

## 使用边界

本调研主要评估教学结构与可复用材料，不以模型 benchmark 排名决定选择。MiniMax H3 的内部架构与训练信息仍应以团队材料为准；未公开的信息不应从外部资料反推。
