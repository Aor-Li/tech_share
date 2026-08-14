# 总览（content/v3）

依据 `specs/c0/content.md`。本章只做导览与贯穿实例，不讲算力公式。

---

浏览大模型训推过程中的 infra 基础，形成简单认知。

---

## 导览图

总览（本页）→ 计算 → 后续（随 `specs/` 增加）。

---

## 参考实例对照

列头是模型名；同一行扫两边。

| | Qwen3-32B | MiniMax H3 |
|---|---|---|
| 类型 | 稠密 decoder-only LLM | 视频 DiT（H3-Omni-Transformer） |
| 参数 | 32.8B（非 embed 31.2B） | 33B dense；AdaLN ≈13B 可缓存 → 推理有效 ≈20B |
| 深度 | 64 层 | 50 block |
| hidden | 5,120 | 5,376 |
| Attention | GQA 64/8 | 56×128 heads |
| FFN | SwiGLU 25,600 | SwiGLU |
| 位置 | — | MM-RoPE |
| 序列 | 文本 token；原生 32K，YaRN 至 131K | 潜空间 token；24fps，最长 15s；开源默认 768p、托管可达 2K |
| 精度 | 默认 BF16；官方 FP8（block 128） | 权重 BF16；DiT 可 FP8；VAE / 文本编码器通常更高精度 |

口径：Qwen3 取官方模型卡；H3 取官方模型卡（33B / AdaLN≈13B）及公开权重结构（50 block / hidden 5,376）。
