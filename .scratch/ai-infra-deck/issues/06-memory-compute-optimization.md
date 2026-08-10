# 06 — ⑤ 显存与计算优化章

**What to build:** 激活重计算(checkpoint)、混合精度、offload 等典型显存优化；FlashAttention 基础（tiling/IO-aware/不物化 N×N）+ MagiAttention flex-flash-attn（FA-3+灵活掩码+分布式）进阶；KV cache 结合 Philoflow SNL 流式讲无限长生成，并对比 LTX-2（文本交叉注意力 K/V 依赖 sigma 故不缓存、时序 K/V 缓存）与 minimax h3，展示 KV cache 大小/策略如何随架构变化。先出 Markdown 大纲（要点+图表清单+具体数字）供作者确认，再转成 `deck/chapters/05-memory-compute-optimization.html`，最后并入 `index.html`。

**Blocked by:** 05 — 训练并行范式章

**Status:** ready-for-agent

- [ ] `.scratch/ai-infra-deck/06-memory-compute-optimization/outline.md` 大纲经作者确认，数字可复算
- [ ] 覆盖激活重计算/混合精度/offload
- [ ] FlashAttention 基础 + MagiAttention flex-flash-attn 进阶都讲到
- [ ] KV cache 结合 Philoflow SNL 讲流式无限长，并做 LTX-2 vs minimax h3 对比，minimax h3 相关内容标注为公开信息推断
- [ ] `deck/chapters/05-memory-compute-optimization.html` 与共享主题同构，可独立打开
- [ ] 已并入 `index.html`，图表数字与大纲一致
