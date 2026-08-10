# 05 — ④ 训练并行范式章

**What to build:** DP/ZeRO、TP、PP、EP 四种主流训练并行模式一章（全场最重，~12 分钟）。DP/ZeRO 用 Philoflow FSDP(=ZeRO-3)、16-GPU 配置举例，并明确说明 Philoflow 未采用 TP/PP 及原因（靠 FSDP 扩展）。EP 用 minimax h3（MoE 视频）+ 经典 MoE LLM 讲，对比 MAGI 的 modality-experts，区分 token-routed experts 与 modality experts、说明 EP 只作用于前者。先出 Markdown 大纲（要点+图表清单+具体数字）供作者确认，再转成 `deck/chapters/04-parallelism-paradigms.html`，最后并入 `index.html`。

**Blocked by:** 04 — 序列维并行章

**Status:** ready-for-agent

- [ ] `.scratch/ai-infra-deck/05-parallelism-paradigms/outline.md` 大纲经作者确认，数字可复算
- [ ] DP/ZeRO 用 Philoflow FSDP 16-GPU 配置举例；明确说明未采用 TP/PP 的原因
- [ ] EP 用 minimax h3 + 经典 MoE LLM 讲，并与 MAGI modality-experts 做对比，区分 routed vs modality experts
- [ ] `deck/chapters/04-parallelism-paradigms.html` 与共享主题同构，可独立打开
- [ ] 已并入 `index.html`，图表数字与大纲一致
