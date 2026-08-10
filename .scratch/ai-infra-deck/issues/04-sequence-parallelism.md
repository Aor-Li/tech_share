# 04 — ③ 序列维并行章

**What to build:** 对比 Ulysses / Ring-Attention / MagiAttention（flex-flash-attn）三种上下文并行（CP）方案，讲清长序列注意力如何跨卡切分及其通讯差异——建立在上一章的通讯代价模型之上。用长视频 token 序列作为落点，贴近作者真实场景。先出 Markdown 大纲（要点+图表清单+具体数字）供作者确认，再转成 `deck/chapters/03-sequence-parallelism.html`，最后并入 `index.html`。

**Blocked by:** 03 — 通讯原语与代价模型章（并行方案的通讯代价推导依赖该章建立的代价模型）

**Status:** ready-for-agent

- [ ] `.scratch/ai-infra-deck/04-sequence-parallelism/outline.md` 大纲经作者确认，通讯代价数字可复算并与第②章代价模型一致
- [ ] Ulysses / Ring-Attention / MagiAttention 三方案对比清晰，落点为长视频 token 序列
- [ ] `deck/chapters/03-sequence-parallelism.html` 与共享主题同构，可独立打开
- [ ] 已并入 `index.html`，图表数字与大纲一致
