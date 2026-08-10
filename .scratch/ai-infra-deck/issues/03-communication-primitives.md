# 03 — ② 通讯原语与代价模型章

**What to build:** all-reduce / all-gather / reduce-scatter / all-to-all / p2p 通讯原语，配 α-β 代价模型，作为后续并行章代价推导的统一基础。关键机制（如 ring all-reduce 数据流）配简单分步动画。先出 Markdown 大纲（要点+图表清单+具体数字）供作者确认，再转成 `deck/chapters/02-communication-primitives.html`，最后并入 `index.html`。

**Blocked by:** 00 — 视觉与骨架基础设施

**Status:** ready-for-agent

- [ ] `.scratch/ai-infra-deck/03-communication-primitives/outline.md` 大纲经作者确认，通讯代价数字可复算
- [ ] 覆盖 all-reduce/all-gather/reduce-scatter/all-to-all/p2p 五种原语 + α-β 代价模型
- [ ] ring all-reduce 数据流有简单分步动画
- [ ] `deck/chapters/02-communication-primitives.html` 与共享主题同构，可独立打开
- [ ] 已并入 `index.html`，图表数字与大纲一致
