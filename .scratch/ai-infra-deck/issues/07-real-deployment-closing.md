# 07 — 结尾：组合成真实部署页

**What to build:** 报告收尾的"组合成真实部署"页，用 Philoflow 16-GPU（FSDP）/ minimax 举例，串联前面各章讲过的技术（通讯代价、序列并行、DP/ZeRO/EP、优化手段），让听众理解这些技术如何在真实系统里协同，而不是孤立知识点。先出 Markdown 大纲（要点+图表清单）供作者确认，再转成 `deck/chapters/06-real-deployment.html`，最后并入 `index.html`。

**Blocked by:** 06 — 显存与计算优化章

**Status:** done

- [x] `.scratch/ai-infra-deck/07-real-deployment/outline.md` 大纲经作者确认
- [x] 综合举例使用的数字与前面各章大纲一致，不产生新的、未在其他章节出现的数字
- [x] `deck/chapters/06-real-deployment.html` 与共享主题同构，可独立打开
- [x] 已并入 `index.html`，作为全场最后一页
