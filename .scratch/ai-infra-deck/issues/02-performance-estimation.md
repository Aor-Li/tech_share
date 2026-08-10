# 02 — ① 性能估计基础章

**What to build:** 显存占用 + 计算量/时间估计一章。用经典 LLM、MAGI（hidden 5120/40层/head_dim 128）、LTX-2（48层）三个真实模型做显存三项分解（模型/激活/KV），每项给完整数字例子；用 roofline 直觉图解释算力/带宽约束的判断。先出 Markdown 大纲（要点+图表清单+具体数字）供作者确认，再转成 `deck/chapters/01-performance-estimation.html`，最后并入 `index.html`。

**Blocked by:** 00 — 视觉与骨架基础设施

**Status:** done

- [x] `.scratch/ai-infra-deck/02-performance-estimation/outline.md` 大纲经作者确认，数字可复算、量纲自洽
- [x] 显存三项分解覆盖经典LLM + MAGI + LTX-2，各有完整数字例子
- [x] roofline 图给出算力/带宽约束的直觉解释
- [x] `deck/chapters/01-performance-estimation.html` 与共享主题同构，可独立打开
- [x] 已并入 `index.html`，图表数字与大纲一致
