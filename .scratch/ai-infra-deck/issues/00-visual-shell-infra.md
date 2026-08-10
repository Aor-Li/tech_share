# 00 — 视觉与骨架基础设施（prefactor）

**What to build:** 打通"章节片段 → 母文件"的全套合并机制：Catppuccin Mocha 深色主题的共享 CSS 变量（base `#1e1e2e`，主强调 Mauve `#cba6f7`，次强调 Teal `#94e2d5`；正文无衬线、术语/数字/代码等宽）、SVG 图表的统一线性风格与组件类名规范、`index.html` 骨架（键盘翻页、页码、章节导航），以及各章片段的模板 `deck/chapters/NN-<slug>.html`（同构、可独立打开、可零成本并入母文件）。用 1-2 个占位页跑通端到端流程：`index.html` 在浏览器打开能看到占位内容、键盘翻页可用、片段单独打开也能正确套用主题。

**Blocked by:** None — can start immediately

**Status:** done

- [x] `index.html`（仓库根）存在，自包含、无需构建即可打开
- [x] 共享 CSS 变量集中定义（Mocha 调色板 + 排版），后续所有章节片段复用同一套变量，不做逐章样式分叉
- [x] `deck/chapters/` 下有片段模板，片段可独立在浏览器打开并正确渲染共享主题
- [x] 母文件支持键盘翻页与页码显示
- [x] 占位页验证了"片段拼接进母文件、无冲突"的合并机制
