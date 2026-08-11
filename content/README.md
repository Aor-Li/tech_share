# content/ — 逐节内容与展示同址存放

这份报告的每一节，把 **内容（Markdown 大纲）** 和 **展示（HTML 片段）** 放在同一个
`content/NN-<slug>/` 目录下，方便对着改内容、调风格时两者始终一致、不再散落两处。

## 布局

```
content/
  theme.css        # 共享主题（Catppuccin Mocha），CSS 变量 + 组件类的唯一真源
  nav.js           # 共享键盘翻页控制器
  _template.html   # 新节骨架：复制到 content/NN-<slug>/slide.html 起手
  NN-<slug>/
    content.md     # 本节 Markdown 大纲（要点 + 图表清单 + 量级数字）
    slide.html     # 本节 HTML 片段（可独立打开，套用共享主题）
```

七节，按 `docs/adr/0001` 的依赖链排序：

| 目录 | 内容 |
|------|------|
| `00-panorama` | 全景地图开场 |
| `01-performance-estimation` | 性能估计基础（显存 / 算力 / roofline） |
| `02-communication-primitives` | 通讯原语与 α-β 代价模型 |
| `03-sequence-parallelism` | 序列维并行（Ulysses / Ring / MagiAttention） |
| `04-parallelism-paradigms` | 训练并行范式（DP/ZeRO · TP · PP · EP） |
| `05-memory-compute-optimization` | 显存与计算优化（重计算 · 混合精度 · offload · FlashAttention · KV cache） |
| `06-real-deployment` | 组合成真实部署（Philoflow 16-GPU / minimax h3） |

## 一节的生命周期

1. 改 `content.md` —— 内容大纲先行，数字在此定稿。
2. 改同址 `slide.html` —— 与大纲同构的展示；`../theme.css` + `../nav.js` 解析到
   `content/` 根，独立打开即可预览本节。
3. 把 `slide.html` 里的 `<section class="slide">` 拷进根 `index.html` 对应位置
   （见其 `<!-- NN — … (content/NN-<slug>/slide.html) -->` 标记），零样式改动。

## 边界

- **交付物** 仍是根 `index.html`（自包含单文件，ADR 0003），它把 `theme.css` +
  `nav.js` 内联、把各节 `<section class="slide">` 拼接进来。
- **样式** 只改 `content/theme.css`，不在单节里加 per-section 覆盖，保证合并后零分叉。
- **工作项**（spec / issues）在开发完成后已移除；其历史保留在 git 中。若开启新一轮开发，
  按 `docs/agents/issue-tracker.md` 约定在 `.scratch/<slug>/` 下重建即可。
