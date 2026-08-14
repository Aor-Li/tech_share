# tech_share

## Deck content

每节的内容与展示同址存放于 `content/vN/NN-<slug>/`（`content.md` + `slide.html`），
共享 `content/theme.css` + `content/nav.js`。交付物是根自包含 `index.html`。见
`content/README.md`。

**内容默认走原理 + 直觉优先**（定量机制留公式 + 拇指法则 + 锚点数字、砍代入演算；
定性机制用一图/一比喻 + 一句 philoflow 落点）——缺省手感、非硬性规范，细节取舍以作者判断为准。

## specs 工作流（修订草稿 → 定版）

**内容的唯一来源是 `specs/`，不是上一版 `content/vN`。**

写或重写某一节的 `content/vN/`（`content.md` + `slide.html`）时：

- 结构、章节边界、论点顺序、图表意图、该出现哪些模型/数字——一律跟当前 `specs/<slug>/`。
- `content/v1`、`v2`……是历史快照，用来回看，**默认不继承**。不要把旧版的叙事主线、小节划分、开场钩子、或把别的 spec 里的材料（例如总览的实例表）搬进本章。
- 只有 `specs/` 里出现 `- [REF] …` 时，才去复用旧版的图、数据口径或组件样式；没有 `[REF]` 就当旧版不存在。
- slug 不要混：`specs/c0/` 是导览/总览，`specs/c1/` 是第一章「计算」，依此类推。落盘到 `content/vN/NN-<slug>/` 时对上号，禁止把 c0 的内容写进 c1 的目录。

一节内容在定版前，反复修订的中间过程记录在 `specs/<slug>/`：

- `outline.md` —— 大纲草稿。`content.md` —— 内容草稿。
- 文件里的**正文 Markdown 是要保留进产出的内容**；`- [关键字] ...` **标记行是作者给 AI 的修改意见**
  （见下"标记约定"），不是内容本身，不会进入任何交付版本。
- 修订是反复进行的：同一份 `outline.md` / `content.md` 会被多轮标记 + 改写，收敛出确认要保留的内容。

每完成一轮修订、内容被确认保留后，落一个版本快照到 `content/vN/<slug>/content.md`
（+ `slide.html`，同 `content/README.md` 的生命周期）：N 从 1 开始，每轮修订对应的确认稿
递增一个版本目录，旧版本保留不删，可回看历史。`content/vN/` 下的文件是纯净产出，不带 `specs/`
里那种修改意见标记行。

### 标记约定（写 `outline.md` / `content.md` 时）

`specs/` 里的内容分三类，靠**位置**（正文 vs 注释 vs 标题）区分，不靠额外语法糖：

1. **正文 Markdown** —— 保留进产出的内容。AI 生成/润色时可以改措辞、调语序、
   补过渡句，但不能改事实、数字、结论方向——正文即事实来源。
2. **`- [关键字] 说明` 标记行** —— 作者给 AI 的修改指令，不进入交付物。
   不用 `<!-- -->` 注释，因为多数编辑器的预览/高亮会把 HTML 注释隐藏或灰掉，扫一遍正文时容易漏看；
   用普通列表项就能在任何编辑器里正常显示，靠**行首方括号关键字**（不是内容列表会用的写法）
   和普通要点列表区分开。紧贴在它作用的段落之前或之后：
   - `- [NOTE] 说明` —— 背景/上下文，仅供理解，不要求改动
   - `- [EDIT] 说明` —— 紧邻的段落需要按说明重写
   - `- [ADD] 说明` —— 在此处插入新内容（说明写清楚要点，AI 起草）
   - `- [CUT] 说明` —— 删除紧邻的段落/小节
   - `- [REF] 路径或说明` —— 引用已有资源，见下
   - 旧文件里残留的 `<!-- -->` 裸注释（如 `c0/content.md`）等价于 `[NOTE]`，历史遗留不必回填，新写的一律用方括号标记行。
3. **图表设计** —— 是要保留进大纲的正文，不是指令，用标题标出：
   - `### 📊 静态图 · <标题>` / `### 📊 静态表 · <标题>`
   - `### 🎛️ 动态图 · <标题>`
   - 标题下用要点列表写清楚：画什么/取什么轴、怎么着色或分组、教学目标是什么。
     AI 据此起草 `slide.html` 里的实现，不替它决定设计意图。
   - 关键结论用 `>` blockquote，`⭐` 前缀标全章最重点、`⚠️` 前缀标最值得对照警示的一条——已在
     `specs/c1/outline.md` 里用开，直接沿用，不用再造新符号。

**引用已有资源**：`- [REF] <路径或说明>` 是**唯一**允许从旧版 `content/vN` 取材的入口（图 / 数据口径 / 配色 / 组件）。没有这条标记，就按 `specs/` 正文和图表标题重新做，不要自行对照旧稿「补全」。例："`- [REF] 配色同 content/v1/05-memory-compute-optimization/slide.html 的显存条形图`"。

```markdown
- [NOTE] 这段是给听众的心理预期，语气要轻
浏览大模型训推过程中的 infra 基础，形成简单认知。

- [ADD] 补一段解释为什么选这两个模型做参照，突出规模差异

### 📊 静态图 · 参考模型对比
- Qwen3-32B 与 MiniMax H3 的参数量/结构并排对比
- 教学目标：建立"稠密 LLM vs 视频 DiT"的直觉锚点

- [REF] 配色沿用 content/v1/00-panorama/slide.html 的模型卡片样式
```

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
