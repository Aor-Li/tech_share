# tech_share

## Deck content

每节的内容与展示同址存放于 `content/NN-<slug>/`（`content.md` + `slide.html`），
共享 `content/theme.css` + `content/nav.js`。交付物是根自包含 `index.html`。见
`content/README.md`。

**内容默认走原理 + 直觉优先**（定量机制留公式 + 拇指法则 + 锚点数字、砍代入演算；
定性机制用一图/一比喻 + 一句 philoflow 落点）——缺省手感、非硬性规范，细节取舍以作者判断为准。

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
