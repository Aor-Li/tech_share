# v2

## 版本元数据

| 项目 | 值 |
| --- | --- |
| 内容基线 | `7b1577d` |
| 包含章节 | 01、02、03（设计预览；包含未确认正文） |
| 快照状态 | 未同步 |
| 来源版本 | v1 |

## 本版本意图

基于当前 `content/` 制作一版可完整浏览的报告设计预览，用更克制的版式、统一的资源账本语言和场景化图表串联资源直觉、训练与推理。

## 优化记录

| 轮次 | 改动 | 触发原因 | 是否更换内容基线 |
| --- | --- | --- | --- |
| 1 | 新建 28 页逐页报告，补齐目录、键盘导航、移动端与打印降级 | 用户要求基于当前 content 创建简洁美观的 v2 展示网页 | 是 |
| 2 | 调整为 35 页现场大字号版：基础字号提升到 15.5–19 px，并拆分计算实例、Checkpointing、FSDP、序列/专家切分、指标口径、Prefill/Decode 和推理多卡等高密度页面 | 用户反馈原版字体过小，现场观看效果不足 | 否 |
| 3 | 将 Roofline、显存生命周期、通信层次、Checkpointing、ZeRO/FSDP、TP/PP、Ring/Ulysses、EP、Prefill/Decode、量化与融合、Speculative、Continuous Batching、Paged KV 和推理扩展等页面改为论文/官方资料启发的 SVG 图解，并在页脚署名 | 用户要求减少报告文字，优先使用网络经典图例展示技术方案 | 否 |
| 4 | 严格按 `content/01-resource-intuition.md`、`02-training.md`、`03-inference.md` 逐节复核并扩展为 48 页；补回计算操作分类、Attention 模块分析、GPU 存储层级、时间双下界、物理路径、collective 语义、训练速度口径、输入利用率、单请求优化四问、prefix reuse、chunked prefill、基准设计与停止条件 | 用户指出图解版没有遵循三章 Markdown 已确认内容 | 否 |

## 正文—页面映射

以下映射以当前 `content/` 三章文件为唯一内容源；图解只用于解释对应结论，不替换、外推或删改正文边界。

| 正文小节 | 页面 ID | 覆盖内容 |
| --- | --- | --- |
| 01 · 1.1 | `three-ledgers` | 现实问题、三本账、单位与估算边界 |
| 01 · 1.2 | `compute-ops`、`compute`、`attention-breakdown`、`compute-example` | GEMM/Elementwise/Reduction、前反向、Self-Attention、2P/6P/6PD、Qwen3/H3 实例 |
| 01 · 1.3 | `memory-hierarchy`、`roofline`、`time-bound`、`memory-life` | H100 存储层级、算术强度、计算/带宽双下界、峰值生命周期 |
| 01 · 1.4 | `communication`、`physical-paths`、`collectives` | 三层模型、节点内/跨节点路径、点对点与 collective、张量布局 |
| 01 · 1.5 | `four-questions` | 四问资源判断卡与可验证假设 |
| 02 · 2.1 | `train-step`、`train-metrics` | 一次参数更新、global batch、显存/时间账、step time/吞吐/MFU/HFU |
| 02 · 2.2 | `single-gpu`、`checkpoint`、`input-utilization` | 混合精度、累积、Activation Checkpointing、padding/packing/数据供给 |
| 02 · 2.3–2.8 | `split-map`、`state-sharding`、`fsdp-life`、`tp`、`pp`、`sequence-expert`、`expert-parallel` | DP/ZeRO/FSDP、TP、PP、SP/CP/Ring/Ulysses、EP 的切分对象与数据流 |
| 02 · 2.9 | `train-choice` | 资源交换表、选择顺序与单变量验证 |
| 03 · 3.1 | `metrics`、`metric-definitions` | 在线/离线目标、TTFT/TPOT/ITL/吞吐/稳定并发口径 |
| 03 · 3.2 | `prefill-decode`、`prefill-compare` | 一次请求流程、KV 生命周期与阶段特征 |
| 03 · 3.3 | `optimization-four-questions`、`single-request`、`speculative` | 量化、kernel/fusion/编译、speculative decoding 的统一四问 |
| 03 · 3.4 | `continuous`、`kv-pool`、`prefix-reuse`、`chunked-prefill` | 同一 A–E 请求组、continuous batching、paged KV、prefix reuse、chunked prefill |
| 03 · 3.5 | `serving-scale`、`serving-sharding` | Replicas 与 TP/PP 的容量、关键路径和适用场景 |
| 03 · 3.6 | `inference-validate`、`benchmark-design`、`validation-order` | 假设、单请求/服务基准、五步读结果与停止条件 |

## 验收检查

对应 `spec/README.md` 展示网页验收项 1–6。

| 序号 | 检查项 | 结果 |
| --- | --- | --- |
| 1 | 章节与基线一致 | 当前三章内容覆盖检查通过，并记录逐节映射；但 01.2 待确认、02/03 正文仍标为草拟，因此本版尚未通过“已同步”交付门槛 |
| 2 | 离线打开、目录与逐页导航 | 已做静态检查；目录、按钮、方向键、PageUp/PageDown、Home/End、M/Escape 均已实现 |
| 3 | 16:9 无溢出或信息墙 | 已按 1380 × 840 自适应画布、48 页大字号拆页和图解优先策略重新编排；待内容确认后做最终视觉验收 |
| 4 | 打印分页与静态降级 | 已实现 landscape 分页；图表均为内嵌 SVG 或 HTML/CSS 静态结构 |
| 5 | 无待填写或调试标记 | 通过 |
| 6 | 署名与高风险信息标记 | 重绘图例页脚保留论文/官方资料署名，来源登记于 REF-008/009；待内容确认后做最终核验 |

## 遗留项

- 01.2、02、03 正文确认并提交后，更新内容基线并重新核对逐页表述。
- 内容确认后再执行浏览器视觉验收，并决定是否将快照状态升级为“已同步”。
