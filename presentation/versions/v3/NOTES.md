# v3

## 版本元数据

| 项目 | 值 |
| --- | --- |
| 内容基线 | `7b1577d` |
| 包含章节 | 01、02、03（设计预览；包含未确认正文） |
| 快照状态 | 未同步 |
| 来源版本 | v2 |

## 本版本意图

以 v2 的版式、字号、配色、导航与现场显示效果为设计基线，逐章选择性吸收 v1 中更好的内容组织与图表表达，形成可与 v1、v2 并排比较的候选整合版。

## 优化记录

| 轮次 | 改动 | 触发原因 | 是否更换内容基线 |
| --- | --- | --- | --- |
| 1 | 完整复制 v2 的 48 页报告并更新版本身份；尚未吸收 v1 内容或图表 | 用户决定保留 v1、v2 作为对照，并创建 v3 作为候选整合版 | 否 |
| 2 | 第 1 页封面移除“算法与数据同学”“30–60 分钟”“v3 · 整合预览”三个标签 | 逐页精简封面信息 | 否 |
| 3 | 第 2 页删除原有流程与三本账卡片，改为可点击、可键盘聚焦的三章节交互路线图 | 导览页只保留美观图示和章节跳转功能 | 否 |
| 4 | 第 3 页移除章节开场副标题，仅保留章节号与主标题 | 精简章节开场信息 | 否 |
| 5 | 第 4 页保留 v3 的视觉系统，将三张并列卡片重构为“计算对象 → 三本账 → 估算与实测边界”的层级图 | 参考 v1 第 4 页的内容结构，增强输入条件与三类资源问题的因果关系 | 否 |
| 6 | 第 4 页顶部对象改为“Kernel / Block / Layer / Model”，删除标题下引导句与图后结论句，将三本账统一为低饱和度蓝、黄绿、珊瑚色块与实色指标标签 | 根据内容负责人对术语、信息密度与配色的逐页反馈 | 否 |
| 7 | 第 4 页三个指标标签统一为全大写：“FLOPS”、“BYTES”、“BYTES / CALL” | 统一并列指标的大小写视觉规则 | 否 |
| 8 | 第 5–8 页保留 v3 导航、字号和蓝/黄绿/珊瑚配色，迁移并重绘 v1 第 5–9 页的三类操作、Self-Attention 数据流与 T/T² 增长图、2P/6P/6PD 分解和 Qwen3/H3 实例条形图；将五页参考内容压缩为四页连续学习路径 | 用户要求参考 v1 第 5–9 页修改 v3 第 5–8 页，并强调风格迁移与排版优化 | 否 |
| 9 | 第 5 页三列公式统一为同尺寸、同基线的彩色条，Reduction 将“≈ n FLOPs”与“depth ≈ log n”合并；三列说明文字保持同一水平基线 | 用户要求第 5 页三列公式与说明文字分别水平对齐 | 否 |
| 10 | 将原第 6 页拆成两页：第 6 页按 v1 的结构重绘完整 Self-Attention 数据流，第 7 页单独呈现投影线性项、Attention 二次项及 2048 token 示例；公式统一使用 batch、token、hidden、heads 完整变量名 | 用户认为 v1 的两页展示更直观，尤其希望恢复 v1 第 6 页的 Attention 结构，并避免变量名简写 | 否 |
| 11 | 第 5 页底部由单一 GEMM backward 示例改为三列 backward 对照：GEMM 明确列出 FWD 2mkn、BWD 4mkn、训练合计 6mkn；Elementwise 与 Reduction 分别标注前后向的线性量级及算子相关常数 | 用户要求补齐三类操作的 backward 计算量说明 | 否 |
| 12 | 第 6–7 页公式改回 B、T、H、h 简写并在两页页脚统一释义；节点名称与叙述保留完整单词，同时放大公式字号、减少不必要换行 | 完整变量名使 Attention 结构图过于拥挤，内容负责人同意在放不下时使用简写 | 否 |
| 13 | 第 6 页流程区按照 v1 第 6 页的方向重新排版：输入向右分支到 Q/K/V，Q/K 汇入 score，softmax 向下连接 prob×Value，Value 横向汇入后继续向右输出；保留原底部合计条，并将公式放入节点或独立留白区以避免与连线重叠 | 内容负责人要求基于 v1 Attention 流程结构调整 v3，同时保持合计部分不变 | 否 |
| 14 | 第 6 页在流程图中补充 QK 到 softmax 的计算量推导：令单头维度 d=H/h，QK 由 BhT² 个长度 d 的点积得到 2BT²H；score 共 BhT² 个元素，缩放、mask 各遍历一次，softmax 将 max、减最大值、exp、sum、归一化粗估为五遍，合计约 7BhT² | 内容负责人要求说明 QK 到 softmax 各项计算量的来源 | 否 |
| 15 | 第 6 页移除 QK 到 softmax 的两行额外推导；将输入、score 与输出 shape 中表示乘法的点号改为紧凑写法，并用冒号分隔标签和公式 | 内容负责人决定不在流程图中增加该推导，并要求减少公式点号以避免显示冲突 | 否 |
| 16 | 修正第 5 页三类操作图示：GEMM 输出矩阵改为与 m×n 形状一致，Elementwise 输出恢复为与输入相同的 3×4 网格；底部栏下移，并统一改为三列 FWD+BWD 合计口径 | 内容负责人要求优化第 5 页，重点修正图式错误，并尝试将合计栏布局到页面下方 | 否 |

## 正文—页面映射

以下映射以当前 `content/` 三章文件为唯一内容源；图解只用于解释对应结论，不替换、外推或删改正文边界。

| 正文小节 | 页面 ID | 覆盖内容 |
| --- | --- | --- |
| 01 · 1.1 | `three-ledgers` | 现实问题、三本账、单位与估算边界 |
| 01 · 1.2 | `compute-ops`、`attention-breakdown`、`attention-growth`、`compute`、`compute-example` | GEMM/Elementwise/Reduction、前反向、Self-Attention 数据流与形状增长、2P/6P/6PD、Qwen3/H3 实例 |
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
| 2 | 离线打开、目录与逐页导航 | 已做静态检查；目录、按钮、方向键、PageUp/PageDown、Home/End、M/Escape 均已实现；本轮复核第 6–10 页目录跳转、连续翻页和新增页后的 49 页计数正常 |
| 3 | 16:9 无溢出或信息墙 | 已按 1380 × 840 自适应画布、49 页大字号拆页和图解优先策略重新编排；第 5 页已在 1280 × 720 下复核矩阵维度、Elementwise 前后形状、Reduction 归约层级和下移后的三列合计栏，第 6–7 页也已完成同尺寸复核；第 6 页按 v1 方向重排后再次确认 Q/K 汇合、Value 横线、softmax 下行箭头、公式与节点之间无重叠、遮挡或异常换行；待内容确认后做全篇最终验收 |
| 4 | 打印分页与静态降级 | 已实现 landscape 分页；图表均为内嵌 SVG 或 HTML/CSS 静态结构；本轮新增的第 6–7 页视觉均为无交互依赖的内联 SVG |
| 5 | 无待填写或调试标记 | 通过 |
| 6 | 署名与高风险信息标记 | 重绘图例页脚保留论文/官方资料署名，来源登记于 REF-008/009；待内容确认后做最终核验 |

## 遗留项

- 按章节比较 v1 与 v2，建立需要吸收的内容、图表和页面拆分清单。
- 在不破坏 v2 视觉系统的前提下，逐章完成选择性整合并记录每轮变化。
- 01.2、02、03 正文确认并提交后，更新内容基线并重新核对逐页表述。
- 内容确认后再执行浏览器视觉验收，并决定是否将快照状态升级为“已同步”。
