# 参考资料索引

## 创作参考

| 编号 | 参考资料 | 相关章节 | 可借鉴内容 | 采纳状态 |
| --- | --- | --- | --- | --- |
| REF-001 | [How to Scale Your Model](articles/how-to-scale-your-model/README.md)（[Google DeepMind 原文](https://jax-ml.github.io/scaling-book/)） | 01 计算；后续性能与并行章节 | Roofline 叙事、TPU/GPU 介绍、分片和训练/推理章节组织 | 待形成参考对照 |
| REF-002 | [大型语言模型的推理演算](articles/llm-inference-arithmetic-zh/README.md)（[OneFlow 中文编译](https://zhuanlan.zhihu.com/p/620170671)） | 01 计算；后续推理章节 | 从模型结构推导容量、时延、KV cache 和通讯成本的方法 | 待形成参考对照 |
| REF-003 | [Transformer Math 101](articles/transformer-math-101/README.md)（[EleutherAI 原文](https://blog.eleuther.ai/transformer-math/)） | 01 计算；后续显存章节 | 训练算力、显存、混合精度和重计算的讲解顺序与公式组织 | 待形成参考对照 |
| REF-004 | [The Ultra-Scale Playbook](articles/ultra-scale-playbook/README.md)（[Hugging Face / nanotron 原文](https://huggingface.co/spaces/nanotron/ultrascale-playbook)） | 后续通讯与并行章节 | 数据、张量、上下文、流水线和专家并行的结构与交互图思路 | 待形成参考对照 |

## 事实依据

| 编号 | 事实依据 | 相关章节 | 可核验主张 | 具体定位 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| FACT-001 | `codes/philoflow-monorepo` | 后续实践章节 | 代码实现及固定提交中的实际配置 | 当前只记录提交 `74124a2`，缺少 `.gitmodules` 上游映射 | 待核验 |
| FACT-002 | [Qwen3 官方发布说明](https://qwenlm.github.io/blog/qwen3/)；[Qwen3-32B 官方配置](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | 00 总览；01 计算 | Qwen3-32B 是 32B dense 模型；层数、隐藏维度、Attention/KV heads 等基础规格 | 发布说明的模型规格表；配置文件中的 `hidden_size`、`intermediate_size`、`num_*` 字段 | 已核验 |
| FACT-003 | [OLMo 2 32B 官方发布说明](https://allenai.org/blog/olmo2-32b)；[官方模型卡](https://huggingface.co/allenai/OLMo-2-0325-32B) | 01 计算；后续通讯与并行章节 | 32B 模型规格、训练集群、吞吐、MFU，以及并行与 checkpoint 工程案例 | 发布说明的 “Training infrastructure”“OLMo-core Trainer” 与 “Training on Google Cloud Engine”；模型卡规格表 | 已核验 |

当前 `01-computation.md` 中的 Transformer GEMM 占比、H100 峰值、一般化 MFU 区间和实时预算尚未登记对应事实依据，均保持“待核验”；Qwen3-32B 规格与 OLMo 2 32B 公开训练案例已登记。

## 章节参考对照

进入章节大纲设计前，选择约 3–5 篇创作参考完成下表；只记录会影响本章结构和表达的内容。

| 参考编号 | 结构亮点 | 可借鉴内容 | 图表思路 | 与本章关系 | 是否采纳 |
| --- | --- | --- | --- | --- | --- |
| REF-NNN | 待分析 | 待分析 | 待分析 | 待分析 | 待决定 |
