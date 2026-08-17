# 参考资料索引

## 创作参考

| 编号 | 参考资料 | 相关章节 | 可借鉴内容 | 采纳状态 |
| --- | --- | --- | --- | --- |
| REF-001 | [How to Scale Your Model](articles/how-to-scale-your-model/README.md)（[Google DeepMind 原文](https://jax-ml.github.io/scaling-book/)） | 01–03 | Roofline、分片和训推分析的完整知识地图；主讲只采纳其中面向直觉与量级估算的部分 | 已采纳（01）；02–03 计划采纳 |
| REF-002 | [大型语言模型的推理演算](articles/llm-inference-arithmetic-zh/README.md)（[OneFlow 中文编译](https://zhuanlan.zhihu.com/p/620170671)） | 01 资源直觉；03 推理 | 01 采纳推理按前向 `2P` 估、逐步生成仍读一遍权重但时间常受搬运限制；03 继续用容量、时延和 KV cache 方法 | 已采纳（01）；03 计划采纳 |
| REF-003 | [Transformer Math 101](articles/transformer-math-101/README.md)（[EleutherAI 原文](https://blog.eleuther.ai/transformer-math/)） | 01 资源直觉；02 训练 | 训练计算、显存、混合精度和重计算的简化讲解顺序 | 已采纳（01）；02 计划采纳 |
| REF-004 | [The Ultra-Scale Playbook](articles/ultra-scale-playbook/README.md)（[Hugging Face / nanotron 原文](https://huggingface.co/spaces/nanotron/ultrascale-playbook)） | 02 训练 | 数据、状态、张量和序列切分的动机与交互图思路；组合并行细节不进入主讲 | 计划采纳 |
| REF-005 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)；[FlashAttention 原论文](https://arxiv.org/abs/2205.14135) | 01 资源直觉；02 训练；03 推理 | 01 用两层内存直觉区分容量与数据移动；02/03 只判断 IO-aware attention 的适用性 | 已采纳（01）；02–03 计划采纳 |
| REF-006 | [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/index.html)；[Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) | 01 资源直觉；02 训练 | 用张量输入输出解释核心通信原语，并按切分对象说明常见并行方案 | 已采纳（01）；02 计划采纳 |
| REF-007 | [Google Scaling Book 推理章](https://jax-ml.github.io/scaling-book/inference/)；[PagedAttention/vLLM 原论文](https://arxiv.org/abs/2309.06180)；[TensorRT-LLM 官方文档](https://nvidia.github.io/TensorRT-LLM/) | 03 推理 | 按工作负载、目标、cache、batching 和复制/切分组织推理内容；调度实现仅作资料入口 | 计划采纳 |

## 事实依据

| 编号 | 事实依据 | 相关章节 | 可核验主张 | 具体定位 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| FACT-001 | `codes/philoflow-monorepo` | 后续实践章节 | 代码实现及固定提交中的实际配置 | 当前只记录提交 `74124a2`，缺少 `.gitmodules` 上游映射 | 待核验 |
| FACT-002 | [Qwen3 官方发布说明](https://qwenlm.github.io/blog/qwen3/)；[Qwen3-32B 官方模型卡](https://huggingface.co/Qwen/Qwen3-32B) | 00 总览；01 资源直觉；02 训练；03 推理 | Qwen3-32B 是 dense 模型；总参数 32.8B，非 embedding 31.2B；64 层、GQA 等基础规格 | 模型卡 “Model Overview” 列出 Number of Parameters 32.8B、Non-Embedding 31.2B、64 layers；发布说明的规格表 | 已核验 |
| FACT-003 | [OLMo 2 32B 官方发布说明](https://allenai.org/blog/olmo2-32b)；[官方模型卡](https://huggingface.co/allenai/OLMo-2-0325-32B) | 02 训练 | 32B 模型规格、训练集群、吞吐、MFU，以及并行与 checkpoint 工程案例 | 发布说明的 “Training infrastructure”“OLMo-core Trainer” 与 “Training on Google Cloud Engine”；模型卡规格表 | 已核验 |
| FACT-004 | [MiniMax H3 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 00 总览；01 资源直觉；02 训练；03 推理 | H3-Omni-Transformer 为约 33B 的 dense 单流 Transformer；约 13B 参数在 AdaLN 相关分支，推理可缓存，因此推理激活约 20B；H3-Encoder 使用 Qwen3-VL-32B 权重 | 模型卡 “H3-Omni-Transformer”“H3-Encoder” 两节 | 已核验 |
| FACT-005 | [NVIDIA H100 产品规格](https://www.nvidia.com/en-us/data-center/h100/)；[NVIDIA Hopper 架构说明](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)；[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 01 资源直觉 | H100 SXM 的 BF16 稠密 Tensor Core 峰值约为 989 TFLOP/s、HBM 带宽为 3.35 TB/s；正文中 GEMM 的 FLOPs、最小逻辑搬运量、算术强度和理论时间下界演算 | H100 产品规格表列出 BF16 1,979 TFLOP/s（脚注为 with sparsity）和 3.35 TB/s；Hopper 架构说明明确稀疏特性将标准 Tensor Core 吞吐翻倍，因此稠密口径约为 1,979 / 2 = 989.5 TFLOP/s，正文按 989 TFLOP/s 取整；CUDA 指南 9.2 节给出理论/有效带宽口径；演算以十进制 SI 单位复核 | 已核验 |

第一章以无品牌 GEMM/Attention 建立方法，并可用 Qwen3-32B 与 H3-Omni-Transformer 的参数量做一次 `2P`/`6P` 量级对照；具体训推工作单元、指标与完整案例分别属于 02、03。H100 峰值、带宽和第一章通用演算已登记为 `FACT-005`；Qwen3-32B 参数口径为 `FACT-002`，H3-Omni-Transformer 参数口径为 `FACT-004`。02 仍需登记 MFU/HFU 与训练案例口径，03 仍需登记推理指标与案例口径，不得使用一般化 MFU 区间或实时预算。

## 章节参考对照

进入章节大纲设计前，选择约 3–5 篇创作参考完成下表；只记录会影响本章结构和表达的内容。

| 参考编号 | 结构亮点 | 可借鉴内容 | 图表思路 | 与本章关系 | 是否采纳 |
| --- | --- | --- | --- | --- | --- |
| REF-001 | 先 Roofline 和硬件，再讲 Transformer 数学、训练、推理与实践 | 保留统一分析入口，但把完整性能模型压缩为通用资源方法和两个场景章 | 简化 Roofline、分片矩阵、训练/推理算术表 | 影响 01–03；为目标听众删除大部分硬件和公式细节 | 是 |
| REF-004 | 从单卡起步，按 DP、TP、CP、PP、EP 解释动机后再组合 | 并行策略先回答“为什么需要”；主讲只保留 DP、FSDP、TP、序列切分 | 切分对象与设备的交互图 | 主要影响 02；PP/EP 和组合搜索降为识别或补充材料 | 是 |
| REF-005 | 从内存层次和 IO 解释性能，不把显存只当静态容量 | 01 保留容量与数据移动的区别，02/03 只判断 IO-aware attention 是否适用 | HBM–片上存储直觉和场景内的前后对照 | 影响 01–03；不再为 FlashAttention 设置独立深入页 | 是 |
| REF-006 | NCCL 定义原语，Megatron Core 定义并行策略与组合 | 核心 collective 只用输入输出讲直觉，再连接训练切分 | collective 张量卡片与切分对象 | 影响 01/02；删除通信成本公式和拓扑调参 | 是 |
| REF-007 | 推理先区分阶段和服务目标，再进入 cache、调度和部署 | prefill/decode 与 H3 先分支；显存节省必须连接并发和吞吐 | 双阶段账本、请求队列和状态池 | 影响 03；复杂调度与分布式 serving 只作资料入口 | 是 |
