# 大模型训练与推理性能分享：整体章节规划调研

## 调研问题

围绕“计算、显存、通信、训练优化、推理优化”设计一份 30–60 分钟的技术分享，重点解决三个问题：

1. 如何安排章节，避免同一技术在多个主题中重复讲解；
2. 哪些概念存在前置依赖，应按什么顺序出现；
3. 各章如何分配篇幅，避免“计算很长、通信或应用很短”等失衡。

本文是结构调研，不替代正式大纲；建议需由内容负责人确认后再进入 `outline/`。

> 后续筛选说明（2026-08-15）：正式草拟大纲已根据“主要面向算法与数据同学、建立基础直觉而非深入实现”的要求进一步压缩为 01 资源直觉、02 训练、03 推理、04 诊断四章。本文保留较完整的知识结构和一手资料入口，作为补充材料候选，不代表当前主讲范围；当前章节与筛选边界以 `outline/index.md` 为准。

## 结论摘要

建议保留“资源基础 → 场景应用”的大方向，但不要把五章当作五个互斥的技术清单。更稳健的结构是：

> **工作负载与指标 → 统一性能模型 → 显存容量与本地数据移动 → 跨设备通信基础 → 分布式训练 → 推理系统 → 综合决策**

其中前三个主体章回答“资源如何计账”，后两个应用章回答“在训练或推理约束下如何组合技术”。每项技术只设一个“原理归属章”；其他章节只引用它造成的成本变化或决策影响。

最重要的边界调整有三项：

- 将原“存储”改名为 **“显存容量与数据移动”**。GPU 性能不仅取决于能否放下张量，也取决于张量在寄存器、片上 SRAM、HBM 和主机内存之间移动多少；CUDA 官方性能指南将内存空间、带宽、访问合并与数据传输作为同一组性能问题，[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)也说明“减少 HBM 与 SRAM 之间的读写”可以在不减少数学 FLOPs 的情况下加速计算。
- “通信”章只讲 **拓扑、点对点/集合通信、通信成本和重叠原则**；DP、TP、PP、SP/CP、EP 等并行方案放到训练或推理应用章。NCCL 官方文档定义的是 AllReduce、AllGather、ReduceScatter 等原语，而 [Megatron Core 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)才负责把它们组合为应用级并行策略。
- “常用优化手段”不要按名词罗列，而应按 **目标约束** 组织：训练先解决“放得下”，再解决“扩得开”，最后解决“跑得满”；推理先区分 prefill/decode，再按单请求时延、并发吞吐和服务成本选择优化。官方 TensorRT-LLM 文档明确指出 [prefill 与 decode 具有不同计算特征](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html)，因此推理不能直接沿用训练的 MFU 或并行叙事。

## 为什么这套结构能减少交叉

### 两条正交轴

整份分享可用两条轴定位内容：

| 轴 | 回答的问题 | 应放内容 |
| --- | --- | --- |
| 资源轴 | 时间和空间花在哪里？ | FLOPs、精度、硬件峰值、显存容量、HBM 流量、通信量、延迟与带宽 |
| 生命周期轴 | 在什么工作负载和目标下做取舍？ | 训练 step、prefill、decode、批处理、并行布局、调度、可靠性 |

资源章讲稳定的分析语言，生命周期章讲组合决策。这样允许一项技术同时影响计算、显存和通信，但不需要在三个地方各讲一次。

[Google DeepMind 的《How to Scale Your Model》](https://jax-ml.github.io/scaling-book/)采用了相近依赖顺序：先讲 Roofline、硬件和分片矩阵，再讲 Transformer 数学，最后分别进入训练与推理。其 [Roofline 章节](https://jax-ml.github.io/scaling-book/roofline/)把执行时间下界表达为计算时间、内存/通信时间中的主导项；原始 [Roofline 论文](https://www.osti.gov/pages/biblio/1407073)则用算术强度连接处理器峰值与片外内存流量。这说明“计算、显存带宽、通信”更适合作为统一模型的三个约束，而不是三套互不相干的百科章节。

### 三层归属规则

对每个容易重复的主题，只允许按以下三层展开：

1. **定义/公式**：在哪一章建立分析语言；
2. **机制**：在哪一章解释它如何改变 FLOPs、bytes、容量或通信；
3. **应用决策**：训练或推理中何时启用、和什么组合、代价是什么。

“机制”只完整讲一次。应用章通过一行成本变化和前向引用复用，不重新解释底层过程。

## 推荐章节规划

### 00 导览：从工作负载到系统约束

**唯一目标**：建立全篇分析路径和指标边界。

建议内容：

- 先定义分析单位：训练的一次 step、LLM 的一次 prefill/decode、视频生成的一次 denoising/流式输出单元；
- 给出统一账本：`工作量（FLOPs）`、`驻留量（bytes）`、`搬运量（bytes）`、`关键路径时间`；
- 区分目标：训练吞吐与完成时间，推理 TTFT/每 token 时延/吞吐/并发/成本；
- 说明 MiniMax H3、Qwen3-32B 和 OLMo 2 32B 的案例角色，不在此展开具体公式。

**依赖作用**：避免后续用“token/s”“MFU”“显存占用”回答彼此不同的问题。

### 01 统一性能模型：计算量、数值精度与硬件上限

**唯一目标**：能从模型算子估算理想计算时间，并判断何时 FLOPs 不是瓶颈。

建议顺序：

1. 从 GEMM 的 `2mkn` 到模型级 FLOPs；
2. 前向、反向与训练 step 的差异；
3. FP32/BF16/FP16/FP8/INT8/INT4 的“表示、存储、计算、累加”四个角色，避免只按位宽列参数表；
4. Tensor Core 峰值、稠密/稀疏口径、实际吞吐；
5. Roofline：算术强度、计算上限、HBM 带宽上限；
6. MFU/HFU 的适用边界，并明确推理还需看延迟和吞吐。

NVIDIA [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)同时把低精度描述为性能与内存优化，且其 [FP8 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)还涉及格式、缩放和前后向使用差异。因此“精度”适合在本章建立共同概念，而训练稳定性与推理量化方案分别留到应用章。

**边界**：这里只给通信时间占位，不讲 DP/TP；只建立显存带宽概念，不展开各类张量的生命周期。

### 02 显存容量与本地数据移动：什么要留、什么要搬

**唯一目标**：会画出张量生命周期，并分别计算“放得下”和“喂得上”。

建议分成两个清楚的小节：

1. **容量账本**
   - 训练：参数、梯度、优化器状态、激活、临时 workspace；
   - 推理：参数、临时激活、KV cache 或模型特有状态、运行时预留；
   - 峰值而非总和：按张量创建、保留、释放的时间轴估算；
2. **数据移动账本**
   - 寄存器/片上 SRAM/HBM/CPU 内存的层次；
   - 读写字节、复用率、访存合并、kernel fusion 与 tiling；
   - 用 FlashAttention 作为“数学不变、数据移动改变”的主例。

[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)把有效带宽写为读写字节数除以时间，并强调内存层次、访问合并和减少主机—设备传输；[FlashAttention](https://arxiv.org/abs/2205.14135)则直接以 HBM–SRAM IO 复杂度解释性能与内存改进。两者共同支持把“容量”和“带宽/搬运”并列，而不是把所有问题都称为“显存优化”。

本章最后只给五类通用杠杆：**少存（压缩/低精度）、晚存（释放/流式化）、不存（重计算）、分开存（分片）、换地方存（offload）**。具体实现归属后续章节：

- 激活重计算的机制在此讲；训练中如何配置留到第 04 章。PyTorch 官方文档明确将 [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)定义为以计算换内存。
- “分片可以降低单卡容量”在此讲；ZeRO/FSDP 的通信过程和训练选择留到第 04 章。
- KV cache 的尺寸公式在此讲；分页、复用、淘汰与调度留到第 05 章。

### 03 跨设备通信基础：从链路到原语

**唯一目标**：能从张量布局推导通信原语，并估算其关键路径成本。

建议顺序：

1. 分层拓扑：卡内数据移动 → 卡间 NVLink/PCIe → 节点间网络；
2. 延迟、带宽、消息大小和并发；
3. P2P send/recv；
4. Broadcast、AllReduce、AllGather、ReduceScatter、AllToAll；
5. ring/tree 等算法只讲直觉，不展开实现细节；
6. 通信与计算重叠：区分总通信量和暴露在关键路径上的通信时间。

[NCCL 官方集合通信说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_298/user-guide/docs/usage/collectives.html)给出了各原语的输入输出语义，并指出 ReduceScatter + AllGather 等价于 AllReduce；NCCL 还会根据拓扑、消息大小、算法和协议做选择，[官方调优说明](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)展示了这些成本维度。因而通信章不宜把“会调用某个集合通信”直接等同于“理解某种并行方案”。

**边界**：本章用一个分片 GEMM 和一个梯度同步示例建立映射即可；DP/TP/PP/SP/CP/EP 的完整动机、内存收益和组合放到训练章，推理中的副本、模型分片和 P/D 解耦放到推理章。

### 04 分布式训练：先放得下，再扩得开，最后跑得满

**唯一目标**：给定模型、序列、卡数和拓扑，能解释训练配置的选择顺序。

建议用决策流程代替“优化技巧列表”：

1. **训练 step 解剖**：前向、反向、优化器更新；建立 micro-batch、global batch、gradient accumulation；
2. **单卡基线**：混合精度、融合算子、activation checkpointing、数据流水线；
3. **模型状态放不下**：DP → 分布式优化器/ZeRO/FSDP；
4. **单层或激活放不下**：TP + SP；长序列再引入 CP；
5. **模型深度仍放不下**：PP 与 micro-batch schedule、bubble；
6. **MoE 特有**：EP、AllToAll、负载均衡；
7. **组合与映射**：并行维度乘积、哪些应留在高速域内、通信重叠；
8. **训练系统工程**：checkpoint、容错和慢节点仅在篇幅允许时作为扩展。

这一顺序有直接的一手资料支撑：

- [ZeRO 论文](https://arxiv.org/abs/1910.02054)将训练内存拆为参数、梯度和优化器状态，并通过逐级分片消除冗余；[DeepSpeed 官方教程](https://www.deepspeed.ai/tutorials/zero/)对应说明 ZeRO-1/2/3 分别增加优化器状态、梯度和参数分片。
- [Megatron-LM 论文](https://arxiv.org/abs/2104.04473)表明 TP、PP、DP 是可组合的，同时选择要受内存、跨节点通信和等待时间约束。
- [Megatron Core 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)按切分维度区分 DP、TP、PP、CP、EP 和 FSDP；[Context Parallel 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)进一步说明 CP 切分全部层的序列激活，而 SP 只涉及部分模块。因此报告中不应把 SP 与 CP 当作同义词。
- [Megatron Bridge 性能指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)的配置建议也是先从 DP 出发，再因容量与拓扑加入 TP/CP/PP/EP；这比按缩写字母顺序介绍更符合真实决策过程。

### 05 推理系统：从一次前向到持续服务

**唯一目标**：能根据模型类型和 SLO，选择单请求、批处理和分布式服务优化。

建议先分支，再合流：

1. **工作负载分支**
   - 自回归 LLM：prefill 与 decode 分开计账；
   - DiT/视频生成：以 step、分辨率、时空 token、CFG/流式窗口等实际工作单元计账；没有的状态机制不要为对齐 LLM 而强套 KV cache；
2. **单次执行优化**
   - 低精度/量化、融合 kernel、编译与图优化；
   - LLM 的 KV cache、Attention kernel、speculative decoding；
   - 模型架构侧变化单列“改变数学问题”，与系统无损优化区分；
3. **并发与调度**
   - 静态 batching → iteration/continuous batching；
   - paged KV、prefix cache、cache eviction；
4. **分布式部署**
   - 请求级 replicas/DP、TP/PP/EP；
   - prefill–decode 解耦、KV 传输与路由；
5. **评估闭环**
   - 质量约束、TTFT、每步时延、吞吐、并发、显存和单位成本一起报告。

关键来源及其结构含义：

- [Google Scaling Book 推理章](https://jax-ml.github.io/scaling-book/inference/)指出推理比训练更受延迟、批大小和内存带宽限制，并单独分析 KV cache，因此不能把训练并行配置直接复制到推理。
- [Orca 论文](https://www.usenix.org/conference/osdi22/presentation/yu)把生成服务的关键改变定位为 iteration-level scheduling，说明“批处理”在训练和在线生成中不是同一种机制。
- [PagedAttention/vLLM 论文](https://arxiv.org/abs/2309.06180)将 KV cache 的动态增长、碎片和共享问题连接到批大小与吞吐；这类内容应归入推理调度，而不只是显存章的一个公式。
- TensorRT-LLM 的 [KV cache 文档](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)把 block pool、复用、offload 和 eviction 作为系统能力；其 [disaggregated serving 文档](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html)又把 prefill/decode 放到不同 worker，并引入 KV 传输，体现了显存与通信在应用层的联合取舍。
- [Speculative Decoding 原论文](https://arxiv.org/abs/2211.17192)针对自回归串行步提出并行验证，属于改变 decode 关键路径的推理算法，不应放进通用计算优化清单。

### 06 收束：一张瓶颈诊断与方案选择图

**唯一目标**：让听众能把五章知识用于新模型，而不是记住一组工具名。

建议用固定流程收束：

1. 明确工作单元与目标指标；
2. 计算 FLOPs、峰值驻留和数据移动；
3. 用 Roofline/时间下界找当前瓶颈；
4. 若单卡不满足，再计算分片后的通信；
5. 选择只改变当前瓶颈的优化，并检查质量、复杂度与成本副作用；
6. 用 profiler/benchmark 验证，回到模型修正估计。

## 易重叠主题的唯一归属

| 主题 | 原理归属章 | 其他章节只保留什么 |
| --- | --- | --- |
| 数值精度/量化 | 01：格式、缩放、计算与存储口径 | 02 写容量影响；04 写训练稳定性；05 写推理质量与 kernel 支持 |
| Roofline / arithmetic intensity | 01 | 04、05 只用它判断方案是否改善瓶颈 |
| FlashAttention / tiling / fusion | 02：本地数据移动机制 | 04、05 只说明训练或推理中的适用形状与效果 |
| Activation checkpointing | 02：计算换容量的机制 | 04 说明训练配置与额外计算代价 |
| 参数/状态分片 | 02：分开存的通用杠杆 | 04 讲 ZeRO/FSDP 的阶段和 collectives；05 讲权重分片部署 |
| KV cache | 02：尺寸与生命周期公式 | 05 讲 paging、reuse、eviction、routing |
| AllReduce/AllGather/ReduceScatter/AllToAll | 03：语义和成本 | 04、05 标注各并行策略使用什么原语，不重画定义 |
| DP/TP/PP/SP/CP/EP | 04：训练动机、收益与组合 | 03 仅以小例连接原语；05 只讲推理目标下的不同选择 |
| 通信重叠 | 03：关键路径与重叠原则 | 04、05 讲可调度的具体阶段及所需 buffer |
| batching | 00：指标和批大小基本概念 | 04 讲 micro/global batch；05 讲 continuous/iteration batching |
| MFU | 01：定义和口径 | 04 作为训练结果指标；05 不用 MFU 取代延迟/吞吐/SLO |

建议每次出现跨章主题时使用同一种旁注：`改变：FLOPs / 驻留 bytes / 搬运 bytes / 通信 bytes / 关键路径`。这样可以让听众看到交叉影响，同时避免重复讲机制。

## 概念依赖顺序

下列依赖最好不要颠倒：

1. **张量形状 → 参数量/FLOPs → 精度字节数**：否则显存与通信量没有共同口径；[Scaling Book 的 Transformer 数学章](https://jax-ml.github.io/scaling-book/transformers/)也从矩阵尺寸同时推导 FLOPs、参数与 KV cache。
2. **峰值能力 → 算术强度/Roofline → 实际利用率**：否则 MFU 容易被误解为通用硬件效率，而忽略访存或通信受限。
3. **显存容量 → 张量生命周期 → 重计算/分片/offload**：否则只会背“省几倍”，无法解释峰值为何改变。
4. **拓扑与 collective → 并行维度**：否则 DP/TP/EP 只剩缩写，无法比较消息大小和高速域映射。
5. **单个并行维度 → 混合并行 → 通信重叠与调度**：重叠依赖具体前后向或 serving schedule，不能先于基本数据依赖。
6. **训练基础 → 训练优化；prefill/decode 差异 → 推理优化**：同一个 TP 或 batching 在两个生命周期的目标函数不同。
7. **理论下界 → profiler 观测 → 优化验证**：优化项应由瓶颈驱动，而不是按“常用技巧”全开。

## 建议篇幅配比

以下配比是教学取舍，不是行业重要性的量化结论。原则是：基础章控制在约一半，应用章占另一半；通信基础故意保持紧凑，把应用级并行的篇幅归还给训练与推理。

| 章节 | 完整版 60 分钟 | 压缩版 30 分钟 | 篇幅理由 |
| --- | ---: | ---: | --- |
| 00 导览 | 4 分钟 | 2 分钟 | 只统一单位、指标与案例 |
| 01 统一性能模型 | 10 分钟 | 5 分钟 | 后续所有估算的共同语言 |
| 02 显存与数据移动 | 9 分钟 | 5 分钟 | 容量与带宽必须分开，但不罗列框架配置 |
| 03 通信基础 | 8 分钟 | 4 分钟 | 只讲拓扑、collective 与成本模型 |
| 04 分布式训练 | 15 分钟 | 8 分钟 | 并行组合最复杂，也是主要听众最直接相关部分 |
| 05 推理系统 | 12 分钟 | 5 分钟 | 完整版覆盖执行、调度和部署；压缩版保留 prefill/decode 与一个服务案例 |
| 06 收束 | 2 分钟 | 1 分钟 | 用诊断流程复盘，不新增概念 |

若现场更偏训练，可从推理章减 2–3 分钟给训练；不建议压缩 01–03 到总时长的三分之一以下，否则后面的并行策略会退化为名词罗列。若 MiniMax H3 是主要实例，还应把推理章内部至少一半时间留给 DiT/视频路径，LLM 的 KV cache 与 serving 技术作为对照，不要反客为主。

## 参考结构对照

| 资料 | 结构特点 | 可借鉴之处 | 不直接照搬之处 |
| --- | --- | --- | --- |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | Roofline → 芯片/分片矩阵 → Transformer 数学 → 训练 → 推理 → 实践 | 先建立统一成本模型，再分训练/推理；贯穿形状与时间估计 | 体量远大于 60 分钟，且以 TPU/JAX 为主，需要压缩和 GPU 化 |
| [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/main/ultra_blog.md) | 单卡起步，依次介绍 DP、TP、CP、PP、EP，再组合和深入 GPU | 每种并行先讲“为什么需要”，最后再做配置搜索 | 重点是训练，不能承担完整推理叙事；按技术逐项讲在短分享中仍可能超时 |
| [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) | 按切分维度列出 DP、TP、PP、CP、EP、FSDP，并说明组合 | 可作为术语边界与训练并行映射的权威来源 | 是产品指南，不是从计算/显存/通信逐层推导的教学叙事 |
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 从测量、带宽、内存空间与访问模式解释单卡性能 | 支撑“容量”和“数据移动”分章内双线表达 | 粒度偏 kernel，需用 Transformer/DiT 算子裁剪案例 |
| [TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/) | 按 KV cache、调度、并行和 disaggregated serving 等系统能力组织 | 推理章应从 prefill/decode、cache 和调度进入，而非重复训练优化列表 | 工具特定；正文应保持机制中立，TensorRT-LLM 只作一手实现案例 |

## 推荐的大纲写作约束

后续若据此调整正式大纲，建议对每章统一填写四项，强制控制范围：

1. **本章只回答的一个问题**；
2. **输入账本**：需要哪些形状、精度、硬件或拓扑信息；
3. **输出账本**：本章最终给出 FLOPs、驻留 bytes、搬运 bytes、通信 bytes 或关键路径中的哪一项；
4. **明确不讲**：哪些机制只前向引用或留到应用章。

每个优化手段使用同一张四格卡片即可：`解决的瓶颈`、`改变的账本项`、`代价/副作用`、`适用工作负载`。这比以框架参数或缩写为主线更容易控制篇幅，也方便 MiniMax H3 与 Qwen3-32B 使用同一分析模板而不强求技术一一对应。

## 来源与核验说明

- 调研日期：2026-08-15。
- 本文优先使用原始论文、官方项目文档和硬件厂商技术文档；Hugging Face Ultra-Scale Playbook 与 Google Scaling Book 主要作为创作结构参考，精确性能数字仍应回到其引用的论文、配置或硬件规格核验。
- 调研完成时没有提前为全部来源分配 `REF-NNN` / `FACT-NNN`；正式大纲采纳本调研后，已在 `references/sources.md` 登记实际影响结构的 `REF-005`–`REF-007`。正文使用的精确主张仍需按既有规则另行登记和核验。
- MiniMax H3 的非公开架构、token 定义、实时目标与实测性能应以团队材料为准；公开 LLM 资料只能提供分析框架，不能用于反推 H3 的具体规格。
