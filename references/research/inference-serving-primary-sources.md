# 第 03 章推理与服务：一手资料核验记录

## 调研范围与口径

- 调研日期：2026-08-18。
- 对应大纲：`outline/03-inference.md`。
- 目标：为正文中的指标口径、KV cache、量化、speculative decoding、持续批处理与 KV 管理、多卡部署提供可定位的一手依据。
- 来源优先级：原论文、会议正式论文页、官方模型卡或固定模型配置、框架官方文档。
- 本文只整理可支持的事实与表达边界，不把论文中的特定系统性能数字外推为通用结论，也不替代目标硬件和目标工作负载上的实测。

## 结论摘要

1. `TTFT`、`TPOT`、`ITL`、端到端延迟和吞吐必须先写明测量边界与聚合方式。客户端 TTFT 不等于纯 prefill kernel 时间，TPOT 也不总等于逐次流式响应间隔。
2. 标准 decoder-only、全上下文 KV cache 的逻辑载荷与层数、KV head 数、head dimension、元素字节数和所有活跃 token 总数线性相关。Qwen3-32B 在 BF16 下是 `256 KiB/token`；单条 32,768-token 序列是 `8 GiB` 的逻辑 KV 载荷。
3. 量化先改变存储与搬运字节，是否加速取决于目标硬件、执行框架、kernel、shape 和阶段；质量是否可接受必须按实际任务、上下文和生成配置验证。
4. 标准 speculative sampling 通过草稿、目标模型并行验证和修正分布保持目标模型的采样分布；“分布相同”不等于随机采样时逐 token、逐次运行结果相同，近似接受规则也不再具有这一保证。
5. continuous batching、paged KV、prefix reuse 和 chunked prefill 改变的是调度粒度、状态池利用率或 prefill/decode 交错方式；它们提供优化机会，不自动保证吞吐与所有延迟指标同时改善。
6. replicas 复制模型并分流独立请求，主要扩总吞吐；TP/PP 切分模型以解决单卡容量或聚合算力，但把通信和同步放入单请求关键路径。选择必须结合互联、batch、输入输出长度和 SLO 实测。

## 1. Prefill、decode 与性能指标

### 1.1 执行阶段

- **Prefill/context**：处理 prompt token，产生后续生成要复用的 KV cache，并得到第一个待采样位置的 logits。prompt 中多个位置可在一次前向中并行处理。
- **Decode/generation**：从已有 logits 采样 token，把新 token 送回模型，追加该 token 的 KV，随后重复；输出位置之间存在自回归串行依赖。
- 这两个阶段常有不同 shape 和资源特征，但不应把“prefill 必然计算受限、decode 必然带宽受限”写成架构无关的定律。短 prompt、大 batch、长上下文 attention、量化、并行和具体 kernel 都可能改变瓶颈。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [Google《How To Scale Your Model》Inference](https://jax-ml.github.io/scaling-book/inference/) | “Prefill and generation”及后续 prefill/generate 接口说明 | prefill 同时处理 prompt 并保存 KV；generation 逐 token 追加 KV | 文中的 TPU 性能模型和“compute/memory bound”判断依赖其模型、batch 与硬件假设 |
| [TensorRT-LLM Disaggregated Serving](https://nvidia.github.io/TensorRT-LLM/latest/features/disagg-serving.html) | 开头 “LLM inference has two stages” | context 计算 prompt KV，generation 逐 token 使用缓存；两阶段可采用不同并行策略 | 阶段拆分会增加 KV 传输，不代表解耦部署必然提升吞吐 |

### 1.2 指标定义

以客户端发请求时刻 `t_send`、收到第一个输出 token 时刻 `t_first`、收到最后一个输出 token 时刻 `t_last`、输出 token 数 `N_out` 为例：

- `TTFT = t_first - t_send`：客户端从发出请求到收到首 token 的时间。
- `E2E latency = t_last - t_send`：从发请求到收到完整响应。
- 对 `N_out > 1`，NVIDIA NIM/GenAI-Perf 的请求级平均口径为

  `TPOT = (E2E latency - TTFT) / (N_out - 1)`。
- `ITL` 更适合指相邻流式输出之间的各个间隔；TPOT 是单请求内这些间隔的平均值。工具也可能把所有请求的 ITL 样本直接汇总，因此 TPOT 的分位数与 ITL 的分位数不是同一个统计量。
- 系统 `output token throughput` 是压测期间完成的输出 token 总数除以压测 wall-clock 时长；`request throughput` 是完成请求数除以时长。还应区分“系统总 output tok/s”“单用户 tok/s”和包含输入 token 的 `total token throughput`。

关键边界：

- 客户端 TTFT 通常包含排队、batching、服务端处理和网络，因此只能近似反映 prefill 体验，不能直接当作纯 prefill kernel 延迟。
- 端到端延迟同时受输入长度、输出长度、排队和流式传输影响；比较时必须固定或报告长度分布与到达模式。
- 当一次流式响应可能包含多个 token（例如 speculative decoding）时，ITL 记录的是响应间隔，而 TPOT 按实际输出 token 摊销，两者可能明显不同。
- `N_out = 1` 时上述 TPOT 分母为零，应报告 TTFT/E2E 而不是伪造 TPOT。
- 吞吐必须注明分子、计时区间、并发/到达率和是否达到延迟 SLO；饱和吞吐不能替代低负载单请求延迟。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [NVIDIA NIM LLM Benchmarking 2.0.0：Metrics](https://docs.nvidia.com/nim/benchmarking/llm/2.0.0/metrics.html) | “Time to First Token”“End-to-End Request Latency”“Inter-token Latency”“Tokens per Second” | TTFT、E2E、ITL/TPOT 公式；系统 TPS 与单用户 TPS 的区别；文档明确不同工具是否纳入 TTFT 会不同 | 这是该工具的测量边界，不是所有 benchmark 的唯一命名规范 |
| [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) | 输出指标说明中的 TTFT、TPOT、ITL 与 speculative decoding 示例 | TPOT 按请求计算后聚合；ITL 汇总相邻流式输出间隔；一次输出含多个 token 时两者会分离 | 不应看到 “ITL/TPOT” 就默认同一组样本和同一分位数 |
| [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2700/user-guide/docs/perf_analyzer/genai-perf/README.html) | “Metrics”表 | output token throughput、request throughput、request latency 的明确分子和计时含义 | 工具输出的均值或 P99 仍需同时给出压测 workload |

## 2. Decoder-only LLM 的 KV cache 容量

### 2.1 变量关系

对每层都保存全历史 K/V、没有 sliding-window/latent-attention 等结构变化的 decoder-only Transformer，逻辑 KV 载荷为：

`KV_bytes = 2 × bytes_per_element × L × H_kv × D_head × T_live`

其中：

- `2`：K 和 V 两份状态；
- `L`：层数；
- `H_kv`：KV head 数，GQA/MQA 应使用 KV head 数而不是 query head 数；
- `D_head`：每个 head 的维度；
- `T_live`：服务实例当前所有活跃序列实际保留的 token 数总和；固定等长 batch 时可写成 `B × S`；
- `bytes_per_element`：KV cache 实际 dtype 的元素字节数，不一定等于权重 dtype。

这个公式给出**逻辑 tensor 载荷**。运行时峰值还可能包含 block 取整、元数据、未填满的尾块、临时张量、allocator/框架预留、CUDA Graph 和通信 buffer；相反，TP 对 KV head/序列维分片、KV 量化、滑动窗口等又可能降低单卡载荷。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [Google Scaling Book：Inference](https://jax-ml.github.io/scaling-book/inference/) | “The main difference is the KV cache”后的公式 | `2 × bytes × head_dim × kv_heads × layers × tokens` | 公式针对标准缓存形态；不能直接套到 MLA、滑窗或压缩 KV |
| [Google Scaling Book：Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) | “Key-Value (KV) caching” | KV 形状可写作 `[2, S, L, K, H]`；GQA 的 `K` 是 KV head 数 | 实际框架的物理 layout 可以不同，但逻辑元素数一致 |

### 2.2 Qwen3-32B 可核验示例

固定到 Qwen 官方仓库初始已验证提交 `1aecd22f7ae87d957e541aaf933345c33d8a3a26`：

- `num_hidden_layers = 64`
- `hidden_size = 5120`
- `num_attention_heads = 64`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `max_position_embeddings = 40960`
- 默认权重 dtype 字段为 `bfloat16`
- 官方模型卡称原生 context 为 32,768 token；131,072 token 需要 YaRN 配置

若 KV cache 也使用 BF16（2 bytes/element），则：

- 每 token：`2 × 2 × 64 × 8 × 128 = 262,144 bytes = 256 KiB`
- 单条 32,768-token 序列：`262,144 × 32,768 = 8,589,934,592 bytes = 8 GiB`
- 按固定 config 的 40,960 个位置：`262,144 × 40,960 = 10,737,418,240 bytes = 10 GiB`

使用边界：

- `8 GiB` 是一条已缓存 32,768 token 序列的全模型逻辑 KV 载荷，不是整个服务进程显存，也不是“额外可用上下文”。
- `10 GiB` 对应固定 config 的 40,960 个位置；它与模型卡的“原生 32,768”是不同口径，正文必须明确采用的是模型卡原生长度还是 checkpoint 配置上限。
- Qwen3-32B 的 `head_dim` 由 config 明确给出为 128，不能用 `hidden_size / num_attention_heads = 80` 代替。
- 同时服务多条不同长度请求时，应按 `256 KiB × Σ实际保留 token` 计算，不应一律按 `max_seq_len × max_batch` 当作实际占用。
- 若框架把 KV 分片到多卡，`8 GiB` 是跨分片总逻辑量；单卡量取决于分片方式。
- 官方模型卡提醒短于 32,768 token 的常见场景不建议无条件启用 YaRN，因为可能影响性能；因此不应把 131,072 写成无条件“原生上下文”。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [Qwen3-32B 固定提交的 `config.json`](https://huggingface.co/Qwen/Qwen3-32B/blob/1aecd22f7ae87d957e541aaf933345c33d8a3a26/config.json) | `head_dim`、`hidden_size`、`num_attention_heads`、`num_hidden_layers`、`num_key_value_heads`、`max_position_embeddings`、`torch_dtype` 字段 | 演算所需的官方结构规格 | `torch_dtype` 是模型配置字段，服务时 KV dtype 仍由框架配置决定 |
| [Qwen3-32B 固定提交模型卡](https://huggingface.co/Qwen/Qwen3-32B/blob/1aecd22f7ae87d957e541aaf933345c33d8a3a26/README.md) | “Model Overview”与长上下文/YaRN 说明 | 64 层、64 Q heads/8 KV heads、原生 32,768 与 YaRN 131,072 的官方口径 | 不应把配置中的 40,960 或 YaRN 扩展长度混称为原生上下文 |

## 3. 量化：适用边界

### 3.1 能支持的主张

- 权重量化可降低模型权重驻留与搬运字节；激活量化、KV cache 量化分别改变不同账本项，不能统称为同一种“INT4 推理”。
- 权重低位存储只有在运行时有匹配的 packing、dequantization 与 GEMM/GEMV kernel 时才可能转化为延迟或吞吐收益。缺少支持时可能回退、拒绝加载，或因动态量化/反量化开销而收益有限。
- 硬件支持按格式和计算路径区分。例如 vLLM 的官方兼容表中，AWQ、GPTQ、Marlin、INT8 W8A8、FP8 W8A8 对 Volta/Turing/Ampere/Ada/Hopper/AMD/CPU 的支持并不相同。
- 量化质量必须相对同模型基线，在实际任务、prompt/输出长度、生成参数和可接受阈值下验证。校准数据和 scale 策略也属于结果条件。

### 3.2 Qwen3-32B 的官方质量证据

[Qwen3-32B-AWQ 官方模型卡](https://huggingface.co/Qwen/Qwen3-32B-AWQ)将 BF16 与 AWQ INT4 并列报告：

- Thinking 模式：LiveBench `74.9 → 73.1`，GPQA `68.4 → 69.0`，MMLU-Redux `90.9 → 90.8`，AIME24 `81.4 → 79.4`。
- Non-Thinking 模式：LiveBench `59.8 → 59.8`，GPQA `54.6 → 53.1`，MMLU-Redux `85.7 → 85.6`。

这组官方结果只足以支持“不同任务上的变化方向和幅度不同，必须逐任务验证”，不能支持“INT4 一律无损”或“固定下降 X%”。模型卡也没有为所有业务任务提供误差条、长上下文和生成稳定性覆盖。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [vLLM 0.27.0 Quantization](https://docs.vllm.ai/en/v0.27.0/features/quantization/) | “Supported Hardware”兼容表 | 量化实现与 GPU 架构/CPU 平台支持是组合关系 | 表会随版本演进；正文引用时应注明框架版本，不推广为所有 runtime |
| [TensorRT-LLM FP8 Quantization Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) | calibration 配置与 quantized KV cache 说明 | 从 FP16 checkpoint 量化可能需要校准数据；更激进的 KV 量化增加质量风险，应验证输出质量 | “支持某格式”不等于任意模型、任意层和任意 shape 都有加速 |
| [AWQ 原论文](https://arxiv.org/abs/2306.00978) | 摘要、§3 方法、§4 实验 | AWQ 是 activation-aware 的低位 weight-only 量化，并依赖相应 kernel 实现加速 | 论文的速度数字绑定其 TinyChat、模型、设备和 batch，不能当作 Qwen3-32B 的通用倍数 |
| [Qwen3-32B-AWQ 官方模型卡](https://huggingface.co/Qwen/Qwen3-32B-AWQ) | “Performance”表与 `Quantization: AWQ 4-bit` | 官方给出的 BF16/AWQ INT4 任务分数差异 | 少量 benchmark 不能替代目标业务、长上下文和生成配置验证 |

## 4. Speculative decoding

### 4.1 标准机制

标准 draft-target speculative sampling 的一轮可概括为：

1. 较快的草稿模型 `q` 自回归提出 `γ` 个候选 token；
2. 目标模型 `p` 用一次并行前向评估这些位置；
3. 从左到右，以 `min(1, p(x)/q(x))` 接受草稿 token；
4. 第一次拒绝时，从 `norm(max(0, p-q))` 修正分布采样并丢弃之后的草稿；
5. 若全部接受，还可从目标模型再产生一个 token。

机制减少的是目标模型必须串行调用的轮数，不保证减少总 FLOPs。草稿模型开销、目标模型并行验证开销、额外 KV/权重驻留和被拒候选的浪费都必须计入。

### 4.2 输出一致性边界

- Leviathan 等的标准算法证明：在精确概率运算和规定的修正采样下，输出 token 的分布与只从目标模型 `p` 采样相同。
- 这是一项**分布保证**。随机采样时，不应表述为“开启 speculative decoding 后每次输出 token 串与基线逐字相同”。
- 贪心解码配合严格逐 token 验证，目标是接受与目标模型 argmax 一致的前缀；但硬件浮点、batch shape、非确定性 kernel 或 logprob 数值差异仍可能改变边界 token。
- relaxed acceptance、ensemble verification 或其他修改接受准则的近似方案不再具有标准算法的精确目标分布保证，应单独标注质量边界。
- 加速依赖草稿成本 `c`、接受率 `α`、草稿长度 `γ`、batch 和硬件；接受率低时会增加无效计算，不能承诺固定倍数。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*](https://proceedings.mlr.press/v202/leviathan23a.html) | Algorithm 1、Theorem 3.5、Appendix A.1；§3.3 walltime analysis | 草稿—并行验证—修正采样机制；目标分布不变；收益由接受率、草稿成本和长度共同决定 | 论文的实验加速比不能外推；“相同分布”不是“相同随机样本” |
| [Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318) | 摘要与 Algorithm 1 | modified rejection sampling 在 hardware numerics 范围内恢复目标分布 | “within hardware numerics”已明确不是 bitwise 等价承诺 |
| [vLLM Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) | “Lossless guarantees of Speculative Decoding” | 理论无损、实现验证、浮点精度与 batch 数值稳定性是不同层次 | 运行间 logprob/输出不稳定不能简单归因于 speculative 算法本身 |
| [TensorRT-LLM Speculative Decoding](https://nvidia.github.io/TensorRT-LLM/1.0.0/features/speculative-decoding.html) | Draft/target 与 greedy sampling 说明 | 目标模型单次验证多个 draft token；严格贪心接受要求 token 匹配 | 文档也含 relaxed acceptance；启用该选项后不能沿用严格一致性表述 |

## 5. 持续服务：batch 与 KV 状态池

### 5.1 Continuous batching / iteration-level scheduling

ORCA 的核心改变是把调度粒度从“完整请求”降到“一次模型迭代”：每轮后完成请求可退出，新请求有机会进入下一轮。selective batching 则把可兼容的 token-wise 操作合批，对 shape/状态不同的 attention 做请求感知处理。

能支持的主张：

- 相比固定请求批，迭代级调度可减少短请求等待长请求完成造成的空槽和排队；
- 动态加入/退出为提高平均 batch 利用率和吞吐提供条件。

边界：

- continuous batching 不是“请求随时无成本插入”；插入点、prefill 与 decode 是否混批、token budget 和抢占策略由实现决定。
- 更大的运行 batch 会增加 KV 容量压力，并可能恶化 TTFT、ITL 或尾延迟；最大稳定并发必须在 SLO 下测量。
- ORCA 的 `36.9×` 是其 GPT-3 175B、对比实现与实验设置下的结果，不适合作为现代框架的通用收益。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [ORCA OSDI 2022 正式论文](https://www.usenix.org/system/files/osdi22-yu.pdf) | §3 的 “Iteration-level scheduling”“Selective batching”，Figure 4–5 | 每迭代重组 batch；完成请求退出、新请求进入；选择性合批 | 原论文的 scheduler、模型和基线是一个具体实现，不定义所有 continuous batching 行为 |
| [ORCA USENIX 论文页](https://www.usenix.org/conference/osdi22/presentation/yu) | 摘要与论文元数据 | iteration-level scheduling 的正式出处与论文页码 521–538 | 摘要中的性能数字必须保留实验上下文 |

### 5.2 PagedAttention / paged KV

PagedAttention 把每条序列的逻辑 KV 分成固定 token 数的 block，用 block table 映射到可不连续的物理 block；运行时按需分配，而不是为最大生成长度预留连续空间。

能支持的主张：

- 避免要求每条请求占用一段最大长度的连续物理 KV；
- 固定块可消除传统意义的外部碎片，内部浪费主要受尾块和 block size 影响；
- block 引用计数还可支持共享相同 prompt/分支的物理 KV；
- 更高的状态池利用率可能允许更大 batch，继而提升吞吐。

边界：

- paging 主要改善容量利用率，不直接减少每个有效 token 的逻辑 KV 元素数。
- block table、非连续访存和专用 attention kernel 有实现成本；block 越大，kernel 效率与尾块浪费/复用粒度之间的取舍越明显。
- 显存节省只有在转化为更多活跃请求、更少 preemption 或更低成本时才成为服务收益；低并发、KV 不占主导时收益可能很小。
- 论文中的 `2–4× throughput` 绑定论文版本的 vLLM、数据集、模型和基线，正文不应写成技术固有倍数。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*](https://doi.org/10.1145/3600006.3613165) | §3 memory challenges；§4.1 PagedAttention；§4.2 KV Cache Manager；§4.3 decoding；§4.4 sharing | 固定 block、逻辑—物理映射、按需增长、尾块浪费、引用共享 | 论文的吞吐数字不是所有模型/框架/负载的保证 |

### 5.3 Prefix caching / KV reuse

Prefix reuse 以“模型和执行上下文一致、token 前缀完全相同”为键，复用已经计算的 prefix KV，从而跳过命中部分的 prefill 计算。vLLM 的实现按 parent hash、block tokens 和额外上下文（如 LoRA ID、多模态输入 hash）识别完整 block；TensorRT-LLM 用 radix tree 保存已填满 block，并用优先级/LRU 淘汰。

收益与边界：

- 只减少**命中的 prefill 部分**，不减少新 suffix 的 prefill，也不消除后续 decode；
- 主要影响重复长前缀请求的 TTFT、prefill 计算和共享状态容量；收益取决于前缀长度、命中率、复用粒度和驻留时间；
- 只有完整 block 能直接共享，block size 越大，kernel 效率可能更高但复用机会和精细度可能下降；
- 保留可复用 KV 占用状态池，内存压力大时会淘汰；大 batch 和长输出会降低命中 block 留存概率；
- 多副本各自维护 KV cache 时，路由到错误副本会丢失复用机会；多租户还需隔离/加盐，不能只用裸 token hash 假设安全。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [vLLM 0.27.0 Automatic Prefix Caching](https://docs.vllm.ai/en/v0.27.0/design/prefix_caching/) | hash key 组成、block 分配与 eviction 部分 | exact token block + parent + extra hashes；引用计数和 LRU free queue | hash 命中不是语义相似命中；不同模型/LoRA/多模态上下文不能直接共享 |
| [TensorRT-LLM KV cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html) | system prompt 示例、LRU、block size、host offload | 只有 full blocks 可共享；内存压力、batch 和输出长度影响留存；offload 有传输成本 | 复用不是永久命中，也不能保证所有请求 TTFT 下降 |

### 5.4 Chunked prefill

Chunked prefill 把长 prompt 的 prefill 拆成多个较小 chunk，并由迭代级 scheduler 与进行中的 decode 交错。Sarathi-Serve 进一步用每轮 token budget 先容纳 decode，再放入部分 prefill，以控制 generation stall。

收益边界：

- 小 chunk/token budget 可限制单轮 prefill 对正在 decode 请求的干扰，改善 ITL/TBT 尾部；
- prefill 与 decode 混合有机会同时利用两类工作负载的不同资源特征，也能让 PP microbatch 工作量更均匀；
- chunk 越小，单个新请求完成全部 prefill 所需轮数越多，TTFT 可能上升；
- 分块会重复读取先前 chunk 的 KV，并增加 kernel launch/调度开销；过小 chunk 还可能降低 GPU 利用率；
- 因而 chunk size/token budget 是 TTFT、ITL、吞吐和 prefill 效率的可调权衡，不应写成“四项指标同时无条件改善”。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://www.usenix.org/system/files/osdi24-agrawal.pdf) | §4.1 Chunked-prefills；§4.2 Stall-free batching；§4.3 token budget，尤其重复 KV read 的说明 | 长 prefill 分块、decode-first token budget、较小 budget 的 ITL 收益和额外 KV read/kernel 开销 | 论文的 capacity 倍数绑定模型、硬件、trace 和 SLO |
| [vLLM 0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/#chunked-prefill) | “Chunked Prefill”与 “Performance Tuning” | decode 优先；较小 budget 倾向更好 ITL，较大 budget 倾向更好 TTFT；可混合 prefill/decode | 文档推荐参数是版本与硬件相关起点，不能当作 Qwen3-32B 的最优配置 |

## 6. Replicas、TP 与 PP 推理部署

### 6.1 三种方式改变的账本

**Replicas / data parallel serving**

- 每个 replica 保存完整模型权重，并处理独立请求批次；
- 增加 replica 主要扩大系统请求吞吐、隔离故障域和吸收突发；
- 单个请求通常只落到一个 replica，因此不会降低它所需的完整模型容量，也不会因“副本数增加”自动缩短单请求模型执行关键路径；
- 各 replica 的 KV cache 独立，负载均衡还要考虑队列和 prefix-cache locality。

**Tensor parallelism (TP)**

- 每层权重/算子切到多卡，降低单卡权重容量并聚合 HBM 带宽与计算；
- 每层需要 All-Reduce 等 collective 同步；较小的局部矩阵计算收益必须超过通信与同步开销；
- 快速卡间互联、模型放在单节点时更有利；跨慢速节点做高 TP 容易让通信进入关键路径；
- TP 可能降低单请求延迟，也可能因小 batch、逐 token 高频同步而变慢，不能只凭卡数判断。

**Pipeline parallelism (PP)**

- 把连续层段放到不同设备，降低每卡权重容量；只在 stage 边界传 activation，通信频率/带宽需求通常低于每层 collective；
- 一条请求仍要顺序穿过所有 stage，并承担边界传输；若 microbatch 不足或各 stage 不均衡，会出现 bubble/低利用率；
- PP 常适合跨节点或 TP 无法整除/互联较慢的容量场景，但不应默认它能降低单请求延迟。

### 6.2 决策边界

1. 模型单卡放得下且单请求 SLO 已满足：先把额外设备作为 replicas 的候选，比较总吞吐、队列、故障隔离与 prefix locality。
2. 单卡放不下：TP/PP 的首要收益是容量；先按互联域选切分，再验证 TTFT、TPOT、峰值显存和稳定并发。
3. 单请求确实需要更低延迟：只有当 TP 聚合的计算/HBM 收益超过 collective 时，切分才可能缩短关键路径。
4. 跨节点：优先考虑“节点内 TP、节点间 PP”作为起点，不是固定答案；NVLink/NVSwitch 等拓扑、stage balance 和 workload 会改变结果。
5. 报告容量时同时列出权重、KV、activation/workspace 和通信 buffer；报告性能时同时列出 TP/PP/replica 数、互联、batch、长度和延迟分位数。

一手资料与定位：

| 来源 | 具体定位 | 可支持主张 | 不应过度概括 |
| --- | --- | --- | --- |
| [vLLM 0.27.0 Data Parallel Deployment](https://docs.vllm.ai/en/v0.27.0/serving/data_parallel_deployment/) | 开头定义与 load balancing/KV cache 说明 | DP replica 复制权重、处理独立 batch；每个 engine 有独立 KV，路由影响 prefix cache | DP 主要扩系统容量，不代表单请求使用多个副本 |
| [TensorRT-LLM Deciding Model Sharding Strategy](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html) | “How to Think about Model Sharding: Communication is Key” | PP 在 stage 边界发送输出；TP 每层需要更重的 All-Reduce；快互联决定 TP 收益能否覆盖通信 | “单节点 TP、跨节点 PP”是文档的起点建议，必须 sanity check |
| [TensorRT-LLM Core Concepts](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html) | Tensor/Pipeline Parallelism 定义 | TP 通常更均衡但需要更高卡间带宽；PP 降低带宽需求但可能负载不均、利用率较低 | 这是机制性比较，不给特定模型的最优并行度 |
| [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) | 单卡、单节点 TP、多节点 TP+PP 建议 | 模型放不下时的容量导向部署起点；无 NVLink 时 PP 可能更合适 | 配置建议不是延迟/吞吐保证 |

## 7. 可直接登记到 `references/sources.md` 的 FACT 条目建议

以下编号承接当前 `FACT-007`，仅为登记建议；本轮不修改 `references/sources.md`。

| 编号 | 事实依据 | 相关章节 | 可核验主张 | 具体定位 | 建议状态 |
| --- | --- | --- | --- | --- | --- |
| FACT-008 | [NVIDIA NIM Metrics](https://docs.nvidia.com/nim/benchmarking/llm/2.0.0/metrics.html)；[vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)；[GenAI-Perf Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2700/user-guide/docs/perf_analyzer/genai-perf/README.html) | 03 推理 | TTFT、E2E、TPOT、ITL、output token/request throughput 的测量边界；TPOT 与 ITL 在聚合及多 token 流式响应时可不同 | NIM 的四个指标小节；vLLM CLI 的 TTFT/TPOT/ITL 公式与 speculative 示例；GenAI-Perf “Metrics”表 | 已核验 |
| FACT-009 | [Qwen3-32B 固定 `config.json`](https://huggingface.co/Qwen/Qwen3-32B/blob/1aecd22f7ae87d957e541aaf933345c33d8a3a26/config.json)；[官方模型卡](https://huggingface.co/Qwen/Qwen3-32B/blob/1aecd22f7ae87d957e541aaf933345c33d8a3a26/README.md)；[Scaling Book KV 公式](https://jax-ml.github.io/scaling-book/inference/) | 03 推理 | 标准 BF16 KV 公式；Qwen3-32B 为 64 层、8 KV heads、head_dim 128，因此逻辑 KV 为 256 KiB/token；模型卡原生 32,768 token 为 8 GiB/序列，固定 config 的 40,960 个位置为 10 GiB/序列 | config 对应字段；模型卡 “Model Overview”与 YaRN 说明；Scaling Book KV cache size 公式 | 已核验 |
| FACT-010 | [vLLM Quantization 0.27.0](https://docs.vllm.ai/en/v0.27.0/features/quantization/)；[TensorRT-LLM FP8 Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html)；[Qwen3-32B-AWQ 模型卡](https://huggingface.co/Qwen/Qwen3-32B-AWQ) | 03 推理 | 量化支持取决于格式、硬件和 runtime/kernel；校准与输出质量需验证；Qwen 官方 BF16/AWQ INT4 分数变化因任务而异 | vLLM “Supported Hardware”；TensorRT-LLM calibration/KV quality 提醒；Qwen “Performance”表 | 已核验 |
| FACT-011 | [Speculative Decoding 原论文](https://proceedings.mlr.press/v202/leviathan23a.html)；[Speculative Sampling 原论文](https://arxiv.org/abs/2302.01318)；[vLLM lossless 说明](https://docs.vllm.ai/en/stable/features/speculative_decoding/) | 03 推理 | draft + target 并行验证减少串行目标调用；标准修正采样保持目标分布；硬件数值和近似接受规则是输出一致性边界 | Algorithm 1、Theorem 3.5、Appendix A.1；Chen 摘要/算法；vLLM “Lossless guarantees” | 已核验 |
| FACT-012 | [ORCA OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf) | 03 推理 | iteration-level scheduling 允许请求在迭代边界进入/退出 batch；selective batching 处理不规则状态 | §3 “Iteration-level scheduling”“Selective batching”，Figure 4–5 | 已核验 |
| FACT-013 | [PagedAttention/vLLM 论文](https://doi.org/10.1145/3600006.3613165)；[vLLM Prefix Caching](https://docs.vllm.ai/en/v0.27.0/design/prefix_caching/)；[TensorRT-LLM KV reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html) | 03 推理 | paged KV 以固定 block 映射非连续物理空间并按需增长；prefix reuse 只复用匹配的完整前缀 block，受 block size、命中和淘汰影响 | PagedAttention §3、§4.1–4.4；vLLM hash/eviction；TensorRT-LLM full-block/LRU/offload 说明 | 已核验 |
| FACT-014 | [Sarathi-Serve OSDI 2024](https://www.usenix.org/system/files/osdi24-agrawal.pdf)；[vLLM Chunked Prefill](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/#chunked-prefill) | 03 推理 | chunked prefill 以 token budget 与 decode 交错；较小 budget 倾向改善 ITL，但会增加轮数、重复 KV read 和固定开销，TTFT/吞吐需权衡 | Sarathi §4.1–4.3；vLLM “Chunked Prefill”“Performance Tuning” | 已核验 |
| FACT-015 | [vLLM DP Deployment](https://docs.vllm.ai/en/v0.27.0/serving/data_parallel_deployment/)；[TensorRT-LLM Sharding Strategy](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html)；[Core Concepts](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html) | 03 推理 | replicas 复制权重并处理独立请求；TP 每层切分并引入 collective，PP 按层段切分并在边界传 activation；容量、互联、通信和延迟需共同比较 | vLLM DP 定义/KV locality；TensorRT-LLM “Communication is Key”；Core Concepts TP/PP 段落 | 已核验 |

## 8. 正文中应避免的高风险表述

1. 避免“TTFT 就是 prefill 时间”。应写“客户端 TTFT 包含排队、服务处理和网络；prefill 是主要组成之一，需另测服务端阶段时间”。
2. 避免“ITL、TPOT、每 token 延迟完全同义”。应注明请求内平均、逐间隔样本和跨请求聚合方式，特别是 speculative 多 token 流式响应。
3. 避免“prefill 一定计算受限，decode 一定显存带宽受限”。应写成常见形态，并保留短 prompt、batch、长 KV、量化、通信和 kernel 的边界。
4. 避免“Qwen3-32B 的 KV cache 是固定 8 GiB”。应写“BF16、标准全上下文缓存、单条 32,768-token 序列的逻辑 KV 为 8 GiB；实际随活跃 token、KV dtype、分片和 block 管理变化”。
5. 避免把 `num_attention_heads=64` 代入 Qwen3-32B 的 GQA KV 公式；应使用 `num_key_value_heads=8`。
6. 避免用 `hidden_size / num_attention_heads = 80` 推断 Qwen3-32B 的 head dimension；官方 config 明确给出 `head_dim=128`。
7. 避免把 Qwen3-32B 的 40,960 配置位置、原生 32,768 context 和 YaRN 131,072 混为同一口径；具体数字必须绑定 checkpoint 与 RoPE/YaRN serving 配置。
8. 避免“INT4/FP8 会按位宽等比例加速且基本无损”。位宽先改变 bytes；速度依赖硬件/kernel/shape，质量必须按实际任务验证。
9. 避免“speculative decoding 生成结果逐字不变”。标准随机算法保证目标分布；严格贪心可追求同 token，但 hardware numerics、batch 和近似接受规则仍是边界。
10. 避免“speculative decoding 一定减少计算量/固定加速 2×”。它主要减少串行目标调用，可能增加总计算；收益取决于接受率和草稿开销。
11. 避免“continuous batching 让新请求随时加入且不会影响延迟”。加入发生在调度边界，prefill 干扰、KV 压力和队列仍可能恶化尾延迟。
12. 避免“PagedAttention 减小每 token 的 KV 大小”。它主要减少预留和碎片，并支持共享；有效 token 的逻辑 KV 元素数不因 paging 自动下降。
13. 避免“prefix cache 命中语义相似 prompt”。标准复用要求模型执行上下文一致且 token 前缀匹配；它不减少 suffix prefill 或 decode。
14. 避免“chunked prefill 同时降低 TTFT、ITL 并提高吞吐”。chunk/token budget 本身就是 TTFT、ITL、吞吐和额外 KV read 之间的权衡。
15. 避免“多一倍 replicas 就让单请求快一倍”。replicas 主要扩系统吞吐和降排队，不切分单请求模型。
16. 避免“TP 卡数越多单请求越快”或“PP 天然低延迟”。TP 的 collective、PP 的 stage 边界/bubble 都可能进入关键路径；互联与 workload 决定净收益。
17. 避免脱离模型、精度、输入/输出长度、batch/并发、到达率、硬件、软件版本和延迟 SLO 引用任何加速倍数。

## 9. 正文采用时的最小验证清单

- 指标：明确客户端/服务端边界、流式方式、均值与 P95/P99、吞吐分子和计时区间。
- Workload：报告输入/输出长度分布、请求到达过程、并发和生成参数。
- KV：报告模型结构、KV dtype、实际活跃 token、block size、分片方式和运行时额外显存。
- 量化：核对模型×格式×硬件×框架版本支持，并跑目标任务质量与速度对照。
- Speculative：报告 draft 方法、草稿长度、接受率、严格/近似验收、额外显存和 target calls/token。
- 服务机制：同时报告 TTFT、TPOT/ITL、E2E、P95/P99、稳定吞吐、最大稳定并发和 KV 池压力。
- 多卡：报告 replicas、TP、PP、节点/互联拓扑、每卡显存、collective/P2P 时间和负载均衡方式。
