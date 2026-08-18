# 03 推理：从单次执行到并发服务

## 迭代状态

| 小节 | 状态 | 本轮约束 |
| --- | --- | --- |
| 3.1 工作单元与目标 | 草拟 | 在线与离线分开；明确指标口径与比较条件 |
| 3.2 一次请求的执行流程与特点 | 草拟 | 只做定性说明，不展开模型和硬件数值估算 |
| 3.3 单请求优化 | 草拟 | 量化、kernel/编译与 speculative decoding 统一回答四个问题 |
| 3.4 并发服务 | 草拟 | 用同一组长短请求串联 continuous batching、paged KV、prefix reuse 与 chunked prefill |
| 3.5 多卡部署 | 草拟 | 只比较副本、TP 与 PP 的容量和关键路径，不展开复杂 serving 架构 |
| 3.6 固定条件验证 | 草拟 | 同时报告速度、容量、质量和成本 |

## 3.1 先定义“快”：工作单元、目标与比较条件

同一个“token/s”可能回答三个不同问题：单个用户看到文本有多快、整台服务器每秒产出多少 token，或离线任务多久完成。开始优化前，先明确服务形态。

| 场景 | 主要目标 | 必须同时观察 |
| --- | --- | --- |
| 在线交互 | 首 token 尽快出现，后续输出平稳，尾延迟满足 SLO | TTFT、TPOT/ITL、端到端时延、P95/P99、稳定并发 |
| 在线非流式 | 请求在截止时间内完成 | 端到端时延、P95/P99、吞吐、超时率 |
| 离线生成 | 给定任务尽快、尽便宜地完成 | 总完成时间、总吞吐、设备时、单位输出成本 |

本章采用以下口径：

- **TTFT（Time To First Token）**：从客户端提交请求，到收到第一个输出 token；这里包含网络、排队、prefill 和产生首 token 的执行时间。若只测服务端或模型阶段，必须另行标明。
- **TPOT（Time Per Output Token）**：首 token 之后，生成其余输出 token 的平均时间；它会掩盖抖动。
- **ITL（Inter-Token Latency）**：相邻输出 token 的时间间隔；在线流式服务应观察其分布，而不只看平均值。
- **端到端时延**：从请求到达到完整输出结束。
- **吞吐**：必须写清是 output tokens/s、input + output tokens/s 还是 requests/s，并说明是单卡、单副本还是整个服务的聚合值。
- **稳定并发**：不是“显存最多塞进多少请求”，而是在给定到达分布和 SLO 下能持续服务的并发。

对自回归 LLM，一次完整生成由一次 prefill 和多次 decode 组成。比较不同方案时，仍需固定模型、输入/输出分布、精度、batch/并发和硬件，否则 TTFT、TPOT 与吞吐没有可比性。

## 3.2 拆解一次 LLM 请求：prefill、decode 与阶段特点

最基本的执行流程如下：

```text
请求到达并完成排队
        │
        ▼
prefill：处理全部输入 token
        │ 建立初始 KV cache，得到首个输出位置的 logits
        ▼
采样并返回第 1 个输出 token
        │
        ▼
decode：读取已有 KV 和最新 token，生成下一个 token
        │ 追加新的 KV，返回 token
        └──────── 重复，直到生成 EOS 或达到输出上限
                         │
                         ▼
                    释放请求状态
```

在基础实现中，prefill 对整段输入执行一次前向计算，建立后续生成所需的 KV cache，并从最后一个输入位置的 logits 采样首个输出 token。之后进入 decode：每一步只新增一个 token，复用已有 KV cache，并把本步产生的 K/V 追加进去。由于下一步必须等上一步的 token 产生，decode 会逐步串行执行。

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 每次处理的内容 | 一段输入 token | 每个活动请求的一个新 token |
| 依赖关系 | 输入位置可在一次前向中并行处理 | 下一个 token 依赖上一个 token，步骤之间串行 |
| KV cache | 为输入上下文建立初始 KV | 每步读取已有 KV，并追加新 KV |
| 主要影响 | 输入越长，通常越影响 TTFT | 输出越长，串行步数越多，通常越影响 TPOT 和端到端时延 |
| 常见执行特征 | 工作规模较大，通常更容易形成较大的矩阵计算 | 单请求单步粒度较小，低 batch 时更容易受权重/KV 搬运和运行时开销影响 |

这些特征不是固定标签：短输入、大 decode batch、超长上下文、量化方式和多卡通信都可能改变实际瓶颈。后续优化的核心，就是根据真实工作负载判断应缩短 prefill、减少 decode 的逐步开销，还是让多个请求更高效地共享执行资源。

## 3.3 单请求优化

### 少搬、少存：量化

量化必须分别说明权重、激活、KV cache、计算和累加格式。

| 四问 | 回答 |
| --- | --- |
| 解决的瓶颈 | 权重或 KV 容量过大；低 batch decode 的 HBM 搬运压力高 |
| 改变的账本 | 每元素字节下降；目标硬件的可用计算吞吐可能变化，也可能增加反量化或缩放工作 |
| 代价与副作用 | 输出质量可能变化；校准数据与量化方法会影响结果；硬件支持不等于目标 shape 有高效 kernel |
| 适用工作负载 | 容量/带宽主导，且目标硬件、框架和实际 shape 有对应 kernel，并通过质量验证 |

位宽可估算静态字节数，不能直接推导加速倍数；格式转换、算子回退和当前瓶颈都会改变收益。KV 量化还需单独验证长上下文质量。

### 提高复用、减少运行时开销：kernel、fusion 与编译

IO-aware kernel、fusion、编译和图执行主要减少中间张量落回 HBM、重复读写和 kernel 启动。

```text
优化前：读输入 → kernel A → 写中间张量 → 读中间张量 → kernel B → 写结果
优化后：读输入 → 融合/分块执行 A+B → 写结果
```

| 四问 | 回答 |
| --- | --- |
| 解决的瓶颈 | 中间搬运、临时张量、kernel 启动或调度开销主导 |
| 改变的账本 | 数学 FLOPs 可能基本不变，实际 HBM bytes、临时峰值和启动次数下降 |
| 代价与副作用 | 编译时间、shape/精度约束、图捕获限制和回退路径增加 |
| 适用工作负载 | 热点 shape 稳定、对应 kernel 成熟，且观测到开销确在关键路径 |

若瓶颈是最多 511 次串行增量 decode，单个 kernel 快 5% 不会变成 5 倍端到端加速；减少小中间张量也不会改写权重搬运下界。

### 减少大模型串行等待：speculative decoding

speculative decoding 让较便宜的 drafter 提议多个 token，再由目标模型并行验证：

```text
drafter:  t1 → t2 → t3 → t4
                     │
target :  一次验证候选区间
                     │
accept :  接受最长有效前缀，并继续生成
```

| 四问 | 回答 |
| --- | --- |
| 解决的瓶颈 | 目标模型每次只推进一个 token 的串行等待 |
| 改变的账本 | 每次目标模型调用可能确认多个 token，但增加 drafter 工作和被拒候选计算 |
| 代价与副作用 | 接受率依赖任务与 drafter；额外模型/头、KV 和调度增加复杂度 |
| 适用工作负载 | 目标模型单步昂贵、候选接受率足够高、验证可有效并行 |

标准 speculative sampling 可保持目标模型的采样分布，但不保证随机轨迹或固定 seed 逐 token 相同。验证需报告平均接受 token 数、drafter 开销、TPOT 与质量。

视频/扩散模型若通过步数、窗口或结构变化降时延，应归入算法—质量取舍，而非目标不变的 kernel 优化。

## 3.4 持续服务

单请求优化回答“这条关键路径能否更短”，服务优化还要回答“不同请求能否共享一次执行，又不让队列和尾延迟失控”。

下面始终使用同一组请求：`A/C` 共享 1024-token system prompt，`A/B/C` 已进入 decode，剩余输出长度分别为 `4/1/3`；`D` 携带长 prompt 新到达，`E` 已完成 prefill、等待 decode。所有 KV 都来自同一个 block pool。

### 从静态 batch 到 continuous batching

```text
静态 batch
step 1: decode A B C
step 2: decode A - C
step 3: decode A - C
step 4: decode A - -
新请求 D/E 即使已到达或就绪，也要等待整批结束

continuous batching + chunked prefill
step 1: decode A B C                    （B 完成，释放 KV blocks）
step 2: decode A C + prefill D[chunk 1] （下一迭代边界补入）
step 3: decode A C + prefill D[chunk 2] （C 完成）
step 4: decode A D E                    （D prefill 完成后进入 decode）
```

continuous batching 在迭代边界移除完成请求并补入工作，让多个请求的“当前一步”共享权重读取和矩阵计算。

收益取决于到达率、长度分布、调度和 KV 容量；更大 batch 可能提高吞吐，也可能增加单步时间和排队。

### KV cache 从连续预留变成状态池

若每个请求都按最大上下文预留连续 KV 空间，短请求会留下内部浪费；动态增长和释放还可能造成外部碎片。paged KV 改为按实际增长领取固定大小的 block。沿用上面的请求：

```text
step 1: [共享前缀 P×2][A][B][C][空][空]
step 2: [共享前缀 P×2][A][D1][C][空][空]  ← B 释放，D chunk 领取
step 4: [共享前缀 P×2][A][D1][D2][E][空]  ← C 释放，D/E 补入
```

`P` 被 A/C 共同引用，物理上只保留一份。paged KV 不减少有效 token 的逻辑 KV，且有元数据、末块和专用 kernel 成本；收益要落到稳定并发、OOM/碎片、吞吐或尾延迟。

### prefix reuse：用容量换掉重复 prefill

上例中的 A/C 共享完全相同的 1024-token system prompt。命中后，C 复用对应 KV，只处理自己的 suffix；在本章 BF16 示例中，这段前缀 KV 约占 `256 MiB`。复用要求模型执行上下文和 token 序列兼容，收益取决于前缀长度、命中率、驻留和淘汰。

prefix reuse 只减少命中的 prefill；低命中率时，冷前缀反而挤占活动 KV。

### 长 prefill 为什么会干扰正在 decode 的请求

上例把 D 的长 prefill 拆成 `D1/D2`，让 A/C 的 decode 继续推进。分块不删除总工作，过小会增加调度、kernel 启动和历史 KV 读取，也可能延后 D 的 TTFT。prefill–decode 解耦能进一步隔离资源，但会引入 KV 传输和路由，本章只列为进阶方向。

在这组请求中，continuous batching 减少空槽，paged KV 回收并重分配状态，prefix reuse 跳过重复 prefill，chunked prefill 缩小长工作块。只有这些变化转化为更大稳定并发、更低 P99、更高吞吐或更低单位成本时，系统层优化才闭环。

## 3.5 从单卡到多卡：增加副本还是切分模型

多一张卡有两种完全不同的用法：保存另一份完整模型接更多请求，或让多张卡共同执行同一份模型。

| 方式 | 每卡放什么 | 主要收益 | 关键代价 | 优先场景 |
| --- | --- | --- | --- | --- |
| 请求级 replicas | 每个副本都有完整权重，各自维护请求 KV | 聚合吞吐、降低排队，并为故障隔离提供条件 | 权重重复；健康检查、路由和容量冗余另有成本 | 模型单卡放得下，单请求执行已达标 |
| Tensor Parallelism | 一层权重和计算沿张量维度切到多卡 | 降低单卡权重压力，聚合 HBM 带宽与计算 | 高频 collective；KV 是否分片取决于 heads、degree 和实现 | 单层/模型放不下，或单请求需要聚合资源 |
| Pipeline Parallelism | 连续层分到不同 stage | 降低每卡驻留层数，可让多个请求占据不同 stage | 单请求仍依次经过各 stage；stage 不均与空泡增加等待 | 深层模型容量跨卡，且有足够并发填充流水 |

模型和目标并发能在单卡放下、单请求执行已达标时，优先比较 replicas；放不下时先比较量化，仍不够再用 TP/PP；只有计算或 HBM 下界确实主导单请求时，才把 TP 当作降时延候选。副本只能通过分流降低排队。TP 每卡少存、少读部分权重，但 KV 是否同步分片取决于 attention 布局和实现；小 batch 的逐 token 路径也更难用大矩阵摊薄固定通信，或与独立工作重叠。

判断切分是否值得，要把每卡节省的权重/FLOPs、聚合 HBM 带宽和可能分片的 KV，与新增 collective、同步、buffer 对照，再看 TTFT/TPOT、每卡吞吐和稳定并发。TP 后单请求更快但每卡吞吐下降，可能是合理的延迟—成本交换；通信让 P99 变差，则不能因“用了更多卡”视为优化。跨节点 TP、KV 传输和复杂路由留到补充材料。

## 3.6 固定条件验证：同时报告速度、容量、质量与成本

一次可信的推理实验先写假设，再只改变一个主要杠杆：

> 在 BF16、低并发 decode 中，实测单步时间远高于 FLOPs 下界并接近权重与 KV 搬运下界；只改变权重格式，验证权重 bytes、峰值显存、TPOT 和质量。

### 单请求基准

| 固定条件 | 报告指标 |
| --- | --- |
| 模型版本、采样配置、输入长度分布、输出长度/上限、精度、硬件、卡数、并行度 | TTFT、TPOT、ITL 分布、端到端时延、峰值显存 |
| 质量边界与测试集 | 任务质量、数值差异或可接受标准 |
| warmup、重复次数和是否包含排队/编译 | P50/P95/P99；需要时同时给估计置信区间 |

### 服务基准

| 固定条件 | 报告指标 |
| --- | --- |
| 请求到达过程、输入/输出联合分布、并发或请求率、SLO、调度配置 | queue time、TTFT、ITL/TPOT、端到端 P95/P99 |
| 副本数、每副本卡数、batch/token 上限、KV pool 配置 | input/output tokens/s、requests/s、最大稳定并发、KV block 占用 |
| 硬件时长与失败口径 | 超时/OOM/拒绝率、GPU-hours、每百万输出 token 或每请求成本 |

验证时按以下顺序读结果：

1. **速度是否改善目标指标**：吞吐提高但 P99 超出 SLO，不是在线目标下的成功。
2. **容量是否真的可用**：显存下降后，batch、活动 token 或稳定并发是否提高。
3. **质量边界是否保持**：量化、近似采样、步数、窗口和输出上限是否改变。
4. **复杂度与稳定性是否可接受**：编译回退、cache miss、碎片、抖动和故障恢复是否恶化。
5. **成本是否改善**：看整个服务的设备时和达标请求，而不是只看单卡峰值 token/s。

平均吞吐高但尾延迟差时，优先按请求长度和阶段拆开 queue、prefill、decode 时间，检查 batch 组成、长 prefill 干扰和状态池压力。优化后输出变化时，先判断工作负载或数学问题是否已经改变，再讨论速度。

停止条件也应预先写清：目标 SLO 已满足且继续优化不降低单位成本；容量已有安全余量；或下一项收益不足以覆盖质量风险和系统复杂度。没有停止条件的“继续榨利用率”很容易用用户延迟换取无效吞吐。

## 适用边界与讲述压缩

本章给出的是 dense decoder-only LLM 的一阶资源模型。真实结果还取决于 attention 结构、kernel、内存分配器、采样、框架调度、拓扑与流量分布。账本用于提出值得验证的假设，不能替代固定 workload 下的 benchmark 和 profiler。

对 MoE、线性/局部注意力、跨层 KV 共享、超长上下文或多模态模型，参数相关 `2P`、KV 公式和切分方式都要按实际激活参数与状态生命周期修正。MiniMax H3 等视频模型只有在公开执行顺序、状态复用边界和质量口径完整时，才适合进入同样的量化账本；本章不根据 LLM 机制反推其实现。

30 分钟压缩版保留：

```text
工作单元与指标
→ prefill / decode 流程与特点
→ 量化 + speculative decoding 的关键路径对照
→ continuous batching + KV block pool
→ replicas 或模型切分
→ 固定 workload 的验证卡
```

单请求优化只讲量化与 speculative decoding；kernel/编译沿用 01 的“数学不变、搬运减少”结论，不另设图。prefix reuse 和 chunked prefill 并入统一并发时间线；视频/扩散只保留工作单元边界卡；prefill–decode 解耦和复杂调度不进入主讲。

## 参考映射

- `REF-001`：prefill/decode 的资源模型、KV cache、batch 与模型切分的统一分析入口；
- `REF-002`：推理参数、KV cache、低 batch decode 权重搬运和时延下界的估算方法；
- `REF-005`：IO-aware Attention 与“数学工作不变、HBM 搬运减少”的解释；
- `REF-007`：按工作负载、cache、continuous batching、paged KV、prefix reuse 与部署方式组织推理；
- `FACT-002`：Qwen3-32B 的参数量、层数、attention heads、KV heads 与 head dim；
- `FACT-005`：H100 SXM 的 BF16 稠密峰值、HBM 容量与带宽，以及本章理论下界的硬件口径；
- `FACT-008`：TTFT、端到端时延和 ITL/TPOT 的计时边界；
- `FACT-009`：量化格式、硬件/kernel 支持与质量验证边界；
- `FACT-010`：speculative decoding 的 draft–verify 过程与输出分布边界；
- `FACT-011`：iteration-level/continuous batching 的调度粒度；
- `FACT-012`：paged KV 与 prefix reuse 的状态管理和收益边界；
- `FACT-013`：chunked prefill 的 token budget 与 TTFT/ITL 取舍；
- `FACT-014`：replicas、TP 与 PP 的切分、通信和部署边界。
