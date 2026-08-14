# 第一章 · 计算（content/v3）

依据 `specs/c1/outline.md`。1.1.3 / 1.2.1 的机制来自 `reference/`（How to Scale Your Model；Ultra-Scale Playbook 精度两图）。
贯穿实例的规格在 `content/v3/00-overview/`，本章只引用。

---

## 0. 本章定位

**如何分析一次大模型输出需要多少算力？**

1. 要算多少次浮点运算（FLOPs 估算）
2. 用什么数据格式算（精度体系）
3. 机器一秒能算多少（硬件算力）
4. 实际能拿到多少（MFU）

---

## 1.1 运算量统计

### 1.1.1 基本概念

一次浮点乘法或加法各计 1 FLOP。乘加（MAC）= 2 FLOPs。

### 1.1.2 基本思路

Transformer 里 95%+ 的浮点运算集中在少数 GEMM 上，「数 FLOPs」≈「数矩阵乘」。

```
[m,k] × [k,n] 的 GEMM = 2mkn FLOPs        （乘、加各算一次）
```

图：每个输出元素 k 次乘 + k 次加；m×n 个输出 → 2mkn。

### 1.1.3 Dense Transformer

据 *How to Scale Your Model*：一条权重为 P 的 GEMM，前向 2P（每 token），反向再算 dW 与 dX，训练合计 6P。

- 每 token 前向 **≈ 2N**（N = 参与逐 token 矩阵乘的参数，不含 embedding 查表）
- 训练 **C ≈ 6ND**（D = token 数）
- 单层典型拆分（hidden = h）：注意力投影 QKVO ≈ 4h²（MHA）；FFN 两矩阵扩张 4 ≈ 8h²，SwiGLU 三矩阵 ≈ 12h²。层内合计常记 **≈12h²**（两矩阵 FFN）。
- 注意力**分数**（QKᵀ、AV）与参数量无关，随 s²。前向口径 **attn / linear ≈ s/(6h)**。对话场景 s≈4K、h≈4K 时约 **0.17**。训练口径因反向把线性放大 3×，阈值更接近 T≈8h。

### 1.1.4 Qwen3-32B

官方：32.8B / 非 embed 31.2B；64 层；h=5,120；GQA 64/8；head_dim 128；SwiGLU intermediate 25,600。

**对 2N / 6ND 的调整**

- 真实单层 ≠ 12h²：Q 维 8,192≠h，GQA 缩小 K/V，SwiGLU 三矩阵且 intermediate=5h。逐层实算 ≈487.6M ×64 = 31.2B，与官方非 embed 一致。每 token 前向用 N≈32.0B（含 lm_head）→ **≈64 GFLOPs**。
- 注意力分数按 d_q=8,192 计，不是 h。短对话仍远小于 1；到 YaRN 131K 分数项会超过线性投影。

### 1.1.5 MiniMax H3 DiT

官方：33B dense；AdaLN ≈13B，调制只依赖 timestep，推理可缓存。公开结构：50 block，hidden 5,376，56×128 heads（投影内维 7,168 > h）。

**对 2N / 6ND 的调整**

- 推理逐 token 路径 **N_eff ≈ 20B**（33B−13B），不是 33B。每 token ≈ **40 GFLOPs**。
- 训练时 AdaLN 要算梯度；扩散训练每样本通常只采一个 timestep，**训练侧不乘采样步数**。
- 潜空间 token 数随分辨率²×时长，attn/linear 从约 **0.94** 到 **10**（takeaway 口径 s/(6h)）。

> 不是 DiT 换公式，是同一公式谁先越过 6h，谁的账先变。

---

## 1.2 精度体系

### 1.2.1 基本概念

参考 Ultra-Scale Playbook 的两张直觉图：位域（符号 / 指数 / 尾数）与「范围 vs 精度」。

| 格式 | 位宽 | 指数 | 尾数 |
|---|---|---|---|
| FP32 | 32 | 8 | 23 |
| TF32 | 19 有效 | 8 | 10 |
| FP16 | 16 | 5 | 10 |
| BF16 | 16 | 8 | 7 |
| FP8 E4M3 | 8 | 4 | 3 |
| FP8 E5M2 | 8 | 5 | 2 |

BF16 与 FP32 同指数宽，用精度换范围，训练通常不需要 loss scaling；FP16 需要。

混合精度：FP32 主权重 + BF16/FP16 计算 + FP32 累加。FP8 必须配 scaling。

### 1.2.2 Qwen3

默认 `bfloat16`。官方 **Qwen3-32B-FP8**：fine-grained FP8，block 128。社区常见再走 AWQ/GPTQ INT4。同代 LLM：训练 BF16，推理 BF16 或 FP8，端侧 INT4。

### 1.2.3 MiniMax H3

权重 BF16。部署时 DiT 可在线 FP8；VAE 与 Qwen3-VL 文本编码器通常不一起量化。AdaLN 不在逐 token 热路径，精度可单独定。扩散步串联，误差按最终画面评估；流式视频 DiT 同此约束。

---

## 1.3 硬件算力

### 1.3.1 Nvidia GPU

```
算力峰值 = SM 数 × 每 SM 每周期 Tensor Core MAC 数 × 2 × 频率
```

### 1.3.2 代际数据

稠密峰值 TFLOPS（spec sheet）：

| GPU | BF16 | FP8 |
|---|---|---|
| A100 | 312 | — |
| H100 SXM | 989 | 1,979 |
| H200 | 989 | 1,979 |
| B200 | 2,250 | 4,500 |

### 1.3.3 数值锚点

> **H100 SXM ≈ 1 PFLOPS BF16（稠密）**——记住这个数就够推大部分账。

### 1.3.4 算力估计的陷阱

1. **稀疏峰值 = 稠密 × 2**，宣传数字常用前者
2. **TF32 只是 FP32 的入口**，实际走 Tensor Core
3. **标称频率 ≠ 持续功耗下的实际频率**

---

## 1.4 MFU：从峰值到现实

```
MFU = 实际 FLOPs/s ÷ 峰值 FLOPs/s
```

重计算的 FLOPs：HFU 计入分子，MFU 不计。

| 场景 | MFU |
|---|---|
| LLM 稠密训练 | 35–50% |
| LLM MoE 训练 | 更低 |
| LLM decode | **< 5%** |
| **DiT 推理** | **天然较高**（始终计算受限） |

### 双栏 MFU 瀑布

左栏 LLM decode、右栏 H3 流式。分段：峰值 → 非 GEMM → 访存受限 → 通讯 → 流水气泡 → 实际 MFU。两栏损失构成不同；各段对应后面章节，兼作路线图。

### 实时性预算

目标：24fps 实时 = 每秒 6,048 token。
起点：单卡 220 PFLOPs / 9 分钟。杠杆：`分辨率 × 步数 × 注意力形态 × 卡数`，一步步逼到实时。

---

## The Takeaways

| | LLM | H3 / 视频 DiT |
|---|---|---|
| 每 token 前向 | 2N | 2N_eff（注意 AdaLN） |
| 训练总量 | 6ND | 6N_eff·D，无步数乘子 |
| attn 占比 | s/(6h) ≈ 0.17 | 0.94 ~ 10 |
| 推理成本 | 2N/token | steps × cfg × (...) |
| decode/流式瓶颈 | 访存受限 | **计算受限** |
