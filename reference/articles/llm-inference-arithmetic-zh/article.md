# 大型语言模型的推理演算

> 原文（英文）：kipply — Transformer Inference Arithmetic  
> 中文编译：OneFlow（杨婷、徐佳渝、贾川）  
> 来源：https://zhuanlan.zhihu.com/p/620170671  
> Fetched: 2026-08-13

---

![](https://pic1.zhimg.com/v2-24884fbee938e761302bc8041942ad74_1440w.jpg)

**作者｜kipply

OneFlow编译

翻译｜杨婷、徐佳渝、贾川**

本文详细阐述了大型语言模型推理性能的几个基本原理，不含任何实验数据或复杂的数学公式，旨在加深读者对相关原理的理解。此外，作者还提出了一种极其简单的推理时延模型，该模型与实证结果拟合度高，可更好地预测和解释Transformer模型的推理过程。

为了更好地阅读本文，读者需了解一些Transformer模型的相关先验知识，比如《图解Transformer》的大部分内容。另外，了解与本文相关的参数计数文章也能更好地帮助读者理解本文内容。

**目录**

*   kv 缓存 (kv cache) 解释了在推理过程中缓存自注意力向量（Self-Attention Vector）所带来的性能优化效果，以及可能导致的权衡（tradeoff）以及容量成本（capacity cost）问题。
*   容量（capacity）考虑了kv缓存的存储成本以及模型权重（model weight）的存储成本之间的联系，并解释了容量大小对模型性能的影响。
*   模型并行（model parallelism）可帮助我们理解张量并行（tensor parallelism），以明确通信成本（cost of communication）。
*   时延计算（latency calculation）需要从其他概念中获得理解，并创建用于确定推理速度底线（floorline）的方程。
*   批量大小（batch size）对性能的影响以及最优批量大小为多少。
*   通过Transformer块（transformer blocks）执行flops（每秒浮点运算次数）计数操作，可以识别对flops速度有实质性贡献的操作。
*   中间内存成本（Intermediate memory cost）涵盖了激活（即激活函数的输出结果）（activations）占用额外内存，以及一些真实基准（real benchmark）测试中的内存带宽成本。
*   对比真实基准测试（comparing against real benchmarks）是指将计算出的内容与Nvidia FasterTransformer基准测试结果进行对比，并确定其中的差异。

## **1. kv 缓存**

采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（kv）向量。这些向量存储在一个矩阵中，通常被称为kv缓存或者past缓存（开源GPT-2的实现称其为past缓存）。past缓存通常表示为：[batch, 2, num_heads, seq_len, features]。

![](https://pic1.zhimg.com/v2-ab65cbf2cc49e79e7ede92d571f674b8_1440w.jpg)

kv缓存是为了避免每次采样token时重新计算键值向量。利用预先计算好的k值和v值，可以节省大量计算时间，尽管这会占用一定的存储空间。每个token所存储的字节数为：

![](https://pic4.zhimg.com/v2-87e5c66794f14930fcd064c86a6a03e9_1440w.jpg)

第一个因子2表示k和v这两个向量。在每一层中我们都要存储这些k，v向量，每个值都为 n_{heads}\times d_{head}

矩阵。然后再乘以2，以计算每个向量所需的字节数（在本文中，我们假设采用16位格式）。

与token嵌入（token embeddings）相乘权重为 W_{k}，W_{v}\in R^{d_{model}\times d_{model}} ，其中每个token嵌入为 t_{e}\in R^{1\times d_{model}} 。这样，我们就可以算出所有层的k和v需进行的浮点运算次数为：

![](https://pic1.zhimg.com/v2-18a816c2afe45a369ff9425ec85c5d24_1440w.jpg)

将 t_{e} 乘以 W_{k} 需要进行

![](https://pica.zhimg.com/v2-6c4c464097a14b47d8e7a16c7c6fbbae_1440w.jpg)

次浮点运算。另一个2表示我们需要重复两次这样的操作，一次用于计算k和一次用于计算v，然后再重复所有层数 n_{layers} 。

> 矩阵乘法（matmul）中的浮点运算次数为多少？
> 
> 矩阵-向量（matrix-vector）乘法的计算公式是2mn

，其中 A\in R^{m\times n}，b\in R^{n}，。对于矩阵-矩阵（matrix-matrix）乘法，计算公式是 2mnp，其中 A\in R^{m\times n}，B\in R^{n\times p}。mn因子十分重要，因为它反映了矩阵乘法中由乘法和加法组成的组合方式，即“乘法(1)-加法(2) 操作组合”。更多内容见讲义（lecture notes）。

这意味着对于一个520亿参数的模型来说 （以Anthropic中的模型为例，dmodel=8192，n_{layers}=64 。），其浮点运算次数为：

![](https://pic2.zhimg.com/v2-0ae5b0d29b7a3839f68423ce9bceb021_1440w.jpg)

假设有一个A100 GPU，其每秒可执行的浮点运算次数为312e12，其内存带宽可达1.5e12字节/秒。以下数字仅涉及kv权重及计算的数值：

![](https://pica.zhimg.com/v2-03d33de346209450d529bcb8a713e500_1440w.jpg)

> Flops vs 内存有界性（Boundedness）
> 
> 英伟达使用了数学带宽这个术语，我觉得这个术语真的很可爱。从技术上讲，这种描述存在于每个内核中，但可以抽象为操作组。 
> 
> Flops vs 内存有界性是Transformer推理和深度学习优化的常见问题。为了完成所需计算，通常需要加载权重，加载过程会占用内存带宽。假设通过加载权重已经得到了很好的优化，那么我们可以在加载权重的同时开始计算。
> 
> 在这种情况下，flop bound意味着一段时间内存中没有任何数据传输；memory bound则意味着没有进行任何计算操作。
> 
> 英伟达使用数学带宽（math bandwidth）来描述该情况，我觉得相当有意思。从技术上讲，这种划分通常是指每个内核（kernel）中的计算量受限，但也可以指这些操作组的计算量受限，将每一组视为抽象意义上的单元。

现在模型架构不再重要了——在给定硬件规格的情况下，我们得到了一个明显的比率208。这意味着，我们计算一个token的kv所需的时间，与处理208个token的时间相同。若低于该值，会出现内存带宽限制，若高于该值，会出现flops限制。如果我们使用剩余的权重来完成完整的前向传递（即运行剩余的transformer），那么结果仍然是208（分子和分母各乘以6）。这一点我们会在后面的章节详细介绍。

下图的交点是208，不过实际上内存线（memory line）会有一些倾斜，这是因为中间计算（intermediate calculation）存在内存成本（上一节讨论过）。

![](https://pic3.zhimg.com/v2-61f426a91a7ac4d28e4ca026ffda979c_1440w.jpg)

对于拥有520亿参数的模型来说，一次完整的前向传递需要

![](https://pica.zhimg.com/v2-38b521a5bee9c44cd63ff2aa1ab7ebe6_1440w.jpg)

毫秒，这是处理208个token所需的时间（实际上我们会使用四个GPU进行并行处理，因此实际需要的时间约为17毫秒，后续章节将做详细介绍）。如果context存在416个token（即双倍token），那么处理时间将翻倍，而处理312个token所需的时间是处理208个token的1.5倍。计算kv缓存的token时，一个token所需的计算成本为模型中传递该token计算成本的1/6。总的来说，这些前向传递（获取logits、嵌入和训练时我们深有体会）非常便宜，因为可以进行并行计算。

相比之下，采样的成本要高得多，因为它需要强制读取每个token的所有权重，进行自回归预测。但这并不意味着时间节省了1/6！假设出现了flops bound，在每个采样步骤中，我们可以少进行

![](https://picx.zhimg.com/v2-babd76f1cec9e8b7379cf9db62263959_1440w.jpg)

次浮点运算，而解码步骤（step）需要进行

![](https://picx.zhimg.com/v2-e6e0e72ab4f4d912b6eb964c43c71513_1440w.jpg)

次浮点运算。

因此，在每个步骤中，我们节省的每秒浮点运算次数是序列中每个token每秒浮点运算次数的1/6（很大！），而且该数值会随着采样token数量的增加而增加。在没有kv缓存的情况下，随着token数量的增加，采样的时间复杂度（time complexity）将以平方级增加。考虑到存储缓存相关的开销和权衡（tradeoffs），以上说法并不全面。如果我们进行小批量设置，可能会出现内存带宽受限，而非flops bound。在这种情况下，我们可能不会使用过去的缓存，而是倾向于重新计算，这会消耗flops（因为我们已经支付了采样的内存成本）。

## **2. 容量**

我们对于GPU中存储的kv缓存和权重有了一定认识，并且了解到GPU容量确实对Transformer模型的推理性能有着重要影响，也就有了充分的理由对其进行评估。一般来说，Nvidia A100 GPU是用于推理的最佳GPU，其容量标准为40GB。虽然有一些GPU的容量高达80GB，并且具有更高的内存带宽（为 2e12 而非 1.5e12），但它们尚未被任何大型云服务提供商采用，因此，于我而言它们缺乏实际价值。将给定的参数计数（parameter count）乘以2，我们就可以获取相应字节数，进而算出拥有520亿参数的模型的权重大小。

![](https://pica.zhimg.com/v2-bc098894962ed1bca7334741cd148a60_1440w.jpg)

![](https://picx.zhimg.com/v2-dbb9fea472235c27dcf9af6952931ee3_1440w.jpg)

不过这显然不能在一个GPU上完成，我们至少需要三个GPU才能加载所有权重（稍后将讨论如何进行分区（sharding））。但这样就只剩下

![](https://pic2.zhimg.com/v2-9ae0e3defdc96dc031305b1eb6abd38b_1440w.jpg)

可用kv缓存了，这足够吗？让我们回到kv缓存内存中每个token的方程式，再次使用520亿参数大小的模型运行。

![](https://picx.zhimg.com/v2-dbb9fea472235c27dcf9af6952931ee3_1440w.jpg)

使用这种GPU设置，我们可以将

![](https://pic2.zhimg.com/v2-8b89f06db8aea59e1b2e42e4dcf68d19_1440w.jpg)

个token存储在kv缓存中。或者我们可以将batch size设置为4，其中每个请求最多包含2048个token（token越少所需batch size越大）。难办的是，我们想要做更高的batch size，但却受到容量限制。batch size越大，GPU处理相同请求所需的时间就越短，就能更有效地利用GPU资源。然而，batch size小，内存又会受到限制。在这种情况下，应该放弃使用kv缓存，选择支付flops成本。

同时使用四个GPU，就能处理

![](https://pic1.zhimg.com/v2-0f4d66b369aa8bacd62ea7cd269ae960_1440w.jpg)

个token。要想batch size更大，处理更多的数据，我们肯定会选择四个GPU。若只用一个GPU，会降低2/3的效率，这显然不是明智的做法。这不仅是batch size的问题，如果有大量的数据需要处理，那应该使用多个模型实例。我们的目标应该是，尽可能让每个实例都可以通过更大的batch size处理，因为无法避免支付存储权重的成本。中间计算步骤会占用一些额外空间，但这些空间可以忽略不计。

## **3. 模型并行**

这方面已有许多相关介绍，所以我就不再详细介绍模型并行（model parallelism）及其实现细节了，仅提供一些有用信息，以帮助读者做性能决策并计算通信成本。模型并行的最终结果：通过内存和flops传输的所有权重成本都被分摊到使用的加速器数量上。我们将采用张量并行（一种模型并行），将模型的中间进行划分。每个加速器将使用其权重分片（shards）尽可能多地执行操作，并在需要同步时进行通信。

相比之下，流水并行（pipeline parallel）更为简单，其中每个GPU将保留模型的一部分层。虽然这种方法的确平衡了权重加载成本，但存在明显缺陷：只有一个GPU运转，其他的都被闲置！在训练过程中，你可以采用流水并行（第一批数据移向下一个GPU时，新一批数据重新于第一个GPU开始），以便有效利用各个GPU。虽然处理多个样本请求时该方法十分有用，但在处理单个样本请求时，这种方法就不太奏效了。此外，无论flops是否受限，流水线并行都无法充分利用内存带宽。简而言之，流水并行适用于通信，而模型并行适用于计算。流水并行会在每个加速器之间进行dmodel次通信，而模型并行会在每个层内进行 N\cdot d_{model} 次通信，其中N是加速器的数量。

A100 GPU的通信带宽为300GB/s，而文档将其标记为600GB/s，这是因为NVIDIA 在每颗芯片上叠加了300GB/s的带宽，同时向外输出300GB/s的带宽，而没有使用双向数值（对于计算，这种方法更为直观）。

![](https://pic3.zhimg.com/v2-821d5b57c7474f24e767e3560c1d54e2_1440w.jpg)

首先，我们在图中黄色区域将token嵌入（token embedding）插入到模型底部。紫色盒子描述了权重在加速器上的分配情况，此处我们使用的是一个非常小的模型，以便能够按比例绘制所有内容。

普遍的做法是，我们可以将两个矩阵X和Y进行划分，并计算shards的乘积，但这并不能完成 X\cdot Y 的矩阵乘法（matmul）。换言之，如果我们只是简单地将shards的乘积连接在一起，那么得到的矩阵就会太大，无法存储或进行计算。相反，如果我们想要进行传输，可以计算出shard sum，并将shard sum进行通信，然后进行连接到输出 X\cdot Y。注意力机制通常涉及多头（head），因此采取并行计算非常直观。几乎不需要通信就能走完大部分注意力层，因为注意力头（attention head）是连接在一起的，需要乘以权重 W_{o} 进行计算。在乘以 \upsilon 后，我们需要将结果乘以shard的 W_{o}，以得到

![](https://pic3.zhimg.com/v2-8001d1aedc91f8e27863b1a53e59f04a_1440w.jpg)

然后，每个加速器会将自身的shard传递给其他加速器，其他加速器也会将其shard返回，其通信成本为

![](https://pica.zhimg.com/v2-dbf071d3fd0e8e36f8666c844535d352_1440w.jpg)

。每个加速器平等分配相加的shard，以获得输出投影（output projection，译者注：输出层的输出被称为输出投影,它是一种将神经网络的输出转换为特定格式的过程。）然后，他们会重复上次进行的通信，各个主机进行连接（近似瞬间就能完成）。其本质与MLP（Multi-Layer Perceptron）层类似！就像我们使用权重 W_{o}将多头注意力（multi-headed attention）的结果投射回长度为 d_{model}的向量一样。同样，我们也使用

![](https://picx.zhimg.com/v2-bb693ceffc9b9139a88e6b89768fdba7_1440w.jpg)

和

![](https://picx.zhimg.com/v2-7961f8002059e6109b9013334424d981_1440w.jpg)

将向量维度扩大4倍，并将其重新投射回原始值大小。因此，在MLP的末端完成了两次同样的通信。我们最终的通信量为

![](https://pica.zhimg.com/v2-78b1cce1de1cdd182d008523b17c90d0_1440w.jpg)

字节。GPU将对整个kv缓存进行划分，并将其分配到各个头部。

## **4. 时延计算**

我们已经较全面地讨论了容量（capacity）、模型并行中的通信，以及一般的计算步骤。接下来我们将其构建到估计时延的方程中！

![](https://pic1.zhimg.com/v2-00ef89d71e84f45e8fb7e17578e73af4_1440w.jpg)

时延计算大多与flops和内存有界性相关。如果每个参数需要进行的乘法运算很少，那么我们可能会受到内存带宽的限制。浮点运算量增加取决于批处理大小和参数数量，而内存只受参数数量的影响。通信方面，关键不是有界性问题，而是增加时延项和吞吐量项（即300GB/s）。由于时延方面的数据不透明，最理想化的估计也是每条消息发送大约需要8微秒。该数据源于Citadel的一篇论文，但该论文针对的是V100 NVLink。受计算因素的影响，计算单个token解码步骤的时延时间需要两个公式：一个用于内存带宽bound（小batch），另一个用于 flops bound（大batch）。处理大batch数据时，我们通常会忽略通信的时延因素。针对小batch（当batch size=1时，可以忽略batch因素）的方程如下（其中N为加速器数量，P为参数数量，b表示字节单位）：

![](https://pic3.zhimg.com/v2-526a33b5023c9b1b9dd716268dc8f82e_1440w.jpg)

因为我们需要通过内存传递所有参数，且每个参数都是2字节，所以这里是2*P。

A_{bm} 是加速器内存带宽，成本在加速器之间分摊。每层有

![](https://pic3.zhimg.com/v2-879352d4b254d07a68b3d7257e1cbdf8_1440w.jpg)

个通信请求，每个请求的时延较小，通常可以忽略。通信还有吞吐成本，但也可以忽略。有时读取kv缓存的时间也会对计算产生重要影响。不过，由于该因素取决于前后token的数量，而数量在同一批次内会有所变化，我们要采样的总token数量也有所不同，因此暂时将其排除在计算之外。这个时间会被计算为内存带宽时间。另一个缺失的内存带宽时间是读取unembeddings，以计算每个采样步骤的logits，即

![](https://pica.zhimg.com/v2-1970e257a11ec668694f3804aa3ab5b6_1440w.jpg)

如前所述，内存实际上并非保持不变，每个批次会使用一些额外的内存来存储中间激活值。我们之所以不计算这些额外内存，是因为其数量深受软件堆栈、编译器优化等因素的影响，很难精确计算。针对大batch（batch size=512）的方程如下（其中 B 是批量大小）：

![](https://pica.zhimg.com/v2-fa37fa8cdc80ca0cbdb028d7783d450c_1440w.jpg)

其中，A_{f}表示加速器的浮点运算能力，A_{c}表示处理单元之间的通信带宽。公式中做了2*P次浮点运算，这从对所有参数进行矩阵相乘也可看出。如前所述，因为

![](https://pic3.zhimg.com/v2-a4a1d48a8f62ff99eeac6ed9683456fe_1440w.jpg)

，所以矩阵向量乘法是2mn。在模型并行部分，每个层需要进行四次（N-1个因子，四舍五入为至N）d_{model}大小向量的通信。考虑到时延可以忽略不计，这里使用吞吐量来计算通信时间。将需要传输的总数据量除以通信带宽，可以得到通信所需时间。接下来，在16个GPU上使用2600亿参数的Gopher模型来进行推理。使用小batch推理时，生成一个token需要22毫秒的时间。通过大batch公式计算出来的通信吞吐量成本约为35微秒，因此可以放心地降低该成本。

![](https://pic4.zhimg.com/v2-2740272eec141cc5d382e71401d54795_1440w.jpg)

对于512的大batch，生成每个token所需的时间为53毫秒（即在62毫秒内生成512个token）。通信的时延成本也为3毫秒（由于消息可以一起准备，因此延时延不会随着batch的增加而增加），这对于减少时延来说有一定的意义，但如果假设通信和计算是并行的，那么这也可以接受。

![](https://pic3.zhimg.com/v2-369e680cde15d6920b65d053fd9f3d16_1440w.jpg)

在假定并行处理的情况下，我们会选择计算和通信中较大的值。因此，我们希望避免通信时间大于计算时间（这是防止随着芯片数量的增加趋近于零时延的机制，最终通信时间将占用越来越多的时间）。但并不能保证所有系统都能够完美地并行处理。这些数值明显比实际应用环境中得到的值要低得多。在这个环境中，程序只能使用被授权的资源，无法对系统或其他程序产生影响），因为它假设了最佳的硬件使用，没有考虑softmax，假设没有通信时延，并忽略了许多其他较小的因素。尽管如此，这些数学推理仍有助于思考如何优化性能以及未来优化所带来的变化。

## **5. Batch sizes**

（Batch sizes）批大小是性能的一个重要因素，尤其是对特定用途性能的理解。我们在前面部分进行了两个计算，用于确定何时是内存带宽bound，何时是flops bound。为了确定主导因素，我们可以对这些数值进行对比。

![](https://pic2.zhimg.com/v2-a610d9e53fed98bf325bdf02d23f8a49_1440w.jpg)

我们正在处理与kv缓存部分相同的比率。内存带宽bound的最小batch size为，A_{bw}/A_{f}=208。该比率非常有用！在足够负载的情况下，我们更倾向于flops bound，因为这样计算效率更高。但是，如果是flops bound，增大batch size并不会提高计算速度。很容易计算出容量中的主流何时从kv缓存转变为权重，但这之间并没有明显的二进制分界点（当kv缓存开始占用更多内存时并不会有特别的变化）。此外，也没有特别重要的通信因素。随着batch size增加，吞吐量开始超过时延，所以我们不再考虑时延因素。

正如之前观察到的，时延变得不重要的时间相对较晚（例如，52B通信成本上的512批大小仍有11%的时延）。将其简化一下，即通信需要在四个不同的步骤中进行，这意味着我们不仅希望计算时间比通信时间长，而且在每个步骤中都是如此（如果我们可以同时进行计算和通信）。为了达到这个目标，我们使用一个更奇特的比率，即每个字节的通信量需要多少次浮点运算。下图是一个很好的计算表，后续内容也将会用到。

![](https://picx.zhimg.com/v2-93e744fc9665ee7be5400374516f385d_1440w.jpg)

A100芯片每字节通信的浮点运算次数为(312e12/300e9)=1040次。我们希望最后一行的值大于硬件每字节的浮点运算次数，以保持flops bound（假设没有memory bound）。对于任何embedding维度超过1024（每个芯片）的模型，相对比较安全！但对于512维度的模型而言，情况就有些棘手。API负载较低时，会出现较小的batch sizes，这时会考虑放弃kv缓存等决策。当API负载较高时，为了优化每个请求的时延，会选择提供最低batch size，以达到flop bound，即使仍有容量剩余。在AlphaCode等大规模推断任务中，我们通常会尽可能多地插入芯片，然后利用这些容量执行最大的batch操作。虽然我经常使用“可能”一词，但这些情况都是绝对存在的。

## **6. Flops计算**

之前：

> 从所有参数的matmul操作来看，我们的flops运算为2*P次。

这是很合理的推理，我们还可以检查transformer步骤来分解推理，以核查是否能得出2P这个结论。以下是对token和layer的计算。准确来说，W_{q}，W_{k}，W_{v}\in R^{d_{model}\times d_{model}}应该表述为 W_{q}^{i}，W_{k}^{i}，W_{v}^{i}\in R^{d_{model}\times d_{head}} ，其中i最大为

![](https://pic2.zhimg.com/v2-60aeb5b33a37017723bcf0954e6a547b_1440w.jpg)

。为了计算时延，我简化了 W_{q}，W_{k}，W_{v}，以包含所有heads。

![](https://picx.zhimg.com/v2-c42c81678baab5b7d841d57f434567b3_1440w.jpg)

将所有flops相加！

![](https://picx.zhimg.com/v2-017ed906207e969ad9ccd468196dc5bf_1440w.jpg)

在8192模型中，flops大约为100B；

![](https://pic3.zhimg.com/v2-fbd2ffe1c9c366853cab384bede18ee4_1440w.jpg)

103079215104除以2约等于515亿。我们得到的结果为515亿，比520亿略少一些，这是因为token（un）embeddings的参数接近了10亿，相比 2\cdot P，

![](https://pic2.zhimg.com/v2-d16d96a32d9a7643c55b49dec04e4151_1440w.jpg)

更适合用来计算时延，但这两者的差异不到2%。该如何计算Z和其他步骤呢？这些都是向量到向量（甚至向量到标量）的运算，所以它们都是围绕为 d_{model}因子建造起来的，而不是为 d_{model^{2}}因子。即使每一层（layer)要运行100次这样的操作，最终也会有一亿次flops, 占已计算flops的0.1%。

## **7. 中间内存成本**

Data Movement Is All You Need（_[https://arxiv.org/pdf/2007.0007 2.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2007.00072.pdf)_，主要内容是优化transformers的低层级数据移动，与本文内容不太相关）一文提出了一个很好的分类操作方法。首先，在大矩阵（包括线性层）当中，张量缩并（tensor contractions）是最重要的，统计归一化（包括softmax和layernorm）次之，最后是逐元素操作，比如偏置（ biases）、dropouts和激活（activations）等。那么我们该如何计算矩阵、layernorms等的时延呢？我们硬件上报告的flops是专门针对乘加运算的，因此即使我们可以计算flops，其结果也不对。

幸运的是，这只是为了占用内存来进行softmax读/写，因为这有利于带宽和flops比率。这是已知的时延因素！在这里我将抛开第一性原则，讨论 Data Movement Is All You Need一文中的表格A.1。我们发现softmax的时延时间略长于qkv的计算时间（softmax时延时间是qkv操作时间的三倍）。这有点令人担忧，可能会影响整个神经网络的性能。

![](https://pic3.zhimg.com/v2-0f39a753b34e6772afeede8658ed13d8_1440w.jpg)

出于同样的原因，softmax会受到内存限制，所以qk、ReLU和dropout的乘法（multiplication）操作也相当昂贵。

> GPU内核融合（Fusion）
> 
> GPU 以“内核”为单位执行操作。内核融合意味着2个内核可以融合为一个，这样我们就可以在内存中重复利用负载，减少冗余负载和存储。例如，一个乘加（multiply-add）是一个内核。但是如果一个乘加有两个内核，一个内核负责加载+加法+存储，另一个内核负责加载+乘法+存储。内核融合以后，我们可以运行加载+加法+乘法+存储，以简化步骤。

通过计算所需的读写次数，我们发现这里的softmax没有完美融合。理论上它可以是一次读取和一次写入（标准次数为四次）。qk是两次读取和一次写入（两次读取可能可以保存）。三比一的比率表示softmax执行的内存传递量多于最佳值。

我这样说是因为，这表明了计算的软件依赖程度，并且需要通过实验来验证，因为从理论上来说成本可能为0。值得注意的是，随着模型大小的增加，操作所花费的时间百分比会迅速降低，因为每层内存将以 d_{model}增加，flops将以为 d_{model^{2}}增加。本文为336M参数模型，

![](https://pic3.zhimg.com/v2-58305a2f0298c3176dab2d6471a26a80_1440w.jpg)

我将“Ours”列中内存限制的所有值的时延相加（包括 element-wise操作）。结果显示，中间步骤占用了43%的时间。可以看到，这些操作在大小为520亿（d_{model}是这个模型的八倍）的模型中没有那么重要。这些memory bound中间操作的持续时间将延长8倍，因为这些操作是长度 d_{model}的向量。但是，flops次数将增加64倍，这意味着flop时间会延长64倍。

![](https://pic4.zhimg.com/v2-4ced6f86a0dc0609773e9f36d58830fd_1440w.jpg)

因此，若不考虑时延，使用 Data Movement Is All You Need中的优化方法，520亿参数模型的推理时延大约会是中间计算的 5%。

## **8. 实际基准对比**

我在语言建模公司工作，我们公司有自己的基础设施和基准，但IP是公司面临的一大难题。可悲的是缺乏可用于模型并行推理的公共基准。目前我只知道Nvidia的FasterTransformer和Microsoft的Deepspeed，可能其他我不了解的论文也提出了一些基准。

无论如何，我们可以根据真实基准来验证计算！因为我只想用2个GPU，所以使用了FasterTransformer来运行了一个130亿参数的模型。该模型执行一连串内核融合，并提供张量并行功能。模型有40层，每层有40个头，每个头的维度为128，总维度大小为5120。这里是配置文件截图，这些截图显示了许多有趣的细节，可能值得单独撰写一篇文章。首先是输出的512个上下文长度（context length）、batch size为1和10个token。对于2个GPU上的一个小batch的token，我们预计为8.4毫秒，大约1毫秒的通信，那么一个GPU则是16.8毫秒，0毫秒的通信。（2x40x12x5120^2/1.5e12）

> 在这里我应该将内存宽带（mem bandwidth）设置为1.555，而不是1.5。

实测结果表明，1个GPU应该是22.0ms，这意味着，我们的猜测的准确率为76%。可以确定，其中一部分时间花在了中间激活操作上，从理论上来说，我们可以获得100%的内存带宽，但实际上并没有。对于这些维度，测试表明我们可以获得高达90%的内存带宽利用率（我们将矩阵乘法的预期成本与单个矩阵乘法内核的持续时间进行比较，由于加载的张量不同，带宽利用率会有所变化）。考虑到这一点，我们预计需要18.5毫秒。加上中间激活操作的成本后（我们可以从测试结果中获得），需要额外的2.2毫秒，总共需要20.7毫秒！为了解释剩下的1.4毫秒，我们考虑到了一些其他的亚毫秒级操作，比如token嵌入、top-(k|p)操作、少于90%的带宽利用率（不想计算平均值，直接取了我能找到的最高带宽利用率），或者是内核启动时间。实验结果显示，使用2个GPU时，总时间为13.5毫秒。

相比一个GPU，这次我们只占了内存宽带的76%，离目标还有很大的差距。为了解决这个问题，我们重新检查了配置文件，发现内存带宽稍微差了一些，因为张量较小获取的带宽也比较少。经过计算，宽带没有达到90%，只有87%，总时间为9.5毫秒，中间的激活时间需要大约2毫秒，这使得总时间为11.7毫秒。剩余的1.5毫秒需要找出通信问题。但是这个问题很容易解决，因为我们之前计算的1毫秒通信时间没有被并行化。根据配置文件的数据，每个层的通信时间为40-50微秒，总共约为1.7毫秒，这就是很好的证明。

上述两种操作的中间激活计数都比应有的要高一些，因为配置文件提供的时延始终略高于原始基准测试运行。基准测试运行的输出为180.86ms（上下文时间：45.45ms）和 283.60ms（上下文时间：63.17ms）。在前向传递过程中，模型需要将所有token发送到每个GPU上，然后每个GPU都会对其进行自己的注意力头计算并存储kv。由于这个过程需要发送大量的数据并进行并行计算，因此预计前向传递需要的时间会比解码步骤长num_tokens/flops_to_bw_ratio倍。

更新的内存带宽为：312e12/（1.5e12x0.9）=231。在1个GPU的设置中，22ms是我们预期的解码步骤，我们可以得到22*（512/231）= 48，并不是63。在2个GPU设置中，我们通过计算得到了更加糟糕的结果：13.5*（512/231）=30ms。单个GPU只缺失了部分kv储存时间。查看配置文件，我们发现kv储存时间每层为18微秒，总共为0.7毫秒。Memsets占了0.2毫秒。我们预计其中一个MLP乘法的flop时间（flop bound）为512x4x5120^2x2/312e12 = 344微秒。实际上，最低浮点运算时间应该是476微秒，也就是说，我们得到了预期flops的72%。

对于attention中的投影（projection），我们期望是512x5120^2x2/312e12 =86微秒。但是在配置文件中，我们发现最低是159微秒。请参阅本文的图14，其中512x4000x4000的最终结果低于150TFLOPs/s。

## **9. 练习**

1.在给定batch size、上下文长度和next_n的情况下，我们该如何计算kv缓存节约的宽带？

2.kv缓存会增加内存时间开销吗？

3.我们是否可以在前向传递时选择memory bound，在采样步骤中选择flops bound？

4.在GPU超过容量所需的情况下，我们应该进行怎样的权衡和计算？例如，一个520亿参数大小的模型（拥有8或16个GPU）。

5.如果我们有计算预测一个token时间的公式。我们应该如何计算执行整个样本的时间呢? 是否应该先在上下文上进行前向传递，再预测所有的请求token？

6.在容量部分，我曾提到中间计算的内存可以忽略不计。那么这些内存到底有多小呢？

7.在batch size部分，我们讨论了每字节通信的token数。如果我们的嵌入维度为512，我们需要做怎样的权衡？

8.假设GPU都连接到了同一主机，但可以像训练那样在主机之间进行GPU通信。AWS 有400GB/s。这种情况该怎么办呢？

9.在模型并行部分，我们可以实际沟通所有分片（shards），然后让每个加速器执行所有添加（不只是其中一部分添加）。这部分会对时延产生怎样的影响呢？10.在batch size为256的4xGPUs上计算520亿的大批量速度。计算约为21毫秒，通信约为4毫秒。

11.从最后一层中取出向量，将其乘以未嵌入矩阵，存储logits，然后进行top-k或top-p采样（需要排序）。对于一个包含520亿参数的模型，这个过程需要多长时间，我们可以在这里并行化什么操作？

12.如何进行分片token嵌入？在输入token嵌入和未嵌入token之间，是否需要以不同的方式划分分片和使用层归一化（Layernorms），这会引起额外的通信开销吗？

其他人都在看

*   [“ChatGPT们”的淘金时代](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247491185%26idx%3D1%26sn%3Ddc25c7666718b30fae825a13861931b5%26chksm%3Dfe419047c9361951d96c2203727cfb4c4dd6a1dea3f52413daae875353a25d6616f114d44456%26scene%3D21%23wechat_redirect)
*   [GPT-4，大增长时代的序幕](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247491126%26idx%3D1%26sn%3D5aa6a7705b5f12ca7148a820ce03d82a%26chksm%3Dfe419000c9361916fbc10c2ea50c8aa7ee3922b23ff20a6ca2ff8f06acefa18e736f78a384e0%26scene%3D21%23wechat_redirect)
*   [GPT-4创造者：第二次改变AI浪潮的方向](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247491133%26idx%3D1%26sn%3D1e967f1c657f8781f8d876515047e5d5%26chksm%3Dfe41900bc936191d7756b74d23c85acba6e8b14f6e4c7ea48033b3959cb43011a9a2d81e0dee%26scene%3D21%23wechat_redirect)
*   [ChatGPT作者Schulman：我们成功的秘密武器](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247490988%26idx%3D1%26sn%3D664b9b90504c75ed0f27b0651f3aad2f%26chksm%3Dfe41939ac9361a8c400c2310c4297ef6a80f626731e32cca276504e4b4b8c9a38c6d63eea3f1%26scene%3D21%23wechat_redirect)
*   [比快更快，开源Stable Diffusion刷新作图速度](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247489843%26idx%3D1%26sn%3Df32143165779a5c96d4861d657286ea6%26chksm%3Dfe419705c9361e13ef901009ec002ac74ea8e7c269f8cb21c6d6edb8671e08acf1a836f76b7f%26scene%3D21%23wechat_redirect)
*   [OneEmbedding:单卡](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247488841%26idx%3D1%26sn%3D6c5de6713dbce2bd25d60db742879368%26chksm%3Dfe419b7fc93612690a667bec72a258ac837c2454cd180c2432c1c190bc236743cf7e4dd0dfee%26scene%3D21%23wechat_redirect)[训练TB级推荐模型不是梦](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247488841%26idx%3D1%26sn%3D6c5de6713dbce2bd25d60db742879368%26chksm%3Dfe419b7fc93612690a667bec72a258ac837c2454cd180c2432c1c190bc236743cf7e4dd0dfee%26scene%3D21%23wechat_redirect)
*   [GLM训练加速：性能最高提升3倍，显存节省1/3](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzU5ODY2MTk3Nw%3D%3D%26mid%3D2247490555%26idx%3D1%26sn%3D280c4ac043a31170236a9d8fba5fc2d2%26chksm%3Dfe4195cdc9361cdb6db1cb3b77c66b45c353b2fb65c4d082fbae9b72fe0b3a441ab9101cb147%26scene%3D21%23wechat_redirect)
**欢迎Star、试用OneFlow:**
