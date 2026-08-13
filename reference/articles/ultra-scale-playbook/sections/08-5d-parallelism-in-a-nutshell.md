## 5D parallelism in a nutshell

Congratulation reader, you have now seen all 5 parallelism strategies you can
use to scale model training:

  1. Data Parallelism (DP) – along the batch dimension
  2. Tensor Parallelism (TP) - along the hidden dimension
  3. Sequence and Context Parallelism (SP/CP) - along the sequence dimension
  4. Pipeline Parallelism (PP) - along the model layers
  5. Expert Parallelism (EP) - along the model experts

As well as the 3 ZeRO strategies which can be combined with Data Parallelism for
memory reduction:

  1. ZeRO-1 – sharding optimizer states among the DP replicas
  2. ZeRO-2 – sharding optimizer states and gradients among the DP replicas
  3. ZeRO-3 – sharding optimizer states, gradients and parameters among the DP replicas

At this stage, one aspect you are probably curious about is how all these
parallelism and ZeRO strategies compare to, and interact with, each other. In
other words, which ones should we use and efficiently combine together, and
which ones should we rather keep separated?

Let’s take a look at the similarities and interplay. We'll start by comparing
Pipeline parallelism are ZeRO-3 side-by-side as they have some very close
similarities but also important differences.

**Pipeline parallelism vs. ZeRO-3 -** Both PP and ZeRO-3 are ways to partition
the model weights over several GPUs and perform communication/computation along
the model depth axis (for example in ZeRO-3, we prefetch the next layer while
computing). This means in both cases full layer operations are computed on each
device, as opposed to TP or EP for instance in which computation are performed
on sub-layer units.

In the following we say “a layer” to simplify what should be in general called
“a set of layer” (as the basis sharding unit of the model).

However, there are a few major differences between PP and ZeRO-3 approaches:

| **ZeRO-3** | **Pipeline Parallelism**  
---|---|---  
Each compute unit stores  | only a fraction of a layer | a full layer  
Communication is used to transfer | weights | activations  
Orchestration | model agnostic | model agnostic  
Implementation challenges | Complex to handle model partitioning and communications | Complex to handle efficient PP schedules  
Scaling considerations | Prefers large mbs and seq\\_len to hide comms | Prefers large \text{grad\\_acc} to hide bubble  
  
As you can see, ZeRO-3 and PP solve the same challenge but involve different
approaches and the choice between both will depend whether you decide to focus
communication either on weights or on activations. While they can be combined,
it's not often done in practice as doing so requires increasing the global batch
size significantly to amortize the communication costs, creating a tradeoff
between global batch size, model size, network bandwidth, and training
efficiency. If you decide to combine them, ZeRO-3 should be configured to keep
the weights in memory during the series of PP micro-batches to minimize as much
as possible un-necessary communication overhead.

On the other hand, ZeRO-1 and ZeRO-2, which focus on optimizer states and
gradients, can be easily combined with Pipeline Parallelism and are
complementary to it. Combining them don't raise any particular new challenge.
For instance, the training of DeepSeek-v3 used PP combined with ZeRO-1 (sic).

**Tensor Parallelism** (with Sequence Parallelism) is naturally complementary
and can be combined with both Pipeline Parallelism and ZeRO-3 as it relies on
the distributive property of matrix multiplications which allows weights and
activations to be sharded and computed independently before being combined.

![TP & SP diagram](/assets/images/5d_nutshell_tp_sp.svg)

The main reason we don't want to use TP only for parallelism is that, in
practice, TP has two limitations we've discussed in the previous sections:
First, since its communication operations are part of the critical path of
computation, it's difficult to scale well beyond a certain point at which
communication overhead begins to dominate. Second, unlike ZeRO and PP which are
model-agnostic, TP requires careful handling of activation sharding - sometimes
along the hidden dimension (in the TP region) and sometimes along the sequence
dimension (in the SP region) - making it more cumbersome to implement correctly
and requiring model-specific knowledge to ensure proper sharding patterns
throughout.

As a consequence, when combining parallelism strategies, TP will typically be
kept for high-speed intra-node communications while ZeRO-3 or PP can be used for
parallelism groups spanning lower speed inter-node communications as their
communication patterns require less bandwidth (for PP) or can be more easily
overlapped with computation (for ZeRO-3). The main consideration when combining
these techniques is to organize the GPU efficiently in groups for each
parallelism dimension to maximize throughput and minimize communication
overhead, while being mindful of TP's scaling limitations. For instance, the
groups of GPUs communicating for TP should be kept inside nodes.

**Context Parallelism** and **Expert Parallelism** also help us shard
activations, and can be seen as complimentary to TP. The first one handles long
sequences while the second enables distributed Mixture of Experts training and
they can be combined together without any particular issue.

**Context Parallelism (CP)** specifically targets the challenge of training with
very long sequences by sharding activations along the sequence dimension across
GPUs. While most operations like MLPs and LayerNorm can process these sharded
sequences independently, attention layers require communication since each token
needs access to keys/values from the full sequence. As we saw in  CP section,
this is handled efficiently through ring attention patterns that overlap
computation and communication. CP is particularly valuable when scaling to
extreme sequence lengths (128k+ tokens) where, even when using full activation
recomputation, the memory requirements for attention would be prohibitive on a
single GPU.

![CP diagram](/assets/images/5d_nutshell_cp.svg)

**Expert Parallelism (EP)** specifically targets the challenge of training
Mixture of Experts (MoE) models by sharding specialized "experts" across GPUs
and dynamically routing tokens to relevant experts during computation. The key
communication operation in EP is the `all-to-all` operations routing tokens to
their assigned experts and gathering the results back. While this operation
introduces some communication overhead, it enables scaling model capacity
significantly since each token is only processed during inference (and training)
by a much smaller fraction of the total parameters. In terms of distributed
training/inference, partitioning experts across GPUs becomes relevant when
models scales to a large number of experts.

For instance DeepSeek V3 uses 256 experts.

![EP diagram](/assets/images/5d_nutshell_ep.svg)

📝 Note

This similarity between EP and DP in terms of input handling is why some
implementations consider Expert Parallelism to be a subgroup of Data
Parallelism, with the key difference being that EP uses specialized expert
routing rather than having all GPUs process inputs through identical model
copies.

**Scope and focus** Let's also quickly summarize the sub-part of the model where
some of these different parallelism strategies have the most impact:

  * Tensor Parallelism (and Sequence Parallelism) affects computation throughout the entire model by sharding both weights and activations.
  * Context Parallelism primarily impacts attention layers since that's where cross-sequence communication is required, with other layers operating independently on sharded sequences.
  * Expert Parallelism primarly affects the MoE layers (which replace standard MLP blocks), leaving attention and other components unchanged
  * Pipeline Parallelism and ZeRO are not especially specific to any sub-module or component with the exception that modules and layers need to be balanced in Pipeline Parallelism, the first and last layers are thus often treated differently due to the additional embedding layers.

**Tensor + Sequence Parallel** | **Context Parallel** | **Expert Parallel**  
---|---|---  
shards weights and activations along hidden/seq dim | shards activations along sequence dim | shards specialized expert weights and activations  
communication for matrix multiply operations (column/row linears) | communication for attention key/values | communication for token routing to experts  
model-specific implementation needed | model-agnostic except for attention | model-agnostic except for MoE layers  
Prefers high-bandwidth intra-node communication | Prefers large sequence lengths | Requires MoEs  
  
**Summarizing it all–** Now what about gathering and combining all the
techniques we've seen in a single diagram combining them all. Yes, we're up for
the challenge!

In this summary diagram, you will find illustrated activations and modules for a
single transformers layer –in it's MoE variant–. We also illustrate the various
directions of parallelism and the communication operations we've been discussing
in all the previous sections.

![image.png](/assets/images/5d_full.svg)

We can also represent side-by-side a **full overview** of the memory savings for
each one of these strategies. We'll plot them with different sequence length as
well as with selective (top) and full (bottom) recomputation so you can see how
they all play with activations:

![5Dparallelism_8Bmemoryusage.svg](/assets/images/5Dparallelism_8Bmemoryusage.svg)

Let's finish this section with a high level view at all of these techniques,
their main underlying idea and major bottleneck:

**Method** | **Memory savings applies specifically on** | **Parallel/sharding dimension** | **Disadvantage**  
---|---|---|---  
DP | Activations (reduce local batch size) | Batch | Limited by max batch size  
PP | Model parameters | Model layers | Idle bubble and complex schedules  
TP/SP | Model parameters and activations | Hidden dimension / Sequence length | Requires high bandwidth communication  
CP | Activations | Sequence length | Add communication overhead in attention modules  
EP | Experts parameters | Expert dimension | Requires MoE layers, add routing communication overhead  
ZeRO-1 | Optimizer states | Sharded among DP replicas | Params communication overhead  
ZeRO-2 | Optimizer states and gradients | Sharded among DP replicas | Params communication overhead  
ZeRO-3 | Optimizer states, gradients, and model parameters | Sharded among DP replicas | Params communication overhead  
  
Clearly, none of these techniques is a silver bullet for magical scaling and
we'll often have to combine them in one way or another. Can we actually come up
with a few rules that would help us find a good starting point to choose among
–and combine– them? This will be the topic of our next section.

