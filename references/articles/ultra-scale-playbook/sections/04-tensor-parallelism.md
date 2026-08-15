## Tensor Parallelism

Your browser does not support the audio element.

To add a podcast feeling to your reading experience, feel free to listen to the
NotebookLM hosts discussing the following sections of this book as you're
reading along.

So we have sharded the model’s parameters, gradients and optimizers states with
ZeRO but we hit a limit once activation memory overtakes our memory budget.
Welcome Tensor Parallelism (TP), a method which shards weights, gradients, and
optimizers states as well as activations and without the need to gather them all
prior to the computation. Seems like a dream! Let’s first have a look at how
Tensor Parallel works with simple matrix multiplications.

Tensor Parallelism leverages the mathematical properties of matrix
multiplication A \times B. To understand how it works, let's examine two
fundamental equations that make this parallelization possible:

\begin{aligned} &\text{1.} \quad A\cdot B = A \cdot \begin{bmatrix} B_1 & B_2 &
\cdots \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots \end{bmatrix} \\\
&\text{2.} \quad A\cdot B =\begin{bmatrix} A_1 & A_2 & \cdots \end{bmatrix}
\begin{bmatrix} B_1 \\\ B_2 \\\ \vdots \end{bmatrix} = \sum_{i=1}^n A_i B_i
\end{aligned}

This means that we can compute matrix product by either 1) multiplying each
column of B individually or 2) multiplying each row individually and combining
the results. In a neural network, the matrix multiplication is more often
represented in the following format: X \times W, where:

  * X represents the input or activation values
  * W represents the weight of the `nn.Linear`

In practice a small example of the operation looks like this:

![TP diagram](/assets/images/tp_diagram.svg)

Let’s see how we can parallelise this operation! In tensor parallelism, tensors
will be split into N shards along a particular dimension and distributed across
N GPUs. Matrices can be split either on the column part or row part leading to
row and column parallelism. One thing we’ll see in the following is that
choosing row or column sharding will require different communications
primitives.

Our first option is to use column-wise sharding (also called **_column-
linear_**): We'll copy the complete input matrices to each worker, requiring an
operation called **_broadcast_** , and split the weight matrix into columns. The
inputs are then multiplied with the partial weight matrices, and the results are
finally combined using an **_all-gather_** operation.

![image.png](/assets/images/tp_diagram2.png)

Here's the code implementation of column wise tensor parallelism:

👉 Column parallel TP implementation in Picotron (Click to expand)

The second option is called row-wise sharding (also called **_row-linear_**): As
the attentive reader might guess, row-linear means that we split the weight
matrix into chunks of rows. However, this also requires us to split the inputs,
which needs a **_scatter_** operation rather than a broadcast as used in column-
linear sharding. The results on each worker are already in the right shape but
need to be summed for the final result, thus requiring an all-reduce operation
in this scenario.

We see here our fourth distributed primitive: **_scatter_**!

![image.png](/assets/images/tp_diagram3.png)

Here's the implementation for row-wise tensor parallelism:

👉 Row parallel TP implementation in Picotron (Click to expand)

Now that we have the basic building blocks of TP, let's have a look at how we
can effectively combine them inside a transformer layer!

### Tensor Parallelism in a Transformer Block

To come up with a strategy to follow, let’s move from a toy example to a real
model building block. A Transformer model is made of two main building blocks :
Feedforward layers (MLP) and Multi-Head Attention (MHA). We can apply tensor
parallelism to both.

The Feedforward part can be parallelized by having a “Column linear” followed by
a “Row Linear” which amounts to a broadcast to copy the input and an all-reduce
in forward. Note that the broadcast isn’t needed in actual training where we can
make sure inputs are already synced across TP ranks. This setup is more
efficient than starting with "Row Linear" followed by "Column Linear" as we can
skip the intermediate all-reduce between both splitted operations.

![image.png](/assets/images/tp_diagram4.png)

Now that we’ve found an efficient schema for the Feedforward part of the
transformer, let’s take a look at the multi-head attention block (MHA).

We can generally follow a similar approach where Q, K, and V matrices are split in a column-parallel fashion, and the output projection is split along the row dimension. With multi-head attention, the column-parallel approach has a very natural interpretation: each worker computes the attention for an individual or a subset of heads. The same approach works as well for [**_multi-query_** (MQA)](https://arxiv.org/abs/1911.02150) or [**_grouped query attention_** (GQA)](https://arxiv.org/abs/2305.13245) where key and values are shared between queries. 

It's worth noting however that the tensor parallelism degree should not exceed
the number of Q/K/V heads because we need intact heads per TP rank (otherwise we
cannot compute the attentions independently on each GPU and we'll need
additional communication operations). In case we’re using GQA, the TP degree
should actually be smaller than the number of K/V heads. For instance, LLaMA-3
8B has 8 Key/Value heads, so the tensor parallelism degree should advantageously
not exceed 8. If we use TP=16 for this model, we will need to duplicate the K/V
heads on each GPU and make sure they stay in sync.

![image.png](/assets/images/tp_full_diagram.png)

Finally note that Tensor Parallelsim is still not a silver bullet for training.
We’ve added several distributed communication primitive directly in the
computation path of our model which are therefore hard to fully hide/overlap
with computation (like we did in ZeRO), our final performances will be the
results of a tradeoff between the computation and memory gains and the added
communication overhead. Let's illustrate this:

![Forward pass in Tensor Parallelism](/assets/images/tp_overlap.svg)

It's possible to partially hide this communication by performing block matrix
multiplication coupled with async communication/computation.

Looking at the timeline of operations in tensor-parallel MLP (same applies for
Attention), we can better understand the tradeoffs involved. In the forward of
each decoder layer, we hit a synchronization point with the AllReduce operation
that cannot be overlapped with computation. This _exposed communication_
overhead is necessary to combine partial results across tensor-parallel ranks
before the final LayerNorm can be applied.

For example, Megatron-LM/Nanotron implement a partial overlapping of all-gather
with FC1 computation where a portion of the matrix multiplication result will
start to be sent to the other GPU while the other part is still being computed.

Tensor parallelism does help reduce activation memory for the matrix
multiplications since the intermediate activations are sharded across GPUs.
However, we still need to gather the full activations for operations like
LayerNorm, which means we're not getting the full memory benefits we could.
Additionally, TP introduces significant communication requirements that heavily
depend on the network infrastructure. The inability to fully hide this
particular AllReduce behind computation means it directly adds to the critical
path of forward propagation.

This area of research is still an active area of research, with recent work like
Domino  exploring novel techniques to maximize this overlap.

Let's take a better look at the trade-off as we scale the TP degree:

While increasing TP leads to reduced per-GPU throughput (left), it enables
processing of larger batch sizes (right), illustrating the trade-off between
computational efficiency and memory availability in distributed training.

In practice and as we see above on the left plot, the communication overhead of
tensor parallelism becomes particularly noticeable as we scale beyond 8 GPUs.
While tensor parallelism within a single node can leverage fast NVLink
interconnects, going across nodes requires slower network connections. We
observe significant drops when moving from TP=8 to TP=16, and an even steeper
decline from TP=16 to TP=32. At higher degrees of parallelism, the communication
overhead becomes so high that it quickly dominates the computation time.

This being said, tensor parallelism provides important benefits for memory usage
by distributing model parameters, gradients, optimizer states and activations
(to some extent) across GPUs. Let's examine this effect on a 70B parameter
model:

Increasing tensor parallelism reduces the memory needed for model parameters,
gradients and optimizer states on each GPU to the point where we can start
fitting a large model on a single node of 8 GPUs.

Is there a way to get even more benefits from this technique? We've seen that
layer normalization and dropout still require gathering the full activations on
each GPU, partially negating the memory savings. We can do better by finding
ways to parallelize these remaining operations as well.

📝 Note

One interesting note about layer normalization in tensor parallel training -
since each TP rank sees the same activations after the all-gather, the layer
norm weights don't actually need an all-reduce to sync their gradients after the
backward pass. They naturally stay in sync across ranks. However, for dropout
operations, we must make sure to sync the random seed across TP ranks to
maintain deterministic behavior.

Let's explore next a small and natural extension to tensor parallelism, called
**Sequence Parallelism** which does exactly that.

### Sequence Parallelism

**Sequence parallelism (SP)** involves splitting the activations and
computations for the parts of the model not handled by tensor parallelism (TP)
such as Dropout and LayerNorm, but along the input sequence dimension rather
than across hidden dimension.

📝 Note

The term Sequence Parallelism is a bit overloaded: the Sequence Parallelism in
this section is tightly coupled to Tensor Parallelism and applies to dropout and
layer norm operation. However, when we will move to longer sequences the
attention computation will become a bottleneck, which calls for techniques such
as Ring-Attention, which are sometimes also called _Sequence Parallelism_ but
we’ll refer to them as _Context Parallelism_ to differentiate the two
approaches. So each time you see sequence parallelism, remember that it is used
together with tensor parallelism (in contrast to context parallelism, which can
be used independently).

This is needed because these operations require access to the full hidden
dimension to compute correctly. For example, LayerNorm needs the full hidden
dimension to compute mean and variance:

\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} +
\beta

where \mu = \text{mean}(x) and \sigma^2 = \text{var}(x) are computed across
hidden dimension h.

So even though these operations are computationally cheap, they still require
significant activation memory since they need the complete hidden dimension. SP
allows us to shard this **memory** burden across GPUs by splitting along the
sequence dimension instead.

In practice we’ll go from the left diagram to the right:

![ in forward: f = no-op ; f* = all-reduce ; g = all-gather ; g* = reduce-
scatter

            in backward: f = all-reduce ; f* = no-op ; g = reduce-scatter ; g* = all-gather
           SP region needs full hidden_dim](/assets/images/tp_sp_diagram.png)

The diagram shows how we transition between tensor-parallel and sequence-
parallel regions using different collective operations (labeled "f" and "g").
The key challenge is managing these transitions efficiently while keeping memory
usage low and maintaining correctness.

In the forward pass:

  * "f" is a no-op (no operation) because activations are already duplicated across ranks
  * "f*" is an all-reduce to synchronize activations and ensure correctness

In the backward pass:

  * "f*" is a no-op because gradients are already duplicated across ranks
  * "f" is an all-reduce to synchronize gradients

These operations "f" and "f*" are called **conjugate** pairs because they
complement each other - when one is a no-op in forward, the other is an all-
reduce in backward, and vice versa.

For sequence parallelism (SP), we use different operations labeled "g" and "g*".
Specifically, we avoid using all-reduce in the SP region since that would
require gathering the full activations and increase our peak memory usage,
defeating the purpose of SP.

So what is actually happening here? As a famous LLM would say, let’s take it
step-by-step:

**Initial LayerNorm (SP Region)**

  * Input tensors X1 _and X2_ (b,s/2,h) enter LayerNorm, already split across sequence dimension
  * Each GPU computes LayerNorm independently on its sequence chunk and give Y1 _and Y2_

**First Transition (SP → TP)**

  * "g" operation (all-gather) combines Y1 _and Y2_ back to full sequence length
  * Restores Y (b,s,h) since column linear needs full hidden dimension h

**First Linear (TP Region)**

  * A1 is a column-linear, so it splits Y along the hidden dimension
  * GeLU is applied independently on each GPU
  * Z1* is (b,s,h/2)

**Second Linear (TP Region)**

  * B1 is a row-linear, so it restores the hidden dimension
  * W1 is (b,s,h)

**Final Transition (TP → SP)**

  * "g*" operation (reduce-scatter) which reduces for previous row-linear correctness while scattering along sequence dimension
  * W1* is (b,s/2,h)

![image.png](/assets/images/tp_sp_diagram_zoomed.png)

A key advantage of sequence parallelism is that it reduces the maximum
activation size we need to store. In tensor parallelism alone, we had to store
activations of shape (b,s,h) at various points. However, with sequence
parallelism, the maximum activation size is reduced to \frac{b \cdot s \cdot
h}{tp} since we always either split along the sequence or hidden dimensions.

It’s a bit difficult to keep track of all the parts that are sharded differently
in TP and TP/SP - believe us, we find it hard to map as well so we made this
small table to summarize how the activations (aka `hidden_states` ) shape change
across hidden dimension h and sequence dimension s during a forward pass:

Region | TP only | TP with SP  
---|---|---  
Enter TP (Column Linear) | h: sharded (weight_out is sharded)  
s: full | h: sharded (weight_out is sharded)  
s: **all-gather** to full  
TP Region | h: sharded  
s: full | h: sharded  
s: full  
Exit TP (Row Linear) | h: full (weight_out is full + **all-reduce** for correctness)  
s: full | h: full (weight_out is full + **reduce-scatter** for correctness)  
s: **reduce-scatter** to sharded  
SP Region | h: full  
s: full | h: full  
s: sharded  
  
And for the embedding layer:

Region | Vanilla TP | TP with SP  
---|---|---  
Embedding Layer (Row Linear sharded on vocab) | h: full (weight_out is full + **all-reduce** for correctness)  
s: full | h: full (weight_out is full + **reduce-scatter** for correctness)  
s: **reduce-scatter** to sharded  
  
By using sequence parallelism, we can achieve even greater activation memory
savings, allowing us to push our batch size and sequence length further than
what would be possible with tensor parallelism alone. Let's see what that means
for our previous 70B model example:

As we can see, we've again strongly reduced the maximum memory usage per GPU,
allowing us to fit sequence lengths of 16k tokens with TP/SP=16, an improvement
over the vanilla TP case! (TP=16 is still a bit large as we've seen in the
previous section, but we'll see how we can improve this in the next section).

One question you may be asking yourself is whether using TP+SP incurs more
communication than vanilla TP? Well, yes and no. In the forward pass of a
vanilla TP we had two all-reduce per transformer block, and in SP we have two
all-gather and two reduce-scatter per transformer block. So SP does twice the
number of communication operations as TP. But since an all-reduce operation can
be broken down into to an all-gather + reduce-scatter (see the A quick focus on
Ring AllReduce section in the appendix) they’re actually equivalent in terms of
communication. Same reasoning for backward as we just use the conjugate of each
operation (no-op ↔ allreduce and allgather ↔ reducescatter).

If you’ve been paying close attention, you’ll notice that we’re talking about 4
comms ops in each layer (2 for Attention and 2 for MLP). This is how the MLP
profiling looks like when using Tensor + Sequence Parallelism:

![tp_sp_overlap.svg](/assets/images/tp_sp_overlap.svg)

Just like vanilla TP, TP+SP can’t easily be overlapped with compute, which makes
throughput heavily dependent on the communication bandwidth. Here again, like
vanilla TO, TP+SP is usually done only within a node (keeping the TP degree
under the number of GPU per nodes, e.g. TP≤8).

We can benchmark how this communication overhead becomes increasingly
problematic as we scale up tensor parallelism. Let’s measure the throughput and
memory utilization as we scale TP with SP for a 3B model with 4096 seqlen:

Here again, there's a trade-off between computational efficiency (left) and
memory capacity (right). While higher parallelism degrees enable processing of
significantly larger batch sizes by reducing the activation memory, they also
reduce per-GPU throughput, in particular above a threshold corresponding to the
number of GPUs per node.

Let’s summarize our observations:

  * for both methods we notice the biggest performance drop when we move from TP=8 to TP=16, because that’s when we move from only communicating within a single node (NVLink), to communicating inter-nodes (EFA)
  * the memory savings in activations when using TP with SP helps us fit far bigger batches than TP alone

**We have seen how TP helps us shard activations across several GPUs by
splitting the attention and feedforward operations along the hidden dimension
and how SP is a natural complement for the remaining operations by splitting
along the sequence dimension.**

📝 Note

Since LayerNorms in the SP region operate on different portions of the sequence,
their gradients will differ across TP ranks. To ensure the weights stay
synchronized, we need to all-reduce their gradients during the backward pass,
similar to how DP ensures weights stay in sync. This is however a small
communication overhead since LayerNorm has relatively few parameters.

However, there are two limits to TP and SP: 1) if we scale the sequence length
the activation memory will still blow up in the TP region and 2) if the model is
too big to fit with TP=8 then we will see a massive slow-down due to the inter-
node connectivity.

We can tackle problem 1) with Context parallelism and problem 2) with Pipeline
parallelism. Let’s first have a look at Context parallelism!

