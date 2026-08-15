## Context Parallelism

With Tensor Parallelism and Sequence Parallelism, we can reduce the memory
requirements per GPU significantly as both model weights and activations are
distributed across GPUs. However, when training models on longer and longer
sequences (e.g. when scaling to 128k or more tokens per sequence) we might still
exceed the memory available on a single node as we still have to process a full
sequence length when we're inside the TP region.

Moreover, even if we use full recomputation of the activations (which comes at a
heavy compute overhead of ~30%), we still need to hold in memory some
activations at the layer boundaries which scale linearly with sequence length.
Let's take a look and see how Context Parallelism can help us:

The core idea of Context Parrallelism is to apply a similar idea to the Sequence
Parallelism approach (aka to split along the sequence length) but to the modules
where we already apply Tensor Parallelism. We will thus split these modules
along two dimensions, thereby also reducing the effect of sequence length. You
will find this approach quite intuitive after all we’ve already convered but...
there is a trick to it so stay awake!

For Context Parallelism; just like Sequence Parallelism, we’ll split the input
along the sequence dimension but we now apply this splitting along the full
model, instead of only the sequence parallel regions of the model as we’ve done
previous with Tensor + Sequence Parallelism.

Splitting the sequence doesn't affect most modules like MLP and LayerNorm, where
each token is processed independently. It also doesn’t require expensive
communication like TP, as only the inputs are split and not the weight matrices.
Just like data parallelism, after computing the gradients, an all-reduce
operation is initiated to synchronize the gradients across the context
parallelism group.

There is one important exception though as we we need to pay particular
attention to the **Attention blocks** (haha.. pun intended :D). In the attention
module each token needs to access key/value pairs from **all** other sequence
tokens or in the case of causal attention at least attends to each previous
token.

Because Context Parallelism splits the inputs along the sequence dimension
across GPUs, the attention module will require full communication between GPUs
to exchange the necessary key/value data.

That sounds very expensive if we do it naively. Is there a way to do this rather
efficiently and fast! Thankfully there is: a core technique to handle this
communication of key/value pairs efficiently is called _Ring Attention_.

📝 Note

Context Parallelism shares some conceptual similarities with Flash Attention
(see later for more details) - both techniques rely on online softmax
computation to reduce memory usage. While Flash Attention focuses on optimizing
the attention computation itself on a single GPU, Context Parallelism achieves
memory reduction by distributing the sequence across multiple GPUs.

### Discovering Ring Attention

In this implementation of the attention mechanism, each GPU first initiates an
asynchronous communication operation to send its key/value pairs to other GPUs.
While waiting for the other GPUs data, it computes the attention score for the
portion of the data it already has in memory. Ideally, a next key/value pair is
received from another GPU before this computation finishes, allowing the GPU to
start the next round of computation immediately after it finishes its first
computation.

Let's illustrate this. We'll suppose we have 4 GPUs and an input of 4 tokens.
Initially, the input sequence is split evenly along the sequence dimension, so
each GPU will have just one token along with its corresponding Q/K/V values.
Leyt's say Q1, K1, and V1 represent the query, key, and value of the first
token, which are located on the 1st GPU. The attention calculation will take 4
time steps to complete. At each time step, each GPU performs these three
successive operations:

  1. Send “current keys and values” to the next machine except during the last time step in a non-blocking manner so we can starts the following step before this step is finished
  2. Locally compute the attention score on the “current keys and values” it already has, which typically involves performing Softmax(\frac{QK^T}{\sqrt{d}}) * V.
  3. Wait to receive keys and values from the previous GPU and then circle back to step 1. where “current keys and values” are now the key/values just received from the previous GPU.

We perform these 3 steps four times to complete the attention calculation.

The whole process with 4 GPUs is shown in the following animation:

![ring-attention.gif](/assets/images/ring-attention.gif)

It's probably obvious to you on this animation why the authors chose to call
this approach Ring Attention.

There is one big problem though which is that a naive implementation of Ring
Attention lead to some strong imbalance between GPU coming from the shape of the
causal attention matrix. Let’s take a look at the SoftMax computation by
considering the attention score matrix with the causal attention mask:

![cp_attnmask.svg](/assets/images/cp_attnmask.svg)

The SoftMax is computed row-wise, which means whenever a GPU has received all
the tokens of a row it can be computed. We see that GPU1 can immediately compute
it as it starts with tokens 1-4 and GPU1 actually doesn’t need to receive any
information from any other GPUs. However, GPU2 will need to wait for the second
round to also receive 1-4 and thus have all values for tokens 1-8. Also, GPU1
seems to perform much less work than all the other GPUs.

Let’s see if we can balance our computations better:

### Zig-Zag Ring Attention – A Balanced Compute Implementation

We need a better way to distribute the input sequences. This can be achieved by
assigning the tokens not purely sequential to the GPUs but by mixing the
ordering a bit such that we have a good mix of early and late tokens on each
GPU. This approach is called Zig-Zag attention and in this new arrangement, the
attention mask will show an even distribution of computation but if you count
the number of colored squares, you’ll see that the computation is now balanced
across all GPUs.

![cp_zigzagmask.svg](/assets/images/cp_zigzagmask.svg)

At the same time we’ll also see that in order to complete all rows, each GPU
will need information from all the other GPUs.

We have two general ways to overlap computation and communication, either by
performing a general all-gather, regrouping all the KV on each GPUs at the same
time (in a Zero-3 type of way) or we gather them one-by-one from each GPU to
each GPU as needed:

![cp_overlap_allgather.svg](/assets/images/cp_overlap_allgather.svg)

![cp_overlap_all2all.svg](/assets/images/cp_overlap_all2all.svg)

The key difference between these two implementations lies in their communication
patterns and memory usage:

**1\. AllGather Implementation:**

  * All GPUs simultaneously gather the complete key/value pairs from all other GPUs
  * Requires more temporary memory as each GPU needs to store the full KV pairs at once
  * Communication happens in one step but with larger memory overhead

**2\. All-to-All (Ring) Implementation:**

  * GPUs exchange KV pairs in a ring-like pattern, one chunk at a time
  * More memory efficient as each GPU only needs to store one additional chunk temporarily
  * Communication is spread out and overlapped with computation, though with some additional base latency overhead from multiple communication steps

The All-to-All approach generally offers better memory efficiency at the cost of
slightly more complex communication patterns, while the AllGather approach is
simpler but requires more temporary memory during the attention computation.

We've now seen how we can split a model across one node with TP to tame large
models and that we can use CP to tame the activation explosion with long
sequences.

However, we still know that TP doesn't scale well across nodes, so what can we
do if the model weights don't easily fit on 1 node? Here come another degree of
parallelism, our forth one, called **Pipeline Parallelism** , to the rescue!

