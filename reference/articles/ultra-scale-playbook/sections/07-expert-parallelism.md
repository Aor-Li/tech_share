## Expert parallelism

This is our last parallelism method to discuss. Before tackling it, if you don't have any exposure to Mixture-of-Experts, feel free to read about them in [this previous, much shorter, blog post](https://huggingface.co/blog/moe) we published some time ago and which should help you better understand the Mixture-of-Experts (MoE) architecture in general.

Mixture-of-expert models have gained recent traction and visibility with models
such as GPT-4, Mixtral or more recently DeepSeek-V3/R1. The basic idea is that
instead of having a single feedforward module per layer we can have several
parallel modules and route tokens through one or the other to be processed
differently.

![ep_moe.png](/assets/images/ep_moe.png)

Illustrationg of a MoE layer taken from the Switch Transformers paper

The design of MoE layers makes it actually easy to implement parallelism across
the experts dimension for what we will call **Expert parallelism** (EP). Since
the feedforward layers are fully independent we can simply put each expert's
feedforward layer on a different worker. Compared to TP it's much more
lightweight, since we don't need to split the matrix multiplication, we just
need to route the hidden states of a token to the right expert.

In practice, EP will typically be used in conjunction with other forms of
parallelism - for instance Data Parallelism. This is because EP only affects the
MoE layers and doesn't shard the input tokens (unlike Context Parallelism which
shards tokens along the sequence length dimension). This means our GPUs would be
doing redundant compute for all the non-MoE blocks if we only used EP. By
combining EP with DP, we can efficiently shard both the experts and the input
batches across our GPUs, as we can see in the simplified diagram below:

![ep_schema.png](/assets/images/ep_schema.png)

Source: A Survey on Mixture of Experts

But let's not get ahead of ourselves - our following section will specifically
talk about all the interactions between different parallelism strategies, so
don't worry if you don't understand yet this last diagram.

In practice, there are a few tricks to make EP work efficiently and they are
closely tied to model design. For instance, DeepSeek-V3 enforces a constraint in
the router, ensuring that each token is sent to at most M nodes (in their case,
4) to keep the tokens on a single node and reduce communication overhead. While
Expert parallelism has been around for a while it is just now gaining new
traction with the MoE architecture gaining more traction.

We plan to add a more complete example of EP in picotron/nanotron soon, so stay
tuned for more!

