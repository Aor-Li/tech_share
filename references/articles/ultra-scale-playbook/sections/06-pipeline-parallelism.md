## Pipeline Parallelism

Your browser does not support the audio element.

To add a podcast feeling to your reading experience, feel free to listen to the
NotebookLM hosts discussing the following sections of this book as you're
reading along.

In the Tensor Parallelism section we saw that trying to scale Tensor parallelism
past the number of GPUs per single node (typically 4 or 8) hit a lower bandwidth
network called “inter-node connection” which can quite strongly impair our
performances. We can see this clearly on e.g. the all-reduce operation when we
benchmark it on our cluster across several nodes (each node has 8 GPUs):

Inter-node communication bandwidth measurements across different node counts,
showing median (lines) and 5th-95th percentile ranges (shaded areas) for
AllReduce, AllGather and ReduceScatter operations.

Sequence and context parallelism can help for long sequences but don’t help much
if the sequence length is not the root cause of our memory issues but rather the
size of the model itself. For large model (70B+), the size of the weights alone
can already push past the limits of the 4-8 GPUs on a single node. We can solve
this issue by summoning the fourth (and last) parallelism dimension: “pipeline
parallelism”.

Pipeline parallelism is a simple but powerful technique - we split our model's
layers across multiple GPUs! For example, if we have 8 GPUs, we could put layers
1-4 on GPU 1, layers 5-8 on GPU 2, and so on. This way, each GPU only needs to
store and process a portion of the model's layers, significantly reducing the
memory requirements per GPU. Let's see the effect of Pipeline Parallelism in
action on the memory usage for a 8B model:

This technique may remind you of our discussion on ZeRO-3 where we split the
model parameters across GPUs. We compare both techniques in details later in the
5D parallelism in a nutshell section.

Looking at the figure above, we notice something interesting: while the model
parameters are nicely split across GPUs, the activation memory remains the same
on each GPU! This is because each GPU still needs to process the full batch of
data, just with different layers. The activations from one GPU's layers will be
sent to the next GPU to continue the forward pass.

This introduces a new type of communication pattern: instead of communicating
parameters like we did with ZeRO-3 in data parallelism, we're now passing
activation tensors sequentially between GPUs in a "pipeline". While conceptually
simple, efficiently implementing this technique is quite tricky. Let's dive
right into the details!

### Splitting layers on various nodes - All forward, all backward

So, let’s say we simply spread the layers on several devices, e.g. a first GPU
will take the first few layers and a second GPU will take the second part of the
models and so on. The forward pass through our model now simply involves
sequentially passing the batch of data along the model and thus successively
using each compute device.

We have a direct first advantage: the required interconnect bandwidth stays
quite low as we only send moderate-sized activations at a handful of location
along the model depth. It can make a huge difference versus e.g. communications
in Tensor Parallelism, which happens several times within each layer.

But maybe you start feeling a glimpse of the troubles to come:
**“sequentially”** and **“successively”**?!? This doesn’t sound very efficient
in the world of parallel computations, especially after our discussion on
computation and communication overlap.

Indeed reader! The main challenge in pipeline parallelism will be how to
efficiently circumvent the sequential nature of PP to keep our GPU busy at all
times and avoid having one GPU computing while the others are waiting. Here is
how our GPU utilization is looking when doing a naive and simple forward and
backward pass through the model (here the numbers indicate the model layers):

![image.png](/assets/images/pp_afab.svg)

An example of Pipeline parallelism for a model with 16 layers distributed across
4 GPUs. The numbers correspond to the layer IDs.

The remaining idle time is indicated in grey and usually called the “bubble” and
the sight of this probably break your heart after we spent so much time
optimizing throughput.

We can quantify how efficient a pipeline setup is by looking at how much time we
loose because of the bubble. Let’s say t_f and t_b are the times for the forward
and backward pass, respectively, as measured for one microbatch and one stage of
the pipeline (a simple assumption is often to have t_b \approx 2 \times t_f
which you can see on the above graph). If we could perfectly parallelize the
ideal total time would be t_{id}=t_f + t_b. However, we can count on the graph
that due to the pipeline bubble there is additional time of
t_{pb}=(p-1)*(t_f+t_b) (where p is the degree of pipeline parallelism, i.e the
number of GPU on the above graph) ie. the time each GPU is waiting while other
GPUs are computing.

We can compute the ratio of the additional bubble time over the ideal time:

r_{bubble} = \frac{(p-1)*(t_f+t_b)}{t_f+t_b} = p-1

As we add more stages the bubble time thus increases and the utilization drops.
As we can see, the bubble can be very large in a naive implementation!

Thankfully, various pipeline parallelism schemes have been designed to **reduce
the size of the bubble**.

Let’s take a first tool out of our toolbox and think about splitting our batch
into smaller bit-sized portions which can be processed in parallel or almost,
like we did before in data parallel for instance. Now when the second GPU is
busy processing micro-batch 1, the first GPU can already start processing micro-
batch 2. Here is a schedule using 8 micro-batches:

![pp_afab2.svg](/assets/images/pp_afab2.svg)

Before the numbers in the diagram indicated the layers but in all pipeline
parallel plots from now including this one it indicates a microbatch. You can
think of each square here to contain several layers as seen in the previous
figure.

The above schedule is called the **_all-forward-all-backward (AFAB)_** schedule
as we first do all forward passes and then only all-backward passes. The
advantage is that forward and backward steps are still generally sequential and
so we're preserving the general organization of our model training code. It
makes this PP implementation one of the simplest to implement.

You can find the full implementation of the AFAB pipeline in picotron:

👉 AFAB PP implementation in Picotron (Click to expand)

Let’s estimate the bubble in this example. The difference with our first example
is that the ideal time to process m microbatches is now t_{id} = m*(t_f+t_b):

r_{bubble} = \frac{(p-1)*(t_f+t_b)}{m*(t_f+t_b)} = \frac{p-1}{m}

As we can see, we can fight some inefficiencies of pipeline stages by adding
more microbatches, reducing the size of the bubble by a factor of m.

However, as annoying as the bubble is the memory storage required for storing
all activation. We need to keep all of the activations in memory until we reach
the backward stage which lead to a quick memory explosion in these
implementations of PP. Can we do better and avoid this memory explosion?

Since the memory explosion is triggered by the activation we store for the
backward pass, let’s try to see if we can start performing the backward pass
while we are still performing other forward part of the computation. This will
allow us to drop some of the activations we need for the backward pass as soon
as possible.

### One-forward-one-backward and LLama 3.1 schemes

This schedule is called **_one-forward-one-backward (1F1B)_** as the
middle/steady state involves alternatively performing one forward and one
backward pass. The general idea is to start performing the backward pass as soon
as possible. The schedule looks like this:

![image.png](/assets/images/pp_1f1b.svg)

If you count carefully you'll see that the bubble still has the same size so our
training efficiency is not significantly improved. However we only need to store
activations for p micro-batches (where p is the degree of pipeline parallelism)
instead of m (where m was the number of microbatches) which can reduce the
activation memory explosion we had in the AFAB schedule. As a consequence we can
add more microbatches which then will actually reduce the bubble.

A major complexity of this setup, visible on the above graph is how forward and
backward passes are not anymore cleanly sequential but performed in parallel
across devices and interleaved. This means we will have to schedule a switch
from forward to backward passes independently on each device instead of in a
simple and common central training loop as usual.

This is one of the reason implementing Pipeline Parallelism usually requires
rather extensive modifications to training code as well as modeling code.

You can find a full implementation of 1F1B in picotron as well:

👉 1F1B PP implementation in Picotron (Click to expand)

Let's take a look at how the 1F1B Pipeline Parallelism schedule scales in
practice with some benchmarks on our cluster:

![Throughput scaling of Pipeline Parallelism with varying microbatch sizes](/assets/images/pp_1f1b_scaling.png)

On the left, with a number of microbatches equal to –or less than– PP degree
minus one (m = p - 1), we see how detrimental the pipeline bubble can be -
performance are low and even drops as we scale PP. The right plot shows that
using many more microbatches than PP degree (m = 32 \gg p - 1) helps improve
low-PP-degree performances while still staying limited at very large PP degree.
In practice it's not possible to arbitrarly increase the number of microbatches
to maintain the ratio of m \gg p - 1 since we're ultimately constrained by the
target global batch size. With a maximal possible number of microbatches as we
add more PP degree, we'll ultimately have to increase the bubble size according
to r_{bubble} = \frac{p - 1}{m}.

Interestingly, at small number of micro-batches the performance only drops by
14% when scaling from one node (p = 8) to two nodes (p = 16) - a much better
scaling than Tensor Parallelism which typically sees around 43% performance
degradation in similar cross-node scenarios. This type of behavior when hitting
the lower-bandwith inter-node network makes Pipeline Parallelism particularly
attractive for distributed training across multiple nodes.

While 1F1B significantly reduces our activation memory footprint, we see on this
last graph that the pipeline bubble remains a major efficiency bottleneck. With
the bubble size still proportional to the number of pipeline stages, we're
leaving valuable GPU compute idle. Can we design an even smarter schedule to
minimize this wasted computation time?

### Interleaving stages

The 1F1B schedule has let us improved memory usage but not much the size of the
idle buddle. Any way we could still push this frontier?

Well it turns out this is possible if we are willing to bring in a few
additional communication operations. Time to talk about **_interleaved
stages_**.

Up to now we’ve sliced our model naively along the model depth dimensions,
hosting for instance layers 1-4 on the first GPU and layers 5-8 on the second
GPU. But there are other ways we could think about slicing our layers, e.g.
having odd layers 1, 3, 5, 7 on the first GPU and even layers 2, 4, 6, 8 on the
second GPU.

This can be seen in general as a kind of “looping pipeline” where a micro-batch
will move in circles from one GPU to the next as it goes through the forward
pass through the model. Let's take a graphical look at how this works:

![pp_1f1b_interleaved.svg](/assets/images/pp_1f1b_interleaved.svg)

An example of interleaved pipeline parallelism for a model with layers
distributed across 4 GPUs. Numbers still correspond to the microbatches IDs but
for clarity we've colored differently the first and the last layers of the model
to illustrate how layers are spread accross GPUs.

As a consequence we see additional communications happening as the model goes
several times through each GPU for the same computation that previously just
took one pass. However, each forward and backward pass is divided by a factor of
v, where v is the number of stages or model chunks per GPUs as we are able to
better interleave forward and backward passes.

\begin{aligned} &t;_{pb} = \frac{(p-1)*(t_f+t_b)}{v} \\\ &r;_{bubble} =
\frac{1}{v}\frac{(p-1)*(t_f+t_b)}{m*(t_f+t_b)} = \frac{p-1}{v*m} \end{aligned}

So we can now decrease the bubble by adding microbatches and interleaved stages,
but note that quantitatively, the amount of communication also increases by v so
it’s a trade off. In the following plot you can see several configurations for a
PP setup with p=8, where the special case of m=1, v=1 corresponds to naive
pipeline parallelism and the configurations with v=1 are AFAB or 1F1B setups and
v \neq 1 are interleaved configurations.

Scheduling also becomes more complex here as we have to decide on a given GPU
and at a given moment whether we are prioritizing earlier micro-batches going
through later layers –meaning that we close the forward and backward loops as
fast as possible (so called “depth-first”, i.e. prioritizing getting batches out
of the model as fast as possible)– or if we prioritize to first have later
micro-batches going through earlier layers (so called “breadth-first” i.e.
prioritizing filling in the pipeline as much as possible). This choice is
explained in detail in the nice "Breadth-Fist Pipeline" paper.

You now have all the elements to understand the pipeline parallelism approach in
Llama 3.1 which is using a one-forward-one-backward setup with interleaved
stages and a priority setting tuneable between depth-first and breadth-first.

![pp_llama3.1_schedule.png](/assets/images/pp_llama3.1_schedule.png)

However, we haven’t reached the end of possible pipeline schedules and recently
some methods have been proposed to **reduce the bubble to virtually zero**!
These techniques were for instance used in the DeepSeek V3/R1 implementation.
Peaked your curiosity? Let’s have a final quick look at these magical schedules
before we leave the world of Pipeline Parallelism!

### Zero Bubble and DualPipe

Even more sophisticated ways to reduce the bubble have recently been proposed
which reached close to a “zero bubble” regime. The secret here is to split at an
even finer-grained level the operations involved in order to interleave them in
the most efficient way. For instance the pipeline implementation approach in
DeepSeek V3/R1, called DualPipe, reaches close to a zero bubble regime.

Ultimate "flex" in DeepSeek V3 technical report where the authors indicate that
their setup "achiev[ed] a near-zero all-to-all communication overhead".

Let’s briefly see how this can work by summarizing the ZeroBubble work which is
a precursor to DualPipe. The base observation of ZeroBubble is that the backward
pass through a matrix multiplication actually involves two separated operations:
backward operation for the inputs (B) and the backward operation for the weights
(W):

While the output of B, the backward pass for the input, is necessary for
performing the backward pass of the lower layers, the backward pass of the
weights, W, is not necessary for the rest of the backward pass and generally
only needs to be performed before the optimiser step. We can see that in the
following diagram:

![image.png](/assets/images/pp_zerobubble_compgraph.png)

This means W can be flexibly scheduled anywhere after the corresponding B of the
same stage. This allows for strategic placement of W to fill the pipeline
bubbles. The ZB-H2 schedule on the top right is an example of (theoretical)
schedule with zero bubble taking advantage for this fine-grained decomposition.

![image.png](/assets/images/pp_zerobubble_ppschedule.png)

On the top (Figure 2 from the ZeroBubble paper): the classical 1F1B schedule,
interleaving forward and backward pass but keeping a coarse-grained backward
pass. On the bottom two graphs (Figure 3 from the ZeroBubble paper), two
variantes of the ZeroBubble schedule, splitting the backward operation in a "B"
and a "W" finer-grained operations. The last schedule, so-called "ZB-H2" is an
example of (theoretical) schedule with zero bubble taking advantage for this
fine-grained decomposition.

DeepSeek’s DualPipe introduced with its V3 technical report  an extension of
this decomposed approach to the additional case of two streams propagating from
both ends of the PP dimension, these streams being interleaved to minimize even
further idle time in the GPUs. This schedule is displayed in the following
scheduling graph and is even more complex than the previous ones:

![image.png](/assets/images/pp_zerobubble_dualpipe.png)

In general, fully optimizing such complex schedules involve carfully measuring
the duration of the various fine-grained operations and solving a ILP to
minimize the final bubble time. See for instance in the ZeroBubble paper for a
discussion of the heuristics and algorithms to perform such a scheduling. As a
result, the ZeroBubble and DualPipe schedules are too complex for us to give
here code snippets but you should start to have a general idea of the concepts
involved.

This concludes our tour into the world of pipeline schedules and bubbles. We
hope you enjoyed this guided tour!

It's now time to turn to the last parallelism method we'll detail and which we
can use to train large models efficiently: **Expert parallelism**.

