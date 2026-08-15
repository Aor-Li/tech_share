# The Ultra-Scale Playbook:  
Training LLMs on GPU Clusters

{{{fragment-banner}}}

We ran over 4000 scaling experiments on up to 512 GPUs and measured throughput
(size of markers) and GPU utilization (color of markers). Note that both are
normalized per model size in this visualization.

Thousands of GPUs humming in perfect harmony. That's what it takes to train today's most powerful AI models – a symphony of computing power that until recently was the exclusive domain of elite research labs. Open source has transformed this landscape, but not completely. Yes, you can download the latest [Llama](https://huggingface.co/meta-llama) or [DeepSeek](https://huggingface.co/deepseek-ai) models. Yes, you can read their [technical](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) and [experiment](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) reports. But the most challenging part – the training code, the knowledge and technics necessary to coordinate GPUs to train these massive systems – remains shrouded in complexity and spread around a series of disconnected papers and often private codebases. 

Reading time: 2-4 days.  
For the best reading experience, we recommend not using a mobile phone.

This open-source book is here to changes that. Starting from the basics, we'll
walk you through the knowledge necessary to scale the training of large language
models from one GPU to tens, hundreds and even thousands of GPUs, illustrating
theory with practical code examples and reproducible benchmarks.

As the size of the clusters used to train these models grew, various techniques
such as data parallelism, tensor parallelism, pipeline parallelism or context
parallelism as well as ZeRO or kernel fusion have been invented to makes sure
that GPUs are highly utilized at all times. This significantly reduces training
time and makes the best use of this expensive hardware. Even more, as the
challenge of scaling up AI training goes beyond just building the initial models
and teams have found that fine-tuning large models on specialized data often
produces the best results, generally involving the same distributed training
techniques. In this book we'll progressively go over all of these techniques
–from the simplest to the most raffined one– while keeping a single story-line
to understand where each method comes from.

If you have questions or remarks open a discussion on the [Community tab](https://huggingface.co/spaces/nanotron/ultrascale-playbook/discussions?status=open&type=discussion)!

We'll assumes you have some simple basic knowledge about current LLM architecture and are roughtly familiar with how deep learning model are trained, but you can be generally new to distributed training. If needed, the basics of model training can be found in great courses found at [DeepLearning.ai](https://www.deeplearning.ai) or on the [PyTorch tutorial sections](https://pytorch.org/tutorials/beginner/basics/intro.html). This book can be seen as the second part of a trilogy following our first blog on processing data for pre-training, the so-called “[FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)”. Having read both blog posts, you should have almost all the core knowledge needed to fully understand how how performing LLMs are being built nowadays, just missing some final spices regarding data mixing and architecture choices to complete the recipe (stay tuned for part three…).

We are extremely thankful to the whole [distill.pub](https://distill.pub/) team for creating the template on which we based this blog post.

The book is built on the following **three general foundations** :

**Quick intros on theory and concepts:** before diving into code and
experiments, we want to understand how each method works at a high level and
what it’s advantages and limits are. You’ll learn about which parts of a
language model eat away your memory and when during training it happens. You’ll
learn how we can solve memory constraints by parallelizing the models and
increase the throughput by scaling up GPUs. As a result you'll understand how
the following widget to compute the memory breakdown of a transformer model
works:

Note that we're still missing Pipeline Parallelism in this widget. To be added
as an exercise for the reader.

Memory usage breakdown

Attention Heads (a):

Mixed Precision:

Micro Batch Size (b):

Sequence Parallelism:

Hidden Dimension (h):

Recomputation: None Selective Full

Feedforward Dimension (h_ff):

Zero: 0 1 2 3

Number of Layers (L):

FF Activation: ReLU GELU SwiGLU

Sequence Length (s):

Vocabulary Size (v):

Tensor Parallelism (t):

Optimizer Parameters (k):

Data Parallelism (d):

Presets: Llama 3 Tiny Llama 3 8B Llama 3 70B Llama 3 405B

(Don't worry if you have no idea what's happening in this widget. That's why
we're here.)

While this widget gives a theoretical breakdown we also made the [following tool](https://huggingface.co/spaces/nanotron/predict_memory) that can be used to predict the memory usage during a training run:

[ ![Predict Memory Tool](/assets/images/predict_memory_tool.png) ](https://huggingface.co/spaces/nanotron/predict_memory)

**Clear code implementations:** theory is one thing, but we discover all kinds
of edge cases and important details when we implement something. That’s why we
link to implementation references where possible. Depending on the case, we’ll
use two code references:

  * the [picotron](https://github.com/huggingface/picotron) repository is built for education, thus it implements concepts usually in single, self-contained short files. 

  * On the other hand, to look at production ready code, we’ll refer to the [nanotron](https://github.com/huggingface/nanotron) implementations which is a production training codebase used at Hugging Face. 

If you want to watch a video on distributed training rather than reading the blog or picotron code checkout [Ferdinand's YouTube channel](https://www.youtube.com/watch?v=u2VSwDDpaBM&list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S).

**Real training efficiency benchmarks:** Finally, how to _actually_ scale your
LLM training depends on your infrastructure, such as the kind of chips,
interconnect etc., and we can’t give a single unified recipe. What we will give
though is a way to benchmark several setups and it is what we have done on our
cluster! We ran over 4100 distributed experiments (over 16k including test runs)
with up to 512 GPUs to scan many possible distributed training layouts and model
sizes.

As you can see, there’s a lot of ground to be covered. Before getting into the
trenches of distributed training let’s take a quick high level look on the
challenges we'll cover in the book.

