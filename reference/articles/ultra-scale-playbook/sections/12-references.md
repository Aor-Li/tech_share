## References

### Landmark LLM Scaling Papers

[**Megatron-LM**](https://arxiv.org/abs/1909.08053)

Introduces tensor parallelism and efficient model parallelism techniques for
training large language models.

[**Megatron-Turing NLG 530B**](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)

Describes the training of a 530B parameter model using a combination of
DeepSpeed and Megatron-LM frameworks.

[**PaLM**](https://arxiv.org/abs/2204.02311)

Introduces Google's Pathways Language Model, demonstrating strong performance
across hundreds of language tasks and reasoning capabilities.

[**Gemini**](https://arxiv.org/abs/2312.11805)

Presents Google's multimodal model architecture capable of processing text,
images, audio, and video inputs.

[**Llama 3**](https://arxiv.org/abs/2407.21783)

The Llama 3 Herd of Models

[**DeepSeek-V3**](https://arxiv.org/abs/2412.19437v1)

DeepSeek's report on architecture and training of the DeepSeek-V3 model.

### Training Frameworks

[**Nanotron**](https://github.com/huggingface/nanotron)

Our framework for training large language models featuring various parallelism
strategies

[**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM)

NVIDIA's framework for training large language models featuring various
parallelism strategies.

[**DeepSpeed**](https://www.deepspeed.ai/)

Microsoft's deep learning optimization library featuring ZeRO optimization
stages and various parallelism strategies.

[**FairScale**](https://github.com/facebookresearch/fairscale/tree/main)

PyTorch extension library for large-scale training, offering various parallelism
and optimization techniques.

[**ColossalAI**](https://colossalai.org/)

Integrated large-scale model training system with various optimization
techniques.

[**torchtitan**](https://github.com/pytorch/torchtitan)

A PyTorch native library for large model training.

[**GPT-NeoX**](https://github.com/EleutherAI/gpt-neox)

EleutherAI's framework for training large language models, used to train GPT-
NeoX-20B.

[**LitGPT**](https://github.com/Lightning-AI/litgpt)

Lightning AI's implementation of state-of-the-art open-source LLMs with focus on
reproducibility.

[**DiLoco**](https://github.com/PrimeIntellect-ai/OpenDiLoCo)

Training language models across compute clusters with DiLoCo.

[**torchgpipe**](https://github.com/kakaobrain/torchgpipe)

A GPipe implementation in PyTorch.

[**OSLO**](https://github.com/EleutherAI/oslo)

OSLO: Open Source for Large-scale Optimization.

### Debugging

[**Speed profiling**](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

Official PyTorch tutorial on using the profiler to analyze model performance and
bottlenecks.

[**Memory profiling**](https://pytorch.org/blog/understanding-gpu-memory-1/)

Comprehensive guide to understanding and optimizing GPU memory usage in PyTorch.

[**Memory profiling walkthrough on a simple example**](https://huggingface.co/blog/train_memory)

Visualize and understand GPU memory in PyTorch.

[**TensorBoard Profiler Tutorial**](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)

Guide to using TensorBoard's profiling tools for PyTorch models.

### Distribution Techniques

[**Data parallelism**](https://siboehm.com/articles/22/data-parallel-training)

Comprehensive explanation of data parallel training in deep learning.

[**ZeRO**](https://arxiv.org/abs/1910.02054)

Introduces Zero Redundancy Optimizer for training large models with memory
optimization.

[**FSDP**](https://arxiv.org/abs/2304.11277)

Fully Sharded Data Parallel training implementation in PyTorch.

[**Tensor and Sequence Parallelism + Selective Recomputation**](https://arxiv.org/abs/2205.05198)

Advanced techniques for efficient large-scale model training combining different
parallelism strategies.

[**Pipeline parallelism**](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism)

NVIDIA's guide to implementing pipeline parallelism for large model training.

[**Breadth first Pipeline Parallelism**](https://arxiv.org/abs/2211.05953)

Includes broad discussions around PP schedules.

[**All-reduce**](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)

Detailed explanation of the ring all-reduce algorithm used in distributed
training.

[**Ring-flash-attention**](https://github.com/zhuzilin/ring-flash-attention)

Implementation of ring attention mechanism combined with flash attention for
efficient training.

[**Ring attention tutorial**](https://coconut-mode.com/posts/ring-attention/)

Tutorial explaining the concepts and implementation of ring attention.

[**ZeRO and 3D**](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/#understanding-performance-tradeoff-between-zero-and-3d-parallelism)

DeepSpeed's guide to understanding tradeoffs between ZeRO and 3D parallelism
strategies.

[**Mixed precision training**](https://arxiv.org/abs/1710.03740)

Introduces mixed precision training techniques for deep learning models.

[**Visualizing 6D Mesh Parallelism**](https://main-horse.github.io/posts/visualizing-6d/)

Explains the collective communication involved in a 6D parallel mesh.

### Hardware

[**Fire-Flyer - a 10,000 PCI chips cluster**](https://www.arxiv.org/abs/2408.14158)

DeepSeek's report on designing a cluster with 10k PCI GPUs.

[**Meta's 24k H100 Pods**](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)

Meta's detailed overview of their massive AI infrastructure built with NVIDIA
H100 GPUs.

[**Semianalysis - 100k H100 cluster**](https://www.semianalysis.com/p/100000-h100-clusters-power-network)

Analysis of large-scale H100 GPU clusters and their implications for AI
infrastructure.

[**Modal GPU Glossary**](https://modal.com/gpu-glossary/readme)

CUDA docs for human

### Others

[**Stas Bekman's Handbook**](https://github.com/stas00/ml-engineering)

Comprehensive handbook covering various aspects of training LLMs.

[**Bloom training chronicles**](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)

Detailed documentation of the BLOOM model training process and challenges.

[**OPT logbook**](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)

Meta's detailed logbook documenting the training process of the OPT-175B model.

[**Harm's law for training smol models longer**](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)

Investigation into the relationship between model size and training overhead.

[**Harm's blog for long context**](https://www.harmdevries.com/post/context-length/)

Investigation into long context training in terms of data and training cost.

[**GPU Mode**](https://www.youtube.com/@GPUMODE/videos)

A GPU reading group and community.

[**EleutherAI Youtube channel**](https://youtube.com/playlist?list=PLvtrkEledFjqOLuDB_9FWL3dgivYqc6-3&si=fKWPotx8BflLAUkf)

ML Scalability & Performance Reading Group

[**Google Jax Scaling book**](https://jax-ml.github.io/scaling-book/)

How to Scale Your Model

[**@fvsmassa & @TimDarcet FSDP**](https://github.com/facebookresearch/capi/blob/main/fsdp.py)

Standalone ~500 LoC FSDP implementation

[**thonking.ai**](https://www.thonking.ai/)

Some of Horace He's blogposts - Making GPUs go BRRR..

[**Aleksa's ELI5 Flash Attention**](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)

Easy explanation of Flash Attention

[**TunibAI's 3D parallelism tutorial**](https://github.com/tunib-ai/large-scale-lm-tutorials)

Large-scale language modeling tutorials with PyTorch.

