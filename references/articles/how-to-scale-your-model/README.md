# How to Scale Your Model

> A Systems View of LLMs on TPUs  
> Austin et al., Google DeepMind, 2025

**Source**: https://jax-ml.github.io/scaling-book/  
**Repo**: https://github.com/jax-ml/scaling-book  
**Fetched**: 2026-08-13（raw Markdown from `main`）

本目录为后续章节稿写作参考，原文未改写。

## Chapters

| File | Chapter |
|---|---|
| `index.md` | Intro / outline |
| `roofline.md` | Ch.1 Roofline analysis |
| `tpus.md` | Ch.2 How to think about TPUs |
| `sharding.md` | Ch.3 Sharded matrices |
| `transformers.md` | Ch.4 Transformer math |
| `training.md` | Ch.5 Parallelize for training |
| `applied-training.md` | Ch.6 Training LLaMA 3 on TPUs |
| `inference.md` | Ch.7 Transformer inference |
| `applied-inference.md` | Ch.8 Serving LLaMA 3 on TPUs |
| `profiling.md` | Ch.9 Profile TPU code |
| `jax-stuff.md` | Ch.10 Programming TPUs in JAX |
| `conclusion.md` | Ch.11 Conclusions & further reading |
| `gpus.md` | Ch.12 How to think about GPUs |

## Citation

```bibtex
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
  and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```
