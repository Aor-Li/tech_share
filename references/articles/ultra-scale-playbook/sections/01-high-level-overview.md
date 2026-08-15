## High level overview

All the techniques we'll cover in this book tackle one or several of the
following three key challenges, which we'll keep bumping into throughout the
book:

  1. **Memory Usage** : it's a hard limitation - if a training step doesn't fit in memory, training cannot proceed
  2. **Compute Efficiency** : we want our hardware to spend most time computing, so we need to reduce time spent on data transfers or waiting for other GPUs to perform work.
  3. **Communication overhead** : we want to minimize communication overhead as it keeps GPUs idle. To archieve this we will try to make best use of intra-node (fast) and inter-node (slower) bandwidths as well as overlap communication with compute as much as possible.

In many places we'll see that we can trade one of these (computation,
communication, memory) for another (e.g. recomputation or Tensor Parallelism).
Finding the right balance is key to scaling training.

As this book is very extensive, we've made a [cheatsheet](assets/images/ultra-cheatsheet.svg) to help you navigate the book and get the general take-away. Keep it close to your heart as you navigate these stormy waters! 

[![Cheatsheet](assets/images/ultra-cheatsheet.svg)](assets/images/ultra-cheatsheet.svg)

