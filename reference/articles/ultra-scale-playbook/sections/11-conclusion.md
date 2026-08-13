## Conclusion

Congratulations, dear reader, you made it to the end! We've completed quite a
journey: we started from understanding how to train a simple model on a single
GPU, all the way to mastering all the intricate techniques used to efficiently
train massive language models like Llama-405B and DeepSeek-V3 on thousands of
GPUs. By now, you can read a diagram, like Llama-3's 4D parallel setup, with
(relative) ease:

![image.png](/assets/images/conclusion_llama3_parallelism.png)

Orchestrating large clusters of GPUs to train LLMs efficiently is no easy feat.
We learned how to optimize computations and communications between GPUs such
that they run with maximum utilization at all times. It involves choosing the
right parallelization strategy for a given model and cluster size, overlapping
communication and computation where possible, and writing custom kernels that
take into account the hardware layout to perform an operation as fast as
possible on the GPU.

You might still believe that this knowledge is a bit niche and only concerns the small set of people that pretrain LLMs. Historically, that may have been true, but as both the [AI builder community](https://huggingface.co) and model sizes are growing rapidly, the community of people using distributed techniques for inference, fine-tuning and training is increasing exponentially as well making distributed training setups more and more common. Diving deeper into all things distributed might thus prove very timely.

This has been a long learning journey, but not just for you! Running thousands
of benchmarks on a GPU cluster was more challenging than we anticipated and we
want to share a few highlights of our own learning experience as well.

### So, what’s next?

You now have good overview of the main distributed training concepts but at the
same time we just scratched to surface of several of these tools and techniques.
There are many ways to dive deep into a subject but here are some steps that we
recommend:

  * Carefully read some of the landmark or very recent papers. You can find a very extenside list of the most impactful papers, blog posts and books in References.
  * Start from scratch and implement an algorithm yourself. Often a method only fully “clicks” if you implemented it yourself.
  * Dive into one of the widely used frameworks and start contributing: fix bugs, answer issues, or implement a new feature. That’s the best way to get in any ML field!

We hope this book helps you get started in distributed training and that you
will train the next generation of awesome models to the hum of your GPU cluster!

* * *

**One last word** for our first readers. We're so happy with this writing piece
that we've decided to distribute a limited number of physical printed editions
of it as a gift for our first readers.

If you are among the first 50 people to fill in your email address below, we'll
contact you later in the year to send you a real physical edition once we've
formatted it as a printed copy.

We expect the book to be around 100-150 pages and to cover the same content as
the blog post but we may also decide to shorten or lengthen it depending on what
make sense as a printed object.

To get your physical copy, please fill in your email address in the following [google form](https://forms.gle/e1GkAShUCtgcwnne8).

Whether you are one of our first readers or coming much later to this blog post,
we've very happy to see that you enjoyed this sharing of knowledge. May the
force of open-source and open-science always be with you.

### Acknowledgements

We thank [Elie](https://huggingface.co/eliebak) for conducting thorough reviews and creating the audio components using NotebookLM. Special thanks to [Hynek](https://huggingface.co/hynky) for optimizing the frontend performance. We also thank [Simon](https://huggingface.co/sbrandeis) for resolving some issues on the hub.

### Discussion page

If you want to discuss the content of this blog post, ask questions, propose changes or just say hi, please open a thread on the [discussion page](https://huggingface.co/spaces/nanotron/ultrascale-playbook/discussions).

