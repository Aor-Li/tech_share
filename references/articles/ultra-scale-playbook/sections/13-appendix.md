## Appendix

### A0: Parallel Programming Crash Course

Throughout the blogpost we scale LLM training from one to hundreds of GPUs. This
will require the communication and synchronization of weights, gradients, and
data between all the machines. There’s a set of distributed patterns to achieve
exactly that called **_collective operations_**. In this section we’ll do a
small crash course of all the operations like _Broadcast, AllReduce, Scatter_
and more. Let’s dive in!

The general setup is that we have a number of independent nodes which could be
CPU cores, GPUs, or compute nodes. Each performs some computation and then we
want to communicate the result or parts of it to the other nodes for the next
computation step (t+1).

![image.png](/assets/images/a0_general.png)

Maybe we need to send the result from one node to all other nodes, or we need to
sum all the intermediate results from each node to report the overall result.
Usually, there is one node with an elevated status that plays a central role,
here denoted with `root` that is the target or source of some operations. Let’s
start with one of the simplest primitives: a broadcast operation.

#### Broadcast

A very common pattern is that you have some data on Node 1 and you want to share
it with all the other nodes so they can do some computation with the data. The
broadcast operation does just that:

![image.png](/assets/images/a0_broadcast.png)

Collective operations are natively provided by PyTorch so we can easily write a
small example that demonstrates how broadcasting works. We first need to
initialize a process group with `dist.initi_process_group` which sets up the
communication backend (we’ll talk about NCCL later), it determines how many
workers (aka nodes) exists and assigns a rank to each one (which we can get with
`dist.get_rank`). Finally, it establishes a connection between the workers.

To showcase the `dist.broadcast` operation, let's create a tensor with non-zero
values on `rank=0` and tensors full of zeros on the other workers. We then
distribute the `rank=0` tensor to all other ranks with `dist.broadcast(tensor,
src=0)` :

import torch import torch.distributed as dist def init_process():
dist.init_process_group(backend='nccl') torch.cuda.set_device(dist.get_rank())
def example_broadcast(): if dist.get_rank() == 0: tensor = torch.tensor([1, 2,
3, 4, 5], dtype=torch.float32).cuda() else: tensor = torch.zeros(5,
dtype=torch.float32).cuda() print(f"Before broadcast on rank {dist.get_rank()}:
{tensor}") dist.broadcast(tensor, src=0) print(f"After broadcast on rank
{dist.get_rank()}: {tensor}") init_process() example_broadcats()

You can run the above script with `torchrun --nproc_per_node=3 dist_op.py`
(you’ll need 3 GPUs for this or change `nproc_per_node` accordingly) and you
should see the following output:

Before broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0') Before
broadcast on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1') Before
broadcast on rank 2: tensor([0., 0., 0., 0., 0.], device='cuda:2') After
broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0') After
broadcast on rank 1: tensor([1., 2., 3., 4., 5.], device='cuda:1') After
broadcast on rank 2: tensor([1., 2., 3., 4., 5.], device='cuda:2')

Great, seems like it works as expected. Note that the rank messages can be
printed out of order as we have no control over which print statement is
executed first (we ordered them here for readability). Now let’s move on to the
Reduce and AllReduce patterns!

#### Reduce & AllReduce

Reduce patterns are among the most fundamental patterns in distributed data
processing. The idea is that you want to combine the data present on each node
through a function `f()` which can be for instance summation or averaging. In
the Reduce paradigm the result is sent to the root node only, whereas in the
AllReduce case the result is broadcasted to all nodes:

![image.png](/assets/images/a0_reduce_allreduce.png)

Of course no magic “free flying” node that can perform this operation and
generally each node does a partial computation in a ring or tree structure of
the nodes. Here is a simple example: let’s say we need to compute a sum of
numbers on each nodes and our nodes are connected in a ring pattern. The first
node sends its number to a neighbour which adds its number to the received
number before forwarding it to the next neighbour. At the end of a round along
the ring of nodes, the first node will receive the total sum.

Here’s the code to run a simple Reduce operation summing the tensors, we specify the operation to use with `op=dist.ReduceOp.SUM` (you can find more information on the supported operations in the [Pytorch docs](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)):

def example_reduce(): tensor = torch.tensor([dist.get_rank() + 1] * 5,
dtype=torch.float32).cuda() print(f"Before reduce on rank {dist.get_rank()}:
{tensor}") dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM) print(f"After reduce
on rank {rank}: {tensor}") init_process() example_reduce()

Note that in the Reduce operation only the tensor on the `dst` node is updated:

Before reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0') Before
reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1') Before reduce on
rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2') After reduce on rank 0:
tensor([6., 6., 6., 6., 6.], device='cuda:0') After reduce on rank 1:
tensor([2., 2., 2., 2., 2.], device='cuda:1') After reduce on rank 2:
tensor([3., 3., 3., 3., 3.], device='cuda:2')

Similarly we can perform an AllReduce (we don’t need to specify a destination in
this case):

def example_all_reduce(): tensor = torch.tensor([dist.get_rank() + 1] * 5,
dtype=torch.float32).cuda() print(f"Before all_reduce on rank {dist.get_rank()}:
{tensor}") dist.all_reduce(tensor, op=dist.ReduceOp.SUM) print(f"After
all_reduce on rank {dist.get_rank()}: {tensor}") init_process()
example_all_reduce()

In this case the result is available on all nodes:

Before all_reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before all_reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before all_reduce on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2') After
all_reduce on rank 0: tensor([6., 6., 6., 6., 6.], device='cuda:0') After
all_reduce on rank 1: tensor([6., 6., 6., 6., 6.], device='cuda:1') After
all_reduce on rank 2: tensor([6., 6., 6., 6., 6.], device='cuda:2')

Now let’s turn to our next distributed communication operation. In many real
cases, each node individually perform many complex computations and we need to
share the final results among nodes. Gather and AllGather are the operations we
want to use in this case. Let’s take a look!

#### Gather & AllGather

Gather and AllGather are quite similar to the Broadcast in that they allow
distributing data among node without modification. The main difference to
Broadcast is that there is not one value we need to share from one node to all
other nodes but each node has an individual chunk of data that we want to either
gather all data on one node (in case of Gather) or gather all data on all nodes
(in the case of AllGather). A picture being worth 1000 words, let’s take a look:

![image.png](/assets/images/a0_gather_allgather.png)

Note that the dashed lines indicate that some data actually doesn’t move at all
(since it’s already present on the node).

In the case of the gather operation we need to prepare a container objects where
the gathered tensors can be stored in this example the `gather_list`:

def example_gather(): tensor = torch.tensor([dist.get_rank() + 1] * 5,
dtype=torch.float32).cuda() if dist.get_rank() == 0: gather_list = [
torch.zeros(5, dtype=torch.float32).cuda() for _ in range(dist.get_world_size())
] else: gather_list = None print(f"Before gather on rank {dist.get_rank()}:
{tensor}") dist.gather(tensor, gather_list, dst=0) if dist.get_rank() == 0:
print(f"After gather on rank 0: {gather_list}") init_process() example_gather()

And we see that the `gather_list` indeed contains the tensors of all ranks:

Before gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0') Before
gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1') Before gather on
rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2') After gather on rank 0:
[tensor([1., 1., 1., 1., 1.], device='cuda:0'), tensor([2., 2., 2., 2., 2.],
device='cuda:0'), tensor([3., 3., 3., 3., 3.], device='cuda:0')]

The only thing we need to change for the AllGather example is that every node
will need a placeholder for the results:

def example_all_gather(): tensor = torch.tensor([dist.get_rank() + 1] * 5,
dtype=torch.float32).cuda() gather_list = [ torch.zeros(5,
dtype=torch.float32).cuda() for _ in range(dist.get_world_size()) ]
print(f"Before all_gather on rank {dist.get_rank()}: {tensor}")
dist.all_gather(gather_list, tensor) print(f"After all_gather on rank
{dist.get_rank()}: {gather_list}") init_process() example_all_gather()

And indeed we can see that now each node has all the data:

Before all_gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Before all_gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Before all_gather on rank 2: tensor([3., 3., 3., 3., 3.], device='cuda:2') After
all_gather on rank 0: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
tensor([2., 2., 2., 2., 2.], device='cuda:0'), tensor([3., 3., 3., 3., 3.],
device='cuda:0')] After all_gather on rank 1: [tensor([1., 1., 1., 1., 1.],
device='cuda:1'), tensor([2., 2., 2., 2., 2.], device='cuda:0'), tensor([3., 3.,
3., 3., 3.], device='cuda:0')] After all_gather on rank 2: [tensor([1., 1., 1.,
1., 1.], device='cuda:2'), tensor([2., 2., 2., 2., 2.], device='cuda:2'),
tensor([3., 3., 3., 3., 3.], device='cuda:2')]

Now what about the inverse of a gather? In this case we would have all the data
on one node and want to distribute/slice it among node, possibly with some
intermediate processing? We can use the Scatter, or in the case of an operation
in between a Reduce Scatter pattern:

#### Scatter & ReduceScatter

As the name subtly suggests, the goal of the Scatter operation is to take data
on one node and distribute slices of it to all other nodes. It’s thus different
from the Broadcast operation which copy data without slicing and it’s the
logical the inverse of the Gather operation.

The ReduceScatter pattern is slightly more complex: imagine you apply an
operation like in the Reduce case but instead of moving the result to just one
node we also distribute it evenly to all nodes:

![image.png](/assets/images/a0_scatter_reducescatter.png)

The Scatter operation is written in code as the opposite of the Gather: instead
of preparing a list of tensors as target we prepare the source data as a list of
tensors we want to distribute. We also need to specify the `src`:

def example_scatter(): if dist.get_rank() == 0: scatter_list = [ torch.tensor([i
+ 1] * 5, dtype=torch.float32).cuda() for i in range(dist.get_world_size()) ]
print(f"Rank 0: Tensor to scatter: {scatter_list}") else: scatter_list = None
tensor = torch.zeros(5, dtype=torch.float32).cuda() print(f"Before scatter on
rank {dist.get_rank()}: {tensor}") dist.scatter(tensor, scatter_list, src=0)
print(f"After scatter on rank {dist.get_rank()}: {tensor}") init_process()
example_scatter()

As a result we can see how the empty tensors got filled with the contents of the
`scatter_list`

Rank 0: Tensor to scatter: [tensor([1., 1., 1., 1., 1.], device='cuda:0'),
tensor([2., 2., 2., 2., 2.], device='cuda:0'), tensor([3., 3., 3., 3., 3.],
device='cuda:0')] Before scatter on rank 0: tensor([0., 0., 0., 0., 0.],
device='cuda:0') Before scatter on rank 1: tensor([0., 0., 0., 0., 0.],
device='cuda:1') Before scatter on rank 2: tensor([0., 0., 0., 0., 0.],
device='cuda:2') After scatter on rank 0: tensor([1., 1., 1., 1., 1.],
device='cuda:0') After scatter on rank 1: tensor([2., 2., 2., 2., 2.],
device='cuda:1') After scatter on rank 2: tensor([3., 3., 3., 3., 3.],
device='cuda:2')

Let’s create more interesting data to demonstrate the ReduceScatter logic: on
each node we create a list of 2-elements vector on each node with a power
exponent and an offset function of the node rank (it’s a bit hard to imagine so
just look below for an example):

def example_reduce_scatter(): rank = dist.get_rank() world_size =
dist.get_world_size() input_tensor = [ torch.tensor([(rank + 1) * i for i in
range(1, 3)], dtype=torch.float32).cuda()**(j+1) for j in range(world_size) ]
output_tensor = torch.zeros(2, dtype=torch.float32).cuda() print(f"Before
ReduceScatter on rank {rank}: {input_tensor}")
dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
print(f"After ReduceScatter on rank {rank}: {output_tensor}") init_process()
example_reduce_scatter()

Let’s print the pattern of data that we created. We also immediately see the
ReduceScatter pattern: the first rank received the sum of the first tensor from
each node, and the second rank contains the sum of the second tensor on each
node and so on:

Before ReduceScatter on rank 0: [tensor([1., 2.], device='cuda:0'), tensor([1.,
4.], device='cuda:0'), tensor([1., 8.], device='cuda:0')] Before ReduceScatter
on rank 1: [tensor([2., 4.], device='cuda:1'), tensor([ 4., 16.],
device='cuda:1'), tensor([ 8., 64.], device='cuda:1')] Before ReduceScatter on
rank 2: [tensor([3., 6.], device='cuda:2'), tensor([ 9., 36.], device='cuda:2'),
tensor([ 27., 216.], device='cuda:2')] After ReduceScatter on rank 0: tensor([
6., 12.], device='cuda:0') After ReduceScatter on rank 1: tensor([14., 56.],
device='cuda:1') After ReduceScatter on rank 2: tensor([ 36., 288.],
device='cuda:2')

Let's have a quick look at a common implementation of AllReduce that uses
ReduceScatter and AllGather: Ring AllReduce.

#### A quick focus on Ring AllReduce

**_Ring AllReduce_** is one specific implementation of AllReduce, optimized for
scalability. Rather than all devices communicating with each other directly,
which could create communication bottlenecks, Ring All-Reduce can be broken down
into two key steps: ReduceScatter and AllGather. Here's how it works:

  1. **ReduceScatter**
     * Each device splits its data (e.g., gradients) into chunks and sends one chunk to its neighbour. Simultaneously, each device receives a chunk from its other neighbour.
     * As each device receives a chunk, it adds (reduces) its corresponding chunk to the received one.
     * This process continues around the ring until each device holds a partially reduced chunk, representing a sum of the gradients across all devices for that chunk.
  2. **AllGather**
     * Now, each device needs to collect the fully reduced chunks from other devices.
     * The devices start sending their reduced chunks to neighbours.
     * Each device forwards the chunks it receives until every device has all the fully reduced chunks, giving each device the complete, summed-up gradient.

Let’s illustrate this with the following gifs, where we have 5 GPUs, each with a
tensor of length 5. The first animation shows the ReduceScatter step, where, at
the end, each GPU receives the reduced results for a specific chunk of data
(orange rectangle).

![image.png](/assets/images/a0_reduce_scatter.gif)

The next animation shows the AllGather step, where, at the end, each GPU obtains
the full results of the AllReduce operation:

![image.png](/assets/images/a0_all_gather.gif)

You may have noticed that each of the N GPUs sends and receives values N-1 times
during both the reduce-scatter and all-gather steps. Each GPU sends \frac{K}{N}
values per transfer, where K is the total number of values in the array being
summed across the GPUs. Therefore, the total amount of data transferred to and
from each GPU is 2 \times (N-1) \times \frac{K}{N}. When N (the number of GPUs)
is large, the total amount of data transferred to and from each GPU is
approximately 2 \times K, where K is the total number of parameters.

**There are two key things to keep in mind for AllReduce:**

  1. The communication cost for AllReduce is approximately 2xK when N (the number of GPUs) is large.
  2. An AllReduce operation can be broken down into a reduce-scatter followed by an all-gather. The communication cost for these two operations is half that of the AllReduce, which is approximately K.

As we can see this implementation can make efficient use of even a limited
bandwidth between nodes.

We now have seen the main building block of distributed operations but before we
see them in action let’s have a look at a special operation used for
synchronization: the Barrier.

#### Barrier

The Barrier is a simple operation to synchronize all nodes. A barrier is not
lifted until all nodes have reached it. Then only are they allowed to continue
with further computations:

![image.png](/assets/images/a0_barrier.png)

We can easily simulate delayed nodes by setting up a different sleep time on
each node and see how long it takes for all of them to pass the barrier:

def example_barrier(): rank = dist.get_rank() t_start = time.time() print(f"Rank
{rank} sleeps {rank} seconds.") time.sleep(rank) # Simulate different processing
times dist.barrier() print(f"Rank {rank} after barrier time delta:
{time.time()-t_start:.4f}") init_process() example_barrier()

We can see that although the first rank didn’t sleep at all it also took it 2sec
to pass the barrier:

Rank 0 sleeps 0 seconds. Rank 1 sleeps 1 seconds. Rank 2 sleeps 2 seconds. Rank
0 after barrier time delta: 2.0025 Rank 1 after barrier time delta: 2.0025 Rank
2 after barrier time delta: 2.0024

We need to be careful with synchronizing all nodes like this, as this defeat the
purpose of parallel independent operations and might thus slow down the whole
processing. In many situations it can be just fine if a fast node already starts
processing the next job as the fast node could be slower in a next iteration
therefore evening out the delay over the whole process.

Before turning to practical distributed training implementations, let’s first
solve a mystery: what the heck is NCCL?

#### NCCL: NVIDIA Collective Communications Library

When training large models on many GPUs we may sometimes strike gold but we will
always encounter nickel (or NCCL 🥁)! What’s is that?

There are several libraries that implement collective communication and are support by PyTorch: there’s the classic **_MPI_** (Message Passing Interface), there’s **_Gloo_** by Meta, and finally there is `NCCL` (NVIDIA Collective Communications Library). They all provide similar functionality in terms of collective communication patterns but are optimized for different hardware setups; NCCL is designed to serve GPU-GPU communication efficiently while MPI and Gloo are setup for CPU-CPU or CPU-GPU communication. PyTorch provides a [great guide](https://pytorch.org/docs/stable/distributed.html#which-backend-to-use) to decide which one to use:

  * GPU training: use NCCL
  * CPU training: use Gloo

There are a few finer points in the decision tree that we leave to the reader to
explore in the PyTorch guide referenced above.

Now that we covered the fundamental operations for distributed training and you
should now be ready to follow the blog post easily.

### A1: Distributed Training Profiling

#### Kernels

Let's begin by assuming for now that the kernels are already integrated into
PyTorch. As a simple example, we can look at the Layer Normalization function
implemented in PyTorch as `torch.nn.functional.layer_norm`. There are several
methods to profile the kernel that underlies this function. The most
straightforward approach might be to use the Python `time` module. However,
since CUDA operations are asynchronous, measuring time with this method will
only capture the overhead associated with launching the kernel in Python, rather
than the actual execution time of the kernel itself.

To address this, we can utilize `torch.cuda.Event` for accurate timing and
employ the `torch.cuda.synchronize()` directive to ensure we wait for the kernel
execution to complete. This approach is demonstrated in the following snippet:

def profile_pytorch(func, input): # Create CUDA events to track time. CUDA
operations are asynchronous, start = torch.cuda.Event(enable_timing=True) #
Event to mark the start time end = torch.cuda.Event(enable_timing=True) # Event
to mark the end time # Warmup to eliminate any overhead from the first run,
which might not reflect # the actual performance. for _ in range(10):
func(input) # Record the start time before executing the function start.record()
func(input) # Call the function we want to profile # Record the end time after
the function has completed end.record() # Synchronize the CUDA operations to
ensure all operations are completed # before measuring the elapsed time.
torch.cuda.synchronize() # Calculate and return the elapsed time in
milliseconds. return start.elapsed_time(end)

A more effective approach to profiling is to utilize the PyTorch Profiler, as
explained previously. For example, consider the following code:

import torch import torch.nn.functional as F def pytorch_layer_norm(input):
return F.layer_norm(input, input.size()[1:]) a = torch.randn(10000,
10000).cuda() with torch.profiler.profile( activities=[
torch.profiler.ProfilerActivity.CPU, # Profile CPU activities
torch.profiler.ProfilerActivity.CUDA, # Profile CUDA activities ], # Define a
schedule for the profiler schedule=torch.profiler.schedule( wait=1, # Wait for 1
iteration before starting to profile warmup=3, # Warm up for 3 iterations to
stabilize performance active=2, # Profile for 2 active iterations repeat=1, #
Repeat the profiling schedule once ),
on_trace_ready=torch.profiler.tensorboard_trace_handler('.'), ) as p: for iter
in range(10): pytorch_layer_norm(a) p.step() # Print a table of the profiling
results, sorted by total CUDA time, limited to the top 10 entries
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))

This would print aggregated profiling results sorted by the total CUDA time, and
the output would be:

![image.png](/assets/images/a1_kernels.png)

You can also try to inspect the trace as we previously mentioned on
`chrome://tracing/`

💡 Tip

If you're new to this tool, you can navigate the trace by using the right and
left arrow keys. Additionally, you can zoom in and out by holding the **Alt**
key while scrolling left or right with your mouse.

After zooming in, you can observe the flow of operations when calling
`layer_norm` in this trace:

![image.png](/assets/images/a1_profile_trace.png)

The sequence begins in the CPU (the upper section) with `aten::layer_norm`,
progressing to `aten::native_layer_norm`, and then transitioning to
`cudaLaunchKernel`. From there, we move on to the GPU, where the
`vectorized_layer_norm_kernel` kernel is called.

📝 Note

You can enable memory profiling by setting `profile_memory` to `True` in the
profiler. However, this can lead to more complex traces.

While the PyTorch Profiler offers a quick performance overview, **NVIDIA Nsight
Compute (ncu)** provides deeper insights into GPU performance, including
detailed execution times and memory usage for each kernel. To run the profiler
it's very simple:

ncu --set full python layer_norm.py

Where `layer_norm.py` is a straightforward file that executes the layer
normalization function. This command will generate log outputs, but a more
effective way to visualize the results is by setting the output flag:

ncu --set full -o output python layer_norm.py

and open the file `output.ncu-rep` with Nsight Compute, you will have a view
that looks like this:

![image.png](/assets/images/a1_ncu.png)

With clear warnings about compute and memory utilization, and how to make the
kernel better in balancing compute and memory and achieve maximal occupancy.

#### CPP extension

If the kernel you want to profile isn't already integrated into PyTorch, you can
use PyTorch's `cpp_extension` module to easily compile and run custom CUDA code.
The process is straightforward—just create your CUDA kernel in a `.cu` file, and
use the `load` function from the `cpp_extension` module to load it in Python.

The `.cu` file would like this for a simple `add` kernel:

#include  #include  #include  __global__ void add_kernel(float* x, float* y,
float* output, int size) { int index = blockIdx.x * blockDim.x + threadIdx.x; if
(index < size) { output[index] = x[index] + y[index]; } } void
add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor output) { int threads =
1024; int blocks = (x.size(0) + threads - 1) / threads;
add_kernel<<>>(x.data_ptr(), y.data_ptr(), output.data_ptr(), x.size(0)); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_cuda", &add;_cuda, "Vector
addition (CUDA)"); }

And the python file to load the kernel:

import torch from torch.utils.cpp_extension import load # Load and compile the
CUDA extension vector_add = load( name="vector_add", sources=["add_kernel.cu"],
verbose=True ) # Define input tensors size = 10000 x = torch.randn(size,
device='cuda') y = torch.randn(size, device='cuda') output = torch.empty(size,
device='cuda') # Run the CUDA kernel vector_add.add_cuda(x, y, output)

Using this method, you can profile the custom CUDA kernel just as we
demonstrated earlier with PyTorch's profiler or NVIDIA tools.

### A2: Typical Scales in LLM Training

Let's get a feel for the typical sizes of things in LLM training. When we talk
about memory or compute, we're often counting "elements" - think of these as
numbers in tensors. To get the actual memory in bytes, you'll need to multiply
by the size of each number (e.g., 2 bytes for bf16, 4 bytes for fp32).

Here are some quick ballpark figures:

  * **Input tokens:** For each batch, we process seq \cdot mbs tokens, where mbs is the micro batch size and seq is the sequence length.
  * **Activations (hidden states):** For a single layer, the hidden state tensor is of size seq \cdot mbs \cdot h elements.
  * **Model weights and gradients:** Each weight matrix in your model (like in linears) is about h^2 elements. This is per weight matrix. Gradients have the same size as weights.
  * **Optimizer states:** For each weight matrix (of elements h^2), if you're using an optimizer like Adam with mixed precision training, it keeps momentum and variance states in fp32 precision (2 \cdot h^2), plus master weights in fp32 (h^2). So total optimizer states will be around (6 \cdot h^2) per weight matrix.
  * **Total model parameters:** For each transformer block: 
    * Attention parameters: 
      * QKV projections: 3h^2 parameters
      * Output projection: h^2 parameters
    * MLP parameters with GLU: 
      * Gate and up projections: 8h^2 parameters (2 matrices of size h \times 4h)
      * Down projection: 4h^2 parameters (1 matrix of size 4h \times h)
    * Total per block: 16h^2 with GLU MLPs, or 12h^2 without GLU
    * For full model: 16h^2 \cdot num\_layers (with GLU)
    * Additional parameters: 
      * Input embeddings: vocab\_size \cdot h
      * LM head: vocab\_size \cdot h (if not tied with input embeddings)
      * Positional embeddings (if used): max\_seq\_len \cdot h
  * **Forward and backward pass compute (FLOPs):** A very rough estimate for the FLOPs in a forward pass is 2 \cdot num\_tokens \cdot num\_params. And backward pass compute is twice as that: 4 \cdot num\_tokens \cdot num\_params.

### A3: Math for Compute/Communication Overlap

Using the formulas from the previous section, we can estimate when computation
and communication can effectively overlap in distributed training. Let's look at
data parallelism (Zero-0) as an example.

#### Data Parallelism Communication Analysis

The total gradient size that needs to be communicated is:

  * Gradients = Parameters ≈ num\_layers \cdot 16h^2

During backward pass, these gradients are communicated in buckets (default
25MB). The communication time to all-reduce each bucket is:

t_{comm} = t_{comm\_bucket} = \frac{bucket\_size \cdot 2(DP-1)}{DP \cdot
peak\_bw}

📝 Note

For bandwidth calculations, we use the bus bandwidth formulas from the [NCCL documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#summary). These formulas account for the specific communication patterns when calculating effective bandwidth between GPUs.

The computation time for backward pass is:

t_{compute} = \frac{4 \cdot num\_tokens \cdot num\_params}{peak\_flops}

For effective overlap, we need:

\frac{t_{comm}}{t_{compute}} = \frac{num\_params}{2 \cdot num\_tokens} \cdot
\frac{DP-1}{DP} \cdot \frac{peak\_flops}{peak\_bw} \leq 1

This ratio helps determine if communication will become a bottleneck in
training. When the ratio is less than 1, communication can be fully overlapped
with computation.

#### ZeRO-3 (FSDP) Communication Analysis

For ZeRO-3, parameters and gradients are sharded across GPUs. Let's analyze the
communication pattern for a model with transformer blocks of size 16h^2
parameters each:

  * For each transformer block in forward pass: 
    * Allgather parameters: 16h^2/DP bytes per rank
  * For each transformer block in backward pass: 
    * Allgather parameters: 16h^2/DP bytes per rank
    * Reducescatter gradients: 16h^2/DP bytes per rank
  * Total communication per block: 3 \cdot 16h^2/DP bytes
  * Total communication for full model: 3 \cdot num\_layers \cdot 16h^2/DP bytes

The communication time for allgather operations is:

t_{comm} = 16h^2 \cdot \frac{DP-1}{DP \cdot peak\_bw}

The computation time for forward pass of one decoder layer is:

t_{compute} = \frac{32 \cdot seq\_len \cdot mbs \cdot h^2}{peak\_flops}

For effective overlap between computation and communication, we need:

\frac{t_{comm}}{t_{compute}} = \frac{1}{2 \cdot seq\_len \cdot mbs} \cdot
\frac{DP-1}{DP} \cdot \frac{peak\_flops}{peak\_bw} \leq 1

When this ratio is less than 1, the communication of parameters for the next
layer can be hidden behind the computation of the current layer.

`

#### TP Communication Analysis

For Tensor Parallel (TP), activations are sharded across GPUs during linears.
Let's analyze the communication pattern:

  * For each column linear in forward pass: 
    * Allgather activations: seq \cdot mbs \cdot h/TP bytes per rank
  * For each column linear in backward pass: 
    * Reducescatter gradients: seq \cdot mbs \cdot h/TP bytes per rank
  * And vice-versa for row linears. Each transformer block has 2 column linears and 2 row linears.
  * Total communication per block: 8 \cdot seq \cdot mbs \cdot h/TP bytes
  * Total communication for full model: 8 \cdot num\_layers \cdot seq \cdot mbs \cdot h/TP bytes

Let's analyze if we can overlap the allgather communication for one layer with
the computation of the next linear. The communication time for allgather
operations is:

t_{comm} = \frac{seq \cdot mbs \cdot h \cdot (TP-1)}{TP \cdot peak\_bw}

While the computation time for the next linear (with parameters h^2) is:

t_{compute} = \frac{2 \cdot seq \cdot mbs \cdot h^2}{TP \cdot peak\_flops}

For effective overlap, we want the communication time to be less than the
compute time:

\frac{t_{comm}}{t_{compute}} = \frac{TP-1}{2 \cdot h} \cdot
\frac{peak\_flops}{peak\_bw} \leq 1

This ratio tells us whether we can successfully hide the allgather communication
behind the computation of the next linear. Interestingly, the ratio only depends
on the hidden size h and tensor parallelism degree TP, not on sequence length or
batch size.

#### PP Communication Analysis

For Pipeline Parallel (PP), activations and gradients are communicated between
pipeline stages. Let's analyze the communication pattern:

  * For each microbatch in forward pass: 
    * Receive and send activations: 2 \cdot seq \cdot mbs \cdot h bytes
  * For each microbatch in backward pass: 
    * Receive and send gradients: 2 \cdot seq \cdot mbs \cdot h bytes
  * Total communication per microbatch: 4 \cdot seq \cdot mbs \cdot h bytes
  * For gradient accumulation steps (gas), total communication: 4 \cdot gas \cdot seq \cdot mbs \cdot h bytes

Let's analyze if we can overlap the communication of activations/gradients with
computation of the next transformer block. The computation time for transformer
blocks in the next pipeline stage is:

t_{compute} = \frac{32 \cdot seq \cdot mbs \cdot h^2 \cdot
num\_layers\_in\_next\_pp}{peak\_flops}

While the communication time for P2P transfer is:

t_{comm} = \frac{seq \cdot mbs \cdot h}{peak\_bw}

For effective overlap, we want:

\frac{t_{comm}}{t_{compute}} = \frac{peak\_flops}{32 \cdot h \cdot
num\_layers\_in\_next\_pp \cdot peak\_bw} \leq 1

Similar to TP, this ratio is independent of sequence length and batch size. It
depends on the hidden size h, number of layers in the next pipeline stage, and
the ratio of compute to P2P bandwidth capabilities of the hardware.

### Citation

For attribution in academic contexts, please cite this work as

    
    
    Tazi et al., "The Ultra-Scale Playbook: Training LLMs on GPU Clusters", 2025.

BibTeX citation

    
    
    @misc{ultrascale_playbook,
          title={The Ultra-Scale Playbook: Training LLMs on GPU Clusters},
          author={Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf},
          year={2025},
    }

