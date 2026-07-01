---
title: "CuTe-ly writing an SM86 SGEMM"
date: 2026-02-05
comment: true
tags: ["CUDA", "C++", "GPU"]
---

I know what you're thinking. A `fp32` GEMM in 2026? For Ampere??

Yes, SGEMM is rarely ever used these days in the ML space given the massive compute throughput gap between CUDA and tensor cores. For reference, an A100 PCIe's (now almost 6 years old) peak theoretical performance for `bf16` is 312 TFLOPs/sec. and 156 TFLOPs/sec. for `tf32`, while `fp32` sits at a "measly" 19.5 TFLOPs/sec. With [mixed-precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) training and quantized inference being the norm, GEMM itself is practically never done in `fp32` today.

*Despite* all of these (good) reasons, SGEMM still acts as a great learning experience for squeezing the most performance out of GPUs (without involving tensor cores, of course). In this blogpost, I will briefly go over the structure of GEMM and iteratively work through how I optimized a kernel using [NVIDIA's CUTLASS CuTe](https://github.com/NVIDIA/cutlass/tree/main) library to match/exceed cuBLAS on *certain* problem shapes on SM86 using my personal RTX 3070 Mobile card.

Disclaimer: this post will be for those relatively new to GPU programming.

## a brief intro on GEMMs
Large and square-ish GEMM (general matrix-multiply) routines are embarrassingly parallel and compute-bound. GPUs being massively parallel SIMT processors, they are the ideal hardware to run such algorithms on quickly. 

> [!NOTE] A Simplified Model
> To see why large GEMMs are compute-bound, consider an SGEMM $C = AB$ of shape $2048 \times 2048 \times 2048$.
> 
> Simplifying the memory operations to loading $A, B$ once and storing to $C$ once from/to DRAM, this results in $(2048^2 \times 3) \times 4$ bytes of movement required. The arithmetic intensity (AI) required can be computed with $2 \times M \times N \times K$, as $K$ inner products & accumulations (using `FMA`) are required along both $M$ and $N$ axes and dividing by the bytes of memory moved. Thus, the AI is $\frac{2(2048^3)}{3(4)(2048^2)} \approx 341$ FLOPs/byte.
> 
> Now, mapping to hardware, we use the A100 PCIe 80 GB. as an example, which has an off-chip DRAM bandwidth of $1.94$ TFLOPs/sec. and peak `fp32` performance of $19.49$ TFLOPs/sec. Theoretically, with no compute roof, DRAM could allow us a blistering $1.94 \frac{\text{TB.}}{\text{sec.}} \times 341 \frac{\text{FLOPs}}{\text{byte}} \approx 661 \frac{\text{TFLOPs}}{\text{sec.}}$. However, since the underlying `fp32` ALUs are capped at $19.49$ TFLOPs/sec., we are compute-bound. Note that in reality, we will almost never reach either of these roofs, as we will later see. 
> ![roofline diagram](/images/posts/sm86_gemm/roofline-dark.png "an example roofline diagram with an `fp32` ridge point (not to scale)")
> 
> For a more detailed explanation of the roofline model, see Modal's [glossary](https://modal.com/gpu-glossary/perf/roofline-model).

So the goal is straightforward: keep compute units busy by making sure data is always there when they need it. This pushes us towards using faster levels of the CUDA memory hierarchy. This includes the non-programmable L1 and L2 caches, and programmable shared memory (SMEM) and registerfile. These on-chip memory regions are much smaller in size, but have far higher throughput than DRAM. A more detailed discussion of these memory levels is available in the [CUDA docs](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

> The basic hardware execution unit of NVIDIA GPUs is the streaming multiprocessor (SM), which contains shared memory, the registerfile, ALUs, LSUs, warp schedulers, and more. Warp schedulers issue instructions for warps to execute in lockstep (barring warp divergence), and can swap executing warps extremely quickly when some are in a blocked state, enabling latency hiding. See the Modal glossary for more details. 

To exploit this memory hierarchy and the large data reuse inherent to GEMM, most large routines follow a similar pattern: tile and move $A, B$ into SMEM and loop along the $K$-axis for each thread/warp to accumulate its own small outer product tile within the SMEM tile. See the basic pseudocode below. With the introduction of tensor cores, the motivation for tiling became stronger, as these specialized hardware units execute warp-wide mini-GEMMs on specific tile shapes. Newer microarchitectures like Hopper (SM90) and Blackwell (SM100+) also have WGMMA, TMA, and TMEM, which are hardware features that can accelerate such routines greatly.

```c++
// SMEM tiles
__shared__ float A_tile[block_M * block_K];
__shared__ float B_tile[block_K * block_N];

// chunking along K-axis
const int tiles = CEIL_DIV(K, block_K);

// thread-local accumulator in registerfile
float C_accum[thread_M * thread_N] = {0.0};
// outer-product operands
float A_register[thread_M] = {0.0};
float B_register[thread_N] = {0.0};

for (int tile = 0; tile < tiles; ++tile) {
    for (int load_tile = 0; load_tile < A_load_tiles; ++load_tile) {
        // ...LDGSTS A SMEM tile
    }
    for (int load_tile = 0; load_tile < B_load_tiles; ++load_tile) {
        // ...LDGSTS B SMEM tile
    }
    __syncthreads();
    for (int k = 0; k < block_K; ++k) {
        for (int m = 0; m < thread_M; ++m) {
            // ...LDS A outer product operand
        }
        for (int n = 0; n < thread_N; ++n) {
            // ...LDS B outer product operand
        }
        for (int m = 0; m < thread_M; ++m) {
            for (int n = 0; n < thread_N; ++n) {
                // ...accumulate outer products
            }
        }
    }
    __syncthreads();
}
for (int m = 0; m < thread_M; ++m) {
    for (int n = 0; n < thread_N; ++n) {
        // ...STG accumulated registerfile
    }
}

// LDG = read from DRAM, STG = write to DRAM
// LDS = read from SMEM, STS = write to SMEM
```

Unfortunately for us, Ampere does not have such features. The closest equivalent are the `cp.async` family of instructions, which allow a warp to asynchronously transfer memory from DRAM to SMEM (one-way), bypassing the registerfile without blocking. Note that `cp.async` requires every thread in a block to cooperatively load a tile into SMEM, unlike TMA. Still, this instruction will come in handy, as we can asynchronously load the next tile of data into a circular SMEM buffer while the current tile is being processed to keep compute units busy. 

We use a similar pipelining technique when moving data to the registerfile from SMEM. We allocate 2 "$K$-slices" worth of operand data (or as we will call them &mdash; blocks) from $A, B$ and have each register-backed buffer ping-pong between compute and reading the next blocks from SMEM (more in-depth discussion later). This diagram from NVIDIA outlines the approach well:

<a id="software-pipeline"></a>
![pipeline diagram](/images/posts/sm86_gemm/software-pipeline.jpg)

## host-side setup
With all that being said, let's setup the kernel! We'll hand off most of the indexing calculations to CuTe to deal with at compile-time, so that we can save some of our precious registers. We'll start with the necessary setup on host-side. This will be a `nn` kernel, where both inputs are not transposed and expected to be in column-major format.
```c++
void nn(int m, int n, int k, 
        float alpha, const float* A, int ldA,
        const float* B, int ldB, float beta,
        float* C, int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<32>{});
    auto stride_A = make_stride(Int<1>{}, ldA);
    auto stride_B = make_stride(ldB, Int<1>{});
    auto stride_C = make_stride(Int<1>{}, ldC);

    constexpr int n_pipes = 3;
    auto sA_layout = make_layout(make_shape(select<0>(cta_shape), select<2>(cta_shape), Int<n_pipes>{}));
    auto sB_layout = make_layout(
        make_shape(size<1>(cta_shape), size<2>(cta_shape), Int<n_pipes>{}),
        make_stride(size<2>(cta_shape), Int<1>{}, size<1>(cta_shape) * size<2>(cta_shape))
    );
    constexpr uint smem_size = (cosize_v<decltype(sA_layout)> + cosize_v<decltype(sB_layout)>) * sizeof(float);
    ...
}
```
We define the SMEM tiling shapes for $A, B$ to be $128 \times 32$, and the strides for their entire tensors in DRAM to be column-major. The DRAM and SMEM $B$ tile shapes are $(N, K)$ rather than the usual $(K, N)$ to consistently keep the reduction mode $K$ on the right, like $A$, which is $(M, K)$. For the SMEM tiles, we'll use a 3-stage circular buffer for both. To use the widest granularity for reads/writes (i.e. 128-bits), both SMEM tiles should also be in column-major. Wider reads and writes lower instruction count and *generally* improve performance. 

> `cp.async` can move memory at 4-byte, 8-byte, and 16-byte granularities. Furthermore, reads AND writes must strictly be to contiguous chunks of memory, unlike with the standard vectorized loads. For the $B$ matrix, some `nn` GEMM implementations read 16 contiguous bytes from DRAM and chunk it into 4 discontiguous 32-byte writes to SMEM to transpose it to enforce a stride-1 along the $N$-axis instead of $K$. This allows for 16-byte reads from SMEM to the registerfile later. Thus, we unfortunately cannot do something similar here.

> [!NOTE] CuTe Layouts
> A CuTe tensor is comprised of an iterator and Layout, which is a potentially nested tuple of a Shape and Stride, which themselves are potentially nested tuples. The layout is convenient because it informs us of the logical shape of a tensor *and* the "contiguity" of the elements, which is very important when writing kernels. However, at its core, a layout is simply a function from $Z^n \rightarrow Z$ that results in a linear offset by computing a dot product between an input $n$-mode tuple (coord) and the layout's stride. Lastly, CuTe uses colexiographical order; it always "fills" a layout starting from the left-most mode moving to the right.
> 
> There are many great resources out there that helped me formally understand the layout concept, like Colfax Research's [paper](https://www.arxiv.org/pdf/2601.05972) and Lei Mao's [blogpost](https://leimao.github.io/article/CuTe-Layout-Algebra/). The two can be pretty dense with the math though, so the good old documentation from the CUTLASS team helped me out here a lot too. 

![smem layouts](/images/posts/sm86_gemm/smem-layouts.png "We can visualize the two SMEM tile layouts below. I've attempted to show the stride-1 direction with the arrows for each layout. That is, they represent how the tensor's physical linear memory is arranged with `sA_layout` and `sB_layout`.")

We sum up the cosizes of each SMEM layout to retrieve the number of bytes of SMEM required for the kernel. We need to dynamically allocate SMEM in our kernel, as CUDA caps static allocation at 48 kilobytes across all microarchitectures (we need $128 \times 32 \times 3 \times 2 \times 4 = 98304$ bytes). 

A side note: Cosize represents the size of the codomain of a layout. When a layout is compact and bijective, the size is equivalent to the cosize. However, there may be times where this is not the case:
- It has "gaps" in the codomain (i.e. non-compact but bijective), in which its cosize will be greater than its size.
- Non-injective layouts, when multiple coordinates in the domain map to the same value in the codomain (e.g. broadcasting), where its cosize is less than its size.

For us, the SMEM layouts are compact and bijective. We still use cosize though to enforce that they require the minimum size of the codomains to back their layouts in physical memory, independent of the tensors' logical size.

```c++
auto copy_A = make_tiled_copy(
    Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, float>{},
    make_layout(make_shape(Int<32>{}, Int<8>{})),
    make_layout(make_shape(Int<4>{}, Int<1>{}))
);
auto copy_B = make_tiled_copy(
    Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>, float>{},
    make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}),
    make_layout(make_shape(Int<1>{}, Int<4>{}), LayoutRight{})
);
auto mma = make_tiled_mma(
    MMA_Atom<UniversalFMA<float>>{},
    Layout<Shape<_16,_16>>{}
);
auto kernel = ampere_sgemm_128x32_3stage<decltype(stride_A), decltype(stride_B), decltype(stride_C),
                                         decltype(sA_layout), decltype(sB_layout), decltype(cta_shape),
                                         decltype(copy_A), decltype(copy_B), decltype(mma)>;
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

dim3 block_dim(size(mma));
dim3 grid_dim(size(ceil_div(m, select<0>(cta_shape))), size(ceil_div(n, select<1>(cta_shape))));
kernel<<<grid_dim, block_dim, smem_size, nullptr>>>(
    m, n, k, alpha, beta, A, stride_A, B, stride_B, C, stride_C, cta_shape, sA_layout, sB_layout, copy_A, copy_B, mma
);
```

Setting up the tiled MMA and copy objects, there's a bit of boilerplate code needed.

`make_tiled_copy` allows us to arrange a group of threads with a certain layout and assign a group of elements for each thread to be responsible for. We wrap a `CopyTraits` object to use the SM80 `cp.async.cg.shared.global.L2::128B` instruction with a read granularity of 128-bits = 16-bytes. The `CopyAtom` then takes this `CopyTraits` and the actual dtype to be used.

The layout of the threads for $A$ is $(32 \times 8)$ in column-major, with each thread responsible for 4 column-major elements, whereas for $B$ both are in row-major arrangement. It is very important to write correct layouts for any problem to ensure read-write access patterns that we actually intended (CuTe will also statically fail if given an impossible access pattern). 

A good way to start writing these is to write out the ideal case, or what we'd expect the tiling to be. To coalesce 128-byte DRAM reads and use thread-local 16-byte loads, there should be a stride of 1 between each thread's 4 `fp32`s and stride of 4 between threads.

![a partition](/images/posts/sm86_gemm/partition.png "Visualizing the layout for $A$ (not to scale, some values omitted)")

Above is an ideal partitioning of the destination tensor. We can see *some* values mapped to Thread 0 in pink. The green represents the first 4 values mapped to Thread 1. Thread 2's values would be right below, and so on, tiling in column-major order. The blue represents the actual stride required in physical linear memory to jump to the next value. The stride-1 between each value within the 4-value blocks allow us to use the wide 16-byte instructions. This layout also maximally coalesces GMEM (global memory) transactions.

> [!IMPORTANT]- Coalescing & Global Memory
> Recall that any GMEM read fetches a contiguous 128-byte cache line into L2 at minimum. When every thread in a warp reads a contiguous, adjacent element, the hardware coalesces those 32 requests into the minimum number of cache lines. With our layout, 32 threads &times; 16 bytes = 512 bytes = 4 cache lines. In the worst case, with strided accesses, each thread could require its own cache line fetch, multiplying transaction count by up to 32. Note that coalescing applies to accesses of threads *within* a warp, not *across* warps.

Just to verify, we can print `copy_A.tidfrg_D(dst_tensor)`:

$$
(256,(4,1),(1,4,3)):(4,(1,0),(0,1024,4096))
$$

The comments in CuTe source code for `tidfrg_S` and `tidfrg_D` explain what these layouts represent:

```c++
// Tile a tensor or a layout from shape
//   (M,N,...)
// to shape
//   (Thr,(FrgV,FrgX),(RestM,RestN,...))
// where
//   Thr:   The logical threads within the tiled copy.
//   FrgV:  The values local to a COPY_ATOM Dst.
//   FrgX:  The values tiled across COPY_ATOMs Dst.
//   RestM: The values tiled in M.
//   RestN: The values tiled in N.
auto tidfrg_D(DTensor&& dtensor) {
    ...
}
```

Corresponding each term to what we saw, we see that this tiled copy uses 256 threads, where each thread is mapped to 4 values local to a copy atom (16-byte `cp.async` atom = 4x `fp32`), 4 values tiled across the K-mode, and 3 values tiled across the shared memory's 3 stages, matching our earlier visualization. Internally, `partition_X` uses the layouts from `tidfrg` by indexing the `Thr`-mode with the thread index in the actual kernel.

> [!IMPORTANT] Occupancy & Parallelism
> You may be asking how we chose the number of threads to use. GPUs have constrained on-chip memory sizes. For compute-bound problems, we typically prefer loading larger tile sizes to faster on-chip memory since data reuse is high. Fetching data from HBM once for larger tiles and reusing it from shared memory or registers helps amortize that memory latency across more arithmetic operations.
> 
> However, this comes at a price &mdash; heavier on-chip memory usage reduces occupancy, the number of warps (group of 32 threads) resident on an SM. Fewer warps means the scheduler may not be able to switch to another warp when one is "blocked", potentially preventing latency hiding. This occupancy vs. arithmetic intensity tradeoff is an important concept that informs us of a kernel's optimal launch configuration.

Lastly, we set up a `TiledMMA` object, which assigns threads to values for matrix-multiply accumulation (MMA). Similar to `TiledCopy`, it takes an `MMA_Trait` wrapping an `MMA_Atom`, which defines the smallest "unit" of computation to be tiled. This allows for easy partitioning of the operand $A$ and $B$ matrices and register allocation for the product $C$ fragment. Since our operands are in `fp32`, which don't have tensor core support, we use the `UniversalFMA<float>` atom here. But, for example, if our operands were to be in half-precision, we'd want to use an atom like `SM80_16x8x16_F16F16F16F16_TN` that wraps a tensor core (TC) instruction.

Each of these atoms intrinsically involve a different number of threads. For `FMA`, it's just 1 thread performing a scalar computation. Ampere TC instructions are at warp-level; they require a warp to cooperatively compute a small matrix product, while Hopper TCs involve a warpgroup (4 warps) and so on. We'll focus on the simple `FMA` case.

Suppose that after we've done some tuning, we decide to use 256 threads in our kernel. Of course, writing a kernel for first time, you wouldn't know the optimal number of threads, but 128 or 256 threads tend to be good starting points for GEMMs. With each thread computing 1 scalar fragment of $C$, a threadblock of 256 threads can independently compute 256 $C$ values "per atom". For simplicity (and other reasons we'll later see), we'll arrange the atom's 256 values to compute a $(16 \times 16)$ fragment of $C$.

![c-mma](/images/posts/sm86_gemm/c-mma.png "`print_latex` helps us visualize the atom. Each thread computes 1 scalar product and accumulates to its 1 fragment register.")

You may be asking: "But I thought our setup was for each threadblock to accumulate a $(128 \times 128)$ tile of $C$?" `TiledMMA` exposes similar methods to `TiledCopy`, where we can partition an arbitrarily-sized input tensor according to this smaller atom layout. In other words, it replicates the atom across given matrices. Partitioning a $(128 \times 128)$ tile of $C$ thus results in an $(8 \times 8)$ tile of atoms. There is also `thrfrg`, which is the MMA equivalent to the `tidfrg` from `TiledCopy` we saw earlier. Printing `thrfrg_C(gC).shape`:

$$
((1, (16, 16)), (1, (8, 8)))
$$

And again, looking at the source code comments:

```c++
// Tile a tensor or a layout from shape
//   (M,N,...)
// to shape
//   ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))
// where
//   ThrV:  The threads local to an MMA. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
//   ThrM:  The threads tiled in M.      layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
//   ThrN:  The threads tiled in N.      layout<2>(ThrLayoutVMNK): ThrN -> thread_idx
//   FrgV:  The values local to an MMA.
//   RestM: The values tiled in M.
//   RestN: The values tiled in N.
auto thrfrg_C(CTensor&& ctensor) const {
    ...
}
```
We see that there is exactly 1 thread and 1 value local to the MMA atom, and 16 threads tiled in the M- and N-modes. To cover the entire threadblock tile, there then must be 8 values tiled in the M- and N-modes too, meaning the $C$ fragment alone requires 64&times; 32-bit registers per thread. That is, each thread is computing an $(8 \times 8)$ outer product. Let's look at `thrfrg_A` now:

$$((1,(16,1)),(1,(8,32,3)))$$

Notice here how *no* threads are tiled in the K-mode. Instead, each thread owns all values in its K-mode, as it requires an accumulation over this mode. Now, we can move onto the block/thread-local setup within the actual kernel.

## device-side setup
```c++
// cta_shape is (Tile M, Tile N, Tile K) = (128, 128, 32)
auto mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(m, k), stride_A));
auto mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(n, k), stride_B));
auto mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(m, n), stride_C));

auto coord = make_coord(blockIdx.y, blockIdx.x, _);
auto gA = local_tile(mA, cta_shape, coord, Step<_1, X, _1>{});
auto gB = local_tile(mB, cta_shape, coord, Step<X, _1, _1>{});
auto gC = local_tile(mC, cta_shape, coord, Step<_1, _1, X>{});
```

This code is boilerplate for GEMMs. We first create the dynamically-shaped global memory tensors. To "assign" each $(M, N)$ tile coord to threadblocks, we create a `Coord` object based on block index, and use the `local_tile` API. Internally, this function is a wrapper on top of `zipped_divide` that take `Coord` and `Step` objects, which specify which tile coord we want to slice out and which modes we want to keep/discard, respectively. Notice the `X` in the `Step` object, which means that the corresponding mode in the `cta_shape` divisor is dropped. Since it's of shape $(\text{Tile M, Tile N, Tile K})$, we specify the `X` at mode-1 for $A$ and mode-0 for $B$.

For example, for `gA`, this means we index mode-0 of the "quotient" tensor of shape $\left(\frac{M}{\text{Tile M}}, \frac{K}{\text{Tile K}}\right)$ by `blockIdx.y` and keep all of mode-1, since each block does a reduction along the K-mode. Thus, we set `Coord` to `(blockIdx.y, blockIdx.x, _)` &mdash; *in CuTe syntax, an underscore indicates we keep all of that mode.* This results in `gA` and `gB` of shape $(128, K)$.

![threadblock-partitioning](/images/posts/sm86_gemm/block-tile.png "A visual of threadblock partitioning (not to scale)")

```c++
extern __shared__ float smem_buffer[];
auto sA = make_tensor(make_smem_ptr(&smem_buffer[0]), sA_layout);
auto sB = make_tensor(make_smem_ptr(&smem_buffer[cosize_v<sALayout>]), sB_layout);

auto tA = copy_A.get_thread_slice(threadIdx.x);
auto tAgA = tA.partition_S(gA);
auto tAsA = tA.partition_D(sA);

auto tB = copy_B.get_thread_slice(threadIdx.x);
auto tBgB = tB.partition_S(gB);
auto tBsB = tB.partition_D(sB);
```

We then setup the shared memory tensors from a single buffer, and partition the source and destination tensors for each thread based on the `TiledCopy` objects we discussed earlier. Printing `tAsA.shape` from any thread gives us $(4, 1), 1, 4, 3$. We can see that this is exactly what we saw from `tidfrg_D`, but without mode-0, since we've now sliced that tensor to get the thread-local partition. Similarly, `tAgA.shape` prints $(4, 1), 1, 4, 32$ when $K = 1024$, as $\frac{1024}{\text{Tile K}} = 32$ (mode-3 represents values tiled across $K$).

```c++
using blockPipes = _2;  // num. of block buffers
...
auto tC = tiled_mma.get_thread_slice(threadIdx.x);
auto tCsA = tC.partition_A(sA);
auto tCsB = tC.partition_B(sB);
auto tCgC = tC.partition_C(gC);
auto tCrA = make_fragment_like(composition(tCsA(_,_,_,0), make_shape(_,_,blockPipes{})));
auto tCrB = make_fragment_like(composition(tCsB(_,_,_,0), make_shape(_,_,blockPipes{})));
auto tCrC = make_fragment_like(tCgC);
fill(tCrC, 0.0);
```

Last component for the setup is now setting up the thread-local MMA registers. `TiledMMA` already gave us a thread-value mapping for $A$, $B$, and $C$, so we just need to slice out each thread-local view and partition the input tensors. Notice that `gC` is referring to the *global memory* pointer for $C$, since we're not tiling it into SMEM, only $A$ and $B$.

You may be asking what this is doing: `composition(tCsA(_,_,_,0), make_shape(_,_,blockPipes{}))`. Recall the pipelining diagram from [before](#software-pipeline), and how each thread owns all of its K-mode. Ideally, we want each thread to begin its load for the next "slice" of its K-mode from SMEM *while* doing compute on this current slice to save some cycles. The cost though is that we require more registers to eliminate the potential scoreboard stall.

> [!NOTE]+ Scoreboard
> In CUDA, the scoreboard is a memory dependency tracking system. It dynamically ensures that instructions that depend on some overlapping memory run in an order that preserves correctness (more on this from [Modal](https://modal.com/gpu-glossary/perf/scoreboard-stall)). Consider this pseudocode:
> ```c++
> // single register
> float a_reg, b_reg;
> for (k = 0 ... Tile K) {
>     a_reg = sA[k];  // LDS
>     b_reg = sB[k];  // LDS
>     c += a_reg * b_reg;  // FMA (stalls until LDS completes)
> }
> ```
> Because the `FMA` on any iteration reads from the same registers that the `LDS` on that iteration writes to, a scoreboard dependency is created &mdash; the `FMA` stalls until both loads complete, despite the fact that they could run on independent hardware units on an SM (`LDS` on load/store units, `FMA` on CUDA cores). Visualizing this for one of the operands...
> ![scoreboard1](/images/posts/sm86_gemm/scoreboard1.png "Due to the memory dependency between the K-tiles, the compiler cannot order `LDS` for $K = 1$ before or during `FMA`")
> ![scoreboard2](/images/posts/sm86_gemm/scoreboard2.png "By allocating the second register block $B$, we can eliminate the dependency, allowing the compiler to interleave the `LDS` for the next tile with the `FMA` for the current tile. ")
> 
> Note we are at the mercy of the compiler here. It's up to us as the programmer to semantically express the dependency-free condition as clearly as possible and verify that the compiler has generated the correct SASS (nsight-compute is good for this). However, ptxas tends to be good at optimizing these kinds of common pipeline routines.
> 
> If you have some experience in CPU performance optimization, you're probably familiar with this kind of software pipelining. What this typically looks like is scheduling a high latency instruction prior to executing another independent instruction, so that the former does not hurt throughput and they can run on different hardware units. This enables what we call ILP (instruction-level parallelism) &mdash; attempting to exploit parallelism on an instruction-scheduling basis within each warp's instruction stream.

## mainloop
We'll now implement the mainloop, where each threadblock loops over its K-tiles and accumulates its individual matrix product. We begin with a prefetch to avoid some extra conditional statements inside the loop.

```c++
uint gmem_tile_idx = 0;
uint gmem_tiles = size<2>(gA);
constexpr uint smem_pipes = size<2>(sA);

// prefetch for first (smem_pipes - 1) pipes
CUTE_UNROLL
for (uint i = 0; i < smem_pipes - 1; ++i) {
    copy(copy_A, tAgA(_,_,_,gmem_tile_idx), tAsA(_,_,_,i));
    copy(copy_B, tBgB(_,_,_,gmem_tile_idx), tBsB(_,_,_,i));
    cp_async_fence();
    --gmem_tiles;
    if (gmem_tiles) {
        ++gmem_tile_idx;
    }
}
```

We initialize some variables for tracking:
- `gmem_tile_idx` to track which global K-tile is next to be loaded
- `gmem_tiles` is the number of total K-tiles, equal to $\frac{K}{\text{Tile K}}$ (cannot be `constexpr`, as $K$ is a runtime value)
- `smem_pipes` is the number of shared memory pipes (in our case &mdash; 3)

In the prefetch loop, we fire off the loads for the first 2 out of 3 pipes. For each `copy` call, we glob the values local to a copy atom, values tiled across Tile M, and values tiled across Tile K by placing an underscore at those modes. We then copy this K-tile at position `gmem_tile_idx` to the corresponding pipe. Lastly, we decrement the number of K-tiles left to load, and only increment the K-tile position if there are more left. Otherwise, the next iteration simply loads the same global K-tile into the next pipe to prevent a segfault.

> [!IMPORTANT]+ Async Copy
> To use `cp.async` effectively, CuTe provides two key functions:
> - `cp_async_fence()` wraps the `cp.async.commit_group` PTX instruction, which "commits all prior initiated but uncommitted `cp.async` instructions into a cp.async-group." (NVIDIA)
>   - This is effectively a code barrier that allows us to create logical groups of loads, so that we can wait for certain groups to finish before doing some work, while intentionally allowing others to continue. 
> - `cp_async_wait<N>()` wraps the `cp.async.wait_group` instruction, which "will cause executing thread to wait till only N or fewer of the most recent cp.async-groups are pending and all the prior cp.async-groups committed by the executing threads are complete." (NVIDIA)
>   - This gives us the aforementioned ability to wait for certain groups to finish. The fences create an implicit queue, and this instruction lets us "pop off" a specified number of pending groups from the front of the queue.
> ![hello](/images/posts/sm86_gemm/cp-async.png "An example of 6 committed groups and a `cp_async_wait` call with N=4. Guarantees that at most 4 groups are pending (may be less) and anything older is completed.")

