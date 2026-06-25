---
title: "CuTe-ly writing an SM86 fp32fp32 GEMM"
date: 2026-02-05
comment: true
tags: ["CUDA", "GPU"]
---

I know what you're thinking. A `fp32fp32` GEMM in 2026? For Ampere??

Yes, yes, SGEMM is rarely ever used these days in the ML space given the massive compute throughput gap between CUDA and tensor cores. For reference, an A100 PCIe's (now almost 6 years old) peak theoretical performance for `bf16` is 312 TFLOPs/sec. and 156 TFLOPs/sec. for `tf32`, while `fp32` sits at a "measly" 19.5 TFLOPs/sec. With [mixed-precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) training and quantized inference being the norm, GEMM itself is practically never done in `fp32` today.

*Despite* all of these (good) reasons, SGEMM still acts as a great learning experience for squeezing the most performance out of GPUs (without involving tensor cores, of course). In this blogpost, I will briefly go over the structure of GEMM and iteratively work through how I optimized a kernel using [NVIDIA's CUTLASS CuTe](https://github.com/NVIDIA/cutlass/tree/main) library to match/exceed cuBLAS on *certain* problem shapes on SM86 using my personal RTX 3070 Mobile card. Fair warning though: this will not only attempt to serve as a lesson on optimizing a kernel but also on using CuTe, as I found it to be a pretty big help when trying to iterate quickly on kernels, although I know some are not the biggest fans of it...

## a brief intro on GEMMs
Large and square-ish GEMM (general matrix-multiply) routines are embarrassingly parallel and compute-bound. GPUs being massively parallel SIMT processors, they are the ideal hardware to run such algorithms on quickly. 

> [!NOTE] A Simplified Model
> To see why large GEMMs are compute-bound, consider an SGEMM $C = AB$ of shape $2048 \times 2048 \times 2048$.
> 
> Simplifying the memory operations to loading $A, B$ once and storing to $C$ once from/to DRAM, this results in $(2048^2 \times 3) \times 4$ bytes of movement required. The arithmetic intensity (AI) required can be computed with $2 \times M \times N \times K$, as $K$ inner products & accumulations (using FMA) are required along both $M$ and $N$ axes and dividing by the bytes of memory moved. Thus, the AI is $\frac{2(2048^3)}{3(4)(2048^2)} \approx 341$ FLOPs/byte.
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

Unfortunately for us, Ampere does not have such features. The closest equivalent are the `cp.async` family of instructions, which allow a warp to asynchronously transfer memory from DRAM to SMEM (one-way), bypassing the registerfile without being blocked. Note that `cp.async` requires every thread in a block to cooperatively load a tile into SMEM, unlike TMA. Still, this instruction will come in handy, as we can asynchronously load the next tile of data into a circular SMEM buffer while the current tile is being processed to keep compute units busy. 

We use a similar pipelining technique when moving data to the registerfile from SMEM. We allocate 2 "$K$-slices" worth of operand data (or as we will call them &mdash; blocks) from $A, B$ and have each register-backed buffer ping-pong between compute and reading the next blocks from SMEM. This diagram from NVIDIA outlines the approach well:

![pipeline diagram](/images/posts/sm86_gemm/software-pipeline.jpg)

## the kernel
With all that being said, let's actually write the kernel! We'll hand off most of the indexing calculations to CuTe to deal with at compile-time, so that we can save some of our precious registers. We'll start with the necessary setup on host-side. This will be a `nn` kernel, where both inputs are not transposed and expected to be in column-major format.
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
We define the SMEM tiling shapes for $A, B$ to be $128 \times 32$, and the strides for their entire tensors in DRAM to be column-major. The DRAM and SMEM $B$ tile shapes are $(N, K)$ rather than the usual $(K, N)$ to consistently keep the reduction mode $K$ on the right, like $A$. For the SMEM tiles, we'll use a 3-stage circular buffer for both. To use the widest granularity for reads/writes (i.e. 128-bits), both SMEM tiles should also be in column-major. Wider reads and writes lower instruction count and *generally* improve performance. 

> `cp.async` can move memory at 4-byte, 8-byte, and 16-byte granularities. Furthermore, reads AND writes must strictly be to contiguous chunks of memory, unlike with the standard vector (e.g. `float2`, `int4`, etc.) types. For the $B$ matrix, some `nn` GEMM implementations read 16 contiguous bytes from DRAM and chunk it into 4 discontiguous 32-byte writes to SMEM to transpose it to enforce a stride-1 along the $N$-axis instead of $K$. This allows for 16-byte reads from SMEM to the registerfile later. Thus, we unfortunately cannot do something similar here.

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

A good way to start writing these is to write out the ideal case, or what we'd expect the tiling to be. To coalesce 128-byte DRAM reads and use thread-local 16-byte loads, there should be a stride of 1 between each thread's 4 `fp32`s and stride of 4 between threads. For example, visualizing the layout for $A$:

![a partition](/images/posts/sm86_gemm/partition.png)

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

Corresponding each term to what we saw, we see that this tiled copy uses 256 threads, where each thread is mapped to 4 values local to a copy atom (16-byte `cp.async` atom = 4x `fp32`), 4 values tiled across the K-mode, and 3 values tiled across the shared memory's 3 stages, matching our earlier visualization. As a side note, the layouts from `tidfrg` are sliced out by each thread by slicing the `Thr`-mode with the thread index within the actual kernel.

> [!IMPORTANT] Occupancy & Parallelism
> You may be asking how we chose the number of threads to use. GPUs have constrained on-chip memory sizes. For compute-bound problems, we typically prefer loading larger tile sizes to faster on-chip memory since data reuse is high. Fetching data from HBM once for larger tiles and reusing it from shared memory or registers helps amortize that memory latency across more arithmetic operations.
> 
> However, this comes at a price &mdash; heavier on-chip memory usage reduces occupancy, the number of warps (group of 32 threads) resident on an SM. Fewer warps means the scheduler may not be able to switch to another warp when one is "blocked", potentially preventing latency hiding. This occupancy vs. arithmetic intensity tradeoff is an important concept that informs us of a kernel's optimal launch configuration.