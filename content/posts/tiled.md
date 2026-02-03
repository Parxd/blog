---
title: "understanding CuTe's TiledCopy + TiledMMA through a simple GEMM"
date: 2025-12-06
comment: true
tags: []
---

`thr_layout_mnk` in `make_tiled_mma(...)` is simply tiled layout of this single MMA
- represents number of total threads in CTA in SGEMM UniversalFMA b/c each thread takes on one value

what does `thr_layout_vmnk` represent?
- v: threads local to an MMA atom (universalFMA requires 1 thread, SM70 QP MMA requires 8 threads, SM80 WMMA requires 32 threads, etc.)
- m: threads tiled in M-mode
- n: threads tiled in N-mode
- k: threads tiled in K-mode

this has nothing to do with the source tensor to be partitioned yet, simply represents a layout of threads

we then SELECT the calling thread's position within this layout of threads when constructing `ThrMMA`
- we do this by calling `thr_layout_vmnk.get_flat_coord(threadIdx.x)`
- finds position within the 4D layout
- for UniversalFMA, V-mode will always be 0, since this particular MMA atom only requires 1 thread to perform
- for SM70 QP MMA, the V-mode ranges [0, 7], since it requires a quadpair (4 x 2) to perform

`TiledMMA.get_slice(...)` constructs `ThrMMA` child class, which takes template parameter `thr_vmnk`, which is what we got from the earlier `get_flat_coord` call
- at this point, we've sliced a thread-local view of `thr_layout_vmnk` as mentioned above

now how does `partition_X(...)` work?

taking the example for `A`, we see that it creates a `thr_tensor`, whose layout is `thrfrg_A(tensor.layout)`, where `tensor` is being partitioned

ABSTRACTING `thrfrg` impl. for now...

it returns a layout of shape `((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))`
- `ThrV`: threads local to the MMA atom
- `ThrM`: threads tiled in M-mode
- `ThrK`: threads tiled in K-mode
- `FrgV`: values local to the MMA atom
- `RestM`: values tiled in M-mode
- `RestN`: values tiled in K-mode