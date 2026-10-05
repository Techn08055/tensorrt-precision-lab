# GEMM

First CUDA-kernel module of this lab (research question 01 in the repo root
[`README.md`](../README.md)). Naive GEMM → tiled → shared memory → register
blocking, benchmarked against cuBLAS at each step, then connected to where GEMM
actually shows up inside [`../resnet/`](../resnet/) — the 1x1 convolutions are
literally a GEMM; the 3x3 ones map to it via implicit GEMM (that connection is
`../conv/`, the next module).

## Status

- [x] GEMM mental model — [`docs/GEMM_MENTAL_MODEL.md`](docs/GEMM_MENTAL_MODEL.md), runnable version at [`docs/understand_gemm.py`](docs/understand_gemm.py) (pure Python + NumPy, no GPU needed)
- [x] Naive kernel spec (design, not yet code) — [`docs/NAIVE_KERNEL_SPEC.md`](docs/NAIVE_KERNEL_SPEC.md)
- [x] Naive kernel implementation + correctness check — [`src/gemm_naive.cu`](src/gemm_naive.cu), [`tests/test_correctness.py`](tests/test_correctness.py) (3/3 sizes pass, max abs error ~3e-6, well under the 1e-3 threshold)
- [x] Naive kernel benchmark (256 → 4096, square sizes) — [`results/benchmark.csv`](results/benchmark.csv): flat at ~570-670 GFLOPS across every size, consistent with a memory-bound kernel (see mental model doc's 0.25 FLOP/byte estimate) rather than one that scales with problem size
- [x] Naive kernel vs pure Python vs NumPy(BLAS), same size — [`results/python_vs_cuda.md`](results/python_vs_cuda.md): up to 17,224x over pure Python at 256³; also shows the naive kernel itself is below its ~570 GFLOPS ceiling at 128³ (not enough threads yet to hide overhead) — the small-problem-size effect from the TensorRT/INT8 side of this lab, showing up again at the kernel level
- [ ] Nsight Compute profile of the naive kernel — confirm the memory-bound read directly instead of inferring it from GFLOPS alone (`ncu` is available on this machine)
- [x] Tiled kernel (shared memory), FP16 in/out with FP32 accumulation — [`src/gemm_tiled_fp16.cu`](src/gemm_tiled_fp16.cu), [`results/tiled_fp16_vs_naive.csv`](results/tiled_fp16_vs_naive.csv): 1.2-1.3x over a same-contract naive FP16 kernel up to 1024, 1.7x at 4096 (2048 varied between runs, 1.34x and 1.64x). Output is bit-identical to the naive kernel and within half-precision rounding of an FP32 CPU reference. Known bug: reads `C` when `beta == 0`, so NaN in an uninitialized `C` propagates
- [x] Register blocking + smaller thread block (same 32x32 tile, each thread computes an RxR patch of `C`) — [`src/gemm_regblock_fp16.cu`](src/gemm_regblock_fp16.cu), [`results/regblock_fp16_comparison.csv`](results/regblock_fp16_comparison.csv): 2x2 and 4x4 are both 2.4-2.9x over the tiled kernel from 512 up (3.2-4.8x over naive); 4x4 is ahead at 4096, 2x2 at 256; 8x8 (168 registers per thread, 16 threads per block) drops back to 1.1-1.7x. Output is bit-identical to the naive and tiled kernels at every size, including 1000 (not a multiple of the tile)
- [x] Tensor cores (WMMA), one 16x16 fragment per warp, loads straight from global memory — [`src/gemm_wmma_fp16.cu`](src/gemm_wmma_fp16.cu), [`results/wmma_fp16_comparison.csv`](results/wmma_fp16_comparison.csv): 5-8x over naive, about 2x over the 4x4 register-blocked scalar kernel up to 2048 and 1.4x at 4096, where its throughput falls from ~5,700-6,400 GFLOPS (2048) to ~3,600
- [x] Tensor cores + shared-memory tiles + 2x2 accumulator fragments per warp — [`src/gemm_wmma_blocked_fp16.cu`](src/gemm_wmma_blocked_fp16.cu), [`results/wmma_blocked_fp16_comparison.csv`](results/wmma_blocked_fp16_comparison.csv): bit-identical to the plain WMMA kernel; 1.3x faster at 512, 1.6x at 1024, 1.8x at 2048, 2.2x at 4096 (~10,100 and ~7,900 GFLOPS at the last two), but 0.65x at 256 (only 16 thread blocks). Staging 32 k-columns per step beats 64 at every size from 512 up
- [x] cuBLAS comparison (`cublasGemmEx`, FP16 in/out, FP32 compute) — in both WMMA result files: ~24,400 GFLOPS at 2048 and ~26,700 at 4096. The blocked WMMA kernel reaches 50% of cuBLAS at 1024, 41% at 2048, 30% at 4096
- [x] Small wins on the blocked WMMA kernel: vectorized tile loads (8 halves per load), `beta == 0` no longer reads `C`, any M/N/K — [`src/gemm_wmma_vec_fp16.cu`](src/gemm_wmma_vec_fp16.cu), [`results/wmma_vec_fp16_comparison.csv`](results/wmma_vec_fp16_comparison.csv): 1.2-1.6x over the blocked kernel (~9,200 GFLOPS at 1024, ~13,000 at 2048, ~10,500 at 4096), which is 60%, 54% and 41% of cuBLAS. Bit-identical to the blocked kernel, and to cuBLAS at 1000/1024/2048/4096. Sizes 1000 and 1001 run correctly; 1001 takes the scalar-load path because its row stride isn't 16-byte aligned. NaN-filled `C` with `beta == 0` gives 0 NaN out. Caveat: the blocked kernel's 2048 number moved between sessions (~10,100 GFLOPS in its own benchmark, ~8,000 here), so compare kernels within one results file, not across files
- [x] Where the final WMMA kernel trails cuBLAS, without hardware counters — [`profiling/ablate_wmma_vec.cu`](profiling/ablate_wmma_vec.cu) (removes one part of the kernel at a time), [`profiling/ablate_wmma_vec.txt`](profiling/ablate_wmma_vec.txt), [`profiling/nsys_kernel_launch_shapes.csv`](profiling/nsys_kernel_launch_shapes.csv) (Nsight Systems trace). At 2048 the `mma_sync` calls alone take ~50% of the kernel's time and that 50% is already about cuBLAS's whole runtime; the rest is shared-memory fragment loads (~30%) and global tile loads (~20%). At 4096 the global tile loads alone take ~77% of the full time and more than cuBLAS's whole runtime. cuBLAS runs `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_nn_align8`: a 256x128 tile per block, 8 warps, 228 registers per thread, 3 staged k-slices in 73,728 bytes of shared memory, against our 64x64 tile, 4 warps, 93 registers, 1 slice
- [ ] Nsight Compute profile of the final WMMA kernel — blocked: GPU performance counters are admin-only on this machine (`RmProfilingAdminOnly: 1`), needs `sudo ncu`
- [ ] Connect to `../conv/` (1x1 conv as GEMM)

## Layout

```
docs/          design docs, written before the code they describe
src/           .cu / .cuh kernel sources
tests/         correctness checks against a NumPy/PyTorch reference
benchmarks/    sweep scripts, write to results/
profiling/     Nsight Systems/Compute output
results/       benchmark.csv, plots — generated, not hand-written
```

## Hardware (same machine as the rest of this lab)

NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9), 6141 MiB VRAM,
driver 580.173.02, CUDA 12.0 (`nvcc`).
