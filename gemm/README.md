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
- [ ] Tiled kernel (shared memory)
- [ ] Further optimization passes (register blocking, vectorized loads, FP16)
- [ ] cuBLAS comparison
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
