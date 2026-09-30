# GEMM — not started

Planned first CUDA-kernel module of this lab: naive GEMM → tiled → shared memory →
register blocking, benchmarked against cuBLAS, then connected to where GEMM actually
shows up inside `resnet/` (the 1x1 convolutions are literally a GEMM; the 3x3 ones map
to it via implicit GEMM).

Nothing here yet. See the repo root [`README.md`](../README.md) for how this fits into
the overall plan and what order things are being tackled in.
