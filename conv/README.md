# Conv2D — not started

Planned second CUDA-kernel module: take one real convolution out of `resnet/` (the
per-layer profiles in `profiling/` already identify the heaviest ones — `Conv_846` was
the single largest layer in every FP32/FP16/INT8 profile) and go from PyTorch layer to
tensor shapes to a naive CUDA kernel to a tiled/shared-memory kernel, benchmarked
against cuDNN and against the TensorRT engines already built in `resnet/`.

Depends on [`../gemm/`](../gemm/) for the 1x1-convolution-as-GEMM connection. Nothing
built yet. See the repo root [`README.md`](../README.md) for how this fits in.
