# Conv2D

Second CUDA-kernel module: take real convolutions out of `resnet/` and run them as
GEMMs, starting from the heaviest layers in the per-layer profiles in `../profiling/`.
Builds on [`../gemm/`](../gemm/); see the repo root [`README.md`](../README.md).

## Status

- [x] `Conv_846` (slowest layer in every FP32/FP16/INT8 profile): 1x1 conv, stride 2, 1024x14x14 → 2048x7x7, which is a GEMM with M=2048, N=49, K=1024 after packing the stride-2 pixels — [`src/conv846_gemm.cu`](src/conv846_gemm.cu) (reuses the step-4 WMMA kernel from `../gemm/` unchanged), [`benchmarks/torch_conv846.py`](benchmarks/torch_conv846.py) (cuDNN baseline), [`results/conv846_gemm.csv`](results/conv846_gemm.csv). FP16, batch 1, no bias, random data at the real shape. Gather + WMMA GEMM: 0.070 ms; gather + cuBLAS: 0.037 ms; cuDNN `conv2d`: 0.027-0.035 ms; TensorRT FP16 layer (from the profile, with bias): 0.043 ms. The gather is 0.003 ms; the gap is the GEMM. Both GEMM outputs are within FP16 rounding of an FP32 CPU conv reference
- [x] Verified on real ResNet50 tensors — [`benchmarks/export_conv846_real.py`](benchmarks/export_conv846_real.py) runs a real calibration image through the ONNX model in onnxruntime and dumps `Conv_846`'s actual input activation, weights, bias and ONNX's own FP32 output to `data/` (gitignored, regenerate with the script); `./conv846_gemm ../data` uses them. Output + bias matches ONNX within 0.0063 at values up to 14.8 (under one FP16 step at that magnitude) — [`results/conv846_gemm_real.csv`](results/conv846_gemm_real.csv)
- [x] Fit the GEMM to the skinny shape. Padding N 49 → 64 (zero columns, so B gets vectorized loads): 0.070 → 0.054 ms. Probe: 2x the columns (32 → 64 blocks) costs only 1.43x the time, so the GPU is underfilled at 32 blocks. Split-K (K sliced across `blockIdx.z`, FP32 partials, then one reduce kernel that adds bias and writes the final 2048x7x7 NCHW output): best at 4 splits / 128 blocks, **0.039 ms for gather + GEMM + reduce + bias**, against 0.037 ms for gather + cuBLAS (no bias) and 0.043 ms for the TensorRT FP16 layer in the profile. 8 splits: 0.041 ms; 16: 0.050; 32: 0.070. Same numbers on random and real data, three runs within 1%
- [x] Fold the stride-2 gather into the split-K kernel's B tile load (`gemm_wmma_splitk<true>`, reads the raw 1024x14x14 input, no gather kernel, no packed buffer): correct (same 0.0062 max error vs ONNX), but **slower**: 0.042 ms against 0.039 ms with the separate gather at the best 4 splits, and slower at every split count except 32. My reading, not profiled: the folded load is 16 scalar stride-2 loads per thread per k-step instead of 2 16-byte loads, and all 32 row-blocks redo that gather for the same B tile, while the separate gather kernel does it once for 0.003 ms. Kept the separate gather as the default
- [ ] 3x3 convs `Conv_842`/`Conv_850`/`Conv_856` (512x7x7, implicit GEMM M=512, N=49, K=4608)
- [ ] Bias + ReLU in the epilogue

## Hardware

Same machine as the rest of this lab: RTX 4050 Laptop GPU (sm_89), CUDA 12.0 `nvcc`,
cuDNN 9.2 through PyTorch 2.12 (no cuDNN C headers installed).
