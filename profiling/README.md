# Profiling

Done. Used TensorRT's own per-layer profiler (`trt.IProfiler` + engine inspector, via
`scripts/profile_layers.py`) and Nsight Systems CUDA kernel traces (`scripts/nsys_run_engine.py`
under `nsys profile`) on the fp32/fp16/int8 engines to find out *why* the FP32→FP16→INT8 speedups
look the way they do.

Full writeup, tables, and the reproduce commands: [`PROFILE_RESULTS.md`](PROFILE_RESULTS.md).

Short version: this model is compute-bound (95%+ of GPU time) at every precision, and INT8 is
genuinely running INT8 tensor-core kernels almost everywhere (only the final FC layer stays FP32).
FP32→FP16 gets close to the full 2x; FP16→INT8 only gets ~1.7x because at batch=1 many of ResNet50's
per-layer conv/GEMM problem sizes are too small to reach peak INT8 tensor-core throughput — the extra
2x math throughput isn't fully convertible to wall-clock time when fixed per-kernel-launch overhead
doesn't shrink along with the bit width. Not caused by reformat overhead or precision fallback, which
are both small (1-4% of GPU time).

Raw data: `layer_profile_{fp32,fp16,int8}.json` (per-layer time + precision), `nsys/*.nsys-rep`
(Nsight Systems traces, open with `nsys-ui`), `nsys/*_cuda_gpu_kern_sum.csv` (kernel time summaries).

**Note on staleness**: this data was captured on the engines *before* the optimizations
described in [`../results/RESULTS.md`](../results/RESULTS.md#optimizations-applied)
(CUDA graph capture/replay, `BuilderFlag.DIRECT_IO`) were applied — the reformat-layer
and launch-overhead findings above are exactly what motivated those two changes, and
the fixed-launch-overhead diagnosis was confirmed directly: CUDA graphs (which remove
that overhead) closed part of the FP16→INT8 gap, from 1.7x to 1.86x. The compute-bound
/ genuine-INT8-kernel conclusions still hold; the per-layer JSON/nsys traces here reflect
the pre-optimization engines and haven't been re-captured against the current ones.

## Follow-up: per-block INT8 sensitivity sweep

Done. Tested whether quantizing only the layers with the least accuracy impact
(rather than the whole network) could keep most of INT8's speed while paying
less of its accuracy cost. Small scattered/contiguous selective-INT8 engines
(9-16 of 53 conv layers) got the accuracy part mostly right — but no speed:
at batch=1 those engines are statistically indistinguishable from full FP16,
because the extra reformat layers TensorRT inserts at each FP16↔INT8 boundary
cost about as much as the INT8 compute saves. The fix was to stop scattering
INT8 in small islands and instead keep only the two most sensitive units (the
stem conv and the first block) pinned to FP16, forcing everything else — 48
of 53 conv layers — to INT8. That engine needs only 3 reformat layers instead
of 5-10, and is a real, **reproducible** 1.58x faster than FP16 (0.381ms vs.
~0.603ms), at an accuracy cost between the small selective engines and full
INT8 (93.75% top-1). Separate finding along the way: the scattered/contiguous
engines turned out to be **bimodal across identical rebuilds** — TensorRT
picks one of two fixed tactics each build, with meaningfully different
accuracy — while the head-FP16 engine reproduced identically across every
rebuild tried. See [`INT8_SENSITIVITY_SWEEP.md`](INT8_SENSITIVITY_SWEEP.md).
