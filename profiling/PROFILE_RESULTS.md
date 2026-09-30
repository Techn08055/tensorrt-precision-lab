# Profiling Results

ResNet50, batch=1, input 3x224x224, NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9),
driver 580.173.02, CUDA 13.0, TensorRT 10.14.1.48.post1.

Two independent measurements, cross-checked against each other:

1. **TensorRT per-layer profiler** (`scripts/profile_layers.py`) — engines rebuilt from ONNX with
   `ProfilingVerbosity.DETAILED` and a `trt.IProfiler` callback attached, so every layer's GPU time
   and precision are visible. Attaching a profiler forces a CUDA sync between every layer, which adds
   overhead and breaks cross-layer overlap — so absolute totals here run higher than real inference
   latency, but the *relative* per-layer/per-category split is meaningful. Raw data:
   `layer_profile_fp32.json`, `layer_profile_fp16.json`, `layer_profile_int8.json`.
2. **Nsight Systems CUDA kernel trace** (`scripts/nsys_run_engine.py`, run under `nsys profile`) —
   traces the actual, already-benchmarked engines (`../resnet/fp32/`, `../resnet/fp16/`, `../resnet/int8/`) with no TensorRT
   instrumentation in the loop, so it reproduces true unblocked latency and gives ground-truth kernel
   names/durations. Raw data: `nsys/{fp32,fp16,int8}.nsys-rep`, `nsys/{fp32,fp16,int8}_cuda_gpu_kern_sum.csv`.

Why both: the TensorRT profiler names layers and reports their precision (so we can see *which* layer
is slow and *why*, e.g. "still running in FP32"); Nsight Systems confirms that number isn't an
artifact of profiler overhead and shows the literal kernel that ran.

## Per-iteration GPU time (Nsight Systems, 300 iterations, no profiler overhead)

| | FP32 | FP16 | INT8 |
|---|---:|---:|---:|
| GPU kernel time (ms) | 2.008 | 0.756 | 0.446 |
| vs. previous precision | — | 2.65x | 1.70x |
| vs. FP32 | 1.00x | 2.65x | 4.50x |

```
FP32  ████████████████████████████████████████  2.008 ms
FP16  ███████████████                            0.756 ms
INT8  ████████                                   0.446 ms
```

These track the end-to-end CUDA-event latency in [`../results/RESULTS.md`](../results/RESULTS.md)
(1.847 / 0.736 / 0.397 ms) closely — the small gap is host-side stream-sync/launch overhead that
Nsight Systems' pure kernel-time sum doesn't include.

**The key number: FP32→FP16 nearly gets the full 2x, but FP16→INT8 only gets 1.7x, not 2x.**
That's the answer to "why wasn't INT8 a uniform 4x" — see below for why.

## Where the time goes (Nsight Systems kernel categories, by name)

| | FP32 | FP16 | INT8 |
|---|---:|---:|---:|
| Conv/GEMM (compute-bound) | 97.2% | 97.7% | 95.0% |
| Reformat/copy (memory-bound) | 2.2% | 1.2% | 1.4% |
| Pooling (memory-bound) | 0.5% | 1.1% | 3.6% |

The TensorRT layer profiler's own category split agrees directionally (compute-bound 92–96% of
per-layer time in all three engines; see `time_by_category_pct` in each `layer_profile_*.json`).

**This model is compute-bound at every precision, overwhelmingly.** Reformatting/layout-conversion
overhead — the thing that bit the version-mismatch lesson in the main README — is real but small
(1–4% of GPU time) and isn't what's capping the INT8 speedup.

## Is INT8 actually running in INT8?

Checked two ways:

- **Kernel names** (Nsight Systems): of 17,920 kernel launches across 300 INT8 iterations, 17,280
  (96.4%) are genuine `i8i8`/`IMMA` integer-tensor-core kernels
  (e.g. `sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw...`). INT8 quantization is not
  silently falling back to FP16/FP32 kernels across the network.
- **Engine inspector** (TensorRT, DETAILED verbosity): of 58 layers in the INT8 engine, all but the
  final classifier layer run with `Int8` output tensors. Two nodes stay in `Float`:
  `node_linear + (Unnamed Layer* 130) [ElementWise]` (the final FC layer's bias-add) and the reshape
  feeding it. This is TensorRT's own choice, not a calibration failure — the last classifier layer is
  a common, deliberate place to keep full precision because quantizing it barely saves compute (it's
  tiny — a 2048x1000 matmul, run once per inference) but directly touches the logits used for argmax.

So the 93.75% top-1 accuracy number in [`../results/RESULTS.md`](../results/RESULTS.md) isn't being
paid for by a fallback layer eating the speedup — the fallback layer is cheap (0.03ms out of ~0.45ms
in the profiler run) and precision loss is concentrated in the quantized conv stack instead.

## Why the FP16→INT8 step is only 1.7x, not 2x

Given the above — 95%+ of GPU time is genuine compute-bound INT8 tensor-core convolution, and
reformat overhead is a rounding error — the shrinking marginal gain isn't caused by data-movement
overhead or precision fallback. It comes down to **kernel efficiency at batch=1**:

- Individual kernel durations in the INT8 trace average roughly 4,000–17,000 ns per launch, across
  hundreds of small conv layers per inference. At batch=1, most of ResNet50's per-layer GEMM/conv
  problem sizes are too small to fill an INT8 tensor-core tile the way they'd fill a larger batch.
- INT8's 2x throughput advantage over FP16 (on tensor cores rated for ~2x INT8-vs-FP16 dense
  throughput) is a *peak* throughput number. It assumes the kernel runs long enough, and the tile is
  full enough, that fixed per-launch overhead (grid/block setup, memory access latency before compute
  saturates) is amortized. Halving the bits per value doesn't halve that fixed overhead.
- FP32→FP16 got closer to the full 2x because FP32 conv kernels on this hardware are further from
  their overhead-bound floor to begin with — there's more headroom to cut.

In short: this model at batch=1 is compute-bound in aggregate, but many of its individual layers are
small enough to be *kernel-launch-and-occupancy-bound* rather than purely math-throughput-bound —
so the last precision cut captures only part of its theoretical ceiling. A batched (batch>1) INT8
benchmark would be the natural follow-up to test this: larger batches should push the FP16→INT8 ratio
closer to 2x by giving each kernel more work to amortize its overhead against.

## Top layers by time (TensorRT per-layer profiler)

FP32 and FP16 top layers are the same convolutions (as expected — same graph, same shapes), just at
roughly a quarter and a half of FP32's per-layer time respectively:

| Layer | FP32 (ms) | FP16 (ms) |
|---|---:|---:|
| `node_Conv_846` | 0.207 | 0.043 |
| `node_Conv_842 + node_relu_41` | 0.135 | 0.043 |
| `node_Conv_856 + node_relu_47` | 0.103 | 0.038 |
| `node_Conv_850 + node_relu_44` | 0.095 | 0.038 |

In the INT8 engine, the single largest layer by profiled time is not a convolution at all — it's the
**input reformat** (`Reformatting CopyNode for Input Tensor 0 to node_Conv_754...`), which converts
the FP32 NCHW input into the INT8 layout the rest of the engine expects. That's expected: a one-time,
fixed-size data-layout conversion doesn't shrink with quantization the way compute does, so as the
compute layers get faster it becomes proportionally more visible — even though in absolute terms
(~0.036 ms out of ~1.1 ms profiled total) it's still small.

## Reproduce

```bash
cd scripts
python profile_layers.py     # rebuilds fp32/fp16/int8 with DETAILED verbosity, per-layer timing + precision
nsys profile --trace=cuda --force-overwrite=true -o ../profiling/nsys/fp32 -- python nsys_run_engine.py --engine ../resnet/fp32/resnet50_fp32.engine --iters 300
nsys profile --trace=cuda --force-overwrite=true -o ../profiling/nsys/fp16 -- python nsys_run_engine.py --engine ../resnet/fp16/resnet50_fp16.engine --iters 300
nsys profile --trace=cuda --force-overwrite=true -o ../profiling/nsys/int8 -- python nsys_run_engine.py --engine ../resnet/int8/resnet50_int8.engine --iters 300
nsys stats --report cuda_gpu_kern_sum --format csv --output ../profiling/nsys ../profiling/nsys/fp32.nsys-rep
# (repeat nsys stats for fp16.nsys-rep and int8.nsys-rep)
```

`profile_layers.py` rebuilds engines rather than reusing the ones under `../resnet/fp32/`,
`../resnet/fp16/`, `../resnet/int8/`: those were built by a different Python `tensorrt` package version than the system
`trtexec`, and per the version-lock lesson in the main README, only the same TensorRT Python API that
built them can deserialize them. `nsys_run_engine.py` sidesteps that entirely by loading the original
benchmarked engines directly — Nsight Systems traces CUDA calls, not TensorRT internals, so it has no
engine-version dependency at all. INT8 rebuild reuses `../calibration/calibration.cache`, so it
doesn't recalibrate from images.

## Note: fixed a gap found along the way

`../models/resnet50.onnx` referenced external weights (`resnet50.onnx.data`) that were missing from
the repo, so the ONNX file alone couldn't be parsed. Re-ran `scripts/export.py` to regenerate both
files before any of the above would build. Separately, `parser.parse(f.read())` (used by
`build_and_bench.py`, `calibrate_int8.py`, `measure_full_metrics.py`) resolves the external-data path
relative to the process's working directory rather than the `.onnx` file's directory, so it fails
unless run with cwd set just right; `profile_layers.py` here uses `parser.parse_from_file(onnx_path)`
instead, which resolves external data relative to the ONNX file itself and works from `scripts/`
regardless of caller cwd. Not fixed in the older scripts since it wasn't broken for them until the
`.onnx.data` file went missing — flagged here as a latent fragility, not fixed proactively for those
three files since that wasn't in the scope of this profiling session.

## Status

Done: per-layer timing breakdown (both instrumented and ground-truth), compute-bound vs memory-bound
classification, INT8 precision-fallback check, and the explanation for the shrinking FP16→INT8 gain.
Open follow-up: repeat the INT8 vs FP16 comparison at batch>1 to test the kernel-occupancy explanation
above.

**Update — optimizations applied based on these findings**: the input-reformat layer noted above
motivated trying `BuilderFlag.DIRECT_IO`, and the launch-overhead explanation for the shrinking
FP16→INT8 gain motivated trying CUDA graph capture/replay. Both are now applied in the production
build/benchmark scripts; see [`../results/RESULTS.md`](../results/RESULTS.md#optimizations-applied)
for the measured effect of each (CUDA graphs: a real, sizeable win, largest for INT8, confirming the
launch-overhead diagnosis above; DIRECT_IO: initially overstated from a single noisy A/B run, real
effect on repeated measurement is small). The per-layer/nsys data in this document was captured
*before* those changes and has not been re-captured against the optimized engines — the diagnosis
still holds, but absolute per-layer numbers here are now historical, not current.
