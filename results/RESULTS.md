# Results

ResNet50, batch=1, input 3x224x224, NVIDIA GeForce RTX 4050 Laptop GPU, driver 580.173.02,
CUDA 13.0, TensorRT 10.14.1.48.post1. Full field-by-field data: `../benchmarks/fp32.csv`,
`fp16.csv`, `int8.csv`.

## Summary table

|            |     FP32 |     FP16 |     INT8 |
| ---------- | -------: | -------: | -------: |
| Latency (ms) | 1.808 | 0.645 | 0.347 |
| Throughput (qps) | 553.24 | 1551.03 | 2880.55 |
| GPU memory (MB) | 140.0 | 72.0 | 52.0 |
| Engine size (MB) | 107.94 | 49.14 | 25.26 |
| Engine build time (s) | 18.6 | 40.4 | 64.8* |
| Top-1 match vs. PyTorch | 1.0 | 1.0 | 0.9375 |

\* INT8 build time here reuses the existing calibration cache; a cold build with a full
calibration pass over 500 real images measured 85.27s. Not directly comparable to the
fp32/fp16 build times, which never calibrate.

Latency/throughput above use CUDA-graph capture/replay plus `BuilderFlag.DIRECT_IO` —
see **Optimizations applied** below. The pre-optimization numbers (eager
`execute_async_v3` per call, no DIRECT_IO) were 1.847 / 0.736 / 0.397 ms.

## Latency (lower is better)

```
FP32  ████████████████████████████████████████  1.808 ms
FP16  ██████████████                            0.645 ms
INT8  ████████                                  0.347 ms
```

## Throughput (higher is better)

```
FP32  ████████                                  553 qps
FP16  ██████████████████████                    1551 qps
INT8  ████████████████████████████████████████  2881 qps
```

## GPU memory footprint (lower is better)

```
FP32  ████████████████████████████████████████   140.0 MB
FP16  █████████████████████                       72.0 MB
INT8  ███████████████                             52.0 MB
```

## Optimizations applied

Two optimizations were tested on top of the base FP32/FP16/INT8 builds, both verified
to produce bit-identical output (max abs diff = 0.0 vs. the unoptimized path, fixed-seed
input, all three precisions) before being folded into `build_and_bench.py` /
`calibrate_int8.py`:

- **CUDA Graph capture/replay** — the `execute_async_v3` kernel-launch sequence is
  captured once into a CUDA graph and replayed per inference, instead of re-issuing the
  ~100+ per-layer kernel launches from Python/CPU on every call. This directly targets
  the fixed per-launch CPU overhead that profiling identified as the reason FP16→INT8
  wasn't a full 2x at batch=1 (small kernels don't shrink the per-launch overhead).
  Measured on this model (batch=1, repeated builds): **fp32 +3-7%, fp16 +11-14%, int8
  +19-30%** (mean/min across trials) — the gain grows as precision drops because
  per-kernel GPU time shrinks while the fixed launch overhead doesn't. This is the
  larger and more reliable of the two optimizations.
- **`BuilderFlag.DIRECT_IO`** — forces the engine's I/O tensors to use the format the
  first/last layer already wants, removing a reformat/copy layer at the input boundary.
  Initially measured as a large win (~9-13%) in a single A/B run, but repeated builds
  (3x each config) showed that number was mostly a cold-start artifact: fp16 shows *no*
  measurable effect once cold-start is excluded (min latency ties at 0.621ms either way),
  and int8 shows a real but modest ~2-4% gain, smaller than TensorRT's own
  build-to-build tactic-selection noise (~±8% across identical rebuilds). Kept because
  it's free and never measured worse — not documented as a major win. See
  [`../profiling/PROFILE_RESULTS.md`](../profiling/PROFILE_RESULTS.md) for the full
  investigation and the reformat-layer analysis that motivated trying it.

Combined effect vs. the original (pre-optimization) baseline: fp32 ~2% faster,
fp16 ~12% faster, int8 ~13% faster. The FP16→INT8 step also improved, from 1.7x to
1.86x, moving closer to the naive 2x-from-halving-bits expectation — consistent with
part of that gap being fixed launch overhead rather than an intrinsic tensor-core
throughput limit.

## Why the numbers look like this

- FP16 gives roughly 2.8x lower latency than FP32 and roughly halves engine size and memory footprint — consistent with using half the bits per weight/activation and TensorRT selecting FP16 tensor core kernels.
- INT8 gives another 1.86x on top of FP16 (not a full 2x) and roughly halves engine size again. Profiling ([`../profiling/PROFILE_RESULTS.md`](../profiling/PROFILE_RESULTS.md)) confirms this model is compute-bound at every precision (95%+ of GPU time in conv/GEMM kernels) and that INT8 genuinely runs INT8 tensor-core kernels almost everywhere — part of the shrinking gain is fixed per-kernel-launch CPU overhead (see **Optimizations applied** above, which closes some of it), and the rest is that at batch=1 many per-layer problem sizes are too small to reach peak INT8 tensor-core throughput over FP16.
- INT8 accuracy: 93.75% top-1 agreement with the PyTorch reference (30/32), vs 100% for both FP32 and FP16. This is the expected quantization cost, and it's a small sample — see the caveat below.

## Accuracy caveat

The accuracy check (`../benchmarks/accuracy_results.json`) ran on 32 real images sampled
from the *same* 500-image calibration pool used to calibrate the INT8 engine — there was
no held-out split. This is a legitimate directional sanity check (real images, real
preprocessing, real classifier), but it is not a rigorous held-out evaluation. A proper
follow-up would calibrate on one Imagenette split and evaluate top-1 match / true-label
accuracy on a disjoint split.

## Open (not run yet)

- **Pipeline diagram** — ONNX → TensorRT export/build diagram. See `../diagrams/README.md`.
- **Batched INT8 vs FP16 comparison** — profiling (see `../profiling/PROFILE_RESULTS.md`) suggests
  the FP16→INT8 gap at batch=1 is capped by small per-layer kernel sizes not reaching peak INT8
  tensor-core throughput; a batch>1 run would test whether a larger batch closes that gap.

## Reproduce

```bash
cd scripts
python export.py
python build_and_bench.py
python calibrate_int8.py
python verify_accuracy_venv.py
python measure_full_metrics.py
```

(all scripts default to the sibling `models/`, `fp32/`, `fp16/`, `int8/`, `calibration/`, `benchmarks/` directories)

Note: `export_onnx_to_trt.py` in `scripts/` is the original `trtexec`-subprocess-based
builder and is kept for reference, but it is version-fragile — if the system `trtexec`
and the TensorRT Python package (used by the other scripts here) drift to different
versions, engines built by one can't be loaded by the other (`Serialization assertion
... Version tag does not match`). `build_and_bench.py` avoids this by building and
benchmarking entirely through the same Python API/version.
