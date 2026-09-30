# AI Inference Performance Lab — ResNet50

A reproducible investigation into how ResNet50 inference actually executes on a GPU —
from CUDA kernels up through TensorRT, precision, and fusion. One model, one GPU,
throughout. Started as a TensorRT FP32/FP16/INT8 precision experiment; growing toward
CUDA kernel work (GEMM, Conv2D) underneath it as the same model, same measurements.

## Research questions

| # | Question | Where | Status |
|---|---|---|---|
| 01 | How does GEMM execute on a GPU? | [`gemm/`](gemm/) | in progress — mental model + naive-kernel spec done, kernel not yet written |
| 02 | How does a ResNet convolution map to GEMM? | [`conv/`](conv/) | not started |
| 03 | What does FP32→FP16→INT8 cost and save? | [`resnet/`](resnet/), this file below | done ([video 1](videos/01_fp32_fp16_int8/)) |
| 04 | What does TensorRT fuse, and is there more to fuse? | [`profiling/`](profiling/) | done |
| 05 | Can quantizing only the safe layers keep INT8's speed? | [`profiling/INT8_SENSITIVITY_SWEEP.md`](profiling/INT8_SENSITIVITY_SWEEP.md) | done ([video 2](videos/02_mixed_precision/)) |
| 06 | What actually dominates ResNet inference at batch=1? | [`profiling/PROFILE_RESULTS.md`](profiling/PROFILE_RESULTS.md) | done |
| 07 | How does any of this change at batch>1? | — | open |

## Repository map

```
resnet/           completed engines: fp32/, fp16/, int8/, selective_int8/ (questions 03, 05, 06)
gemm/             CUDA GEMM kernel lab (question 01) -- not started
conv/             CUDA Conv2D kernel lab (question 02) -- not started, depends on gemm/
models/           ResNet50 weights + ONNX export
calibration/      INT8 calibration cache + images
scripts/          build / benchmark / calibrate / profile / sweep -- one script per step
benchmarks/       raw measured numbers (csv/json)
profiling/        per-layer timing, Nsight traces, the sensitivity-sweep + fusion writeups
results/          the narrative writeup of the fp32/fp16/int8 results below
videos/           two-part video documentary built from this lab:
                    01_fp32_fp16_int8/  -- this precision module
                    02_mixed_precision/ -- the sensitivity-sweep/fusion follow-up
```

Everything below this point documents the **FP32/FP16/INT8 module** (question 03 above,
the first and so far most complete piece of the lab).

## Question

How much faster, smaller, and cheaper can the same model get by moving from FP32 → FP16 → INT8 in TensorRT — and what does it cost in accuracy?

Same model. Same GPU. Same input. Only the precision changes.

## Hardware

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9, 6141 MiB / 5770.8 MiB usable)
- Driver: 580.173.02
- CUDA: 13.0

## Software

- TensorRT: 10.14.1.48.post1 (Python API, `pip install tensorrt`)
- PyTorch / torchvision: used only for ONNX export and as the FP32 accuracy reference
- Calibration dataset: [Imagenette2-160](https://github.com/fastai/imagenette) (fast.ai), 500 real JPEGs, 50 per class across 10 classes

## Model

ResNet50 (`torchvision.models.resnet50`), exported to ONNX with a dynamic batch dimension, input name `input`, output name `output`, input shape `3x224x224`. No normalization layer is baked into the graph — callers normalize inputs themselves (ImageNet mean/std).

## Methodology

1. **Export** — PyTorch ResNet50 → ONNX (`scripts/export.py`).
2. **FP32 / FP16** — built directly from ONNX via the TensorRT Python Builder API, batch=1 fixed optimization profile (`scripts/build_and_bench.py`).
3. **INT8** — built via the same Builder API with a custom `trt.IInt8EntropyCalibrator2` (`scripts/calibrate_int8.py`), calibrated on 500 real preprocessed images (resize shorter side to 256, center-crop 224, normalize, CHW). A custom calibrator was necessary because `trtexec` alone can't calibrate against real images — it needs either a pre-existing cache or a Python/C++ calibrator implementation.
4. **Benchmark** — GPU-side latency timed with CUDA events (20 warmup + 200 timed iterations), batch=1, all three engines run through the same TensorRT Python API version to avoid engine-serialization version skew (see note below).
5. **Accuracy** — each engine's output compared against the PyTorch FP32 reference on real images, using top-1 argmax agreement and raw output diff (`scripts/verify_accuracy_venv.py`).
6. **Extra metrics** — engine build time, engine file size, and GPU memory footprint during inference, measured directly rather than estimated (`scripts/measure_full_metrics.py`).

## Results

| | FP32 | FP16 | INT8 |
|---|---:|---:|---:|
| Latency (ms) | 1.808 | 0.645 | 0.347 |
| Throughput (qps) | 553.24 | 1551.03 | 2880.55 |
| GPU memory (MB) | 140.0 | 72.0 | 52.0 |
| Engine size (MB) | 107.94 | 49.14 | 25.26 |
| Engine build time (s) | 18.6 | 40.4 | 64.8* |
| Top-1 match vs. PyTorch | 1.0 | 1.0 | 0.9375 |

\* reuses the existing calibration cache; a cold build with a full calibration pass over
500 images measured 85.27s, not comparable 1:1 to fp32/fp16 build time.

Latency/throughput use CUDA-graph capture/replay + `BuilderFlag.DIRECT_IO` (see
**Optimization** below); pre-optimization numbers were 1.847 / 0.736 / 0.397 ms.

Full breakdown, bar charts, and the accuracy caveat: [`results/RESULTS.md`](results/RESULTS.md). Raw per-precision data: [`benchmarks/fp32.csv`](benchmarks/fp32.csv), [`fp16.csv`](benchmarks/fp16.csv), [`int8.csv`](benchmarks/int8.csv).

## Why did performance change?

FP16 gives ~2.8x lower latency than FP32, and roughly halves engine size and memory — consistent with half the bits per value and TensorRT selecting tensor-core FP16 kernels. INT8 gives another 1.86x on top of FP16, not a full 2x — partly because of fixed per-kernel-launch CPU overhead (mitigated below by CUDA graphs) and partly because at batch=1 many per-layer problem sizes are too small to reach peak INT8 tensor-core throughput. See profiling and optimization sections below.

## Profiling

Done. Per-layer timing (TensorRT profiler) and GPU kernel traces (Nsight Systems) on all three
engines show the model is compute-bound at every precision (95%+ of GPU time in conv/GEMM kernels),
and INT8 is genuinely running INT8 tensor-core kernels almost everywhere. FP32→FP16 gets close to
the full 2x; FP16→INT8 only got ~1.7x pre-optimization because at batch=1 many per-layer problem
sizes are too small to reach peak INT8 tensor-core throughput, and because fixed per-kernel-launch
CPU overhead doesn't shrink with bit width. See [`profiling/PROFILE_RESULTS.md`](profiling/PROFILE_RESULTS.md).

## Follow-up: does per-block INT8 sensitivity analysis buy anything?

Done. Since full INT8 costs real accuracy (93.75% top-1) and FP16→INT8's speedup
already shrinks at batch=1, tested whether quantizing only the layers that don't
hurt accuracy (found via a per-block sensitivity sweep, comparing each candidate
block's INT8 output back to an all-FP16 baseline) could keep most of INT8's speed
while paying less of its accuracy cost. First answer: small selective-INT8
engines (9-16 of 53 conv layers, picked by accuracy sensitivity alone) get the
accuracy part mostly right — but no speed: TensorRT inserts extra reformat
layers at each FP16↔INT8 boundary (1 in FP16, 10 in a 5-block scattered
selective engine) and their cost cancels the INT8 compute saving. The fix:
stop scattering INT8 in small islands and instead keep only the two most
sensitive units (stem + first block) pinned to FP16, forcing everything else
to INT8 — that engine needs just 3 reformat layers and is a real, reproducible
1.58x faster than FP16, at an accuracy cost between the small engines and full
INT8. Second finding along the way: the small scattered/contiguous engines
are **bimodal across identical rebuilds** (TensorRT lands on one of two fixed
tactics with different accuracy each build) while the head-FP16 engine
reproduced identically every time — fewer INT8 islands turned out to mean
more reproducible, not just faster. Full writeup:
[`profiling/INT8_SENSITIVITY_SWEEP.md`](profiling/INT8_SENSITIVITY_SWEEP.md).

## Optimization

Done. Two optimizations tested, both verified bit-identical to the unoptimized output:

- **CUDA Graph capture/replay** — captures the whole per-layer kernel-launch sequence
  once and replays it as a single graph launch per inference, removing fixed CPU
  launch overhead between kernels. Measured: fp32 +3-7%, fp16 +11-14%, int8 +19-30%
  (mean/min, repeated builds) — the effect grows as precision drops, confirming
  launch overhead (not tensor-core throughput) was the limiter that shrunk at batch=1.
- **`BuilderFlag.DIRECT_IO`** — removes a reformat/copy layer at the input boundary.
  A single-run A/B test initially suggested a large (~9-13%) win; repeated builds
  showed that was mostly a cold-start artifact — the real effect is ~2-4% for int8 and
  not measurably different from zero for fp16/fp32, smaller than TensorRT's own
  build-to-build tactic-selection noise. Kept because it's free, documented honestly
  as a minor win rather than the originally-measured large one.

Combined: ~2%/~12%/~13% faster than the original fp32/fp16/int8 baseline, and the
FP16→INT8 ratio improved from 1.7x to 1.86x. Full writeup, including the corrected
DIRECT_IO investigation: [`results/RESULTS.md`](results/RESULTS.md#optimizations-applied).

## Reproduce

```bash
cd scripts
python export.py                 # PyTorch -> ../models/resnet50.onnx
python build_and_bench.py        # builds fp32/fp16, benchmarks all three engines
python calibrate_int8.py         # builds ../resnet/int8/resnet50_int8.engine from ../calibration/images
python verify_accuracy_venv.py   # accuracy check vs PyTorch reference, 32 real images
python measure_full_metrics.py   # engine build time, size, GPU memory footprint
```

All scripts default to reading/writing the sibling `models/`, `resnet/fp32/`, `resnet/fp16/`, `resnet/int8/`, `calibration/`, and `benchmarks/` directories — run them from `scripts/` with no extra arguments, or pass flags to override any path.

Requires a working NVIDIA driver + `nvidia-smi`, and `pip install tensorrt torch torchvision pillow numpy cuda-python` in your environment.

## Lessons learned

- **Real images matter for INT8 calibration.** Calibrating on random noise or personal photos would either produce meaningless quantization ranges or non-reproducible results — Imagenette gives real ImageNet-like statistics in a small (~5MB copied here), redistributable dataset.
- **TensorRT engines are version-locked to the exact build that serialized them.** An engine built with one Python `tensorrt` package version won't deserialize with a different-versioned `trtexec` or runtime (`Serialization assertion ... Version tag does not match`). This bit us mid-project when a system driver upgrade also bumped the system TensorRT install out from under previously-built engines — the fix was to build and benchmark everything through one consistent Python API/version rather than mixing `trtexec` (system) and the Python API (venv).
- **INT8 accuracy checks need held-out data to mean anything.** The 93.75% top-1 number here is a real, honest measurement, but it reused calibration images as "test" images — a proper eval needs a disjoint split, which is flagged as follow-up work, not silently glossed over.
- **A single random-noise sample is a bad accuracy smoke test for INT8 specifically** — quantization ranges are calibrated on real image statistics, so noise is exactly the out-of-distribution input where argmax flips happen regardless of engine quality, and n=1 has zero statistical power. Switching to a batch of real images was necessary to get a meaningful number.

## Status

FP32, FP16, INT8 builds, benchmarking, accuracy checks, profiling, optimization (CUDA graphs, DIRECT_IO), and per-block INT8 sensitivity sweep + selective-quantization follow-up: done. Pipeline diagram: open, tracked in `diagrams/README.md`. Open follow-up: repeat the selective-INT8 latency test at batch>1, where per-layer problem sizes are large enough that partial quantization might actually beat FP16 (see `profiling/INT8_SENSITIVITY_SWEEP.md`); repeat the INT8 accuracy check against a held-out split disjoint from calibration images.
