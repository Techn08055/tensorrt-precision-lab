# Video Dossier — FP32 → FP16 → INT8: What Precision Actually Changes

## Format Directive (read first — binding on Director/Writer/Animator)

The creator explicitly rejected a Veritasium-style structure (cold open on a paradox,
non-linear jumps, mystery-reveal loops). Requested format instead:

> "Linear, going and explaining, adding curiosity while I build the code."

Concretely this means:
- **Progression is linear**: concept → math → code → run it → look at the real number →
  next concept. No flash-forward hooks, no "but that's not the whole story" cold opens.
- **Curiosity is seeded inline, not structural.** A curiosity beat is a sentence dropped
  right before the code that resolves it, not a teaser withheld for act three.
- **The code build IS the spine.** This project has a real, working repo
  (`tensorrt-precision-lab/`) with real ResNet50 engines, real benchmark numbers, and a
  real calibration cache already on disk — the video should narrate building/running it,
  not simulate a build in a sandbox. Screen capture of the actual scripts running is the
  primary visual, with diagrams layered in for the parts that happen inside the bit
  pattern / inside the GPU where there's nothing to point a camera at.
- Downstream phases (Director/Writer/Animator) should treat this dossier's "Experiment
  Plan" section as the literal code-build sequence for the video, in order.

---

## 1. Executive Summary & Verified Facts

### What this video is actually about
Not "TensorRT tutorial." The spine is one question, answered with a real, reproducible
experiment already sitting on disk in this repo: **the same ResNet50 model, unchanged,
run through TensorRT at FP32, FP16, and INT8 — what does each precision format actually
do to the numbers, and why does the resulting speed/size/accuracy tradeoff look the way
it does, instead of the "clean" 2x/4x someone would naively expect?**

### Verified facts — number formats (textbook, high confidence)

| | FP32 | FP16 | INT8 |
|---|---:|---:|---:|
| Total bits | 32 | 16 | 8 |
| Sign bits | 1 | 1 | (signed int, no separate sign field) |
| Exponent bits | 8 | 5 | — (no exponent; integer) |
| Mantissa/fraction bits | 23 | 10 | — |
| Exponent bias | 127 | 15 | — |
| Normal exponent range | −126 to +127 | −14 to +15 | — |
| Largest finite value | ≈3.4028235×10³⁸ | 65504 | 127 (signed) |
| Smallest positive normal | ≈1.17549435×10⁻³⁸ | ≈6.10×10⁻⁵ | — |
| Smallest positive subnormal | ≈1.4013×10⁻⁴⁵ (2⁻¹⁴⁹) | ≈5.96×10⁻⁸ | — |
| Needs a scale/zero-point to represent arbitrary reals? | No | No | **Yes** |

- Formula for a normal FP32/FP16 value: `x = (-1)^s × (1+f) × 2^(E−bias)`, where `f` is
  built from the stored fraction bits (`f = Σ bᵢ·2⁻ⁱ`) and the leading `1.` is implicit
  (not stored) for normal numbers — this is why 23 stored fraction bits buy 24 bits of
  effective precision.
- `E=0` and `E=all-ones` are both reserved (subnormals/zero, and infinity/NaN
  respectively) — this is *why* the normal exponent range is `[1, 2^bits−2]` shifted by
  the bias, not the naive `[0, 2^bits−1]`.
- FP32→FP16 is a **format conversion** (no calibration, no scale, no zero-point — the
  16-bit layout itself defines the number). FP32→INT8 is **quantization**: INT8 has no
  exponent, so an external scale (and optionally a zero-point) must be supplied to map a
  continuous range onto 256 integers.
- Symmetric quantization: `s = max(|x|)/127`, `q = round(x/s)`, dequant `x̂ = q·s`. Zero
  maps to zero exactly.
- Asymmetric quantization: `s = (max−min)/255`, `q = round(x/s) + z` where
  `z = round(qmin − min/s)`. Needed when the real range isn't centered on zero (e.g. ReLU
  outputs, which are ≥0) — symmetric quantization would burn half the INT8 range on
  values that never occur.
- Per-tensor = one scale for a whole tensor; per-channel = one scale per output channel.
  Per-channel costs a handful of extra floats and can meaningfully cut error when
  different channels have very different magnitudes (common for conv weights).
- **Documented NVIDIA behavior** (TensorRT developer guide; not independently verified
  against this repo's own engine-inspector dump — flagged as an open code-audit item
  below): TensorRT applies **per-channel** scaling automatically to convolution /
  deconvolution / fully-connected / MatMul **weights** when INT8 is enabled, regardless
  of calibrator, while **activations** get **per-tensor** scaling from whatever the
  calibrator (or explicit Q/DQ nodes) supplies. If true of this repo's engine, it means
  the symmetric/asymmetric and per-tensor/per-channel axes from the source notes aren't
  hypothetical for this project — weights and activations in the *same* INT8 ResNet50
  engine are being quantized two different ways simultaneously.
- `IInt8EntropyCalibrator2` (used in `scripts/calibrate_int8.py`) picks each tensor's
  clipping threshold by minimizing KL-divergence between the original float histogram and
  the quantized one (Szymon Migacz, NVIDIA, "8-bit Inference with TensorRT," 2017) — it is
  *not* simple min/max. This directly explains why calibration can choose a threshold
  smaller than the tensor's true max: entropy calibration deliberately clips outliers if
  doing so preserves more resolution for the bulk of the distribution.

### Verified facts — this repo's actual experiment (ResNet50, batch=1, RTX 4050 Laptop, TensorRT 10.14.1)

| | FP32 | FP16 | INT8 |
|---|---:|---:|---:|
| Latency (ms) | 1.808 | 0.645 | 0.347 |
| Throughput (qps) | 553.24 | 1551.03 | 2880.55 |
| GPU memory (MB) | 140.0 | 72.0 | 52.0 |
| Engine size (MB) | 107.94 | 49.14 | 25.26 |
| Engine build time (s) | 18.6 | 40.4 | 64.8 (cache reuse) / 85.27 (cold calibration) |
| Top-1 match vs. PyTorch FP32 reference | 1.0 | 1.0 | 0.9375 |
| Max abs output diff vs. PyTorch | 0.0279 | 0.0867 | 2.396 |

- Source: `results/RESULTS.md`, `benchmarks/*.json`, `benchmarks/*.csv` — all numbers
  above are measured, not estimated, with a documented methodology (CUDA events, 20
  warmup + 200 timed iters, batch=1, all three engines built/run through one consistent
  TensorRT Python API version to dodge engine-serialization version skew).
- **Ground-truth GPU kernel time** (Nsight Systems, no profiler instrumentation
  overhead, 300 iters): FP32 2.008 ms, FP16 0.756 ms (2.65x), INT8 0.446 ms (1.70x on top
  of FP16, 4.50x total vs FP32). This is the real, unblocked number — separate from the
  TensorRT layer-profiler numbers, which run higher because attaching a profiler forces a
  CUDA sync between every layer.
- **This model is compute-bound at every single precision**: 95–97.7% of GPU time is in
  conv/GEMM kernels at FP32, FP16, *and* INT8. Reformat/copy overhead — the thing you'd
  guess is "the INT8 tax" — is only 1.2–2.2% of GPU time across all three.
- **INT8 is genuinely running INT8 tensor-core kernels, not silently falling back**: of
  17,920 kernel launches across 300 INT8 iterations, 17,280 (96.4%) are literal
  `i8i8`/`IMMA` integer-tensor-core kernels (kernel name checked directly in the Nsight
  trace, e.g. `sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw...`).
- **Exactly 2 of 58 layers in the INT8 engine stay in Float**: the final FC layer's
  bias-add (`node_linear + (Unnamed Layer* 130) [ElementWise]`, 0.034 ms) and the reshape
  feeding it. This is TensorRT's own deliberate choice (keep the tiny, logit-producing
  last layer in full precision), not a calibration failure — confirmed via
  `engine.create_engine_inspector()` output at `ProfilingVerbosity.DETAILED`.
- **CUDA graph capture/replay measured effect**: fp32 +3–7%, fp16 +11–14%, int8 +19–30%
  — the win *grows* as precision drops, because per-kernel GPU time shrinks with bit
  width but the fixed per-launch CPU overhead between kernels does not.
- **`BuilderFlag.DIRECT_IO` — a documented correction of the project's own earlier
  finding**: a single A/B run first measured a ~9–13% win; repeating the build 3x per
  config showed that was mostly a cold-start artifact — FP16 shows *zero* measurable
  effect once cold-start is excluded, INT8 shows a real but modest 2–4% gain, smaller than
  TensorRT's own build-to-build tactic-selection noise (~±8% across identical rebuilds).
  This is on record in the repo (`results/RESULTS.md`, `profiling/PROFILE_RESULTS.md`),
  not hidden — the project corrected its own initially-overstated claim before publishing.
- The calibration cache (`calibration/calibration.cache`) is a **plain-text file**: 127
  lines of `tensor_name: <8-hex-digit float32 bit pattern>`, one calibrated scale-defining
  value per tensor in the whole 58-layer graph. It is trivially human-readable and
  decodable with a five-line Python script (`int(hex, 16)` → reinterpret as IEEE-754
  bits) — this is the literal, on-disk evidence of "calibration produces a scale per
  tensor," not an abstraction.

## 2. The Curiosity Engine

Ranked by curiosity voltage (highest first). Each includes the resolving evidence
already sitting in this repo — no hypothetical needed.

1. **"INT8 should be 2x faster than FP16 — TensorRT tensor cores are literally rated for
   2x INT8-vs-FP16 throughput. It only got 1.7x. Where did the other 0.3x go?"** (Mystery,
   very high voltage — this is the natural "wait, that doesn't add up" moment right after
   revealing the headline numbers.) Resolution: not precision fallback (96.4% of kernels
   are genuine INT8), not reformat overhead (1.4% of GPU time) — it's kernel-launch
   overhead and occupancy. At batch=1, ResNet50's individual conv layers are often too
   small to fill an INT8 tensor-core tile fully; the fixed cost of launching a kernel
   doesn't shrink when the kernel's own math gets 2x cheaper, so the *ratio* of
   fixed-cost to compute-cost gets worse as compute shrinks. CUDA graphs (removing the
   host-side launch overhead specifically) recovered part of the gap: 1.7x → 1.86x.
2. **Misconception: "FP16 and INT8 are the same kind of thing, just smaller."** Almost
   every learner's first mental model. False: FP16 is still a floating-point format (its
   own bits fully define every value, no external metadata) — going FP32→FP16 needs zero
   calibration data. FP16→INT8 needs a calibration dataset, a chosen calibration
   algorithm (entropy/percentile/max), and per-tensor scale files, because INT8 has no
   exponent and cannot represent scale on its own. This is exactly why the repo has an
   `IInt8EntropyCalibrator2` class and 500 real JPEGs, but no equivalent machinery for the
   FP16 build — the code asymmetry between `build_and_bench.py` (FP32/FP16, no
   calibration) and `calibrate_int8.py` (INT8, full calibration loop) is direct, visual
   proof of the conceptual asymmetry.
3. **Paradox: the project's own "fix" data point contradicts its first measurement.**
   `BuilderFlag.DIRECT_IO` was first measured as a 9–13% win, then re-measured (more
   carefully) as ~2–4% for INT8 and statistically indistinguishable from zero for
   FP16/FP32. Two honestly-run experiments on the same flag gave different-looking
   answers. Great engineering-methodology beat: single-run A/B tests on GPU latency are
   dangerously noisy (cold-start effects, TensorRT's own build-to-build tactic selection
   noise of ~±8%), and the fix was simply to repeat the build 3x per config.
4. **Surprising fact: the per-tensor INT8 scale for the *entire model* is sitting in a
   127-line plain-text file you can `cat` and hand-decode with 5 lines of Python.** No
   black box. Great "build the code" moment — literally open `calibration.cache`, show
   the raw hex, write a decoder inline, and get out a real number.
5. **Engineering decision — why does `calibrate_int8.py` set `BuilderFlag.FP16` even
   though it's building an INT8 engine?** (Line: `config.set_flag(trt.BuilderFlag.FP16)
   # allows fallback for layers without an INT8 kernel`.) Problem: not every layer has an
   INT8-optimized kernel implementation. Constraint: TensorRT must still produce a valid
   engine even for those layers. Solution: allow FP16 as a fallback precision *within* an
   INT8 build. Tradeoff: this is exactly the mechanism behind the "2 of 58 layers stay in
   Float" finding — it's not a bug, it's the fallback flag doing its documented job on the
   one layer where TensorRT judged full precision worth keeping.
6. **Tradeoff, stated numerically, no hand-waving: accuracy cost of INT8 here is a
   drop from 100% → 93.75% top-1 agreement with the PyTorch reference, and a max output
   diff that jumps from 0.087 (FP16) to 2.396 (INT8)** — roughly 27x larger a single-value
   error than FP16's own error over FP32. This is a real, measured number, not "some
   accuracy is lost" hand-waving.
7. **Mystery: why is the single largest layer in the INT8 engine, by profiled time, not
   a convolution at all?** It's `Reformatting CopyNode for Input Tensor 0 to
   node_Conv_754...` (0.036 ms, 3.25% of profiled total) — a fixed-cost, one-time
   NCHW→INT8-layout conversion at the very front door of the network. It doesn't shrink
   with quantization the way compute does, so as everything else gets faster, this fixed
   cost becomes proportionally more visible. Nice illustration of Amdahl's-law-flavored
   thinking applied to a real profile.
8. **Paradox: version-locking bit the project mid-experiment.** An engine serialized by
   one TensorRT Python package version refuses to deserialize under a different-versioned
   `trtexec` (`Serialization assertion ... Version tag does not match`) — a system driver
   upgrade silently bumped the system TensorRT install out from under already-built
   engines. Two tools, "same" TensorRT, incompatible binary files. Fixed by routing
   everything through one consistent Python API version.
9. **Misconception: "a bad accuracy smoke test means the INT8 engine is broken."** The
   project deliberately documents a counter-lesson: testing INT8 accuracy with a single
   batch of random Gaussian noise is close to meaningless, because INT8 quantization
   ranges are calibrated against *real image statistics* — noise is exactly the
   out-of-distribution input where argmax flips happen regardless of engine quality, and
   n=1 has zero statistical power anyway. Real images were required to get a trustworthy
   number.
10. **Tradeoff, disclosed rather than hidden: the 93.75% top-1 number is not a rigorous
    held-out eval.** The 32 accuracy-check images were resampled from the same 500-image
    pool used to calibrate the INT8 engine — a legitimate directional check, but not proof
    of generalization. The repo flags this itself as open follow-up work rather than
    silently presenting 93.75% as the final word. Good beat for modeling intellectual
    honesty on camera instead of overclaiming.

## 3. Skeptic Findings & Contradictions

- **Claim: "INT8 gives 4x the speed of FP32."** — False as a general claim; true only as
  the vs.-FP32 ratio 4.50x measured *here* (Nsight ground truth), for *this* model, *this*
  batch size (1), and *this* GPU. The dossier's own profiling section explains why it
  isn't a clean 4x from "4x fewer bits": the FP16→INT8 leg only contributed 1.70x, not
  2x, because of batch=1 kernel-occupancy limits. Skeptic mode conclusion: never state the
  4.5x number without immediately attaching "at batch=1, on an RTX 4050 Laptop GPU" — a
  batch>1 run (explicitly flagged as *not yet run* in `results/RESULTS.md`, "Open (not
  run yet)") could push the ratio closer to a true 2x per step. Do not imply that
  follow-up already happened.
- **Claim: "TensorRT applies per-channel weight scales / per-tensor activation
  scales automatically."** — This is standard, widely-documented NVIDIA behavior, but it
  has **not been independently verified inside this specific engine** in this session
  (would require pulling per-channel scale arrays out of the engine-inspector JSON for a
  conv layer, which wasn't captured). Flagged explicitly as PLAUSIBLE, not CONFIRMED, and
  added to the Experiment Plan below as a live on-camera check rather than an asserted
  fact.
- **Claim (my own decode attempt): "the input tensor's calibrated dynamic range is
  0.026."** — Two internally-consistent-but-different decodes of the same 8 hex
  characters are both mechanically valid depending on assumed byte order (0.0261 vs.
  0.2293). Cross-checking against the *known* plausible range of this specific tensor
  (ImageNet-normalized pixels, roughly [−2.12, +2.64] per `IMAGENET_MEAN`/`STD` in the
  repo's own `preprocess_image`) doesn't cleanly favor either candidate — both are far
  below the naive expected max, which is itself explained by entropy calibration
  deliberately clipping outliers rather than a decode bug. **Do not present a specific
  decoded number as ground truth on camera without deriving it live and sanity-checking
  it against the known input range** — this is exactly the kind of "verify by running it,
  not by trusting the first plausible-looking number" moment the video should model, not
  paper over. See Experiment Plan #1.
- **Whose evidence, who disagrees**: NVIDIA's own TensorRT best-practices docs
  acknowledge INT8 speedups are workload- and batch-size-dependent and don't promise a
  flat 4x — this repo's finding is consistent with, not contradicting, NVIDIA's own
  framing. No real disagreement found; the risk is *creator* overclaiming past what
  NVIDIA itself claims, not a factual dispute in the field.
- **Counterexample check**: could the "compute-bound at every precision" finding be an
  artifact of this being a small, laptop-GPU, batch=1 run rather than a general truth
  about CNN inference? Yes — plausible. Larger batches or a memory-bandwidth-heavy
  architecture (e.g., a model with many elementwise/attention ops relative to conv/GEMM)
  could easily flip to memory-bound. The video should state the compute-bound finding as
  true *for this model, this batch size, this GPU*, not as a universal law.

## 4. Experiment Plan (build-along sequence, linear order)

Numbered in the literal order they should appear on screen, matching the "explain, then
build" format directive. Each rung either already has a script in this repo, or is a
short new one to write live.

1. **Visualize it — decode a real calibration scale on camera.** Open
   `calibration/calibration.cache`, show the raw text (127 lines, `name: hexfloat`).
   Live-write ~5 lines of Python: `bits = int(hex_str, 16)`, reinterpret as IEEE-754 float
   (`struct.unpack('>f', struct.pack('>I', bits))`), print it next to `amax/127` as the
   scale. Immediately sanity-check the resulting number against the known input range
   (ImageNet-normalized, roughly ±2.1 to ±2.6) and flag the discrepancy honestly on camera
   as evidence of entropy calibration's outlier-clipping behavior rather than hiding it —
   this *is* the "range → scale" pipeline from the source material, pulled out of a real
   file instead of a slide.
2. **Reproduce at toy scale — the quantization loop by hand, then in code.** Take the
   worked example already in the source notes (`[-1.5, -0.7, -0.2, 0.0, 0.3, 0.8, 1.2]`,
   symmetric INT8, scale = 1.5/127 ≈ 0.01181, `q=round(0.8/s)=68`, dequant ≈0.803, error
   ≈−0.003) and write it as an 8-line NumPy function on screen. Then run the *same*
   function against a real slice of activation values pulled from one of this repo's own
   ONNX tensors (via `onnxruntime` or by hooking a PyTorch forward hook on
   `resnet50.pth`) to show the toy math and the real model aren't different things.
3. **Code audit — confirm per-channel weights vs. per-tensor activations inside this
   engine.** Extend `profile_layers.py`'s existing `engine.create_engine_inspector()`
   call (already imported and used) to dump full layer info for one INT8 conv layer and
   look for a per-channel scale array vs. a single scalar for its input tensor. This
   directly resolves the PLAUSIBLE/CONFIRMED gap flagged in the Skeptic section — do this
   before claiming the per-channel/per-tensor split as fact on camera.
4. **Benchmark it — the headline numbers, live.** Run `scripts/build_and_bench.py`
   on camera (or show a captured run — full cold build takes real minutes) to produce the
   1.808 / 0.645 / 0.347 ms table. This is the emotional payoff beat, placed *after* the
   audience understands what FP32/FP16/INT8 mechanically are, per the linear-not-teased
   format directive.
5. **Code audit — is INT8 actually running INT8?** Show the Nsight Systems kernel-name
   grep already done for this repo (`i8i8`/`IMMA` kernel names, 96.4% of 17,920 launches)
   as the literal proof, not an assumption. This is the single best "don't trust the
   label, check the kernel" beat in the whole video.
6. **Benchmark it — the "why not 4x" follow-up.** Show the Nsight per-precision GPU-time
   table (2.008 / 0.756 / 0.446 ms) and the compute-bound-percentage table (97/97.7/95%)
   side by side to walk through the batch=1 kernel-occupancy explanation for the
   FP16→INT8 shortfall (1.70x vs. a theoretical 2x).
7. **Reproduce at toy scale — reconstruct the CUDA-graph win with a stopwatch mental
   model.** No new code needed beyond what's in `build_and_bench.py`'s `benchmark()`
   already (it has both the `use_cuda_graph=True/False` paths) — toggle the flag on
   camera and show the fp32/fp16/int8 percentage gains directly, reinforcing "fixed
   overhead doesn't shrink with bit width."
8. **Open, flagged as future work on camera, not run in this video**: a batch>1 rerun of
   the FP16-vs-INT8 comparison, to test whether larger batches close the 1.7x-not-2x gap
   — stated explicitly in `results/RESULTS.md` as not yet done. Good place to end the
   video with a genuine, unresolved next question rather than a manufactured cliffhanger.

## 5. Audience Shield (critical questions, ranked beginner → expert)

**Beginner tier**
1. "Why can't we just always use INT8 if it's faster and smaller?" → Because it costs
   real accuracy (100%→93.75% top-1 here) and needs representative calibration data;
   FP16 is the "almost free" middle option (0% top-1 loss measured here, ~2.8x speedup).
2. "Is FP16 quantization?" → No — it's a different floating-point *format*, not a
   quantization scheme; no scale/zero-point/calibration involved.
3. "What is a 'scale' actually, physically?" → The size, in real FP32 units, of one INT8
   step. Multiply an integer by it to get back to (approximately) the original float.
4. "Why 127 and not 128 for the symmetric denominator?" → Signed INT8 range is
   [−128, 127]; symmetric quantization intentionally sacrifices the −128 codepoint so
   that +max and −max are treated symmetrically and zero still maps exactly to zero.
5. "What's a zero-point, in one sentence?" → The INT8 integer that represents real-world
   zero, when the float range isn't centered on zero (e.g., post-ReLU values ≥0).
6. "Does quantization change the model's weights permanently, or just at inference?" →
   For post-training static quantization (what this repo does), the original FP32
   weights/ONNX file are untouched; a separate INT8 engine artifact is built from them.
7. "What does 'calibration' need as input?" → Representative real data run through the
   network once, purely to observe activation ranges — not labels, not gradients, not
   training.
8. "Why does the video say FP16 'roughly halves' memory but the measured numbers are
   140→72 MB, not 140→70?" → Fixed overhead (context buffers, workspace) doesn't halve
   even when tensor storage does; the ratio approaches but doesn't hit exactly 2x.

**Intermediate tier**
9. "If INT8 tensor cores are rated 2x FP16 throughput, why did this repo only measure
   1.7x?" → Answered at length in the Curiosity Engine #1 / Experiment Plan #6 — batch=1
   kernel-launch overhead and occupancy, not a broken engine.
10. "How do you know the INT8 engine isn't secretly running most layers in FP16?" →
    Checked the actual CUDA kernel names in the Nsight trace (96.4% are `i8i8`/IMMA
    kernels) rather than trusting the builder flag.
11. "Why does the calibrator need `BuilderFlag.FP16` set if it's building an INT8
    engine?" → Fallback path for any layer lacking an INT8 kernel implementation — see
    Curiosity Engine #5.
12. "Isn't comparing accuracy on calibration images itself cheating?" → Legitimate
    concern, and the repo says so explicitly — flagged as a real limitation, not resolved
    in this video (Curiosity Engine #10 / Skeptic section).
13. "Does batch size change any of these numbers?" → Almost certainly yes for the
    INT8/FP16 ratio specifically (small per-layer kernels at batch=1 don't reach peak
    throughput); explicitly untested here — see Experiment Plan #8.
14. "What GPU/software was this run on, and does that matter?" → RTX 4050 Laptop GPU
    (compute capability 8.9), driver 580.173.02, CUDA 13.0, TensorRT 10.14.1 — absolute
    numbers are hardware-specific; the *shape* of the finding (compute-bound, INT8
    genuinely running, launch-overhead-limited gain) is the transferable part.
15. "Why does engine *build* time go up as precision goes down (18.6s → 40.4s →
    64.8–85.3s)?" → More work per build: FP16 searches a larger tactic space including
    FP16 kernels; INT8 additionally runs a full calibration pass over real images before
    building.

**Expert tier**
16. "Is the per-channel/per-tensor split actually happening in this engine, or is that
    just textbook TensorRT behavior being assumed?" → Explicitly marked PLAUSIBLE not
    CONFIRMED in this dossier — resolving it is Experiment Plan #3, a real code-audit
    step, not assumed.
17. "Single-run A/B microbenchmarks on GPU kernels are notoriously noisy — how do you
    know DIRECT_IO's measured effect isn't noise?" → The project caught exactly this on
    itself: first single-run measurement (9–13%) was later shown to be a cold-start
    artifact once repeated 3x per config, revised down to ~2–4% for INT8 and ~0 for
    FP16/FP32, against a documented ~±8% build-to-build tactic-selection noise floor.
18. "Why is entropy (KL-divergence) calibration preferred over simple min/max for
    INT8?" → Min/max is outlier-sensitive — a single extreme activation value blows out
    the whole tensor's range and wastes resolution on the common case; entropy
    calibration (Migacz 2017) picks a clipping threshold that minimizes information loss
    across the *distribution*, at the cost of clipping (and thus systematically
    misrepresenting) the true extremes.
19. "Could this model become memory-bound instead of compute-bound at a different batch
    size or on different hardware?" → Yes, plausibly — this finding is scoped to
    batch=1 on this specific GPU; see Skeptic section counterexample check.
20. "Version-locked engine serialization — is that a TensorRT quirk or standard for
    compiled-artifact inference runtimes generally?" → Standard pattern for any format
    that bakes in build-time tactic selection tied to a specific library build (similar
    class of issue to, e.g., pickled ML models tied to a library version) — not
    TensorRT-specific in kind, though the specific error string is.
