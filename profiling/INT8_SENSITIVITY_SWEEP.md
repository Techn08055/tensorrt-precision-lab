# INT8 Sensitivity Sweep: Can Only the Safe Layers Be Quantized?

## Question

The [profiling writeup](PROFILE_RESULTS.md) found FP16→INT8 only gets ~1.7-1.86x
(not 2x) at batch=1, and separately that full INT8 costs real accuracy (93.75%
top-1 vs. PyTorch, [`../results/RESULTS.md`](../results/RESULTS.md)). Follow-up
question: can most of INT8's speed be kept while only quantizing the layers that
don't hurt accuracy, leaving the accuracy-sensitive ones at FP16?

Short answer: **the accuracy part works as expected; the speed part doesn't.**
Selectively quantizing only the layers with zero measured accuracy impact
recovers roughly none of full INT8's latency win at batch=1. Details below.

## Method: per-block sensitivity sweep

TensorRT supports per-layer precision constraints (`ILayer.precision` +
`BuilderFlag.OBEY_PRECISION_CONSTRAINTS`) instead of one global INT8 flag, so
"only some layers INT8" is directly buildable.

To find *which* layers are safe, `scripts/sweep_int8_sensitivity.py`:

1. Builds a **baseline engine pinned to FP16 everywhere** (every conv layer's
   `.precision` explicitly set to `HALF`, not just the default TensorRT would
   pick with the FP16 flag on).
2. Groups ResNet50's 53 conv layers into **17 units** (the stem conv + 16
   bottleneck blocks — 3, 4, 6, 3 per stage, with a downsample conv on each
   stage's first block), derived directly from the ONNX graph's node inputs
   rather than assumed, since TensorRT's execution order does not match ONNX
   definition order for parallel branches (the profiler's `index` field is
   scheduling order; the sweep needed *graph* order to find block boundaries).
3. For each unit, builds a mixed engine with **only that block's conv layers
   forced to INT8** (reusing the existing `calibration.cache` for scales) and
   every other conv layer still pinned to FP16.
4. Compares each mixed engine's output to the **FP16 baseline**, not the FP32
   PyTorch reference. This isolates the quantization error *that block alone*
   introduces, instead of mixing it with the ordinary fp32→fp16 gap.
5. Verifies via the engine inspector that the block actually landed in `Int8`
   output dtype each time (OBEY_PRECISION_CONSTRAINTS forces it, but this
   catches a silent fallback rather than assuming the flag worked).

Per-layer (all 53 convs individually) would have been more granular but ~3x
more engine builds for less actionable signal than block-level, since a real
deployment decision is "quantize this residual block or not," not individual
convs within it.

## Sweep results (17 units, 64 real images, sorted least → most sensitive)

| Block | Convs | Flip rate vs FP16 baseline | Mean \|Δlogit\| | Top-1 vs PyTorch |
|---|---:|---:|---:|---:|
| layer2.1 | 3 | 0.0% | 0.0060 | 100.0% |
| layer3.0 | 4 | 0.0% | 0.0084 | 100.0% |
| layer3.5 | 3 | 0.0% | 0.0085 | 100.0% |
| layer3.4 | 3 | 0.0% | 0.0091 | 100.0% |
| layer4.1 | 3 | 0.0% | 0.0147 | 100.0% |
| layer3.3 | 3 | 1.6% | 0.0087 | 98.4% |
| layer2.2 | 3 | 4.7% | 0.0373 | 95.3% |
| layer2.3 | 3 | 4.7% | 0.0375 | 95.3% |
| layer3.1 | 3 | 4.7% | 0.0376 | 95.3% |
| layer3.2 | 3 | 4.7% | 0.0377 | 95.3% |
| layer1.2 | 3 | 4.7% | 0.0377 | 95.3% |
| layer1.1 | 3 | 6.3% | 0.0380 | 93.8% |
| layer2.0 | 4 | 6.3% | 0.0382 | 93.8% |
| layer4.0 | 4 | 4.7% | 0.0391 | 95.3% |
| layer4.2 | 3 | 6.3% | 0.0412 | 93.8% |
| layer1.0 | 4 | 1.6% | 0.0466 | 98.4% |
| **stem** | 1 | **7.8%** | **0.0611** | **92.2%** |

Raw data: [`int8_sensitivity_sweep.json`](int8_sensitivity_sweep.json).

The **stem conv is the single most sensitive layer** — expected, it's the
layer closest to raw pixel statistics rather than learned, more uniform
intermediate feature statistics, and matches standard mixed-precision-
quantization practice of keeping the first/last layers at higher precision.
Five blocks (`layer2.1`, `layer3.0`, `layer3.4`, `layer3.5`, `layer4.1`) show
**zero top-1 flips** over 64 images — the "safe" set used below.

Two caveats worth stating plainly:

- The 64 eval images are drawn from the same pool used for INT8 calibration,
  not a held-out split (same known limitation as
  [`../README.md`](../README.md#lessons-learned)).
- **Top-1 flip counts at n=64 are dominated by a few near-tie images.** In the
  FP16 baseline, 13 of the 64 images have a top-1 vs. top-2 logit margin under
  0.5 (images 21 and 30 are at 0.031 and 0.035 — effectively coin flips). Every
  image that flips in the selective and full-INT8 engines below is one of these
  low-margin images. So a "4.7%" flip rate is really 3 images, and 1 image
  (1.6%) is the resolution of the whole column. The **mean |Δlogit|** column is
  the more trustworthy signal here, because it doesn't hinge on a tie.
  Also worth noting: 9 of the 17 blocks report a nearly identical max |Δlogit|
  (3.789–3.813), close to what the stem gives (3.93) and to the worst single
  image under full INT8 (image 15, 3.931). That looks like one fragile image (or
  one clipped outlier activation on the shared residual stream) rather than 9
  independent block effects. Per-image results weren't saved for the sweep
  engines, so this is a hypothesis to test, not a finding.

## Follow-up: build and measure a selective-INT8 engine

`scripts/build_selective_int8.py` builds an engine with only chosen blocks
quantized to INT8, everything else pinned to FP16, `BuilderFlag.DIRECT_IO` on
to match the production fp32/fp16/int8 builds. Two variants were tried:

- **Scattered**: the 5 individually-safest blocks from the sweep table above
  (`layer2.1`, `layer3.0`, `layer3.4`, `layer3.5`, `layer4.1`) — 16 of 53 conv
  layers, forming 4 separate INT8 islands (`layer3.4`/`layer3.5` are adjacent).
- **Contiguous**: `layer3.3`→`layer3.4`→`layer3.5`, a consecutive run within
  one stage (`layer3.3` had a small 1.6% flip rate individually, included to
  make the run contiguous) — 9 of 53 conv layers, one INT8 island. This tests
  the boundary-reformat-cost hypothesis directly.

(An earlier version of this section said the scattered engine crossed a
precision boundary "10 times" and the contiguous one "2 times". That was
counted by hand from block placement and never measured — the topological count
is actually 8 vs. 2. What TensorRT *actually inserted* is measured below and is
different again.)

Latency was benchmarked with the same CUDA-graph-capture methodology as
[`../results/RESULTS.md`](../results/RESULTS.md) — but a first single-run pass
gave a **misleadingly clean "0.607ms selective ≈ 0.606ms FP16" result that
didn't survive repetition**, so every number below is a mean of 5 repeated
trials (matching the rigor the CUDA-graph/DIRECT_IO investigation in
[`../results/RESULTS.md`](../results/RESULTS.md#optimizations-applied)
already established was necessary here). Accuracy checked with the existing
`verify_accuracy_venv.py` against the same 64 images.

| | FP16 (full) | Selective, scattered (5 blocks) | **Selective, contiguous (3 blocks)** | INT8 (full) |
|---|---:|---:|---:|---:|
| GPU latency, mean of 5 (ms) | 0.618¹ | 0.606 | **0.602** | 0.335 |
| GPU latency, min of 5 (ms) | 0.590 | 0.606 | **0.601** | 0.334 |
| Top-1 vs PyTorch (n=64) | 100.0% | 96.9% | **98.4%** | 90.6% |
| Mean abs output diff | 0.0023 | 0.0213 | **0.0123** | 0.0724 |

¹ FP16's first of 5 trials read 0.683ms, a cold-start outlier consistent with
the noise this project's own DIRECT_IO write-up already flagged; excluding it,
FP16's steady-state mean is ~0.602ms — statistically the same as the
contiguous variant.

Raw data: [`selective_int8_meta.json`](selective_int8_meta.json),
[`selective_int8_contiguous_meta.json`](selective_int8_contiguous_meta.json),
[`../benchmarks/selective_int8_repeated_benchmark.json`](../benchmarks/selective_int8_repeated_benchmark.json)
(the 5-trial data), [`../benchmarks/selective_int8_contiguous_accuracy.json`](../benchmarks/selective_int8_contiguous_accuracy.json).

**Accuracy: both variants recover most of the gap to FP16, but top-1 can't
separate them.** Both selective engines sit far closer to FP16 than full INT8
does (mean abs diff 0.021 / 0.012 vs. 0.072). Top-1 says 96.9% (scattered) vs.
98.4% (contiguous), but that is 2 flipped images vs. 1, and all of them are
near-tie images (scattered flips images 21 and 30, margins 0.031 / 0.035;
contiguous flips image 30 only; full INT8's six flips all have margins under
0.5). So the top-1 difference between the two selective engines is one coin
flip, not evidence. The mean |Δlogit| gap (0.021 vs. 0.012) and max |Δlogit|
(0.98 vs. 0.29) are continuous and do favor contiguous, but that is still one
pair of engines, not a controlled test.

**Measured reformat cost (this is the part that explains the latency).**
Counting `Reformatting…` layers in each serialized engine and summing their
time under TensorRT's per-layer profiler (4 rounds, reformat time was stable
across rounds):

| | FP16 | Scattered | Contiguous | INT8 (full) |
|---|---:|---:|---:|---:|
| Engine layers | 58 | 67 | 62 | 58 |
| Reformat layers | 1 | **10** | **5** | 2 |
| Reformat time (ms, profiled) | 0.014 | **0.090** | **0.038** | 0.011 |

TensorRT inserted 9 extra reformat layers for the scattered engine and 4 for
the contiguous one, costing roughly +0.076 ms and +0.024 ms, i.e. about 12% and
4% of a 0.60 ms inference. Those extra copies are the same size as the compute
time the INT8 layers save, which is why the net latency doesn't move.

**Latency: no real win at batch=1.** With CUDA graphs the two selective engines
land at 0.606 / 0.602 ms against FP16's steady-state ~0.602 ms. Without CUDA
graphs (5 trials each) it's 0.650 (scattered) / 0.641 (contiguous) / 0.645
(FP16) ms, so contiguous is ahead of scattered by about 0.009 ms in that setup
and level with FP16 within noise; scattered is at best level with FP16. These
sub-1% gaps are also smaller than the ~±8% build-to-build variation the DIRECT_IO
investigation measured, and each selective engine was built only once — the
trial-to-trial spread only covers run-to-run noise on that one file. So
"contiguous is faster than scattered" is **not established**; the reformat-layer
counts above (10 vs. 5) are the stronger evidence. Full
INT8 is 0.335 ms (0.377 without graphs) in both setups, so the gain is real
when everything is INT8 and disappears when INT8 is scattered through an FP16
graph. The evidence points at **reformat overhead cancelling the INT8 compute
saving**, and contiguity helps because it needs fewer reformats.

What this does *not* establish: I earlier attributed the flat latency mainly to
"kernel occupancy at batch=1" ([`PROFILE_RESULTS.md`](PROFILE_RESULTS.md)).
That may still explain why full INT8 gets 1.86x not 2x, but nothing here
verifies it for the selective engines, and the profiler's summed layer time
for FP16 was too noisy across rounds (0.727–0.866 ms) to split "saved compute"
from "added reformat" precisely. Treat the reformat counts and times as the
solid part and the exact split as soft.

The latency and accuracy numbers in this section are each from one build of
scattered and one build of contiguous — see the correction immediately below:
both configurations turned out to be bimodal across rebuilds, so treat the
specific values above as one sample from a two-point distribution, not a
fixed property of "scattered" vs. "contiguous."

## Correction: scattered/contiguous accuracy is bimodal across rebuilds

While verifying that `build_selective_int8.py` was actually pinning the
intended layers (it was missing `ProfilingVerbosity.DETAILED`, fixed and
re-verified via the engine inspector — every intended-INT8 conv genuinely
came out `Int8`, every intended-FP16 one genuinely came out `Half`), rebuilding
the *same* scattered and contiguous configurations gave **different accuracy
numbers than the ones reported above**. That turned into its own investigation.

Rebuilding scattered 6 times and contiguous 5 times (identical config, same
calibration cache, same ONNX file each time) does not produce a scatter of
close values — it produces two **discrete, repeatable modes**:

| | Scattered — Mode A (3/6 builds) | Scattered — Mode B (3/6 builds) | Contiguous — Mode A (3/5 builds) | Contiguous — Mode B (2/5 builds) |
|---|---:|---:|---:|---:|
| Mean abs output diff | 0.021 | 0.044 | 0.012 | 0.040 |
| Top-1 vs PyTorch (n=64) | 95.3–96.9% | 93.75% (identical both times) | 98.4% | 95.3% |
| GPU latency, steady state (ms) | ~0.593 | ~0.572 | ~0.586 | ~0.577 |

Every build lands cleanly in one mode or the other — never in between. The two
Mode-B scattered builds gave the *exact same* mean abs diff (0.043675) to six
decimal places, which is what gave this away: that's not measurement noise,
that's TensorRT picking one of two specific tactic/fusion strategies at build
time, each with its own fixed numerical behavior. The lower-accuracy mode is
also consistently the faster one in both engines (by ~2-4%) — a real, small
speed/accuracy tradeoff between the two tactics, dwarfed by the head-FP16 gap
below either way. **The scattered and contiguous engine files currently saved
in this repo both happen to be Mode B** (the worse-accuracy one); the
"96.9%"/"98.4%" numbers earlier in this doc were Mode A, from each
configuration's first build.

This means the "contiguous (98.4%) is more accurate than scattered (96.9%)"
comparison drawn earlier — already softened once, to "one near-tie image, not
a real difference" — needs softening further: that comparison used one build
of each, and each is independently a coin flip between two modes roughly a
build apart from each other. The right comparison is mode-range vs. mode-range,
and even there, contiguous's two modes (98.4%/95.3%) both sit at or above
scattered's corresponding modes (96.9%/93.75%), so "contiguous tends to be a
little more accurate than scattered" still holds directionally — just not from
a single-build measurement, and not by the specific numbers first reported.

**Why head-FP16 doesn't have this problem**: 3 separate rebuilds of it gave
*identical* results to six decimal places every time (0.049257 mean abs diff,
93.75% top-1, ~0.38 ms latency). It has one FP16→INT8 cutover and 3 reformat
layers; scattered/contiguous have 4 and 1 separate INT8 islands with 10 and 5
reformat layers, leaving TensorRT far more fusion/tactic choices at build time.
Fewer islands isn't just faster and more accurate — it also appears to be more
*reproducible*, for the same underlying reason.

Not tested: whether the full FP16 and full INT8 production engines (built by
`build_and_bench.py`/`calibrate_int8.py`, not `build_selective_int8.py`) show
the same bimodality. This investigation only rebuilt the mixed-precision
engines multiple times; the pure single-precision ones were each built once
per this session, same as before.

## Follow-up: keep only the sensitive head in FP16

The scattered/contiguous results above both point the same way: the problem
isn't *how much* is INT8, it's how many times the graph switches. So instead
of carving out small INT8 *islands* inside an FP16 graph, `build_selective_int8.py
--blocks layer1.1 layer1.2 layer2.0 layer2.1 layer2.2 layer2.3 layer3.0 layer3.1
layer3.2 layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2` keeps only the
two most sensitive units from the sweep (`stem`, `layer1.0`) pinned to FP16 and
forces everything else — 48 of 53 conv layers — to INT8. That's one FP16 region
at the front and one INT8 region for the rest: structurally closer to "full
INT8 with a protected head" than to the scattered/contiguous experiments above.

| | FP16 | Scattered (16 convs) | Contiguous (9 convs) | **Head-FP16 (48 convs)** | INT8 (full, 53 convs) |
|---|---:|---:|---:|---:|---:|
| Reformat layers | 1 | 10 | 5 | **3** | 2 |
| GPU latency, mean of 5 (ms) | 0.623¹ | 0.572–0.593² | 0.577–0.586² | **0.381 (fixed)** | 0.336 |
| Top-1 vs PyTorch (n=64) | 100.0% | 93.75–96.9%² | 95.3–98.4%² | **93.75% (fixed)** | 90.6% |
| Mean abs output diff | 0.0023 | 0.021–0.044² | 0.012–0.040² | **0.0493 (fixed)** | 0.0724 |

¹ FP16's first of 5 trials read 0.702 ms in this run (another cold-start
outlier, same pattern noted throughout this doc); steady-state mean of the
other 4 is ~0.603 ms.

² Scattered and contiguous are **bimodal across rebuilds** — see "Correction:
scattered/contiguous accuracy is bimodal across rebuilds" above. These ranges
span the two modes observed; head-FP16 is the only mixed-precision variant
that gave identical numbers across every rebuild tested.

Raw data: [`selective_int8_headfp16_meta.json`](selective_int8_headfp16_meta.json),
[`../benchmarks/selective_int8_headfp16_benchmark.json`](../benchmarks/selective_int8_headfp16_benchmark.json),
[`../benchmarks/selective_int8_headfp16_accuracy.json`](../benchmarks/selective_int8_headfp16_accuracy.json).

**This is the first variant that actually wins on latency.** 0.381 ms is a real
1.58x speedup over FP16's ~0.603 ms steady state, not noise — the 5-trial std
on this engine is 0.0006 ms, tighter than the gap it needs to clear. It gets most
of the way to full INT8's 0.336 ms while only quantizing 91% as many conv layers,
because it needs just 3 reformat layers instead of full INT8's 2 turning into
10 or 5 when the INT8 region gets fragmented into islands.

The accuracy cost is real and lands where you'd expect for converting 48 of 53
convs: worse than either mode of the two small selective engines (93.75% vs.
93.75–96.9%/95.3–98.4% top-1), better than full INT8 (90.6%, 0.072). Its 4
flipped images (21, 30, 53, 63) are a subset of full INT8's 6 flips, and —
consistent with the near-tie finding above — all 4 have a top-1/top-2 logit
margin under 0.5, so this isn't a new failure mode, it's the same near-tie
images being pushed over by more accumulated quantization noise. Unlike the
scattered/contiguous numbers, this one is not a single-build snapshot — it was
reproduced identically across 3 separate rebuilds.

So the honest framing: this isn't "selective quantization beats full INT8" —
it's a genuine speed/accuracy point *between* FP16 and full INT8, reached by
controlling how many conv layers are quantized rather than *which* specific
ones. Whether that's a better tradeoff than full INT8 depends on how much the
6.25-point top-1 gap actually matters for the deployment; it is unambiguously
a better tradeoff than the scattered/contiguous engines, which bought accuracy
but zero speed.

## Conclusion

Per-block INT8 sensitivity analysis found a plausible safe/sensitive split
(the stem conv is the clear outlier on the continuous metric; a handful of
mid-network blocks show near-zero drift), with the caveat above that the
top-1 flip column is mostly near-tie images. Quantizing only a handful of
"safe" blocks scattered or clustered through an otherwise-FP16 graph doesn't
buy latency at **batch=1**: TensorRT inserts extra reformat layers at each
FP16↔INT8 boundary, and their cost is about the size of the compute saved.
But quantizing *almost everything except the sensitive head* does win — one
FP16→INT8 switch near the front instead of several scattered ones recovers
most of full INT8's speed at a real but smaller accuracy cost. The lever that
mattered was never "how much accuracy-safe compute is in INT8" on its own;
it was how many times the graph switches format.

A second, independent finding fell out of double-checking the first one:
scattered/contiguous engines are **bimodal across identical rebuilds** —
TensorRT lands on one of two fixed tactic choices with meaningfully different
accuracy (and a small, opposite-direction latency difference) each time, while
the head-FP16 engine reproduced identically across every rebuild tried. So the
fewer-islands design isn't just faster and (on average) more accurate — it's
also more *trustworthy*, in the specific sense that a single build of it means
something, where a single build of the scattered/contiguous engines doesn't.
That reframes the earlier "contiguous is more accurate than scattered because
it needs fewer reformats" claim: directionally still true across the two
configurations' modes, but not something a single build of either can show on
its own, which is exactly the trap the original single-build numbers fell into.

Open follow-ups: repeat at batch>1 ([`../README.md`](../README.md#status));
save per-image results in the sweep to test the fragile-image hypothesis;
sweep the FP16/INT8 cut point (just `stem`, vs. `stem`+`layer1.0`, vs. further
into the network) to map out the actual speed/accuracy tradeoff curve instead
of one point on it; check whether the full FP16/INT8 production engines show
the same build-to-build bimodality the mixed engines did.

## Reproduce

```bash
cd scripts
python sweep_int8_sensitivity.py --batch 64        # ~6 min, writes ../profiling/int8_sensitivity_sweep.json

# scattered (5 individually-safest blocks)
python build_selective_int8.py

# contiguous (a consecutive 3-block run within one stage)
python build_selective_int8.py --blocks layer3.3 layer3.4 layer3.5 \
  --engine ../selective_int8/resnet50_selective_int8_contiguous.engine \
  --meta-out ../profiling/selective_int8_contiguous_meta.json

python verify_accuracy_venv.py --engines ../fp16/resnet50_fp16.engine \
  ../selective_int8/resnet50_selective_int8.engine \
  ../selective_int8/resnet50_selective_int8_contiguous.engine \
  ../int8/resnet50_int8.engine --batch 64 \
  --results-file ../benchmarks/selective_int8_contiguous_accuracy.json

# head-FP16 (only stem + layer1.0 stay FP16, everything else INT8)
python build_selective_int8.py \
  --blocks layer1.1 layer1.2 layer2.0 layer2.1 layer2.2 layer2.3 layer3.0 layer3.1 \
  layer3.2 layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2 \
  --engine ../selective_int8/resnet50_selective_int8_headfp16.engine \
  --meta-out ../profiling/selective_int8_headfp16_meta.json

python verify_accuracy_venv.py --engines ../fp16/resnet50_fp16.engine \
  ../selective_int8/resnet50_selective_int8_headfp16.engine \
  ../int8/resnet50_int8.engine --batch 64 \
  --results-file ../benchmarks/selective_int8_headfp16_accuracy.json
```

Latency was measured with `build_and_bench.py`'s `benchmark()` function
(CUDA-graph capture/replay, 20 warmup + 200 timed iterations, batch=1) called
directly against all four engine files, **5 times each** — not a standalone
script here, since it's a short reuse of existing code, but the repeat count
matters: a single run per engine gave a misleadingly clean result (see above),
and this project's own DIRECT_IO investigation already established that a
single-run A/B here isn't trustworthy.
