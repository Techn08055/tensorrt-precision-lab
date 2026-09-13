# Video Metadata — FP32 → FP16 → INT8: What Precision Actually Changes

Grounded in `video_script.md` (169 scenes, 18 chapters + close) and the measured
numbers in `video_dossier.md`. Nothing below states a number, tool, or result the
script doesn't say.

## A. Title options

1. I Made ResNet50 5x Faster With TensorRT — Here's What It Cost
2. FP32 → FP16 → INT8: What Quantization Actually Costs You
3. Why My INT8 Model Wasn't 2x Faster Than FP16
4. I Decoded a TensorRT Calibration File, Byte by Byte
5. TensorRT Rates INT8 at 2x FP16. I Measured 1.7x.
6. The Real Cost of Quantizing ResNet50 to INT8
7. TensorRT INT8: 5x Faster, 4x Smaller, 94% as Accurate
8. I Measured Every TensorRT Precision Mode on My Own GPU
9. Quantization Isn't Free: FP32 vs FP16 vs INT8, Measured
10. My "Big Win" TensorRT Flag Was Mostly a Cold-Start Artifact

**Top 5, ranked:**

1. **#5 — "TensorRT Rates INT8 at 2x FP16. I Measured 1.7x."** Strongest curiosity gap: states a spec-sheet claim and contradicts it with a real measurement in the same breath. High technical credibility (a developer who knows the "2x tensor core" claim will click to find the gap). Search intent is narrower but exactly matches Ch.13's core reversal.
2. **#3 — "Why My INT8 Model Wasn't 2x Faster Than FP16"** Same reversal, phrased as a question — broader search intent ("INT8 not faster than FP16"), still concrete and credible, no exaggeration.
3. **#1 — "I Made ResNet50 5x Faster With TensorRT — Here's What It Cost"** Leads with the one big, true number (5x, Ch.11/18), "here's what it cost" sets up the accuracy-honesty payoff (Ch.16) without giving it away. Broadest audience fit (anyone searching "TensorRT speedup").
4. **#7 — "TensorRT INT8: 5x Faster, 4x Smaller, 94% as Accurate"** All three numbers are real and specific (Ch.11, Ch.16) — highest search-intent density (people search exact tradeoff numbers), but lower curiosity gap since it front-loads the payoff.
5. **#4 — "I Decoded a TensorRT Calibration File, Byte by Byte"** Narrower audience (people already past "what is quantization") but strong technical credibility and a genuine hands-on hook (Ch.8's surprising 0.026 decode) that differentiates from generic quantization explainers.

## B. Thumbnail text options

1. NOT 2x
2. 5x FASTER
3. 1.7x, NOT 2x
4. 94% ACCURATE
5. THE CATCH
6. I MEASURED IT
7. BYTE BY BYTE
8. REAL NUMBERS
9. WRONG GUESS
10. WHY NOT 2x?

## C. Description

Same model. Same weights. Three different number formats — and a 5x speed gap between
them that shouldn't be that simple. This is what actually happens inside a TensorRT
engine when you quantize ResNet50 from FP32 down to FP16 and INT8, verified against the
real files this repo produced, not a slide deck.

Most quantization explainers stop at "INT8 is faster, but less accurate." This one goes
further: it opens the actual calibration cache file, decodes a real scale value by hand,
checks TensorRT's own engine-inspector output to confirm per-channel weight scaling is
really happening, traces literal CUDA kernel names to prove the INT8 engine is really
running INT8, and catches one of its own optimization results overstating its win on
the first try.

**In this video you'll learn:**
- What a 32-bit float's sign/exponent/mantissa fields actually store, and why FP16 is
  just a smaller version of the same idea while INT8 is a fundamentally different one
- How symmetric and asymmetric quantization scales are computed, with a scale and
  zero-point worked by hand and then in code
- Why weights get a per-channel scale and activations get one per-tensor scale in this
  engine, confirmed directly from TensorRT's engine inspector JSON, not just the docs
- How entropy calibration picks a scale from 500 real images, and why the resulting
  number can look "too small" and still be correct
- Why FP16→INT8 only measured 1.7x faster instead of the rated 2x, and what that gap has
  to do with kernel launch overhead at batch size 1
- What CUDA graphs actually fix, and what they don't
- Why one optimization flag's first-run result didn't survive being measured three times

**Key Takeaways:**
- ResNet50 on TensorRT (RTX 4050 laptop GPU): FP32 1.808ms → FP16 0.645ms → INT8
  0.347ms — a 5x latency drop end to end
- Engine size drops from 107.94MB (FP32) to 49.14MB (FP16) to 25.26MB (INT8)
- FP32 and FP16 match the reference PyTorch model on every tested image; INT8 matches
  about 94% of the time, with the worst single-value error over 27x FP16's
- The measured FP16→INT8 speedup (1.7x, later ~1.9x with CUDA graphs) falls short of
  the hardware's rated 2x INT8-over-FP16 tensor-core throughput because batch-size-1
  convolution layers are too small to fully use the tensor cores, so fixed kernel
  launch overhead eats a larger share of a smaller kernel
- Nsight Systems kernel traces confirm ~96% of GPU kernel launches in the INT8 engine
  are genuine INT8 ("i8i8") kernels, not a fallback
- The TensorRT engine inspector confirms per-channel scales for weights (one value per
  output channel) and a single per-tensor scale for activations
- A DIRECT_IO build flag that looked like a 9–13% win on a single run dropped to
  roughly 0% (FP16) and 2–4% (INT8) once the build was repeated three times per config
- The accuracy test set overlaps with the calibration set, a caveat stated directly
  rather than left out, along with the untested question of whether a larger batch
  size closes more of the FP16→INT8 gap

**Chapters:**
[00:00:00] Orientation — One Model, Three Precisions
[TIMESTAMP NEEDED] What A Float Actually Is (FP32)
[TIMESTAMP NEEDED] Shrinking The Format (FP16)
[TIMESTAMP NEEDED] Where Floats Stop Working (Why INT8 Is Different)
[TIMESTAMP NEEDED] The Simplest Mapping (Symmetric Quantization)
[TIMESTAMP NEEDED] When Zero Isn't In The Middle (Asymmetric Quantization)
[TIMESTAMP NEEDED] One Scale Or Many? (Per-Tensor vs Per-Channel)
[TIMESTAMP NEEDED] Where The Scale Actually Comes From (Calibration)
[TIMESTAMP NEEDED] Opening The Calibration File
[TIMESTAMP NEEDED] From Toy Numbers To Real Activations
[TIMESTAMP NEEDED] Checking TensorRT's Own Homework (Engine Inspector Audit)
[TIMESTAMP NEEDED] The Benchmark (build_and_bench.py)
[TIMESTAMP NEEDED] Is INT8 Actually Running In INT8?
[TIMESTAMP NEEDED] Why Isn't It 4x?
[TIMESTAMP NEEDED] Removing The Launch Overhead (CUDA Graphs)
[TIMESTAMP NEEDED] The Flag That Wasn't As Good As It Looked (DIRECT_IO)
[TIMESTAMP NEEDED] What Accuracy Actually Costs
[TIMESTAMP NEEDED] What's Still Open
[TIMESTAMP NEEDED] Close

*(Only Chapter 0's start is known by definition — 00:00:00. The rest need to be filled
in once the final render's real cut points exist; do not paste guessed values into
YouTube's chapter field, they'll be wrong.)*

GitHub:
[INSERT LINK]

Resources:
[INSERT LINKS]

[INSERT SOCIAL LINKS]

Every number in this video came from scripts sitting in that repo, calibration cache
included — if you run the bigger-batch version yourself, drop what you got in the
comments.

## D. Hashtags

`#MachineLearning` `#DeepLearning` `#CUDA` `#TensorRT` `#INT8Quantization`

## E. Search tags

TensorRT, INT8 quantization, TensorRT INT8, FP16 vs INT8, FP32 vs FP16 vs INT8, model
quantization, quantization calibration, entropy calibration, per-channel quantization,
per-tensor quantization, TensorRT engine inspector, ResNet50 benchmark, GPU inference
optimization, CUDA graphs, Nsight Systems, CUDA kernel launch overhead, TensorRT
calibration cache, neural network quantization explained, deep learning inference
latency, quantization accuracy loss

## F. SEO / AEO check

**Primary search query:** TensorRT INT8 quantization benchmark

**Secondary search queries:**
- FP16 vs INT8 TensorRT speed
- why isn't INT8 2x faster than FP16
- TensorRT calibration cache explained
- TensorRT per-channel vs per-tensor quantization

**One-sentence answer for AI search:** Running the same ResNet50 model through
TensorRT at FP32, FP16, and INT8 on an RTX 4050 laptop GPU measured a 5x latency drop
(1.808ms → 0.347ms) and roughly a quarter the engine size from FP32 to INT8, with INT8
matching the full-precision model's output on about 94% of tested images and a
FP16-to-INT8 speedup of only ~1.7x (vs. the hardware's rated 2x) due to kernel launch
overhead dominating at batch size 1.

## G. Final recommended package

```
TITLE: TensorRT Rates INT8 at 2x FP16. I Measured 1.7x.
THUMBNAIL TEXT: NOT 2x
THUMBNAIL CONCEPT: Split frame — left half a clean "2x" spec-sheet stat card, right half
the real measured ratio number with a red strike/correction mark over the 2x; creator
reacting in the corner. Hand off to forward-logic-thumbnail for the actual image.
WHY THEY WORK TOGETHER: the title states the spec-sheet claim and the measured
contradiction in one breath; the thumbnail text is the three-word punch of that same
contradiction ("NOT 2x") without repeating the title's wording, so title and thumbnail
reinforce instead of duplicate.
PRIMARY SEARCH QUERY: TensorRT INT8 quantization benchmark
DESCRIPTION: <see section C above>
HASHTAGS: #MachineLearning #DeepLearning #CUDA #TensorRT #INT8Quantization
TAGS: TensorRT, INT8 quantization, TensorRT INT8, FP16 vs INT8, FP32 vs FP16 vs INT8,
model quantization, quantization calibration, entropy calibration, per-channel
quantization, per-tensor quantization, TensorRT engine inspector, ResNet50 benchmark,
GPU inference optimization, CUDA graphs, Nsight Systems, CUDA kernel launch overhead,
TensorRT calibration cache, neural network quantization explained, deep learning
inference latency, quantization accuracy loss
```
