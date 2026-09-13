# Video Storyboard — FP32 → FP16 → INT8: What Precision Actually Changes

Source: `video_dossier.md`. Format directive from the dossier is binding and is honored
throughout this storyboard, see note below before the Blueprint.

## Format note (carried forward, not re-litigated)

This is deliberately **not** built as a Veritasium-style cold-open-mystery video. There is
no flash-forward hook, no withheld reveal saved for a "twist" at minute 12. The chapters
below still move from a question to an answer to the next question, because that's how
understanding a topic like this actually proceeds, one honest step at a time, but every
question is asked and answered *in the order the audience would naturally hit it* while
watching the concept get explained and then built. The belief-reversal beats required by
this skill's format are all **real reversals already on record in this repo's own
history** (the DIRECT_IO re-measurement, the FP16→INT8 shortfall, entropy calibration's
below-max clipping). They are disclosed at the point they're discovered during the build,
not staged as suspense.

**Narration voice note:** every "Narration (draft)" line below is written to sound like
Christy actually talking through an experiment as it happens, thinking out loud,
correcting herself, reacting to a number as she sees it, not like polished voiceover copy.
Keep that register in the final script. Two specific things to keep watching for, since
earlier drafts of this file drifted into them: (1) em dashes used as a default connector
between clauses instead of just picking a period or a comma, and (2) the same handful of
"naturalizing" words (actually, honestly, genuinely, turns out) repeated on nearly every
line until they stop sounding natural and start sounding like a tic. A little of both goes
a long way; more than that reads as performed casualness rather than the real thing.

**Pacing note (added on revision):** the math-heavy chapters (1, 2, 4, 5, 9) were
trimmed after review, the video was carrying too much formula derivation and too many
repeated worked examples back to back for a general audience to sit through comfortably
before reaching any real payoff. Each of those chapters now makes its point with one
clean example instead of two or three, and Chapter 9 (which mostly re-derived math
already taught in Chapter 4) is now a short real-data sanity check instead of a full
replay of the quantization loop. The teaching content that matters, range → scale →
integer → dequantize → error, is still fully covered; the repetition and the exhaustive
edge-case derivations are what got cut. Chapters 3, 6, 7, 8, and 10 are concept- and
code-driven rather than equation-driven and were left as they were.

Primary visual language: Remotion-rendered code blocks and terminal output standing in
for the real repo (not raw screen-capture footage; see `hyperframes-animation`/
`video-animator` conventions), cut against A-Roll of Christy explaining/narrating and
reacting to real numbers as they come out of the actual scripts.

---

## 1. The Story Blueprint

Plain-language version of the chapter arc: what question we're on, what we figure out,
and what question that leaves us with next. Chapter identities and order are unchanged
by the pacing trim above, only how many scenes each one takes to make its point.

| Ch. | Title | We start out wondering | What we find out | Which leaves us wondering | What flips our expectations |
|---|---|---|---|---|---|
| 0 | Orientation — One Model, Three Precisions | What actually changes when you flip that switch? | Here's the finished results table, before any of it makes sense yet | Why do these numbers look the way they do? | — |
| 1 | What A Float Actually Is (FP32) | What is a "32-bit float," really? | Three parts, a sign, an exponent, and a fraction, combine through one formula to build any number | If FP32 already covers such a huge range, why would anyone shrink it? | — |
| 2 | Shrinking The Format (FP16) | What do you actually lose by using fewer bits? | Smaller range, less precision, but it's still a normal float, nothing extra needed to read it | FP16 still has room for an exponent. INT8 doesn't. What happens then? | — |
| 3 | Where Floats Stop Working (Why INT8 Is Different) | Can we just keep shrinking the format down to 8 bits? | No. There's no room left for an exponent at all. INT8 is a bare integer that has no idea what range it's supposed to cover | If INT8 can't describe its own range, something else has to. What? | "Smaller float" intuition breaks: INT8 isn't a smaller float, it's a different kind of thing entirely |
| 4 | The Simplest Mapping (Symmetric Quantization) | What's the least amount of extra info needed to turn floats into integers? | One extra number, a scale, is enough, as long as the range is already centered on zero | What if the range isn't centered on zero? | — |
| 5 | When Zero Isn't In The Middle (Asymmetric Quantization) | What goes wrong when min and max aren't symmetric? | Add one more number, a zero-point, and it's fixed | Do we need one scale for a whole tensor, or can we do better than that? | — |
| 6 | One Scale Or Many? (Per-Tensor vs Per-Channel) | Is one scale enough for a weight tensor where different channels have wildly different sizes? | Giving each channel its own scale fixes it, and TensorRT is documented to do exactly this automatically for weights | Is that actually happening inside this specific engine, or are we just repeating the docs? | Flagged here, not answered yet. Comes back in Chapter 10 |
| 7 | Where The Scale Actually Comes From (Calibration) | Where do these scale numbers even come from for activations, which depend on live input data? | Run real images through the model once and watch what ranges show up, using a smarter method than plain min/max | Could that smarter method end up picking a number smaller than the tensor's real max? | — |
| 8 | Opening The Calibration File | Is "calibration produces a scale per tensor" just a concept, or can we actually see one? | It's a real, plain-text file sitting on disk, and we open it to decode a real number, live | The decoded number looks way too small. Did we mess up the decode, or is something else going on? | Two "correct-looking" ways to decode the same bytes give two different, both-too-small numbers. That turns out to be expected, not a bug |
| 9 | From Toy Numbers To Real Activations | Does the textbook quantization math actually match what happens inside a real model? | Yes. The exact same small function works on a real slice of this model's own activations | We assumed per-channel weights are real. Let's go check that properly. | — |
| 10 | Checking TensorRT's Own Homework (Engine Inspector Audit) | (carried over from Ch.6) Is this engine really using per-channel scales for weights and one scale for activations? | Open the engine's own inspector output for a real layer and look directly | Now that we actually understand how this works, how much faster does INT8 make the model? | Confirms (or corrects) the textbook claim using this engine's own data, instead of trusting a PDF |
| 11 | The Benchmark (build_and_bench.py) | How much faster does INT8 actually make this model? | 1.808 ms → 0.645 ms → 0.347 ms. INT8 ends up around 5x faster than FP32 | INT8 tensor cores are supposedly rated for 2x FP16 speed, but the FP16→INT8 step here is only... | — |
| 12 | Is INT8 Actually Running In INT8? | Before explaining a shortfall, is the INT8 engine even doing what it claims? | Yes, checked directly by kernel name: over 96% of what ran are genuine INT8 kernels | Then why isn't the speed-up a clean 2x on top of FP16? | Answers the skeptical objection before anyone has to raise it |
| 13 | Why Isn't It 4x? | Why is the FP16→INT8 step only 1.70x, when the hardware is rated for 2x? | The model's doing real heavy compute, but at batch size 1 each layer is too small to use the hardware fully. And a fixed launch cost doesn't shrink just because the numbers got smaller | Can that fixed launch cost be removed without touching the model or the batch size? | — |
| 14 | Removing The Launch Overhead (CUDA Graphs) | Can that fixed launch overhead actually be removed? | Yes. Bundling the kernel launches into one "graph" cuts a lot of it; INT8 gains 19-30% from doing this | Every optimization tried so far worked exactly as expected. Did they all, really? | — |
| 15 | The Flag That Wasn't As Good As It Looked (DIRECT_IO) | Did every optimization tried actually help as much as it first looked like it did? | No. One flag looked like a big win on the first try, but measuring it properly (repeating the test) showed the real gain was much smaller, almost nothing for FP16/FP32 | We've spent this whole video chasing speed. What did that speed actually cost? | The real reversal here: the project catching its own first, overstated result |
| 16 | What Accuracy Actually Costs | What did all that speed cost, in accuracy? | Accuracy barely moves for FP16, but INT8 does lose some. About 6% of predictions change, and even that number comes with an honest caveat about how it was tested | Given everything we've now measured, what's still untested, and why does it matter? | Honesty beat: the number is real, but its limits are stated out loud, not hidden |
| 17 | What's Still Open | What's the one thing this whole investigation never actually tested? | Nobody's tried a bigger batch size yet. The theory from Ch.13 says it should help close the gap further, but that's still a guess, not a result | (handed to the viewer, not answered) | — |
| 18 | Close | — | Recap: range, scale, integer, dequantize, error. This time tied to a real, working engine instead of just a slide | — | — |

**Belief-reversal quota check**: Ch.3 (INT8 isn't "a smaller float"), Ch.8 (the decode
ambiguity / aggressive clipping), Ch.10 (per-channel claim checked against real engine
data), Ch.13 (rated 2x ≠ measured 2x), Ch.15 (DIRECT_IO self-correction). Five reversals
across the runtime, and each one is something that genuinely happened during this
project, not manufactured for pacing.

---

## 2. The Full Scene List

Scene numbering is continuous across chapters. Estimated total runtime: ~16 minutes
(169 scenes after the pacing trim, ~5.6s average).

### Chapter 0 — Orientation: One Model, Three Precisions (Scenes 1–8)

```
Scene 1 — Chapter 0
Duration:            10s (estimate)
Goal:                Open on the real artifact, not a hook
Purpose:             Orient
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy speaking to camera, laptop/terminal visible in frame or
                     picture-in-picture. Lower-third text fades in: "tensorrt-precision-lab".
Narration (draft):   "This is a folder I've been running experiments in for the last
                     few days. Same ResNet50, same GPU, three separate TensorRT
                     engines. The only thing that changes between them is the number
                     format."
Transition:          Cut
```
```
Scene 2 — Chapter 0
Duration:            6s (estimate)
Goal:                Show the repo is real, not illustrative
Purpose:             Establish credibility
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Remotion-rendered terminal, `find . -maxdepth 2` typing in,
                     directory tree populating: models/, fp32/, fp16/, int8/,
                     calibration/, benchmarks/, profiling/.
Narration (draft):   "Every number I show you today came out of these folders.
                     Nothing here is made up for the video."
Transition:          Cut
```
```
Scene 3 — Chapter 0
Duration:            8s (estimate)
Goal:                State the question plainly, no teaser
Purpose:             Frame
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera. Text overlay lower-third: "FP32 → FP16 → INT8 —
                     what actually changes?"
Narration (draft):   "Here's what I want to figure out: what does 'precision' even
                     mean inside these files, and what does changing it really cost
                     you, and save you?"
Transition:          Cut
```
```
Scene 4 — Chapter 0
Duration:            7s (estimate)
Goal:                Show the headline table, unexplained
Purpose:             Set the destination without spoiling the mechanism
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Table renders in: Latency 1.808 / 0.645 / 0.347 ms; Engine size
                     107.94 / 49.14 / 25.26 MB; Top-1 1.0 / 1.0 / 0.9375.
Narration (draft):   "Here's where we end up. These are the real numbers I measured.
                     They probably don't mean much yet, but by the end you'll know
                     exactly why each one looks like this."
Transition:          Hard cut
```
```
Scene 5 — Chapter 0
Duration:            5s (estimate)
Goal:                Name the hardware/software honestly
Purpose:             Ground the numbers
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Small spec card: RTX 4050 Laptop GPU, driver 580.173.02, CUDA 13.0,
                     TensorRT 10.14.1.
Narration (draft):   "Quick disclaimer: it's just a laptop GPU. These exact numbers
                     are specific to this hardware. The reasoning behind them isn't,
                     though."
Transition:          Cut
```
```
Scene 6 — Chapter 0
Duration:            6s (estimate)
Goal:                Preview the path without giving away answers
Purpose:             Roadmap
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, three labeled icons animate in left-to-right:
                     "bits" → "scale" → "engine."
Narration (draft):   "Roughly how we're going to get there: start at the bit level,
                     work up to a scale and a calibration file, then go run these
                     engines."
Transition:          Cut
```
```
Scene 7 — Chapter 0
Duration:            5s (estimate)
Goal:                Set expectations: linear build, not a highlight reel
Purpose:             Frame tone
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, plain shot, no overlay.
Narration (draft):   "I'm not going to jump-cut to the punchline here. We're building
                     this the way you'd have to build it, step by step."
Transition:          Cut
```
```
Scene 8 — Chapter 0
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "1. What a float actually is."
Narration (draft):   (none — chapter card)
Transition:          Fade
```

### Chapter 1 — What A Float Actually Is (FP32) (Scenes 9–17)

Trimmed from 16 scenes to 9: one worked example instead of three, and the max/min
range is stated as a fact instead of derived bit-by-bit.

```
Scene 9 — Chapter 1
Duration:            6s (estimate)
Goal:                Pose the question
Purpose:             Frame
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Before we even get near INT8, I want to nail down one thing
                     properly. What IS a 32-bit float, at the bit level?"
Transition:          Cut
```
```
Scene 10 — Chapter 1
Duration:            8s (estimate)
Goal:                Show the 3-field layout and the reconstruction formula together
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      32 boxes animate into 3 groups, labeled sign (1 bit), exponent
                     (8 bits), mantissa (23 bits); the formula
                     x = (-1)^s × (1+f) × 2^(E-127) builds directly below it.
Narration (draft):   "Thirty-two bits, split into three jobs: a sign, an exponent,
                     and a fraction. One formula turns those bits back into a real
                     number."
Transition:          Cut
```
```
Scene 11 — Chapter 1
Duration:            7s (estimate)
Goal:                Explain sign and the exponent bias quickly, together
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Sign bit highlights briefly with (-1)^0/(-1)^1; then the
                     exponent field highlights with a "subtract 127" label.
Narration (draft):   "Sign's just plus or minus. The exponent's a little weirder. It
                     stores a shifted number, and you subtract 127 to get the real
                     exponent. That's the trick."
Transition:          Cut
```
```
Scene 12 — Chapter 1
Duration:            6s (estimate)
Goal:                Explain the implicit leading 1
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Fraction bits "01000..." shown; "1." prepended in a different color
                     with label "implicit — not stored."
Narration (draft):   "Here's a nice detail. You only store twenty-three fraction
                     bits, but you get twenty-four bits of precision, because that
                     leading 1 is just implied. It's free."
Transition:          Cut
```
```
Scene 13 — Chapter 1
Duration:            7s (estimate)
Goal:                One concrete worked example
Purpose:             Build understanding through a single concrete case
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Bit pattern 0|01111111|000...0 shown; each field highlighted in
                     sequence; result "= 1.0" lands.
Narration (draft):   "Let's try one. Sign's zero, exponent lands on zero, fraction's
                     all zeros. So it's just... one point zero."
Transition:          Cut
```
```
Scene 14 — Chapter 1
Duration:            7s (estimate)
Goal:                State the range as a fact instead of deriving every bound
Purpose:             Teach, keep pacing tight
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Number line stretches from a tiny tick near zero to a huge tick
                     far to the right, labeled "~10⁻³⁸" and "~10³⁸".
Narration (draft):   "Push that same formula to its limits and FP32 covers roughly
                     ten to the minus thirty-eight, all the way up to ten to the
                     plus thirty-eight. Huge range. We don't need to grind through
                     every edge case to know that."
Transition:          Cut
```
```
Scene 15 — Chapter 1
Duration:            6s (estimate)
Goal:                Summarize range vs precision distinction
Purpose:             Consolidate
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; two-word overlay: "RANGE" / "PRECISION" pinned
                     apart.
Narration (draft):   "Worth keeping these two things separate in your head: how big
                     a number can get, and how finely you can tell two numbers
                     apart. FP32's generous on both."
Transition:          Cut
```
```
Scene 16 — Chapter 1
Duration:            5s (estimate)
Goal:                Ask the pivot question
Purpose:             Bridge to Ch.2
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So if it's this generous, why would anyone ever want to shrink
                     it?"
Transition:          Cut
```
```
Scene 17 — Chapter 1
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "2. Shrinking the format — FP16."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 2 — Shrinking The Format (FP16) (Scenes 18–24)

Trimmed from 10 scenes to 7: the range-shrink explanation and the max-value payoff are
now one scene, and the "no calibration needed" point and the "it's just a format
conversion" point are now one scene.

```
Scene 18 — Chapter 2
Duration:            6s (estimate)
Goal:                Answer the "why shrink" question directly
Purpose:             Teach
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "The simple answer: half the bits means half the memory. And
                     on hardware built for it, roughly half the compute time too."
Transition:          Cut
```
```
Scene 19 — Chapter 2
Duration:            6s (estimate)
Goal:                Show the FP16 layout side-by-side with FP32
Purpose:             Compare
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Two stacked bit diagrams: FP32 (1/8/23) above FP16 (1/5/10),
                     aligned so the shrink is visually obvious.
Narration (draft):   "Same basic idea as FP32, just fewer bits in two of the three
                     fields."
Transition:          Cut
```
```
Scene 20 — Chapter 2
Duration:            7s (estimate)
Goal:                Show the range shrinking and land the max value together
Purpose:             Teach + payoff
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Exponent range visibly shrinks on a number line, then "65504"
                     lands next to FP32's "3.4×10³⁸" for scale contrast.
Narration (draft):   "Fewer exponent bits, much smaller range. Work through the
                     same math and FP16 tops out around sixty-five thousand. That's
                     it."
Transition:          Cut
```
```
Scene 21 — Chapter 2
Duration:            8s (estimate)
Goal:                Concrete overflow example
Purpose:             Make the consequence tangible
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      x = 100000 (FP32, fits fine) → arrow into FP16 box → "OVERFLOW →
                     +∞" stamps in red.
Narration (draft):   "So a number FP32 handles without blinking, say a hundred
                     thousand, just doesn't fit in FP16 at all."
Transition:          Cut
```
```
Scene 22 — Chapter 2
Duration:            7s (estimate)
Goal:                Critical clarifying fact, combined with the "it's just a format" point
Purpose:             Set up the contrast that drives the rest of the video
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay text: "NO scale. NO calibration."
Narration (draft):   "And notice, I didn't need anything extra for any of this. No
                     outside data. Those sixteen bits describe the number on their
                     own. It's just a format conversion."
Transition:          Cut
```
```
Scene 23 — Chapter 2
Duration:            6s (estimate)
Goal:                Pivot question toward INT8
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "But FP16 still has an exponent field in there. So what
                     happens when we go all the way down to 8 bits, and there's
                     just no room left for one at all?"
Transition:          Cut
```
```
Scene 24 — Chapter 2
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "3. Where floats stop working."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 3 — Where Floats Stop Working (Why INT8 Is Different) (Scenes 25–32)

Unchanged from the original cut, renumbered only.

```
Scene 25 — Chapter 3
Duration:            7s (estimate)
Goal:                State the naive expectation, then break it
Purpose:             Belief reversal #1
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay text fades in: "INT8 = an even smaller
                     float?" then a red strike-through crosses it.
Narration (draft):   "Honestly, my first assumption was that INT8 is basically FP16,
                     but smaller. It's not. It's not that at all."
Transition:          Cut
```
```
Scene 26 — Chapter 3
Duration:            6s (estimate)
Goal:                Show INT8 has no exponent field at all
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      8 boxes, no subdivisions, labeled simply "signed integer,
                     -128 to 127."
Narration (draft):   "INT8 is just a plain signed integer. There's no
                     sign-exponent-mantissa split anymore. It has no idea what
                     range it's supposed to represent."
Transition:          Cut
```
```
Scene 27 — Chapter 3
Duration:            7s (estimate)
Goal:                Pose the mapping problem visually
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      A continuous FP32 number line [-2.0, +2.0] compresses down onto
                     256 discrete tick marks labeled -128..127.
Narration (draft):   "So the problem becomes this: how do you squeeze a continuous
                     range of real numbers down into just two hundred fifty-six
                     integers?"
Transition:          Cut
```
```
Scene 28 — Chapter 3
Duration:            6s (estimate)
Goal:                Name the missing ingredient
Purpose:             Set up next chapter
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; single word lands: "SCALE."
Narration (draft):   "You need something extra. Some number, stored alongside the
                     integers, that says what one INT8 step is actually worth."
Transition:          Cut
```
```
Scene 29 — Chapter 3
Duration:            6s (estimate)
Goal:                Bridge into the repo — this isn't hypothetical
Purpose:             Ground the abstraction
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Remotion-rendered file tree zooms into `scripts/calibrate_int8.py`.
Narration (draft):   "This repo has a whole script dedicated to exactly that
                     problem. Nothing like it exists for the FP16 build."
Transition:          Cut
```
```
Scene 30 — Chapter 3
Duration:            8s (estimate)
Goal:                Show the code-asymmetry as visual proof
Purpose:             Payoff of the "different kind of thing" claim
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Split screen: `build_and_bench.py` (short, no calibration) vs
                     `calibrate_int8.py` (calibrator class, image loop) — visibly
                     different shapes.
Narration (draft):   "Look at the difference. One file just flips a builder flag.
                     The other one loads five hundred real images. That's the whole
                     asymmetry, right there in the code."
Transition:          Cut
```
```
Scene 31 — Chapter 3
Duration:            5s (estimate)
Goal:                Set the three-question agenda for the math section
Purpose:             Roadmap
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; three items list in: "range → scale →
                     mapping."
Narration (draft):   "Alright, let's build that scale ourselves, starting with the
                     easiest version of the problem."
Transition:          Cut
```
```
Scene 32 — Chapter 3
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "4. The simplest mapping — symmetric quantization."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 4 — The Simplest Mapping (Symmetric Quantization) (Scenes 33–40)

Trimmed from 12 scenes to 8: one worked value instead of two (drops the separate
0.5/-0.5 pass), and quantize + dequantize + error now land in a single scene instead
of three.

```
Scene 33 — Chapter 4
Duration:            6s (estimate)
Goal:                Set up the symmetric case
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Number line [-2.0, +2.0], visibly centered on 0.
Narration (draft):   "Easiest case first: a range that's already sitting centered
                     on zero."
Transition:          Cut
```
```
Scene 34 — Chapter 4
Duration:            7s (estimate)
Goal:                Derive the scale formula and land the numeric value together
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      s = max(|x|) / 127 builds; 2 / 127 substituted; ≈0.01575 lands.
Narration (draft):   "One number needed: take the largest absolute value, divide it
                     by a hundred twenty-seven. That gives us about 0.01575."
Transition:          Cut
```
```
Scene 35 — Chapter 4
Duration:            8s (estimate)
Goal:                Quantize one value, dequantize it, and show the error in one pass
Purpose:             Build understanding through a single concrete case, close the loop
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      1.0 → round(1.0/0.01575)=64 animates; then 64×0.01575≈1.008
                     lands next to the original 1.000, diff highlighted in a small
                     red box.
Narration (draft):   "Try it on one point zero. That becomes sixty-four. Multiply
                     back by the scale and you get about one point zero zero
                     eight. That little gap is the quantization error."
Transition:          Cut
```
```
Scene 36 — Chapter 4
Duration:            5s (estimate)
Goal:                Name why zero is special here
Purpose:             Consolidate
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "0.0 → 0 exactly."
Narration (draft):   "One nice thing: zero always lands exactly on zero here. No
                     offset needed."
Transition:          Cut
```
```
Scene 37 — Chapter 4
Duration:            8s (estimate)
Goal:                Build the code live — first live-code moment of the video
Purpose:             Curiosity seeded right before the resolving code (per format directive)
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Remotion-rendered editor: a 5-line Python function types in —
                     `def quantize_symmetric(x, s): return round(x/s)`.
Narration (draft):   "Okay, enough talking about it. Let me just write the actual
                     function."
Transition:          Cut
```
```
Scene 38 — Chapter 4
Duration:            6s (estimate)
Goal:                Run it, confirm it matches the hand math
Purpose:             Verify
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal: `quantize_symmetric(1.0, 0.01575)` → `64` prints.
Narration (draft):   "Run it, and yep, same sixty-four we got by hand."
Transition:          Cut
```
```
Scene 39 — Chapter 4
Duration:            6s (estimate)
Goal:                Preview reuse of the function later, then pivot to the asymmetric case
Purpose:             Bridge, set up Chapter 9's real-data check
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; number line with min=-1, max=+3 fades in
                     visibly off-center behind her.
Narration (draft):   "Hang onto this function, we'll reuse it on a real activation
                     soon. But first: what happens when the real range isn't
                     centered on zero at all?"
Transition:          Cut
```
```
Scene 40 — Chapter 4
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "5. When zero isn't in the middle."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 5 — When Zero Isn't In The Middle (Asymmetric Quantization) (Scenes 41–47)

Trimmed from 10 scenes to 7: scale and zero-point are derived in one scene instead of
two, and the code-build/run pair is now a single scene.

```
Scene 41 — Chapter 5
Duration:            7s (estimate)
Goal:                Show the waste from forcing symmetry
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Range [-1, +3] shown, then a wider forced-symmetric [-3, +3] box
                     overlays it, with the wasted [-3,-1] region shaded.
Narration (draft):   "If you force this into a symmetric range, you end up wasting
                     something like a third of your INT8 codes on values that never
                     even show up."
Transition:          Cut
```
```
Scene 42 — Chapter 5
Duration:            8s (estimate)
Goal:                Derive the asymmetric scale and the zero-point together
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      s = (max-min)/(qmax-qmin) = 4/255 ≈ 0.01569 builds; then
                     z = round(qmin - min/s) ≈ -64 builds beside it, arrow points
                     from FP32 zero to INT8 -64.
Narration (draft):   "So the scale spans the whole range instead, across all two
                     hundred fifty-five steps. And now zero doesn't land on integer
                     zero anymore, it lands wherever this zero-point number says it
                     should."
Transition:          Cut
```
```
Scene 43 — Chapter 5
Duration:            6s (estimate)
Goal:                Show the full formula with zero-point
Purpose:             Consolidate
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      q = round(x/s) + z lands beside the symmetric q = round(x/s) for
                     direct comparison.
Narration (draft):   "One extra term added on. That's the whole difference between
                     the two."
Transition:          Cut
```
```
Scene 44 — Chapter 5
Duration:            6s (estimate)
Goal:                Name a realistic case where this matters
Purpose:             Ground the abstraction
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "ReLU output ≥ 0."
Narration (draft):   "And this comes up constantly in real networks. Anything right
                     after a ReLU is never negative."
Transition:          Cut
```
```
Scene 45 — Chapter 5
Duration:            7s (estimate)
Goal:                Live-code the asymmetric function and confirm it, in one pass
Purpose:             Reinforce build-along format
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Editor types `def quantize_asymmetric(x, s, z): return
                     round(x/s) + z`; terminal confirms output matches the
                     hand-derived value.
Narration (draft):   "Barely different code, just one extra piece, and it matches
                     what we got by hand."
Transition:          Cut
```
```
Scene 46 — Chapter 5
Duration:            5s (estimate)
Goal:                Pivot to the "how many scales" question
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Okay, one scale per tensor. But is one scale always enough?"
Transition:          Cut
```
```
Scene 47 — Chapter 5
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "6. One scale, or many?"
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 6 — One Scale Or Many? (Per-Tensor vs Per-Channel) (Scenes 48–57)

Unchanged from the original cut, renumbered only.

```
Scene 48 — Chapter 6
Duration:            7s (estimate)
Goal:                Show the problem case
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Two channel ranges shown stacked: Channel 1 [-0.3,+0.3], Channel 2
                     [-10,+8] — visibly very different scales.
Narration (draft):   "Picture one weight tensor where one channel's values are
                     tiny, and a different channel's values are huge."
Transition:          Cut
```
```
Scene 49 — Chapter 6
Duration:            6s (estimate)
Goal:                Show per-tensor's failure mode
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      One shared scale forces channel 1's tiny values into a handful of
                     INT8 codes near zero, visibly wasted resolution.
Narration (draft):   "If you use one shared scale for the whole thing, that small
                     channel basically gets no usable resolution at all."
Transition:          Cut
```
```
Scene 50 — Chapter 6
Duration:            6s (estimate)
Goal:                Introduce per-channel as the fix
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Same two channels now each get their own scale, both filling
                     their INT8 range fully.
Narration (draft):   "But give each channel its own scale, and now both of them get
                     to use their full range."
Transition:          Cut
```
```
Scene 51 — Chapter 6
Duration:            7s (estimate)
Goal:                Introduce the claim about TensorRT specifically
Purpose:             Set up the investigation, honestly flagged
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay text: "NVIDIA docs say: weights =
                     per-channel, activations = per-tensor."
Narration (draft):   "NVIDIA's own docs say TensorRT does exactly this
                     automatically for weights, while activations still just get
                     one scale each, from the calibrator."
Transition:          Cut
```
```
Scene 52 — Chapter 6
Duration:            7s (estimate)
Goal:                Flag the epistemic status explicitly, on camera
Purpose:             Model intellectual honesty (dossier Skeptic section)
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay stamp: "DOCUMENTED — NOT YET VERIFIED
                     IN THIS ENGINE."
Narration (draft):   "Now, I haven't checked that inside this specific engine yet.
                     So instead of just repeating what the docs say, let's go look
                     for ourselves."
Transition:          Cut
```
```
Scene 53 — Chapter 6
Duration:            6s (estimate)
Goal:                Name the tool that will answer it
Purpose:             Roadmap
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `engine.create_engine_inspector()` line highlights in
                     `profile_layers.py`.
Narration (draft):   "TensorRT ships a tool for exactly this kind of question. It's
                     called the engine inspector."
Transition:          Cut
```
```
Scene 54 — Chapter 6
Duration:            5s (estimate)
Goal:                Defer the check, not forget it
Purpose:             Set audience expectation
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Answered in Chapter 10."
Narration (draft):   "We'll answer this properly once the engine's built, a few
                     chapters from now. Promise I won't forget."
Transition:          Cut
```
```
Scene 55 — Chapter 6
Duration:            6s (estimate)
Goal:                Summarize the two axes so far
Purpose:             Consolidate
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      2x2 grid fills in: symmetric/asymmetric × per-tensor/per-channel.
Narration (draft):   "So really there's two separate choices here: where the
                     scale's centered, and how many scales you're using."
Transition:          Cut
```
```
Scene 56 — Chapter 6
Duration:            5s (estimate)
Goal:                Pivot to where scale values come from for activations
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "None of this tells us where these numbers come from for
                     activations, though. That's calibration."
Transition:          Cut
```
```
Scene 57 — Chapter 6
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "7. Where the scale actually comes from —
                     calibration."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 7 — Where The Scale Actually Comes From (Calibration) (Scenes 58–69)

Unchanged from the original cut, renumbered only.

```
Scene 58 — Chapter 7
Duration:            6s (estimate)
Goal:                State the activation problem
Purpose:             Teach
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Weights, you can just go inspect ahead of time. Activations
                     are trickier. They depend entirely on whatever input you feed
                     the network."
Transition:          Cut
```
```
Scene 59 — Chapter 7
Duration:            7s (estimate)
Goal:                Introduce the calibration pipeline visually
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Flow diagram: calibration images → run network → observe ranges
                     → scales → INT8 engine.
Narration (draft):   "So the idea is, you run real data through the network once,
                     just to watch what ranges show up."
Transition:          Cut
```
```
Scene 60 — Chapter 7
Duration:            7s (estimate)
Goal:                Show the actual calibration dataset in this repo
Purpose:             Ground the abstraction
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `calibration/images/` folder listing scrolls, real JPEG
                     filenames visible (Imagenette).
Narration (draft):   "In this repo it's five hundred real JPEGs, fifty per class
                     across ten classes. Not synthetic noise."
Transition:          Cut
```
```
Scene 61 — Chapter 7
Duration:            7s (estimate)
Goal:                Raise the naive approach: why not just min/max
Purpose:             Set up the reversal
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Just use min/max?"
Narration (draft):   "Simplest thing you'd try: just track the min and max you see
                     and scale to that. Turns out TensorRT doesn't do that by
                     default."
Transition:          Cut
```
```
Scene 62 — Chapter 7
Duration:            8s (estimate)
Goal:                Show why min/max fails with an outlier
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Distribution of small values 0.01–0.6 shown dense near zero, with
                     one outlier point at 100.0 far to the right; scale bar stretches
                     to cover it, crushing the dense region into a sliver.
Narration (draft):   "One weird outlier value blows the whole range out, and
                     suddenly every normal value gets crammed into a tiny sliver of
                     INT8's resolution."
Transition:          Cut
```
```
Scene 63 — Chapter 7
Duration:            7s (estimate)
Goal:                Introduce entropy calibration as the fix
Purpose:             Teach, resolve the reversal
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Two overlapping histograms (original float distribution vs
                     quantized) with a "KL divergence" label between them, minimized.
Narration (draft):   "So instead, entropy calibration picks a threshold that keeps
                     the quantized distribution as close as possible to the
                     original, even if that means deliberately clipping the extreme
                     outlier."
Transition:          Cut
```
```
Scene 64 — Chapter 7
Duration:            5s (estimate)
Goal:                Name the source
Purpose:             Credibility
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Citation card: "Szymon Migacz, NVIDIA — '8-bit Inference with
                     TensorRT,' 2017."
Narration (draft):   "This isn't some new trick, either. It's a method NVIDIA
                     published back in 2017."
Transition:          Cut
```
```
Scene 65 — Chapter 7
Duration:            7s (estimate)
Goal:                Show the actual calibrator class in this repo
Purpose:             Ground the abstraction, build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `class ImageCalibrator(trt.IInt8EntropyCalibrator2):` and
                     `get_batch()` highlighted in `calibrate_int8.py`.
Narration (draft):   "Here's the actual class from the repo. Every batch it hands
                     over to TensorRT is eight real, preprocessed images."
Transition:          Cut
```
```
Scene 66 — Chapter 7
Duration:            6s (estimate)
Goal:                Show the preprocessing detail
Purpose:             Ground the abstraction
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `preprocess_image()` highlighted: resize→crop→normalize→CHW
                     steps labeled.
Narration (draft):   "Resize, center-crop, normalize with ImageNet stats. Literally
                     the same preprocessing the model expects at inference time."
Transition:          Cut
```
```
Scene 67 — Chapter 7
Duration:            6s (estimate)
Goal:                Show the cache read/write hooks
Purpose:             Set up Chapter 8
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `write_calibration_cache()` highlighted, arrow points forward to
                     a file icon labeled "calibration.cache".
Narration (draft):   "And once calibration's done, it just writes everything it
                     learned out to one file."
Transition:          Cut
```
```
Scene 68 — Chapter 7
Duration:            5s (estimate)
Goal:                Pose the next question
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So, 'calibration produces a scale per tensor.' Is that just a
                     concept I'm describing, or can I show you it?"
Transition:          Cut
```
```
Scene 69 — Chapter 7
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "8. Opening the calibration file."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 8 — Opening The Calibration File (Scenes 70–79)

Unchanged from the original cut, renumbered only.

```
Scene 70 — Chapter 8
Duration:            6s (estimate)
Goal:                Open the raw file
Purpose:             Reveal, ground the abstraction (Experiment Plan #1)
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal: `cat calibration/calibration.cache | head -5` prints
                     plain text lines "tensor_name: hex".
Narration (draft):   "Let's just open it up. And it's not some binary blob. It's
                     literally a plain text file."
Transition:          Cut
```
```
Scene 71 — Chapter 8
Duration:            5s (estimate)
Goal:                Count the lines
Purpose:             Scale the claim
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `wc -l calibration.cache` → `128`.
Narration (draft):   "One line per tensor, across this whole fifty-eight-layer
                     graph."
Transition:          Cut
```
```
Scene 72 — Chapter 8
Duration:            6s (estimate)
Goal:                Pick one line to decode
Purpose:             Focus
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Line highlights: `input: 3cd56a3e`.
Narration (draft):   "Let's decode one of these. Let's do the input tensor."
Transition:          Cut
```
```
Scene 73 — Chapter 8
Duration:            8s (estimate)
Goal:                Live-write the decoder
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Editor types: `bits = int("3cd56a3e", 16)` then
                     `struct.unpack('>f', struct.pack('>I', bits))`.
Narration (draft):   "Should only take about five lines: read the hex as an
                     integer, then reinterpret those exact same bits as a float."
Transition:          Cut
```
```
Scene 74 — Chapter 8
Duration:            5s (estimate)
Goal:                Run it, get a number
Purpose:             Payoff
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal prints `0.0260516...`.
Narration (draft):   "Okay, run it. 0.026. So that's apparently the calibrated
                     dynamic range for the input tensor."
Transition:          Cut
```
```
Scene 75 — Chapter 8
Duration:            8s (estimate)
Goal:                Sanity-check against the known expected range — trigger the reversal
Purpose:             Belief reversal #2, model verification discipline
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, visibly pausing on the number; overlay:
                     "Expected ~±2.1 to ±2.6. Got 0.026?"
Narration (draft):   "Wait, hang on. This input's ImageNet-normalized, its real
                     range should be something like plus or minus two and a half.
                     0.026 is way off from that. Did I mess up the decode?"
Transition:          Cut
```
```
Scene 76 — Chapter 8
Duration:            8s (estimate)
Goal:                Show the ambiguity is real, not a mistake to hide
Purpose:             Skeptic-mode honesty beat
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Same 8 hex characters decoded two mechanically-valid ways side by
                     side: 0.0261 vs 0.2293, both flagged "valid interpretation."
Narration (draft):   "So I tried it the other byte-order way too, and honestly both
                     readings are defensible. They give two different numbers, and
                     neither one matches what I expected."
Transition:          Cut
```
```
Scene 77 — Chapter 8
Duration:            8s (estimate)
Goal:                Resolve with the real explanation, not hand-waving
Purpose:             Resolution
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Entropy calibration clips outliers
                     — on purpose."
Narration (draft):   "But this is exactly what entropy calibration is supposed to
                     do. It can pick a threshold way below the true max if that
                     gives better resolution for most of the data. Small doesn't
                     mean broken here."
Transition:          Cut
```
```
Scene 78 — Chapter 8
Duration:            5s (estimate)
Goal:                State the honest takeaway
Purpose:             Consolidate the reversal
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Honestly, the lesson isn't the exact number. It's that when
                     something looks surprising, you go check it instead of
                     trusting the first plausible-looking answer."
Transition:          Cut
```
```
Scene 79 — Chapter 8
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "9. From toy numbers to real activations."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 9 — From Toy Numbers To Real Activations (Scenes 80–83)

Trimmed from 10 scenes to 4. The original version re-derived the whole
range-scale-quantize-dequantize-error loop on the toy array a second time before
touching real data, which is exactly the kind of repetition the pacing trim was meant
to remove. Chapter 4 already fully teaches that loop; this chapter's only job is to
check it against something real.

```
Scene 80 — Chapter 9
Duration:            6s (estimate)
Goal:                Pose the "is this really what happens inside the model" question
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, small thumbnail of the Chapter 4 toy array
                     beside her.
Narration (draft):   "All that math, does the real model actually produce numbers
                     like this? Let's find out instead of just assuming."
Transition:          Cut
```
```
Scene 81 — Chapter 9
Duration:            8s (estimate)
Goal:                Pull a real activation slice and run the existing function on it
Purpose:             Build-along, ground the toy math in reality
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      A short PyTorch forward-hook snippet on `resnet50.pth` captures
                     one conv layer's output tensor; `quantize_symmetric` (from
                     Chapter 4) runs on it directly; output prints.
Narration (draft):   "Grab a real slice of activations straight out of this
                     ResNet50, and run the exact same function on it. Same eight
                     lines. Real numbers."
Transition:          Cut
```
```
Scene 82 — Chapter 9
Duration:            7s (estimate)
Goal:                Consolidate the point, then pivot back to the deferred per-channel question
Purpose:             Land the takeaway, bridge to Chapter 10
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Now — that open question from
                     Chapter 6."
Narration (draft):   "No separate, more-real version of the math hiding anywhere.
                     This is it. Time to go check what TensorRT actually did with
                     per-channel scaling, that question from a few chapters back."
Transition:          Cut
```
```
Scene 83 — Chapter 9
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "10. Checking TensorRT's own homework."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 10 — Checking TensorRT's Own Homework (Engine Inspector Audit) (Scenes 84–93)

Unchanged from the original cut, renumbered only.

```
Scene 84 — Chapter 10
Duration:            6s (estimate)
Goal:                Restate the exact question being tested
Purpose:             Frame
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay recalls: "Weights: per-channel?
                     Activations: per-tensor?"
Narration (draft):   "Here's the claim I want to test: weights get their own scale
                     per channel, activations get just one scale for the whole
                     tensor."
Transition:          Cut
```
```
Scene 85 — Chapter 10
Duration:            7s (estimate)
Goal:                Show the tool
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `engine.create_engine_inspector()` and
                     `inspector.get_layer_information(i, JSON)` highlighted in
                     `profile_layers.py`.
Narration (draft):   "Turns out TensorRT will hand you the full JSON description of
                     any layer it built, if you ask it nicely."
Transition:          Cut
```
```
Scene 86 — Chapter 10
Duration:            7s (estimate)
Goal:                Pick one conv layer to inspect
Purpose:             Focus
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      One `CaskConvolution` layer from the INT8 engine highlighted in
                     a layer list.
Narration (draft):   "Let's pull the full record for one actual INT8 convolution
                     layer."
Transition:          Cut
```
```
Scene 87 — Chapter 10
Duration:            8s (estimate)
Goal:                Show the JSON dump
Purpose:             Reveal
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Raw JSON scrolls, weight-scale field highlighted showing an
                     array of values (one per output channel) vs. the input-tensor
                     scale field showing a single value.
Narration (draft):   "Look at the shapes here. The weight scale is an array, one
                     number per output channel. The activation scale is just a
                     single number."
Transition:          Cut
```
```
Scene 88 — Chapter 10
Duration:            6s (estimate)
Goal:                Confirm the claim explicitly
Purpose:             Resolution of Ch.6's open question
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay stamp changes from "DOCUMENTED — NOT
                     YET VERIFIED" to "CONFIRMED, this engine."
Narration (draft):   "So here's the confirmation. This specific engine really is
                     mixing per-channel weights with per-tensor activations,
                     exactly like the docs describe."
Transition:          Cut
```
```
Scene 89 — Chapter 10
Duration:            6s (estimate)
Goal:                Note the honest caveat if it were NOT confirmed as expected
Purpose:             Preserve integrity regardless of outcome (director instruction)
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; small footnote overlay: "If your own engine
                     inspector shows something different — trust that, not this
                     video."
Narration (draft):   "If you run this yourself on a different model or TensorRT
                     version and get something different, that's kind of the whole
                     point of checking it yourself. Trust your own inspector output
                     over mine."
Transition:          Cut
```
```
Scene 90 — Chapter 10
Duration:            6s (estimate)
Goal:                Consolidate the full mapping of concepts to this repo
Purpose:             Recap before the benchmark payoff
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Full 2x2 grid from Ch.6 re-appears, each quadrant now stamped
                     with a checkmark and a repo file reference.
Narration (draft):   "So at this point, every concept from the first half of this
                     video is tied to a real file sitting in this repo."
Transition:          Cut
```
```
Scene 91 — Chapter 10
Duration:            5s (estimate)
Goal:                Reframe the remaining agenda
Purpose:             Roadmap
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Which brings up the real question: how much does any of this
                     help?"
Transition:          Cut
```
```
Scene 92 — Chapter 10
Duration:            5s (estimate)
Goal:                Name the next script
Purpose:             Bridge
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `scripts/build_and_bench.py` highlighted in the file tree.
Narration (draft):   "Time to go run the benchmark."
Transition:          Cut
```
```
Scene 93 — Chapter 10
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "11. The benchmark."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 11 — The Benchmark (build_and_bench.py) (Scenes 94–105)

Unchanged from the original cut, renumbered only.

```
Scene 94 — Chapter 11
Duration:            6s (estimate)
Goal:                Show the benchmark methodology briefly
Purpose:             Credibility before the reveal
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `benchmark()` function highlighted: 20 warmup + 200 timed iters,
                     CUDA events.
Narration (draft):   "Quick note on how this is measured: twenty warmup runs, two
                     hundred timed ones, using CUDA events, not just a Python
                     stopwatch."
Transition:          Cut
```
```
Scene 95 — Chapter 11
Duration:            6s (estimate)
Goal:                Run the FP32 build/benchmark
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal: `Building fp32 engine...` then `fp32: {'throughput_qps':
                     553.24, 'gpu_latency_mean_ms': 1.808}`.
Narration (draft):   "Alright, FP32 first. One point eight milliseconds."
Transition:          Cut
```
```
Scene 96 — Chapter 11
Duration:            6s (estimate)
Goal:                Run the FP16 build/benchmark
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal: `fp16: {'throughput_qps': 1551.03, 'gpu_latency_mean_ms':
                     0.645}`.
Narration (draft):   "FP16. About zero point six five milliseconds. Almost three
                     times faster already."
Transition:          Cut
```
```
Scene 97 — Chapter 11
Duration:            6s (estimate)
Goal:                Run/reuse the INT8 benchmark
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal: `int8: {'throughput_qps': 2880.55, 'gpu_latency_mean_ms':
                     0.347}`.
Narration (draft):   "And INT8. About zero point three five."
Transition:          Cut
```
```
Scene 98 — Chapter 11
Duration:            7s (estimate)
Goal:                Land the full comparison, with a genuine reaction
Purpose:             Emotional payoff, placed after understanding (per format directive)
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, table overlays beside: 1.808 / 0.645 / 0.347
                     ms, bar chart animating in proportion.
Narration (draft):   "So FP32 to INT8, that's five times faster. On the exact same
                     weights, same model."
Transition:          Cut
```
```
Scene 99 — Chapter 11
Duration:            6s (estimate)
Goal:                Show engine size and memory too
Purpose:             Broaden the payoff beyond latency
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Engine size bars: 107.94 / 49.14 / 25.26 MB; memory bars: 140 /
                     72 / 52 MB.
Narration (draft):   "And latency's only part of it. Smaller on disk, smaller in
                     GPU memory too. Both roughly halve at each step."
Transition:          Cut
```
```
Scene 100 — Chapter 11
Duration:            6s (estimate)
Goal:                Note build time tradeoff honestly
Purpose:             Balance the picture
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Build time bars: 18.6 / 40.4 / 64.8s (or 85.3s cold).
Narration (draft):   "The one thing that gets worse is build time. Makes sense,
                     INT8 has to calibrate first."
Transition:          Cut
```
```
Scene 101 — Chapter 11
Duration:            6s (estimate)
Goal:                Do the naive math out loud
Purpose:             Set up the reversal
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, doing mental math on screen: "FP16→INT8 should
                     be another 2x..."
Narration (draft):   "Okay so, FP16 to INT8 halves the bits again. So logically,
                     this step should roughly double the speed too, right?"
Transition:          Cut
```
```
Scene 102 — Chapter 11
Duration:            6s (estimate)
Goal:                Show the actual ratio, breaking the expectation
Purpose:             Belief reversal setup (payoff in Ch.13)
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      0.645 / 0.347 = 1.86x computed on screen, next to a ghosted
                     "expected: 2x".
Narration (draft):   "Let's check. About one point nine x. Close. Not quite two
                     x."
Transition:          Cut
```
```
Scene 103 — Chapter 11
Duration:            5s (estimate)
Goal:                Refuse to hand-wave the gap
Purpose:             Frame the investigation ahead
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "That gap's small enough you could just shrug it off. I don't
                     want to shrug it off."
Transition:          Cut
```
```
Scene 104 — Chapter 11
Duration:            5s (estimate)
Goal:                Pre-empt the "is it even really INT8" objection
Purpose:             Bridge, sets up Ch.12
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "But before I go blaming hardware limits, let's rule out
                     something dumber first. Is this engine even really running in
                     INT8?"
Transition:          Cut
```
```
Scene 105 — Chapter 11
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "12. Is INT8 actually running in INT8?"
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 12 — Is INT8 Actually Running In INT8? (Scenes 106–115)

Unchanged from the original cut, renumbered only.

```
Scene 106 — Chapter 12
Duration:            6s (estimate)
Goal:                State why this check matters
Purpose:             Teach
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Because setting a builder flag that says 'use INT8' doesn't
                     prove every layer got an INT8 kernel."
Transition:          Cut
```
```
Scene 107 — Chapter 12
Duration:            6s (estimate)
Goal:                Introduce Nsight Systems as ground truth
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `nsys profile ... nsys_run_engine.py` command highlighted.
Narration (draft):   "So let's use Nsight Systems. It traces the literal CUDA
                     kernels that ran, no TensorRT instrumentation getting in the
                     way."
Transition:          Cut
```
```
Scene 108 — Chapter 12
Duration:            7s (estimate)
Goal:                Show a real kernel name
Purpose:             Payoff — the evidence itself
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Long kernel name from the trace types in:
                     `sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32_nchw...`,
                     `i8i8` substring highlighted.
Narration (draft):   "'i8i8.' Right there in the kernel name. That's the actual
                     kernel that ran on the GPU. I didn't have to take TensorRT's
                     word for it."
Transition:          Cut
```
```
Scene 109 — Chapter 12
Duration:            6s (estimate)
Goal:                Show the aggregate count
Purpose:             Quantify the claim
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      "17,280 / 17,920 kernel launches = 96.4%" lands.
Narration (draft):   "Across three hundred iterations, about ninety-six percent of
                     every single kernel launch genuinely runs on integer tensor
                     cores."
Transition:          Cut
```
```
Scene 110 — Chapter 12
Duration:            7s (estimate)
Goal:                Show what the remaining fraction actually is
Purpose:             Completeness, avoids overclaiming
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Engine inspector output: 58 layers, 2 flagged "Float" —
                     `node_linear` bias-add and its reshape.
Narration (draft):   "The only two layers that stay in float are the tiny final
                     classifier bias-add. TensorRT chose to leave those two in
                     full precision on purpose."
Transition:          Cut
```
```
Scene 111 — Chapter 12
Duration:            6s (estimate)
Goal:                Explain why that layer is exempted
Purpose:             Teach
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "0.034ms out of ~0.45ms."
Narration (draft):   "Makes sense, actually. It's cheap to keep at full precision,
                     and it directly produces the logits your argmax depends on.
                     Worth protecting."
Transition:          Cut
```
```
Scene 112 — Chapter 12
Duration:            6s (estimate)
Goal:                Rule out the fallback explanation for the 1.7x gap
Purpose:             Close this sub-investigation cleanly
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So that missing 0.3x isn't hiding in some fallback layer.
                     It's got to be something else."
Transition:          Cut
```
```
Scene 113 — Chapter 12
Duration:            6s (estimate)
Goal:                Also rule out reformat overhead
Purpose:             Narrow the search
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Kernel-category pie/bar: Conv/GEMM 95%, Reformat 1.4%, Pooling
                     3.6%.
Narration (draft):   "And it's not reformat overhead either. That's like one and a
                     half percent of GPU time. Barely anything."
Transition:          Cut
```
```
Scene 114 — Chapter 12
Duration:            5s (estimate)
Goal:                Pose the narrowed question
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So it's ninety-five percent compute-bound, genuinely running
                     INT8 kernels. So why isn't it two x?"
Transition:          Cut
```
```
Scene 115 — Chapter 12
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "13. Why isn't it 4x?"
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 13 — Why Isn't It 4x? (Scenes 116–127)

Unchanged from the original cut, renumbered only.

```
Scene 116 — Chapter 13
Duration:            6s (estimate)
Goal:                Show the ground-truth Nsight per-precision times
Purpose:             Establish the real numbers, separate from earlier CUDA-event numbers
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Table: GPU kernel time 2.008 / 0.756 / 0.446 ms; ratios 2.65x,
                     1.70x.
Narration (draft):   "Pure GPU kernel time, no profiler overhead in the way: about
                     two point six x for the first step, one point seven x for the
                     second."
Transition:          Cut
```
```
Scene 117 — Chapter 13
Duration:            6s (estimate)
Goal:                State the hardware promise explicitly
Purpose:             Sharpen the mystery
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Tensor cores: rated ~2x INT8 vs
                     FP16 throughput."
Narration (draft):   "These tensor cores are literally rated for double the FP16
                     throughput in INT8. On paper this should've been the easy
                     step."
Transition:          Cut
```
```
Scene 118 — Chapter 13
Duration:            6s (estimate)
Goal:                Introduce the key word: peak vs. achieved
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Two bars: "peak throughput" (tall, theoretical) vs "achieved
                     throughput" (shorter, real), gap shaded and labeled.
Narration (draft):   "But 'rated for 2x' is a peak number. And peak numbers assume
                     the kernel runs long enough to get there."
Transition:          Cut
```
```
Scene 119 — Chapter 13
Duration:            7s (estimate)
Goal:                Show the batch=1 problem concretely
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      ResNet50 conv layer diagram at batch=1 shown as a small tile that
                     doesn't fill a larger tensor-core tile outline.
Narration (draft):   "At batch size one, most of ResNet50's individual conv layers
                     are just small. Too small to fully fill a tensor-core tile."
Transition:          Cut
```
```
Scene 120 — Chapter 13
Duration:            7s (estimate)
Goal:                Explain fixed launch overhead
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Timeline bar: [launch overhead][compute] shown at two different
                     compute lengths (FP16 long bar, INT8 half-length bar) — overhead
                     segment stays the same fixed width both times.
Narration (draft):   "Every kernel launch has some fixed setup cost attached to
                     it. Halving the math time doesn't halve that fixed cost."
Transition:          Cut
```
```
Scene 121 — Chapter 13
Duration:            6s (estimate)
Goal:                Show the real per-kernel duration numbers
Purpose:             Ground the explanation in data
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      "Individual INT8 kernel durations: ~4,000–17,000 ns, hundreds of
                     small conv layers per inference."
Narration (draft):   "We're talking microsecond-scale kernels here. That fixed
                     overhead eats up a much bigger fraction of a small kernel than
                     a big one."
Transition:          Cut
```
```
Scene 122 — Chapter 13
Duration:            6s (estimate)
Goal:                Explain why FP32→FP16 got closer to 2x by contrast
Purpose:             Complete the explanation symmetrically
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "FP32 kernels start further from the
                     overhead floor."
Narration (draft):   "Which is also why FP32 got closer to a clean 2x earlier. Its
                     kernels were slower to begin with, more room to cut before
                     hitting that same floor."
Transition:          Cut
```
```
Scene 123 — Chapter 13
Duration:            6s (estimate)
Goal:                Consolidate the full explanation
Purpose:             Resolution
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay recap: "Compute-bound overall. Launch-
                     overhead-bound per layer at batch=1."
Narration (draft):   "So overall, the model's compute-bound. But at batch one, a
                     lot of individual layers are launch-overhead-bound instead."
Transition:          Cut
```
```
Scene 124 — Chapter 13
Duration:            6s (estimate)
Goal:                Pose the actionable question
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So if launch overhead's the real limiter here, can we get rid
                     of it, without touching the model at all?"
Transition:          Cut
```
```
Scene 125 — Chapter 13
Duration:            5s (estimate)
Goal:                Name the technique
Purpose:             Roadmap
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      "CUDA Graphs" title card, small icon of a linked kernel chain.
Narration (draft):   "Turns out, yeah. CUDA graphs."
Transition:          Cut
```
```
Scene 126 — Chapter 13
Duration:            5s (estimate)
Goal:                Flag this is testable, not theoretical
Purpose:             Set expectations for build-along
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "And this repo's already got both code paths, so we can just
                     compare them directly."
Transition:          Cut
```
```
Scene 127 — Chapter 13
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "14. Removing the launch overhead — CUDA graphs."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 14 — Removing The Launch Overhead (CUDA Graphs) (Scenes 128–137)

Unchanged from the original cut, renumbered only.

```
Scene 128 — Chapter 14
Duration:            7s (estimate)
Goal:                Explain the mechanism
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Diagram: ~100+ individual kernel launches (each with its own
                     overhead icon) collapse into one "graph launch" arrow.
Narration (draft):   "Idea is: capture the whole sequence of per-layer kernel
                     launches just once, then replay it as a single graph launch
                     every time after that."
Transition:          Cut
```
```
Scene 129 — Chapter 14
Duration:            6s (estimate)
Goal:                Show the exact code toggle
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `benchmark(..., use_cuda_graph=True)` parameter highlighted in
                     `build_and_bench.py`.
Narration (draft):   "It's already sitting right here in the repo's own benchmark
                     function. One flag."
Transition:          Cut
```
```
Scene 130 — Chapter 14
Duration:            6s (estimate)
Goal:                Show the capture/replay code briefly
Purpose:             Ground the mechanism
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `cudaStreamBeginCapture` / `cudaGraphInstantiate` /
                     `cudaGraphLaunch` lines highlighted in sequence.
Narration (draft):   "Capture once, instantiate once, then just launch it over and
                     over."
Transition:          Cut
```
```
Scene 131 — Chapter 14
Duration:            6s (estimate)
Goal:                Verify bit-identical output before trusting the speed number
Purpose:             Model rigor
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "max abs diff = 0.0 — verified."
Narration (draft):   "Before I trust any speed number out of this, worth saying:
                     it's already been checked bit-identical against the normal
                     path."
Transition:          Cut
```
```
Scene 132 — Chapter 14
Duration:            7s (estimate)
Goal:                Toggle and show fp32 result
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal shows before/after fp32 latency, delta "+3-7%"
                     highlighted.
Narration (draft):   "FP32 first. Small gain, three to seven percent."
Transition:          Cut
```
```
Scene 133 — Chapter 14
Duration:            6s (estimate)
Goal:                Show fp16 result
Purpose:             Build-along
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Terminal delta "+11-14%" highlighted.
Narration (draft):   "FP16's bigger. Eleven to fourteen percent."
Transition:          Cut
```
```
Scene 134 — Chapter 14
Duration:            7s (estimate)
Goal:                Show int8 result — the confirming payoff
Purpose:             Resolution of Ch.13's investigation
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, visibly satisfied; overlay: "INT8: +19-30%."
Narration (draft):   "And INT8, the one with the smallest, most overhead-bound
                     kernels, gets the biggest win of all. Which basically
                     confirms the whole theory."
Transition:          Cut
```
```
Scene 135 — Chapter 14
Duration:            6s (estimate)
Goal:                Show the ratio improvement
Purpose:             Close the loop from Ch.11's setup
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      "1.70x → 1.86x" animates, moving visibly closer to a ghosted
                     "2.0x" target line.
Narration (draft):   "So the FP16-to-INT8 ratio moves, from one point seven up to
                     almost one point nine."
Transition:          Cut
```
```
Scene 136 — Chapter 14
Duration:            5s (estimate)
Goal:                Note it's real but partial, not a full fix
Purpose:             Prevent overclaiming
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Closer to two x. Not all the way there. That batch-size
                     limit is still real."
Transition:          Cut
```
```
Scene 137 — Chapter 14
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "15. The flag that wasn't as good as it looked."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 15 — The Flag That Wasn't As Good As It Looked (DIRECT_IO) (Scenes 138–147)

Unchanged from the original cut, renumbered only.

```
Scene 138 — Chapter 15
Duration:            6s (estimate)
Goal:                Introduce the second optimization
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `config.set_flag(trt.BuilderFlag.DIRECT_IO)` highlighted.
Narration (draft):   "One more thing was tried: a flag called DIRECT_IO. It
                     removes a reformat layer sitting right at the input boundary."
Transition:          Cut
```
```
Scene 139 — Chapter 15
Duration:            6s (estimate)
Goal:                Recall what motivated trying it
Purpose:             Connect to earlier finding
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Callback to the "biggest layer isn't a conv, it's a reformat"
                     finding, small thumbnail from earlier chapter.
Narration (draft):   "This came from something we noticed earlier. Remember, the
                     biggest single layer in the INT8 profile wasn't a convolution,
                     it was an input reformat."
Transition:          Cut
```
```
Scene 140 — Chapter 15
Duration:            7s (estimate)
Goal:                Show the first measurement — a big win
Purpose:             Set up the reversal
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, genuinely upbeat; overlay: "First A/B run:
                     ~9-13% faster!"
Narration (draft):   "First test looked really promising. Nine to thirteen percent
                     faster."
Transition:          Cut
```
```
Scene 141 — Chapter 15
Duration:            7s (estimate)
Goal:                Introduce doubt: single-run measurement
Purpose:             Teach measurement discipline
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "...from a single run."
Narration (draft):   "One run, though. On a GPU. That alone should already make
                     you a little nervous."
Transition:          Cut
```
```
Scene 142 — Chapter 15
Duration:            7s (estimate)
Goal:                Show the repeated measurement — the reversal itself
Purpose:             Belief reversal #5, the real repo-sourced one
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      "3x repeated builds per config" label, then the "9-13%" number
                     visibly shrinks and gets crossed toward "~2-4% (int8), ~0%
                     (fp16/fp32)".
Narration (draft):   "So, repeat the build three times per config, and the win
                     mostly evaporates. FP16 ties exactly. INT8 keeps a real, but
                     small, two to four percent."
Transition:          Cut
```
```
Scene 143 — Chapter 15
Duration:            7s (estimate)
Goal:                Name the actual culprit behind the first measurement
Purpose:             Explain, don't just reveal
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Two causes listed: "cold-start artifact" and "TensorRT's own
                     ±8% build-to-build tactic noise", both larger than the real
                     effect.
Narration (draft):   "Turns out that first number was mostly a cold-start
                     artifact, and it's smaller than the noise TensorRT's own
                     tactic selection already produces between identical rebuilds
                     anyway."
Transition:          Cut
```
```
Scene 144 — Chapter 15
Duration:            6s (estimate)
Goal:                Show the flag was kept anyway, and why
Purpose:             Resolve without discarding the work
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Kept — free, never measured worse."
Narration (draft):   "It's still in the build scripts, for what it's worth. It's
                     free, never measured worse. Just not the big win it looked
                     like at first."
Transition:          Cut
```
```
Scene 145 — Chapter 15
Duration:            6s (estimate)
Goal:                Name the meta-lesson explicitly
Purpose:             Land the reversal's real point
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Same discipline as that calibration-cache decode earlier. A
                     good-looking first number still has to get double-checked."
Transition:          Cut
```
```
Scene 146 — Chapter 15
Duration:            5s (estimate)
Goal:                Pivot to accuracy
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Okay, we've spent this whole video chasing speed. What did
                     all of it cost us?"
Transition:          Cut
```
```
Scene 147 — Chapter 15
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "16. What accuracy actually costs."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 16 — What Accuracy Actually Costs (Scenes 148–157)

Unchanged from the original cut, renumbered only.

```
Scene 148 — Chapter 16
Duration:            6s (estimate)
Goal:                Explain the methodology
Purpose:             Credibility
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      `verify_accuracy_venv.py` highlighted: PyTorch reference vs each
                     TensorRT engine, top-1 argmax + raw diff.
Narration (draft):   "Every engine's output gets compared directly against the
                     original PyTorch model, on real images."
Transition:          Cut
```
```
Scene 149 — Chapter 16
Duration:            6s (estimate)
Goal:                Show fp32/fp16 results
Purpose:             Establish the baseline
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Top-1 match: fp32 1.0, fp16 1.0; max abs diff 0.028 vs 0.087.
Narration (draft):   "FP32 and FP16 both match the reference on every single
                     image. No surprises there."
Transition:          Cut
```
```
Scene 150 — Chapter 16
Duration:            7s (estimate)
Goal:                Show int8 result
Purpose:             Payoff — the real cost
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Top-1 match: int8 0.9375; max abs diff 2.396 highlighted next to
                     fp16's 0.087 for scale contrast.
Narration (draft):   "INT8, though. Two out of thirty-two images flip. That's
                     about ninety-four percent still matching. And the worst
                     single-value error is over twenty-seven times bigger than
                     FP16's."
Transition:          Cut
```
```
Scene 151 — Chapter 16
Duration:            6s (estimate)
Goal:                Contextualize: this is expected, not alarming
Purpose:             Balance
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "That's roughly the expected cost of quantization. Not a red
                     flag on its own."
Transition:          Cut
```
```
Scene 152 — Chapter 16
Duration:            7s (estimate)
Goal:                Disclose the real methodological limitation
Purpose:             Honesty beat, matches dossier Skeptic finding
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Tested on calibration images. Not
                     held-out."
Narration (draft):   "But here's the part I don't want to gloss over: these
                     thirty-two test images were pulled from the same five hundred
                     used for calibration. That's not really a clean, held-out
                     test."
Transition:          Cut
```
```
Scene 153 — Chapter 16
Duration:            6s (estimate)
Goal:                Explain why that matters
Purpose:             Teach
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Venn diagram: calibration set and "test" set shown as fully
                     overlapping circles.
Narration (draft):   "A model's always going to look a little better on data it's
                     already seen something similar to."
Transition:          Cut
```
```
Scene 154 — Chapter 16
Duration:            6s (estimate)
Goal:                Show the disclosed follow-up
Purpose:             Model honest scoping
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      README excerpt: "flagged as follow-up work, not silently glossed
                     over."
Narration (draft):   "And that's written down in the repo itself as open
                     follow-up work, not buried in a footnote somewhere."
Transition:          Cut
```
```
Scene 155 — Chapter 16
Duration:            6s (estimate)
Goal:                Callback: why random noise was a bad first accuracy test
Purpose:             Reinforce the earlier lesson about representative data
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "n=1 random noise ≠ a real test."
Narration (draft):   "Early on, a single batch of random noise got tried as a
                     quick smoke test. That's exactly the wrong input, since the
                     calibration ranges are built entirely from real image
                     statistics."
Transition:          Cut
```
```
Scene 156 — Chapter 16
Duration:            5s (estimate)
Goal:                Consolidate the accuracy chapter
Purpose:             Land the takeaway
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So, a real cost, measured honestly, and its limits stated
                     out loud."
Transition:          Cut
```
```
Scene 157 — Chapter 16
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "17. What's still open."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 17 — What's Still Open (Scenes 158–165)

Unchanged from the original cut, renumbered only.

```
Scene 158 — Chapter 17
Duration:            6s (estimate)
Goal:                Name the single biggest untested variable
Purpose:             Teach
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "batch=1, always."
Narration (draft):   "One thing worth flagging: every single number in this video
                     was measured at batch size one."
Transition:          Cut
```
```
Scene 159 — Chapter 17
Duration:            7s (estimate)
Goal:                Restate the hypothesis this predicts
Purpose:             Connect back to Ch.13's explanation
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Ch.13's "small kernel, fixed overhead" diagram reappears, now
                     with a ghosted larger-batch tile that fills the tensor-core tile
                     fully.
Narration (draft):   "The whole explanation for that 1.7x shortfall was that
                     batch-one kernels are too small. So logically, a bigger batch
                     should close some of that gap."
Transition:          Cut
```
```
Scene 160 — Chapter 17
Duration:            5s (estimate)
Goal:                State plainly this hasn't been run
Purpose:             Honesty
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera; overlay: "Not run yet."
Narration (draft):   "That's a prediction, though. Not a result. I haven't run
                     it."
Transition:          Cut
```
```
Scene 161 — Chapter 17
Duration:            6s (estimate)
Goal:                Frame this as a genuine open question, not manufactured suspense
Purpose:             Consolidate format directive
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "And I'm not saying that to leave you hanging for views. It's
                     genuinely just written down as the next step in this repo's
                     own README."
Transition:          Cut
```
```
Scene 162 — Chapter 17
Duration:            5s (estimate)
Goal:                Point at the repo as the artifact, not just the video
Purpose:             Distribution beat, minimal
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      GitHub repo README scrolls briefly, "Reproduce" section
                     visible.
Narration (draft):   "Every script that made every number in this video is
                     sitting in this repo, with the exact commands to reproduce
                     it."
Transition:          Cut
```
```
Scene 163 — Chapter 17
Duration:            5s (estimate)
Goal:                One clear, non-pushy prompt
Purpose:             Light CTA, matches "don't oversell" lesson from source material
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "So if you end up running the bigger-batch version yourself,
                     I'd like to know what you get."
Transition:          Cut
```
```
Scene 164 — Chapter 17
Duration:            5s (estimate)
Goal:                Begin winding toward the recap
Purpose:             Bridge
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "Alright, let's put the whole chain back together one more
                     time before we wrap up."
Transition:          Cut
```
```
Scene 165 — Chapter 17
Duration:            4s (estimate)
Goal:                Chapter break
Purpose:             Pace
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chapter card: "18. Close."
Narration (draft):   (none)
Transition:          Fade
```

### Chapter 18 — Close (Scenes 166–169)

Unchanged from the original cut, renumbered only.

```
Scene 166 — Chapter 18
Duration:            8s (estimate)
Goal:                Recap the core mental model, now earned rather than asserted
Purpose:             Consolidate the entire video
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Chain animates in one final time: RANGE → SCALE → INTEGER VALUE →
                     DEQUANTIZE → ERROR — each arrow briefly flashing back to the
                     scene number where it was proven, not just stated.
Narration (draft):   "Range. Scale. Integer value. Dequantize. Error. We didn't
                     just draw this chain out. We ran every link of it."
Transition:          Cut
```
```
Scene 167 — Chapter 18
Duration:            6s (estimate)
Goal:                Restate the headline result once more, now fully explained
Purpose:             Payoff callback to Scene 4
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera, same framing as Scene 4's table but now no
                     question marks.
Narration (draft):   "FP32 to INT8: five times faster, a quarter the size, still
                     matching the original about nineteen times out of twenty. And
                     now you know exactly why every one of those numbers looks the
                     way it does."
Transition:          Cut
```
```
Scene 168 — Chapter 18
Duration:            6s (estimate)
Goal:                Close on the honest, unresolved thread
Purpose:             End on genuine curiosity, not a manufactured hook
Visual Source:       A-Roll (Hybrid Mode)
Visual Content:      Christy on camera.
Narration (draft):   "The batch-size question's still sitting there, unanswered.
                     That's probably where this picks back up next."
Transition:          Cut
```
```
Scene 169 — Chapter 18
Duration:            5s (estimate)
Goal:                End card
Purpose:             Close
Visual Source:       B-Roll (Motion Graphics)
Visual Content:      Simple end card: channel mark, repo name, no subscribe-animation
                     clutter.
Narration (draft):   (none)
Transition:          Fade to black
```
