# Video Script: FP32 → FP16 → INT8, What Precision Actually Changes

Mapped directly to `video_storyboard.md` (169 scenes after the pacing trim, 18
chapters + close). No cold open, no zero-noun mystery hook. Per the binding format
directive carried from `video_dossier.md` and confirmed by the creator: this video is
linear and names its subject in the first line. Every reversal below is a real thing
that happened during this project, so `[PAUSE]` marks a genuine beat, not a
manufactured cliffhanger.

Delivery cues used throughout: an energy label opens each chapter; `[PAUSE]` marks a
real beat (a hesitation, a moment to let a number land); **bold** marks the one word or
number per line worth landing hard. Everything else is plain conversational delivery.
Silent chapter-card scenes are noted but carry no voiceover.

**Revision note:** Chapters 1, 2, 4, 5, and 9 were shortened after review, the earlier
draft carried too much formula derivation and too many worked examples back to back.
Each now makes its point with one clean example instead of two or three; Chapter 9 in
particular used to re-run the entire quantization loop on the toy array a second time
before finally touching real data, and now goes straight to the real-data check.
Chapters 3, 6, 7, 8, and 10 (concept- and code-driven, not equation-driven) are
unchanged from the previous draft.

---

## Chapter 0: Orientation, One Model and Three Precisions [QUIET/CALM]

(Scene 1 - A-Roll)
I've been running experiments for the last few days. Using
ResNet50, laptop GPU, three separate TensorRT engines. The only thing that changes
between them is the number format.

(Scene 2 - B-Roll: terminal, directory tree)
Every number I show you today came out of these folders. Nothing here is made up
for the video.

(Scene 3 - A-Roll)
Here's what I want to figure out. What does **precision** even mean inside these
files? And what does changing it really cost you, and save you?

(Scene 4 - B-Roll: results table)
Here's where we end up. These are the real numbers I measured. [PAUSE] They
probably don't mean much yet. By the end, you'll know exactly why each one looks
like this.

(Scene 5 - B-Roll: spec card)
Quick disclaimer. It's just a laptop GPU. These exact numbers are specific to this
hardware. The reasoning behind them isn't, though.

(Scene 6 - A-Roll)
Roughly how we're going to get there: start at the bit level, work up to a scale
and a calibration file, then go run these engines.

(Scene 7 - A-Roll)
I'm not going to jump-cut to the punchline here. We're building this the way
you'd have to build it, step by step.

(Scene 8 - B-Roll: chapter card, no VO)

---

## Chapter 1: What A Float Actually Is (FP32) [QUIET/CALM]

(Scene 9 - A-Roll)
Before we even get near INT8, I want to nail down one thing properly. What IS a
32-bit float, at the bit level?

(Scene 10 - B-Roll: bit diagram + formula)
Thirty-two bits, split into three jobs: a sign, an exponent, and a fraction. One
formula turns those bits back into a real number.

(Scene 11 - B-Roll: sign and exponent highlights)
Sign's just plus or minus. The exponent's a little weirder, it stores a shifted
number, and you subtract 127 to get the real exponent. That's the trick.

(Scene 12 - B-Roll: fraction bits)
Here's a nice detail. You only store twenty-three fraction bits, but you get
**twenty-four** bits of precision, because that leading 1 is just implied. It's
free.

(Scene 13 - B-Roll: worked example, bit pattern)
Let's try one. Sign's zero, exponent lands on zero, fraction's all zeros. So it's
just... **one point zero.**

(Scene 14 - B-Roll: range stated on a number line)
Push that same formula to its limits and FP32 covers roughly ten to the minus
thirty-eight, all the way up to **ten to the plus thirty-eight.** Huge range. We
don't need to grind through every edge case to know that.

(Scene 15 - A-Roll)
Worth keeping these two things separate in your head: how big a number can get,
and how finely you can tell two numbers apart. FP32's generous on both.

(Scene 16 - A-Roll)
So if it's this generous, why would anyone ever want to shrink it?

(Scene 17 - B-Roll: chapter card, no VO)

---

## Chapter 2: Shrinking The Format (FP16) [QUIET/CALM]

(Scene 18 - A-Roll)
The simple answer: half the bits means half the memory. And on hardware built
for it, roughly half the compute time too.

(Scene 19 - B-Roll: FP32 vs FP16 bit diagram)
Same basic idea as FP32, just fewer bits in two of the three fields.

(Scene 20 - B-Roll: range shrinks, max value lands)
Fewer exponent bits, much smaller range. Work through the same math and FP16
tops out around **sixty-five thousand.** That's it.

(Scene 21 - B-Roll: overflow example)
So a number FP32 handles without blinking, say a hundred thousand, just doesn't
fit in FP16 at all.

(Scene 22 - A-Roll)
And notice, I didn't need anything extra for any of this. No outside data. Those
sixteen bits describe the number on their own. It's just a format conversion.

(Scene 23 - A-Roll)
But FP16 still has an exponent field in there. So what happens when we go all
the way down to 8 bits, and there's just no room left for one at all?

(Scene 24 - B-Roll: chapter card, no VO)

---

## Chapter 3: Where Floats Stop Working (Why INT8 Is Different) [QUIET/CALM]

(Scene 25 - A-Roll)
My first assumption was that INT8 is basically FP16, but smaller. [PAUSE] It's
not. It's not that at all.

(Scene 26 - B-Roll: plain integer box)
INT8 is just a plain signed integer. There's no sign-exponent-mantissa split
anymore. It has no idea what range it's supposed to represent.

(Scene 27 - B-Roll: number line compresses)
So the problem becomes this: how do you squeeze a continuous range of real
numbers down into just two hundred fifty-six integers?

(Scene 28 - A-Roll)
You need something extra. Some number, stored alongside the integers, that says
what one INT8 step is actually worth.

(Scene 29 - B-Roll: file tree)
This repo has a whole script dedicated to exactly that problem. Nothing like it
exists for the FP16 build.

(Scene 30 - B-Roll: split screen, two scripts)
Look at the difference. One file just flips a builder flag. The other one loads
five hundred real images. That's the whole asymmetry, right there in the code.

(Scene 31 - A-Roll)
Alright, let's build that scale ourselves, starting with the easiest version of
the problem.

(Scene 32 - B-Roll: chapter card, no VO)

---

## Chapter 4: The Simplest Mapping (Symmetric Quantization) [QUIET/CALM]

(Scene 33 - B-Roll: centered number line)
Easiest case first: a range that's already sitting centered on zero.

(Scene 34 - B-Roll: scale formula)
One number needed: take the largest absolute value, divide it by a hundred
twenty-seven. That gives us about **0.01575.**

(Scene 35 - B-Roll: quantize, dequantize, error)
Try it on one point zero. That becomes **sixty-four.** Multiply back by the
scale and you get about one point zero zero eight. That little gap is the
quantization error.

(Scene 36 - A-Roll)
One nice thing: zero always lands exactly on zero here. No offset needed.

(Scene 37 - B-Roll: editor, function types in)
Okay, enough talking about it. Let me just write the actual function.

(Scene 38 - B-Roll: terminal output)
Run it, and yep, same sixty-four we got by hand.

(Scene 39 - A-Roll)
Hang onto this function, we'll reuse it on a real activation soon. But first:
what happens when the real range isn't centered on zero at all?

(Scene 40 - B-Roll: chapter card, no VO)

---

## Chapter 5: When Zero Isn't In The Middle (Asymmetric Quantization) [QUIET/CALM]

(Scene 41 - B-Roll: forced-symmetric waste)
If you force this into a symmetric range, you end up wasting something like a
third of your INT8 codes on values that never even show up.

(Scene 42 - B-Roll: asymmetric scale and zero-point)
So the scale spans the whole range instead, across all two hundred fifty-five
steps. And now zero doesn't land on integer zero anymore, it lands wherever this
zero-point number says it should.

(Scene 43 - B-Roll: formula comparison)
One extra term added on. That's the whole difference between the two.

(Scene 44 - A-Roll)
And this comes up constantly in real networks. Anything right after a ReLU is
never negative.

(Scene 45 - B-Roll: editor + terminal)
Barely different code, just one extra piece, and it matches what we got by
hand.

(Scene 46 - A-Roll)
Okay, one scale per tensor. But is one scale always enough?

(Scene 47 - B-Roll: chapter card, no VO)

---

## Chapter 6: One Scale Or Many? (Per-Tensor vs Per-Channel) [QUIET/CALM]

(Scene 48 - B-Roll: two channel ranges)
Picture one weight tensor where one channel's values are tiny, and a different
channel's values are huge.

(Scene 49 - B-Roll: wasted resolution)
If you use one shared scale for the whole thing, that small channel basically
gets no usable resolution at all.

(Scene 50 - B-Roll: per-channel fix)
But give each channel its own scale, and now both of them get to use their full
range.

(Scene 51 - A-Roll)
NVIDIA's own docs say TensorRT does exactly this automatically for weights,
while activations still just get one scale each, from the calibrator.

(Scene 52 - A-Roll)
Now, I haven't checked that inside this specific engine yet. [PAUSE] So instead
of just repeating what the docs say, let's go look for ourselves.

(Scene 53 - B-Roll: engine inspector call highlighted)
TensorRT ships a tool for exactly this kind of question. It's called the engine
inspector.

(Scene 54 - A-Roll)
We'll answer this properly once the engine's built. A few chapters from now.
Promise I won't forget.

(Scene 55 - B-Roll: 2x2 grid)
So really there's two separate choices here: where the scale's centered, and how
many scales you're using.

(Scene 56 - A-Roll)
None of this tells us where these numbers come from for activations, though.
That's calibration.

(Scene 57 - B-Roll: chapter card, no VO)

---

## Chapter 7: Where The Scale Actually Comes From (Calibration) [QUIET/CALM]

(Scene 58 - A-Roll)
Weights, you can just go inspect ahead of time. Activations are trickier. They
depend entirely on whatever input you feed the network.

(Scene 59 - B-Roll: calibration flow diagram)
So the idea is, you run real data through the network once, just to watch what
ranges show up.

(Scene 60 - B-Roll: image folder listing)
In this repo it's **five hundred** real JPEGs, fifty per class across ten
classes. Not synthetic noise.

(Scene 61 - A-Roll)
Simplest thing you'd try: just track the min and max you see and scale to that.
Turns out TensorRT doesn't do that by default.

(Scene 62 - B-Roll: outlier distribution)
One weird outlier value blows the whole range out, and suddenly every normal
value gets crammed into a tiny sliver of INT8's resolution.

(Scene 63 - B-Roll: overlapping histograms)
So instead, entropy calibration picks a threshold that keeps the quantized
distribution as close as possible to the original. Even if that means
deliberately clipping the extreme outlier.

(Scene 64 - B-Roll: citation card)
This isn't some new trick, either. It's a method NVIDIA published back in 2017.

(Scene 65 - B-Roll: calibrator class highlighted)
Here's the actual class from the repo. Every batch it hands over to TensorRT is
eight real, preprocessed images.

(Scene 66 - B-Roll: preprocessing steps)
Resize, center-crop, normalize with ImageNet stats. Literally the same
preprocessing the model expects at inference time.

(Scene 67 - B-Roll: cache write highlighted)
And once calibration's done, it just writes everything it learned out to one
file.

(Scene 68 - A-Roll)
So, "calibration produces a scale per tensor." Is that just a concept I'm
describing? Or can I show you it?

(Scene 69 - B-Roll: chapter card, no VO)

---

## Chapter 8: Opening The Calibration File [QUIET/CALM, shifting to SLOW REVEAL at Scene 75]

(Scene 70 - B-Roll: cat command, plain text output)
Let's just open it up. And it's not some binary blob. It's literally a plain text
file.

(Scene 71 - B-Roll: line count)
One line per tensor, across this whole fifty-eight-layer graph.

(Scene 72 - B-Roll: one line highlighted)
Let's decode one of these. Let's do the input tensor.

(Scene 73 - B-Roll: decoder types in)
Should only take about five lines: read the hex as an integer, then reinterpret
those exact same bits as a float.

(Scene 74 - B-Roll: terminal output)
Okay, run it. **0.026.** So that's apparently the calibrated dynamic range for
the input tensor.

(Scene 75 - A-Roll) [SLOW REVEAL]
Wait. [PAUSE] Hang on. This input's ImageNet-normalized, its real range should be
something like plus or minus two and a half. 0.026 is way off from that. Did I
mess up the decode?

(Scene 76 - B-Roll: two decode candidates side by side)
So I tried it the other byte-order way too, and honestly both readings are
defensible. They give two different numbers, and neither one matches what I
expected.

(Scene 77 - A-Roll)
But this is exactly what entropy calibration is supposed to do. It can pick a
threshold way below the true max if that gives better resolution for most of the
data. [PAUSE] Small doesn't mean broken here.

(Scene 78 - A-Roll)
Honestly, the lesson isn't the exact number. It's that when something looks
surprising, you go check it instead of trusting the first plausible-looking
answer.

(Scene 79 - B-Roll: chapter card, no VO)

---

## Chapter 9: From Toy Numbers To Real Activations [QUIET/CALM]

(Scene 80 - A-Roll)
All that math, does the real model actually produce numbers like this? Let's
find out instead of just assuming.

(Scene 81 - B-Roll: PyTorch hook + function reused)
Grab a real slice of activations straight out of this ResNet50, and run the
exact same function on it. Same eight lines. Real numbers.

(Scene 82 - A-Roll)
No separate, more-real version of the math hiding anywhere. This is it. Time to
go check what TensorRT actually did with per-channel scaling, that question
from a few chapters back.

(Scene 83 - B-Roll: chapter card, no VO)

---

## Chapter 10: Checking TensorRT's Own Homework (Engine Inspector Audit) [QUIET/CALM, lifting to HIGH ENERGY at Scene 88]

(Scene 84 - A-Roll)
Here's the claim I want to test: weights get their own scale per channel,
activations get just one scale for the whole tensor.

(Scene 85 - B-Roll: inspector call highlighted)
Turns out TensorRT will hand you the full JSON description of any layer it
built, if you ask it nicely.

(Scene 86 - B-Roll: one conv layer highlighted)
Let's pull the full record for one actual INT8 convolution layer.

(Scene 87 - B-Roll: JSON scrolls)
Look at the shapes here. The weight scale is an array, one number per output
channel. The activation scale is just a single number.

(Scene 88 - A-Roll) [HIGH ENERGY]
So here's the confirmation. **This specific engine** really is mixing
per-channel weights with per-tensor activations, exactly like the docs
describe.

(Scene 89 - A-Roll)
If you run this yourself on a different model or TensorRT version and get
something different, that's kind of the whole point of checking it yourself.
Trust your own inspector output over mine.

(Scene 90 - B-Roll: 2x2 grid stamped)
So at this point, every concept from the first half of this video is tied to a
real file sitting in this repo.

(Scene 91 - A-Roll)
Which brings up the real question: how much does any of this help?

(Scene 92 - B-Roll: file tree highlight)
Time to go run the benchmark.

(Scene 93 - B-Roll: chapter card, no VO)

---

## Chapter 11: The Benchmark (build_and_bench.py) [HIGH ENERGY]

(Scene 94 - B-Roll: benchmark function highlighted)
Quick note on how this is measured: twenty warmup runs, two hundred timed ones,
using CUDA events. Not just a Python stopwatch.

(Scene 95 - B-Roll: terminal, fp32 result)
Alright, FP32 first. **One point eight milliseconds.**

(Scene 96 - B-Roll: terminal, fp16 result)
FP16. About zero point six five milliseconds. Almost three times faster
already.

(Scene 97 - B-Roll: terminal, int8 result)
And INT8. [PAUSE] **About zero point three five.**

(Scene 98 - A-Roll)
So FP32 to INT8, that's **five times faster.** On the exact same weights, same
model.

(Scene 99 - B-Roll: size and memory bars)
And latency's only part of it. Smaller on disk, smaller in GPU memory too. Both
roughly halve at each step.

(Scene 100 - B-Roll: build time bars)
The one thing that gets worse is build time. Makes sense, INT8 has to calibrate
first.

(Scene 101 - A-Roll)
Okay so, FP16 to INT8 halves the bits again. So logically, this step should
roughly double the speed too, right?

(Scene 102 - B-Roll: ratio computed on screen)
Let's check. **About one point nine x.** Close. Not quite two x.

(Scene 103 - A-Roll)
That gap's small enough you could just shrug it off. I don't want to shrug it
off.

(Scene 104 - A-Roll)
But before I go blaming hardware limits, let's rule out something dumber first.
Is this engine even really running in INT8?

(Scene 105 - B-Roll: chapter card, no VO)

---

## Chapter 12: Is INT8 Actually Running In INT8? [QUIET/CALM, investigative]

(Scene 106 - A-Roll)
Because setting a builder flag that says "use INT8" doesn't prove every layer
got an INT8 kernel.

(Scene 107 - B-Roll: nsys command highlighted)
So let's use Nsight Systems. It traces the literal CUDA kernels that ran. No
TensorRT instrumentation getting in the way.

(Scene 108 - B-Roll: kernel name types in)
"i8i8." Right there in the kernel name. [PAUSE] That's the actual kernel that ran
on the GPU. I didn't have to take TensorRT's word for it.

(Scene 109 - B-Roll: count lands)
Across three hundred iterations, **about ninety-six percent** of every single
kernel launch genuinely runs on integer tensor cores.

(Scene 110 - B-Roll: engine inspector, 2 float layers)
The only two layers that stay in float are the tiny final classifier bias-add.
TensorRT chose to leave those two in full precision on purpose.

(Scene 111 - A-Roll)
Makes sense, actually. It's cheap to keep at full precision, and it directly
produces the logits your argmax depends on. Worth protecting.

(Scene 112 - A-Roll)
So that missing 0.3x isn't hiding in some fallback layer. It's got to be
something else.

(Scene 113 - B-Roll: kernel category bar)
And it's not reformat overhead either. That's like one and a half percent of GPU
time. Barely anything.

(Scene 114 - A-Roll)
So it's ninety-five percent compute-bound, genuinely running INT8 kernels. So why
isn't it two x?

(Scene 115 - B-Roll: chapter card, no VO)

---

## Chapter 13: Why Isn't It 4x? [QUIET/CALM, investigative, SLOW REVEAL at Scene 123]

(Scene 116 - B-Roll: Nsight per-precision table)
Pure GPU kernel time, no profiler overhead in the way: about **two point six x**
for the first step, one point seven x for the second.

(Scene 117 - A-Roll)
These tensor cores are literally rated for **double** the FP16 throughput in
INT8. On paper this should've been the easy step.

(Scene 118 - B-Roll: peak vs achieved bars)
But "rated for 2x" is a peak number. And peak numbers assume the kernel runs long
enough to get there.

(Scene 119 - B-Roll: small tile diagram)
At batch size one, most of ResNet50's individual conv layers are just small. Too
small to fully fill a tensor-core tile.

(Scene 120 - B-Roll: timeline bar)
Every kernel launch has some fixed setup cost attached to it. Halving the math
time doesn't halve that fixed cost.

(Scene 121 - B-Roll: kernel duration numbers)
We're talking microsecond-scale kernels here. That fixed overhead eats up a much
bigger fraction of a small kernel than a big one.

(Scene 122 - A-Roll)
Which is also why FP32 got closer to a clean 2x earlier. Its kernels were slower
to begin with. More room to cut before hitting that same floor.

(Scene 123 - A-Roll) [SLOW REVEAL]
So overall, the model's compute-bound. [PAUSE] But at batch one, a lot of
individual layers are launch-overhead-bound instead.

(Scene 124 - A-Roll)
So if launch overhead's the real limiter here, can we get rid of it, without
touching the model at all?

(Scene 125 - B-Roll: CUDA graphs title card)
Turns out, yeah. **CUDA graphs.**

(Scene 126 - A-Roll)
And this repo's already got both code paths, so we can just compare them
directly.

(Scene 127 - B-Roll: chapter card, no VO)

---

## Chapter 14: Removing The Launch Overhead (CUDA Graphs) [MODERATE ENERGY, building to HIGH ENERGY at Scene 134]

(Scene 128 - B-Roll: kernel launches collapse into one graph)
Idea is: capture the whole sequence of per-layer kernel launches just once, then
replay it as a single graph launch every time after that.

(Scene 129 - B-Roll: code flag highlighted)
It's already sitting right here in the repo's own benchmark function. One flag.

(Scene 130 - B-Roll: capture/replay code)
Capture once, instantiate once, then just launch it over and over.

(Scene 131 - A-Roll)
Before I trust any speed number out of this, worth saying: it's already been
checked bit-identical against the normal path.

(Scene 132 - B-Roll: fp32 delta)
FP32 first. Small gain. Three to seven percent.

(Scene 133 - B-Roll: fp16 delta)
FP16's bigger. Eleven to fourteen percent.

(Scene 134 - A-Roll) [HIGH ENERGY]
And INT8, the one with the smallest, most overhead-bound kernels, gets the
**biggest win of all.** [PAUSE] Which basically confirms the whole theory.

(Scene 135 - B-Roll: ratio animates)
So the FP16-to-INT8 ratio moves, from one point seven up to **almost one point
nine.**

(Scene 136 - A-Roll)
Closer to two x. Not all the way there. That batch-size limit is still real.

(Scene 137 - B-Roll: chapter card, no VO)

---

## Chapter 15: The Flag That Wasn't As Good As It Looked (DIRECT_IO) [MODERATE ENERGY, SLOW REVEAL at Scene 142]

(Scene 138 - B-Roll: DIRECT_IO flag highlighted)
One more thing was tried: a flag called DIRECT_IO. It removes a reformat layer
sitting right at the input boundary.

(Scene 139 - B-Roll: callback thumbnail)
This came from something we noticed earlier. Remember, the biggest single layer
in the INT8 profile wasn't a convolution. It was an input reformat.

(Scene 140 - A-Roll)
First test looked really promising. **Nine to thirteen percent** faster.

(Scene 141 - A-Roll)
One run, though. On a GPU. [PAUSE] That alone should already make you a little
nervous.

(Scene 142 - B-Roll: number shrinks) [SLOW REVEAL]
So, repeat the build three times per config, and the win mostly evaporates.
[PAUSE] FP16 ties exactly. INT8 keeps a real, but small, two to four percent.

(Scene 143 - B-Roll: two causes listed)
Turns out that first number was mostly a cold-start artifact. And it's smaller
than the noise TensorRT's own tactic selection already produces between identical
rebuilds anyway.

(Scene 144 - A-Roll)
It's still in the build scripts, for what it's worth. It's free, never measured
worse. Just not the big win it looked like at first.

(Scene 145 - A-Roll)
Same discipline as that calibration-cache decode earlier. A good-looking first
number still has to get double-checked.

(Scene 146 - A-Roll)
Okay, we've spent this whole video chasing speed. What did all of it cost us?

(Scene 147 - B-Roll: chapter card, no VO)

---

## Chapter 16: What Accuracy Actually Costs [QUIET/CALM, honest]

(Scene 148 - B-Roll: accuracy script highlighted)
Every engine's output gets compared directly against the original PyTorch model,
on real images.

(Scene 149 - B-Roll: fp32/fp16 results)
FP32 and FP16 both match the reference on every single image. No surprises
there.

(Scene 150 - B-Roll: int8 result)
INT8, though. Two out of thirty-two images flip. That's **about ninety-four
percent** still matching. And the worst single-value error is over twenty-seven
times bigger than FP16's.

(Scene 151 - A-Roll)
That's roughly the expected cost of quantization. Not a red flag on its own.

(Scene 152 - A-Roll)
But here's the part I don't want to gloss over: these thirty-two test images were
pulled from the same five hundred used for calibration. That's not really a
clean, held-out test.

(Scene 153 - B-Roll: Venn diagram, overlapping circles)
A model's always going to look a little better on data it's already seen
something similar to.

(Scene 154 - B-Roll: README excerpt)
And that's written down in the repo itself as open follow-up work. Not buried in
a footnote somewhere.

(Scene 155 - A-Roll)
Early on, a single batch of random noise got tried as a quick smoke test. That's
exactly the wrong input, since the calibration ranges are built entirely from
real image statistics.

(Scene 156 - A-Roll)
So, a real cost, measured honestly, and its limits stated out loud.

(Scene 157 - B-Roll: chapter card, no VO)

---

## Chapter 17: What's Still Open [QUIET/CALM]

(Scene 158 - A-Roll)
One thing worth flagging: every single number in this video was measured at
batch size one.

(Scene 159 - B-Roll: ghosted larger-batch tile)
The whole explanation for that 1.7x shortfall was that batch-one kernels are too
small. So logically, a bigger batch should close some of that gap.

(Scene 160 - A-Roll)
That's a prediction, though. Not a result. I haven't run it.

(Scene 161 - A-Roll)
And I'm not saying that to leave you hanging for views. It's genuinely just
written down as the next step in this repo's own README.

(Scene 162 - B-Roll: README scrolls)
Every script that made every number in this video is sitting in this repo, with
the exact commands to reproduce it.

(Scene 163 - A-Roll)
So if you end up running the bigger-batch version yourself, I'd like to know what
you get.

(Scene 164 - A-Roll)
Alright, let's put the whole chain back together one more time before we wrap
up.

(Scene 165 - B-Roll: chapter card, no VO)

---

## Chapter 18: Close [QUIET/CALM, warm]

(Scene 166 - B-Roll: full chain animates)
Range. Scale. Integer value. Dequantize. Error. We didn't just draw this chain
out. We ran every link of it.

(Scene 167 - A-Roll)
FP32 to INT8: **five times faster,** a quarter the size, still matching the
original about **nineteen times out of twenty.** [PAUSE] And now you know exactly
why every one of those numbers looks the way it does.

(Scene 168 - A-Roll)
The batch-size question's still sitting there, unanswered. That's probably where
this picks back up next.

(Scene 169 - B-Roll: end card, no VO)
