# Thumbnail — FP32 → FP16 → INT8

Before/after: **2X** (RATED) → **1.7X** (MEASURED). Color mode: `problem-to-solution`
— the rated spec is the unverified claim (muted red/gray), the measured number is the
real, checked-yourself truth (bright yellow-lime).

## Variant 1 — Split left/right, connecting arrow

Canvas 1280×720, 16:9. Background near-black (#0b0f14), an extremely subtle geometric
circuit/grid texture at ~4% opacity, never busy. Frame split by a thin vertical divider
line. Left half: the number "2X" in bold geometric sans-serif, muted red-gray (#6b7280
with a faint #7f1d1d undertone), no glow, deliberately flat/inert — tiny label beneath
it reads "RATED" in small subtle type. Right half: the number "1.7X" massive, bold
geometric sans-serif, vivid yellow-lime (#d4ff3f), subtle glow for readability, tiny
label beneath reads "MEASURED". A thin arrow motif connects left to right at mid-height,
rendered in the same lime accent, implying "claim → reality." Behind both numbers, a
simplified 2D GPU/chip silhouette split down the same divider: left half of the chip
mostly dark/inactive line art, right half mostly lit with lime circuit traces. No text
besides "2X", "1.7X", "RATED", "MEASURED" — four elements total, nothing else. No
people, no photorealism, no logos or brand marks, no UI/dashboard chrome, no watermark,
no clutter.

## Variant 2 — Struck-through claim, single dominant number

Canvas 1280×720, 16:9. Background near-black (#050608), no visible texture. Centered
composition: "2X" rendered large in muted red-gray, with a single bold diagonal
strike-line through it (the spec-sheet claim being corrected). Below and slightly
larger, "1.7X" in vivid yellow-lime with a restrained glow — this is the dominant
element of the frame, sized larger than the struck "2X" above it. A simplified 2D GPU
chip silhouette sits centered behind both numbers, low-opacity dark line art on the
"2X" side blending into brighter lime circuit-trace linework directly behind "1.7X",
so the illustration reads as one continuous chip transitioning from inert to active
top-to-bottom. Small supporting label beneath everything: "INT8 vs FP16". No people, no
photorealism, no real logos/brand marks, no dashboards or UI chrome, no watermark, no
extra text beyond the four short labels described.

## Variant 3 — Two stat cards, chip silhouette behind

Canvas 1280×720, 16:9. Background near-black (#08090c) with a faint radial vignette
darkening the corners, no grid texture. Two flat rectangular "stat card" panels sit
side by side with a thin lime divider line between them (not skeuomorphic, just a
sharp-edged rectangle of a slightly lighter dark tone than the background). Left card:
"2X" in muted red-gray bold sans-serif, small "RATED" label beneath, card rendered
visually "dimmer" (lower contrast) than the right card. Right card: "1.7X" in vivid
yellow-lime bold sans-serif with restrained glow, small "MEASURED" label beneath, card
rendered at full contrast/brightness. A single simplified 2D GPU/tensor-core silhouette
spans behind both cards at low opacity, mostly dark line art behind the left card and
lit lime circuit traces behind the right card, tying the two panels together as one
chip. No people, no photorealism, no real brand/vendor logos, no dashboard or UI
chrome, no watermark, no additional text or clutter beyond the four labels.

---

HyperFrames-native alternative: this can also be built as a fully deterministic vector
composition — reusing the `stat-reveal` block from `~/video-templates/compositions/`
and captured at 1280×720 via `hyperframes snapshot` — no diffusion model, no risk of
hallucinated logos or artifacts, fully on-brand and directly editable. Say the word and
I'll build that version instead of / alongside the AI-image prompts above.
