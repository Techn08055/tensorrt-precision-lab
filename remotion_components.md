# Remotion Components — Forward Logic: "FP32 → FP16 → INT8"

Source: `video_storyboard.md` + `video_script.md`. Built chapter-by-chapter per the
Animator contract — this file currently covers **Chapter 0 only**. Pausing after this
chapter before continuing (see note at the end of this file).

## Project config

```
Composition id:  Main
width:           1920
height:          1080
fps:             30
```

## Narration audio

Corrected narration lives at `audio/narration_normal_speed.wav` (extracted from the
creator's raw take, which was recorded/exported at 1.4x — slowed back to 1/1.4x to
restore natural speaking pace; original 897.2s → corrected 1255.6s, ~20m56s total).

**A-Roll is a designed animation, not camera footage.** Per the creator: A-Roll scenes
are built by the animator as motion-graphic compositions, driven only by the corrected
audio — the source recording's video track is not used at all, not even as a
placeholder-for-later. `<CameraPlaceholder>` below is the actual, deliberate look for
these scenes (a treatment to refine later, not a stand-in waiting for a real take).
Per-scene narration timing below is estimated from the storyboard's duration budget
(10/6/8/7/5/6/5/4s), not yet word-aligned to the actual audio — real alignment (via
silence detection / a transcript) should happen once final cut points exist; real
voice duration overrides these estimates at that point.

## Shared primitives

```tsx
// CameraPlaceholder.tsx
import { AbsoluteFill } from "remotion";

export const CameraPlaceholder: React.FC<{ label?: string }> = ({ label }) => (
  <AbsoluteFill
    style={{
      background: "linear-gradient(135deg, #14181f 0%, #1f2937 100%)",
      justifyContent: "center",
      alignItems: "center",
    }}
  >
    <div
      style={{
        fontFamily: "Inter, sans-serif",
        fontSize: 28,
        color: "#4b5563",
        letterSpacing: 2,
        textTransform: "uppercase",
        border: "1px dashed #374151",
        padding: "12px 24px",
        borderRadius: 8,
      }}
    >
      {label}
    </div>
  </AbsoluteFill>
);
```
`label` is optional — most A-Roll scenes render the dark gradient plain, undressed by
design; pass a `label` only for a scene that specifically calls for on-screen text in
that slot (none currently do, `<LowerThird>` handles the actual on-screen copy).

```tsx
// LowerThird.tsx
import { interpolate, useCurrentFrame, Easing } from "remotion";

export const LowerThird: React.FC<{ text: string; fromFrame?: number }> = ({
  text,
  fromFrame = 0,
}) => {
  const frame = useCurrentFrame() - fromFrame;
  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const translateY = interpolate(frame, [0, 15], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <div
      style={{
        position: "absolute",
        left: 80,
        bottom: 90,
        opacity,
        translate: `0px ${translateY}px`,
        fontFamily: "Inter, sans-serif",
        fontSize: 40,
        fontWeight: 600,
        color: "#f9fafb",
        textShadow: "0px 4px 14px rgba(0,0,0,0.7)",
        background: "rgba(17,24,39,0.55)",
        padding: "10px 22px",
        borderRadius: 6,
      }}
    >
      {text}
    </div>
  );
};
```

```tsx
// ChapterCard.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const ChapterCard: React.FC<{ number: number; title: string; durationInFrames: number }> = ({
  number,
  title,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [0, 15, durationInFrames - 15, durationInFrames],
    [0, 1, 1, 0],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    }
  );

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity, fontFamily: "Inter, sans-serif", fontSize: 44, color: "#f9fafb", textAlign: "center" }}>
        <div style={{ color: "#6b7280", fontSize: 24, marginBottom: 12 }}>CHAPTER {number}</div>
        {title}
      </div>
    </AbsoluteFill>
  );
};
```
Reused for every chapter-card scene from Chapter 1 onward (`Ch0Scene8` predates this
shared version and is left as-is rather than reworked without cause).

## Scene 1 — Chapter 0 (A-Roll, 0–300f / 10s)

```tsx
// scenes/Ch0Scene1.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch0Scene1: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="tensorrt-precision-lab" fromFrame={30} />
  </AbsoluteFill>
);
```
Narration: "This is a folder I've been running experiments in for the last few days.
Same ResNet50, same GPU, three separate TensorRT engines. The only thing that changes
between them is the number format."

## Scene 2 — Chapter 0 (B-Roll, 300–480f / 6s)

```tsx
// scenes/Ch0Scene2.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const DIRS = ["models/", "fp32/", "fp16/", "int8/", "calibration/", "benchmarks/", "profiling/"];
const COMMAND = "find . -maxdepth 2";

export const Ch0Scene2: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 25], [0, COMMAND.length], {
    extrapolateRight: "clamp",
  }));

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 32 }}>
        $ {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ marginTop: 24 }}>
        {DIRS.map((dir, i) => {
          const start = 30 + i * 10;
          const opacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const translateX = interpolate(frame, [start, start + 10], [-12, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={dir}
              style={{ opacity, translate: `${translateX}px 0px`, color: "#e5e7eb", fontSize: 28, padding: "4px 0" }}
            >
              ./{dir}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Every number I show you today came out of these folders. Nothing here is
made up for the video."

## Scene 3 — Chapter 0 (A-Roll, 480–720f / 8s)

```tsx
// scenes/Ch0Scene3.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch0Scene3: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="FP32 → FP16 → INT8 — what actually changes?" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "Here's what I want to figure out. What does precision even mean inside
these files? And what does changing it really cost you, and save you?"

## Scene 4 — Chapter 0 (B-Roll, 720–930f / 7s)

```tsx
// scenes/Ch0Scene4.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const ROWS = [
  { label: "Latency (ms)", fp32: "1.808", fp16: "0.645", int8: "0.347" },
  { label: "Engine size (MB)", fp32: "107.94", fp16: "49.14", int8: "25.26" },
  { label: "Top-1 accuracy", fp32: "1.0", fp16: "1.0", int8: "0.9375" },
];

export const Ch0Scene4: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "Inter, sans-serif", color: "#f9fafb" }}>
        <div style={{ display: "grid", gridTemplateColumns: "320px repeat(3, 200px)", fontSize: 28 }}>
          <div />
          {["FP32", "FP16", "INT8"].map((h) => (
            <div key={h} style={{ fontWeight: 700, textAlign: "center", color: "#93c5fd" }}>{h}</div>
          ))}
          {ROWS.map((row, i) => {
            const start = 20 + i * 25;
            const opacity = interpolate(frame, [start, start + 15], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            return (
              <React.Fragment key={row.label}>
                <div style={{ opacity, padding: "10px 0" }}>{row.label}</div>
                <div style={{ opacity, textAlign: "center" }}>{row.fp32}</div>
                <div style={{ opacity, textAlign: "center" }}>{row.fp16}</div>
                <div style={{ opacity, textAlign: "center" }}>{row.int8}</div>
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Here's where we end up. These are the real numbers I measured. [PAUSE]
They probably don't mean much yet. By the end, you'll know exactly why each one looks
like this."

## Scene 5 — Chapter 0 (B-Roll, 930–1080f / 5s)

```tsx
// scenes/Ch0Scene5.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const SPECS = [
  ["GPU", "RTX 4050 Laptop GPU"],
  ["Driver", "580.173.02"],
  ["CUDA", "13.0"],
  ["TensorRT", "10.14.1"],
];

export const Ch0Scene5: React.FC = () => {
  const frame = useCurrentFrame();
  const cardOpacity = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: cardOpacity,
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "32px 48px",
          fontFamily: "monospace",
          fontSize: 26,
          color: "#e5e7eb",
        }}
      >
        {SPECS.map(([k, v]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 60, padding: "6px 0" }}>
            <span style={{ color: "#6b7280" }}>{k}</span>
            <span>{v}</span>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Quick disclaimer. It's just a laptop GPU. These exact numbers are specific
to this hardware. The reasoning behind them isn't, though."

## Scene 6 — Chapter 0 (A-Roll, 1080–1260f / 6s)

```tsx
// scenes/Ch0Scene6.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

const STEPS = ["bits", "scale", "engine"];

export const Ch0Scene6: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 120 }}>
        <div style={{ display: "flex", gap: 40 }}>
          {STEPS.map((step, i) => {
            const start = 15 + i * 20;
            const scale = interpolate(frame, [start, start + 12], [0.6, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.34, 1.56, 0.64, 1),
            });
            const opacity = interpolate(frame, [start, start + 12], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            return (
              <div
                key={step}
                style={{
                  scale,
                  opacity,
                  fontFamily: "Inter, sans-serif",
                  fontSize: 32,
                  color: "#f9fafb",
                  background: "rgba(17,24,39,0.75)",
                  border: "1px solid #374151",
                  borderRadius: 999,
                  padding: "14px 32px",
                }}
              >
                {step}
                {i < STEPS.length - 1 && <span style={{ color: "#6b7280", marginLeft: 24 }}>→</span>}
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Roughly how we're going to get there: start at the bit level, work up to
a scale and a calibration file, then go run these engines."

## Scene 7 — Chapter 0 (A-Roll, 1260–1410f / 5s)

```tsx
// scenes/Ch0Scene7.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch0Scene7: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "I'm not going to jump-cut to the punchline here. We're building this the
way you'd have to build it, step by step."

## Scene 8 — Chapter 0 (B-Roll, chapter card, 1410–1530f / 4s, no VO)

```tsx
// scenes/Ch0Scene8.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch0Scene8: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 15, 105, 120], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity, fontFamily: "Inter, sans-serif", fontSize: 44, color: "#f9fafb", textAlign: "center" }}>
        <div style={{ color: "#6b7280", fontSize: 24, marginBottom: 12 }}>CHAPTER 1</div>
        What a float actually is.
      </div>
    </AbsoluteFill>
  );
};
```

## Chapter 0 assembly

```tsx
// chapters/Chapter0.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch0Scene1 } from "../scenes/Ch0Scene1";
import { Ch0Scene2 } from "../scenes/Ch0Scene2";
import { Ch0Scene3 } from "../scenes/Ch0Scene3";
import { Ch0Scene4 } from "../scenes/Ch0Scene4";
import { Ch0Scene5 } from "../scenes/Ch0Scene5";
import { Ch0Scene6 } from "../scenes/Ch0Scene6";
import { Ch0Scene7 } from "../scenes/Ch0Scene7";
import { Ch0Scene8 } from "../scenes/Ch0Scene8";

// Cut points in frames at 30fps, from the storyboard's per-scene duration estimate.
const CUTS = [0, 300, 480, 720, 930, 1080, 1260, 1410, 1530];
const SCENES = [Ch0Scene1, Ch0Scene2, Ch0Scene3, Ch0Scene4, Ch0Scene5, Ch0Scene6, Ch0Scene7, Ch0Scene8];

export const Chapter0: React.FC = () => (
  <AbsoluteFill>
    {/* Chapter 0 starts at the top of the corrected narration file (0s). */}
    <Audio src={staticFile("narration_normal_speed.wav")} trimAfter={1530} />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 1 — What A Float Actually Is (FP32)

Scenes 9–17 (9 scenes, 56s / 1680f at 30fps). Continues immediately after Chapter 0 in
the master timeline (global offset 1530f / 51s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 9 — Chapter 1 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch1Scene9.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch1Scene9: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Before we even get near INT8, I want to nail down one thing properly.
What IS a 32-bit float, at the bit level?"

## Scene 10 — Chapter 1 (B-Roll, 180–420f / 8s)

```tsx
// scenes/Ch1Scene10.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

type Field = { label: string; bits: number; color: string };
const FIELDS: Field[] = [
  { label: "sign (1 bit)", bits: 1, color: "#f97316" },
  { label: "exponent (8 bits)", bits: 8, color: "#60a5fa" },
  { label: "mantissa (23 bits)", bits: 23, color: "#4ade80" },
];

export const Ch1Scene10: React.FC = () => {
  const frame = useCurrentFrame();
  const boxesIn = interpolate(frame, [0, 50], [0, 32], { extrapolateRight: "clamp" });
  const formulaOpacity = interpolate(frame, [140, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  let drawn = 0;

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", gap: 3 }}>
        {FIELDS.flatMap((field) =>
          Array.from({ length: field.bits }, (_, i) => {
            const index = drawn++;
            const shown = index < boxesIn;
            return (
              <div
                key={`${field.label}-${i}`}
                style={{
                  width: 24,
                  height: 40,
                  background: shown ? field.color : "#1f2937",
                  opacity: shown ? 1 : 0.3,
                  borderRadius: 3,
                }}
              />
            );
          })
        )}
      </div>
      <div style={{ display: "flex", gap: 32, marginTop: 24, fontFamily: "Inter, sans-serif", fontSize: 22 }}>
        {FIELDS.map((field) => (
          <div key={field.label} style={{ color: field.color }}>
            {field.label}
          </div>
        ))}
      </div>
      <div
        style={{
          opacity: formulaOpacity,
          marginTop: 44,
          fontFamily: "monospace",
          fontSize: 34,
          color: "#f9fafb",
        }}
      >
        x = (-1)<sup>s</sup> × (1+f) × 2<sup>(E-127)</sup>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Thirty-two bits, split into three jobs: a sign, an exponent, and a
fraction. One formula turns those bits back into a real number."

## Scene 11 — Chapter 1 (B-Roll, 420–630f / 7s)

```tsx
// scenes/Ch1Scene11.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch1Scene11: React.FC = () => {
  const frame = useCurrentFrame();
  const signOpacity = interpolate(frame, [0, 15, 80, 95], [0, 1, 1, 0.35], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const expOpacity = interpolate(frame, [95, 115], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: signOpacity, fontFamily: "monospace", fontSize: 40, color: "#f97316" }}>
        (-1)<sup>0</sup> / (-1)<sup>1</sup> — sign: plus or minus
      </div>
      <div
        style={{
          opacity: expOpacity,
          marginTop: 40,
          fontFamily: "monospace",
          fontSize: 40,
          color: "#60a5fa",
          textAlign: "center",
        }}
      >
        E − 127
        <div style={{ fontSize: 22, color: "#93c5fd", marginTop: 8 }}>
          stored value is shifted — subtract 127 to get the real exponent
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Sign's just plus or minus. The exponent's a little weirder. It stores a
shifted number, and you subtract 127 to get the real exponent. That's the trick."

## Scene 12 — Chapter 1 (B-Roll, 630–810f / 6s)

```tsx
// scenes/Ch1Scene12.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch1Scene12: React.FC = () => {
  const frame = useCurrentFrame();
  const leadingOpacity = interpolate(frame, [30, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "monospace", fontSize: 48, color: "#f9fafb" }}>
        <span style={{ opacity: leadingOpacity, color: "#4ade80" }}>1.</span>
        01000...
      </div>
      <div
        style={{
          opacity: leadingOpacity,
          marginTop: 16,
          fontFamily: "Inter, sans-serif",
          fontSize: 22,
          color: "#4ade80",
          textTransform: "uppercase",
          letterSpacing: 1,
        }}
      >
        implicit — not stored
      </div>
      <div style={{ marginTop: 28, fontFamily: "Inter, sans-serif", fontSize: 24, color: "#9ca3af" }}>
        23 stored bits → 24 bits of precision
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Here's a nice detail. You only store twenty-three fraction bits, but you
get twenty-four bits of precision, because that leading 1 is just implied. It's
free."

## Scene 13 — Chapter 1 (B-Roll, 810–1020f / 7s)

```tsx
// scenes/Ch1Scene13.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const GROUPS = [
  { text: "0", color: "#f97316", start: 0 },
  { text: "01111111", color: "#60a5fa", start: 25 },
  { text: "000...0", color: "#4ade80", start: 50 },
];

export const Ch1Scene13: React.FC = () => {
  const frame = useCurrentFrame();
  const resultOpacity = interpolate(frame, [150, 175], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", gap: 16, fontFamily: "monospace", fontSize: 44 }}>
        {GROUPS.map((g) => {
          const scale = interpolate(frame, [g.start, g.start + 15], [0.85, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          const opacity = interpolate(frame, [g.start, g.start + 15], [0.3, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <span key={g.text} style={{ color: g.color, scale, opacity }}>
              {g.text}
            </span>
          );
        })}
      </div>
      <div style={{ opacity: resultOpacity, marginTop: 36, fontFamily: "Inter, sans-serif", fontSize: 52, color: "#f9fafb" }}>
        = 1.0
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Let's try one. Sign's zero, exponent lands on zero, fraction's all zeros.
So it's just... one point zero."

## Scene 14 — Chapter 1 (B-Roll, 1020–1230f / 7s)

```tsx
// scenes/Ch1Scene14.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch1Scene14: React.FC = () => {
  const frame = useCurrentFrame();
  const lineWidth = interpolate(frame, [10, 60], [0, 1400], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelsOpacity = interpolate(frame, [60, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 1400, height: 4, background: "#374151" }}>
        <div style={{ position: "absolute", left: 0, top: 0, width: lineWidth, height: 4, background: "#60a5fa" }} />
        <div
          style={{
            opacity: labelsOpacity,
            position: "absolute",
            left: -20,
            top: 20,
            fontFamily: "monospace",
            fontSize: 26,
            color: "#93c5fd",
          }}
        >
          ~10⁻³⁸
        </div>
        <div
          style={{
            opacity: labelsOpacity,
            position: "absolute",
            right: -20,
            top: 20,
            fontFamily: "monospace",
            fontSize: 26,
            color: "#93c5fd",
          }}
        >
          ~10³⁸
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Push that same formula to its limits and FP32 covers roughly ten to the
minus thirty-eight, all the way up to ten to the plus thirty-eight. Huge range. We
don't need to grind through every edge case to know that."

## Scene 15 — Chapter 1 (A-Roll, 1230–1410f / 6s)

```tsx
// scenes/Ch1Scene15.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch1Scene15: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div style={{ opacity, display: "flex", gap: 60, fontFamily: "Inter, sans-serif", fontSize: 30, fontWeight: 700 }}>
          <span style={{ color: "#60a5fa", background: "rgba(17,24,39,0.75)", padding: "10px 22px", borderRadius: 999 }}>
            RANGE
          </span>
          <span style={{ color: "#4ade80", background: "rgba(17,24,39,0.75)", padding: "10px 22px", borderRadius: 999 }}>
            PRECISION
          </span>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Worth keeping these two things separate in your head: how big a number
can get, and how finely you can tell two numbers apart. FP32's generous on both."

## Scene 16 — Chapter 1 (A-Roll, 1410–1560f / 5s)

```tsx
// scenes/Ch1Scene16.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch1Scene16: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So if it's this generous, why would anyone ever want to shrink it?"

## Scene 17 — Chapter 1 (B-Roll, chapter card, 1560–1680f / 4s, no VO)

```tsx
// scenes/Ch1Scene17.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch1Scene17: React.FC = () => (
  <ChapterCard number={2} title="Shrinking the format — FP16." durationInFrames={120} />
);
```

## Chapter 1 assembly

```tsx
// chapters/Chapter1.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch1Scene9 } from "../scenes/Ch1Scene9";
import { Ch1Scene10 } from "../scenes/Ch1Scene10";
import { Ch1Scene11 } from "../scenes/Ch1Scene11";
import { Ch1Scene12 } from "../scenes/Ch1Scene12";
import { Ch1Scene13 } from "../scenes/Ch1Scene13";
import { Ch1Scene14 } from "../scenes/Ch1Scene14";
import { Ch1Scene15 } from "../scenes/Ch1Scene15";
import { Ch1Scene16 } from "../scenes/Ch1Scene16";
import { Ch1Scene17 } from "../scenes/Ch1Scene17";

// Local cut points (frames, 30fps) for this chapter's own 0..1680 timeline.
const CUTS = [0, 180, 420, 630, 810, 1020, 1230, 1410, 1560, 1680];
const SCENES = [
  Ch1Scene9,
  Ch1Scene10,
  Ch1Scene11,
  Ch1Scene12,
  Ch1Scene13,
  Ch1Scene14,
  Ch1Scene15,
  Ch1Scene16,
  Ch1Scene17,
];

// Global offset: Chapter 0 runs an estimated 1530f (51s) before this chapter starts.
const CHAPTER_START_GLOBAL = 1530;

export const Chapter1: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1680}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

**Chapter 1 done (17/169 scenes, 2/18 chapters).** Pausing again per the Animator
contract before Chapter 2 (Shrinking The Format — FP16) — say the word to keep going,
or name how many chapters to batch at once.

---

# Chapters 2–18

# Chapter 2 — Shrinking The Format (FP16)

Scenes 18–24 (7 scenes, 44s / 1320f at 30fps). Picks up at global offset 3210f
(107s) in the master timeline and pulls that slice of
`audio/narration_normal_speed.wav`.

## Scene 18 — Chapter 2 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch2Scene18.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch2Scene18: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "The simple answer: half the bits means half the memory. And on hardware
built for it, roughly half the compute time too."

## Scene 19 — Chapter 2 (B-Roll, 180–360f / 6s)

```tsx
// scenes/Ch2Scene19.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

type Field = { label: string; bits: number; color: string };
const FP32_FIELDS: Field[] = [
  { label: "sign", bits: 1, color: "#f97316" },
  { label: "exponent", bits: 8, color: "#60a5fa" },
  { label: "mantissa", bits: 23, color: "#4ade80" },
];
const FP16_FIELDS: Field[] = [
  { label: "sign", bits: 1, color: "#f97316" },
  { label: "exponent", bits: 5, color: "#60a5fa" },
  { label: "mantissa", bits: 10, color: "#4ade80" },
];

const BitRow: React.FC<{ fields: Field[]; total: number; startFrame: number; frame: number }> = ({
  fields,
  total,
  startFrame,
  frame,
}) => {
  const boxesIn = interpolate(frame, [startFrame, startFrame + 40], [0, total], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  let drawn = 0;
  return (
    <div style={{ display: "flex", gap: 3 }}>
      {fields.flatMap((field) =>
        Array.from({ length: field.bits }, (_, i) => {
          const index = drawn++;
          const shown = index < boxesIn;
          return (
            <div
              key={`${field.label}-${i}`}
              style={{
                width: 24,
                height: 40,
                background: shown ? field.color : "#1f2937",
                opacity: shown ? 1 : 0.3,
                borderRadius: 3,
              }}
            />
          );
        })
      )}
    </div>
  );
};

export const Ch2Scene19: React.FC = () => {
  const frame = useCurrentFrame();
  const labelOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
        <div>
          <div style={{ opacity: labelOpacity, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#6b7280", marginBottom: 8 }}>
            FP32 — 1 / 8 / 23
          </div>
          <BitRow fields={FP32_FIELDS} total={32} startFrame={0} frame={frame} />
        </div>
        <div>
          <div style={{ opacity: labelOpacity, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#6b7280", marginBottom: 8 }}>
            FP16 — 1 / 5 / 10
          </div>
          <BitRow fields={FP16_FIELDS} total={16} startFrame={40} frame={frame} />
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Same basic idea as FP32, just fewer bits in two of the three fields."

## Scene 20 — Chapter 2 (B-Roll, 360–570f / 7s)

```tsx
// scenes/Ch2Scene20.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch2Scene20: React.FC = () => {
  const frame = useCurrentFrame();
  const fullWidth = interpolate(frame, [0, 30], [0, 1400], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const shrunkWidth = interpolate(frame, [50, 90], [1400, 260], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const lineWidth = frame < 50 ? fullWidth : shrunkWidth;

  const valuesOpacity = interpolate(frame, [120, 145], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 1400, height: 4, background: "#374151" }}>
        <div
          style={{
            position: "absolute",
            left: (1400 - lineWidth) / 2,
            top: 0,
            width: lineWidth,
            height: 4,
            background: "#60a5fa",
          }}
        />
      </div>
      <div style={{ display: "flex", gap: 100, marginTop: 60, opacity: valuesOpacity }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "monospace", fontSize: 30, color: "#93c5fd" }}>~3.4 × 10³⁸</div>
          <div style={{ fontFamily: "Inter, sans-serif", fontSize: 18, color: "#6b7280", marginTop: 6 }}>FP32 max</div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "monospace", fontSize: 40, color: "#f9fafb", fontWeight: 700 }}>65504</div>
          <div style={{ fontFamily: "Inter, sans-serif", fontSize: 18, color: "#6b7280", marginTop: 6 }}>FP16 max</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Fewer exponent bits, much smaller range. Work through the same math and
FP16 tops out around sixty-five thousand. That's it."

## Scene 21 — Chapter 2 (B-Roll, 570–810f / 8s)

```tsx
// scenes/Ch2Scene21.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch2Scene21: React.FC = () => {
  const frame = useCurrentFrame();
  const valueOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const arrowProgress = interpolate(frame, [40, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const stampScale = interpolate(frame, [90, 105], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const stampOpacity = interpolate(frame, [90, 105], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 40 }}>
        <div style={{ opacity: valueOpacity, textAlign: "center" }}>
          <div style={{ fontFamily: "monospace", fontSize: 40, color: "#93c5fd" }}>100000</div>
          <div style={{ fontFamily: "Inter, sans-serif", fontSize: 18, color: "#6b7280", marginTop: 6 }}>fits FP32 fine</div>
        </div>
        <div
          style={{
            width: 80 * arrowProgress,
            height: 3,
            background: "#6b7280",
            opacity: arrowProgress > 0 ? 1 : 0,
          }}
        />
        <div
          style={{
            scale: stampScale,
            opacity: stampOpacity,
            textAlign: "center",
            border: "2px solid #ef4444",
            borderRadius: 10,
            padding: "18px 28px",
          }}
        >
          <div style={{ fontFamily: "monospace", fontSize: 30, color: "#ef4444", fontWeight: 700 }}>
            OVERFLOW → +∞
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So a number FP32 handles without blinking, say a hundred thousand, just
doesn't fit in FP16 at all."

## Scene 22 — Chapter 2 (A-Roll, 810–1020f / 7s)

```tsx
// scenes/Ch2Scene22.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch2Scene22: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [15, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div style={{ opacity, display: "flex", gap: 24, fontFamily: "Inter, sans-serif", fontSize: 30, fontWeight: 700 }}>
          <span style={{ color: "#f9fafb", background: "rgba(17,24,39,0.75)", padding: "10px 22px", borderRadius: 999 }}>
            NO scale.
          </span>
          <span style={{ color: "#f9fafb", background: "rgba(17,24,39,0.75)", padding: "10px 22px", borderRadius: 999 }}>
            NO calibration.
          </span>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "And notice, I didn't need anything extra for any of this. No outside
data. Those sixteen bits describe the number on their own. It's just a format
conversion."

## Scene 23 — Chapter 2 (A-Roll, 1020–1200f / 6s)

```tsx
// scenes/Ch2Scene23.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch2Scene23: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "But FP16 still has an exponent field in there. So what happens when we go
all the way down to 8 bits, and there's just no room left for one at all?"

## Scene 24 — Chapter 2 (B-Roll, chapter card, 1200–1320f / 4s, no VO)

```tsx
// scenes/Ch2Scene24.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch2Scene24: React.FC = () => (
  <ChapterCard number={3} title="Where floats stop working." durationInFrames={120} />
);
```

## Chapter 2 assembly

```tsx
// chapters/Chapter2.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch2Scene18 } from "../scenes/Ch2Scene18";
import { Ch2Scene19 } from "../scenes/Ch2Scene19";
import { Ch2Scene20 } from "../scenes/Ch2Scene20";
import { Ch2Scene21 } from "../scenes/Ch2Scene21";
import { Ch2Scene22 } from "../scenes/Ch2Scene22";
import { Ch2Scene23 } from "../scenes/Ch2Scene23";
import { Ch2Scene24 } from "../scenes/Ch2Scene24";

// Local cut points (frames, 30fps) for this chapter's own 0..1320 timeline.
const CUTS = [0, 180, 360, 570, 810, 1020, 1200, 1320];
const SCENES = [
  Ch2Scene18,
  Ch2Scene19,
  Ch2Scene20,
  Ch2Scene21,
  Ch2Scene22,
  Ch2Scene23,
  Ch2Scene24,
];

// Global offset: chapters 0-1 run an estimated 3210f (107s) before this chapter starts.
const CHAPTER_START_GLOBAL = 3210;

export const Chapter2: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1320}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 3 — Where Floats Stop Working (Why INT8 Is Different)

Scenes 25–32 (8 scenes, 49s / 1470f at 30fps). Picks up at global offset 4530f
(151s) in the master timeline, immediately after Chapter 2.

## Scene 25 — Chapter 3 (A-Roll, 0–210f / 7s)

```tsx
// scenes/Ch3Scene25.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch3Scene25: React.FC = () => {
  const frame = useCurrentFrame();
  const textOpacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const strikeProgress = interpolate(frame, [95, 130], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div style={{ position: "relative", opacity: textOpacity }}>
          <div
            style={{
              fontFamily: "Inter, sans-serif",
              fontSize: 32,
              fontWeight: 700,
              color: "#f9fafb",
              background: "rgba(17,24,39,0.75)",
              padding: "14px 28px",
              borderRadius: 8,
            }}
          >
            INT8 = an even smaller float?
          </div>
          <div
            style={{
              position: "absolute",
              left: 28,
              top: "50%",
              height: 4,
              width: `${strikeProgress}%`,
              background: "#ef4444",
            }}
          />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Honestly, my first assumption was that INT8 is basically FP16, but
smaller. It's not. It's not that at all."

## Scene 26 — Chapter 3 (B-Roll, 210–390f / 6s)

```tsx
// scenes/Ch3Scene26.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch3Scene26: React.FC = () => {
  const frame = useCurrentFrame();
  const boxesIn = interpolate(frame, [0, 35], [0, 8], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const labelOpacity = interpolate(frame, [50, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", gap: 6 }}>
        {Array.from({ length: 8 }, (_, i) => {
          const shown = i < boxesIn;
          return (
            <div
              key={i}
              style={{
                width: 48,
                height: 64,
                background: "#1f2937",
                border: "2px solid #9ca3af",
                opacity: shown ? 1 : 0.2,
                borderRadius: 4,
              }}
            />
          );
        })}
      </div>
      <div style={{ opacity: labelOpacity, marginTop: 32, fontFamily: "Inter, sans-serif", fontSize: 26, color: "#9ca3af" }}>
        signed integer, -128 to 127
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "INT8 is just a plain signed integer. There's no sign-exponent-mantissa
split anymore. It has no idea what range it's supposed to represent."

## Scene 27 — Chapter 3 (B-Roll, 390–600f / 7s)

```tsx
// scenes/Ch3Scene27.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const TICK_COUNT = 32; // sampled ticks standing in for the 256 discrete steps

export const Ch3Scene27: React.FC = () => {
  const frame = useCurrentFrame();
  const lineOpacity = interpolate(frame, [0, 15, 90, 110], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const ticksOpacity = interpolate(frame, [100, 125], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelsOpacity = interpolate(frame, [140, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 1400, height: 80 }}>
        <div
          style={{
            opacity: lineOpacity,
            position: "absolute",
            left: 0,
            top: 38,
            width: 1400,
            height: 4,
            background: "#60a5fa",
          }}
        />
        <div style={{ opacity: ticksOpacity, position: "absolute", left: 0, top: 20, width: 1400, display: "flex", justifyContent: "space-between" }}>
          {Array.from({ length: TICK_COUNT }, (_, i) => (
            <div key={i} style={{ width: 3, height: 40, background: "#93c5fd" }} />
          ))}
        </div>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", width: 1400, opacity: labelsOpacity, marginTop: 8 }}>
        <span style={{ fontFamily: "monospace", fontSize: 24, color: "#93c5fd" }}>-128</span>
        <span style={{ fontFamily: "Inter, sans-serif", fontSize: 20, color: "#6b7280" }}>256 integers</span>
        <span style={{ fontFamily: "monospace", fontSize: 24, color: "#93c5fd" }}>127</span>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So the problem becomes this: how do you squeeze a continuous range of
real numbers down into just two hundred fifty-six integers?"

## Scene 28 — Chapter 3 (A-Roll, 600–780f / 6s)

```tsx
// scenes/Ch3Scene28.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch3Scene28: React.FC = () => {
  const frame = useCurrentFrame();
  const scale = interpolate(frame, [30, 50], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const opacity = interpolate(frame, [30, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div
          style={{
            scale,
            opacity,
            fontFamily: "Inter, sans-serif",
            fontSize: 72,
            fontWeight: 700,
            color: "#f97316",
            letterSpacing: 4,
          }}
        >
          SCALE.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "You need something extra. Some number, stored alongside the integers,
that says what one INT8 step is actually worth."

## Scene 29 — Chapter 3 (B-Roll, 780–960f / 6s)

```tsx
// scenes/Ch3Scene29.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const FILES = [
  "build_and_bench.py",
  "calibrate_int8.py",
  "export_onnx_to_trt.py",
  "export.py",
  "measure_full_metrics.py",
  "nsys_run_engine.py",
  "profile_layers.py",
  "verify_accuracy_venv.py",
];
const COMMAND = "find scripts/ -maxdepth 1 -type f";

export const Ch3Scene29: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 25], [0, COMMAND.length], {
    extrapolateRight: "clamp",
  }));
  const zoomScale = interpolate(frame, [110, 150], [1, 1.5], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const dimOthers = interpolate(frame, [110, 150], [1, 0.25], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 30 }}>
        $ {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ marginTop: 24 }}>
        {FILES.map((file, i) => {
          const start = 30 + i * 8;
          const isTarget = file === "calibrate_int8.py";
          const opacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          return (
            <div
              key={file}
              style={{
                opacity: opacity * (isTarget ? 1 : dimOthers),
                scale: isTarget ? zoomScale : 1,
                transformOrigin: "left center",
                color: isTarget ? "#4ade80" : "#e5e7eb",
                fontSize: 26,
                padding: "4px 0",
              }}
            >
              ./scripts/{file}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "This repo has a whole script dedicated to exactly that problem. Nothing
like it exists for the FP16 build."

## Scene 30 — Chapter 3 (B-Roll, 960–1200f / 8s)

```tsx
// scenes/Ch3Scene30.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LEFT_LINES = [
  "config = builder.create_builder_config()",
  "config.set_flag(trt.BuilderFlag.INT8)",
  "engine = builder.build_engine(network, config)",
];
const RIGHT_LINES = [
  "class Int8Calibrator(trt.IInt8EntropyCalibrator2):",
  "    def __init__(self, image_dir):",
  "        self.images = load_images(image_dir)  # 500 real images",
  "    def get_batch(self, names):",
  "        return next_batch(self.images)",
  "    def read_calibration_cache(self): ...",
  "    def write_calibration_cache(self, cache): ...",
];

export const Ch3Scene30: React.FC = () => {
  const frame = useCurrentFrame();
  const leftOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const rightOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", flexDirection: "row" }}>
      <div style={{ flex: 1, padding: "80px 40px", opacity: leftOpacity }}>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 20, color: "#6b7280", marginBottom: 16 }}>
          build_and_bench.py
        </div>
        {LEFT_LINES.map((line, i) => (
          <div key={i} style={{ fontFamily: "monospace", fontSize: 20, color: "#e5e7eb", padding: "3px 0" }}>
            {line}
          </div>
        ))}
      </div>
      <div style={{ width: 1, background: "#374151" }} />
      <div style={{ flex: 1, padding: "80px 40px", opacity: rightOpacity }}>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 20, color: "#6b7280", marginBottom: 16 }}>
          calibrate_int8.py
        </div>
        {RIGHT_LINES.map((line, i) => (
          <div key={i} style={{ fontFamily: "monospace", fontSize: 20, color: "#e5e7eb", padding: "3px 0" }}>
            {line}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Look at the difference. One file just flips a builder flag. The other
one loads five hundred real images. That's the whole asymmetry, right there in the
code."

## Scene 31 — Chapter 3 (A-Roll, 1200–1350f / 5s)

```tsx
// scenes/Ch3Scene31.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

const STEPS = ["range", "scale", "mapping"];

export const Ch3Scene31: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 120 }}>
        <div style={{ display: "flex", gap: 40 }}>
          {STEPS.map((step, i) => {
            const start = 10 + i * 15;
            const scale = interpolate(frame, [start, start + 12], [0.6, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.34, 1.56, 0.64, 1),
            });
            const opacity = interpolate(frame, [start, start + 12], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            return (
              <div
                key={step}
                style={{
                  scale,
                  opacity,
                  fontFamily: "Inter, sans-serif",
                  fontSize: 32,
                  color: "#f9fafb",
                  background: "rgba(17,24,39,0.75)",
                  border: "1px solid #374151",
                  borderRadius: 999,
                  padding: "14px 32px",
                }}
              >
                {step}
                {i < STEPS.length - 1 && <span style={{ color: "#6b7280", marginLeft: 24 }}>→</span>}
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Alright, let's build that scale ourselves, starting with the easiest
version of the problem."

## Scene 32 — Chapter 3 (B-Roll, chapter card, 1350–1470f / 4s, no VO)

```tsx
// scenes/Ch3Scene32.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch3Scene32: React.FC = () => (
  <ChapterCard number={4} title="The simplest mapping — symmetric quantization." durationInFrames={120} />
);
```

## Chapter 3 assembly

```tsx
// chapters/Chapter3.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch3Scene25 } from "../scenes/Ch3Scene25";
import { Ch3Scene26 } from "../scenes/Ch3Scene26";
import { Ch3Scene27 } from "../scenes/Ch3Scene27";
import { Ch3Scene28 } from "../scenes/Ch3Scene28";
import { Ch3Scene29 } from "../scenes/Ch3Scene29";
import { Ch3Scene30 } from "../scenes/Ch3Scene30";
import { Ch3Scene31 } from "../scenes/Ch3Scene31";
import { Ch3Scene32 } from "../scenes/Ch3Scene32";

// Local cut points (frames, 30fps) for this chapter's own 0..1470 timeline.
const CUTS = [0, 210, 390, 600, 780, 960, 1200, 1350, 1470];
const SCENES = [
  Ch3Scene25,
  Ch3Scene26,
  Ch3Scene27,
  Ch3Scene28,
  Ch3Scene29,
  Ch3Scene30,
  Ch3Scene31,
  Ch3Scene32,
];

// Global offset: chapters 0-2 run an estimated 4530f (151s) before this chapter starts.
const CHAPTER_START_GLOBAL = 4530;

export const Chapter3: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1470}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 4 — The Simplest Mapping (Symmetric Quantization)

Scenes 33–40 (8 scenes, 50s / 1500f at 30fps). Picks up at global offset 6000f
(200s) in the master timeline, immediately after Chapter 3.

## Scene 33 — Chapter 4 (B-Roll, 0–180f / 6s)

```tsx
// scenes/Ch4Scene33.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch4Scene33: React.FC = () => {
  const frame = useCurrentFrame();
  const lineWidth = interpolate(frame, [10, 60], [0, 1000], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelsOpacity = interpolate(frame, [60, 85], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const zeroOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 1000, height: 4, background: "#374151" }}>
        <div
          style={{
            position: "absolute",
            left: (1000 - lineWidth) / 2,
            top: 0,
            width: lineWidth,
            height: 4,
            background: "#4ade80",
          }}
        />
        <div
          style={{
            opacity: zeroOpacity,
            scale: zeroOpacity,
            position: "absolute",
            left: "50%",
            top: -8,
            width: 20,
            height: 20,
            marginLeft: -10,
            borderRadius: "50%",
            background: "#f9fafb",
          }}
        />
        <div style={{ opacity: labelsOpacity, position: "absolute", left: -20, top: 20, fontFamily: "monospace", fontSize: 26, color: "#93c5fd" }}>
          -2.0
        </div>
        <div style={{ opacity: zeroOpacity, position: "absolute", left: "50%", top: 20, marginLeft: -14, fontFamily: "monospace", fontSize: 26, color: "#f9fafb" }}>
          0
        </div>
        <div style={{ opacity: labelsOpacity, position: "absolute", right: -20, top: 20, fontFamily: "monospace", fontSize: 26, color: "#93c5fd" }}>
          +2.0
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Easiest case first: a range that's already sitting centered on zero."

## Scene 34 — Chapter 4 (B-Roll, 180–390f / 7s)

```tsx
// scenes/Ch4Scene34.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch4Scene34: React.FC = () => {
  const frame = useCurrentFrame();
  const formulaOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const substOpacity = interpolate(frame, [70, 95], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const resultOpacity = interpolate(frame, [140, 165], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: formulaOpacity, fontFamily: "monospace", fontSize: 40, color: "#f9fafb" }}>
        s = max(|x|) / 127
      </div>
      <div style={{ opacity: substOpacity, marginTop: 24, fontFamily: "monospace", fontSize: 40, color: "#93c5fd" }}>
        s = 2 / 127
      </div>
      <div style={{ opacity: resultOpacity, marginTop: 24, fontFamily: "monospace", fontSize: 52, color: "#4ade80", fontWeight: 700 }}>
        s ≈ 0.01575
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "One number needed: take the largest absolute value, divide it by a
hundred twenty-seven. That gives us about 0.01575."

## Scene 35 — Chapter 4 (B-Roll, 390–630f / 8s)

```tsx
// scenes/Ch4Scene35.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch4Scene35: React.FC = () => {
  const frame = useCurrentFrame();
  const quantOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const quantResultOpacity = interpolate(frame, [55, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const dequantOpacity = interpolate(frame, [110, 135], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const errorOpacity = interpolate(frame, [170, 195], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: quantOpacity, fontFamily: "monospace", fontSize: 32, color: "#e5e7eb" }}>
        round(1.0 / 0.01575) = <span style={{ opacity: quantResultOpacity, color: "#4ade80", fontWeight: 700 }}>64</span>
      </div>
      <div style={{ opacity: dequantOpacity, marginTop: 24, fontFamily: "monospace", fontSize: 32, color: "#e5e7eb" }}>
        64 × 0.01575 ≈ 1.008
      </div>
      <div
        style={{
          opacity: errorOpacity,
          marginTop: 28,
          display: "flex",
          alignItems: "center",
          gap: 20,
          fontFamily: "monospace",
          fontSize: 24,
        }}
      >
        <span style={{ color: "#9ca3af" }}>original 1.000</span>
        <span
          style={{
            color: "#ef4444",
            border: "1px solid #ef4444",
            borderRadius: 6,
            padding: "6px 14px",
          }}
        >
          +0.008 error
        </span>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Try it on one point zero. That becomes sixty-four. Multiply back by the
scale and you get about one point zero zero eight. That little gap is the
quantization error."

## Scene 36 — Chapter 4 (A-Roll, 630–780f / 5s)

```tsx
// scenes/Ch4Scene36.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch4Scene36: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div
          style={{
            opacity,
            fontFamily: "monospace",
            fontSize: 36,
            fontWeight: 700,
            color: "#4ade80",
            background: "rgba(17,24,39,0.75)",
            padding: "12px 26px",
            borderRadius: 8,
          }}
        >
          0.0 → 0 exactly.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "One nice thing: zero always lands exactly on zero here. No offset
needed."

## Scene 37 — Chapter 4 (B-Roll, 780–1020f / 8s)

```tsx
// scenes/Ch4Scene37.tsx
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

const LINES = [
  "def quantize_symmetric(x, s):",
  '    """Map a float to its INT8 code."""',
  "    q = x / s",
  "    q = round(q)",
  "    return q",
];
const FULL_TEXT = LINES.join("\n");

export const Ch4Scene37: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(
    interpolate(frame, [10, 180], [0, FULL_TEXT.length], { extrapolateRight: "clamp" })
  );
  const shownText = FULL_TEXT.slice(0, typedChars);
  const shownLines = shownText.split("\n");

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      {shownLines.map((line, i) => (
        <div key={i} style={{ color: "#e5e7eb", fontSize: 30, minHeight: 40 }}>
          {line}
          {i === shownLines.length - 1 && (
            <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
          )}
        </div>
      ))}
    </AbsoluteFill>
  );
};
```
Narration: "Okay, enough talking about it. Let me just write the actual function."

## Scene 38 — Chapter 4 (B-Roll, 1020–1200f / 6s)

```tsx
// scenes/Ch4Scene38.tsx
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

const COMMAND = "quantize_symmetric(1.0, 0.01575)";

export const Ch4Scene38: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 35], [0, COMMAND.length], {
    extrapolateRight: "clamp",
  }));
  const resultOpacity = interpolate(frame, [60, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 32 }}>
        &gt;&gt;&gt; {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ opacity: resultOpacity, marginTop: 16, color: "#f9fafb", fontSize: 40, fontWeight: 700 }}>
        64
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Run it, and yep, same sixty-four we got by hand."

## Scene 39 — Chapter 4 (A-Roll, 1200–1380f / 6s)

```tsx
// scenes/Ch4Scene39.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch4Scene39: React.FC = () => {
  const frame = useCurrentFrame();
  const lineOpacity = interpolate(frame, [30, 55], [0, 0.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div style={{ opacity: lineOpacity, position: "relative", width: 900, height: 4, background: "#374151" }}>
          <div style={{ position: "absolute", left: 200, top: -8, width: 20, height: 20, marginLeft: -10, borderRadius: "50%", background: "#93c5fd" }} />
          <div style={{ position: "absolute", left: 200, top: 20, marginLeft: -22, fontFamily: "monospace", fontSize: 22, color: "#93c5fd" }}>
            min -1
          </div>
          <div style={{ position: "absolute", right: 100, top: -8, width: 20, height: 20, marginRight: -10, borderRadius: "50%", background: "#93c5fd" }} />
          <div style={{ position: "absolute", right: 100, top: 20, marginRight: -20, fontFamily: "monospace", fontSize: 22, color: "#93c5fd" }}>
            max +3
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Hang onto this function, we'll reuse it on a real activation soon. But
first: what happens when the real range isn't centered on zero at all?"

## Scene 40 — Chapter 4 (B-Roll, chapter card, 1380–1500f / 4s, no VO)

```tsx
// scenes/Ch4Scene40.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch4Scene40: React.FC = () => (
  <ChapterCard number={5} title="When zero isn't in the middle." durationInFrames={120} />
);
```

## Chapter 4 assembly

```tsx
// chapters/Chapter4.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch4Scene33 } from "../scenes/Ch4Scene33";
import { Ch4Scene34 } from "../scenes/Ch4Scene34";
import { Ch4Scene35 } from "../scenes/Ch4Scene35";
import { Ch4Scene36 } from "../scenes/Ch4Scene36";
import { Ch4Scene37 } from "../scenes/Ch4Scene37";
import { Ch4Scene38 } from "../scenes/Ch4Scene38";
import { Ch4Scene39 } from "../scenes/Ch4Scene39";
import { Ch4Scene40 } from "../scenes/Ch4Scene40";

// Local cut points (frames, 30fps) for this chapter's own 0..1500 timeline.
const CUTS = [0, 180, 390, 630, 780, 1020, 1200, 1380, 1500];
const SCENES = [
  Ch4Scene33,
  Ch4Scene34,
  Ch4Scene35,
  Ch4Scene36,
  Ch4Scene37,
  Ch4Scene38,
  Ch4Scene39,
  Ch4Scene40,
];

// Global offset: chapters 0-3 run an estimated 6000f (200s) before this chapter starts.
const CHAPTER_START_GLOBAL = 6000;

export const Chapter4: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1500}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 5 — When Zero Isn't In The Middle (Asymmetric Quantization)

Scenes 41–47 (7 scenes, 43s / 1290f at 30fps). Continues immediately after Chapter 4 in
the master timeline (global offset 7500f / 250s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Shared primitive: RangeBar

```tsx
// RangeBar.tsx
import { interpolate, useCurrentFrame, Easing } from "remotion";

type Segment = {
  from: number;
  to: number;
  color: string;
  revealFrame: number;
  height?: number;
  outline?: boolean;
  dashed?: boolean;
};
type Tick = { value: number; label: string; revealFrame: number; color?: string };

export const RangeBar: React.FC<{
  min: number;
  max: number;
  width?: number;
  segments: Segment[];
  ticks?: Tick[];
  fromFrame?: number;
}> = ({ min, max, width = 1400, segments, ticks = [], fromFrame = 0 }) => {
  const frame = useCurrentFrame() - fromFrame;
  const toPx = (v: number) => ((v - min) / (max - min)) * width;

  return (
    <div style={{ position: "relative", width, height: 4, background: "#374151" }}>
      {segments.map((seg, i) => {
        const segFrame = frame - seg.revealFrame;
        const h = seg.height ?? 8;
        const fullWidth = toPx(seg.to) - toPx(seg.from);
        const w = interpolate(segFrame, [0, 25], [0, fullWidth], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.16, 1, 0.3, 1),
        });
        const opacity = interpolate(segFrame, [0, 12], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: toPx(seg.from),
              top: -h / 2 + 2,
              width: w,
              height: h,
              opacity,
              borderRadius: 4,
              background: seg.outline ? "transparent" : seg.color,
              border: seg.outline ? `2px ${seg.dashed ? "dashed" : "solid"} ${seg.color}` : "none",
            }}
          />
        );
      })}
      {ticks.map((tick) => {
        const opacity = interpolate(frame, [tick.revealFrame, tick.revealFrame + 15], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            key={tick.value}
            style={{
              position: "absolute",
              left: toPx(tick.value) - 40,
              top: 20,
              width: 80,
              textAlign: "center",
              opacity,
              fontFamily: "monospace",
              fontSize: 22,
              color: tick.color ?? "#93c5fd",
            }}
          >
            {tick.label}
          </div>
        );
      })}
    </div>
  );
};
```
A thin animated axis reused for every range/scale comparison in Chapters 5–6: a dim
track, colored `segments` that grow in from a `revealFrame` (optionally `outline` +
`dashed` for a boundary box rather than a fill), and fading-in `ticks` for endpoint
labels. Imported as `../RangeBar` from scene files.

## Scene 41 — Chapter 5 (B-Roll, 0–210f / 7s)

```tsx
// scenes/Ch5Scene41.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { RangeBar } from "../RangeBar";

export const Ch5Scene41: React.FC = () => {
  const frame = useCurrentFrame();
  const actualLabelOpacity = interpolate(frame, [5, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const symLabelOpacity = interpolate(frame, [75, 95], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const wastedLabelOpacity = interpolate(frame, [130, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: actualLabelOpacity, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#93c5fd", marginBottom: 20 }}>
        actual value range
      </div>
      <RangeBar
        min={-3}
        max={3}
        segments={[
          { from: -1, to: 3, color: "#60a5fa", revealFrame: 10, height: 10 },
          { from: -3, to: 3, color: "#6b7280", revealFrame: 75, height: 70, outline: true, dashed: true },
          { from: -3, to: -1, color: "rgba(239,68,68,0.35)", revealFrame: 120, height: 70 },
        ]}
        ticks={[
          { value: -3, label: "-3", revealFrame: 75, color: "#9ca3af" },
          { value: -1, label: "-1", revealFrame: 10, color: "#93c5fd" },
          { value: 3, label: "+3", revealFrame: 10, color: "#93c5fd" },
        ]}
      />
      <div style={{ opacity: symLabelOpacity, marginTop: 100, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#6b7280" }}>
        forced symmetric range [-3, +3]
      </div>
      <div style={{ opacity: wastedLabelOpacity, marginTop: 10, fontFamily: "Inter, sans-serif", fontSize: 24, color: "#ef4444", fontWeight: 700 }}>
        ~1/3 of the codes, wasted
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "If you force this into a symmetric range, you end up wasting something like
a third of your INT8 codes on values that never even show up."

## Scene 42 — Chapter 5 (B-Roll, 210–450f / 8s)

```tsx
// scenes/Ch5Scene42.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch5Scene42: React.FC = () => {
  const frame = useCurrentFrame();
  const scaleOpacity = interpolate(frame, [10, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const zeroOpacity = interpolate(frame, [95, 115], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const arrowOpacity = interpolate(frame, [175, 195], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: scaleOpacity, fontFamily: "monospace", fontSize: 34, color: "#60a5fa", textAlign: "center" }}>
        s = (max − min) / (qmax − qmin) = 4 / 255 ≈ 0.01569
      </div>
      <div style={{ opacity: zeroOpacity, marginTop: 32, fontFamily: "monospace", fontSize: 34, color: "#f97316", textAlign: "center" }}>
        z = round(qmin − min/s) ≈ -64
      </div>
      <div style={{ opacity: arrowOpacity, marginTop: 56, display: "flex", alignItems: "center", gap: 28, fontFamily: "monospace", fontSize: 28 }}>
        <span style={{ color: "#9ca3af" }}>FP32 zero → 0.0</span>
        <span style={{ color: "#6b7280", fontSize: 36 }}>→</span>
        <span style={{ color: "#f97316", fontWeight: 700 }}>INT8 → -64</span>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So the scale spans the whole range instead, across all two hundred fifty-five
steps. And now zero doesn't land on integer zero anymore, it lands wherever this
zero-point number says it should."

## Scene 43 — Chapter 5 (B-Roll, 450–630f / 6s)

```tsx
// scenes/Ch5Scene43.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch5Scene43: React.FC = () => {
  const frame = useCurrentFrame();
  const symOpacity = interpolate(frame, [10, 28], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const asymOpacity = interpolate(frame, [55, 73], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const extraTermScale = interpolate(frame, [90, 105], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const noteOpacity = interpolate(frame, [120, 138], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: symOpacity, fontFamily: "monospace", fontSize: 36, color: "#9ca3af" }}>
        symmetric:&nbsp;&nbsp; q = round(x / s)
      </div>
      <div style={{ opacity: asymOpacity, marginTop: 24, fontFamily: "monospace", fontSize: 36, color: "#f9fafb" }}>
        asymmetric: q = round(x / s)
        <span style={{ scale: extraTermScale, display: "inline-block", color: "#f97316", fontWeight: 700 }}> + z</span>
      </div>
      <div style={{ opacity: noteOpacity, marginTop: 44, fontFamily: "Inter, sans-serif", fontSize: 24, color: "#6b7280" }}>
        one extra term — that's the whole difference
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "One extra term added on. That's the whole difference between the two."

## Scene 44 — Chapter 5 (A-Roll, 630–810f / 6s)

```tsx
// scenes/Ch5Scene44.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch5Scene44: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="ReLU output ≥ 0" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "And this comes up constantly in real networks. Anything right after a ReLU
is never negative."

## Scene 45 — Chapter 5 (B-Roll, 810–1020f / 7s)

```tsx
// scenes/Ch5Scene45.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CODE = "def quantize_asymmetric(x, s, z): return round(x/s) + z";

export const Ch5Scene45: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(
    interpolate(frame, [0, 80], [0, CODE.length], { extrapolateRight: "clamp" })
  );
  const terminalOpacity = interpolate(frame, [110, 130], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const matchOpacity = interpolate(frame, [165, 185], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#e5e7eb", fontSize: 26 }}>
        {CODE.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ opacity: terminalOpacity, marginTop: 60, borderTop: "1px solid #374151", paddingTop: 30 }}>
        <div style={{ color: "#22c55e", fontSize: 24 }}>
          &gt;&gt;&gt; quantize_asymmetric(3.0, 0.01569, -64)
        </div>
        <div style={{ color: "#f9fafb", fontSize: 24, marginTop: 8 }}>127</div>
        <div style={{ opacity: matchOpacity, color: "#4ade80", fontSize: 22, marginTop: 16 }}>
          ✓ matches the hand-derived value
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Barely different code, just one extra piece, and it matches what we got by
hand."

## Scene 46 — Chapter 5 (A-Roll, 1020–1170f / 5s)

```tsx
// scenes/Ch5Scene46.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch5Scene46: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Okay, one scale per tensor. But is one scale always enough?"

## Scene 47 — Chapter 5 (B-Roll, chapter card, 1170–1290f / 4s, no VO)

```tsx
// scenes/Ch5Scene47.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch5Scene47: React.FC = () => (
  <ChapterCard number={6} title="One scale, or many?" durationInFrames={120} />
);
```

## Chapter 5 assembly

```tsx
// chapters/Chapter5.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch5Scene41 } from "../scenes/Ch5Scene41";
import { Ch5Scene42 } from "../scenes/Ch5Scene42";
import { Ch5Scene43 } from "../scenes/Ch5Scene43";
import { Ch5Scene44 } from "../scenes/Ch5Scene44";
import { Ch5Scene45 } from "../scenes/Ch5Scene45";
import { Ch5Scene46 } from "../scenes/Ch5Scene46";
import { Ch5Scene47 } from "../scenes/Ch5Scene47";

// Local cut points (frames, 30fps) for this chapter's own 0..1290 timeline.
const CUTS = [0, 210, 450, 630, 810, 1020, 1170, 1290];
const SCENES = [
  Ch5Scene41,
  Ch5Scene42,
  Ch5Scene43,
  Ch5Scene44,
  Ch5Scene45,
  Ch5Scene46,
  Ch5Scene47,
];

// Global offsets given for Chapter 5: 7500f–8790f.
const CHAPTER_START_GLOBAL = 7500;
const CHAPTER_END_GLOBAL = 8790;

export const Chapter5: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_END_GLOBAL}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 6 — One Scale Or Many? (Per-Tensor vs Per-Channel)

Scenes 48–57 (10 scenes, 59s / 1770f at 30fps). Continues immediately after Chapter 5 in
the master timeline (global offset 8790f / 293s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 48 — Chapter 6 (B-Roll, 0–210f / 7s)

```tsx
// scenes/Ch6Scene48.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { RangeBar } from "../RangeBar";

export const Ch6Scene48: React.FC = () => {
  const frame = useCurrentFrame();
  const label1Opacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const label2Opacity = interpolate(frame, [95, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: label1Opacity, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#93c5fd" }}>
        Channel 1 — tiny values
      </div>
      <RangeBar
        min={-10}
        max={10}
        segments={[{ from: -0.3, to: 0.3, color: "#60a5fa", revealFrame: 10, height: 14 }]}
        ticks={[
          { value: -0.3, label: "-0.3", revealFrame: 10, color: "#93c5fd" },
          { value: 0.3, label: "+0.3", revealFrame: 10, color: "#93c5fd" },
        ]}
      />
      <div style={{ opacity: label2Opacity, marginTop: 80, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#4ade80" }}>
        Channel 2 — huge values
      </div>
      <RangeBar
        min={-10}
        max={10}
        segments={[{ from: -10, to: 8, color: "#4ade80", revealFrame: 95, height: 14 }]}
        ticks={[
          { value: -10, label: "-10", revealFrame: 95, color: "#4ade80" },
          { value: 8, label: "+8", revealFrame: 95, color: "#4ade80" },
        ]}
      />
    </AbsoluteFill>
  );
};
```
Narration: "Picture one weight tensor where one channel's values are tiny, and a
different channel's values are huge."

## Scene 49 — Chapter 6 (B-Roll, 210–390f / 6s)

```tsx
// scenes/Ch6Scene49.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { RangeBar } from "../RangeBar";

export const Ch6Scene49: React.FC = () => {
  const frame = useCurrentFrame();
  const sharedLabelOpacity = interpolate(frame, [5, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const calloutOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: sharedLabelOpacity, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>
        one shared scale, sized for Channel 2
      </div>
      <RangeBar
        min={-10}
        max={8}
        segments={[
          { from: -10, to: 8, color: "#374151", revealFrame: 10, height: 14, outline: true },
          { from: -0.3, to: 0.3, color: "#ef4444", revealFrame: 60, height: 34 },
        ]}
        ticks={[
          { value: -10, label: "-10", revealFrame: 10, color: "#6b7280" },
          { value: 8, label: "+8", revealFrame: 10, color: "#6b7280" },
        ]}
      />
      <div style={{ opacity: calloutOpacity, marginTop: 90, fontFamily: "Inter, sans-serif", fontSize: 24, color: "#ef4444", fontWeight: 700, textAlign: "center" }}>
        Channel 1's whole range → a handful of codes near zero
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "If you use one shared scale for the whole thing, that small channel
basically gets no usable resolution at all."

## Scene 50 — Chapter 6 (B-Roll, 390–570f / 6s)

```tsx
// scenes/Ch6Scene50.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { RangeBar } from "../RangeBar";

export const Ch6Scene50: React.FC = () => {
  const frame = useCurrentFrame();
  const label1Opacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const label2Opacity = interpolate(frame, [75, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: label1Opacity, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#4ade80" }}>
        Channel 1 — own scale, full range used
      </div>
      <RangeBar
        min={-0.3}
        max={0.3}
        segments={[{ from: -0.3, to: 0.3, color: "#4ade80", revealFrame: 10, height: 14 }]}
        ticks={[
          { value: -0.3, label: "-0.3", revealFrame: 10, color: "#4ade80" },
          { value: 0.3, label: "+0.3", revealFrame: 10, color: "#4ade80" },
        ]}
      />
      <div style={{ opacity: label2Opacity, marginTop: 80, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#4ade80" }}>
        Channel 2 — own scale, full range used
      </div>
      <RangeBar
        min={-10}
        max={8}
        segments={[{ from: -10, to: 8, color: "#4ade80", revealFrame: 75, height: 14 }]}
        ticks={[
          { value: -10, label: "-10", revealFrame: 75, color: "#4ade80" },
          { value: 8, label: "+8", revealFrame: 75, color: "#4ade80" },
        ]}
      />
    </AbsoluteFill>
  );
};
```
Narration: "But give each channel its own scale, and now both of them get to use their
full range."

## Scene 51 — Chapter 6 (A-Roll, 570–780f / 7s)

```tsx
// scenes/Ch6Scene51.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch6Scene51: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="NVIDIA docs: weights = per-channel, activations = per-tensor" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "NVIDIA's own docs say TensorRT does exactly this automatically for weights,
while activations still just get one scale each, from the calibrator."

## Scene 52 — Chapter 6 (A-Roll, 780–990f / 7s)

```tsx
// scenes/Ch6Scene52.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch6Scene52: React.FC = () => {
  const frame = useCurrentFrame();
  const scale = interpolate(frame, [25, 45], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const opacity = interpolate(frame, [25, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", paddingBottom: 40 }}>
        <div
          style={{
            scale,
            opacity,
            rotate: "-4deg",
            fontFamily: "Inter, sans-serif",
            fontSize: 30,
            fontWeight: 700,
            color: "#ef4444",
            border: "3px solid #ef4444",
            borderRadius: 10,
            padding: "16px 32px",
            letterSpacing: 1,
            textTransform: "uppercase",
            background: "rgba(17,24,39,0.75)",
          }}
        >
          Documented — not yet verified in this engine
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Now, I haven't checked that inside this specific engine yet. [PAUSE] So
instead of just repeating what the docs say, let's go look for ourselves."

## Scene 53 — Chapter 6 (B-Roll, 990–1170f / 6s)

```tsx
// scenes/Ch6Scene53.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LINES = [
  "# profile_layers.py",
  "engine = load_engine(engine_path)",
  "context = engine.create_execution_context()",
  "inspector = engine.create_engine_inspector()",
  "print(inspector.get_engine_information(...))",
];

export const Ch6Scene53: React.FC = () => {
  const frame = useCurrentFrame();
  const highlightOpacity = interpolate(frame, [60, 85], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelOpacity = interpolate(frame, [100, 120], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace", fontSize: 26 }}>
      {LINES.map((line, i) => {
        const isTarget = i === 3;
        const lineOpacity = interpolate(frame, [10 + i * 10, 25 + i * 10], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            key={line}
            style={{
              opacity: lineOpacity,
              color: i === 0 ? "#6b7280" : "#e5e7eb",
              background: isTarget ? `rgba(96,165,250,${highlightOpacity * 0.18})` : "transparent",
              borderLeft: isTarget ? `3px solid rgba(96,165,250,${highlightOpacity})` : "3px solid transparent",
              padding: "6px 12px",
            }}
          >
            {line}
          </div>
        );
      })}
      <div style={{ opacity: labelOpacity, marginTop: 30, color: "#60a5fa", fontFamily: "Inter, sans-serif", fontSize: 22 }}>
        the engine inspector
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "TensorRT ships a tool for exactly this kind of question. It's called the
engine inspector."

## Scene 54 — Chapter 6 (A-Roll, 1170–1320f / 5s)

```tsx
// scenes/Ch6Scene54.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch6Scene54: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Answered in Chapter 10" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "We'll answer this properly once the engine's built. A few chapters from now.
Promise I won't forget."

## Scene 55 — Chapter 6 (B-Roll, 1320–1500f / 6s)

```tsx
// scenes/Ch6Scene55.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CELLS = [
  { row: "per-tensor", col: "symmetric" },
  { row: "per-tensor", col: "asymmetric" },
  { row: "per-channel", col: "symmetric" },
  { row: "per-channel", col: "asymmetric" },
];

export const Ch6Scene55: React.FC = () => {
  const frame = useCurrentFrame();
  const headerOpacity = interpolate(frame, [5, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: headerOpacity, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af", marginBottom: 20 }}>
        two separate choices
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 280px)", gap: 20 }}>
        {CELLS.map((cell, i) => {
          const start = 35 + i * 22;
          const scale = interpolate(frame, [start, start + 16], [0.8, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          const opacity = interpolate(frame, [start, start + 16], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={`${cell.row}-${cell.col}`}
              style={{
                scale,
                opacity,
                background: "#111827",
                border: "1px solid #374151",
                borderRadius: 10,
                padding: "24px 16px",
                textAlign: "center",
                fontFamily: "Inter, sans-serif",
              }}
            >
              <div style={{ color: "#93c5fd", fontSize: 20 }}>{cell.col}</div>
              <div style={{ color: "#4ade80", fontSize: 20, marginTop: 6 }}>{cell.row}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So really there's two separate choices here: where the scale's centered,
and how many scales you're using."

## Scene 56 — Chapter 6 (A-Roll, 1500–1650f / 5s)

```tsx
// scenes/Ch6Scene56.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch6Scene56: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "None of this tells us where these numbers come from for activations,
though. That's calibration."

## Scene 57 — Chapter 6 (B-Roll, chapter card, 1650–1770f / 4s, no VO)

```tsx
// scenes/Ch6Scene57.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch6Scene57: React.FC = () => (
  <ChapterCard number={7} title="Where the scale actually comes from — calibration." durationInFrames={120} />
);
```

## Chapter 6 assembly

```tsx
// chapters/Chapter6.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch6Scene48 } from "../scenes/Ch6Scene48";
import { Ch6Scene49 } from "../scenes/Ch6Scene49";
import { Ch6Scene50 } from "../scenes/Ch6Scene50";
import { Ch6Scene51 } from "../scenes/Ch6Scene51";
import { Ch6Scene52 } from "../scenes/Ch6Scene52";
import { Ch6Scene53 } from "../scenes/Ch6Scene53";
import { Ch6Scene54 } from "../scenes/Ch6Scene54";
import { Ch6Scene55 } from "../scenes/Ch6Scene55";
import { Ch6Scene56 } from "../scenes/Ch6Scene56";
import { Ch6Scene57 } from "../scenes/Ch6Scene57";

// Local cut points (frames, 30fps) for this chapter's own 0..1770 timeline.
const CUTS = [0, 210, 390, 570, 780, 990, 1170, 1320, 1500, 1650, 1770];
const SCENES = [
  Ch6Scene48,
  Ch6Scene49,
  Ch6Scene50,
  Ch6Scene51,
  Ch6Scene52,
  Ch6Scene53,
  Ch6Scene54,
  Ch6Scene55,
  Ch6Scene56,
  Ch6Scene57,
];

// Global offsets given for Chapter 6: 8790f–10560f.
const CHAPTER_START_GLOBAL = 8790;
const CHAPTER_END_GLOBAL = 10560;

export const Chapter6: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_END_GLOBAL}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 7 — Where The Scale Actually Comes From (Calibration)

Scenes 58–69 (12 scenes, 75s / 2250f at 30fps). Continues after Chapter 6 at global
offset 10560f (352s) and pulls that same slice of `audio/narration_normal_speed.wav`.

## Scene 58 — Chapter 7 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch7Scene58.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch7Scene58: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Weights, you can just go inspect ahead of time. Activations are trickier.
They depend entirely on whatever input you feed the network."

## Scene 59 — Chapter 7 (B-Roll, 180–390f / 7s)

```tsx
// scenes/Ch7Scene59.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const STEPS = [
  { label: "calibration images", color: "#93c5fd" },
  { label: "run network", color: "#93c5fd" },
  { label: "observe ranges", color: "#93c5fd" },
  { label: "scales", color: "#4ade80" },
  { label: "INT8 engine", color: "#4ade80" },
];

export const Ch7Scene59: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", gap: 20 }}>
        {STEPS.map((step, i) => {
          const start = 15 + i * 20;
          const scale = interpolate(frame, [start, start + 12], [0.6, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          const opacity = interpolate(frame, [start, start + 12], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div key={step.label} style={{ display: "flex", alignItems: "center", gap: 20 }}>
              <div
                style={{
                  scale,
                  opacity,
                  fontFamily: "Inter, sans-serif",
                  fontSize: 24,
                  fontWeight: 600,
                  color: step.color,
                  background: "#111827",
                  border: `1px solid ${step.color}`,
                  borderRadius: 999,
                  padding: "14px 24px",
                  whiteSpace: "nowrap",
                }}
              >
                {step.label}
              </div>
              {i < STEPS.length - 1 && <span style={{ opacity, color: "#6b7280", fontSize: 26 }}>→</span>}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So the idea is, you run real data through the network once, just to watch
what ranges show up."

## Scene 60 — Chapter 7 (B-Roll, 390–600f / 7s)

```tsx
// scenes/Ch7Scene60.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const COMMAND = "ls calibration/images";
const FILES = [
  "n01440764_ILSVRC2012_val_00028158.JPEG",
  "n02102040_ILSVRC2012_val_00002315.JPEG",
  "n02979186_ILSVRC2012_val_00023674.JPEG",
  "n03000684_ILSVRC2012_val_00029918.JPEG",
  "n03028079_ILSVRC2012_val_00016854.JPEG",
  "n03394916_ILSVRC2012_val_00005554.JPEG",
  "n03417042_ILSVRC2012_val_00039069.JPEG",
  "n03425413_ILSVRC2012_val_00018006.JPEG",
];

export const Ch7Scene60: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 20], [0, COMMAND.length], { extrapolateRight: "clamp" }));
  const statOpacity = interpolate(frame, [170, 190], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 32 }}>
        $ {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ marginTop: 20 }}>
        {FILES.map((file, i) => {
          const start = 25 + i * 12;
          const opacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          return (
            <div key={file} style={{ opacity, color: "#9ca3af", fontSize: 22, padding: "2px 0" }}>
              {file}
            </div>
          );
        })}
      </div>
      <div
        style={{
          opacity: statOpacity,
          marginTop: 28,
          fontFamily: "Inter, sans-serif",
          fontSize: 26,
          color: "#f9fafb",
          fontWeight: 600,
        }}
      >
        500 images · 50 per class · 10 classes
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "In this repo it's five hundred real JPEGs, fifty per class across ten
classes. Not synthetic noise."

## Scene 61 — Chapter 7 (A-Roll, 600–810f / 7s)

```tsx
// scenes/Ch7Scene61.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch7Scene61: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Just use min/max?" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "Simplest thing you'd try: just track the min and max you see and scale to
that. Turns out TensorRT doesn't do that by default."

## Scene 62 — Chapter 7 (B-Roll, 810–1050f / 8s)

```tsx
// scenes/Ch7Scene62.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const VALUES = [0.03, 0.08, 0.12, 0.15, 0.19, 0.22, 0.27, 0.31, 0.35, 0.4, 0.44, 0.49, 0.53, 0.58];
const OUTLIER = 100.0;
const AXIS_WIDTH = 1400;

export const Ch7Scene62: React.FC = () => {
  const frame = useCurrentFrame();
  const domainMax = interpolate(frame, [70, 140], [0.6, OUTLIER], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const outlierOpacity = interpolate(frame, [70, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const sliverOpacity = interpolate(frame, [160, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const sliverWidth = interpolate(domainMax, [0.6, 100], [AXIS_WIDTH, AXIS_WIDTH * (0.6 / 100)], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: labelOpacity, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af", marginBottom: 40 }}>
        activation values seen during calibration
      </div>
      <div style={{ position: "relative", width: AXIS_WIDTH, height: 4, background: "#374151" }}>
        {VALUES.map((v, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: (v / domainMax) * AXIS_WIDTH - 5,
              top: -8,
              width: 10,
              height: 10,
              borderRadius: 999,
              background: "#60a5fa",
            }}
          />
        ))}
        <div
          style={{
            position: "absolute",
            left: (OUTLIER / domainMax) * AXIS_WIDTH - 7,
            top: -10,
            width: 14,
            height: 14,
            borderRadius: 999,
            background: "#ef4444",
            opacity: outlierOpacity,
            boxShadow: "0 0 14px rgba(239,68,68,0.7)",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: (OUTLIER / domainMax) * AXIS_WIDTH - 30,
            top: 16,
            opacity: outlierOpacity,
            fontFamily: "monospace",
            fontSize: 20,
            color: "#ef4444",
          }}
        >
          100.0
        </div>
      </div>
      <div style={{ position: "relative", width: AXIS_WIDTH, marginTop: 56 }}>
        <div
          style={{
            opacity: sliverOpacity,
            width: sliverWidth,
            height: 18,
            background: "#f97316",
            borderRadius: 4,
          }}
        />
        <div
          style={{
            opacity: sliverOpacity,
            marginTop: 10,
            fontFamily: "Inter, sans-serif",
            fontSize: 20,
            color: "#f97316",
          }}
        >
          every normal value crammed into this sliver of INT8's resolution
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "One weird outlier value blows the whole range out, and suddenly every
normal value gets crammed into a tiny sliver of INT8's resolution."

## Scene 63 — Chapter 7 (B-Roll, 1050–1260f / 7s)

```tsx
// scenes/Ch7Scene63.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const ORIGINAL = [10, 22, 38, 55, 70, 62, 48, 30, 18, 8];
const QUANTIZED = [12, 24, 40, 58, 74, 66, 50, 20, 6, 2];
const BAR_W = 46;
const GAP = 18;

export const Ch7Scene63: React.FC = () => {
  const frame = useCurrentFrame();
  const barsIn = interpolate(frame, [0, 50], [0, ORIGINAL.length], { extrapolateRight: "clamp" });
  const labelOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const badgeScale = interpolate(frame, [110, 140], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "flex-end", gap: GAP, height: 220 }}>
        {ORIGINAL.map((h, i) => {
          const shown = i < barsIn;
          return (
            <div key={i} style={{ position: "relative", width: BAR_W, height: 220 }}>
              <div
                style={{
                  position: "absolute",
                  bottom: 0,
                  width: BAR_W,
                  height: shown ? h * 2.4 : 0,
                  background: "rgba(96,165,250,0.55)",
                  borderRadius: 4,
                }}
              />
              <div
                style={{
                  position: "absolute",
                  bottom: 0,
                  width: BAR_W,
                  height: shown ? QUANTIZED[i] * 2.4 : 0,
                  background: "rgba(74,222,128,0.55)",
                  borderRadius: 4,
                  mixBlendMode: "screen",
                }}
              />
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 32, marginTop: 20, fontFamily: "Inter, sans-serif", fontSize: 20 }}>
        <span style={{ color: "#93c5fd" }}>● original (float)</span>
        <span style={{ color: "#4ade80" }}>● quantized</span>
      </div>
      <div
        style={{
          opacity: labelOpacity,
          scale: badgeScale,
          marginTop: 32,
          fontFamily: "monospace",
          fontSize: 26,
          color: "#f9fafb",
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 999,
          padding: "10px 26px",
        }}
      >
        KL divergence — minimized
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So instead, entropy calibration picks a threshold that keeps the
quantized distribution as close as possible to the original, even if that means
deliberately clipping the extreme outlier."

## Scene 64 — Chapter 7 (B-Roll, 1260–1410f / 5s)

```tsx
// scenes/Ch7Scene64.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch7Scene64: React.FC = () => {
  const frame = useCurrentFrame();
  const cardOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: cardOpacity,
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "36px 56px",
          fontFamily: "Inter, sans-serif",
          textAlign: "center",
          maxWidth: 900,
        }}
      >
        <div style={{ color: "#f9fafb", fontSize: 30, fontWeight: 600 }}>Szymon Migacz, NVIDIA</div>
        <div style={{ color: "#9ca3af", fontSize: 24, marginTop: 12, fontStyle: "italic" }}>
          "8-bit Inference with TensorRT"
        </div>
        <div style={{ color: "#6b7280", fontSize: 22, marginTop: 8 }}>2017</div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "This isn't some new trick, either. It's a method NVIDIA published back
in 2017."

## Scene 65 — Chapter 7 (B-Roll, 1410–1620f / 7s)

```tsx
// scenes/Ch7Scene65.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LINES = [
  { text: "class ImageCalibrator(trt.IInt8EntropyCalibrator2):", highlight: true },
  { text: "    def get_batch_size(self):" },
  { text: "        return self.batch_size" },
  { text: "" },
  { text: "    def get_batch(self, names):", highlight: true },
  { text: "        batch = np.stack([preprocess_image(f) for f in batch_files])" },
];

export const Ch7Scene65: React.FC = () => {
  const frame = useCurrentFrame();
  const cardOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const highlightOpacity = interpolate(frame, [40, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: cardOpacity,
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "28px 36px",
          fontFamily: "monospace",
          fontSize: 24,
        }}
      >
        {LINES.map((line, i) => (
          <div
            key={i}
            style={{
              padding: "5px 12px",
              borderRadius: 6,
              background: line.highlight ? `rgba(96,165,250,${highlightOpacity * 0.18})` : "transparent",
              boxShadow: line.highlight ? `inset 0 0 0 1px rgba(96,165,250,${highlightOpacity * 0.6})` : "none",
              color: line.highlight ? "#93c5fd" : "#e5e7eb",
              whiteSpace: "pre",
            }}
          >
            {line.text || " "}
          </div>
        ))}
      </div>
      <div style={{ marginTop: 24, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#9ca3af" }}>
        every batch: eight real, preprocessed images
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Here's the actual class from the repo. Every batch it hands over to
TensorRT is eight real, preprocessed images."

## Scene 66 — Chapter 7 (B-Roll, 1620–1800f / 6s)

```tsx
// scenes/Ch7Scene66.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const STEPS = ["resize", "center-crop", "normalize", "CHW"];

export const Ch7Scene66: React.FC = () => {
  const frame = useCurrentFrame();
  const codeOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: codeOpacity,
          fontFamily: "monospace",
          fontSize: 28,
          color: "#93c5fd",
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "18px 32px",
        }}
      >
        def preprocess_image(path, size=224):
      </div>
      <div style={{ display: "flex", gap: 28, marginTop: 44 }}>
        {STEPS.map((step, i) => {
          const start = 40 + i * 20;
          const scale = interpolate(frame, [start, start + 12], [0.6, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          const opacity = interpolate(frame, [start, start + 12], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div key={step} style={{ display: "flex", alignItems: "center", gap: 28 }}>
              <div
                style={{
                  scale,
                  opacity,
                  fontFamily: "Inter, sans-serif",
                  fontSize: 24,
                  fontWeight: 600,
                  color: "#4ade80",
                  background: "rgba(17,24,39,0.75)",
                  border: "1px solid #374151",
                  borderRadius: 999,
                  padding: "12px 26px",
                }}
              >
                {step}
              </div>
              {i < STEPS.length - 1 && <span style={{ opacity, color: "#6b7280", fontSize: 24 }}>→</span>}
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 28, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#9ca3af" }}>
        same preprocessing the model expects at inference time
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Resize, center-crop, normalize with ImageNet stats. Literally the same
preprocessing the model expects at inference time."

## Scene 67 — Chapter 7 (B-Roll, 1800–1980f / 6s)

```tsx
// scenes/Ch7Scene67.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch7Scene67: React.FC = () => {
  const frame = useCurrentFrame();
  const codeOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const arrowOpacity = interpolate(frame, [60, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fileScale = interpolate(frame, [80, 100], [0.6, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const fileOpacity = interpolate(frame, [80, 100], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 40 }}>
        <div
          style={{
            opacity: codeOpacity,
            fontFamily: "monospace",
            fontSize: 22,
            color: "#93c5fd",
            background: "#111827",
            border: "1px solid #374151",
            borderRadius: 12,
            padding: "18px 28px",
            whiteSpace: "pre",
          }}
        >
          {"def write_calibration_cache(self, cache):\n    with open(self.cache_file, \"wb\") as f:"}
        </div>
        <span style={{ opacity: arrowOpacity, color: "#6b7280", fontSize: 40 }}>→</span>
        <div style={{ scale: fileScale, opacity: fileOpacity, textAlign: "center" }}>
          <div
            style={{
              width: 64,
              height: 80,
              margin: "0 auto",
              background: "#111827",
              border: "1px solid #4ade80",
              borderRadius: 4,
              position: "relative",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                right: 0,
                width: 0,
                height: 0,
                borderStyle: "solid",
                borderWidth: "0 16px 16px 0",
                borderColor: "transparent #0b0f14 transparent transparent",
              }}
            />
          </div>
          <div style={{ marginTop: 10, fontFamily: "monospace", fontSize: 20, color: "#4ade80" }}>
            calibration.cache
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "And once calibration's done, it just writes everything it learned out
to one file."

## Scene 68 — Chapter 7 (A-Roll, 1980–2130f / 5s)

```tsx
// scenes/Ch7Scene68.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch7Scene68: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So, 'calibration produces a scale per tensor.' Is that just a concept
I'm describing, or can I show you it?"

## Scene 69 — Chapter 7 (B-Roll, chapter card, 2130–2250f / 4s, no VO)

```tsx
// scenes/Ch7Scene69.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch7Scene69: React.FC = () => (
  <ChapterCard number={8} title="Opening the calibration file." durationInFrames={120} />
);
```

## Chapter 7 assembly

```tsx
// chapters/Chapter7.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch7Scene58 } from "../scenes/Ch7Scene58";
import { Ch7Scene59 } from "../scenes/Ch7Scene59";
import { Ch7Scene60 } from "../scenes/Ch7Scene60";
import { Ch7Scene61 } from "../scenes/Ch7Scene61";
import { Ch7Scene62 } from "../scenes/Ch7Scene62";
import { Ch7Scene63 } from "../scenes/Ch7Scene63";
import { Ch7Scene64 } from "../scenes/Ch7Scene64";
import { Ch7Scene65 } from "../scenes/Ch7Scene65";
import { Ch7Scene66 } from "../scenes/Ch7Scene66";
import { Ch7Scene67 } from "../scenes/Ch7Scene67";
import { Ch7Scene68 } from "../scenes/Ch7Scene68";
import { Ch7Scene69 } from "../scenes/Ch7Scene69";

// Local cut points (frames, 30fps) for this chapter's own 0..2250 timeline.
const CUTS = [0, 180, 390, 600, 810, 1050, 1260, 1410, 1620, 1800, 1980, 2130, 2250];
const SCENES = [
  Ch7Scene58,
  Ch7Scene59,
  Ch7Scene60,
  Ch7Scene61,
  Ch7Scene62,
  Ch7Scene63,
  Ch7Scene64,
  Ch7Scene65,
  Ch7Scene66,
  Ch7Scene67,
  Ch7Scene68,
  Ch7Scene69,
];

// Global offset given by the Animator contract for Chapter 7.
const CHAPTER_START_GLOBAL = 10560;

export const Chapter7: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 2250}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 8 — Opening The Calibration File

Scenes 70–79 (10 scenes, 63s / 1890f at 30fps). Continues immediately after Chapter 7
in the master timeline (global offset 12810f / 427s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 70 — Chapter 8 (B-Roll, 0–180f / 6s)

```tsx
// scenes/Ch8Scene70.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const COMMAND = "cat calibration/calibration.cache | head -5";
const LINES = [
  "TRT-101401-EntropyCalibration2",
  "input: 3cd56a3e",
  "fc.bias_output: 3a2e3bf0",
  "ONNXTRT_Broadcast_output: 3a2e3bf0",
  "node_linear_output: 3d6ca735",
];

export const Ch8Scene70: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 35], [0, COMMAND.length], { extrapolateRight: "clamp" }));
  const noteOpacity = interpolate(frame, [140, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 30 }}>
        $ {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ marginTop: 24 }}>
        {LINES.map((line, i) => {
          const start = 40 + i * 12;
          const opacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          return (
            <div key={line} style={{ opacity, color: "#e5e7eb", fontSize: 26, padding: "3px 0" }}>
              {line}
            </div>
          );
        })}
      </div>
      <div style={{ opacity: noteOpacity, marginTop: 24, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>
        plain text — not a binary blob
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Let's just open it up. And it's not some binary blob. It's literally a
plain text file."

## Scene 71 — Chapter 8 (B-Roll, 180–330f / 5s)

```tsx
// scenes/Ch8Scene71.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const COMMAND = "wc -l calibration/calibration.cache";

export const Ch8Scene71: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 30], [0, COMMAND.length], { extrapolateRight: "clamp" }));
  const resultScale = interpolate(frame, [55, 75], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const resultOpacity = interpolate(frame, [55, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const labelOpacity = interpolate(frame, [90, 105], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center", fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 30 }}>
        $ {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ scale: resultScale, opacity: resultOpacity, marginTop: 30, fontSize: 96, fontWeight: 700, color: "#f9fafb" }}>
        128
      </div>
      <div style={{ opacity: labelOpacity, marginTop: 14, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>
        one line per tensor, across the whole graph
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "One line per tensor, across this whole fifty-eight-layer graph."

## Scene 72 — Chapter 8 (B-Roll, 330–510f / 6s)

```tsx
// scenes/Ch8Scene72.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LINES = [
  "TRT-101401-EntropyCalibration2",
  "input: 3cd56a3e",
  "fc.bias_output: 3a2e3bf0",
  "ONNXTRT_Broadcast_output: 3a2e3bf0",
  "node_linear_output: 3d6ca735",
  "getitem: 3e9fe293",
];

export const Ch8Scene72: React.FC = () => {
  const frame = useCurrentFrame();
  const listOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const highlightOpacity = interpolate(frame, [50, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const highlightScale = interpolate(frame, [50, 70], [1, 1.06], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: listOpacity, fontFamily: "monospace", fontSize: 26 }}>
        {LINES.map((line) => {
          const isTarget = line.startsWith("input:");
          return (
            <div
              key={line}
              style={{
                scale: isTarget ? highlightScale : 1,
                color: isTarget ? "#93c5fd" : "#4b5563",
                background: isTarget ? `rgba(96,165,250,${highlightOpacity * 0.15})` : "transparent",
                boxShadow: isTarget ? `inset 0 0 0 1px rgba(96,165,250,${highlightOpacity * 0.6})` : "none",
                borderRadius: 6,
                padding: "6px 16px",
                margin: "4px 0",
              }}
            >
              {line}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Let's decode one of these. Let's do the input tensor."

## Scene 73 — Chapter 8 (B-Roll, 510–750f / 8s)

```tsx
// scenes/Ch8Scene73.tsx
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

const LINE1 = 'bits = int("3cd56a3e", 16)';
const LINE2 = "struct.unpack('>f', struct.pack('>I', bits))";

export const Ch8Scene73: React.FC = () => {
  const frame = useCurrentFrame();
  const chars1 = Math.floor(
    interpolate(frame, [10, 90], [0, LINE1.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  );
  const chars2 = Math.floor(
    interpolate(frame, [110, 210], [0, LINE2.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  );
  const cursorLine = chars1 < LINE1.length ? 1 : 2;
  const cursorOn = frame % 20 < 10;

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "36px 44px",
          fontFamily: "monospace",
          fontSize: 28,
          minWidth: 900,
        }}
      >
        <div style={{ color: "#e5e7eb" }}>
          {LINE1.slice(0, chars1)}
          {cursorLine === 1 && <span style={{ opacity: cursorOn ? 1 : 0 }}>_</span>}
        </div>
        <div style={{ color: "#4ade80", marginTop: 14 }}>
          {LINE2.slice(0, chars2)}
          {cursorLine === 2 && <span style={{ opacity: cursorOn ? 1 : 0 }}>_</span>}
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Should only take about five lines: read the hex as an integer, then
reinterpret those exact same bits as a float."

## Scene 74 — Chapter 8 (B-Roll, 750–900f / 5s)

```tsx
// scenes/Ch8Scene74.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch8Scene74: React.FC = () => {
  const frame = useCurrentFrame();
  const resultOpacity = interpolate(frame, [10, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const resultScale = interpolate(frame, [10, 30], [0.85, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const labelOpacity = interpolate(frame, [60, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center", fontFamily: "monospace" }}>
      <div style={{ color: "#6b7280", fontSize: 26 }}>$ python3 decode.py</div>
      <div style={{ opacity: resultOpacity, scale: resultScale, marginTop: 20, fontSize: 68, fontWeight: 700, color: "#f9fafb" }}>
        0.0260516...
      </div>
      <div style={{ opacity: labelOpacity, marginTop: 20, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>
        calibrated dynamic range for the input tensor
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Okay, run it. 0.026. So that's apparently the calibrated dynamic range
for the input tensor."

## Scene 75 — Chapter 8 (A-Roll, 900–1140f / 8s)

```tsx
// scenes/Ch8Scene75.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch8Scene75: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Expected ~±2.1 to ±2.6. Got 0.026?" fromFrame={35} />
  </AbsoluteFill>
);
```
Narration: "Wait. [PAUSE] Hang on. This input's ImageNet-normalized, its real range
should be something like plus or minus two and a half. 0.026 is way off from that.
Did I mess up the decode?"

## Scene 76 — Chapter 8 (B-Roll, 1140–1380f / 8s)

```tsx
// scenes/Ch8Scene76.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CANDIDATES = [
  { label: "big-endian read", value: "0.0261", start: 15 },
  { label: "little-endian read", value: "0.2293", start: 40 },
];

export const Ch8Scene76: React.FC = () => {
  const frame = useCurrentFrame();
  const hexOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: hexOpacity, fontFamily: "monospace", fontSize: 26, color: "#6b7280", marginBottom: 40 }}>
        3c d5 6a 3e
      </div>
      <div style={{ display: "flex", gap: 48 }}>
        {CANDIDATES.map((c) => {
          const opacity = interpolate(frame, [c.start, c.start + 15], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const scale = interpolate(frame, [c.start, c.start + 15], [0.9, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          return (
            <div
              key={c.label}
              style={{
                opacity,
                scale,
                background: "#111827",
                border: "1px solid #374151",
                borderRadius: 12,
                padding: "28px 40px",
                textAlign: "center",
                fontFamily: "monospace",
              }}
            >
              <div style={{ fontSize: 40, color: "#f9fafb", fontWeight: 700 }}>{c.value}</div>
              <div style={{ marginTop: 10, fontFamily: "Inter, sans-serif", fontSize: 18, color: "#9ca3af" }}>{c.label}</div>
              <div
                style={{
                  marginTop: 14,
                  fontFamily: "Inter, sans-serif",
                  fontSize: 16,
                  color: "#93c5fd",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                }}
              >
                valid interpretation
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So I tried it the other byte-order way too, and honestly both readings
are defensible. They give two different numbers, and neither one matches what I
expected."

## Scene 77 — Chapter 8 (A-Roll, 1380–1620f / 8s)

```tsx
// scenes/Ch8Scene77.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch8Scene77: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Entropy calibration clips outliers — on purpose." fromFrame={30} />
  </AbsoluteFill>
);
```
Narration: "But this is exactly what entropy calibration is supposed to do. It can
pick a threshold way below the true max if that gives better resolution for most of
the data. [PAUSE] Small doesn't mean broken here."

## Scene 78 — Chapter 8 (A-Roll, 1620–1770f / 5s)

```tsx
// scenes/Ch8Scene78.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch8Scene78: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Honestly, the lesson isn't the exact number. It's that when something
looks surprising, you go check it instead of trusting the first plausible-looking
answer."

## Scene 79 — Chapter 8 (B-Roll, chapter card, 1770–1890f / 4s, no VO)

```tsx
// scenes/Ch8Scene79.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch8Scene79: React.FC = () => (
  <ChapterCard number={9} title="From toy numbers to real activations." durationInFrames={120} />
);
```

## Chapter 8 assembly

```tsx
// chapters/Chapter8.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch8Scene70 } from "../scenes/Ch8Scene70";
import { Ch8Scene71 } from "../scenes/Ch8Scene71";
import { Ch8Scene72 } from "../scenes/Ch8Scene72";
import { Ch8Scene73 } from "../scenes/Ch8Scene73";
import { Ch8Scene74 } from "../scenes/Ch8Scene74";
import { Ch8Scene75 } from "../scenes/Ch8Scene75";
import { Ch8Scene76 } from "../scenes/Ch8Scene76";
import { Ch8Scene77 } from "../scenes/Ch8Scene77";
import { Ch8Scene78 } from "../scenes/Ch8Scene78";
import { Ch8Scene79 } from "../scenes/Ch8Scene79";

// Local cut points (frames, 30fps) for this chapter's own 0..1890 timeline.
const CUTS = [0, 180, 330, 510, 750, 900, 1140, 1380, 1620, 1770, 1890];
const SCENES = [
  Ch8Scene70,
  Ch8Scene71,
  Ch8Scene72,
  Ch8Scene73,
  Ch8Scene74,
  Ch8Scene75,
  Ch8Scene76,
  Ch8Scene77,
  Ch8Scene78,
  Ch8Scene79,
];

// Global offset given by the Animator contract for Chapter 8.
const CHAPTER_START_GLOBAL = 12810;

export const Chapter8: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1890}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 9 — From Toy Numbers To Real Activations

Scenes 80–83 (4 scenes, 25s / 750f at 30fps). Continues immediately after Chapter 8 in
the master timeline (global offset 14700f / 490s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 80 — Chapter 9 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch9Scene80.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

const TOY = [2.1, -0.8, 1.4, 0.3, -1.6];

export const Ch9Scene80: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [20, 35], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <div
        style={{
          position: "absolute",
          top: 90,
          right: 90,
          opacity,
          background: "rgba(17,24,39,0.85)",
          border: "1px solid #374151",
          borderRadius: 10,
          padding: "18px 24px",
          fontFamily: "monospace",
        }}
      >
        <div style={{ color: "#6b7280", fontSize: 18, marginBottom: 8 }}>Ch.4 — toy array</div>
        <div style={{ color: "#e5e7eb", fontSize: 24 }}>[{TOY.join(", ")}]</div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "All that math, does the real model actually produce numbers like this?
Let's find out instead of just assuming."

## Shared primitive: CodeBlock

```tsx
// CodeBlock.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

type CodeBlockProps = {
  title: string;
  lines: string[];
  highlightLines?: number[];
  fromFrame?: number;
};

export const CodeBlock: React.FC<CodeBlockProps> = ({
  title,
  lines,
  highlightLines = [],
  fromFrame = 0,
}) => {
  const frame = useCurrentFrame() - fromFrame;

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ background: "#111827", border: "1px solid #374151", borderRadius: 10, overflow: "hidden", minWidth: 900 }}>
        <div style={{ padding: "10px 20px", borderBottom: "1px solid #374151", fontFamily: "monospace", fontSize: 18, color: "#6b7280" }}>
          {title}
        </div>
        <div style={{ padding: "20px 28px" }}>
          {lines.map((line, i) => {
            const start = 10 + i * 6;
            const opacity = interpolate(frame, [start, start + 12], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            const translateX = interpolate(frame, [start, start + 12], [-10, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const highlighted = highlightLines.includes(i);
            return (
              <div
                key={i}
                style={{
                  opacity,
                  translate: `${translateX}px 0px`,
                  fontFamily: "monospace",
                  fontSize: 24,
                  color: highlighted ? "#f9fafb" : "#9ca3af",
                  background: highlighted ? "rgba(96,165,250,0.12)" : "transparent",
                  borderLeft: highlighted ? "3px solid #60a5fa" : "3px solid transparent",
                  padding: "4px 12px",
                  whiteSpace: "pre",
                }}
              >
                {line}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Reused for every code-highlight scene from Chapter 9 onward (scenes 81, 85, 94).

## Scene 81 — Chapter 9 (B-Roll, 180–420f / 8s)

```tsx
// scenes/Ch9Scene81.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CodeBlock } from "../CodeBlock";

const LINES = [
  "activation = {}",
  "def hook(module, inp, out):",
  '    activation["conv1"] = out.detach()',
  "model.conv1.register_forward_hook(hook)",
  "model(real_image_batch)          # resnet50.pth",
  'raw = activation["conv1"]',
  "q, scale = quantize_symmetric(raw)",
  "print(q.dtype, scale)",
];

export const Ch9Scene81: React.FC = () => {
  const frame = useCurrentFrame();
  const resultOpacity = interpolate(frame, [190, 210], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CodeBlock title="hook_and_quantize.py" lines={LINES} highlightLines={[6]} />
      <div
        style={{
          position: "absolute",
          bottom: 90,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: resultOpacity,
          fontFamily: "monospace",
          fontSize: 26,
          color: "#4ade80",
        }}
      >
        &gt;&gt;&gt; real activations, quantized. int8.
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Grab a real slice of activations straight out of this ResNet50, and run
the exact same function on it. Same eight lines. Real numbers."

## Scene 82 — Chapter 9 (A-Roll, 420–630f / 7s)

```tsx
// scenes/Ch9Scene82.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch9Scene82: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Now — that open question from Chapter 6." fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "No separate, more-real version of the math hiding anywhere. This is it.
Time to go check what TensorRT actually did with per-channel scaling, that question
from a few chapters back."

## Scene 83 — Chapter 9 (B-Roll, chapter card, 630–750f / 4s, no VO)

```tsx
// scenes/Ch9Scene83.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch9Scene83: React.FC = () => (
  <ChapterCard number={10} title="Checking TensorRT's own homework." durationInFrames={120} />
);
```

## Chapter 9 assembly

```tsx
// chapters/Chapter9.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch9Scene80 } from "../scenes/Ch9Scene80";
import { Ch9Scene81 } from "../scenes/Ch9Scene81";
import { Ch9Scene82 } from "../scenes/Ch9Scene82";
import { Ch9Scene83 } from "../scenes/Ch9Scene83";

// Local cut points (frames, 30fps) for this chapter's own 0..750 timeline.
const CUTS = [0, 180, 420, 630, 750];
const SCENES = [Ch9Scene80, Ch9Scene81, Ch9Scene82, Ch9Scene83];

// Global offset: Chapters 0-8 run an estimated 14700f (490s) before this chapter starts.
const CHAPTER_START_GLOBAL = 14700;

export const Chapter9: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 750}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 10 — Checking TensorRT's Own Homework (Engine Inspector Audit)

Scenes 84–93 (10 scenes, 60s / 1800f at 30fps). Continues immediately after Chapter 9
in the master timeline (global offset 15450f / 515s) and pulls that same slice of
`audio/narration_normal_speed.wav`. Tone shifts from QUIET/CALM to HIGH ENERGY starting
at Scene 88 — the confirmation beat — reflected below in tighter, bouncier pop-ins from
that scene on.

## Scene 84 — Chapter 10 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch10Scene84.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch10Scene84: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Weights: per-channel? Activations: per-tensor?" fromFrame={25} />
  </AbsoluteFill>
);
```
Narration: "Here's the claim I want to test: weights get their own scale per channel,
activations get just one scale for the whole tensor."

## Scene 85 — Chapter 10 (B-Roll, 180–390f / 7s)

```tsx
// scenes/Ch10Scene85.tsx
import { CodeBlock } from "../CodeBlock";

const LINES = [
  "inspector = engine.create_engine_inspector()",
  "for i in range(engine.num_layers):",
  "    info = json.loads(",
  "        inspector.get_layer_information(",
  "            i, trt.LayerInformationFormat.JSON",
  "        )",
  "    )",
];

export const Ch10Scene85: React.FC = () => (
  <CodeBlock title="profile_layers.py" lines={LINES} highlightLines={[0, 3, 4]} />
);
```
Narration: "Turns out TensorRT will hand you the full JSON description of any layer it
built, if you ask it nicely."

## Scene 86 — Chapter 10 (B-Roll, 390–600f / 7s)

```tsx
// scenes/Ch10Scene86.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LAYERS = ["Reformat", "CaskConvolution", "PointWise", "Scale", "CaskConvolution", "ElementWise", "Pooling"];
const PICKED = 4;

export const Ch10Scene86: React.FC = () => {
  const frame = useCurrentFrame();
  const boxScale = interpolate(frame, [90, 110], [1, 1.04], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "monospace", fontSize: 26 }}>
        {LAYERS.map((layer, i) => {
          const start = 10 + i * 10;
          const opacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const picked = i === PICKED;
          return (
            <div
              key={i}
              style={{
                opacity,
                scale: picked ? boxScale : 1,
                color: picked ? "#f9fafb" : "#6b7280",
                background: picked ? "rgba(249,115,22,0.15)" : "transparent",
                border: picked ? "1px solid #f97316" : "1px solid transparent",
                borderRadius: 6,
                padding: "8px 24px",
                margin: "3px 0",
              }}
            >
              [{i}] {layer}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Let's pull the full record for one actual INT8 convolution layer."

## Scene 87 — Chapter 10 (B-Roll, 600–840f / 8s)

```tsx
// scenes/Ch10Scene87.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const JSON_LINES = [
  { text: "{", indent: 0 },
  { text: '"Name": "layer4.1.conv2",', indent: 1 },
  { text: '"LayerType": "CaskConvolution",', indent: 1 },
  { text: '"Inputs": [{ "Format/Datatype": "Int8",', indent: 1 },
  { text: '  "scale": s }],', indent: 2, highlight: "activation" },
  { text: '"Weights": { "Type": "Int8",', indent: 1 },
  { text: '  "scale": [ s0, s1, s2, ..., sN ] },', indent: 2, highlight: "weight" },
  { text: "}", indent: 0 },
];

export const Ch10Scene87: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "monospace", fontSize: 26, minWidth: 860 }}>
        {JSON_LINES.map((line, i) => {
          const start = 15 + i * 14;
          const opacity = interpolate(frame, [start, start + 12], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const color =
            line.highlight === "activation" ? "#93c5fd" : line.highlight === "weight" ? "#4ade80" : "#e5e7eb";
          return (
            <div key={i} style={{ opacity, color, paddingLeft: line.indent * 28, fontWeight: line.highlight ? 700 : 400 }}>
              {line.text}
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 60, marginTop: 30, fontFamily: "Inter, sans-serif", fontSize: 22 }}>
        <div style={{ opacity: interpolate(frame, [130, 150], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }), color: "#4ade80" }}>
          weight scale — array, one per channel
        </div>
        <div style={{ opacity: interpolate(frame, [150, 170], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }), color: "#93c5fd" }}>
          activation scale — single number
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Look at the shapes here. The weight scale is an array, one number per
output channel. The activation scale is just a single number."

## Scene 88 — Chapter 10 (A-Roll, 840–1020f / 6s) [HIGH ENERGY]

```tsx
// scenes/Ch10Scene88.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch10Scene88: React.FC = () => {
  const frame = useCurrentFrame();
  const oldOpacity = interpolate(frame, [0, 10, 20, 30], [1, 1, 0.3, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const stampScale = interpolate(frame, [22, 34], [0.5, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const stampOpacity = interpolate(frame, [22, 32], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const stampRotate = interpolate(frame, [22, 34], [-10, -4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div style={{ position: "relative", width: 620, height: 70 }}>
          <div
            style={{
              position: "absolute",
              opacity: oldOpacity,
              color: "#9ca3af",
              fontFamily: "Inter, sans-serif",
              fontSize: 26,
              letterSpacing: 1,
              textDecoration: "line-through",
            }}
          >
            DOCUMENTED — NOT YET VERIFIED
          </div>
          <div
            style={{
              position: "absolute",
              opacity: stampOpacity,
              scale: stampScale,
              rotate: `${stampRotate}deg`,
              color: "#4ade80",
              fontFamily: "Inter, sans-serif",
              fontWeight: 800,
              fontSize: 34,
              letterSpacing: 1,
              border: "3px solid #4ade80",
              borderRadius: 8,
              padding: "8px 20px",
            }}
          >
            CONFIRMED — this engine.
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "So here's the confirmation. This specific engine really is mixing
per-channel weights with per-tensor activations, exactly like the docs describe."

## Scene 89 — Chapter 10 (A-Roll, 1020–1200f / 6s)

```tsx
// scenes/Ch10Scene89.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch10Scene89: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [15, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <div
        style={{
          position: "absolute",
          left: 80,
          bottom: 90,
          opacity,
          maxWidth: 760,
          fontFamily: "Inter, sans-serif",
          fontSize: 24,
          color: "#9ca3af",
          background: "rgba(17,24,39,0.6)",
          padding: "12px 20px",
          borderRadius: 6,
        }}
      >
        If your own engine inspector shows something different — trust that, not this
        video.
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "If you run this yourself on a different model or TensorRT version and get
something different, that's kind of the whole point of checking it yourself. Trust
your own inspector output over mine."

## Scene 90 — Chapter 10 (B-Roll, 1200–1380f / 6s)

```tsx
// scenes/Ch10Scene90.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CELLS = [
  { row: "Weights", col: "Per-Channel", confirmed: true, file: "calibrate_int8.py" },
  { row: "Weights", col: "Per-Tensor", confirmed: false, file: null },
  { row: "Activations", col: "Per-Channel", confirmed: false, file: null },
  { row: "Activations", col: "Per-Tensor", confirmed: true, file: "profile_layers.py" },
];

export const Ch10Scene90: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "grid", gridTemplateColumns: "220px 220px", gridTemplateRows: "160px 160px", gap: 4 }}>
        {CELLS.map((cell, i) => {
          const start = 10 + i * 15;
          const opacity = interpolate(frame, [start, start + 12], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const checkScale = interpolate(frame, [start + 20, start + 32], [0.5, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          return (
            <div
              key={i}
              style={{
                opacity,
                background: "#111827",
                border: cell.confirmed ? "1px solid #4ade80" : "1px solid #374151",
                borderRadius: 8,
                padding: 16,
                fontFamily: "Inter, sans-serif",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <div style={{ color: "#9ca3af", fontSize: 18 }}>
                {cell.row} × {cell.col}
              </div>
              {cell.confirmed && (
                <div style={{ scale: checkScale, color: "#4ade80", fontFamily: "monospace", fontSize: 20 }}>
                  ✓ {cell.file}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So at this point, every concept from the first half of this video is tied
to a real file sitting in this repo."

## Scene 91 — Chapter 10 (A-Roll, 1380–1530f / 5s)

```tsx
// scenes/Ch10Scene91.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch10Scene91: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Which brings up the real question: how much does any of this help?"

## Scene 92 — Chapter 10 (B-Roll, 1530–1680f / 5s)

```tsx
// scenes/Ch10Scene92.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const FILES = ["calibrate_int8.py", "export_onnx_to_trt.py", "profile_layers.py", "build_and_bench.py", "measure_full_metrics.py"];
const COMMAND = "$ ls scripts/";

export const Ch10Scene92: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 20], [0, COMMAND.length], { extrapolateRight: "clamp" }));

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#22c55e", fontSize: 30 }}>
        {COMMAND.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ marginTop: 24 }}>
        {FILES.map((file, i) => {
          const start = 25 + i * 10;
          const picked = file === "build_and_bench.py";
          const opacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          return (
            <div
              key={file}
              style={{
                opacity,
                fontSize: 26,
                color: picked ? "#f9fafb" : "#6b7280",
                background: picked ? "rgba(96,165,250,0.15)" : "transparent",
                borderLeft: picked ? "3px solid #60a5fa" : "3px solid transparent",
                padding: "5px 14px",
              }}
            >
              {file}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Time to go run the benchmark."

## Scene 93 — Chapter 10 (B-Roll, chapter card, 1680–1800f / 4s, no VO)

```tsx
// scenes/Ch10Scene93.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch10Scene93: React.FC = () => (
  <ChapterCard number={11} title="The benchmark." durationInFrames={120} />
);
```

## Chapter 10 assembly

```tsx
// chapters/Chapter10.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch10Scene84 } from "../scenes/Ch10Scene84";
import { Ch10Scene85 } from "../scenes/Ch10Scene85";
import { Ch10Scene86 } from "../scenes/Ch10Scene86";
import { Ch10Scene87 } from "../scenes/Ch10Scene87";
import { Ch10Scene88 } from "../scenes/Ch10Scene88";
import { Ch10Scene89 } from "../scenes/Ch10Scene89";
import { Ch10Scene90 } from "../scenes/Ch10Scene90";
import { Ch10Scene91 } from "../scenes/Ch10Scene91";
import { Ch10Scene92 } from "../scenes/Ch10Scene92";
import { Ch10Scene93 } from "../scenes/Ch10Scene93";

// Local cut points (frames, 30fps) for this chapter's own 0..1800 timeline.
const CUTS = [0, 180, 390, 600, 840, 1020, 1200, 1380, 1530, 1680, 1800];
const SCENES = [
  Ch10Scene84,
  Ch10Scene85,
  Ch10Scene86,
  Ch10Scene87,
  Ch10Scene88,
  Ch10Scene89,
  Ch10Scene90,
  Ch10Scene91,
  Ch10Scene92,
  Ch10Scene93,
];

// Global offset: Chapters 0-9 run an estimated 15450f (515s) before this chapter starts.
const CHAPTER_START_GLOBAL = 15450;

export const Chapter10: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1800}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 11 — The Benchmark (build_and_bench.py)

Scenes 94–105 (12 scenes, 69s / 2070f at 30fps). Continues immediately after Chapter 10
in the master timeline (global offset 17250f / 575s) and pulls that same slice of
`audio/narration_normal_speed.wav`. Marked HIGH ENERGY throughout — pop-ins run tighter
and bouncier than Chapters 0–9 while staying inside the same easing vocabulary.

## Scene 94 — Chapter 11 (B-Roll, 0–180f / 6s)

```tsx
// scenes/Ch11Scene94.tsx
import { CodeBlock } from "../CodeBlock";

const LINES = [
  "def benchmark(engine_path, batch=1,",
  "              warmup=20, iters=200):",
  "    ...",
  "    for _ in range(warmup):",
  "        context.execute_async_v3(stream)",
  "    cudart.cudaEventRecord(start, stream)",
  "    for _ in range(iters):",
  "        context.execute_async_v3(stream)",
  "    cudart.cudaEventRecord(end, stream)",
];

export const Ch11Scene94: React.FC = () => (
  <CodeBlock title="build_and_bench.py" lines={LINES} highlightLines={[1, 5, 8]} />
);
```
Narration: "Quick note on how this is measured: twenty warmup runs, two hundred timed
ones, using CUDA events. Not just a Python stopwatch."

## Shared primitive: TerminalBlock

```tsx
// TerminalBlock.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

type OutputLine = { text: string; color?: string };
type TerminalBlockProps = {
  command: string;
  outputLines: OutputLine[];
  fromFrame?: number;
};

export const TerminalBlock: React.FC<TerminalBlockProps> = ({ command, outputLines, fromFrame = 0 }) => {
  const frame = useCurrentFrame() - fromFrame;
  const typedChars = Math.floor(
    interpolate(frame, [0, 20], [0, command.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  );
  const typingDone = typedChars >= command.length;

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "monospace", fontSize: 30, minWidth: 1000 }}>
        <div style={{ color: "#22c55e" }}>
          $ {command.slice(0, typedChars)}
          {!typingDone && <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>}
        </div>
        <div style={{ marginTop: 20 }}>
          {outputLines.map((line, i) => {
            const start = 20 + i * 14;
            const opacity = interpolate(frame, [start, start + 12], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            return (
              <div key={i} style={{ opacity, color: line.color ?? "#e5e7eb", padding: "4px 0" }}>
                {line.text}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Reused for the three build/benchmark terminal scenes (95, 96, 97).

## Scene 95 — Chapter 11 (B-Roll, 180–360f / 6s)

```tsx
// scenes/Ch11Scene95.tsx
import { TerminalBlock } from "../TerminalBlock";

export const Ch11Scene95: React.FC = () => (
  <TerminalBlock
    command="python build_and_bench.py --precision fp32"
    outputLines={[
      { text: "Building fp32 engine...", color: "#9ca3af" },
      { text: "fp32: {'throughput_qps': 553.24, 'gpu_latency_mean_ms': 1.808}", color: "#f9fafb" },
    ]}
  />
);
```
Narration: "Alright, FP32 first. One point eight milliseconds."

## Scene 96 — Chapter 11 (B-Roll, 360–540f / 6s)

```tsx
// scenes/Ch11Scene96.tsx
import { TerminalBlock } from "../TerminalBlock";

export const Ch11Scene96: React.FC = () => (
  <TerminalBlock
    command="python build_and_bench.py --precision fp16"
    outputLines={[
      { text: "Building fp16 engine...", color: "#9ca3af" },
      { text: "fp16: {'throughput_qps': 1551.03, 'gpu_latency_mean_ms': 0.645}", color: "#4ade80" },
    ]}
  />
);
```
Narration: "FP16. About zero point six five milliseconds. Almost three times faster
already."

## Scene 97 — Chapter 11 (B-Roll, 540–720f / 6s)

```tsx
// scenes/Ch11Scene97.tsx
import { TerminalBlock } from "../TerminalBlock";

export const Ch11Scene97: React.FC = () => (
  <TerminalBlock
    command="python build_and_bench.py --precision int8"
    outputLines={[
      { text: "Building int8 engine...", color: "#9ca3af" },
      { text: "int8: {'throughput_qps': 2880.55, 'gpu_latency_mean_ms': 0.347}", color: "#f97316" },
    ]}
  />
);
```
Narration: "And INT8. [PAUSE] About zero point three five."

## Shared primitive: BarCompare

```tsx
// BarCompare.tsx
import { useCurrentFrame, interpolate, Easing } from "remotion";

type Bar = { label: string; value: number; display: string; color: string };
type BarCompareProps = {
  title?: string;
  bars: Bar[];
  maxValue: number;
  fromFrame?: number;
  maxBarWidth?: number;
};

export const BarCompare: React.FC<BarCompareProps> = ({ title, bars, maxValue, fromFrame = 0, maxBarWidth = 700 }) => {
  const frame = useCurrentFrame() - fromFrame;

  return (
    <div style={{ fontFamily: "Inter, sans-serif" }}>
      {title && <div style={{ color: "#9ca3af", fontSize: 24, marginBottom: 20, textAlign: "center" }}>{title}</div>}
      {bars.map((bar, i) => {
        const start = 8 + i * 12;
        const width = interpolate(frame, [start, start + 22], [0, (bar.value / maxValue) * maxBarWidth], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.16, 1, 0.3, 1),
        });
        const labelOpacity = interpolate(frame, [start, start + 10], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div key={bar.label} style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 16 }}>
            <div style={{ opacity: labelOpacity, width: 90, color: "#93c5fd", fontSize: 22, textAlign: "right" }}>
              {bar.label}
            </div>
            <div style={{ width, height: 34, background: bar.color, borderRadius: 4 }} />
            <div style={{ opacity: labelOpacity, color: "#f9fafb", fontSize: 22, fontFamily: "monospace" }}>
              {bar.display}
            </div>
          </div>
        );
      })}
    </div>
  );
};
```
A plain (non-`AbsoluteFill`) primitive so it can be embedded full-bleed (scenes 99, 100)
or positioned beside `CameraPlaceholder` in hybrid mode (scene 98). Bar color always
maps FP32→`#93c5fd`, FP16→`#4ade80`, INT8→`#f97316`, matching the three-precision
palette already used for engine-level results.

## Scene 98 — Chapter 11 (A-Roll, 720–930f / 7s)

```tsx
// scenes/Ch11Scene98.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { BarCompare } from "../BarCompare";

const BARS = [
  { label: "FP32", value: 1.808, display: "1.808 ms", color: "#93c5fd" },
  { label: "FP16", value: 0.645, display: "0.645 ms", color: "#4ade80" },
  { label: "INT8", value: 0.347, display: "0.347 ms", color: "#f97316" },
];

export const Ch11Scene98: React.FC = () => {
  const frame = useCurrentFrame();
  const bigOpacity = interpolate(frame, [130, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const bigScale = interpolate(frame, [130, 145], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <div style={{ position: "absolute", right: 100, top: 140 }}>
        <BarCompare bars={BARS} maxValue={1.808} maxBarWidth={420} />
      </div>
      <div
        style={{
          position: "absolute",
          left: 100,
          bottom: 130,
          opacity: bigOpacity,
          scale: bigScale,
          fontFamily: "Inter, sans-serif",
          fontWeight: 800,
          fontSize: 64,
          color: "#f97316",
        }}
      >
        5× faster
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So FP32 to INT8, that's five times faster. On the exact same weights, same
model."

## Scene 99 — Chapter 11 (B-Roll, 930–1110f / 6s)

```tsx
// scenes/Ch11Scene99.tsx
import { AbsoluteFill } from "remotion";
import { BarCompare } from "../BarCompare";

const SIZE_BARS = [
  { label: "FP32", value: 107.94, display: "107.94 MB", color: "#93c5fd" },
  { label: "FP16", value: 49.14, display: "49.14 MB", color: "#4ade80" },
  { label: "INT8", value: 25.26, display: "25.26 MB", color: "#f97316" },
];
const MEM_BARS = [
  { label: "FP32", value: 140, display: "140 MB", color: "#93c5fd" },
  { label: "FP16", value: 72, display: "72 MB", color: "#4ade80" },
  { label: "INT8", value: 52, display: "52 MB", color: "#f97316" },
];

export const Ch11Scene99: React.FC = () => (
  <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
    <div style={{ display: "flex", gap: 100 }}>
      <BarCompare title="Engine size (disk)" bars={SIZE_BARS} maxValue={107.94} maxBarWidth={340} />
      <BarCompare title="GPU memory" bars={MEM_BARS} maxValue={140} maxBarWidth={340} fromFrame={20} />
    </div>
  </AbsoluteFill>
);
```
Narration: "And latency's only part of it. Smaller on disk, smaller in GPU memory too.
Both roughly halve at each step."

## Scene 100 — Chapter 11 (B-Roll, 1110–1290f / 6s)

```tsx
// scenes/Ch11Scene100.tsx
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { BarCompare } from "../BarCompare";

const BUILD_BARS = [
  { label: "FP32", value: 18.6, display: "18.6 s", color: "#93c5fd" },
  { label: "FP16", value: 40.4, display: "40.4 s", color: "#4ade80" },
  { label: "INT8", value: 64.8, display: "64.8 s", color: "#f97316" },
];

export const Ch11Scene100: React.FC = () => {
  const frame = useCurrentFrame();
  const noteOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <BarCompare title="Build time" bars={BUILD_BARS} maxValue={64.8} maxBarWidth={600} />
      <div style={{ opacity: noteOpacity, marginTop: 20, fontFamily: "monospace", fontSize: 20, color: "#6b7280" }}>
        (INT8 cold build with fresh calibration: ~85.3s)
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "The one thing that gets worse is build time. Makes sense, INT8 has to
calibrate first."

## Scene 101 — Chapter 11 (A-Roll, 1290–1470f / 6s)

```tsx
// scenes/Ch11Scene101.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch11Scene101: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [15, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div
          style={{
            opacity,
            fontFamily: "monospace",
            fontSize: 34,
            color: "#f9fafb",
            background: "rgba(17,24,39,0.75)",
            padding: "14px 28px",
            borderRadius: 8,
          }}
        >
          FP16 → INT8 should be another 2×...
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "Okay so, FP16 to INT8 halves the bits again. So logically, this step
should roughly double the speed too, right?"

## Scene 102 — Chapter 11 (B-Roll, 1470–1650f / 6s)

```tsx
// scenes/Ch11Scene102.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch11Scene102: React.FC = () => {
  const frame = useCurrentFrame();
  const eqOpacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const resultScale = interpolate(frame, [60, 75], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const resultOpacity = interpolate(frame, [60, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const ghostOpacity = interpolate(frame, [100, 120], [0, 0.5], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: eqOpacity, fontFamily: "monospace", fontSize: 40, color: "#e5e7eb" }}>
        0.645 / 0.347 ={" "}
        <span style={{ scale: resultScale, opacity: resultOpacity, color: "#f97316", fontWeight: 700 }}>1.86×</span>
      </div>
      <div style={{ opacity: ghostOpacity, marginTop: 30, fontFamily: "monospace", fontSize: 28, color: "#6b7280" }}>
        expected: 2×
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Let's check. About one point nine x. Close. Not quite two x."

## Scene 103 — Chapter 11 (A-Roll, 1650–1800f / 5s)

```tsx
// scenes/Ch11Scene103.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch11Scene103: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "That gap's small enough you could just shrug it off. I don't want to
shrug it off."

## Scene 104 — Chapter 11 (A-Roll, 1800–1950f / 5s)

```tsx
// scenes/Ch11Scene104.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch11Scene104: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "But before I go blaming hardware limits, let's rule out something dumber
first. Is this engine even really running in INT8?"

## Scene 105 — Chapter 11 (B-Roll, chapter card, 1950–2070f / 4s, no VO)

```tsx
// scenes/Ch11Scene105.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch11Scene105: React.FC = () => (
  <ChapterCard number={12} title="Is INT8 actually running in INT8?" durationInFrames={120} />
);
```

## Chapter 11 assembly

```tsx
// chapters/Chapter11.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch11Scene94 } from "../scenes/Ch11Scene94";
import { Ch11Scene95 } from "../scenes/Ch11Scene95";
import { Ch11Scene96 } from "../scenes/Ch11Scene96";
import { Ch11Scene97 } from "../scenes/Ch11Scene97";
import { Ch11Scene98 } from "../scenes/Ch11Scene98";
import { Ch11Scene99 } from "../scenes/Ch11Scene99";
import { Ch11Scene100 } from "../scenes/Ch11Scene100";
import { Ch11Scene101 } from "../scenes/Ch11Scene101";
import { Ch11Scene102 } from "../scenes/Ch11Scene102";
import { Ch11Scene103 } from "../scenes/Ch11Scene103";
import { Ch11Scene104 } from "../scenes/Ch11Scene104";
import { Ch11Scene105 } from "../scenes/Ch11Scene105";

// Local cut points (frames, 30fps) for this chapter's own 0..2070 timeline.
const CUTS = [0, 180, 360, 540, 720, 930, 1110, 1290, 1470, 1650, 1800, 1950, 2070];
const SCENES = [
  Ch11Scene94,
  Ch11Scene95,
  Ch11Scene96,
  Ch11Scene97,
  Ch11Scene98,
  Ch11Scene99,
  Ch11Scene100,
  Ch11Scene101,
  Ch11Scene102,
  Ch11Scene103,
  Ch11Scene104,
  Ch11Scene105,
];

// Global offset: Chapters 0-10 run an estimated 17250f (575s) before this chapter starts.
const CHAPTER_START_GLOBAL = 17250;

export const Chapter11: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 2070}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 12 — Is INT8 Actually Running In INT8?

Scenes 106–115 (10 scenes, 59s / 1770f at 30fps). Continues after Chapter 11 in the
master timeline (global offset 19320f) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 106 — Chapter 12 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch12Scene106.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch12Scene106: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Because setting a builder flag that says 'use INT8' doesn't prove every
layer got an INT8 kernel."

## Scene 107 — Chapter 12 (B-Roll, 180–360f / 6s)

```tsx
// scenes/Ch12Scene107.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const PRE = "$ nsys profile ... ";
const HIGHLIGHT = "nsys_run_engine.py";
const FULL = PRE + HIGHLIGHT;

export const Ch12Scene107: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 40], [0, FULL.length], {
    extrapolateRight: "clamp",
  }));
  const shown = FULL.slice(0, typedChars);
  const preShown = shown.slice(0, PRE.length);
  const highlightShown = shown.slice(PRE.length);

  const boxOpacity = interpolate(frame, [55, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const subOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center", fontFamily: "monospace" }}>
      <div style={{ fontSize: 32, color: "#22c55e" }}>
        {preShown}
        <span
          style={{
            color: boxOpacity > 0 ? "#0b0f14" : "#22c55e",
            background: `rgba(96,165,250,${boxOpacity})`,
            borderRadius: 4,
            padding: "0 4px",
          }}
        >
          {highlightShown}
        </span>
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0, color: "#22c55e" }}>_</span>
      </div>
      <div style={{ opacity: subOpacity, marginTop: 28, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>
        traces the literal CUDA kernels — no TensorRT instrumentation in the way
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So let's use Nsight Systems. It traces the literal CUDA kernels that
ran, no TensorRT instrumentation getting in the way."

## Scene 108 — Chapter 12 (B-Roll, 360–570f / 7s)

```tsx
// scenes/Ch12Scene108.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const PREFIX = "sm80_xmma_fprop_implicit_gemm_interleaved_";
const HIGHLIGHT = "i8i8";
const SUFFIX = "_i8i32_f32_nchw...";
const FULL = PREFIX + HIGHLIGHT + SUFFIX;

export const Ch12Scene108: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(interpolate(frame, [0, 90], [0, FULL.length], {
    extrapolateRight: "clamp",
  }));
  const shown = FULL.slice(0, typedChars);
  const prefixShown = shown.slice(0, PREFIX.length);
  const highlightShown = shown.slice(PREFIX.length, PREFIX.length + HIGHLIGHT.length);
  const suffixShown = shown.slice(PREFIX.length + HIGHLIGHT.length);

  const pulse = interpolate(frame, [100, 115, 130, 145], [1, 1.15, 1, 1.08], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const highlightColor = interpolate(frame, [95, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const captionOpacity = interpolate(frame, [150, 170], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center", padding: 100 }}>
      <div style={{ fontFamily: "monospace", fontSize: 26, color: "#e5e7eb", textAlign: "center", maxWidth: 1500, wordBreak: "break-all" }}>
        {prefixShown}
        <span
          style={{
            scale: pulse,
            display: "inline-block",
            color: highlightColor > 0 ? "#0b0f14" : "#e5e7eb",
            background: `rgba(249,115,22,${highlightColor})`,
            borderRadius: 4,
            padding: "0 2px",
            fontWeight: 700,
          }}
        >
          {highlightShown}
        </span>
        {suffixShown}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div style={{ opacity: captionOpacity, marginTop: 36, fontFamily: "Inter, sans-serif", fontSize: 24, color: "#9ca3af" }}>
        the actual kernel that ran on the GPU
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "'i8i8.' Right there in the kernel name. [PAUSE] That's the actual
kernel that ran on the GPU. I didn't have to take TensorRT's word for it."

## Scene 109 — Chapter 12 (B-Roll, 570–750f / 6s)

```tsx
// scenes/Ch12Scene109.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const BAR_WIDTH = 1000;

export const Ch12Scene109: React.FC = () => {
  const frame = useCurrentFrame();
  const fillWidth = interpolate(frame, [15, 90], [0, BAR_WIDTH * 0.964], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const numbersOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const pctScale = interpolate(frame, [90, 105], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const pctOpacity = interpolate(frame, [90, 105], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: numbersOpacity, fontFamily: "monospace", fontSize: 28, color: "#9ca3af", marginBottom: 20 }}>
        17,280 / 17,920 kernel launches
      </div>
      <div style={{ position: "relative", width: BAR_WIDTH, height: 28, background: "#1f2937", borderRadius: 999 }}>
        <div style={{ width: fillWidth, height: 28, background: "#4ade80", borderRadius: 999 }} />
      </div>
      <div style={{ opacity: pctOpacity, scale: pctScale, marginTop: 28, fontFamily: "Inter, sans-serif", fontSize: 64, fontWeight: 700, color: "#4ade80" }}>
        96.4%
      </div>
      <div style={{ opacity: pctOpacity, marginTop: 8, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>
        genuinely running on integer tensor cores
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Across three hundred iterations, about ninety-six percent of every
single kernel launch genuinely runs on integer tensor cores."

## Scene 110 — Chapter 12 (B-Roll, 750–960f / 7s)

```tsx
// scenes/Ch12Scene110.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const TOTAL_LAYERS = 58;
const FLOAT_INDICES = [56, 57]; // final classifier bias-add + its reshape

export const Ch12Scene110: React.FC = () => {
  const frame = useCurrentFrame();
  const boxesIn = interpolate(frame, [0, 70], [0, TOTAL_LAYERS], { extrapolateRight: "clamp" });
  const labelOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", flexWrap: "wrap", width: 460, gap: 6 }}>
        {Array.from({ length: TOTAL_LAYERS }, (_, i) => {
          const shown = i < boxesIn;
          const isFloat = FLOAT_INDICES.includes(i);
          return (
            <div
              key={i}
              style={{
                width: 32,
                height: 32,
                borderRadius: 4,
                background: !shown ? "#1f2937" : isFloat ? "#ef4444" : "#4ade80",
                opacity: shown ? 1 : 0.3,
              }}
            />
          );
        })}
      </div>
      <div style={{ opacity: labelOpacity, marginTop: 32, fontFamily: "Inter, sans-serif", fontSize: 26, color: "#f9fafb", textAlign: "center" }}>
        58 layers — <span style={{ color: "#ef4444" }}>2 flagged Float</span>
        <div style={{ fontSize: 20, color: "#9ca3af", marginTop: 8 }}>
          final classifier bias-add and its reshape
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "The only two layers that stay in float are the tiny final classifier
bias-add. TensorRT chose to leave those two in full precision on purpose."

## Scene 111 — Chapter 12 (A-Roll, 960–1140f / 6s)

```tsx
// scenes/Ch12Scene111.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch12Scene111: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="0.034ms out of ~0.45ms" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "Makes sense, actually. It's cheap to keep at full precision, and it
directly produces the logits your argmax depends on. Worth protecting."

## Scene 112 — Chapter 12 (A-Roll, 1140–1320f / 6s)

```tsx
// scenes/Ch12Scene112.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch12Scene112: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So that missing 0.3x isn't hiding in some fallback layer. It's got to
be something else."

## Scene 113 — Chapter 12 (B-Roll, 1320–1500f / 6s)

```tsx
// scenes/Ch12Scene113.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const SEGMENTS = [
  { label: "Conv/GEMM", pct: 95.0, color: "#60a5fa" },
  { label: "Reformat", pct: 1.4, color: "#6b7280" },
  { label: "Pooling", pct: 3.6, color: "#4ade80" },
];
const BAR_WIDTH = 1200;

export const Ch12Scene113: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", width: BAR_WIDTH, height: 56, borderRadius: 10, overflow: "hidden", background: "#1f2937" }}>
        {SEGMENTS.map((seg, i) => {
          const start = 10 + i * 25;
          const segWidth = interpolate(frame, [start, start + 30], [0, (seg.pct / 100) * BAR_WIDTH], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          return <div key={seg.label} style={{ width: segWidth, height: 56, background: seg.color }} />;
        })}
      </div>
      <div style={{ display: "flex", gap: 48, marginTop: 28, fontFamily: "Inter, sans-serif", fontSize: 22 }}>
        {SEGMENTS.map((seg, i) => {
          const start = 10 + i * 25;
          const opacity = interpolate(frame, [start + 20, start + 40], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div key={seg.label} style={{ opacity, color: seg.color }}>
              {seg.label} — {seg.pct}%
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "And it's not reformat overhead either. That's like one and a half
percent of GPU time. Barely anything."

## Scene 114 — Chapter 12 (A-Roll, 1500–1650f / 5s)

```tsx
// scenes/Ch12Scene114.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch12Scene114: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So it's ninety-five percent compute-bound, genuinely running INT8
kernels. So why isn't it two x?"

## Scene 115 — Chapter 12 (B-Roll, chapter card, 1650–1770f / 4s, no VO)

```tsx
// scenes/Ch12Scene115.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch12Scene115: React.FC = () => (
  <ChapterCard number={13} title="Why isn't it 4x?" durationInFrames={120} />
);
```

## Chapter 12 assembly

```tsx
// chapters/Chapter12.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch12Scene106 } from "../scenes/Ch12Scene106";
import { Ch12Scene107 } from "../scenes/Ch12Scene107";
import { Ch12Scene108 } from "../scenes/Ch12Scene108";
import { Ch12Scene109 } from "../scenes/Ch12Scene109";
import { Ch12Scene110 } from "../scenes/Ch12Scene110";
import { Ch12Scene111 } from "../scenes/Ch12Scene111";
import { Ch12Scene112 } from "../scenes/Ch12Scene112";
import { Ch12Scene113 } from "../scenes/Ch12Scene113";
import { Ch12Scene114 } from "../scenes/Ch12Scene114";
import { Ch12Scene115 } from "../scenes/Ch12Scene115";

// Local cut points (frames, 30fps) for this chapter's own 0..1770 timeline.
const CUTS = [0, 180, 360, 570, 750, 960, 1140, 1320, 1500, 1650, 1770];
const SCENES = [
  Ch12Scene106,
  Ch12Scene107,
  Ch12Scene108,
  Ch12Scene109,
  Ch12Scene110,
  Ch12Scene111,
  Ch12Scene112,
  Ch12Scene113,
  Ch12Scene114,
  Ch12Scene115,
];

// Global offset: Chapter 12 starts 19320f into the master timeline.
const CHAPTER_START_GLOBAL = 19320;

export const Chapter12: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 1770}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 13 — Why Isn't It 4x?

Scenes 116–127 (12 scenes, 70s / 2100f at 30fps). Continues immediately after
Chapter 12 in the master timeline (global offset 21090f) and pulls that same slice
of `audio/narration_normal_speed.wav`. Investigative, quiet/calm throughout, with a
SLOW REVEAL beat at Scene 123 — the video's core reversal (measured FP16→INT8
speedup was 1.7x, not the rated 2x) — so that scene's overlay is paced with a
genuinely held gap between its two lines, not a quick double-reveal.

## Scene 116 — Chapter 13 (B-Roll, 0–180f / 6s)

```tsx
// scenes/Ch13Scene116.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const ROWS = [
  { label: "GPU kernel time (ms)", fp32: "2.008", fp16: "0.756", int8: "0.446" },
  { label: "vs. previous precision", fp32: "—", fp16: "2.65x", int8: "1.70x" },
];

export const Ch13Scene116: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "Inter, sans-serif", color: "#f9fafb" }}>
        <div style={{ display: "grid", gridTemplateColumns: "360px repeat(3, 200px)", fontSize: 28 }}>
          <div />
          {["FP32", "FP16", "INT8"].map((h) => (
            <div key={h} style={{ fontWeight: 700, textAlign: "center", color: "#93c5fd" }}>{h}</div>
          ))}
          {ROWS.map((row, i) => {
            const start = 20 + i * 30;
            const opacity = interpolate(frame, [start, start + 15], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            return (
              <React.Fragment key={row.label}>
                <div style={{ opacity, padding: "10px 0" }}>{row.label}</div>
                <div style={{ opacity, textAlign: "center", fontFamily: "monospace" }}>{row.fp32}</div>
                <div style={{ opacity, textAlign: "center", fontFamily: "monospace" }}>{row.fp16}</div>
                <div style={{ opacity, textAlign: "center", fontFamily: "monospace", color: "#4ade80" }}>{row.int8}</div>
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Pure GPU kernel time, no profiler overhead in the way: about two point
six x for the first step, one point seven x for the second."

## Scene 117 — Chapter 13 (A-Roll, 180–360f / 6s)

```tsx
// scenes/Ch13Scene117.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch13Scene117: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Tensor cores: rated ~2x INT8 vs FP16 throughput" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "These tensor cores are literally rated for double the FP16 throughput
in INT8. On paper this should've been the easy step."

## Scene 118 — Chapter 13 (B-Roll, 360–540f / 6s)

```tsx
// scenes/Ch13Scene118.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CHART_HEIGHT = 400;
const PEAK_HEIGHT = 400;
const ACHIEVED_HEIGHT = 220;

export const Ch13Scene118: React.FC = () => {
  const frame = useCurrentFrame();
  const peakH = interpolate(frame, [10, 55], [0, PEAK_HEIGHT], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const achievedH = interpolate(frame, [30, 75], [0, ACHIEVED_HEIGHT], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const gapOpacity = interpolate(frame, [90, 115], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", display: "flex", alignItems: "flex-end", gap: 100, height: CHART_HEIGHT }}>
        <div
          style={{
            position: "absolute",
            left: -20,
            right: -20,
            bottom: achievedH,
            height: peakH - achievedH,
            opacity: gapOpacity,
            background: "rgba(249,115,22,0.14)",
            borderTop: "1px dashed #f97316",
            borderBottom: "1px dashed #f97316",
          }}
        />
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ width: 140, height: peakH, background: "#374151", borderRadius: "6px 6px 0 0" }} />
          <div style={{ marginTop: 16, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#93c5fd", textAlign: "center" }}>
            PEAK
            <div style={{ fontSize: 15, color: "#6b7280" }}>theoretical</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ width: 140, height: achievedH, background: "#4ade80", borderRadius: "6px 6px 0 0" }} />
          <div style={{ marginTop: 16, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#4ade80", textAlign: "center" }}>
            ACHIEVED
            <div style={{ fontSize: 15, color: "#6b7280" }}>measured</div>
          </div>
        </div>
      </div>
      <div style={{ opacity: gapOpacity, marginTop: 24, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#f97316" }}>
        the gap between rated and real
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "But 'rated for 2x' is a peak number. And peak numbers assume the
kernel runs long enough to get there."

## Scene 119 — Chapter 13 (B-Roll, 540–750f / 7s)

```tsx
// scenes/Ch13Scene119.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const TILE_OUTLINE = 320;
const WORKLOAD_SIZE = 120;

export const Ch13Scene119: React.FC = () => {
  const frame = useCurrentFrame();
  const outlineOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const fillScale = interpolate(frame, [30, 60], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const labelOpacity = interpolate(frame, [90, 115], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          position: "relative",
          width: TILE_OUTLINE,
          height: TILE_OUTLINE,
          border: "2px dashed #60a5fa",
          opacity: outlineOpacity,
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: WORKLOAD_SIZE,
            height: WORKLOAD_SIZE,
            scale: fillScale,
            background: "#4ade80",
            borderRadius: 4,
          }}
        />
      </div>
      <div style={{ display: "flex", gap: 48, marginTop: 28, fontFamily: "Inter, sans-serif", fontSize: 20 }}>
        <div style={{ opacity: outlineOpacity, color: "#93c5fd" }}>tensor-core tile</div>
        <div style={{ opacity: labelOpacity, color: "#4ade80" }}>batch=1 conv layer</div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "At batch size one, most of ResNet50's individual conv layers are just
small. Too small to fully fill a tensor-core tile."

## Scene 120 — Chapter 13 (B-Roll, 750–960f / 7s)

```tsx
// scenes/Ch13Scene120.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const OVERHEAD_WIDTH = 90;
const FP16_COMPUTE = 560;
const INT8_COMPUTE = 280;

export const Ch13Scene120: React.FC = () => {
  const frame = useCurrentFrame();
  const fp16Width = interpolate(frame, [10, 55], [0, FP16_COMPUTE], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const int8Width = interpolate(frame, [70, 105], [0, INT8_COMPUTE], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelOpacity = interpolate(frame, [120, 140], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const row = (label: string, computeWidth: number, color: string) => (
    <div key={label} style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 24 }}>
      <div style={{ width: 60, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#9ca3af" }}>{label}</div>
      <div style={{ display: "flex", height: 44 }}>
        <div style={{ width: OVERHEAD_WIDTH, height: 44, background: "#f97316", borderRadius: "6px 0 0 6px" }} />
        <div style={{ width: computeWidth, height: 44, background: color, borderRadius: "0 6px 6px 0" }} />
      </div>
    </div>
  );

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div>
        {row("FP16", fp16Width, "#60a5fa")}
        {row("INT8", int8Width, "#4ade80")}
      </div>
      <div style={{ opacity: labelOpacity, marginTop: 12, display: "flex", gap: 40, fontFamily: "Inter, sans-serif", fontSize: 20 }}>
        <div style={{ color: "#f97316" }}>launch overhead — fixed</div>
        <div style={{ color: "#9ca3af" }}>compute — shrinks with precision</div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Every kernel launch has some fixed setup cost attached to it. Halving
the math time doesn't halve that fixed cost."

## Scene 121 — Chapter 13 (B-Roll, 960–1140f / 6s)

```tsx
// scenes/Ch13Scene121.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch13Scene121: React.FC = () => {
  const frame = useCurrentFrame();
  const lineWidth = interpolate(frame, [10, 60], [0, 1100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelsOpacity = interpolate(frame, [60, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const subOpacity = interpolate(frame, [95, 115], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 1100, height: 4, background: "#374151" }}>
        <div style={{ position: "absolute", left: 0, top: 0, width: lineWidth, height: 4, background: "#4ade80" }} />
        <div
          style={{
            opacity: labelsOpacity,
            position: "absolute",
            left: -10,
            top: 20,
            fontFamily: "monospace",
            fontSize: 24,
            color: "#9ca3af",
          }}
        >
          ~4,000 ns
        </div>
        <div
          style={{
            opacity: labelsOpacity,
            position: "absolute",
            right: -10,
            top: 20,
            fontFamily: "monospace",
            fontSize: 24,
            color: "#9ca3af",
          }}
        >
          ~17,000 ns
        </div>
      </div>
      <div style={{ opacity: subOpacity, marginTop: 60, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#f9fafb", textAlign: "center" }}>
        hundreds of small conv layers per inference
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "We're talking microsecond-scale kernels here. That fixed overhead eats
up a much bigger fraction of a small kernel than a big one."

## Scene 122 — Chapter 13 (A-Roll, 1140–1320f / 6s)

```tsx
// scenes/Ch13Scene122.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch13Scene122: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="FP32 kernels start further from the overhead floor" fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "Which is also why FP32 got closer to a clean 2x earlier. Its kernels
were slower to begin with, more room to cut before hitting that same floor."

## Scene 123 — Chapter 13 (A-Roll, SLOW REVEAL, 1320–1500f / 6s)

```tsx
// scenes/Ch13Scene123.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch13Scene123: React.FC = () => {
  const frame = useCurrentFrame();
  const line1Opacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const line1TranslateY = interpolate(frame, [20, 40], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  // Held beat: line 1 sits alone on screen from frame 40 to frame 112 — about
  // 2.4s — a genuine pause before the reversal lands, not a quick double-reveal.
  // This is the video's core reversal: measured FP16→INT8 was 1.7x, not 2x.
  const line2Opacity = interpolate(frame, [112, 135], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const line2TranslateY = interpolate(frame, [112, 135], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 130 }}>
        <div
          style={{
            opacity: line1Opacity,
            translate: `0px ${line1TranslateY}px`,
            fontFamily: "Inter, sans-serif",
            fontSize: 34,
            fontWeight: 600,
            color: "#93c5fd",
            background: "rgba(17,24,39,0.7)",
            padding: "10px 22px",
            borderRadius: 6,
            marginBottom: 14,
          }}
        >
          Compute-bound overall.
        </div>
        <div
          style={{
            opacity: line2Opacity,
            translate: `0px ${line2TranslateY}px`,
            fontFamily: "Inter, sans-serif",
            fontSize: 34,
            fontWeight: 600,
            color: "#f97316",
            background: "rgba(17,24,39,0.7)",
            padding: "10px 22px",
            borderRadius: 6,
          }}
        >
          Launch-overhead-bound per layer at batch=1.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "So overall, the model's compute-bound. [PAUSE] But at batch one, a lot
of individual layers are launch-overhead-bound instead."

## Scene 124 — Chapter 13 (A-Roll, 1500–1680f / 6s)

```tsx
// scenes/Ch13Scene124.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch13Scene124: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So if launch overhead's the real limiter here, can we get rid of it,
without touching the model at all?"

## Scene 125 — Chapter 13 (B-Roll, 1680–1830f / 5s)

```tsx
// scenes/Ch13Scene125.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const NODES = 5;

export const Ch13Scene125: React.FC = () => {
  const frame = useCurrentFrame();
  const titleOpacity = interpolate(frame, [50, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center" }}>
        {Array.from({ length: NODES }, (_, i) => {
          const start = i * 8;
          const nodeOpacity = interpolate(frame, [start, start + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const nodeScale = interpolate(frame, [start, start + 10], [0.4, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          const lineOpacity = interpolate(frame, [start + 6, start + 16], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <React.Fragment key={i}>
              {i > 0 && <div style={{ opacity: lineOpacity, width: 36, height: 2, background: "#4ade80" }} />}
              <div
                style={{
                  opacity: nodeOpacity,
                  scale: nodeScale,
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  background: "#4ade80",
                }}
              />
            </React.Fragment>
          );
        })}
      </div>
      <div
        style={{
          opacity: titleOpacity,
          marginTop: 36,
          fontFamily: "Inter, sans-serif",
          fontSize: 48,
          fontWeight: 700,
          color: "#f9fafb",
          letterSpacing: 1,
        }}
      >
        CUDA GRAPHS
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Turns out, yeah. CUDA graphs."

## Scene 126 — Chapter 13 (A-Roll, 1830–1980f / 5s)

```tsx
// scenes/Ch13Scene126.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch13Scene126: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "And this repo's already got both code paths, so we can just compare
them directly."

## Scene 127 — Chapter 13 (B-Roll, chapter card, 1980–2100f / 4s, no VO)

```tsx
// scenes/Ch13Scene127.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch13Scene127: React.FC = () => (
  <ChapterCard number={14} title="Removing the launch overhead — CUDA graphs." durationInFrames={120} />
);
```

## Chapter 13 assembly

```tsx
// chapters/Chapter13.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch13Scene116 } from "../scenes/Ch13Scene116";
import { Ch13Scene117 } from "../scenes/Ch13Scene117";
import { Ch13Scene118 } from "../scenes/Ch13Scene118";
import { Ch13Scene119 } from "../scenes/Ch13Scene119";
import { Ch13Scene120 } from "../scenes/Ch13Scene120";
import { Ch13Scene121 } from "../scenes/Ch13Scene121";
import { Ch13Scene122 } from "../scenes/Ch13Scene122";
import { Ch13Scene123 } from "../scenes/Ch13Scene123";
import { Ch13Scene124 } from "../scenes/Ch13Scene124";
import { Ch13Scene125 } from "../scenes/Ch13Scene125";
import { Ch13Scene126 } from "../scenes/Ch13Scene126";
import { Ch13Scene127 } from "../scenes/Ch13Scene127";

// Local cut points (frames, 30fps) for this chapter's own 0..2100 timeline.
const CUTS = [0, 180, 360, 540, 750, 960, 1140, 1320, 1500, 1680, 1830, 1980, 2100];
const SCENES = [
  Ch13Scene116,
  Ch13Scene117,
  Ch13Scene118,
  Ch13Scene119,
  Ch13Scene120,
  Ch13Scene121,
  Ch13Scene122,
  Ch13Scene123,
  Ch13Scene124,
  Ch13Scene125,
  Ch13Scene126,
  Ch13Scene127,
];

// Global offset: Chapter 13 starts 21090f into the master timeline
// (immediately after Chapter 12 ends).
const CHAPTER_START_GLOBAL = 21090;

export const Chapter13: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_START_GLOBAL + 2100}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 14 — Removing The Launch Overhead (CUDA Graphs)

Scenes 128–137 (10 scenes, 60s / 1800f at 30fps). Continues the master timeline at
global offset 23190f (~773s) and pulls that slice of `audio/narration_normal_speed.wav`.

## Scene 128 — Chapter 14 (B-Roll, 0–210f / 7s)

```tsx
// scenes/Ch14Scene128.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const COLS = 13;
const ROWS = 8;
const COUNT = COLS * ROWS; // ~100+ individual kernel launches

export const Ch14Scene128: React.FC = () => {
  const frame = useCurrentFrame();

  const launchesOpacity = interpolate(frame, [0, 40, 95, 130], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const launchesScale = interpolate(frame, [95, 140], [1, 0.5], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  const graphOpacity = interpolate(frame, [120, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const graphScale = interpolate(frame, [120, 150], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: launchesOpacity,
          scale: launchesScale,
          display: "grid",
          gridTemplateColumns: `repeat(${COLS}, 18px)`,
          gap: 6,
        }}
      >
        {Array.from({ length: COUNT }, (_, i) => {
          const start = Math.floor(i / 3);
          const itemOpacity = interpolate(frame, [start, start + 8], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          return (
            <div
              key={i}
              style={{
                width: 18,
                height: 18,
                borderRadius: 4,
                background: "#1f2937",
                border: "1px solid #374151",
                opacity: itemOpacity,
              }}
            />
          );
        })}
      </div>
      <div
        style={{
          position: "absolute",
          opacity: graphOpacity,
          scale: graphScale,
          fontFamily: "Inter, sans-serif",
          fontSize: 34,
          fontWeight: 700,
          color: "#f9fafb",
          background: "rgba(17,24,39,0.9)",
          border: "1px solid #4ade80",
          borderRadius: 999,
          padding: "18px 40px",
        }}
      >
        one graph launch <span style={{ color: "#4ade80" }}>→</span>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Idea is: capture the whole sequence of per-layer kernel launches just
once, then replay it as a single graph launch every time after that."

## Scene 129 — Chapter 14 (B-Roll, 210–390f / 6s)

```tsx
// scenes/Ch14Scene129.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch14Scene129: React.FC = () => {
  const frame = useCurrentFrame();
  const boxOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const highlightAlpha = interpolate(frame, [45, 65], [0, 0.18], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: boxOpacity,
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "32px 44px",
          fontFamily: "monospace",
          fontSize: 26,
        }}
      >
        <div style={{ color: "#6b7280", marginBottom: 10 }}>build_and_bench.py</div>
        <div style={{ color: "#e5e7eb" }}>
          benchmark(engine, inputs,
          <br />
          &nbsp;&nbsp;warmup=50, iters=500,
          <br />
          &nbsp;&nbsp;
          <span
            style={{
              background: `rgba(96,165,250,${highlightAlpha})`,
              color: "#60a5fa",
              borderRadius: 4,
              padding: "2px 4px",
            }}
          >
            use_cuda_graph=True
          </span>
          )
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "It's already sitting right here in the repo's own benchmark function.
One flag."

## Scene 130 — Chapter 14 (B-Roll, 390–570f / 6s)

```tsx
// scenes/Ch14Scene130.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LINES = [
  { code: "cudaStreamBeginCapture(stream);", start: 20 },
  { code: "cudaGraphInstantiate(&exec, graph);", start: 60 },
  { code: "cudaGraphLaunch(exec, stream);", start: 100 },
];

export const Ch14Scene130: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "36px 48px",
          fontFamily: "monospace",
          fontSize: 27,
        }}
      >
        {LINES.map((line) => {
          const opacity = interpolate(frame, [line.start, line.start + 14], [0.3, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const highlightAlpha = interpolate(
            frame,
            [line.start, line.start + 14, line.start + 34],
            [0, 0.18, 0.06],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={line.code}
              style={{
                opacity,
                color: "#e5e7eb",
                padding: "10px 14px",
                marginBottom: 8,
                borderRadius: 6,
                background: `rgba(96,165,250,${highlightAlpha})`,
              }}
            >
              {line.code}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Capture once, instantiate once, then just launch it over and over."

## Scene 131 — Chapter 14 (A-Roll, 570–750f / 6s)

```tsx
// scenes/Ch14Scene131.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch14Scene131: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="max abs diff = 0.0 — verified" fromFrame={25} />
  </AbsoluteFill>
);
```
Narration: "Before I trust any speed number out of this, worth saying: it's already
been checked bit-identical against the normal path."

## Shared primitive: DeltaStat

Reused for the two back-to-back B-Roll result reveals (Scenes 132, 133) — a
precision label over a large monospace delta, in the house "positive result"
green.

```tsx
// DeltaStat.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const DeltaStat: React.FC<{ label: string; delta: string; accent?: string }> = ({
  label,
  delta,
  accent = "#4ade80",
}) => {
  const frame = useCurrentFrame();
  const labelOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const deltaScale = interpolate(frame, [25, 50], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const deltaOpacity = interpolate(frame, [25, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: labelOpacity,
          fontFamily: "Inter, sans-serif",
          fontSize: 26,
          color: "#9ca3af",
          textTransform: "uppercase",
          letterSpacing: 2,
          marginBottom: 18,
        }}
      >
        {label}
      </div>
      <div
        style={{
          opacity: deltaOpacity,
          scale: deltaScale,
          fontFamily: "monospace",
          fontSize: 72,
          fontWeight: 700,
          color: accent,
        }}
      >
        {delta}
      </div>
    </AbsoluteFill>
  );
};
```

## Scene 132 — Chapter 14 (B-Roll, 750–960f / 7s)

```tsx
// scenes/Ch14Scene132.tsx
import { DeltaStat } from "../DeltaStat";

export const Ch14Scene132: React.FC = () => <DeltaStat label="FP32" delta="+3–7%" />;
```
Narration: "FP32 first. Small gain. Three to seven percent."

## Scene 133 — Chapter 14 (B-Roll, 960–1140f / 6s)

```tsx
// scenes/Ch14Scene133.tsx
import { DeltaStat } from "../DeltaStat";

export const Ch14Scene133: React.FC = () => <DeltaStat label="FP16" delta="+11–14%" />;
```
Narration: "FP16's bigger. Eleven to fourteen percent."

## Scene 134 — Chapter 14 (A-Roll, 1140–1350f / 7s)

```tsx
// scenes/Ch14Scene134.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch14Scene134: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="INT8: +19–30%" fromFrame={30} />
  </AbsoluteFill>
);
```
Narration: "And INT8, the one with the smallest, most overhead-bound kernels, gets
the **biggest win of all.** [PAUSE] Which basically confirms the whole theory."

## Scene 135 — Chapter 14 (B-Roll, 1350–1530f / 6s)

```tsx
// scenes/Ch14Scene135.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const MIN = 1.5;
const MAX = 2.05;
const TRACK_WIDTH = 1000;
const toX = (v: number) => ((v - MIN) / (MAX - MIN)) * TRACK_WIDTH;

export const Ch14Scene135: React.FC = () => {
  const frame = useCurrentFrame();
  const trackOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const markerX = interpolate(frame, [25, 90], [toX(1.7), toX(1.86)], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelValue = interpolate(frame, [25, 90], [1.7, 1.86], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: trackOpacity,
          position: "relative",
          width: TRACK_WIDTH,
          height: 6,
          background: "#1f2937",
          borderRadius: 3,
        }}
      >
        <div
          style={{
            position: "absolute",
            left: toX(2.0),
            top: -22,
            width: 2,
            height: 50,
            background: "#374151",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: toX(2.0) - 24,
            top: -56,
            fontFamily: "monospace",
            fontSize: 20,
            color: "#6b7280",
          }}
        >
          2.0x target
        </div>
        <div
          style={{
            position: "absolute",
            left: markerX - 9,
            top: -12,
            width: 18,
            height: 18,
            borderRadius: 9,
            background: "#4ade80",
            boxShadow: "0 0 16px rgba(74,222,128,0.6)",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: markerX - 40,
            top: 26,
            width: 120,
            textAlign: "center",
            fontFamily: "monospace",
            fontSize: 32,
            fontWeight: 700,
            color: "#f9fafb",
          }}
        >
          {labelValue.toFixed(2)}x
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So the FP16-to-INT8 ratio moves, from one point seven up to **almost
one point nine.**"

## Scene 136 — Chapter 14 (A-Roll, 1530–1680f / 5s)

```tsx
// scenes/Ch14Scene136.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch14Scene136: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Closer to two x. Not all the way there. That batch-size limit is still
real."

## Scene 137 — Chapter 14 (B-Roll, chapter card, 1680–1800f / 4s, no VO)

```tsx
// scenes/Ch14Scene137.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch14Scene137: React.FC = () => (
  <ChapterCard number={15} title="The flag that wasn't as good as it looked." durationInFrames={120} />
);
```

## Chapter 14 assembly

```tsx
// chapters/Chapter14.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch14Scene128 } from "../scenes/Ch14Scene128";
import { Ch14Scene129 } from "../scenes/Ch14Scene129";
import { Ch14Scene130 } from "../scenes/Ch14Scene130";
import { Ch14Scene131 } from "../scenes/Ch14Scene131";
import { Ch14Scene132 } from "../scenes/Ch14Scene132";
import { Ch14Scene133 } from "../scenes/Ch14Scene133";
import { Ch14Scene134 } from "../scenes/Ch14Scene134";
import { Ch14Scene135 } from "../scenes/Ch14Scene135";
import { Ch14Scene136 } from "../scenes/Ch14Scene136";
import { Ch14Scene137 } from "../scenes/Ch14Scene137";

// Local cut points (frames, 30fps) for this chapter's own 0..1800 timeline.
const CUTS = [0, 210, 390, 570, 750, 960, 1140, 1350, 1530, 1680, 1800];
const SCENES = [
  Ch14Scene128,
  Ch14Scene129,
  Ch14Scene130,
  Ch14Scene131,
  Ch14Scene132,
  Ch14Scene133,
  Ch14Scene134,
  Ch14Scene135,
  Ch14Scene136,
  Ch14Scene137,
];

// Global offset: this chapter's slice of the narration file, 23190f–24990f.
const GLOBAL_START = 23190;
const GLOBAL_END = 24990;

export const Chapter14: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={GLOBAL_START}
      trimAfter={GLOBAL_END}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 15 — The Flag That Wasn't As Good As It Looked (DIRECT_IO)

Scenes 138–147 (10 scenes, 61s / 1830f at 30fps). Continues immediately after
Chapter 14 at global offset 24990f (~833s).

## Scene 138 — Chapter 15 (B-Roll, 0–180f / 6s)

```tsx
// scenes/Ch15Scene138.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch15Scene138: React.FC = () => {
  const frame = useCurrentFrame();
  const boxOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const highlightAlpha = interpolate(frame, [45, 65], [0, 0.18], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const noteOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: boxOpacity,
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 12,
          padding: "34px 46px",
          fontFamily: "monospace",
          fontSize: 27,
        }}
      >
        <div style={{ color: "#6b7280", marginBottom: 10 }}>build_engine.py</div>
        <div style={{ color: "#e5e7eb" }}>
          config.set_flag(
          <br />
          &nbsp;&nbsp;
          <span
            style={{
              background: `rgba(96,165,250,${highlightAlpha})`,
              color: "#60a5fa",
              borderRadius: 4,
              padding: "2px 4px",
            }}
          >
            trt.BuilderFlag.DIRECT_IO
          </span>
          <br />
          )
        </div>
      </div>
      <div
        style={{
          opacity: noteOpacity,
          marginTop: 28,
          fontFamily: "Inter, sans-serif",
          fontSize: 22,
          color: "#9ca3af",
        }}
      >
        removes the reformat layer at the input boundary
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "One more thing was tried: a flag called DIRECT_IO. It removes a
reformat layer sitting right at the input boundary."

## Scene 139 — Chapter 15 (B-Roll, 180–360f / 6s)

```tsx
// scenes/Ch15Scene139.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const BARS = [
  { label: "conv", height: 70, color: "#60a5fa" },
  { label: "reformat", height: 150, color: "#f97316" },
];

export const Ch15Scene139: React.FC = () => {
  const frame = useCurrentFrame();
  const cardOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const captionOpacity = interpolate(frame, [90, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: cardOpacity,
          background: "#111827",
          border: "1px solid #374151",
          borderRadius: 14,
          padding: "28px 40px",
        }}
      >
        <div
          style={{
            fontFamily: "Inter, sans-serif",
            fontSize: 18,
            color: "#6b7280",
            textTransform: "uppercase",
            letterSpacing: 1.5,
            marginBottom: 18,
          }}
        >
          earlier — int8 layer profile
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 28, height: 160 }}>
          {BARS.map((bar, i) => {
            const start = 25 + i * 20;
            const h = interpolate(frame, [start, start + 30], [0, bar.height], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            return (
              <div key={bar.label} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{ width: 60, height: h, background: bar.color, borderRadius: "4px 4px 0 0" }} />
                <div style={{ marginTop: 10, fontFamily: "monospace", fontSize: 18, color: "#9ca3af" }}>
                  {bar.label}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ opacity: captionOpacity, marginTop: 20, fontFamily: "Inter, sans-serif", fontSize: 20, color: "#f97316" }}>
          biggest single layer — not a convolution
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "This came from something we noticed earlier. Remember, the biggest
single layer in the INT8 profile wasn't a convolution. It was an input reformat."

## Scene 140 — Chapter 15 (A-Roll, 360–570f / 7s)

```tsx
// scenes/Ch15Scene140.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch15Scene140: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="First A/B run: ~9–13% faster!" fromFrame={25} />
  </AbsoluteFill>
);
```
Narration: "First test looked really promising. **Nine to thirteen percent**
faster."

## Scene 141 — Chapter 15 (A-Roll, 570–780f / 7s)

```tsx
// scenes/Ch15Scene141.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch15Scene141: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="...from a single run." fromFrame={30} />
  </AbsoluteFill>
);
```
Narration: "One run, though. On a GPU. [PAUSE] That alone should already make you
a little nervous."

## Scene 142 — Chapter 15 (B-Roll, 780–990f / 7s)

The honest-reversal beat — slow reveal, held rather than rushed.

```tsx
// scenes/Ch15Scene142.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch15Scene142: React.FC = () => {
  const frame = useCurrentFrame();

  const labelOpacity = interpolate(frame, [0, 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  const bigOpacity = interpolate(frame, [25, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  // the reversal: the first number shrinks, dims, and gets struck through
  const shrinkScale = interpolate(frame, [95, 140], [1, 0.55], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const shrinkAmount = interpolate(frame, [95, 140], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const strikeWidth = interpolate(frame, [100, 135], [0, 100], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  const realOpacity = interpolate(frame, [150, 175], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const realTranslateY = interpolate(frame, [150, 175], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  // blend the confident green toward a discredited gray as it shrinks
  const r = 74 + (156 - 74) * shrinkAmount;
  const g = 222 + (163 - 222) * shrinkAmount;
  const b = 128 + (175 - 128) * shrinkAmount;

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          opacity: labelOpacity,
          position: "absolute",
          top: 220,
          fontFamily: "Inter, sans-serif",
          fontSize: 24,
          color: "#9ca3af",
          letterSpacing: 1,
        }}
      >
        3× repeated builds per config
      </div>

      <div style={{ position: "relative", opacity: bigOpacity, scale: shrinkScale }}>
        <div style={{ fontFamily: "monospace", fontSize: 88, fontWeight: 700, color: `rgb(${r}, ${g}, ${b})` }}>
          9–13%
        </div>
        <div
          style={{
            position: "absolute",
            left: "0%",
            top: "50%",
            width: `${strikeWidth}%`,
            height: 4,
            background: "#6b7280",
          }}
        />
      </div>

      <div
        style={{
          opacity: realOpacity,
          translate: `0px ${realTranslateY}px`,
          marginTop: 56,
          display: "flex",
          gap: 60,
          fontFamily: "monospace",
          fontSize: 32,
        }}
      >
        <div style={{ color: "#9ca3af" }}>
          FP16: <span style={{ color: "#f9fafb" }}>ties, ~0%</span>
        </div>
        <div style={{ color: "#9ca3af" }}>
          INT8: <span style={{ color: "#4ade80" }}>+2–4%</span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "So, repeat the build three times per config, and the win mostly
evaporates. [PAUSE] FP16 ties exactly. INT8 keeps a real, but small, two to four
percent."

## Scene 143 — Chapter 15 (B-Roll, 990–1200f / 7s)

```tsx
// scenes/Ch15Scene143.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CAUSES = [
  { title: "cold-start artifact", detail: "first build/run pays warmup costs the rest don't", start: 20 },
  { title: "TensorRT's own tactic noise", detail: "±8% between identical rebuilds — already bigger than the effect", start: 70 },
];

export const Ch15Scene143: React.FC = () => {
  const frame = useCurrentFrame();
  const noteOpacity = interpolate(frame, [150, 175], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 24, width: 1100 }}>
        {CAUSES.map((cause) => {
          const opacity = interpolate(frame, [cause.start, cause.start + 18], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          const translateX = interpolate(frame, [cause.start, cause.start + 18], [-16, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });
          return (
            <div
              key={cause.title}
              style={{
                opacity,
                translate: `${translateX}px 0px`,
                background: "#111827",
                border: "1px solid #374151",
                borderRadius: 12,
                padding: "20px 28px",
              }}
            >
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: 26, fontWeight: 700, color: "#f97316" }}>
                {cause.title}
              </div>
              <div style={{ fontFamily: "Inter, sans-serif", fontSize: 20, color: "#9ca3af", marginTop: 6 }}>
                {cause.detail}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ opacity: noteOpacity, marginTop: 36, fontFamily: "monospace", fontSize: 24, color: "#6b7280" }}>
        both bigger than the real effect
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Turns out that first number was mostly a cold-start artifact. And
it's smaller than the noise TensorRT's own tactic selection already produces
between identical rebuilds anyway."

## Scene 144 — Chapter 15 (A-Roll, 1200–1380f / 6s)

```tsx
// scenes/Ch15Scene144.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch15Scene144: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Kept — free, never measured worse." fromFrame={25} />
  </AbsoluteFill>
);
```
Narration: "It's still in the build scripts, for what it's worth. It's free,
never measured worse. Just not the big win it looked like at first."

## Scene 145 — Chapter 15 (A-Roll, 1380–1560f / 6s)

```tsx
// scenes/Ch15Scene145.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch15Scene145: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Same discipline as that calibration-cache decode earlier. A
good-looking first number still has to get double-checked."

## Scene 146 — Chapter 15 (A-Roll, 1560–1710f / 5s)

```tsx
// scenes/Ch15Scene146.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch15Scene146: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Okay, we've spent this whole video chasing speed. What did all of it
cost us?"

## Scene 147 — Chapter 15 (B-Roll, chapter card, 1710–1830f / 4s, no VO)

```tsx
// scenes/Ch15Scene147.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch15Scene147: React.FC = () => (
  <ChapterCard number={16} title="What accuracy actually costs." durationInFrames={120} />
);
```

## Chapter 15 assembly

```tsx
// chapters/Chapter15.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch15Scene138 } from "../scenes/Ch15Scene138";
import { Ch15Scene139 } from "../scenes/Ch15Scene139";
import { Ch15Scene140 } from "../scenes/Ch15Scene140";
import { Ch15Scene141 } from "../scenes/Ch15Scene141";
import { Ch15Scene142 } from "../scenes/Ch15Scene142";
import { Ch15Scene143 } from "../scenes/Ch15Scene143";
import { Ch15Scene144 } from "../scenes/Ch15Scene144";
import { Ch15Scene145 } from "../scenes/Ch15Scene145";
import { Ch15Scene146 } from "../scenes/Ch15Scene146";
import { Ch15Scene147 } from "../scenes/Ch15Scene147";

// Local cut points (frames, 30fps) for this chapter's own 0..1830 timeline.
const CUTS = [0, 180, 360, 570, 780, 990, 1200, 1380, 1560, 1710, 1830];
const SCENES = [
  Ch15Scene138,
  Ch15Scene139,
  Ch15Scene140,
  Ch15Scene141,
  Ch15Scene142,
  Ch15Scene143,
  Ch15Scene144,
  Ch15Scene145,
  Ch15Scene146,
  Ch15Scene147,
];

// Global offset: this chapter's slice of the narration file, 24990f–26820f.
const GLOBAL_START = 24990;
const GLOBAL_END = 26820;

export const Chapter15: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={GLOBAL_START}
      trimAfter={GLOBAL_END}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 16 — What Accuracy Actually Costs

Scenes 148–157 (10 scenes, 59s / 1770f at 30fps). Continues from Chapter 15 in the
master timeline at global offset 26820f (894s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 148 — Chapter 16 (B-Roll, 0–180f / 6s)

```tsx
// scenes/Ch16Scene148.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LINES = [
  { text: "ref  = torch_model(image)", color: "#9ca3af" },
  { text: "for engine in [fp32, fp16, int8]:", color: "#e5e7eb" },
  { text: "    out   = engine.infer(image)", color: "#e5e7eb" },
  { text: "    top1  = argmax(out) == argmax(ref)", color: "#60a5fa" },
  { text: "    diff  = abs(out - ref).max()", color: "#60a5fa" },
];

export const Ch16Scene148: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ color: "#6b7280", fontSize: 24, marginBottom: 28 }}>verify_accuracy_venv.py</div>
      {LINES.map((line, i) => {
        const start = 15 + i * 18;
        const opacity = interpolate(frame, [start, start + 14], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
          easing: Easing.bezier(0.16, 1, 0.3, 1),
        });
        const translateX = interpolate(frame, [start, start + 14], [-10, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        return (
          <div
            key={line.text}
            style={{ opacity, translate: `${translateX}px 0px`, color: line.color, fontSize: 28, padding: "5px 0" }}
          >
            {line.text}
          </div>
        );
      })}
    </AbsoluteFill>
  );
};
```
Narration: "Every engine's output gets compared directly against the original PyTorch
model, on real images."

## Scene 149 — Chapter 16 (B-Roll, 180–360f / 6s)

```tsx
// scenes/Ch16Scene149.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const ROWS = [
  { label: "Top-1 match", fp32: "1.0", fp16: "1.0" },
  { label: "Max abs diff", fp32: "0.028", fp16: "0.087" },
];

export const Ch16Scene149: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: "Inter, sans-serif", color: "#f9fafb" }}>
        <div style={{ display: "grid", gridTemplateColumns: "280px repeat(2, 200px)", fontSize: 28 }}>
          <div />
          {["FP32", "FP16"].map((h) => (
            <div key={h} style={{ fontWeight: 700, textAlign: "center", color: "#93c5fd" }}>{h}</div>
          ))}
          {ROWS.map((row, i) => {
            const start = 20 + i * 25;
            const opacity = interpolate(frame, [start, start + 15], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            return (
              <React.Fragment key={row.label}>
                <div style={{ opacity, padding: "10px 0" }}>{row.label}</div>
                <div style={{ opacity, textAlign: "center", color: "#4ade80" }}>{row.fp32}</div>
                <div style={{ opacity, textAlign: "center", color: "#4ade80" }}>{row.fp16}</div>
              </React.Fragment>
            );
          })}
        </div>
        <div
          style={{
            opacity: interpolate(frame, [90, 110], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            }),
            marginTop: 32,
            fontSize: 24,
            color: "#6b7280",
            textAlign: "center",
          }}
        >
          every image, both formats
        </div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "FP32 and FP16 both match the reference on every single image. No surprises
there."

## Scene 150 — Chapter 16 (B-Roll, 360–570f / 7s)

```tsx
// scenes/Ch16Scene150.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch16Scene150: React.FC = () => {
  const frame = useCurrentFrame();

  const tableOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const matchScale = interpolate(frame, [15, 30], [0.85, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const fp16Bar = interpolate(frame, [70, 100], [0, 32], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const int8Bar = interpolate(frame, [100, 140], [0, 880], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelOpacity = interpolate(frame, [130, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: tableOpacity, fontFamily: "Inter, sans-serif", color: "#f9fafb", textAlign: "center" }}>
        <div style={{ fontSize: 26, color: "#9ca3af" }}>INT8 top-1 match</div>
        <div style={{ scale: matchScale, fontFamily: "monospace", fontSize: 64, color: "#ef4444", marginTop: 6 }}>
          0.9375
        </div>
        <div style={{ fontSize: 22, color: "#6b7280", marginTop: 4 }}>30 / 32 — two images flip</div>
      </div>

      <div style={{ marginTop: 56, fontFamily: "monospace", fontSize: 22, color: "#9ca3af" }}>max abs diff</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 12, width: 900 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ width: 60, fontFamily: "monospace", fontSize: 20, color: "#93c5fd" }}>FP16</div>
          <div style={{ width: fp16Bar, height: 22, background: "#60a5fa", borderRadius: 4 }} />
          <div style={{ fontFamily: "monospace", fontSize: 20, color: "#93c5fd" }}>0.087</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ width: 60, fontFamily: "monospace", fontSize: 20, color: "#f87171" }}>INT8</div>
          <div style={{ width: int8Bar, height: 22, background: "#ef4444", borderRadius: 4 }} />
          <div style={{ fontFamily: "monospace", fontSize: 20, color: "#f87171" }}>2.396</div>
        </div>
      </div>
      <div style={{ opacity: labelOpacity, marginTop: 20, fontFamily: "Inter, sans-serif", fontSize: 22, color: "#6b7280" }}>
        27x bigger than FP16's worst error
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "INT8, though. Two out of thirty-two images flip. That's **about ninety-four
percent** still matching. And the worst single-value error is over twenty-seven times
bigger than FP16's."

## Scene 151 — Chapter 16 (A-Roll, 570–750f / 6s)

```tsx
// scenes/Ch16Scene151.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch16Scene151: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "That's roughly the expected cost of quantization. Not a red flag on its
own."

## Scene 152 — Chapter 16 (A-Roll, 750–960f / 7s)

```tsx
// scenes/Ch16Scene152.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch16Scene152: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Tested on calibration images. Not held-out." fromFrame={25} />
  </AbsoluteFill>
);
```
Narration: "But here's the part I don't want to gloss over: these thirty-two test
images were pulled from the same five hundred used for calibration. That's not really
a clean, held-out test."

## Scene 153 — Chapter 16 (B-Roll, 960–1140f / 6s)

```tsx
// scenes/Ch16Scene153.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch16Scene153: React.FC = () => {
  const frame = useCurrentFrame();
  const drift = interpolate(frame, [0, 50], [70, 14], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelOpacity = interpolate(frame, [55, 75], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 560, height: 360 }}>
        <div
          style={{
            position: "absolute",
            left: 280 - drift - 170,
            top: 40,
            width: 340,
            height: 340,
            borderRadius: "50%",
            border: "2px solid #60a5fa",
            background: "rgba(96,165,250,0.12)",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: 280 + drift - 170,
            top: 40,
            width: 340,
            height: 340,
            borderRadius: "50%",
            border: "2px solid #4ade80",
            background: "rgba(74,222,128,0.12)",
          }}
        />
      </div>
      <div style={{ opacity: labelOpacity, display: "flex", gap: 70, marginTop: 10, fontFamily: "Inter, sans-serif", fontSize: 22 }}>
        <span style={{ color: "#93c5fd" }}>calibration — 500 images</span>
        <span style={{ color: "#4ade80" }}>"test" — 32 images</span>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "A model's always going to look a little better on data it's already seen
something similar to."

## Scene 154 — Chapter 16 (B-Roll, 1140–1320f / 6s)

```tsx
// scenes/Ch16Scene154.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const LINE = "test images overlap with the calibration set — needs a real held-out split";

export const Ch16Scene154: React.FC = () => {
  const frame = useCurrentFrame();
  const typedChars = Math.floor(
    interpolate(frame, [20, 90], [0, LINE.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  );
  const headerOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const tagOpacity = interpolate(frame, [100, 120], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", padding: 100, fontFamily: "monospace" }}>
      <div style={{ opacity: headerOpacity, color: "#6b7280", fontSize: 24 }}>README.md</div>
      <div style={{ opacity: headerOpacity, color: "#9ca3af", fontSize: 28, marginTop: 24 }}>## Known limitations</div>
      <div style={{ color: "#e5e7eb", fontSize: 28, marginTop: 14 }}>
        - [ ] {LINE.slice(0, typedChars)}
        <span style={{ opacity: frame % 20 < 10 ? 1 : 0 }}>_</span>
      </div>
      <div
        style={{
          opacity: tagOpacity,
          marginTop: 36,
          fontFamily: "Inter, sans-serif",
          fontSize: 22,
          color: "#4ade80",
          textTransform: "uppercase",
          letterSpacing: 1,
        }}
      >
        flagged in the repo — not glossed over
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "And that's written down in the repo itself as open follow-up work. Not
buried in a footnote somewhere."

## Scene 155 — Chapter 16 (A-Roll, 1320–1500f / 6s)

```tsx
// scenes/Ch16Scene155.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch16Scene155: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="n=1 random noise ≠ real test." fromFrame={20} />
  </AbsoluteFill>
);
```
Narration: "Early on, a single batch of random noise got tried as a quick smoke test.
That's exactly the wrong input, since the calibration ranges are built entirely from
real image statistics."

## Scene 156 — Chapter 16 (A-Roll, 1500–1650f / 5s)

```tsx
// scenes/Ch16Scene156.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch16Scene156: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So, a real cost, measured honestly, and its limits stated out loud."

## Scene 157 — Chapter 16 (B-Roll, chapter card, 1650–1770f / 4s, no VO)

```tsx
// scenes/Ch16Scene157.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch16Scene157: React.FC = () => (
  <ChapterCard number={17} title="What's still open." durationInFrames={120} />
);
```

## Chapter 16 assembly

```tsx
// chapters/Chapter16.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch16Scene148 } from "../scenes/Ch16Scene148";
import { Ch16Scene149 } from "../scenes/Ch16Scene149";
import { Ch16Scene150 } from "../scenes/Ch16Scene150";
import { Ch16Scene151 } from "../scenes/Ch16Scene151";
import { Ch16Scene152 } from "../scenes/Ch16Scene152";
import { Ch16Scene153 } from "../scenes/Ch16Scene153";
import { Ch16Scene154 } from "../scenes/Ch16Scene154";
import { Ch16Scene155 } from "../scenes/Ch16Scene155";
import { Ch16Scene156 } from "../scenes/Ch16Scene156";
import { Ch16Scene157 } from "../scenes/Ch16Scene157";

// Local cut points (frames, 30fps) for this chapter's own 0..1770 timeline.
const CUTS = [0, 180, 360, 570, 750, 960, 1140, 1320, 1500, 1650, 1770];
const SCENES = [
  Ch16Scene148,
  Ch16Scene149,
  Ch16Scene150,
  Ch16Scene151,
  Ch16Scene152,
  Ch16Scene153,
  Ch16Scene154,
  Ch16Scene155,
  Ch16Scene156,
  Ch16Scene157,
];

// Global offsets, cumulative across the full 19-chapter video.
const CHAPTER_START_GLOBAL = 26820;
const CHAPTER_END_GLOBAL = 28590;

export const Chapter16: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_END_GLOBAL}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 17 — What's Still Open

Scenes 158–165 (8 scenes, 43s / 1290f at 30fps). Continues immediately after Chapter
16 in the master timeline at global offset 28590f (953s) and pulls that same slice of
`audio/narration_normal_speed.wav`.

## Scene 158 — Chapter 17 (A-Roll, 0–180f / 6s)

```tsx
// scenes/Ch17Scene158.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch17Scene158: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="batch=1, always." fromFrame={25} />
  </AbsoluteFill>
);
```
Narration: "One thing worth flagging: every single number in this video was measured
at batch size one."

## Scene 159 — Chapter 17 (B-Roll, 180–390f / 7s)

```tsx
// scenes/Ch17Scene159.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch17Scene159: React.FC = () => {
  const frame = useCurrentFrame();
  const outlineOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const smallTileScale = interpolate(frame, [15, 35], [0.7, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.34, 1.56, 0.64, 1),
  });
  const ghostOpacity = interpolate(frame, [90, 130], [0, 0.55], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const labelOpacity = interpolate(frame, [140, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ position: "relative", width: 400, height: 400 }}>
        <div
          style={{
            opacity: outlineOpacity,
            position: "absolute",
            inset: 0,
            border: "2px dashed #374151",
            borderRadius: 8,
          }}
        />
        <div
          style={{
            opacity: ghostOpacity,
            position: "absolute",
            inset: 6,
            border: "2px dashed #4ade80",
            background: "rgba(74,222,128,0.08)",
            borderRadius: 6,
          }}
        />
        <div
          style={{
            scale: smallTileScale,
            position: "absolute",
            left: 20,
            top: 20,
            width: 110,
            height: 110,
            background: "#60a5fa",
            borderRadius: 6,
          }}
        />
      </div>
      <div style={{ display: "flex", gap: 60, marginTop: 24, opacity: labelOpacity, fontFamily: "Inter, sans-serif", fontSize: 22 }}>
        <span style={{ color: "#93c5fd" }}>batch=1 kernel</span>
        <span style={{ color: "#4ade80" }}>bigger batch (predicted)</span>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "The whole explanation for that 1.7x shortfall was that batch-one kernels
are too small. So logically, a bigger batch should close some of that gap."

## Scene 160 — Chapter 17 (A-Roll, 390–540f / 5s)

```tsx
// scenes/Ch17Scene160.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";
import { LowerThird } from "../LowerThird";

export const Ch17Scene160: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
    <LowerThird text="Not run yet." fromFrame={15} />
  </AbsoluteFill>
);
```
Narration: "That's a prediction, though. Not a result. I haven't run it."

## Scene 161 — Chapter 17 (A-Roll, 540–720f / 6s)

```tsx
// scenes/Ch17Scene161.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch17Scene161: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "And I'm not saying that to leave you hanging for views. It's genuinely
just written down as the next step in this repo's own README."

## Scene 162 — Chapter 17 (B-Roll, 720–870f / 5s)

```tsx
// scenes/Ch17Scene162.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const COMMANDS = [
  "python calibrate_int8.py",
  "python build_and_bench.py",
  "python verify_accuracy_venv.py",
];

export const Ch17Scene162: React.FC = () => {
  const frame = useCurrentFrame();
  const scrollY = interpolate(frame, [0, 130], [0, -120], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", overflow: "hidden" }}>
      <div style={{ position: "absolute", left: 100, top: 100 + scrollY, fontFamily: "monospace" }}>
        <div style={{ color: "#6b7280", fontSize: 24 }}>README.md</div>
        <div style={{ color: "#9ca3af", fontSize: 30, marginTop: 40 }}>## Reproduce</div>
        {COMMANDS.map((cmd) => (
          <div key={cmd} style={{ color: "#4ade80", fontSize: 26, marginTop: 18 }}>
            $ {cmd}
          </div>
        ))}
        <div style={{ color: "#9ca3af", fontSize: 24, marginTop: 40 }}>## Known limitations</div>
        <div style={{ color: "#e5e7eb", fontSize: 24, marginTop: 14 }}>- [ ] try a bigger batch size</div>
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Every script that made every number in this video is sitting in this
repo, with the exact commands to reproduce it."

## Scene 163 — Chapter 17 (A-Roll, 870–1020f / 5s)

```tsx
// scenes/Ch17Scene163.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch17Scene163: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "So if you end up running the bigger-batch version yourself, I'd like to
know what you get."

## Scene 164 — Chapter 17 (A-Roll, 1020–1170f / 5s)

```tsx
// scenes/Ch17Scene164.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch17Scene164: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "Alright, let's put the whole chain back together one more time before we
wrap up."

## Scene 165 — Chapter 17 (B-Roll, chapter card, 1170–1290f / 4s, no VO)

```tsx
// scenes/Ch17Scene165.tsx
import { ChapterCard } from "../ChapterCard";

export const Ch17Scene165: React.FC = () => (
  <ChapterCard number={18} title="Close." durationInFrames={120} />
);
```

## Chapter 17 assembly

```tsx
// chapters/Chapter17.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch17Scene158 } from "../scenes/Ch17Scene158";
import { Ch17Scene159 } from "../scenes/Ch17Scene159";
import { Ch17Scene160 } from "../scenes/Ch17Scene160";
import { Ch17Scene161 } from "../scenes/Ch17Scene161";
import { Ch17Scene162 } from "../scenes/Ch17Scene162";
import { Ch17Scene163 } from "../scenes/Ch17Scene163";
import { Ch17Scene164 } from "../scenes/Ch17Scene164";
import { Ch17Scene165 } from "../scenes/Ch17Scene165";

// Local cut points (frames, 30fps) for this chapter's own 0..1290 timeline.
const CUTS = [0, 180, 390, 540, 720, 870, 1020, 1170, 1290];
const SCENES = [
  Ch17Scene158,
  Ch17Scene159,
  Ch17Scene160,
  Ch17Scene161,
  Ch17Scene162,
  Ch17Scene163,
  Ch17Scene164,
  Ch17Scene165,
];

// Global offsets, cumulative across the full 19-chapter video.
const CHAPTER_START_GLOBAL = 28590;
const CHAPTER_END_GLOBAL = 29880;

export const Chapter17: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_END_GLOBAL}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

# Chapter 18 — Close

Scenes 166–169 (4 scenes, 25s / 750f at 30fps). Continues immediately after Chapter
17 in the master timeline at global offset 29880f (996s) and pulls that same slice of
`audio/narration_normal_speed.wav`. This is the final chapter of the video — global
offset 30630f (1021s) is the last frame of the whole 19-chapter timeline.

## Scene 166 — Chapter 18 (B-Roll, 0–240f / 8s)

```tsx
// scenes/Ch18Scene166.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

const CHAIN = [
  { label: "RANGE", color: "#60a5fa", tag: "Ch.4" },
  { label: "SCALE", color: "#60a5fa", tag: "Ch.4" },
  { label: "INTEGER VALUE", color: "#4ade80", tag: "Ch.9" },
  { label: "DEQUANTIZE", color: "#4ade80", tag: "Ch.10" },
  { label: "ERROR", color: "#ef4444", tag: "Ch.16" },
];

export const Ch18Scene166: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        {CHAIN.map((node, i) => {
          const start = 15 + i * 22;
          const scale = interpolate(frame, [start, start + 14], [0.6, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.34, 1.56, 0.64, 1),
          });
          const opacity = interpolate(frame, [start, start + 14], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const tagStart = 130 + i * 16;
          const tagOpacity = interpolate(
            frame,
            [tagStart, tagStart + 10, tagStart + 24, tagStart + 34],
            [0, 1, 1, 0],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div key={node.label} style={{ display: "flex", alignItems: "center", gap: 20 }}>
              <div style={{ position: "relative", scale, opacity }}>
                <div
                  style={{
                    fontFamily: "Inter, sans-serif",
                    fontSize: 26,
                    fontWeight: 700,
                    color: node.color,
                    background: "rgba(17,24,39,0.85)",
                    border: `1px solid ${node.color}`,
                    borderRadius: 999,
                    padding: "14px 26px",
                    whiteSpace: "nowrap",
                  }}
                >
                  {node.label}
                </div>
                <div
                  style={{
                    opacity: tagOpacity,
                    position: "absolute",
                    left: "50%",
                    translate: "-50% 0px",
                    top: -36,
                    fontFamily: "monospace",
                    fontSize: 18,
                    color: "#6b7280",
                  }}
                >
                  {node.tag}
                </div>
              </div>
              {i < CHAIN.length - 1 && <span style={{ color: "#374151", fontSize: 26 }}>→</span>}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```
Narration: "Range. Scale. Integer value. Dequantize. Error. We didn't just draw this
chain out. We ran every link of it."

## Scene 167 — Chapter 18 (A-Roll, 240–420f / 6s)

```tsx
// scenes/Ch18Scene167.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

const ROWS = [
  { label: "Latency", from: "1.808 ms", to: "0.347 ms", tag: "5x faster" },
  { label: "Engine size", from: "107.94 MB", to: "25.26 MB", tag: "1/4 the size" },
  { label: "Top-1 accuracy", from: "1.0", to: "0.9375", tag: "~19/20 match" },
];

export const Ch18Scene167: React.FC = () => {
  const frame = useCurrentFrame();
  const cardOpacity = interpolate(frame, [10, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill>
      <CameraPlaceholder />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div
          style={{
            opacity: cardOpacity,
            background: "rgba(17,24,39,0.82)",
            border: "1px solid #374151",
            borderRadius: 12,
            padding: "32px 48px",
            fontFamily: "Inter, sans-serif",
          }}
        >
          {ROWS.map((row, i) => {
            const start = 40 + i * 25;
            const opacity = interpolate(frame, [start, start + 15], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            });
            return (
              <div key={row.label} style={{ opacity, display: "flex", alignItems: "center", gap: 28, padding: "10px 0" }}>
                <span style={{ color: "#9ca3af", fontSize: 22, width: 160 }}>{row.label}</span>
                <span style={{ fontFamily: "monospace", fontSize: 24, color: "#e5e7eb" }}>
                  {row.from} → {row.to}
                </span>
                <span style={{ color: "#4ade80", fontSize: 20, fontWeight: 700 }}>{row.tag}</span>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
```
Narration: "FP32 to INT8: **five times faster,** a quarter the size, still matching
the original about **nineteen times out of twenty.** [PAUSE] And now you know exactly
why every one of those numbers looks the way it does."

## Scene 168 — Chapter 18 (A-Roll, 420–600f / 6s)

```tsx
// scenes/Ch18Scene168.tsx
import { AbsoluteFill } from "remotion";
import { CameraPlaceholder } from "../CameraPlaceholder";

export const Ch18Scene168: React.FC = () => (
  <AbsoluteFill>
    <CameraPlaceholder />
  </AbsoluteFill>
);
```
Narration: "The batch-size question's still sitting there, unanswered. That's
probably where this picks back up next."

## Scene 169 — Chapter 18 (B-Roll, end card, 600–750f / 5s, no VO)

```tsx
// scenes/Ch18Scene169.tsx
import { AbsoluteFill, useCurrentFrame, interpolate, Easing } from "remotion";

export const Ch18Scene169: React.FC = () => {
  const frame = useCurrentFrame();
  const markOpacity = interpolate(frame, [0, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const markScale = interpolate(frame, [0, 40], [0.97, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const ruleWidth = interpolate(frame, [30, 60], [0, 120], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const repoOpacity = interpolate(frame, [55, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  // A slow, deliberate fade to black over the final ~1.3s — the video's actual
  // last beat, not an abrupt cutoff.
  const blackOut = interpolate(frame, [110, 150], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  return (
    <AbsoluteFill style={{ background: "#0b0f14", justifyContent: "center", alignItems: "center" }}>
      <div style={{ opacity: markOpacity, scale: markScale, textAlign: "center" }}>
        <div style={{ fontFamily: "Inter, sans-serif", fontSize: 40, fontWeight: 700, color: "#f9fafb", letterSpacing: 1 }}>
          forward_logic
        </div>
        <div style={{ width: ruleWidth, height: 2, background: "#4ade80", margin: "18px auto" }} />
        <div style={{ opacity: repoOpacity, fontFamily: "monospace", fontSize: 22, color: "#6b7280" }}>
          tensorrt-precision-lab
        </div>
      </div>
      <AbsoluteFill style={{ background: "#000000", opacity: blackOut }} />
    </AbsoluteFill>
  );
};
```
Narration: (none)

## Chapter 18 assembly

```tsx
// chapters/Chapter18.tsx
import { AbsoluteFill, Sequence } from "remotion";
import { Audio, staticFile } from "@remotion/media";
import { Ch18Scene166 } from "../scenes/Ch18Scene166";
import { Ch18Scene167 } from "../scenes/Ch18Scene167";
import { Ch18Scene168 } from "../scenes/Ch18Scene168";
import { Ch18Scene169 } from "../scenes/Ch18Scene169";

// Local cut points (frames, 30fps) for this chapter's own 0..750 timeline.
const CUTS = [0, 240, 420, 600, 750];
const SCENES = [Ch18Scene166, Ch18Scene167, Ch18Scene168, Ch18Scene169];

// Global offsets, cumulative across the full 19-chapter video. This is the final
// chapter — CHAPTER_END_GLOBAL (30630f) is the last frame of the whole video.
const CHAPTER_START_GLOBAL = 29880;
const CHAPTER_END_GLOBAL = 30630;

export const Chapter18: React.FC = () => (
  <AbsoluteFill>
    <Audio
      src={staticFile("narration_normal_speed.wav")}
      trimBefore={CHAPTER_START_GLOBAL}
      trimAfter={CHAPTER_END_GLOBAL}
    />
    {SCENES.map((Scene, i) => (
      <Sequence key={i} from={CUTS[i]} durationInFrames={CUTS[i + 1] - CUTS[i]}>
        <Scene />
      </Sequence>
    ))}
  </AbsoluteFill>
);
```

---

---

**All 18 chapters + close done. 169 scenes total, 30630 frames / 1021.0s (~17m1s) master timeline, audio track `narration_normal_speed.wav` (1255.6s) — narration runs longer than the coded scene timeline; real word-level alignment against final narration timestamps is still deferred to the assembly pass.**
