# Naive GEMM — Spec (design before code)

Scope: **only** the naive, one-thread-per-output-element kernel (Phase B/C/D from
the plan). Tiling, shared memory, FP16, and the cuBLAS comparison are explicitly
out of scope for this spec — separate specs when we get there, so this one stays
reviewable in one sitting.

## What gets built

```
gemm/
├── src/
│   ├── gemm_naive.cu      # the kernel + a CLI driver (build C = A*B once, print timing)
│   └── gemm_utils.cuh     # shared helpers: CUDA error-checking macro, host-side
│                           #   matrix alloc/fill/print, used by every kernel version
│                           #   we write later too (not just this one)
├── tests/
│   └── test_correctness.py   # calls the compiled binary, compares against NumPy
├── benchmarks/
│   └── benchmark.py          # runs the binary across sizes, writes results/benchmark.csv
└── results/
    └── benchmark.csv         # produced by benchmark.py, not hand-written
```

## The kernel itself

Precision: **FP32** only for this first version (matches the mental-model doc's
FLOP/byte math; FP16 is Phase G, later).

Signature:

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K);
```

Thread mapping, exactly as in `GEMM_MENTAL_MODEL.md`:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;  // i, 0..M-1
int col = blockIdx.x * blockDim.x + threadIdx.x;  // j, 0..N-1
if (row >= M || col >= N) return;                 // guard for non-divisible sizes

float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = acc;
```

Layout: both A and B are row-major, C is row-major. No transpose flags, no
strides — the simplest possible layout, matching what `A.numpy()` / `B.numpy()`
give you from PyTorch/NumPy by default, so the correctness check has no layout
surprises to debug.

Block size: start with `16x16` threads per block (256 threads/block, a
conventional starting point — not tuned, no claim it's optimal, just a sane
default for a *correctness* pass; block-size sweeps belong to the tiled version's
benchmarking, not this one). Grid size computed to cover `M x N` with that block
size, rounding up (`(M + 15) / 16`, `(N + 15) / 16`).

**Intentionally not done in this version** (each is a later phase, not an
oversight): no shared memory, no register blocking, no vectorized loads
(`float4`), no multiple-output-per-thread, no stream/graph overlap, no error
handling beyond a basic `cudaGetLastError()` check after launch. If it's not in
this list or the code block above, it's out of scope for `gemm_naive.cu`.

## Correctness methodology

Reference: NumPy `A @ B` in float32 (not float64 — comparing fp32 GPU output
against a float64 CPU reference would show a "gap" that's actually just precision
mismatch between the reference and the thing under test, not a real bug).

```
max_abs_error = max(|C_cuda - C_ref|)
max_rel_error = max(|C_cuda - C_ref| / (|C_ref| + eps))
```

Pass threshold: `max_abs_error < 1e-3` for sizes up to at least 2048 (fp32
accumulation error grows with K; 1e-3 is a starting threshold, not a law — if a
legitimate size fails it because of accumulation order rather than a real bug,
that's a finding to report, not a threshold to quietly loosen without saying so).

`tests/test_correctness.py` runs this at a small size (e.g. 64x64x64, fast, good
for catching a broken kernel immediately) and prints PASS/FAIL with both error
numbers — matching the plan's requirement that correctness is provable, not
asserted.

## Benchmark methodology

Square sizes only for this pass (`M = N = K`), matching the plan:
`256, 512, 1024, 2048, 4096`. (4096^3 fp32 GEMM is ~137 GFLOP of work and ~200MB
across A+B+C — worth confirming it fits in this GPU's 6GB before assuming it runs;
`benchmark.py` should fail loudly, not hang, if an allocation doesn't fit.)

Per size, per run:
- CUDA event timing around the kernel launch only (not allocation, not the H2D/D2H
  copies) — matches how latency was measured throughout the TensorRT side of this
  repo (`scripts/build_and_bench.py`'s `benchmark()`), for a consistent methodology
  across the whole lab.
- Report **mean of 5 runs after 2 warmup runs**, not a single measurement — this
  repo already learned the hard way (TensorRT build-to-build tactic noise,
  DIRECT_IO's overstated first measurement) that a single run on this GPU is not
  trustworthy. Same discipline applies here from the start instead of relearning it.
- Derived metric: `GFLOPS = (2*M*N*K) / (time_seconds * 1e9)`.

Output: `gemm/results/benchmark.csv` with columns
`M, N, K, mean_ms, min_ms, gflops`.

## What this spec deliberately leaves as an open question

Whether the naive kernel is memory-bound in *practice* on this GPU (not just in
the back-of-envelope math in the mental-model doc) is a profiling question, not
something this spec asserts as already proven — that's Phase E, using Nsight
Compute on the compiled binary, after this version exists and is benchmarked.

## Sign-off

If this matches what you had in mind, next step is writing `gemm_utils.cuh` and
`gemm_naive.cu` from this spec, compiling with `nvcc`, and running the
correctness test before touching benchmarks — same order as the plan's own
"naive → verify → benchmark" sequence.
