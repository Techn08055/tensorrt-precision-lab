# GEMM Mental Model

One page. This is what has to be true in your head before writing a single line of
CUDA — if any of this is fuzzy, the naive kernel design that follows won't make sense.

Everything below is runnable, not just asserted: `python understand_gemm.py` (needs
only NumPy) prints the triple-loop definition checked against `A @ B`, the FLOPs/bytes/
arithmetic-intensity numbers computed for real sizes, the reuse counts proven by
actually counting reads instead of just claiming them, and a timed pure-Python-vs-BLAS
comparison that previews why the CUDA-vs-cuBLAS comparison later in this lab matters.

## The operation

```
C = A x B
A: [M, K]
B: [K, N]
C: [M, N]
```

Every output element is a dot product of one row of A and one column of B:

```
C[i,j] = sum over k=0..K-1 of A[i,k] * B[k,j]
```

M x N output elements, each one a K-length dot product.

## Cost, in numbers

- **FLOPs**: each output element does K multiplies + K adds = 2K FLOPs. Total:
  `2 * M * N * K` FLOPs for the whole matrix.
- **Memory read, naive**: if every thread independently re-reads the row/column it
  needs from global memory with no reuse, that's `M*N*K` reads of A elements and
  `M*N*K` reads of B elements (each of A's M*K elements gets re-read N times; each
  of B's K*N elements gets re-read M times). This is the number that tiling exists
  to shrink — see "why this matters" below.
- **Memory required**: `A` is `M*K*4` bytes (fp32), `B` is `K*N*4`, `C` is `M*N*4`.
- **Arithmetic intensity** (FLOPs per byte moved) at the naive-reuse level:

  ```
  AI = 2*M*N*K / ((M*N*K + M*N*K) * 4)  =  2*M*N*K / (8*M*N*K)  =  0.25 FLOP/byte
  ```

  That's a rough floor, not the real naive-kernel number (see below) — it assumes
  every read comes from global memory with zero reuse. This GPU's fp32 peak is
  roughly 2560 CUDA cores x ~2 GHz x 2 FLOP/cycle (~10 TFLOP/s class), against
  global memory bandwidth in the ~200-250 GB/s range for a laptop 4050. At
  0.25 FLOP/byte, the naive kernel is nowhere near compute-bound — it's
  memory-bound, and badly so. That gap is the entire reason tiling exists, and is
  the thing Phase E (profiling) should show directly rather than just asserting.

## Why GPU parallelism applies at all

Every output element `C[i,j]` is independent of every other output element. No
output depends on another output being computed first. That's what makes this
"embarrassingly parallel" — M*N independent dot products, each one internally
sequential (K steps that must accumulate in order... actually don't even need to be
in order, floating-point addition isn't associative in the bit-exact sense, but
summing in any order is a valid GEMM implementation; only the K reduction itself is
serial *per output element*, not across elements).

## Why memory reuse matters (the whole story ahead of time)

Look at two output elements in the same row: `C[i,j]` and `C[i,j+1]`. Both need the
*entire* row `A[i,:]` — the same K values of A, read twice (once per output
element) if nothing is cached. Scale that up: every element in row `i` of C reads
the same K values of `A[i,:]`, and every element in column `j` of C reads the same
K values of `B[:,j]`. A naive kernel where every thread independently re-fetches
from global memory throws this reuse away entirely. Tiling (Phase F) is the fix:
load a block of A and a block of B into fast on-chip shared memory once, and let
every thread in that block reuse it many times before moving to the next tile.

This is the one sentence that matters for the whole rest of the project: **the
naive kernel is correct but wastes memory bandwidth on data every neighboring
thread already fetched.** Everything from tiling onward is exploiting that reuse.

## CUDA thread mapping (for the naive kernel specifically)

```
one CUDA thread  ->  one output element C[i,j]
thread loops over k = 0..K-1, accumulates into a local (register) sum
writes the final sum once to C[i,j] in global memory
```

Grid/block shape: a 2D grid of 2D thread blocks, one thread per (i,j) pair,
`i` and `j` derived from `blockIdx`/`threadIdx` in the usual
`row = blockIdx.y * blockDim.y + threadIdx.y`,
`col = blockIdx.x * blockDim.x + threadIdx.x` pattern.

Per-thread global memory traffic in this design: K reads from A's row `i` (all
threads in the same block-row re-read the *same* K values independently) + K reads
from B's column `j` (all threads in the same block-column re-read the *same* K
values independently) + 1 write to C. Nothing is shared between threads. That's
the specific inefficiency Phase E's profiler numbers should quantify, and Phase F's
shared-memory tiling should fix.

## What "done" looks like for this doc

Anyone should be able to read this page and answer, without looking anything up:
what M/N/K mean, why the naive kernel is memory-bound, what one thread computes,
and why tiling is the next step — before seeing any code.
