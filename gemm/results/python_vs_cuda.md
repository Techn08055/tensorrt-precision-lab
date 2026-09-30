# Pure Python vs NumPy (BLAS) vs CUDA naive — same size, same data

Produced by [`../benchmarks/compare_python_vs_cuda.py`](../benchmarks/compare_python_vs_cuda.py).
Pure Python is O(size³) at the Python-object level, so this stays at sizes small
enough to finish in seconds — a different question than the
[`benchmark.py`](../benchmarks/benchmark.py) sweep (256→4096), which asks how the
CUDA kernel scales, not how it compares to a from-scratch baseline.

| Size | Implementation | Time | GFLOPS | Speedup vs pure Python |
|---|---|---:|---:|---:|
| 128³ | pure Python (triple loop) | 166.6 ms | 0.025 | 1x |
| 128³ | NumPy (BLAS, CPU) | 0.433 ms | 9.68 | 384x |
| 128³ | CUDA naive | 0.0132 ms | 318.6 | 12,657x |
| 256³ | pure Python (triple loop) | 1011.9 ms | 0.033 | 1x |
| 256³ | NumPy (BLAS, CPU) | 0.461 ms | 72.77 | 2,195x |
| 256³ | CUDA naive | 0.0587 ms | 571.2 | 17,224x |

Correctness cross-check (pure Python vs BLAS, not the CUDA kernel's own test):
max abs error ~1e-5 at both sizes — same computation, fp32 rounding only.

## Two things worth noting, not just the headline speedup

**The CUDA naive kernel's GFLOPS itself roughly doubles from 128³ to 256³** (318.6
→ 571.2), even though it's the *same* untiled, unoptimized kernel. At 128×128
output there are only 16,384 threads total; at 256×256 there are 65,536 — more
threads to hide per-thread memory latency and fixed launch/scheduling overhead
behind. 571 GFLOPS at 256³ matches the dedicated sweep's 256³ result (572.992,
[`benchmark.csv`](benchmark.csv)) almost exactly, confirming both scripts measure
the same thing consistently. This is the same "too small to reach peak throughput
at small problem sizes" pattern this lab already found on the TensorRT/INT8 side
(`profiling/PROFILE_RESULTS.md`) — showing up again here, at the kernel level
instead of the whole-model level.

**BLAS's speedup over pure Python (384x–2,195x) already makes the point this
whole lab is circling back to** before a GPU is even involved: same computation,
radically different implementation, radically different speed. The CUDA-vs-BLAS
and (later) CUDA-naive-vs-cuBLAS comparisons are the same question asked again,
one level down.
