"""
understand_gemm.py -- run this to *see* what GEMM_MENTAL_MODEL.md describes,
instead of just reading it. Every number this prints is computed here, not
copied from the markdown doc -- if the doc and this script ever disagree,
trust this script and go fix the doc.

No GPU needed. No CUDA. Just Python + NumPy, on purpose: the point is to build
intuition for what GEMM *is* and why memory reuse matters, before any of that
gets buried in CUDA syntax. The actual GPU kernel is gemm/src/gemm_naive.cu --
this script is upstream of that, not a replacement for it.

Run: python understand_gemm.py
"""

import time
import numpy as np


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. The definition, spelled out with an actual loop -- not the @ operator.
#    This is deliberately the slowest possible way to write it, because the
#    goal here is to make C[i,j] = sum_k A[i,k]*B[k,j] undeniable, not fast.
# ---------------------------------------------------------------------------

def gemm_pure_python(A, B):
    M, K = len(A), len(A[0])
    K2, N = len(B), len(B[0])
    assert K == K2, f"inner dimensions must match: A is Mx{K}, B is {K2}xN"

    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc
    return C


def demo_definition():
    section("1. The definition (a real loop, not the @ operator)")
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]        # 2x3
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]   # 3x2
    C = gemm_pure_python(A, B)
    print("A (2x3):", A)
    print("B (3x2):", B)
    print("C = A@B (2x2):", C)

    C_ref = (np.array(A) @ np.array(B)).tolist()
    assert C == C_ref, f"triple-loop GEMM disagrees with NumPy: {C} vs {C_ref}"
    print("Matches NumPy's A @ B. Good -- the loop above IS what @ computes.")

    print("\nOne output element, by hand: C[0][0] should be")
    print("  A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]")
    print(f"  = {A[0][0]}*{B[0][0]} + {A[0][1]}*{B[1][0]} + {A[0][2]}*{B[2][0]}"
          f" = {A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]}")
    print(f"  C[0][0] from the loop = {C[0][0]}")


# ---------------------------------------------------------------------------
# 2. FLOPs, bytes, and arithmetic intensity -- the formulas from
#    GEMM_MENTAL_MODEL.md, computed for a real size instead of left symbolic.
# ---------------------------------------------------------------------------

def gemm_cost(M, N, K, dtype_bytes=4):
    flops = 2 * M * N * K
    bytes_A = M * K * dtype_bytes
    bytes_B = K * N * dtype_bytes
    bytes_C = M * N * dtype_bytes
    # Naive-reuse assumption from the mental model doc: every one of A's M*K
    # elements gets re-read N times (once per output column), every one of
    # B's K*N elements gets re-read M times (once per output row).
    naive_reads = M * N * K + M * N * K  # A-reads + B-reads, in elements
    naive_bytes_moved = naive_reads * dtype_bytes + bytes_C
    arithmetic_intensity = flops / naive_bytes_moved
    return {
        "flops": flops,
        "bytes_A": bytes_A, "bytes_B": bytes_B, "bytes_C": bytes_C,
        "naive_bytes_moved": naive_bytes_moved,
        "arithmetic_intensity": arithmetic_intensity,
    }


def demo_cost():
    section("2. Cost: FLOPs, bytes, arithmetic intensity")
    for size in [256, 1024, 4096]:
        c = gemm_cost(size, size, size)
        print(f"\nM=N=K={size}")
        print(f"  FLOPs:              {c['flops']:,}  ({c['flops']/1e9:.2f} GFLOP)")
        print(f"  A + B + C bytes:     {c['bytes_A']+c['bytes_B']+c['bytes_C']:,} "
              f"({(c['bytes_A']+c['bytes_B']+c['bytes_C'])/1e6:.1f} MB, if each were read once)")
        print(f"  Bytes actually moved (naive, no reuse): {c['naive_bytes_moved']:,} "
              f"({c['naive_bytes_moved']/1e9:.2f} GB)")
        print(f"  Arithmetic intensity: {c['arithmetic_intensity']:.3f} FLOP/byte")
    print("\nThis number doesn't change with size -- it's a property of the naive")
    print("access pattern, not of any particular M/N/K. That's why the CUDA")
    print("benchmark (gemm/results/benchmark.csv) is flat across every size:")
    print("the kernel is memory-bound at every size, not just the big ones.")


# ---------------------------------------------------------------------------
# 3. Reuse, counted rather than asserted: how many times does the naive
#    access pattern actually re-read each element of A and B?
# ---------------------------------------------------------------------------

def count_reads(M, N, K):
    reads_A = np.zeros((M, K), dtype=np.int64)
    reads_B = np.zeros((K, N), dtype=np.int64)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                reads_A[i, k] += 1  # A[i,k] read once per (i,j,k) triple
                reads_B[k, j] += 1  # B[k,j] read once per (i,j,k) triple
    return reads_A, reads_B


def demo_reuse():
    section("3. Reuse, counted (not just claimed)")
    M, N, K = 4, 5, 3
    reads_A, reads_B = count_reads(M, N, K)
    print(f"M={M}, N={N}, K={K}")
    print("\nHow many times each element of A gets read, in the naive access pattern:")
    print(reads_A)
    print(f"Every row of A is read exactly N={N} times (once per output column) --")
    print(f"check: every value above equals N? {bool(np.all(reads_A == N))}")

    print("\nHow many times each element of B gets read:")
    print(reads_B)
    print(f"Every column of B is read exactly M={M} times (once per output row) --")
    print(f"check: every value above equals M? {bool(np.all(reads_B == M))}")

    print("\nThis is the concrete version of the mental-model doc's reuse argument:")
    print("C[i,j] and C[i,j+1] both read all of row A[i,:] -- the counts above")
    print("show that isn't a one-off coincidence, it happens for every row/column,")
    print("exactly N and M times respectively. Tiling exists to make each of")
    print("those repeated reads come from fast shared memory instead of global")
    print("memory after the first one.")


# ---------------------------------------------------------------------------
# 4. Why any of this matters: pure-Python loops vs NumPy (BLAS), timed.
#    This is the Python-level preview of "naive CUDA vs cuBLAS" -- same shape
#    of question, answerable right now with no GPU.
# ---------------------------------------------------------------------------

def demo_speed_gap():
    section("4. Why this matters: naive loops vs BLAS, timed")
    size = 96  # pure-Python triple loop is O(size^3) Python-level ops -- keep it small
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, size=(size, size)).astype(np.float32)
    B = rng.uniform(-1, 1, size=(size, size)).astype(np.float32)

    t0 = time.perf_counter()
    C_naive = gemm_pure_python(A.tolist(), B.tolist())
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    C_blas = A @ B
    t_blas = time.perf_counter() - t0

    max_err = float(np.abs(np.array(C_naive) - C_blas).max())
    flops = 2 * size**3
    print(f"Size {size}x{size}x{size} ({flops/1e9:.3f} GFLOP of work)")
    print(f"  pure-Python triple loop: {t_naive*1000:.1f} ms  ({flops/t_naive/1e9:.4f} GFLOPS)")
    print(f"  NumPy (A @ B, BLAS):     {t_blas*1000:.4f} ms  ({flops/t_blas/1e9:.2f} GFLOPS)")
    print(f"  speedup: {t_naive/t_blas:,.0f}x")
    print(f"  max abs error vs BLAS: {max_err:.2e} (should be tiny -- same computation)")
    print("\nBLAS gets its speed from decades of the same optimizations this lab")
    print("is building up to on the GPU: cache/shared-memory tiling, register")
    print("blocking, vectorized loads. gemm/src/gemm_naive.cu is the CUDA")
    print("equivalent of the slow loop above -- cuBLAS is the equivalent of A @ B.")


if __name__ == "__main__":
    demo_definition()
    demo_cost()
    demo_reuse()
    demo_speed_gap()
    print("\nDone. Next: gemm/docs/NAIVE_KERNEL_SPEC.md and gemm/src/gemm_naive.cu")
    print("do the same triple loop, but on the GPU, one thread per (i,j).")
