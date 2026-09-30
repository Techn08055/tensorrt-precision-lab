"""
compare_python_vs_cuda.py -- three-way GEMM speed comparison at the same size:
pure-Python triple loop, NumPy (BLAS), and the compiled CUDA naive kernel
(gemm/src/gemm_naive).

Pure Python is O(size^3) at the Python-object level, so this intentionally runs
at a size small enough to finish in a few seconds -- comparing at the 4096 size
from gemm/benchmarks/benchmark.py would mean waiting for Python to do ~137 GFLOP
one multiply-add at a time. This answers a different question than that sweep:
not "how does the CUDA kernel scale", but "how much does even a naive GPU kernel
already buy you over the honest from-scratch baseline."
"""

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs"))
from understand_gemm import gemm_pure_python  # the exact loop from understand_gemm.py, not re-implemented here

OUT_RE = re.compile(r"mean_ms=([\d.]+) min_ms=([\d.]+) gflops=([\d.]+)")


def run_cuda(binary, size, warmup, iters):
    result = subprocess.run(
        [binary, "--M", str(size), "--N", str(size), "--K", str(size),
         "--warmup", str(warmup), "--iters", str(iters)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CUDA binary failed (exit {result.returncode}): {result.stderr}")
    m = OUT_RE.search(result.stdout)
    if not m:
        raise RuntimeError(f"couldn't parse CUDA output: {result.stdout!r}")
    mean_ms, _min_ms, gflops = (float(x) for x in m.groups())
    return mean_ms, gflops


def main():
    ap = argparse.ArgumentParser(description="Pure-Python vs NumPy(BLAS) vs CUDA naive GEMM, same size")
    ap.add_argument("--size", type=int, default=128,
                     help="M=N=K for the comparison (kept small -- pure Python is O(size^3))")
    ap.add_argument("--binary", default="../src/gemm_naive")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        print(f"CUDA binary not found at {args.binary} -- build it first:")
        print(f"  nvcc -O3 -arch=sm_89 ../src/gemm_naive.cu -o {args.binary}")
        sys.exit(1)

    size = args.size
    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, size=(size, size)).astype(np.float32)
    B = rng.uniform(-1, 1, size=(size, size)).astype(np.float32)
    flops = 2 * size ** 3

    print(f"M=N=K={size}  ({flops / 1e9:.4f} GFLOP of work)\n")

    # 1. Pure Python -- the honest from-scratch baseline
    t0 = time.perf_counter()
    C_py = gemm_pure_python(A.tolist(), B.tolist())
    t_py = time.perf_counter() - t0
    gflops_py = flops / t_py / 1e9

    # 2. NumPy / BLAS -- decades-optimized CPU reference
    t0 = time.perf_counter()
    C_blas = A @ B
    t_blas = time.perf_counter() - t0
    gflops_blas = flops / t_blas / 1e9

    # 3. CUDA naive kernel -- mean of `iters` runs after `warmup`, same
    #    discipline as the rest of this repo (see NAIVE_KERNEL_SPEC.md)
    t_cuda_ms, gflops_cuda = run_cuda(args.binary, size, args.warmup, args.iters)

    max_err = float(np.abs(np.array(C_py) - C_blas).max())

    rows = [
        ("pure Python (triple loop)", t_py * 1000, gflops_py),
        ("NumPy (BLAS, CPU)", t_blas * 1000, gflops_blas),
        ("CUDA naive (this lab)", t_cuda_ms, gflops_cuda),
    ]

    print(f"{'implementation':<28}{'time (ms)':>12}{'GFLOPS':>10}{'speedup vs pure Python':>24}")
    for name, ms, gf in rows:
        speedup = t_py * 1000 / ms
        print(f"{name:<28}{ms:>12.4f}{gf:>10.4f}{speedup:>23,.0f}x")

    print(f"\npure-Python vs BLAS max abs error: {max_err:.2e} (sanity check -- the CUDA")
    print("kernel has its own dedicated correctness test: gemm/tests/test_correctness.py)")


if __name__ == "__main__":
    main()
