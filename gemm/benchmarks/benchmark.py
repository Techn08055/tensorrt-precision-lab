"""Benchmark sweep for gemm_naive across gemm/docs/NAIVE_KERNEL_SPEC.md's sizes.

Runs the compiled binary in its own random-data benchmark mode (no file I/O in
the loop -- that would measure disk speed, not the kernel) across square sizes,
each as its own process invocation with 2 warmup + 5 timed runs (matching the
spec), and writes gemm/results/benchmark.csv.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

SIZES = [256, 512, 1024, 2048, 4096]
OUT_RE = re.compile(
    r"M=(\d+) N=(\d+) K=(\d+) mean_ms=([\d.]+) min_ms=([\d.]+) gflops=([\d.]+)"
)


def run_size(binary, size, warmup, iters):
    result = subprocess.run(
        [binary, "--M", str(size), "--N", str(size), "--K", str(size),
         "--warmup", str(warmup), "--iters", str(iters)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  size={size}: binary exited {result.returncode}\n{result.stderr}")
        return None
    m = OUT_RE.search(result.stdout)
    if not m:
        print(f"  size={size}: couldn't parse output: {result.stdout!r}")
        return None
    M, N, K, mean_ms, min_ms, gflops = m.groups()
    return {
        "M": int(M), "N": int(N), "K": int(K),
        "mean_ms": float(mean_ms), "min_ms": float(min_ms), "gflops": float(gflops),
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark sweep for gemm_naive")
    ap.add_argument("--binary", default="../src/gemm_naive")
    ap.add_argument("--out", default="../results/benchmark.csv")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Binary not found: {args.binary}")
        sys.exit(1)

    rows = []
    for size in args.sizes:
        print(f"M=N=K={size} ...", end=" ", flush=True)
        row = run_size(args.binary, size, args.warmup, args.iters)
        if row is None:
            continue
        print(f"mean={row['mean_ms']:.4f}ms  {row['gflops']:.1f} GFLOPS")
        rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["M", "N", "K", "mean_ms", "min_ms", "gflops"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
