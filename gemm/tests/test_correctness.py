"""Correctness check for gemm_naive: compiled binary's C = A @ B vs a NumPy fp32
reference, at a size small enough to run in well under a second.

Per gemm/docs/NAIVE_KERNEL_SPEC.md: reference is computed in float32, not
float64 -- comparing fp32 GPU output against a float64 reference would show a
"gap" that's a precision mismatch between the two references, not a bug in the
kernel under test.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

ABS_TOL = 1e-3


def run_case(binary, M, N, K, tmp_dir, seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)

    a_path = os.path.join(tmp_dir, "a.bin")
    b_path = os.path.join(tmp_dir, "b.bin")
    c_path = os.path.join(tmp_dir, "c.bin")
    A.tofile(a_path)
    B.tofile(b_path)

    result = subprocess.run(
        [binary, "--M", str(M), "--N", str(N), "--K", str(K),
         "--a-file", a_path, "--b-file", b_path, "--c-out", c_path,
         "--warmup", "1", "--iters", "1"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  binary exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}")
        return False

    C_cuda = np.fromfile(c_path, dtype=np.float32).reshape(M, N)
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)

    abs_err = np.abs(C_cuda - C_ref)
    rel_err = abs_err / (np.abs(C_ref) + 1e-8)
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    passed = max_abs < ABS_TOL

    status = "PASS" if passed else "FAIL"
    print(f"  M={M} N={N} K={K}  max_abs_error={max_abs:.6g}  max_rel_error={max_rel:.6g}  {status}")
    return passed


def main():
    ap = argparse.ArgumentParser(description="Correctness check for gemm_naive")
    ap.add_argument("--binary", default="../src/gemm_naive")
    ap.add_argument("--tmp-dir", default="/tmp")
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Binary not found: {args.binary} (build it first: nvcc -O3 -arch=sm_89 ../src/gemm_naive.cu -o {args.binary})")
        sys.exit(1)

    cases = [
        (64, 64, 64, 0),      # square, small -- the primary fast check
        (32, 64, 96, 1),      # non-square, exercises row/col guard paths differently
        (65, 65, 65, 2),      # not a multiple of the 16x16 block size -- boundary guard test
    ]

    print("Correctness\n------------")
    all_passed = True
    for M, N, K, seed in cases:
        ok = run_case(args.binary, M, N, K, args.tmp_dir, seed)
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("ALL PASS")
    else:
        print("FAILED -- see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
