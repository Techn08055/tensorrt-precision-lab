"""Run every exported ResNet50 conv through our kernel (../src/conv_gemm), im2col + cuBLAS, and a
one-layer TensorRT engine (trt_single_layer.py), and write one table: ../results/all_layers.csv.

Each layer runs REPEATS times per method and the fastest run is kept: the GPU's clock state changes
from one process to the next (~1.33x, see ../README.md), and the minimum is the fast-state number.
Run: python run_all_layers.py   (after export_conv_real.py all, and building ../src/conv_gemm)
"""
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DRIVER = os.path.join(HERE, "..", "src", "conv_gemm")
REPEATS = 2

layers = sorted((d for d in os.listdir(DATA) if os.path.exists(os.path.join(DATA, d, "meta.txt"))),
                key=lambda d: int(d.split("_")[1]))
rows = []
for layer in layers:
    best = None
    for _ in range(REPEATS):
        out = subprocess.run([DRIVER, os.path.join(DATA, layer)], capture_output=True, text=True).stdout
        res = dict(kv.split("=") for kv in out.strip().splitlines()[-1].split()[1:])
        if best is None or float(res["ours_ms"]) < float(best["ours_ms"]):
            best = res | {"cublas_ms": min(float(res["cublas_ms"]), float(best["cublas_ms"])) if best else res["cublas_ms"]}
        else:
            best["cublas_ms"] = min(float(best["cublas_ms"]), float(res["cublas_ms"]))
    trt = []
    for _ in range(REPEATS):
        out = subprocess.run([sys.executable, os.path.join(HERE, "trt_single_layer.py"), layer],
                             capture_output=True, text=True).stdout
        m = re.search(r"graph replay ([\d.]+) ms", out)
        e = re.search(r"max\|out - onnx\| ([\d.e-]+)", out)
        trt.append((float(m.group(1)), float(e.group(1))))
    trt_ms, trt_err = min(trt)
    row = {k: best[k] for k in ("layer", "cin", "h", "cout", "k", "s", "ho", "relu", "gflop", "stages", "splits", "implicit",
                                "ours_ms", "ours_err", "im2col_ms", "cublas_ms", "max_ref", "ok")}
    row |= {"trt_ms": f"{trt_ms:.5f}", "trt_err": f"{trt_err:.5g}",
            "trt_over_ours": f"{trt_ms / float(best['ours_ms']):.3f}"}
    rows.append(row)
    print(f"{layer}: {row['cin']}x{row['h']} -> {row['cout']}x{row['ho']} k{row['k']} s{row['s']}  "
          f"ours {float(row['ours_ms']):.4f} ms (split {row['splits']}, ok {row['ok']})  "
          f"cublas {float(row['cublas_ms']):.4f}  trt {trt_ms:.4f}  -> ours is {float(row['trt_over_ours']):.2f}x TRT speed",
          flush=True)

with open(os.path.join(HERE, "..", "results", "all_layers.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
tot = {k: sum(float(r[k]) for r in rows) for k in ("ours_ms", "cublas_ms", "trt_ms", "im2col_ms")}
print(f"\n{len(rows)} layers, all ok: {all(r['ok'] == '1' for r in rows)}")
print(f"sum of per-layer times: ours {tot['ours_ms']:.3f} ms (of which im2col {tot['im2col_ms']:.3f}), "
      f"im2col+cuBLAS {tot['cublas_ms']:.3f} ms, TensorRT one-layer engines {tot['trt_ms']:.3f} ms")
