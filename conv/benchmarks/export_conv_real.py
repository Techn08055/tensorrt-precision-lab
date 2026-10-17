"""Dump real tensors for one ResNet50 conv node from the ONNX model, for the drivers in ../src/.

Runs one real calibration image through onnxruntime (FP32, CPU) with the node's input and
output exposed as graph outputs, and writes to ../data/<node>/ (e.g. ../data/Conv_846/):
  x.bin      fp16  Cin x H x W        real input activation
  w.bin      fp16  Cout x (Cin*kh*kw) real weights (BN already folded in the export)
  b.bin      fp32  Cout               real bias
  y_ref.bin  fp32  Cout x Ho x Wo     ONNX's own FP32 output of the node: the reference
Run: python export_conv846_real.py [Conv_846|Conv_850|...] [image_path]
"""
import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from calibrate_int8 import preprocess_image  # same preprocessing as the engines were calibrated with

short = sys.argv[1] if len(sys.argv) > 1 else "Conv_846"
NODE = "node_" + short
OUT_DIR = os.path.join(ROOT, "conv", "data", short)

model = onnx.load(os.path.join(ROOT, "models", "resnet50.onnx"))
node = next(n for n in model.graph.node if n.name == NODE)
x_name, w_name, b_name = node.input
attr = {a.name: list(a.ints) for a in node.attribute if a.ints}
stride, pad = attr.get("strides", [1, 1])[0], attr.get("pads", [0, 0, 0, 0])[0]
init = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
w, b = init[w_name], init[b_name]

for name in (x_name, node.output[0]):
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

image = sys.argv[2] if len(sys.argv) > 2 else sorted(
    glob.glob(os.path.join(ROOT, "calibration", "images", "**", "*.JPEG"), recursive=True))[0]
x, y = sess.run([x_name, node.output[0]], {sess.get_inputs()[0].name: preprocess_image(image)[None]})

# self-check: the dumped tensors reproduce ONNX's output via im2col + GEMM in NumPy
co, ci, kh, kw = w.shape
_, _, ho, wo = y.shape
xp = np.pad(x[0], ((0, 0), (pad, pad), (pad, pad)))
cols = np.stack([xp[:, i:i + stride * ho:stride, j:j + stride * wo:stride]
                 for i in range(kh) for j in range(kw)], axis=1).reshape(ci * kh * kw, ho * wo)
y_np = w.reshape(co, -1) @ cols + b[:, None]
assert np.allclose(y_np, y.reshape(co, -1), atol=1e-3), np.abs(y_np - y.reshape(co, -1)).max()

os.makedirs(OUT_DIR, exist_ok=True)
x[0].astype(np.float16).tofile(os.path.join(OUT_DIR, "x.bin"))
w.reshape(co, -1).astype(np.float16).tofile(os.path.join(OUT_DIR, "w.bin"))
b.astype(np.float32).tofile(os.path.join(OUT_DIR, "b.bin"))
y[0].astype(np.float32).tofile(os.path.join(OUT_DIR, "y_ref.bin"))
print(f"{short}: x {x.shape[1:]} w {w.shape} stride {stride} pad {pad} -> y {y.shape[1:]}")
print(f"image {os.path.relpath(image, ROOT)}")
print(f"x range [{x.min():.3f}, {x.max():.3f}], zeros {np.mean(x == 0):.1%}")
print(f"y_ref range [{y.min():.3f}, {y.max():.3f}]  -> wrote {os.path.relpath(OUT_DIR, ROOT)}")
