"""Dump real Conv_846 tensors from the ResNet50 ONNX model for ../src/conv846_gemm.cu.

Runs one real calibration image through onnxruntime (FP32, CPU) with Conv_846's input and
output exposed as graph outputs, and writes to ../data/:
  x.bin      fp16  1024x14x14  real input activation
  w.bin      fp16  2048x1024   real weights (BN already folded in the export)
  b.bin      fp32  2048        real bias
  y_ref.bin  fp32  2048x7x7    ONNX's own FP32 output of the node: the reference
Run: python export_conv846_real.py [image_path]
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

NODE = "node_Conv_846"
OUT_DIR = os.path.join(ROOT, "conv", "data")

model = onnx.load(os.path.join(ROOT, "models", "resnet50.onnx"))
node = next(n for n in model.graph.node if n.name == NODE)
x_name, w_name, b_name = node.input
init = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
w, b = init[w_name], init[b_name]
assert w.shape == (2048, 1024, 1, 1) and b.shape == (2048,), (w.shape, b.shape)

for name in (x_name, node.output[0]):
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

image = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob(os.path.join(ROOT, "calibration", "images", "**", "*.JPEG"), recursive=True))[0]
x, y = sess.run([x_name, node.output[0]], {sess.get_inputs()[0].name: preprocess_image(image)[None]})
assert x.shape == (1, 1024, 14, 14) and y.shape == (1, 2048, 7, 7), (x.shape, y.shape)

# self-check: the dumped weights reproduce ONNX's output (1x1 stride-2 conv = GEMM on every other pixel)
y_np = w.reshape(2048, 1024) @ x[0, :, ::2, ::2].reshape(1024, 49) + b[:, None]
assert np.allclose(y_np, y.reshape(2048, 49), atol=1e-3), np.abs(y_np - y.reshape(2048, 49)).max()

os.makedirs(OUT_DIR, exist_ok=True)
x[0].astype(np.float16).tofile(os.path.join(OUT_DIR, "x.bin"))
w.reshape(2048, 1024).astype(np.float16).tofile(os.path.join(OUT_DIR, "w.bin"))
b.astype(np.float32).tofile(os.path.join(OUT_DIR, "b.bin"))
y[0].astype(np.float32).tofile(os.path.join(OUT_DIR, "y_ref.bin"))
print(f"image {os.path.relpath(image, ROOT)}")
print(f"x range [{x.min():.3f}, {x.max():.3f}], zeros {np.mean(x == 0):.1%} (post-ReLU)")
print(f"w range [{w.min():.4f}, {w.max():.4f}]  b range [{b.min():.3f}, {b.max():.3f}]")
print(f"y_ref range [{y.min():.3f}, {y.max():.3f}]  -> wrote {OUT_DIR}")
