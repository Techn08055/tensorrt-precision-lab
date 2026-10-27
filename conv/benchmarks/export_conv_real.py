"""Dump real tensors for ResNet50 conv nodes from the ONNX model, for the drivers in ../src/.

Runs one real calibration image through onnxruntime (FP32, CPU) with each node's input and
output exposed as graph outputs, and writes to ../data/<node>/ (e.g. ../data/Conv_846/):
  x.bin      fp16  Cin x H x W        real input activation
  w.bin      fp16  Cout x (Cin*kh*kw) real weights (BN already folded in the export)
  b.bin      fp32  Cout               real bias
  y_ref.bin  fp32  Cout x Ho x Wo     ONNX's own FP32 output of the node: the reference
  meta.txt   "cin h cout k stride pad ho relu"   relu = 1 if the node's output feeds a Relu
Run: python export_conv_real.py all                 every conv in the model (53)
     python export_conv_real.py Conv_846 Conv_850   just these
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

model = onnx.load(os.path.join(ROOT, "models", "resnet50.onnx"))
convs = [n for n in model.graph.node if n.op_type == "Conv"]
wanted = sys.argv[1:] or ["Conv_846"]
if wanted != ["all"]:
    convs = [n for n in convs if n.name[len("node_"):] in wanted]
    assert len(convs) == len(wanted), f"unknown node in {wanted}"
init = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
consumers = {}
for n in model.graph.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n.op_type)
graph_inputs = {i.name for i in model.graph.input}

names = sorted({t for n in convs for t in (n.input[0], n.output[0])} - graph_inputs)
for name in names:
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
image_path = sorted(glob.glob(os.path.join(ROOT, "calibration", "images", "**", "*.JPEG"), recursive=True))[0]
image = preprocess_image(image_path)[None]
acts = dict(zip(names, sess.run(names, {sess.get_inputs()[0].name: image})))
acts.update({g: image for g in graph_inputs})

for node in convs:
    short = node.name[len("node_"):]
    x, y = acts[node.input[0]], acts[node.output[0]]
    w, b = init[node.input[1]], init[node.input[2]]
    attr = {a.name: (list(a.ints) or a.i) for a in node.attribute}
    stride, pad = attr["strides"][0], attr["pads"][0]
    assert attr.get("group", 1) in (1, []) and set(attr["dilations"]) == {1}, short
    assert len(set(attr["strides"])) == 1 and len(set(attr["pads"])) == 1 and x.shape[2] == x.shape[3], short
    co, ci, k, _ = w.shape
    ho = y.shape[2]

    # self-check: the dumped tensors reproduce ONNX's output via im2col + GEMM in NumPy
    xp = np.pad(x[0], ((0, 0), (pad, pad), (pad, pad)))
    cols = np.stack([xp[:, i:i + stride * ho:stride, j:j + stride * ho:stride]
                     for i in range(k) for j in range(k)], axis=1).reshape(ci * k * k, ho * ho)
    y_np = w.reshape(co, -1) @ cols + b[:, None]
    assert np.allclose(y_np, y.reshape(co, -1), atol=2e-3), (short, np.abs(y_np - y.reshape(co, -1)).max())

    out = os.path.join(ROOT, "conv", "data", short)
    os.makedirs(out, exist_ok=True)
    x[0].astype(np.float16).tofile(os.path.join(out, "x.bin"))
    w.reshape(co, -1).astype(np.float16).tofile(os.path.join(out, "w.bin"))
    b.astype(np.float32).tofile(os.path.join(out, "b.bin"))
    y[0].astype(np.float32).tofile(os.path.join(out, "y_ref.bin"))
    relu = int(consumers.get(node.output[0], []) == ["Relu"])
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"{ci} {x.shape[2]} {co} {k} {stride} {pad} {ho} {relu}\n")
    print(f"{short}: {ci}x{x.shape[2]}x{x.shape[2]} -> {co}x{ho}x{ho}  k{k} s{stride} p{pad} relu{relu}")
print(f"image {os.path.relpath(image_path, ROOT)}; wrote {len(convs)} layers to conv/data/")
