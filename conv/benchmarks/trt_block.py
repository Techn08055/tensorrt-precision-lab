"""TensorRT baseline for one whole ResNet50 bottleneck block, timed like the CUDA drivers.

Block (last stage, 7x7): x -> Conv_848 1x1 -> ReLU -> Conv_850 3x3 -> ReLU -> Conv_852 1x1 -> + x -> ReLU
Builds one FP16 engine from the real weights in ../data/Conv_8{48,50,52}/ (TensorRT is free to fuse
inside it), checks it against ONNX's output of node_relu_45, and times it with CUDA-graph replay.
Also writes the block input and ONNX's block output to ../data/block_848/ for the CUDA block driver.
Run: python trt_block.py
"""
import glob
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
DATA = os.path.join(HERE, "..", "data")
CONVS = ["Conv_848", "Conv_850", "Conv_852"]

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from calibrate_int8 import preprocess_image  # noqa: E402  same preprocessing as the per-layer export

# block input and ONNX's own block output, from the same real image as the per-layer data
model = onnx.load(os.path.join(ROOT, "models", "resnet50.onnx"))
for name in ("relu_42", "relu_45"):
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
image = sorted(glob.glob(os.path.join(ROOT, "calibration", "images", "**", "*.JPEG"), recursive=True))[0]
x, y_ref = sess.run(["relu_42", "relu_45"], {sess.get_inputs()[0].name: preprocess_image(image)[None]})
# same tensor as Conv_848's exported input, up to onnxruntime running a differently optimized graph
x_layer = np.fromfile(os.path.join(DATA, "Conv_848", "x.bin"), np.float16).reshape(x.shape[1:]).astype(np.float32)
x_diff = np.abs(x_layer - x[0]).max()
assert x_diff <= 1e-2 * max(np.abs(x).max(), 1.0), f"block input differs from Conv_848's input by {x_diff}"
os.makedirs(os.path.join(DATA, "block_848"), exist_ok=True)
x[0].astype(np.float16).tofile(os.path.join(DATA, "block_848", "x.bin"))
y_ref[0].astype(np.float32).tofile(os.path.join(DATA, "block_848", "y_ref.bin"))

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network(0)
inp = net.add_input("x", trt.float16, (1, 2048, 7, 7))
t = inp
for i, name in enumerate(CONVS):
    cin, h, cout, k, s, pad, ho, relu = map(int, open(os.path.join(DATA, name, "meta.txt")).read().split())
    w = np.fromfile(os.path.join(DATA, name, "w.bin"), np.float16).reshape(cout, cin, k, k).astype(np.float32)
    b = np.fromfile(os.path.join(DATA, name, "b.bin"), np.float32)
    conv = net.add_convolution_nd(t, cout, (k, k), trt.Weights(w), trt.Weights(b))
    conv.stride_nd, conv.padding_nd = (s, s), (pad, pad)
    t = conv.get_output(0)
    if i == len(CONVS) - 1:
        t = net.add_elementwise(t, inp, trt.ElementWiseOperation.SUM).get_output(0)
    t = net.add_activation(t, trt.ActivationType.RELU).get_output(0)
t.name = "y"
net.mark_output(t)
t.dtype = trt.float16
cfg = builder.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(net, cfg))
assert engine is not None, "engine build failed"
info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))

ctx = engine.create_execution_context()
dx = torch.from_numpy(x).half().cuda()
dy = torch.empty(1, 2048, 7, 7, device="cuda", dtype=torch.half)
ctx.set_tensor_address("x", dx.data_ptr())
ctx.set_tensor_address("y", dy.data_ptr())
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
err = np.abs(dy.float().cpu().numpy() - y_ref).max()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    for _ in range(50):
        ctx.execute_async_v3(stream.cuda_stream)
with torch.cuda.stream(stream):
    for _ in range(3):
        graph.replay()
    stream.synchronize()
    s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()  # nsys --capture-range=cudaProfilerApi: timed runs only
    s0.record(stream)
    for _ in range(5):
        graph.replay()
    s1.record(stream)
    s1.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
ms = s0.elapsed_time(s1) / 250

print(f"block_848 TensorRT {trt.__version__} FP16 engine: {ms:.4f} ms (graph replay)  "
      f"max|out - onnx| {err:.4g}  (max|onnx| {np.abs(y_ref).max():.2f})  [input vs Conv_848 export: {x_diff:.3g}]")
for layer in info["Layers"]:
    print(f"  layer: {layer.get('Name', '?')[:90]}")
