"""TensorRT baseline for one ResNet50 conv, timed the same way as the CUDA drivers in ../src/.

Builds a one-layer FP16 engine from the real weights/bias in ../data/<layer>/ (conv + ReLU where
the full ResNet engine fuses one), checks it against ONNX's output, and times it two ways:
  back-to-back: mean of 50 execute_async_v3 calls between CUDA events (same as the C++ drivers,
                but each call also pays Python + TensorRT enqueue overhead on the CPU)
  graph:        the same 50 calls captured in one CUDA graph and replayed (no CPU overhead)
Run: python trt_single_layer.py Conv_850
"""
import json
import os
import sys

import numpy as np
import tensorrt as trt
import torch

layer = sys.argv[1] if len(sys.argv) > 1 else "Conv_850"
data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", layer)
# meta.txt from export_conv_real.py: Cin H Cout kernel stride pad Ho relu (relu fused in the full engine)
cin, h, cout, k, s, pad, ho, relu = map(int, open(os.path.join(data, "meta.txt")).read().split())
x = np.fromfile(os.path.join(data, "x.bin"), np.float16).reshape(1, cin, h, h)
w = np.fromfile(os.path.join(data, "w.bin"), np.float16).reshape(cout, cin, k, k).astype(np.float32)
b = np.fromfile(os.path.join(data, "b.bin"), np.float32)
y_ref = np.fromfile(os.path.join(data, "y_ref.bin"), np.float32).reshape(1, cout, ho, ho)
if relu:
    y_ref = np.maximum(y_ref, 0)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
inp = network.add_input("x", trt.float16, (1, cin, h, h))
conv = network.add_convolution_nd(inp, cout, (k, k), trt.Weights(w), trt.Weights(b))
conv.stride_nd, conv.padding_nd = (s, s), (pad, pad)
out = network.add_activation(conv.get_output(0), trt.ActivationType.RELU).get_output(0) if relu else conv.get_output(0)
out.name = "y"
network.mark_output(out)
out.dtype = trt.float16  # after mark_output, or TensorRT ignores it
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
engine = trt.Runtime(logger).deserialize_cuda_engine(builder.build_serialized_network(network, config))
assert engine is not None, "engine build failed"
assert engine.get_tensor_dtype("y") == trt.float16 and engine.get_tensor_dtype("x") == trt.float16
info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
kernels = [(l.get("Name", "?"), l.get("TacticName", "")) for l in info["Layers"]]

ctx = engine.create_execution_context()
dx = torch.from_numpy(x).cuda()
dy = torch.empty(1, cout, ho, ho, device="cuda", dtype=torch.half)
ctx.set_tensor_address("x", dx.data_ptr())
ctx.set_tensor_address("y", dy.data_ptr())
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
err = np.abs(dy.float().cpu().numpy() - y_ref).max()


def timed(fn, iters=50):
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
        stream.synchronize()
        s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record(stream)
        for _ in range(iters):
            fn()
        s1.record(stream)
    s1.synchronize()
    return s0.elapsed_time(s1) / iters


t_b2b = timed(lambda: ctx.execute_async_v3(stream.cuda_stream))
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    for _ in range(50):
        ctx.execute_async_v3(stream.cuda_stream)
t_graph = timed(graph.replay, iters=5) / 50

print(f"{layer} TensorRT {trt.__version__} FP16 one-layer engine ({'conv+relu' if relu else 'conv'}): "
      f"max|out - onnx| {err:.4g}")
print(f"  back-to-back {t_b2b:.4f} ms   graph replay {t_graph:.4f} ms")
for name, tactic in kernels:
    print(f"  layer: {name[:70]}  tactic: {tactic[:80]}")
