import argparse
import glob
import json
import os

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from PIL import Image

INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Layer types dominated by tensor-core / FMA math throughput.
COMPUTE_BOUND_TYPES = {
    "CaskConvolution", "CaskConvActPool", "CaskGemmConvolution",
    "Convolution", "MatrixMultiply", "CaskGemm",
}
# Layer types dominated by moving bytes rather than doing math on them.
MEMORY_BOUND_TYPES = {
    "Reformat", "Shuffle", "ElementWise", "Pooling", "PointWise",
    "PointWiseV2", "Scale", "Softmax", "TopK", "Slice", "Concatenation",
    "NoOp", "Cast",
}


def check(err):
    if isinstance(err, cudart.cudaError_t) and err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return err


def classify(layer_type):
    if layer_type in COMPUTE_BOUND_TYPES:
        return "compute-bound"
    if layer_type in MEMORY_BOUND_TYPES:
        return "memory-bound"
    return "other"


def preprocess_image(path, size=224):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)  # CHW


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Reuses the calibration.cache written by calibrate_int8.py — no recalibration pass needed."""

    def __init__(self, image_dir, cache_file, batch_size=8):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.files = sorted(glob.glob(os.path.join(image_dir, "**", "*.JPEG"), recursive=True))
        self.current_index = 0
        nbytes = batch_size * int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
        err, self.device_input = cudart.cudaMalloc(nbytes)
        check(err)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.files):
            return None
        batch_files = self.files[self.current_index : self.current_index + self.batch_size]
        batch = np.ascontiguousarray(np.stack([preprocess_image(f) for f in batch_files]))
        check(cudart.cudaMemcpy(self.device_input, batch.ctypes.data, batch.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0])
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        pass  # cache already exists from calibrate_int8.py; nothing new to persist

    def __del__(self):
        if hasattr(self, "device_input"):
            cudart.cudaFree(self.device_input)


class LayerProfiler(trt.IProfiler):
    """Accumulates per-layer GPU time across iterations via TensorRT's built-in layer timer."""

    def __init__(self):
        super().__init__()
        self.total_ms = {}

    def report_layer_time(self, layer_name, ms):
        self.total_ms[layer_name] = self.total_ms.get(layer_name, 0.0) + ms


def build_engine(onnx_path, precision, batch, workspace_mb, calib_images, calib_cache):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, (batch, *INPUT_SHAPE), (batch, *INPUT_SHAPE), (batch, *INPUT_SHAPE))
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allows fallback for layers without an INT8 kernel
        calibrator = ImageCalibrator(calib_images, calib_cache)
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)

    print(f"Building {precision} engine with DETAILED profiling verbosity...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Engine build failed for {precision}")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)


def profile_engine(engine, batch, warmup, iters):
    context = engine.create_execution_context()
    context.set_input_shape(INPUT_NAME, (batch, *INPUT_SHAPE))
    layer_profiler = LayerProfiler()
    context.profiler = layer_profiler

    in_nbytes = batch * int(np.prod(INPUT_SHAPE)) * 4
    out_shape = tuple(context.get_tensor_shape(OUTPUT_NAME))
    out_nbytes = int(np.prod(out_shape)) * 4

    err, d_in = cudart.cudaMalloc(in_nbytes)
    check(err)
    err, d_out = cudart.cudaMalloc(out_nbytes)
    check(err)
    context.set_tensor_address(INPUT_NAME, int(d_in))
    context.set_tensor_address(OUTPUT_NAME, int(d_out))

    host_in = np.random.standard_normal((batch, *INPUT_SHAPE)).astype(np.float32)
    check(cudart.cudaMemcpy(d_in, host_in.ctypes.data, in_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0])

    err, stream = cudart.cudaStreamCreate()
    check(err)

    for _ in range(warmup):
        context.execute_async_v3(stream)
    check(cudart.cudaStreamSynchronize(stream)[0])

    # Layer-time callbacks require a sync per iteration, so this is slower than
    # the free-running benchmark in build_and_bench.py -- expected for profiling.
    for _ in range(iters):
        context.execute_async_v3(stream)
        check(cudart.cudaStreamSynchronize(stream)[0])

    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    cudart.cudaStreamDestroy(stream)

    inspector = engine.create_engine_inspector()
    layers = []
    for i in range(engine.num_layers):
        info = json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        name = info.get("Name", f"layer_{i}")
        outputs = info.get("Outputs") or [{}]
        layer_type = info.get("LayerType", "Unknown")
        layers.append({
            "index": i,
            "name": name,
            "type": layer_type,
            "output_dtype": outputs[0].get("Format/Datatype", "Unknown"),
            "tactic": info.get("TacticName"),
            "avg_ms": round(layer_profiler.total_ms.get(name, 0.0) / iters, 5),
            "category": classify(layer_type),
        })

    total_ms = sum(l["avg_ms"] for l in layers)
    for l in layers:
        l["pct_of_total"] = round(100 * l["avg_ms"] / total_ms, 2) if total_ms else 0.0
    layers.sort(key=lambda l: -l["avg_ms"])

    return layers, total_ms


def summarize(precision, layers, total_ms):
    by_category_ms = {}
    for l in layers:
        by_category_ms[l["category"]] = by_category_ms.get(l["category"], 0.0) + l["avg_ms"]

    non_native_dtype = {"fp32": "Float", "fp16": "Half", "int8": "Int8"}[precision]
    fallback_layers = [
        l for l in layers
        if l["output_dtype"] not in (non_native_dtype, "Unknown") and l["category"] != "other"
    ]

    return {
        "precision": precision,
        "num_layers": len(layers),
        "total_avg_ms": round(total_ms, 5),
        "time_by_category_ms": {k: round(v, 5) for k, v in by_category_ms.items()},
        "time_by_category_pct": {k: round(100 * v / total_ms, 2) for k, v in by_category_ms.items()} if total_ms else {},
        "fallback_layer_count": len(fallback_layers),
        "fallback_layers": [{"name": l["name"], "dtype": l["output_dtype"], "avg_ms": l["avg_ms"]} for l in fallback_layers],
        "top_layers": layers[:15],
    }


def main():
    ap = argparse.ArgumentParser(description="Per-layer TensorRT timing + precision breakdown for fp32/fp16/int8")
    ap.add_argument("--onnx", default="../models/resnet50.onnx")
    ap.add_argument("--calib-images", default="../calibration/images")
    ap.add_argument("--calib-cache", default="../calibration/calibration.cache")
    ap.add_argument("--out-dir", default="../profiling")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for precision in args.precisions:
        engine = build_engine(args.onnx, precision, args.batch, args.workspace_mb, args.calib_images, args.calib_cache)
        print(f"Profiling {precision} ({args.warmup} warmup + {args.iters} timed iterations)...")
        layers, total_ms = profile_engine(engine, args.batch, args.warmup, args.iters)
        summary = summarize(precision, layers, total_ms)

        out_path = os.path.join(args.out_dir, f"layer_profile_{precision}.json")
        with open(out_path, "w") as f:
            json.dump({"batch": args.batch, "iters": args.iters, **summary, "layers": layers}, f, indent=2)
        print(f"  total per-iter (profiler sum): {total_ms:.4f} ms")
        print(f"  by category: {summary['time_by_category_pct']}")
        if summary["fallback_layer_count"]:
            print(f"  {summary['fallback_layer_count']} layers not running at native {precision} precision")
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
