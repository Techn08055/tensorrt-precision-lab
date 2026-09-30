"""
Per-block INT8 sensitivity sweep for ResNet50.

Question: which layers contribute little to accuracy, so only *those* could be
quantized to INT8 while the rest of the network stays at higher precision?

Method: build a baseline engine forced to FP16 everywhere (via per-layer
`precision` + OBEY_PRECISION_CONSTRAINTS, so it's pinned rather than TensorRT's
free choice). Then, one residual block at a time, force just that block's
conv layers to INT8 (reusing the existing calibration.cache for scales) while
everything else stays pinned to FP16. Compare each mixed engine's output back
to the FP16 baseline (not the FP32 PyTorch reference -- that isolates the
quantization error *this block* introduces, rather than mixing it with the
fp32->fp16 gap). A block with near-zero drift vs baseline is safe to quantize;
a block with a big top-1 flip rate or logit shift is not.

Blocks are ONNX Conv node names grouped by residual block (stem, then
layer{1..4}.{i}), derived directly from the ONNX graph rather than assumed,
since TensorRT execution order does not match ONNX definition order for
branches that run in parallel (see block map construction below).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import onnx
import tensorrt as trt
from cuda.bindings import runtime as cudart
from PIL import Image
from torchvision.models import resnet50
import torch

INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STAGE_SIZES = [3, 4, 6, 3]  # ResNet50 bottleneck counts per stage

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def check(err):
    if isinstance(err, cudart.cudaError_t) and err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return err


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
    return arr.transpose(2, 0, 1).astype(np.float32)


def load_real_batch(image_dir, n, seed):
    files = sorted(glob.glob(os.path.join(image_dir, "**", "*.JPEG"), recursive=True))
    rng = np.random.default_rng(seed)
    picked = rng.choice(files, size=min(n, len(files)), replace=False)
    return np.stack([preprocess_image(f) for f in picked]).astype(np.float32)


def pytorch_reference(weights_path, x):
    model = resnet50()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Reuses the calibration.cache written by calibrate_int8.py -- no recalibration pass needed."""

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
        pass

    def __del__(self):
        if hasattr(self, "device_input"):
            cudart.cudaFree(self.device_input)


def build_block_map(onnx_path):
    """Group the 53 Conv nodes into stem + 16 bottleneck blocks, in ONNX
    definition order (== PyTorch Bottleneck.forward() trace order: conv1,
    conv2, conv3, then downsample(x) if present -- confirmed by inspecting
    each node's input tensor name in the exported graph)."""
    model = onnx.load(onnx_path, load_external_data=False)
    convs = [n.name for n in model.graph.node if n.op_type == "Conv"]

    blocks = {"stem": [convs[0]]}
    idx = 1
    for stage_i, n_blocks in enumerate(STAGE_SIZES, start=1):
        for b in range(n_blocks):
            size = 4 if b == 0 else 3  # first block of each stage has a downsample conv
            blocks[f"layer{stage_i}.{b}"] = convs[idx : idx + size]
            idx += size

    consumed = sum(len(v) for v in blocks.values())
    assert consumed == len(convs), f"conv count mismatch: grouped {consumed}, found {len(convs)} in ONNX graph"
    return blocks


def build_mixed_engine(onnx_path, all_conv_names, int8_layer_names, workspace_mb, calib_images, calib_cache):
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
    config.set_flag(trt.BuilderFlag.FP16)

    calibrator = None
    if int8_layer_names:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        calibrator = ImageCalibrator(calib_images, calib_cache)
        config.int8_calibrator = calibrator
    else:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Fixed at batch=1: run_engine() executes one sample at a time regardless of
    # how many images the sweep evaluates, so the engine only ever needs batch=1.
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, (1, *INPUT_SHAPE), (1, *INPUT_SHAPE), (1, *INPUT_SHAPE))
    config.add_optimization_profile(profile)
    if int8_layer_names:
        config.set_calibration_profile(profile)

    # Only constrain the conv layers -- forcing precision on Shape/Constant/etc.
    # layers errors out (they must stay Int64/whatever their data requires), and
    # leaving activation/elementwise/pooling ops unconstrained is fine: with every
    # conv pinned, TensorRT has no freedom left to sneak INT8 in via a conv the
    # sweep didn't intend, and non-conv ops just adopt whatever fits their neighbors.
    int8_set = set(int8_layer_names)
    pinned, skipped = 0, 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.name not in all_conv_names:
            continue
        try:
            layer.precision = trt.DataType.INT8 if layer.name in int8_set else trt.DataType.HALF
            pinned += 1
        except Exception:
            skipped += 1

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine, pinned, skipped


def run_engine(engine, x):
    context = engine.create_execution_context()
    context.set_input_shape(INPUT_NAME, (1, *INPUT_SHAPE))

    in_nbytes = int(np.prod(INPUT_SHAPE)) * 4
    out_shape = tuple(context.get_tensor_shape(OUTPUT_NAME))
    out_nbytes = int(np.prod(out_shape)) * 4

    err, d_in = cudart.cudaMalloc(in_nbytes)
    check(err)
    err, d_out = cudart.cudaMalloc(out_nbytes)
    check(err)
    context.set_tensor_address(INPUT_NAME, int(d_in))
    context.set_tensor_address(OUTPUT_NAME, int(d_out))

    err, stream = cudart.cudaStreamCreate()
    check(err)

    outs = []
    for i in range(x.shape[0]):
        sample = np.ascontiguousarray(x[i : i + 1])
        check(cudart.cudaMemcpy(d_in, sample.ctypes.data, in_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0])
        context.execute_async_v3(stream)
        check(cudart.cudaStreamSynchronize(stream)[0])
        host_out = np.empty(out_shape, dtype=np.float32)
        check(cudart.cudaMemcpy(host_out.ctypes.data, d_out, out_nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)[0])
        outs.append(host_out.copy())

    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    cudart.cudaStreamDestroy(stream)
    return np.concatenate(outs, axis=0)


def check_actual_precision(engine, block_layer_names):
    """Verify OBEY_PRECISION_CONSTRAINTS actually landed the block in Int8 (not a silent fallback)."""
    inspector = engine.create_engine_inspector()
    dtypes = []
    for i in range(engine.num_layers):
        info = json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        if not isinstance(info, dict):
            continue
        name = info.get("Name", "")
        if any(bn in name for bn in block_layer_names):
            outputs = info.get("Outputs") or [{}]
            dtypes.append(outputs[0].get("Format/Datatype", "Unknown"))
    return dtypes


def main():
    ap = argparse.ArgumentParser(description="Per-block INT8 sensitivity sweep for ResNet50")
    ap.add_argument("--onnx", default="../models/resnet50.onnx")
    ap.add_argument("--weights", default="../models/resnet50.pth")
    ap.add_argument("--calib-images", default="../calibration/images")
    ap.add_argument("--calib-cache", default="../calibration/calibration.cache")
    ap.add_argument("--out", default="../profiling/int8_sensitivity_sweep.json")
    ap.add_argument("--batch", type=int, default=64, help="number of real images to eval each engine on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--blocks", nargs="+", default=None, help="subset of block names to sweep (default: all)")
    args = ap.parse_args()

    block_map = build_block_map(args.onnx)
    print(f"Block map: {len(block_map)} units, {sum(len(v) for v in block_map.values())} conv layers total")
    for name, layers in block_map.items():
        print(f"  {name}: {layers}")

    targets = args.blocks if args.blocks else list(block_map.keys())
    all_conv_names = set(name for layers in block_map.values() for name in layers)

    x = load_real_batch(args.calib_images, args.batch, args.seed)
    ref = pytorch_reference(args.weights, x)
    ref_top1 = ref.argmax(axis=1)

    print(f"\nBuilding FP16 baseline (all layers pinned to HALF)...")
    t0 = time.time()
    baseline_engine, pinned, skipped = build_mixed_engine(
        args.onnx, all_conv_names, [], args.workspace_mb, args.calib_images, args.calib_cache
    )
    print(f"  built in {time.time()-t0:.1f}s ({pinned} layers pinned, {skipped} skipped)")
    baseline_out = run_engine(baseline_engine, x)
    baseline_top1 = baseline_out.argmax(axis=1)
    baseline_vs_pytorch = float((baseline_top1 == ref_top1).mean())
    print(f"  baseline FP16 top-1 vs PyTorch: {baseline_vs_pytorch:.4f}")

    results = {
        "batch": args.batch,
        "seed": args.seed,
        "baseline_top1_vs_pytorch": baseline_vs_pytorch,
        "blocks": {},
    }

    for name in targets:
        layer_names = block_map[name]
        print(f"\n=== Block '{name}' -> INT8 ({len(layer_names)} conv layers), rest pinned FP16 ===")
        t0 = time.time()
        engine, pinned, skipped = build_mixed_engine(
            args.onnx, all_conv_names, layer_names, args.workspace_mb, args.calib_images, args.calib_cache
        )
        build_s = time.time() - t0
        actual_dtypes = check_actual_precision(engine, layer_names)
        out = run_engine(engine, x)
        top1 = out.argmax(axis=1)

        flip_vs_baseline = float((top1 != baseline_top1).mean())
        mean_abs_diff_vs_baseline = float(np.abs(out - baseline_out).mean())
        max_abs_diff_vs_baseline = float(np.abs(out - baseline_out).max())
        top1_vs_pytorch = float((top1 == ref_top1).mean())

        results["blocks"][name] = {
            "layers": layer_names,
            "num_conv_layers": len(layer_names),
            "build_seconds": round(build_s, 1),
            "actual_output_dtypes": actual_dtypes,
            "flip_rate_vs_fp16_baseline": round(flip_vs_baseline, 4),
            "mean_abs_logit_diff_vs_baseline": round(mean_abs_diff_vs_baseline, 6),
            "max_abs_logit_diff_vs_baseline": round(max_abs_diff_vs_baseline, 6),
            "top1_vs_pytorch": round(top1_vs_pytorch, 4),
        }
        print(f"  build {build_s:.1f}s | dtypes seen: {set(actual_dtypes)} | "
              f"flip-vs-baseline {flip_vs_baseline:.4f} | mean|Δlogit| {mean_abs_diff_vs_baseline:.5f} | "
              f"top1-vs-pytorch {top1_vs_pytorch:.4f}")

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
