"""
Builds a "selective INT8" ResNet50 engine: only the bottleneck blocks that the
per-block sensitivity sweep (sweep_int8_sensitivity.py,
../profiling/int8_sensitivity_sweep.json) found safe -- zero top-1 flips vs. an
all-FP16 baseline over 64 real images -- are quantized to INT8. Every other
layer, including the stem conv and the classifier, stays pinned to FP16.

This is the follow-up experiment the sweep was for: does selectively quantizing
only the low-sensitivity blocks recover meaningful latency over full FP16
without paying full INT8's accuracy cost?
"""

import argparse
import json
import os

import tensorrt as trt

from sweep_int8_sensitivity import ImageCalibrator, INPUT_NAME, INPUT_SHAPE, build_block_map

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Blocks with 0% top-1 flip rate vs. the FP16 baseline in the sensitivity sweep.
SAFE_BLOCKS = ["layer2.1", "layer3.0", "layer3.4", "layer3.5", "layer4.1"]


def build_selective_engine(onnx_path, engine_path, safe_blocks, workspace_mb, calib_images, calib_cache):
    block_map = build_block_map(onnx_path)
    all_conv_names = set(name for layers in block_map.values() for name in layers)
    int8_names = set(name for b in safe_blocks for name in block_map[b])
    print(f"Quantizing {len(safe_blocks)} blocks to INT8 ({len(int8_names)} conv layers "
          f"of {len(all_conv_names)} total); rest pinned to FP16.")

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
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)  # matches production fp32/fp16/int8 builds

    calibrator = ImageCalibrator(calib_images, calib_cache)
    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, (1, *INPUT_SHAPE), (1, *INPUT_SHAPE), (1, *INPUT_SHAPE))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.name not in all_conv_names:
            continue
        layer.precision = trt.DataType.INT8 if layer.name in int8_names else trt.DataType.HALF
        pinned += 1

    print(f"Building selective-INT8 engine ({pinned} conv layers pinned)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved {engine_path}")
    return int8_names


def main():
    ap = argparse.ArgumentParser(description="Build a selective (block-level) INT8 ResNet50 engine")
    ap.add_argument("--onnx", default="../models/resnet50.onnx")
    ap.add_argument("--engine", default="../resnet/selective_int8/resnet50_selective_int8.engine")
    ap.add_argument("--calib-images", default="../calibration/images")
    ap.add_argument("--calib-cache", default="../calibration/calibration.cache")
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--blocks", nargs="+", default=SAFE_BLOCKS)
    ap.add_argument("--meta-out", default="../profiling/selective_int8_meta.json")
    args = ap.parse_args()

    int8_names = build_selective_engine(
        args.onnx, args.engine, args.blocks, args.workspace_mb, args.calib_images, args.calib_cache
    )

    with open(args.meta_out, "w") as f:
        json.dump({"blocks_quantized": args.blocks, "conv_layers_quantized": sorted(int8_names)}, f, indent=2)
    print(f"Saved {args.meta_out}")


if __name__ == "__main__":
    main()
