import json
import os
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = (3, 224, 224)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def check(err):
    if isinstance(err, cudart.cudaError_t) and err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return err


def build_and_time(onnx_path, precision, batch=1, workspace_mb=2048):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    parser.parse_from_file(onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, (batch, *INPUT_SHAPE), (batch, *INPUT_SHAPE), (batch, *INPUT_SHAPE))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    return build_s


def gpu_mem_during_inference(engine_path, batch=1):
    check(cudart.cudaDeviceSynchronize())
    err, free_before, total = cudart.cudaMemGetInfo()
    check(err)

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    context.set_input_shape(INPUT_NAME, (batch, *INPUT_SHAPE))

    in_nbytes = batch * int(np.prod(INPUT_SHAPE)) * 4
    out_shape = tuple(context.get_tensor_shape(OUTPUT_NAME))
    out_nbytes = int(np.prod(out_shape)) * 4

    err, d_in = cudart.cudaMalloc(in_nbytes)
    check(err)
    err, d_out = cudart.cudaMalloc(out_nbytes)
    check(err)
    context.set_tensor_address(INPUT_NAME, int(d_in))
    context.set_tensor_address(OUTPUT_NAME, int(d_out))

    x = np.random.standard_normal((batch, *INPUT_SHAPE)).astype(np.float32)
    check(cudart.cudaMemcpy(d_in, x.ctypes.data, in_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0])

    err, stream = cudart.cudaStreamCreate()
    check(err)
    context.execute_async_v3(stream)
    check(cudart.cudaStreamSynchronize(stream)[0])

    err, free_after, _ = cudart.cudaMemGetInfo()
    check(err)

    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    cudart.cudaStreamDestroy(stream)
    del context, engine, runtime

    used_mb = (free_before - free_after) / (1024 * 1024)
    return round(used_mb, 2), round(total / (1024 * 1024), 1)


def main():
    metrics = {}

    print("Timing FP32 build...")
    metrics.setdefault("fp32", {})["engine_build_time_s"] = round(build_and_time("../models/resnet50.onnx", "fp32"), 3)
    print("Timing FP16 build...")
    metrics.setdefault("fp16", {})["engine_build_time_s"] = round(build_and_time("../models/resnet50.onnx", "fp16"), 3)
    print("(INT8 build time measured separately by calibrate_int8.py's own run; see progress log)")

    for name, engine_file in [
        ("fp32", "../resnet/fp32/resnet50_fp32.engine"),
        ("fp16", "../resnet/fp16/resnet50_fp16.engine"),
        ("int8", "../resnet/int8/resnet50_int8.engine"),
    ]:
        used_mb, total_mb = gpu_mem_during_inference(engine_file)
        metrics.setdefault(name, {})["gpu_mem_used_mb"] = used_mb
        metrics[name]["gpu_total_mem_mb"] = total_mb
        print(f"{name}: GPU memory used during inference = {used_mb} MB (of {total_mb} MB total)")

    for name, engine_file in [
        ("fp32", "../resnet/fp32/resnet50_fp32.engine"),
        ("fp16", "../resnet/fp16/resnet50_fp16.engine"),
        ("int8", "../resnet/int8/resnet50_int8.engine"),
    ]:
        metrics.setdefault(name, {})["engine_size_mb"] = round(os.path.getsize(engine_file) / (1024 * 1024), 2)

    with open("../benchmarks/full_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved ../benchmarks/full_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
