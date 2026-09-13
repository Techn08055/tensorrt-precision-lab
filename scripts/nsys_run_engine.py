import argparse

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


def main():
    ap = argparse.ArgumentParser(description="Run a serialized engine in a free-running loop, meant to be wrapped by `nsys profile`")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    context.set_input_shape(INPUT_NAME, (args.batch, *INPUT_SHAPE))

    in_nbytes = args.batch * int(np.prod(INPUT_SHAPE)) * 4
    out_shape = tuple(context.get_tensor_shape(OUTPUT_NAME))
    out_nbytes = int(np.prod(out_shape)) * 4

    err, d_in = cudart.cudaMalloc(in_nbytes)
    check(err)
    err, d_out = cudart.cudaMalloc(out_nbytes)
    check(err)
    context.set_tensor_address(INPUT_NAME, int(d_in))
    context.set_tensor_address(OUTPUT_NAME, int(d_out))

    host_in = np.random.standard_normal((args.batch, *INPUT_SHAPE)).astype(np.float32)
    check(cudart.cudaMemcpy(d_in, host_in.ctypes.data, in_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0])

    err, stream = cudart.cudaStreamCreate()
    check(err)

    for _ in range(args.warmup):
        context.execute_async_v3(stream)
    check(cudart.cudaStreamSynchronize(stream)[0])

    for _ in range(args.iters):
        context.execute_async_v3(stream)
    check(cudart.cudaStreamSynchronize(stream)[0])

    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    cudart.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()
