import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from PIL import Image
from torchvision.models import resnet50

INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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


def run_engine_per_sample(engine_path, x):
    """Run each sample individually at batch=1 so this works regardless of an
    engine's optimization-profile batch range (fp32/fp16 here are fixed at
    batch=1; the int8 engine supports 1-8)."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
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


def main():
    ap = argparse.ArgumentParser(description="Compare TensorRT engine outputs against the PyTorch reference")
    ap.add_argument("--weights", default="../models/resnet50.pth")
    ap.add_argument("--engines", nargs="+", default=["../resnet/fp32/resnet50_fp32.engine", "../resnet/fp16/resnet50_fp16.engine", "../resnet/int8/resnet50_int8.engine"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-file", default="../benchmarks/accuracy_results.json")
    ap.add_argument("--real-images", default="../calibration/images")
    ap.add_argument("--synthetic", action="store_true", help="use random Gaussian noise instead of real images")
    args = ap.parse_args()

    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        x = rng.standard_normal((args.batch, *INPUT_SHAPE)).astype(np.float32)
    else:
        x = load_real_batch(args.real_images, args.batch, args.seed)

    ref = pytorch_reference(args.weights, x)
    ref_top1 = ref.argmax(axis=1)

    results = {}
    for engine_path in args.engines:
        name = Path(engine_path).stem
        out = run_engine_per_sample(engine_path, x)
        top1 = out.argmax(axis=1)
        results[name] = {
            "max_abs_diff": float(np.abs(out - ref).max()),
            "mean_abs_diff": float(np.abs(out - ref).mean()),
            "top1_match_rate": float((top1 == ref_top1).mean()),
        }

    input_kind = "synthetic Gaussian noise" if args.synthetic else f"real images ({args.real_images})"
    print(f"\n=== Accuracy vs PyTorch reference (batch={args.batch}, seed={args.seed}, input={input_kind}) ===")
    print(f"{'engine':<20}{'max_abs_diff':<16}{'mean_abs_diff':<16}{'top1_match_rate'}")
    for name, stats in results.items():
        print(f"{name:<20}{stats['max_abs_diff']:<16.6f}{stats['mean_abs_diff']:<16.6f}{stats['top1_match_rate']}")

    with open(args.results_file, "w") as f:
        json.dump({"batch": args.batch, "seed": args.seed, "input": input_kind, "results": results}, f, indent=2)
    print(f"\nSaved accuracy results to {args.results_file}")


if __name__ == "__main__":
    main()
