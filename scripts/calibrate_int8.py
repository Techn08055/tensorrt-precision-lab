import argparse
import glob
import os

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from PIL import Image

INPUT_NAME = "input"
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
    return arr.transpose(2, 0, 1).astype(np.float32)  # CHW


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds real preprocessed images to the TensorRT builder for INT8 range calibration."""

    def __init__(self, image_dir, cache_file, batch_size=8):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.files = sorted(
            glob.glob(os.path.join(image_dir, "**", "*.JPEG"), recursive=True)
        )
        if not self.files:
            raise RuntimeError(f"No .JPEG calibration images found under {image_dir}")
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
        batch = np.stack([preprocess_image(f) for f in batch_files]).astype(np.float32)
        batch = np.ascontiguousarray(batch)

        err = cudart.cudaMemcpy(
            self.device_input,
            batch.ctypes.data,
            batch.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )[0]
        check(err)

        self.current_index += self.batch_size
        print(f"  calibration batch {self.current_index}/{len(self.files)}")
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reusing existing calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Saved calibration cache to {self.cache_file}")

    def __del__(self):
        if hasattr(self, "device_input"):
            cudart.cudaFree(self.device_input)


def build_int8_engine(
    onnx_path,
    engine_path,
    calib_image_dir,
    calib_cache,
    calib_batch_size,
    min_batch,
    opt_batch,
    max_batch,
    workspace_mb,
):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # allows fallback for layers without an INT8 kernel
    # Forces the engine to consume/produce I/O tensors in the format the first/last
    # layer actually wants, instead of inserting a reformat copy layer at the boundary.
    # Bit-identical output; measured effect across repeated builds is small (~2-4%
    # for INT8, within noise for FP16/FP32) - kept because it's free, not because
    # it's a big win (see profiling docs).
    config.set_flag(trt.BuilderFlag.DIRECT_IO)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        (min_batch, *INPUT_SHAPE),
        (opt_batch, *INPUT_SHAPE),
        (max_batch, *INPUT_SHAPE),
    )
    config.add_optimization_profile(profile)

    calibrator = ImageCalibrator(calib_image_dir, calib_cache, batch_size=calib_batch_size)
    config.int8_calibrator = calibrator
    # calibration itself always runs at a fixed shape — use the opt batch size
    config.set_calibration_profile(profile)

    print(f"Building INT8 engine from {onnx_path} (calibrating on {len(calibrator.files)} images)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved TensorRT INT8 engine to {engine_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build an INT8 TensorRT engine from an ONNX model, calibrated on real images"
    )
    parser.add_argument("--onnx", default="../models/resnet50.onnx")
    parser.add_argument("--engine", default="../int8/resnet50_int8.engine")
    parser.add_argument("--calib-images", default="../calibration/images")
    parser.add_argument("--calib-cache", default="../calibration/calibration.cache")
    parser.add_argument("--calib-batch-size", type=int, default=8)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    args = parser.parse_args()

    build_int8_engine(
        args.onnx,
        args.engine,
        args.calib_images,
        args.calib_cache,
        args.calib_batch_size,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.workspace_mb,
    )


if __name__ == "__main__":
    main()
