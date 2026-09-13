import argparse
import shutil
import subprocess

INPUT_NAME = "input"
INPUT_SHAPE = "3x224x224"


def build_engine(onnx_path, engine_path, fp16, min_batch, opt_batch, max_batch, workspace_mb):
    if shutil.which("trtexec") is None:
        raise RuntimeError("trtexec not found on PATH. Install TensorRT first.")

    shapes = {
        "min": f"{INPUT_NAME}:{min_batch}x{INPUT_SHAPE}",
        "opt": f"{INPUT_NAME}:{opt_batch}x{INPUT_SHAPE}",
        "max": f"{INPUT_NAME}:{max_batch}x{INPUT_SHAPE}",
    }

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
        f"--minShapes={shapes['min']}",
        f"--optShapes={shapes['opt']}",
        f"--maxShapes={shapes['max']}",
    ]
    if fp16:
        cmd.append("--fp16")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved TensorRT engine to {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert an ONNX model to a TensorRT engine via trtexec")
    parser.add_argument("--onnx", default="resnet50.onnx", help="Path to input ONNX model")
    parser.add_argument("--engine", default="resnet50.engine", help="Path to output TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    args = parser.parse_args()

    build_engine(
        args.onnx,
        args.engine,
        args.fp16,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.workspace_mb,
    )


if __name__ == "__main__":
    main()
