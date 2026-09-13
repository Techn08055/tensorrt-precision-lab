import torch
from torchvision.models import resnet50

WEIGHTS_PATH = "../models/resnet50.pth"
OUTPUT_PATH = "../models/resnet50.onnx"


def main():
    model = resnet50()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    batch_size = torch.export.Dim("batch_size")

    torch.onnx.export(
        model,
        (dummy_input,),
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {0: batch_size}},
    )
    print(f"Saved ONNX model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
