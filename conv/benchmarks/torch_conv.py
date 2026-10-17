"""cuDNN baseline via PyTorch for the ResNet50 convs in ../src/, FP16, batch 1, no bias.

Same shapes and timing method as the CUDA drivers (mean of 50 back-to-back launches).
Run: python torch_conv.py [Conv_846|Conv_850]
"""
import sys

import torch
import torch.nn.functional as F

# node: (Cin, H, Cout, kernel, stride, pad)
LAYERS = {"Conv_846": (1024, 14, 2048, 1, 2, 0), "Conv_850": (512, 7, 512, 3, 1, 1)}
node = sys.argv[1] if len(sys.argv) > 1 else "Conv_846"
cin, h, cout, k, stride, pad = LAYERS[node]

torch.backends.cudnn.benchmark = True  # let cuDNN pick its fastest algorithm for this shape
x = torch.rand(1, cin, h, h, device="cuda", dtype=torch.half) * 2 - 1
w = (torch.rand(cout, cin, k, k, device="cuda", dtype=torch.half) * 2 - 1) * 0.05

def conv():
    return F.conv2d(x, w, stride=stride, padding=pad)

ref = F.conv2d(x.float(), w.float(), stride=stride, padding=pad)
err = (conv().float() - ref).abs().max().item()
assert conv().shape == (1, cout, 7, 7)

for _ in range(20):
    conv()
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(50):
    conv()
e.record()
e.synchronize()
ms = s.elapsed_time(e) / 50
flops = 2 * cout * 49 * cin * k * k
print(f"{node} cuDNN {torch.backends.cudnn.version()} fp16 conv2d: {ms:.4f} ms  {flops / ms / 1e6:.1f} GFLOPS  "
      f"max|out - fp32| {err:.4g}")
