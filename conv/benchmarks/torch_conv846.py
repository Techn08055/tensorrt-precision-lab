"""cuDNN baseline for Conv_846 via PyTorch: 1x1 stride-2 conv, 1024x14x14 -> 2048x7x7, FP16, no bias.

Same shape and timing method as ../src/conv846_gemm.cu (mean of 50 back-to-back launches).
Run: python torch_conv846.py
"""
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True  # let cuDNN pick its fastest algorithm for this shape
x = torch.rand(1, 1024, 14, 14, device="cuda", dtype=torch.half) * 2 - 1
w = (torch.rand(2048, 1024, 1, 1, device="cuda", dtype=torch.half) * 2 - 1) * 0.05

def conv():
    return F.conv2d(x, w, stride=2)

ref = F.conv2d(x.float(), w.float(), stride=2)
err = (conv().float() - ref).abs().max().item()
assert conv().shape == (1, 2048, 7, 7)

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
flops = 2 * 2048 * 49 * 1024
print(f"cuDNN {torch.backends.cudnn.version()} fp16 conv2d: {ms:.4f} ms  {flops / ms / 1e6:.1f} GFLOPS  "
      f"max|out - fp32| {err:.4g}")
