"""
An RMSNorm PyTorch reference implementation.
This script then does forward/back and writes everything to file so we can
develop the CPU version, and eventually the GPU kernel as well.
"""

import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        rstd = torch.rsqrt(mean_sq)
        norm = x * rstd
        return norm

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def rmsnorm_backward(x, w, dout, eps):
    # recompute the rstd, norm (or we could cache it in the forward pass)
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True) + eps  # (B, T, 1)
    rstd = torch.rsqrt(mean_sq)  # (B, T, 1)
    norm = x * rstd  # (B, T, C)
    # gradients for weights
    dw = (dout * norm).sum((0, 1))  # (C)
    # gradients for input
    dnorm = dout * w  # (B, T, C)
    dx = dnorm - norm * (dnorm * norm).mean(dim=-1, keepdim=True)
    dx *= rstd
    return dx, dw

# -----------------------------------------------------------------------------

# seed the rng
torch.manual_seed(42)

B = 4
T = 64
C = 256
eps = 1e-5

inp = torch.randn(B, T, C, dtype=torch.float32)
inp.requires_grad = True

# rmsnorm
m = RMSNorm(C, eps=eps)
out = m(inp)

# loss can just be a weighted sum, with some fixed weights
wei = torch.randn_like(out, dtype=torch.float32)
loss = (out * wei).sum()
loss.backward()

# let's now do the backward pass manually
# backprop starts with the output gradient, which is exactly wei because of the loss functions
dx, dw = rmsnorm_backward(inp, m.weight, wei, eps)
# let's assert that the gradients match
assert torch.allclose(dx, inp.grad, atol=1e-6)
assert torch.allclose(dw, m.weight.grad, atol=1e-6)
print("RMSNorm gradients match")
print("first 5 elements of dx comparison:")
print(dx.view(-1)[:5].tolist())
print(inp.grad.view(-1)[:5].tolist())
print("first 5 elements of dw comparison:")
print(dw.view(-1)[:5].tolist())
print(m.weight.grad.view(-1)[:5].tolist())
print("dx error:", (inp.grad.view(-1) - dx.view(-1)).abs().max().item())
print("dw error:", (m.weight.grad.view(-1) - dw.view(-1)).abs().max().item())

# save to .bin file so we can check correctness in C land
int_header = np.zeros(16, dtype=np.int32) # for ints
float_header = np.zeros(16, dtype=np.float32) # for floats
int_header[0] = 20240925 # magic number
int_header[1] = B
int_header[2] = T
int_header[3] = C
float_header[0] = eps

# write the hyperparameters, inputs, output, and input gradients to file
results_file = "rmsnorm.bin"
with open(results_file, "wb") as f:
    f.write(int_header.tobytes()) # 16 int32
    f.write(float_header.tobytes()) # 16 float32
    f.write(inp.detach().cpu().numpy().tobytes()) # B * T * C
    f.write(out.detach().cpu().numpy().tobytes()) # B * T * C
    f.write(wei.detach().cpu().numpy().tobytes()) # B * T * C
    f.write(inp.grad.detach().cpu().numpy().tobytes()) # B * T * C
    f.write(m.weight.grad.detach().cpu().numpy().tobytes()) # C
print("Saved results to %s" % results_file)
