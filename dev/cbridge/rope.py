"""
Minimal example for developing RoPE.
Basically we cherry-pick / copy paste the critical portions from train_llama3.py
This script then does forward/back and writes everything to file so we can
develop the CPU version, and eventually the GPU kernel as well.
"""

import math
import torch
import numpy as np

# -----------------------------------------------------------------------------
# hyperparameters

# Llama 3.1 8B config, simplified a tiny bit by removing spurious parameters
class LlamaConfig:
    version: str = "3.1"
    block_size: int = 8192
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 4096
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True

config = LlamaConfig()

# -----------------------------------------------------------------------------
# freqs_cis

def apply_scaling(freqs: torch.Tensor):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis_complex(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    """ the complex valued version of this """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    return freqs_cis

def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    """ the much simpler real-valued-only version """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)  # real part (end, dim // 2)
    freqs_sin = torch.sin(freqs)  # imaginary part (end, dim // 2)
    freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=-1) # (end, dim // 2, 2)
    return freqs_cis

# -----------------------------------------------------------------------------
# RoPE forward pass

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb_complex(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)

def apply_rotary_emb_real(x: torch.Tensor, freqs_cis: torch.Tensor):
    xf = x.float()
    x_real, x_imag = xf[..., ::2], xf[..., 1::2]
    freqs_cos, freqs_sin = freqs_cis[..., 0], freqs_cis[..., 1]
    freqs_cos = reshape_for_broadcast(freqs_cos, x_real)
    freqs_sin = reshape_for_broadcast(freqs_sin, x_real)
    x_out_real = x_real * freqs_cos - x_imag * freqs_sin
    x_out_imag = x_real * freqs_sin + x_imag * freqs_cos
    x_out = torch.stack([x_out_real, x_out_imag], dim=-1)
    return x_out.flatten(3).type_as(x)

# -----------------------------------------------------------------------------
# forward, backward, save a little example

B = 4
T = 64
head_dim = config.n_embd // config.n_head

# input here is (B, T, n_head, head_dim)
# for example in Llama 3.1 8B, this could be B=4, T=64, n_head=32, head_dim=128
inp = torch.randn(B, T, config.n_head, head_dim, dtype=torch.float32)
inp.requires_grad = True

# compare the two versions of RoPE
# 1. out_complex is the original PyTorch version that I think is way too complex
# 2. out_real is the simplified version that I think is correct
# -----------------------------------------------------------------------------
# (1)
freqs_cis = precompute_freqs_cis_complex(
    head_dim,
    config.block_size * 2,
    config.rope_theta,
    config.use_scaled_rope,
)
freqs_cis = freqs_cis[:T] # (T, head_dim // 2) of complex64
out_complex = apply_rotary_emb_complex(inp, freqs_cis)

# -----------------------------------------------------------------------------
# (2)
freqs_cis = precompute_freqs_cis_real(
    head_dim,
    config.block_size * 2,
    config.rope_theta,
    config.use_scaled_rope,
)
freqs_cis = freqs_cis[:T] # (T, head_dim // 2, 2)
out_real = apply_rotary_emb_real(inp, freqs_cis)
# -----------------------------------------------------------------------------
assert torch.allclose(out_complex, out_real, atol=1e-6)
print("RoPE simplified version OK")
out = out_real # ok to use this one
# -----------------------------------------------------------------------------

# calculate the loss and gradients
wei = torch.randn_like(out, dtype=torch.float32)
loss = (out * wei).sum()
loss.backward()

# save to .bin file so we can check correctness in C land
int_header = np.zeros(16, dtype=np.int32) # for ints
float_header = np.zeros(16, dtype=np.float32) # for floats
int_header[0] = 20240924 # magic number
int_header[1] = B
int_header[2] = T
int_header[3] = config.n_embd
int_header[4] = config.n_head
int_header[5] = config.use_scaled_rope
float_header[0] = config.rope_theta

# write the hyperparameters, inputs, output, and input gradients to file
results_file = "rope.bin"
with open(results_file, "wb") as f:
    f.write(int_header.tobytes()) # 16 int32
    f.write(float_header.tobytes()) # 16 float32
    f.write(inp.detach().cpu().numpy().tobytes()) # B * T * n_head * head_dim
    f.write(freqs_cis.detach().cpu().numpy().tobytes()) # T * head_dim // 2 * 2
    f.write(out.detach().cpu().numpy().tobytes()) # B * T * n_head * head_dim
    f.write(wei.detach().cpu().numpy().tobytes()) # B * T * n_head * head_dim
    f.write(inp.grad.detach().cpu().numpy().tobytes()) # B * T * n_head * head_dim
print("Saved results to %s" % results_file)
