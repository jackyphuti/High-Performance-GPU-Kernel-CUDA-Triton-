"""
Unit Tests for Normalization, Activation, and RoPE Kernels
"""

import math
import pytest
import torch
from triton.rms_norm import rms_norm_triton
from triton.swiglu import swiglu_triton
from triton.rope import apply_rotary_emb_triton

try:
    import custom_kernels as ck
except ImportError:
    ck = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
def test_rms_norm():
    torch.manual_seed(42)
    B, S, D = 4, 128, 4096
    x = torch.randn((B, S, D), device="cuda", dtype=torch.float32)
    w = torch.randn((D,), device="cuda", dtype=torch.float32)
    eps = 1e-6

    # Reference
    variance = x.pow(2).mean(-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps) * w

    # Triton
    actual_triton = rms_norm_triton(x, w, eps)
    assert torch.allclose(actual_triton, expected, atol=1e-3, rtol=1e-3)

    # CUDA
    if ck is not None and ck.ops.HAS_CUDA_EXT:
        actual_cuda = ck.rms_norm(x, w, eps)
        assert torch.allclose(actual_cuda, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
def test_swiglu():
    torch.manual_seed(42)
    B, S, D2 = 2, 64, 2048
    x = torch.randn((B, S, D2), device="cuda", dtype=torch.float32)

    # Reference
    D = D2 // 2
    a, b = x[..., :D], x[..., D:]
    expected = torch.nn.functional.silu(a) * b

    # Triton
    actual_triton = swiglu_triton(x)
    assert torch.allclose(actual_triton, expected, atol=1e-4, rtol=1e-4)

    # CUDA
    if ck is not None and ck.ops.HAS_CUDA_EXT:
        actual_cuda = ck.swiglu(x)
        assert torch.allclose(actual_cuda, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
def test_rope():
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 128, 64
    x = torch.randn((B, H, S, D), device="cuda", dtype=torch.float32)
    cos = torch.randn((S, D // 2), device="cuda", dtype=torch.float32)
    sin = torch.randn((S, D // 2), device="cuda", dtype=torch.float32)

    # Reference
    half_d = D // 2
    x1, x2 = x[..., :half_d], x[..., half_d:]
    expected1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    expected2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
    expected = torch.cat([expected1, expected2], dim=-1)

    actual = apply_rotary_emb_triton(x, cos, sin)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
