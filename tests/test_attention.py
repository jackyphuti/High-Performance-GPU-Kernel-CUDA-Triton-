"""
Unit Tests for FlashAttention-2 (Triton & CUDA)
"""

import math
import pytest
import torch
from triton.flash_attention import flash_attention_triton

try:
    import custom_kernels as ck
except ImportError:
    ck = None


def reference_attention(q, k, v, is_causal=True):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
        scores = scores + mask
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
@pytest.mark.parametrize("B,H,N,D,is_causal", [
    (1, 4, 128, 64, True),
    (2, 8, 256, 64, True),
    (1, 4, 128, 64, False),
    (2, 4, 512, 64, True),
])
def test_triton_flash_attention(B, H, N, D, is_causal):
    torch.manual_seed(42)
    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

    expected = reference_attention(q, k, v, is_causal=is_causal)
    actual = flash_attention_triton(q, k, v, is_causal=is_causal)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2), \
        f"Max diff: {(actual - expected).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
def test_cuda_flash_attention():
    if ck is None or not ck.ops.HAS_CUDA_EXT:
        pytest.skip("CUDA extension not built")

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 128, 64
    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float32)

    expected = reference_attention(q, k, v, is_causal=True)
    actual = ck.flash_attention(q, k, v, is_causal=True)

    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
