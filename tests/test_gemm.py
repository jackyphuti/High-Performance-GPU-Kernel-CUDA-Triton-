"""
Unit Tests for GEMM Kernels (Triton & CUDA)
"""

import pytest
import torch
from triton.gemm import matmul_triton

try:
    import custom_kernels as ck
except ImportError:
    ck = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 512, 256),
    (512, 256, 1024),
    (1024, 1024, 1024),
    (137, 241, 159), # Arbitrary non-power-of-2 dimensions
])
def test_triton_gemm(M, N, K):
    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    expected = torch.matmul(a, b)
    actual = matmul_triton(a, b)

    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2), \
        f"Mismatch: Max diff = {(actual - expected).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for testing")
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
])
def test_cuda_gemm(M, N, K):
    if ck is None or not ck.ops.HAS_CUDA_EXT:
        pytest.skip("CUDA extension not built")

    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    expected = torch.matmul(a, b)

    actual_tiled = ck.gemm(a, b, backend="cuda_tiled")
    assert torch.allclose(actual_tiled, expected, atol=1e-3, rtol=1e-3)

    actual_reg = ck.gemm(a, b, backend="cuda_reg_tiled")
    assert torch.allclose(actual_reg, expected, atol=1e-3, rtol=1e-3)
