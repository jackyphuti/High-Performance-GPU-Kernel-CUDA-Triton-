"""
Fused RMSNorm Kernel in OpenAI Triton
Computes: y = (x / sqrt(mean(x^2) + eps)) * w
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X, W, Y,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Offset pointers for current row
    row_x = X + row_idx * stride_x_row
    row_y = Y + row_idx * stride_y_row

    # Accumulate sum of squares
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)

    # Compute variance / RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms = tl.rsqrt(mean_sq + eps)

    # Normalize and scale
    y = x * rms * w
    tl.store(row_y + offs, y.to(Y.dtype.element_ty), mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Triton Fused RMSNorm wrapper.
    Args:
        x: (..., N) Input tensor on CUDA
        weight: (N,) Weight tensor on CUDA
        eps: Small epsilon for numerical stability
    Returns:
        y: Normalized tensor with same shape as x
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    assert x.shape[-1] == weight.shape[0], "Hidden dimensions must match"

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape

    y_2d = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    _rms_norm_fwd_kernel[grid](
        x_2d, weight, y_2d,
        x_2d.stride(0), y_2d.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,
    )
    return y_2d.reshape(orig_shape)
