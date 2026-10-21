"""
Fused SwiGLU Activation in OpenAI Triton
Computes: SwiGLU(x, y) = (x * sigmoid(x)) * y
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    In,
    Out,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    M,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)

    offs_d = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D

    # In is shape [M, 2*D]
    ptr_x = In + row_idx * stride_in_m + offs_d * stride_in_n
    ptr_y = In + row_idx * stride_in_m + (offs_d + D) * stride_in_n

    x = tl.load(ptr_x, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(ptr_y, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)
    out = silu_x * y

    ptr_out = Out + row_idx * stride_out_m + offs_d * stride_out_n
    tl.store(ptr_out, out.to(Out.dtype.element_ty), mask=mask)


def swiglu_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton Fused SwiGLU wrapper.
    Args:
        x: (..., 2*D) Input tensor where the last dimension is split into (x, y)
    Returns:
        out: (..., D) SwiGLU activated tensor
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.shape[-1] % 2 == 0, "Last dimension must be even"

    orig_shape = list(x.shape)
    D = orig_shape[-1] // 2
    x_2d = x.reshape(-1, 2 * D)
    M = x_2d.shape[0]

    out_2d = torch.empty((M, D), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024
    grid = (M, triton.cdiv(D, BLOCK_SIZE))

    _swiglu_kernel[grid](
        x_2d,
        out_2d,
        x_2d.stride(0),
        x_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        M,
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    orig_shape[-1] = D
    return out_2d.reshape(orig_shape)
