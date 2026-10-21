"""
Fused Rotary Position Embedding (RoPE) in OpenAI Triton
Computes in-place or out-of-place 2D rotation on Q and K tensors.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    X,
    Cos,
    Sin,
    Out,
    stride_x_b,
    stride_x_h,
    stride_x_s,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    stride_o_d,
    S,
    D,
    HALF_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_s = tl.program_id(0)  # Sequence index
    pid_bh = tl.program_id(1)  # Batch * Head index

    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < HALF_D

    # Base pointers
    base_x = X + pid_bh * stride_x_h + pid_s * stride_x_s
    base_o = Out + pid_bh * stride_o_h + pid_s * stride_o_s

    # Load first half and second half of embedding dimension
    x1 = tl.load(base_x + offs_d * stride_x_d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base_x + (offs_d + HALF_D) * stride_x_d, mask=mask, other=0.0).to(
        tl.float32
    )

    # Load cos and sin
    cos = tl.load(
        Cos + pid_s * stride_cos_s + offs_d * stride_cos_d, mask=mask, other=1.0
    ).to(tl.float32)
    sin = tl.load(
        Sin + pid_s * stride_sin_s + offs_d * stride_sin_d, mask=mask, other=0.0
    ).to(tl.float32)

    # Apply 2D complex plane rotation:
    # out1 = x1 * cos - x2 * sin
    # out2 = x2 * cos + x1 * sin
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    tl.store(base_o + offs_d * stride_o_d, out1.to(Out.dtype.element_ty), mask=mask)
    tl.store(
        base_o + (offs_d + HALF_D) * stride_o_d,
        out2.to(Out.dtype.element_ty),
        mask=mask,
    )


def apply_rotary_emb_triton(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding to tensor x.
    Args:
        x: (Batch, Heads, SeqLen, HeadDim)
        cos: (SeqLen, HeadDim // 2)
        sin: (SeqLen, HeadDim // 2)
    Returns:
        out: Rotated tensor
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    B, H, S, D = x.shape
    HALF_D = D // 2

    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(HALF_D)

    grid = (S, B * H)
    _rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        S,
        D,
        HALF_D=HALF_D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
