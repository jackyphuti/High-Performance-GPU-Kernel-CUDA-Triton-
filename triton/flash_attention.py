"""
FlashAttention-2 Forward Implementation in OpenAI Triton
Implements the Online Softmax algorithm to prevent materializing O(N^2) attention matrices.
"""

import math

import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_q = (
        off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    off_k = (
        off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    )
    off_v = (
        off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    )

    # Pointers to Q, K, V
    q = tl.load(Q + off_q, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Online softmax statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Scale Q by sm_scale
    q = (q * sm_scale).to(Q.dtype.element_ty)

    # Loop over KV sequence length
    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(0, end_n, BLOCK_N):
        curr_offs_n = start_n + offs_n

        # Load K and V tiles
        k = tl.load(
            K + off_k + start_n * stride_kn,
            mask=curr_offs_n[None, :] < N_CTX,
            other=0.0,
        )
        v = tl.load(
            V + off_v + start_n * stride_vn,
            mask=curr_offs_n[:, None] < N_CTX,
            other=0.0,
        )

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, float("-inf"))

        # Online Softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Scale old accumulator
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Accumulate P * V
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        m_i = m_ij

    # Normalize output accumulator
    acc = acc / l_i[:, None]

    # Write output back to HBM
    off_o = (
        off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    )
    tl.store(Out + off_o, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True
) -> torch.Tensor:
    """
    Triton FlashAttention-2 forward wrapper.
    Args:
        q: (Batch, Heads, SeqLen, HeadDim)
        k: (Batch, Heads, SeqLen, HeadDim)
        v: (Batch, Heads, SeqLen, HeadDim)
        is_causal: Whether to apply causal masking
    Returns:
        out: (Batch, Heads, SeqLen, HeadDim)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA"
    assert q.shape == k.shape == v.shape, "Shapes of Q, K, V must match"

    Z, H, N_CTX, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 64 if D <= 64 else 32
    BLOCK_N = 64 if D <= 64 else 32

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        L,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=2,
    )
    return out
