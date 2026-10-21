import torch

# Try importing the compiled C++/CUDA extension
try:
    import custom_kernels_cuda as _C

    HAS_CUDA_EXT = True
except ImportError:
    _C = None
    HAS_CUDA_EXT = False

# Try importing Triton kernels
try:
    from triton.flash_attention import flash_attention_triton
    from triton.gemm import matmul_triton
    from triton.rms_norm import rms_norm_triton
    from triton.rope import apply_rotary_emb_triton
    from triton.swiglu import swiglu_triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def gemm(a: torch.Tensor, b: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """
    Unified GEMM dispatcher:
    backend: 'auto', 'cuda_reg_tiled', 'cuda_tiled', 'cuda_naive', 'triton', 'cublas'
    """
    if backend == "auto":
        if HAS_CUDA_EXT and a.dtype == torch.float32:
            return _C.gemm_reg_tiled(a, b)
        elif HAS_TRITON:
            return matmul_triton(a, b)
        else:
            return torch.matmul(a, b)

    if backend == "cuda_reg_tiled":
        assert HAS_CUDA_EXT, "CUDA extension not built. Run 'pip install -e .'"
        return _C.gemm_reg_tiled(a, b)
    elif backend == "cuda_tiled":
        assert HAS_CUDA_EXT, "CUDA extension not built. Run 'pip install -e .'"
        return _C.gemm_tiled(a, b)
    elif backend == "cuda_naive":
        assert HAS_CUDA_EXT, "CUDA extension not built. Run 'pip install -e .'"
        return _C.gemm_naive(a, b)
    elif backend == "triton":
        assert HAS_TRITON, "Triton not available."
        return matmul_triton(a, b)
    elif backend == "cublas":
        return torch.matmul(a, b)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Unified FlashAttention-2 dispatcher.
    """
    if backend == "auto":
        if HAS_CUDA_EXT and q.dtype == torch.float32 and q.shape[-1] == 64:
            return _C.flash_attention(q, k, v, is_causal)
        elif HAS_TRITON:
            return flash_attention_triton(q, k, v, is_causal=is_causal)
        else:
            scale = 1.0 / (q.shape[-1] ** 0.5)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            if is_causal:
                seq_len = q.shape[-2]
                mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=q.device),
                    diagonal=1,
                )
                scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            return torch.matmul(p, v)

    if backend == "cuda":
        assert HAS_CUDA_EXT
        return _C.flash_attention(q, k, v, is_causal)
    elif backend == "triton":
        assert HAS_TRITON
        return flash_attention_triton(q, k, v, is_causal)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, backend: str = "auto"
) -> torch.Tensor:
    """
    Unified RMSNorm dispatcher.
    """
    if backend == "auto":
        if HAS_CUDA_EXT and x.dtype == torch.float32:
            return _C.rms_norm(x, weight, eps)
        elif HAS_TRITON:
            return rms_norm_triton(x, weight, eps)
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + eps) * weight

    if backend == "cuda":
        assert HAS_CUDA_EXT
        return _C.rms_norm(x, weight, eps)
    elif backend == "triton":
        assert HAS_TRITON
        return rms_norm_triton(x, weight, eps)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def swiglu(x: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """
    Unified SwiGLU activation dispatcher.
    """
    if backend == "auto":
        if HAS_CUDA_EXT and x.dtype == torch.float32:
            return _C.swiglu(x)
        elif HAS_TRITON:
            return swiglu_triton(x)
        else:
            d = x.shape[-1] // 2
            a, b = x[..., :d], x[..., d:]
            return torch.nn.functional.silu(a) * b

    if backend == "cuda":
        assert HAS_CUDA_EXT
        return _C.swiglu(x)
    elif backend == "triton":
        assert HAS_TRITON
        return swiglu_triton(x)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Unified RoPE dispatcher.
    """
    if HAS_TRITON:
        return apply_rotary_emb_triton(x, cos, sin)
    else:
        half_d = x.shape[-1] // 2
        x1, x2 = x[..., :half_d], x[..., half_d:]
        cos_expanded = cos[:, None, :] if cos.dim() == 2 else cos
        sin_expanded = sin[:, None, :] if sin.dim() == 2 else sin
        out1 = x1 * cos_expanded - x2 * sin_expanded
        out2 = x2 * cos_expanded + x1 * sin_expanded
        return torch.cat([out1, out2], dim=-1)
