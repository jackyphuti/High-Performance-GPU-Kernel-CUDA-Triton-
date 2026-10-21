"""
High-Performance GPU Kernel Library (CUDA C++ & OpenAI Triton)
"""

from .ops import (
    flash_attention,
    gemm,
    rms_norm,
    rope,
    swiglu,
)

__version__ = "0.2.0"
__all__ = [
    "flash_attention",
    "gemm",
    "rms_norm",
    "rope",
    "swiglu",
]
