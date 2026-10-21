"""
High-Performance GPU Kernel Library (CUDA C++ & OpenAI Triton)
"""

from .ops import (
    gemm,
    flash_attention,
    rms_norm,
    swiglu,
    rope,
)

__version__ = "0.2.0"
__all__ = [
    "gemm",
    "flash_attention",
    "rms_norm",
    "swiglu",
    "rope",
]
