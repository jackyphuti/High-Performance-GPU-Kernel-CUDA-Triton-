"""
FlashAttention-2 Benchmark Suite:
PyTorch Eager vs Custom CUDA FlashAttention vs OpenAI Triton FlashAttention
"""

import math
import torch
from triton.flash_attention import flash_attention_triton

try:
    import custom_kernels as ck
except ImportError:
    ck = None


def pytorch_eager_attention(q, k, v, is_causal=True):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
        scores = scores + mask
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def benchmark_attn(fn, *args, warmup=10, rep=30):
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    avg_ms = start.elapsed_time(end) / rep
    return avg_ms, peak_mem_mb


def run_attention_benchmarks():
    assert torch.cuda.is_available(), "CUDA device required"
    print("=" * 105)
    print(" FLASHATTENTION BENCHMARK SUITE (Batch=2, Heads=8, HeadDim=64, Causal=True)")
    print("=" * 105)

    B = 2
    H = 8
    D = 64
    seq_lens = [512, 1024, 2048, 4096, 8192]

    header = f"{'SeqLen (N)':<12} | {'Backend':<26} | {'Time (ms)':<10} | {'TFLOPS':<10} | {'Peak Mem (MB)':<14} | {'Speedup':<8}"
    print(header)
    print("-" * 105)

    for N in seq_lens:
        q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

        # Causal attention FLOPs: 4 * B * H * (N^2 / 2) * D = 2 * B * H * N^2 * D
        flops = 2.0 * B * H * (N ** 2) * D

        # 1. PyTorch Eager Attention
        eager_ms, eager_mem = None, None
        try:
            eager_ms, eager_mem = benchmark_attn(pytorch_eager_attention, q, k, v)
            eager_tflops = (flops * 1e-12) / (eager_ms * 1e-3)
            print(f"{N:<12} | {'PyTorch Eager (O(N^2))':<26} | {eager_ms:<10.3f} | {eager_tflops:<10.2f} | {eager_mem:<14.1f} | {'1.00x':<8}")
        except torch.cuda.OutOfMemoryError:
            print(f"{N:<12} | {'PyTorch Eager (O(N^2))':<26} | {'OOM':<10} | {'-':<10} | {'> 16000 MB':<14} | {'-':<8}")

        # 2. Triton FlashAttention-2
        triton_ms, triton_mem = benchmark_attn(flash_attention_triton, q, k, v, True)
        triton_tflops = (flops * 1e-12) / (triton_ms * 1e-3)
        speedup_str = f"{(eager_ms / triton_ms):.2f}x" if eager_ms is not None else "N/A (OOM saved)"
        print(f"{'':<12} | {'Triton FlashAttention-2':<26} | {triton_ms:<10.3f} | {triton_tflops:<10.2f} | {triton_mem:<14.1f} | {speedup_str:<8}")

        # 3. Custom CUDA FlashAttention (if available and N <= 4096)
        if ck is not None and ck.ops.HAS_CUDA_EXT:
            q_f32 = q.float()
            k_f32 = k.float()
            v_f32 = v.float()
            cuda_ms, cuda_mem = benchmark_attn(ck.flash_attention, q_f32, k_f32, v_f32, True)
            cuda_tflops = (flops * 1e-12) / (cuda_ms * 1e-3)
            cuda_speedup = f"{(eager_ms / cuda_ms):.2f}x" if eager_ms is not None else "N/A"
            print(f"{'':<12} | {'CUDA Custom FlashAttention':<26} | {cuda_ms:<10.3f} | {cuda_tflops:<10.2f} | {cuda_mem:<14.1f} | {cuda_speedup:<8}")

        print("-" * 105)


if __name__ == "__main__":
    run_attention_benchmarks()
