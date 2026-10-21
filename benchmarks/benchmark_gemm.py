"""
Comprehensive GEMM Benchmark Suite:
Custom CUDA vs OpenAI Triton vs NVIDIA cuBLAS (torch.matmul)
"""

import torch

try:
    import custom_kernels as ck
except ImportError:
    ck = None

from triton.gemm import matmul_triton


def benchmark_fn(fn, *args, warmup=10, rep=50):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / rep


def run_gemm_benchmarks():
    assert torch.cuda.is_available(), "CUDA device required"
    device_name = torch.cuda.get_device_name(0)
    print("=" * 105)
    print(f" GEMM BENCHMARK SUITE | Device: {device_name}")
    print("=" * 105)

    shapes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    header = f"{'Shape (M=N=K)':<18} | {'Backend':<22} | {'Time (ms)':<10} | {'TFLOPS':<10} | {'Bandwidth (GB/s)':<18} | {'Speedup':<8}"
    print(header)
    print("-" * 105)

    for M, N, K in shapes:
        a_f32 = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b_f32 = torch.randn((K, N), device="cuda", dtype=torch.float32)

        a_f16 = a_f32.to(torch.float16)
        b_f16 = b_f32.to(torch.float16)

        total_flops = 2.0 * M * N * K
        min_bytes = (M * K + K * N + M * N) * 4

        # 1. PyTorch cuBLAS (FP32)
        cublas_ms = benchmark_fn(torch.matmul, a_f32, b_f32)
        cublas_tflops = (total_flops * 1e-12) / (cublas_ms * 1e-3)
        cublas_bw = (min_bytes * 1e-9) / (cublas_ms * 1e-3)
        print(
            f"{M}x{N}x{K:<8} | {'cuBLAS (FP32)':<22} | {cublas_ms:<10.3f} | {cublas_tflops:<10.2f} | {cublas_bw:<18.1f} | {'1.00x':<8}"
        )

        # 2. OpenAI Triton (FP16)
        triton_ms = benchmark_fn(matmul_triton, a_f16, b_f16)
        triton_tflops = (total_flops * 1e-12) / (triton_ms * 1e-3)
        triton_bw = (min_bytes * 0.5 * 1e-9) / (triton_ms * 1e-3)
        triton_speedup = cublas_ms / triton_ms
        print(
            f"{'':<18} | {'Triton (FP16 Autotuned)':<22} | {triton_ms:<10.3f} | {triton_tflops:<10.2f} | {triton_bw:<18.1f} | {f'{triton_speedup:.2f}x':<8}"
        )

        # 3. Custom CUDA Kernels (if compiled)
        if ck is not None and ck.ops.HAS_CUDA_EXT:
            for name, backend in [
                ("CUDA Reg-Tiled (FP32)", "cuda_reg_tiled"),
                ("CUDA Shared-Tiled", "cuda_tiled"),
                ("CUDA Naive", "cuda_naive"),
            ]:
                if backend == "cuda_naive" and M > 2048:
                    continue  # Skip huge naive runs
                ms = benchmark_fn(ck.gemm, a_f32, b_f32, backend=backend)
                tflops = (total_flops * 1e-12) / (ms * 1e-3)
                bw = (min_bytes * 1e-9) / (ms * 1e-3)
                speedup = cublas_ms / ms
                print(
                    f"{'':<18} | {name:<22} | {ms:<10.3f} | {tflops:<10.2f} | {bw:<18.1f} | {f'{speedup:.2f}x':<8}"
                )

        print("-" * 105)


if __name__ == "__main__":
    run_gemm_benchmarks()
