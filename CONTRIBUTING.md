# Contributing to High-Performance GPU Kernels (CUDA & Triton)

First off, thank you for considering contributing! We are building an industry-grade, cutting-edge open-source repository for high-performance GPU kernels in **CUDA C++**, **CUTLASS**, and **OpenAI Triton**.

Whether you work at an AI lab, semiconductor company, startup, or are an independent GPU hacker, your contributions are welcome.

---

## 🗺️ Kernel Roadmap & Wishlist

We actively invite PRs for the following targets:

| Area | Topic | Target Tech | Status |
| :--- | :--- | :--- | :--- |
| **Attention** | FlashDecoding (Paged KV-Cache Decode) | CUDA C++ / Triton | 🟡 Planned |
| **Attention** | FlashAttention-2 Backward Pass | CUDA / Triton | 🟡 Planned |
| **Quantization**| W4A16 GEMV (AWQ / GPTQ Dequantization) | CUDA C++ | 🟡 Wanted |
| **GEMM** | FP8 GEMM with E4M3 / E5M2 | sm_89 / sm_90 MMA | 🟡 Wanted |
| **GEMM** | Async TMA (Tensor Memory Accelerator) GEMM | sm_90 Hopper | 🟡 Wanted |
| **Loss** | Fused Cross-Entropy Loss (Online Softmax) | Triton / CUDA | 🟡 Wanted |
| **Norm** | Fused GroupNorm & InstanceNorm | Triton | 🟡 Wanted |

---

## 🛠️ Development Setup

### 1. Prerequisites
- Linux or Google Colab environment with an NVIDIA GPU (T4, A10, A100, H100, RTX 3090/4090)
- CUDA Toolkit 11.8+ or 12.0+
- Python 3.8+ with PyTorch 2.0+ and Triton 2.1+

### 2. Clone and Install in Editable Mode
```bash
git clone https://github.com/jacky/High-Performance-GPU-Kernel-CUDA-Triton-.git
cd High-Performance-GPU-Kernel-CUDA-Triton-

# Install dependencies
pip install -r requirements.txt

# Build and install PyTorch CUDA extension in editable mode
pip install -e .
```

---

## 📋 Kernel Development Checklist

Before submitting a Pull Request for a new GPU kernel, ensure:

1. **Numerical Correctness**:
   - Compare output against `torch` reference with `torch.allclose()`.
   - Test both standard shapes (e.g. powers of 2) and arbitrary odd dimensions (boundary masking).
2. **Benchmarking**:
   - Add a benchmark entry comparing against PyTorch Eager / cuBLAS / cuDNN.
   - Measure Runtime ($ms$), Throughput ($TFLOPS$), and Memory Bandwidth ($GB/s$).
3. **Hardware Efficiency**:
   - Profile with `ncu` (Nsight Compute) to verify zero unintentional bank conflicts and high SM/DRAM utilization.
4. **Code Quality**:
   - Format C++/CUDA code with `clang-format -i csrc/**/*.cu csrc/**/*.cuh`.
   - Format Python code with `black .` and `ruff check .`.
   - Add a unit test in `tests/test_<kernel_name>.py`.

---

## 🚀 Submitting a Pull Request

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/hopper-fp8-gemm
   ```
2. Commit your changes:
   ```bash
   git commit -m "feat(gemm): add sm_90 FP8 GEMM kernel with TMA support"
   ```
3. Push to your fork and submit a PR:
   ```bash
   git push origin feature/hopper-fp8-gemm
   ```
4. Include benchmark numbers and target GPU specs in the PR description!
