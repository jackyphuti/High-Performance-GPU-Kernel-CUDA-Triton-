# Contributing to High-Performance GPU Kernels (CUDA and Triton)

Thank you for your interest in contributing to this repository. This project aims to maintain production-grade, low-level implementations in **CUDA C++**, **CUTLASS**, and **OpenAI Triton**.

Contributions from industry engineers, academic researchers, and systems developers are welcome.

---

## Kernel Roadmap and Wishlist

Pull requests are actively encouraged for the following milestones:

| Area | Topic | Target Technology | Status |
| :--- | :--- | :--- | :--- |
| **Attention** | FlashDecoding (Paged KV-Cache Decode) | CUDA C++ / Triton | Planned |
| **Attention** | FlashAttention-2 Backward Pass | CUDA / Triton | Planned |
| **Quantization**| W4A16 GEMV (AWQ / GPTQ Dequantization) | CUDA C++ | Open for Contribution |
| **GEMM** | FP8 GEMM with E4M3 / E5M2 | sm_89 / sm_90 MMA | Open for Contribution |
| **GEMM** | Async TMA (Tensor Memory Accelerator) GEMM | sm_90 Hopper | Open for Contribution |
| **Loss** | Fused Cross-Entropy Loss (Online Softmax) | Triton / CUDA | Open for Contribution |
| **Norm** | Fused GroupNorm and InstanceNorm | Triton | Open for Contribution |

---

## Development Setup

### 1. Prerequisites
- Linux or Google Colab environment with an NVIDIA GPU (Tesla T4, A10, A100, H100, RTX 3090/4090)
- CUDA Toolkit 11.8+ or 12.0+
- Python 3.8+ with PyTorch 2.0+ and Triton 2.1+

### 2. Clone and Install in Editable Mode
```bash
git clone https://github.com/jackyphuti/High-Performance-GPU-Kernel-CUDA-Triton-.git
cd High-Performance-GPU-Kernel-CUDA-Triton-

# Install dependencies
pip install -r requirements.txt

# Build and install PyTorch CUDA extension in editable mode
pip install -e .
```

---

## Kernel Development Checklist

Before submitting a Pull Request for a new GPU kernel, ensure the following requirements are met:

1. **Numerical Correctness**:
   - Compare outputs against reference PyTorch operations using `torch.allclose()`.
   - Test both power-of-two dimensions and non-aligned arbitrary matrix shapes.
2. **Benchmarking**:
   - Include benchmark comparisons against PyTorch Eager, cuBLAS, or cuDNN.
   - Report Runtime ($ms$), Throughput ($TFLOPS$), and Effective Memory Bandwidth ($GB/s$).
3. **Hardware Efficiency**:
   - Profile with NVIDIA Nsight Compute (`ncu`) to ensure high compute/memory utilization and absence of unintended shared memory bank conflicts.
4. **Code Quality**:
   - Format Python source files using `black .` and `ruff check .`.
   - Maintain C++/CUDA adherence to standard modern CUDA guidelines.
   - Include unit tests under the `tests/` directory.

---

## Submitting a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/hopper-fp8-gemm
   ```
2. Commit your changes with descriptive messages:
   ```bash
   git commit -m "feat(gemm): add sm_90 FP8 GEMM kernel with TMA support"
   ```
3. Push to your fork and submit a Pull Request:
   ```bash
   git push origin feature/hopper-fp8-gemm
   ```
4. Detail your benchmark results, profiling metrics, and hardware specifications in the Pull Request description.
