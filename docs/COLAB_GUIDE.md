# Google Colab Setup and Execution Guide

This guide walks through configuring and running the CUDA C++ and OpenAI Triton kernels on a free **NVIDIA Tesla T4 GPU** in Google Colab.

---

## 1. Quickstart: Open the Walkthrough Notebook

You can open the pre-built notebook directly in Google Colab using either of the following approaches:

### Option A: Open directly via GitHub URL
1. Go to [Google Colab](https://colab.research.google.com).
2. Click on the **GitHub** tab.
3. Enter the repository URL:
   ```text
   https://github.com/jackyphuti/High-Performance-GPU-Kernel-CUDA-Triton-
   ```
4. Select [`notebooks/t4_sgemm_walkthrough.ipynb`](notebooks/t4_sgemm_walkthrough.ipynb).

### Option B: Upload Manually
1. Download `notebooks/t4_sgemm_walkthrough.ipynb` from the repository.
2. In Google Colab, select **File -> Upload notebook** and upload the file.

---

## 2. Enabling the NVIDIA Tesla T4 GPU

By default, newly created Colab notebooks run on a CPU-only instance. To attach an NVIDIA GPU:

1. In the top navigation bar, click **Runtime**.
2. Select **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.

Wait 5 to 10 seconds while Google Colab allocates a GPU virtual machine.

---

## 3. Verifying the GPU Environment

In the first cell of your notebook, execute:

```bash
!nvidia-smi
```

### Expected Output:
```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05     CUDA Version: 12.2     |
|-----------------------------------+------------------------+----------------------------+
| GPU  Name                  Mode   | Bus-Id          Disp.A | Volatile Uncorr. ECC       |
| Fan  Temp   Perf          Pwr:Usage/Cap | Memory-Usage     | GPU-Util  Compute M.       |
|===================================+========================+============================|
|   0  Tesla T4              Off    | 00000000:00:04.0   Off |                          0 |
| N/A   42C    P8              9W /  70W |      0MiB / 15360MiB |      0%      Default    |
+-----------------------------------+------------------------+----------------------------+
```

Verify that the CUDA compiler (`nvcc`) is available:

```bash
!nvcc --version
```

---

## 4. Method 1: Running Standalone CUDA Benchmarks

To compile and benchmark the self-contained SGEMM suite directly from a Colab cell:

### Step 1: Write the Source Code
Run the `%%writefile sgemm_t4.cu` cell in the notebook. This generates the CUDA C++ file containing:
- Baseline Naive SGEMM
- 2D Shared-Memory Tiled SGEMM ($32 \times 32$)
- 2D Register Micro-Tiled SGEMM ($128 \times 128$ block, $8 \times 8$ thread tile, `float4`)
- Turing Hardware Tensor Cores (`nvcuda::wmma`)
- CUDA Event timing harness and CPU verification

### Step 2: Compile with Architecture Flag `sm_75`
```bash
!nvcc -arch=sm_75 -O3 sgemm_t4.cu -o sgemm_t4
```

### Step 3: Run the Benchmark Executable
```bash
!./sgemm_t4
```

### Expected Results on Tesla T4 ($M = N = K = 2048$):
```text
=======================================================================================================
Starting Kernel Benchmarks (Averaged over 20 iterations):
-------------------------------------------------------------------------------------------------------
1. Naive SGEMM                   | Time:  248.512 ms | Throughput:    0.07 TFLOPS (    69.1 GFLOPS) | Bandwidth:    3.5 GB/s
Verification PASSED (Max absolute difference: 0.00014)
-------------------------------------------------------------------------------------------------------
2. Shared Memory Tiled SGEMM     | Time:    6.012 ms | Throughput:    2.86 TFLOPS (  2857.3 GFLOPS) | Bandwidth:   14.3 GB/s
Verification PASSED (Max absolute difference: 0.00018)
-------------------------------------------------------------------------------------------------------
3. Register-Tiled SGEMM (8x8)    | Time:    2.709 ms | Throughput:    6.35 TFLOPS (  6348.5 GFLOPS) | Bandwidth:   31.7 GB/s
Verification PASSED (Max absolute difference: 0.00021)
-------------------------------------------------------------------------------------------------------
4. Turing Tensor Cores (WMMA)    | Time:    0.350 ms | Throughput:   49.10 TFLOPS ( 49102.3 GFLOPS) | Bandwidth:  245.5 GB/s
Verification PASSED (Max absolute difference: 0.0124)
-------------------------------------------------------------------------------------------------------
```

---

## 5. Method 2: Cloning the Full Repository in Colab

If you want to run the full PyTorch CUDA extension, OpenAI Triton kernels, and benchmark suites, run the following commands inside a notebook cell:

```bash
# Clone the repository
!git clone https://github.com/jackyphuti/High-Performance-GPU-Kernel-CUDA-Triton-.git
%cd High-Performance-GPU-Kernel-CUDA-Triton-

# Install dependencies
!pip install -r requirements.txt

# Compile PyTorch CUDA extension
!pip install -e .
```

### Running the GEMM Benchmark Suite
Compares custom CUDA kernels and OpenAI Triton against `torch.matmul` (cuBLAS):
```bash
!python benchmarks/benchmark_gemm.py
```

### Running the FlashAttention-2 Benchmark Suite
Demonstrates $O(N^2)$ PyTorch Eager memory scaling vs $O(N)$ FlashAttention:
```bash
!python benchmarks/benchmark_attention.py
```

### Running Unit Tests
Verifies numerical equivalence against PyTorch reference implementations:
```bash
!pytest tests/
```

---

## 6. Troubleshooting Common Colab Errors

### Error: `/bin/bash: line 1: nvidia-smi: command not found`
- **Cause**: The notebook is operating in CPU mode.
- **Solution**: Follow Section 2 above to switch the runtime to **T4 GPU** via **Runtime -> Change runtime type**.

### Error: `/bin/bash: line 1: nvcc: command not found`
- **Cause**: The CUDA binary path is missing from your shell environment PATH variable.
- **Solution**: Add the CUDA binary directory to PATH before compiling:
  ```bash
  !export PATH=/usr/local/cuda/bin:$PATH && nvcc -arch=sm_75 -O3 sgemm_t4.cu -o sgemm_t4 && ./sgemm_t4
  ```

### First-Time Triton Kernel Latency
- **Observation**: The first execution of an OpenAI Triton kernel takes 1–3 seconds longer than subsequent runs.
- **Reason**: Triton compiles JIT kernels to PTX and machine code on the first invocation and caches the compiled binary. Warmup iterations in the benchmark suite handle this automatically.

### GPU Memory Disconnection / CUDA Out of Memory
- **Limit**: Free Google Colab instances provide 15 GB of GPU VRAM and a 12-hour session limit.
- **Solution**: If testing matrix dimensions larger than $N = 8192$, monitor VRAM usage with `!nvidia-smi` and avoid allocating multiple unreferenced matrices simultaneously.
