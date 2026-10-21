# Deep Architecture & GPU Kernel Engineering Guide

This document breaks down the core low-level GPU architectural concepts powering this repository.

---

## 1. NVIDIA GPU Memory Hierarchy

Every kernel optimization in this repository targets a specific layer in the physical GPU memory hierarchy:

```
[ GPU Registers: ~0-1 cycles latency | Aggregate Bandwidth: > 30 TB/s ]
                             │
[ L1 Cache / Shared Memory: ~19-30 cycles latency | Bandwidth: ~15-20 TB/s ]
                             │
[ L2 Cache (Unified): ~100-200 cycles latency | Bandwidth: ~3-6 TB/s ]
                             │
[ Global Device Memory (GDDR6 / HBM3): ~200-500 cycles latency | Bandwidth: 0.3 - 3.3 TB/s ]
```

### Key Latency and Bandwidth Realities:
- **Global Memory (DRAM/HBM)** is slow and narrow compared to compute. If a kernel requests data from global memory on every instruction, the arithmetic units sit idle waiting for memory responses (**Memory Stall**).
- **Shared Memory (SRAM)** is located directly on each Streaming Multiprocessor (SM). Staging data into shared memory allows all 1024 threads in a block to reuse cached elements, converting high-latency DRAM trips into single-digit nanosecond SRAM lookups.
- **Registers** are the fastest storage on the GPU. By caching small matrix sub-tiles ($8 \times 8$) in registers, threads perform outer products with zero memory bus arbitration.

---

## 2. Matrix Multiplication: From Memory-Bound to Compute-Bound

### Theoretical Arithmetic Intensity
For $C = A \times B$ where $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$:
- **FLOPs**: $2 M N K$
- **Theoretical Data Footprint**: $(M K + K N + M N) \times \text{sizeof(float)}$

### Why Naive SGEMM Fails:
In Naive GEMM, thread $(i, j)$ loops $K$ times, loading $A(i, k)$ and $B(k, j)$ from DRAM on every iteration.
$$\text{Memory Traffic}_{\text{naive}} = 2 M N K \times 4\text{ bytes}$$
$$\text{Arithmetic Intensity}_{\text{naive}} = \frac{2 M N K\text{ FLOPs}}{8 M N K\text{ Bytes}} = 0.25\text{ FLOP/Byte}$$

On an NVIDIA Tesla T4 ($320\text{ GB/s}$ memory bandwidth):
$$\text{Max Throughput}_{\text{naive}} = 320\text{ GB/s} \times 0.25\text{ FLOP/Byte} = \mathbf{80\text{ GFLOPS}} \quad (\approx 0.08\text{ TFLOPS})$$
The GPU is 99% idle!

---

### The 3-Tier Tiling Architecture

To achieve $>80\%$ of theoretical peak hardware performance, we apply multi-level tiling:

```
+--------------------------------------------------------------+
| 1. Grid Level: Global Matrix C (M x N)                      |
|    Partitioned across Thread Blocks into Block Tiles (BM x BN)|
+--------------------------------------------------------------+
                               │
+--------------------------------------------------------------+
| 2. Block Level (Shared Memory):                             |
|    Loads BM x BK from A and BK x BN from B into __shared__   |
|    Synchronizes threads with __syncthreads()                 |
+--------------------------------------------------------------+
                               │
+--------------------------------------------------------------+
| 3. Thread Level (Register Micro-Tiling):                    |
|    Each thread computes TM x TN (8 x 8 = 64 elements)        |
|    in private registers.                                     |
|    Issues float4 (128-bit) vectorized global loads.          |
+--------------------------------------------------------------+
```

---

## 3. FlashAttention & Online Softmax Mathematical Derivation

Standard attention computes:
$$S = \frac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = P V$$

Materializing $S$ and $P$ requires $O(N^2)$ memory storage. For $N=8192$, $N^2 \times 2\text{ bytes} \approx 134\text{ MB}$ per head per batch. Across 32 heads and batch size 4, this exceeds $17\text{ GB}$ of DRAM allocations!

### The Online Softmax Algorithm (Milakov & Gimelshein / Tri Dao)
Instead of waiting for the full row of $S$ to calculate $\max(x_i)$ and $\sum e^{x_i - \max}$, we update running statistics block-by-block.

Let block $1$ have max $m^{(1)}$ and sum $l^{(1)} = \sum_{j} e^{S_{ij} - m^{(1)}}$.
When block $2$ arrives with local max $\tilde{m}$ and local sum $\tilde{l}$:
1. **New Global Maximum**:
   $$m^{(2)} = \max(m^{(1)}, \tilde{m})$$
2. **Rescaling Factor**:
   $$\alpha = e^{m^{(1)} - m^{(2)}}, \quad \tilde{\alpha} = e^{\tilde{m} - m^{(2)}}$$
3. **New Global Normalizer**:
   $$l^{(2)} = l^{(1)} \alpha + \tilde{l} \tilde{\alpha}$$
4. **Running Output Update**:
   $$O^{(2)} = O^{(1)} \alpha + \tilde{P}_{\text{unnorm}} V_2$$

At the final block, the true normalized output is simply:
$$O_{\text{final}} = \frac{O^{(K)}}{l^{(K)}}$$

**Result**: We compute exact attention in $O(N)$ memory directly inside GPU SRAM without ever writing the $N \times N$ matrix to DRAM!

---

## 4. OpenAI Triton Compilation Model

Why does OpenAI Triton match or outperform hand-written CUDA for many workloads?
1. **Block-Level Abstraction**: Triton operates at the block level rather than the thread level. The compiler handles warp assignment, register allocation, and barrier synchronizations automatically.
2. **Block Swizzling**:
   Standard row-major block execution causes cache evictions. Triton swizzles `program_id` in 2D groups (`GROUP_M`), ensuring that blocks scheduled concurrently share rows of $A$ and columns of $B$ in L2 cache.
3. **Automatic Software Pipelining (`num_stages=3/4`)**:
   Triton pipelines global memory loads and tensor core computations across multiple stages, hiding load latency behind previous stage FMAs.
