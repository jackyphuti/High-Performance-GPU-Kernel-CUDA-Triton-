# GPU Kernel Profiling Guide: Nsight Compute & Nsight Systems

To optimize custom GPU kernels to peak hardware limits, you must inspect hardware performance counters using **NVIDIA Nsight Compute (`ncu`)** and **NVIDIA Nsight Systems (`nsys`)**.

---

## 1. Speed of Light (SOL) Analysis

The **Speed of Light (SOL)** percentage shows what fraction of the hardware's theoretical maximum bandwidth or compute throughput your kernel is sustaining:

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active,gpu__dram_throughput.avg.pct_of_peak_sustained_active ./sgemm_t4
```

### Interpreting SOL Results:
- **SM Throughput > 75%, DRAM Throughput < 30%**:
  Kernel is **Compute-Bound**. You have achieved high arithmetic intensity! Optimizations should focus on instruction scheduling, register usage, and FMA pipe saturation.
- **DRAM Throughput > 75%, SM Throughput < 30%**:
  Kernel is **Memory-Bound**. Data transfers are saturating the memory bus. Optimizations should focus on shared memory tiling, data reuse, and compression.
- **Both SM and DRAM < 40%**:
  Kernel is **Latency-Bound** or suffering from low occupancy / warp stalls (e.g. excessive barrier synchronization `__syncthreads()`, shared memory bank conflicts, or instruction starvation).

---

## 2. Detecting Shared Memory Bank Conflicts

Shared memory has 32 banks ($4\text{ bytes}$ wide). If multiple threads in a warp access different words in the same bank, accesses are serialized.

Run the following command to measure bank conflicts:
```bash
ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
./sgemm_t4
```

### Remediation:
- If `op_ld.sum > 0`: Threads are reading conflicting banks.
- **Fix**: Pad the shared memory array by 1 element:
  ```cuda
  // Bank conflict prone if stride is a multiple of 32
  __shared__ float s_A[32][32];

  // Bank conflict free (offsets columns across banks)
  __shared__ float s_A[32][33];
  ```

---

## 3. Warp Scheduler Stall Analysis

When an SM cannot issue instructions for a warp, the warp is stalled. Profile the stall reasons:

```bash
ncu --metrics \
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
./sgemm_t4
```

| Stall Reason | Root Cause | Fix |
| :--- | :--- | :--- |
| **Long Scoreboard** | Waiting on global memory DRAM read (`LDG`) | Double-buffering, prefetching, or register tiling |
| **Short Scoreboard** | Waiting on shared memory load (`LDS`) | Unroll inner loops, tile in registers |
| **Barrier** | Stalled at `__syncthreads()` | Balance workload between threads, eliminate excessive barriers |
| **Math Pipe Throttle** | Compute pipeline is 100% full (Ideal!) | Kernel is saturating hardware arithmetic units |

---

## 4. End-to-End Tracing with Nsight Systems

To visualize kernel execution timeline, host-to-device memory copies (`cudaMemcpy`), and stream concurrency:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=timeline_report \
  --force-overwrite=true \
  python benchmarks/benchmark_gemm.py
```
Open `timeline_report.nsys-rep` in the NVIDIA Nsight Systems GUI to inspect execution queues.
