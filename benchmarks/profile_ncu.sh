#!/usr/bin/env bash
# ==============================================================================
# NVIDIA Nsight Compute (ncu) Profiling Script for High-Performance GPU Kernels
# Analyzes:
# 1. Speed of Light (SOL): SM Compute vs DRAM Throughput
# 2. Warp Scheduler Stalls (Long Scoreboard, Short Scoreboard, Wait, Barrier)
# 3. Shared Memory Bank Conflicts (l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum)
# 4. Memory Coalescing & L1/L2 Cache Hit Rates
# ==============================================================================

set -euo pipefail

EXE=${1:-"./sgemm_t4"}

echo "========================================================================"
echo " Starting Nsight Compute Profiling on: ${EXE}"
echo "========================================================================"

# Profile Speed of Light (SOL) and Memory Workload Analysis
ncu \
  --set full \
  --target-processes all \
  --metrics \
sm__throughput.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  --export profile_report \
  --force-overwrite \
  "${EXE}"

echo "Profiling report generated: profile_report.ncu-rep"
echo "Open with NVIDIA Nsight Compute GUI or view metrics above."
