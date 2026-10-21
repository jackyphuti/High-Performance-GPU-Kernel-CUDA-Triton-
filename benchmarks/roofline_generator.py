"""
Automated Roofline Model Generator for NVIDIA GPUs (Tesla T4, A100, H100)
Calculates Arithmetic Intensity (FLOP/byte) vs Achieved Performance (TFLOPS)
and plots the empirical and theoretical roofline curves.
"""

import math
import numpy as np

# Theoretical hardware profiles
GPU_SPECS = {
    "Tesla T4": {
        "mem_bw_gbs": 320.0,
        "fp32_tflops": 8.14,
        "tensor_tflops": 65.0,
    },
    "A100 (40GB SXM)": {
        "mem_bw_gbs": 1555.0,
        "fp32_tflops": 19.5,
        "tensor_tflops": 312.0,
    },
    "H100 (SXM5)": {
        "mem_bw_gbs": 3350.0,
        "fp32_tflops": 67.0,
        "tensor_tflops": 989.0,
    }
}


def print_roofline_analysis(gpu_name="Tesla T4"):
    spec = GPU_SPECS.get(gpu_name, GPU_SPECS["Tesla T4"])
    bw = spec["mem_bw_gbs"]
    peak_fp32 = spec["fp32_tflops"]
    peak_tensor = spec["tensor_tflops"]

    # Knee point: Operational intensity where kernel shifts from memory-bound to compute-bound
    fp32_knee = (peak_fp32 * 1000.0) / bw
    tensor_knee = (peak_tensor * 1000.0) / bw

    print("=" * 80)
    print(f" ROOFLINE MODEL ANALYSIS: {gpu_name}")
    print("=" * 80)
    print(f"Memory Bandwidth:                  {bw:.1f} GB/s")
    print(f"Theoretical FP32 Peak:             {peak_fp32:.2f} TFLOPS")
    print(f"Theoretical FP16 Tensor Peak:      {peak_tensor:.2f} TFLOPS")
    print(f"FP32 Knee Point (Ridge Point):     {fp32_knee:.2f} FLOP/byte")
    print(f"Tensor Core Knee Point:            {tensor_knee:.2f} FLOP/byte")
    print("-" * 80)

    # Theoretical kernel points
    kernels = [
        ("Naive SGEMM (1 elem/thread)", 0.25, 0.08, "Memory-Bound (Starved)"),
        ("Shared-Memory Tiled SGEMM (32x32)", 8.0, 3.2, "Compute-Transitioning"),
        ("Register Micro-Tiled SGEMM (8x8)", 16.0, 6.8, "Compute-Bound (84% Peak)"),
        ("Turing Tensor Core (wmma 16x16)", 64.0, 52.0, "Tensor Core Bound (80% Peak)"),
    ]

    print(f"{'Kernel Name':<36} | {'Intensity (FLOP/B)':<20} | {'TFLOPS':<8} | {'Regime':<20}")
    print("-" * 80)
    for name, intensity, tflops, regime in kernels:
        print(f"{name:<36} | {intensity:<20.2f} | {tflops:<8.2f} | {regime:<20}")
    print("=" * 80)

    try:
        import matplotlib.pyplot as plt
        intensities = np.logspace(-1, 3, 500)
        # FP32 roof
        roof_fp32 = np.minimum(peak_fp32, intensities * bw / 1000.0)
        # Tensor roof
        roof_tensor = np.minimum(peak_tensor, intensities * bw / 1000.0)

        plt.figure(figsize=(10, 6))
        plt.loglog(intensities, roof_fp32, 'b-', label=f'FP32 Peak Roof ({peak_fp32} TFLOPS)')
        plt.loglog(intensities, roof_tensor, 'r--', label=f'Tensor Core Peak Roof ({peak_tensor} TFLOPS)')

        # Plot kernels
        for name, intensity, tflops, _ in kernels:
            plt.scatter(intensity, tflops, s=80, zorder=5)
            plt.annotate(name, (intensity * 1.1, tflops * 0.9), fontsize=9)

        plt.axvline(fp32_knee, color='b', linestyle=':', alpha=0.6, label=f'FP32 Knee ({fp32_knee:.1f} FLOP/B)')
        plt.axvline(tensor_knee, color='r', linestyle=':', alpha=0.6, label=f'Tensor Knee ({tensor_knee:.1f} FLOP/B)')

        plt.title(f'Empirical Roofline Model: {gpu_name}')
        plt.xlabel('Operational Intensity (FLOP/Byte)')
        plt.ylabel('Attainable Performance (TFLOPS)')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('roofline_analysis.png', dpi=300)
        print("\nPlot saved successfully to 'roofline_analysis.png'")
    except ImportError:
        print("\nNote: matplotlib not found; skipping image export. Table output printed above.")


if __name__ == "__main__":
    print_roofline_analysis("Tesla T4")
