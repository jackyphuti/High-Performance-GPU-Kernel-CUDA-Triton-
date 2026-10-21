#pragma once

#include "common.cuh"

/**
 * Fused SwiGLU Activation Kernel
 * 
 * Formula: SwiGLU(x, y) = (x * sigmoid(x)) * y
 * Commonly used in LLaMA, Mistral, Gemma, and DeepSeek MLP architectures.
 * 
 * Inputs:
 * - In: Packed input tensor (N, 2 * D) where first half is x and second half is y
 * - Out: Output tensor (N, D)
 * Optimized with 128-bit vectorized float4 operations.
 */

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void swiglu_kernel(
    const float *__restrict__ In,
    float *__restrict__ Out,
    int total_elements, // N * D
    int D
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < total_elements) {
        int row = idx / D;
        int col = idx % D;

        // In is shape [N, 2D]
        const float *in_row = In + row * (2 * D);
        float4 x_vec = *reinterpret_cast<const float4*>(in_row + col);
        float4 y_vec = *reinterpret_cast<const float4*>(in_row + D + col);

        float4 out_vec;
        out_vec.x = silu(x_vec.x) * y_vec.x;
        out_vec.y = silu(x_vec.y) * y_vec.y;
        out_vec.z = silu(x_vec.z) * y_vec.z;
        out_vec.w = silu(x_vec.w) * y_vec.w;

        *reinterpret_cast<float4*>(Out + idx) = out_vec;
    } else {
        // Scalar boundary fallback
        for (int i = 0; i < 4 && (idx + i) < total_elements; ++i) {
            int cur_idx = idx + i;
            int row = cur_idx / D;
            int col = cur_idx % D;
            float x = In[row * (2 * D) + col];
            float y = In[row * (2 * D) + D + col];
            Out[cur_idx] = silu(x) * y;
        }
    }
}

inline void launch_swiglu(
    const float *In, float *Out,
    int N, int D,
    cudaStream_t stream = 0
) {
    int total_elements = N * D;
    int threads = 256;
    int blocks = ceil_div(ceil_div(total_elements, 4), threads);
    swiglu_kernel<<<blocks, threads, 0, stream>>>(In, Out, total_elements, D);
}
