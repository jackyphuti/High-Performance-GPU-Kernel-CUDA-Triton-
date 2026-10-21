#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * Robust CUDA error checking macro.
 */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA Error at %s:%d -> %s (%s)\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err),         \
                         cudaGetErrorName(err));                              \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                   \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA Kernel Launch Error at %s:%d -> %s\n", \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

/**
 * Ceil division helper for grid calculation.
 */
inline __host__ __device__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * Warp-level reduction sum using shuffle down instructions.
 */
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp-level reduction max using shuffle down instructions.
 */
template <typename T>
__inline__ __device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}
