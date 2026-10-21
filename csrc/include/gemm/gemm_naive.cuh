#pragma once

#include "common.cuh"

/**
 * Naive SGEMM baseline:
 * One thread calculates one element C[row, col].
 * Every multiply-accumulate loads directly from global memory.
 * Arithmetic intensity: 0.25 FLOP/byte (Severely memory bandwidth bound).
 */
__global__ void sgemm_naive_kernel(int M, int N, int K, const float *A, const float *B, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

inline void launch_sgemm_naive(int M, int N, int K, const float *A, const float *B, float *C, cudaStream_t stream = 0) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
    sgemm_naive_kernel<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
