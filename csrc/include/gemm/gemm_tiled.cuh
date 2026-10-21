#pragma once

#include "common.cuh"

/**
 * 2D Shared-Memory Tiled SGEMM:
 * Matrices A and B are partitioned into square tiles of size TILE_DIM x TILE_DIM.
 * Staged into __shared__ memory, reusing each loaded element TILE_DIM times.
 * Arithmetic intensity increases by TILE_DIMx (up to 8.0 FLOP/byte for TILE_DIM=32).
 */
template <int TILE_DIM = 32>
__global__ void sgemm_tiled_kernel(int M, int N, int K, const float *A, const float *B, float *C) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;
    int numTiles = ceil_div(K, TILE_DIM);

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

inline void launch_sgemm_tiled(int M, int N, int K, const float *A, const float *B, float *C, cudaStream_t stream = 0) {
    constexpr int TILE_DIM = 32;
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(ceil_div(N, TILE_DIM), ceil_div(M, TILE_DIM));
    sgemm_tiled_kernel<TILE_DIM><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
