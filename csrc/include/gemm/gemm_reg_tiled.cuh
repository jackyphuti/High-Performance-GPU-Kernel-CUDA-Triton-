#pragma once

#include "common.cuh"

/**
 * 2D Register-Tiled (Thread-Tiled / Micro-Tiled) SGEMM
 * 
 * Architectural Hierarchy:
 * - Block Tile: BM=128, BN=128, BK=8
 * - Thread Tile: TM=8, TN=8 (each thread maintains an 8x8 sub-tile in private registers)
 * - Threads per Block: (128/8) * (128/8) = 16 * 16 = 256 threads (8 warps)
 * - Memory Traffic: Coalesced 128-bit vectorized float4 global loads.
 * - Compute Intensity: 64 FMAs per 16 shared memory loads (4 FMAs per smem load).
 */
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void sgemm_reg_tiled_kernel(int M, int N, int K, const float *A, const float *B, float *C) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int threadCol = threadIdx.x; // 0..15
    int threadRow = threadIdx.y; // 0..15

    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Vectorized mapping: 256 threads loading 1024 floats (1 float4 per thread)
    int loadA_row = tid / 2;
    int loadA_col = (tid % 2) * 4;
    int loadB_row = tid / 32;
    int loadB_col = (tid % 32) * 4;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // 1. Cooperative Vectorized Load A -> Shared Memory
        int currA_row = blockRow + loadA_row;
        int currA_col = bkIdx + loadA_col;
        if (currA_row < M && (currA_col + 3) < K) {
            float4 valA = *reinterpret_cast<const float4*>(&A[currA_row * K + currA_col]);
            s_A[loadA_row][loadA_col + 0] = valA.x;
            s_A[loadA_row][loadA_col + 1] = valA.y;
            s_A[loadA_row][loadA_col + 2] = valA.z;
            s_A[loadA_row][loadA_col + 3] = valA.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s_A[loadA_row][loadA_col + i] = (currA_row < M && (currA_col + i) < K) 
                                                ? A[currA_row * K + (currA_col + i)] : 0.0f;
            }
        }

        // 2. Cooperative Vectorized Load B -> Shared Memory
        int currB_row = bkIdx + loadB_row;
        int currB_col = blockCol + loadB_col;
        if (currB_row < K && (currB_col + 3) < N) {
            float4 valB = *reinterpret_cast<const float4*>(&B[currB_row * N + currB_col]);
            s_B[loadB_row][loadB_col + 0] = valB.x;
            s_B[loadB_row][loadB_col + 1] = valB.y;
            s_B[loadB_row][loadB_col + 2] = valB.z;
            s_B[loadB_row][loadB_col + 3] = valB.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s_B[loadB_row][loadB_col + i] = (currB_row < K && (currB_col + i) < N) 
                                                ? B[currB_row * N + (currB_col + i)] : 0.0f;
            }
        }

        __syncthreads();

        // 3. Compute outer product into private thread registers
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = s_A[threadRow * TM + i][dotIdx];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = s_B[dotIdx][threadCol * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // 4. Store 8x8 sub-tile to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = blockRow + threadRow * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = blockCol + threadCol * TN + j;
            if (r < M && c < N) {
                C[r * N + c] = regC[i][j];
            }
        }
    }
}

inline void launch_sgemm_reg_tiled(int M, int N, int K, const float *A, const float *B, float *C, cudaStream_t stream = 0) {
    dim3 block(BN / TN, BM / TM); // (16, 16) = 256 threads
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_reg_tiled_kernel<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
