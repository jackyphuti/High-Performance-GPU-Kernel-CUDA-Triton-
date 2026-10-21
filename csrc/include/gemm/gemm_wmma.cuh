#pragma once

#include "common.cuh"
#include <mma.h>
#include <cuda_fp16.h>

/**
 * Hardware Tensor Core GEMM using nvcuda::wmma
 * 
 * Target: Turing / Ampere / Ada / Hopper (sm_75+)
 * Tile configuration: 16x16x16 WMMA Tile per warp
 * Precision: FP16 Inputs (half), FP32 Accumulator (float)
 * Hardware Throughput: ~65 TFLOPS on Tesla T4
 */
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block computes a 32x32 tile of C using 4 warps (128 threads)
// Warp layout: 2x2 warps -> each warp computes a 16x16 WMMA tile
__global__ void wmma_gemm_kernel(int M, int N, int K, const half *A, const half *B, float *C) {
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 2; // 0 or 1
    int warpCol = warpId % 2; // 0 or 1

    int cRow = blockIdx.y * 32 + warpRow * WMMA_M;
    int cCol = blockIdx.x * 32 + warpCol * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        if (cRow < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + cRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + cCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

inline void launch_wmma_gemm(int M, int N, int K, const half *A, const half *B, float *C, cudaStream_t stream = 0) {
    dim3 block(32, 4); // 4 warps = 128 threads
    dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
    wmma_gemm_kernel<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
