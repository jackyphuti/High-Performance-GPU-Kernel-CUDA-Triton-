/**
 * High-Performance SGEMM on NVIDIA Tesla T4 (Turing sm_75)
 * 
 * Features:
 * 1. Naive SGEMM baseline
 * 2. 2D Shared-Memory Tiled SGEMM (TILE_DIM = 32)
 * 3. 2D Register-Tiled (Thread-Tiled) SGEMM (128x128 block, 8x8 thread tile, float4 vectorized)
 * 4. Turing Hardware Tensor Core GEMM (nvcuda::wmma, FP16 inputs, FP32 accumulator)
 * 5. Full host harness with CUDA Event timers and CPU verification
 *
 * Compilation on Google Colab / Linux with T4:
 *   nvcc -arch=sm_75 -O3 sgemm_t4.cu -o sgemm_t4
 *   ./sgemm_t4
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ============================================================================
// 1. NAIVE SGEMM
// Grid:  (ceil(N / 32), ceil(M / 32))
// Block: (32, 32) -> 1024 threads
// Each thread computes 1 element of C.
// Arithmetic intensity: 2 FLOPs / (2 * 4 bytes) = 0.25 FLOP/byte (Memory-bound)
// ============================================================================
__global__ void sgemm_naive(int M, int N, int K, const float *A, const float *B, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= col < N
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 <= row < M

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row, k] is a broadcast read across threads in the same row
            // B[k, col] is a coalesced read across adjacent threads in a warp
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 2. 2D SHARED-MEMORY TILED SGEMM
// Tile Size: 32x32
// Block: (32, 32) -> 1024 threads
// Shared Memory: 2 * (32 * 32 * 4) = 8,192 bytes (8 KB)
// Arithmetic intensity: 0.25 * 32 = 8.0 FLOP/byte (32x reduction in DRAM reads)
// ============================================================================
template <int TILE_DIM>
__global__ void sgemm_shared_tiled(int M, int N, int K, const float *A, const float *B, float *C) {
    // Statically allocated shared memory for tiles of A and B
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // Cooperative global memory loading into shared memory
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Synchronize to ensure all threads have finished loading tiles into shared memory
        __syncthreads();

        // Accumulate partial dot-products from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            // s_B[k][tx] is conflict-free: thread tx in warp reads bank tx
            // s_A[ty][k] is a shared memory broadcast: all threads in the same row read same address
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Synchronize before writing next tile into shared memory (prevents RAW / WAR hazards)
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 3. 2D REGISTER-TILED (MICRO-TILED) SGEMM WITH VECTORIZED LOADS
// Block Tile: BM=128, BN=128, BK=8
// Thread Tile: TM=8, TN=8 (each thread calculates an 8x8 sub-tile in registers)
// Threads per block: (128/8) * (128/8) = 16 * 16 = 256 threads (8 warps)
// Shared memory: (128*8 + 8*128) * 4 bytes = 8 KB
// Memory traffic: Each thread loads 4 floats from A and 4 from B using float4
// Arithmetic intensity: (8*8 FMAs) / (8 loads from s_A + 8 loads from s_B) = 4 FMAs per smem load!
// ============================================================================
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void sgemm_register_tiled(int M, int N, int K, const float *A, const float *B, float *C) {
    // 256 threads per block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Shared memory buffers
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // Thread coordinates within the block for computing C
    int threadCol = threadIdx.x; // 0..15
    int threadRow = threadIdx.y; // 0..15

    // Register accumulation buffer for 8x8 tile
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // Block starting offsets in C
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Loading mapping:
    // A tile is BM x BK = 128 x 8 = 1024 floats. 256 threads -> 4 floats per thread.
    // We map each thread to load 1 float4 (4 floats)
    int loadA_row = tid / 2;         // 0..127
    int loadA_col = (tid % 2) * 4;   // 0 or 4

    // B tile is BK x BN = 8 x 128 = 1024 floats. 256 threads -> 4 floats per thread.
    int loadB_row = tid / 32;        // 0..7
    int loadB_col = (tid % 32) * 4;  // 0, 4, 8, ..., 124

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // 1. Cooperative Vectorized Load A -> Shared Memory using float4
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

        // 2. Cooperative Vectorized Load B -> Shared Memory using float4
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

        // 3. Compute outer products from Shared Memory into Registers
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load TM elements from s_A into thread-private registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = s_A[threadRow * TM + i][dotIdx];
            }

            // Load TN elements from s_B into thread-private registers
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = s_B[dotIdx][threadCol * TN + j];
            }

            // Compute TM x TN = 64 FMAs entirely in registers!
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

    // 4. Store accumulated results to global memory C
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

// ============================================================================
// 4. HARDWARE TENSOR CORE GEMM (nvcuda::wmma) ON TURING sm_75
// Shape: 16x16x16 WMMA Tile per warp
// Precision: FP16 Inputs (half), FP32 Accumulator (float)
// Turing T4 peak: ~65 TFLOPS
// ============================================================================
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block computes 32x32 of C using 4 warps (128 threads)
// Warp grid: 2x2 warps -> each warp computes a 16x16 WMMA tile
__global__ void wmma_gemm_t4(int M, int N, int K, const half *A, const half *B, float *C) {
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 2; // 0 or 1
    int warpCol = warpId % 2; // 0 or 1

    int cRow = blockIdx.y * 32 + warpRow * WMMA_M;
    int cCol = blockIdx.x * 32 + warpCol * WMMA_N;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in chunks of WMMA_K (16)
    for (int k = 0; k < K; k += WMMA_K) {
        if (cRow < M && k < K) {
            // Load matrix tiles into fragments
            wmma::load_matrix_sync(a_frag, A + cRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + cCol, N);

            // Execute hardware Tensor Core Matrix Multiply-Accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store the resulting 16x16 tile to global memory
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// ============================================================================
// CPU REFERENCE AND VERIFICATION HARNESS
// ============================================================================
void cpu_gemm(int M, int N, int K, const float *A, const float *B, float *C) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

bool verify_matrix(int M, int N, const float *expected, const float *actual, float tol = 1e-3f) {
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            std::cout << "Mismatch at index " << i << " -> expected: " << expected[i] 
                      << ", actual: " << actual[i] << ", diff: " << diff << std::endl;
            return false;
        }
    }
    std::cout << "Verification PASSED (Max absolute difference: " << max_diff << ")" << std::endl;
    return true;
}

// Helper to convert float to half on CPU
void float2half_cpu(int size, const float *src, half *dst) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

// ============================================================================
// BENCHMARKING HARNESS
// ============================================================================
int main() {
    // Matrix dimensions (Multiples of 128 for optimal hardware alignment)
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    std::cout << "========================================================\n";
    std::cout << "   GPU Performance Engineering: Matrix Multiplication\n";
    std::cout << "   Target Architecture: NVIDIA Tesla T4 (sm_75, Turing)\n";
    std::cout << "   Problem Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "========================================================\n\n";

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Host memory allocation
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    // Initialize with normalized random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    // Quick CPU reference on a 512x512 submatrix for fast verification
    std::cout << "Computing CPU reference on 512x512 submatrix for correctness check...\n";
    const int VERIF_DIM = 512;
    cpu_gemm(VERIF_DIM, VERIF_DIM, VERIF_DIM, h_A.data(), h_B.data(), h_C_ref.data());
    std::cout << "CPU reference complete.\n\n";

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // Prepare FP16 buffers for Tensor Cores
    half *d_A_half, *d_B_half;
    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    float2half_cpu(M * K, h_A.data(), h_A_half.data());
    float2half_cpu(K * N, h_B.data(), h_B_half.data());

    CUDA_CHECK(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_half, K * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A_half, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_half, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // CUDA Timing Events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int WARMUP_ITERS = 5;
    const int BENCH_ITERS = 20;
    double total_flops = 2.0 * static_cast<double>(M) * N * K;

    auto benchmark_kernel = [&](const char *name, auto launch_func, bool is_wmma = false) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            launch_func();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < BENCH_ITERS; ++i) {
            launch_func();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCH_ITERS;

        // Calculate GFLOPS and TFLOPS
        double gflops = (total_flops * 1e-9) / (avg_ms * 1e-3);
        double tflops = gflops / 1000.0;

        // Theoretical memory traffic (min A + B + C read/written once)
        double min_bytes = (static_cast<double>(M) * K + static_cast<double>(K) * N + static_cast<double>(M) * N) * sizeof(float);
        double effective_bandwidth = (min_bytes * 1e-9) / (avg_ms * 1e-3);

        std::cout << std::left << std::setw(32) << name 
                  << " | Time: " << std::fixed << std::setprecision(3) << std::setw(8) << avg_ms << " ms"
                  << " | Throughput: " << std::setw(7) << std::setprecision(2) << tflops << " TFLOPS"
                  << " (" << std::setw(8) << std::setprecision(1) << gflops << " GFLOPS)"
                  << " | Bandwidth: " << std::setw(6) << std::setprecision(1) << effective_bandwidth << " GB/s"
                  << std::endl;

        // Verify correctness against CPU reference
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
        verify_matrix(VERIF_DIM, VERIF_DIM, h_C_ref.data(), h_C.data(), is_wmma ? 1e-1f : 1e-3f);
        std::cout << "-------------------------------------------------------------------------------------------------------\n";
    };

    std::cout << "-------------------------------------------------------------------------------------------------------\n";
    std::cout << "Starting Kernel Benchmarks (Averaged over " << BENCH_ITERS << " iterations):\n";
    std::cout << "-------------------------------------------------------------------------------------------------------\n";

    // 1. Benchmark Naive SGEMM
    {
        dim3 block(32, 32);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        benchmark_kernel("1. Naive SGEMM", [&]() {
            sgemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
        });
    }

    // 2. Benchmark Shared Memory Tiled SGEMM
    {
        constexpr int TILE_DIM = 32;
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        benchmark_kernel("2. Shared Memory Tiled SGEMM", [&]() {
            sgemm_shared_tiled<TILE_DIM><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
        });
    }

    // 3. Benchmark Register-Tiled (Thread-Tiled) SGEMM
    {
        dim3 block(BN / TN, BM / TM); // (16, 16) = 256 threads
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        benchmark_kernel("3. Register-Tiled SGEMM (8x8)", [&]() {
            sgemm_register_tiled<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
        });
    }

    // 4. Benchmark Hardware Tensor Core GEMM (sm_75)
    {
        // 4 warps per block (128 threads) computing a 32x32 block of C
        dim3 block(32, 4);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        benchmark_kernel("4. Turing Tensor Cores (WMMA)", [&]() {
            wmma_gemm_t4<<<grid, block>>>(M, N, K, d_A_half, d_B_half, d_C);
        }, true);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_half));
    CUDA_CHECK(cudaFree(d_B_half));

    std::cout << "\nBenchmark complete. Tesla T4 theoretical peak: 8.1 FP32 TFLOPS | ~65 FP16 Tensor TFLOPS.\n";
    return 0;
}
