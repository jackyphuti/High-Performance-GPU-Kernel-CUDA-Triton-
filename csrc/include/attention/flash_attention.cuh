#pragma once

#include "common.cuh"
#include <cfloat>

/**
 * FlashAttention Forward Pass (Self-Attention with Online Softmax)
 * 
 * Algorithm: Tri Dao et al. (2022, 2023)
 * Solves O(N^2) memory explosion by avoiding materialization of the full N x N attention matrix.
 * 
 * Matrix Dimensions:
 * - Q: (B, H, N, d)
 * - K: (B, H, N, d)
 * - V: (B, H, N, d)
 * - O: (B, H, N, d)
 * 
 * Tile Sizes:
 * - Br: Query block size (rows)
 * - Bc: Key/Value block size (cols)
 * - d: Head dimension (e.g. 64 or 128)
 */
template <int Br = 32, int Bc = 32, int d = 64>
__global__ void flash_attention_fwd_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int N,
    float scale,
    bool is_causal
) {
    // Each block processes 1 batch, 1 head, and a Br-sized chunk of queries
    int b = blockIdx.z; // Batch index
    int h = blockIdx.y; // Head index
    int q_block_idx = blockIdx.x; // Block along sequence length N

    int tx = threadIdx.x; // 0..Br-1 (thread represents 1 query row in the block)

    // Statically allocated Shared Memory for Q, K, V tiles
    __shared__ float s_Q[Br][d];
    __shared__ float s_K[Bc][d];
    __shared__ float s_V[Bc][d];

    // Global memory offset for current (batch, head)
    int bh_offset = (b * gridDim.y + h) * N * d;
    const float *q_ptr = Q + bh_offset;
    const float *k_ptr = K + bh_offset;
    const float *v_ptr = V + bh_offset;
    float *o_ptr = O + bh_offset;

    int q_row = q_block_idx * Br + tx;

    // Load Q tile into shared memory
    if (q_row < N) {
        #pragma unroll
        for (int i = 0; i < d; ++i) {
            s_Q[tx][i] = q_ptr[q_row * d + i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < d; ++i) {
            s_Q[tx][i] = 0.0f;
        }
    }

    // Thread-private registers for online softmax and output accumulation
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_reg[d] = {0.0f};

    int num_kv_blocks = ceil_div(N, Bc);

    // Loop over Key/Value blocks
    for (int kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
        // Causal masking early exit: if Key tile starts strictly after Query tile, skip
        if (is_causal && (kv_idx * Bc > (q_block_idx + 1) * Br - 1)) {
            break;
        }

        // Cooperatively load K and V tiles into shared memory
        int kv_row = kv_idx * Bc + tx;
        if (kv_row < N) {
            #pragma unroll
            for (int i = 0; i < d; ++i) {
                s_K[tx][i] = k_ptr[kv_row * d + i];
                s_V[tx][i] = v_ptr[kv_row * d + i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < d; ++i) {
                s_K[tx][i] = 0.0f;
                s_V[tx][i] = 0.0f;
            }
        }

        __syncthreads();

        // 1. Compute S = Q * K^T (size Br x Bc) for this thread's query row
        float S_row[Bc];
        float m_curr = -FLT_MAX;

        #pragma unroll
        for (int j = 0; j < Bc; ++j) {
            int k_col = kv_idx * Bc + j;
            if (k_col < N && (!is_causal || k_col <= q_row)) {
                float score = 0.0f;
                #pragma unroll
                for (int i = 0; i < d; ++i) {
                    score += s_Q[tx][i] * s_K[j][i];
                }
                score *= scale;
                S_row[j] = score;
                if (score > m_curr) m_curr = score;
            } else {
                S_row[j] = -FLT_MAX;
            }
        }

        // 2. Online Softmax Update (Milakov & Gimelshein / Tri Dao)
        // m_new = max(m_prev, m_curr)
        float m_new = (m_prev > m_curr) ? m_prev : m_curr;
        float alpha = __expf(m_prev - m_new);

        float l_curr = 0.0f;
        #pragma unroll
        for (int j = 0; j < Bc; ++j) {
            if (S_row[j] != -FLT_MAX) {
                float p = __expf(S_row[j] - m_new);
                S_row[j] = p; // Re-use S_row to store unnormalized probabilities P
                l_curr += p;
            } else {
                S_row[j] = 0.0f;
            }
        }

        // l_new = l_prev * exp(m_prev - m_new) + l_curr
        float l_new = l_prev * alpha + l_curr;

        // 3. Update Output Accumulator:
        // O = O * alpha + P * V
        #pragma unroll
        for (int i = 0; i < d; ++i) {
            float pv = 0.0f;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                pv += S_row[j] * s_V[j][i];
            }
            o_reg[i] = o_reg[i] * alpha + pv;
        }

        m_prev = m_new;
        l_prev = l_new;

        __syncthreads();
    }

    // 4. Final normalization by l_prev and write-back to Global Memory O
    if (q_row < N) {
        float inv_l = 1.0f / (l_prev + 1e-6f);
        #pragma unroll
        for (int i = 0; i < d; ++i) {
            o_ptr[q_row * d + i] = o_reg[i] * inv_l;
        }
    }
}

inline void launch_flash_attention(
    const float *Q, const float *K, const float *V, float *O,
    int B, int H, int N, int d,
    bool is_causal = true,
    cudaStream_t stream = 0
) {
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int D = 64;

    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    dim3 block(Br);
    dim3 grid(ceil_div(N, Br), H, B);

    flash_attention_fwd_kernel<Br, Bc, D><<<grid, block, 0, stream>>>(
        Q, K, V, O, N, scale, is_causal
    );
}
