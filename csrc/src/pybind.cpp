#include <torch/extension.h>
#include <cuda_runtime.h>

#include "../include/gemm/gemm_naive.cuh"
#include "../include/gemm/gemm_tiled.cuh"
#include "../include/gemm/gemm_reg_tiled.cuh"
#include "../include/gemm/gemm_wmma.cuh"
#include "../include/attention/flash_attention.cuh"
#include "../include/norm/layernorm_rmsnorm.cuh"
#include "../include/activations/swiglu.cuh"

// Macro for input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// GEMM Bindings
// ============================================================================
torch::Tensor sgemm_naive_torch(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner matrix dimensions (K) must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    launch_sgemm_naive(M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

torch::Tensor sgemm_tiled_torch(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    launch_sgemm_tiled(M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

torch::Tensor sgemm_reg_tiled_torch(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    launch_sgemm_reg_tiled(M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

// ============================================================================
// FlashAttention Binding
// ============================================================================
torch::Tensor flash_attention_torch(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool is_causal) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    TORCH_CHECK(Q.dim() == 4, "Tensors must be 4D: (Batch, Heads, SeqLen, Dim)");
    TORCH_CHECK(Q.size(3) == 64, "Currently optimized for head dimension d=64");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto O = torch::empty_like(Q);
    launch_flash_attention(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        B, H, N, d, is_causal
    );
    return O;
}

// ============================================================================
// Normalization Bindings
// ============================================================================
torch::Tensor rms_norm_torch(torch::Tensor X, torch::Tensor W, float eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(W);

    int hidden_dim = W.size(0);
    int num_tokens = X.numel() / hidden_dim;

    auto Y = torch::empty_like(X);
    launch_rms_norm(X.data_ptr<float>(), W.data_ptr<float>(), Y.data_ptr<float>(), num_tokens, hidden_dim, eps);
    return Y;
}

torch::Tensor layer_norm_torch(torch::Tensor X, torch::Tensor W, torch::Tensor B, float eps) {
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    if (B.defined()) CHECK_INPUT(B);

    int hidden_dim = W.size(0);
    int num_tokens = X.numel() / hidden_dim;

    auto Y = torch::empty_like(X);
    const float *b_ptr = B.defined() ? B.data_ptr<float>() : nullptr;
    launch_layer_norm(X.data_ptr<float>(), W.data_ptr<float>(), b_ptr, Y.data_ptr<float>(), num_tokens, hidden_dim, eps);
    return Y;
}

// ============================================================================
// Activation Binding
// ============================================================================
torch::Tensor swiglu_torch(torch::Tensor In) {
    CHECK_INPUT(In);
    int D = In.size(-1) / 2;
    int N = In.numel() / (2 * D);

    auto out_sizes = In.sizes().vec();
    out_sizes.back() = D;
    auto Out = torch::empty(out_sizes, In.options());

    launch_swiglu(In.data_ptr<float>(), Out.data_ptr<float>(), N, D);
    return Out;
}

// ============================================================================
// PyBind11 Module Registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-Performance Custom CUDA & Triton GPU Kernels";

    // GEMM
    m.def("gemm_naive", &sgemm_naive_torch, "Naive SGEMM (CUDA)");
    m.def("gemm_tiled", &sgemm_tiled_torch, "Shared Memory Tiled SGEMM (CUDA)");
    m.def("gemm_reg_tiled", &sgemm_reg_tiled_torch, "Register-Tiled Micro-SGEMM (CUDA)");

    // Attention
    m.def("flash_attention", &flash_attention_torch, "FlashAttention v2 Forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal") = true);

    // Normalization
    m.def("rms_norm", &rms_norm_torch, "Fused RMSNorm (CUDA)",
          py::arg("X"), py::arg("W"), py::arg("eps") = 1e-6f);
    m.def("layer_norm", &layer_norm_torch, "Fused LayerNorm (CUDA)",
          py::arg("X"), py::arg("W"), py::arg("B") = torch::Tensor(), py::arg("eps") = 1e-5f);

    // Activation
    m.def("swiglu", &swiglu_torch, "Fused SwiGLU Activation (CUDA)", py::arg("In"));
}
