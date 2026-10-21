cmake_minimum_required(VERSION 3.18)
project(HighPerformanceGPUKernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Architecture compilation flags
    # sm_75: Tesla T4, RTX 2080
    # sm_80: A100
    # sm_86: RTX 3080/3090, A6000
    # sm_89: RTX 4090, L40
    # sm_90a: H100
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()

    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    include_directories(
        csrc/include
        ${CUDA_INCLUDE_DIRS}
    )

    # Standalone GEMM Benchmark Executable
    add_executable(sgemm_benchmark
        sgemm_t4.cu
    )

    set_target_properties(sgemm_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(sgemm_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -lineinfo
        >
    )
else()
    message(STATUS "CUDA compiler (nvcc) not found; skipping CUDA executable targets.")
endif()
