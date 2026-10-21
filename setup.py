import os
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect GPU architectures
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")
os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-lineinfo",
    ],
}

ext_modules = []
if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
    ext_modules.append(
        CUDAExtension(
            name="custom_kernels_cuda",
            sources=[
                "csrc/src/pybind.cpp",
            ],
            include_dirs=[
                os.path.abspath("csrc/include"),
            ],
            extra_compile_args=extra_compile_args,
        )
    )

setup(
    name="custom_kernels",
    version="0.2.0",
    author="GPU Performance Engineering Community",
    description="High-Performance GPU Kernels in CUDA C++ and OpenAI Triton",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python") + ["triton"],
    package_dir={"": "python", "triton": "triton"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
