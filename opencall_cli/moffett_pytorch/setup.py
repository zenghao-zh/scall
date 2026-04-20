from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="moffett_ae_sigmoid_tanh_cuda",
    py_modules=["load_moffett_ae"],
    ext_modules=[
        CUDAExtension(
            name="moffett_ae_sigmoid_tanh_cuda",
            sources=[
                "moffett_ae_torch.cpp",
                "ae_sigmoid_tanh_rcp_cpu.cpp",
                "ae_sigmoid_tanh_rcp_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
