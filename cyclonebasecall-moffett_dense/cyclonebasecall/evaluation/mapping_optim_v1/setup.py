# -*- coding:utf-8 -*-
from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy as np
setup(
    name='cam_v1',
    # ext_modules=cythonize("cam_v1.pyx", include_path=[np.get_include()]),
    ext_modules=cythonize(
        Extension(
            "cam_v1",
            sources=["cam_v1.pyx"],
            include_dirs=[np.get_include()]
        )
    ),
    install_requires=["numpy"],
    zip_safe=False,
)