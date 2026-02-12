from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from Cython.Build import cythonize
import numpy as np

package_name = "cyclonebasecall"
version = "2.0"
require_file = "./requirements.txt"

with open(require_file) as f:
    requirements = f.read().splitlines()

setup(
    name=package_name,
    version=version,
    packages=find_packages(include=["cyclonebasecall", "cyclonebasecall.*"]),
    # install_requires=requirements,
    author='yanxu',
    author_email='yanxu@genomics.cn',
    ext_modules=cythonize(
        Extension(
            'cyclonebasecall.evaluation.mapping_optim_v1.cam_v1',
            sources=['cyclonebasecall/evaluation/mapping_optim_v1/cam_v1.pyx'],
            include_dirs=[np.get_include()], )
    ),
    install_requires=["numpy"],
    zip_safe=False,
)
