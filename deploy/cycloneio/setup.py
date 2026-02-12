from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from Cython.Build import cythonize
import numpy as np

package_name = "cycloneio"
version = "0.2.2"
require_file = "./requirements.txt"

with open(require_file) as f:
    requirements = f.read().splitlines()

setup(
    name=package_name,
    version=version,
    packages=find_packages(include=["test", "cycloneio", "cycloneio.*"]),
    zip_safe = False
)
