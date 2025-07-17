from setuptools import setup, find_packages, Extension
import pybind11
import os

# Find OpenCV
import cv2
opencv_config = os.popen('pkg-config --cflags --libs opencv4').read().strip().split()

# Extract include and library paths
include_dirs = [pybind11.get_include()]
library_dirs = []
libraries = []
for flag in opencv_config:
    if flag.startswith('-I'):
        include_dirs.append(flag[2:])
    elif flag.startswith('-L'):
        library_dirs.append(flag[2:])
    elif flag.startswith('-l'):
        libraries.append(flag[2:])

# Extension module
ext_modules = [
    Extension(
        'spec_encoding',
        ['airtio'+os.sep+'spec_encoding.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries + ['gomp'],  # OpenMP
        extra_compile_args=['-O3', '-Wall',  '-std=c++20', '-fPIC', '-march=native', '-ffast-math'],
        extra_link_args=[],
        language='c++'
    )
]

setup(
    name="airtio",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "PyGObject",
        "python-uinput",
        "displayarray",
        "sounddevice",
        "PyV4L2Cam @ git+https://github.com/SimLeek/PyV4L2Cam"
    ],
    ext_modules=ext_modules
)