"""Build the Cython monotonic-alignment kernel (used only during training).

    python setup.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension("core", ["core.pyx"], include_dirs=[numpy.get_include()])
setup(name="monotonic_align", ext_modules=cythonize([ext], language_level=3))
