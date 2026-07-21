"""Compiles the Cython monotonic-alignment kernel at install time.
Package metadata + console entry point live in pyproject.toml.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [Extension(
            "matcha.utils.monotonic_align.core",
            ["matcha/utils/monotonic_align/core.pyx"],
            include_dirs=[numpy.get_include()],
        )],
        language_level=3,
    ),
)
