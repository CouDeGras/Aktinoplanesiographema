from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

ext_modules = cythonize(
    [
        Extension(
            name="aktino.core",
            sources=["aktino/core.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"],
        )
    ],
    compiler_directives={"language_level": "3"},
)

setup(
    name="aktino",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["numpy", "Pillow"],
    zip_safe=False,
)
