#To build the cython code in the .pyx file, type in the terminal:
#"python setup.py build_ext --inplace"
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("cython_finite_diff_lap", ["cython_finite_diff_lap.pyx"],
                include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions, annotate=True),
)
