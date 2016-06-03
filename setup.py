from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [Extension(
                "ptest",
                sources=["ptest.pyx"],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"]
            )]

setup(
    ext_modules = cythonize(extensions)
)

