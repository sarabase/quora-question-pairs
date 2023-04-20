from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("tf_idf.pyx"),
)
