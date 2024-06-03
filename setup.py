from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Levenberg-Marquardt fitter',
    ext_modules=cythonize("levmarpy/qrsolv.pyx"),
)


