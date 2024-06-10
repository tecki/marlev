from setuptools import setup
from Cython.Build import cythonize

setup(
    name='levmarpy',
    version='0.1',
    description='Levenberg-Marquardt fitter',
    ext_modules=cythonize("levmarpy/qrsolv.pyx"),
    author='Martin Teichmann',
    author_email='martin.teichmann@xfel.eu',
)


