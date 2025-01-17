from setuptools import setup
from Cython.Build import cythonize

setup(
    name='marlev',
    version='0.2',
    description='Levenberg-Marquardt fitter',
    ext_modules=cythonize("marlev/_qrsolv.pyx"),
    author='Martin Teichmann',
    author_email='martin.teichmann@xfel.eu',
)


