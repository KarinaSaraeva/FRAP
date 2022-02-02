from setuptools import setup, Extension
import numpy as np

module = Extension('Fricks_Law', sources=['Fricks_Law.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='jetbrains',
    ext_modules=[
        Extension("Fricks_Law", ["Fricks_Law.pyx"],
                  include_dirs=[np.get_include()]),
    ],
)
