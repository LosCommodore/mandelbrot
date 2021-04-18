from setuptools import setup, Extension

module = Extension('mandel_py', sources=['mandel_py.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='jetbrains',
    ext_modules=[module]
)