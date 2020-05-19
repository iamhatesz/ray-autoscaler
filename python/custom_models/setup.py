from setuptools import setup, find_packages

setup(
    name='custom_models',
    version='1.0.0',
    packages=find_packages(),
    install_requires=['ray', 'torch']
)
