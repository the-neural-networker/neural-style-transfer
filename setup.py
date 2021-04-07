#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='nst',
    version='0.1.0',
    description='Implementation of Neural Style Transfer using PyTorch',
    author='Abhiroop Tejomay',
    author_email='abhirooptejomay@gmail.com',
    url='https://github.com/visualCalculus/neural-style-transfer',
    packages=find_packages(include=["nst", "nst.*"]),
)

