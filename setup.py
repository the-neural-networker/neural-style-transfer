#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='neural-style-transfer',
    version='0.1.0',
    description='Implementation of Neural Style Transfer using PyTorch',
    author='Abhiroop Tejomay',
    author_email='abhirooptejomay@gmail.com',
    url='https://github.com/visualCalculus/neural-style-transfer',
    install_requires=['torch', 'torchvision', 
                    'numpy', 'matplotlib', 
                    'pillow', 'tqdm'],
    packages=find_packages(),
)

