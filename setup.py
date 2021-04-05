#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='nst',
    version='0.1.0',
    description='Implementation of Neural Style Transfer using PyTorch',
    author='Abhiroop Tejomay',
    author_email='abhirooptejomay@gmail.com',
    url='https://github.com/visualCalculus/neural-style-transfer',
    install_requires=['torch==1.7.0', 'torchvision==0.8.0', 
                    'numpy==1.19.5', 'matplotlib', 
                    'pillow', 'tqdm'],
    packages=find_packages(),
)

