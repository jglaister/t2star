#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup
Setup for t2star package
Author: Jeffrey Glaister
"""
from glob import glob
from setuptools import setup, find_packages

args = dict(
    name='estimate_t2star',
    version='0.1',
    description="T2star processing",
    author='Jeffrey Glaister',
    author_email='jeff.glaister@gmail.com',
    url='https://github.com/jglaister/t2star'
)

setup(install_requires=['nipype', 'numpy', 'nibabel', 'scipy', 'sklearn'],
      packages=['estimate_t2star'],
      scripts=glob('bin/*'), **args)