# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:40:12 2016

@author: Brian
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("hello_world.pyx")
)