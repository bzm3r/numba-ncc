# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:08:49 2016

@author: Brian
"""

import subprocess

command = ['python', 'setup.py', 'build_ext', '--inplace']

subprocess.call(command)