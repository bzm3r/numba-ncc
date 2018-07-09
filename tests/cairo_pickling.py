# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 16:13:56 2018

@author: Brian
"""

import cairo #pycairo
import dill

x = cairo.Matrix()
print(x)

dill.detect.trace(True)

dill.dumps(cairo.Matrix())

