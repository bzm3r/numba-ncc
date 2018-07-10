# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:49:06 2018

@author: Brian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
import numba as nb

def f(x, a, b, c):
    return a*np.exp(b*x) + c

a, b, c = sciopt.curve_fit(f, [-0.1, 0, 0.1], [80000, 8000, 1000])[0]

def make_g(a, b, c):
    @nb.jit(nopython=True)
    def g(x):
        return a*np.exp(b*x) + c
    
    return g

g= make_g(a, b, c)

xs = np.linspace(-0.15, 0.15, num=1000)
ys = np.array([g(x) for x in xs])
plt.plot(xs, ys)
plt.show()