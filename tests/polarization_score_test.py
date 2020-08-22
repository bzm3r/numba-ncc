# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:02:35 2017

@author: Brian
"""


import numpy as np
import matplotlib.pyplot as plt
import numba as nb


def f(offset, width):
    return lambda x: np.exp(-1.0 * (x - offset) ** 2 / width)


#@nb.jit(nopython=True)
def piecewise_linear_function(x):
    defining_xs = [0.0, 0.2, 0.33, 0.7, 0.9, 1.0]
    defining_ys = [0.0, 0.2, 1.0, 1.0, 0.2, 0.0]

    num_points = len(defining_xs)

    if x < defining_xs[0]:
        return 0.0
    elif x > defining_xs[-1]:
        return 0.0
    else:
        for j in range(1, num_points):
            x2 = defining_xs[j]

            if x <= x2:
                x1 = defining_xs[j - 1]
                y1 = defining_ys[j - 1]
                y2 = defining_ys[j]

                return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


xs = np.linspace(0.0, 1.0, num=100)
ys = [piecewise_linear_function(x) for x in xs]

plt.plot(xs, ys)
plt.show()
