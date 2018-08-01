# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:25:46 2018

@author: Brian
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

def make_linear_gradient_function(source_x, source_y, max_value, slope):
    @nb.jit(nopython=True)
    def f(x):
        d = np.sqrt((x[0] - source_x)**2 + (x[1] - source_y)**2)
        calc_value = max_value - slope*d
        
        if calc_value > max_value:
            return max_value
        elif calc_value < 0.0:
            return 0.0
        else:
            return calc_value
            
    return f

test_slopes = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

xs = np.linspace(0.0, 1000.0, num=100)
array_xs = np.zeros((xs.shape[0], xs.shape[0]), dtype=np.float64)
array_xs[:,0] = xs

for slope in test_slopes:
    linear_function = make_linear_gradient_function(0.0, 0.0, 10000.0, slope*40.0*10000.0)
    plt.plot(xs, [linear_function(x) for x in array_xs])
    
plt.show()