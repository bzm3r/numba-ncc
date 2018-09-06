# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 19:57:54 2017

@author: Brian
"""

import numpy as np
import numba as nb

@nb.jit(nopython=True)
def is_angle_between_range(a, b, angle):
    mod_a, mod_b = a%(2*np.pi), b%(2*np.pi)
    mod_angle = angle%(2*np.pi)
    
    if mod_b < mod_a:
        if (0 <= mod_angle <= mod_b) or (mod_a <= mod_angle <= 2*np.pi):
            return 1
        else:
            return 0
    else:
        if mod_a <= mod_angle <= mod_b:
            return 1
        else:
            return 0
        
a, b = [-0.5*np.pi, 0.5*np.pi]
num_nodes = 3
index_directions = np.linspace(0, 2*np.pi, num=num_nodes)

is_node_ok = [is_angle_between_range(a, b, x) for x in index_directions]

for x, b in zip(index_directions, is_node_ok):
    print("{}, {}".format(b, x/np.pi))