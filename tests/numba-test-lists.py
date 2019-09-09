# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:37:34 2017

@author: Brian
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def stupid_list():
    a_list = []

    for n in range(10):
        # does not work
        # a_list.append(np.random.rand(5, 5))

        # does work
        # a_list.append(n)

        # does work
        a_list.append((n, n + 1, n + 2))

    return a_list
