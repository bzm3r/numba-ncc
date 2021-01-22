# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07 17:47:39 2015

@author: Brian
"""

import numpy as np


# ==============================================================
def calculate_normalized_randomization_factors(size):
    rfs = np.random.random(size)
    return rfs / np.sum(rfs)


# ===============================================================


def is_numeric(s):
    try:
        float(s)
        return True

    except ValueError:
        return False

#@nb.jit(nopython=True)
def make_node_coords_array_given_xs_and_ys(num_nodes, xs, ys):
    node_coords = np.empty((num_nodes, 2), dtype=np.float64)

    for i in range(num_nodes):
        node_coord = node_coords[i]
        node_coord[0] = xs[i]
        node_coord[1] = ys[i]

    return node_coords
