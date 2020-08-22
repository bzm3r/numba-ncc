# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 16:56:45 2017

@author: Brian
"""

import scipy.spatial as space
import numpy as np
import time
import numba as nb

# points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
#
# dt = space.Delaunay(points)
#
# for simplex, neighbor_list in zip(dt.simplices, dt.neighbors):
#    pass


#@nb.jit(nopython=True)
def is_prospective_edge_already_counted(prospective_edge, edges):
    for ei in range(edges.shape[0]):
        edge = edges[ei]
        if edge[0] == -1:
            return False
        else:
            if np.all(edge == prospective_edge):
                return True

    return False


def determine_edges(dt):
    edges = -1 * np.ones(
        (dt.points.shape[0] + dt.simplices.shape[0] - 1, 2), dtype=np.int64
    )
    sorted_simplices = np.sort(dt.simplices, axis=1)
    pairs = np.array([[0, 1], [0, 2], [1, 2]])

    ei = 0
    for simplex in sorted_simplices:
        for pair in pairs:
            if ei == edges.shape[0]:
                break

            prospective_edge = simplex[pair]

            if not is_prospective_edge_already_counted(prospective_edge, edges):
                edges[ei] = prospective_edge
                ei += 1
        else:  # http://psung.blogspot.ca/2007/12/for-else-in-python.html
            continue  # executed only if the loop ended normally
        break  # only executed if continue under else statement was skipped

    return edges


#@nb.jit(nopython=True)
def calculate_edge_lengths(points, edges):
    edge_lengths = np.zeros(edges.shape[0], dtype=np.float64)

    for ei in range(edges.shape[0]):
        edge = edges[ei]
        x, y = points[edge[0]] - points[edge[1]]
        edge_lengths[ei] = np.sqrt(x * x + y * y)

    return edge_lengths


points = np.random.rand(
    49, 2
)  # np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64)
dt = space.Delaunay(points)

times = []
for n in range(20):
    st = time.time()
    edges = determine_edges(dt)
    edge_lengths = np.average(calculate_edge_lengths(points, edges))
    et = time.time()
    times.append(et - st)

print("time taken: {} s".format(np.average(times)))
