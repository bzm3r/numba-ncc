# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:08:23 2019

@author: Brian
"""

import numpy as np


def determine_num_nodes_in_protrusion_group(
    num_nodes_in_cell, left_boundary, right_boundary
):
    if left_boundary > right_boundary:
        num_nodes_right = right_boundary + 1
        num_nodes_left = num_nodes_in_cell - left_boundary

        return num_nodes_right + num_nodes_left
    else:
        return (right_boundary - left_boundary) + 1


def determine_central_protrusion_group_nodes(
    num_nodes_in_cell, num_nodes_in_group, left_boundary, right_boundary
):
    central_node_delta = num_nodes_in_group / 2
    from_left_center = (left_boundary + central_node_delta) % num_nodes_in_cell
    from_right_center = (right_boundary - central_node_delta) % num_nodes_in_cell
    return (
        int(np.floor(from_left_center) % num_nodes_in_cell),
        int(np.ceil(from_right_center) % num_nodes_in_cell),
    )


num_nodes_in_cell = 16
left_boundary = np.random.randint(0, 16)
print("left_boundary: {}".format(left_boundary))
right_boundary = np.random.randint(0, 16)
print("right_boundary: {}".format(right_boundary))

calculated_num_nodes_in_group = determine_num_nodes_in_protrusion_group(
    num_nodes_in_cell, left_boundary, right_boundary
)
print("calculated_num_nodes_in_group: {}".format(calculated_num_nodes_in_group))
central_protrusion_group_nodes = determine_central_protrusion_group_nodes(
    num_nodes_in_cell, calculated_num_nodes_in_group, left_boundary, right_boundary
)
print("central_protrusion_group_nodes: {}".format(central_protrusion_group_nodes))
