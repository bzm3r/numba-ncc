# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:47:26 2017

@author: Brian
"""

import core.geometry as geometry
import numpy as np
import time

num_cells = 50
num_nodes_per_cell = 16
all_cells_node_coords = np.random.rand(50, 16, 2)
num_threads = 4

def format_time(num):
    return np.round(num, decimals=1)

st = time.time()
cells_bounding_box_array = geometry.create_initial_bounding_box_polygon_array(num_cells, num_nodes_per_cell, all_cells_node_coords)
et = time.time()
print("create_initial_bounding_box_polygon_array: {}s".format(format_time(et - st)))

last_updated_cell_index = np.random.randint(0, 50)
print("cell index to update: ", last_updated_cell_index)

st = time.time()

all_geometry_tasks = np.array(geometry.create_dist_and_line_segment_interesection_test_args(num_cells, num_nodes_per_cell), dtype=np.int64)
geometry_tasks_per_cell = np.array([geometry.create_dist_and_line_segment_interesection_test_args_relative_to_specific_cell(ci, num_cells, num_nodes_per_cell) for ci in range(num_cells)], dtype=np.int64)

#distance_squared_matrix, line_segment_intersection_matrix = geometry.create_initial_line_segment_intersection_and_dist_squared_matrices(num_threads, all_geometry_tasks, num_cells, num_nodes_per_cell, cells_bounding_box_array, all_cells_node_coords)

distance_squared_matrix, line_segment_intersection_matrix = geometry.create_initial_line_segment_intersection_and_dist_squared_matrices_old(num_cells, num_nodes_per_cell, cells_bounding_box_array, all_cells_node_coords)
et = time.time()

print("create_initial_line_segment_intersection_and_dist_squared_matrices: {}s".format(format_time(et - st)))

st = time.time()

for ci in range(num_cells):
#    distance_squared_matrix, line_segment_intersection_matrix = geometry.update_line_segment_intersection_and_dist_squared_matrices(num_threads, geometry_tasks_per_cell[ci], num_cells, num_nodes_per_cell, all_cells_node_coords, cells_bounding_box_array, distance_squared_matrix, line_segment_intersection_matrix, sequential=False)
    distance_squared_matrix, line_segment_intersection_matrix =  geometry.update_line_segment_intersection_and_dist_squared_matrices_old(ci, num_cells, num_nodes_per_cell, all_cells_node_coords, cells_bounding_box_array, distance_squared_matrix, line_segment_intersection_matrix)
et = time.time()

print("update_line_segment_intersection_and_dist_squared_matrices: {}s".format(format_time(et - st)))