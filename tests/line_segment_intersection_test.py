# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:53:49 2017

@author: Brian
"""

import numpy as np
import matplotlib.pyplot as plt
import core.geometry as geometry
import math

num_nodes = 16
cell_radius = 20
num_cells_width = 2
num_cells_height = 2

def make_cell_grid(num_nodes, cell_radius, num_cells_width, num_cells_height):
    cell_diameter = 2*cell_radius
    origin_xs = cell_radius + np.arange(num_cells_width)*cell_diameter
    origin_ys = cell_radius + np.arange(num_cells_height)*cell_diameter
    
    cells = []
    
    for ox in origin_xs:
        for oy in origin_ys:
            cells.append([[ox + cell_radius*np.cos(theta), oy + cell_radius*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, endpoint=False, num=num_nodes)])
            
    return cells

cell_grid = make_cell_grid(num_nodes, cell_radius, num_cells_width, num_cells_height)

fig, ax = plt.subplots()

for cell_coords in cell_grid:
    for i in range(num_nodes):
        ip1 = (i + 1)%num_nodes
        ni_coords = cell_coords[i]
        nip1_coords = cell_coords[ip1]
        ax.plot([ni_coords[0]], [ni_coords[1]], color='k', marker='.')
        ax.plot([ni_coords[0], nip1_coords[0]], [ni_coords[1], nip1_coords[1]], color='k')

num_cells = num_cells_width*num_cells_height

bboxes = geometry.create_initial_bounding_box_polygon_array(num_cells, num_nodes, np.array(cell_grid))


line_segment_intersection_matrix = geometry.create_initial_line_segment_intersection_matrix(num_cells, num_nodes, bboxes, np.array(cell_grid))

for ci in [0]:
    for ni in range(num_nodes):
        for other_ci in [3]:
            for other_ni in range(num_nodes):
                if line_segment_intersection_matrix[ci][ni][other_ci][other_ni] == 0:
                    ax.plot([cell_grid[ci][ni][0], cell_grid[other_ci][other_ni][0]], [cell_grid[ci][ni][1], cell_grid[other_ci][other_ni][1]], color='g')
        
ax.set_aspect('equal')
        
plt.show()
    

