# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:56:32 2017

@author: Brian
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np


def draw_cell_arrangement(ax, origin, draw_space_factor, num_cells, box_height, box_width, corridor_height, box_y_placement_factor):
    scale_factor = box_width*1.2#num_cells*1.2
    bh = draw_space_factor*(box_height/scale_factor)
    ch = draw_space_factor*(corridor_height/scale_factor)
    bw = draw_space_factor*(box_width/scale_factor)
    cw = draw_space_factor*(1.2*box_width/scale_factor)
    
    origin[0] = origin[0] - cw*0.5

    corridor_boundary_coords = np.array([[cw, 0.], [0., 0.], [0., ch], [cw, ch]], dtype=np.float64) + origin
    
    corridor_boundary_patch = mpatch.Polygon(corridor_boundary_coords, closed=False, fill=False, color='r', ls='solid')
    ax.add_artist(corridor_boundary_patch)
    
    box_origin = origin + np.array([0.0, (ch - bh)*box_y_placement_factor])
    
    cell_radius = 0.5*draw_space_factor*(1./scale_factor)
    cell_placement_delta = cell_radius*2
    y_delta = np.array([0., cell_placement_delta])
    x_delta = np.array([cell_placement_delta, 0.])
    
    cell_origin = box_origin + np.array([cell_radius, cell_radius])
    y_delta_index = 0
    x_delta_index = 0
    
    for ci in range(num_cells):
        cell_patch = mpatch.Circle(cell_origin + y_delta_index*y_delta + x_delta_index*x_delta, radius=cell_radius, color='k', fill=False, ls='solid')
        ax.add_artist(cell_patch)
        
        if y_delta_index == box_height - 1:
            y_delta_index = 0
            x_delta_index += 1
        else:
            y_delta_index += 1
            
# =============================================================================

nrows = 5
fig, axarr = plt.subplots(nrows=nrows, sharex=True)
         
# draw cells in a corridor
tests = [(16, 1, 16, 4, 0.0), (16, 1, 16, 4, 0.5), (16, 2, 8, 4, 0.0), (16, 2, 8, 4, 0.5), (16, 3, 6, 4, 0.0), (16, 3, 6, 4, 0.5), (16, 4, 4, 4, 0.0)]
artists = []
origin = np.array([1.0, 0.25])

for i, test in enumerate(tests):
    nc, bh, bw, ch, byp = test
    o = origin + i*np.array([1.0, 0.0])
    artists.append(draw_cell_arrangement(axarr[-1], o, 0.8, nc, bh, bw, ch, byp))

axarr[-1].set_aspect('equal') # don't squish the circles!
axarr[-1].set_xlim([0.0, (len(tests) + 0.8*0.5)*1.2])
axarr[-1].set_ylim([0.0, (np.max([0.8*(t[1]/(t[2]*1.2)) for t in tests]) + origin[1])*1.2])

# create some random data for boxplots
data_sets = [[np.random.randint(1, high=10, size=10)*np.random.rand(10) for y in range(nrows - 1)] for x in tests]

for ax in axarr[:-1]:
    ax.boxplot(data_sets)
