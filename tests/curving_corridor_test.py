# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:47:33 2017

@author: Brian
"""

import matplotlib.pyplot as plt
import numpy as np

def generate_bottom_and_top_curves(x_offset, y_offset, curve_start_x, curve_radius, height_corridor, resolution):
    outer_radius = curve_radius + 0.5*height_corridor
    inner_radius = outer_radius - height_corridor
    thetas = np.linspace(0.75*2*np.pi, 2*np.pi, num=resolution)
    
    bottom_curve_untranslated = []
    for theta in thetas:
        bottom_curve_untranslated.append([outer_radius*np.cos(theta), outer_radius*np.sin(theta)])
        
    top_curve_unstranslated = []
    for theta in thetas[::-1]:
        top_curve_unstranslated.append([inner_radius*np.cos(theta), inner_radius*np.sin(theta)])
        
    y_offset = y_offset + 0.5*height_corridor + curve_radius
    bottom_curve = np.array(bottom_curve_untranslated, dtype=np.float64) + [x_offset, y_offset]
    top_curve = np.array(top_curve_unstranslated, dtype=np.float64) + [x_offset, y_offset]
    
    arc_base_line_length = 2*curve_radius*np.sin(np.abs(thetas[1] - thetas[0]))
    length_within_curve = resolution*arc_base_line_length
    
    return bottom_curve, top_curve, length_within_curve

def make_curving_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, curve_start_x, curve_radius, resolution):
    
    if make_migr_space_poly == True:
        bottom_curve, top_curve, length_within_curve = generate_bottom_and_top_curves(corridor_x_offset + curve_start_x, corridor_y_offset, curve_start_x, curve_radius, height_corridor, resolution)
        
        remaining_corridor = width_corridor - length_within_curve
        if remaining_corridor < 1e-16:
            raise Exception("Corridor is not long enough to fit curve!")
            
        bottom_left = [[0 + corridor_x_offset, 0 + corridor_y_offset]]
        bottom_right = [[bottom_curve[-1][0], bottom_curve[-1][1] + remaining_corridor]]
        top_right = [[bottom_curve[-1][0] - height_corridor, bottom_curve[-1][1] + remaining_corridor]]
        top_left = [[0 + corridor_x_offset, 0 + corridor_y_offset + height_corridor]]
        
        full_polygon = np.zeros((0, 2), dtype=np.float64)
        
        for curve in [bottom_left, bottom_curve, bottom_right, top_right, top_curve, top_left]:
            full_polygon = np.append(full_polygon, curve, axis=0)
        
    return full_polygon*1e-6

def make_bottleneck_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, first_slope_start, first_slope_end, second_slope_start, second_slope_end, bottleneck_factor):
    migr_space_poly = np.array([])
    
    if make_migr_space_poly == True:
        bottleneck_y_dip = 0.5*height_corridor*(1. - bottleneck_factor)
        remaining_corridor = width_corridor - (first_slope_start + first_slope_end + second_slope_start + second_slope_end)
        
        if remaining_corridor < 1e-16:
            raise Exception("Width of the corridor is not long enough to support bottleneck!")
            
        bottom_left = [0 + corridor_x_offset, 0 + corridor_y_offset]
        first_slope_start_bottom = [0 + corridor_x_offset + first_slope_start, 0 + corridor_y_offset]
        first_slope_end_bottom = [0 + corridor_x_offset + first_slope_start + first_slope_end, 0 + corridor_y_offset + bottleneck_y_dip]
        second_slope_start_bottom = [0 + corridor_x_offset + first_slope_start + first_slope_end + second_slope_start, 0 + corridor_y_offset + bottleneck_y_dip]
        second_slope_end_bottom = [0 + corridor_x_offset + first_slope_start + first_slope_end + second_slope_start + second_slope_end, 0 + corridor_y_offset]
        bottom_right = [0 + corridor_x_offset + first_slope_start + first_slope_end + second_slope_start + second_slope_end + remaining_corridor, 0 + corridor_y_offset]
        top_right = [0 + corridor_x_offset + first_slope_start + first_slope_end + second_slope_start + second_slope_end + remaining_corridor, 0 + corridor_y_offset + height_corridor]
        second_slope_end_top = [0 + corridor_x_offset + first_slope_start + first_slope_end + second_slope_start + second_slope_end, 0 + corridor_y_offset + height_corridor]
        second_slope_start_top = [0 + corridor_x_offset + first_slope_start + first_slope_end + second_slope_start, 0 + corridor_y_offset + height_corridor - bottleneck_y_dip]
        first_slope_end_top = [0 + corridor_x_offset + first_slope_start + first_slope_end, 0 + corridor_y_offset + height_corridor - bottleneck_y_dip]
        first_slope_start_top = [0 + corridor_x_offset + first_slope_start, 0 + corridor_y_offset + height_corridor]
        top_left = [0 + corridor_x_offset, 0 + corridor_y_offset + height_corridor]
        
        migr_space_poly = np.array([bottom_left, first_slope_start_bottom, first_slope_end_bottom, second_slope_start_bottom, second_slope_end_bottom, bottom_right, top_right, second_slope_end_top, second_slope_start_top, first_slope_end_top, first_slope_start_top, top_left])*1e-6
        
    return migr_space_poly

make_migr_space_poly = True
first_slope_start=40*4*2
first_slope_end = 40*4*1
second_slope_start = 40*4*1
second_slope_end = 40*4*1
bottleneck_factor = 0.5
width_corridor = 4*40*10
height_corridor = 4*40
corridor_x_offset = 10
corridor_y_offset = 10
curve_start_x = first_slope_start
curve_radius = 3*40
resolution = 10

#migr_space_poly = make_bottleneck_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, first_slope_start, first_slope_end, second_slope_start, second_slope_end, bottleneck_factor)/1e-6

migr_space_poly = make_curving_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, curve_start_x, curve_radius, resolution)/1e-6

fig, ax = plt.subplots()
ax.fill(migr_space_poly[:,0], migr_space_poly[:,1])
ax.set_aspect('equal')
fig.show()