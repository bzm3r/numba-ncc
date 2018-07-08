# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:50:09 2017

@author: Brian
"""

import numpy as np
import numba as nb

@nb.jit(nopython=True)  
def calculate_polygon_bounding_box(polygon):
    num_vertices = polygon.shape[0]
    
    min_x = 0.0
    max_x = 0.0
    
    min_y = 0.0
    max_y = 0.0
    
    for i in range(num_vertices):
        this_x, this_y = polygon[i]
        if i == 0:
            min_x = this_x
            max_x = min_x
            
            min_y = this_y
            max_y = min_y
        else:
            if this_x < min_x:
                min_x = this_x
            elif this_x > max_x:
                max_x = this_x
            
            if this_y < min_y:
                min_y = this_y
            elif this_y > max_y:
                max_y = this_y
    
    return min_x, max_x, min_y, max_y


@nb.jit(nopython=True)  
def is_point_in_polygon_bounding_box(test_point, num_vertices, polygon):
    
    min_x, max_x, min_y, max_y = calculate_polygon_bounding_box(num_vertices, polygon)
    
    return is_point_in_polygon_bounding_box(test_point, min_x, max_x, min_y, max_y)

@nb.jit(nopython=True)  
def is_point_in_polygon_bounding_box_given_bounding_box(test_point, min_x, max_x, min_y, max_y):                
    tp_x = test_point[0]
    tp_y = test_point[1]
    
    if (min_x < tp_x < max_x) and (min_y < tp_y < max_y):
        return 1
    else:
        return 0
    
@nb.jit(nopython=True)         
def is_left(p0, p1, p2):
    '''
    Input:  three points P0, P1, and P2
    Return: > 0 for P2 left of the line through P0 to P1
            = 0 for P2 on the line
            < 0 for P2 right of the line
    '''
    p0x, p0y = p0
    p1x, p1y = p1
    p2x, p2y = p2
    
    return ((p1x - p0x)*(p2y - p0y) - (p2x - p0x)*(p1y - p0y))

@nb.jit(nopython=True)   
def is_point_in_polygon_without_bb_check(test_point, num_vertices, polygon):
    wn = 0
    test_point_y = test_point[1]
    
    # count number of intersections of positive-x direction ray emanating from test_point with polygon edges
    for i in range(num_vertices):
        p_start = polygon[i]
        p_end = polygon[(i + 1)%num_vertices]
        
        p_start_y = p_start[1]
        
        p_end_y = p_end[1]
        
        if p_start_y <= test_point_y < p_end_y:
            # upward crossing
            is_tp_left_of_edge = is_left(p_start, p_end, test_point)
            
            if is_tp_left_of_edge > 0:
                # positive-x direction ray emanating from test_point wil intersect with this edge if left of it
                wn = wn + 1
        elif p_end_y < test_point_y <= p_start_y:
            # downward crossing
            is_tp_left_of_edge = is_left(p_start, p_end, test_point)
            
            if is_tp_left_of_edge < 0:
                # positive-x direction ray emanating from test_point will intersect with this edge if left of it
                wn = wn - 1
        else:
            # no intersection
            wn = wn
                
    if wn == 0:
        return 0
    else:
        return 1
    
@nb.jit(nopython=True)    
def is_point_in_polygon_given_bounding_box(test_point, num_vertices, polygon, min_x, max_x, min_y, max_y):
    is_test_point_in_poly_bb = is_point_in_polygon_bounding_box_given_bounding_box(test_point, min_x, max_x, min_y, max_y)
    
    if is_test_point_in_poly_bb == 0:
        return 0
    else:
        return is_point_in_polygon_without_bb_check(test_point, num_vertices, polygon)
    
@nb.jit(nopython=True)  
def are_points_inside_polygon(num_points, points, num_poly_vertices, polygon):
    results = np.zeros(num_points, dtype=np.int64)
    
    min_x, max_x, min_y, max_y = calculate_polygon_bounding_box(polygon)
    
    for index in range(num_points):
        test_point = points[index]
        
        results[index] = is_point_in_polygon_given_bounding_box(test_point, num_poly_vertices, polygon, min_x, max_x, min_y, max_y)
        
    return results

@nb.jit(nopython=True)  
def check_if_nodes_inside_other_cells(this_cell_index, num_nodes, num_cells, all_cells_node_coords):
    are_nodes_inside_other_cells = np.zeros((num_nodes, num_cells), dtype=np.int64)
    
    this_cell_node_coords = all_cells_node_coords[this_cell_index]
    
    for other_ci in range(num_cells):
        if other_ci !=  this_cell_index:
            are_nodes_inside_current_cell = are_points_inside_polygon(num_nodes, this_cell_node_coords, num_nodes, all_cells_node_coords[other_ci])
            for ni in range(num_nodes):
                are_nodes_inside_other_cells[ni][other_ci] = are_nodes_inside_current_cell[ni]
                    
    return are_nodes_inside_other_cells

def generate_cell_coords(num_nodes, r, ox, oy):
    return np.array([[ox + r*np.cos(theta), oy + r*np.sin(theta)] for theta in np.linspace(0, 2*np.pi,num=num_nodes, endpoint=False)])
    

# =============================================================================

num_nodes = 10
num_cells = 2

cc0 = generate_cell_coords(num_nodes, 1., 0.0, 0.0)
cc1 = generate_cell_coords(num_nodes, 0.5, 0.0, 0.0)

all_cells_node_coords = np.array([cc0, cc1])

result = check_if_nodes_inside_other_cells(1, num_nodes, num_cells, all_cells_node_coords)

@nb.jit(nopython=True)
def testfn0(num_nodes):
    newarray = np.zeros(num_nodes, dtype=np.int64)
    
    return newarray

@nb.jit(nopython=True)
def testfn(a, b):
    x = -1*np.ones((a, b), dtype=np.int64)
    
    for i in range(a):
        y = testfn0(num_nodes)
        for j in range(b):
            x[i][j] = y[j]
            
    return x

print(testfn(num_nodes, num_cells))