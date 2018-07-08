# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:01:31 2018

@author: Brian
"""

import core.geometry as geometry
import numpy as np

# -------------------------------------
@nb.jit(nopython=True)
def calculate_gp_adjacency_matrix(coa_grid_points, coa_data_per_gridpoint, resolution, similarity_threshold=0.98):
    num_gps = coa_grid_points.shape[0]
    gp_adjacency_matrix = -2*np.ones((num_gps, num_gps), dtype=np.int16)
    
    for i in range(num_gps):
        ith_gp = coa_grid_points[i]
        for j in range(num_gps):
            if i == j:
                gp_adjacency_matrix[i][j] = -1
            elif gp_adjacency_matrix[i][j] == -2:
                dist = geometry.calculate_dist_between_points_given_coords(ith_gp, coa_grid_points[j])
                coa_at_i, coa_at_j = coa_data_per_gridpoint[i], coa_data_per_gridpoint[j]
                coa_ratio = 0.0
                if coa_at_i < coa_at_j:
                    coa_ratio = coa_at_i/coa_at_j
                else:
                    coa_ratio = coa_at_j/coa_at_i
                                                           
                if np.abs(resolution - dist) < 1e-12 and coa_ratio > similarity_threshold:    
                    gp_adjacency_matrix[i][j] = 1
                    gp_adjacency_matrix[j][i] = 1
                else:
                    gp_adjacency_matrix[i][j] = 0
                    gp_adjacency_matrix[j][i] = 0
                    
    return gp_adjacency_matrix

# --------------------------------------
def get_relative_of_j_around_i_in_polygon_matrix(i, j, ith_gp, jth_gp, i_offset):
    relative = jth_gp - ith_gp
    
    dir_offset, offset = 0.0
    for k in range(2):
        if np.abs(relative[k] - 0) > 1e-12:
            dir_offset, offset = k, relative[k]
            break
    
    if np.abs(offset - 0.0) < 1e-16:
        return i_offset
    elif offset > 0:
        if dir_offset == 0:
            return np.array([1, 0]) + i_offset
        else:
            return np.array([0, 1]) + i_offset
    else:
        if dir_offset == 0:
            return np.array([-1, 0]) + i_offset
        else:
            return np.array([0, -1]) + i_offset
        
# --------------------------------------
def get_adjacency_tile(focus_index, focus_index_gp, relative_to_focus_adjacency, coa_grid_points, gps_processing_status, focus_index_offset_in_polygon_matrix):
         
    adjacent_to = [k for k, adjacent in relative_to_focus_adjacency if adjacent == 1 and gps_processing_status[k] == 0]
    assert(len(adjacent_to)) < 5
    
    relative_indices_per_gpi = [(k, get_relative_of_j_around_i_in_polygon_matrix(focus_index, k, focus_index_gp, coa_grid_points[k])) for k in adjacent_to]
    
    return adjacent_to, relative_indices_per_gpi

# --------------------------------------
    
def get_relative_index(relative_indices_per_gpi, i):
    for k, relative_index in relative_indices_per_gpi:
        if k == i:
            return relative_index
# --------------------------------------
            
def convert_polygon_matrix_into_polygon(origin, polygon_matrix):
    cursor_direction = np.array([1, 0])
    
    polygon_vertices = []
    pm_size = polygon_matrix.shape[0]
    
    start = -1*np.ones(2, dtype=np.int16)
    start_vertex = -1
    for i in range(pm_size):
        for j in range(pm_size):
            cell_content = polygon_matrix[i][j]
            if cell_content > 0:
                start = np.array([i, j])
                start_vertex = cell_content
                break
    
    if start_vertex < 0:
        raise Exception("No start found!")
    
    polygon_vertices.append(start)
    
    done = False
    while not done:
        new_vertex = start + cursor_direction
        
        if new_vertex not in polygon_vertices:
            polygon_vertices.append(new_vertex)
            if polygon_matrix[start + cursor_direction + cursor_direction] != 1:
                cursor_direction = geometry.rotate_2D_vector_CCW(cursor_direction)
        else:
            done = True
            
    return np.array(polygon_vertices)
        
# --------------------------------------

def convert_polygon_matrices_into_polygons(polygon_matrices, coa_grid_points, resolution):
    raw_start_vertex_indices_and_polygons = [convert_polygon_matrix_into_polygon(pm) for pm in polygon_matrices]
    
    polygons = []
    for vertex_index, raw_poly in raw_start_vertex_indices_and_polygons:
        origin = coa_grid_points[vertex_index] + -1*0.5*np.array([resolution, resolution])
        polygons.append(raw_poly + origin)
        
    return polygons

# --------------------------------------
            
def get_simplified_polygon_matrices_and_coa_data(coa_grid_points, coa_data_per_gridpoint, resolution):
    num_gps = coa_grid_points.shape[0]
    gp_adjacency_matrix = calculate_gp_adjacency_matrix(coa_grid_points, resolution)
    gps_processing_status = np.zeros(num_gps, dtype=np.bool)
    
    polygon_matrices = []
    average_coa_data_per_polygon = []

    for i in range(num_gps):
        if gps_processing_status[i] == 0:
            current_polygon_matrix_size = 0
            coa_data = 0
        
            gpi_indices = [i]
            relative_indices_per_gpi = [(i, np.array([0, 0]))]
            adjacent_to = [i]
            
            while len(adjacent_to) != 0:
                j = adjacent_to.pop(0)
                gpi_indices.append(j)
                
                j_relative_index = get_relative_index(relative_indices_per_gpi, j)
                sub_adjacent_to, sub_relative_indices_per_gpi =  get_adjacency_tile(j, coa_grid_points[j], gp_adjacency_matrix[j], coa_grid_points, gps_processing_status, j_relative_index)
                gps_processing_status[j] = 1
                
                adjacent_to += sub_adjacent_to
                relative_indices_per_gpi += sub_relative_indices_per_gpi
                
                if current_polygon_matrix_size == 0:
                    current_polygon_matrix_size += 3
                else:
                    current_polygon_matrix_size += 1
                    
                coa_data.append(coa_data_per_gridpoint[j])
        
            if current_polygon_matrix_size > 0:
                polygon_matrix = np.zeros((current_polygon_matrix_size, current_polygon_matrix_size), dtype=np.int16)
                
                polygon_center = np.array([0, 0]) + (int(current_polygon_matrix_size/2) + 1)
                for k, rk in relative_indices_per_gpi:
                    polygon_matrix[polygon_center + rk] = k
                    
                polygon_matrices.append(polygon_matrix)
                average_coa_data_per_polygon.append(np.average(coa_data))
            
    simplified_polygons = convert_polygon_matrices_into_polygons(polygon_matrices, resolution)
            
    return simplified_polygons, average_coa_data_per_polygon