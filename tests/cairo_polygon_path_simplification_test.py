# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:01:31 2018

@author: Brian
"""

import core.geometry as geometry
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.sparse as sparsemat

@nb.jit(nopython=True)
def float_cartesian_product(xs, ys):
    num_xs = xs.shape[0]
    num_ys = ys.shape[0]
    products = np.zeros((num_xs*num_ys, 2), dtype=np.float64)
    
    for i in range(xs.shape[0]):
        x = xs[i]
        for j in range(ys.shape[0]):
            y = ys[j]
            products[i*num_ys + j][0] = x
            products[i*num_ys + j][1] = y
            
    return products

@nb.jit(nopython=True)
def calculate_gp_adjacency_vector_relative_to_focus(focus_index, coa_grid_points, coa_data_per_gridpoint, resolution):
    num_gps = coa_grid_points.shape[0]
    gp_adjacency_vector = -1*np.ones(4, dtype=np.int16)
    overres = 1.05*resolution
    
    ith_gp = coa_grid_points[focus_index]
    counter = 0
    for j in range(num_gps):
        if j != focus_index:
            dist = geometry.calculate_dist_between_points_given_vectors(ith_gp, coa_grid_points[j])
                                                       
            if dist < overres:    
                gp_adjacency_vector[counter] = j
                counter += 1
                    
    return gp_adjacency_vector

# --------------------------------------
def get_relative_of_j_around_i_in_polygon_matrix(i, j, ith_gp, jth_gp, i_offset):
    relative = jth_gp - ith_gp
    
    dir_offset, offset = 0, 0.0
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
@nb.jit(nopython=True)
def numba_get_data_adjacent_gpis(focus_index, focus_adjacency_vector, coa_data_per_gridpoint, similarity_threshold, gps_processing_status, current_polygon_member_gpis):
    pmgpis_coa = -1*np.ones_like(current_polygon_member_gpis, dtype=np.float64)    
    for i, gpi in enumerate(current_polygon_member_gpis):
        if not gpi < 0:
            pmgpis_coa[i] = coa_data_per_gridpoint[gpi]
            
    data_adjacency_factors = -1*np.ones_like(focus_adjacency_vector, dtype=np.float64)
    for i, gpi in enumerate(focus_adjacency_vector):
        if not gpi < 0:
            cursor = 0
            dafs = -1*np.ones_like(pmgpis_coa)
            coa_at_gpi = coa_data_per_gridpoint[gpi]
            
            for member_coa in pmgpis_coa:
                if not member_coa < 0:
                    if member_coa == 0:
                        dafs[cursor] = np.abs(member_coa - coa_at_gpi)/1e-16
                    else:
                        dafs[cursor] = np.abs(member_coa - coa_at_gpi)/member_coa
                    cursor += 1
                
            max_dafks = np.max(dafs)
            if not max_dafks < 0:
                data_adjacency_factors[i] = max_dafks
    
    data_adjacent_gpis = -1*np.ones_like(focus_adjacency_vector, dtype=np.int64)
    for i, daf in enumerate(data_adjacency_factors):
        gpi = focus_adjacency_vector[i]
        if not daf < 0 and daf < similarity_threshold and gps_processing_status[gpi] == 0:
            data_adjacent_gpis[i] = gpi

    return data_adjacent_gpis

# ---------------------------------
    
def get_data_adjacent_gpis(focus_index, focus_adjacency_vector, coa_data_per_gridpoint, similarity_threshold, gps_processing_status, current_polygon_member_gpis):
    adjacent_gps = [x for x in np.arange(focus_adjacency_vector.shape[0]) if ((not focus_adjacency_vector[x] < 0) and (not gps_processing_status[x]))]
            
    pmgpis_coa = [coa_data_per_gridpoint[x] for x in current_polygon_member_gpis]
    
    data_adjacency_factors = []
    for k in adjacent_gps:
        coa_k = coa_data_per_gridpoint[k]
        dafsk = [np.abs(coa - coa_k)/coa for coa in pmgpis_coa]
        data_adjacency_factors.append(np.max(dafsk))
    
    data_adjacent_gpis = [k for k, daf in zip(adjacent_gps, data_adjacency_factors) if daf < similarity_threshold]
    
    return data_adjacent_gpis

# --------------------------------------
    
def get_adjacency_tile(focus_index, focus_index_gp, focus_index_offset_in_polygon_matrix, relative_to_focus_adjacency_vector, coa_grid_points, coa_data_per_gridpoint, gps_processing_status, similarity_threshold, current_polygon_member_gpis):
         
    adjacent_to = numba_get_data_adjacent_gpis(focus_index, relative_to_focus_adjacency_vector, coa_data_per_gridpoint, similarity_threshold, gps_processing_status, np.array(current_polygon_member_gpis))
    
    return [x for x in adjacent_to if x != -1]

# --------------------------------------
def get_relative_index(relative_indices_per_gpi, i):
    for k, relative_index in relative_indices_per_gpi:
        if k == i:
            return relative_index
        
# --------------------------------------

def rotate_cursor_direction_CW(cursor_direction):
    x, y = cursor_direction
    
    return np.array([y, -1*x], dtype=np.int64)

# --------------------------------------

def rotate_cursor_direction_CCW(cursor_direction):
    x, y = cursor_direction
    return np.array([-1*y, x], dtype=np.int64)

# --------------------------------------
def is_cursor_location_valid(cursor, polygon_matrix):
    x, y = cursor
    if np.any(cursor < 0) or x >= polygon_matrix.shape[0] or y >= polygon_matrix.shape[1] or polygon_matrix[x][y] < 0:
        return False
    
    return True

# --------------------------------------
def one_tile_greedy_cursor_rotate(cursor_direction, cursor, polygon_matrix):    
    # would like to make a CCW turn
    cursor_direction = rotate_cursor_direction_CCW(cursor_direction)
    new_cursor = cursor_direction + cursor
    
    if is_cursor_location_valid(new_cursor, polygon_matrix):
        return cursor_direction, new_cursor
    else:
        cursor_direction = rotate_cursor_direction_CW(cursor_direction)
    
    # could not make a CCW turn here, what about if we move a teeny bit?
    new_cursor = cursor + rotate_cursor_direction_CCW(cursor_direction) + cursor_direction
    if is_cursor_location_valid(new_cursor, polygon_matrix):
        return rotate_cursor_direction_CCW(cursor_direction), new_cursor
        
    # could not make a CCW turn, can continue to point in same direction?
    new_cursor = cursor_direction + cursor
    if is_cursor_location_valid(new_cursor, polygon_matrix):
        return cursor_direction, new_cursor
    else:
        # continuing to point in direction we are going failed, turn CW
        cursor_direction = rotate_cursor_direction_CW(cursor_direction)
        return cursor_direction, cursor
    
    return cursor_direction, cursor

# --------------------------------------
#@nb.jit(nopython=True)  
def convert_polygon_matrix_into_polygon(polygon_matrix, coa_grid_points, resolution):
    cursor_direction = np.array([0, -1], dtype=np.int64)
    
    num_gps = 0
    for x in polygon_matrix.ravel():
        if not x < 0:
            num_gps += 1
    
    polygon_vertices = np.zeros((num_gps*4, 2), dtype=np.float64)
    x_size, y_size = polygon_matrix.shape 
    
    #print("===========")
    start_vertex = np.zeros(2, dtype=np.int64)
    start_vertex_index = -1
    cursor = np.zeros(2, dtype=np.int64)
    loop_break = False
    for i in range(x_size):
        i_ = x_size - 1 - i
        for j in range(y_size):
            j_ = y_size - 1 - j
#            print("-----------")
#            print("i_,j_: {}".format((i_, j_)))
            cell_content = polygon_matrix[i_][j_]
            #print("polygon_matrix[i_][j_]: {}".format(cell_content))
            if cell_content >= 0:
                start_vertex_index = cell_content
                start_vertex = coa_grid_points[cell_content] + 0.5*resolution*np.ones(2, dtype=np.float64)
                cursor[0] = i_
                cursor[1] = j_
                loop_break = True
                break
        if loop_break:
            break
    
    if start_vertex_index < 0:
        return np.zeros((0, 2), dtype=np.float64)

    polygon_vertices[0] = start_vertex
    polygon_vertices[1] = start_vertex + resolution*cursor_direction
    polygon_vertex_index = 2
    
#    print("-----------")
#    print("poly_matrix")
#    print(polygon_matrix)
    
    done = False
    while not done:
        #print("***********")
        cursor_direction, cursor = one_tile_greedy_cursor_rotate(cursor_direction, cursor, polygon_matrix)
        #print("cursor_direction: {}".format(cursor_direction))
        new_vertex = polygon_vertices[polygon_vertex_index - 1] + resolution*cursor_direction
        
#        print("-----------")
#        print("start_vertex: {}".format(start_vertex))
#        print("new_vertex: {}".format(new_vertex))
#        print("-----------")
#        print("new cursor: {}".format(cursor))
        
        if not np.all(np.abs(new_vertex - start_vertex) < 1e-10):
            polygon_vertices[polygon_vertex_index] = new_vertex
            polygon_vertex_index += 1
        else:
            done = True
        #print("***********")
        
    #print("poly_vertices")
    #print(polygon_vertices[:polygon_vertex_index])
    #print("===========")
    return polygon_vertices[:polygon_vertex_index]
        
# --------------------------------------

def convert_polygon_matrices_into_polygons(polygon_matrices, coa_grid_points, resolution):
    polygons = [convert_polygon_matrix_into_polygon(pm, coa_grid_points, resolution) for pm in polygon_matrices]
    return polygons

# --------------------------------------

def get_simplified_polygon_matrices_and_coa_data(coa_grid_points, coa_data_per_gridpoint, resolution, similarity_threshold):
    num_gps = coa_grid_points.shape[0]
    gps_processing_status = np.zeros(num_gps, dtype=np.int16)
    
    polygon_matrices = []
    average_coa_data_per_polygon = []

    for i in range(num_gps):
        if gps_processing_status[i] == 0:
            gp_process_percentage = 100*np.round(np.sum(gps_processing_status)/num_gps, decimals=8)
            print("gps processed: {}%".format(gp_process_percentage))
            coa_data = []
        
            gpi_indices = []
            relative_indices_per_gpi = [(i, np.array([0, 0]))]
            adjacent_to = [i]
            
            while len(adjacent_to) != 0:
                print("current len of adjacent_to: {}".format(len(adjacent_to)))
                j = adjacent_to.pop(0)
                this_gp = coa_grid_points[j]
                gpi_indices.append(j)
                
                j_relative_index = get_relative_index(relative_indices_per_gpi, j)
                # focus_index, focus_index_gp, relative_to_focus_adjacency, coa_grid_points, gps_processing_status, focus_index_offset_in_polygon_matrix
                relative_to_focus_adjacency_vector = calculate_gp_adjacency_vector_relative_to_focus(j, coa_grid_points, coa_data_per_gridpoint, resolution)
                sub_adjacent_to =  get_adjacency_tile(j, this_gp, j_relative_index, relative_to_focus_adjacency_vector, coa_grid_points, coa_data_per_gridpoint, gps_processing_status, similarity_threshold, gpi_indices)
                sub_adjacent_to = [x for x in sub_adjacent_to if x not in adjacent_to]
                sub_relative_indices_per_gpi = [(k, get_relative_of_j_around_i_in_polygon_matrix(j, k, this_gp, coa_grid_points[k], j_relative_index)) for k in sub_adjacent_to]
                gps_processing_status[j] = 1
                
                adjacent_to += sub_adjacent_to
                relative_indices_per_gpi += sub_relative_indices_per_gpi
                    
                coa_data.append(coa_data_per_gridpoint[j])
        
            relative_indices = np.array([ri for _, ri in relative_indices_per_gpi])
            x_size = np.max(relative_indices[:,0]) - np.min(relative_indices[:,0]) + 1
            y_size = np.max(relative_indices[:,1]) - np.min(relative_indices[:,1]) + 1
            
            polygon_matrix = -1*np.ones((x_size, y_size), dtype=np.int16)
            polygon_center = np.zeros(2, dtype=np.int64)
            
            x_min = np.min(relative_indices[:,0])
            if x_min < 0:
                polygon_center[0] = np.abs(x_min)
                
            y_min = np.min(relative_indices[:,1])
            if y_min < 0:
                polygon_center[1] = np.abs(y_min)
                
            for k, rk in relative_indices_per_gpi:
                x, y = polygon_center + rk
                polygon_matrix[x][y] = k
            
            #polygon_matrix = np.transpose([np.flip(x, 0) for x in polygon_matrix])
            polygon_matrices.append(polygon_matrix)
            average_coa_data_per_polygon.append(np.average(coa_data))
    
    print("Converting matrices into polygons...")
    simplified_polygons = convert_polygon_matrices_into_polygons(polygon_matrices, coa_grid_points, resolution)
            
    return simplified_polygons, average_coa_data_per_polygon

# --------------------------------------                  
    
resolution = 100

refresh = False
graph = True
load = False

if refresh and not load:
    grid_size = 3
    x_point_bdries = resolution*np.arange(grid_size + 1)
    y_point_bdries = resolution*np.arange(grid_size + 1)
    
    x_points = np.array([(x1 + x2)/2. for x1, x2 in zip(x_point_bdries[1:], x_point_bdries[:-1])], dtype=np.float64)
    y_points = np.array([(y1 + y2)/2. for y1, y2 in zip(y_point_bdries[1:], y_point_bdries[:-1])], dtype=np.float64)
    
    all_grid_points = float_cartesian_product(x_points, y_points)
elif load:
    all_grid_points = np.load("test_all_gridpoints.npy")
    x_point_bdries = np.sort([gp[0] - resolution*0.5 for gp in all_grid_points])
    y_point_bdries = np.sort([gp[1] - resolution*0.5 for gp in all_grid_points])

num_grid_points = all_grid_points.shape[0]

if graph:
    fig, ax = plt.subplots()
    
    for xbdry, ybdry in zip(x_point_bdries, y_point_bdries):
        ax.axvline(xbdry, color='k')
        ax.axhline(ybdry, color='k')
        
    ax.set_aspect('equal')
    ax.set_xlim([np.min(x_point_bdries), np.max(x_point_bdries)])
    ax.set_ylim([np.min(y_point_bdries), np.max(y_point_bdries)])
    
    for i, gp in enumerate(all_grid_points):
        ax.plot(gp[:1], gp[1:], color='r', marker='.', ls='')

if refresh and not load:
    num_valid_gps = np.random.randint(int(0.5*num_grid_points), num_grid_points)
    valid_gp_true_indices = np.random.choice(np.arange(num_grid_points), size=num_valid_gps, replace=False)
    
    valid_gps = all_grid_points[valid_gp_true_indices]
    
    valid_gp_indices = np.arange(num_valid_gps)
    
    family1size = int((num_valid_gps + 1)*np.random.rand())
    family1_indices = np.sort(np.random.choice(valid_gp_indices, size=family1size, replace=False))
    
    family2size = num_valid_gps - family1size
    if num_valid_gps - family1size > 0:
        family2_indices = np.array([x for x in valid_gp_indices if x not in family1_indices])
    else:
        family2_indices = np.zeros([], dtype=np.int16)
    
    family1 = valid_gps[family1_indices]
    family2 = valid_gps[family2_indices]
    
#    coa_data = np.array([np.round(np.random.lognormal(mean=0.0, sigma=0.1), decimals=2) if (i in family1_indices) else np.round(np.random.lognormal(mean=0.1, sigma=0.1), decimals=2) for i in range(num_valid_gps)])
    coa_data = np.array([1.0 if (i in family1_indices) else 2.0 for i in range(num_valid_gps)])
    similarity_threshold = 0.5
elif load:
    valid_gps = np.load("test_coa_gridpoints.npy")
    coa_data = np.load("test_coa_data_per_gridpoint.npy")
    similarity_threshold = 0.05

if graph and not load:
    if family1size > 0:
        for i, gp in enumerate(family1):
            ax.plot(gp[:1], gp[1:], color='b', marker='.', ls='')
            gpi = family1_indices[i]
            ax.annotate("{}, {}".format(gpi, coa_data[gpi]), gp)
    
    if family2size > 0:
        for i, gp in enumerate(family2):
            ax.plot(gp[:1], gp[1:], color='g', marker='.', ls='')
            gpi = family2_indices[i]
            ax.annotate("{}, {}".format(gpi, coa_data[gpi]), gp)  
    fig.savefig("path_simplification_test_matrix.png")
elif graph and load:
    for gpi, gp in enumerate(valid_gps):
        ax.plot(gp[:1], gp[1:], color='b', marker='.', ls='')
        ax.annotate("{},\n{}".format(gpi, np.round(coa_data[gpi], decimals=2)), gp)
    
    fig.savefig("path_simplification_test_matrix.png")
    
fig.show()

simplified_polygons, average_coa_data_per_polygon = get_simplified_polygon_matrices_and_coa_data(valid_gps, coa_data, resolution, similarity_threshold)

if graph:
    for polygon in simplified_polygons:
        polygon = np.append(polygon, polygon[:1], axis=0)
        ax.plot(polygon[:,0], polygon[:,1], linewidth=3)
        
    fig.savefig("path_simplification_test_matrix.png")

