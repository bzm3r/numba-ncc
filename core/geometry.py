# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:22:06 2015

@author: Brian
"""
from __future__ import division
import numpy as np
import math
import numba as nb

# ----------------------------------------------------------------

def calculate_cluster_centroid(points):
    return np.array([np.average(points[:,0]), np.average(points[:,1])])

# ----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_centroid(num_vertices, polygon_coords):
    A = calculate_polygon_area(num_vertices, polygon_coords)
    
    Cx = 0.0
    Cy = 0.0
    
    centroid = np.empty(2, dtype=np.float64)
    
    for vi in range(num_vertices):
        vi_plus_1 = (vi + 1)%num_vertices
        
        x_i, y_i = polygon_coords[vi]
        x_i_plus_1, y_i_plus_1 = polygon_coords[vi_plus_1]
        
        coord_combination = x_i*y_i_plus_1 - x_i_plus_1*y_i
        
        Cx = Cx + (x_i + x_i_plus_1)*coord_combination
        Cy = Cy + (y_i + y_i_plus_1)*coord_combination
    
    area_factor = (1/(6*A))
    
    centroid[0] = area_factor*Cx
    centroid[1] = area_factor*Cy
    
    return centroid
        
# ----------------------------------------------------------------
@nb.jit(nopython=True)  
def determine_rotation_matrix_to_rotate_vector1_to_lie_along_vector2(vector1, vector2):
    mag1 = calculate_2D_vector_mag(vector1)
    mag2 = calculate_2D_vector_mag(vector2)
    
    u1, u2 = vector1
    v1, v2 = vector2
    
    dot_prod = u1*v1 + u2*v2
    cross_prod_mag = u1*v2 - u2*v1
    
    mag1_times_mag2 = mag1*mag2
    sin_theta = cross_prod_mag/(mag1_times_mag2) 
    cos_theta = dot_prod/(mag1_times_mag2)
    
    rotation_matrix = np.zeros((2, 2), dtype=np.float64)
    
    rotation_matrix[0][0] = cos_theta
    rotation_matrix[0][1] = -1*sin_theta
    rotation_matrix[1][0] = sin_theta
    rotation_matrix[1][1] = cos_theta
    
    return rotation_matrix

# ----------------------------------------------------------------
@nb.jit(nopython=True)   
def rotate_vector(theta, vector):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    x, y = vector
    
    new_x = cos_theta*x - sin_theta*y
    new_y = sin_theta*x + cos_theta*y
    
    new_vector = np.empty(2)
    new_vector[0] = new_x
    new_vector[1] = new_y
    
    return new_vector
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def rotate_2D_vector_CCW(vector):
    x, y = vector
    
    result_vector = np.empty(2, dtype=np.float64)
    
    result_vector[0] = -1.0*y
    result_vector[1] = x
    
    return result_vector
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def rotate_2D_vectors_CCW(vectors):
    num_vectors = vectors.shape[0]
    rotated_vectors = np.empty_like(vectors)
    
    for i in range(num_vectors):
        rotated_vector = rotate_2D_vector_CCW(vectors[i])
        for j in range(2):
            rotated_vectors[i, j] = rotated_vector[j]
        
    return rotated_vectors
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_2D_vector_mag(vector):
    x = vector[0]
    y = vector[1]
    
    return math.sqrt(x*x + y*y)
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def calculate_2D_vector_mags(num_vectors, vectors):
    vector_mags = np.empty(num_vectors)
    
    for i in range(num_vectors):
        vector_mags[i] = calculate_2D_vector_mag(vectors[i])
        
    return vector_mags
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_2D_vector_direction(vector):
    x = vector[0]
    y = vector[1]
    
    abs_x = abs(x)
    abs_y = abs(y)
    
    if abs_x < 1e-10:
        if abs_y < 1e-10:
            return 0.0
        else:
            phi = np.pi/2
    else:
        phi = np.arctan(abs_y/abs_x)
            
    if x >= 0:
        if y >= 0:
            return phi
        else:
            return 2*np.pi - phi
    else:
        if y >= 0:
            return np.pi - phi
        else:
            return np.pi + phi
            
# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def calculate_2D_vector_directions(num_vectors, vectors):
    vector_dirns = np.empty(num_vectors)
    
    for i in range(num_vectors):
        vector_dirns[i] = calculate_2D_vector_direction(vectors[i])
        
    return vector_dirns
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def normalize_2D_vector(vector):
    normalized_vector = np.empty(2, dtype=np.float64)

    mag = calculate_2D_vector_mag(vector)
    x, y = vector

    normalized_vector[0] = x/mag
    normalized_vector[1] = y/mag
    
    return normalized_vector
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)         
def normalize_vectors(num_vectors, vectors):
    normalized_vectors = np.empty((num_vectors, 2), dtype=np.float64)
    
    for i in range(num_vectors):
        vector = vectors[i]
        
        normalized_vector = normalize_2D_vector(vector)
        
        normalized_vectors[i, 0] = normalized_vector[0]
        normalized_vectors[i, 1] = normalized_vector[1]
        
    return normalized_vectors
    
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_edge_vectors(num_nodes, node_coords):
    edge_displacement_vectors = np.empty((num_nodes, 2), dtype=np.float64)
    for i in range(num_nodes):
        i_plus1 = (i+1)%num_nodes
        
        xA, yA = node_coords[i]
        xB, yB = node_coords[i_plus1]
        
        edge_displacement_vectors[i, 0] = xB - xA
        edge_displacement_vectors[i, 1] = yB - yA
        
    return edge_displacement_vectors
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def get_vector_normals(num_vectors, vectors):
    normal_to_vectors = np.empty((num_vectors, 2), dtype=np.float64)
    
    for i in range(num_vectors):
        x, y = vectors[i]
        normal_to_vectors[i, 0] = -1*y
        normal_to_vectors[i, 1] = x
        
    return normal_to_vectors

# -----------------------------------------------------------------     
@nb.jit(nopython=True)  
def roll_2D_vectors(roll_number, num_vectors, vectors):
    rolled_vectors = np.empty((num_vectors, 2), dtype=np.float64)
    
    for i in range(num_vectors):
        rolled_i = (i - roll_number)%num_vectors
        x, y = vectors[rolled_i]
        rolled_vectors[i, 0] = x
        rolled_vectors[i, 1] = y
        
    return rolled_vectors

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def sum_2D_vectors(num_elements, v1s, v2s):
    result = np.empty((num_elements, 2), dtype=np.float64)
    
    for i in range(num_elements):
        v1 = v1s[i]
        v2 = v2s[i]
        
        v1x, v1y = v1
        v2x, v2y = v2
        
        result[i, 0] = v1x + v2x
        result[i, 1] = v1y + v2y
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def difference_2D_vectors(num_elements, v1s, v2s):
    result = np.empty((num_elements, 2), dtype=np.float64)
    
    for i in range(num_elements):
        v1 = v1s[i]
        v2 = v2s[i]
        
        v1x, v1y = v1
        v2x, v2y = v2
        
        result[i, 0] = v1x - v2x
        result[i, 1] = v1y - v2y
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_unit_inside_pointing_vecs(node_coords):
    num_nodes = node_coords.shape[0]
    
    unit_inside_pointing_vecs = np.empty((num_nodes, 2), dtype=np.float64)
    
    tangent_vector = np.empty(2, dtype=np.float64)
    
    for i in range(num_nodes):
        i_plus_1 = (i + 1)%num_nodes
        
        edge_vector_to_plus = node_coords[i_plus_1] - node_coords[i]
        edge_vector_to_plus_normalized = normalize_2D_vector(edge_vector_to_plus)
        
        i_minus_1 = (i - 1)%num_nodes
        
        edge_vector_from_minus = node_coords[i] - node_coords[i_minus_1]
        edge_vectors_from_minus_normalized = normalize_2D_vector(edge_vector_from_minus)
        
        tangent_vector[0] = edge_vector_to_plus_normalized[0] + edge_vectors_from_minus_normalized[0]
        tangent_vector[1] = edge_vector_to_plus_normalized[1] + edge_vectors_from_minus_normalized[1]
        
        tangent_vector_normalized = normalize_2D_vector(tangent_vector)

        x_part, y_part = rotate_2D_vector_CCW(tangent_vector_normalized)
        unit_inside_pointing_vecs[i, 0] = x_part
        unit_inside_pointing_vecs[i, 1] = y_part
    
    return unit_inside_pointing_vecs

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_unit_inside_pointing_vecs_per_timestep(node_coords_per_timestep):
    num_timesteps = node_coords_per_timestep.shape[0]
    num_nodes = node_coords_per_timestep.shape[1]
    
    unit_inside_pointing_vecs_per_timestep = np.empty((num_timesteps, num_nodes, 2), dtype=np.float64)
    
    for t in range(num_timesteps):
        unit_inside_pointing_vecs_per_timestep[t] = calculate_unit_inside_pointing_vecs(node_coords_per_timestep[t])
    
    return unit_inside_pointing_vecs_per_timestep

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_tangent_vecs_and_inside_pointing_vecs(num_nodes, node_coords):

    unit_inside_pointing_vecs = np.empty((num_nodes, 2), dtype=np.float64)
    tangent_vectors = np.empty((num_nodes, 2), dtype=np.float64)
    
    tangent_vector = np.empty(2, dtype=np.float64)
    for i in range(num_nodes):
        i_plus_1 = (i + 1)%num_nodes
        
        edge_vector_to_plus = calculate_vector_from_p1_to_p2_given_vectors(node_coords[i], node_coords[i_plus_1])
        edge_vector_to_plus_normalized = normalize_2D_vector(edge_vector_to_plus)
        
        i_minus_1 = (i - 1)%num_nodes
        
        edge_vector_from_minus = calculate_vector_from_p1_to_p2_given_vectors(node_coords[i_minus_1], node_coords[i])
        edge_vectors_from_minus_normalized = normalize_2D_vector(edge_vector_from_minus)
        
        tangent_vector[0] = edge_vector_to_plus_normalized[0] + edge_vectors_from_minus_normalized[0]
        tangent_vector[1] = edge_vector_to_plus_normalized[1] + edge_vectors_from_minus_normalized[1]
        
        tangent_vector_normalized = normalize_2D_vector(tangent_vector)
        tangent_vectors[i, 0], tangent_vectors[i, 1] = tangent_vector_normalized
        
        normal_to_tangent_vector = rotate_2D_vector_CCW(tangent_vector_normalized)
        unit_inside_pointing_vecs[i, 0], unit_inside_pointing_vecs[i, 1] = normal_to_tangent_vector
    
    return unit_inside_pointing_vecs, tangent_vectors
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def check_sign_projection_of_a_on_b(a, b):
    '''
    The magnitude of the projection of a on b is given by:
        (a \dot b)/|b|
    
    However, |b| is always >= 0.
    
    Thus, the sign is determined solely by the sign of a \dot b.
    '''
    if (a[0]*b[0] + a[1]*b[1]) > 0:
        return 1
    else:
        return 0
    
# -----------------------------------------------------------------
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
    
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def calculate_bounding_box_polygon(polygon):
    num_vertices = polygon.shape[0]
    
    min_x = 0.0
    max_x = 0.0
    
    min_y = 0.0
    max_y = 0.0
    
    for i in range(num_vertices):
        if i == 0:
            min_x = polygon[i][0]
            max_x = min_x
            
            min_y = polygon[i][1]
            max_y = min_y
        else:
            this_x = polygon[i][0]
            this_y = polygon[i][1]
            
            if this_x < min_x:
                min_x = this_x
            elif this_x > max_x:
                max_x = this_x
            
            if this_y < min_y:
                min_y = this_y
            elif this_y > max_y:
                max_y = this_y
                
    bounding_box_polygon = np.zeros((4, 2), dtype=np.float64)
    
    for i in range(4):
        if i < 2:
            bounding_box_polygon[i, 1] = min_y
            if i%2 == 0:
                bounding_box_polygon[i, 0] = min_x
            else:
                bounding_box_polygon[i, 0] = max_x 
        else:
            bounding_box_polygon[i, 1] = max_y
            if i%2 == 1:
                bounding_box_polygon[i, 0] = min_x
            else:
                bounding_box_polygon[i, 0] = max_x 
    
    
    return bounding_box_polygon

# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def create_initial_bounding_box_polygon_array(num_cells, num_nodes_per_cell, environment_cells_node_coords):
    bounding_box_polygon_array = np.zeros((num_cells, 4), dtype=np.float64)

    for ci in range(num_cells):
        bounding_box_polygon_array[ci] = calculate_polygon_bounding_box(environment_cells_node_coords[ci])
        
    return bounding_box_polygon_array

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def is_point_in_polygon_bounding_box_given_bounding_box(test_point, min_x, max_x, min_y, max_y):                
    tp_x = test_point[0]
    tp_y = test_point[1]
    
    if (min_x < tp_x < max_x) and (min_y < tp_y < max_y):
        return 1
    else:
        return 0
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def is_point_in_polygon_bounding_box(test_point, num_vertices, polygon):
    
    min_x, max_x, min_y, max_y = calculate_polygon_bounding_box(num_vertices, polygon)
    
    return is_point_in_polygon_bounding_box(test_point, min_x, max_x, min_y, max_y)
        
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def is_point_in_polygon(test_point, num_vertices, polygon):
    is_test_point_in_poly_bb = is_point_in_polygon_bounding_box(test_point, num_vertices, polygon)
    
    if is_test_point_in_poly_bb == 0:
        return 0
    else:
        return is_point_in_polygon_without_bb_check(test_point, num_vertices, polygon)
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def is_point_in_polygon_given_bounding_box(test_point, num_vertices, polygon, min_x, max_x, min_y, max_y):
    is_test_point_in_poly_bb = is_point_in_polygon_bounding_box_given_bounding_box(test_point, min_x, max_x, min_y, max_y)
    
    if is_test_point_in_poly_bb == 0:
        return 0
    else:
        return is_point_in_polygon_without_bb_check(test_point, num_vertices, polygon)

# -----------------------------------------------------------------       
@nb.jit(nopython=True)  
def is_point_in_polygons(point, polygons):
    num_polygons = polygons.shape[0]
    if num_polygons != 0:
        for pi in range(num_polygons):
            polygon = polygons[pi]
            if is_point_in_polygon(point, polygon.shape[0], polygon):
                return 1
            else:
                continue
    else:
        return 0
    
    return 0
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def are_points_inside_polygon(num_points, points, num_poly_vertices, polygon):
    results = np.zeros(num_points, dtype=np.int64)
    
    min_x, max_x, min_y, max_y = calculate_polygon_bounding_box(polygon)
    
    for index in range(num_points):
        test_point = points[index]
        
        results[index] = is_point_in_polygon_given_bounding_box(test_point, num_poly_vertices, polygon, min_x, max_x, min_y, max_y)
        
    return results

# -----------------------------------------------------------------
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
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def are_nodes_inside_physical_boundary(this_cell_index, num_nodes, all_cells_node_coords, exists_space_physical_bdry_polygon, space_physical_bdry_polygon):
    
    if exists_space_physical_bdry_polygon:
        return are_points_inside_polygon(num_nodes, all_cells_node_coords[this_cell_index], space_physical_bdry_polygon.shape[0], space_physical_bdry_polygon)
    else:
        return np.ones(num_nodes, dtype=np.int64)
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def calculate_polygon_area(num_vertices, vertex_coords):
    '''
    http://geomalgorithms.com/a01-_area.html
    '''
    area = 0
    for i in range(num_vertices):
        j = (i + 1)%num_vertices
        k = (i - 1)%num_vertices
        area += vertex_coords[i, 0] * (vertex_coords[j, 1] - vertex_coords[k, 1])
        
    return area*0.5
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def calculate_edgeplus_lengths(num_nodes, node_coords):
    edgeplus_lengths = np.zeros(num_nodes, dtype=np.float64)
    
    for ni in range(num_nodes):
        this_node_coord = node_coords[ni]
        plus_node_coord = node_coords[(ni + 1)%num_nodes]

        edgeplus_lengths[ni] = calculate_dist_between_points_given_vectors(this_node_coord, plus_node_coord)
    
    return edgeplus_lengths
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)     
def calculate_edgeminus_lengths(num_nodes, node_coords):
    edgeminus_lengths = np.zeros(num_nodes, dtype=np.float64)
    
    for ni in range(num_nodes):
        this_node_coord = node_coords[ni]
        minus_node_coord = node_coords[(ni - 1)%num_nodes]

        edgeminus_lengths[ni] = calculate_dist_between_points_given_vectors(this_node_coord, minus_node_coord)
    
    return edgeminus_lengths

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_average_edge_lengths(num_nodes, node_coords):
    edgeplus_lengths = calculate_edgeplus_lengths(num_nodes, node_coords)
    edgeminus_lengths = calculate_edgeminus_lengths(num_nodes, node_coords)
    
    average_edge_lengths = np.zeros(num_nodes, dtype=np.float64)
    
    for ni in range(num_nodes):
        average_edge_lengths[ni] = 0.5*edgeplus_lengths[ni] + 0.5*edgeminus_lengths[ni]
        
    return average_edge_lengths
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_perimeter(num_nodes, node_coords):
    edgeplus_lengths  = calculate_edgeplus_lengths(num_nodes, node_coords)

    perimeter = 0.0  
    for ni in range(num_nodes):
        perimeter += edgeplus_lengths[ni]
    
    return perimeter
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def multiply_vectors_by_scalars(num_elements, vectors, scalars):
    result = np.empty((num_elements, 2))
    for i in range(num_elements):
        x, y = vectors[i]
        scalar = scalars[i]
        result[i, 0] = scalar*x
        result[i, 1] = scalar*y
        
    return result
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def multiply_vectors_by_scalar(num_elements, vectors, scalar):
    result = np.empty((num_elements, 2))
    
    for i in range(num_elements):
        x, y = vectors[i]
        result[i, 0] = scalar*x
        result[i, 1] = scalar*y
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    
    return dx*dx + dy*dy
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def create_initial_distance_squared_matrix(num_cells, num_nodes_per_cell, init_all_cells_node_coords):
    distance_squared_matrix = -1*np.ones((num_cells, num_nodes_per_cell, num_cells, num_nodes_per_cell), dtype=np.float64)
    info_update_tracker = np.zeros_like(distance_squared_matrix, dtype=np.int64)
    
    for ci in range(num_cells):
        for ni in range(num_nodes_per_cell):
            for other_ci in range(num_cells):
                if ci != other_ci:
                    relevant_info_update_tracker_slice = info_update_tracker[ci][ni][other_ci]
                    for other_ni in range(num_nodes_per_cell):
                        if relevant_info_update_tracker_slice[other_ni] != 1:
                            info_update_tracker[ci][ni][other_ci][other_ni] = 1
                            info_update_tracker[other_ci][other_ni][ci][ni] = 1
                            
                            this_node = init_all_cells_node_coords[ci][ni]
                            other_node = init_all_cells_node_coords[other_ci][other_ni]
                            
                            squared_dist = calculate_squared_dist(this_node, other_node)
                            distance_squared_matrix[ci][ni][other_ci][other_ni] = squared_dist
                            distance_squared_matrix[other_ci][other_ni][ci][ni] = squared_dist
                            
    return distance_squared_matrix

# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def update_distance_squared_matrix(last_updated_cell_index, num_cells, num_nodes_per_cell, all_cells_node_coords, distance_squared_matrix):
    
    # 1 if info has been updated, 0 otherwise
    info_update_tracker = np.zeros_like(distance_squared_matrix, dtype=np.int64)
    
    for ni in range(num_nodes_per_cell):
        for other_ci in range(num_cells):
            if other_ci != last_updated_cell_index:
                relevant_info_update_tracker_slice = info_update_tracker[last_updated_cell_index][ni][other_ci]
                for other_ni in range(num_nodes_per_cell):
                    if relevant_info_update_tracker_slice[other_ni] != 1:
                        info_update_tracker[last_updated_cell_index][ni][other_ci][other_ni] = 1
                        info_update_tracker[other_ci][other_ni][last_updated_cell_index][ni] = 1
                        
                        this_node = all_cells_node_coords[last_updated_cell_index][ni]
                        other_node = all_cells_node_coords[other_ci][other_ni]
                        
                        squared_dist = calculate_squared_dist(this_node, other_node)
                        distance_squared_matrix[last_updated_cell_index][ni][other_ci][other_ni] = squared_dist
                        distance_squared_matrix[other_ci][other_ni][last_updated_cell_index][ni] = squared_dist
                            
    return distance_squared_matrix
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def calculate_vector_from_p1_to_p2_given_vectors(p1, p2):
    p1x = p1[0]
    p1y = p1[1]
    
    p2x = p2[0]
    p2y = p2[1]
    
    result = np.empty(2, dtype=np.float64)
    
    result[0] = p2x - p1x
    result[1] = p2y - p1y
    
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_vector_from_p1_to_p2_given_coords(p1x, p1y, p2x, p2y):
    
    result = np.empty(2, dtype=np.float64)
    
    result[0] = p2x - p1x
    result[1] = p2y - p1y
    
    return result
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def calculate_projection_of_a_on_b(a, b):
    b_mag = calculate_2D_vector_mag(b)
    
    return (a[0]*b[0] + a[1]*b[1])/b_mag
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_dist_between_points_given_vectors(p1, p2):
    displacement_vector = p2 - p1
    distance = calculate_2D_vector_mag(displacement_vector)
    
    return distance

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_dist_between_points_given_coords(p1x, p1y, p2x, p2y):
    displacement_vector = calculate_vector_from_p1_to_p2_given_coords(p1x, p1y, p2x, p2y)
    distance = calculate_2D_vector_mag(displacement_vector)
    
    return distance
    
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)          
def find_closest_node_on_other_cells_for_each_node_on_this_cell(num_cells, num_nodes, this_ci, dist_squared_array):
    closest_nodes_on_other_cells = -1*np.ones((num_nodes, num_cells), dtype=np.int64)
    
    for ni in range(num_nodes):
        this_node_dist_squared_array = dist_squared_array[ni]
        for ci in range(num_cells):
            if ci != this_ci:
                closest_node = -1
                closest_node_dist = -1
                
                this_node_dist_squared_wrt_other_cell = this_node_dist_squared_array[ci]
                for other_ni in range(num_nodes):
                    dist_bw_nodes = this_node_dist_squared_wrt_other_cell[other_ni]
                    
                    if other_ni == 0 or dist_bw_nodes < closest_node_dist:
                        closest_node = other_ni
                        closest_node_dist = dist_bw_nodes
                
                closest_nodes_on_other_cells[ni][ci] = closest_node
    
    return closest_nodes_on_other_cells
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)                            
def calculate_closest_point_dist_squared(num_nodes, this_nc, other_cell_node_coords, closest_ni):
    closest_ni_plus1 = (closest_ni + 1)%num_nodes  
    closest_ni_minus1 = (closest_ni - 1)%num_nodes
    
    closest_nc = other_cell_node_coords[closest_ni]
    closest_nc_plus1 = other_cell_node_coords[closest_ni_plus1]
    closest_nc_minus1 = other_cell_node_coords[closest_ni_minus1]
    
    closest_to_this = this_nc - closest_nc
    closest_to_plus1 = closest_nc_plus1 - closest_nc
    closest_to_minus1 = closest_nc_minus1 - closest_nc
    
    proj_for_plus1 = calculate_projection_of_a_on_b(closest_to_this, closest_to_plus1)
    
    if 0 < proj_for_plus1 and proj_for_plus1 < 1:
        closest_pc = closest_nc + proj_for_plus1*closest_to_plus1
        return calculate_squared_dist(this_nc, closest_pc)
        
    proj_for_minus1 = calculate_projection_of_a_on_b(closest_to_this, closest_to_minus1)
    if 0 < proj_for_minus1 and proj_for_minus1 < 1:
        closest_pc = closest_nc + proj_for_minus1*closest_to_minus1
        return calculate_squared_dist(this_nc, closest_pc)
    
    return -1
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)      
def do_close_points_to_each_node_on_other_cells_exist(num_cells, num_nodes, this_ci, this_cell_node_coords, dist_squared_array, closeness_dist_squared_criteria, all_cells_node_coords, are_nodes_inside_other_cells):
    close_points_exist = np.zeros((num_nodes, num_cells), dtype=np.int64)
    
    closest_nodes_on_other_cells = find_closest_node_on_other_cells_for_each_node_on_this_cell(num_cells, num_nodes, this_ci, dist_squared_array)
    
    for ni in range(num_nodes):
        closest_nodes_to_this_node = closest_nodes_on_other_cells[ni]
        relevant_dist_squared_array_slice = dist_squared_array[ni]
        this_nc = this_cell_node_coords[ni]
        
        for ci in range(num_cells):
            if ci != this_ci:
                closest_ni = closest_nodes_to_this_node[ci]
                closest_node_dist = relevant_dist_squared_array_slice[ci][closest_ni]
                
                other_cell_node_coords = all_cells_node_coords[ci]
                
                if closest_node_dist < closeness_dist_squared_criteria:
                    close_points_exist[ni][ci] = 1
                    continue
                
                closest_point_dist = calculate_closest_point_dist_squared(num_nodes, this_nc, other_cell_node_coords, closest_ni)
                if closest_point_dist != -1 and closest_point_dist < closeness_dist_squared_criteria:
                    close_points_exist[ni][ci] = 1
                    continue
                
                if are_nodes_inside_other_cells[ni][ci] == 1:
                    close_points_exist[ni][ci] = 1
                    continue
        
    return close_points_exist
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def check_if_line_segment_intersects_polygon(a, b, normal_to_line_segment, polygon, ignore_vertex_index):
    num_vertices = polygon.shape[0]

    nls_x, nls_y = normal_to_line_segment
            
    for vi in range(num_vertices):
        next_index = (vi + 1)%num_vertices
        
        if ignore_vertex_index != -1:
            if vi == ignore_vertex_index:
                continue
            if next_index == ignore_vertex_index:
                continue
        
        this_coords = polygon[vi]
        next_coords = polygon[next_index]
        
        is_left_a = is_left(this_coords, next_coords, a)
        is_left_b = is_left(this_coords, next_coords, b)
        
        
        if is_left_a < 0 and is_left_b < 0:
            continue
        elif is_left_a > 0 and is_left_b > 0:
            continue
        
        if is_left_a == 0 or is_left_b == 0:
            return True
            
        alpha_x, alpha_y = (a - this_coords)
        beta_x, beta_y = (next_coords - this_coords)
        
        denominator = beta_x*nls_x + beta_y*nls_y
        if denominator == 0:
            return False
            
        t = (alpha_x*nls_x + alpha_y*nls_y)/denominator
        
        if 0 <= t < 1:
            return True
        else:
            continue
            
    return False
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)
def cross_product_2D(a, b):
    return (a[0]*b[1]) - (a[1]*b[0])

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def is_given_vector_between_others(x, alpha, beta):
    if cross_product_2D(alpha, x) > 0 and cross_product_2D(x, beta) > 0:
        return True
    else:
        return False
        
# -----------------------------------------------------------------
        
@nb.jit(nopython=True)
def check_if_line_segment_from_node_self_intersects(start_coord, end_coord, polygon_coords, start_coord_index):
    num_vertices = polygon_coords.shape[0]
    si_minus1 = (start_coord_index - 1)%num_vertices
    si_plus1 = (start_coord_index + 1)%num_vertices
    
    si_plus1_coord = polygon_coords[si_plus1]
    si_minus1_coord = polygon_coords[si_minus1]
    
    #edge_vector_to_plus = calculate_vector_from_p1_to_p2_given_vectors(start_coord, si_plus1_coord)
    #edge_vector_to_plus_normalized = normalize_2D_vector(edge_vector_to_plus)
    
    #edge_vector_from_minus = calculate_vector_from_p1_to_p2_given_vectors(si_minus1_coord, start_coord)
    #edge_vector_from_minus_normalized = normalize_2D_vector(edge_vector_from_minus)
    
    #tangent_vector = edge_vector_to_plus_normalized + edge_vector_from_minus_normalized
    rough_tangent_vector = si_plus1_coord - si_minus1_coord
    inside_pointing_vector = rotate_2D_vector_CCW(rough_tangent_vector)
    
    v = end_coord - start_coord
    
    if is_given_vector_between_others(v, inside_pointing_vector, si_minus1_coord - start_coord):
        return 1
    elif is_given_vector_between_others(v, si_plus1_coord - start_coord, inside_pointing_vector):
        return 1
    else:
        return 0
        
# -----------------------------------------------------------------
        
@nb.jit(nopython=True)
def check_if_line_segment_intersects_box(start, end, min_x, min_y, max_x, max_y):
    
    sx, sy = start
    ex, ey = end
    
    if sx < min_x and ex < min_x:
        return False
    if sx > max_x and ex > max_x:
        return False
    if sy < min_y and ey < min_y:
        return False
    if sy > max_y and ey > max_y:
        return False
        
    m = (sy - ey)/(sx - ex)
    b = sy - m*sx

    if min_y < m*min_x + b < max_y:
        return True
    elif min_y < m*max_x + b < max_y:
        return True
    elif min_x < (min_y - b)/m < max_x:
        return True
    elif min_x < (max_y - b)/m < max_x:
        return True
        
    return False
    
@nb.jit(nopython=True)      
def check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(pi_a, vi_a, pi_b, vi_b, all_polygon_coords, all_polygons_bounding_box_coords):
    coords_a = all_polygon_coords[pi_a, vi_a]
    coords_b = all_polygon_coords[pi_b, vi_b]
    
    normal_to_line_segment = rotate_2D_vector_CCW(coords_b - coords_a)
    
    if check_if_line_segment_from_node_self_intersects(coords_a, coords_b, all_polygon_coords[pi_a], vi_a):
        return 1
    if check_if_line_segment_from_node_self_intersects(coords_b, coords_a, all_polygon_coords[pi_b], vi_b):
        return 1
#    if check_if_line_segment_intersects_polygon(coords_a, coords_b, normal_to_line_segment, all_polygon_coords[pi_a], vi_a):
#        return 1
#    elif check_if_line_segment_intersects_polygon(coords_a, coords_b, normal_to_line_segment, all_polygon_coords[pi_b], vi_b):
#        return 1
    else:
        num_polygons = all_polygon_coords.shape[0]
        for pi in range(num_polygons):
            if pi == pi_a or pi == pi_b:
                continue
            else:
                min_x, max_x, min_y, max_y = all_polygons_bounding_box_coords[pi]
                if check_if_line_segment_intersects_box(coords_a, coords_b, min_x, max_x, min_y, max_y):
                    #check_if_line_segment_intersects_polygon(coords_a, coords_b, normal_to_line_segment, this_poly_bounding_box, -1):
                    if check_if_line_segment_intersects_polygon(coords_a, coords_b, normal_to_line_segment, all_polygon_coords[pi], -1):
                        return 1
                    else:
                        return 0
                else:
                    return 0
        
        return 0
    
    return 0

# -----------------------------------------------------------------
  
@nb.jit(nopython=True)      
def create_initial_line_segment_intersection_matrix(num_cells, num_nodes_per_cell, init_cells_bounding_box_array, init_all_cells_node_coords):
    line_segment_intersection_matrix = -1*np.ones((num_cells, num_nodes_per_cell, num_cells, num_nodes_per_cell), dtype=np.int64)
    info_update_tracker = np.zeros_like(line_segment_intersection_matrix, dtype=np.int64)
    
    for ci in range(num_cells):
        for ni in range(num_nodes_per_cell):
            for other_ci in range(num_cells):
                if ci != other_ci:
                    relevant_info_update_tracker_slice = info_update_tracker[ci][ni][other_ci]
                    for other_ni in range(num_nodes_per_cell):
                        if relevant_info_update_tracker_slice[other_ni] != 1:
                            info_update_tracker[ci][ni][other_ci][other_ni] = 1
                            info_update_tracker[other_ci][other_ni][ci][ni] = 1
                            
                            does_line_segment_between_nodes_intersect = check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(ci, ni, other_ci, other_ni, init_all_cells_node_coords, init_cells_bounding_box_array)
                            
                            line_segment_intersection_matrix[ci][ni][other_ci][other_ni] = does_line_segment_between_nodes_intersect
                            line_segment_intersection_matrix[other_ci][other_ni][ci][ni] = does_line_segment_between_nodes_intersect

                            
    return line_segment_intersection_matrix

# -----------------------------------------------------------------
 
@nb.jit(nopython=True)      
def update_line_segment_intersection_matrix(last_updated_cell_index, num_cells, num_nodes_per_cell, all_cells_node_coords, cells_bounding_box_array, line_segment_intersection_matrix):
    
    # 1 if info has been updated, 0 otherwise
    info_update_tracker = np.zeros_like(line_segment_intersection_matrix, dtype=np.int64)
    
    for ni in range(num_nodes_per_cell):
        for other_ci in range(num_cells):
            if other_ci != last_updated_cell_index:
                relevant_info_update_tracker_slice = info_update_tracker[last_updated_cell_index][ni][other_ci]
                for other_ni in range(num_nodes_per_cell):
                    if relevant_info_update_tracker_slice[other_ni] != 1:
                        info_update_tracker[last_updated_cell_index][ni][other_ci][other_ni] = 1
                        info_update_tracker[other_ci][other_ni][last_updated_cell_index][ni] = 1
                        
                        does_line_segment_between_nodes_intersect = check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(last_updated_cell_index, ni, other_ci, other_ni, all_cells_node_coords, cells_bounding_box_array)
                            
                        line_segment_intersection_matrix[last_updated_cell_index][ni][other_ci][other_ni] = does_line_segment_between_nodes_intersect
                        line_segment_intersection_matrix[other_ci][other_ni][last_updated_cell_index][ni] = does_line_segment_between_nodes_intersect
                            
    return line_segment_intersection_matrix
    
# -----------------------------------------------------------------
    
@nb.jit(nopython=True)      
def create_initial_line_segment_intersection_and_dist_squared_matrices(num_cells, num_nodes_per_cell, init_cells_bounding_box_array, init_all_cells_node_coords):
    distance_squared_matrix = -1*np.ones((num_cells, num_nodes_per_cell, num_cells, num_nodes_per_cell), dtype=np.float64)
    line_segment_intersection_matrix = -1*np.ones((num_cells, num_nodes_per_cell, num_cells, num_nodes_per_cell), dtype=np.int64)
    
    info_update_tracker = np.zeros_like(distance_squared_matrix, dtype=np.int64)
    
    for ci in range(num_cells):
        for ni in range(num_nodes_per_cell):
            for other_ci in range(num_cells):
                if ci != other_ci:
                    relevant_info_update_tracker_slice = info_update_tracker[ci][ni][other_ci]
                    for other_ni in range(num_nodes_per_cell):
                        if relevant_info_update_tracker_slice[other_ni] != 1:
                            info_update_tracker[ci][ni][other_ci][other_ni] = 1
                            info_update_tracker[other_ci][other_ni][ci][ni] = 1
                            
                            does_line_segment_between_nodes_intersect = check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(ci, ni, other_ci, other_ni, init_all_cells_node_coords, init_cells_bounding_box_array)
                            
                            line_segment_intersection_matrix[ci][ni][other_ci][other_ni] = does_line_segment_between_nodes_intersect
                            line_segment_intersection_matrix[other_ci][other_ni][ci][ni] = does_line_segment_between_nodes_intersect
                                
                            this_node = init_all_cells_node_coords[ci][ni]
                            other_node = init_all_cells_node_coords[other_ci][other_ni]
                            
                            squared_dist = calculate_squared_dist(this_node, other_node)
                            distance_squared_matrix[ci][ni][other_ci][other_ni] = squared_dist
                            distance_squared_matrix[other_ci][other_ni][ci][ni] = squared_dist
                            
    return distance_squared_matrix, line_segment_intersection_matrix

# -----------------------------------------------------------------
    
@nb.jit(nopython=True)      
def update_line_segment_intersection_and_dist_squared_matrices(last_updated_cell_index, num_cells, num_nodes_per_cell, all_cells_node_coords, cells_bounding_box_array, distance_squared_matrix, line_segment_intersection_matrix):
    
    # 1 if info has been updated, 0 otherwise
    info_update_tracker = np.zeros_like(line_segment_intersection_matrix, dtype=np.int64)
    
    for ni in range(num_nodes_per_cell):
        for other_ci in range(num_cells):
            if other_ci != last_updated_cell_index:
                relevant_info_update_tracker_slice = info_update_tracker[last_updated_cell_index][ni][other_ci]
                for other_ni in range(num_nodes_per_cell):
                    if relevant_info_update_tracker_slice[other_ni] != 1:
                        info_update_tracker[last_updated_cell_index][ni][other_ci][other_ni] = 1
                        info_update_tracker[other_ci][other_ni][last_updated_cell_index][ni] = 1
                        
                        does_line_segment_between_nodes_intersect = check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(last_updated_cell_index, ni, other_ci, other_ni, all_cells_node_coords, cells_bounding_box_array)
                            
                        line_segment_intersection_matrix[last_updated_cell_index][ni][other_ci][other_ni] = does_line_segment_between_nodes_intersect
                        line_segment_intersection_matrix[other_ci][other_ni][last_updated_cell_index][ni] = does_line_segment_between_nodes_intersect
                            
                        this_node = all_cells_node_coords[last_updated_cell_index][ni]
                        other_node = all_cells_node_coords[other_ci][other_ni]
                        
                        squared_dist = calculate_squared_dist(this_node, other_node)
                        distance_squared_matrix[last_updated_cell_index][ni][other_ci][other_ni] = squared_dist
                        distance_squared_matrix[other_ci][other_ni][last_updated_cell_index][ni] = squared_dist
                            
    return distance_squared_matrix, line_segment_intersection_matrix
    
    
    
if __name__ == '__main__':
    print "=============== geometry.py ===================="
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    unit_inside_pointing_vecs = calculate_unit_inside_pointing_vecs(4, polygon)
    print "================================================"