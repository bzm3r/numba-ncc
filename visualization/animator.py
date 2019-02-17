import cairo #pycairo package
import numpy as np
import numba as nb
import core.geometry as geometry
import os, shutil
import subprocess
import time
from . import colors
import core.hardio as hardio
import core.utilities as cu
import core.chemistry as chemistry
import threading
import general.utilities as gu
from pathos.multiprocessing import ProcessPool 

def fsquared(x):
    return x*x

def convert_numpy_array_to_pycairo_matrix(np_array):
    xx, xy = np_array[0]
    yx, yy = np_array[1]
    x0, y0 = np_array[2]
    
    return cairo.Matrix(xx=xx, xy=xy, yx=yx, yy=yy, x0=x0, y0=y0)

def convert_pycairo_matrix_numpy_array(cairo_matrix):
    xx, xy, yx, yy, x0, y0 = cairo_matrix.xx, cairo_matrix.xy, cairo_matrix.yx, cairo_matrix.yy, cairo_matrix.x0, cairo_matrix.y0
    
    return np.array([[xx, xy], [yx, yy], [x0, y0]])

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
            

def copy_worker(tasks):
    for task in tasks:
        shutil.copy(*task)

#@nb.jit(nopython=True)
def create_transformation_matrix_entries(scale_x, scale_y, rotation_theta, translation_x, translation_y, plate_width, plate_height):
    sin_theta = np.sin(rotation_theta)
    cos_theta = np.cos(rotation_theta)
    
    xx = scale_x*cos_theta
    xy = -1*scale_y*sin_theta
    yx = scale_x*sin_theta
    yy = scale_y*cos_theta
    x0 = scale_x*translation_x
    y0 = scale_y*translation_y #+ scale_y*plate_height
    
    return xx, xy, x0, yx, yy, y0
    
# -------------------------------------

#@nb.jit()
def draw_polygon_jit(polygon_coords, polygon_edge_and_vertex_color, polygon_fill_color, polygon_line_width, context):
    context.new_path()
    r, g, b = polygon_edge_and_vertex_color
    context.set_source_rgb(r, g, b)
    context.set_line_width(polygon_line_width)

    for n, coord in enumerate(polygon_coords):
        x, y = coord
        if n == 0:
            context.move_to(x, y)
        else:
            context.line_to(x, y)
            
    context.close_path()
    context.stroke_preserve()
    context.set_source_rgb(*polygon_fill_color)
    context.fill()
    
# -------------------------------------

@nb.jit()
def draw_dot_jit(centre_coords, polygon_edge_and_vertex_color, polygon_line_width, context):
    context.new_path()
    r, g, b = polygon_edge_and_vertex_color
    context.set_source_rgb(r, g, b)
    context.set_line_width(polygon_line_width)

    context.arc(centre_coords[0], centre_coords[1], polygon_line_width, 0.0, 2*np.pi)
    context.stroke_preserve()
    context.fill()
    
# -------------------------------------

@nb.jit()
def draw_circle_jit(centre_coords, circle_radius, polygon_edge_and_vertex_color, polygon_line_width, context):
    context.new_path()
    r, g, b = polygon_edge_and_vertex_color
    context.set_source_rgb(r, g, b)
    context.set_line_width(polygon_line_width)

    context.arc(centre_coords[0], centre_coords[1], circle_radius, 0.0, 2*np.pi)
    context.stroke()

# -------------------------------------

@nb.jit()
def draw_centroid_trail_jit(centroid_line_width, centroid_color, centroid_coords, context):
    context.new_path()
    r, g, b = centroid_color
    context.set_source_rgb(r, g, b)
    context.set_line_width(centroid_line_width)
    
    init_point_moved_to = False
    for centroid_coord in centroid_coords:
        x, y = centroid_coord
        if not init_point_moved_to:
            init_point_moved_to = True
            context.move_to(x, y)
        else:
            context.line_to(x, y)
            
    context.stroke()
            
# -------------------------------------
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
            gp_process_percentage = np.round(100*np.sum(gps_processing_status)/num_gps, decimals=6)
            print("gps processed: {}%".format(gp_process_percentage))
            coa_data = []
        
            gpi_indices = []
            relative_indices_per_gpi = [(i, np.array([0, 0]))]
            adjacent_to = [i]
            
            while len(adjacent_to) != 0:
                #print("current len of adjacent_to: {}".format(len(adjacent_to)))
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
    
#@nb.jit()    
def draw_coa_overlay(coa_overlay_color, resolution, coa_grid_points, coa_data_per_gridpoint, context):
    r, g, b = coa_overlay_color
    standard_tl = 0.01*resolution*np.array([-0.5, 0.5])
    
#    simplified_coa_polygons, simplified_coa_polygon_data = get_simplified_polygon_matrices_and_coa_data(coa_grid_points, coa_data_per_gridpoint, resolution, 0.05)
#    
#    print("COA overlay simplification benefit: {}".format(1.0 - np.round(len(simplified_coa_polygons)/coa_grid_points.shape[0], decimals=2)))
#    
#    alphas = np.array([(coa/(16*24.0)) for coa in simplified_coa_polygon_data])
#    alphas = 0.8*alphas/np.max(alphas)
#    
#    for polygon, a in zip(simplified_coa_polygons, alphas):
#        context.new_path()
#        for n, coord in enumerate(polygon):
#            x, y = coord
#            if n == 0:
#                context.move_to(x, y)
#            else:
#                context.line_to(x, y)
#            
#        context.close_path()
#        if a > 0.01:
#            context.set_source_rgba(r, g, b, a)
#            context.fill()          
    
    alphas = np.array([(coa/(16*24.0)) if coa > 0 else 0.0 for coa in coa_data_per_gridpoint])
    alphas = 0.8*alphas/np.max(alphas)
    for gp, a in zip(coa_grid_points, alphas):
        if a > 0.01:
            rect_tl = gp + standard_tl
            context.set_source_rgba(r, g, b, a)
            context.new_path()
            context.rectangle(rect_tl[0], rect_tl[1], resolution, resolution)
            context.fill()
    
    
# -------------------------------------
        
class AnimationCell():
    def __init__(self, polygon_edge_and_vertex_color, polygon_fill_color, rgtpase_colors, rgtpase_background_shine_color, velocity_colors, centroid_color, coa_color, chemoattractant_color, hidden, show_rgtpase, show_velocities, show_centroid_trail, show_coa, show_chemoattractant, polygon_line_width, rgtpase_line_width, velocity_line_width, centroid_line_width, coa_line_width, chemoattractant_line_width):
        self.hidden = hidden
        
        self.show_velocities = show_velocities
        self.show_rgtpase = show_rgtpase
        self.show_centroid_trail = show_centroid_trail
        self.show_coa = show_coa
        self.show_chemoattractant = show_chemoattractant
        
        self.polygon_edge_and_vertex_color = polygon_edge_and_vertex_color
        self.polygon_fill_color = polygon_fill_color
        self.polygon_line_width = polygon_line_width
   
        self.rgtpase_colors = rgtpase_colors
        self.rgtpase_background_shine_color = rgtpase_background_shine_color
        self.rgtpase_line_width = rgtpase_line_width

        self.velocity_colors = velocity_colors
        self.velocity_line_width = velocity_line_width
 
        self.centroid_color = centroid_color
        self.centroid_line_width = centroid_line_width
        
        self.coa_color = coa_color
        self.coa_line_width = coa_line_width

        self.chemoattractant_color = chemoattractant_color
        self.chemoattractant_line_width = chemoattractant_line_width

    # -------------------------------------
        
    def hide(self):
        self.hidden = True
    
    # -------------------------------------
    
    def unhide(self):
        self.hidden = False
        
    # -------------------------------------
    
    def draw_cell_polygon(self, context, polygon_coords):
        draw_polygon_jit(polygon_coords, self.polygon_edge_and_vertex_color, self.polygon_fill_color, self.polygon_line_width, context)
        
    # -------------------------------------
        
    def draw_rgtpase(self, context, polygon_coords, rgtpase_line_coords_per_gtpase):
        if type(self.rgtpase_background_shine_color) != type(None):
            context.set_line_width(self.rgtpase_line_width*1.1)
            offset_coords = rgtpase_line_coords_per_gtpase[-1]
            offset_directions = [0, -1, 0, -1]
            r, g, b = self.rgtpase_background_shine_color
            
            for i, rgtpase_line_coords in enumerate(rgtpase_line_coords_per_gtpase[:-1]):
                offset_direction = offset_directions[i]
                
                #if offset_direction != - 1:
                context.set_source_rgb(r, g, b)
                for polygon_coord, rgtpase_line_coord, offset_coord in zip(polygon_coords, rgtpase_line_coords, offset_coords):
                    x0, y0 = polygon_coord + offset_direction*offset_coord
                    x1, y1 = rgtpase_line_coord
                    
                    context.new_path()
                    context.move_to(x0, y0)
                    context.line_to(x1, y1)
                    context.stroke()
                        
        context.set_line_width(self.rgtpase_line_width)
        offset_coords = rgtpase_line_coords_per_gtpase[-1]
        offset_directions = [0, -1, 0, -1]
        
        for i, rgtpase_line_coords in enumerate(rgtpase_line_coords_per_gtpase[:-1]):
            offset_direction = offset_directions[i]
            
            #if offset_direction != - 1:
            r, g, b = self.rgtpase_colors[i]
            context.set_source_rgb(r, g, b)
            for polygon_coord, rgtpase_line_coord, offset_coord in zip(polygon_coords, rgtpase_line_coords, offset_coords):
                x0, y0 = polygon_coord + offset_direction*offset_coord
                x1, y1 = rgtpase_line_coord
                
                context.new_path()
                context.move_to(x0, y0)
                context.line_to(x1, y1)
                context.stroke()
            
    # -------------------------------------
        
    def draw_rgtpase_showing_rac_random_spikes(self, context, polygon_coords, rgtpase_line_coords_per_gtpase, rac_random_spikes_info, rac_random_spike_color=(0, 153, 0)):        
        context.set_line_width(self.rgtpase_line_width)
        offset_coords = rgtpase_line_coords_per_gtpase[-1]
        offset_directions = [0, -1, 0, -1]
        
        for i, rgtpase_line_coords in enumerate(rgtpase_line_coords_per_gtpase[:-1]):
            offset_direction = offset_directions[i]
            #if offset_direction != -1:
            default_rgb = self.rgtpase_colors[i]
            context.set_source_rgb(*default_rgb)
            
            
            for n, drawing_data in enumerate(zip(polygon_coords, rgtpase_line_coords, offset_coords)):
                polygon_coord, rgtpase_line_coord, offset_coord = drawing_data
                if i == 0:
                    if rac_random_spikes_info[n] > 1:
                        context.set_source_rgb(*rac_random_spike_color)
                    else:
                        context.set_source_rgb(*default_rgb)
                    
                x0, y0 = polygon_coord + offset_direction*offset_coord
                x1, y1 = rgtpase_line_coord
                
                context.new_path()
                context.move_to(x0, y0)
                context.line_to(x1, y1)
                context.stroke()
                
    # -------------------------------------
        
    def draw_coa(self, context, polygon_coords, coa_line_coords):        
        context.set_line_width(self.coa_line_width)
        
        r, g, b = self.coa_color
        context.set_source_rgb(r, g, b)
        
        for polygon_coord, coa_line_coord in zip(polygon_coords, coa_line_coords):
            x0, y0 = polygon_coord
            x1, y1 = coa_line_coord
            
            context.new_path()
            context.move_to(x0, y0)
            context.line_to(x1, y1)
            context.stroke()

    # -------------------------------------

    def draw_chemoattractant(self, context, polygon_coords, chemoattractant_line_coords):
        context.set_line_width(self.coa_line_width)

        r, g, b = self.chemoattractant_color
        context.set_source_rgb(r, g, b)

        for polygon_coord, chemoattractant_line_coord in zip(polygon_coords, chemoattractant_line_coords):
            x0, y0 = polygon_coord
            x1, y1 = polygon_coord + 0.1*chemoattractant_line_coord

            context.new_path()
            context.move_to(x0, y0)
            context.line_to(x1, y1)
            context.stroke()
                
    # -------------------------------------
    
    def draw_velocities(self, context, polygon_coords, velocity_line_coords_per_forcetype):
        context.set_line_width(self.velocity_line_width)
        for i, velocity_line_coords, polygon_coords, in enumerate(velocity_line_coords_per_forcetype):
            r, g, b = self.velocity_colors[i]
            context.set_source_rgb(r, g, b)
            
            for polygon_coord, velocity_line_coord in zip(polygon_coords, velocity_line_coords):
                x0, y0 = polygon_coord
                x1, y1 = velocity_line_coord
                
                context.new_path()
                context.move_to(x0, y0)
                context.line_to(x1, y1)
                context.stroke()
                
    # -------------------------------------
                
    def draw_centroid_trail(self, context, centroid_coords_per_frame):
        draw_centroid_trail_jit(self.centroid_line_width, self.centroid_color, centroid_coords_per_frame, context)
            
    # -------------------------------------
    
    def draw(self, context, polygon_coords, rgtpase_line_coords_per_label, rac_random_spikes_info, velocity_line_coords_per_label, centroid_coords_per_frame, coa_line_coords, chemoattractant_line_coords):
        if self.hidden == False:
            
            self.draw_cell_polygon(context, polygon_coords)
            
            if (self.show_velocities == True):
                self.draw_velocities(context, polygon_coords, velocity_line_coords_per_label)
                
            if (self.show_rgtpase == True):
                if type(rac_random_spikes_info) != type(None):
                    self.draw_rgtpase_showing_rac_random_spikes(context, polygon_coords, rgtpase_line_coords_per_label, rac_random_spikes_info)
                else:
                    self.draw_rgtpase(context, polygon_coords, rgtpase_line_coords_per_label)
            
            if (self.show_centroid_trail == True):
                self.draw_centroid_trail(context, centroid_coords_per_frame)
                
            if (self.show_coa == True):
                self.draw_coa(context, polygon_coords, coa_line_coords)

            if (self.show_chemoattractant == True):
                self.draw_chemoattractant(context, polygon_coords, chemoattractant_line_coords)
                
            return True
        else:
            return False
            
    # -------------------------------------

# -------------------------------------

def prepare_velocity_data(num_nodes, eta, velocity_scale, cell_index, timesteps, polygon_coords_per_timestep, storefile_path):
    scale = (velocity_scale/eta)
    
    num_timesteps = timesteps.shape[0]
    
    VF, VEFplus, VEFminus, VF_rgtpase, VF_cytoplasmic = np.empty((num_timesteps, num_nodes, 2), dtype=np.float64), np.empty((num_timesteps, num_nodes, 2), dtype=np.float64), np.empty((num_timesteps, num_nodes, 2), dtype=np.float64), np.empty((num_timesteps, num_nodes, 2), dtype=np.float64), np.empty((num_timesteps, num_nodes, 2), dtype=np.float64)
    
    polygon_x = polygon_coords_per_timestep[:,:,0]
    polygon_y = polygon_coords_per_timestep[:,:,1]
    
    VF[:,:,0] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "F_x", storefile_path) + polygon_x
    VF[:,:,1] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "F_y", storefile_path) + polygon_y
    
    VEFplus[:,:,0] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "EFplus_x", storefile_path) + polygon_x
    VEFplus[:,:1] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "EFplus_y", storefile_path) + polygon_y
    
    VEFminus[:,:,0] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "EFminus_x", storefile_path) + polygon_x
    VEFminus[:,:,1] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "EFminus_y", storefile_path) + polygon_y
    
    VF_rgtpase[:,:,0] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "F_rgtpase_x", storefile_path) + polygon_x
    VF_rgtpase[:,:,1] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "F_rgtpase_y", storefile_path) + polygon_y
    
    VF_cytoplasmic[:,:,0] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "F_cytoplasmic_x", storefile_path) + polygon_x
    VF_cytoplasmic[:,:,1] = scale*hardio.get_data_for_tsteps(cell_index, timesteps, "F_cytoplasmic_y", storefile_path) + polygon_y

    return VF, VEFplus, VEFminus, VF_rgtpase, VF_cytoplasmic
    
# -------------------------------------

def tile_scalar_array_for_multiplication(given_array):
    given_array = given_array[:,:,np.newaxis]
    given_array = np.tile(given_array, (1, 1, 2))
    
    return given_array
    
def prepare_rgtpase_data(rgtpase_scale, cell_index, unique_undrawn_timesteps, polygon_coords_per_timestep, offset_magnitude, storefile_path):    
    rac_membrane_active_mag = tile_scalar_array_for_multiplication(rgtpase_scale*hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "rac_membrane_active", storefile_path))
    rac_membrane_inactive_mag = tile_scalar_array_for_multiplication(rgtpase_scale*hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "rac_membrane_inactive", storefile_path))
    rho_membrane_active_mag = tile_scalar_array_for_multiplication(rgtpase_scale*hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "rho_membrane_active", storefile_path))
    rho_membrane_inactive_mag = tile_scalar_array_for_multiplication(rgtpase_scale*hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "rho_membrane_inactive", storefile_path))

    unit_inside_pointing_vecs = geometry.calculate_unit_inside_pointing_vecs_per_timestep(polygon_coords_per_timestep)
    
    num_timesteps = unit_inside_pointing_vecs.shape[0]
    num_nodes = unit_inside_pointing_vecs.shape[1]
    
    unit_inside_pointing_vecs = unit_inside_pointing_vecs.reshape((num_timesteps*num_nodes, 2))
    
    normal_to_uivs = geometry.rotate_2D_vectors_CCW(unit_inside_pointing_vecs)
    normal_to_uivs = normal_to_uivs.reshape((num_timesteps, num_nodes, 2))
    unit_inside_pointing_vecs = unit_inside_pointing_vecs.reshape((num_timesteps, num_nodes, 2))
        
    offset_vecs = offset_magnitude*normal_to_uivs #0*normal_to_uivs
    
    positive_offset = offset_vecs + polygon_coords_per_timestep
    negative_offset = -1*offset_vecs + polygon_coords_per_timestep
    
    rac_membrane_active = -1*rac_membrane_active_mag*unit_inside_pointing_vecs + positive_offset
    rac_membrane_inactive = -1*rac_membrane_inactive_mag*unit_inside_pointing_vecs + negative_offset
    rho_membrane_active = positive_offset +  rho_membrane_active_mag*unit_inside_pointing_vecs
    rho_membrane_inactive = negative_offset +  rho_membrane_inactive_mag*unit_inside_pointing_vecs
    
    return rac_membrane_active, rac_membrane_inactive, rho_membrane_active, rho_membrane_inactive, offset_vecs

# -------------------------------------
    
def prepare_coa_data(coa_scale, cell_index, unique_undrawn_timesteps, polygon_coords_per_timestep, storefile_path):
    coa_mag = coa_scale*hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "coa_signal", storefile_path)

    unit_inside_pointing_vecs = geometry.calculate_unit_inside_pointing_vecs_per_timestep(polygon_coords_per_timestep)
    
    coa_signal = geometry.multiply_vectors_by_scalars(unit_inside_pointing_vecs, coa_mag) + polygon_coords_per_timestep
    
    return coa_signal


def prepare_chemoattractant_data(chemoattractant_scale, cell_index, unique_undrawn_timesteps, polygon_coords_per_timestep, offset_magnitude, storefile_path):
    chemoattractant_mag = chemoattractant_scale * hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "chemoattractant_signal_on_nodes", storefile_path)

    unit_inside_pointing_vecs = geometry.calculate_unit_inside_pointing_vecs_per_timestep(polygon_coords_per_timestep)

    num_timesteps = unit_inside_pointing_vecs.shape[0]
    num_nodes = unit_inside_pointing_vecs.shape[1]

    unit_inside_pointing_vecs = unit_inside_pointing_vecs.reshape((num_timesteps * num_nodes, 2))

    normal_to_uivs = geometry.rotate_2D_vectors_CCW(unit_inside_pointing_vecs)
    normal_to_uivs = normal_to_uivs.reshape((num_timesteps, num_nodes, 2))

    chemoattractant_mag = chemoattractant_mag.reshape((num_timesteps*num_nodes,))
    chemoattractant_vecs = geometry.multiply_vectors_by_scalars(-1*unit_inside_pointing_vecs, chemoattractant_mag)
    chemoattractant_vecs = chemoattractant_vecs.reshape((num_timesteps, num_nodes, 2))
    unit_inside_pointing_vecs = unit_inside_pointing_vecs.reshape((num_timesteps, num_nodes, 2))

    offset_vecs = offset_magnitude * normal_to_uivs  # 0*normal_to_uivs

    positive_offset = offset_vecs + polygon_coords_per_timestep

    return -1*chemoattractant_vecs
    
def prepare_rac_random_spike_data(cell_index, unique_undrawn_timesteps, storefile_path):
    rac_random_spike_info = hardio.get_data_for_tsteps(cell_index, unique_undrawn_timesteps, "randomization_rac_kgtp_multipliers", storefile_path)
    
    return rac_random_spike_info

# -----------------------------------------------------------------
    
#@nb.jit(nopython=True)
def grid_point_validation_worker(space_migratory_bdry_polygon, space_physical_bdry_polygon, grid_point_chunk, grid_point_indices_chunk, is_grid_point_valid_array):
    #print("gpis: {}".format(grid_point_indices_chunk))
    init_gpi = grid_point_indices_chunk[0]
    
    for gpi in grid_point_indices_chunk:
        gpindex = gpi - init_gpi
        gp = grid_point_chunk[gpindex]
        grid_point_valid = 1
        
#        if space_migratory_bdry_polygon.shape[0] > 0:
#            if geometry.is_point_in_polygon(gp, space_migratory_bdry_polygon) == 0:
#                grid_point_valid = 0
        
        if space_physical_bdry_polygon.shape[0] > 0:
            if geometry.is_point_in_polygon(gp, space_physical_bdry_polygon) == 1:
                grid_point_valid = 0
        
        if grid_point_valid == 1:
            is_grid_point_valid_array[gpi] = 1
            
    return 

# ---------------------------------------------------------------      
def rearrange_data_per_x_per_y_into_data_per_y_per_x(nparray):
    shape = nparray.shape

    if len(shape) >= 2:
        index_transposition = [1, 0] + [x for x in range(2, len(shape))]
        return np.transpose(nparray, axes=index_transposition)
    else:
        return nparray
    
# ------------------------------------------------------------------

@nb.jit(nopython=True)
def calculate_polygon_bbs(polygon_coords):
    bbs = np.zeros((polygon_coords.shape[0], 4), dtype=np.float64)
    for pi in range(polygon_coords.shape[0]):
        bbs[pi] = geometry.calculate_polygon_bounding_box(polygon_coords[pi])
        
    return bbs

# ------------------------------------------------------------------
@nb.jit(nopython=True)
def coa_data_calculation_worker(timepoint_indices_chunk, polygon_coords_per_cell_chunk, valid_grid_points, space_migratory_bdry_polygon, space_physical_bdry_polygon, max_coa_signal, too_close_dist_squared, coa_distribution_exponent, coa_intersection_exponent, cell_dependent_coa_signal_strengths, pw, ph):
    num_timepoints = timepoint_indices_chunk.shape[0]
    num_cells = polygon_coords_per_cell_chunk.shape[1]
    num_vertices = polygon_coords_per_cell_chunk.shape[2]
    num_valid_grid_points = valid_grid_points.shape[0]
    
    max_ds = (np.log(0.01)/coa_distribution_exponent)**2
    
    relevant_grid_point_coa_data = np.zeros((num_timepoints, num_valid_grid_points), dtype=np.float64)
    
    for ti in range(num_timepoints):
        relevant_cell_polygons = polygon_coords_per_cell_chunk[ti]
        relevant_cell_polygon_bbs = calculate_polygon_bbs(relevant_cell_polygons)
        
        for gi in range(num_valid_grid_points):
            gp = valid_grid_points[gi]
            
            coa_data_at_gp = 0.0
            
            for ci in range(num_cells):
                break_loop = False
                signal_strength = cell_dependent_coa_signal_strengths[ci]
                relevant_cell_polygon = relevant_cell_polygons[ci]
                
                for vi in range(num_vertices):                    
                    vp = relevant_cell_polygon[vi]
                    
                    ds = geometry.calculate_squared_dist(gp, vp)
                    
                    if not (too_close_dist_squared < ds < max_ds):
                        continue
                    
                    num_intersections = geometry.check_if_line_segment_going_from_vertex_of_one_polygon_to_point_passes_through_any_polygon(ci, vi, gp, relevant_cell_polygons, relevant_cell_polygon_bbs, space_migratory_bdry_polygon, space_physical_bdry_polygon)
                    intersection_factor = (1./(num_intersections + 1.)**coa_intersection_exponent)
                    
                    if intersection_factor < 0.01:
                        continue
                    
                    break_loop, coa_data_at_gp = chemistry.calculate_coa_signal(max_coa_signal, 0, coa_distribution_exponent, coa_data_at_gp, signal_strength, ds, intersection_factor)
                        
                    if break_loop:
                        break
                    
                if break_loop:
                    break
            
            if coa_data_at_gp > 1.0:
                relevant_grid_point_coa_data[ti][gi] = coa_data_at_gp
    
    return relevant_grid_point_coa_data
                                
# ------------------------------------------------------------------

def generate_coa_data_depending_on_grid_point_relevancy(unique_undrawn_timesteps, all_grid_points, space_migratory_bdry_polygon, space_physical_bdry_polygon, polygon_coords_per_timepoint_per_cell, max_coa_signal, too_close_dist_squared, coa_distribution_exponent, coa_intersection_exponent, cell_dependent_coa_signal_strengths, storefile_path, num_processes=4, sequential=False, pw=0, ph=0):
    is_grid_point_valid_array = np.zeros(all_grid_points.shape[0], dtype=np.bool)
    num_all_grid_points = all_grid_points.shape[0]
    
    chunksize = num_processes*15
    
    grid_point_indices = np.arange(num_all_grid_points)
    chunked_grid_point_indices = gu.chunkify(chunksize, grid_point_indices)
    chunked_grid_points = gu.chunkify(chunksize, all_grid_points)
    
    # Create argument tuples for each input chunk
    grid_point_validation_tasks = [(space_migratory_bdry_polygon, space_physical_bdry_polygon, grid_point_chunk, grid_point_indices_chunk, is_grid_point_valid_array) for grid_point_chunk, grid_point_indices_chunk in zip(chunked_grid_points, chunked_grid_point_indices)]

    # Spawn one thread per chunk
    if not sequential:
        threads = [threading.Thread(target=grid_point_validation_worker, args=c) for c in grid_point_validation_tasks]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for task in grid_point_validation_tasks:
            grid_point_validation_worker(*task)
    
    del grid_point_validation_tasks
    
    
    valid_grid_points = np.array([gp for i, gp in enumerate(all_grid_points) if is_grid_point_valid_array[i] == 1])
    #num_valid_grid_points = valid_grid_points.shape[0]
    del is_grid_point_valid_array
    del all_grid_points
    
    success, grid_point_coa_data = hardio.get_coa_overlay_data(storefile_path, valid_grid_points)
    
    if not success:
        print("Could not find stored coa overlay data.")
        polygon_coords_per_cell_per_timepoint = rearrange_data_per_x_per_y_into_data_per_y_per_x(polygon_coords_per_timepoint_per_cell)
        num_cells = polygon_coords_per_cell_per_timepoint.shape[1]
        num_cell_poly_vertices = polygon_coords_per_timepoint_per_cell.shape[2]
        num_timepoints = unique_undrawn_timesteps.shape[0]
        
        timepoint_indices = np.arange(num_timepoints)

        chunked_timepoint_indices = gu.chunkify(chunksize, timepoint_indices)
        chunked_polygon_coords_per_cell = gu.chunkify(chunksize, polygon_coords_per_cell_per_timepoint)
        
        per_node_cell_info = []
        for ci in range(num_cells):
            per_node_cell_info += [ci]*num_cell_poly_vertices
        per_node_cell_info = np.array(per_node_cell_info, dtype=np.int64)
        
        coa_data_calculation_worker_tasks = [(timepoint_indices_chunk, polygon_coords_per_cell_chunk, valid_grid_points, space_migratory_bdry_polygon, space_physical_bdry_polygon, max_coa_signal, too_close_dist_squared, coa_distribution_exponent, coa_intersection_exponent, cell_dependent_coa_signal_strengths, pw, ph) for polygon_coords_per_cell_chunk, timepoint_indices_chunk in zip(chunked_polygon_coords_per_cell, chunked_timepoint_indices) if timepoint_indices_chunk.shape[0] > 0]
        
        if not sequential and num_timepoints > 10:
            print("Executing in parallel.")
            st = time.time()
            
            pool = ProcessPool(nodes=4)
            results = pool.map(coa_data_calculation_worker, coa_data_calculation_worker_tasks)          
            
            et = time.time()
            print("Time taken: {}s.".format(np.round(et - st, decimals=1)))
        else:
            st = time.time()
            print("Executing in sequence.")
            results = []
            for task in coa_data_calculation_worker_tasks:
                results.append(coa_data_calculation_worker(*task))
                
            et = time.time()
            print("Time taken: {}s.".format(np.round(et - st, decimals=1)))
    else:
        print("Found stored coa overlay data successfully.")
    
    grid_point_coa_data = np.concatenate(results)
    return valid_grid_points, grid_point_coa_data


            

# ------------------------------------------------------------------
         
def generate_coa_overlay_data(unique_undrawn_timesteps, plate_width, plate_height, max_coa_signal, too_close_dist_squared, coa_distribution_exponent, coa_intersection_exponent, cell_dependent_coa_signal_strengths, polygon_coords_per_timepoint_per_cell, space_migratory_bdry_polygon, space_physical_bdry_polygon, storefile_path, resolution=1):
    print("Dividing space into grid points...")
    num_x_points = int(plate_width/resolution) + 1
    num_y_points = int(plate_height/resolution) + 1
    x_point_bdries = resolution*np.arange(num_x_points)
    y_point_bdries = resolution*np.arange(num_y_points)
    
    x_points = np.array([(x1 + x2)/2. for x1, x2 in zip(x_point_bdries[1:], x_point_bdries[:-1])], dtype=np.float64)
    y_points = np.array([(y1 + y2)/2. for y1, y2 in zip(y_point_bdries[1:], y_point_bdries[:-1])], dtype=np.float64)
    
    all_grid_points = float_cartesian_product(x_points, y_points)
    
    relevant_grid_points, relevant_grid_point_coa_data = generate_coa_data_depending_on_grid_point_relevancy(unique_undrawn_timesteps, all_grid_points, space_migratory_bdry_polygon, space_physical_bdry_polygon, polygon_coords_per_timepoint_per_cell, max_coa_signal, too_close_dist_squared, coa_distribution_exponent, coa_intersection_exponent, cell_dependent_coa_signal_strengths, storefile_path, pw=plate_width, ph=plate_height)

    return relevant_grid_points, relevant_grid_point_coa_data
    
# -------------------------------------

def draw_timestamp(timestep, timestep_length, text_color, font_size, global_scale, img_width, img_height, context):
    text_r, text_g, text_b = text_color
    context.set_source_rgb(text_r, text_g, text_b)
    context.select_font_face("Consolas", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(font_size*global_scale)

    timestamp_string = "t = {} min".format(int(np.round(timestep*timestep_length/60.0)))
    #timestamp_string = "NT = {} ".format(np.round(timestep))
    text_x_bearing, text_y_bearing, text_width, text_height = context.text_extents(timestamp_string)[:4]
    context.move_to((img_width - 1.2*text_width), (img_height - 1.2*text_height))
    context.show_text(timestamp_string)
    
    return

# -------------------------------------

def draw_animation_frame(task):
    
    timestep_index, timestep, timestep_length, font_color, font_size, global_scale, plate_width, plate_height, image_height_in_pixels, image_width_in_pixels, transform_matrix, animation_cells, polygon_coords_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, rac_random_spikes_info_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell, space_physical_bdry_polygon, space_migratory_bdry_polygon, chemoattractant_source_location, chemotaxis_target_radius, chemoattractant_line_coords_per_timepoint_per_cell, coa_overlay_color, coa_overlay_resolution, coa_grid_points, coa_data_per_gridpoint, background_color, migratory_bdry_color, physical_bdry_color, chemoattractant_dot_color, unique_timesteps, global_image_dir, global_image_name_format_str, image_format = task
    
    if image_format == ".png":
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, image_width_in_pixels, image_height_in_pixels)
    elif image_format == ".svg":
        surface = cairo.SVGSurface(os.path.join(global_image_dir, global_image_name_format_str.format(timestep)), image_width_in_pixels, image_height_in_pixels)
            
    context = cairo.Context(surface)
        
    context.set_source_rgb(*background_color)
    context.paint()

    # timestep, timestep_length, text_color, font_size, global_scale, img_width, img_height, context
    draw_timestamp(timestep, timestep_length, font_color, font_size, global_scale, image_width_in_pixels, image_height_in_pixels, context)
    
    context.transform(convert_numpy_array_to_pycairo_matrix(transform_matrix))
        
    if space_physical_bdry_polygon.shape[0] != 0:
        context.new_path()
        draw_polygon_jit(space_physical_bdry_polygon, physical_bdry_color, background_color, 2, context)
        
    if space_migratory_bdry_polygon.shape[0] != 0:
        context.new_path()
        #polygon_coords, polygon_edge_and_vertex_color, polygon_fill_color, polygon_line_width, context
        draw_polygon_jit(space_migratory_bdry_polygon, migratory_bdry_color, background_color, 2, context)
        
    if type(coa_data_per_gridpoint) == np.ndarray:
        if coa_grid_points.shape[0] > 0:
            draw_coa_overlay(coa_overlay_color, coa_overlay_resolution, coa_grid_points, coa_data_per_gridpoint, context)
    
    for cell_index, anicell in enumerate(animation_cells):
        
        if type(rgtpase_line_coords_per_label_per_timepoint_per_cell) != np.ndarray:
            rgtpase_data = None
        else:
            rgtpase_data = rgtpase_line_coords_per_label_per_timepoint_per_cell[cell_index][timestep_index]
            
        if type(rac_random_spikes_info_per_timepoint_per_cell) != np.ndarray:
            rac_random_spikes_info = None
        else:
            rac_random_spikes_info = rac_random_spikes_info_per_timepoint_per_cell[cell_index][timestep_index]
            
        if type(velocity_line_coords_per_label_per_timepoint_per_cell) != np.ndarray:
            velocity_data = None
        else:
            velocity_data = velocity_line_coords_per_label_per_timepoint_per_cell[cell_index][timestep_index]
        
        if type(centroid_coords_per_timepoint_per_cell) != np.ndarray:
            centroid_data = None
        else:
            centroid_data = centroid_coords_per_timepoint_per_cell[cell_index][unique_timesteps[:timestep+1]]
        
        if type(coa_line_coords_per_timepoint_per_cell) != np.ndarray:
            coa_data = None
        else:
            coa_data = coa_line_coords_per_timepoint_per_cell[cell_index][timestep_index]

        if type(chemoattractant_line_coords_per_timepoint_per_cell) != np.ndarray:
            chemoattractant_data = None
        else:
            chemoattractant_data = chemoattractant_line_coords_per_timepoint_per_cell[cell_index][timestep_index]
        
        anicell.draw(context, polygon_coords_per_timepoint_per_cell[cell_index][timestep_index], rgtpase_data, rac_random_spikes_info, velocity_data, centroid_data, coa_data, chemoattractant_data)
    
    if len(chemoattractant_source_location) != 0:
        context.new_path()
        draw_dot_jit(chemoattractant_source_location, chemoattractant_dot_color, 2, context)
        
        if chemotaxis_target_radius > 0.0:
            draw_circle_jit(chemoattractant_source_location, chemotaxis_target_radius, chemoattractant_dot_color, 2, context)
        
    if image_format == ".svg":
        #context.show_page()
        surface.finish()
    else:
        image_fp = os.path.join(global_image_dir, global_image_name_format_str.format(timestep))
        surface.write_to_png(image_fp)
            
def draw_animation_frame_for_given_timesteps(tasks):
    
    for task in tasks:
        draw_animation_frame(task)

# --------------------------------------------------------------------------  
  
def make_progress_str(progress, len_progress_bar=20, progress_char="-"):
    num_progress_chars = int(progress*len_progress_bar)
    return "|" + progress_char*num_progress_chars + " "*(len_progress_bar - num_progress_chars) + "|"
    

# --------------------------------------------------------------------------     
    
class EnvironmentAnimation():
    def __init__(self, general_animation_save_folder_path, environment_name, num_cells, num_nodes, max_num_timepoints, cell_group_indices, cell_Ls, cell_etas, cell_skip_dynamics, env_storefile_path, global_scale=1, plate_height_in_micrometers=400, plate_width_in_micrometers=600, rotation_theta=0.0, translation_x=10, translation_y=10, velocity_scale=1, rgtpase_scale=1, coa_scale=1, chemoattractant_scale=1, show_velocities=False, show_rgtpase=False, show_inactive_rgtpase=False, show_centroid_trail=False, show_coa=False, show_chemoattractant=False, color_each_group_differently=False, show_rac_random_spikes=False, only_show_cells=[], background_color=colors.RGB_WHITE, chemoattractant_dot_color=colors.RGB_LIGHT_GREEN, migratory_bdry_color=colors.RGB_BRIGHT_RED, physical_bdry_color=colors.RGB_BLACK, default_cell_polygon_fill_color=colors.RGB_WHITE, cell_polygon_edge_and_vertex_colors=[], default_cell_polygon_edge_and_vertex_color=colors.RGB_BLACK, rgtpase_colors=[colors.RGB_BRIGHT_BLUE, colors.RGB_LIGHT_BLUE, colors.RGB_BRIGHT_RED, colors.RGB_LIGHT_RED], rgtpase_background_shine_color=None, velocity_colors=[colors.RGB_ORANGE, colors.RGB_LIGHT_GREEN, colors.RGB_LIGHT_GREEN, colors.RGB_CYAN, colors.RGB_MAGENTA], coa_color=colors.RGB_DARK_GREEN, font_size=16, font_color=colors.RGB_BLACK, offset_scale=0.2, polygon_line_width=1, rgtpase_line_width=1, velocity_line_width=1, coa_line_width=1, chemoattractant_line_width=1, show_physical_bdry_polygon=False, space_physical_bdry_polygon=np.array([]), space_migratory_bdry_polygon=np.array([]), chemoattractant_source_location=np.array([]), centroid_colors_per_cell=[], centroid_line_width=1, show_coa_overlay=True, max_coa_signal=-1.0, coa_too_close_dist_squared=1e-12, coa_distribution_exponent=0.0, coa_intersection_exponent=0.0, coa_overlay_color=colors.RGB_LIGHT_GREEN, coa_overlay_resolution=1, cell_dependent_coa_signal_strengths=[], short_video_length_definition=2000.0, short_video_duration=5.0, timestep_length=None, fps=30, origin_offset_in_pixels=np.zeros(2), string_together_pictures_into_animation=True, allowed_drift_before_geometry_recalc=-1.0, specific_timesteps_to_draw_as_svg=[], chemotaxis_target_radius=-1):
        self.global_scale = global_scale
        self.rotation_theta = rotation_theta
        self.translation_x = translation_x
        self.translation_y = translation_y
        
        self.plate_height_in_micrometers = plate_height_in_micrometers
        self.plate_width_in_micrometers = plate_width_in_micrometers
        
        self.string_together_into_animation = string_together_pictures_into_animation
        self.animation_name = environment_name + '_animation.mp4'
        self.short_video_length_definition = short_video_length_definition
        self.short_video_duration = short_video_duration
        self.fps = 30
        self.origin_offset_in_pixels = origin_offset_in_pixels
        
        self.image_height_in_pixels = np.int(np.round(plate_height_in_micrometers*global_scale, decimals=0))
        if self.image_height_in_pixels%2 == 1:
            self.image_height_in_pixels += 1
        self.image_width_in_pixels = np.int(np.round(plate_width_in_micrometers*global_scale, decimals=0))
        if self.image_width_in_pixels%2 == 1:
            self.image_width_in_pixels += 1
        
        self.transform_matrix = convert_pycairo_matrix_numpy_array(self.calculate_transform_matrix())
        
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.max_num_timepoints = max_num_timepoints
        
        self.show_velocities = show_velocities
        self.show_rgtpase = show_rgtpase
        self.show_inactive_rgtpase = show_inactive_rgtpase
        self.show_centroid_trail = show_centroid_trail
        self.show_coa = show_coa
        self.show_chemoattractant = show_chemoattractant
        
        self.velocity_scale = velocity_scale
        self.rgtpase_scale = rgtpase_scale
        self.coa_scale = coa_scale
        self.chemoattractant_scale = chemoattractant_scale
        
        self.only_show_cells = only_show_cells
        self.background_color = background_color
        self.chemoattractant_dot_color = chemoattractant_dot_color
        self.chemotaxis_target_radius = chemotaxis_target_radius
        self.default_cell_polygon_edge_and_vertex_color = default_cell_polygon_edge_and_vertex_color
        self.default_cell_polygon_fill_color = default_cell_polygon_fill_color
        self.rgtpase_colors = rgtpase_colors
        self.velocity_colors = velocity_colors
        self.coa_color = coa_color
        self.font_color = font_color
        self.rgtpase_background_shine_color = rgtpase_background_shine_color
        self.migratory_bdry_color = migratory_bdry_color
        self.physical_bdry_color = physical_bdry_color
        
        self.show_rac_random_spikes = show_rac_random_spikes
        self.show_coa_overlay = show_coa_overlay
        
        self.cell_polygon_edge_and_vertex_colors = []
        if color_each_group_differently == True and len(cell_polygon_edge_and_vertex_colors) == 0:
            for ci in range(num_cells):
                self.cell_polygon_edge_and_vertex_colors.append((ci, colors.color_list_cell_groups10[cell_group_indices[ci]%10]))
        else:
            self.cell_polygon_edge_and_vertex_colors = cell_polygon_edge_and_vertex_colors
        
        self.velocity_labels = ["F", "EFplus", "EFminus", "F_rgtpase", "F_cytoplasmic"]
        self.num_velocity_labels = len(self.velocity_labels)
        
        self.rgtpase_labels = ["rac_membrane_active", "rac_membrane_inactive", "rho_membrane_active", "rho_membrane_inactive"]
        self.num_rgtpase_labels = len(self.rgtpase_labels)
        
        self.font_size = font_size
        
        self.offset_scale = offset_scale
        self.polygon_line_width = polygon_line_width
        self.rgtpase_line_width = rgtpase_line_width
        self.velocity_line_width = velocity_line_width
        self.centroid_line_width = centroid_line_width
        self.coa_line_width = coa_line_width
        self.chemoattractant_line_width = chemoattractant_line_width
        
        self.show_physical_bdry_polygon = show_physical_bdry_polygon
        if self.show_physical_bdry_polygon == True:
            self.space_physical_bdry_polygon = space_physical_bdry_polygon/1e-6
        else:
            self.space_physical_bdry_polygon = np.array([], dtype=np.float64)
        self.space_migratory_bdry_polygon = space_migratory_bdry_polygon/1e-6
        self.chemoattractant_source_location = chemoattractant_source_location
        
        self.timestep_length = timestep_length
            
        self.cell_etas = cell_etas
        self.cell_Ls = cell_Ls
        self.offset_magnitudes = np.array(self.cell_Ls)*self.offset_scale
        self.cell_skip_dynamics = cell_skip_dynamics
        self.storefile_path = env_storefile_path
        
        self.global_image_dir = os.path.join(general_animation_save_folder_path, "images_global")
        
        if not os.path.exists(self.global_image_dir):
            os.makedirs(self.global_image_dir)
        else:
            shutil.rmtree(self.global_image_dir)
            os.makedirs(self.global_image_dir)
            
        self.coa_overlay_data_store_path = os.path.join(general_animation_save_folder_path, "coa_overlay_data.hdf5")
            
        self.gathered_info = np.zeros((self.max_num_timepoints, self.num_cells), dtype=np.int64)
        self.animation_cells = np.empty(self.num_cells, dtype=object)
        self.image_drawn_array = self.determine_drawn_timesteps(np.zeros(self.max_num_timepoints, dtype=np.int64))
        
        self.coa_overlay_info = (self.plate_width_in_micrometers, self.plate_height_in_micrometers, max_coa_signal, coa_too_close_dist_squared, coa_distribution_exponent, cell_dependent_coa_signal_strengths)
        self.coa_overlay_color = coa_overlay_color
        self.coa_overlay_resolution = coa_overlay_resolution
        self.max_coa_signal = max_coa_signal
        self.coa_too_close_dist_squared = coa_too_close_dist_squared
        self.coa_distribution_exponent = coa_distribution_exponent
        self.cell_dependent_coa_signal_strengths = cell_dependent_coa_signal_strengths
        self.coa_intersection_exponent = coa_intersection_exponent
        
        self.allowed_drift_before_geometry_recalc = allowed_drift_before_geometry_recalc
        self.specific_timesteps_to_draw_as_svg = specific_timesteps_to_draw_as_svg
    
    # ---------------------------------------------------------------------
    
    def determine_drawn_timesteps(self, image_drawn_array=np.array([])):
        if image_drawn_array.shape[0] == 0:
            image_drawn_array = np.zeros(self.max_num_timepoints, dtype=np.int64)
        
        drawn_timepoints = [int(fn[10:-4]) for fn in os.listdir(self.global_image_dir)]
        image_drawn_array[drawn_timepoints] = 1
        
        return image_drawn_array
        
    # ---------------------------------------------------------------------
        
    def calculate_transform_matrix(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.image_width_in_pixels, self.image_height_in_pixels)
        context = cairo.Context(surface)
        context.translate(0, self.image_height_in_pixels)
        context.scale(1, -1)
        context.scale(self.global_scale, self.global_scale)
        context.rotate(self.rotation_theta)
        context.translate(self.translation_x, self.translation_y)
#        xx, xy, x0, yx, yy, y0 = create_transformation_matrix_entries(self.global_scale, self.global_scale, self.rotation_theta + np.pi, self.translation_x, self.translation_y, self.plate_width_in_micrometers, self.plate_width_in_micrometers)
        #return cairo.Matrix(xx, yx, xy, yy, x0, y0)
        return context.get_matrix()
    
    # ---------------------------------------------------------------------
        
    def gather_data(self, timestep_to_draw_till, unique_undrawn_timesteps):
        polygon_coords_per_timepoint_per_cell = np.zeros((self.num_cells, unique_undrawn_timesteps.shape[0], self.num_nodes, 2), dtype=np.float64)
        if self.show_velocities:
            velocity_line_coords_per_label_per_timepoint_per_cell = np.zeros((self.num_cells, unique_undrawn_timesteps.shape[0], self.num_velocity_labels, self.num_nodes, 2))
        else:
            velocity_line_coords_per_label_per_timepoint_per_cell = None
            
        if self.show_centroid_trail:
            centroid_coords_per_timepoint_per_cell = np.empty((self.num_cells, timestep_to_draw_till, 2), dtype=np.float64)
        else:
            centroid_coords_per_timepoint_per_cell = None
            
        if self.show_rgtpase:
            rgtpase_line_coords_per_label_per_timepoint_per_cell = np.zeros((self.num_cells, unique_undrawn_timesteps.shape[0], self.num_rgtpase_labels + 1, self.num_nodes, 2))
        else:
            rgtpase_line_coords_per_label_per_timepoint_per_cell = None
        
        if self.show_rac_random_spikes:
            rac_random_spike_info_per_timepoint_per_cell = np.zeros((self.num_cells, unique_undrawn_timesteps.shape[0], self.num_nodes))
        else:
            rac_random_spike_info_per_timepoint_per_cell = None
            
        if self.show_coa:
            coa_line_coords_per_timepoint_per_cell = np.zeros((self.num_cells, unique_undrawn_timesteps.shape[0], self.num_nodes, 2), dtype=np.float64)
        else:
            coa_line_coords_per_timepoint_per_cell = None

        if self.show_chemoattractant:
            chemoattractant_line_coords_per_timepoint_per_cell = np.zeros((self.num_cells, unique_undrawn_timesteps.shape[0], self.num_nodes, 2), dtype=np.float64)
        else:
            chemoattractant_line_coords_per_timepoint_per_cell = None
            
        coa_grid_points, coa_grid_point_data = np.array([]), np.array([])
        
        print("Gathering data for visualizing cells...")
        for cell_index in range(self.num_cells):
            L = self.cell_Ls[cell_index]
            this_cell_polygon_coords_per_timestep = L*hardio.get_node_coords_for_given_tsteps(cell_index, unique_undrawn_timesteps, self.storefile_path)
            
            polygon_coords_per_timepoint_per_cell[cell_index,:,:,:] = this_cell_polygon_coords_per_timestep
            
            if self.show_centroid_trail:
                centroid_coords_per_timepoint_per_cell[cell_index,:,:] = cu.calculate_cell_centroids_until_tstep(cell_index, timestep_to_draw_till, self.storefile_path)
            
            if self.show_velocities:
                eta = self.cell_etas[cell_index]
                velocity_data_for_undrawn_timesteps = prepare_velocity_data(self.num_nodes, eta, self.velocity_scale, cell_index, unique_undrawn_timesteps, this_cell_polygon_coords_per_timestep, self.storefile_path)
                
                for x in range(self.num_velocity_labels):
                    velocity_line_coords_per_label_per_timepoint_per_cell[cell_index,:,x,:,:] = velocity_data_for_undrawn_timesteps[x]
                    
            if self.show_rgtpase:
                offset_magnitude = self.offset_magnitudes[cell_index]
                rgtpase_data_for_undrawn_timesteps = prepare_rgtpase_data(self.rgtpase_scale, cell_index, unique_undrawn_timesteps, this_cell_polygon_coords_per_timestep, self.global_scale*offset_magnitude, self.storefile_path)
                
                for x in range(self.num_rgtpase_labels + 1):
                    rgtpase_line_coords_per_label_per_timepoint_per_cell[cell_index,:,x,:,:] = rgtpase_data_for_undrawn_timesteps[x]

                    
            if self.show_rac_random_spikes:
                rac_random_spike_info_per_timepoint_per_cell[cell_index,:,:] = prepare_rac_random_spike_data(cell_index, unique_undrawn_timesteps, self.storefile_path)
                
            if self.show_coa:
                coa_line_coords_per_timepoint_per_cell[cell_index,:,:,:] = prepare_coa_data(self.coa_scale, cell_index, unique_undrawn_timesteps, this_cell_polygon_coords_per_timestep, self.storefile_path)

            if self.show_chemoattractant:
                chemoattractant_line_coords_per_timepoint_per_cell[cell_index,:,:,:] = prepare_chemoattractant_data(self.chemoattractant_scale, cell_index, unique_undrawn_timesteps, this_cell_polygon_coords_per_timestep, offset_magnitude, self.storefile_path)
                
        if self.show_coa_overlay:
#            assert(self.allowed_drift_before_geometry_recalc >= 0)
#            allowed_drift_before_geometry_recalc = self.allowed_drift_before_geometry_recalc
#            
#            recalc_timepoints = []
#            prev_centroids = centroid_coords_per_timepoint_per_cell[:,0,:]
#            recalc_geometry = True
#            centroid_drifts = 0.0
#            
#            for t in range(timestep_to_draw_till):
#                curr_centroids = centroid_coords_per_timepoint_per_cell[:,t,:]
#                delta_drifts = geometry.calculate_centroid_dift(prev_centroids, curr_centroids)
#                prev_centroids = curr_centroids
#                if recalc_geometry:
#                    centroid_drifts = 0.0
#                    recalc_geometry = False
#                    recalc_timepoints.append(t)
#                else:
#                    centroid_drifts += delta_drifts
#                    
#                if centroid_drifts > allowed_drift_before_geometry_recalc:
#                    recalc_geometry = True
#            
#            coa_overlay_timepoints = np.zeros_like(unique_undrawn_timesteps)
#            curr_overlay_timepoint = 0
#            cursor = 0
#            for i, t in enumerate(unique_undrawn_timesteps):
#                if t >= recalc_timepoints[cursor]:
#                    curr_overlay_timepoint = t
#                    cursor += 1
#                coa_overlay_timepoints[i] = curr_overlay_timepoint
#                
#            coa_overlay_timepoints = sorted(list(set(list(coa_overlay_timepoints))))
                
            coa_grid_points, coa_grid_point_data = generate_coa_overlay_data(unique_undrawn_timesteps, self.plate_width_in_micrometers, self.plate_height_in_micrometers, self.max_coa_signal, self.coa_too_close_dist_squared, self.coa_distribution_exponent, self.coa_intersection_exponent, self.cell_dependent_coa_signal_strengths, polygon_coords_per_timepoint_per_cell, self.space_migratory_bdry_polygon, self.space_physical_bdry_polygon, self.coa_overlay_data_store_path, resolution=self.coa_overlay_resolution)
                
        print("Done gathering visualization data.")              
        return polygon_coords_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, rac_random_spike_info_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell, coa_grid_points, coa_grid_point_data, chemoattractant_line_coords_per_timepoint_per_cell
                                    
    # ---------------------------------------------------------------------
                                    
    def create_animation_cells(self):
        animation_cells = []
        
        len_only_show_cells = len(self.only_show_cells)
        len_cell_poly_colors = len(self.cell_polygon_edge_and_vertex_colors)
        
        for cell_index in range(self.num_cells):
            if len_only_show_cells == 0:
                hidden = False
            elif (cell_index in self.only_show_cells):
                hidden = False
            else:
                hidden = True
                
            if self.cell_skip_dynamics[cell_index] == True:
                show_velocities = False
                show_rgtpase = False
                show_centroid_trail = False
                show_coa = False
                show_chemoattractant = False
            else:
                show_velocities = self.show_velocities
                show_rgtpase = self.show_rgtpase
                show_centroid_trail = self.show_centroid_trail
                show_coa = self.show_coa
                show_chemoattractant = self.show_chemoattractant
            
            polygon_edge_and_vertex_color = self.default_cell_polygon_edge_and_vertex_color
            if len_cell_poly_colors > 0:
                for i in range(len_cell_poly_colors):
                    if self.cell_polygon_colors[i][0] == cell_index:
                        polygon_edge_and_vertex_color = self.cell_polygon_colors[i][1]
                        
            colors.color_list20[cell_index%20]
            animation_cell = AnimationCell(polygon_edge_and_vertex_color, self.default_cell_polygon_fill_color, self.rgtpase_colors, self.rgtpase_background_shine_color, self.velocity_colors, colors.color_list20[cell_index%20], self.coa_color, self.chemoattractant_dot_color, hidden, show_rgtpase, show_velocities, show_centroid_trail, show_coa, show_chemoattractant, self.polygon_line_width, self.rgtpase_line_width, self.velocity_line_width, self.centroid_line_width, self.coa_line_width, self.chemoattractant_line_width)
            
            animation_cells.append(animation_cell)
            
        return animation_cells
            
    # ---------------------------------------------------------------------
        
    def create_animation_from_data(self, animation_save_folder_path, animation_file_name, timestep_to_draw_till=None, duration=None, num_threads=8, multithread=True):        
        if timestep_to_draw_till == None:
            timestep_to_draw_till = self.environment.num_timepoints
            
        if duration == None or duration == 'auto': 
            if timestep_to_draw_till*self.timestep_length < self.short_video_length_definition:
                duration = self.short_video_duration
            else:
                duration = (timestep_to_draw_till*self.timestep_length/self.short_video_length_definition)*self.short_video_duration
                
        num_frames = duration*self.fps
        
        if len(self.specific_timesteps_to_draw_as_svg) == 0:
            unique_timesteps = np.sort(np.array([x for x in list(set(np.linspace(0, timestep_to_draw_till, num=num_frames, endpoint=False, dtype=np.int64)))]))
            num_unique_timesteps = unique_timesteps.shape[0]
            
            unique_undrawn_timesteps = np.array([x for x in unique_timesteps if self.image_drawn_array[x] == 0])
            image_format = ".png"
            
            local_image_dir = os.path.join(animation_save_folder_path, "images_n={}_fps={}_t={}".format(num_unique_timesteps, self.fps, duration))
        
            if not os.path.exists(local_image_dir):
                os.makedirs(local_image_dir)
            else:
                shutil.rmtree(local_image_dir)
                os.makedirs(local_image_dir)
                
            max_local_image_number_length = len(str(num_unique_timesteps))
            local_image_name_format_str = "img{{:0>{}}}.png".format(max_local_image_number_length) 
            
            global_image_dir = self.global_image_dir
            
            max_global_image_number_length = len(str(self.max_num_timepoints))
            global_image_name_format_str = "global_img{{:0>{}}}.png".format(max_global_image_number_length)
        else:
            local_image_dir = None
            image_format = ".svg"
            local_image_name_format_str = None
            unique_undrawn_timesteps = np.array(self.specific_timesteps_to_draw_as_svg)
            unique_timesteps = np.arange(0, timestep_to_draw_till)
            global_image_dir = self.global_image_dir + "-svg"
            
            if not os.path.exists(global_image_dir):
                os.makedirs(global_image_dir)
            else:
                shutil.rmtree(global_image_dir)
                os.makedirs(global_image_dir)
                
            max_global_image_number_length = len(str(self.max_num_timepoints))
            global_image_name_format_str = "global_img{{:0>{}}}.svg".format(max_global_image_number_length)
        
        polygon_coords_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, rac_random_spike_info_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell, coa_grid_points, coa_data_per_grid_point_per_timepoint, chemoattractant_line_coords_per_timepoint_per_cell = self.gather_data(timestep_to_draw_till, unique_undrawn_timesteps)

        animation_cells = self.create_animation_cells()
        
        timestep_length = self.timestep_length
        font_color = self.font_color
        font_size = self.font_size
        global_scale = self.global_scale
        plate_width = self.plate_width_in_micrometers
        plate_height = self.plate_height_in_micrometers
        image_width_in_pixels = self.image_width_in_pixels
        image_height_in_pixels = self.image_height_in_pixels
        transform_matrix = self.transform_matrix
        space_physical_bdry_polygon = self.space_physical_bdry_polygon
        space_migratory_bdry_polygon = self.space_migratory_bdry_polygon
        chemoattractant_source_location = self.chemoattractant_source_location
        chemotaxis_target_radius = self.chemotaxis_target_radius
        coa_overlay_color = self.coa_overlay_color
        coa_overlay_resolution = self.coa_overlay_resolution
        
        background_color = self.background_color
        migratory_bdry_color = self.migratory_bdry_color
        physical_bdry_color = self.physical_bdry_color
        chemoattractant_dot_color = self.chemoattractant_dot_color
        
        image_prep_st = time.time()       
        st = time.time()
        print("Drawing undrawn images....")

        drawing_tasks = []
        for i, t in enumerate(unique_undrawn_timesteps):
            this_image_coa_data = None
            if self.show_coa_overlay:
                this_image_coa_data = coa_data_per_grid_point_per_timepoint[i]
                
            self.image_drawn_array[t] = 1

            drawing_tasks.append((i, t, timestep_length, font_color, font_size, global_scale, plate_width, plate_height, image_height_in_pixels, image_width_in_pixels, transform_matrix, animation_cells, polygon_coords_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, rac_random_spike_info_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell, space_physical_bdry_polygon, space_migratory_bdry_polygon, chemoattractant_source_location, chemotaxis_target_radius, chemoattractant_line_coords_per_timepoint_per_cell, coa_overlay_color, coa_overlay_resolution, coa_grid_points, this_image_coa_data, background_color, migratory_bdry_color, physical_bdry_color, chemoattractant_dot_color, unique_timesteps, global_image_dir, global_image_name_format_str, image_format))

        pool = ProcessPool(nodes=4)
        pool.map(draw_animation_frame, drawing_tasks)

#         for task in drawing_tasks:
#             draw_animation_frame(task)
            
        et = time.time()
        print("Time taken to draw images: {} s".format(np.round(et - st, decimals=3)))
        
        if len(self.specific_timesteps_to_draw_as_svg) == 0:
            st = time.time()
            copying_tasks = [] 
            print("Copying pre-drawn images...")                
            for i, t in enumerate(unique_timesteps):
                assert(self.image_drawn_array[t] == 1)
                copying_tasks.append((os.path.join(global_image_dir, global_image_name_format_str.format(t)), os.path.join(local_image_dir, local_image_name_format_str.format(i))))
                
            num_tasks = len(copying_tasks)
            chunklen = (num_tasks + num_threads - 1) // num_threads
            # Create argument tuples for each input chunk
            chunks = []
            for i in range(num_threads):
                chunks.append(copying_tasks[i*chunklen:(i+1)*chunklen])
                
            threads = [threading.Thread(target=copy_worker, args=(c,)) for c in chunks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            et = time.time() 
            print("Time taken to copy images: {} s".format(np.round(et - st, decimals=3)))
            
            image_prep_et = time.time()
            
            print("Done preparing images. Total time taken: {}s".format(np.round(image_prep_et - image_prep_st, decimals=3)))
            
            if self.string_together_into_animation == True and len(self.specific_timesteps_to_draw_as_svg) == 0:
                animation_output_path = os.path.join(animation_save_folder_path, animation_file_name)
                
                print("Stringing together pictures...")
                
                command = [ 'ffmpeg',
                '-y', # (optional) overwrite output file if it exists,
                '-framerate', str(float(num_unique_timesteps)/duration),
                '-i', os.path.join(local_image_dir, 'img%0{}d.png'.format(max_local_image_number_length)),
                '-r', str(self.fps), # frames per second
                '-an', # Tells FFMPEG not to expect any audio
                '-threads', str(4),
                '-vcodec', 'libx264', 
                '-pix_fmt', 'yuv420p',
                animation_output_path ]
                
                subprocess.call(command)
            