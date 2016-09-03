from __future__ import division
import cairo
import numpy as np
import numba as nb
import core.geometry as geometry
import os, shutil
import subprocess
import time
import colors
import sys
import core.hardio as hardio
import analysis.utilities as analysis_utils

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

@nb.jit()
def draw_polygon_jit(polygon_coords, polygon_color, polygon_line_width, context):
    context.new_path()
    r, g, b = polygon_color
    context.set_source_rgb(r, g, b)
    context.set_line_width(polygon_line_width)

    for n, coord in enumerate(polygon_coords):
        x, y = coord
        if n == 0:
            context.move_to(x, y)
        else:
            context.line_to(x, y)
            
    context.close_path()
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
    
class AnimationCell():
    def __init__(self, polygon_color, rgtpase_colors, velocity_colors, centroid_color, coa_color, hidden, show_rgtpase, show_velocities, show_centroid_trail, show_coa, polygon_line_width, rgtpase_line_width, velocity_line_width, centroid_line_width, coa_line_width):
        self.hidden = hidden
        
        self.show_velocities = show_velocities
        self.show_rgtpase = show_rgtpase
        self.show_centroid_trail = show_centroid_trail
        self.show_coa = show_coa
        
        self.polygon_color = polygon_color
        self.polygon_line_width = polygon_line_width
   
        self.rgtpase_colors = rgtpase_colors
        self.rgtpase_line_width = rgtpase_line_width

        self.velocity_colors = velocity_colors
        self.velocity_line_width = velocity_line_width
 
        self.centroid_color = centroid_color
        self.centroid_line_width = centroid_line_width
        
        self.coa_color = coa_color
        self.coa_line_width = coa_line_width

    # -------------------------------------
        
    def hide(self):
        self.hidden = True
    
    # -------------------------------------
    
    def unhide(self):
        self.hidden = False
        
    # -------------------------------------
    
    def draw_cell_polygon(self, context, polygon_coords):
        draw_polygon_jit(polygon_coords, self.polygon_color, self.polygon_line_width, context)
        
    # -------------------------------------
        
    def draw_rgtpase(self, context, polygon_coords, rgtpase_line_coords_per_gtpase):        
        context.set_line_width(self.rgtpase_line_width)
        offset_coords = rgtpase_line_coords_per_gtpase[-1]
        offset_directions = [1, -1, 1, -1]
        
        for i, rgtpase_line_coords in enumerate(rgtpase_line_coords_per_gtpase[:-1]):
            offset_direction = offset_directions[i]
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
    
    def draw(self, context, polygon_coords, rgtpase_line_coords_per_label, velocity_line_coords_per_label, centroid_coords_per_frame, coa_line_coords):
        if self.hidden == False:
            
            self.draw_cell_polygon(context, polygon_coords)
            
            if (self.show_velocities == True):
                self.draw_velocities(context, polygon_coords, velocity_line_coords_per_label)
                
            if (self.show_rgtpase == True):
                self.draw_rgtpase(context, polygon_coords, rgtpase_line_coords_per_label)
            
            if (self.show_centroid_trail == True):
                self.draw_centroid_trail(context, centroid_coords_per_frame)
                
            if (self.show_coa == True):
                self.draw_coa(context, polygon_coords, coa_line_coords)
                
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
        
    offset_vecs = offset_magnitude*normal_to_uivs
    
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
    
    coa_signal = coa_mag*unit_inside_pointing_vecs + polygon_coords_per_timestep
    
    return coa_signal
# -------------------------------------

def draw_timestamp(timestep, timestep_length, text_color, font_size, global_scale, img_width, img_height, context):
    text_r, text_g, text_b = text_color
    context.set_source_rgb(text_r, text_g, text_b)
    context.select_font_face("Consolas", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(font_size/global_scale)

    timestamp_string = "T = {} min".format(np.round(timestep*timestep_length/60.0))
    #timestamp_string = "NT = {} ".format(np.round(timestep))
    text_x_bearing, text_y_bearing, text_width, text_height = context.text_extents(timestamp_string)[:4]
    context.move_to((img_width - 1.2*text_width), (img_height - 1.2*text_height))
    context.show_text(timestamp_string)
    
    return

# -------------------------------------
   
def draw_animation_frame_for_given_timestep(timestep_index, timestep, timestep_length, font_color, font_size, global_scale, plate_width, plate_height, image_height_in_pixels, image_width_in_pixels, transform_matrix, animation_cells, polygon_coords_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell, space_physical_bdry_polygon, space_migratory_bdry_polygon, unique_timesteps, global_image_dir, global_image_name_format_str):
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, image_width_in_pixels, image_height_in_pixels)
    context = cairo.Context(surface)
        
    context.set_source_rgb(*colors.RGB_WHITE)
    context.paint()

    # timestep, timestep_length, text_color, font_size, global_scale, img_width, img_height, context
    draw_timestamp(timestep, timestep_length, font_color, font_size, global_scale, image_width_in_pixels, image_height_in_pixels, context)
    
    context.transform(transform_matrix)
    
    for cell_index, anicell in enumerate(animation_cells):
        
        if rgtpase_line_coords_per_label_per_timepoint_per_cell == None:
            rgtpase_data = None
        else:
            rgtpase_data = rgtpase_line_coords_per_label_per_timepoint_per_cell[cell_index][timestep_index]
            
        if velocity_line_coords_per_label_per_timepoint_per_cell == None:
            velocity_data = None
        else:
            velocity_data = velocity_line_coords_per_label_per_timepoint_per_cell[cell_index][timestep_index]
        
        if centroid_coords_per_timepoint_per_cell == None:
            centroid_data = None
        else:
            centroid_data = centroid_coords_per_timepoint_per_cell[cell_index][unique_timesteps[:timestep_index+1]]
        
        if coa_line_coords_per_timepoint_per_cell == None:
            coa_data = None
        else:
            coa_data = coa_line_coords_per_timepoint_per_cell[cell_index][timestep_index]
            
        anicell.draw(context, polygon_coords_per_timepoint_per_cell[cell_index][timestep_index], rgtpase_data, velocity_data, centroid_data, coa_data)
        
    if space_physical_bdry_polygon.shape[0] != 0:
        context.new_path()
        draw_polygon_jit(space_physical_bdry_polygon/1e-6, colors.RGB_BLACK, 2, context)
        
    if space_migratory_bdry_polygon.shape[0] != 0:
        context.new_path()
        draw_polygon_jit(space_migratory_bdry_polygon/1e-6, colors.RGB_BRIGHT_RED, 2, context)
    
    image_fp = os.path.join(global_image_dir, global_image_name_format_str.format(timestep))
    surface.write_to_png(image_fp)

# ------------------------------------- 
  
def make_progress_str(progress, len_progress_bar=20, progress_char="-"):
    num_progress_chars = int(progress*len_progress_bar)
    return "|" + progress_char*num_progress_chars + " "*(len_progress_bar - num_progress_chars) + "|"
    

# -------------------------------------    
    
class EnvironmentAnimation():
    def __init__(self, general_animation_save_folder_path, environment_name, num_cells, num_nodes, max_num_timepoints, cell_group_indices, cell_Ls, cell_etas, cell_skip_dynamics, env_storefile_path, global_scale=1, plate_height_in_micrometers=400, plate_width_in_micrometers=600, rotation_theta=0.0, translation_x=10, translation_y=10, velocity_scale=1, rgtpase_scale=1, coa_scale=1, show_velocities=False, show_rgtpase=False, show_centroid_trail=False, show_coa=True, color_each_group_differently=False, only_show_cells=[], background_color=colors.RGB_WHITE, cell_polygon_colors=[], default_cell_polygon_color=(0,0,0), rgtpase_colors=[colors.RGB_BRIGHT_BLUE, colors.RGB_LIGHT_BLUE, colors.RGB_BRIGHT_RED, colors.RGB_LIGHT_RED], velocity_colors=[colors.RGB_ORANGE, colors.RGB_LIGHT_GREEN, colors.RGB_LIGHT_GREEN, colors.RGB_CYAN, colors.RGB_MAGENTA], coa_color=colors.RGB_DARK_GREEN, font_size=16, font_color=colors.RGB_BLACK, offset_scale=0.2, polygon_line_width=1, rgtpase_line_width=1, velocity_line_width=1, coa_line_width=1, space_physical_bdry_polygon=np.array([]), space_migratory_bdry_polygon=np.array([]), centroid_colors_per_cell=[], centroid_line_width=1, short_video_length_definition=2000.0, short_video_duration=5.0, timestep_length=None, fps=30, origin_offset_in_pixels=np.zeros(2), string_together_pictures_into_animation=True):        
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
        self.image_width_in_pixels = np.int(np.round(plate_width_in_micrometers*global_scale, decimals=0))
        
        self.transform_matrix = self.calculate_transform_matrix()
        
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.max_num_timepoints = max_num_timepoints
        
        self.show_velocities = show_velocities
        self.show_rgtpase = show_rgtpase
        self.show_centroid_trail = show_centroid_trail
        self.show_coa = show_coa
        
        self.velocity_scale = velocity_scale
        self.rgtpase_scale = rgtpase_scale
        self.coa_scale = coa_scale
        
        self.only_show_cells = only_show_cells
        self.background_color = background_color
        
        self.cell_polygon_colors = []
        if color_each_group_differently == True and len(cell_polygon_colors) == 0:
            for ci in xrange(num_cells):
                self.cell_polygon_colors.append((ci, colors.color_list_cell_groups10[cell_group_indices[ci]%10]))
        else:
            self.cell_polygon_colors = cell_polygon_colors
        
        self.velocity_labels = ["F", "EFplus", "EFminus", "F_rgtpase", "F_cytoplasmic"]
        self.num_velocity_labels = len(self.velocity_labels)
        
        self.rgtpase_labels = ["rac_membrane_active", "rac_membrane_inactive", "rho_membrane_active", "rho_membrane_inactive"]
        self.num_rgtpase_labels = len(self.rgtpase_labels)
        
        self.default_cell_polygon_color = default_cell_polygon_color
        self.rgtpase_colors = rgtpase_colors
        self.velocity_colors = velocity_colors
        self.coa_color = coa_color
        
        self.font_size = font_size
        self.font_color = font_color
        
        self.offset_scale = offset_scale
        self.polygon_line_width = polygon_line_width
        self.rgtpase_line_width = rgtpase_line_width
        self.velocity_line_width = velocity_line_width
        self.centroid_line_width = centroid_line_width
        self.coa_line_width = coa_line_width
        
        self.space_physical_bdry_polygon = space_physical_bdry_polygon
        self.space_migratory_bdry_polygon = space_migratory_bdry_polygon
        
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
            
        self.gathered_info = np.zeros((self.max_num_timepoints, self.num_cells), dtype=np.int64)
        self.animation_cells = np.empty(self.num_cells, dtype=object)
        self.image_drawn_array = self.determine_drawn_timesteps(np.zeros(self.max_num_timepoints, dtype=np.int64))
        self.cell_offset_magnitudes = np.zeros(num_cells, dtype=np.float64)
        
    
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
            
        if self.show_coa:
            coa_line_coords_per_timepoint_per_cell = np.zeros((self.num_cells, self.max_num_timepoints, self.num_nodes, 2), dtype=np.float64)
        else:
            coa_line_coords_per_timepoint_per_cell = None
        
        for cell_index in range(self.num_cells):
            L = self.cell_Ls[cell_index]
            
            polygon_coords_per_timestep = L*hardio.get_node_coords_for_given_tsteps(cell_index, unique_undrawn_timesteps, self.storefile_path)
            
            polygon_coords_per_timepoint_per_cell[cell_index,:,:,:] = polygon_coords_per_timestep
            
            if self.show_centroid_trail:
                centroid_coords_per_timepoint_per_cell[cell_index,:,:] = analysis_utils.calculate_cell_centroids_until_tstep(cell_index, timestep_to_draw_till, self.storefile_path)
            
            if self.show_velocities:
                eta = self.cell_etas[cell_index]
                velocity_data_for_undrawn_timesteps = prepare_velocity_data(self.num_nodes, eta, self.velocity_scale, cell_index, unique_undrawn_timesteps, polygon_coords_per_timestep, self.storefile_path)
                
                for x in xrange(self.num_velocity_labels):
                    velocity_line_coords_per_label_per_timepoint_per_cell[cell_index,:,x,:,:] = velocity_data_for_undrawn_timesteps[x]
                    
            if self.show_rgtpase:
                offset_magnitude = self.offset_magnitudes[cell_index]
                rgtpase_data_for_undrawn_timesteps = prepare_rgtpase_data(self.rgtpase_scale, cell_index, unique_undrawn_timesteps, polygon_coords_per_timestep, offset_magnitude, self.storefile_path)
                
                for x in xrange(self.num_rgtpase_labels + 1):
                    rgtpase_line_coords_per_label_per_timepoint_per_cell[cell_index,:,x,:,:] = rgtpase_data_for_undrawn_timesteps[x]
                
            if self.show_coa:
                coa_line_coords_per_timepoint_per_cell[cell_index,:,:,:] = prepare_coa_data(self.coa_scale, cell_index, unique_undrawn_timesteps, polygon_coords_per_timepoint_per_cell, self.storefile_path)
                
                        
        return polygon_coords_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell
                                    
                                    
    # ---------------------------------------------------------------------
                                    
    def create_animation_cells(self):
        animation_cells = []
        
        len_only_show_cells = len(self.only_show_cells)
        len_cell_poly_colors = len(self.cell_polygon_colors)
        
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
            else:
                show_velocities = self.show_velocities
                show_rgtpase = self.show_rgtpase
                show_centroid_trail = self.show_centroid_trail
                show_coa = self.show_coa
            
            polygon_color = self.default_cell_polygon_color
            if len_cell_poly_colors > 0:
                for i in range(len_cell_poly_colors):
                    if self.cell_polygon_colors[i][0] == cell_index:
                        polygon_color = self.cell_polygon_colors[i][1]
                        
            colors.color_list20[cell_index%20]
            animation_cell = AnimationCell(polygon_color, self.rgtpase_colors, self.velocity_colors, colors.color_list20[cell_index%20], self.coa_color, hidden, show_rgtpase, show_velocities, show_centroid_trail, show_coa, self.polygon_line_width, self.rgtpase_line_width, self.velocity_line_width, self.centroid_line_width, self.coa_line_width)
            
            animation_cells.append(animation_cell)
            
        return animation_cells
            
    # ---------------------------------------------------------------------
        
    def create_animation_from_data(self, animation_save_folder_path, timestep_to_draw_till=None, duration=None):
        
        if timestep_to_draw_till == None:
            timestep_to_draw_till = self.environment.num_timepoints
            
        if duration == None or duration == 'auto': 
            if timestep_to_draw_till*self.timestep_length < self.short_video_length_definition:
                duration = self.short_video_duration
            else:
                duration = (timestep_to_draw_till*self.timestep_length/self.short_video_length_definition)*self.short_video_duration
                
        num_frames = duration*self.fps
            
        unique_timesteps = np.sort(np.array([x for x in list(set(np.linspace(0, timestep_to_draw_till, num=num_frames, endpoint=False, dtype=np.int64)))]))
        num_unique_timesteps = unique_timesteps.shape[0]
        
        unique_undrawn_timesteps = np.array([x for x in unique_timesteps if self.image_drawn_array[x] == 0])
        
        polygon_coords_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell = self.gather_data(timestep_to_draw_till, unique_undrawn_timesteps)

        animation_cells = self.create_animation_cells()
        
        local_image_dir = os.path.join(animation_save_folder_path, "images_n={}_fps={}_t={}".format(num_unique_timesteps, self.fps, duration))
        
        if not os.path.exists(local_image_dir):
            os.makedirs(local_image_dir)
        else:
            shutil.rmtree(local_image_dir)
            os.makedirs(local_image_dir)
        
        max_global_image_number_length = len(str(self.max_num_timepoints))
        global_image_name_format_str = "global_img{{:0>{}}}.png".format(max_global_image_number_length)
        
        max_local_image_number_length = len(str(num_unique_timesteps))
        local_image_name_format_str = "img{{:0>{}}}.png".format(max_local_image_number_length) 
        
        timestep_length = self.timestep_length
        font_color = self.font_color
        font_size = self.font_size
        global_scale = self.global_scale
        plate_width = self.plate_width_in_micrometers
        plate_height = self.plate_height_in_micrometers
        global_image_dir = self.global_image_dir
        image_width_in_pixels = self.image_width_in_pixels
        image_height_in_pixels = self.image_height_in_pixels
        transform_matrix = self.transform_matrix
        space_physical_bdry_polygon = self.space_physical_bdry_polygon
        space_migratory_bdry_polygon = self.space_migratory_bdry_polygon
        
        image_prep_st = time.time()        
        print "Drawing undrawn images...."
        for i, t in enumerate(unique_undrawn_timesteps):
            progress_str = make_progress_str(float(i)/num_unique_timesteps)
            sys.stdout.write("\r")
            sys.stdout.write(progress_str)
            sys.stdout.flush()
            assert(self.image_drawn_array[t] == 0)
            self.image_drawn_array[t] = 1
            draw_animation_frame_for_given_timestep(i, t, timestep_length, font_color, font_size, global_scale, plate_width, plate_height, image_height_in_pixels, image_width_in_pixels, transform_matrix, animation_cells, polygon_coords_per_timepoint_per_cell, rgtpase_line_coords_per_label_per_timepoint_per_cell, velocity_line_coords_per_label_per_timepoint_per_cell, centroid_coords_per_timepoint_per_cell, coa_line_coords_per_timepoint_per_cell, space_physical_bdry_polygon, space_migratory_bdry_polygon, unique_timesteps, global_image_dir, global_image_name_format_str)
            print ""

        print "Copying pre-drawn images..."                
        for i, t in enumerate(unique_timesteps):
            progress_str = make_progress_str(float(i)/num_unique_timesteps)
            sys.stdout.write("\r")
            sys.stdout.write(progress_str)
            sys.stdout.flush()
            assert(self.image_drawn_array[t] == 1)
            print (i, t)
            shutil.copy(os.path.join(global_image_dir, global_image_name_format_str.format(t)), os.path.join(local_image_dir, local_image_name_format_str.format(i)))
            print ""
        
        image_prep_et = time.time()
        
        print "Done preparing images. Time taken: {}s".format(np.round(image_prep_et - image_prep_st, decimals=1))
        
        animation_output_path = os.path.join(animation_save_folder_path, self.animation_name)
        
        if self.string_together_into_animation == True:
            print "Stringing together pictures..."
            
            command = [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists,
            '-framerate', str(float(num_unique_timesteps)/duration),
            '-i', os.path.join(local_image_dir, 'img%0{}d.png'.format(max_local_image_number_length)),
            '-r', str(self.fps), # frames per second
            '-an', # Tells FFMPEG not to expect any audio
            '-threads', str(4),
            '-vcodec', 'libx264', 
            animation_output_path ]
            
            subprocess.call(command)
            