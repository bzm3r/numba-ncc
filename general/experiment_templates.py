# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 16:00:16 2016

@author: Brian Merchant
"""

import numpy as np
import general.exec_utils as eu
import core.utilities as cu
import visualization.datavis as datavis
import os
import copy
import numba as nb
import dill
import core.parameterorg as cporg
import visualization.colors as colors

global_randomization_scheme_dict = {'m': 'kgtp_rac_multipliers', 'w': 'wipeout'}

# =======================================================================

def setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, velocity_scale=1, rgtpase_scale_factor=312.5, coa_scale_factor=62.5, show_velocities=False, show_rgtpase=True, show_inactive_rgtpase=False, show_centroid_trail=False, show_rac_random_spikes=False, show_coa=False, color_each_group_differently=False, only_show_cells=[], polygon_line_width=1, space_physical_bdry_polygon=np.empty((0, 0), dtype=np.float64), space_migratory_bdry_polygon=np.empty((0, 0), dtype=np.float64), short_video_length_definition_factor=1000.0, short_video_duration=5.0, fps=30, string_together_pictures_into_animation=True, show_coa_overlay=False, max_coa_signal=-1.0, coa_too_close_dist_squared=0.0, coa_overlay_resolution=10.0, cell_dependent_coa_signal_strengths=[], coa_distribution_exponent=0.0, coa_intersection_exponent=0.0, background_color=colors.RGB_WHITE, chemoattractant_dot_color=colors.RGB_DARK_GREEN, default_cell_polygon_edge_and_vertex_color=colors.RGB_BLACK, default_cell_polygon_fill_color=colors.RGB_WHITE, rgtpase_colors=[colors.RGB_BRIGHT_BLUE, colors.RGB_LIGHT_BLUE, colors.RGB_BRIGHT_RED, colors.RGB_LIGHT_RED], velocity_colors=[colors.RGB_ORANGE, colors.RGB_LIGHT_GREEN, colors.RGB_LIGHT_GREEN, colors.RGB_CYAN, colors.RGB_MAGENTA], coa_color=colors.RGB_DARK_GREEN, font_color=colors.RGB_BLACK, coa_overlay_color=colors.RGB_LIGHT_GREEN, rgtpase_background_shine_color=None, migratory_bdry_color=colors.RGB_BRIGHT_RED, physical_bdry_color=colors.RGB_BLACK, allowed_drift_before_geometry_recalc=-1.0, specific_timesteps_to_draw_as_svg=[], chemoattractant_source_location=[], chemotaxis_target_radius=-1.0):
    
    return dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', velocity_scale), ('rgtpase_scale', global_scale*rgtpase_scale_factor), ('coa_scale', global_scale*coa_scale_factor), ('show_velocities', show_velocities), ('show_rgtpase', show_rgtpase), ('show_centroid_trail', show_centroid_trail), ('show_rac_random_spikes', show_rac_random_spikes), ('show_coa', show_coa), ('color_each_group_differently', color_each_group_differently), ('only_show_cells', only_show_cells), ('polygon_line_width', polygon_line_width),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', short_video_duration), ('timestep_length', timestep_length), ('fps', fps), ('string_together_pictures_into_animation', string_together_pictures_into_animation), ('show_coa_overlay', show_coa_overlay), ('max_coa_signal', max_coa_signal), ('coa_too_close_dist_squared', coa_too_close_dist_squared), ('coa_distribution_exponent', coa_distribution_exponent), ('coa_intersection_exponent', coa_intersection_exponent), ('coa_overlay_resolution', coa_overlay_resolution), ('cell_dependent_coa_signal_strengths', cell_dependent_coa_signal_strengths), ('background_color', background_color), ('chemoattractant_dot_color', chemoattractant_dot_color), ('default_cell_polygon_edge_and_vertex_color', default_cell_polygon_edge_and_vertex_color), ('default_cell_polygon_fill_color', default_cell_polygon_fill_color), ('rgtpase_colors', rgtpase_colors), ('velocity_colors', velocity_colors), ('coa_color', coa_color), ('font_color', font_color), ('rgtpase_background_shine_color', rgtpase_background_shine_color), ('migratory_bdry_color', migratory_bdry_color), ('physical_bdry_color', physical_bdry_color), ('coa_overlay_color', coa_overlay_color), ('allowed_drift_before_geometry_recalc', allowed_drift_before_geometry_recalc), ('specific_timesteps_to_draw_as_svg', specific_timesteps_to_draw_as_svg), ('chemoattractant_source_location', chemoattractant_source_location), ('chemotaxis_target_radius', chemotaxis_target_radius), ('show_inactive_rgtpase', show_inactive_rgtpase)])

def make_default_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset):
    migr_space_poly = np.zeros((0, 0), dtype=np.float64)
    
    if make_migr_space_poly == True:
        bottom_left = [0 + corridor_x_offset, 0 + corridor_y_offset]
        bottom_right = [width_corridor + corridor_x_offset, 0 + corridor_y_offset]
        top_right = [width_corridor + corridor_x_offset, height_corridor + corridor_y_offset]
        top_left = [0 + corridor_x_offset, height_corridor + corridor_y_offset]
        migr_space_poly = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float64)*1e-6
            
    return migr_space_poly

#=====================================================================
    
def make_bottleneck_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, first_slope_start, first_slope_end, second_slope_start, second_slope_end, bottleneck_factor):
    migr_space_poly = np.zeros((0, 0), dtype=np.float64)
    
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

#=====================================================================
    
def make_obstacle_migr_polygon_and_phys_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, obstacle_x_start, obstacle_width, passage_space):
    migr_space_poly = np.zeros((0, 0), dtype=np.float64)
    
    if make_migr_space_poly == True:
        bottom_left = [0 + corridor_x_offset, 0 + corridor_y_offset]
        bottom_right = [width_corridor + corridor_x_offset, 0 + corridor_y_offset]
        top_right = [width_corridor + corridor_x_offset, height_corridor + corridor_y_offset]
        top_left = [0 + corridor_x_offset, height_corridor + corridor_y_offset]
        migr_space_poly = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float64)*1e-6
    
    obstacle_height = height_corridor - 2*passage_space
    if obstacle_height < 1e-16:
        raise Exception("Passage space is too wide to fit within corridor: corridor height = {}, passage space = {}".format(height_corridor, passage_space))
    bottom_left = [corridor_x_offset + obstacle_x_start, corridor_y_offset + passage_space]
    bottom_right = [corridor_x_offset + obstacle_x_start + obstacle_width, corridor_y_offset + passage_space]
    top_right = [corridor_x_offset + obstacle_x_start + obstacle_width, corridor_y_offset + passage_space + obstacle_height]
    top_left = [corridor_x_offset + obstacle_x_start, corridor_y_offset + passage_space + obstacle_height]
    phys_space_poly = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float64)*1e-6
        
    return migr_space_poly, phys_space_poly

#=====================================================================
    
def generate_bottom_and_top_curves(x_offset, y_offset, curve_start_x, curve_radius, height_corridor, resolution, curve_direction):
    outer_radius = curve_radius + 0.5*height_corridor
    inner_radius = outer_radius - height_corridor
    
    if curve_direction == 1:
        bottom_thetas = np.linspace(0.75*2*np.pi, 2*np.pi, num=resolution)
        top_thetas = np.flip(bottom_thetas, 0)
        bottom_curve_radius = outer_radius
        top_curve_radius = inner_radius
    else:
        top_thetas = np.linspace(0, 0.25*2*np.pi, num=resolution)
        bottom_thetas = np.flip(top_thetas, 0)
        bottom_curve_radius = inner_radius
        top_curve_radius = outer_radius
        
    
    bottom_curve_untranslated = bottom_curve_radius*np.array([[np.cos(t), np.sin(t)] for t in bottom_thetas])
    top_curve_untranslated = top_curve_radius*np.array([[np.cos(t), np.sin(t)] for t in top_thetas])
    
    origin = bottom_curve_untranslated[0]
    bottom_curve_untranslated = bottom_curve_untranslated - origin
    top_curve_untranslated = top_curve_untranslated - origin

    bottom_curve = np.array(bottom_curve_untranslated, dtype=np.float64) + [x_offset, y_offset]
    top_curve = np.array(top_curve_untranslated, dtype=np.float64) + [x_offset, y_offset]
    
    arc_base_line_length = 2*curve_radius*np.sin(np.abs(bottom_thetas[1] - bottom_thetas[0]))
    length_within_curve = resolution*arc_base_line_length
    
    return bottom_curve, top_curve, length_within_curve


# =================================================================
    
def make_curving_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, curve_start_x, curve_radius, resolution, curve_direction):
    
    if make_migr_space_poly == True:
        bottom_curve, top_curve, length_within_curve = generate_bottom_and_top_curves(corridor_x_offset + curve_start_x, corridor_y_offset, curve_start_x, curve_radius, height_corridor, resolution, curve_direction)
        
#        import matplotlib.pyplot as plt
#        fig, ax = plt.subplots()
#        ax.plot(bottom_curve[:,0], bottom_curve[:,1], label='bot', color='r')
#        ax.plot(top_curve[:,0], top_curve[:,1], label='top', color='g')
#        ax.plot(bottom_curve[:1,0], bottom_curve[:1,1], color='r', marker='.')
#        ax.plot(top_curve[:1,0], top_curve[:1,1], color='g', marker='.')
#        ax.legend(loc='best')
#        fig.savefig("B:\\numba-ncc\\output\\2018_FEB_10\\SET=0\\top_and_bot-post.png")
        
        remaining_corridor = width_corridor - length_within_curve
        if remaining_corridor < 1e-16:
            raise Exception("Corridor is not long enough to fit curve!")
        
        bottom_left = [[0 + corridor_x_offset, 0 + corridor_y_offset]]
        bottom_right = [[bottom_curve[-1][0], bottom_curve[-1][1] + curve_direction*remaining_corridor]]
        top_right = [[bottom_right[0][0] - curve_direction*height_corridor, bottom_right[0][1]]]
        top_left = [[0 + corridor_x_offset, 0 + corridor_y_offset + height_corridor]]
        
        full_polygon = np.zeros((0, 2), dtype=np.float64)
        
        
#        fig, ax = plt.subplots()
        #labels = ["bottom_left", "bottom_curve", "bottom_right", "top_right", "top_curve", "top_left"]
        for curve in [bottom_left, bottom_curve, bottom_right, top_right, top_curve, top_left]:
            #ax.plot([x[0] for x in curve], [x[1] for x in curve], label=labels.pop(), marker='.')
            full_polygon = np.append(full_polygon, curve, axis=0)
#        
#        ax.legend(loc='best')
#        fig.set_size_inches(10, 10)
#        fig.savefig("B:\\numba-ncc\\output\\2018_FEB_10\\SET=0\\full_poly.png")
        
    return full_polygon*1e-6

#=====================================================================

def make_space_polygons(corridor_definition, make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset):
    
    if corridor_definition == None or len(corridor_definition) == 0:
        corridor_definition = ["default"]
        make_migr_space_poly = False
        
    if corridor_definition[0] == "default":
        return make_default_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset),  np.zeros((0, 0), dtype=np.float64)
    
    elif corridor_definition[0] == "bottleneck":
        if len(corridor_definition[1:]) != 5:
            raise Exception("Not enough parameters given to make bottleneck corridor.")
            
        first_slope_start, first_slope_end, second_slope_start, second_slope_end, bottleneck_factor = corridor_definition[1:]
        
        #make_migr_space_poly, make_phys_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, first_slope_start, first_slope_end, second_slope_start, second_slope_end, bottleneck_factor
        return make_bottleneck_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, first_slope_start, first_slope_end, second_slope_start, second_slope_end, bottleneck_factor), np.zeros((0, 0), dtype=np.float64)
    
    elif corridor_definition[0] == "regular curve":
        curve_start, curve_radius, resolution, curve_direction = corridor_definition[1:]
        
        return make_curving_migr_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, curve_start, curve_radius, resolution, curve_direction), np.zeros((0, 0), dtype=np.float64)
    
    elif corridor_definition[0] == "obstacle":
        obstacle_x_start, obstacle_width, passage_space = corridor_definition[1:]
        
        return make_obstacle_migr_polygon_and_phys_polygon(make_migr_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, obstacle_x_start, obstacle_width, passage_space)  
    else:
        raise Exception("Unknown boundary polygon definition given: {}".format(corridor_definition[0]))

#=====================================================================
        
def make_centered_physical_obstacle(width_factor, height_factor, x_offset, height_corridor, corridor_x_offset, corridor_y_offset, cell_diameter):
    phys_space_poly = np.zeros((0, 0), dtype=np.float64)
    
    bottom_y = corridor_y_offset + 0.5*height_corridor*(1 - cell_diameter*height_factor)
    top_y = bottom_y + cell_diameter*height_corridor
    
    bottom_left = [x_offset + corridor_x_offset, bottom_y]
    bottom_right = [x_offset + cell_diameter*width_factor + corridor_x_offset, bottom_y]
    top_right = [x_offset + cell_diameter*width_factor + corridor_x_offset, top_y]
    top_left = [x_offset + corridor_x_offset, top_y]
    
    phys_space_poly = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float64)*1e-6
    
    return phys_space_poly
    
#=====================================================================

def define_group_boxes_and_corridors(corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, x_placement_option, y_placement_option, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None], migratory_bdry_x_offset=None, migratory_bdry_y_offset=None):
    test_lists = [num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes]
    test_list_labels = ['num_cells_in_boxes', 'box_heights', 'box_widths', 'x_space_between_boxes']
    allowed_placement_options = ["CENTER", "CENTRE", "ORIGIN", "OVERRIDE"]
    
    if len(box_x_offsets) == 0:
        box_x_offsets = [0.0 for x in range(num_boxes)]
    elif len(box_x_offsets) != num_boxes:
        raise Exception("Incorrect number of box_x_offsets given!")
    if len(box_y_offsets) == 0:
        box_y_offsets = [0.0 for x in range(num_boxes)]
    elif len(box_y_offsets) != num_boxes:
        raise Exception("Incorrect number of box_y_offsets given!")
    
    for test_list_label, test_list in zip(test_list_labels, test_lists):
        if test_list_label == "x_space_between_boxes":
            required_len = num_boxes - 1
        else:
            required_len = num_boxes
            
        if len(test_list) != required_len:
            raise Exception("{} length is not the required length (should be {}, got {}).".format(test_list_label, required_len, len(test_list)))

  
    for axis, placement_option in zip(["x", "y"], [x_placement_option, y_placement_option]):
        if placement_option not in allowed_placement_options:
            raise Exception("Given {} placement option not an allowed placement option!\nGiven: {},\nAllowed: {}".format(axis, placement_option, allowed_placement_options))
            
    if x_placement_option != "OVERRIDE":    
        first_box_offset = 0.0
        if x_placement_option == "ORIGIN":
            first_box_offset = origin_x_offset
        else:
            first_box_offset = 0.5*plate_width - 0.5*(np.sum(box_widths) + np.sum(x_space_between_boxes))
            
        for box_index in range(num_boxes):
            if box_index > 0:
                box_x_offsets[box_index] = first_box_offset + x_space_between_boxes[box_index - 1] + np.sum(box_widths[:box_index]) + np.sum(x_space_between_boxes[:(box_index - 1)])
            else:
                box_x_offsets[box_index] = first_box_offset
                
    if y_placement_option != "OVERRIDE":
        for box_index in range(num_boxes):
            if y_placement_option == "ORIGIN":
                box_y_offsets[box_index] = origin_y_offset
            else:
                box_y_offsets[box_index] = 0.5*plate_height - 0.5*np.max(box_heights)

    make_migr_poly = True
    if migratory_corridor_size == [None, None]:
        make_migr_poly = False
        
    width_migr_corridor, height_migr_corridor = migratory_corridor_size

    if migratory_bdry_x_offset == None:
        migratory_bdry_x_offset = origin_x_offset
    if migratory_bdry_y_offset == None:
        migratory_bdry_y_offset = origin_y_offset
    
    space_migratory_bdry_polygon, space_physical_bdry_polygon = make_space_polygons(corridor_definition, make_migr_poly, width_migr_corridor, height_migr_corridor, migratory_bdry_x_offset, migratory_bdry_y_offset)
    
    return np.arange(num_boxes), box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon
        
# ===========================================================================

def produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals):
    if timesteps_between_generation_of_intermediate_visuals == None:
        produce_intermediate_visuals = np.array([])
    else:
        if type(timesteps_between_generation_of_intermediate_visuals) != int:
            raise Exception("Non-integer value given for timesteps_between_generation_of_intermediate_visuals")
        else:
            if timesteps_between_generation_of_intermediate_visuals > 0:
                produce_intermediate_visuals = np.arange(timesteps_between_generation_of_intermediate_visuals, num_timesteps, step=timesteps_between_generation_of_intermediate_visuals)
            else:
                raise Exception("timesteps_between_generation_of_intermediate_visuals <= 0!")
                
    return produce_intermediate_visuals

# ===========================================================================

def update_pd_with_randomization_info(pd, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w):
    global global_randomization_scheme_dict
    
    if randomization_scheme in ['m', 'w', None]:
        if randomization_scheme != None:
            pd.update([('randomization_scheme', global_randomization_scheme_dict[randomization_scheme])])
        else:
            pd.update([('randomization_scheme', None)])
        
        if randomization_scheme == 'm' or randomization_scheme == None:
            pd.update([('randomization_time_mean', randomization_time_mean_m)])
            pd.update([('randomization_time_variance_factor', randomization_time_variance_factor_m)])
            pd.update([('randomization_magnitude', randomization_magnitude_m)])
        elif randomization_scheme == 'w':
            pd.update([('randomization_time_mean', randomization_time_mean_w)])
            pd.update([('randomization_time_variance_factor', randomization_time_variance_factor_w)])
            pd.update([('randomization_magnitude', 1)])
    else:
        raise Exception("Unknown randomization_scheme given: {} (should be either 'm' or 'w')").format(randomization_scheme)
            
    return pd
            

# ===========================================================================

def fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_overrides_dict):
    if randomization_scheme in ['m', 'w']:
        experiment_name = experiment_name_format_string.format("rand-{}".format(randomization_scheme))
    else:
        experiment_name = experiment_name_format_string.format("rand-{}".format('no'))
        
    return experiment_name
# ===========================================================================

def setup_polarization_experiment(parameter_dict, total_time_in_hours=1, timestep_length=2, cell_diameter=40, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=None, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, init_rho_gtpase_conditions=None):    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [1]
    box_heights = [1*cell_diameter]
    box_widths = [1*cell_diameter]

    x_space_between_boxes = []
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, "CENTER", "CENTER")
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon*1e-6
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon*1e-6
    
    environment_wide_variable_defns = {'parameter_explorer_run': True, 'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': False, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, 'parameter_explorer_init_rho_gtpase_conditions': init_rho_gtpase_conditions, 'cell_placement_method': ""}
    
    cell_dependent_coa_signal_strengths_defn_dict = dict([(x, default_coa) for x in boxes])
    intercellular_contact_factor_magnitudes_defn_dict = dict([(x, default_cil) for x in boxes])
    
    biased_rgtpase_distrib_defn_dict = {'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}
    
    user_cell_group_defn = {'cell_group_name': 0, 'num_cells': 1, 'init_cell_radius': cell_diameter*0.5*1e-6, 'cell_group_bounding_box': np.array([box_x_offsets[0], box_x_offsets[0] + box_widths[0], box_y_offsets[0], box_heights[0] + box_y_offsets[0]])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dict, 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dict, 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dict, 'parameter_dict': parameter_dict} 
        
    return (environment_wide_variable_defns, user_cell_group_defn)

# ===========================================================================

def single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, parameter_dict, base_output_dir="B:\\numba-ncc\\output\\", no_randomization=False, total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, justify_parameters=True, run_experiments=True, remake_graphs=False, remake_animation=False, show_centroid_trail=False, show_randomized_nodes=False, convergence_test=False, Tr_vs_Tp_test=False, do_final_analysis=True, biased_rgtpase_distrib_defn_dict={'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}, zoomed_in=False, show_coa_overlay=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if zoomed_in == True:
        plate_width = 250
        plate_height = 250
        global_scale = 4
    else:
        plate_width = 1000
        plate_height = 1000
        global_scale = 1
        
    if convergence_test:
        experiment_name_format_string = "convergence_{}_NN={}_".format(sub_experiment_number, parameter_dict['num_nodes']) +"{}"
    elif Tr_vs_Tp_test:
        experiment_name_format_string = "Tr_vs_Tp_{}_Tr={}_".format(sub_experiment_number, int(parameter_dict["randomization_time_mean"])) +"{}"
    elif zoomed_in:
        experiment_name_format_string = "single_cell_{}_zi_".format(sub_experiment_number) +"{}"
    else:
        experiment_name_format_string = "single_cell_{}_".format(sub_experiment_number) +"{}"
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [1]
    box_heights = [1*cell_diameter]
    box_widths = [1*cell_diameter]

    x_space_between_boxes = []

    #corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, x_placement_option, y_placement_option, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None], migratory_bdry_x_offset=None, migratory_bdry_y_offset=None
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(None, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, "CENTER", "CENTER")
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, 'cell_placement_method': "", 'convergence_test': convergence_test}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[biased_rgtpase_distrib_defn_dict]]
    parameter_dict_per_sub_experiment = [[parameter_dict]]
    experiment_descriptions_per_subexperiment = ["from experiment template: single cell, no randomization"]
    chemoattractant_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)
    
    cell_dependent_coa_signal_strengths = []
    cc = -1
    for cgd in user_cell_group_defns:
        pass

    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=show_randomized_nodes, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), cell_dependent_coa_signal_strengths=cell_dependent_coa_signal_strengths, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, run_experiments=run_experiments, new_num_timesteps=num_timesteps, justify_parameters=justify_parameters, remake_graphs=remake_graphs, remake_animation=remake_animation)
    
    if do_final_analysis:
        centroids_persistences_speeds_per_repeat = []
        for rpt_number in range(num_experiment_repeats):
            environment_name = "RPT={}".format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
            
            print(("Analyzing repeat number: ", rpt_number))
            time_unit, centroids_persistences_speeds = cu.analyze_single_cell_motion(relevant_environment, storefile_path, no_randomization)
            
            centroids_persistences_speeds_per_repeat.append(centroids_persistences_speeds)
            # ================================================================
            
        datavis.present_collated_single_cell_motion_data(centroids_persistences_speeds_per_repeat, experiment_dir, total_time_in_hours, time_unit)

    print("Done.")
    return experiment_name
    

# ===========================================================================

def collate_single_cell_test_data(num_experiment_repeats, experiment_dir):
    cell_full_speeds_per_repeat = []
    cell_rac_active_max_conc_per_repeat = []
    cell_rho_active_max_conc_per_repeat = []
    cell_rac_inactive_max_conc_per_repeat = []
    cell_rho_inactive_max_conc_per_repeat = []
    
    for rpt_number in range(num_experiment_repeats):
        environment_name = "RPT={}".format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        
        data_dict_pickle_path = os.path.join(environment_dir, "general_data_dict.pkl")
        data_dict = None
        with open(data_dict_pickle_path, 'rb') as f:
            data_dict = dill.load(f)
            
        if data_dict == None:
            raise Exception("Unable to load data_dict at path: {}".format(data_dict_pickle_path))
            
        cell_full_speeds_per_repeat.append(data_dict["cell_full_speeds"][0])
        cell_rac_active_max_conc_per_repeat.append(data_dict["avg_max_conc_rac_membrane_active_0"])
        cell_rac_inactive_max_conc_per_repeat.append(data_dict["avg_max_conc_rac_membrane_inactive_0"])
        cell_rho_active_max_conc_per_repeat.append(data_dict["avg_max_conc_rho_membrane_active_0"])
        cell_rho_inactive_max_conc_per_repeat.append(data_dict["avg_max_conc_rho_membrane_inactive_0"])

    
    return cell_full_speeds_per_repeat, cell_rac_active_max_conc_per_repeat, cell_rac_inactive_max_conc_per_repeat, cell_rho_active_max_conc_per_repeat, cell_rho_inactive_max_conc_per_repeat

# ===========================================================================

def convergence_test_simple(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_nodes=np.array([]), default_coa=0.0, default_cil=0.0, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, num_experiment_repeats=5, biased_rgtpase_distrib_defn_dict={'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}):
    
    num_tests = len(test_num_nodes)
    
    cell_speeds = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    active_racs = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    active_rhos = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    inactive_racs = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    inactive_rhos = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, nn in enumerate(test_num_nodes):
        print("=========")
        print(("nn = {}".format(nn)))
        
        this_parameter_dict = copy.deepcopy(parameter_dict)
        this_parameter_dict.update([("num_nodes", nn)])
        experiment_name = single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, this_parameter_dict, base_output_dir=base_output_dir, no_randomization=no_randomization, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, show_centroid_trail=True, convergence_test=True, do_final_analysis=do_final_analysis, biased_rgtpase_distrib_defn_dict=biased_rgtpase_distrib_defn_dict)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        cell_full_speeds_per_repeat, cell_rac_active_max_conc_per_repeat, cell_rac_inactive_max_conc_per_repeat, cell_rho_active_max_conc_per_repeat, cell_rho_inactive_max_conc_per_repeat = collate_single_cell_test_data(num_experiment_repeats, experiment_dir)
        
        num_speed_points = cell_full_speeds_per_repeat[0].shape[0]
        cell_speeds[xi] = [np.average(cfs[num_speed_points/2:]) for cfs in cell_full_speeds_per_repeat]
        
        num_rgtpase_points = cell_rac_active_max_conc_per_repeat[0].shape[0]
        active_racs[xi] = [np.average(dats[num_rgtpase_points/2:]) for dats in cell_rac_active_max_conc_per_repeat]
        active_rhos[xi] = [np.average(dats[num_rgtpase_points/2:]) for dats in cell_rho_active_max_conc_per_repeat]
        inactive_racs[xi] = [np.average(dats[num_rgtpase_points/2:]) for dats in cell_rac_inactive_max_conc_per_repeat]
        inactive_rhos[xi] = [np.average(dats[num_rgtpase_points/2:]) for dats in cell_rho_inactive_max_conc_per_repeat]


    print("=========")
    
    datavis.graph_convergence_test_data(sub_experiment_number, test_num_nodes, cell_speeds, active_racs, active_rhos, inactive_racs, inactive_rhos, save_dir=experiment_set_directory)
        
        
    print("Complete.")

# ============================================================================

def Tr_vs_Tp_test(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_Trs=np.array([]), num_experiment_repeats=50, default_coa=0.0, default_cil=0.0, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False):
    
    num_tests = len(test_Trs)
    
    average_cell_persistence_times = np.zeros(num_tests, dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, Tr in enumerate(test_Trs):
        print("=========")
        print(("Tr = {}".format(Tr)))
        
        this_parameter_dict = copy.deepcopy(parameter_dict)
        this_parameter_dict.update([("randomization_time_mean", Tr)])
        experiment_name = single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, this_parameter_dict, base_output_dir=base_output_dir, no_randomization=no_randomization, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, show_centroid_trail=True, convergence_test=False, Tr_vs_Tp_test=True, do_final_analysis=do_final_analysis)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        cell_persistence_ratios_per_repeat, cell_persistence_times_per_repeat, cell_speeds_per_repeat, average_cell_rac_activity_per_repeat, average_cell_rho_activity_per_repeat, average_cell_rac_inactivity_per_repeat, average_cell_rho_inactivity_per_repeat = collate_single_cell_test_data(num_experiment_repeats, experiment_dir)
        
        average_cell_persistence_times[xi] = np.average(cell_persistence_times_per_repeat)


    print("=========")
    
    datavis.graph_Tr_vs_Tp_test_data(sub_experiment_number, test_Trs, average_cell_persistence_times, save_dir=experiment_set_directory)

        
    print("Complete.")
    
# ============================================================================

def two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, corridor_definition=["default"], num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, show_coa_overlay=False, justify_parameters=True):
    global_scale = 4
    
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    experiment_name_format_string = "cil_symm_{}_CIL={}_COA={}".format(sub_experiment_number, default_cil, default_coa) + "_{}"
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 2
    num_cells_in_boxes = [1, 1]
    box_heights = [1*cell_diameter]*num_boxes
    box_widths = [1*cell_diameter]*num_boxes

    x_space_between_boxes = [1*cell_diameter]
    plate_width, plate_height = 20*cell_diameter*1.2, 3*cell_diameter

    #define_group_boxes_and_corridors(corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, x_placement_option, y_placement_option, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None], migratory_bdry_x_offset=None, migratory_bdry_y_offset=None)
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, "CENTRE", "ORIGIN",  migratory_corridor_size=[100*cell_diameter, cell_diameter], migratory_bdry_x_offset=-1*50*cell_diameter, origin_y_offset=25)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": ""}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]), 0.2]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 0.2]}]]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: single cell, no randomization"]
    chemoattractant_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 2
        
    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'])  
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation, run_experiments=run_experiments, justify_parameters=justify_parameters)
    
    if do_final_analysis:
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        datavis.graph_intercellular_distance_after_first_collision(all_cell_centroids_per_repeat, timestep_length*(1./60.0), cell_diameter, save_dir=experiment_dir)
            # ================================================================
        
        #time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22, general_data_structure=None
        time_unit = "min."
        datavis.present_collated_cell_motion_data(time_unit, np.array(all_cell_centroids_per_repeat), np.array(all_cell_persistence_ratios_per_repeat), np.array(all_cell_persistence_times_per_repeat), np.array(all_cell_speeds_per_repeat), all_cell_protrusion_lifetimes_and_directions_per_repeat, np.array(group_centroid_per_timestep_per_repeat), np.array(group_persistence_ratio_per_repeat), np.array(group_persistence_time_per_repeat), experiment_dir, total_time_in_hours)
        
        drift_args = (timestep_length, parameter_dict["init_cell_radius"]*2/1e-6, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, experiment_dir, total_time_in_hours)
        
        datavis.present_collated_group_centroid_drift_data(*drift_args, min_ylim=-1500.0, max_ylim=1500.0)
        

    print("Done.")
    
    
# ============================================================================
def make_no_rgtpase_parameter_dict(parameter_dict):
    no_rgtpase_parameter_dict = copy.deepcopy(parameter_dict)
    no_rgtpase_parameter_dict.update([('C_total', 1.), ('H_total', 1.),('max_force_rac', 1.0)])
    #no_rgtpase_parameter_dict.update([('C_total', 1.), ('H_total', 1.), ('threshold_rac_activity_multiplier', 100.), ('threshold_rho_activity_multiplier', 100.), ('hill_exponent', 3.), ('tension_mediated_rac_inhibition_half_strain', 5.0), ('kgtp_rac_multiplier', 1e-6), ('kgtp_rho_multiplier', 1e-6), ('kdgtp_rac_multiplier', 1e-6), ('kdgtp_rho_multiplier', 1e-6), ('kgtp_rac_autoact_multiplier', 1e-6), ('kgtp_rho_autoact_multiplier', 1e-6), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1e-6), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1e-6), ('tension_mediated_rac_inhibition_half_strain', 5.0),('tension_mediated_rac_inhibition_magnitude', 1e-6), ('max_force_rac', 1e-6),('eta', 2.9*10000.0), ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 12.0), ('randomization_node_percentage', 0.25)])
    
    return no_rgtpase_parameter_dict
    
def collision_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=0.8, run_experiments=True, remake_graphs=False, remake_animation=False, show_coa_overlay=False):
    global_scale = 4
    
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    experiment_name_format_string = "collision_test_{}".format(sub_experiment_number, default_cil, default_coa) + "_{}"
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 2
    num_cells_in_boxes = [1, 1]
    box_heights = [1*cell_diameter]*num_boxes
    box_widths = [1*cell_diameter]*num_boxes

    x_space_between_boxes = [0*cell_diameter]
    plate_width, plate_height = 10*cell_diameter*1.2, 3*cell_diameter

    # plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None], migratory_bdry_x_offset=None, migratory_bdry_y_offset=None
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, "OVERRIDE", "ORIGIN", physical_bdry_polygon_extra=20, box_x_offsets=[10 + 3*cell_diameter, 10 + (3 + 1 + 1)*cell_diameter],  migratory_corridor_size=[10*cell_diameter, migr_bdry_height_factor*cell_diameter], make_only_migratory_corridor=True, origin_y_offset=25)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": ""}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}, {'default': ['unbiased random', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 1.0]}]]
    parameter_dict_per_sub_experiment = [[parameter_dict, make_no_rgtpase_parameter_dict(parameter_dict)]]
    experiment_descriptions_per_subexperiment = ["from experiment template: single cell, no randomization"]
    chemoattractant_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 4
        
    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'])  
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation, run_experiments=run_experiments, justify_parameters=False)

    print("Done.")
    
# ============================================================================

def block_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, dist_from_block=1.0, init_polarization="opposite", num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, remake_graphs=False, remake_animation=False, run_experiments=True, show_coa_overlay=False, justify_parameters=True):

    acceptable_init_polarization_opts = ["opposite", "random", "o", "r"]
    if init_polarization not in acceptable_init_polarization_opts:
        raise Exception("init_polarization not in acceptable list of options: {}. Given: {}.".format(acceptable_init_polarization_opts, init_polarization))
        
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    experiment_name_format_string = "block_coa_test_{}_{}_NC={}_D={}_COA={}_CIL={}".format(sub_experiment_number, "{}", block_cells_width*block_cells_height, dist_from_block, default_coa, default_cil)
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 2
    num_cells_in_boxes = [block_cells_width*block_cells_height, 1]
    box_heights = [block_cells_height*cell_diameter, cell_diameter]
    box_widths = [block_cells_width*cell_diameter, cell_diameter]

    x_space_between_boxes = [dist_from_block*cell_diameter]
    
    min_x_space_required = box_widths[0] + x_space_between_boxes[0] + box_widths[1]
    plate_width = 2*(box_widths[0] + x_space_between_boxes[0] + box_widths[1])
    plate_height = plate_width

    box_x_offsets = [0.5*(plate_width - min_x_space_required), 0.5*(plate_width - min_x_space_required) + min_x_space_required + box_widths[1]]
    box_y_offsets = [plate_height/2. - box_heights[0]/2., plate_height/2. - box_heights[1]/2.]
    #(corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, x_placement_option, y_placement_option, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None], migratory_bdry_x_offset=None, migratory_bdry_y_offset=None)
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon  = define_group_boxes_and_corridors(["default"], plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, "OVERRIDE", "OVERRIDE", box_x_offsets=box_x_offsets, box_y_offsets=box_y_offsets, origin_y_offset=0.0)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": ""}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    if init_polarization in ["r", "random"]:
        biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]*num_boxes]
    elif init_polarization in ["o", "opposite"]:
        biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}]]
        
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    chemoattractant_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        this_pd = copy.deepcopy(parameter_dict_per_sub_experiment[si][bi])
        if bi == 0:
            this_pd.update([("skip_dynamics", True)])
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': this_pd} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 1
        
    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'])  
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation, run_experiments=run_experiments, justify_parameters=justify_parameters)
    
    print("Done.")

# =============================================================================

def many_cells_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, auto_calculate_num_cells=True, num_cells=None, remake_graphs=False, run_experiments=True, remake_animation=False, show_centroid_trail=True, show_rac_random_spikes=False, cell_placement_method="", show_coa_overlay=False, justify_parameters=True):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if auto_calculate_num_cells:
        num_cells = box_height*box_width
    else:
        if num_cells == None:
            raise Exception("Auto-calculation of cell number turned off, but num_cells not given!")
            
    experiment_name_format_string = "coa_test_{}_{}_NC={}_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, default_coa, default_cil)
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [box_height*cell_diameter]
    box_widths = [box_width*cell_diameter]

    x_space_between_boxes = []
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(["default"], plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, "CENTER", "CENTER")
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": cell_placement_method}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    chemoattractant_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 1
    
    cell_dependent_coa_signal_strengths = []
    for cgi, cgd in enumerate(user_cell_group_defns):
        signal_strength = cgd['interaction_factors_coa_per_celltype'][cgi]
        for ci in range(cgd['num_cells']):
            cell_dependent_coa_signal_strengths.append(signal_strength)
    # np.log(parameters_dict['coa_sensing_value_at_dist'])/self.coa_sensing_dist_at_value
    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, cell_dependent_coa_signal_strengths=cell_dependent_coa_signal_strengths, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'], coa_overlay_resolution=1.0)  
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation, run_experiments=run_experiments, justify_parameters=justify_parameters)
    
    experiment_name_format_string = "RPT={}"
    cell_centroids_persistences_speeds_per_repeat = []
    group_centroid_per_timestep_per_repeat = []
    group_centroid_x_per_timestep_per_repeat = []
    min_x_centroid_per_timestep_per_repeat = []
    max_x_centroid_per_timestep_per_repeat = []
    group_speed_per_timestep_per_repeat = []
    group_persistence_ratio_per_repeat = []
    group_persistence_time_per_repeat = []
    cell_persistence_ratios_per_repeat = []
    cell_persistence_times_per_repeat = []
    cell_speeds_per_repeat = []
    cell_centroids_per_repeat = []
    cell_protrusion_lifetimes_and_directions_per_repeat = []
    
    for rpt_number in range(num_experiment_repeats):
        print(("Analyzing repeat {}...".format(rpt_number)))
        environment_name = experiment_name_format_string.format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        storefile_path = eu.get_storefile_path(environment_dir)
        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
        
        time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_centroid_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds   = cu.analyze_cell_motion(relevant_environment, storefile_path, si, rpt_number)
        
        cell_centroids_persistences_speeds_per_repeat.append(centroids_persistences_speeds)
        group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
        group_centroid_x_per_timestep_per_repeat.append(group_centroid_x_per_timestep)
        min_x_centroid_per_timestep_per_repeat.append(min_x_centroid_per_timestep)
        max_x_centroid_per_timestep_per_repeat.append(max_x_centroid_per_timestep)
        group_speed_per_timestep_per_repeat.append(group_speed_per_timestep)
        
        group_persistence_ratio_per_repeat.append(group_persistence_ratio)
        
        group_persistence_time_per_repeat.append(group_persistence_time)
        cell_centroids_per_repeat.append([x[0] for x in centroids_persistences_speeds])
        cell_persistence_ratios_per_repeat.append([x[1][0] for x in centroids_persistences_speeds])
        cell_persistence_times_per_repeat.append([x[1][1] for x in centroids_persistences_speeds])
        cell_speeds_per_repeat.append([x[2] for x in centroids_persistences_speeds])
        cell_protrusion_lifetimes_and_directions_per_repeat.append([x[3] for x in centroids_persistences_speeds])
        # ================================================================
    
    #time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22
    datavis.present_collated_cell_motion_data(time_unit, np.array(cell_centroids_per_repeat), np.array(cell_persistence_ratios_per_repeat), np.array(cell_persistence_times_per_repeat),  np.array(cell_speeds_per_repeat), cell_protrusion_lifetimes_and_directions_per_repeat, np.array(group_centroid_per_timestep_per_repeat), np.array(group_persistence_ratio_per_repeat), np.array(group_persistence_time_per_repeat), experiment_dir, total_time_in_hours)
    
    print("Done.")
    
    return experiment_name

# =============================================================================
def make_linear_gradient_function(source_x, source_y, max_value, slope):
    @nb.jit(nopython=True)
    def f(x):
        d = np.sqrt((x[0] - source_x)**2 + (x[1] - source_y)**2)
        calc_value = max_value*(1 - slope*d)
        
        if calc_value > max_value:
            return max_value
        elif calc_value < 0.0:
            return 0.0
        else:
            return calc_value
            
    return f

def make_normal_gradient_function(source_x, source_y, gaussian_width, gaussian_height):    
    widthsq = gaussian_width*gaussian_width
    
    @nb.jit(nopython=True)
    def f(x):
        dsq = (x[0] - source_x)**2 + (x[1] - source_y)**2
        return gaussian_height*np.exp(-1*dsq/(2*widthsq))
    
    return f

def make_chemoattractant_gradient_function(source_type='', source_x=np.nan, source_y=np.nan, max_value=np.nan, slope=np.nan, gaussian_width=np.nan, gaussian_height=np.nan):
    if source_type == '':
        return lambda x: 0.0
    elif source_type == "normal":
        if not np.any(np.isnan([source_x, source_y, gaussian_width, gaussian_height])):
            return make_normal_gradient_function(source_x, source_y, gaussian_width, gaussian_height)
        else:
            raise Exception("Normal chemoattractant gradient function requested, but definition is not filled out properly!")
    elif source_type == "linear":
        if not np.any(np.isnan([source_x, source_y, max_value, slope])):
            return make_linear_gradient_function(source_x, source_y, max_value, slope)
        else:
            raise Exception("Linear chemoattractant gradient function requested, but definition is not filled out properly!")
# ============================================================================

def collate_final_analysis_data(num_experiment_repeats, experiment_dir):
    all_cell_centroids_per_repeat = []
    all_cell_persistence_ratios_per_repeat = []
    all_cell_persistence_times_per_repeat = []
    all_cell_speeds_per_repeat = []
    all_cell_protrusion_lifetimes_and_directions_per_repeat = []
    group_centroid_per_timestep_per_repeat = []
    group_centroid_x_per_timestep_per_repeat = []
    min_x_centroid_per_timestep_per_repeat = []
    max_x_centroid_per_timestep_per_repeat = []
    group_speed_per_timestep_per_repeat = []
    fit_group_x_velocity_per_repeat = []
    group_persistence_ratio_per_repeat = []
    group_persistence_time_per_repeat = []
    cell_separations_per_repeat = []
    transient_end_times_per_repeat = []
    areal_strains_per_cell_per_repeat = []
    
    for rpt_number in range(num_experiment_repeats):
        environment_name = "RPT={}".format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        
        data_dict_pickle_path = os.path.join(environment_dir, "general_data_dict.pkl")
        data_dict = None
        with open(data_dict_pickle_path, 'rb') as f:
            data_dict = dill.load(f)
            
        if data_dict == None:
            raise Exception("Unable to load data_dict at path: {}".format(data_dict_pickle_path))
        
        all_cell_centroids_per_tstep = data_dict["all_cell_centroids_per_tstep"]
        all_cell_centroids_per_repeat.append(all_cell_centroids_per_tstep)
        all_cell_persistence_ratios_per_repeat.append(data_dict["all_cell_persistence_ratios"])
        all_cell_persistence_times_per_repeat.append(data_dict["all_cell_persistence_times"])
        all_cell_speeds_per_repeat.append(data_dict["all_cell_speeds"])
        all_cell_protrusion_lifetimes_and_directions_per_repeat.append(data_dict["all_cell_protrusion_lifetimes_and_directions"])
        #centroids_persistences_speeds_protrusionlifetimes_per_repeat.append(centroids_persistences_speeds_protrusionlifetimes)
        group_centroid_per_timestep = data_dict["group_centroid_per_tstep"]
        group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
        group_centroid_x_per_timestep_per_repeat.append(group_centroid_per_timestep[:, 0])
        min_x_centroid_per_timestep_per_repeat.append(np.min(all_cell_centroids_per_tstep[:, :, 0], axis=0))
        max_x_centroid_per_timestep_per_repeat.append(np.max(all_cell_centroids_per_tstep[:, :, 0], axis=0))
        group_speed_per_timestep_per_repeat.append(data_dict["group_speeds"])
        fit_group_x_velocity_per_repeat.append(data_dict["fit_group_x_velocity"])
        
        group_persistence_ratio_per_repeat.append(data_dict["group_persistence_ratio"])
        
        group_persistence_time_per_repeat.append(data_dict["group_persistence_time"])
        cell_separations_per_repeat.append(data_dict["cell_separation_mean"])
        transient_end_times_per_repeat.append(data_dict["transient_end"])
        #areal_strains_per_cell_per_repeat.append(data_dict["all_cell_areal_strains"])
        
    
    return all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat

# ============================================================================

def collate_corridor_convergence_data(num_experiment_repeats, experiment_dir):
    all_cell_centroids_per_repeat = []
    all_cell_persistence_ratios_per_repeat = []
    all_cell_persistence_times_per_repeat = []
    all_cell_speeds_per_repeat = []
    all_cell_protrusion_lifetimes_and_directions_per_repeat = []
    group_centroid_per_timestep_per_repeat = []
    group_centroid_x_per_timestep_per_repeat = []
    min_x_centroid_per_timestep_per_repeat = []
    max_x_centroid_per_timestep_per_repeat = []
    group_speed_per_timestep_per_repeat = []
    fit_group_x_velocity_per_repeat = []
    group_persistence_ratio_per_repeat = []
    group_persistence_time_per_repeat = []
    cell_separations_per_repeat = []
    cell_full_speeds_per_repeat = []
    cell_rac_active_max_conc_per_repeat = []
    cell_rho_active_max_conc_per_repeat = []
    cell_rac_inactive_max_conc_per_repeat = []
    cell_rho_inactive_max_conc_per_repeat = []
    
    for rpt_number in range(num_experiment_repeats):
        environment_name = "RPT={}".format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        
        data_dict_pickle_path = os.path.join(environment_dir, "general_data_dict.pkl")
        data_dict = None
        with open(data_dict_pickle_path, 'rb') as f:
            data_dict = dill.load(f)
            
        if data_dict == None:
            raise Exception("Unable to load data_dict at path: {}".format(data_dict_pickle_path))
        
        all_cell_centroids_per_tstep = data_dict["all_cell_centroids_per_tstep"]
        all_cell_centroids_per_repeat.append(all_cell_centroids_per_tstep)
        all_cell_persistence_ratios_per_repeat.append(data_dict["all_cell_persistence_ratios"])
        all_cell_persistence_times_per_repeat.append(data_dict["all_cell_persistence_times"])
        all_cell_speeds_per_repeat.append(data_dict["all_cell_speeds"])
        all_cell_protrusion_lifetimes_and_directions_per_repeat.append(data_dict["all_cell_protrusion_lifetimes_and_directions"])
        #centroids_persistences_speeds_protrusionlifetimes_per_repeat.append(centroids_persistences_speeds_protrusionlifetimes)
        group_centroid_per_timestep = data_dict["group_centroid_per_tstep"]
        group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
        group_centroid_x_per_timestep_per_repeat.append(group_centroid_per_timestep[:, 0])
        min_x_centroid_per_timestep_per_repeat.append(np.min(all_cell_centroids_per_tstep[:, :, 0], axis=1))
        max_x_centroid_per_timestep_per_repeat.append(np.max(all_cell_centroids_per_tstep[:, :, 0], axis=1))
        group_speed_per_timestep_per_repeat.append(data_dict["group_speeds"])
        fit_group_x_velocity_per_repeat.append(data_dict["fit_group_x_velocity"])
        
        group_persistence_ratio_per_repeat.append(data_dict["group_persistence_ratio"])
        
        group_persistence_time_per_repeat.append(data_dict["group_persistence_time"])
        cell_separations_per_repeat.append(data_dict["cell_separation_mean"])
        
        cell_full_speeds_per_repeat.append(data_dict["cell_full_speeds"])
        cell_rac_active_max_conc_per_repeat.append([data_dict["avg_max_conc_rac_membrane_active_{}".format(k)] for k in range(2)])
        cell_rac_inactive_max_conc_per_repeat.append([data_dict["avg_max_conc_rac_membrane_inactive_{}".format(k)] for k in range(2)])
        cell_rho_active_max_conc_per_repeat.append([data_dict["avg_max_conc_rho_membrane_active_{}".format(k)] for k in range(2)])
        cell_rho_inactive_max_conc_per_repeat.append([data_dict["avg_max_conc_rho_membrane_inactive_{}".format(k)] for k in range(2)])
        
    
    return all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, cell_full_speeds_per_repeat, cell_rac_active_max_conc_per_repeat, cell_rac_inactive_max_conc_per_repeat, cell_rho_active_max_conc_per_repeat, cell_rho_inactive_max_conc_per_repeat

# ============================================================================

def make_chemoattractant_source_info_tag_for_experiment_name(chemoattractant_source_definition):
    if len(chemoattractant_source_definition.keys()) == 0:
        return ''
    else:
        if chemoattractant_source_definition['source_type'] == "linear":
            return "-CS=L-{}-{}-{}".format(chemoattractant_source_definition["x_offset_in_corridor"], chemoattractant_source_definition["max_value"], chemoattractant_source_definition["slope"])
        
        elif chemoattractant_source_definition['source_type'] == 'normal':
            return "-CS=N-{}-{}-{}".format(chemoattractant_source_definition["x_offset_in_corridor"], chemoattractant_source_definition["gaussian_width"], chemoattractant_source_definition["gaussian_height"])
        else:
            return "-CS=ERROR"
      
# ============================================================================
            
def corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, corridor_height=None, box_width=4, box_height=4, box_y_placement_factor=0.0, cell_placement_method="", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25, box_x_offset=0, num_cells=0, corridor_definition=["default"], run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, convergence_test=False, biased_rgtpase_distrib_defn_dict={'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}, graph_group_centroid_splits=False, max_animation_corridor_length=None, global_scale=1, show_coa_overlay=False, coa_overlay_resolution=10, justify_parameters=True, colorscheme="normal", specific_timesteps_to_draw_as_svg=[], chemoattractant_source_definition=[]):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if num_cells == 0:
        raise Exception("No cells!")
    
    if corridor_height == None:
        corridor_height = box_height
        
    if corridor_height < box_height:
        raise Exception("Corridor height is less than box height!")
        
    if corridor_height == box_height:
        box_y_placement_factor = 0.0
        
    accepted_cell_placement_methods = ["", "r"]
    if cell_placement_method not in accepted_cell_placement_methods:
        raise Exception("Unknown placement method given: {}, expected one of {}".format(cell_placement_method, accepted_cell_placement_methods))
    
    if convergence_test == True:
        experiment_name_format_string = "corr_conv_{}_{}_NN={}_CIL={}_COA={}".format(sub_experiment_number, "{}", parameter_dict['num_nodes'], np.round(default_cil, decimals=3), np.round(default_coa, decimals=3))
    else:
        if cell_placement_method == "":
            experiment_name_format_string = "cm_{}_{}_NC=({}, {}, {}, {}, {}){}_COA={}_CIL={}{}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, corridor_height, box_y_placement_factor, cell_placement_method, np.round(default_coa, decimals=3), np.round(default_cil, decimals=3), make_chemoattractant_source_info_tag_for_experiment_name(chemoattractant_source_definition))
        else:
            experiment_name_format_string = "cm_{}_{}_NC=({}, {}, {}, {}, {})({}, {}, {})_COA={}_CIL={}{}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, corridor_height, box_y_placement_factor, cell_placement_method, max_placement_distance_factor, init_random_cell_placement_x_factor, np.round(default_coa, decimals=3), np.round(default_cil, decimals=3), make_chemoattractant_source_info_tag_for_experiment_name(chemoattractant_source_definition))
        
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [box_height*cell_diameter]
    box_widths = [box_width*cell_diameter]

    x_space_between_boxes = []

    if not convergence_test:
        if max_animation_corridor_length == None:
            plate_width, plate_height = min(2000, max(1000, box_widths[0]*8)), (corridor_height*cell_diameter + 40 + 100)
        else:
            plate_width, plate_height = max_animation_corridor_length, (corridor_height*cell_diameter + 40 + 100)
    else:
        plate_width, plate_height = 600, (corridor_height*cell_diameter + 40 + 100)

    origin_y_offset = 55
    physical_bdry_polygon_extra = 20
    
    initial_x_placement_options = "ORIGIN"
    initial_y_placement_options = "OVERRIDE"
    
    box_y_offsets = [box_y_placement_factor*(corridor_height - box_height)*cell_diameter + origin_y_offset]

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon  = define_group_boxes_and_corridors(corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, box_y_offsets=box_y_offsets, physical_bdry_polygon_extra=physical_bdry_polygon_extra, origin_y_offset=origin_y_offset, migratory_corridor_size=[box_widths[0]*100, corridor_height*cell_diameter], make_only_migratory_corridor=convergence_test)
    
    if corridor_definition[0] == "regular curve":
        migr_poly_xs = space_migratory_bdry_polygon[:,0]
        #migr_poly_ys = space_migratory_bdry_polygon[:,1]
        curve_radius = corridor_definition[2]
        plate_width = (np.max(migr_poly_xs/1e-6) - np.min(migr_poly_xs/1e-6))*1.1
        if max_animation_corridor_length == None:
            min_height = corridor_height + 0.5*corridor_height + curve_radius
            plate_height = (min_height + (750 - min_height))*1.1
        else:
            min_height = corridor_height + 0.5*corridor_height + curve_radius
            plate_height = (min_height + (max_animation_corridor_length - min_height))*1.1
        
        if corridor_definition[-1] == -1:
            origin_y_offset = plate_height - origin_y_offset - corridor_height*cell_diameter
            box_y_offsets = [box_y_placement_factor*(corridor_height - box_height)*cell_diameter + origin_y_offset]

            boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon  = define_group_boxes_and_corridors(corridor_definition, plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, box_y_offsets=box_y_offsets, physical_bdry_polygon_extra=physical_bdry_polygon_extra, origin_y_offset=origin_y_offset, migratory_corridor_size=[box_widths[0]*100, corridor_height*cell_diameter], make_only_migratory_corridor=convergence_test)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": cell_placement_method, "max_placement_distance_factor": max_placement_distance_factor, "init_random_cell_placement_x_factor": init_random_cell_placement_x_factor, "convergence_test": convergence_test, "graph_group_centroid_splits": graph_group_centroid_splits}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[biased_rgtpase_distrib_defn_dict]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]

    chemoattractant_source_location = np.array([])
    if 'x_offset_in_corridor' in chemoattractant_source_definition.keys():
        chemoattractant_source_definition['source_x'] = box_x_offsets[0] + chemoattractant_source_definition['x_offset_in_corridor']
        del chemoattractant_source_definition['x_offset_in_corridor']
        chemoattractant_source_definition['source_y'] = box_y_offsets[0] + corridor_height*cell_diameter*0.5
        chemoattractant_source_location = np.array([chemoattractant_source_definition['source_x'], chemoattractant_source_definition['source_y']])
    
    chemoattractant_gradient_fn_per_subexperiment = [make_chemoattractant_gradient_function(**chemoattractant_source_definition)]
        
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    if not convergence_test:
        global_scale = global_scale
    else:
        global_scale = 2
    
    cell_dependent_coa_signal_strengths = []
    for cgi, cgd in enumerate(user_cell_group_defns):
        signal_strength = cgd['interaction_factors_coa_per_celltype'][cgi]
        for ci in range(cgd['num_cells']):
            cell_dependent_coa_signal_strengths.append(signal_strength)
            
    if colorscheme == "normal":
        animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'], coa_overlay_resolution=coa_overlay_resolution, cell_dependent_coa_signal_strengths=cell_dependent_coa_signal_strengths, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, specific_timesteps_to_draw_as_svg=specific_timesteps_to_draw_as_svg, chemoattractant_source_location=chemoattractant_source_location) 
    elif colorscheme == "scifi":
        animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_centroid_trail=True, show_rac_random_spikes=False, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'], background_color=colors.RGB_BLACK, chemoattractant_dot_color=colors.RGB_DARK_GREEN, default_cell_polygon_edge_and_vertex_color=colors.RGB_BLACK, default_cell_polygon_fill_color=colors.RGB_CYAN, rgtpase_colors=[colors.RGB_BRIGHT_BLUE, colors.RGB_LIGHT_BLUE, colors.RGB_BRIGHT_RED, colors.RGB_LIGHT_RED], velocity_colors=[colors.RGB_ORANGE, colors.RGB_LIGHT_GREEN, colors.RGB_LIGHT_GREEN, colors.RGB_CYAN, colors.RGB_MAGENTA], coa_color=colors.RGB_DARK_GREEN, font_color=colors.RGB_BLACK, coa_overlay_color=colors.RGB_CYAN, rgtpase_background_shine_color=colors.RGB_WHITE, coa_overlay_resolution=coa_overlay_resolution, cell_dependent_coa_signal_strengths=cell_dependent_coa_signal_strengths, show_rgtpase=False, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, specific_timesteps_to_draw_as_svg=specific_timesteps_to_draw_as_svg, chemoattractant_source_location=chemoattractant_source_location) 
    else:
        raise Exception("Unknown colorscheme given: {}. Expected one of [{}, {}]".format(colorscheme, "normal", "scifi"))
    
    if corridor_definition[0] == "obstacle":
        animation_settings.update([("show_physical_bdry_polygon", True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, run_experiments=run_experiments, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation, justify_parameters=justify_parameters)
    
    drift_args = None
    if do_final_analysis:
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
            # ================================================================
        
        #time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22, general_data_structure=None
        time_unit = "min."
        datavis.present_collated_cell_motion_data(time_unit, np.array(all_cell_centroids_per_repeat), np.array(all_cell_persistence_ratios_per_repeat), np.array(all_cell_persistence_times_per_repeat), np.array(all_cell_speeds_per_repeat), all_cell_protrusion_lifetimes_and_directions_per_repeat, np.array(group_centroid_per_timestep_per_repeat), np.array(group_persistence_ratio_per_repeat), np.array(group_persistence_time_per_repeat), experiment_dir, total_time_in_hours)
        
        drift_args = (timestep_length, parameter_dict["init_cell_radius"]*2/1e-6, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, experiment_dir, total_time_in_hours)
        
        datavis.present_collated_group_centroid_drift_data(*drift_args)
        

    print("Done.")
    
    return experiment_name, drift_args

# ============================================================================
    
def no_corridor_chemoattraction_test(date_str, experiment_number, sub_experiment_number, parameter_dict, chemoattractant_source_definition, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, box_y_placement_factor=0.0, cell_placement_method="", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25, box_x_offset=0, num_cells=0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, biased_rgtpase_distrib_defn_dict={'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}, graph_group_centroid_splits=False, max_animation_corridor_length=None, global_scale=1, show_coa_overlay=False, coa_overlay_resolution=10, justify_parameters=True, colorscheme="normal", specific_timesteps_to_draw_as_svg=[], chemotaxis_target_radius=-1.0, show_centroid_trail=False):
    
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if num_cells == 0:
        raise Exception("No cells!")
        
    accepted_cell_placement_methods = ["", "r"]
    if cell_placement_method not in accepted_cell_placement_methods:
        raise Exception("Unknown placement method given: {}, expected one of {}".format(cell_placement_method, accepted_cell_placement_methods))
        
    if cell_placement_method == "":
        experiment_name_format_string = "ch_{}_{}_NC=({}, {}, {}, {}){}_COA={}_CIL={}{}_S={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, box_y_placement_factor, cell_placement_method, np.round(default_coa, decimals=3), np.round(default_cil, decimals=3), make_chemoattractant_source_info_tag_for_experiment_name(chemoattractant_source_definition), seed)
    else:
        experiment_name_format_string = "ch_{}_{}_NC=({}, {}, {}, {})({}, {}, {})_COA={}_CIL={}{}_S={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, box_y_placement_factor, cell_placement_method, max_placement_distance_factor, init_random_cell_placement_x_factor, np.round(default_coa, decimals=3), np.round(default_cil, decimals=3), make_chemoattractant_source_info_tag_for_experiment_name(chemoattractant_source_definition), seed)
        
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [box_height*cell_diameter]
    box_widths = [box_width*cell_diameter]

    x_space_between_boxes = []

    if 'x_offset_in_corridor':
        source_x_location = chemoattractant_source_definition["x_offset_in_corridor"]
        plate_width = max(2.2*source_x_location, 5*box_width)
        initial_x_placement_options = "OVERRIDE"
        box_x_offsets = [1.1*source_x_location - 0.5*box_width]
    else:
        plate_width = 5*box_width
        initial_x_placement_options = "ORIGIN"
        box_x_offsets = []
        
    plate_height = plate_width

    origin_y_offset = 55
    initial_y_placement_options = "CENTER"

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors([], plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, origin_y_offset=origin_y_offset, box_x_offsets=box_x_offsets)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": cell_placement_method, "max_placement_distance_factor": max_placement_distance_factor, "init_random_cell_placement_x_factor": init_random_cell_placement_x_factor, "convergence_test": False, "graph_group_centroid_splits": graph_group_centroid_splits}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[biased_rgtpase_distrib_defn_dict]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]

    chemoattractant_source_location = np.array([])
    if 'x_offset_in_corridor' in chemoattractant_source_definition.keys():
        chemoattractant_source_definition['source_x'] = box_x_offsets[0] + chemoattractant_source_definition['x_offset_in_corridor']
        del chemoattractant_source_definition['x_offset_in_corridor']
        chemoattractant_source_definition['source_y'] = box_y_offsets[0] + box_height*cell_diameter*0.5
        chemoattractant_source_location = np.array([chemoattractant_source_definition['source_x'], chemoattractant_source_definition['source_y']])
    
    chemoattractant_gradient_fn_per_subexperiment = [make_chemoattractant_gradient_function(**chemoattractant_source_definition)]
        
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)
    
    cell_dependent_coa_signal_strengths = []
    for cgi, cgd in enumerate(user_cell_group_defns):
        signal_strength = cgd['interaction_factors_coa_per_celltype'][cgi]
        for ci in range(cgd['num_cells']):
            cell_dependent_coa_signal_strengths.append(signal_strength)
            
    if colorscheme == "normal":
        animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'], coa_overlay_resolution=coa_overlay_resolution, cell_dependent_coa_signal_strengths=cell_dependent_coa_signal_strengths, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, specific_timesteps_to_draw_as_svg=specific_timesteps_to_draw_as_svg, chemoattractant_source_location=chemoattractant_source_location, chemotaxis_target_radius=chemotaxis_target_radius, show_centroid_trail=show_centroid_trail) 
    elif colorscheme == "scifi":
        animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_centroid_trail=True, show_rac_random_spikes=False, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'], background_color=colors.RGB_BLACK, chemoattractant_dot_color=colors.RGB_DARK_GREEN, default_cell_polygon_edge_and_vertex_color=colors.RGB_BLACK, default_cell_polygon_fill_color=colors.RGB_CYAN, rgtpase_colors=[colors.RGB_BRIGHT_BLUE, colors.RGB_LIGHT_BLUE, colors.RGB_BRIGHT_RED, colors.RGB_LIGHT_RED], velocity_colors=[colors.RGB_ORANGE, colors.RGB_LIGHT_GREEN, colors.RGB_LIGHT_GREEN, colors.RGB_CYAN, colors.RGB_MAGENTA], coa_color=colors.RGB_DARK_GREEN, font_color=colors.RGB_BLACK, coa_overlay_color=colors.RGB_CYAN, rgtpase_background_shine_color=colors.RGB_WHITE, coa_overlay_resolution=coa_overlay_resolution, cell_dependent_coa_signal_strengths=cell_dependent_coa_signal_strengths, show_rgtpase=False, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, specific_timesteps_to_draw_as_svg=specific_timesteps_to_draw_as_svg, chemoattractant_source_location=chemoattractant_source_location, chemotaxis_target_radius=chemotaxis_target_radius) 
    else:
        raise Exception("Unknown colorscheme given: {}. Expected one of [{}, {}]".format(colorscheme, "normal", "scifi"))
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, run_experiments=run_experiments, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation, justify_parameters=justify_parameters)
    
    drift_args = None
    if do_final_analysis:
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
            # ================================================================
        
        #time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22, general_data_structure=None
        time_unit = "min."
        datavis.present_collated_cell_motion_data(time_unit, np.array(all_cell_centroids_per_repeat), np.array(all_cell_persistence_ratios_per_repeat), np.array(all_cell_persistence_times_per_repeat), np.array(all_cell_speeds_per_repeat), all_cell_protrusion_lifetimes_and_directions_per_repeat, np.array(group_centroid_per_timestep_per_repeat), np.array(group_persistence_ratio_per_repeat), np.array(group_persistence_time_per_repeat), experiment_dir, total_time_in_hours, chemoattraction_source_coords=chemoattractant_source_location)
        
        drift_args = (timestep_length, parameter_dict["init_cell_radius"]*2/1e-6, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, experiment_dir, total_time_in_hours)
        
        datavis.present_collated_group_centroid_drift_data(*drift_args)
        

    print("Done.")
    
    return experiment_name, drift_args, environment_wide_variable_defns, chemoattractant_source_definition['source_x'], chemoattractant_source_definition['source_y']


# ============================================================================

def corridor_migration_symmetric_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, corridor_height=None, box_width=4, box_height=4, box_y_placement_factor=0.0, cell_placement_method="", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25, box_x_offset=0, num_cells=0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, biased_rgtpase_distrib_defn_dict={'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}, graph_group_centroid_splits=False, max_animation_corridor_length=None, show_coa_overlay=False, global_scale=1, chemoattractant_source_definition=None):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if num_cells == 0:
        raise Exception("No cells!")
    
    if corridor_height == None:
        corridor_height = box_height
        
    if corridor_height < box_height:
        raise Exception("Corridor height is less than box height!")
        
    if corridor_height == box_height:
        box_y_placement_factor = 0.0
        
    accepted_cell_placement_methods = ["", "r"]
    if cell_placement_method not in accepted_cell_placement_methods:
        raise Exception("Unknown placement method given: {}, expected one of {}".format(cell_placement_method, accepted_cell_placement_methods))
    
    convergence_test = False
    
    if cell_placement_method == "":
        experiment_name_format_string = "cmsym_{}_{}_NC=({}, {}, {}, {}, {}){}_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, corridor_height, box_y_placement_factor, cell_placement_method, np.round(default_coa, decimals=3), np.round(default_cil, decimals=3))
    else:
        experiment_name_format_string = "cmsym_{}_{}_NC=({}, {}, {}, {}, {})({}, {}, {})_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, corridor_height, box_y_placement_factor, cell_placement_method, max_placement_distance_factor, init_random_cell_placement_x_factor, np.round(default_coa, decimals=3), np.round(default_cil, decimals=3))
        
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [box_height*cell_diameter]
    box_widths = [box_width*cell_diameter]

    x_space_between_boxes = []

    if max_animation_corridor_length == None:
        plate_width, plate_height = min(2000, max(1000, box_widths[0]*8)), (corridor_height*cell_diameter + 40 + 100)
    else:
        plate_width, plate_height = max_animation_corridor_length, (corridor_height*cell_diameter + 40 + 100)

    origin_y_offset = 55
    physical_bdry_polygon_extra = 20
    
    initial_x_placement_options = "CENTER"
    initial_y_placement_options = "OVERRIDE"
    
    box_y_offsets = [box_y_placement_factor*(corridor_height - box_height)*cell_diameter + origin_y_offset]

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon  = define_group_boxes_and_corridors(["default"], plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, box_y_offsets=box_y_offsets, physical_bdry_polygon_extra=physical_bdry_polygon_extra, origin_y_offset=origin_y_offset, migratory_corridor_size=[box_widths[0]*100, corridor_height*cell_diameter], make_only_migratory_corridor=convergence_test, migratory_bdry_x_offset=-1*0.5*box_widths[0]*100)

    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": cell_placement_method, "max_placement_distance_factor": max_placement_distance_factor, "init_random_cell_placement_x_factor": init_random_cell_placement_x_factor, "convergence_test": convergence_test, "graph_group_centroid_splits": graph_group_centroid_splits}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[biased_rgtpase_distrib_defn_dict]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    
    chemoattractant_gradient_fn_per_subexperiment = [make_chemoattractant_gradient_function(chemoattractant_source_definition)]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)


    global_scale = 2
        
    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), rgtpase_scale_factor=0.75*np.sqrt(global_scale)*312.5, coa_intersection_exponent=parameter_dict['coa_intersection_exponent'])  
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, run_experiments=run_experiments, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)
    
    drift_args = None
    if do_final_analysis:
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
            # ================================================================
        
        #time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22, general_data_structure=None
        time_unit = "min."
        datavis.present_collated_cell_motion_data(time_unit, np.array(all_cell_centroids_per_repeat), np.array(all_cell_persistence_ratios_per_repeat), np.array(all_cell_persistence_times_per_repeat), np.array(all_cell_speeds_per_repeat), all_cell_protrusion_lifetimes_and_directions_per_repeat, np.array(group_centroid_per_timestep_per_repeat), np.array(group_persistence_ratio_per_repeat), np.array(group_persistence_time_per_repeat), experiment_dir, total_time_in_hours)
        
        drift_args = (timestep_length, parameter_dict["init_cell_radius"]*2/1e-6, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, experiment_dir, total_time_in_hours)
        
        datavis.present_collated_group_centroid_drift_data(*drift_args, min_ylim=-1500., max_ylim=1500.)
        

    print("Done.")
    
    return experiment_name, drift_args

# =============================================================================

def chemotaxis_threshold_test_magnitudes(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_x_offset_in_corridor=625.0, test_chemo_magnitudes=[], test_chemo_slope=0.0016, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, default_coa=0.0, default_cil=60.0, chemotaxis_target_radius=160.0, box_width=1, box_height=1, box_y_placement_factor=0.5, num_cells=1):
    
    test_chemo_magnitudes = sorted(test_chemo_magnitudes)
    
    chemotaxis_success_ratios = np.zeros(len(test_chemo_magnitudes), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    closest_to_source_per_repeat_per_magslope = np.zeros((len(test_chemo_magnitudes), num_experiment_repeats), dtype=np.float64)
    protrusion_lifetimes_and_directions_per_magslope = []
    
    for xi, chm in enumerate(test_chemo_magnitudes):
        print("=========")
        print("mag: {}".format(chm))
        experiment_name, drift_args, environment_wide_variable_defns, source_x, source_y = no_corridor_chemoattraction_test(date_str, experiment_number, sub_experiment_number, parameter_dict, chemoattractant_source_definition={'source_type': 'linear', 'x_offset_in_corridor': test_x_offset_in_corridor, 'max_value': chm, 'slope': test_chemo_slope}, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, box_y_placement_factor=box_y_placement_factor, num_cells=num_cells, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=True, chemotaxis_target_radius=chemotaxis_target_radius, show_centroid_trail=False)
        
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        experiment_name_format_string = "RPT={}"
        
        if run_experiments == False:
            if not os.path.exists(experiment_dir):
                raise Exception("Experiment directory does not exist.")
            else:
                for rpt_number in range(num_experiment_repeats):
                    environment_name = experiment_name_format_string.format(rpt_number)
                    environment_dir = os.path.join(experiment_dir, environment_name)
                    if not os.path.exists(environment_dir):
                        raise Exception("Environment directory does not exist.")
                
                    storefile_path = eu.get_storefile_path(environment_dir)
                    if not os.path.isfile(storefile_path):
                        raise Exception("Storefile does not exist.")
                    
                    relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                    if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                        raise Exception("Simulation is not complete.")
            
                print("Data exists.")
        
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)

        protrusion_lifetimes_and_directions = []
        for protrusion_lifetime_dirn_per_cell in all_cell_protrusion_lifetimes_and_directions_per_repeat:
            for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                for l, d in protrusion_lifetime_dirn:
                    protrusion_lifetimes_and_directions.append((l, d))

        datavis.graph_protrusion_lifetimes_radially(protrusion_lifetimes_and_directions, 12,
                                            save_dir=experiment_dir, save_name="all_cells_protrusion_life_dir")
        
        chemotaxis_success_per_repeat = []
        closest_to_source_per_repeat = []
        for rpt_number in range(num_experiment_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            #empty_env_pickle_path, produce_intermediate_visuals, produce_final_visuals, environment_wide_variable_defns, simulation_execution_enabled=False
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
            
            chemotaxis_success, closest_to_source  = cu.analyze_chemotaxis_success(relevant_environment, storefile_path, rpt_number, source_x, source_y, chemotaxis_target_radius)
            
            chemotaxis_success_per_repeat.append(chemotaxis_success)
            closest_to_source_per_repeat.append(closest_to_source)

        success_protrusion_lifetimes_and_directions = []
        fail_protrusion_lifetimes_and_directions = []
        for i, protrusion_lifetime_dirn_per_cell in enumerate(all_cell_protrusion_lifetimes_and_directions_per_repeat):
            if chemotaxis_success_per_repeat[i] == 1:
                for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                    for l, d in protrusion_lifetime_dirn:
                        success_protrusion_lifetimes_and_directions.append((l, d))
            else:
                for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                    for l, d in protrusion_lifetime_dirn:
                        fail_protrusion_lifetimes_and_directions.append((l, d))

        datavis.graph_protrusion_lifetimes_radially(success_protrusion_lifetimes_and_directions, 12,
                                                    save_dir=experiment_dir, save_name="successful_cells_protrusion_lifetime_dirn_N={}".format(np.sum(chemotaxis_success_per_repeat)))

        datavis.graph_protrusion_lifetimes_radially(fail_protrusion_lifetimes_and_directions, 12,
                                                    save_dir=experiment_dir,
                                                    save_name="fail_cells_protrusion_lifetime_dirn_N={}".format(num_experiment_repeats - np.sum(chemotaxis_success_per_repeat)))
            
        chemotaxis_success_ratios[xi] = np.sum(chemotaxis_success_per_repeat)/num_experiment_repeats
        closest_to_source_per_repeat_per_magslope[xi] = closest_to_source_per_repeat

    print("=========")
    # sub_experiment_number, test_chemo_magnitudes, test_chemo_slope, chemotaxis_success_ratios, box_width, box_height, num_cells, save_dir=None, fontsize=22
    datavis.graph_chemotaxis_efficiency_data(sub_experiment_number, test_chemo_magnitudes, [test_chemo_slope]*len(test_chemo_magnitudes), chemotaxis_success_ratios, box_width, box_height, num_cells, save_dir=experiment_set_directory)
    datavis.graph_chemotaxis_efficiency_data_using_violins(sub_experiment_number, test_chemo_magnitudes, [test_chemo_slope]*len(test_chemo_magnitudes), closest_to_source_per_repeat_per_magslope, box_width, box_height, num_cells, save_dir=experiment_set_directory)
    #datavis.graph_chemotaxis_protrusion_lifetimes(sub_experiment_number, test_chemo_magnitudes, [test_chemo_slope]*len(test_chemo_magnitudes), protrusion_lifetimes_and_directions_per_magslope, box_width, box_height, num_cells, save_dir=experiment_set_directory)
    
    print("Complete.")
    
# =============================================================================

def chemotaxis_threshold_test_slopes(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_x_offset_in_corridor=625.0, test_chemo_slopes=[], num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, default_coa=0.0, default_cil=60.0, chemotaxis_target_radius=160.0, box_width=1, box_height=1, box_y_placement_factor=0.5, num_cells=1, halfmax_dist=625.0):
    
    test_chemo_slopes = sorted(test_chemo_slopes)
    
    chemotaxis_success_ratios = np.zeros(len(test_chemo_slopes), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    closest_to_source_per_repeat_per_magslope = np.zeros((len(test_chemo_slopes), num_experiment_repeats), dtype=np.float64)
    protrusion_lifetimes_per_magslope = []
    required_magnitudes = [slope*halfmax_dist*2 for slope in test_chemo_slopes]
    
    for xi, chs in enumerate(test_chemo_slopes):
        print("=========")
        print("slope: {}".format(chs))
        experiment_name, drift_args, environment_wide_variable_defns, source_x, source_y = no_corridor_chemoattraction_test(date_str, experiment_number, sub_experiment_number, parameter_dict, chemoattractant_source_definition={'source_type': 'linear', 'x_offset_in_corridor': test_x_offset_in_corridor, 'max_value': required_magnitudes[xi], 'slope': chs}, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, box_y_placement_factor=box_y_placement_factor, num_cells=num_cells, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=True, chemotaxis_target_radius=chemotaxis_target_radius, show_centroid_trail=False)
        
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        experiment_name_format_string = "RPT={}"
        
        if run_experiments == False:
            if not os.path.exists(experiment_dir):
                raise Exception("Experiment directory does not exist.")
            else:
                for rpt_number in range(num_experiment_repeats):
                    environment_name = experiment_name_format_string.format(rpt_number)
                    environment_dir = os.path.join(experiment_dir, environment_name)
                    if not os.path.exists(environment_dir):
                        raise Exception("Environment directory does not exist.")
                
                    storefile_path = eu.get_storefile_path(environment_dir)
                    if not os.path.isfile(storefile_path):
                        raise Exception("Storefile does not exist.")
                    
                    relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                    if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                        raise Exception("Simulation is not complete.")
            
                print("Data exists.")
                
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        protrusion_lifetimes = []
        for protrusion_lifetime_dirn_per_cell in all_cell_protrusion_lifetimes_and_directions_per_repeat:
            for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                for l, d in protrusion_lifetime_dirn:
                    protrusion_lifetimes.append(l/60.0)
        
        protrusion_lifetimes_per_magslope.append(protrusion_lifetimes)
        
        chemotaxis_success_per_repeat = []
        closest_to_source_per_repeat = []
        for rpt_number in range(num_experiment_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            #empty_env_pickle_path, produce_intermediate_visuals, produce_final_visuals, environment_wide_variable_defns, simulation_execution_enabled=False
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
            
            chemotaxis_success, closest_to_source  = cu.analyze_chemotaxis_success(relevant_environment, storefile_path, rpt_number, source_x, source_y, chemotaxis_target_radius)
            
            chemotaxis_success_per_repeat.append(chemotaxis_success)
            closest_to_source_per_repeat.append(closest_to_source)
            
        chemotaxis_success_ratios[xi] = np.sum(chemotaxis_success_per_repeat)/num_experiment_repeats
        closest_to_source_per_repeat_per_magslope[xi] = closest_to_source_per_repeat

    print("=========")
    # sub_experiment_number, test_chemo_magnitudes, test_chemo_slope, chemotaxis_success_ratios, box_width, box_height, num_cells, save_dir=None, fontsize=22
    datavis.graph_chemotaxis_efficiency_data(sub_experiment_number, test_chemo_slopes, required_magnitudes, chemotaxis_success_ratios, box_width, box_height, num_cells, save_dir=experiment_set_directory)
    datavis.graph_chemotaxis_efficiency_data_using_violins(sub_experiment_number, required_magnitudes, test_chemo_slopes, closest_to_source_per_repeat_per_magslope, box_width, box_height, num_cells, save_dir=experiment_set_directory)
    datavis.graph_chemotaxis_protrusion_lifetimes(sub_experiment_number, required_magnitudes, test_chemo_slopes, protrusion_lifetimes_per_magslope, box_width, box_height, num_cells, save_dir=experiment_set_directory)
    
    print("Complete.")
    
# =============================================================================

def two_cell_chemotaxis_threshold_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_x_offset_in_corridor=625.0, test_chemo_magnitudes=[], test_chemo_slope=0.0016, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, default_coa=0.0, default_cil=60.0):
    
    test_chemo_magnitudes = sorted(test_chemo_magnitudes)
    
    chemotaxis_success_ratios = np.zeros(len(test_chemo_magnitudes), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, chm in enumerate(test_chemo_magnitudes):
        print("=========")
        print("mag: {}".format(chm))
        experiment_name, drift_args, environment_wide_variable_defns, source_x, source_y = no_corridor_chemoattraction_test(date_str, experiment_number, sub_experiment_number, parameter_dict, chemoattractant_source_definition={'source_type': 'linear', 'x_offset_in_corridor': test_x_offset_in_corridor, 'max_value': chm, 'slope': test_chemo_slope}, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=2, box_height=1, box_y_placement_factor=0.5, num_cells=2, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=True)
        
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        experiment_name_format_string = "RPT={}"
        
        if run_experiments == False:
            if not os.path.exists(experiment_dir):
                raise Exception("Experiment directory does not exist.")
            else:
                for rpt_number in range(num_experiment_repeats):
                    environment_name = experiment_name_format_string.format(rpt_number)
                    environment_dir = os.path.join(experiment_dir, environment_name)
                    if not os.path.exists(environment_dir):
                        raise Exception("Environment directory does not exist.")
                
                    storefile_path = eu.get_storefile_path(environment_dir)
                    if not os.path.isfile(storefile_path):
                        raise Exception("Storefile does not exist.")
                    
                    relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                    if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                        raise Exception("Simulation is not complete.")
            
                print("Data exists.")
                
        chemotaxis_success_per_repeat = []
        for rpt_number in range(num_experiment_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            #empty_env_pickle_path, produce_intermediate_visuals, produce_final_visuals, environment_wide_variable_defns, simulation_execution_enabled=False
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
            
            chemotaxis_success  = cu.analyze_chemotaxis_success(relevant_environment, storefile_path, rpt_number, source_x, source_y)
            
            chemotaxis_success_per_repeat.append(chemotaxis_success)
            
        chemotaxis_success_ratios[xi] = np.sum(chemotaxis_success_per_repeat)/num_experiment_repeats



    print("=========")
    
    datavis.graph_chemotaxis_efficiency_data(sub_experiment_number, test_chemo_magnitudes, test_chemo_slope, chemotaxis_success_ratios, save_dir=experiment_set_directory)
    
    print("Complete.")
    
# =============================================================================

def corridor_migration_fixed_cells_vary_coa_cil(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_coas=[], test_cils=[], num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, auto_calculate_num_cells=True, num_cells=None, run_experiments=True, remake_graphs=False, remake_animation=False):
    
    test_coas = sorted(test_coas)
    test_cils = sorted(test_cils)
    
    average_cell_persistence = np.zeros((len(test_cils), len(test_coas)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, test_cil in enumerate(test_cils):
        for yi, test_coa in enumerate(test_coas):
            print("=========")
            print(("COA = {}, CIL = {}".format(test_coa, test_cil)))
            experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=test_coa, default_cil=test_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, auto_calculate_num_cells=auto_calculate_num_cells, num_cells=num_cells, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation)
            
            experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
            experiment_name_format_string = experiment_name + "_RPT={}"
            
            if run_experiments == False:
                if not os.path.exists(experiment_dir):
                    print("Experiment directory does not exist.")
                    average_cell_persistence[xi, yi] = np.nan
                    continue
                else:
                    no_data = False
                    for rpt_number in range(num_experiment_repeats):
                        environment_name = experiment_name_format_string.format(rpt_number)
                        environment_dir = os.path.join(experiment_dir, environment_name)
                        if not os.path.exists(environment_dir):
                            no_data = True
                            print("Environment directory does not exist.")
                            break
                    
                        storefile_path = eu.get_storefile_path(environment_dir)
                        if not os.path.isfile(storefile_path):
                            no_data = True
                            print("Storefile does not exist.")
                            break
                        
                        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                        if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                            print("Simulation is not complete.")
                            no_data = True
                            break
                    if no_data:
                        average_cell_persistence[xi, yi] = np.nan
                        continue
                
                    print("Data exists.")
            all_cell_persistences = []
            for rpt_number in range(num_experiment_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
                storefile_path = eu.get_storefile_path(environment_dir)
                relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                
                time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds  = cu.analyze_cell_motion(relevant_environment, storefile_path, 0, rpt_number)
                
                all_cell_persistences += [x[1] for x in centroids_persistences_speeds]
                
            avg_p = np.average(all_cell_persistences)
            average_cell_persistence[xi, yi] = avg_p



    print("=========")
    
    if num_cells == None:
        num_cells = box_height*box_width
    
    datavis.graph_fixed_cells_vary_coa_cil_data(sub_experiment_number, test_cils, test_coas, average_cell_persistence, num_cells, box_width, box_height, save_dir=experiment_set_directory)
        
    print("Complete.")

# =============================================================================

def corridor_migration_fixed_cells_vary_corridor_height(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_cells=[], test_heights=[], coa_dict=[], default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis_after_running_experiments=False):
    
    test_num_cells = sorted(test_num_cells)
    test_heights = sorted(test_heights)
    
    average_cell_persistence_ratios = np.zeros((len(test_num_cells), len(test_heights)), dtype=np.float64)
    average_cell_persistence_times = np.zeros((len(test_num_cells), len(test_heights)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, tnc in enumerate(test_num_cells):
        for yi, th in enumerate(test_heights):
            calculated_width = int(np.ceil(float(tnc)/th))
            print("=========")
            print(("num_cells = {}, height = {}".format(tnc, th)))
            experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[tnc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=calculated_width, box_height=th, num_cells=tnc, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis_after_running_experiments)
            
            experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
            experiment_name_format_string = experiment_name + "_RPT={}"
            
            if run_experiments == False:
                if not os.path.exists(experiment_dir):
                    print("Experiment directory does not exist.")
                    average_cell_persistence_ratios[xi, yi] = np.nan
                    average_cell_persistence_times[xi, yi] = np.nan
                    continue
                else:
                    no_data = False
                    for rpt_number in range(num_experiment_repeats):
                        environment_name = experiment_name_format_string.format(rpt_number)
                        environment_dir = os.path.join(experiment_dir, environment_name)
                        if not os.path.exists(environment_dir):
                            no_data = True
                            print("Environment directory does not exist.")
                            break
                    
                        storefile_path = eu.get_storefile_path(environment_dir)
                        if not os.path.isfile(storefile_path):
                            no_data = True
                            print("Storefile does not exist.")
                            break
                        
                        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                        if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                            print("Simulation is not complete.")
                            no_data = True
                            break
                    if no_data:
                        average_cell_persistence_ratios[xi, yi] = np.nan
                        average_cell_persistence_times[xi, yi] = np.nan
                        continue
                
                    print("Data exists.")
            all_cell_persistence_ratios = []
            all_cell_persistence_times = []
            for rpt_number in range(num_experiment_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
                storefile_path = eu.get_storefile_path(environment_dir)
                relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, produce_graphs, produce_animation, environment_wide_variable_defns)
                
                time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_centroid_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds   = cu.analyze_cell_motion(relevant_environment, storefile_path, 0, rpt_number)
                
                all_cell_persistence_ratios += [x[1][0] for x in centroids_persistences_speeds]
                all_cell_persistence_times += [x[1][1] for x in centroids_persistences_speeds]
                
            avg_pr = np.average(all_cell_persistence_ratios)
            avg_pt = np.average(all_cell_persistence_times)
            average_cell_persistence_ratios[xi, yi] = avg_pr
            average_cell_persistence_times[xi, yi] = avg_pt

    print("=========")
    
    #graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    datavis.graph_confinement_data_persistence_ratios(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_ratios, save_dir=experiment_set_directory)
    #datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
    
# =============================================================================

def corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_heights=[], test_num_cells=[], coa_dict={}, default_cil=40.0, num_experiment_repeats=1, particular_repeats=[], timesteps_between_generation_of_intermediate_visuals=None, graph_x_dimension="test_num_cells", produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, max_animation_corridor_length=None, global_scale=1, cell_placement_method=""):
    
    assert(len(test_num_cells) == len(test_heights))
    
    if graph_x_dimension == "test_num_cells":
        test_num_cells = sorted(test_num_cells)
    elif graph_x_dimension == "test_heights":
        test_heights = sorted(test_heights)
    else:
        raise Exception("Unexpected graph_x_dimension: {}".format(graph_x_dimension))
    
    num_tests = len(test_num_cells)
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    areal_strains = []
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    all_experiment_drift_args = []
    
    for xi, tnc_and_th in enumerate(zip(test_num_cells, test_heights)):
        tnc, th = tnc_and_th
        tw = max(1, int(np.ceil(float(tnc)/th)))
        print("=========")
        print(("num_cells = {}, height = {}, width = {}".format(tnc, th, tw)))
        
        experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[tnc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=tw, box_height=th, num_cells=tnc, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, max_animation_corridor_length=max_animation_corridor_length, global_scale=global_scale, cell_placement_method=cell_placement_method)
        
        all_experiment_drift_args.append(drift_args)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        areal_strains.append(areal_strains_per_cell_per_repeat)

    print("=========")
    
#    graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    #datavis.graph_combined_group_drifts(all_experiment_drift_args, "cell_number_change_data", experiment_set_label, save_dir=experiment_set_directory)
    datavis.graph_cell_number_change_data(sub_experiment_number, test_num_cells, test_heights, graph_x_dimension, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, areal_strains, experiment_set_label, save_dir=experiment_set_directory)
        
#    datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
# =============================================================================

def corridor_migration_coa_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_coas=[], default_cil=0.0, box_height=1, box_width=4, corridor_height=1, num_cells=4, num_experiment_repeats=1, particular_repeats=[], timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, max_animation_corridor_length=None, global_scale=1, cell_placement_method="", show_coa_overlay=False, justify_parameters=True):

    
    num_tests = len(test_coas)
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    areal_strains = []
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    all_experiment_drift_args = []
    
    for xi, tcoa in enumerate(test_coas):
        print("=========")
        print(("coa = {}".format(tcoa)))
        
        experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=tcoa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, num_cells=num_cells, corridor_height=corridor_height, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, max_animation_corridor_length=max_animation_corridor_length, global_scale=global_scale, cell_placement_method=cell_placement_method, justify_parameters=justify_parameters)
        
        all_experiment_drift_args.append(drift_args)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        areal_strains.append(areal_strains_per_cell_per_repeat)

    print("=========")
    
#    graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    #datavis.graph_combined_group_drifts(all_experiment_drift_args, "cell_number_change_data", experiment_set_label, save_dir=experiment_set_directory)
    datavis.graph_coa_variation_test_data(sub_experiment_number, test_coas, default_cil, corridor_height, num_cells, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, areal_strains, experiment_set_label, save_dir=experiment_set_directory)
        
#    datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
# =============================================================================

def corridor_migration_cil_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_cils=[], default_coa=0.0, box_height=1, box_width=4, corridor_height=1, num_cells=4, num_experiment_repeats=1, particular_repeats=[], timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, max_animation_corridor_length=None, global_scale=1, cell_placement_method="", show_coa_overlay=False, justify_parameters=True):

    
    num_tests = len(test_cils)
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    areal_strains = []
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    all_experiment_drift_args = []
    
    for xi, tcil in enumerate(test_cils):
        print("=========")
        print(("cil = {}".format(tcil)))
        
        experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=tcil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, num_cells=num_cells, corridor_height=corridor_height, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, max_animation_corridor_length=max_animation_corridor_length, global_scale=global_scale, cell_placement_method=cell_placement_method, justify_parameters=justify_parameters)
        
        all_experiment_drift_args.append(drift_args)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        areal_strains.append(areal_strains_per_cell_per_repeat)

    print("=========")
    
#    graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    #datavis.graph_combined_group_drifts(all_experiment_drift_args, "cell_number_change_data", experiment_set_label, save_dir=experiment_set_directory)
    datavis.graph_cil_variation_test_data(sub_experiment_number, test_cils, default_coa, corridor_height, num_cells, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, areal_strains, experiment_set_label, save_dir=experiment_set_directory)
        
#    datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
# =============================================================================

def corridor_migration_vertex_choice_tests(date_str, experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_cil=0.0, default_coa=0.0, sub_experiment_numbers=[], test_vertex_choice_ratios_and_randomization_magnitudes=[], box_height=1, box_width=4, corridor_height=1, num_cells=4, num_experiment_repeats=1, particular_repeats=[], timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, max_animation_corridor_length=None, global_scale=1, cell_placement_method="", show_coa_overlay=False, justify_parameters=True):

    assert(len(sub_experiment_numbers) == len(test_vertex_choice_ratios_and_randomization_magnitudes))
    num_tests = len(test_vertex_choice_ratios_and_randomization_magnitudes)
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    areal_strains = []
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    all_experiment_drift_args = []
    
    for xi, sen_tvr_trm in enumerate(zip(sub_experiment_numbers, test_vertex_choice_ratios_and_randomization_magnitudes)):
        sen, tvr_trm = sen_tvr_trm
        tvr, trm = tvr_trm
        print("=========")
        print(("tvr = {}, trm = {}".format(tvr, trm)))
        
        this_pd = copy.deepcopy(parameter_dict)
        this_pd.update([("randomization_node_percentage", tvr), ("randomization_magnitude", trm)])
        
        experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sen, this_pd, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, num_cells=num_cells, corridor_height=corridor_height, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, max_animation_corridor_length=max_animation_corridor_length, global_scale=global_scale, cell_placement_method=cell_placement_method, justify_parameters=justify_parameters)
        
        all_experiment_drift_args.append(drift_args)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        areal_strains.append(areal_strains_per_cell_per_repeat)

    print("=========")
    
#    graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    #datavis.graph_combined_group_drifts(all_experiment_drift_args, "cell_number_change_data", experiment_set_label, save_dir=experiment_set_directory)
    datavis.graph_vertex_choice_variation_test_data(test_vertex_choice_ratios_and_randomization_magnitudes, default_cil, default_coa, corridor_height, num_cells, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, areal_strains, experiment_set_label, save_dir=experiment_set_directory)
        
#    datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
    
# =============================================================================

def corridor_migration_parameter_set_test(date_str, experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, labels=[], sub_experiment_numbers=[], parameter_update_dicts=[], default_coas=[], default_cils=[], num_cells=2, box_height=1, box_width=2, corridor_height=1, num_experiment_repeats=1, particular_repeats=[], timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, max_animation_corridor_length=None, global_scale=1, cell_placement_method="", show_coa_overlay=False, justify_parameters=True):
    
    assert(len(labels) == len(sub_experiment_numbers) == len(default_coas) == len(default_cils) == len(parameter_update_dicts))
    
    num_tests = len(labels)
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    areal_strains = []
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    all_experiment_drift_args = []
    
    for xi, l_sen_pud_dcoa_dcil in enumerate(zip(labels, sub_experiment_numbers, parameter_update_dicts, default_coas, default_cils)):
        label, sen, pud, dcoa, dcil = l_sen_pud_dcoa_dcil
        print("=========")
        print(("label: {}\nsen: {}\n".format(label, sen)))
 
        this_pd = copy.deepcopy(parameter_dict)
        this_pd.update(pud)
        experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sen, this_pd, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=dcoa, default_cil=dcil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, num_cells=num_cells, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, max_animation_corridor_length=max_animation_corridor_length, global_scale=global_scale, cell_placement_method=cell_placement_method, justify_parameters=justify_parameters)
        
        all_experiment_drift_args.append(drift_args)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        areal_strains.append(areal_strains_per_cell_per_repeat)

    print("=========")
    
#    graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    #datavis.graph_combined_group_drifts(all_experiment_drift_args, "cell_number_change_data", experiment_set_label, save_dir=experiment_set_directory)
    datavis.graph_corridor_migration_parameter_test_data(labels, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, areal_strains, experiment_set_label, save_dir=experiment_set_directory)
        
#    datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
# =============================================================================
    

def convergence_test_corridor(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_nodes=[], coa_dict={}, default_cil=40.0, default_interaction_factor_migr_bdry_contact=30.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, biased_rgtpase_distrib_defn_dict={'default': ['unbiased uniform', np.array([0, 2*np.pi]), 0.3]}):
    
    num_tests = len(test_num_nodes)
    num_timepoints = int(total_time_in_hours*3600.0/timestep_length) + 1
    
    cell_full_speeds = np.zeros((num_tests, 2), dtype=np.float64)
    active_racs = np.zeros((num_tests, 2), dtype=np.float64)
    active_rhos = np.zeros((num_tests, 2), dtype=np.float64)
    inactive_racs = np.zeros((num_tests, 2), dtype=np.float64)
    inactive_rhos = np.zeros((num_tests, 2), dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, nn in enumerate(test_num_nodes):
        print("=========")
        print(("nn = {}".format(nn)))
        
        this_parameter_dict = copy.deepcopy(parameter_dict)
        this_parameter_dict['num_nodes'] = np.array([nn], dtype=np.int64)[0]
        this_parameter_dict['interaction_factor_migr_bdry_contact'] =  default_interaction_factor_migr_bdry_contact/nn
        experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, this_parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2]/nn, default_cil=default_cil/nn, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=2, box_height=1, num_cells=2, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, convergence_test=True, biased_rgtpase_distrib_defn_dict=biased_rgtpase_distrib_defn_dict)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, cell_full_speeds_per_repeat, cell_rac_active_max_conc_per_repeat, cell_rac_inactive_max_conc_per_repeat, cell_rho_active_max_conc_per_repeat, cell_rho_inactive_max_conc_per_repeat = collate_corridor_convergence_data(num_experiment_repeats, experiment_dir)
        
        #cell_x_positions = [cps[:,0] for cps in all_cell_centroids_per_repeat]
        #cell_positions[xi] = [np.array([xs_per_t[:, k] for k in range(2)]) for xs_per_t in cell_x_positions]
        
        cell_full_speeds[xi] = [np.average(cell_full_speeds_per_repeat[0][k][int((num_timepoints - 1)/2):]) for k in range(2)]
        active_racs[xi] = [np.average(cell_rac_active_max_conc_per_repeat[0][k][int(num_timepoints/2):]) for k in range(2)]
        inactive_racs[xi] = [np.average(cell_rac_inactive_max_conc_per_repeat[0][k][int(num_timepoints/2):]) for k in range(2)]
        active_rhos[xi] = [np.average(cell_rho_active_max_conc_per_repeat[0][k][int(num_timepoints/2):]) for k in range(2)]
        inactive_rhos[xi] = [np.average(cell_rho_inactive_max_conc_per_repeat[0][k][int(num_timepoints/2):]) for k in range(2)]


    print("=========")
    
    datavis.graph_corridor_convergence_test_data(sub_experiment_number, test_num_nodes, cell_full_speeds, active_racs, active_rhos, inactive_racs, inactive_rhos, save_dir=experiment_set_directory)
        
        
    print("Complete.")
    
# =============================================================================

def corridor_migration_init_conditions_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_cells=[], test_heights=[], max_placement_distance_factors=[], init_random_cell_placement_x_factors=[], test_widths=[], corridor_heights=[], box_placement_factors=[], coa_dict={}, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False):
    
    num_tests = len(test_num_cells)
    assert(np.all([len(x) == num_tests for x in [test_heights, test_widths, corridor_heights, box_placement_factors, max_placement_distance_factors, init_random_cell_placement_x_factors]]))
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    transient_end_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    tests = list(zip(test_num_cells, test_heights, test_widths, corridor_heights, box_placement_factors, max_placement_distance_factors, init_random_cell_placement_x_factors))
    for xi, nc_th_tw_ch_bpy_mpdf_ircpx in enumerate(tests):
        nc, th, tw, ch, bpy, mpdf, ircpx = nc_th_tw_ch_bpy_mpdf_ircpx
        print("=========")
        print(("nc = {}, h = {}, w = {}, ch = {}, bpy = {}".format(nc, th, tw, ch, bpy)))
        if th != "r":
            experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[nc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=tw, box_height=th, num_cells=nc, corridor_height=ch, box_y_placement_factor=bpy, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)
        else:
            experiment_name, drift_args = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[nc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=tw, box_height=ch, num_cells=nc, corridor_height=ch, box_y_placement_factor=bpy, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis, cell_placement_method="r", max_placement_distance_factor=mpdf, init_random_cell_placement_x_factor=ircpx)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        transient_end_times[xi] = transient_end_times_per_repeat

    print("=========")
    
    #graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    datavis.graph_init_condition_change_data(sub_experiment_number, tests, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, transient_end_times, experiment_set_label, save_dir=experiment_set_directory)
        
    #datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print("Complete.")
    
#==============================================================================

def corridor_migration_multigroup_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, box_width=2, box_height=1, cell_diameter=40, num_groups=2, verbose=True, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=1.2, default_intra_group_cil=20, default_inter_group_cil=40, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_animation=True, produce_graphs=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, animation_time_resolution='normal', remake_graphs=False, remake_animation=False):    
    num_cells_in_group = box_width*box_height
    
    experiment_name_format_string = "multigroup_corridor_migration_{}_NC=({}, {})_NG={}".format("{}", box_width, box_height, num_groups)
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = num_groups
    num_cells_in_boxes = [num_cells_in_group, num_cells_in_group]
    box_heights = [box_height*cell_diameter]*num_boxes
    box_widths = [box_width*cell_diameter]*num_boxes

    x_space_between_boxes = []
    box_x_offsets = [10]*num_boxes
    box_y_offsets = [30 + box_height*cell_diameter*n for n in range(num_boxes)]
    plate_width, plate_height = box_widths[0]*10*1.5, (np.sum(box_heights) + 40)*2.4

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "OVERRIDE", "OVERRIDE", origin_y_offset=30, migratory_corridor_size=[box_widths[0]*100, np.sum(box_heights)], physical_bdry_polygon_extra=20, box_x_offsets=box_x_offsets, box_y_offsets=box_y_offsets)
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    
    ic_dict_tuple_list = []    
    for n in range(num_boxes):
        tuple_list = []
        for m in range(num_boxes):
            if (n == m):
                tuple_list.append((m, default_intra_group_cil))
            else:
                tuple_list.append((m, default_inter_group_cil))
                
        ic_dict_tuple_list.append((n, dict(tuple_list)))
                
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict(ic_dict_tuple_list)]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}]*num_groups]
    parameter_override_dicts_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: corridor migration, multi-group"]
    chemoattractant_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []
    
    si = 0
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'init_cell_radius': cell_diameter*0.5*1e-6, 'C_total': 3e6, 'H_total': 1.5e6, 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'intercellular_contact_factor_magnitudes_defn': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'cell_dependent_coa_signal_strengths_defn': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_override_dict': parameter_override_dicts_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 1
    
    if animation_time_resolution not in ['normal', 'high', 'adaptive']:
        raise Exception("Unknown animation_time_resolution specified: {}".format(animation_time_resolution))
    elif animation_time_resolution == 'normal':
        short_video_length_definition = 1000.0*timestep_length
        short_video_duration = 2.0
    elif 'high':
        short_video_length_definition = 100.0*timestep_length
        short_video_duration = 4.0
    elif 'adaptive':
        short_video_length_definition = int(0.1*num_timesteps)*timestep_length
        short_video_duration = 2.0
        
        
    animation_settings = setup_animation_settings(timestep_length, global_scale, plate_height, plate_width, show_rac_random_spikes=False, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, string_together_pictures_into_animation=True, show_coa_overlay=show_coa_overlay, coa_too_close_dist_squared=1, coa_distribution_exponent=np.log(parameter_dict['coa_sensing_value_at_dist'])/(parameter_dict['coa_sensing_dist_at_value']/1e-6), coa_intersection_exponent=parameter_dict['coa_intersection_exponent'])  
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, baseline_parameter_dict, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, chemoattractant_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_graphs=produce_graphs, produce_animation=produce_animation, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)

    print("Done.")    

# =============================================================================
    
def non_linear_to_linear_parameter_comparison(date_str, experiment_number, parameter_dicts, base_output_dir="B:\\numba-ncc\\output\\"):
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    kgtp_rac_multipliers = [pd['kgtp_rac_multiplier'] for pd in parameter_dicts]
    kgtp_rho_multipliers = [pd['kgtp_rho_multiplier'] for pd in parameter_dicts]
    kgtp_rac_autoact_multipliers = [pd['kgtp_rac_autoact_multiplier'] for pd in parameter_dicts]
    kgtp_rho_autoact_multipliers = [pd['kgtp_rho_autoact_multiplier'] for pd in parameter_dicts]
    kdgtp_rac_multipliers = [pd['kdgtp_rac_multiplier'] for pd in parameter_dicts]
    kdgtp_rho_multipliers = [pd['kdgtp_rho_multiplier'] for pd in parameter_dicts]
    kdgtp_rho_mediated_rac_inhib_multipliers = [pd['kdgtp_rho_mediated_rac_inhib_multiplier'] for pd in parameter_dicts]
    kdgtp_rac_mediated_rho_inhib_multipliers = [pd['kdgtp_rac_mediated_rho_inhib_multiplier'] for pd in parameter_dicts]
    
    datavis.graph_nonlin_to_lin_parameter_comparison(kgtp_rac_multipliers, kgtp_rho_multipliers, kgtp_rac_autoact_multipliers, kgtp_rho_autoact_multipliers, kdgtp_rac_multipliers, kdgtp_rho_multipliers, kdgtp_rho_mediated_rac_inhib_multipliers, kdgtp_rac_mediated_rho_inhib_multipliers, save_dir=experiment_set_directory)
    
    print("Done.")
    
