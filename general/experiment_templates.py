# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 16:00:16 2016

@author: Brian Merchant
"""

import numpy as np
import general.exec_utils as eu
import analysis.utilities as au
import visualization.datavis as datavis
import os


def define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, default_x_position, default_y_position, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None]):
    if len(box_heights) < num_boxes:
        raise StandardError("Number of boxes is greater than number of box heights given.")
        
    if len(box_widths) < num_boxes:
        raise StandardError("Number of boxes is greater than number of box widths given.")
        
    if len(x_space_between_boxes) < num_boxes - 1:
        raise StandardError("Not enough x_space_between_boxes given (missing {}).".format((num_boxes - 1) - len(x_space_between_boxes)))
    
    default_positions = ["CENTER", "ORIGIN", "OVERRIDE"]
    
    if default_x_position not in default_positions:
        raise StandardError("Default x-position of boxes is not in list {} (given: {}).".format(default_positions, default_x_position))
        
    if default_y_position not in default_positions:
        raise StandardError("Default y-position of boxes is not in list {} (given: {}).".format(default_positions, default_y_position))
        
    boxes = np.arange(num_boxes)
    
    if default_x_position == "ORIGIN":
        x_offset = origin_x_offset
    elif default_x_position == "CENTER":
        x_offset = 0.5*(plate_width - 0.5*box_widths[0])
    elif default_x_position == "OVERRIDE":
        if box_x_offsets == []:
            raise StandardError("default_x_position is OVERRIDE, but x_offset is not given!")
        
    if default_y_position == "ORIGIN":
        y_offset = origin_y_offset
    elif default_x_position == "CENTER":
        y_offset = 0.5*(plate_height - 0.5*box_heights[0])
    elif default_y_position == "OVERRIDE":
        if box_y_offsets == []:
            raise StandardError("default_x_position is OVERRIDE, but x_offset is not given!")
    
    if default_x_position != "OVERRIDE":
        box_x_offsets = [0]*num_boxes
        for bi in boxes:
            if bi == 0:
                box_x_offsets[bi] = x_offset
            else:
                box_x_offsets[bi] = x_offset + x_space_between_boxes[bi-1]

    if default_y_position != "OVERRIDE":
        box_y_offsets = [y_offset]*num_boxes
    
    if make_only_migratory_corridor:
        make_migr_poly = True
        make_phys_poly = False
    else:
        make_migr_poly = True
        make_phys_poly = True
    
    if migratory_corridor_size == [None, None]:
        make_migr_poly = False
        make_phys_poly = False
        
    width_migr_corridor, height_migr_corridor = migratory_corridor_size

    space_migratory_bdry_polygon, space_physical_bdry_polygon = eu.make_space_polygons(make_migr_poly, make_phys_poly, width_migr_corridor, height_migr_corridor, origin_x_offset, origin_y_offset, physical_bdry_polygon_extra=physical_bdry_polygon_extra)

    return boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon

# ===========================================================================

def produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals):
    if timesteps_between_generation_of_intermediate_visuals == None:
        produce_intermediate_visuals = np.array([])
    else:
        if type(timesteps_between_generation_of_intermediate_visuals) != int:
            raise StandardError("Non-integer value given for timesteps_between_generation_of_intermediate_visuals")
        else:
            if timesteps_between_generation_of_intermediate_visuals > 0:
                produce_intermediate_visuals = np.arange(timesteps_between_generation_of_intermediate_visuals, num_timesteps, step=timesteps_between_generation_of_intermediate_visuals)
            else:
                raise StandardError("timesteps_between_generation_of_intermediate_visuals <= 0!")
                
    return produce_intermediate_visuals

# ===========================================================================

def fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, parameter_overrides_dict):
    if randomization==True:
        rand_type = None
        if parameter_overrides_dict['randomization_scheme'] == 'wipeout':
            rand_type = 'w'
        else:
            rand_type = 'm'
        experiment_name = experiment_name_format_string.format("rand-{}".format(rand_type))
    else:
        experiment_name = experiment_name_format_string.format("(randomization, {})".format(None))
        
    return experiment_name
# ===========================================================================
    
def single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, base_output_dir="C:\\cygwin\\home\\Brian Merchant\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):
    
    experiment_name_format_string = "single_cell_{}"
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, parameter_overrides_dict)
        
    parameter_overrides_dict = eu.update_pd(parameter_overrides_dict, 'randomization', parameter_overrides_dict['randomization'], randomization)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [1]
    box_heights = [1*cell_diameter]
    box_widths = [1*cell_diameter]

    x_space_between_boxes = []
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths,x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER")
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]]
    parameter_override_dicts_per_sub_experiment = [[parameter_overrides_dict]]
    experiment_descriptions_per_subexperiment = ["from experiment template: single cell, no randomization"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env)
    
    experiment_name_format_string = experiment_name + "_RPT={}"
    extracted_results = []
    for rpt_number in xrange(num_experiment_repeats):
        environment_name = experiment_name_format_string.format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        storefile_path = eu.get_storefile_path(environment_dir)
        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
        
        analysis_data = au.analyze_single_cell_motion(relevant_environment, storefile_path, si, rpt_number)
        
        extracted_results.append(analysis_data)
        # ================================================================
        
    datavis.present_collated_single_cell_motion_data(extracted_results, experiment_dir)

    print "Done."
    
    
# ============================================================================
    
def two_cells_cil_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, base_output_dir="C:\\cygwin\\home\\Brian Merchant\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, default_coa=0, default_cil=20, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):
    
    experiment_name_format_string = "cil_test_{}"
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, parameter_overrides_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 2
    num_cells_in_boxes = [1, 1]
    box_heights = [1*cell_diameter]*num_boxes
    box_widths = [1*cell_diameter]*num_boxes

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = 10*cell_diameter*1.2, 2.4*cell_diameter

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "OVERRIDE", "ORIGIN", physical_bdry_polygon_extra=20, box_x_offsets=[10 + 3*cell_diameter, 10 + (3 + 1 + 2)*cell_diameter],  migratory_corridor_size=[10*cell_diameter, 1*cell_diameter])
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 1.0]}]]
    parameter_override_dicts_per_sub_experiment = [[parameter_overrides_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: two cells, cil test"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
                
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env)

    print "Done."    
    
# ============================================================================
    
def many_cells_coa_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, base_output_dir="C:\\cygwin\\home\\Brian Merchant\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_on_box_side=4, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, default_coa=0.2, default_cil=20, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):
    
    experiment_name_format_string = "coa_test_{}"
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, parameter_overrides_dict)
            
    parameter_overrides_dict = eu.update_pd(parameter_overrides_dict, 'randomization', parameter_overrides_dict['randomization'], randomization)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells_on_box_side*num_cells_on_box_side]
    box_heights = [num_cells_on_box_side*cell_diameter]*num_boxes
    box_widths = [num_cells_on_box_side*cell_diameter]*num_boxes

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER")
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 1.0]}]]
    parameter_override_dicts_per_sub_experiment = [[parameter_overrides_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: two cells, cil test"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
                
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env)

    print "Done."    
    

# ============================================================================
    
def corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, base_output_dir="C:\\cygwin\\home\\Brian Merchant\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=1, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, default_coa=1.2, default_cil=20, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):
    
    total_num_cells = num_cells_width*num_cells_height
    
    experiment_name_format_string = "corridor_migration_{}_NC=({}, {})".format("{}", num_cells_width, num_cells_height)
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, parameter_overrides_dict)
        
    parameter_overrides_dict = eu.update_pd(parameter_overrides_dict, 'randomization', parameter_overrides_dict['randomization'], randomization)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [total_num_cells]
    box_heights = [num_cells_height*cell_diameter]*num_boxes
    box_widths = [num_cells_width*cell_diameter]*num_boxes

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = box_widths[0]*10*1.5, box_heights[0]*2.4

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "ORIGIN", "ORIGIN", migratory_corridor_size=[box_widths[0]*100, box_heights[0]], physical_bdry_polygon_extra=20)
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 1.0]}]]
    parameter_override_dicts_per_sub_experiment = [[parameter_overrides_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: two cells, cil test"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
                
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env)

    print "Done."    
    
    