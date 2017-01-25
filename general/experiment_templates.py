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

global_randomization_scheme_dict = {'m': 'kgtp_rac_multipliers', 'w': 'wipeout'}

def define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, default_x_position, default_y_position, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None]):
    if len(box_heights) < num_boxes:
        raise StandardError("Number of boxes is greater than number of box heights given.")
        
    if len(box_widths) < num_boxes:
        raise StandardError("Number of boxes is greater than number of box widths given.")

    
    default_positions = ["CENTER", "ORIGIN", "OVERRIDE"]
    
    if default_x_position not in default_positions:
        raise StandardError("Default x-position of boxes is not in list {} (given: {}).".format(default_positions, default_x_position))
        
    if default_y_position not in default_positions:
        raise StandardError("Default y-position of boxes is not in list {} (given: {}).".format(default_positions, default_y_position))
        
    if len(x_space_between_boxes) < num_boxes - 1 and default_x_position != "OVERRIDE":
        raise StandardError("Not enough x_space_between_boxes given (missing {}).".format((num_boxes - 1) - len(x_space_between_boxes)))
        
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

def update_pd_with_randomization_info(pd, randomization, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w):
    global global_randomization_scheme_dict
    pd = eu.update_pd(pd, 'randomization', pd['randomization'], randomization)
    
    if randomization:
        if randomization_scheme in ['m', 'w']:
            pd = eu.update_pd(pd, 'randomization_scheme', pd['randomization_scheme'], global_randomization_scheme_dict[randomization_scheme])
            
            if randomization_scheme == 'm':
                pd = eu.update_pd(pd, 'randomization_time_mean', pd['randomization_time_mean'], randomization_time_mean_m)
                pd = eu.update_pd(pd, 'randomization_time_variance_factor', pd['randomization_time_variance_factor'], randomization_time_variance_factor_m)
                pd = eu.update_pd(pd, 'randomization_magnitude', pd['randomization_magnitude'], randomization_magnitude_m)
            else:
                pd = eu.update_pd(pd, 'randomization_time_mean', pd['randomization_time_mean'], randomization_time_mean_w)
                pd = eu.update_pd(pd, 'randomization_time_variance_factor', pd['randomization_time_variance_factor'], randomization_time_variance_factor_w)
        else:
            raise StandardError("Unknown randomization_scheme given: {} (should be either 'm' or 'w')").format(randomization_scheme)
            
    return pd
            

# ===========================================================================

def fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, randomization_scheme, parameter_overrides_dict):
    if randomization==True:
        experiment_name = experiment_name_format_string.format("rand-{}".format(randomization_scheme))
    else:
        experiment_name = experiment_name_format_string.format("rand-{}".format('no'))
        
    return experiment_name
# ===========================================================================
    
def single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=20.0, randomization_time_variance_factor_m=0.01, randomization_magnitude_m=0.75*25, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):    
    experiment_name_format_string = "single_cell_{}"
    
    parameter_overrides_dict = update_pd_with_randomization_info(parameter_overrides_dict, randomization, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w)
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, randomization_scheme, parameter_overrides_dict)
    
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
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_rac_random_spikes', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps)
    
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
    
def two_cells_cil_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=20.0, randomization_time_variance_factor_m=0.01, randomization_magnitude_m=0.75*25, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, migr_bdry_height_factor=0.8, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=20, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):
    
    experiment_name_format_string = "cil_test_CIL={}_COA={}".format(default_cil, default_coa) + "_{}"

    parameter_overrides_dict = update_pd_with_randomization_info(parameter_overrides_dict, randomization, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w)
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, randomization_scheme, parameter_overrides_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 2
    num_cells_in_boxes = [1, 1]
    box_heights = [1*cell_diameter]*num_boxes
    box_widths = [1*cell_diameter]*num_boxes

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = 10*cell_diameter*1.2, 2.4*cell_diameter

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "OVERRIDE", "ORIGIN", physical_bdry_polygon_extra=20, box_x_offsets=[10 + 3*cell_diameter, 10 + (3 + 1 + 2)*cell_diameter],  migratory_corridor_size=[10*cell_diameter, migr_bdry_height_factor*cell_diameter])
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_rac_random_spikes', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
                
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps)

    print "Done."    
    
# ============================================================================
    
def many_cells_coa_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=20.0, randomization_time_variance_factor_m=0.01, randomization_magnitude_m=0.75*25, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=4, num_cells_height=4, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0.2, default_cil=20, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True):    
    experiment_name_format_string = "coa_test_{}_NC={}_COA={}".format("{}", num_cells_width*num_cells_height, default_coa)
    
    parameter_overrides_dict = update_pd_with_randomization_info(parameter_overrides_dict, randomization, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w)
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, randomization_scheme, parameter_overrides_dict)
            
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells_width*num_cells_height]
    box_heights = [num_cells_height*cell_diameter]*num_boxes
    box_widths = [num_cells_width*cell_diameter]*num_boxes

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER")
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 1.0]}]]
    parameter_override_dicts_per_sub_experiment = [[parameter_overrides_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: many cell coa test"]
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_rac_random_spikes', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 2.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
                
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps)

    print "Done."    
    
# ============================================================================
    
def corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=20.0, randomization_time_variance_factor_m=0.01, randomization_magnitude_m=0.75*25, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=1, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=1.2, default_cil=20, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, animation_time_resolution='normal'):    
    total_num_cells = num_cells_width*num_cells_height
    
    experiment_name_format_string = "corridor_migration_{}_NC=({}, {})_COA={}".format("{}", num_cells_width, num_cells_height, default_coa)
    
    parameter_overrides_dict = update_pd_with_randomization_info(parameter_overrides_dict, randomization, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w)
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, randomization_scheme, parameter_overrides_dict)
            
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [total_num_cells]
    box_heights = [num_cells_height*cell_diameter]*num_boxes
    box_widths = [num_cells_width*cell_diameter]*num_boxes

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = min(2000, box_widths[0]*10*1.5), (box_heights[0] + 40 + 100)

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "ORIGIN", "ORIGIN", origin_y_offset=30, migratory_corridor_size=[box_widths[0]*100, box_heights[0]], physical_bdry_polygon_extra=20)
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'num_nodes': num_nodes, 'verbose': verbose, 'closeness_dist_squared_criteria': closeness_dist_squared_criteria, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
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
    
    if animation_time_resolution not in ['normal', 'high', 'adaptive']:
        raise StandardError("Unknown animation_time_resolution specified: {}".format(animation_time_resolution))
    elif animation_time_resolution == 'normal':
        short_video_length_definition = 1000.0*timestep_length
        short_video_duration = 2.0
    elif 'high':
        short_video_length_definition = 100.0*timestep_length
        short_video_duration = 4.0
    elif 'adaptive':
        short_video_length_definition = int(0.1*num_timesteps)*timestep_length
        short_video_duration = 2.0
        
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_rac_random_spikes', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', short_video_length_definition), ('short_video_duration', short_video_duration), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
                
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps)

    print "Done."

# ============================================================================
    
def corridor_migration_multigroup_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=20.0, randomization_time_variance_factor_m=0.01, randomization_magnitude_m=0.75*25, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=1, cell_diameter=40, num_groups=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=1.2, default_intra_group_cil=20, default_inter_group_cil=40, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, animation_time_resolution='normal'):    
    num_cells_in_group = num_cells_width*num_cells_height
    
    experiment_name_format_string = "multigroup_corridor_migration_{}_NC=({}, {})_NG={}".format("{}", num_cells_width, num_cells_height, num_groups)
    
    parameter_overrides_dict = update_pd_with_randomization_info(parameter_overrides_dict, randomization, randomization_scheme, randomization_time_mean_m, randomization_time_variance_factor_m, randomization_magnitude_m, randomization_time_mean_w, randomization_time_variance_factor_w)
    
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization, randomization_scheme, parameter_overrides_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = num_groups
    num_cells_in_boxes = [num_cells_in_group, num_cells_in_group]
    box_heights = [num_cells_height*cell_diameter]*num_boxes
    box_widths = [num_cells_width*cell_diameter]*num_boxes

    x_space_between_boxes = []
    box_x_offsets = [10]*num_boxes
    box_y_offsets = [30 + num_cells_height*cell_diameter*n for n in range(num_boxes)]
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
    parameter_override_dicts_per_sub_experiment = [[parameter_overrides_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: corridor migration, multi-group"]
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
    
    if animation_time_resolution not in ['normal', 'high', 'adaptive']:
        raise StandardError("Unknown animation_time_resolution specified: {}".format(animation_time_resolution))
    elif animation_time_resolution == 'normal':
        short_video_length_definition = 1000.0*timestep_length
        short_video_duration = 2.0
    elif 'high':
        short_video_length_definition = 100.0*timestep_length
        short_video_duration = 4.0
    elif 'adaptive':
        short_video_length_definition = int(0.1*num_timesteps)*timestep_length
        short_video_duration = 2.0
        
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', True), ('show_rac_random_spikes', True), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', short_video_length_definition), ('short_video_duration', short_video_duration), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True), ('color_each_group_differently', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_overrides_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps)

    print "Done."    
    
    