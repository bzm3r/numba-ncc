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

global_randomization_scheme_dict = {'m': 'kgtp_rac_multipliers', 'w': 'wipeout'}

def define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, default_x_position_calculation_type, default_y_position_calculation_type, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None]):
    if len(box_heights) < num_boxes:
        raise StandardError("Number of boxes is greater than number of box heights given.")
        
    if len(box_widths) < num_boxes:
        raise StandardError("Number of boxes is greater than number of box widths given.")

    
    allowable_default_position_calculation_types = ["CENTER", "ORIGIN", "OVERRIDE"]
    
    if default_x_position_calculation_type not in allowable_default_position_calculation_types:
        raise StandardError("Default x-position of boxes is not in list {} (given: {}).".format(default_x_position_calculation_type, default_x_position_calculation_type))
        
    if default_y_position_calculation_type not in allowable_default_position_calculation_types:
        raise StandardError("Default y-position of boxes is not in list {} (given: {}).".format(default_x_position_calculation_type, default_y_position_calculation_type))
        
    if len(x_space_between_boxes) < num_boxes - 1 and default_x_position_calculation_type != "OVERRIDE":
        raise StandardError("Not enough x_space_between_boxes given (missing {}).".format((num_boxes - 1) - len(x_space_between_boxes)))
        
    boxes = np.arange(num_boxes)
    
    if default_x_position_calculation_type == "ORIGIN":
        x_offset = origin_x_offset
    elif default_x_position_calculation_type == "CENTER":
        x_offset = 0.5*(plate_width - 0.5*box_widths[0])
    elif default_x_position_calculation_type == "OVERRIDE":
        if box_x_offsets == []:
            raise StandardError("default_x_position_calculation_type is OVERRIDE, but x_offset is not given!")
        
    if default_y_position_calculation_type == "ORIGIN":
        y_offset = origin_y_offset
    elif default_y_position_calculation_type == "CENTER":
        y_offset = 0.5*(plate_height - 0.5*box_heights[0])
    elif default_y_position_calculation_type == "OVERRIDE":
        if box_y_offsets == []:
            raise StandardError("default_y_position_calculation_type is OVERRIDE, but y_offset is not given!")
    
    if default_x_position_calculation_type != "OVERRIDE":
        box_x_offsets = [0]*num_boxes
        for bi in boxes:
            if bi == 0:
                box_x_offsets[bi] = x_offset
            else:
                box_x_offsets[bi] = x_offset + x_space_between_boxes[bi-1]

    if default_y_position_calculation_type != "OVERRIDE":
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
        raise StandardError("Unknown randomization_scheme given: {} (should be either 'm' or 'w')").format(randomization_scheme)
            
    return pd
            

# ===========================================================================

def fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_overrides_dict):
    if randomization_scheme in ['m', 'w']:
        experiment_name = experiment_name_format_string.format("rand-{}".format(randomization_scheme))
    else:
        experiment_name = experiment_name_format_string.format("rand-{}".format('no'))
        
    return experiment_name
# ===========================================================================

def setup_polarization_experiment(parameter_dict, total_time_in_hours=1, timestep_length=2, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=None, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, init_rho_gtpase_conditions=None):    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [1]
    box_heights = [1*cell_diameter]
    box_widths = [1*cell_diameter]

    x_space_between_boxes = []
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths,x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER")
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon*1e-6
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon*1e-6
    
    environment_wide_variable_defns = {'parameter_explorer_run': True, 'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': False, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, 'parameter_explorer_init_rho_gtpase_conditions': init_rho_gtpase_conditions}
    
    cell_dependent_coa_signal_strengths_defn_dict = dict([(x, default_coa) for x in boxes])
    intercellular_contact_factor_magnitudes_defn_dict = dict([(x, default_cil) for x in boxes])
    
    biased_rgtpase_distrib_defn_dict = {'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}
    
    user_cell_group_defn = {'cell_group_name': 0, 'num_cells': 1, 'init_cell_radius': cell_diameter*0.5*1e-6, 'cell_group_bounding_box': np.array([box_x_offsets[0], box_x_offsets[0] + box_widths[0], box_y_offsets[0], box_heights[0] + box_y_offsets[0]])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dict, 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dict, 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dict, 'parameter_dict': parameter_dict} 
        
    return (environment_wide_variable_defns, user_cell_group_defn)
    
def single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, parameter_dict, base_output_dir="A:\\numba-ncc\\output\\", no_randomization=False, total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, justify_parameters=True, remake_visualizations=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
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
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths,x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER")
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]]
    parameter_dict_per_sub_experiment = [[parameter_dict]]
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
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 1
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, experiment_name, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, justify_parameters=justify_parameters, remake_visualizations=remake_visualizations)
    
    experiment_name_format_string = experiment_name + "_RPT={}"
    extracted_results = []
    for rpt_number in xrange(num_experiment_repeats):
        environment_name = experiment_name_format_string.format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        storefile_path = eu.get_storefile_path(environment_dir)
        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
        
        analysis_data = cu.analyze_single_cell_motion(relevant_environment, storefile_path, si, rpt_number)
        
        extracted_results.append(analysis_data)
        # ================================================================
        
    datavis.present_collated_single_cell_motion_data(extracted_results, experiment_dir, total_time_in_hours)

    print "Done."
    
    
# ============================================================================

def two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=0.8, remake_visualizations=False):    
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    experiment_name_format_string = "cil_test_{}_CIL={}_COA={}".format(sub_experiment_number, default_cil, default_coa) + "_{}"
    
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

    x_space_between_boxes = [2*cell_diameter]
    plate_width, plate_height = 10*cell_diameter*1.2, 2.4*cell_diameter

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "OVERRIDE", "ORIGIN", physical_bdry_polygon_extra=20, box_x_offsets=[10 + 3*cell_diameter, 10 + (3 + 1 + 2)*cell_diameter],  migratory_corridor_size=[10*cell_diameter, migr_bdry_height_factor*cell_diameter])
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]), 1.0]}, {'default': ['biased uniform', np.array([-0.25*np.pi, 0.25*np.pi]) + np.pi, 1.0]}]]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
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
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': num_cells_in_boxes[bi], 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'interaction_factors_intercellular_contact_per_celltype': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'interaction_factors_coa_per_celltype': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_dict': parameter_dict_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    global_scale = 1
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, experiment_name, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_visualizations=remake_visualizations)

    print "Done."
    
# ============================================================================

def block_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, remake_visualizations=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    experiment_name_format_string = "block_coa_test_{}_{}_NC={}_COA={}_CIL={}".format(sub_experiment_number, "{}", block_cells_width*block_cells_height, default_coa, default_cil)
    
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

    x_space_between_boxes = [3*cell_diameter]
    plate_width, plate_height = 1000, 1000

    box_x_offsets = [plate_width/2. - box_widths[0], plate_width/2. + x_space_between_boxes[0]]
    box_y_offsets = [plate_height/2. - box_heights[0]/2., plate_height/2. - box_heights[1]/2.]
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "OVERRIDE", "OVERRIDE", box_x_offsets=box_x_offsets, box_y_offsets=box_y_offsets)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, experiment_name, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_visualizations=remake_visualizations)
    
    experiment_name_format_string = experiment_name + "_RPT={}"
    extracted_results = []
    for rpt_number in xrange(num_experiment_repeats):
        environment_name = experiment_name_format_string.format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        storefile_path = eu.get_storefile_path(environment_dir)
        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
        
        analysis_data = cu.analyze_single_cell_motion(relevant_environment, storefile_path, si, rpt_number)
        
        extracted_results.append(analysis_data)
        # ================================================================
        
    datavis.present_collated_single_cell_motion_data(extracted_results, experiment_dir)

    print "Done."

# =============================================================================

def many_cells_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=4, auto_calculate_num_cells=True, num_cells=None, remake_visualizations=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if auto_calculate_num_cells:
        num_cells = num_cells_height*num_cells_width
    else:
        if num_cells == None:
            raise StandardError("Auto-calculation of cell number turned off, but num_cells not given!")
            
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
    box_heights = [num_cells_height*cell_diameter]
    box_widths = [num_cells_width*cell_diameter]

    x_space_between_boxes = []
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths,x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER")
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, experiment_name, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_visualizations=remake_visualizations)
    
    print "Done."
    
    return experiment_name

# =============================================================================
def make_linear_gradient_function(source_definition):
    source_x, source_y, cutoff_radius, min_value, max_value = source_definition
    
    @nb.jit(nopython=True)
    def f(x):
        d = np.sqrt((x[0] - source_x)**2 + (x[1] + source_y)**2)
        if d > cutoff_radius:
            return min_value
        else:
            return (max_value - min_value)*(1. - (d/cutoff_radius)) + min_value
        
    return f
        
# =============================================================================

def coa_factor_variation_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_coas=[], default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=4, num_cells_to_test=[], skip_low_coa=False, max_normalized_group_area=3.0, run_experiments=True, remake_visualizations=False):
    
    test_coas = sorted(test_coas)
    test_coas.reverse()
    
    average_cell_group_area_data = max_normalized_group_area*np.ones((len(num_cells_to_test), len(test_coas)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, num_cells in enumerate(num_cells_to_test):
        square_size = int(np.ceil(np.sqrt(num_cells)))
        skip_remaining_coas = False
        
        for yi, test_coa in enumerate(test_coas):
            if skip_low_coa == True:
                if skip_remaining_coas == True:
                    continue
            
            if run_experiments == True:
                experiment_name = many_cells_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=test_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, num_cells_width=square_size, num_cells_height=square_size, auto_calculate_num_cells=False, num_cells=num_cells, remake_visualizations=remake_visualizations)
                
            experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
            
            experiment_name_format_string = experiment_name + "_RPT={}"
            
            if not os.path.exists(experiment_dir):
                average_cell_group_area_data[xi, yi] = np.nan
                continue
            else:
                no_data = False
                for rpt_number in range(num_experiment_repeats):
                    environment_name = experiment_name_format_string.format(rpt_number)
                    environment_dir = os.path.join(experiment_dir, environment_name)
                    if not os.path.exists(environment_dir):
                        no_data = True
                        break
                
                    storefile_path = eu.get_storefile_path(environment_dir)
                    if not os.path.isfile(storefile_path):
                        no_data = True
                        break
                    
                    relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                    if (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                        no_data = True
                        break
                if no_data:
                    average_cell_group_area_data[xi, yi] = np.nan
                    continue
            
            
            group_areas_averaged_over_time = []
            for rpt_number in range(num_experiment_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
            
                storefile_path = eu.get_storefile_path(environment_dir)
                
                relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                
                group_area_over_time = cu.calculate_normalized_group_area_over_time(relevant_environment.num_cells, relevant_environment.curr_tpoint + 1, storefile_path)
                
                group_areas_averaged_over_time.append(np.average(group_area_over_time))
            avg_grp_a = np.average(group_areas_averaged_over_time)
            
            if avg_grp_a > max_normalized_group_area:
                skip_remaining_coas = True
            else:
                average_cell_group_area_data[xi, yi] = avg_grp_a

    if skip_low_coa == False:
        max_normalized_group_area = np.inf
        
    datavis.graph_coa_variation_test_data(sub_experiment_number, num_cells_to_test, test_coas, average_cell_group_area_data, save_dir=experiment_set_directory, max_normalized_group_area=max_normalized_group_area)
        
    print "Done."

# ============================================================================

def corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=4, auto_calculate_num_cells=True, num_cells=None, run_experiments=True, remake_visualizations=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if auto_calculate_num_cells:
        num_cells = num_cells_height*num_cells_width
    else:
        if num_cells == None:
            raise StandardError("Auto-calculation of cell number turned off, but num_cells not given!")
            
    experiment_name_format_string = "corridor_migration_{}_{}_NC=({}, {}, {})_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, num_cells_width, num_cells_height, default_coa, default_cil)
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [num_cells_height*cell_diameter]
    box_widths = [num_cells_width*cell_diameter]

    x_space_between_boxes = [2*cell_diameter]
    width_factor = 1.5
    if np.sum(num_cells_in_boxes) == 2:
        width_factor = 3
    else:
        width_factor = 1.5
        
    plate_width, plate_height = min(1000, box_widths[0]*10*width_factor), (box_heights[0] + 40 + 100)

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "ORIGIN", "ORIGIN", origin_y_offset=30, migratory_corridor_size=[box_widths[0]*100, box_heights[0]], physical_bdry_polygon_extra=20)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    if run_experiments == True:
        eu.run_template_experiments(experiment_dir, experiment_name, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_visualizations=remake_visualizations)
        
        experiment_name_format_string = experiment_name + "_RPT={}"
        extracted_cell_motion_results = []
        group_centroid_per_timestep_per_repeat = []
        min_x_centroid_per_timestep_per_repeat = []
        max_x_centroid_per_timestep_per_repeat = []
        for rpt_number in xrange(num_experiment_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
            
            min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_per_timestep, centroids_and_persistences  = cu.analyze_cell_motion(relevant_environment, storefile_path, si, rpt_number)
            
            extracted_cell_motion_results += centroids_and_persistences
            group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
            min_x_centroid_per_timestep_per_repeat.append(min_x_centroid_per_timestep)
            max_x_centroid_per_timestep_per_repeat.append(max_x_centroid_per_timestep)
            # ================================================================
            
        datavis.present_collated_cell_motion_data(extracted_cell_motion_results, experiment_dir, total_time_in_hours)
        datavis.present_collated_group_centroid_drift_data(timestep_length, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_per_timestep_per_repeat, experiment_dir, total_time_in_hours)

    print "Done."
    
    return experiment_name


# ============================================================================

def chemoattractant_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=4, auto_calculate_num_cells=True, num_cells=None, run_experiments=True, chemoattractant_source_definition=None, autocalculate_chemoattractant_source_y_position=True, migratory_box_width=2000, migratory_box_height=400, remake_visualizations=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if auto_calculate_num_cells:
        num_cells = num_cells_height*num_cells_width
    else:
        if num_cells == None:
            raise StandardError("Auto-calculation of cell number turned off, but num_cells not given!")
            
    experiment_name_format_string = "chatr_{}_{}_NC=({}, {}, {})_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, num_cells_width, num_cells_height, default_coa, default_cil)
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
    experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
    
    total_time = total_time_in_hours*3600
    num_timesteps = int(total_time/timestep_length)
    
    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [num_cells_height*cell_diameter]
    box_widths = [num_cells_width*cell_diameter]

    x_space_between_boxes = [2*cell_diameter]
        
    plate_width, plate_height = 2000, 700
    
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "ORIGIN", "CENTER", origin_y_offset=30, migratory_corridor_size=[plate_width, plate_height], physical_bdry_polygon_extra=20)
    
    if autocalculate_chemoattractant_source_y_position == True and chemoattractant_source_definition != None:
        chemoattractant_source_definition[1] = (space_migratory_bdry_polygon[0][1] + space_migratory_bdry_polygon[2][1])/(2e-6)
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc}
    
    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [[dict([(x, default_coa) for x in boxes])]*num_boxes]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [dict([(n, cil_dict) for n in range(num_boxes)])]
    
    biased_rgtpase_distrib_defn_dicts = [[{'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]}]*num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict]*num_boxes]
    experiment_descriptions_per_subexperiment = ["from experiment template: coa test"]
    if chemoattractant_source_definition != None:
        if type(chemoattractant_source_definition) != list or len(chemoattractant_source_definition) != 5:
            raise StandardError("Invalid chemoattractant source definition given: {}".format(chemoattractant_source_definition))
        external_gradient_fn_per_subexperiment = [make_linear_gradient_function(chemoattractant_source_definition)]
    else:
        external_gradient_fn_per_subexperiment = [lambda x: 0.0]
    
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
    
    chemoattractant_source_location = np.array([])
    if chemoattractant_source_definition != None:
        chemoattractant_source_location = np.array(chemoattractant_source_definition[:2])
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('chemoattractant_source_location', chemoattractant_source_location), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    if run_experiments == True:
        eu.run_template_experiments(experiment_dir, experiment_name, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_visualizations=remake_visualizations)
        
        experiment_name_format_string = experiment_name + "_RPT={}"
        extracted_cell_motion_results = []
        group_centroid_per_timestep_per_repeat = []
        min_x_centroid_per_timestep_per_repeat = []
        max_x_centroid_per_timestep_per_repeat = []
        for rpt_number in xrange(num_experiment_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
            
            min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_per_timestep, centroids_and_persistences  = cu.analyze_cell_motion(relevant_environment, storefile_path, si, rpt_number)
            
            extracted_cell_motion_results += centroids_and_persistences
            group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
            min_x_centroid_per_timestep_per_repeat.append(min_x_centroid_per_timestep)
            max_x_centroid_per_timestep_per_repeat.append(max_x_centroid_per_timestep)
            # ================================================================
            
        datavis.present_collated_cell_motion_data(extracted_cell_motion_results, experiment_dir, total_time_in_hours)
        datavis.present_collated_group_centroid_drift_data(timestep_length, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_per_timestep_per_repeat, experiment_dir, total_time_in_hours)

    print "Done."
    
    return experiment_name

# =============================================================================

def corridor_migration_fixed_cells_vary_coa_cil(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_coas=[], test_cils=[], num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=4, auto_calculate_num_cells=True, num_cells=None, run_experiments=True, remake_visualizations=False):
    
    test_coas = sorted(test_coas)
    test_cils = sorted(test_cils)
    
    average_cell_persistence = np.zeros((len(test_cils), len(test_coas)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, test_cil in enumerate(test_cils):
        for yi, test_coa in enumerate(test_coas):
            print "========="
            print "COA = {}, CIL = {}".format(test_coa, test_cil)
            experiment_name = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=test_coa, default_cil=test_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, num_cells_width=num_cells_width, num_cells_height=num_cells_height, auto_calculate_num_cells=auto_calculate_num_cells, num_cells=num_cells, run_experiments=run_experiments, remake_visualizations=remake_visualizations)
            
            experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
            experiment_name_format_string = experiment_name + "_RPT={}"
            
            if run_experiments == False:
                if not os.path.exists(experiment_dir):
                    print "Experiment directory does not exist."
                    average_cell_persistence[xi, yi] = np.nan
                    continue
                else:
                    no_data = False
                    for rpt_number in range(num_experiment_repeats):
                        environment_name = experiment_name_format_string.format(rpt_number)
                        environment_dir = os.path.join(experiment_dir, environment_name)
                        if not os.path.exists(environment_dir):
                            no_data = True
                            print "Environment directory does not exist."
                            break
                    
                        storefile_path = eu.get_storefile_path(environment_dir)
                        if not os.path.isfile(storefile_path):
                            no_data = True
                            print "Storefile does not exist."
                            break
                        
                        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                        if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                            print "Simulation is not complete."
                            no_data = True
                            break
                    if no_data:
                        average_cell_persistence[xi, yi] = np.nan
                        continue
                
                    print "Data exists."
            all_cell_persistences = []
            for rpt_number in range(num_experiment_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
                storefile_path = eu.get_storefile_path(environment_dir)
                relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                
                centroids_and_persistences = cu.analyze_cell_motion(relevant_environment, storefile_path, 0, rpt_number)
                
                all_cell_persistences += [x[1] for x in centroids_and_persistences]
                
            avg_p = np.average(all_cell_persistences)
            average_cell_persistence[xi, yi] = avg_p



    print "========="
    
    if num_cells == None:
        num_cells = num_cells_height*num_cells_width
    
    datavis.graph_fixed_cells_vary_coa_cil_data(sub_experiment_number, test_cils, test_coas, average_cell_persistence, num_cells, num_cells_width, num_cells_height, save_dir=experiment_set_directory)
        
    print "Complete."

# =============================================================================

def corridor_migration_fixed_cells_vary_corridor_height(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_cells=[], test_heights=[], coa_dict=[], default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_visualizations=False):
    
    test_num_cells = sorted(test_num_cells)
    test_heights = sorted(test_heights)
    
    average_cell_persistence = np.zeros((len(test_num_cells), len(test_heights)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, tnc in enumerate(test_num_cells):
        for yi, th in enumerate(test_heights):
            calculated_width = int(np.ceil(float(tnc)/th))
            print "========="
            print "num_cells = {}, height = {}".format(tnc, th)
            experiment_name = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[tnc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, num_cells_width=calculated_width, num_cells_height=th, auto_calculate_num_cells=False, num_cells=tnc, run_experiments=run_experiments, remake_visualizations=remake_visualizations)
            
            experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
            experiment_name_format_string = experiment_name + "_RPT={}"
            
            if run_experiments == False:
                if not os.path.exists(experiment_dir):
                    print "Experiment directory does not exist."
                    average_cell_persistence[xi, yi] = np.nan
                    continue
                else:
                    no_data = False
                    for rpt_number in range(num_experiment_repeats):
                        environment_name = experiment_name_format_string.format(rpt_number)
                        environment_dir = os.path.join(experiment_dir, environment_name)
                        if not os.path.exists(environment_dir):
                            no_data = True
                            print "Environment directory does not exist."
                            break
                    
                        storefile_path = eu.get_storefile_path(environment_dir)
                        if not os.path.isfile(storefile_path):
                            no_data = True
                            print "Storefile does not exist."
                            break
                        
                        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                        if not (relevant_environment.simulation_complete() and (relevant_environment.curr_tpoint*relevant_environment.T/3600.) == total_time_in_hours):
                            print "Simulation is not complete."
                            no_data = True
                            break
                    if no_data:
                        average_cell_persistence[xi, yi] = np.nan
                        continue
                
                    print "Data exists."
            all_cell_persistences = []
            for rpt_number in range(num_experiment_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
                storefile_path = eu.get_storefile_path(environment_dir)
                relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                
                min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_per_timestep, centroids_and_persistences = cu.analyze_cell_motion(relevant_environment, storefile_path, 0, rpt_number)
                
                all_cell_persistences += [x[1] for x in centroids_and_persistences]
                
            avg_p = np.average(all_cell_persistences)
            average_cell_persistence[xi, yi] = avg_p



    print "========="
    
    #graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, num_cells_width, num_cells_height, save_dir=None)
    datavis.graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, save_dir=experiment_set_directory)
        
    print "Complete."
    
#==============================================================================

def corridor_migration_multigroup_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=1, cell_diameter=40, num_groups=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=1.2, default_intra_group_cil=20, default_inter_group_cil=40, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, animation_time_resolution='normal', remake_visualizations=False):    
    num_cells_in_group = num_cells_width*num_cells_height
    
    experiment_name_format_string = "multigroup_corridor_migration_{}_NC=({}, {})_NG={}".format("{}", num_cells_width, num_cells_height, num_groups)
    
    if no_randomization:
        parameter_dict.update([('randomization_scheme', None)])
        
    randomization_scheme = parameter_dict['randomization_scheme']
    experiment_name = fill_experiment_name_format_string_with_randomization_info(experiment_name_format_string, randomization_scheme, parameter_dict)
    
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
    parameter_override_dicts_per_sub_experiment = [[parameter_dict]*num_boxes]
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
        
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_rac_random_spikes', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', short_video_length_definition), ('short_video_duration', short_video_duration), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True), ('color_each_group_differently', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
    
    eu.run_template_experiments(experiment_dir, experiment_name, baseline_parameter_dict, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_visualizations=remake_visualizations)

    print "Done."    
    
    
