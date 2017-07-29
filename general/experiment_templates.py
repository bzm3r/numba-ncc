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

global_randomization_scheme_dict = {'m': 'kgtp_rac_multipliers', 'w': 'wipeout'}

def define_corridors_and_group_boxes_for_corridor_migration_tests(plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, physical_bdry_polygon_extra=10, origin_x_offset=10, origin_y_offset=10, box_x_offsets=[], box_y_offsets=[], make_only_migratory_corridor=False, migratory_corridor_size=[None, None], migratory_bdry_x_offset=None, migratory_bdry_y_offset=None):
    test_lists = [num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options]
    test_list_labels = ['num_cells_in_boxes', 'box_heights', 'box_widths', 'x_space_between_boxes', 'initial_x_placement_options', 'initial_y_placement_options']
    allowed_placement_options = ["CENTER", "ORIGIN", "OVERRIDE"]
    
    if len(box_x_offsets) == 0:
        box_x_offsets = [0.0 for x in range(num_boxes)]
    elif len(box_x_offsets) != num_boxes:
        raise StandardError("Incorrect number of box_x_offsets given!")
    if len(box_y_offsets) == 0:
        box_y_offsets = [0.0 for x in range(num_boxes)]
    elif len(box_y_offsets) != num_boxes:
        raise StandardError("Incorrect number of box_y_offsets given!")
    
    for test_list_label, test_list in zip(test_list_labels, test_lists):
        if test_list_label == "x_space_between_boxes":
            required_len = num_boxes - 1
        else:
            required_len = num_boxes
            
        if len(test_list) != required_len:
            raise StandardError("{} length is not the required length (should be {}, got {}).".format(test_list_label, required_len, len(test_list)))
            
        if test_list_label in ["initial_x_placement_options", "initial_y_placement_options"]:
            for placement_option in test_list:
                if placement_option not in allowed_placement_options:
                    raise StandardError("Given placement option not an allowed placement option!")
            
    for box_index, placement_option in enumerate(initial_x_placement_options):
        if placement_option == "ORIGIN":
            box_x_offsets[box_index] = origin_x_offset
        elif placement_option == "CENTER":
            box_x_offsets[box_index] = 0.5*(plate_width - 0.5*box_widths[0])

            
    for box_index, placement_option in enumerate(initial_y_placement_options):
        if placement_option == "ORIGIN":
            box_y_offsets[box_index] = origin_y_offset
        elif placement_option == "CENTER":
            box_y_offsets[box_index] = 0.5*(plate_height - 0.5*box_heights[0])
            

    make_migr_poly, make_phys_poly = False, False  
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

    if migratory_bdry_x_offset == None:
        migratory_bdry_x_offset = origin_x_offset
    if migratory_bdry_y_offset == None:
        migratory_bdry_y_offset = origin_y_offset
        
    space_migratory_bdry_polygon, space_physical_bdry_polygon = eu.make_space_polygons(make_migr_poly, make_phys_poly, width_migr_corridor, height_migr_corridor, migratory_bdry_x_offset, migratory_bdry_y_offset, physical_bdry_polygon_extra=physical_bdry_polygon_extra)
    
    return np.arange(num_boxes), box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon
        
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

# ===========================================================================

def single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, parameter_dict, base_output_dir="A:\\numba-ncc\\output\\", no_randomization=False, total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, justify_parameters=True, run_experiments=True, remake_graphs=False, remake_animation=False, show_centroid_trail=False, show_randomized_nodes=False, plate_width=1000, plate_height=1000, global_scale=1, convergence_test=False, Tr_vs_Tp_test=False, do_final_analysis=True):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    if convergence_test:
        experiment_name_format_string = "convergence_{}_NN={}_".format(sub_experiment_number, parameter_dict['num_nodes']) +"{}"
    elif Tr_vs_Tp_test:
        experiment_name_format_string = "Tr_vs_Tp_{}_Tr={}_".format(sub_experiment_number, int(parameter_dict["randomization_time_mean"])) +"{}"
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

    #num_boxes, num_cells_in_boxes, box_heights, box_widths,x_space_between_boxes, plate_width, plate_height, "CENTER", "CENTER"
    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_corridors_and_group_boxes_for_corridor_migration_tests(plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, ["CENTER"], ["CENTER"])
    
    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, 'cell_placement_method': ""}
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', np.sqrt(global_scale)*312.5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', show_centroid_trail), ('show_rac_random_spikes', show_randomized_nodes), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, run_experiments=run_experiments, new_num_timesteps=num_timesteps, justify_parameters=justify_parameters, remake_graphs=remake_graphs, remake_animation=remake_animation)
    
    if do_final_analysis:
        centroids_persistences_speeds_per_repeat = []
        for rpt_number in xrange(num_experiment_repeats):
            environment_name = "RPT={}".format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
            
            print "Analyzing repeat number: ", rpt_number
            time_unit, centroids_persistences_speeds = cu.analyze_single_cell_motion(relevant_environment, storefile_path, no_randomization)
            
            centroids_persistences_speeds_per_repeat.append(centroids_persistences_speeds)
            # ================================================================
            
        datavis.present_collated_single_cell_motion_data(centroids_persistences_speeds_per_repeat, experiment_dir, total_time_in_hours, time_unit)

    print "Done."
    return experiment_name
    

# ===========================================================================

def collate_single_cell_test_data(num_experiment_repeats, experiment_dir):
    cell_persistence_ratios_per_repeat = []
    cell_persistence_times_per_repeat = []
    cell_speeds_per_repeat = []
    average_cell_rac_activity_per_repeat = []
    average_cell_rho_activity_per_repeat = []
    average_cell_rac_inactivity_per_repeat = []
    average_cell_rho_inactivity_per_repeat = []
    
    for rpt_number in xrange(num_experiment_repeats):
        environment_name = "RPT={}".format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        
        data_dict_pickle_path = os.path.join(environment_dir, "general_data_dict.pkl")
        data_dict = None
        with open(data_dict_pickle_path, 'rb') as f:
            data_dict = dill.load(f)
            
        if data_dict == None:
            raise StandardError("Unable to load data_dict at path: {}".format(data_dict_pickle_path))
        
        cell_persistence_ratios_per_repeat.append(data_dict["all_cell_persistence_ratios"][0])
        cell_persistence_times_per_repeat.append(data_dict["all_cell_persistence_times"][0])
        cell_speeds_per_repeat.append(data_dict["all_cell_latter_half_speeds"][0])
        
        average_cell_rac_activity_per_repeat.append(data_dict["average_rac_membrane_active_{}".format(0)])
        average_cell_rho_activity_per_repeat.append(data_dict["average_rho_membrane_active_{}".format(0)])
        average_cell_rac_inactivity_per_repeat.append(data_dict["average_rac_membrane_inactive_{}".format(0)])
        average_cell_rho_inactivity_per_repeat.append(data_dict["average_rho_membrane_inactive_{}".format(0)])
    
    return cell_persistence_ratios_per_repeat, cell_persistence_times_per_repeat, cell_speeds_per_repeat, average_cell_rac_activity_per_repeat, average_cell_rho_activity_per_repeat, average_cell_rac_inactivity_per_repeat, average_cell_rho_inactivity_per_repeat

# ===========================================================================

def convergence_test(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_nodes=np.array([]), default_coa=0.0, default_cil=0.0, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False):
    
    num_tests = len(test_num_nodes)
    
    average_cell_speeds = np.zeros(num_tests, dtype=np.float64)
    average_active_racs = np.zeros(num_tests, dtype=np.float64)
    average_active_rhos = np.zeros(num_tests, dtype=np.float64)
    average_inactive_racs = np.zeros(num_tests, dtype=np.float64)
    average_inactive_rhos = np.zeros(num_tests, dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, nn in enumerate(test_num_nodes):
        print "========="
        print "nn = {}".format(nn)
        
        this_parameter_dict = copy.deepcopy(parameter_dict)
        this_parameter_dict.update([("num_nodes", nn)])
        experiment_name = single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, this_parameter_dict, base_output_dir=base_output_dir, no_randomization=no_randomization, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, show_centroid_trail=True, convergence_test=True, do_final_analysis=do_final_analysis)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        cell_persistence_ratios_per_repeat, cell_persistence_times_per_repeat, cell_speeds_per_repeat, average_cell_rac_activity_per_repeat, average_cell_rho_activity_per_repeat, average_cell_rac_inactivity_per_repeat, average_cell_rho_inactivity_per_repeat = collate_single_cell_test_data(1, experiment_dir)
        
        average_cell_speeds[xi] = cell_speeds_per_repeat[0]
        average_active_racs[xi] = average_cell_rac_activity_per_repeat[0]
        average_active_rhos[xi] = average_cell_rho_activity_per_repeat[0]
        average_inactive_racs[xi] = average_cell_rac_inactivity_per_repeat[0]
        average_inactive_rhos[xi] = average_cell_rho_inactivity_per_repeat[0]


    print "========="
    
    datavis.graph_convergence_test_data(sub_experiment_number, test_num_nodes, average_cell_speeds, average_active_racs, average_active_rhos, average_inactive_racs, average_inactive_rhos, save_dir=experiment_set_directory)
        
        
    print "Complete."

# ============================================================================

def Tr_vs_Tp_test(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_Trs=np.array([]), num_experiment_repeats=50, default_coa=0.0, default_cil=0.0, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False):
    
    num_tests = len(test_Trs)
    
    average_cell_persistence_times = np.zeros(num_tests, dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, Tr in enumerate(test_Trs):
        print "========="
        print "Tr = {}".format(Tr)
        
        this_parameter_dict = copy.deepcopy(parameter_dict)
        this_parameter_dict.update([("randomization_time_mean", Tr)])
        experiment_name = single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, this_parameter_dict, base_output_dir=base_output_dir, no_randomization=no_randomization, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, show_centroid_trail=True, convergence_test=False, Tr_vs_Tp_test=True, do_final_analysis=do_final_analysis)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        cell_persistence_ratios_per_repeat, cell_persistence_times_per_repeat, cell_speeds_per_repeat, average_cell_rac_activity_per_repeat, average_cell_rho_activity_per_repeat, average_cell_rac_inactivity_per_repeat, average_cell_rho_inactivity_per_repeat = collate_single_cell_test_data(experiment_dir)
        
        average_cell_persistence_times[xi] = np.average(cell_persistence_times_per_repeat)


    print "========="
    
    datavis.graph_Tr_vs_Tp_test_data(sub_experiment_number, test_Trs, average_cell_persistence_times, save_dir=experiment_set_directory)

        
    print "Complete."
    
# ============================================================================

def two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=0.8, remake_graphs=False, remake_animation=False):    
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
    plate_width, plate_height = 10*cell_diameter*1.2, 3*cell_diameter

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_group_boxes_and_corridors(num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, plate_width, plate_height, "OVERRIDE", "ORIGIN", physical_bdry_polygon_extra=20, box_x_offsets=[10 + 3*cell_diameter, 10 + (3 + 1 + 2)*cell_diameter],  migratory_corridor_size=[10*cell_diameter, migr_bdry_height_factor*cell_diameter], make_only_migratory_corridor=True, origin_y_offset=25)
    
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
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)

    print "Done."
    
# ============================================================================

def block_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, remake_graphs=False, remake_animation=False):
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
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)
    
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

def many_cells_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, auto_calculate_num_cells=True, num_cells=None, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_rac_random_spikes=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if auto_calculate_num_cells:
        num_cells = box_height*box_width
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
    box_heights = [box_height*cell_diameter]
    box_widths = [box_width*cell_diameter]

    x_space_between_boxes = []
    plate_width, plate_height = 1000, 1000

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon = define_corridors_and_group_boxes_for_corridor_migration_tests(plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, ["CENTER"], ["CENTER"])
    
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
        
    animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', show_centroid_trail), ('show_rac_random_spikes', show_rac_random_spikes), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 1000.0*timestep_length), ('short_video_duration', 5.0), ('timestep_length', timestep_length), ('fps', 30), ('string_together_pictures_into_animation', True)])
    
    produce_intermediate_visuals = produce_intermediate_visuals_array(num_timesteps, timesteps_between_generation_of_intermediate_visuals)
        
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)
    
    experiment_name_format_string = experiment_name + "_RPT={}"
    cell_centroids_persistences_speeds_per_repeat = []
    group_centroid_per_timestep_per_repeat = []
    group_centroid_x_per_timestep_per_repeat = []
    min_x_centroid_per_timestep_per_repeat = []
    max_x_centroid_per_timestep_per_repeat = []
    group_speed_per_timestep_per_repeat = []
    group_persistence_ratio_per_repeat = []
    group_persistence_time_per_repeat = []
    
    for rpt_number in xrange(num_experiment_repeats):
        print "Analyzing repeat {}...".format(rpt_number)
        environment_name = experiment_name_format_string.format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        storefile_path = eu.get_storefile_path(environment_dir)
        relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
        
        time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_centroid_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds   = cu.analyze_cell_motion(relevant_environment, storefile_path, si, rpt_number)
        
        cell_centroids_persistences_speeds_per_repeat.append(centroids_persistences_speeds)
        group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
        group_centroid_x_per_timestep_per_repeat.append(group_centroid_x_per_timestep)
        min_x_centroid_per_timestep_per_repeat.append(min_x_centroid_per_timestep)
        max_x_centroid_per_timestep_per_repeat.append(max_x_centroid_per_timestep)
        group_speed_per_timestep_per_repeat.append(group_speed_per_timestep)
        
        group_persistence_ratio_per_repeat.append(group_persistence_ratio)
        
        group_persistence_time_per_repeat.append(group_persistence_time)
        # ================================================================
    
    datavis.present_collated_cell_motion_data(time_unit, cell_centroids_persistences_speeds_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours)
    
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

def coa_factor_variation_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_coas=[], default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells_to_test=[], skip_low_coa=False, max_normalized_group_area=3.0, run_experiments=True, remake_graphs=False, remake_animation=False):
    
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
                experiment_name = many_cells_coa_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=test_coa, default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=square_size, box_height=square_size, auto_calculate_num_cells=False, num_cells=num_cells, remake_graphs=remake_graphs, remake_animation=remake_animation)
                
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
    
    for rpt_number in xrange(num_experiment_repeats):
        environment_name = "RPT={}".format(rpt_number)
        environment_dir = os.path.join(experiment_dir, environment_name)
        
        data_dict_pickle_path = os.path.join(environment_dir, "general_data_dict.pkl")
        data_dict = None
        with open(data_dict_pickle_path, 'rb') as f:
            data_dict = dill.load(f)
            
        if data_dict == None:
            raise StandardError("Unable to load data_dict at path: {}".format(data_dict_pickle_path))
        
        
        #storefile_path = eu.get_storefile_path(environment_dir)
        #relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
        
        #time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_centroid_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds_protrusionlifetimes  = cu.analyze_cell_motion(relevant_environment, storefile_path, si, rpt_number)
        
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
        transient_end_times_per_repeat.append(data_dict["transient_end"])
        
    
    return all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat

def corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, corridor_height=None, box_width=4, box_height=4, box_y_placement_factor=0.0, cell_placement_method="", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25, box_x_offset=0, num_cells=0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if num_cells == 0:
        raise StandardError("No cells!")
    
    if corridor_height == None:
        corridor_height = box_height
        
    if corridor_height < box_height:
        raise StandardError("Corridor height is less than box height!")
        
    if corridor_height == box_height:
        box_y_placement_factor = 0.0
        
    accepted_cell_placement_methods = ["", "r"]
    if cell_placement_method not in accepted_cell_placement_methods:
        raise StandardError("Unknown placement method given: {}, expected one of {}".format(cell_placement_method, accepted_cell_placement_methods))
        
    if cell_placement_method == "":
        experiment_name_format_string = "cm_{}_{}_NC=({}, {}, {}, {}, {}){}_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, corridor_height, box_y_placement_factor, cell_placement_method, default_coa, default_cil)
    else:
        experiment_name_format_string = "cm_{}_{}_NC=({}, {}, {}, {}, {})({}, {}, {})_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, corridor_height, box_y_placement_factor, cell_placement_method, max_placement_distance_factor, init_random_cell_placement_x_factor, default_coa, default_cil)
        
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

    plate_width, plate_height = min(2000, max(1000, box_widths[0]*8)), (corridor_height*cell_diameter + 40 + 100)

    origin_y_offset = 55
    physical_bdry_polygon_extra = 20
    
    initial_x_placement_options = ["ORIGIN" for x in range(num_boxes)]
    initial_y_placement_options = ["OVERRIDE" for x in range(num_boxes)]
    
    box_y_offsets = [box_y_placement_factor*(corridor_height - box_height)*cell_diameter + origin_y_offset]

    boxes, box_x_offsets, box_y_offsets, space_migratory_bdry_polygon, space_physical_bdry_polygon  = define_corridors_and_group_boxes_for_corridor_migration_tests(plate_width, plate_height, num_boxes, num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes, initial_x_placement_options, initial_y_placement_options, box_y_offsets=box_y_offsets, physical_bdry_polygon_extra=physical_bdry_polygon_extra, origin_y_offset=origin_y_offset, migratory_corridor_size=[box_widths[0]*100, corridor_height*cell_diameter])

    parameter_dict['space_physical_bdry_polygon'] = space_physical_bdry_polygon
    parameter_dict['space_migratory_bdry_polygon'] = space_migratory_bdry_polygon
    
    environment_wide_variable_defns = {'num_timesteps': num_timesteps, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': timestep_length, 'verbose': verbose, 'integration_params': integration_params, 'max_timepoints_on_ram': max_timepoints_on_ram, 'seed': seed, 'allowed_drift_before_geometry_recalc': allowed_drift_before_geometry_recalc, "cell_placement_method": cell_placement_method, "max_placement_distance_factor": max_placement_distance_factor, "init_random_cell_placement_x_factor": init_random_cell_placement_x_factor}
    
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
    
    eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, run_experiments=run_experiments, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)
        
    if do_final_analysis:
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
            # ================================================================
        
        #time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22, general_data_structure=None
        time_unit = "min."
        datavis.present_collated_cell_motion_data(time_unit, np.array(all_cell_centroids_per_repeat), np.array(all_cell_persistence_ratios_per_repeat), np.array(all_cell_persistence_times_per_repeat), np.array(all_cell_speeds_per_repeat), all_cell_protrusion_lifetimes_and_directions_per_repeat, np.array(group_centroid_per_timestep_per_repeat), np.array(group_persistence_ratio_per_repeat), np.array(group_persistence_time_per_repeat), experiment_dir, total_time_in_hours)
        datavis.present_collated_group_centroid_drift_data(timestep_length, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, experiment_dir, total_time_in_hours)

    print "Done."
    
    return experiment_name


# ============================================================================

def chemoattractant_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, auto_calculate_num_cells=True, num_cells=None, run_experiments=True, chemoattractant_source_definition=None, autocalculate_chemoattractant_source_y_position=True, migratory_box_width=2000, migratory_box_height=400, remake_graphs=False, remake_animation=False):
    cell_diameter = 2*parameter_dict["init_cell_radius"]/1e-6
    
    if auto_calculate_num_cells:
        num_cells = box_height*box_width
    else:
        if num_cells == None:
            raise StandardError("Auto-calculation of cell number turned off, but num_cells not given!")
            
    experiment_name_format_string = "chatr_{}_{}_NC=({}, {}, {})_COA={}_CIL={}".format(sub_experiment_number, "{}", num_cells, box_width, box_height, default_coa, default_cil)
    
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
        eu.run_template_experiments(experiment_dir, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)
        
        experiment_name_format_string = experiment_name + "_RPT={}"
        cell_centroids_persistences_speeds_per_repeat = []
        group_centroid_per_timestep_per_repeat = []
        group_centroid_x_per_timestep_per_repeat = []
        min_x_centroid_per_timestep_per_repeat = []
        max_x_centroid_per_timestep_per_repeat = []
        group_speed_per_timestep_per_repeat = []
        group_persistence_ratio_per_repeat = []
        group_persistence_time_per_repeat = []
        
        for rpt_number in xrange(num_experiment_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
            
            time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_centroid_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds   = cu.analyze_cell_motion(relevant_environment, storefile_path, si, rpt_number)
            
            cell_centroids_persistences_speeds_per_repeat.append(centroids_persistences_speeds)
            group_centroid_per_timestep_per_repeat.append(group_centroid_per_timestep)
            group_centroid_x_per_timestep_per_repeat.append(group_centroid_x_per_timestep)
            min_x_centroid_per_timestep_per_repeat.append(min_x_centroid_per_timestep)
            max_x_centroid_per_timestep_per_repeat.append(max_x_centroid_per_timestep)
            group_speed_per_timestep_per_repeat.append(group_speed_per_timestep)
            
            group_persistence_ratio_per_repeat.append(group_persistence_ratio)
            
            group_persistence_time_per_repeat.append(group_persistence_time)
            # ================================================================
        
        datavis.present_collated_cell_motion_data(time_unit, cell_centroids_persistences_speeds_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours)
        datavis.present_collated_group_centroid_drift_data(timestep_length, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, experiment_dir, total_time_in_hours)

    print "Done."
    
    return experiment_name

# =============================================================================

def corridor_migration_fixed_cells_vary_coa_cil(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_coas=[], test_cils=[], num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, auto_calculate_num_cells=True, num_cells=None, run_experiments=True, remake_graphs=False, remake_animation=False):
    
    test_coas = sorted(test_coas)
    test_cils = sorted(test_cils)
    
    average_cell_persistence = np.zeros((len(test_cils), len(test_coas)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, test_cil in enumerate(test_cils):
        for yi, test_coa in enumerate(test_coas):
            print "========="
            print "COA = {}, CIL = {}".format(test_coa, test_cil)
            experiment_name = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=test_coa, default_cil=test_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=box_width, box_height=box_height, auto_calculate_num_cells=auto_calculate_num_cells, num_cells=num_cells, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation)
            
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
                
                time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds  = cu.analyze_cell_motion(relevant_environment, storefile_path, 0, rpt_number)
                
                all_cell_persistences += [x[1] for x in centroids_persistences_speeds]
                
            avg_p = np.average(all_cell_persistences)
            average_cell_persistence[xi, yi] = avg_p



    print "========="
    
    if num_cells == None:
        num_cells = box_height*box_width
    
    datavis.graph_fixed_cells_vary_coa_cil_data(sub_experiment_number, test_cils, test_coas, average_cell_persistence, num_cells, box_width, box_height, save_dir=experiment_set_directory)
        
    print "Complete."

# =============================================================================

def corridor_migration_fixed_cells_vary_corridor_height(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_cells=[], test_heights=[], coa_dict=[], default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis_after_running_experiments=False):
    
    test_num_cells = sorted(test_num_cells)
    test_heights = sorted(test_heights)
    
    average_cell_persistence_ratios = np.zeros((len(test_num_cells), len(test_heights)), dtype=np.float64)
    average_cell_persistence_times = np.zeros((len(test_num_cells), len(test_heights)), dtype=np.float64)
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, tnc in enumerate(test_num_cells):
        for yi, th in enumerate(test_heights):
            calculated_width = int(np.ceil(float(tnc)/th))
            print "========="
            print "num_cells = {}, height = {}".format(tnc, th)
            experiment_name = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[tnc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=calculated_width, box_height=th, num_cells=tnc, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis_after_running_experiments)
            
            experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
            experiment_name_format_string = experiment_name + "_RPT={}"
            
            if run_experiments == False:
                if not os.path.exists(experiment_dir):
                    print "Experiment directory does not exist."
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
                        average_cell_persistence_ratios[xi, yi] = np.nan
                        average_cell_persistence_times[xi, yi] = np.nan
                        continue
                
                    print "Data exists."
            all_cell_persistence_ratios = []
            all_cell_persistence_times = []
            for rpt_number in range(num_experiment_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
                storefile_path = eu.get_storefile_path(environment_dir)
                relevant_environment = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)
                
                time_unit, min_x_centroid_per_timestep, max_x_centroid_per_timestep, group_centroid_x_per_timestep, group_centroid_per_timestep, group_speed_per_timestep, group_persistence_ratio, group_persistence_time, centroids_persistences_speeds   = cu.analyze_cell_motion(relevant_environment, storefile_path, 0, rpt_number)
                
                all_cell_persistence_ratios += [x[1][0] for x in centroids_persistences_speeds]
                all_cell_persistence_times += [x[1][1] for x in centroids_persistences_speeds]
                
            avg_pr = np.average(all_cell_persistence_ratios)
            avg_pt = np.average(all_cell_persistence_times)
            average_cell_persistence_ratios[xi, yi] = avg_pr
            average_cell_persistence_times[xi, yi] = avg_pt



    print "========="
    
    #graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    datavis.graph_confinement_data_persistence_ratios(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_ratios, save_dir=experiment_set_directory)
    #datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print "Complete."
    
    
# =============================================================================

def corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_heights=[], test_num_cells=[], coa_dict={}, default_cil=40.0, num_experiment_repeats=1, particular_repeats=[], timesteps_between_generation_of_intermediate_visuals=None, graph_x_dimension="test_num_cells", produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False):
    
    assert(len(test_num_cells) == len(test_heights))
    
    if graph_x_dimension == "test_num_cells":
        test_num_cells = sorted(test_num_cells)
    elif graph_x_dimension == "test_heights":
        test_heights = sorted(test_heights)
    else:
        raise StandardError("Unexpected graph_x_dimension: {}".format(graph_x_dimension))
    
    num_tests = len(test_num_cells)
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    for xi, tnc_and_th in enumerate(zip(test_num_cells, test_heights)):
        tnc, th = tnc_and_th
        tw = max(1, int(np.ceil(float(tnc)/th)))
        print "========="
        print "num_cells = {}, height = {}, width = {}".format(tnc, th, tw)
        
        experiment_name = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[tnc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=tw, box_height=th, num_cells=tnc, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat

    print "========="
    
    #graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    datavis.graph_cell_number_change_data(sub_experiment_number, test_num_cells, test_heights, graph_x_dimension, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, experiment_set_label, save_dir=experiment_set_directory)
        
    #datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print "Complete."
    
# =============================================================================

def corridor_migration_init_conditions_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="", no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, test_num_cells=[], test_heights=[], test_widths=[], corridor_heights=[], box_placement_factors=[], coa_dict={}, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False):
    
    num_tests = len(test_num_cells)
    assert(np.all([len(x) == num_tests for x in [test_heights, test_widths, corridor_heights, box_placement_factors]]))
    
    group_persistence_ratios = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    group_persistence_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    fit_group_x_velocities = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    cell_separations = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    transient_end_times = np.zeros((num_tests, num_experiment_repeats), dtype=np.float64)
    
    experiment_set_directory = eu.get_experiment_set_directory_path(base_output_dir, date_str, experiment_number)
    
    tests = zip(test_num_cells, test_heights, test_widths, corridor_heights, box_placement_factors)
    for xi, nc_th_tw_ch_bpy in enumerate(tests):
        nc, th, tw, ch, bpy = nc_th_tw_ch_bpy
        print "========="
        print "nc = {}, h = {}, w = {}, ch = {}, bpy = {}".format(nc, th, tw, ch, bpy)
        
        experiment_name = corridor_migration_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=no_randomization, base_output_dir=base_output_dir, total_time_in_hours=total_time_in_hours, timestep_length=timestep_length, verbose=verbose, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[nc], default_cil=default_cil, num_experiment_repeats=num_experiment_repeats, timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, box_width=tw, box_height=th, num_cells=nc, corridor_height=ch, box_y_placement_factor=bpy, run_experiments=run_experiments, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)
            
        experiment_dir = eu.get_template_experiment_directory_path(base_output_dir, date_str, experiment_number, experiment_name)
        
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat,all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat = collate_final_analysis_data(num_experiment_repeats, experiment_dir)
        
        group_persistence_ratios[xi] = group_persistence_ratio_per_repeat
        group_persistence_times[xi] = group_persistence_time_per_repeat
        fit_group_x_velocities[xi] = fit_group_x_velocity_per_repeat
        cell_separations[xi] = cell_separations_per_repeat
        transient_end_times[xi] = transient_end_times_per_repeat

    print "========="
    
    #graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, num_cells, box_width, box_height, save_dir=None)
    
    datavis.graph_init_condition_change_data(sub_experiment_number, tests, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, transient_end_times, experiment_set_label, save_dir=experiment_set_directory)
        
    #datavis.graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence_times, save_dir=experiment_set_directory)
        
    print "Complete."
    
#==============================================================================

def corridor_migration_multigroup_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_dict, no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, box_width=2, box_height=1, cell_diameter=40, num_groups=2, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=1.2, default_intra_group_cil=20, default_inter_group_cil=40, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, animation_time_resolution='normal', remake_graphs=False, remake_animation=False):    
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
    
    eu.run_template_experiments(experiment_dir, baseline_parameter_dict, parameter_dict, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=num_experiment_repeats, animation_settings=animation_settings, produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, full_print=full_print, delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env, extend_simulation=True, new_num_timesteps=num_timesteps, remake_graphs=remake_graphs, remake_animation=remake_animation)

    print "Done."    
    
    
