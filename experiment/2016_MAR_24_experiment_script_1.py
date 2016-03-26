from __future__ import division
import numpy as np
import os
import shutil
import time
from general.experiment_executor import *
import analysis
import general
import visualization.datavis as datavis

EXPERIMENT_NUMBER = 2
experiment_description = "large group"

BASE_OUTPUT_DIR = "A:\\numba-ncc\\output\\"
DATE_STR = "2016_MAR_24"

experiment_dir = get_experiment_directory_path(BASE_OUTPUT_DIR, DATE_STR, EXPERIMENT_NUMBER)

TOTAL_TIME = 50*60*60#32000#(60*60)*3
TIMESTEP_LENGTH = (1/0.5)                                                                                                                                                                                                                                                                                                                                           
NUM_TIMESTEPS = int(TOTAL_TIME/TIMESTEP_LENGTH)
NUM_NODES_PER_CELL = 16

CELL_DIAMETER = 40
NUM_BOXES = 1
NUM_CELLS_IN_BOXES = [50]

boxes = np.arange(NUM_BOXES)
box_heights = [5*CELL_DIAMETER, 4*CELL_DIAMETER]
box_widths = [10*CELL_DIAMETER, 4*CELL_DIAMETER]

x_space_between_boxes = [2*CELL_DIAMETER, 2*CELL_DIAMETER, 2*CELL_DIAMETER]

physical_bdry_polygon_extra = 5

x_offset = 10#500
y_offset = 10

box_x_offsets = [0]*NUM_BOXES
for bi in boxes:
    if bi == 0:
        box_x_offsets[bi] = x_offset #- 1*CELL_DIAMETER - x_space_between_boxes[0]
    else:
        box_x_offsets[bi] = x_offset + x_space_between_boxes[bi-1] #- 1*CELL_DIAMETER - x_space_between_boxes[0]
        
box_y_offsets = [y_offset, y_offset - 2*CELL_DIAMETER, y_offset + 2*CELL_DIAMETER, y_offset + 1.5*CELL_DIAMETER]

WIDTH_MIGR_CORRIDOR = 4*box_widths[0]
HEIGHT_MIGR_CORRIDOR = box_heights[0]

make_migr_poly = True
make_phys_poly = False

space_migratory_bdry_polygon, space_physical_bdry_polygon = make_space_polygons(make_migr_poly, make_phys_poly, WIDTH_MIGR_CORRIDOR, HEIGHT_MIGR_CORRIDOR, 10, 10)

if make_migr_poly == False:
    WIDTH_MIGR_CORRIDOR = None
    HEIGHT_MIGR_CORRIDOR = None

environment_wide_variable_defns = {'num_timesteps': NUM_TIMESTEPS, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': TIMESTEP_LENGTH, 'num_nodes_per_cell': NUM_NODES_PER_CELL, 'verbose': True, 'integration_params': {'atol': 1e-4, 'rtol': 1e-4}, 'closeness_dist_squared_criteria': (1e-6)**2}

base_parameter_dict = dict([('halfmax_coa_sensing_dist_multiplier', 4.4), ('kdgdi_rac_estimate_multiplier', 0.2), ('kdgdi_rho_estimate_multiplier', 0.2), ('kgdi_rac_estimate_multiplier', 1), ('kgdi_rho_estimate_multiplier', 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 500), ('kdgtp_rac_multiplier', 20), ('kdgtp_rho_mediated_rac_inhib_multiplier', 3000), ('kdgtp_rho_multiplier', 20), ('kgtp_rac_autoact_multiplier', 250), ('kgtp_rac_multiplier', 20), ('kgtp_rho_autoact_multiplier', 125), ('kgtp_rho_multiplier', 20), ('max_protrusive_node_velocity', 1e-06), ('randomization', False), ('randomization_centre', 0.15), ('randomization_depth', 1.0), ('randomization_function_type', 0), ('randomization_scheme', 0), ('randomization_time_mean', 20), ('randomization_time_variance_factor', 0.25), ('randomization_width', 10), ('randomization_width_baseline', 2), ('randomization_width_halfmax_threshold', 0.3), ('randomization_width_hf_exponent', 3), ('sigma_rac', 2e-05), ('sigma_rho_multiplier', 0.2), ('force_rac_exp', 3), ('force_rho_exp', 3), ('force_rac_threshold_multiplier', 0.5), ('force_rho_threshold_multiplier', 0.5), ('skip_dynamics', False), ('stiffness_cytoplasmic', 100), ('stiffness_edge', 5e-10), ('tension_fn_type', 0), ('tension_mediated_rac_hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('threshold_rac_autoact_multiplier', 0.5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5), ('threshold_rho_autoact_multiplier', 0.5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5), ('coa_sensitivity_percent_drop_over_cell_diameter', 0.25), ('coa_belt_offset_multiplier', 1.5)])
    
cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = []#[0.025]
intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = []
biased_rgtpase_distrib_defn_dicts = []
parameter_override_dicts_per_sub_experiment = []
experiment_descriptions_per_subexperiment = []
external_gradient_fn_per_subexperiment = []

# sub-experiment 0
experiment_descriptions_per_subexperiment += ['''set randomization_time_mean = 30, which means that mean run time is 30 minutes''']
coa = 2*0.05*0.5*0.5
other_cil = 4
self_cil = 3
cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment += [[dict([(x, coa) for x in boxes])]*NUM_BOXES]#[1.85]
intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment += [{0: {0: self_cil}}]
biased_rgtpase_distrib_defn_dicts += [[{'default': ['unbiased random', np.array([-np.pi/4, np.pi/4]) + np.pi/32 + np.pi, 0.1]}]]
external_gradient_fn_per_subexperiment += [lambda x: 0]
parameter_override_dicts_per_sub_experiment += [[update_pd_with_keyvalue_tuples(base_parameter_dict, [('halfmax_coa_sensing_dist_multiplier', 4.4, 4.4), ('kdgdi_rac_estimate_multiplier', 0.2, 1.0*0.2), ('kdgdi_rho_estimate_multiplier', 0.2, 1.0*0.2), ('kgdi_rac_estimate_multiplier', 1, 1.0), ('kgdi_rho_estimate_multiplier', 1, 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 500, 800), ('kdgtp_rac_multiplier', 20, 20*(2/3.)), ('kdgtp_rho_mediated_rac_inhib_multiplier', 3000, 500), ('kdgtp_rho_multiplier', 20, 20*(2/3.)), ('kgtp_rac_autoact_multiplier', 250, 1.35*200*1.2), ('kgtp_rac_multiplier', 20, 20), ('kgtp_rho_autoact_multiplier', 125, 105), ('kgtp_rho_multiplier', 20, 20), ('max_protrusive_node_velocity', 1e-06, 0.25e-6), ('randomization', False, True), ('randomization_centre', 0.15, 0.15), ('randomization_depth', 1.0, 1.0), ('randomization_function_type', 0, 0), ('randomization_scheme', 0, 0), ('randomization_time_mean', 20, 30), ('randomization_time_variance_factor', 0.25, 0.15), ('randomization_width', 10, 10), ('randomization_width_baseline', 2, 2), ('randomization_width_halfmax_threshold', 0.3, 0.3), ('randomization_width_hf_exponent', 3, 3), ('sigma_rac', 2e-05, 2.5e-05), ('sigma_rho_multiplier', 0.2, 0.2), ('force_rac_exp', 3, 3), ('force_rho_exp', 3, 3), ('force_rac_threshold_multiplier', 0.5, 0.9*1.75), ('force_rho_threshold_multiplier', 0.5, 0.9*1.75), ('skip_dynamics', False, False), ('stiffness_edge', 5e-10, 3.8e-10), ('tension_fn_type', 0, 5), ('tension_mediated_rac_inhibition_half_strain', 0.05, 0.05*0.6), ('threshold_rac_autoact_multiplier', 0.5, 0.5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5, 0.5), ('threshold_rho_autoact_multiplier', 0.5, 0.3), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5, 0.5), ('coa_sensitivity_percent_drop_over_cell_diameter', 0.25, -1), ('coa_belt_offset_multiplier', 1.5, 1.5)]), update_pd_with_keyvalue_tuples(base_parameter_dict, [('skip_dynamics', False, True)])]]

num_sub_experiments = len(parameter_override_dicts_per_sub_experiment)

environment_name_format_strings = []
user_cell_group_defns_per_subexperiment = []
for si in xrange(num_sub_experiments):
    user_cell_group_defns = []
    
    environment_name_format_strings.append( form_base_environment_name_format_string(EXPERIMENT_NUMBER, np.sum(NUM_CELLS_IN_BOXES), str(np.round(TOTAL_TIME/(3600.), decimals=2))+"h", NUM_TIMESTEPS, NUM_NODES_PER_CELL, HEIGHT_MIGR_CORRIDOR, WIDTH_MIGR_CORRIDOR))
    
    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]
        
        cell_group_dict = {'cell_group_name': bi, 'num_cells': NUM_CELLS_IN_BOXES[bi], 'init_cell_radius': CELL_DIAMETER*0.5*1e-6, 'C_total': 3e6, 'H_total': 1.5e6, 'cell_group_bounding_box': np.array([this_box_x_offset, this_box_x_offset + this_box_width, this_box_y_offset, this_box_height + this_box_y_offset])*1e-6, 'intercellular_contact_factor_magnitudes_defn': intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[si][bi], 'cell_dependent_coa_signal_strengths_defn': cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[si][bi], 'biased_rgtpase_distrib_defns': biased_rgtpase_distrib_defn_dicts[si][bi], 'parameter_override_dict': parameter_override_dicts_per_sub_experiment[si][bi]} 
        
        user_cell_group_defns.append(cell_group_dict)
        
    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

# =======================================================================

if __name__ == '__main__':
    RUN_EXPERIMENTS = True
    RUN_ANALYSIS = False
    NUM_EXPERIMENT_REPEATS = 1
    if RUN_EXPERIMENTS == True:
        global_scale = 1
        plate_height = 40*5*1.2
        plate_width = 4*40*10*1.2
        
        animation_settings = dict([('global_scale', global_scale), ('plate_height_in_micrometers', plate_height), ('plate_width_in_micrometers', plate_width), ('velocity_scale', 1), ('rgtpase_scale', global_scale*62.5*5), ('coa_scale', global_scale*62.5), ('show_velocities', False), ('show_rgtpase', True), ('show_centroid_trail', False), ('show_coa', False), ('color_each_group_differently', False), ('only_show_cells', []), ('polygon_line_width', 1),  ('space_physical_bdry_polygon', space_physical_bdry_polygon), ('space_migratory_bdry_polygon', space_migratory_bdry_polygon), ('short_video_length_definition', 2000.0*TIMESTEP_LENGTH), ('short_video_duration', 5.0), ('timestep_length', TIMESTEP_LENGTH), ('fps', 30), ('string_together_pictures_into_animation', True), ('sequential', True), ('num_processes', 4)])
    
        run_experiments(experiment_dir, environment_name_format_strings, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=NUM_EXPERIMENT_REPEATS, elapsed_timesteps_before_producing_intermediate_graphs=1000, elapsed_timesteps_before_producing_intermediate_animations=1000, animation_settings=animation_settings, produce_intermediate_visuals=True, produce_final_visuals=True, full_print=True, elapsed_timesteps_before_hard_saving_env=1000, elapsed_timesteps_before_soft_saving_env=None, delete_and_rerun_experiments_without_stored_env=False)
        
    if False == True:
        # ================================================================
        for si in xrange(len(parameter_override_dicts_per_sub_experiment)):
            environment_name_format_string = environment_name_format_strings[si]
            extracted_results = []
            group_labels = []
            for rpt_number in xrange(NUM_EXPERIMENT_REPEATS):
                group_labels.append("({}, {})".format(si, rpt_number))
                environment_name = environment_name_format_string.format(si, rpt_number)
                environment_dir = get_environment_directory_path(experiment_dir, environment_name)
                
                relevant_environment = retrieve_environment(environment_name, environment_dir)
                
                analysis_data = analysis.analyze_single_cell_motion(experiment_dir, si, rpt_number, relevant_environment)
                
                extracted_results.append(analysis_data)
        # ================================================================
        
            datavis.present_collated_single_cell_motion_data(extracted_results, experiment_dir=experiment_dir)
                
            