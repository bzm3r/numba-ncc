from __future__ import division
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os
import copy

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



closeness_dist_squared_criteria = (0.5e-6)**2

parameter_dict = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.8), ('init_rgtpase_membrane_active_frac', 0.1), ('init_rgtpase_membrane_inactive_frac', 0.1), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('tension_mediated_rac_inhibition_magnitude', 1.0), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 30.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25), ('randomization_type', 'r'), ('coa_intersection_exponent', 2.0), ('strain_calculation_type', 0)])

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 20.0

remake_animation = False
remake_graphs = False
do_final_analysis = True

default_cil = 60.0
default_interaction_factor_migr_bdry_contact = 16.0
integration_params = {'rtol': 1e-4}

base_output_dir = "A:\\numba-ncc\\output" 

parameter_dict.update([('kgtp_rac_multiplier', 12.0),
  ('kgtp_rho_multiplier', 14.0),
  ('kdgtp_rac_multiplier', 4.0),
  ('kdgtp_rho_multiplier', 30.0),
  ('kgtp_rac_autoact_multiplier', 250.0),
  ('kgtp_rho_autoact_multiplier', 195.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
  ('tension_mediated_rac_inhibition_magnitude', 40.0),
  ('max_force_rac', 3000.0),
  ('eta', 2.9*10000.0),
  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 12.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 0

coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir=base_output_dir, total_time_in_hours=2., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)

ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, graph_group_centroid_splits=True)

#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 8.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 1
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 2
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.095),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 3
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.09),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 4
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.085),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 5
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.08),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 6
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 45.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 7
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 45.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 8
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 50.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 9
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 7000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 10
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)
#
#parameter_dict.update([('kgtp_rac_multiplier', 12.0),
#  ('kgtp_rho_multiplier', 14.0),
#  ('kdgtp_rac_multiplier', 4.0),
#  ('kdgtp_rho_multiplier', 30.0),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 195.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0), ('tension_mediated_rac_inhibition_half_strain', 0.1),
#  ('tension_mediated_rac_inhibition_magnitude', 40.0),
#  ('max_force_rac', 3000.0),
#  ('eta', 2.9*10000.0),
#  ('stiffness_edge', 6000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25)])
#
#sub_experiment_number = 11
#
#coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=False)



