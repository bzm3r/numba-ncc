from __future__ import division
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os
import copy

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



closeness_dist_squared_criteria = (0.5e-6)**2

parameter_dict = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('tension_mediated_rac_inhibition_magnitude', 1.0), ('strain_calculation_type', 1),('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 30.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25), ('randomization_type', 'r'), ('coa_intersection_exponent', 2.0)])

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 20.0
remake_visualizations = False

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 23.0),
  ('kdgtp_rac_multiplier', 40.0),
  ('kdgtp_rho_multiplier', 40.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.25000000000000006),
  ('kgtp_rac_autoact_multiplier', 250.0),
  ('kgtp_rho_autoact_multiplier', 0.9*170.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 900.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.095000000000000001),
  ('stiffness_edge', 8000.0), ('randomization_magnitude', 10.0), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_node_percentage', 0.45)])

sub_experiment_number = 0
ets.chemoattractant_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=4983, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, num_cells=1, migratory_box_width=1000, migratory_box_height=400, auto_calculate_num_cells=False, chemoattractant_source_definition=[500., 0., 1000., 0., 1.], autocalculate_chemoattractant_source_y_position=True, remake_visualizations=remake_visualizations)

ets.chemoattractant_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=4983, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, num_cells=3, migratory_box_width=1000, migratory_box_height=400, auto_calculate_num_cells=False, chemoattractant_source_definition=[500., 0., 1000., 0., 1.], autocalculate_chemoattractant_source_y_position=True, remake_visualizations=remake_visualizations)

ets.chemoattractant_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=4983, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, num_cells=9, migratory_box_width=1000, migratory_box_height=400, auto_calculate_num_cells=False, chemoattractant_source_definition=[500., 0., 1000., 0., 1.], autocalculate_chemoattractant_source_y_position=True, remake_visualizations=remake_visualizations)

ets.chemoattractant_test(date_str, experiment_number, 2, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=4983, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=False, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, num_cells=9, migratory_box_width=1000, migratory_box_height=400, auto_calculate_num_cells=False, chemoattractant_source_definition=[500., 0., 1000., 0., 2.], autocalculate_chemoattractant_source_y_position=True, remake_visualizations=remake_visualizations)


#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=2., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0, remake_visualizations=remake_visualizations)

#ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=7, num_cells_height=7, remake_visualizations=remake_visualizations)

#ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=7, num_cells_height=7, remake_visualizations=remake_visualizations)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=2, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=2, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=20.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=16.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=False)


#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=20.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=8.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=False)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=False)

#parameter_dict.update([('kgtp_rac_multiplier', 5.0),
#  ('kgtp_rho_multiplier', 23.0),
#  ('kdgtp_rac_multiplier', 40.0),
#  ('kdgtp_rho_multiplier', 40.0),
#  ('threshold_rac_activity_multiplier', 0.35000000000000014),
#  ('threshold_rho_activity_multiplier', 0.25000000000000006),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 0.9*170.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 900.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
#  ('tension_mediated_rac_inhibition_half_strain', 0.095000000000000001),
#  ('stiffness_edge', 8000.0), ('randomization_magnitude', 10.0), ('randomization_time_mean', 45.0), ('randomization_time_variance_factor', 0.25), ('randomization_node_percentage', 0.45), ('randomization_scheme', 'w')])
#
#sub_experiment_number = 1
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=4.0, default_cil=80.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=2, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=2, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=16.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=remake_visualizations)
#
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=remake_visualizations)
#
#
#parameter_dict.update([('kgtp_rac_multiplier', 5.0),
#  ('kgtp_rho_multiplier', 23.0),
#  ('kdgtp_rac_multiplier', 40.0),
#  ('kdgtp_rho_multiplier', 40.0),
#  ('threshold_rac_activity_multiplier', 0.35000000000000014),
#  ('threshold_rho_activity_multiplier', 0.25000000000000006),
#  ('kgtp_rac_autoact_multiplier', 250.0),
#  ('kgtp_rho_autoact_multiplier', 0.9*170.0),
#  ('kdgtp_rac_mediated_rho_inhib_multiplier', 900.0),
#  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
#  ('tension_mediated_rac_inhibition_half_strain', 0.095000000000000001),
#  ('stiffness_edge', 8000.0), ('randomization_magnitude', 10.0), ('randomization_time_mean', 2*45.0), ('randomization_time_variance_factor', 0.25), ('randomization_node_percentage', 0.45), ('randomization_scheme', 'w')])
#
#sub_experiment_number = 2
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=4.0, default_cil=80.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=2, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=2, remake_visualizations=remake_visualizations)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=16.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=remake_visualizations)
#
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=remake_visualizations)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
