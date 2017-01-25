from __future__ import division
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import analysis.utilities as au
import visualization.datavis as datavis
import os

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)

baseline_parameter_dict = dict([('coa_sensing_dist_at_value', 100), ('coa_sensing_value_at_dist', 0.25),  ('kdgdi_rac_estimate_multiplier', 0.2), ('kdgdi_rho_estimate_multiplier', 0.2), ('kgdi_rac_estimate_multiplier', 1), ('kgdi_rho_estimate_multiplier', 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 500), ('kdgtp_rac_multiplier', 20), ('kdgtp_rho_mediated_rac_inhib_multiplier', 3000), ('kdgtp_rho_multiplier', 20), ('kgtp_rac_autoact_multiplier', 250), ('kgtp_rac_multiplier', 20), ('kgtp_rho_autoact_multiplier', 125), ('kgtp_rho_multiplier', 20), ('max_protrusive_node_velocity', 1e-06), ('randomization', False), ('randomization_scheme', 'wipeout'), ('randomization_time_mean', 20), ('randomization_time_variance_factor', 0.25), ('randomization_magnitude', 1.0), ('sigma_rac', 2e-05), ('sigma_rho_multiplier', 0.2), ('force_adh_constant', 1.0), ('force_rac_exp', 3), ('force_rho_exp', 3), ('force_rac_threshold_multiplier', 0.5), ('force_rho_threshold_multiplier', 0.5), ('skip_dynamics', False), ('stiffness_cytoplasmic', 1e-1), ('stiffness_edge', 5e-10), ('tension_fn_type', 0), ('tension_mediated_rac_hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('threshold_rac_autoact_multiplier', 0.5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5), ('threshold_rho_autoact_multiplier', 0.5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5), ('max_coa_signal', 2.0)])

parameter_overrides_dict = eu.update_pd_with_keyvalue_tuples(baseline_parameter_dict, [('coa_sensing_dist_at_value', 100, 100), ('coa_sensing_value_at_dist', 0.25, 0.25), ('kdgdi_rac_estimate_multiplier', 0.2, 1.0*0.2), ('kdgdi_rho_estimate_multiplier', 0.2, 1.0*0.2), ('kgdi_rac_estimate_multiplier', 1, 1.0), ('kgdi_rho_estimate_multiplier', 1, 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 500, 500), ('kdgtp_rac_multiplier', 20, 20*(2/3.)), ('kdgtp_rho_mediated_rac_inhib_multiplier', 3000, 1000), ('kdgtp_rho_multiplier', 20, 20*(2/3.)), ('kgtp_rac_autoact_multiplier', 250, 2*100), ('kgtp_rac_multiplier', 20, 20), ('kgtp_rho_autoact_multiplier', 125, 100), ('kgtp_rho_multiplier', 20, 20), ('max_protrusive_node_velocity', 1e-06, (3./2.)*0.25e-6), ('randomization_time_mean', 20, 20), ('randomization_time_variance_factor', 0.25, 0.01), ('randomization_magnitude', 1.0, 1.0), ('sigma_rac', 2e-05, 3.2e-05), ('sigma_rho_multiplier', 0.2, 0.2), ('force_adh_constant', 1.0, 0.0), ('force_rac_exp', 3, 3), ('force_rho_exp', 3, 3), ('force_rac_threshold_multiplier', 0.5, 0.45), ('force_rho_threshold_multiplier', 0.5, 0.45), ('skip_dynamics', False, False), ('stiffness_edge', 5e-10, 0.5*3.8e-10), ('stiffness_cytoplasmic', 1e-1, 10), ('tension_fn_type', 0, 6), ('tension_mediated_rac_inhibition_half_strain', 0.05, 0.05), ('threshold_rac_autoact_multiplier', 0.5, 0.5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5, 0.5), ('threshold_rho_autoact_multiplier', 0.5, 0.3), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5, 0.5), ('max_coa_signal', 2.0, -1.0)])

default_coa = 0.5
default_cil = 40
closeness_dist_squared_criteria = (2e-6)**2

randomization_time_mean_m = 20.0
randomization_time_variance_factor_m = 0.1
randomization_magnitude_m = 5

max_timepoints_on_ram = 100
seed = 36
allowed_drift_before_geometry_recalc = 1.0

ets.single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=False, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=False, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=25, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=False, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=50, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=False, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=100, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.single_cell_polarization_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.two_cells_cil_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=False, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, num_nodes=16, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, migr_bdry_height_factor=0.8, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.many_cells_coa_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=1, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.many_cells_coa_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=2, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.many_cells_coa_test(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=10, num_cells_height=3, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=100, default_coa=0.65*0.25*0.35*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=1, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=default_coa, default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=4, num_cells_height=1, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.75*default_coa, default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6, timestep_length=2, num_nodes=16, num_cells_width=2, num_cells_height=2, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=1.0, default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12, timestep_length=2, num_nodes=16, num_cells_width=4, num_cells_height=4, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5*0.25*0.75*default_coa, default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=2000, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12*1.5, timestep_length=2, num_nodes=16, num_cells_width=10, num_cells_height=3, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.65*0.25*0.35*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=2000, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12*1.5, timestep_length=2, num_nodes=16, num_cells_width=6, num_cells_height=6, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.65*0.25*0.35*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=2000, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12*1.5, timestep_length=2, num_nodes=16, num_cells_width=6, num_cells_height=6, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.65*0.25*0.3*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=2000, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12*1.5, timestep_length=2, num_nodes=16, num_cells_width=9, num_cells_height=4, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.65*0.25*0.3*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=2000, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.corridor_migration_experiment(date_str, experiment_number, baseline_parameter_dict, parameter_overrides_dict, randomization=True, randomization_scheme='m', randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12*1.5, timestep_length=2, num_nodes=16, num_cells_width=9, num_cells_height=4, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.65*0.25*0.4*default_coa, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=2000, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)







