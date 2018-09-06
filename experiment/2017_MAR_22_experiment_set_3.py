
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



parameter_dict = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 10), ('closeness_dist_squared_criteria', 0.25e-12), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-7), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 1.), ('skip_dynamics', False)])

closeness_dist_squared_criteria = (2e-6)**2

randomization_time_mean_m = 20.0
randomization_time_variance_factor_m = 0.1
randomization_magnitude_m = 1.5

max_timepoints_on_ram = 100
seed = 36
allowed_drift_before_geometry_recalc = 1.0

parameter_dict.update([('kgtp_rac_multiplier', 1.),
 ('kgtp_rho_multiplier', 1.),
 ('kdgtp_rac_multiplier', 1.),
 ('kdgtp_rho_multiplier', 1.),
 ('threshold_rac_activity_multiplier', 0.4),
 ('threshold_rho_activity_multiplier', 0.4),
 ('kgtp_rac_autoact_multiplier', 1),
 ('kgtp_rho_autoact_multiplier', 1),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 1),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1)])

ets.single_cell_polarization_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 5.),
 ('kgtp_rho_multiplier', 1.),
 ('kdgtp_rac_multiplier', 1.),
 ('kdgtp_rho_multiplier', 1.),
 ('threshold_rac_activity_multiplier', 0.4),
 ('threshold_rho_activity_multiplier', 0.4),
 ('kgtp_rac_autoact_multiplier', 1),
 ('kgtp_rho_autoact_multiplier', 1),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 1),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1)])

ets.single_cell_polarization_test(date_str, experiment_number, 1, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 1.),
 ('kgtp_rho_multiplier', 1.),
 ('kdgtp_rac_multiplier', 1.),
 ('kdgtp_rho_multiplier', 1.),
 ('threshold_rac_activity_multiplier', 0.4),
 ('threshold_rho_activity_multiplier', 0.4),
 ('kgtp_rac_autoact_multiplier', 1.),
 ('kgtp_rho_autoact_multiplier', 1.),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 1.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 5.)])

ets.single_cell_polarization_test(date_str, experiment_number, 2, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 1.),
 ('kgtp_rho_multiplier', 1.),
 ('kdgtp_rac_multiplier', 1.),
 ('kdgtp_rho_multiplier', 1.),
 ('threshold_rac_activity_multiplier', 0.4),
 ('threshold_rho_activity_multiplier', 0.4),
 ('kgtp_rac_autoact_multiplier', 1.),
 ('kgtp_rho_autoact_multiplier', 1.),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 5.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1.)])

ets.single_cell_polarization_test(date_str, experiment_number, 3, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


parameter_dict.update([('kgtp_rac_multiplier', 1.5460521835862795),
 ('kgtp_rho_multiplier', 1.0),
 ('kdgtp_rac_multiplier', 1.0),
 ('kdgtp_rho_multiplier', 1.1857857806619774),
 ('threshold_rac_activity_multiplier', 0.10000000000000001),
 ('threshold_rho_activity_multiplier', 0.15736778660774906),
 ('kgtp_rac_autoact_multiplier', 3.8792904404960207),
 ('kgtp_rho_autoact_multiplier', 1.0971039072208564),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 2.251215234399099),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 3.4088073906802219),
 ('tension_mediated_rac_inhibition_half_strain', 0.01)])


ets.single_cell_polarization_test(date_str, experiment_number, 4, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


parameter_dict.update([('kgtp_rac_multiplier', 20.0),
 ('kgtp_rho_multiplier', 20.0),
 ('kdgtp_rac_multiplier', 20*(2/3.)),
 ('kdgtp_rho_multiplier', 20*(2/3.)),
 ('threshold_rac_activity_multiplier', 0.5),
 ('threshold_rho_activity_multiplier', 0.5),
 ('kgtp_rac_autoact_multiplier', 200.),
 ('kgtp_rho_autoact_multiplier', 100.),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 500.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.),
 ('tension_mediated_rac_inhibition_half_strain', 0.05)])


ets.single_cell_polarization_test(date_str, experiment_number, 5, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


parameter_dict.update([('kgtp_rac_multiplier', 20.0),
 ('kgtp_rho_multiplier', 20.0),
 ('kdgtp_rac_multiplier', 20*(2/3.)),
 ('kdgtp_rho_multiplier', 20*(2/3.)),
 ('threshold_rac_activity_multiplier', 0.5),
 ('threshold_rho_activity_multiplier', 0.5),
 ('kgtp_rac_autoact_multiplier', 200.),
 ('kgtp_rho_autoact_multiplier', 100.),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 500.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.),
 ('tension_mediated_rac_inhibition_half_strain', 0.02)])


ets.single_cell_polarization_test(date_str, experiment_number, 6, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 20.0),
 ('kgtp_rho_multiplier', 20.0),
 ('kdgtp_rac_multiplier', 15.),
 ('kdgtp_rho_multiplier', 15.),
 ('threshold_rac_activity_multiplier', 0.5),
 ('threshold_rho_activity_multiplier', 0.5),
 ('kgtp_rac_autoact_multiplier', 250.),
 ('kgtp_rho_autoact_multiplier', 120.),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.),
 ('tension_mediated_rac_inhibition_half_strain', 0.05)])


ets.single_cell_polarization_test(date_str, experiment_number, 7, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 7.0),
 ('kgtp_rho_multiplier', 20.0),
 ('kdgtp_rac_multiplier', 1.0),
 ('kdgtp_rho_multiplier', 18.0),
 ('threshold_rac_activity_multiplier', 0.80000000000000004),
 ('threshold_rho_activity_multiplier', 0.79000000000000004),
 ('kgtp_rac_autoact_multiplier', 104.0),
 ('kgtp_rho_autoact_multiplier', 276.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 688.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 973.0),
 ('tension_mediated_rac_inhibition_half_strain', 0.050000000000000003)])


ets.single_cell_polarization_test(date_str, experiment_number, 8, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, 8, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m*4, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 20.0),
 ('kgtp_rho_multiplier', 3.0),
 ('kdgtp_rac_multiplier', 2.0),
 ('kdgtp_rho_multiplier', 20.0),
 ('threshold_rac_activity_multiplier', 0.65000000000000002),
 ('threshold_rho_activity_multiplier', 0.73999999999999999),
 ('kgtp_rac_autoact_multiplier', 491.0),
 ('kgtp_rho_autoact_multiplier', 718.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 883.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 773.0),
 ('tension_mediated_rac_inhibition_half_strain', 0.044999999999999998)])


ets.single_cell_polarization_test(date_str, experiment_number, 9, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, 9, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 8.0),
 ('kgtp_rho_multiplier', 11.0),
 ('kdgtp_rac_multiplier', 2.0),
 ('kdgtp_rho_multiplier', 20.0),
 ('threshold_rac_activity_multiplier', 0.79000000000000004),
 ('threshold_rho_activity_multiplier', 0.80000000000000004),
 ('kgtp_rac_autoact_multiplier', 542.0),
 ('kgtp_rho_autoact_multiplier', 831.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 860.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 599.0),
 ('tension_mediated_rac_inhibition_half_strain', 0.045000000000000005)])


ets.single_cell_polarization_test(date_str, experiment_number, 10, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, 10, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


parameter_dict.update([('kgtp_rac_multiplier', 18.0),
 ('kgtp_rho_multiplier', 12.0),
 ('kdgtp_rac_multiplier', 1.0),
 ('kdgtp_rho_multiplier', 10.0),
 ('threshold_rac_activity_multiplier', 0.55999999999999983),
 ('threshold_rho_activity_multiplier', 0.58999999999999986),
 ('kgtp_rac_autoact_multiplier', 265.0),
 ('kgtp_rho_autoact_multiplier', 169.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 841.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
 ('tension_mediated_rac_inhibition_half_strain', 0.050000000000000003)])


ets.single_cell_polarization_test(date_str, experiment_number, 11, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, 11, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


parameter_dict.update([('kgtp_rac_multiplier', 20.0),
 ('kgtp_rho_multiplier', 10.0),
 ('kdgtp_rac_multiplier', 1.0),
 ('kdgtp_rho_multiplier', 20.0),
 ('threshold_rac_activity_multiplier', 0.56999999999999973),
 ('threshold_rho_activity_multiplier', 0.79000000000000004),
 ('kgtp_rac_autoact_multiplier', 163.0),
 ('kgtp_rho_autoact_multiplier', 909.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 486.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 954.0),
 ('tension_mediated_rac_inhibition_half_strain', 0.049999999999999996)])


ets.single_cell_polarization_test(date_str, experiment_number, 12, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, 12, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m*4, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 3.0),
 ('kgtp_rho_multiplier', 4.0),
 ('kdgtp_rac_multiplier', 7.0),
 ('kdgtp_rho_multiplier', 20.0),
 ('threshold_rac_activity_multiplier', 0.1),
 ('threshold_rho_activity_multiplier', 0.50999999999999979),
 ('kgtp_rac_autoact_multiplier', 972.0),
 ('kgtp_rho_autoact_multiplier', 343.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 387.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 253.0),
 ('tension_mediated_rac_inhibition_half_strain', 0.050000000000000003)])


ets.single_cell_polarization_test(date_str, experiment_number, 13, parameter_dict, randomization_scheme=None, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, 13, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)


#ets.single_cell_polarization_test(date_str, experiment_number, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1, timestep_length=2, verbose=True, integration_params={'rtol': 1e-8}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5, default_cil=10, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

#ets.two_cells_cil_test(date_str, experiment_number, parameter_dict, randomization_scheme='m', randomization_time_mean_m=20.0, randomization_time_variance_factor_m=0.01, randomization_magnitude_m=0.75*25, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3, timestep_length=2, cell_diameter=40, verbose=True, closeness_dist_squared_criteria=(1e-6)**2, integration_params={'rtol': 1e-4}, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=1.0, default_coa=0, default_cil=0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=0.8)
