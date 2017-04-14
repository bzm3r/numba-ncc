from __future__ import division
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



closeness_dist_squared_criteria = (1e-6)**2

parameter_dict = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', 6.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 20.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 0.55*1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 1.0), ('skip_dynamics', False)])

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1
randomization_magnitude_m = 6.0

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 2.0

#parameter_dict.update([('kgtp_rac_multiplier', 7.0),
# ('kgtp_rho_multiplier', 20.0),
# ('kdgtp_rac_multiplier', 1.0),
# ('kdgtp_rho_multiplier', 18.0),
# ('threshold_rac_activity_multiplier', 0.80),
# ('threshold_rho_activity_multiplier', 0.79),
# ('kgtp_rac_autoact_multiplier', 104.0),
# ('kgtp_rho_autoact_multiplier', 0.5*276.0),
# ('kdgtp_rac_mediated_rho_inhib_multiplier', 688.0),
# ('kdgtp_rho_mediated_rac_inhib_multiplier', 973.0), 
# ('tension_mediated_rac_inhibition_half_strain', 0.04)])

parameter_dict.update([('kgtp_rac_multiplier', 7.0),
 ('kgtp_rho_multiplier', 20.0),
 ('kdgtp_rac_multiplier', 1.0),
 ('kdgtp_rho_multiplier', 18.0),
 ('threshold_rac_activity_multiplier', 0.80),
 ('threshold_rho_activity_multiplier', 0.79),
 ('kgtp_rac_autoact_multiplier', 104.0),
 ('kgtp_rho_autoact_multiplier', 0.5*276.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 688.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 973.0), 
 ('tension_mediated_rac_inhibition_half_strain', 0.04)])


sub_experiment_number = 0

ets.two_cells_cil_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme='m', randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=3., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

ets.corridor_migration_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=1, num_cells_width=2)

ets.corridor_migration_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=1, num_cells_width=4)

ets.corridor_migration_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=1, num_cells_width=3)

ets.corridor_migration_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=2, num_cells_width=2)


ets.many_cells_coa_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=1, num_cells_width=4)

ets.many_cells_coa_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=2, num_cells_width=3)

ets.corridor_migration_test(date_str, experiment_number, 0, parameter_dict, randomization_scheme="m", randomization_magnitude_m=randomization_magnitude_m, randomization_time_mean_m=randomization_time_mean_m, randomization_time_variance_factor_m=randomization_time_variance_factor_m, randomization_time_mean_w=40.0, randomization_time_variance_factor_w=0.25, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=12., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2, 'atol': 1e-16}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40., num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_height=2, num_cells_width=3)




