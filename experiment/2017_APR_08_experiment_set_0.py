from __future__ import division
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os
import copy

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



closeness_dist_squared_criteria = (0.5e-6)**2

parameter_dict = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 20.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 9.0)])

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 2.0

parameter_dict.update([('kgtp_rac_multiplier', 7.0),
 ('kgtp_rho_multiplier', 20.0),
 ('kdgtp_rac_multiplier', 1.0),
 ('kdgtp_rho_multiplier', 18.0),
 ('threshold_rac_activity_multiplier', 0.80),
 ('threshold_rho_activity_multiplier', 0.79),
 ('kgtp_rac_autoact_multiplier', 104.0),
 ('kgtp_rho_autoact_multiplier', 0.5*276.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 688.0),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 1.5*973.0), 
 ('tension_mediated_rac_inhibition_half_strain', 0.04),
  ('randomization_time_mean', 40.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 9.0)])


sub_experiment_number = 0

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 19.0),
  ('kgtp_rho_multiplier', 20.0),
  ('kdgtp_rac_multiplier', 7.0),
  ('kdgtp_rho_multiplier', 17.0),
  ('threshold_rac_activity_multiplier', 0.5),
  ('threshold_rho_activity_multiplier', 0.75000000000000011),
  ('kgtp_rac_autoact_multiplier', 230.0),
  ('kgtp_rho_autoact_multiplier', 200.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.10000000000000001),
  ('randomization_time_mean', 40.0),
  ('randomization_time_variance_factor', 0.1),
  ('randomization_magnitude', 5.0),
  ('stiffness_edge', 8000.0)])


sub_experiment_number = 1

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 6.5625),
   ('kgtp_rho_multiplier', 6.0),
   ('kdgtp_rac_multiplier', 1.5),
   ('kdgtp_rho_multiplier', 13.484375),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 95.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 7.5),
   ('stiffness_edge', 7750.0)])


sub_experiment_number = 2

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 12.2783203125),
   ('kgtp_rho_multiplier', 4.9375),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 5.0),
   ('threshold_rac_activity_multiplier', 0.43999023437499996),
   ('threshold_rho_activity_multiplier', 0.659912109375),
   ('kgtp_rac_autoact_multiplier', 101.6796875),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1430.078125),
   ('tension_mediated_rac_inhibition_half_strain', 0.051250000000000004),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 7925.78125)])


sub_experiment_number = 3

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 12.2783203125),
   ('kgtp_rho_multiplier', 4.9375),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 5.0),
   ('threshold_rac_activity_multiplier', 0.43999023437499996),
   ('threshold_rho_activity_multiplier', 0.659912109375),
   ('kgtp_rac_autoact_multiplier', 101.6796875),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1430.078125),
   ('tension_mediated_rac_inhibition_half_strain', 0.051250000000000004),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 6000)])


sub_experiment_number = 4

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

parameter_dict.update([('kgtp_rac_multiplier', 12.2783203125),
   ('kgtp_rho_multiplier', 4.9375),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 5.0),
   ('threshold_rac_activity_multiplier', 0.43999023437499996),
   ('threshold_rho_activity_multiplier', 0.659912109375),
   ('kgtp_rac_autoact_multiplier', 101.6796875),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1430.078125),
   ('tension_mediated_rac_inhibition_half_strain', 0.07),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 7925.)])


sub_experiment_number = 5

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)
















