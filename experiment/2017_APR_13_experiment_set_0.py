from __future__ import division
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os
import copy

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



closeness_dist_squared_criteria = (0.5e-6)**2

parameter_dict = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 40.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 9.0), ('randomization_node_percentage', 0.25)])

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 5.0


parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 40.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 10.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])


sub_experiment_number = 0

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=2836, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

'''
Cell is not able to rapidly resolve fronts, especially if they are on opposite sides. Persistence is low, but not too low. In the next experiment, I wonder what happens if I reduce the randomization time mean (since it seems that the random fronts are strong enough that they won't disappear on their own).
'''

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 20.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 10.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])


sub_experiment_number = 1

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=2836, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

'''
A thought occurs: there is no need for cells to be able to mitigate nascent random protrusions on their own -- rather, they should be able to mitigate the new fronts that result from the presence of temporary random spikes. 

So, this experiment was quite promising in this regard, because I was able to spot some cases where the cell was able to mitigate new fronts resulting from the presence of temporary random spikes.

What if the spikes were stronger, and shorter?
'''

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 12.5),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])


sub_experiment_number = 2

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=2836, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

'''
This experiment was very promising. However, spike strength needs to be increased further, so that stronger residual fronts are created.
'''

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 20.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])


sub_experiment_number = 3

ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=2836, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=6.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True)

'''
The results were here were promising. How much COA is needed to keep these guys together?
'''

ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1)

ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1)

ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=12.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1)

'''
I think COA=12.0 is enough, and COA=24.0 is good too. Anyway, how do these cells respond to CIL? I reduce closeness_dist_squared_criteria to (0.5e-6)**2 (so, the COA tests will have to be re-run).
'''

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

'''
Well, RhoA doesn't really activate upon contact. So, the cells don't realy re-polarize effectively, because they might quickly re-polarize towards contact again.

Does this matter really? I am not so sure, but I think it doesn't quite match experimental results.
'''

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 2*8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 20.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 4

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=1359, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 4*8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 20.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 5

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=1359, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.75*0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 20.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 6

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=1359, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.5*0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 20.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 7

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=1359, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)


parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 3*8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.5*0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 10.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 20.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 8

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=1359, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

parameter_dict.update([('kgtp_rac_multiplier', 5.0),
  ('kgtp_rho_multiplier', 3*8.0),
  ('kdgtp_rac_multiplier', 39.0),
  ('kdgtp_rho_multiplier', 35.0),
  ('threshold_rac_activity_multiplier', 0.35000000000000014),
  ('threshold_rho_activity_multiplier', 0.5*0.65000000000000002),
  ('kgtp_rac_autoact_multiplier', 300.0),
  ('kgtp_rho_autoact_multiplier', 5*15.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
  ('randomization_time_mean', 15.0),
  ('randomization_time_variance_factor', 0.10000000000000001),
  ('randomization_magnitude', 15.0),
  ('stiffness_edge', 8000.0), ('randomization_node_percentage', 0.25)])

sub_experiment_number = 9

ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=12.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)

#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=1.5, timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0)





