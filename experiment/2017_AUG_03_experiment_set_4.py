
import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os
import copy

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)



closeness_dist_squared_criteria = (0.5e-6)**2

parameter_dict = dict([('num_nodes', 20), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.8), ('init_rgtpase_membrane_active_frac', 0.1), ('init_rgtpase_membrane_inactive_frac', 0.1), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('tension_mediated_rac_inhibition_magnitude', 1.0), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 30.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 10.0), ('randomization_node_percentage', 0.25), ('randomization_type', 'r'), ('coa_intersection_exponent', 2.0), ('strain_calculation_type', 0)])

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 20.0

remake_animation = False
remake_graphs = False
do_final_analysis = True

default_cil = 960.0
default_interaction_factor_migr_bdry_contact = 30.0*16.0
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

coa_dict = {49: 8.0*16, 36: 10.0*16, 25: 12.0*16, 16: 14.0*16, 9: 16.0*16, 4: 24.0*16, 2: 24.0*16, 1: 24.0*16}

test_num_nodes = np.arange(23) + 3
ets.convergence_test_corridor(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="AR=1", no_randomization=True, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_num_nodes=test_num_nodes, coa_dict=coa_dict, default_cil=default_cil, default_interaction_factor_migr_bdry_contact=default_interaction_factor_migr_bdry_contact, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=False, biased_rgtpase_distrib_defn_dict={'default': ['unbiased random', np.array([0, 2*np.pi]), 0.3]})