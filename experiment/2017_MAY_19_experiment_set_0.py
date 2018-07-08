
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
integration_params = {'rtol': 1e-4}

base_output_dir = "B:\\numba-ncc\\output" 

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

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[1], default_cil=60.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=1, box_height=1, num_cells=1, corridor_height=1, box_y_placement_factor=0.5, remake_graphs=False, remake_animation=True, do_final_analysis=do_final_analysis, colorscheme="scifi", show_coa_overlay=True, coa_overlay_resolution=2)


#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=0.8, run_experiments=True, remake_graphs=False, remake_animation=True)
#
#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0, run_experiments=True, remake_graphs=False, remake_animation=True)
#
#
#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=0.8, run_experiments=True, remake_graphs=False, remake_animation=False)
##
#ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=3., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0, run_experiments=True, remake_graphs=False, remake_animation=False)
##
#ets.two_cells_cil_test_symmetric(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=False, remake_graphs=False, remake_animation=False)


#ets.coa_factor_variation_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_coas=[0.0, coa_dict[49]], default_cil=default_cil, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_to_test=[49], skip_low_coa=False, max_normalized_group_area=3.0, run_experiments=False, remake_graphs=False, remake_animation=False)

#ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[49], default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=7, box_height=7, num_cells=49, remake_graphs=True, remake_animation=True)

#
#ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=7, box_height=7, num_cells=49, remake_graphs=True, remake_animation=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=6, box_height=3, num_cells=16, corridor_height=4, box_y_placement_factor=0.5, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)
#
#ets.corridor_migration_symmetric_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=20, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=True, do_final_analysis=True)

#ets.corridor_migration_symmetric_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=20, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=True, do_final_analysis=True)

#ets.corridor_migration_symmetric_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=20, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=1, num_cells=4, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=True, do_final_analysis=True)

#ets.corridor_migration_symmetric_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=20, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=1, box_height=1, num_cells=1, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=True, do_final_analysis=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="r", max_placement_distance_factor=1.5, init_random_cell_placement_x_factor=0.25)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=6, box_height=6, num_cells=36, corridor_height=6, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="r", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="r", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=20, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=6, box_height=6, num_cells=36, corridor_height=6, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=False, cell_placement_method="r")


#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=8, box_height=2, num_cells=16, corridor_height=4, box_y_placement_factor=0.5, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=8, box_height=2, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=20.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=8, box_height=2, num_cells=16, corridor_height=2, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)


#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=16, box_height=1, num_cells=16, corridor_height=4, box_y_placement_factor=0.5, remake_graphs=remake_graphs, remake_animation=False, do_final_analysis=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=16, box_height=1, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.5, remake_graphs=False, remake_animation=True, do_final_analysis=do_final_analysis)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=10.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=6, box_height=6, num_cells=36, corridor_height=6, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=do_final_analysis)
#
#test_heights = [2, 3, 4, 5, 6, 7]
#test_num_cells = [h**2 for h in test_heights]
#ets.corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="AR=1", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_heights=test_heights, test_num_cells=test_num_cells, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=True)


#test_heights = [2, 3, 4, 5, 6, 7]
#test_num_cells = [h**2 for h in test_heights]
#ets.corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="AR=1_r", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_heights=test_heights, test_num_cells=test_num_cells, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="r")

#test_heights = [5, 6, 7]
#test_num_cells = [h**2 for h in test_heights]
#ets.corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="AR=1", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_heights=test_heights, test_num_cells=test_num_cells, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=True, remake_animation=True, do_final_analysis=True)
#
#test_heights = np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
#test_num_cells = 16*np.ones_like(test_heights, dtype=np.int64)
#ets.corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="N=16_H=X", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_heights=test_heights, test_num_cells=test_num_cells, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=False, graph_x_dimension="test_heights")
###
#test_heights = np.array([2, 4, 8], dtype=np.int64)
#test_num_cells = 16*np.ones_like(test_heights, dtype=np.int64)
#ets.corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="N=16_H=X", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_heights=test_heights, test_num_cells=test_num_cells, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=True, do_final_analysis=False, graph_x_dimension="test_heights", max_animation_corridor_length=600, global_scale=2)
#
#test_heights = np.array([4, 5, 6, 7, 8])
#test_num_cells = 36*np.ones_like(test_heights, dtype=np.int64)
#ets.corridor_migration_collective_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="N=36_H=X", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_heights=test_heights, test_num_cells=test_num_cells, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=False, graph_x_dimension="test_heights")
#
#num_cells = 16
#corridor_height = 4
#test_heights = [4, "r", "r"]
#init_random_cell_placement_x_factors = [0.0, 0.25, 0.0]
#max_placement_distance_factors = [1.0, 1.0, 15.0]
#test_num_cells = num_cells*np.ones(len(test_heights), dtype=np.int64)
#box_placement_factors = 0.5*np.ones(len(test_heights), dtype=np.float64)
#test_widths = np.array([int(np.ceil(num_cells/th)) if th != "r" else int(np.sqrt(num_cells)) for th in test_heights], dtype=np.int64)
#corridor_heights = corridor_height*np.ones(len(test_heights), dtype=np.int64)
#ets.corridor_migration_init_conditions_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="NC=16", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=20., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_num_cells=test_num_cells, test_heights=test_heights, max_placement_distance_factors=max_placement_distance_factors, init_random_cell_placement_x_factors=init_random_cell_placement_x_factors, test_widths=test_widths, corridor_heights=corridor_heights, box_placement_factors=box_placement_factors, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=10 , timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=True)

#num_cells = 16
#corridor_height = 4
#test_heights = np.array([5, 6, 7, 8], dtype=np.int64)
#test_num_cells = num_cells*np.ones(test_heights.shape[0], dtype=np.int64)
#box_placement_factors = 0.5*np.ones(test_heights.shape[0], dtype=np.float64)
#test_widths = np.array([int(np.ceil(num_cells/th)) for th in test_heights], dtype=np.int64)
#corridor_heights = corridor_height*np.ones(test_heights.shape[0], dtype=np.int64)
#ets.corridor_migration_init_conditions_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="NC=16", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_num_cells=test_num_cells, test_heights=test_heights, test_widths=test_widths, corridor_heights=corridor_heights, box_placement_factors=box_placement_factors, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=10 , timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=True, remake_animation=True, do_final_analysis=True)


#num_cells = 16
#corridor_height = 4
#test_heights = np.array([2, 3], dtype=np.int64)
#test_num_cells = num_cells*np.ones(test_heights.shape[0], dtype=np.int64)
#box_placement_factors = 0.0*np.ones(test_heights.shape[0], dtype=np.float64)
#test_widths = np.array([int(np.ceil(num_cells/th)) for th in test_heights], dtype=np.int64)
#corridor_heights = corridor_height*np.ones(test_heights.shape[0], dtype=np.int64)
#ets.corridor_migration_init_conditions_tests(date_str, experiment_number, sub_experiment_number, parameter_dict, experiment_set_label="NC=16", no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_num_cells=test_num_cells, test_heights=test_heights, test_widths=test_widths, corridor_heights=corridor_heights, box_placement_factors=box_placement_factors, coa_dict=coa_dict, default_cil=default_cil, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=True, do_final_analysis=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=10.0, default_cil=60.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=6, box_height=3, num_cells=16, remake_graphs=remake_graphs, remake_animation=remake_animation)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=10.0, default_cil=60.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, remake_graphs=remake_graphs, remake_animation=remake_animation)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=60.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=1, box_height=1, num_cells=1, remake_graphs=remake_graphs, remake_animation=remake_animation)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=14.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=16, box_height=1, num_cells=16, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=True, remake_animation=False, do_final_analysis=False, graph_group_centroid_splits=True)

#ets.corridor_migration_symmetric_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="r", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25)
#
#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=False, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="r", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.block_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, dist_from_block=3.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, run_experiments=True, remake_graphs=False, remake_animation=False)
#
#ets.block_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, dist_from_block=6.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, run_experiments=True, remake_graphs=False, remake_animation=False)
#
#ets.block_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, dist_from_block=12.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, run_experiments=True, remake_graphs=False, remake_animation=False)
#
#ets.block_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5*0.5*coa_dict[16], default_cil=60.0, dist_from_block=3.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, run_experiments=True, remake_graphs=False, remake_animation=False)
#
#
#ets.block_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5*0.5*coa_dict[16], default_cil=60.0, dist_from_block=2.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, block_cells_width=4, block_cells_height=4, run_experiments=True, remake_graphs=False, remake_animation=False)

#ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5*0.5*coa_dict[16], default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, remake_graphs=True, remake_animation=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16]*0.5*0.5, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, remake_graphs=False, remake_animation=False, do_final_analysis=do_final_analysis)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=1, num_cells=4, corridor_height=1, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)
##
#for i in range(11, 17):
#    ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=i*8.0, default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=1, num_cells=4, corridor_height=1, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)
##    
#for i in range(11, 17):
#    ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=24.0, default_cil=i*30.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=1, num_cells=4, corridor_height=1, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)

#for i in range(17):
#    ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=i*8.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)

#for i in range(8):
#    ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=(3 + i)*8.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=3, box_height=3, num_cells=9, corridor_height=3, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis)


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
  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 12.0), ('randomization_node_percentage', 0.25), ('force_rho_multiplier', 0.0)])

sub_experiment_number = 1

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

#ets.two_cells_cil_test_symmetric(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False)

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

sub_experiment_number = 2

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

#ets.two_cells_cil_test_symmetric(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil*0.2, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False)


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

sub_experiment_number = 3
coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

#ets.two_cells_cil_test_symmetric(date_str, experiment_number, sub_experiment_number, parameter_dict, no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=4., timestep_length=2, verbose=True, closeness_dist_squared_criteria=closeness_dist_squared_criteria, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil*0.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False)

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
  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 2*12.0), ('randomization_node_percentage', 1/16.)])

sub_experiment_number = 4

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)


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
  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 12.0), ('randomization_node_percentage', 8/16.)])

sub_experiment_number = 5

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=True, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=2, num_cells=8, corridor_height=2, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, corridor_height=4, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=10, box_height=2, num_cells=20, corridor_height=2, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)
##


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
  ('stiffness_edge', 8000.0), ('randomization_time_mean', 20.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 3*12.0), ('randomization_node_percentage', 1/16.)])

sub_experiment_number = 6

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}
#
#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=True)


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
  ('stiffness_edge', 8000.0), ('randomization_time_mean', 10.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 3*12.0), ('randomization_node_percentage', 1/16.)])

sub_experiment_number = 7

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

#ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=False, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=True)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=3, box_height=1, num_cells=3, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.5*coa_dict[2], default_cil=60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=2, box_height=1, num_cells=2, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.75*coa_dict[2], default_cil=2*60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=3, box_height=2, num_cells=6, corridor_height=2, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=3*60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=3, box_height=2, num_cells=6, corridor_height=2, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)


#ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=3*60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=2, num_cells=8, corridor_height=2, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#for i in range(9):
#    nc = i + 2
#    ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=3*60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=nc, box_height=2, num_cells=nc*2, corridor_height=2, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)

#for i in range(9):
#    nc = i + 2
#    ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=3*60.0, num_experiment_repeats=20, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=nc, box_height=1, num_cells=nc, corridor_height=1, box_y_placement_factor=0.0, run_experiments=True, remake_graphs=False, remake_animation=False, do_final_analysis=True, cell_placement_method="", max_placement_distance_factor=15.0, init_random_cell_placement_x_factor=0.0)



