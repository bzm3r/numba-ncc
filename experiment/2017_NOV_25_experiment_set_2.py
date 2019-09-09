import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os
import copy

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)


closeness_dist_squared_criteria = (0.5e-6) ** 2

parameter_dict = dict(
    [
        ("num_nodes", 16),
        ("init_cell_radius", 20e-6),
        ("C_total", 2.5e6),
        ("H_total", 1e6),
        ("init_rgtpase_cytosol_frac", 0.8),
        ("init_rgtpase_membrane_active_frac", 0.1),
        ("init_rgtpase_membrane_inactive_frac", 0.1),
        ("diffusion_const", 0.1e-12),
        ("kgdi_multiplier", 1),
        ("kdgdi_multiplier", 1),
        ("kgtp_rac_multiplier", 1.0),
        ("kgtp_rac_autoact_multiplier", 200),
        ("kdgtp_rac_multiplier", 5.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 1000),
        ("threshold_rac_activity_multiplier", 0.4),
        ("kgtp_rho_multiplier", 10.0),
        ("kgtp_rho_autoact_multiplier", 100),
        ("kdgtp_rho_multiplier", 2.5),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 1000.0),
        ("threshold_rho_activity_multiplier", 0.4),
        ("hill_exponent", 3),
        ("tension_mediated_rac_inhibition_half_strain", 0.05),
        ("tension_mediated_rac_inhibition_magnitude", 1.0),
        ("max_coa_signal", -1.0),
        ("coa_sensing_dist_at_value", 110e-6),
        ("coa_sensing_value_at_dist", 0.5),
        ("interaction_factor_migr_bdry_contact", 30.0),
        ("closeness_dist_squared_criteria", closeness_dist_squared_criteria),
        ("length_3D_dimension", 10e-6),
        ("stiffness_edge", 5000),
        ("stiffness_cytoplasmic", 1e-5),
        ("eta", 1e5),
        ("max_force_rac", 10e3),
        ("force_rho_multiplier", 0.2),
        ("force_adh_const", 0.0),
        ("skip_dynamics", False),
        ("randomization_scheme", "m"),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 10.0),
        ("randomization_node_percentage", 0.25),
        ("randomization_type", "r"),
        ("coa_intersection_exponent", 2.0),
        ("strain_calculation_type", 0),
    ]
)

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 20.0

remake_animation = False
remake_graphs = False
do_final_analysis = True

default_cil = 60.0
integration_params = {"rtol": 1e-4}

base_output_dir = "B:\\numba-ncc\\output"

labels = ["COA-", "hypo", "standard", "hyper"]
default_coas = [0.0, 3 * 24.0, 24.0, 24.0]
default_cils = [60.0, 2 * 60.0, 60.0, 60.0]
hypo_set = dict(
    [
        ("kgtp_rac_multiplier", 12.0 * 0.1),
        ("kgtp_rho_multiplier", 14.0 * 0.1),
        ("kdgtp_rac_multiplier", 4.0 * 0.1),
        ("kdgtp_rho_multiplier", 30.0 * 0.1),
        ("kgtp_rac_autoact_multiplier", 250.0 * 0.1),
        ("kgtp_rho_autoact_multiplier", 195.0 * 0.1),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 200.0 * 0.1),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0 * 0.1),
        ("tension_mediated_rac_inhibition_half_strain", 0.1),
        ("tension_mediated_rac_inhibition_magnitude", 2 * 40.0),
        ("max_force_rac", 3000.0),
        ("eta", 2.9 * 10000.0),
        ("stiffness_edge", 8000.0),
        ("randomization_time_mean", 20.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 3 * 12.0),
        ("randomization_node_percentage", 0.25),
    ]
)
standard_set = dict(
    [
        ("kgtp_rac_multiplier", 12.0),
        ("kgtp_rho_multiplier", 14.0),
        ("kdgtp_rac_multiplier", 4.0),
        ("kdgtp_rho_multiplier", 30.0),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 195.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 200.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.1),
        ("tension_mediated_rac_inhibition_magnitude", 40.0),
        ("max_force_rac", 3000.0),
        ("eta", 2.9 * 10000.0),
        ("stiffness_edge", 8000.0),
        ("randomization_time_mean", 20.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 12.0),
        ("randomization_node_percentage", 0.25),
    ]
)
hyper_set = dict(
    [
        ("kgtp_rac_multiplier", 12.0 * 10),
        ("kgtp_rho_multiplier", 14.0 * 10),
        ("kdgtp_rac_multiplier", 4.0 * 10),
        ("kdgtp_rho_multiplier", 30.0 * 10),
        ("kgtp_rac_autoact_multiplier", 250.0 * 10),
        ("kgtp_rho_autoact_multiplier", 195.0 * 10),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 200.0 * 10),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0 * 10),
        ("tension_mediated_rac_inhibition_half_strain", 0.1),
        ("tension_mediated_rac_inhibition_magnitude", 40.0),
        ("max_force_rac", 3000.0),
        ("eta", 2.9 * 10000.0),
        ("stiffness_edge", 8000.0),
        ("randomization_time_mean", 20.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 6.0),
        ("randomization_node_percentage", 0.25),
    ]
)
parameter_update_dicts = [
    copy.deepcopy(standard_set),
    hypo_set,
    standard_set,
    hyper_set,
]

# date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="B:\\numba-ncc\\output\\", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=(3 + i)*8.0, default_cil=60.0, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=3, box_height=3, num_cells=9, corridor_height=3, box_y_placement_factor=0.0, remake_graphs=remake_graphs, remake_animation=remake_animation, do_final_analysis=do_final_analysis

ets.corridor_migration_parameter_set_test(
    date_str,
    experiment_number,
    parameter_dict,
    experiment_set_label="hypo_hyper_stand",
    no_randomization=False,
    base_output_dir="B:\\numba-ncc\\output\\",
    total_time_in_hours=20,
    timestep_length=2,
    verbose=True,
    closeness_dist_squared_criteria=closeness_dist_squared_criteria,
    integration_params=integration_params,
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    labels=labels,
    sub_experiment_numbers=[3, 0, 1, 2],
    parameter_update_dicts=parameter_update_dicts,
    default_coas=default_coas,
    default_cils=default_cils,
    num_cells=4,
    box_height=1,
    box_width=4,
    corridor_height=1,
    num_experiment_repeats=20,
    particular_repeats=[],
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    run_experiments=False,
    remake_graphs=False,
    remake_animation=False,
    do_final_analysis=True,
    justify_parameters=False,
)
