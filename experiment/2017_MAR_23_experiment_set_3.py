import numpy as np
import general.exec_utils as eu
import general.experiment_templates as ets
import os

scriptname = os.path.basename(__file__)[:-3]
date_str, experiment_number = eu.get_date_and_experiment_number(scriptname)


parameter_dict = dict(
    [
        ("num_nodes", 16),
        ("init_cell_radius", 20e-6),
        ("C_total", 2.5e6),
        ("H_total", 1e6),
        ("init_rgtpase_cytosol_frac", 0.6),
        ("init_rgtpase_membrane_active_frac", 0.2),
        ("init_rgtpase_membrane_inactive_frac", 0.2),
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
        ("max_coa_signal", -1),
        ("coa_sensing_dist_at_value", 110e-6),
        ("coa_sensing_value_at_dist", 0.5),
        ("interaction_factor_migr_bdry_contact", 2.0),
        ("closeness_dist_squared_criteria", 0.25e-12),
        ("length_3D_dimension", 10e-6),
        ("stiffness_edge", 5000),
        ("stiffness_cytoplasmic", 1e-7),
        ("eta", 0.55 * 1e5),
        ("max_force_rac", 10e3),
        ("force_rho_multiplier", 0.2),
        ("force_adh_const", 0.0),
        ("skip_dynamics", False),
    ]
)

closeness_dist_squared_criteria = (2e-6) ** 2

randomization_time_mean_m = 20.0
randomization_time_variance_factor_m = 0.1
randomization_magnitude_m = 1.5

max_timepoints_on_ram = 100
seed = 36
allowed_drift_before_geometry_recalc = 1.0


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 20.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.0),
        ("kdgtp_rho_multiplier", 20.0),
        ("threshold_rac_activity_multiplier", 0.56999999999999973),
        ("threshold_rho_activity_multiplier", 0.79000000000000004),
        ("kgtp_rac_autoact_multiplier", 163.0),
        ("kgtp_rho_autoact_multiplier", 100.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 486.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 954.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.049999999999999996),
    ]
)


sub_experiment_number = 0

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    parameter_dict,
    randomization_scheme=None,
    randomization_time_mean_m=randomization_time_mean_m,
    randomization_time_variance_factor_m=randomization_time_variance_factor_m,
    randomization_time_mean_w=40.0,
    randomization_time_variance_factor_w=0.25,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=3,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-8},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=seed,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=0.5,
    default_cil=10,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
)

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    parameter_dict,
    randomization_scheme="m",
    randomization_magnitude_m=randomization_magnitude_m * 4,
    randomization_time_mean_m=randomization_time_mean_m,
    randomization_time_variance_factor_m=randomization_time_variance_factor_m,
    randomization_time_mean_w=40.0,
    randomization_time_variance_factor_w=0.25,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=3,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-8},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=seed,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=0.5,
    default_cil=10,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
)


ets.two_cells_cil_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    parameter_dict,
    randomization_scheme=None,
    randomization_magnitude_m=randomization_magnitude_m * 4,
    randomization_time_mean_m=randomization_time_mean_m,
    randomization_time_variance_factor_m=randomization_time_variance_factor_m,
    randomization_time_mean_w=40.0,
    randomization_time_variance_factor_w=0.25,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=1,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-8},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=seed,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=0.5,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
)


ets.two_cells_cil_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    parameter_dict,
    randomization_scheme=None,
    randomization_magnitude_m=randomization_magnitude_m * 4,
    randomization_time_mean_m=randomization_time_mean_m,
    randomization_time_variance_factor_m=randomization_time_variance_factor_m,
    randomization_time_mean_w=40.0,
    randomization_time_variance_factor_w=0.25,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=1,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-8},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=seed,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=0.5,
    default_cil=40.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
)

ets.two_cells_cil_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    parameter_dict,
    randomization_scheme=None,
    randomization_magnitude_m=randomization_magnitude_m * 4,
    randomization_time_mean_m=randomization_time_mean_m,
    randomization_time_variance_factor_m=randomization_time_variance_factor_m,
    randomization_time_mean_w=40.0,
    randomization_time_variance_factor_w=0.25,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=1,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-8},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=seed,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=0.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
)
