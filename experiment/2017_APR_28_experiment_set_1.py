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
    ]
)

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 100
seed = 2836
allowed_drift_before_geometry_recalc = 20.0
remake_visualizations = False

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 10.0),
        ("kdgtp_rho_multiplier", 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 0

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Cell polarizes, but there are two issues. 

1) Strain is too high.

2) RhoA is too active (above 0.4 level).

I am going to tackle the second issue. How can RhoA activation rates be reduced? 

a) decrease kgtp_rho
b) increase kdgtp_rho
c) decrease kgtp_rho_autoact

Of all three, I think c) is the best because RhoA doesn't have as much dampening it out as Rac1 does, so its really the auto-activation that is causing it to overload like that. 
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 10.0),
        ("kdgtp_rho_multiplier", 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 1

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Solved the Rho issue, but Rac1 is now a bit too much. To deal with this, I could:
    
a) reduce kgtp_rac

b) increase kdgtp_rac
"""


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 0.5 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 10.0),
        ("kdgtp_rho_multiplier", 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 2

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 3

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Increasing kdgtp_rac was far more effective, but RhoA (0.4) is much more active than Rac1 (~0.3) now. This wouldn't really be a bad thing, but it is scrunching up the cell seriously. I think in general RhoA should only be active around the 0.1 level. Will test the effect of reducing kgtp_rho, increading kdgtp_rho, and reducing kgtp_rho_autoact
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 0.5 * 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 4

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 1.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 5

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.25 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 6

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Decreasing kgtp_rho had minimal effect. Increasing kdgtp_rho had reasonable effect. Decreasing kdgtp_rho_autoact was even better. I would stick with decreasing kdgtp_rho, but I think it will be useful in the future, when trying to properly implement CIL. So, will try to reduce kdgtp_rho further.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.0 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 7

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("stiffness_edge", 4000.0),
        ("randomization_magnitude", 10.0),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_node_percentage", 0.5),
    ]
)

sub_experiment_number = 8

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Increasing kdgtp_rho by 2.5 times is enough. 

Now, several issues need fixing. Rac1 is very active (0.4) and forming multiple strong fronts. Strain inhibition really doesn't seem to do much to help out. What if we increase the magnitude of strain inhibition?
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.0 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
    ]
)

sub_experiment_number = 9

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.0 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 3.0),
    ]
)

sub_experiment_number = 10

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Increasing tension inhibition magnitude is very effective in reducing overall strain, and controlling Rac1. However, now RhoA is beginning to be problematic (not too much, but a bit). So, increasing RhoA kdgtp again by a bit.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.25 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
    ]
)

sub_experiment_number = 11

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
    ]
)

sub_experiment_number = 12

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
    ]
)

sub_experiment_number = 13

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Okay, that seems to take care of the issue, but Rac1 fronts are not really able to develop into a single cohesive unit, and cell strain is quite high. 

Does this mean that Rac1 is applying too much of a force?
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
        ("max_force_rac", 0.5 * 10e3),
    ]
)

sub_experiment_number = 14

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)


"""
Things are looking better, but still the cell strain is simply too high...reducing max_force_rac further
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
        ("max_force_rac", 0.25 * 10e3),
    ]
)

sub_experiment_number = 15

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.0),
        ("max_force_rac", 0.1 * 10e3),
    ]
)

sub_experiment_number = 16

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Finally, we have strains in the ~0.1 range. Okay, but now the issue that is occurring is that we are way above the half strain mark, and YET, yet there isn't enough strain inhibition. So, we increase the strain inhibition magnitude a bit.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 2.5),
        ("max_force_rac", 0.1 * 10e3),
    ]
)

sub_experiment_number = 17

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 3.0),
        ("max_force_rac", 0.1 * 10e3),
    ]
)

sub_experiment_number = 18

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 3.5),
        ("max_force_rac", 0.1 * 10e3),
    ]
)

sub_experiment_number = 19

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
    ]
)

sub_experiment_number = 20

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Ladies and gentlemen, we have polarization! However, the cell speed is far too low, and it may be that RhoA values are too high for randomization. In any case, let us first solve the issue of cell speed being too low, by tuning eta.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.5 * 1e5),
    ]
)

sub_experiment_number = 21

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.25 * 1e5),
    ]
)

sub_experiment_number = 22

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 23

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
I think eta is low enough, but we don't have enough Rac1 activation. 
"""


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.25 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 24

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.5 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 25

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 2.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 1.5 * 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 26

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Decreasing kdgtp_rac, or increasing kgtp_rac are both fruitful here. So, we choose to decrease increase kgtp_rac, because it seems that the cell polarizes faster in this case. Actually, one solution might be to try and decrease kgtp_rho, because it seems there is a bit too much of it around. 
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.0 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 27

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.25 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 28

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Okay, now RhoA is sufficiently suppressed. However, a Rac1 front isn't quite as strong as we'd like it to be. Currently, cell speed is 2.0, so let us try increasing Rac activation a bit.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.25 * 10.0),
        ("kdgtp_rho_multiplier", 3.25 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 29

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.5 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.25 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 30

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.25 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.25 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 31

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=1,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)
"""
Okay, weirdly enough, in both these cases, we see a small but significant increase in RhoA. What's up with that? It may be that RhoA was not yet sufficiently suppressed, so I am running more runs of experiment 28.

Indeed, it turns out that RhoA was not yet sufficiently suppressed in experiment 28, as the low Rho seem was mostly due to initial conditions. So, let us try to complete suppression of Rho.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 32

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.75 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 33

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
We use the parameter set for experiment 32, and increase Rac1 activation a bit.
"""


parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.25 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 34

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.5 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 35

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)


"""
Trying to slightly improve on 34.
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.3 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 36

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

"""
Sometimes, the polarization isn't quite there, and for some reason, the RhoA is too active, but whatever. Let's just fix how the polarization isn't always there, by making Rac1 a teeny bit more powerful in terms of its strain creating ability?
"""

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.3 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.0),
        ("max_force_rac", 0.15 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 37

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.3 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.25),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 38

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.3 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.5),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 39

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.3 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.75),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 40

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.5 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.75),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 41

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

parameter_dict.update(
    [
        ("kgtp_rac_multiplier", 1.75 * 10.0),
        ("kgtp_rho_multiplier", 10.0),
        ("kdgtp_rac_multiplier", 1.5 * 10.0),
        ("kdgtp_rho_multiplier", 3.5 * 10.0),
        ("threshold_rac_activity_multiplier", 0.2),
        ("threshold_rho_activity_multiplier", 0.2),
        ("kgtp_rac_autoact_multiplier", 250.0),
        ("kgtp_rho_autoact_multiplier", 0.5 * 250.0),
        ("kdgtp_rac_mediated_rho_inhib_multiplier", 2000.0),
        ("kdgtp_rho_mediated_rac_inhib_multiplier", 2000.0),
        ("tension_mediated_rac_inhibition_half_strain", 0.02),
        ("tension_mediated_rac_inhibition_magnitude", 4.75),
        ("max_force_rac", 0.1 * 10e3),
        ("eta", 0.1 * 1e5),
    ]
)

sub_experiment_number = 42

ets.single_cell_polarization_test(
    date_str,
    experiment_number,
    sub_experiment_number,
    copy.deepcopy(parameter_dict),
    no_randomization=True,
    base_output_dir="A:\\numba-ncc\\output\\",
    total_time_in_hours=2.0,
    timestep_length=2,
    verbose=True,
    integration_params={"rtol": 1e-2},
    max_timepoints_on_ram=max_timepoints_on_ram,
    seed=None,
    allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
    default_coa=4.0,
    default_cil=80.0,
    num_experiment_repeats=3,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_final_visuals=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    remake_visualizations=remake_visualizations,
)

# ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=2., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, migr_bdry_height_factor=1.0, remake_visualizations=remake_visualizations)

# ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=0.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=7, num_cells_height=7, remake_visualizations=remake_visualizations)

# ets.many_cells_coa_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=7, num_cells_height=7, remake_visualizations=remake_visualizations)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=2, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=2, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=20.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=3, num_cells_height=3, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=16.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=False)


# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=20.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=8.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=False)

# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=False)

# parameter_dict.update([('kgtp_rac_multiplier', 5.0),
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
# sub_experiment_number = 1
#
# ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=4.0, default_cil=80.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=2, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=2, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=16.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=remake_visualizations)
#
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=remake_visualizations)
#
#
# parameter_dict.update([('kgtp_rac_multiplier', 5.0),
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
# sub_experiment_number = 2
#
# ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=4.0, default_cil=80.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=1, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=32.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=2, num_cells_height=2, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=28.0, default_cil=40.0, num_experiment_repeats=5, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=4, num_cells_height=2, remake_visualizations=remake_visualizations)
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=16.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=5, num_cells_height=3, remake_visualizations=remake_visualizations)
#
#
# ets.corridor_migration_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="A:\\numba-ncc\\output\\", total_time_in_hours=6., timestep_length=2, verbose=True, integration_params={'rtol': 1e-2}, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=20.0, default_coa=12.0, default_cil=40.0, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_final_visuals=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, num_cells_width=10, num_cells_height=5, remake_visualizations=remake_visualizations)
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
