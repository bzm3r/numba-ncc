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
        ("max_coa_signal", -1.0),
        ("max_chemoattractant_signal", 0.0),
    ]
)

randomization_time_mean_m = 40.0
randomization_time_variance_factor_m = 0.1

max_timepoints_on_ram = 6000
seed = 2836
allowed_drift_before_geometry_recalc = 20.0

remake_animation = False
remake_graphs = False
do_final_analysis = True

default_cil = 60.0
integration_params = {"rtol": 1e-4}

base_output_dir = "B:\\numba-ncc\\output"

parameter_dict.update(
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
        ("enable_chemoattractant_shielding_effect", False),
        ("chemoattractant_shielding_effect_length_squared", (40e-6) ** 2),
        ("chemoattractant_mediated_coa_dampening_factor", 0.0),
        ("chemoattractant_mediated_coa_production_factor", 0.0),
    ]
)

sub_experiment_number = 0

coa_dict = {49: 8.0, 36: 9.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0, 2: 24.0, 1: 24.0}

if __name__ == "__main__":
    uniform_initial_polarization = False
    test_chemotaxis_magnitude = 2.75
    slope = 0.02 / 40.0
    x_offset_in_corrdior = 625.0
    parameter_dict.update(
        [
            ("chemoattractant_mediated_coa_dampening_factor", 0.0),
            ("enable_chemoattractant_shielding_effect", False),
            ("randomization_scheme", "m"),
        ]
    )
    standard_rps = [
        ("randomization_scheme", "m"),
        ("randomization_time_mean", 40.0),
        ("randomization_time_variance_factor", 0.1),
        ("randomization_magnitude", 10.0),
        ("randomization_node_percentage", 0.25),
    ]
    parameter_dict.update(standard_rps)

    test_chemotaxis_magnitudes = [0.0]
    test_randomization_parameters = [
        [("randomization_scheme", None)],
    ]
    test_num_cells_responsive_to_chemoattractant = [-1]

    test_cell_group_sizes = [1]
    test_cell_group_widths = [1]
    test_cell_group_heights = [1]
    num_experiment_repeats = [1]
    #intercellular_interaction_knockdown_cases = [(1.0, 1.0)]
    intercellular_interaction_knockdown_cases = [
        (0.0, 0.0),
    ]
    #intercellular_interaction_knockdown_cases = [(0.0, 1.0), (0.1, 1.0), (0.15, 1.0), (0.2, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (1.0, 1.0)]
    # intercellular_interaction_knockdown_cases = cil_knockdown + [(1.0, 1.0), (1.0, 0.75), (1.0, 0.5), (1.0, 0.25), (1.0, 0.0)]
    # intercellular_interaction_knockdown_cases =
    test_variants = []
    info_tag = ""

    #        ets.many_cells_coa_test(date_str, experiment_number, 1, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, remake_graphs=False, remake_animation=True, show_centroid_trail=True)

    ets.rust_comparison_test(
        date_str,
        experiment_number,
        sub_experiment_number,
        parameter_dict,
        num_cells_responsive_to_chemoattractant=-1,
        uniform_initial_polarization=False,
        no_randomization=True,
        base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output",
        total_time_in_hours=6.0,
        timestep_length=2,
        verbose=True,
        integration_params=integration_params,
        max_timepoints_on_ram=max_timepoints_on_ram,
        seed=None,
        allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
        default_coa=0,
        default_cil=0,
        num_experiment_repeats=1,
        timesteps_between_generation_of_intermediate_visuals=None,
        produce_animation=True,
        produce_polarization_animation=False,
        produce_graphs={
            "cell specific": True,
            "all cell speeds": True,
            "group area/cell separation": True,
            "centroid related data": {
                "velocity alignment": True,
                "persistence time": True,
                "general group info": True,
                "centroid drift": True,
                "old interaction quantification": True,
                "simple interaction quantification": True,
            },
            "protrusion existence": True,
            "protrusion bias": True,
        },
        specific_timesteps_to_draw=[],
        full_print=True,
        delete_and_rerun_experiments_without_stored_env=True,
        box_width=1,
        box_height=1,
        box_y_placement_factor=0.0,
        cell_placement_method="",
        max_placement_distance_factor=1.0,
        init_random_cell_placement_x_factor=0.25,
        num_cells=1,
        run_experiments=True,
        remake_graphs=False,
        remake_animation=False,
        remake_polarization_animation=False,
        remake_specific_timestep_snapshots=False,
        do_final_analysis=True,
        remake_final_analysis_graphs=False,
        biased_rgtpase_distrib_defn_dict={
            "default": ["biased nodes", [0, 1, 2, 3], 0.3]
        },
        graph_group_centroid_splits=False,
        global_scale=1,
        show_coa_overlay=False,
        coa_overlay_resolution=10,
        justify_parameters=True,
        colorscheme="normal",
        chemotaxis_target_radius=-1.0,
        show_centroid_trail=False,
        show_chemoattractant=True,
        show_protrusion_existence=True,
    )
