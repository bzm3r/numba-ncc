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
    # parameter_dict.update([('chemoattractant_mediated_coa_dampening_factor', 0.0), ('enable_chemoattractant_shielding_effect', False), ('randomization_scheme', 'w')])
    #
    #    parameter_dict.update([('randomization_scheme', 'm')])
    #    ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=False, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=True, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=True, zoomed_in=False)

    # ets.single_cell_polarization_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir=base_output_dir, total_time_in_hours=4., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=0.0, default_cil=0.0, num_experiment_repeats=50, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, remake_graphs=True, remake_animation=False, show_centroid_trail=True, show_randomized_nodes=True, zoomed_in=False)

    # ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=3., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=True)
    ##
    # ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=3., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False)

    # x_offset_in_corrdior = 625.0
    # slope = 0.02/40.0
    #
    # chm = 2.75
    # ets.no_corridor_chemoattraction_test(date_str, experiment_number, sub_experiment_number, parameter_dict, {'source_type': 'linear', 'x_offset_in_corridor': x_offset_in_corrdior, 'max_value': chm, 'slope': slope}, no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=10, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, timesteps_between_generation_of_intermediate_visuals=None, num_experiment_repeats=20, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=True, remake_animation=False, default_coa=coa_dict[2], default_cil=default_cil, chemotaxis_target_radius=160.0, box_width=1, box_height=1, box_y_placement_factor=0.5, num_cells=1)

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

    a, b = 0, 1
    test_chemotaxis_magnitudes = [0.0]  # [0.0, 5.0, 7.5, 10.0][a:b]
    c, d = 1, 2
    test_randomization_parameters = [
        [
            ("randomization_scheme", "m"),
            ("randomization_time_mean", 40.0),
            ("randomization_time_variance_factor", 0.1),
            ("randomization_magnitude", 5.0),
            ("randomization_node_percentage", 0.25),
        ],
        [
            ("randomization_scheme", "m"),
            ("randomization_time_mean", 40.0),
            ("randomization_time_variance_factor", 0.1),
            ("randomization_magnitude", 10.0),
            ("randomization_node_percentage", 0.25),
        ],
        [
            ("randomization_scheme", "m"),
            ("randomization_time_mean", 40.0),
            ("randomization_time_variance_factor", 0.1),
            ("randomization_magnitude", 20.0),
            ("randomization_node_percentage", 0.25),
        ],
        [
            ("randomization_scheme", "m"),
            ("randomization_time_mean", 160.0),
            ("randomization_time_variance_factor", 0.1),
            ("randomization_magnitude", 5.0),
            ("randomization_node_percentage", 0.25),
        ],
        [("randomization_scheme", None)],
    ][c:d]

    #    [
    #            [("randomization_scheme", "m"),
    #        ("randomization_time_mean", 10.0),
    #        ("randomization_time_variance_factor", 0.1),
    #        ("randomization_magnitude", 20.0),
    #        ("randomization_node_percentage", 0.25)],
    #             [("randomization_scheme", "m"),
    #        ("randomization_time_mean", 10.0),
    #        ("randomization_time_variance_factor", 0.1),
    #        ("randomization_magnitude", 10.0),
    #        ("randomization_node_percentage", 0.25)],
    #             [("randomization_scheme", "m"),
    #        ("randomization_time_mean", 40.0),
    #        ("randomization_time_variance_factor", 0.1),
    #        ("randomization_magnitude", 10.0),
    #        ("randomization_node_percentage", 0.25)],
    #              [("randomization_scheme", "m"),
    #        ("randomization_time_mean", 80.0),
    #        ("randomization_time_variance_factor", 0.1),
    #        ("randomization_magnitude", 10.0),
    #        ("randomization_node_percentage", 0.25)],
    #             [("randomization_scheme", "m"),
    #        ("randomization_time_mean", 160.0),
    #        ("randomization_time_variance_factor", 0.1),
    #        ("randomization_magnitude", 10.0),
    #        ("randomization_node_percentage", 0.25)],
    #              [("randomization_scheme", None)],
    #        ][c:d]

    x, y = 4, 5
    test_cell_group_sizes = [1, 2, 4, 9, 16, 25, 36, 49][x:y]
    test_cell_group_widths = [1, 2, 2, 3, 4, 5, 6, 7][x:y]
    test_cell_group_heights = [1, 1, 2, 3, 4, 5, 6, 7][x:y]
    num_experiment_repeats = [50, 50, 50, 50, 20, 50, 50, 50][x:y]
    intercellular_interaction_knockdown_cases = [(0.0, 1.0)]
    test_variants = []
    info_tag = ""

    #        ets.many_cells_coa_test(date_str, experiment_number, 1, copy.deepcopy(parameter_dict), no_randomization=True, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[16], default_cil=default_cil, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, box_width=4, box_height=4, num_cells=16, remake_graphs=False, remake_animation=True, show_centroid_trail=True)

    ets.chemotaxis_no_corridor_tests(
        date_str,
        experiment_number,
        sub_experiment_number,
        copy.deepcopy(parameter_dict),
        no_randomization=False,
        uniform_initial_polarization=uniform_initial_polarization,
        base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output",
        total_time_in_hours=10,
        timestep_length=2,
        verbose=True,
        integration_params=integration_params,
        max_timepoints_on_ram=max_timepoints_on_ram,
        seed=None,
        allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
        test_x_offset_in_corridor=x_offset_in_corrdior,
        test_chemotaxis_magnitudes=test_chemotaxis_magnitudes,
        test_randomization_parameters=test_randomization_parameters,
        test_chemotaxis_slope=slope,
        num_experiment_repeats=num_experiment_repeats,
        timesteps_between_generation_of_intermediate_visuals=None,
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
        produce_animation=True,
        full_print=False,
        delete_and_rerun_experiments_without_stored_env=True,
        run_experiments=True,
        remake_graphs={
            "cell specific": False,
            "all cell speeds": False,
            "group area/cell separation": False,
            "centroid related data": {
                "velocity alignment": False,
                "persistence time": False,
                "general group info": False,
                "centroid drift": False,
                "old interaction quantification": False,
                "simple interaction quantification": False,
            },
            "protrusion existence": False,
            "protrusion bias": False,
        },
        remake_animation=False,
        redo_chemotaxis_analysis=False,
        do_final_analysis=True,
        remake_final_analysis_graphs=False,
        intercellular_interaction_knockdown_cases=intercellular_interaction_knockdown_cases,
        default_coas=[coa_dict[x] for x in test_cell_group_sizes],
        default_cils=[default_cil for x in test_cell_group_sizes],
        chemotaxis_target_radius=160.0,
        box_y_placement_factor=0.5,
        num_cells=test_cell_group_sizes,
        box_widths=test_cell_group_widths,
        box_heights=test_cell_group_heights,
        info_tag=info_tag,
        show_protrusion_existence=False,
    )

#    ets.two_cells_cil_test(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=3., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, default_coa=coa_dict[2], default_cil=default_cil, num_experiment_repeats=3, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=True, no_corridor=True)

# ets.chemotaxis_threshold_test_magnitudes(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_x_offset_in_corridor=x_offset_in_corrdior, test_chemotaxis_magnitudes=test_chemotaxis_magnitudes, test_chemotaxis_slope=slope, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=True, default_coas=[coa for coa in [coa_dict[1], coa_dict[2], coa_dict[4], coa_dict[9], coa_dict[16]][M:N]], default_cil=default_cil, chemotaxis_target_radius=160.0, \  #   box_y_placement_factor=0.5,  #   num_cells=[1, 2,  #   4, 9,  #   16][M:N],  #   box_widths=[1, 2, 2,  #   3, 4][M:N],  #   box_heights=[1, 1,  #   2, 3, 4][M:N])

# ets.chemotaxis_threshold_test_magnitudes(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_x_offset_in_corridor=x_offset_in_corrdior, test_chemotaxis_magnitudes=test_chemotaxis_magnitudes, test_chemotaxis_slope=slope, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=False, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, default_coas=[coa_dict[16]], default_cil=default_cil, chemotaxis_target_radius=160.0, box_y_placement_factor=0.5, num_cells=[16], box_widths=[4], box_heights=[4])  #  # test_chemotaxis_magnitudes = [2.5]  #  # ets.chemotaxis_threshold_test_magnitudes(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10.0, timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_x_offset_in_corridor=x_offset_in_corrdior, test_chemotaxis_magnitudes=test_chemotaxis_magnitudes, test_chemotaxis_slope=slope, num_experiment_repeats=10, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=False, default_coas=[coa_dict[1], coa_dict[4], coa_dict[9], coa_dict[16]], default_cil=default_cil, chemotaxis_target_radius=160.0, box_y_placement_factor=0.5, num_cells=[1, 4, 9, 16],  #   box_widths=[1, 2, 3, 4], box_heights=[1, 2, 3, 4])

# import time  # st = time.time()  #  # ets.chemotaxis_test_group_sizes(date_str, experiment_number, sub_experiment_number, copy.deepcopy(parameter_dict), no_randomization=False, base_output_dir="C:\\Users\\bhmer\\Desktop\\numba-ncc\\output", total_time_in_hours=10., timestep_length=2, verbose=True, integration_params=integration_params, max_timepoints_on_ram=max_timepoints_on_ram, seed=None, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, test_x_offset_in_corridor=x_offset_in_corrdior, test_chemotaxis_magnitude=test_chemotaxis_magnitude, test_chemotaxis_slope=slope, num_experiment_repeats=1, timesteps_between_generation_of_intermediate_visuals=None, produce_graphs=True, produce_animation=True, full_print=True, delete_and_rerun_experiments_without_stored_env=True, run_experiments=True, remake_graphs=False, remake_animation=True, default_coa_dict=coa_dict, default_cil=default_cil, chemotaxis_target_radius=160.0, box_y_placement_factor=0.5, num_cells=[4])  # et = time.time()  # print("time taken: {}s".format(np.round(et - st, decimals=2)))
