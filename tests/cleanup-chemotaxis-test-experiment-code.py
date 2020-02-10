# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:37:52 2019

@author: Brian
"""
import numpy as np


def setup_initial_rgtpase_bias(uniform_initial_polarization):
    if uniform_initial_polarization == False:
        return {"default": ["unbiased random", np.array([0, 2 * np.pi]), 0.3]}
    else:
        return {"default": ["unbiased uniform", np.array([0, 2 * np.pi]), 0.3]}


def determine_chemotaxis_success_info_save_paths(experiment_dir):
    chemotaxis_success_save_fp = os.path.join(
        experiment_dir, "chemotaxis_success_per_repeat.np"
    )
    chemotaxis_min_dist_save_fp = os.path.join(
        experiment_dir, "chemotaxis_min_dist_per_repeat.np"
    )

    return (chemotaxis_success_save_fp, chemotaxis_min_dist_save_fp)


def perform_chemotaxis_analysis(
    run_experiments,
    experiment_dir,
    num_repeats,
    produce_graphs,
    produce_animation,
    environment_wide_variable_defns,
    total_time_in_hours,
    chemotaxis_success_ratios_per_variant_per_num_cells,
    chemotaxis_min_distances_per_variant_per_num_cells,
    source_x,
    source_y,
    chemotaxis_target_radius,
):
    chemotaxis_success_save_fp, chemotaxis_min_dist_save_fp = determine_chemotaxis_success_info_save_paths(
        experiment_dir
    )

    experiment_name_format_string = "RPT={}"
    chemotaxis_success_per_repeat = None

    chemotaxis_success_ratios = []
    chemotaxis_min_distances = []

    if run_experiments == False:
        if not os.path.exists(experiment_dir):
            raise Exception("Experiment directory does not exist.")
        else:
            for rpt_number in range(num_repeats):
                environment_name = experiment_name_format_string.format(rpt_number)
                environment_dir = os.path.join(experiment_dir, environment_name)
                if not os.path.exists(environment_dir):
                    raise Exception("Environment directory does not exist.")

                storefile_path = eu.get_storefile_path(environment_dir)
                if not os.path.isfile(storefile_path):
                    raise Exception("Storefile does not exist.")

                relevant_environment = eu.retrieve_environment(
                    eu.get_pickled_env_path(environment_dir),
                    False,
                    produce_graphs,
                    produce_animation,
                    environment_wide_variable_defns,
                )
                if not (
                    relevant_environment.simulation_complete()
                    and (
                        relevant_environment.curr_tpoint
                        * relevant_environment.T
                        / 3600.0
                    )
                    == total_time_in_hours
                ):
                    raise Exception("Simulation is not complete.")

                chemotaxis_success_per_repeat = np.load(
                    chemotaxis_success_save_fp + ".npy"
                )
                chemotaxis_min_dist_per_repeat = np.load(
                    chemotaxis_min_dist_save_fp + ".npy"
                )
    else:
        all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_centroid_x_per_timestep_per_repeat, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_speed_per_timestep_per_repeat, fit_group_x_velocity_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, cell_separations_per_repeat, transient_end_times_per_repeat, areal_strains_per_cell_per_repeat = collate_final_analysis_data(
            num_repeats, experiment_dir
        )

        protrusion_lifetimes_and_directions = []
        for (
            protrusion_lifetime_dirn_per_cell
        ) in all_cell_protrusion_lifetimes_and_directions_per_repeat:
            for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                for l, d in protrusion_lifetime_dirn:
                    protrusion_lifetimes_and_directions.append((l, d))

        datavis.graph_protrusion_lifetimes_radially(
            protrusion_lifetimes_and_directions,
            12,
            total_time_in_hours * 60.0,
            save_dir=experiment_dir,
            save_name="all_cells_protrusion_life_dir",
        )

        chemotaxis_success_per_repeat = []
        chemotaxis_min_dist_per_repeat = []
        for rpt_number in range(num_repeats):
            environment_name = experiment_name_format_string.format(rpt_number)
            environment_dir = os.path.join(experiment_dir, environment_name)
            storefile_path = eu.get_storefile_path(environment_dir)
            # empty_env_pickle_path, produce_intermediate_visuals, produce_final_visuals, environment_wide_variable_defns, simulation_execution_enabled=False
            relevant_environment = eu.retrieve_environment(
                eu.get_pickled_env_path(environment_dir),
                False,
                produce_graphs,
                produce_animation,
                environment_wide_variable_defns,
            )

            success, closest_to_source = cu.analyze_chemotaxis_success(
                relevant_environment,
                storefile_path,
                rpt_number,
                source_x,
                source_y,
                chemotaxis_target_radius,
            )

            chemotaxis_success_per_repeat.append(success)
            chemotaxis_min_dist_per_repeat.append(closest_to_source)

        success_protrusion_lifetimes_and_directions = []
        fail_protrusion_lifetimes_and_directions = []
        for i, protrusion_lifetime_dirn_per_cell in enumerate(
            all_cell_protrusion_lifetimes_and_directions_per_repeat
        ):
            if chemotaxis_success_per_repeat[i] == 1:
                for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                    for l, d in protrusion_lifetime_dirn:
                        success_protrusion_lifetimes_and_directions.append((l, d))
            else:
                for protrusion_lifetime_dirn in protrusion_lifetime_dirn_per_cell:
                    for l, d in protrusion_lifetime_dirn:
                        fail_protrusion_lifetimes_and_directions.append((l, d))

        datavis.graph_protrusion_lifetimes_radially(
            success_protrusion_lifetimes_and_directions,
            12,
            total_time_in_hours * 60.0,
            save_dir=experiment_dir,
            save_name="successful_cells_protrusion_lifetime_dirn_N={}".format(
                np.sum(chemotaxis_success_per_repeat)
            ),
        )

        datavis.graph_protrusion_lifetimes_radially(
            fail_protrusion_lifetimes_and_directions,
            12,
            total_time_in_hours * 60.0,
            save_dir=experiment_dir,
            save_name="fail_cells_protrusion_lifetime_dirn_N={}".format(
                num_repeats - np.sum(chemotaxis_success_per_repeat)
            ),
        )
        np.save(chemotaxis_success_save_fp, chemotaxis_success_per_repeat)
        np.save(chemotaxis_min_dist_save_fp, chemotaxis_min_dist_per_repeat)

    chemotaxis_success_ratios.append(
        np.sum(chemotaxis_success_per_repeat[:num_repeats]) / num_repeats
    )
    chemotaxis_min_distances.append(chemotaxis_min_dist_per_repeat)

    chemotaxis_success_ratios_per_variant_per_num_cells.append(
        copy.deepcopy(chemotaxis_success_ratios)
    )
    chemotaxis_min_distances_per_variant_per_num_cells.append(
        copy.deepcopy(chemotaxis_min_distances)
    )

    return (
        chemotaxis_success_ratios,
        chemotaxis_min_distances,
        chemotaxis_success_ratios_per_variant_per_num_cells,
        chemotaxis_min_distances_per_variant_per_num_cells,
    )


def chemotaxis_no_corridor_tests(
    date_str,
    experiment_number,
    sub_experiment_number,
    parameter_dict,
    no_randomization=False,
    uniform_initial_polarization=False,
    base_output_dir="B:\\numba-ncc\\output\\",
    total_time_in_hours=3,
    timestep_length=2,
    verbose=True,
    integration_params={"method": "odeint", "rtol": 1e-4},
    max_timepoints_on_ram=10,
    seed=None,
    allowed_drift_before_geometry_recalc=1.0,
    test_x_offset_in_corridor=625.0,
    test_chemotaxis_magntidues=[],
    test_randomization_parameters=[],
    test_chemotaxis_slope=0.0016,
    num_experiment_repeats=10,
    timesteps_between_generation_of_intermediate_visuals=None,
    produce_animation=True,
    produce_graphs=True,
    full_print=True,
    delete_and_rerun_experiments_without_stored_env=True,
    run_experiments=True,
    remake_graphs=False,
    remake_animation=False,
    default_coas=[],
    default_cil=60.0,
    chemotaxis_target_radius=160.0,
    box_widths=[],
    box_heights=[],
    box_y_placement_factor=0.5,
    num_cells=[],
):

    assert len(num_cells) == len(box_widths) == len(box_heights)

    num_cases = len(num_cells)
    if type(num_experiment_repeats) == int:
        num_experiment_repeats = [num_experiment_repeats] * num_cases

    chemotaxis_success_ratios_per_variant_per_num_cells = []
    chemotaxis_min_distances_per_variant_per_num_cells = []

    experiment_set_directory = eu.get_experiment_set_directory_path(
        base_output_dir, date_str, experiment_number
    )

    biased_rgtpase_distrib_defn_dict = setup_initial_rgtpase_bias(
        uniform_initial_polarization
    )

    if not (
        len(test_chemotaxis_magntidues) == 1 or len(test_randomization_parameters) == 1
    ):
        raise Exception(
            "Currently, chemotaxis_no_corridors_test is designed to handle one of test_chemotaxis_magntidues or test_randomization_parameters having more than one case."
        )

    if len(test_chemotaxis_magntidues) == 1:
        test_variants = test_chemotaxis_magntidues
        variant_label = "mag"
        test_randomization_parameters = [
            copy.deepcopy(parameter_dict).update(test_randomization_parameters[0])
        ]
    else:
        test_variants = test_randomization_parameters
        variant_label = "rand"
        test_randomization_parameters = [
            copy.deepcopy(parameter_dict).update(rp)
            for rp in test_randomization_parameters
        ]

    for nci, nr, nc, bh, bw in zip(
        np.arange(num_cases), num_experiment_repeats, num_cells, box_heights, box_widths
    ):
        for xi, vv in enumerate(test_variants):
            print("=========")
            print("{}: {}".format(variant_label, vv))

            if variant_label == "mag":
                chm = vv
                pd = test_randomization_parameters[0]
            else:
                chm = test_chemotaxis_magntidues[0]
                pd = vv

            experiment_name, drift_args, environment_wide_variable_defns, source_x, source_y = no_corridor_chemoattraction_test(
                date_str,
                experiment_number,
                sub_experiment_number,
                pd,
                chemoattractant_source_definition={
                    "source_type": "linear",
                    "x_offset_in_corridor": test_x_offset_in_corridor,
                    "max_value": chm,
                    "slope": test_chemotaxis_slope,
                },
                no_randomization=no_randomization,
                base_output_dir=base_output_dir,
                total_time_in_hours=total_time_in_hours,
                timestep_length=timestep_length,
                verbose=verbose,
                integration_params=integration_params,
                max_timepoints_on_ram=max_timepoints_on_ram,
                seed=seed,
                allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc,
                default_coa=default_coas[nci],
                default_cil=default_cil,
                num_experiment_repeats=nr,
                timesteps_between_generation_of_intermediate_visuals=timesteps_between_generation_of_intermediate_visuals,
                produce_graphs=produce_graphs,
                produce_animation=produce_animation,
                full_print=full_print,
                delete_and_rerun_experiments_without_stored_env=delete_and_rerun_experiments_without_stored_env,
                box_width=bw,
                box_height=bh,
                box_y_placement_factor=box_y_placement_factor,
                num_cells=nc,
                run_experiments=run_experiments,
                remake_graphs=remake_graphs,
                remake_animation=remake_animation,
                do_final_analysis=True,
                chemotaxis_target_radius=chemotaxis_target_radius,
                show_centroid_trail=False,
                biased_rgtpase_distrib_defn_dict=biased_rgtpase_distrib_defn_dict,
            )

            experiment_dir = eu.get_template_experiment_directory_path(
                base_output_dir, date_str, experiment_number, experiment_name
            )

            chemotaxis_success_ratios_per_variant_per_num_cells, chemotaxis_min_distances_per_variant_per_num_cells = perform_chemotaxis_analysis(
                run_experiments,
                experiment_dir,
                nr,
                produce_graphs,
                produce_animation,
                environment_wide_variable_defns,
                total_time_in_hours,
                chemotaxis_success_ratios_per_variant_per_num_cells,
                chemotaxis_min_distances_per_variant_per_num_cells,
                source_x,
                source_y,
                chemotaxis_target_radius,
            )

    print("=========")
    datavis.graph_chemotaxis_efficiency_data(
        test_variants,
        test_variants_label,
        [test_chemotaxis_slope] * len(test_variants),
        chemotaxis_success_ratios_per_variant_per_num_cells,
        num_experiment_repeats,
        num_cells,
        box_widths,
        box_heights,
        save_dir=experiment_set_directory,
    )

    datavis.graph_chemotaxis_closest_distance_data(
        test_variants,
        test_variants_label,
        [test_chemotaxis_slope] * len(test_variants),
        chemotaxis_min_distances_per_variant_per_num_cells,
        num_experiment_repeats,
        num_cells,
        box_widths,
        box_heights,
        save_dir=experiment_set_directory,
    )

    print("Complete.")
