# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 16:53:39 2015

@author: Brian
"""


import core.parameterorg as parameterorg
import core.environment as environment
import numpy as np
import visualization.datavis as datavis
import visualization.animator as animator
import os
import copy
import multiprocessing as mp
import shutil
import core.hardio as hardio
import dill

MONTHS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]


def convert_parameter_override_dictionary_into_keyvalue_tuple_list(
    base_parameter_override_dict, other_parameter_override_dict
):
    labels_in_base_dict = list(base_parameter_override_dict.keys())
    labels_in_other_dict = list(other_parameter_override_dict.keys())

    keyvalue_tuple_list = []
    for label in labels_in_base_dict:
        if label in labels_in_other_dict:
            orig_value = base_parameter_override_dict[label]
            new_value = other_parameter_override_dict[label]
            if orig_value != new_value:
                keyvalue_tuple_list.append((label, orig_value, new_value))

    return keyvalue_tuple_list


# =============================================================================


def update_pd(pd, key, orig_value, new_value):
    if pd[key] != orig_value:
        raise Exception(
            "Key {} does not have orig_value {}, instead {}".format(
                key, orig_value, pd[key]
            )
        )
    else:
        new_pd = copy.deepcopy(pd)
        new_pd.update([(key, new_value)])
        del pd

        return new_pd


# =============================================================================


def update_pd_with_keyvalue_tuples(pd, keyvalue_tuples):
    new_pd = copy.deepcopy(pd)

    for keyvalue_tuple in keyvalue_tuples:
        key, orig_value, new_value = keyvalue_tuple

        if key == "sigma_rac":
            pass

        if pd[key] != orig_value:
            raise Exception(
                "Key {} does not have orig_value {}, instead {}".format(
                    key, orig_value, pd[key]
                )
            )
        else:
            new_pd.update([(key, new_value)])

    return new_pd


# =============================================================================


def make_experiment_description_file(
    experiment_description,
    environment_dir,
    environment_wide_variable_defns,
    user_cell_group_defns,
):
    notes_fp = os.path.join(environment_dir, "experiment_notes.txt")
    notes_content = []

    notes_content.append("======= EXPERIMENT DESCRIPTION: {} =======\n\n")
    notes_content.append(experiment_description + "\n\n")
    notes_content.append("======= CELL GROUPS IN EXPERIMENT: {} =======\n\n")
    notes_content.append(repr(environment_wide_variable_defns) + "\n\n")

    num_cell_groups = len(user_cell_group_defns)
    notes_content.append(
        "======= CELL GROUPS IN EXPERIMENT: {} =======\n\n".format(num_cell_groups)
    )
    notes_content.append(repr(user_cell_group_defns) + "\n\n")

    notes_content.append("======= VARIABLE SETTINGS =======\n\n")

    for n, cell_group_defn in enumerate(user_cell_group_defns):
        pd = cell_group_defn["parameter_override_dict"]
        sorted_pd_keys = sorted(pd.keys())
        tuple_list = [(key, pd[key]) for key in sorted_pd_keys]
        notes_content.append("CELL_GROUP: {}\n".format(n))
        notes_content.append(repr(tuple_list) + "\n\n")

    with open(notes_fp, "w") as notes_file:
        notes_file.writelines(notes_content)


# ===========================================================================


def make_template_experiment_description_file(
    experiment_description,
    environment_dir,
    parameter_dict,
    environment_wide_variable_defns,
    user_cell_group_defns,
):
    notes_fp = os.path.join(environment_dir, "experiment_notes.txt")
    notes_content = []

    notes_content.append("======= EXPERIMENT DESCRIPTION: {} =======\n\n")
    notes_content.append(experiment_description + "\n\n")
    notes_content.append("======= CELL GROUPS IN EXPERIMENT: {} =======\n\n")
    notes_content.append(repr(environment_wide_variable_defns) + "\n\n")

    num_cell_groups = len(user_cell_group_defns)
    notes_content.append(
        "======= CELL GROUPS IN EXPERIMENT: {} =======\n\n".format(num_cell_groups)
    )
    notes_content.append(repr(user_cell_group_defns) + "\n\n")

    notes_content.append("======= PARAMETER DICT =======\n\n")
    sorted_keys = sorted(parameter_dict.keys())
    tuple_list = [(key, parameter_dict[key]) for key in sorted_keys]
    notes_content.append(repr(tuple_list) + "\n\n")

    notes_content.append("======= VARIABLE SETTINGS =======\n\n")

    for n, cell_group_defn in enumerate(user_cell_group_defns):
        print((list(cell_group_defn.keys())))
        pd = cell_group_defn["parameter_dict"]
        sorted_pd_keys = sorted(pd.keys())
        tuple_list = [(key, pd[key]) for key in sorted_pd_keys]
        notes_content.append("CELL_GROUP: {}\n".format(n))
        notes_content.append(repr(tuple_list) + "\n\n")

    with open(notes_fp, "w") as notes_file:
        notes_file.writelines(notes_content)


# =============================================================================


def make_analysis_description_file(
    analysis_dir, analysis_description, environment_dirs
):
    notes_fp = os.path.join(analysis_dir, "analysis_notes.txt")
    notes_content = []

    notes_content.append("======= ANALYSIS DESCRIPTION: {} =======\n\n")
    notes_content.append(analysis_description + "\n\n")

    notes_content.append("======= ENVIRONMENT DIRS: {} =======\n\n")
    for environment_dir in environment_dirs:
        notes_content.append(environment_dir + "\n\n")

    with open(notes_fp, "w") as notes_file:
        notes_file.writelines(notes_content)


# =============================================================================


def form_base_environment_name_format_string(
    experiment_number,
    num_cells_total,
    total_time,
    num_timesteps,
    num_nodes,
    height_corridor,
    width_corridor,
):

    base_env_name_format_string = (
        "EXP{}".format(experiment_number)
        + "({},{})"
        + "_NC={}_TT={}_NT={}_NN={}".format(
            num_cells_total, total_time, num_timesteps, num_nodes
        )
    )

    if height_corridor == None or width_corridor == None:
        base_env_name_format_string += "_(None)"
    else:
        base_env_name_format_string += "_({}x{})".format(
            height_corridor, width_corridor
        )

    return base_env_name_format_string


# ========================================================================


def get_experiment_directory_path(base_output_dir, date_str, experiment_number):
    return os.path.join(
        base_output_dir, "{}\\{}".format(date_str, "EXP_{}".format(experiment_number))
    )


# ========================================================================


def get_experiment_set_directory_path(base_output_dir, date_str, experiment_number):
    return os.path.join(
        base_output_dir, "{}\\SET={}".format(date_str, experiment_number)
    )


# ========================================================================


def get_template_experiment_directory_path(
    base_output_dir, date_str, experiment_number, experiment_name
):
    return os.path.join(
        base_output_dir,
        "{}\\SET={}\\{}".format(date_str, experiment_number, experiment_name),
    )


# ========================================================================


def get_analysis_directory_path(base_output_dir, date_str, analysis_number):
    return os.path.join(
        base_output_dir, "{}\\{}".format(date_str, "ANA_{}".format(analysis_number))
    )


# ========================================================================


def get_environment_directory_path(experiment_directory_path, environment_name):
    return os.path.join(experiment_directory_path, environment_name)


def get_storefile_path(env_dir):
    return os.path.join(env_dir, "store.hdf5")


def get_pickled_env_path(env_dir):
    return os.path.join(env_dir, "environment.pkl")


# ========================================================================


def run_experiments(
    experiment_directory,
    environment_name_format_strings,
    environment_wide_variable_defns,
    user_cell_group_defns_per_subexperiment,
    experiment_descriptions_per_subexperiment,
    chemoattractant_signal_fn_per_subexperiment,
    num_experiment_repeats=1,
    elapsed_timesteps_before_producing_intermediate_graphs=2500,
    elapsed_timesteps_before_producing_intermediate_animations=5000,
    produce_intermediate_visuals=True,
    produce_graphs=True,
    produce_animation=True,
    full_print=False,
    delete_and_rerun_experiments_without_stored_env=True,
    extend_simulation=False,
    new_num_timesteps=None,
):

    for repeat_number in range(num_experiment_repeats):
        for subexperiment_index, user_cell_group_defns in enumerate(
            user_cell_group_defns_per_subexperiment
        ):
            environment_name_format_string = environment_name_format_strings[
                subexperiment_index
            ]
            environment_name = environment_name_format_string.format(
                subexperiment_index, repeat_number
            )
            environment_dir = os.path.join(experiment_directory, environment_name)

            PO_set_string = "P0 SET {}, RPT {}".format(
                subexperiment_index, repeat_number
            )
            an_environment = None
            if os.path.exists(environment_dir):
                print((PO_set_string + " directory exists."))

                storefile_path = os.path.join(environment_dir, "store.hdf5")
                env_pkl_path = os.path.join(environment_dir, "environment.pkl")

                if os.path.exists(storefile_path) and os.path.exists(env_pkl_path):
                    print((PO_set_string + " stored environment exists."))
                    an_environment = retrieve_environment(
                        env_pkl_path,
                        produce_graphs,
                        produce_animation,
                        produce_intermediate_visuals,
                        simulation_execution_enabled=run_experiments,
                    )
                    if run_experiments:
                        print("Checking to see if simulation has been completed...")
                        if an_environment.simulation_complete() == True:

                            if extend_simulation != True:
                                print("Simulation has been completed. Continuing...")
                                del an_environment
                                continue
                            else:
                                assert new_num_timesteps != None
                                assert new_num_timesteps >= an_environment.num_timesteps
                                print("Extending simulation run time...")
                                an_environment.extend_simulation_runtime(
                                    new_num_timesteps
                                )
                                assert an_environment.simulation_complete() == False
                        else:
                            print("Simulation incomplete. Finishing...")
                            if extend_simulation == True:
                                assert new_num_timesteps != None
                                assert new_num_timesteps >= an_environment.num_timesteps
                                print("Extending simulation run time...")
                                an_environment.extend_simulation_runtime(
                                    new_num_timesteps
                                )
                                assert an_environment.simulation_complete() == False
                else:
                    if delete_and_rerun_experiments_without_stored_env == True:
                        print(
                            (
                                PO_set_string
                                + " directory exists, but stored environment missing -- deleting and re-running experiment."
                            )
                        )
                        shutil.rmtree(environment_dir)
                    else:
                        print(
                            (
                                PO_set_string
                                + " directory exists, but stored environment missing. Continuing regardless."
                            )
                        )
                        continue

            print(("RUNNING " + PO_set_string))
            print(("environment_dir: {}".format(environment_dir)))

            if an_environment == None:
                os.makedirs(environment_dir)
                autogenerate_debug_file(environment_dir)

                make_experiment_description_file(
                    experiment_descriptions_per_subexperiment[subexperiment_index],
                    environment_dir,
                    environment_wide_variable_defns,
                    user_cell_group_defns,
                )

                print("Creating environment...")
                parameter_overrides = [
                    x["parameter_override_dict"] for x in user_cell_group_defns
                ]

                an_environment = parameterorg.make_environment_given_user_cell_group_defns(
                    environment_name=environment_name,
                    parameter_overrides=parameter_overrides,
                    environment_dir=environment_dir,
                    user_cell_group_defns=user_cell_group_defns,
                    chemoattractant_signal_fn=chemoattractant_signal_fn_per_subexperiment[
                        subexperiment_index
                    ],
                    **environment_wide_variable_defns
                )

            an_environment.full_print = full_print

            simulation_time = an_environment.execute_system_dynamics(
                animation_settings,
                produce_intermediate_visuals=produce_intermediate_visuals,
                produce_graphs=produce_graphs,
                produce_animation=produce_animation,
                elapsed_timesteps_before_producing_intermediate_graphs=elapsed_timesteps_before_producing_intermediate_graphs,
                elapsed_timesteps_before_producing_intermediate_animations=elapsed_timesteps_before_producing_intermediate_animations,
            )

            print(("Simulation run time: {}s".format(simulation_time)))


# =======================================================================


def run_simple_experiment_and_return_cell_worker(task_defn):
    if task_defn != None:
        environment_wide_variable_defns, user_cell_group_defn = task_defn
        return run_simple_experiment_and_return_cell(
            environment_wide_variable_defns, user_cell_group_defn
        )
    else:
        return None


# =======================================================================


def run_simple_experiment_and_return_cell(
    environment_wide_variable_defns, user_cell_group_defn
):

    an_environment = parameterorg.make_environment_given_user_cell_group_defns(
        user_cell_group_defns=[user_cell_group_defn], **environment_wide_variable_defns
    )
    an_environment.full_print = False

    an_environment.execute_system_dynamics(
        {}, produce_intermediate_visuals=False, produce_final_visuals=False
    )

    return an_environment.cells_in_environment[0]


# ====================================================================


def remake_graphics(
    remake_graphs, remake_animation, remake_polarization_animation, remake_specific_timestep_snpashots, environment_dir, an_environment, animation_settings
):
    curr_tpoint = an_environment.curr_tpoint
    draw_tpoint = curr_tpoint + 1
    visuals_save_dir = os.path.join(environment_dir, "T={}".format(draw_tpoint))
    visuals_save_dir_polarization = os.path.join(environment_dir, "polarization-T={}".format(draw_tpoint))

    cell_group_indices = []
    cell_Ls = []
    cell_Ts = []
    cell_etas = []
    cell_skip_dynamics = []

    for a_cell in an_environment.cells_in_environment:
        cell_group_indices.append(a_cell.cell_group_index)
        cell_Ls.append(a_cell.L / 1e-6)
        cell_Ts.append(a_cell.T)
        cell_etas.append(a_cell.eta)
        cell_skip_dynamics.append(a_cell.skip_dynamics)

    images_global_dir = os.path.join(environment_dir, "images_global")
    images_global_dir_polarization = os.path.join(environment_dir, "images_global_polarization")

    if remake_animation:
        if os.path.exists(images_global_dir):
            shutil.rmtree(images_global_dir)

        if os.path.exists(visuals_save_dir):
            shutil.rmtree(visuals_save_dir)

    if remake_polarization_animation:
        if os.path.exists(images_global_dir_polarization):
            shutil.rmtree(images_global_dir_polarization)

        if os.path.exists(visuals_save_dir_polarization):
            shutil.rmtree(visuals_save_dir_polarization)

    ani_sets = an_environment.animation_settings
    ani_sets.update(animation_settings)

    for outdated_keywords in ["specific_timesteps_to_draw_as_svg"]:
        ani_sets.pop(outdated_keywords, None)

    animation_object = animator.EnvironmentAnimation(
        an_environment.environment_dir,
        an_environment.environment_name,
        an_environment.num_cells,
        an_environment.num_nodes,
        an_environment.num_timepoints,
        cell_group_indices,
        cell_Ls,
        cell_Ts,
        cell_etas,
        cell_skip_dynamics,
        an_environment.storefile_path,
        an_environment.data_dict_pickle_path,
        **ani_sets
    )

    an_environment.do_data_analysis_and_make_visuals(
        draw_tpoint,
        visuals_save_dir,
        ani_sets,
        animation_object,
        remake_graphs,
        remake_animation,
        remake_polarization_animation,
        remake_specific_timestep_snpashots,
    )


def determine_environment_name_and_dir(
    repeat_number, experiment_directory, template_experiment_name_format_string
):
    environment_name = template_experiment_name_format_string.format(repeat_number)
    environment_dir = os.path.join(experiment_directory, environment_name)

    return environment_name, environment_dir


def check_validity_of_new_num_timesteps(new_num_timesteps):
    type_num_new_timesteps = type(new_num_timesteps)

    if type_num_new_timesteps != int:
        raise Exception(
            "Type of new_num_timesteps is {}!".format(type_num_new_timesteps)
        )
    elif new_num_timesteps < 1:
        raise Exception("new_num_timesteps < 1, given: {}".format(new_num_timesteps))


def check_if_simulation_exists_and_is_complete(
    environment_dir,
    experiment_string,
    environment_wide_variable_defns,
    produce_graphs,
    produce_animation,
    produce_polarization_animation,
    produce_intermediate_visuals,
    extend_simulation,
    new_num_timesteps,
    delete_and_rerun_experiments_without_stored_env,
    run_experiments,
):
    if not run_experiments:
        return "check aborted, no simulation execution expected", None, False

    if not os.path.exists(environment_dir):
        return "environment dir does not exist", None, False

    storefile_path = os.path.join(environment_dir, "store.hdf5")
    # data_dict_pickle_path = os.path.join(environment_dir, "general_data_dict.pkl")
    env_pkl_path = os.path.join(environment_dir, "environment.pkl")

    if not os.path.exists(storefile_path):
        if delete_and_rerun_experiments_without_stored_env == True:
            shutil.rmtree(environment_dir)
            return "storefile does not exist, rerun requested", None, False
        else:
            return "storefile does not exist", None, False

    if not os.path.exists(env_pkl_path):
        if delete_and_rerun_experiments_without_stored_env == True:
            shutil.rmtree(environment_dir)
            return "pickled environment does not exist, rerun requested", None, False
        else:
            return "pickled environment does not exist", None, False

    an_environment = retrieve_environment(
        env_pkl_path,
        produce_graphs,
        produce_animation,
        produce_polarization_animation,
        produce_intermediate_visuals,
        environment_wide_variable_defns,
        simulation_execution_enabled=run_experiments,
    )

    if an_environment.simulation_complete() == True:
        if extend_simulation != True:
            return "simulation complete", an_environment, True
        else:
            check_validity_of_new_num_timesteps(new_num_timesteps)

            if new_num_timesteps > an_environment.num_timesteps:
                an_environment.extend_simulation_runtime(new_num_timesteps)
                return "simulation incomplete", an_environment, False
            elif an_environment.simulation_complete():
                return "simulation complete", an_environment, True
    else:
        return "simulation incomplete", an_environment, False


def create_environment(
    environment_name,
    environment_dir,
    experiment_description,
    parameter_dict,
    justify_parameters,
    environment_wide_variable_defns,
    user_cell_group_defns,
    animation_settings,
):
    os.makedirs(environment_dir)
    autogenerate_debug_file(environment_dir)
    make_template_experiment_description_file(
        experiment_description,
        environment_dir,
        parameter_dict,
        environment_wide_variable_defns,
        user_cell_group_defns,
    )

    an_environment = parameterorg.make_environment_given_user_cell_group_defns(
        animation_settings,
        environment_name=environment_name,
        environment_dir=environment_dir,
        user_cell_group_defns=user_cell_group_defns,
        justify_parameters=justify_parameters,
        **environment_wide_variable_defns
    )

    return an_environment, animation_settings


def run_template_experiments(
    experiment_directory,
    parameter_dict,
    environment_wide_variable_defns,
    user_cell_group_defns_per_subexperiment,
    experiment_descriptions_per_subexperiment,
    num_experiment_repeats=1,
    elapsed_timesteps_before_producing_intermediate_graphs=2500,
    elapsed_timesteps_before_producing_intermediate_animations=5000,
    animation_settings={},
    produce_intermediate_visuals=True,
    produce_graphs=True,
    produce_animation=True,
    produce_polarization_animation=True,
    full_print=False,
    delete_and_rerun_experiments_without_stored_env=True,
    run_experiments=False,
    extend_simulation=False,
    new_num_timesteps=None,
    justify_parameters=True,
    remake_graphs=False,
    remake_animation=False,
    remake_polarization_animation=False,
    remake_specific_timestep_snapshots=False,
):
    template_experiment_name_format_string = "RPT={}"
    for repeat_number in range(num_experiment_repeats):
        for subexperiment_index, user_cell_group_defns in enumerate(
            copy.deepcopy(user_cell_group_defns_per_subexperiment)
        ):
            environment_name, environment_dir = determine_environment_name_and_dir(
                repeat_number,
                experiment_directory,
                template_experiment_name_format_string,
            )

            experiment_string = "RPT {}".format(repeat_number)
            message, an_environment, simulation_complete = check_if_simulation_exists_and_is_complete(
                environment_dir,
                experiment_string,
                environment_wide_variable_defns,
                produce_graphs,
                produce_animation,
                produce_polarization_animation,
                produce_intermediate_visuals,
                extend_simulation,
                new_num_timesteps,
                delete_and_rerun_experiments_without_stored_env,
                run_experiments,
            )

            print(("{}: {}".format(environment_name, message)))

            if message in [
                "environment dir does not exist",
                "pickled environment does not exist, rerun requested",
                "storefile does not exist, rerun requested",
                "simulation incomplete",
            ]:
                if run_experiments:
                    "running simulation..."
                    if message != "simulation incomplete":
                        an_environment, animation_settings = create_environment(
                            environment_name,
                            environment_dir,
                            experiment_descriptions_per_subexperiment[
                                subexperiment_index
                            ],
                            parameter_dict,
                            justify_parameters,
                            environment_wide_variable_defns,
                            user_cell_group_defns,
                            animation_settings,
                        )

                    an_environment.full_print = full_print

                    simulation_time = an_environment.execute_system_dynamics(
                        animation_settings,
                        produce_intermediate_visuals=produce_intermediate_visuals,
                        produce_graphs=produce_graphs,
                        produce_animation=produce_animation,
                        produce_polarization_animation=produce_polarization_animation,
                        elapsed_timesteps_before_producing_intermediate_graphs=elapsed_timesteps_before_producing_intermediate_graphs,
                        elapsed_timesteps_before_producing_intermediate_animations=elapsed_timesteps_before_producing_intermediate_animations,
                    )
                    print(("Simulation run time: {}s".format(simulation_time)))

            if (
                message == "simulation incomplete" and run_experiments
            ) or message == "simulation complete":
                remake_graphics(
                    remake_graphs,
                    remake_animation,
                    remake_polarization_animation,
                    remake_specific_timestep_snapshots,
                    environment_dir,
                    an_environment,
                    animation_settings,
                )


# =====================================================================


def raise_scriptname_error(scriptname, message=None):
    if message == None:
        raise Exception(
            "Given script name does not follow template YYYY_MMM_DD_experiment_script_X: {}".format(
                scriptname
            )
        )
    else:
        raise Exception("{}: {}".format(message, scriptname))


def check_if_date_is_proper(date_pieces, scriptname):
    if not date_pieces[0].isnumeric():
        raise_scriptname_error(scriptname, "Year is not a number")
    if not len(date_pieces[0]) == 4:
        raise_scriptname_error(
            scriptname, "Incorrect number of digits in year (4 needed)"
        )
    if not date_pieces[1].upper() in MONTHS:
        raise_scriptname_error(scriptname, "Unknown month given")
    if not date_pieces[2].isnumeric():
        raise_scriptname_error(scriptname, "Day is not a number")
    if not len(date_pieces[2]) == 2:
        raise_scriptname_error(
            scriptname, "Incorrect number of digits in month (2 needed)"
        )


# ================================================================


def check_if_experiment_number_is_proper(expnum, scriptname):
    if not expnum.isnumeric():
        raise_scriptname_error(scriptname, "Experiment number is not a number")


# ================================================================


def get_date_and_experiment_number(scriptname):
    name_pieces = scriptname.split("_")
    if len(name_pieces) != 6:
        raise_scriptname_error(scriptname)
    else:
        date_pieces = name_pieces[:3]
        # check_if_date_is_proper(date_pieces, scriptname)
        DATE_STR = "_".join([piece.upper() for piece in date_pieces])

        # check_if_experiment_number_is_proper(name_pieces[5], scriptname)
        EXPERIMENT_NUMBER = int(name_pieces[5])

        return DATE_STR, EXPERIMENT_NUMBER


# ================================================================


def load_empty_env(empty_env_pickle_path):
    env = environment.Environment(shell_environment=True)
    env.load_from_pickle(empty_env_pickle_path)

    return env


# ================================================================


def retrieve_environment(
    empty_env_pickle_path,
    produce_intermediate_visuals,
    produce_graphs,
    produce_animation,
    produce_polarization_animation,
    environment_wide_variable_defns,
    simulation_execution_enabled=False,
):
    env = load_empty_env(empty_env_pickle_path)

    if env != None:
        env.environment_dir = os.path.split(empty_env_pickle_path)[0]
        env.init_from_store(
            environment_wide_variable_defns,
            simulation_execution_enabled=simulation_execution_enabled,
        )
        env.produce_intermediate_visuals = produce_intermediate_visuals
        env.produce_graphs = produce_graphs
        env.produce_animation = produce_animation
        env.produce_polarization_animation = produce_polarization_animation
    else:
        raise Exception(
            "Could not load environment pickle file at: {}".format(
                empty_env_pickle_path
            )
        )

    return env


# =======================================================================


def get_subexperiment_number_from_folder_string(experiment_number, folder_string):
    info_tokens = folder_string.split("_")

    experiment_number_token = info_tokens[0]

    subexperiment_number, repeat_number = [
        int(x)
        for x in (
            experiment_number_token[len("EXP{}".format(experiment_number)) + 1 : -1]
        ).split(",")
    ]

    return subexperiment_number, repeat_number


# =======================================================================


def get_environment_dirs_given_relevant_experiment_info(
    base_output_dir, relevant_experiment_info
):
    env_dirs = []

    for experiment_info_tuple in relevant_experiment_info:
        date_str, experiment_number, subexp_number = experiment_info_tuple
        experiment_directory = os.path.join(
            base_output_dir, date_str, "EXP_{}".format(experiment_number)
        )

        for d in os.listdir(experiment_directory):
            env_dir = os.path.join(experiment_directory, d)
            if os.path.isdir(env_dir):
                subexperiment_number, repeat_number = get_subexperiment_number_from_folder_string(
                    experiment_number, d
                )

                if subexperiment_number == subexp_number:
                    env_dirs.append(os.path.join(env_dir))

    return env_dirs


def autogenerate_debug_file(environment_dir):
    with open(os.path.join(environment_dir, "debug_script.py"), "w") as f:
        lines = [
            "# -*- coding: utf-8 -*-",
            "# autogenerated debug file",
            "import general.exec_utils as eu",
            "import os",
            "",
            "environment_dir = os.getcwd()",
            "storefile_path = eu.get_storefile_path(environment_dir)",
            "env = eu.retrieve_environment(eu.get_pickled_env_path(environment_dir), False, False)",
        ]
        lines = [line + "\n" for line in lines]
        f.writelines(lines)
