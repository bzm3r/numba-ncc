# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 16:53:39 2015

@author: Brian
"""


import core.parameterorg as parameterorg
import os
import copy
import shutil

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
    "DEC",]

# ========================================================================
def raise_scriptname_error(scriptname, message=None):
    if message is None:
        raise Exception(
            "Given script name does not follow template YYYY_MMM_DD_experiment_script_X: {}".format(
                scriptname
            )
        )
    else:
        raise Exception("{}: {}".format(message, scriptname))

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

def get_template_experiment_directory_path(
    base_output_dir, date_str, experiment_number, experiment_name
):
    return os.path.join(
        base_output_dir,
        "{}\\SET={}\\{}".format(date_str, experiment_number, experiment_name),
    )


# ========================================================================

def get_environment_directory_path(experiment_directory_path, environment_name):
    return os.path.join(experiment_directory_path, environment_name)

# ========================================================================

def determine_environment_name_and_dir(
    repeat_number, experiment_directory, template_experiment_name_format_string
):
    environment_name = template_experiment_name_format_string.format(repeat_number)
    environment_dir = os.path.join(experiment_directory, environment_name)

    return environment_name, environment_dir

def create_environment(
    environment_name,
    environment_dir,
        justify_parameters,
    environment_wide_variable_defns,
    user_cell_group_defns,
):
    if os.path.exists(environment_dir):
        shutil.rmtree(environment_dir)
    os.makedirs(environment_dir)

    an_environment = parameterorg.make_environment_given_user_cell_group_defns(
        environment_name=environment_name,
        environment_dir=environment_dir,
        user_cell_group_defns=user_cell_group_defns,
        justify_parameters=justify_parameters,
        **environment_wide_variable_defns
    )

    return an_environment


def run_template_experiments(
    experiment_directory,
        environment_wide_variable_defns,
    user_cell_group_defns_per_subexperiment,
        num_experiment_repeats=1,
    justify_parameters=True,
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

            an_environment = create_environment(
                environment_name,
                environment_dir,
                justify_parameters,
                environment_wide_variable_defns,
                user_cell_group_defns,
            )

            an_environment.execute_system_dynamics()
