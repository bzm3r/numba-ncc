# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 16:00:16 2016

@author: Brian Merchant
"""

import numpy as np
import general.exec_utils as eu

global_randomization_scheme_dict = {"m": "kgtp_rac_multipliers", "w": "wipeout"}

# =======================================================================

def define_group_boxes_and_corridors(
        plate_width,
        plate_height,
        num_boxes,
        num_cells_in_boxes,
        box_heights,
        box_widths,
        x_space_between_boxes,
        x_placement_option,
        y_placement_option,
        origin_x_offset=10,
        origin_y_offset=10,
        box_x_offsets=None,
        box_y_offsets=None,
):
    if box_x_offsets is None:
        box_x_offsets = []
    if box_y_offsets is None:
        box_y_offsets = []
    test_lists = [num_cells_in_boxes, box_heights, box_widths, x_space_between_boxes]
    test_list_labels = [
        "num_cells_in_boxes",
        "box_heights",
        "box_widths",
        "x_space_between_boxes",
    ]
    allowed_placement_options = ["CENTER", "CENTRE", "ORIGIN", "OVERRIDE"]

    if len(box_x_offsets) == 0:
        box_x_offsets = [0.0 for _ in range(num_boxes)]
    elif len(box_x_offsets) != num_boxes:
        raise Exception("Incorrect number of box_x_offsets given!")
    if len(box_y_offsets) == 0:
        box_y_offsets = [0.0 for _ in range(num_boxes)]
    elif len(box_y_offsets) != num_boxes:
        raise Exception("Incorrect number of box_y_offsets given!")

    for test_list_label, test_list in zip(test_list_labels, test_lists):
        if test_list_label == "x_space_between_boxes":
            required_len = num_boxes - 1
        else:
            required_len = num_boxes

        if len(test_list) != required_len:
            raise Exception(
                "{} length is not the required length (should be {}, got {}).".format(
                    test_list_label, required_len, len(test_list)
                )
            )

    for axis, placement_option in zip(
            ["x", "y"], [x_placement_option, y_placement_option]
    ):
        if placement_option not in allowed_placement_options:
            raise Exception(
                "Given {} placement option not an allowed placement option!\nGiven: {},\nAllowed: {}".format(
                    axis, placement_option, allowed_placement_options
                )
            )

    if x_placement_option != "OVERRIDE":
        if x_placement_option == "ORIGIN":
            first_box_offset = origin_x_offset
        else:
            first_box_offset = 0.5 * plate_width - 0.5 * (
                    np.sum(box_widths) + np.sum(x_space_between_boxes)
            )

        for box_index in range(num_boxes):
            if box_index > 0:
                box_x_offsets[box_index] = (
                        first_box_offset
                        + x_space_between_boxes[box_index - 1]
                        + np.sum(box_widths[:box_index])
                        + np.sum(x_space_between_boxes[: (box_index - 1)])
                )
            else:
                box_x_offsets[box_index] = first_box_offset

    if y_placement_option != "OVERRIDE":
        for box_index in range(num_boxes):
            if y_placement_option == "ORIGIN":
                box_y_offsets[box_index] = origin_y_offset
            else:
                box_y_offsets[box_index] = 0.5 * plate_height - 0.5 * np.max(
                    box_heights
                )

    return (
        np.arange(num_boxes),
        box_x_offsets,
        box_y_offsets,
    )


# ===========================================================================


def stringify_randomization_info(parameter_dict):
    randomization_scheme, t, tv = (
        parameter_dict["randomization_scheme"],
        parameter_dict["randomization_time_mean"],
        parameter_dict["randomization_time_variance_factor"],
    )
    if randomization_scheme == "m":
        mag, np = (
            parameter_dict["randomization_magnitude"],
            parameter_dict["randomization_node_percentage"],
        )
        return "-rand-{}-(t={}-tv={}-mag={}-np={})".format(
            randomization_scheme, t, tv, mag, np
        )
    elif randomization_scheme is None:
        return "-no-rand"
    else:
        raise Exception("Unrecognized randomization scheme.")

# ===========================================================================

def rust_comparison_test(
        date_str,
        experiment_number,
        sub_experiment_number,
        parameter_dict,
        num_cells_responsive_to_chemoattractant=-1,
        uniform_initial_polarization=False,
        no_randomization=False,
        base_output_dir="B:\\Desktop\\numba-ncc\\output",
        total_time_in_hours=3,
        timestep_length=2,
        integration_params=None,
        allowed_drift_before_geometry_recalc=1.0,
        default_coa=0,
        default_cil=0,
        num_experiment_repeats=1,
        box_width=4,
        box_height=4,
        max_placement_distance_factor=1.0,
        num_cells=0,
        biased_rgtpase_distrib_defn_dict=None,
        justify_parameters=True,
):
    if integration_params is None:
        integration_params = {"rtol": 1e-4}
    if biased_rgtpase_distrib_defn_dict is None:
        biased_rgtpase_distrib_defn_dict = {
            "default": ["biased nodes", [0, 1, 2, 3], 0.3]
        }
    cell_diameter = 2 * parameter_dict["init_cell_radius"] / 1e-6

    if num_cells == 0:
        raise Exception("No cells!")

    if no_randomization:
        parameter_dict.update([("randomization_scheme", None)])

    experiment_name = "rt_{}_NC=({}, {}, {})_COA={}_CIL={}".format(
        sub_experiment_number,
        num_cells,
        box_width,
        box_height,
        np.round(default_coa, decimals=3),
        np.round(default_cil, decimals=3),
    ) + stringify_randomization_info(parameter_dict)
    if uniform_initial_polarization:
        experiment_name += "-UIP"
    if not (num_cells_responsive_to_chemoattractant >= num_cells or num_cells_responsive_to_chemoattractant < 0):
        experiment_name += "-NRESP={}".format(num_cells_responsive_to_chemoattractant)

    experiment_dir = eu.get_template_experiment_directory_path(
        base_output_dir, date_str, experiment_number, experiment_name
    )

    total_time = total_time_in_hours * 3600
    num_timesteps = int(total_time / timestep_length)

    num_boxes = 1
    num_cells_in_boxes = [num_cells]
    box_heights = [box_height * cell_diameter]
    box_widths = [box_width * cell_diameter]

    x_space_between_boxes = []

    plate_width = 5 * box_width
    initial_x_placement_options = "ORIGIN"
    box_x_offsets = [0.0]
    plate_height = plate_width

    origin_y_offset = 0.0
    initial_y_placement_options = "ORIGIN"

    boxes, box_x_offsets, box_y_offsets = define_group_boxes_and_corridors(
        plate_width,
        plate_height,
        num_boxes,
        num_cells_in_boxes,
        box_heights,
        box_widths,
        x_space_between_boxes,
        initial_x_placement_options,
        initial_y_placement_options,
        origin_y_offset=origin_y_offset,
        box_x_offsets=box_x_offsets,
    )

    environment_wide_variable_defns = {
        "num_timesteps": num_timesteps,
        "T": timestep_length,
        "integration_params": integration_params,
        "allowed_drift_before_geometry_recalc": allowed_drift_before_geometry_recalc,
        "max_placement_distance_factor": max_placement_distance_factor,
    }

    cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment = [
        [dict([(x, default_coa) for x in boxes])] * num_boxes
    ]
    # intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [{0: {0: default_cil, 1: default_cil}, 1: {0: default_cil, 1: default_cil}}]
    cil_dict = dict([(n, default_cil) for n in range(num_boxes)])
    intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment = [
        dict([(n, cil_dict) for n in range(num_boxes)])
    ]

    biased_rgtpase_distrib_defn_dicts = [[biased_rgtpase_distrib_defn_dict] * num_boxes]
    parameter_dict_per_sub_experiment = [[parameter_dict] * num_boxes]

    user_cell_group_defns_per_subexperiment = []
    user_cell_group_defns = []

    si = 0

    for bi in boxes:
        this_box_x_offset = box_x_offsets[bi]
        this_box_y_offset = box_y_offsets[bi]
        this_box_width = box_widths[bi]
        this_box_height = box_heights[bi]

        cell_group_dict = {
            "cell_group_name": bi,
            "num_cells": num_cells_in_boxes[bi],
            "num_cells_responsive_to_chemoattractant": num_cells_responsive_to_chemoattractant,
            "cell_group_bounding_box": np.array(
                [
                    this_box_x_offset,
                    this_box_x_offset + this_box_width,
                    this_box_y_offset,
                    this_box_height + this_box_y_offset,
                ]
            )
                                       * 1e-6,
            "interaction_factors_intercellular_contact_per_celltype":
                intercellular_contact_factor_magnitudes_defn_dicts_per_sub_experiment[
                    si
                ][
                    bi
                ],
            "interaction_factors_coa_per_celltype": cell_dependent_coa_signal_strengths_defn_dicts_per_sub_experiment[
                si
            ][
                bi
            ],
            "biased_rgtpase_distrib_defns": biased_rgtpase_distrib_defn_dicts[si][bi],
            "parameter_dict": parameter_dict_per_sub_experiment[si][bi],
        }

        user_cell_group_defns.append(cell_group_dict)

    user_cell_group_defns_per_subexperiment.append(user_cell_group_defns)

    cell_dependent_coa_signal_strengths = []
    for cgi, cgd in enumerate(user_cell_group_defns):
        signal_strength = cgd["interaction_factors_coa_per_celltype"][cgi]
        for ci in range(cgd["num_cells"]):
            cell_dependent_coa_signal_strengths.append(signal_strength)

    eu.run_template_experiments(
        experiment_dir,
        environment_wide_variable_defns,
        user_cell_group_defns_per_subexperiment,
        num_experiment_repeats=num_experiment_repeats,
        justify_parameters=justify_parameters,
    )

    print("Done.")

    return (
        experiment_name,
        environment_wide_variable_defns,
    )

