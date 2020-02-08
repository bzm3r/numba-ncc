# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:21:52 2015

@author: brian
"""

import numpy as np
from . import geometry as geometry
from . import hardio as hardio
from . import parameterorg as parameterorg
import general.moore_data_table as moore_data_table
import numba as nb
import copy
import scipy.spatial as space
import scipy.optimize as scipio
import threading
import math
import time


# ==============================================================================
@nb.jit(nopython=True)
def calculate_centroids_per_tstep(node_coords_per_tstep):
    num_tsteps = node_coords_per_tstep.shape[0]

    centroids_per_tstep = np.zeros((num_tsteps, 2), dtype=np.float64)

    for ti in range(num_tsteps):
        cx, cy = geometry.calculate_centroid(node_coords_per_tstep[ti])
        centroids_per_tstep[ti][0] = cx
        centroids_per_tstep[ti][1] = cy

    return centroids_per_tstep


# ==============================================================================
def calculate_cell_centroids_for_all_time(cell_index, storefile_path):
    node_coords_per_tstep = hardio.get_node_coords_for_all_tsteps(
        cell_index, storefile_path
    )
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)

    return centroids_per_tstep


# ==============================================================================
def calculate_cell_centroids_until_tstep(cell_index, max_tstep, storefile_path):
    node_coords_per_tstep = hardio.get_node_coords_until_tstep(
        cell_index, max_tstep, storefile_path
    )
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)

    return centroids_per_tstep


# ==============================================================================
@nb.jit(nopython=True)
def calculate_cell_centroids_for_given_times(cell_index, tsteps, storefile_path):
    node_coords_per_tstep = hardio.get_node_coords_for_given_tsteps(
        cell_index, tsteps, storefile_path
    )
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)

    return centroids_per_tstep


# ==============================================================================


@nb.jit(nopython=True)
def calculate_velocities(position_per_tstep, T):
    num_tpoints = position_per_tstep.shape[0]
    velocities = np.zeros((num_tpoints - 1, 2), dtype=np.float64)

    for ti in range(num_tpoints - 1):
        velocities[ti] = (position_per_tstep[ti + 1] - position_per_tstep[ti]) / T

    return velocities


# ==============================================================================
def calculate_cell_speeds_until_tstep(cell_index, max_tstep, storefile_path, T, L):
    node_coords_per_tstep = hardio.get_node_coords_until_tstep(
        cell_index, max_tstep, storefile_path
    )
    centroid_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep) * L

    num_tsteps = node_coords_per_tstep.shape[0]

    timepoints = np.arange(num_tsteps - 1) * T

    velocities = calculate_velocities(centroid_per_tstep, T)

    speeds = np.linalg.norm(velocities, axis=1)

    return timepoints, speeds


# ==============================================================================


def calculate_polarization_information_until_timestep(
    cell_index, node_coords_per_tstep, storefile_path, T, max_tstep=None,
):
    centroid_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)

    velocity_per_timestep = calculate_velocities(centroid_per_tstep, T)

    _, _, _, polarization_vector_per_timestep = collate_protrusion_data_for_cell(
        cell_index, T, storefile_path, max_tstep=max_tstep
    )

    return centroid_per_tstep, velocity_per_timestep, polarization_vector_per_timestep


# @nb.jit()
def determine_centroid_types_per_timestep(centroid_coords_per_cell_per_timepoint):
    centroid_types_per_timestep = np.zeros(
        (
            centroid_coords_per_cell_per_timepoint.shape[0],
            centroid_coords_per_cell_per_timepoint.shape[1],
        ),
        dtype=np.uint8,
    )

    for ti in range(centroid_coords_per_cell_per_timepoint.shape[0]):
        this_tstep_centroids = centroid_coords_per_cell_per_timepoint[ti]
        convex_hull_vertices = space.ConvexHull(this_tstep_centroids).vertices
        centroid_types_per_timestep[ti, convex_hull_vertices] = 1

    return centroid_types_per_timestep


@nb.jit(nopython=True)
def calculate_polarization_rating_old(
    rac_membrane_active, rho_membrane_active, num_nodes
):
    max_rac = 0.0
    sum_rac = 0.0
    for i in range(num_nodes):
        this_node_rac_active = rac_membrane_active[i]
        sum_rac = sum_rac + this_node_rac_active
        if this_node_rac_active > max_rac:
            max_rac = this_node_rac_active

    sum_rho = 0.0
    for i in range(num_nodes):
        sum_rho = sum_rho + rho_membrane_active[i]

    significant_rac = np.zeros(num_nodes, dtype=np.int64)
    normalized_rac = rac_membrane_active / max_rac
    for i in range(num_nodes):
        if normalized_rac[i] > 0.2:
            significant_rac[i] = 1

    num_rac_fronts = 0
    front_starts = np.zeros(num_nodes, dtype=np.int64)
    for ni in range(num_nodes):
        ni_plus1 = (ni + 1) % num_nodes
        if significant_rac[ni] == 0 and significant_rac[ni_plus1] == 1:
            front_starts[num_rac_fronts] = ni_plus1
            num_rac_fronts += 1

    if num_rac_fronts == 0:
        return 0.0

    front_strengths = np.zeros(num_rac_fronts, dtype=np.float64)
    front_widths = np.zeros(num_rac_fronts, dtype=np.float64)
    for fi in range(num_rac_fronts):
        front_strength = 0.0
        front_width = 0.0
        i = front_starts[fi]

        while significant_rac[i] != 0:
            front_strength += normalized_rac[i]
            front_width += 1.0
            i = (i + 1) % num_nodes

        front_strengths[fi] = front_strength / front_width
        front_widths[fi] = front_width

    max_front_strength = np.max(front_strengths)
    normalized_front_strengths = front_strengths / max_front_strength

    rac_amount_score = 1.0
    if sum_rac > 0.3:
        rac_amount_score = np.exp((sum_rac - 0.3) * np.log(0.1) / 0.1)

    rho_amount_score = 1.0
    if sum_rho > 0.1:
        rho_amount_score = np.exp((sum_rho - 0.1) * np.log(0.1) / 0.1)

    if num_rac_fronts == 1:
        front_width_rating = front_widths[0] / (num_nodes / 3.0)

        if front_width_rating > 1.0:
            front_width_rating = 1.0 - score_function(1.0, 2.0, front_width_rating)

        return (
            front_width_rating * rac_amount_score * rho_amount_score
        )  # (front_width_rating)*rac_amount_score*rho_amount_score

    elif num_rac_fronts > 1:
        worst_distance_between_fronts = (num_nodes - np.sum(front_widths)) / float(
            num_rac_fronts
        )
        distance_between_fronts_score = np.zeros(num_rac_fronts, dtype=np.float64)

        for fi in range(num_rac_fronts):
            this_si = front_starts[fi]
            this_width = front_widths[fi]
            fi_plus1 = (fi + 1) % (num_rac_fronts)
            next_si = front_starts[fi_plus1]
            if next_si < this_si:
                next_si = num_nodes + next_si
            dist_bw_fronts = next_si - (this_si + this_width)

            this_fs = normalized_front_strengths[fi]
            next_fs = normalized_front_strengths[fi_plus1]
            relevant_fs = 1.0
            if this_fs < next_fs:
                relevant_fs = this_fs
            else:
                relevant_fs = next_fs

            score = dist_bw_fronts * relevant_fs / worst_distance_between_fronts
            if score > 1.0:
                distance_between_fronts_score[fi] = 1.0
            else:
                distance_between_fronts_score[fi] = score

        combined_dist_bw_fronts_score = 1.0
        if num_rac_fronts == 2:
            combined_dist_bw_fronts_score = 1.0 - np.min(distance_between_fronts_score)
        else:
            combined_dist_bw_fronts_score = (
                1.0 - np.sum(distance_between_fronts_score) / num_rac_fronts
            )

        total_front_width = np.sum(front_widths) + 0.0
        front_width_rating = total_front_width / (num_nodes / 3.0)

        if front_width_rating > 1.0:
            front_width_rating = 1.0 - score_function(1.0, 2.0, front_width_rating)

        return (
            combined_dist_bw_fronts_score
            * front_width_rating
            * rac_amount_score
            * rho_amount_score
        )  # combined_dist_bw_fronts_score*front_width_rating*rac_amount_score*rho_amount_score

    return 0.0


# ==============================================================================
@nb.jit(nopython=True)
def polarization_score_pwl_fn(x):
    defining_xs = [0.0, 0.2, 0.33, 0.7, 0.9, 1.0]
    defining_ys = [0.0, 0.2, 1.0, 1.0, 0.2, 0.0]

    num_points = len(defining_xs)

    if x < defining_xs[0]:
        return 0.0
    elif x > defining_xs[-1]:
        return 0.0
    else:
        for j in range(1, num_points):
            x2 = defining_xs[j]

            if x <= x2:
                x1 = defining_xs[j - 1]
                y1 = defining_ys[j - 1]
                y2 = defining_ys[j]

                return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


@nb.jit(nopython=True)
def scaled_and_translated_bump_function(offset, width, x):
    if offset - np.sqrt(width) < x < offset + np.sqrt(width):
        return np.exp(-1.0 * (1 / (1.0 - ((x - offset) ** 2.0) / width)))
    else:
        return 0.0


@nb.jit(nopython=True)
def calculate_polarization_rating(rac_membrane_active, rho_membrane_active, num_nodes):
    max_rac = 0.0
    sum_rac = 0.0

    for i in range(num_nodes):
        this_node_rac_active = rac_membrane_active[i]
        sum_rac = sum_rac + this_node_rac_active
        if this_node_rac_active > max_rac:
            max_rac = this_node_rac_active

    sum_rho = 0.0
    for i in range(num_nodes):
        sum_rho = sum_rho + rho_membrane_active[i]

    significant_rac = np.zeros(num_nodes, dtype=np.int64)
    normalized_rac = rac_membrane_active / max_rac
    for i in range(num_nodes):
        if normalized_rac[i] > 0.2:
            significant_rac[i] = 1

    num_rac_fronts = 0
    front_starts = np.zeros(num_nodes, dtype=np.int64)
    for ni in range(num_nodes):
        ni_plus1 = (ni + 1) % num_nodes
        if significant_rac[ni] == 0 and significant_rac[ni_plus1] == 1:
            front_starts[num_rac_fronts] = ni_plus1
            num_rac_fronts += 1

    if num_rac_fronts == 0:
        return 0.0

    front_strengths = np.zeros(num_rac_fronts, dtype=np.float64)
    front_widths = np.zeros(num_rac_fronts, dtype=np.float64)
    for fi in range(num_rac_fronts):
        front_strength = 0.0
        front_width = 0.0
        i = front_starts[fi]

        while significant_rac[i] != 0:
            front_strength += normalized_rac[i]
            front_width += 1.0
            i = (i + 1) % num_nodes

        front_strengths[fi] = front_strength / front_width
        front_widths[fi] = front_width

    max_front_strength = np.max(front_strengths)
    normalized_front_strengths = front_strengths / max_front_strength

    rac_amount_score = 1.0
    if sum_rac > 0.4:
        rac_amount_score = np.exp((sum_rac - 0.4) * np.log(0.1) / 0.1)

    rho_amount_score = 1.0
    if sum_rho > 0.4:
        rho_amount_score = np.exp((sum_rho - 0.4) * np.log(0.1) / 0.1)

    if num_rac_fronts == 1:
        single_front_width = (1.0 * front_widths[0]) / num_nodes

        front_width_rating = polarization_score_pwl_fn(single_front_width)

        return (
            front_width_rating * rac_amount_score * rho_amount_score
        )  # (front_width_rating)*rac_amount_score*rho_amount_score

    elif num_rac_fronts > 1:
        worst_distance_between_fronts = (num_nodes - np.sum(front_widths)) / float(
            num_rac_fronts
        )
        distance_between_fronts_score = np.zeros(num_rac_fronts, dtype=np.float64)

        for fi in range(num_rac_fronts):
            this_si = front_starts[fi]
            this_width = front_widths[fi]
            fi_plus1 = (fi + 1) % (num_rac_fronts)
            next_si = front_starts[fi_plus1]
            if next_si < this_si:
                next_si = num_nodes + next_si
            dist_bw_fronts = next_si - (this_si + this_width)

            this_fs = normalized_front_strengths[fi]
            next_fs = normalized_front_strengths[fi_plus1]
            relevant_fs = 1.0
            if this_fs < next_fs:
                relevant_fs = this_fs
            else:
                relevant_fs = next_fs

            score = dist_bw_fronts * relevant_fs / worst_distance_between_fronts
            if score > 1.0:
                distance_between_fronts_score[fi] = 1.0
            else:
                distance_between_fronts_score[fi] = score

        combined_dist_bw_fronts_score = 1.0
        if num_rac_fronts == 2:
            combined_dist_bw_fronts_score = 1.0 - np.min(distance_between_fronts_score)
        else:
            combined_dist_bw_fronts_score = (
                1.0 - np.sum(distance_between_fronts_score) / num_rac_fronts
            )

        total_front_width = (np.sum(front_widths) + 0.0) / num_nodes
        front_width_rating = polarization_score_pwl_fn(total_front_width)

        return (
            combined_dist_bw_fronts_score
            * front_width_rating
            * rac_amount_score
            * rho_amount_score
        )  # combined_dist_bw_fronts_score*front_width_rating*rac_amount_score*rho_amount_score

    return 0.0


# ==============================================================================


def calculate_rgtpase_polarity_score_from_rgtpase_data(
    rac_membrane_active_per_tstep,
    rho_membrane_active_per_tstep,
    significant_difference=0.1,
    weigh_by_timepoint=False,
):
    num_nodes = rac_membrane_active_per_tstep.shape[1]

    scores_per_tstep = np.array(
        [
            calculate_polarization_rating(
                rac_membrane_active, rho_membrane_active, num_nodes
            )
            for rac_membrane_active, rho_membrane_active in zip(
                rac_membrane_active_per_tstep, rho_membrane_active_per_tstep
            )
        ]
    )

    averaged_score = np.average(scores_per_tstep)

    return averaged_score, scores_per_tstep


@nb.jit(nopython=True)
def score_function(min_cutoff, max_cutoff, x):
    if x > max_cutoff:
        return 1.0
    elif x < min_cutoff:
        return 0.0
    else:
        # 0.0 = m*min + b
        # 1.0 = m*max + b
        # 1.0 = m*max - m*min
        # 1.0/(max - min) = m
        # b = -m*min
        return (x - min_cutoff) / (max_cutoff - min_cutoff)


def calculate_parameter_exploration_score_from_cell(
    a_cell,
    significant_difference=0.1,
    num_data_points_from_end=None,
    weigh_by_timepoint=False,
):
    if num_data_points_from_end == None:
        rac_membrane_active_per_tstep = a_cell.system_history[
            :, :, parameterorg.rac_membrane_active_index
        ]
        rho_membrane_active_per_tstep = a_cell.system_history[
            :, :, parameterorg.rho_membrane_active_index
        ]
    else:
        rac_membrane_active_per_tstep = a_cell.system_history[
            -num_data_points_from_end:, :, parameterorg.rac_membrane_active_index
        ]
        rho_membrane_active_per_tstep = a_cell.system_history[
            -num_data_points_from_end:, :, parameterorg.rho_membrane_active_index
        ]

    polarity_score = score_function(
        0.0,
        0.7,
        calculate_rgtpase_polarity_score_from_rgtpase_data(
            rac_membrane_active_per_tstep, rho_membrane_active_per_tstep
        )[0],
    )

    persistence_score = 0.0
    velocity_score = 0.0

    node_coords_per_tstep = a_cell.system_history[
        :, :, [parameterorg.x_index, parameterorg.y_index]
    ]
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)

    cell_centroids = centroids_per_tstep * a_cell.L / 1e-6
    num_tsteps = cell_centroids.shape[0]

    net_displacement = cell_centroids[num_tsteps - 1] - cell_centroids[0]
    net_displacement_mag = np.linalg.norm(net_displacement)

    distance_per_tstep = np.linalg.norm(
        cell_centroids[1:] - cell_centroids[: num_tsteps - 1], axis=1
    )
    net_distance = np.sum(distance_per_tstep)

    persistence = net_displacement_mag / net_distance

    velocities = distance_per_tstep * (60.0 / a_cell.T)
    average_velocity = np.average(velocities)

    persistence_score = 1.0 - score_function(0.5, 1.0, persistence)
    if velocity_score > 3.5:
        velocity_score = 1.0 - score_function(3.5, 5.0, velocity_score)
    else:
        velocity_score = score_function(0.0, 2.5, average_velocity)

    avg_strain = np.average(
        np.average(
            a_cell.system_history[:, :, parameterorg.local_strains_index], axis=1
        )
    )

    strain_score = 1.0

    if strain_score > 0.1:
        strain_score = 1.0 - score_function(0.1, 0.2, avg_strain)

    return polarity_score, persistence_score, velocity_score, strain_score


def calculate_parameter_exploration_score_from_cell_no_randomization_variant_old(
    a_cell, should_be_polarized_by_in_hours=0.5
):
    T = a_cell.T
    should_be_polarized_by_tstep = int((should_be_polarized_by_in_hours * 3600.0) / T)

    rac_membrane_active_per_tstep = a_cell.system_history[
        (should_be_polarized_by_tstep + 1) :, :, parameterorg.rac_membrane_active_index
    ]
    rho_membrane_active_per_tstep = a_cell.system_history[
        (should_be_polarized_by_tstep + 1) :, :, parameterorg.rho_membrane_active_index
    ]
    polarity_score_global = calculate_rgtpase_polarity_score_from_rgtpase_data(
        rac_membrane_active_per_tstep, rho_membrane_active_per_tstep
    )[0]

    polarity_score_at_SBPBP_tstep = polarity_score_global
    if T != 0:
        rac_membrane_active_around_SBPB_tstep = a_cell.system_history[
            (should_be_polarized_by_tstep - 2) : (should_be_polarized_by_tstep + 2),
            :,
            parameterorg.rac_membrane_active_index,
        ]
        rho_membrane_active_around_SBPB_tstep = a_cell.system_history[
            (should_be_polarized_by_tstep - 2) : (should_be_polarized_by_tstep + 2),
            :,
            parameterorg.rho_membrane_active_index,
        ]
        polarity_score_at_SBPBP_tstep = calculate_rgtpase_polarity_score_from_rgtpase_data(
            rac_membrane_active_around_SBPB_tstep, rho_membrane_active_around_SBPB_tstep
        )[
            0
        ]

    speed_score = 0.0

    node_coords_per_tstep = a_cell.system_history[
        :, :, [parameterorg.x_index, parameterorg.y_index]
    ]
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)

    cell_centroids = centroids_per_tstep * a_cell.L / 1e-6
    num_tsteps = cell_centroids.shape[0]

    velocities = (cell_centroids[1:] - cell_centroids[: num_tsteps - 1]) * (
        60.0 / a_cell.T
    )
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.average(speeds)

    if avg_speed > 3.5:
        speed_score = 1.0 - score_function(3.5, 5.0, avg_speed)
    else:
        speed_score = score_function(0.0, 2.5, avg_speed)

    avg_strain = np.average(
        np.average(
            a_cell.system_history[
                (should_be_polarized_by_tstep + 1) :,
                :,
                parameterorg.local_strains_index,
            ],
            axis=1,
        )
    )

    strain_score = 1.0
    if strain_score > 0.1:
        strain_score = 1.0 - score_function(0.1, 0.2, avg_strain)

    return (
        polarity_score_global,
        polarity_score_at_SBPBP_tstep,
        speed_score,
        strain_score,
    )


def calculate_parameter_exploration_score_from_cell_no_randomization_variant(
    a_cell, should_be_polarized_by_in_hours=0.5
):
    T = a_cell.T
    should_be_polarized_by_tstep = int((should_be_polarized_by_in_hours * 3600.0) / T)

    rac_membrane_active_per_tstep = a_cell.system_history[
        (should_be_polarized_by_tstep + 1) :, :, parameterorg.rac_membrane_active_index
    ]
    rho_membrane_active_per_tstep = a_cell.system_history[
        (should_be_polarized_by_tstep + 1) :, :, parameterorg.rho_membrane_active_index
    ]
    polarity_score_global = calculate_rgtpase_polarity_score_from_rgtpase_data(
        rac_membrane_active_per_tstep, rho_membrane_active_per_tstep
    )[0]

    return polarity_score_global, 0.0, 0.0, 0.0


def calculate_rgtpase_polarity_score(
    cell_index,
    storefile_path,
    significant_difference=0.1,
    max_tstep=None,
    weigh_by_timepoint=False,
):
    rac_membrane_active_per_tstep = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rac_membrane_active", storefile_path
    )
    rho_membrane_active_per_tstep = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rho_membrane_active", storefile_path
    )

    return calculate_rgtpase_polarity_score_from_rgtpase_data(
        rac_membrane_active_per_tstep,
        rho_membrane_active_per_tstep,
        significant_difference=significant_difference,
        weigh_by_timepoint=weigh_by_timepoint,
    )


# ==============================================================================


def calculate_average_rgtpase_activity(cell_index, storefile_path):
    rac_data = hardio.get_data(cell_index, None, "rac_membrane_active", storefile_path)
    sum_rac_over_nodes = np.sum(rac_data, axis=1)
    avg_sum_rac_over_nodes = np.average(sum_rac_over_nodes)

    rho_data = hardio.get_data(cell_index, None, "rho_membrane_active", storefile_path)
    sum_rho_over_nodes = np.sum(rho_data, axis=1)
    avg_sum_rho_over_nodes = np.average(sum_rho_over_nodes)

    return avg_sum_rac_over_nodes, avg_sum_rho_over_nodes


# ==============================================================================


def calculate_total_displacement(num_nodes, cell_index, storefile_path):
    init_node_coords = np.transpose(
        hardio.get_node_coords(cell_index, 0, storefile_path)
    )
    final_node_coords = np.transpose(
        hardio.get_node_coords(cell_index, -1, storefile_path)
    )

    init_centroid = geometry.calculate_centroid(num_nodes, init_node_coords)
    final_centroid = geometry.calculate_centroid(num_nodes, final_node_coords)

    return np.linalg.norm(init_centroid - final_centroid)


# ==============================================================================


def calculate_sum_displacement_per_interval(
    num_nodes, num_timesteps, cell_index, num_timesteps_to_pick, storefile_path
):
    timepoints_of_interest = np.linspace(
        0, num_timesteps, num=num_timesteps_to_pick, dtype=np.int64
    )

    ncs_of_interest = [
        np.transpose(hardio.get_node_coords(cell_index, x, storefile_path))
        for x in timepoints_of_interest
    ]
    centroids_of_interest = np.array(
        [geometry.calculate_centroid(num_nodes, x) for x in ncs_of_interest]
    )

    return np.sum(
        [
            np.linalg.norm(x)
            for x in centroids_of_interest[1:] - centroids_of_interest[:-1]
        ]
    )


# ==============================================================================


def calculate_migry_boundary_violation_score(
    num_nodes, num_timesteps, cell_index, storefile_path
):
    migr_bdry_contact_data = hardio.get_data(
        cell_index, None, "migr_bdry_contact", storefile_path
    )

    x = migr_bdry_contact_data - 1.0
    y = x > 1e-10
    y = np.array(y, dtype=np.int64)

    return np.sum(y) / (num_timesteps * num_nodes)


# ==============================================================================


def calculate_average_total_strain(num_nodes, cell_index, storefile_path):
    node_coords_per_timestep = hardio.get_node_coords_for_all_tsteps(
        cell_index, storefile_path
    )

    init_perimeter = geometry.calculate_perimeter(
        num_nodes, node_coords_per_timestep[0]
    )
    total_strains = (
        np.array(
            [
                geometry.calculate_perimeter(num_nodes, x)
                for x in node_coords_per_timestep
            ]
        )
        / init_perimeter
    )
    avg_total_strain = np.average(total_strains)

    return avg_total_strain - 1.0


# ==============================================================================


def calculate_acceleration(num_timepoints, num_nodes, L, T, cell_index, storefile_path):
    init_point = 0
    mid_point = np.int(num_timepoints * 0.5)
    final_point = -1

    init_node_coords = np.transpose(
        hardio.get_node_coords(cell_index, init_point, storefile_path)
    )
    mid_node_coords = np.transpose(
        hardio.get_node_coords(cell_index, mid_point, storefile_path)
    )
    final_node_coords = np.transpose(
        hardio.get_node_coords(cell_index, final_point, storefile_path)
    )

    init_centroid = geometry.calculate_centroid(num_nodes, init_node_coords)
    mid_centroid = geometry.calculate_centroid(num_nodes, mid_node_coords)
    final_centroid = geometry.calculate_centroid(num_nodes, final_node_coords)

    acceleration = (
        np.linalg.norm((final_centroid - mid_centroid)) * L * 1e6
        - np.linalg.norm((mid_centroid - init_centroid)) * L * 1e6
    ) / (0.5 * num_timepoints * T / 60.0) ** 2

    return np.abs(acceleration)


# ==============================================================================


def score_distance_travelled(cell_index, storefile_path):
    xs = hardio.get_data(cell_index, None, "x", storefile_path)
    ys = hardio.get_data(cell_index, None, "y", storefile_path)

    x_disps = xs[1:] - xs[:-1]
    y_disps = ys[1:] - ys[:-1]

    dists = np.sqrt(x_disps * x_disps + y_disps * y_disps)

    total_dist_magnitude = np.sum(dists)

    sum_x_disps = np.sum(x_disps)
    sum_y_disps = np.sum(y_disps)

    total_disp_magnitude = np.sqrt(
        sum_x_disps * sum_x_disps + sum_y_disps * sum_y_disps
    )

    return total_dist_magnitude, total_disp_magnitude


# ==============================================================================


def get_event_tsteps(event_type, cell_index, storefile_path):
    relevant_data_per_tstep = None

    if event_type == "ic-contact":
        ic_mags = hardio.get_data(
            cell_index, None, "intercellular_contact_factor_magnitudes", storefile_path
        )
        relevant_data_per_tstep = np.any(ic_mags > 1, axis=1)
    elif event_type == "randomization":
        polarity_loss_occurred = hardio.get_data(
            cell_index, None, "polarity_loss_occurred", storefile_path
        )
        relevant_data_per_tstep = np.any(polarity_loss_occurred, axis=1)

    if relevant_data_per_tstep == None:
        raise Exception("Unknown event type given!")

    event_tsteps = [
        n
        for n in range(relevant_data_per_tstep.shape[0])
        if relevant_data_per_tstep[n] == 1
    ]

    return event_tsteps


# ==============================================================================


def determine_contact_start_end(T, contact_tsteps, min_tstep, max_tstep):
    contact_start_end_tuples = []

    current_start = None
    last_contact_tstep = None

    absolute_last_contact_tstep = contact_tsteps[-1]

    for contact_tstep in contact_tsteps:
        if (
            current_start == None
            and contact_tstep != min_tstep
            and contact_tstep != max_tstep
        ):
            current_start = contact_tstep

        if contact_tstep == absolute_last_contact_tstep:
            contact_start_end_tuples.append((current_start, last_contact_tstep))
            continue

        if last_contact_tstep != None:
            if contact_tstep - 1 != last_contact_tstep:
                if last_contact_tstep != max_tstep and last_contact_tstep != min_tstep:
                    contact_start_end_tuples.append((current_start, last_contact_tstep))
                current_start = contact_tstep

        last_contact_tstep = contact_tstep

    contact_start_end_tuples = [
        x
        for x in contact_start_end_tuples
        if (T * (x[1] - x[0]) > 30.0) and (x[1] != max_tstep)
    ]

    return contact_start_end_tuples


# ==============================================================================


def calculate_kinematics(delta_t, centroid_pos_plus1s, centroid_pos_minus1s):
    delta_pos = centroid_pos_plus1s - centroid_pos_minus1s
    velocities = delta_pos / (2 * delta_t)
    accelerations = delta_pos / (delta_t ** 2)

    return velocities, accelerations


# ==============================================================================


def are_all_elements_of_required_type(required_type, given_list):
    for element in given_list:
        if type(element) != required_type:
            return False

    return True


# ==============================================================================


def is_ascending(num_elements, test_list):
    for n in range(num_elements - 1):
        if test_list[n] > test_list[n + 1]:
            return False

    return True


# ==============================================================================


def determine_relevant_table_points(x, labels, type="row"):
    num_labels = len(labels)

    if not (is_ascending(num_labels, labels)):
        raise Exception("Labels are not ascending!")

    lower_bound_index = None
    for n in range(num_labels - 1):
        if x > labels[n]:
            lower_bound_index = n
            break

    upper_bound_index = None
    if lower_bound_index != None:
        for n in range(lower_bound_index + 1, num_labels):
            if x < labels[n]:
                upper_bound_index = n
                break
    else:
        for n in range(num_labels):
            if x < labels[n]:
                upper_bound_index = n
                break

    if lower_bound_index == None and upper_bound_index != None:
        return upper_bound_index, lower_bound_index
    else:
        return lower_bound_index, upper_bound_index


# ==============================================================================


def determine_index_lb_ub_for_value_given_list(given_value, num_elements, given_list):
    ilb, iub = -1, -1

    for n, value in enumerate(given_list):
        if n == 0:
            if given_value < value:
                ilb, iub = n, n
            else:
                ilb, iub = n, n + 1
            continue
        elif n == num_elements - 1:
            if given_value > value:
                ilb, iub = n, n
        else:
            if given_value == value:
                ilb, iub = n, n
            elif given_list[n - 1] < given_value < value:
                ilb, iub = n - 1, n
                break

    if ilb == -1 or iub == -1 or ilb >= num_elements or iub >= num_elements:
        raise Exception("Did not find one of ilb, iub!")
    else:
        return ilb, iub


# ==============================================================================


def determine_probability_given_N_Rstar(N, Rstar):
    data_dict = moore_data_table.moore_data_table_dict

    rlabels = moore_data_table.moore_row_labels
    num_rlabels = moore_data_table.num_row_labels

    clabels = moore_data_table.moore_col_labels
    num_clabels = moore_data_table.num_col_labels

    ylb, yub = determine_index_lb_ub_for_value_given_list(N, num_rlabels, rlabels)
    N_lb = rlabels[ylb]
    if ylb != ylb:
        N_ub = rlabels[yub]
    else:
        N_ub = N_lb

    rstars_lb = [data_dict[(N_lb, clabel)] for clabel in clabels]
    if yub != ylb:
        rstars_ub = [data_dict[(N_ub, clabel)] for clabel in clabels]
    else:
        rstars_ub = rstars_lb

    x_lb_lb, x_lb_ub = determine_index_lb_ub_for_value_given_list(
        Rstar, num_clabels, rstars_lb
    )
    if yub != ylb:
        x_ub_lb, x_ub_ub = determine_index_lb_ub_for_value_given_list(
            Rstar, num_clabels, rstars_ub
        )
    else:
        x_ub_lb, x_ub_ub = x_lb_lb, x_lb_ub

    if x_lb_lb != x_lb_ub:
        prob_lb = clabels[x_lb_lb] + (
            (clabels[x_lb_ub] - clabels[x_lb_lb])
            / (rstars_lb[x_lb_ub] - rstars_lb[x_lb_lb])
        ) * (Rstar - rstars_lb[x_lb_lb])
    else:
        prob_lb = clabels[x_lb_lb]

    if x_ub_lb != x_ub_ub:
        prob_ub = clabels[x_ub_lb] + (
            (clabels[x_ub_ub] - clabels[x_ub_lb])
            / (rstars_ub[x_ub_ub] - rstars_ub[x_ub_lb])
        ) * (Rstar - rstars_ub[x_ub_lb])
    else:
        prob_ub = clabels[x_ub_lb]

    if yub != ylb:
        prob = prob_lb + ((prob_ub - prob_lb) / (rlabels[yub] - rlabels[ylb])) * (
            N - rlabels[ylb]
        )
    else:
        prob = prob_lb

    return prob


# ==============================================================================


def calculate_polar_velocities(velocities):
    polar_velocities = np.empty_like(velocities)

    num_velocities = velocities.shape[0]
    velocity_mags = geometry.calculate_2D_vector_mags(num_velocities, velocities)
    velocity_thetas = geometry.calculate_2D_vector_directions(
        num_velocities, velocities
    )

    polar_velocities[:, 0] = velocity_mags
    polar_velocities[:, 1] = velocity_thetas

    return polar_velocities


# ==============================================================================


def calculate_null_hypothesis_probability(velocities):
    N = velocities.shape[0]

    polar_velocities = calculate_polar_velocities(velocities)

    sorted_polar_velocities = np.array(sorted(polar_velocities, key=lambda x: x[0]))

    transformed_polar_velocities = np.array(
        [[n + 1, pv[1]] for n, pv in enumerate(sorted_polar_velocities)]
    )

    transformed_rs = transformed_polar_velocities[:, 0]
    thetas = transformed_polar_velocities[:, 1]

    X = np.dot(transformed_rs, np.cos(thetas))
    Y = np.dot(transformed_rs, np.sin(thetas))

    R = np.sqrt(X ** 2 + Y ** 2)
    Rstar = R / (N ** (3.0 / 2.0))

    return determine_probability_given_N_Rstar(N, Rstar)


# =================================================================================


def get_ic_contact_data(cell_index, storefile_path, max_tstep=None):
    if max_tstep == None:
        ic_contact_data = hardio.get_data(
            cell_index, None, "intercellular_contact_factor_magnitudes", storefile_path
        )
    else:
        ic_contact_data = hardio.get_data_until_timestep(
            cell_index,
            max_tstep,
            "intercellular_contact_factor_magnitudes",
            storefile_path,
        )

    return np.array(np.any(ic_contact_data > 1, axis=1), dtype=np.int64)


# =================================================================================


def determine_contact_start_ends(ic_contact_data):
    num_ic_data = ic_contact_data.shape[0]
    contact_start_end_arrays = np.zeros((num_ic_data, 2), dtype=np.int64)
    num_contact_start_end_arrays = 0

    in_contact = False
    contact_start = -1

    for n in range(num_ic_data):
        if ic_contact_data[n] == 1:
            if in_contact == False:
                contact_start = n
                in_contact = True
            else:
                continue
        else:
            if in_contact == True:
                contact_start_end_arrays[num_contact_start_end_arrays][
                    0
                ] = contact_start
                contact_start_end_arrays[num_contact_start_end_arrays][1] = n
                num_contact_start_end_arrays += 1

                in_contact = False
            else:
                continue

    return contact_start_end_arrays[:num_contact_start_end_arrays]


# =================================================================================


def smoothen_contact_start_end_tuples(
    contact_start_end_arrays, min_tsteps_between_arrays=1
):
    smoothened_contact_start_end_arrays = np.zeros_like(contact_start_end_arrays)
    num_start_end_arrays = contact_start_end_arrays.shape[0]

    num_smoothened_contact_start_end_arrays = 0
    for n in range(num_start_end_arrays - 1):
        this_start, this_end = contact_start_end_arrays[n]
        next_start, next_end = contact_start_end_arrays[n + 1]

        if (next_start - this_end) < min_tsteps_between_arrays:
            smoothened_contact_start_end_arrays[
                num_smoothened_contact_start_end_arrays
            ][0] = this_start
            smoothened_contact_start_end_arrays[
                num_smoothened_contact_start_end_arrays
            ][1] = next_end
            num_smoothened_contact_start_end_arrays += 1
        else:
            if n == 0:
                smoothened_contact_start_end_arrays[
                    num_smoothened_contact_start_end_arrays
                ][0] = this_start
                smoothened_contact_start_end_arrays[
                    num_smoothened_contact_start_end_arrays
                ][1] = this_end
                num_smoothened_contact_start_end_arrays += 1

            smoothened_contact_start_end_arrays[
                num_smoothened_contact_start_end_arrays
            ][0] = next_start
            smoothened_contact_start_end_arrays[
                num_smoothened_contact_start_end_arrays
            ][1] = next_end
            num_smoothened_contact_start_end_arrays += 1

    return smoothened_contact_start_end_arrays[:num_smoothened_contact_start_end_arrays]


# =================================================================================


def get_assessable_contact_start_end_tuples(
    smoothened_contact_start_end_arrays,
    data_max_tstep,
    min_tsteps_needed_to_calculate_kinematics=2,
):
    num_smoothened_arrays = smoothened_contact_start_end_arrays.shape[0]

    if num_smoothened_arrays == 0:
        return np.zeros((0, 2), dtype=np.int64)

    assessable_contact_start_end_arrays = np.zeros_like(
        smoothened_contact_start_end_arrays
    )

    num_assessable_arrays = 0
    for n in range(num_smoothened_arrays):
        this_start, this_end = smoothened_contact_start_end_arrays[n]
        if n == 0:
            if this_start + 1 > min_tsteps_needed_to_calculate_kinematics:
                if n == num_smoothened_arrays - 1:
                    if (
                        data_max_tstep - this_end
                    ) + 1 > min_tsteps_needed_to_calculate_kinematics:
                        assessable_contact_start_end_arrays[num_assessable_arrays][
                            0
                        ] = this_start
                        assessable_contact_start_end_arrays[num_assessable_arrays][
                            1
                        ] = this_end
                        num_assessable_arrays += 1
            continue
        else:
            last_start, last_end = smoothened_contact_start_end_arrays[n - 1]
            if n == num_smoothened_arrays - 1:
                if (
                    data_max_tstep - this_end
                ) + 1 > min_tsteps_needed_to_calculate_kinematics:
                    if (
                        this_start - last_end
                    ) + 1 > min_tsteps_needed_to_calculate_kinematics:
                        assessable_contact_start_end_arrays[num_assessable_arrays][
                            0
                        ] = this_start
                        assessable_contact_start_end_arrays[num_assessable_arrays][
                            1
                        ] = this_end
                        num_assessable_arrays += 1
            else:
                next_start, next_end = smoothened_contact_start_end_arrays[n + 1]
                if (
                    next_start - this_end
                ) + 1 > min_tsteps_needed_to_calculate_kinematics:
                    if (
                        this_start - last_end
                    ) + 1 > min_tsteps_needed_to_calculate_kinematics:
                        assessable_contact_start_end_arrays[num_assessable_arrays][
                            0
                        ] = this_start
                        assessable_contact_start_end_arrays[num_assessable_arrays][
                            1
                        ] = this_end
                        num_assessable_arrays += 1

    return assessable_contact_start_end_arrays[:num_assessable_arrays]


# =================================================================================


def calculate_3_point_kinematics(
    last_centroid, next_centroid, delta_tsteps, tstep_length
):
    delta_centroid = next_centroid - last_centroid

    delta_t = delta_tsteps * tstep_length / 60.0

    acceleration = delta_centroid / (delta_t * delta_t)
    velocity = delta_centroid / (2 * delta_t)

    return acceleration, velocity


# =================================================================================


def calculate_contact_pre_post_kinematics(
    assessable_contact_start_end_arrays,
    cell_centroids_per_tstep,
    delta_tsteps,
    tstep_length,
):
    num_start_end_tuples = assessable_contact_start_end_arrays.shape[0]

    pre_velocities = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    post_velocities = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    pre_accelerations = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    post_accelerations = np.zeros((num_start_end_tuples, 2), dtype=np.float64)

    for n in range(num_start_end_tuples):
        start_tstep, end_tstep = assessable_contact_start_end_arrays[n]

        delta_tsteps_doubled = int(delta_tsteps + delta_tsteps)
        pre_minus1_centroid = cell_centroids_per_tstep[
            start_tstep - delta_tsteps_doubled
        ]
        pre_plus1_centroid = cell_centroids_per_tstep[start_tstep]

        pre_acceleration, pre_velocity = calculate_3_point_kinematics(
            pre_minus1_centroid, pre_plus1_centroid, delta_tsteps, tstep_length
        )
        pre_accelerations[n] = pre_acceleration
        pre_velocities[n] = pre_velocity

        post_plus1_centroid = cell_centroids_per_tstep[end_tstep + delta_tsteps_doubled]
        post_minus1_centroid = cell_centroids_per_tstep[end_tstep]

        post_acceleration, post_velocity = calculate_3_point_kinematics(
            post_minus1_centroid, post_plus1_centroid, delta_tsteps, tstep_length
        )

        post_accelerations[n] = post_acceleration
        post_velocities[n] = post_velocity

    return pre_velocities, post_velocities, pre_accelerations, post_accelerations


# =================================================================================


def rotate_contact_kinematics_data_st_pre_lies_along_given_and_post_maintains_angle_to_pre(
    pre_data, post_data, given_vector
):
    num_elements = pre_data.shape[0]
    aligned_pre_data = np.zeros_like(pre_data)
    aligned_post_data = np.zeros_like(post_data)

    for n in range(num_elements):
        pre_datum = pre_data[n]
        post_datum = post_data[n]

        rot_mat = geometry.determine_rotation_matrix_to_rotate_vector1_to_lie_along_vector2(
            pre_datum, given_vector
        )

        aligned_pre_data[n] = np.dot(rot_mat, pre_datum)
        aligned_post_data[n] = np.dot(rot_mat, post_datum)

    return aligned_pre_data, aligned_post_data


# =============================================================================


@nb.jit(nopython=True)
def get_min_max_x_centroid_per_timestep(all_cell_centroid_xs):
    min_x_centroid_per_timestep = np.zeros(
        all_cell_centroid_xs.shape[1], dtype=np.float64
    )
    max_x_centroid_per_timestep = np.zeros(
        all_cell_centroid_xs.shape[1], dtype=np.float64
    )

    num_cells = all_cell_centroid_xs.shape[0]
    num_timesteps = all_cell_centroid_xs.shape[1]

    for ti in range(num_timesteps):
        min_x_centroid_index = 0
        max_x_centroid_index = 0
        min_x_value = 0.0
        max_x_value = 0.0

        for ci in range(num_cells):
            this_x_value = all_cell_centroid_xs[ci][ti]
            if ci == 0:
                min_x_value = this_x_value
                max_x_value = this_x_value
            else:
                if this_x_value < min_x_value:
                    min_x_centroid_index = ci
                    min_x_value = this_x_value
                if this_x_value > max_x_value:
                    max_x_centroid_index = ci
                    max_x_value = this_x_value

        min_x_centroid_per_timestep[ti] = all_cell_centroid_xs[min_x_centroid_index][ti]
        max_x_centroid_per_timestep[ti] = all_cell_centroid_xs[max_x_centroid_index][ti]

    return min_x_centroid_per_timestep, max_x_centroid_per_timestep


def analyze_single_cell_motion(
    relevant_environment, storefile_path, no_randomization, time_unit="min."
):
    if time_unit == "min.":
        T = relevant_environment.T / 60.0
    elif time_unit == "sec":
        T = relevant_environment.T
    else:
        raise Exception("Unknown time unit given: ", time_unit)

    cell_centroids = (
        calculate_cell_centroids_for_all_time(0, storefile_path)
        * relevant_environment.cells_in_environment[0].L
        / 1e-6
    )
    num_tsteps = cell_centroids.shape[0]

    net_displacement = cell_centroids[num_tsteps - 1] - cell_centroids[0]
    net_displacement_mag = np.linalg.norm(net_displacement)

    cell_centroid_displacements = cell_centroids[1:] - cell_centroids[:-1]
    distance_per_tstep = np.linalg.norm(cell_centroid_displacements, axis=1)
    net_distance = np.sum(distance_per_tstep)

    cell_speeds = distance_per_tstep / T

    if net_distance > 0.0:
        if not no_randomization:
            (
                positive_ns,
                positive_das,
            ) = calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
                cell_centroid_displacements
            )
            persistence_time, positive_ts = estimate_persistence_time(
                T, positive_ns, positive_das
            )
        else:
            persistence_time = np.nan

        persistence_ratio = net_displacement_mag / net_distance
    else:
        persistence_ratio = np.nan
        persistence_time = np.nan

    return (
        time_unit,
        (cell_centroids, (persistence_ratio, persistence_time), cell_speeds),
    )


def analyze_cell_motion(
    relevant_environment,
    storefile_path,
    subexperiment_index,
    rpt_number,
    time_unit="min.",
):
    num_cells = relevant_environment.num_cells

    if time_unit == "min.":
        T = relevant_environment.T / 60.0
    elif time_unit == "sec":
        T = relevant_environment.T
    else:
        raise Exception("Unknown time unit given: ", time_unit)

    centroids_persistences_speeds_protrusionlifetimes = []
    for n in range(num_cells):
        print("    Analyzing cell {}...".format(n))
        if n == 48:
            pass

        cell_centroids = (
            calculate_cell_centroids_for_all_time(n, storefile_path)
            * relevant_environment.cells_in_environment[n].L
            / 1e-6
        )
        num_tsteps = cell_centroids.shape[0]

        net_displacement = cell_centroids[num_tsteps - 1] - cell_centroids[0]
        net_displacement_mag = np.linalg.norm(net_displacement)

        distance_per_tstep = np.linalg.norm(
            cell_centroids[1:] - cell_centroids[: num_tsteps - 1], axis=1
        )
        speeds = distance_per_tstep / T
        net_distance = np.sum(distance_per_tstep)

        if net_distance > 0.0:
            persistence_ratio = net_displacement_mag / net_distance
            this_cell_centroid_displacements = cell_centroids[1:] - cell_centroids[:-1]
            (
                this_cell_positive_ns,
                this_cell_positive_das,
            ) = calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
                this_cell_centroid_displacements
            )
            persistence_time, this_cell_positive_ts = estimate_persistence_time(
                T, this_cell_positive_ns, this_cell_positive_das
            )
        else:
            persistence_time = np.nan
            persistence_ratio = np.nan

        protrusion_lifetime_and_average_directions = collate_protrusion_data_for_cell(
            n, T, storefile_path
        )

        centroids_persistences_speeds_protrusionlifetimes.append(
            (
                cell_centroids,
                (persistence_ratio, persistence_time),
                speeds,
                protrusion_lifetime_and_average_directions,
            )
        )

    all_cell_centroids = np.array(
        [x[0] for x in centroids_persistences_speeds_protrusionlifetimes]
    )
    all_cell_centroid_xs = np.array(
        [x[0][:, 0] for x in centroids_persistences_speeds_protrusionlifetimes]
    )

    if num_cells > 1:
        group_centroid_per_timestep = np.array(
            [
                geometry.calculate_cluster_centroid(all_cell_centroids[:, t, :])
                for t in range(all_cell_centroids.shape[1])
            ]
        )
        (
            min_x_centroid_per_timestep,
            max_x_centroid_per_timestep,
        ) = get_min_max_x_centroid_per_timestep(all_cell_centroid_xs)
        group_centroid_x_per_timestep = group_centroid_per_timestep[:, 0]

        init_group_centroid_per_timestep = group_centroid_per_timestep[0]
        relative_group_centroid_per_timestep = (
            group_centroid_per_timestep - init_group_centroid_per_timestep
        )
        group_centroid_displacements_per_timestep = (
            relative_group_centroid_per_timestep[1:]
            - relative_group_centroid_per_timestep[:-1]
        )

        group_net_displacement = (
            relative_group_centroid_per_timestep[-1]
            - relative_group_centroid_per_timestep[0]
        )
        group_net_displacement_mag = np.linalg.norm(group_net_displacement)
        group_net_distance = np.sum(
            np.linalg.norm(
                relative_group_centroid_per_timestep[1:]
                - relative_group_centroid_per_timestep[:-1],
                axis=1,
            )
        )
        group_persistence_ratio = group_net_displacement_mag / group_net_distance

        (
            group_positive_ns,
            group_positive_das,
        ) = calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
            group_centroid_displacements_per_timestep
        )
        group_persistence_time, group_positive_ts = estimate_persistence_time(
            T, group_positive_ns, group_positive_das
        )

        group_velocities = calculate_velocities(group_centroid_per_timestep, T)
        group_speed_per_timestep = np.linalg.norm(group_velocities, axis=1)
    else:
        group_centroid_x_per_timestep = all_cell_centroid_xs[0]
        min_x_centroid_per_timestep, max_x_centroid_per_timestep = (
            np.nan * np.empty_like(group_centroid_x_per_timestep),
            np.nan * np.empty_like(group_centroid_x_per_timestep),
        )

        group_centroid_per_timestep = all_cell_centroids[0]
        group_velocities = calculate_velocities(group_centroid_per_timestep, T)
        group_speed_per_timestep = np.linalg.norm(group_velocities, axis=1)
        group_persistence_ratio = np.nan
        group_persistence_time = np.nan

    return (
        time_unit,
        min_x_centroid_per_timestep,
        max_x_centroid_per_timestep,
        group_centroid_x_per_timestep,
        group_centroid_per_timestep,
        group_speed_per_timestep,
        group_persistence_ratio,
        group_persistence_time,
        centroids_persistences_speeds_protrusionlifetimes,
    )


# ===========================================================================


def determine_run_and_tumble_periods(
    avg_strain_per_tstep,
    polarization_score_per_tstep,
    tumble_period_strain_threshold,
    tumble_period_polarization_threshold,
):
    num_tsteps = polarization_score_per_tstep.shape[0]

    tumble_period_found = False
    associated_run_period_found = False

    tumble_info = -1 * np.ones((int(num_tsteps / 2), 3), dtype=np.int64)
    run_and_tumble_pair_index = 0

    for ti in range(num_tsteps):
        this_tstep_is_tumble = (
            polarization_score_per_tstep[ti] <= tumble_period_polarization_threshold
            and avg_strain_per_tstep[ti] <= tumble_period_strain_threshold
        )

        if tumble_period_found == False and this_tstep_is_tumble:
            tumble_period_found = True
            tumble_info[run_and_tumble_pair_index, 0] = ti
        else:
            if associated_run_period_found == False and (not this_tstep_is_tumble):
                associated_run_period_found = True
                tumble_info[run_and_tumble_pair_index, 1] = ti
            elif associated_run_period_found == True and this_tstep_is_tumble:
                tumble_info[run_and_tumble_pair_index, 2] = ti
                run_and_tumble_pair_index += 1
                tumble_info[run_and_tumble_pair_index, 0] = ti
                tumble_period_found = True
                associated_run_period_found = False

    num_run_and_tumble_pairs = run_and_tumble_pair_index + 1
    for i in range(3):
        if tumble_info[run_and_tumble_pair_index, i] == -1:
            num_run_and_tumble_pairs -= 1
            break

    return_tumble_info = -1 * np.ones((num_run_and_tumble_pairs, 3), dtype=np.int64)
    for pi in range(num_run_and_tumble_pairs):
        tumble_info_tuple = tumble_info[pi]
        for i in range(3):
            return_tumble_info[pi, i] = tumble_info_tuple[i]

    return return_tumble_info


# ===========================================================================


def calculate_run_and_tumble_statistics(
    num_nodes,
    T,
    L,
    cell_index,
    storefile_path,
    cell_centroids=None,
    max_tstep=None,
    significant_difference=2.5e-2,
    tumble_period_strain_threshold=0.3,
    tumble_period_polarization_threshold=0.6,
):
    rac_membrane_active_per_tstep = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rac_membrane_active", storefile_path
    )
    rho_membrane_active_per_tstep = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rho_membrane_active", storefile_path
    )
    avg_strain_per_tstep = np.average(
        hardio.get_data_until_timestep(
            cell_index, max_tstep, "local_strains", storefile_path
        ),
        axis=1,
    )
    polarization_score_per_tstep = np.array(
        [
            calculate_polarization_rating(
                rac_membrane_active,
                rho_membrane_active,
                num_nodes,
                significant_difference=significant_difference,
            )
            for rac_membrane_active, rho_membrane_active in zip(
                rac_membrane_active_per_tstep, rho_membrane_active_per_tstep
            )
        ]
    )

    tumble_periods_info = determine_run_and_tumble_periods(
        avg_strain_per_tstep,
        polarization_score_per_tstep,
        tumble_period_strain_threshold,
        tumble_period_polarization_threshold,
    )

    tumble_periods = [(tpi[1] - tpi[0]) * T for tpi in tumble_periods_info]
    run_periods = [(tpi[2] - tpi[1]) * T for tpi in tumble_periods_info]

    if cell_centroids == None:
        cell_centroids = (
            calculate_cell_centroids_for_all_time(cell_index, storefile_path) * L
        )

    tumble_centroids = [cell_centroids[tpi[0] : tpi[1]] for tpi in tumble_periods_info]
    net_tumble_displacement_mags = [
        np.linalg.norm(tccs[-1] - tccs[0]) for tccs in tumble_centroids
    ]
    mean_tumble_period_speeds = [
        np.average(np.linalg.norm((tccs[1:] - tccs[:-1]) / T, axis=1))
        for tccs in tumble_centroids
    ]

    run_centroids = [cell_centroids[tpi[1] : tpi[2]] for tpi in tumble_periods_info]
    net_run_displacement_mags = [
        np.linalg.norm(rccs[-1] - rccs[0]) for rccs in run_centroids
    ]
    mean_run_period_speeds = [
        np.average(np.linalg.norm((rccs[1:] - rccs[:-1]) / T, axis=1))
        for rccs in run_centroids
    ]

    return (
        tumble_periods,
        run_periods,
        net_tumble_displacement_mags,
        mean_tumble_period_speeds,
        net_run_displacement_mags,
        mean_run_period_speeds,
    )


# =============================================================================


@nb.jit(nopython=True)
def normalize_rgtpase_data_per_tstep(rgtpase_data):
    num_tpoints = rgtpase_data.shape[0]

    normalized_rgtpase_data = np.zeros_like(rgtpase_data, dtype=np.float64)
    for ti in range(num_tpoints):
        this_tstep_rgtpase_data = rgtpase_data[ti]
        normalized_rgtpase_data[ti] = this_tstep_rgtpase_data / np.max(
            this_tstep_rgtpase_data
        )

    return normalized_rgtpase_data


@nb.jit(nopython=True)
def determine_protrusion_existence_and_direction(
    normalized_rac_membrane_active_per_tstep,
    rac_membrane_active_per_tstep,
    rho_membrane_active_per_tstep,
    uivs_per_node_per_timestep,
):
    num_tpoints = normalized_rac_membrane_active_per_tstep.shape[0]
    num_nodes = normalized_rac_membrane_active_per_tstep.shape[1]
    protrusion_existence_per_tstep = np.zeros((num_tpoints, num_nodes), dtype=np.int64)
    protrusion_direction_per_tstep = np.zeros(
        (num_tpoints, num_nodes, 2), dtype=np.float64
    )

    for ti in range(num_tpoints):
        relevant_normalized_rac_actives = normalized_rac_membrane_active_per_tstep[ti]
        relevant_rac_actives = rac_membrane_active_per_tstep[ti]
        relevant_rho_actives = rho_membrane_active_per_tstep[ti]

        for ni in range(num_nodes):
            if (
                relevant_rac_actives[ni] > relevant_rho_actives[ni]
            ) and relevant_normalized_rac_actives[ni] > 0.25:
                protrusion_existence_per_tstep[ti][ni] = 1
                protrusion_direction_per_tstep[ti][ni] = (
                    -1.0 * uivs_per_node_per_timestep[ti][ni]
                )

    return protrusion_existence_per_tstep, protrusion_direction_per_tstep


@nb.jit(nopython=True)
def determine_protrusion_node_index_and_tpoint_start_ends(
    protrusion_existence_per_tstep,
):
    num_tpoints = protrusion_existence_per_tstep.shape[0]
    num_nodes = protrusion_existence_per_tstep.shape[1]

    protrusion_node_index_and_tpoint_start_ends = np.zeros(
        (num_tpoints * num_nodes, 3), dtype=np.int64
    )
    num_protrusions = -1

    for ni in range(num_nodes):
        protrusion_start = False
        for ti in range(num_tpoints):
            if protrusion_existence_per_tstep[ti][ni] == 1:
                if protrusion_start == False:
                    protrusion_start = True
                    num_protrusions += 1
                    protrusion_node_index_and_tpoint_start_ends[num_protrusions][0] = ni
                    protrusion_node_index_and_tpoint_start_ends[num_protrusions][1] = ti
            else:
                if protrusion_start == True:
                    protrusion_start = False
                    protrusion_node_index_and_tpoint_start_ends[num_protrusions][2] = ti

            if ti == (num_tpoints - 1):
                if protrusion_start == True:
                    protrusion_node_index_and_tpoint_start_ends[num_protrusions][2] = (
                        ti + 1
                    )

    return protrusion_node_index_and_tpoint_start_ends[: (num_protrusions + 1)]


@nb.jit(nopython=True)
def determine_protrusion_lifetimes_and_average_directions(
    T, protrusion_node_index_and_tpoint_start_ends, protrusion_direction_per_tstep
):
    num_protrusions = protrusion_node_index_and_tpoint_start_ends.shape[0]
    protrusion_lifetime_and_average_directions = np.zeros(
        (num_protrusions, 2), dtype=np.float64
    )

    for pi in range(num_protrusions):
        ni, ti_start, ti_end = (
            protrusion_node_index_and_tpoint_start_ends[pi][0],
            protrusion_node_index_and_tpoint_start_ends[pi][1],
            protrusion_node_index_and_tpoint_start_ends[pi][2],
        )
        lifetime = (ti_end - ti_start) * T / 60.0
        directions = protrusion_direction_per_tstep[ti_start:ti_end, ni]
        direction_xs = directions[:, 0]
        direction_ys = directions[:, 1]
        average_direction = np.zeros(2, dtype=np.float64)
        average_direction[0] = np.sum(direction_xs) / directions.shape[0]
        average_direction[1] = np.sum(direction_ys) / directions.shape[0]

        protrusion_lifetime_and_average_directions[pi][0] = lifetime
        protrusion_lifetime_and_average_directions[pi][
            1
        ] = geometry.calculate_2D_vector_direction(average_direction)

    return protrusion_lifetime_and_average_directions


@nb.jit(nopython=True)
def determine_protrusion_groups(protrusion_existence_per_node):
    num_nodes = protrusion_existence_per_node.shape[0]
    visited_nodes = np.zeros(num_nodes, dtype=np.int64)

    for ni in range(num_nodes):
        if protrusion_existence_per_node[ni] == 0:
            visited_nodes[ni] = 1

    num_protrusion_groups = 0
    protrusion_groups = np.zeros((int(num_nodes / 2), 2), dtype=np.int64)

    tracking_group = False
    currently_tracking_protrusion_index = 0

    while not np.all(visited_nodes):
        if not tracking_group:
            for ni in range(num_nodes):
                vn = visited_nodes[ni] == 1
                if not vn:
                    tracking_group = True
                    currently_tracking_protrusion_index = ni
                    visited_nodes[ni] = 1
                    break
                else:
                    visited_nodes[ni] = 1
        else:
            found_left_boundary = False
            found_right_boundary = False

            left_boundary = currently_tracking_protrusion_index
            right_boundary = currently_tracking_protrusion_index

            for ni in range(num_nodes):
                if found_left_boundary:
                    break

                li = (left_boundary - 1) % num_nodes

                if protrusion_existence_per_node[li]:
                    left_boundary = li
                else:
                    found_left_boundary = True

                visited_nodes[li] = 1

            for ni in range(num_nodes):
                if found_right_boundary:
                    break

                ri = (right_boundary + 1) % num_nodes

                if protrusion_existence_per_node[ri]:
                    right_boundary = ri
                else:
                    found_right_boundary = True

                visited_nodes[ri] = 1

            protrusion_groups[num_protrusion_groups][0] = left_boundary
            protrusion_groups[num_protrusion_groups][1] = right_boundary
            num_protrusion_groups += 1
            tracking_group = False
            currently_tracking_protrusion_index = 0

    return num_protrusion_groups, protrusion_groups


@nb.jit(nopython=True)
def determine_num_nodes_in_protrusion_group(
    num_nodes_in_cell, left_boundary, right_boundary
):
    if left_boundary > right_boundary:
        num_nodes_right = right_boundary + 1
        num_nodes_left = num_nodes_in_cell - left_boundary

        return num_nodes_right + num_nodes_left
    else:
        return (right_boundary - left_boundary) + 1


@nb.jit(nopython=True)
def determine_nodes_in_protrusion_group(
    num_nodes_in_cell, left_boundary, right_boundary, protrusion_existence
):
    if left_boundary > right_boundary:
        num_nodes_right = right_boundary + 1
        num_nodes_left = num_nodes_in_cell - left_boundary

        num_nodes_in_protrusion_group = num_nodes_right + num_nodes_left
    else:
        num_nodes_in_protrusion_group = (right_boundary - left_boundary) + 1

    nodes_in_protrusion_group = np.zeros(num_nodes_in_protrusion_group, dtype=np.int64)
    for xi in range(num_nodes_in_protrusion_group):
        protrusive_node = (left_boundary + xi) % num_nodes_in_cell
        nodes_in_protrusion_group[xi] = protrusive_node

    return nodes_in_protrusion_group


@nb.jit(nopython=True)
def determine_if_protrusion_group_is_same_as_focus_protrusion_group(group, focus_group):
    if group[0] == focus_group[0] and group[1] == focus_group[1]:
        return True
    else:
        return False


def determine_if_protrusion_group_lies_in_focus_protrusion_group(group, focus_group):
    if focus_group[0] > focus_group[1]:
        right_boundary_within = group[1] <= focus_group[1]
        left_boundary_within = (group[0] >= 0) or (group[0] >= focus_group[0])
    else:
        right_boundary_within = group[1] <= focus_group[1]
        left_boundary_within = group[0] >= focus_group[0]

    return right_boundary_within and left_boundary_within


@nb.jit(nopython=True)
def determine_central_protrusion_group_nodes(
    num_nodes_in_cell, num_nodes_in_group, left_boundary, right_boundary
):
    central_node_delta = num_nodes_in_group / 2
    from_left_center = (left_boundary + central_node_delta) % num_nodes_in_cell
    from_right_center = (right_boundary - central_node_delta) % num_nodes_in_cell
    return (
        int(np.floor(from_left_center) % num_nodes_in_cell),
        int(np.ceil(from_right_center) % num_nodes_in_cell),
    )


@nb.jit(nopython=True)
def determine_if_protrusion_group_is_related_to_focus_group(
    num_nodes_in_cell,
    num_nodes_in_group,
    num_nodes_in_focus_group,
    group_central_node,
    focus_group_central_node,
):
    central_nodes_delta = min(
        (group_central_node - focus_group_central_node) % 16,
        (focus_group_central_node - group_central_node) % 16,
    )

    if central_nodes_delta < num_nodes_in_focus_group / 4:
        return central_nodes_delta, True
    else:
        return 0, False


# @nb.jit(nopython=True)
def determine_best_related_protrusion_group(
    num_nodes_in_cell, focus_group, groups_to_compare_against
):
    num_nodes_in_focus_group = determine_num_nodes_in_protrusion_group(
        num_nodes_in_cell, focus_group
    )
    num_groups = groups_to_compare_against.shape[0]
    num_nodes_in_group_per_group = np.zeros(num_groups, dtype=np.int64)

    for gi in range(num_groups):
        num_nodes_in_group_per_group[gi] = determine_num_nodes_in_protrusion_group(
            num_nodes_in_cell, groups_to_compare_against[gi]
        )

    group_central_nodes = np.zeros(num_groups, dtype=np.int64)
    for gi in range(num_groups):
        group_central_nodes[gi] = determine_central_protrusion_group_node(
            num_nodes_in_cell,
            num_nodes_in_group_per_group[gi],
            groups_to_compare_against[gi],
        )

    focus_group_central_node = determine_central_protrusion_group_node(
        num_nodes_in_cell, num_nodes_in_focus_group, focus_group
    )

    num_possible_related_groups = 0
    possible_related_group_scores = np.zeros(num_groups, dtype=np.int64)
    possible_related_group_indices = np.zeros(num_groups, dtype=np.int64)
    for gi in enumerate(groups_to_compare_against):
        group = groups_to_compare_against[gi]
        if determine_if_protrusion_group_is_same_as_focus_protrusion_group(
            group, focus_group
        ):
            return gi
        else:
            delta, is_related = determine_if_protrusion_group_is_related_to_focus_group(
                num_nodes_in_cell,
                num_nodes_in_group_per_group[gi],
                num_nodes_in_focus_group,
                group_central_nodes[gi],
                focus_group_central_node,
            )

            if is_related:
                possible_related_group_scores[num_possible_related_groups] = delta
                possible_related_group_indices[num_possible_related_groups] = gi
                num_possible_related_groups += 1

    possible_related_group_scores = possible_related_group_scores[
        :num_possible_related_groups
    ]
    possible_related_group_indices = possible_related_group_indices[
        :num_possible_related_groups
    ]
    if num_possible_related_groups == 0:
        return -1
    else:
        sorted_indices = np.argsort(possible_related_group_scores)
        return possible_related_group_indices[sorted_indices[0]]


@nb.jit(nopython=True)
def determine_protrusion_group_direction(
    num_nodes_in_cell,
    protrusion_directions,
    protrusion_existence,
    left_boundary,
    right_boundary,
):
    num_nodes_in_protrusion_group = determine_num_nodes_in_protrusion_group(
        num_nodes_in_cell, left_boundary, right_boundary
    )
    lc, rc = determine_central_protrusion_group_nodes(
        num_nodes_in_cell, num_nodes_in_protrusion_group, left_boundary, right_boundary
    )

    if not (protrusion_existence[lc] == 1 and protrusion_existence[rc] == 1):
        raise Exception("calculated central nodes are not protrusive!")

    if lc == rc:
        return num_nodes_in_protrusion_group, protrusion_directions[lc]
    else:
        return num_nodes_in_protrusion_group, (protrusion_directions[lc] + protrusion_directions[rc]) / 2


@nb.jit(nopython=True)
def determine_protrusion_group_direction_old(
    num_nodes_in_cell,
    protrusion_directions,
    protrusion_existence,
    left_boundary,
    right_boundary,
):
    nodes_in_protrusion_group = determine_nodes_in_protrusion_group(
        num_nodes_in_cell, left_boundary, right_boundary, protrusion_existence
    )
    num_nodes_in_protrusion_group = nodes_in_protrusion_group.shape[0]

    avg_protrusion_direction = np.zeros(2, dtype=np.float64)

    for ni in nodes_in_protrusion_group:
        avg_protrusion_direction += (
            protrusion_directions[ni] / num_nodes_in_protrusion_group
        )

    return avg_protrusion_direction


# @nb.jit(nopython=True)
# def determine_protrusion_group_lifetimes_and_directions(
#    T, protrusion_existence_per_tstep, protrusion_direction_per_tstep
# ):
#    num_timesteps = protrusion_existence_per_tstep.shape[0]
#    last_timestep = num_timesteps - 1
#
#    num_nodes = protrusion_existence_per_tstep.shape[1]
#    max_num_protrusion_groups = int(num_nodes/2)
#
#    num_protrusion_groups_per_timestep = np.zeros(num_timesteps, dtype=np.int64)
#    protrusion_groups_per_timestep = np.zeros((num_timesteps, max_num_protrusion_groups, 2), dtype=np.int64)
#    for ti in range(num_timesteps):
#        pe = protrusion_existence_per_tstep[ti]
#        num_pgs, pgs = determine_protrusion_groups(pe)
#        num_protrusion_groups_per_timestep[ti] = num_pgs
#        protrusion_groups_per_timestep[ti] = pgs
#
#    max_num_organized_protrusion_groups = np.sum(num_protrusion_groups_per_timestep)
#    num_organized_protrusion_groups = 0
#    num_timesteps_per_organized_protrusion_groups = np.zeros(max_num_organized_protrusion_groups, dtype=np.int64)
#    organized_protrusion_groups = np.zeros((max_num_organized_protrusion_groups, num_timesteps, 3), dtype=np.int64)
#    visited_protrusion_groups = np.ones((num_timesteps, max_num_protrusion_groups), dtype=np.int64)
#
#    for ti in range(num_timesteps):
#        num_protrusion_groups = num_protrusion_groups_per_timestep[ti]
#        for pi in range(num_protrusion_groups):
#            visited_protrusion_groups[ti][pi] = 0
#
#    tracking_protrusion_group = False
#    tracked_protrusion_group_timestep = -1
#    tracked_protrusion_group = np.zeros(2, dtype=np.int64)
#    num_nodes_in_cell = protrusion_existence_per_tstep.shape[1]
#
#    while not np.all(visited_protrusion_groups == 1):
#        if not tracking_protrusion_group:
#            for xi in range(num_timesteps):
#                ti = last_timestep - xi
#                npg = num_protrusion_groups_per_timestep[ti]
#                pgs = protrusion_groups_per_timestep[ti]
#
#                for pi, pg in enumerate(pgs[:npg]):
#                    if visited_protrusion_groups[ti][pi]:
#                        continue
#                    else:
#                        visited_protrusion_groups[ti][pi] = 1
#                        tracking_protrusion_group = True
#                        tracked_protrusion_group_timestep = ti
#
#                        tracked_protrusion_group[0] = pg[0]
#                        tracked_protrusion_group[1] = pg[1]
#                        break
#
#                if tracking_protrusion_group:
#                    break
#        else:
#            num_related_protrusion_groups = 1
#            related_protrusion_groups = np.zeros((num_timesteps, 3), dtype=np.int64)
#            related_protrusion_groups[0][0] = tracked_protrusion_group_timestep
#            related_protrusion_groups[0][1] = tracked_protrusion_group[0]
#            related_protrusion_groups[0][2] = tracked_protrusion_group[1]
#
#            last_ti = tracked_protrusion_group_timestep
#            for xi in range(last_ti):
#                ti = last_ti - (xi + 1)
#                protrusion_groups_to_compare_against = protrusion_groups_per_timestep[ti][:num_protrusion_groups_per_timestep[ti]]
#
#                best_related_protrusion_group_index = determine_best_related_protrusion_group(num_nodes_in_cell, related_protrusion_groups[num_related_protrusion_groups - 1][1:], protrusion_groups_to_compare_against)
#
#                if best_related_protrusion_group_index != -1:
#                    pg = protrusion_groups_to_compare_against[best_related_protrusion_group_index]
#                    related_protrusion_groups[num_related_protrusion_groups][0] = ti
#                    related_protrusion_groups[num_related_protrusion_groups][1] = pg[0]
#                    related_protrusion_groups[num_related_protrusion_groups][2] = pg[1]
#                    num_related_protrusion_groups += 1
#                    visited_protrusion_groups[ti][best_related_protrusion_group_index] = 1
#                else:
#                    break
#
#            num_timesteps_per_organized_protrusion_groups[num_organized_protrusion_groups] = num_related_protrusion_groups
#            organized_protrusion_groups[num_organized_protrusion_groups][:num_related_protrusion_groups] = np.flip(related_protrusion_groups[:num_related_protrusion_groups], axis=0)
#            num_organized_protrusion_groups += 1
#            tracking_protrusion_group = False
#
#    protrusion_group_lifetimes_and_directions = np.zeros((num_organized_protrusion_groups, 2), dtype=np.float64)
#    for nopg in num_organized_protrusion_groups:
#        nts = num_timesteps_per_organized_protrusion_groups[nopg]
#        lifetime = nts*T/60.0
#        protrusion_group_lifetimes_and_directions[nopg][0] = lifetime
#
#        directions = np.zeros(nts, dtype=np.float64)
#        opgs = organized_protrusion_groups[nopg][:nts]
#        for xi in range(nts):
#            ti = opgs[xi][0]
#            directions[xi] = determine_protrusion_group_direction(num_nodes_in_cell, protrusion_direction_per_tstep[ti], opgs[xi][1:])
#
#        protrusion_group_lifetimes_and_directions[nopg][1] = np.average(directions)
#
#
#    return protrusion_group_lifetimes_and_directions

@nb.jit(nopython=True)
def determine_polarization_vectors_of_cells(protrusion_existence_per_cell, protrusion_directions_per_cell):
    num_cells = protrusion_existence_per_cell.shape[0]
    num_nodes_in_cell = protrusion_existence_per_cell.shape[1]
    max_num_protrusion_groups = int(num_nodes_in_cell / 2)

    num_protrusion_groups_per_cell = np.zeros(num_cells, dtype=np.int64)
    protrusion_group_boundaries_per_cell = np.zeros((num_cells, max_num_protrusion_groups, 2), dtype=np.int64)

    for ci in range(num_cells):
        pe = protrusion_existence_per_cell[ci]
        num_pgs, pgs = determine_protrusion_groups(pe)
        num_protrusion_groups_per_cell[ci] = num_pgs
        protrusion_group_boundaries_per_cell[ci] = pgs

    polarization_vectors_per_cell = np.zeros((num_cells, 2), dtype=np.float64)

    for ci in range(num_cells):
        num_protrusion_groups = num_protrusion_groups_per_cell[ci]
        relevant_protrusion_groups = protrusion_group_boundaries_per_cell[ci][
                                     :num_protrusion_groups
                                     ]
        relevant_protrusion_directions = protrusion_directions_per_cell[ci]
        relevant_protrusion_existence = protrusion_existence_per_cell[ci]

        protrusion_group_direction_vectors = np.zeros(
            (num_protrusion_groups, 2), dtype=np.float64
        )
        for gi in range(num_protrusion_groups):
            pg = relevant_protrusion_groups[gi]
            num_nodes_in_protrusion_group, protrusion_direction_vector = determine_protrusion_group_direction(
                num_nodes_in_cell,
                relevant_protrusion_directions,
                relevant_protrusion_existence,
                pg[0],
                pg[1],
            )
            protrusion_group_direction = geometry.calculate_2D_vector_direction(
                protrusion_direction_vector
            )
            protrusion_group_direction_vectors[gi] = (
                                                                 num_nodes_in_protrusion_group / num_nodes_in_cell) * protrusion_direction_vector

        polarization_vectors_per_cell[ci] = np.sum(
            protrusion_group_direction_vectors, axis=0
        )

    return polarization_vectors_per_cell

# def determine_spatial_location_type_of_cells_using_all_node_info(node_positions_per_cell):
#     num_cells = node_positions_per_cell.shape[0]
#     num_nodes = node_positions_per_cell.shape[1]
#
#     flattened_node_positions_per_cell = node_positions_per_cell.reshape((num_cells*num_nodes, 2))
#
#     classified_cells = np.zeros(num_cells, dtype=np.int64)
#     cell_classifications = np.zeros(num_cells, dtype=np.int64)
#
#     dtri = space.Delaunay(flattened_node_positions_per_cell)
#     chull = space.ConvexHull(flattened_node_positions_per_cell)
#
#     for x in chull.vertices:
#         ci = x % num_cells
#         ni = (x - ci)

def get_delaunay_neighbours(ix, dtri):
    helper = dtri.vertex_neighbor_vertices
    index_pointers = helper[0]
    indices = helper[1]

    return indices[index_pointers[ix]:index_pointers[ix+1]]

def determine_spatial_location_type_of_cells_using_cell_centroid_info(centroid_positions_per_cell_per_timestep):
    num_timesteps = centroid_positions_per_cell_per_timestep.shape[0]
    num_cells = centroid_positions_per_cell_per_timestep.shape[1]

    cell_classifications_per_cell_per_timestep = np.zeros((num_timesteps, num_cells), dtype=np.int64)
    delaunay_neighbours_per_cell_per_timestep = np.zeros((num_timesteps, num_cells, num_cells), dtype=np.int64)

    for ti in range(num_timesteps):
        cell_classifications_per_cell = np.zeros(num_cells, dtype=np.int64)
        centroid_positions_per_cell = centroid_positions_per_cell_per_timestep[ti]
        dtri = space.Delaunay(centroid_positions_per_cell)
        chull = space.ConvexHull(centroid_positions_per_cell)

        delaunay_neighbours_per_cell = np.zeros((num_cells, num_cells), dtype=np.int64)

        for ci in range(num_cells):
            this_cell_delaunay_neighbours = get_delaunay_neighbours(ci, dtri)
            num_delaunay_neighbours = this_cell_delaunay_neighbours.shape[0]
            delaunay_neighbours_per_cell[ci][0] = num_delaunay_neighbours
            delaunay_neighbours_per_cell[ci][1:(num_delaunay_neighbours + 1)] = this_cell_delaunay_neighbours

        for i in chull.vertices:
            neighbours = delaunay_neighbours_per_cell[i]
            relative_position_vectors = centroid_positions_per_cell[neighbours] - centroid_positions_per_cell[i]
            distance_to_neighbours = np.linalg.norm(relative_position_vectors, axis=1)
            closest = np.argmin(distance_to_neighbours)

            cell_classifications_per_cell[i] = 1
            cell_classifications_per_cell[neighbours] = 1

        cell_classifications_per_cell_per_timestep[ti] = cell_classifications_per_cell
        delaunay_neighbours_per_cell_per_timestep[ti] = delaunay_neighbours_per_cell

    return cell_classifications_per_cell_per_timestep, delaunay_neighbours_per_cell_per_timestep

@nb.jit(nopython=True)
def determine_polarization_vectors_and_protrusion_group_directions(
    protrusion_existence_per_tstep, protrusion_directions_per_tstep
):
    num_timesteps = protrusion_existence_per_tstep.shape[0]

    num_nodes_in_cell = protrusion_existence_per_tstep.shape[1]
    max_num_protrusion_groups = int(num_nodes_in_cell / 2)

    num_protrusion_groups_per_timestep = np.zeros(num_timesteps, dtype=np.int64)
    protrusion_groups_per_timestep = np.zeros(
        (num_timesteps, max_num_protrusion_groups, 2), dtype=np.int64
    )
    for ti in range(num_timesteps):
        pe = protrusion_existence_per_tstep[ti]
        num_pgs, pgs = determine_protrusion_groups(pe)
        num_protrusion_groups_per_timestep[ti] = num_pgs
        protrusion_groups_per_timestep[ti] = pgs

    polarization_vectors_per_timestep = np.zeros((num_timesteps, 2), dtype=np.float64)

    all_protrusion_group_directions = np.zeros(
        num_timesteps * max_num_protrusion_groups, dtype=np.float64
    )
    num_protrusion_group_directions = 0
    for ti in range(num_timesteps):
        num_protrusion_groups = num_protrusion_groups_per_timestep[ti]
        relevant_protrusion_groups = protrusion_groups_per_timestep[ti][
            :num_protrusion_groups
        ]
        relevant_protrusion_directions = protrusion_directions_per_tstep[ti]
        relevant_protrusion_existence = protrusion_existence_per_tstep[ti]

        protrusion_group_direction_vectors = np.zeros(
            (num_protrusion_groups, 2), dtype=np.float64
        )
        for gi in range(num_protrusion_groups):
            pg = relevant_protrusion_groups[gi]
            num_nodes_in_protrusion_group, protrusion_direction_vector = determine_protrusion_group_direction(
                num_nodes_in_cell,
                relevant_protrusion_directions,
                relevant_protrusion_existence,
                pg[0],
                pg[1],
            )
            protrusion_group_direction = geometry.calculate_2D_vector_direction(
                protrusion_direction_vector
            )
            protrusion_group_direction_vectors[gi] = num_nodes_in_protrusion_group*protrusion_direction_vector
            all_protrusion_group_directions[
                num_protrusion_group_directions
            ] = protrusion_group_direction
            num_protrusion_group_directions += 1

        polarization_vectors_per_timestep[ti] = np.sum(
            protrusion_group_direction_vectors, axis=0
        )

    return (
        all_protrusion_group_directions[:num_protrusion_group_directions],
        polarization_vectors_per_timestep,
    )


# @nb.jit(nopython=True)
def calculate_average_cil_signal(
    node_index, normalized_cil_signals_per_tstep, end_buffer_a, end_buffer_b
):
    neighbour_cil_signals = np.zeros((end_buffer_b - end_buffer_a, 3), dtype=np.float64)
    num_nodes = normalized_cil_signals_per_tstep.shape[1]
    ni_p1 = (node_index + 1) % num_nodes
    ni_m1 = (node_index - 1) % num_nodes

    neighbour_cil_signals[:, 0] = normalized_cil_signals_per_tstep[
        end_buffer_a:end_buffer_b, node_index
    ]
    neighbour_cil_signals[:, 1] = normalized_cil_signals_per_tstep[
        end_buffer_a:end_buffer_b, ni_p1
    ]
    neighbour_cil_signals[:, 2] = normalized_cil_signals_per_tstep[
        end_buffer_a:end_buffer_b, ni_m1
    ]

    return np.average(np.average(neighbour_cil_signals, axis=1))


def determine_likely_protrusion_start_end_causes(
    protrusion_node_index_and_tpoint_start_ends,
    normalized_rac_membrane_active_per_tstep,
    coa_signal_per_tstep,
    cil_signal_per_tstep,
    randomization_factors_per_tstep,
):
    num_protrusions = protrusion_node_index_and_tpoint_start_ends.shape[0]
    protrusion_start_end_causes = []
    num_nodes = normalized_rac_membrane_active_per_tstep.shape[1]

    global_avg_coa = np.average(coa_signal_per_tstep)
    if not (global_avg_coa > 1.1):
        normalized_coa_signals_per_tstep = coa_signal_per_tstep / 1e16
    else:
        normalized_coa_signals_per_tstep = coa_signal_per_tstep / global_avg_coa

    global_avg_cil = np.average(cil_signal_per_tstep)
    if not (global_avg_cil > 1.1):
        normalized_cil_signals_per_tstep = cil_signal_per_tstep / 1e16
    else:
        normalized_cil_signals_per_tstep = cil_signal_per_tstep / global_avg_cil

    max_rand = np.max(randomization_factors_per_tstep)
    if not (max_rand > 1.1):
        normalized_randomization_factors_per_tstep = (
            randomization_factors_per_tstep / 1e16
        )
    else:
        normalized_randomization_factors_per_tstep = (
            randomization_factors_per_tstep / max_rand
        )

    for pi in range(num_protrusions):
        ni, ti_start, ti_end = protrusion_node_index_and_tpoint_start_ends[pi]

        last_protrusion_at_this_node_end_tstep = -1
        if pi != 0:
            for x in range(pi):
                lpi = pi - (x + 1)
                if protrusion_node_index_and_tpoint_start_ends[lpi][0] == ni:
                    last_protrusion_at_this_node_end_tstep = protrusion_node_index_and_tpoint_start_ends[
                        lpi
                    ][
                        2
                    ]

        start_causes = []
        if ti_start == 0:
            start_causes.append("init")
        else:
            if last_protrusion_at_this_node_end_tstep != -1:
                start_buffer_a, start_buffer_b = (
                    last_protrusion_at_this_node_end_tstep
                    + int(0.75 * (ti_start - last_protrusion_at_this_node_end_tstep)),
                    ti_start,
                )
            else:
                start_buffer_a, start_buffer_b = max([0, ti_start - 50]), ti_start

            average_coa_signal = np.average(
                normalized_coa_signals_per_tstep[
                    start_buffer_a : (start_buffer_b + 1), ni
                ]
            )
            average_randomization_factor = np.average(
                normalized_randomization_factors_per_tstep[
                    start_buffer_a : (start_buffer_b + 1), ni
                ]
            )

            start_cause_found = False
            if average_coa_signal > 0.25 and global_avg_coa * average_coa_signal > 1.1:
                start_causes.append("coa")
                start_cause_found = True

            if (
                average_randomization_factor > 0.25
                and average_randomization_factor * max_rand > 1.0
            ):
                start_causes.append("rand")
                start_cause_found = True

            if not start_cause_found:
                this_node_rac = normalized_rac_membrane_active_per_tstep[
                    start_buffer_a : (start_buffer_b + 1), ni
                ]
                p1_neigher_node_rac = normalized_rac_membrane_active_per_tstep[
                    start_buffer_a : (start_buffer_b + 1), (ni + 1) % num_nodes
                ]
                m1_neigher_node_rac = normalized_rac_membrane_active_per_tstep[
                    start_buffer_a : (start_buffer_b + 1), (ni - 1) % num_nodes
                ]
                if np.average(
                    np.append(p1_neigher_node_rac, m1_neigher_node_rac)
                ) > np.average(this_node_rac):
                    start_causes.append("neighbour")

        end_buffer_a, end_buffer_b = int(0.5 * (ti_end - ti_start) + ti_start), ti_end

        average_cil_signal = calculate_average_cil_signal(
            ni, normalized_cil_signals_per_tstep, end_buffer_a, end_buffer_b
        )

        end_causes = []
        if average_cil_signal > 0.25 and global_avg_cil * average_cil_signal > 1.1:
            end_causes.append("cil")
        else:
            end_causes.append("other")

        protrusion_start_end_causes.append(
            (copy.deepcopy(start_causes), copy.deepcopy(end_causes))
        )

    return protrusion_start_end_causes


def collate_protrusion_data_for_cell(cell_index, T, storefile_path, max_tstep=None):
    (
        rac_membrane_active_per_tstep,
        rho_membrane_active_per_tstep,
        uivs_per_node_per_timestep,
    ) = hardio.get_multiple_data_until_timestep(
        cell_index,
        max_tstep,
        ["rac_membrane_active", "rho_membrane_active", "unit_in_vec"],
        ["n", "n", "v"],
        storefile_path,
    )

    normalized_rac_membrane_active_per_tstep = normalize_rgtpase_data_per_tstep(
        rac_membrane_active_per_tstep
    )
    (
        protrusion_existence_per_tstep,
        protrusion_directions_per_tstep,
    ) = determine_protrusion_existence_and_direction(
        normalized_rac_membrane_active_per_tstep,
        rac_membrane_active_per_tstep,
        rho_membrane_active_per_tstep,
        uivs_per_node_per_timestep,
    )

    protrusion_node_index_and_tpoint_start_ends = determine_protrusion_node_index_and_tpoint_start_ends(
        protrusion_existence_per_tstep
    )
    protrusion_lifetime_and_average_directions = determine_protrusion_lifetimes_and_average_directions(
        T, protrusion_node_index_and_tpoint_start_ends, protrusion_directions_per_tstep
    )

    (
        all_protrusion_group_directions,
        polarization_vectors_per_timestep,
    ) = determine_polarization_vectors_and_protrusion_group_directions(
        protrusion_existence_per_tstep, protrusion_directions_per_tstep
    )

    return (
        protrusion_existence_per_tstep,
        protrusion_lifetime_and_average_directions,
        all_protrusion_group_directions,
        polarization_vectors_per_timestep,
    )


def collate_protrusion_data(num_cells, T, storefile_path, max_tstep=None):
    protrusion_existence_per_tstep_per_cell = []
    protrusion_lifetime_and_average_directions_per_cell = []
    protrusion_group_directions_per_cell = []

    for cell_index in range(num_cells):
        (
            protrusion_existence_per_tstep,
            protrusion_lifetime_and_average_directions,
            protrusion_group_directions,
            _,
        ) = collate_protrusion_data_for_cell(
            cell_index, T, storefile_path, max_tstep=max_tstep
        )
        protrusion_lifetime_and_average_directions_per_cell.append(
            protrusion_lifetime_and_average_directions
        )
        protrusion_group_directions_per_cell.append(protrusion_group_directions)
        protrusion_existence_per_tstep_per_cell.append(protrusion_existence_per_tstep)

    return (
        protrusion_existence_per_tstep_per_cell,
        protrusion_lifetime_and_average_directions_per_cell,
        protrusion_group_directions_per_cell,
    )


# ==============================================================================


def calculate_cell_speeds_and_directions_until_tstep(
    cell_index, max_tstep, storefile_path, T, L
):
    node_coords_per_tstep = hardio.get_node_coords_until_tstep(
        cell_index, max_tstep, storefile_path
    )
    centroid_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep) * L

    velocities = calculate_velocities(centroid_per_tstep, T)
    speeds = np.linalg.norm(velocities, axis=1)
    directions = geometry.calculate_2D_vector_directions(
        velocities.shape[0], velocities
    )

    return speeds, directions


# =====================================================


def calculate_all_cell_speeds_and_directions_until_tstep(
    num_cells, max_tstep, storefile_path, T, cell_Ls
):
    all_cells_speed_and_directions = []
    for ci in range(num_cells):
        all_cells_speed_and_directions.append(
            calculate_cell_speeds_and_directions_until_tstep(
                ci, max_tstep, storefile_path, T, cell_Ls[ci]
            )
        )

    return all_cells_speed_and_directions


# =============================================================================


def calculate_normalized_group_area_over_time(
    num_cells, num_timepoints, storefile_path
):
    all_cell_centroids_per_tstep = np.zeros(
        (num_timepoints, num_cells, 2), dtype=np.float64
    )

    # ------------------------

    for ci in range(num_cells):
        cell_centroids_per_tstep = calculate_cell_centroids_until_tstep(
            ci, num_timepoints, storefile_path
        )

        all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep

    # ------------------------
    if num_cells < 0:
        raise Exception("Negative number of cells given!")

    if num_cells == 0:
        return np.zeros(num_timepoints, dtype=np.float64)
    elif num_cells == 1:
        return np.ones(num_timepoints, dtype=np.float64)
    elif num_cells == 2:
        distance_between_cells_at_all_timesteps = np.linalg.norm(
            all_cell_centroids_per_tstep[:, 0, :]
            - all_cell_centroids_per_tstep[:, 1, :],
            axis=1,
        )

        return (
            distance_between_cells_at_all_timesteps
            / distance_between_cells_at_all_timesteps[0]
        )
    else:
        delaunay_triangulations_per_tstep = []
        for cell_centroids in all_cell_centroids_per_tstep:
            try:
                delaunay_triangulations_per_tstep.append(space.Delaunay(cell_centroids))
            except:
                delaunay_triangulations_per_tstep.append(None)

        convex_hull_areas_per_tstep = []
        for dt, all_cell_centroids in zip(
            delaunay_triangulations_per_tstep, all_cell_centroids_per_tstep
        ):
            if dt != None:
                simplices = all_cell_centroids[dt.simplices]
                simplex_areas = np.array(
                    [
                        geometry.calculate_polygon_area(simplex.shape[0], simplex)
                        for simplex in simplices
                    ]
                )
                convex_hull_areas_per_tstep.append(
                    np.round(np.sum(simplex_areas), decimals=3)
                )
            else:
                convex_hull_areas_per_tstep.append(np.nan)
        return np.array(convex_hull_areas_per_tstep) / convex_hull_areas_per_tstep[0]


# =============================================================================


@nb.jit(nopython=True)
def is_prospective_edge_already_counted(prospective_edge, edges):
    for ei in range(edges.shape[0]):
        edge = edges[ei]
        if edge[0] == -1:
            return False
        else:
            if np.all(edge == prospective_edge):
                return True

    return False


# =====================================================


def determine_edges(dt):
    edges = -1 * np.ones(
        (dt.points.shape[0] + dt.simplices.shape[0] - 1, 2), dtype=np.int64
    )
    sorted_simplices = np.sort(dt.simplices, axis=1)
    pairs = np.array([[0, 1], [0, 2], [1, 2]])

    ei = 0
    for simplex in sorted_simplices:
        for pair in pairs:
            if ei == edges.shape[0]:
                break

            prospective_edge = simplex[pair]

            if not is_prospective_edge_already_counted(prospective_edge, edges):
                edges[ei] = prospective_edge
                ei += 1
        else:  # http://psung.blogspot.ca/2007/12/for-else-in-python.html
            continue  # executed only if the loop ended normally
        break  # only executed if continue under else statement was skipped

    return edges


# =====================================================


@nb.jit(nopython=True)
def calculate_edge_lengths(points, edges):
    edge_lengths = np.zeros(edges.shape[0], dtype=np.float64)

    for ei in range(edges.shape[0]):
        edge = edges[ei]
        x, y = points[edge[0]] - points[edge[1]]
        edge_lengths[ei] = np.sqrt(x * x + y * y)

    return edge_lengths


# =====================================================


def calculate_simple_intercellular_separations_and_subgroups(
    init_cell_group_separation, all_cell_centroids
):
    num_cells = all_cell_centroids.shape[0]
    intercellular_separations = np.zeros(num_cells - 1, dtype=np.float64)
    relevant_x_centroids = all_cell_centroids[:, 0]
    sorted_cell_indices = sorted(
        np.arange(num_cells), key=lambda x: relevant_x_centroids[x]
    )

    subgroups = []
    current_subgroup = [0]
    for n in range(num_cells - 1):
        cia = sorted_cell_indices[n]
        cib = sorted_cell_indices[n + 1]
        v = all_cell_centroids[cia] - all_cell_centroids[cib]
        d = math.sqrt(v[0] * v[0] + v[1] * v[1])

        if d > 3.0 * init_cell_group_separation:
            if len(current_subgroup) > 0:
                subgroups.append(copy.deepcopy(current_subgroup))
                current_subgroup = [n + 1]
        else:
            current_subgroup.append(n + 1)

        intercellular_separations[n] = d
    subgroups.append(current_subgroup)

    return intercellular_separations, subgroups


# =====================================================

# @nb.jit(nopython=True)
def calculate_simple_intercellular_separations_and_subgroups_per_timestep(
    init_cell_group_separation, all_cell_centroids_per_tstep
):
    num_timepoints = all_cell_centroids_per_tstep.shape[0]
    num_cells = all_cell_centroids_per_tstep.shape[1]
    intercellular_separations_per_timestep = np.zeros(
        (num_timepoints, num_cells - 1), dtype=np.float64
    )

    subgroups_per_timestep = []
    for ti in range(num_timepoints):
        relevant_cell_centroids = all_cell_centroids_per_tstep[ti]
        (
            intercellular_separations_per_timestep[ti],
            subgroups,
        ) = calculate_simple_intercellular_separations_and_subgroups(
            init_cell_group_separation, relevant_cell_centroids
        )
        subgroups_per_timestep.append(subgroups)

    return intercellular_separations_per_timestep, subgroups_per_timestep


# =====================================================


@nb.jit(nopython=True)
def determine_group_aspect_ratio_per_tstep(all_cell_coords_per_tstep):
    num_timepoints = all_cell_coords_per_tstep.shape[0]
    group_aspect_ratio_per_tstep = np.zeros(num_timepoints, dtype=np.float64)

    for ti in range(num_timepoints):
        cell_coords = all_cell_coords_per_tstep[ti]
        xs = cell_coords[:, :, 0]
        ys = cell_coords[:, :, 1]

        w = np.max(xs) - np.min(xs)
        h = np.max(ys) - np.min(ys)

        group_aspect_ratio_per_tstep[ti] = w / h

    return group_aspect_ratio_per_tstep


# =====================================================


def calculate_group_aspect_ratio_over_time(
    num_cells, num_nodes, num_timepoints, storefile_path
):
    all_cell_coords_per_tstep = np.zeros(
        (num_timepoints, num_cells, num_nodes, 2), dtype=np.float64
    )

    for ci in range(num_cells):
        all_cell_coords_per_tstep[:, ci, :, :] = hardio.get_node_coords_until_tstep(
            ci, num_timepoints, storefile_path
        )

    group_aspect_ratio_per_tstep = determine_group_aspect_ratio_per_tstep(
        all_cell_coords_per_tstep
    )

    return group_aspect_ratio_per_tstep


# =====================================================


def calculate_normalized_group_area_and_average_cell_separation_over_time(
    cell_radius, num_cells, num_timepoints, storefile_path, get_subgroups=False
):
    all_cell_centroids_per_tstep = np.zeros(
        (num_timepoints, num_cells, 2), dtype=np.float64
    )

    # ------------------------

    for ci in range(num_cells):
        cell_centroids_per_tstep = calculate_cell_centroids_until_tstep(
            ci, num_timepoints, storefile_path
        )

        all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep

    # ------------------------
    if num_cells < 0:
        raise Exception("Negative number of cells given!")

    cell_subgroups_per_timestep = []
    if num_cells == 0:
        return (
            np.zeros(num_timepoints, dtype=np.float64),
            np.zeros(num_timepoints, dtype=np.float64),
            cell_subgroups_per_timestep,
        )
    elif num_cells == 1:
        return (
            np.ones(num_timepoints, dtype=np.float64),
            np.ones(num_timepoints, dtype=np.float64),
            None,
        )
    elif num_cells == 2:
        distance_between_cells_at_all_timesteps = np.linalg.norm(
            all_cell_centroids_per_tstep[:, 0, :]
            - all_cell_centroids_per_tstep[:, 1, :],
            axis=1,
        )

        return (
            np.nan * np.zeros(num_timepoints, dtype=np.float64),
            distance_between_cells_at_all_timesteps
            / distance_between_cells_at_all_timesteps[0],
            cell_subgroups_per_timestep,
        )
    else:
        delaunay_triangulations_per_tstep = []
        simple_intercellular_separations_per_tstep = []
        initial_intercellular_separation = 2 * cell_radius
        # simple_separation_calculation = False
        init_delaunay_success = False
        for ti, cell_centroids in enumerate(all_cell_centroids_per_tstep):
            try:
                delaunay_triangulations_per_tstep.append(space.Delaunay(cell_centroids))
                simple_intercellular_separations_per_tstep.append(None)
                if ti == 0:
                    init_delaunay_success = True
            except:
                delaunay_triangulations_per_tstep.append(None)
                (
                    separations,
                    subgroups,
                ) = calculate_simple_intercellular_separations_and_subgroups(
                    initial_intercellular_separation, cell_centroids
                )
                simple_intercellular_separations_per_tstep.append(separations)

            if not init_delaunay_success and get_subgroups:
                (
                    separations,
                    subgroups,
                ) = calculate_simple_intercellular_separations_and_subgroups(
                    initial_intercellular_separation, cell_centroids
                )
                cell_subgroups_per_timestep.append(copy.deepcopy(subgroups))

        convex_hull_areas_per_tstep = []
        average_cell_separation_per_tstep = []
        min_cell_separation_tstep0 = -1.0
        for ti, all_cell_centroids in enumerate(all_cell_centroids_per_tstep):
            simple_intercellular_separations = simple_intercellular_separations_per_tstep[
                ti
            ]
            dt = delaunay_triangulations_per_tstep[ti]

            if type(simple_intercellular_separations) != type(None):
                convex_hull_areas_per_tstep.append(np.nan)
                cell_separations = simple_intercellular_separations_per_tstep[ti]
                average_cell_separation_per_tstep.append(np.average(cell_separations))
                if ti == 0:
                    min_cell_separation_tstep0 = np.min(cell_separations)

            if dt != None:
                dt = delaunay_triangulations_per_tstep[ti]
                simplices = all_cell_centroids[dt.simplices]
                simplex_areas = np.array(
                    [geometry.calculate_polygon_area(simplex) for simplex in simplices]
                )
                convex_hull_areas_per_tstep.append(
                    np.round(np.sum(simplex_areas), decimals=3)
                )
                edges = determine_edges(dt)
                edge_lengths = calculate_edge_lengths(dt.points, edges)

                if ti == 0:
                    min_cell_separation_tstep0 = np.min(edge_lengths)

                average_cell_separation_per_tstep.append(np.average(edge_lengths))

        return (
            np.array(convex_hull_areas_per_tstep) / convex_hull_areas_per_tstep[0],
            np.array(average_cell_separation_per_tstep) / min_cell_separation_tstep0,
            cell_subgroups_per_timestep,
        )


# =============================================================================


@nb.jit(nopython=True)
def calculate_cos_theta_for_direction_autocorr_coeffs(a, b):
    ax, ay = a
    bx, by = b

    norm_a = np.sqrt(ax * ax + ay * ay)
    norm_b = np.sqrt(bx * bx + by * by)

    if norm_a < 1e-6:
        a = np.random.rand(2)
        ax, ay = a
        norm_a = np.sqrt(ax * ax + ay * ay)

    if norm_b < 1e-6:
        b = np.random.rand(2)
        bx, by = b
        norm_b = np.sqrt(bx * bx + by * by)

    ax_, ay_ = a / norm_a
    bx_, by_ = b / norm_b

    return ax_ * bx_ + ay_ * by_


# =====================================================


@nb.jit(nopython=True)
def calculate_direction_autocorr_coeffs_for_persistence_time(displacements):
    N = displacements.shape[0]

    all_das = np.zeros(N, dtype=np.float64)
    first_negative_n = -1

    for n in range(N):
        sum_cos_thetas = 0.0
        m = 0.0

        i = 0
        while i + n < N:
            cos_theta = calculate_cos_theta_for_direction_autocorr_coeffs(
                displacements[i], displacements[i + n]
            )
            sum_cos_thetas += cos_theta
            m += 1
            i += 1

        da = (1.0 / m) * sum_cos_thetas
        if da < 0.0 and first_negative_n == -1:
            first_negative_n = n
            break

        all_das[n] = da

    if first_negative_n == -1:
        first_negative_n = N

    return all_das[:first_negative_n]


# =====================================================


@nb.jit(nopython=True, nogil=True)
def calculate_direction_autocorr_coeff_parallel_worker(N, ns, dacs, displacements):
    for n in ns:
        m = 0.0
        sum_cos_thetas = 0.0
        i = 0

        while i + n < N:
            cos_theta = calculate_cos_theta_for_direction_autocorr_coeffs(
                displacements[i], displacements[i + n]
            )
            sum_cos_thetas += cos_theta
            m += 1
            i += 1

        dacs[n] = (1.0 / m) * sum_cos_thetas


# =====================================================


@nb.jit(nopython=True, nogil=True)
def calculate_protrusion_lifetime_autocorr_coeff_parallel_worker(
    N, ns, acs, protrusion_lifetimes_per_direction
):
    for n in ns:
        m = 0.0
        sum_cos_thetas = 0.0
        i = 0

        while i + n < N:
            cos_theta = calculate_cos_theta_for_direction_autocorr_coeffs(
                displacements[i], displacements[i + n]
            )
            sum_cos_thetas += cos_theta
            m += 1
            i += 1

        dacs[n] = (1.0 / m) * sum_cos_thetas


# =====================================================


@nb.jit(nopython=True)
def find_first_negative_n(dacs):
    for n, dac in enumerate(dacs):
        if n < 0.0:
            return n


# =====================================================


def calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
    displacements, num_threads=4
):
    N = displacements.shape[0]
    dacs = np.ones(N, dtype=np.float64)

    task_indices = np.arange(1, N, 30)  # np.linspace(1, N, num=N/30.0, dtype=np.int64)
    chunklen = (N + num_threads - 1) // num_threads

    chunks = []
    for i in range(num_threads):
        chunk = [
            N,
            task_indices[i * chunklen : (i + 1) * chunklen],
            dacs,
            displacements,
        ]
        chunks.append(chunk)

    threads = [
        threading.Thread(
            target=calculate_direction_autocorr_coeff_parallel_worker, args=c
        )
        for c in chunks
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    np.append([0], task_indices)
    dacs = dacs[task_indices]

    first_negative_index = find_first_negative_n(dacs)
    return task_indices[:first_negative_index], dacs[:first_negative_index]


# =====================================================


def calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
    displacements, num_threads=4
):
    N = displacements.shape[0]
    dacs = np.ones(N, dtype=np.float64)

    task_indices = np.arange(1, N, 30)  # np.linspace(1, N, num=N/30.0, dtype=np.int64)
    chunklen = (N + num_threads - 1) // num_threads

    chunks = []
    for i in range(num_threads):
        chunk = [
            N,
            task_indices[i * chunklen : (i + 1) * chunklen],
            dacs,
            displacements,
        ]
        chunks.append(chunk)

    threads = [
        threading.Thread(
            target=calculate_direction_autocorr_coeff_parallel_worker, args=c
        )
        for c in chunks
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    np.append([0], task_indices)
    dacs = dacs[task_indices]

    first_negative_index = find_first_negative_n(dacs)
    return task_indices[:first_negative_index], dacs[:first_negative_index]


# =====================================================


def estimate_persistence_time(timestep, indices, positive_autocorr_factors):
    ts = indices * timestep
    #    A = np.zeros((ts.shape[0], 2), dtype=np.float64)
    #    A[:, 0] = ts
    #    pt = -1./(np.linalg.lstsq(A, np.log(positive_das))[0][0])
    try:
        popt, pcov = scipio.curve_fit(
            lambda t, pt: np.exp(-1.0 * t / pt), ts, positive_autocorr_factors
        )
        pt = popt[0]
    except:
        pt = np.nan

    return pt, ts


# =====================================================


def calculate_mean_and_deviation(data):
    mean = np.average(data)
    deviation = np.sqrt(np.var(data))

    return mean, deviation


# =====================================================

# def analyze_chemotaxis_success(relevant_environment, storefile_path, rpt_number, source_x, source_y,
#                               chemotaxis_score_cutoff_radius):
#    num_cells = relevant_environment.num_cells
#
#    node_coords_per_tstep_per_cell = []
#    for ci in range(num_cells):
#        print(("    Analyzing cell {}...".format(ci)))
#
#        node_coords_per_tstep = hardio.get_node_coords_for_all_tsteps(ci, storefile_path)
#        node_coords_per_tstep_per_cell.append(copy.deepcopy(node_coords_per_tstep))
#
#    normalized_areas, normalized_cell_separations, cell_subgroups = calculate_normalized_group_area_and_average_cell_separation_over_time(
#        20,
#        num_cells,
#        node_coords_per_tstep_per_cell[0].shape[0],
#        storefile_path,
#        get_subgroups=False,
#    )
#
#    min_node_distance_from_source_per_tstep_per_cell = np.array([np.min(np.linalg.norm(node_coords_per_tstep - np.array([source_x, source_y]), axis=2), axis=1) for node_coords_per_tstep in node_coords_per_tstep_per_cell])
#
#    min_dist_tstep_per_cell = np.argmin(min_node_distance_from_source_per_tstep_per_cell, axis=1)
#    min_dist_per_cell = np.array([min_node_distance_from_source_per_tstep_per_cell[ci][min_idx] for ci, min_idx in enumerate(min_dist_tstep_per_cell)])
#    closest_cell = np.argmin(min_dist_per_cell)
#    min_dist_tstep = min_dist_tstep_per_cell[closest_cell]
#    #min_dist_check = np.min(np.linalg.norm(node_coords_per_tstep_per_cell[closest_cell][min_dist_tstep] - np.array([source_x, source_y]), axis=1))
#    min_distance = min_dist_per_cell[closest_cell]
#
#    if min_distance < chemotaxis_score_cutoff_radius:
#        return 1.0/normalized_areas[min_dist_tstep], min_distance
#    else:
#        return 0.0, min_distance

# def analyze_chemotaxis_success(
#    relevant_environment,
#    storefile_path,
#    rpt_number,
#    source_x,
#    source_y,
#    chemotaxis_score_cutoff_radius,
# ):
#    st = time.time()
#    print("Analyzing chemotaxis data...")
#    num_cells = relevant_environment.num_cells
#
#
#    distance_from_chemoattractant_source_per_timestep_per_cell_centroid = []
#    min_distance_and_timestep_per_cell = []
#    for ci in range(num_cells):
#        cell_centroids = calculate_cell_centroids_for_all_time(ci, storefile_path)*relevant_environment.cells_in_environment[ci].L/1e-6
#        if ci == 0:
#            num_timesteps = cell_centroids.shape[0]
#
#        distance_from_chemoattractant_source_per_timestep = np.linalg.norm(cell_centroids - np.array([source_x, source_y]), axis=1)
#
#        min_dist_timestep = np.argmin(distance_from_chemoattractant_source_per_timestep)
#        min_distance_and_timestep_per_cell.append((min_dist_timestep, distance_from_chemoattractant_source_per_timestep[min_dist_timestep]))
#
#        distance_from_chemoattractant_source_per_timestep_per_cell_centroid.append(copy.deepcopy(distance_from_chemoattractant_source_per_timestep))
#
#    sorted_min_distance_and_timestep_per_cell = sorted(min_distance_and_timestep_per_cell, key=lambda x: x[1])
#
#    min_dist_timestep_over_all_cells = sorted_min_distance_and_timestep_per_cell[0][0]
#
#    dist_of_cells_at_min_dist_timestep = [distance_from_chemoattractant_source_per_timestep_per_cell_centroid[i][min_dist_timestep_over_all_cells] for i in range(num_cells)]
#
#    scoring_function = lambda x: 1.0 - ((x**5)/(chemotaxis_score_cutoff_radius**5 + x**5))
#    normalized_areas, normalized_cell_separations, cell_subgroups = calculate_normalized_group_area_and_average_cell_separation_over_time(
#        20,
#        num_cells,
#        num_timesteps,
#        storefile_path,
#        get_subgroups=False,
#    )
#
#    cell_chemotaxis_scores = [scoring_function(d)/normalized_areas[min_dist_timestep_over_all_cells] for d in dist_of_cells_at_min_dist_timestep]
#
#    et = time.time()
#    print("Done. time taken = {} s".format(np.round(et - st, decimals=2)))
#    return cell_chemotaxis_scores, sorted_min_distance_and_timestep_per_cell[0][1]


@nb.jit(nopython=True)
def calculate_cluster_centroid_x_per_tstep_given_all_cell_centroids_per_tstep(
    all_cell_centroids_per_tstep,
):
    num_cells = all_cell_centroids_per_tstep.shape[0]
    num_tsteps = all_cell_centroids_per_tstep.shape[1]

    cluster_centroids_per_tstep = np.empty((num_tsteps, 2), dtype=np.float64)
    cell_centroids_at_tstep = np.empty((num_cells, 2), dtype=np.float64)

    for t in range(num_tsteps):
        cell_centroids_at_tstep = all_cell_centroids_per_tstep[:, t]
        cluster_centroids_per_tstep[t][0] = (
            np.sum(cell_centroids_at_tstep[:, 0]) / num_cells
        )
        cluster_centroids_per_tstep[t][1] = (
            np.sum(cell_centroids_at_tstep[:, 1]) / num_cells
        )

    return cluster_centroids_per_tstep


def calculate_normalized_group_area_and_average_cell_separation_at_tstep(
    cell_radius, num_cells, init_cell_centroids, cell_centroids_at_tstep
):
    if num_cells < 0:
        raise Exception("Negative number of cells given!")

    if num_cells == 0:
        return 0.0
    elif num_cells == 1:
        return 1.0
    elif num_cells == 2:
        init_distance_between_cells = np.linalg.norm(
            init_cell_centroids[0] - init_cell_centroids[1]
        )

        final_distance_between_cells = np.linalg.norm(
            cell_centroids_at_tstep[0] - cell_centroids_at_tstep[1]
        )

        return final_distance_between_cells / init_distance_between_cells
    else:
        init_delaunay_triangulation = space.Delaunay(init_cell_centroids)
        final_delaunay_triangulation = space.Delaunay(cell_centroids_at_tstep)

        init_simplices = init_cell_centroids[init_delaunay_triangulation.simplices]
        init_convex_hull_area = np.sum(
            [geometry.calculate_polygon_area(simplex) for simplex in init_simplices]
        )

        final_simplices = cell_centroids_at_tstep[
            final_delaunay_triangulation.simplices
        ]
        final_convex_hull_area = np.sum(
            [geometry.calculate_polygon_area(simplex) for simplex in final_simplices]
        )

        return final_convex_hull_area / init_convex_hull_area


def analyze_chemotaxis_success_at_cluster_level(
    relevant_environment,
    storefile_path,
    rpt_number,
    source_x,
    source_y,
    chemotaxis_score_cutoff_radius,
    T,
):
    st = time.time()
    print("=======================================")
    print("Analyzing chemotaxis data for repeat {}".format(rpt_number))
    num_cells = relevant_environment.num_cells

    all_cell_centroids_per_tstep = []
    for ci in range(num_cells):
        cell_centroids_per_tstep = (
            calculate_cell_centroids_for_all_time(ci, storefile_path)
            * relevant_environment.cells_in_environment[ci].L
            / 1e-6
        )

        if ci == 0:
            num_tsteps = cell_centroids_per_tstep.shape[0]

        all_cell_centroids_per_tstep.append(copy.deepcopy(cell_centroids_per_tstep))

    all_cell_centroids_per_tstep = np.array(all_cell_centroids_per_tstep)
    all_cell_straight_line_dist_to_end = np.array(
        [np.linalg.norm(x[num_tsteps] - x[0]) for x in all_cell_centroids_per_tstep]
    )
    all_cell_total_dist = np.array(
        [
            np.sum(np.linalg.norm(x[1:] - x[: (num_tsteps - 1)], axis=1))
            for x in all_cell_centroids_per_tstep
        ]
    )
    all_cell_persistence_ratios = (
        all_cell_straight_line_dist_to_end / all_cell_total_dist
    )
    cluster_centroids_per_tstep = calculate_cluster_centroid_x_per_tstep_given_all_cell_centroids_per_tstep(
        all_cell_centroids_per_tstep
    )
    cluster_displacements_per_tstep = (
        cluster_centroids_per_tstep[1:]
        - cluster_centroids_per_tstep[: (num_tsteps - 1)]
    )
    cluster_velocities_per_tstep = cluster_displacements_per_tstep / (T / 60.0)
    cluster_speeds_per_tstep = np.linalg.norm(cluster_velocities_per_tstep, axis=1)
    cluster_x_speeds_per_tstep = np.abs(cluster_velocities_per_tstep[:, 0])
    average_cluster_speed = np.average(cluster_speeds_per_tstep)
    average_cluster_x_speed = np.average(cluster_x_speeds_per_tstep)

    straight_line_dist_to_end = np.linalg.norm(
        cluster_centroids_per_tstep[num_tsteps] - cluster_centroids_per_tstep[0]
    )
    cluster_distances_per_tstep = np.linalg.norm(
        cluster_displacements_per_tstep, axis=1
    )
    total_dist = np.sum(cluster_distances_per_tstep)
    cluster_persistence_ratio = straight_line_dist_to_end / total_dist

    et = time.time()
    print("Done. time taken = {} s".format(np.round(et - st, decimals=2)))

    return (
        average_cluster_speed,
        average_cluster_x_speed,
        cluster_persistence_ratio,
        all_cell_persistence_ratios,
    )


def analyze_chemotactic_success_simple(
    relevant_environment,
    storefile_path,
    rpt_number,
    source_x,
    source_y,
    chemotaxis_score_cutoff_radius,
):
    num_cells = relevant_environment.num_cells

    all_node_coords = np.empty((0, 2), dtype=np.float64)
    for ci in range(num_cells):
        print(("    Analyzing cell {}...".format(ci)))

        node_coords_per_tstep = hardio.get_node_coords_for_all_tsteps(
            ci, storefile_path
        )

        for ni in range(node_coords_per_tstep.shape[1]):
            all_node_coords = np.append(
                all_node_coords, node_coords_per_tstep[:, ni, :], axis=0
            )
        # cell_centroids = calculate_cell_centroids_for_all_time(n, storefile_path)*relevant_environment.cells_in_environment[n].L/1e-6

        # all_cell_centroids = np.append(all_cell_centroids, cell_centroids, axis=0)

    distance_from_chemoattractant_source_per_timestep = np.linalg.norm(
        all_node_coords - np.array([source_x, source_y]), axis=1
    )

    min_distance = np.min(distance_from_chemoattractant_source_per_timestep)

    if min_distance < chemotaxis_score_cutoff_radius:
        return 1.0
    else:
        return 0.0


# =================================================================
@nb.jit(nopython=True)
def calculate_intercellular_distance_matrix(cell_centroid_coordinates):

    num_cells = cell_centroid_coordinates.shape[0]
    calculation_complete = np.zeros((num_cells, num_cells), dtype=np.int64)
    intercellular_distance_matrix = np.zeros((num_cells, num_cells), dtype=np.float64)

    for ci in range(num_cells):
        this_position = cell_centroid_coordinates[ci]
        for other_ci in range(num_cells):
            if ci == other_ci:
                continue
            else:
                if calculation_complete[ci][other_ci] != 1:
                    other_position = cell_centroid_coordinates[other_ci]
                    icd = geometry.calculate_dist_between_points_given_vectors(
                        this_position, other_position
                    )
                    intercellular_distance_matrix[ci][other_ci] = icd
                    intercellular_distance_matrix[other_ci][ci] = icd
                    calculation_complete[ci][other_ci] = 1
                    calculation_complete[other_ci][ci] = 1

    return intercellular_distance_matrix


@nb.jit(nopython=True)
def find_snapshot_index(unique_snapshot_timesteps, timestep):
    for i in range(unique_snapshot_timesteps.shape[0]):
        if unique_snapshot_timesteps[i] == timestep:
            return i


# @nb.jit(nopython=True)
# def calculate_velocity_alignments(time_deltas, distance_radii, all_cell_centroids_per_tstep, T, L):
#    num_drs = distance_radii.shape[0]
#    num_tds = time_deltas.shape[0]
#    num_timesteps = all_cell_centroids_per_tstep.shape[1]
#    num_cells = all_cell_centroids_per_tstep.shape[0]
#
#    num_unique_snapshots = 0
#    unique_snapshot_timesteps = np.zeros(num_timesteps, dtype=np.int64)
#    num_snapshots_per_td = np.zeros(num_tds, dtype=np.int64)
#    tds = np.zeros(num_tds, dtype=np.int64)
#    for xi in range(num_tds):
#        td = int(np.round(time_deltas[xi]*60.0/T))
#        tds[xi] = td
#        num_snapshots = int(num_timesteps/td)
#        num_snapshots_per_td[xi] = num_snapshots
#        initial_snapshot = int(td/2)
#
#        for yi in range(num_snapshots):
#            snapshot_timestep = initial_snapshot + yi*td
#
#            snapshot_is_unique = True
#            for zi in range(num_unique_snapshots):
#                if unique_snapshot_timesteps[zi] == snapshot_timestep:
#                    snapshot_is_unique = False
#
#            if snapshot_is_unique:
#                unique_snapshot_timesteps[num_unique_snapshots] = snapshot_timestep
#                num_unique_snapshots += 1
#
#    unique_snapshot_timesteps = unique_snapshot_timesteps[:num_unique_snapshots]
#    velocity_alignments = np.zeros((num_drs, num_tds, num_cells*num_cells*np.max(num_snapshots_per_td)), dtype=np.float64)
#    num_data_points_per_case = np.zeros((num_drs, num_tds), dtype=np.int64)
#
#    for xi in range(num_drs):
#        for yi in range(num_tds):
#            num_snapshots = num_snapshots_per_td[yi]
#            num_va_data = 0
#            va_data = np.zeros(num_cells*num_cells*num_snapshots, dtype=np.float64)
#            td = tds[yi]
#            for si in range(num_snapshots - 1):
#                et = int(td/2) + (si + 1)*td
#                cell_centroids_st = all_cell_centroids_per_tstep[:, et - td, :]
#                cell_centroids_et = all_cell_centroids_per_tstep[:, et, :]
#                cell_displacements = cell_centroids_et - cell_centroids_st
#                cell_directions = geometry.normalize_vectors(cell_displacements)
#
#                for focus_ci in range(num_cells):
#                    num_relevant_cells = 0
#                    relevant_cell_indices = np.zeros(num_cells, dtype=np.int64)
#                    focus_ci_direction = cell_directions[focus_ci]
#
#                    if focus_ci_direction[0] == np.nan or focus_ci_direction[1] == np.nan:
#                        continue
#
#                    for other_ci in range(num_cells):
#                        if other_ci == focus_ci:
#                            continue
#
#                        other_ci_direction = cell_directions[other_ci]
#                        if other_ci_direction[0] == np.nan or other_ci_direction[1] == np.nan:
#                            continue
#
#                        relevant_cell_indices[num_relevant_cells] = other_ci
#                        num_relevant_cells += 1
#
#                    for other_ci in relevant_cell_indices[:num_relevant_cells]:
#                        va_data[num_va_data] = np.dot(cell_directions[other_ci], focus_ci_direction)
#                        num_va_data += 1
#
#            velocity_alignments[xi][yi][:num_va_data] = va_data[:num_va_data]
#            num_data_points_per_case[xi][yi] = num_va_data
#
#
#    return velocity_alignments, num_data_points_per_case


@nb.jit(nopython=True)
def calculate_velocity_alignments(
    max_num_closest_neighbours,
    velocity_alignment_time_deltas,
    all_cell_centroids_per_tstep,
    T,
    L,
):
    num_mncs = max_num_closest_neighbours.shape[0]
    num_tds = velocity_alignment_time_deltas.shape[0]
    num_timesteps = all_cell_centroids_per_tstep.shape[1]
    num_cells = all_cell_centroids_per_tstep.shape[0]

    num_unique_snapshots = 0
    unique_snapshot_timesteps = np.zeros(num_timesteps, dtype=np.int64)
    num_snapshots_per_td = np.zeros(num_tds, dtype=np.int64)
    tds = np.zeros(num_tds, dtype=np.int64)
    for xi in range(num_tds):
        td = int(np.round(velocity_alignment_time_deltas[xi] * 60.0 / T))
        tds[xi] = td
        num_snapshots = int(num_timesteps / td)
        num_snapshots_per_td[xi] = num_snapshots
        initial_snapshot = int(td / 2)

        for yi in range(num_snapshots):
            snapshot_timestep = initial_snapshot + yi * td

            snapshot_is_unique = True
            for zi in range(num_unique_snapshots):
                if unique_snapshot_timesteps[zi] == snapshot_timestep:
                    snapshot_is_unique = False

            if snapshot_is_unique:
                unique_snapshot_timesteps[num_unique_snapshots] = snapshot_timestep
                num_unique_snapshots += 1

    unique_snapshot_timesteps = unique_snapshot_timesteps[:num_unique_snapshots]
    intercellular_distance_matrices = np.zeros(
        (num_unique_snapshots, num_cells, num_cells), dtype=np.float64
    )
    for zi in range(num_unique_snapshots):
        timestep = unique_snapshot_timesteps[zi]
        cell_centroids_at_tstep = all_cell_centroids_per_tstep[:, timestep, :]
        intercellular_distance_matrices[zi] = calculate_intercellular_distance_matrix(
            cell_centroids_at_tstep
        )

    velocity_alignments = np.zeros(
        (
            num_mncs,
            num_tds,
            num_cells
            * np.max(max_num_closest_neighbours)
            * np.max(num_snapshots_per_td),
        ),
        dtype=np.float64,
    )
    num_data_points_per_case = np.zeros((num_mncs, num_tds), dtype=np.int64)

    for xi in range(num_mncs):
        num_closest_neighbours = max_num_closest_neighbours[xi]
        for yi in range(num_tds):
            num_snapshots = num_snapshots_per_td[yi]
            num_va_data = 0
            va_data = np.zeros(
                num_cells * num_closest_neighbours * num_snapshots, dtype=np.float64
            )
            td = tds[yi]
            for si in range(num_snapshots - 1):
                et = int(td / 2) + (si + 1) * td
                cell_centroids_st = all_cell_centroids_per_tstep[:, et - td, :]
                cell_centroids_et = all_cell_centroids_per_tstep[:, et, :]
                cell_displacements = cell_centroids_et - cell_centroids_st
                cell_directions = geometry.normalize_vectors(cell_displacements)
                snapshot_index = find_snapshot_index(unique_snapshot_timesteps, et)
                relevant_intercellular_distance_matrix = intercellular_distance_matrices[
                    int(snapshot_index)
                ]

                for focus_ci in range(num_cells):
                    num_relevant_cells = 0
                    focus_ci_direction = cell_directions[focus_ci]

                    if (
                        focus_ci_direction[0] == np.nan
                        or focus_ci_direction[1] == np.nan
                    ):
                        continue

                    relevant_cell_indices = np.zeros(6, dtype=np.int64)
                    relevant_intercellular_distances = relevant_intercellular_distance_matrix[
                        focus_ci
                    ]
                    cell_indices_sorted_by_closest = np.argsort(
                        relevant_intercellular_distances
                    )

                    for other_ci in cell_indices_sorted_by_closest:
                        if other_ci == focus_ci:
                            continue

                        other_ci_direction = cell_directions[other_ci]
                        if (
                            other_ci_direction[0] == np.nan
                            or other_ci_direction[1] == np.nan
                        ):
                            continue

                        relevant_cell_indices[num_relevant_cells] = other_ci
                        num_relevant_cells += 1

                        if num_relevant_cells == num_closest_neighbours:
                            break

                    for other_ci in relevant_cell_indices[:num_relevant_cells]:
                        va_data[num_va_data] = np.dot(
                            cell_directions[other_ci], focus_ci_direction
                        )
                        num_va_data += 1

            velocity_alignments[xi][yi][:num_va_data] = va_data[:num_va_data]
            num_data_points_per_case[xi][yi] = num_va_data

    return velocity_alignments, num_data_points_per_case


@nb.jit(nopython=True)
def determine_intercellular_interactions_between_cells(
    all_cell_node_positions, all_cell_centroids, cil_cutoff, coa_cutoff
):
    num_cells = all_cell_node_positions.shape[0]
    calculation_tracker = np.zeros((num_cells, num_cells), dtype=np.int64)
    cil_interaction_matrix = np.zeros((num_cells, num_cells), dtype=np.int64)
    coa_only_interaction_matrix = np.zeros((num_cells, num_cells), dtype=np.int64)

    cell_bounding_boxes = np.zeros((num_cells, 4), dtype=np.float64)
    for ci in range(num_cells):
        x0, x1, y0, y1 = geometry.calculate_polygon_bounding_box(
            all_cell_node_positions[ci]
        )

    for ci in range(num_cells):
        this_cell_centroid = all_cell_centroids[ci]
        for other_ci in range(num_cells):
            if ci == other_ci:
                continue
            else:
                if calculation_tracker[ci][other_ci] == 0:
                    dist_between_centroids = geometry.calculate_dist_between_points_given_vectors(
                        this_cell_centroid, all_cell_centroids[other_ci]
                    )

                    if dist_between_centroids < cil_cutoff:
                        cil_interaction_matrix[ci][other_ci] = 1
                        cil_interaction_matrix[other_ci][ci] = 1
                    else:
                        if dist_between_centroids < coa_cutoff:
                            coa_interaction_result = 1
                            for xci in range(num_cells):
                                if xci == ci or xci == other_ci:
                                    continue
                                else:
                                    x0, x1, y0, y1 = cell_bounding_boxes[xci]
                                    check_result = geometry.check_if_line_segment_intersects_box(
                                        this_cell_centroid,
                                        all_cell_centroids[other_ci],
                                        x0,
                                        x1,
                                        y0,
                                        y1,
                                    )

                                    if check_result == 1:
                                        coa_interaction_result = 0
                                        break

                            coa_only_interaction_matrix[ci][
                                other_ci
                            ] = coa_interaction_result
                            coa_only_interaction_matrix[other_ci][
                                ci
                            ] = coa_interaction_result

                    calculation_tracker[ci][other_ci] = 1
                    calculation_tracker[other_ci][ci] = 1

    return cil_interaction_matrix, coa_only_interaction_matrix


@nb.jit(nopython=True)
def determine_cil_interactions_between_cells(all_cell_centroids, cil_cutoff):
    num_cells = all_cell_centroids.shape[0]
    calculation_tracker = np.zeros((num_cells, num_cells), dtype=np.int64)
    cil_interaction_matrix = np.zeros((num_cells, num_cells), dtype=np.int64)

    for ci in range(num_cells):
        this_cell_centroid = all_cell_centroids[ci]
        for other_ci in range(num_cells):
            if ci == other_ci:
                continue
            else:
                if calculation_tracker[ci][other_ci] == 0:
                    dist_between_centroids = geometry.calculate_dist_between_points_given_vectors(
                        this_cell_centroid, all_cell_centroids[other_ci]
                    )

                    if dist_between_centroids < cil_cutoff:
                        cil_interaction_matrix[ci][other_ci] = 1
                        cil_interaction_matrix[other_ci][ci] = 1

                    calculation_tracker[ci][other_ci] = 1
                    calculation_tracker[other_ci][ci] = 1

    return cil_interaction_matrix


@nb.jit(nopython=True)
def calculate_frequency_of_cil_interaction_over_simulation(
    all_cell_centroids_per_tstep, cil_cutoff, T
):
    num_timesteps = all_cell_centroids_per_tstep.shape[1]
    num_cells = all_cell_centroids_per_tstep.shape[0]
    simulation_length = num_timesteps * T

    interaction_frequency_per_cell = np.zeros(num_cells, dtype=np.float64)
    cil_interaction_matrices_per_tstep = np.zeros(
        (num_timesteps, num_cells, num_cells), dtype=np.int64
    )

    for ti in range(num_timesteps):
        all_cell_centroids = all_cell_centroids_per_tstep[:, ti, :]
        cil_interaction_matrices_per_tstep[
            ti
        ] = determine_cil_interactions_between_cells(all_cell_centroids, cil_cutoff)

    for ci in range(num_cells):
        num_contacts = 0
        contact_tracker = np.zeros(num_cells, dtype=np.int64)
        for ti in range(num_timesteps):
            relevant_contact_info = cil_interaction_matrices_per_tstep[ti][ci]
            for oci in range(num_cells):
                if oci == ci:
                    continue
                else:
                    if relevant_contact_info[oci] == contact_tracker[oci]:
                        continue
                    elif relevant_contact_info[oci] == 1 and contact_tracker[oci] == 0:
                        num_contacts += 1
                        contact_tracker[oci] = 1
                    elif relevant_contact_info[oci] == 0 and contact_tracker[oci] == 1:
                        contact_tracker[oci] = 0

        interaction_frequency_per_cell[ci] = num_contacts / simulation_length

    return interaction_frequency_per_cell


@nb.jit(nopython=True)
def calculate_num_interactions_per_cell_per_tstep(
    all_cell_node_positions_per_tstep,
    all_cell_centroids_per_tstep,
    cil_cutoff,
    coa_cutoff,
    delta_tsteps_between_snapshots,
):
    num_timesteps = all_cell_centroids_per_tstep.shape[1]
    num_cells = all_cell_centroids_per_tstep.shape[0]

    relevant_tsteps = np.arange(0, num_timesteps, delta_tsteps_between_snapshots)
    num_cil_interactions_per_cell_per_snapshot = np.zeros(
        (relevant_tsteps.shape[0], num_cells), dtype=np.int64
    )
    num_coa_only_interactions_per_cell_per_snapshot = np.zeros(
        (relevant_tsteps.shape[0], num_cells), dtype=np.int64
    )

    for xti in range(relevant_tsteps.shape[0]):
        ti = relevant_tsteps[xti]
        all_cell_node_positions = all_cell_node_positions_per_tstep[:, ti, :, :]
        all_cell_centroids = all_cell_centroids_per_tstep[:, ti, :]
        (
            this_tstep_cil_interaction_matrix,
            this_tstep_coa_interaction_matrix,
        ) = determine_intercellular_interactions_between_cells(
            all_cell_node_positions, all_cell_centroids, cil_cutoff, coa_cutoff
        )

        for ci in range(num_cells):
            num_cil_interactions = 0
            num_coa_only_interactions = 0
            relevant_cil_interaction_data = this_tstep_cil_interaction_matrix[ci]
            relevant_coa_interaction_data = this_tstep_coa_interaction_matrix[ci]

            for other_ci in range(num_cells):
                if other_ci == ci:
                    continue
                else:
                    if relevant_cil_interaction_data[other_ci] == 1:
                        num_cil_interactions += 1
                    else:
                        if relevant_coa_interaction_data[other_ci] == 1:
                            num_coa_only_interactions += 1

            num_cil_interactions_per_cell_per_snapshot[xti][ci] = num_cil_interactions
            num_coa_only_interactions_per_cell_per_snapshot[xti][
                ci
            ] = num_coa_only_interactions

    num_interactions_per_cell_per_snapshot = (
        num_cil_interactions_per_cell_per_snapshot
        + num_coa_only_interactions_per_cell_per_snapshot
    )

    return (
        num_interactions_per_cell_per_snapshot,
        num_cil_interactions_per_cell_per_snapshot,
        num_coa_only_interactions_per_cell_per_snapshot,
    )


@nb.jit(nopython=True)
def find_index_of_max_element_in_velocity_alignment_matrix(
    num_x, num_y, va_matrix, num_va_data_points
):
    average_va_matrix = np.zeros((num_x, num_y), dtype=np.float64)
    for xi in range(num_x):
        for yi in range(num_y):
            num_dpoints = num_va_data_points[xi][yi]

            if num_dpoints > 0:
                average_va_matrix[xi][yi] = (
                    np.sum(va_matrix[xi][yi][:num_dpoints]) / num_dpoints
                )
            else:
                average_va_matrix[xi][yi] = 0.0

    va_max = 0.0
    va_max_xi = 0
    va_max_yi = 0

    for xi in range(num_x):
        for yi in range(num_y):
            if average_va_matrix[xi][yi] > va_max:
                va_max_xi = xi
                va_max_yi = yi

    return va_max, va_max_xi, va_max_yi


@nb.jit(nopython=True)
def determine_velocity_alignment_max_radii_and_min_times_over_all_repeats(
    velocity_alignment_radii,
    velocity_alignment_times,
    velocity_alignments_per_repeat,
    num_velocity_alignment_data_points_per_repeat,
):
    num_r = velocity_alignment_radii.shape[0]
    num_t = velocity_alignment_times.shape[0]
    num_repeats = velocity_alignments_per_repeat.shape[0]
    max_va_rs = np.zeros(num_repeats, dtype=np.float64)
    max_va_ts = np.zeros(num_repeats, dtype=np.float64)
    max_vas = np.zeros(num_repeats, dtype=np.float64)

    for nr in range(num_repeats):
        va_max, ri, ti = find_index_of_max_element_in_velocity_alignment_matrix(
            num_r,
            num_t,
            velocity_alignments_per_repeat[nr],
            num_velocity_alignment_data_points_per_repeat[nr],
        )
        max_vas[nr] = va_max
        max_va_rs[nr] = velocity_alignment_radii[ri]
        max_va_ts[nr] = velocity_alignment_times[ti]

    return max_vas, max_va_rs, max_va_ts
