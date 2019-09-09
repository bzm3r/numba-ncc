# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:21:52 2015

@author: brian
"""

# from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import core.utilities as cu
import os
from . import colors
import core.geometry as geometry
import core.hardio as hardio
from matplotlib import cm
import matplotlib.patches as mpatch
import numba as nb
from matplotlib import gridspec as mgs
from matplotlib import ticker as ticker
import datetime
import copy

plt.ioff()


def set_fontsize(fontsize):
    matplotlib.rcParams.update({"font.size": fontsize})


def write_lines_to_file(fp, lines):
    with open(fp, "w+") as f:
        f.writelines(lines)


def show_or_save_fig(
    fig, figsize, save_dir, base_name, extra_label, figtype="eps", bbox_inches="tight"
):
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(*figsize)
        if extra_label != "":
            extra_label = "_{}".format(extra_label)

        if figtype == "eps":
            save_path = os.path.join(
                save_dir, "{}{}".format(base_name, extra_label) + ".eps"
            )
            print("save_path: ", save_path)
            fig.savefig(
                save_path, format="eps", forward=True, bbox_inches="tight", dpi=1000
            )
            plt.close(fig)
            plt.close("all")
        elif figtype == "png":
            save_path = os.path.join(
                save_dir, "{}{}".format(base_name, extra_label) + ".png"
            )
            print("save_path: ", save_path)
            if bbox_inches == "tight":
                fig.savefig(save_path, forward=True, bbox_inches="tight")
            else:
                fig.savefig(save_path, forward=True)
            plt.close(fig)
            plt.close("all")


# =============================================================================


def add_to_general_data_structure(general_data_structure, key_value_tuples):
    if general_data_structure != None:
        if type(general_data_structure) != dict:
            raise Exception(
                "general_data_structure is not dict, instead: {}".format(
                    type(general_data_structure)
                )
            )
        else:
            general_data_structure.update(key_value_tuples)


# =============================================================================


def graph_group_area_and_cell_separation_over_time_and_determine_subgroups(
    num_cells,
    num_nodes,
    num_timepoints,
    T,
    storefile_path,
    save_dir=None,
    fontsize=22,
    general_data_structure=None,
    graph_group_centroid_splits=False,
):
    set_fontsize(fontsize)
    normalized_areas, normalized_cell_separations, cell_subgroups = cu.calculate_normalized_group_area_and_average_cell_separation_over_time(
        20,
        num_cells,
        num_timepoints,
        storefile_path,
        get_subgroups=graph_group_centroid_splits,
    )
    group_aspect_ratios = cu.calculate_group_aspect_ratio_over_time(
        num_cells, num_nodes, num_timepoints, storefile_path
    )
    # normalized_areas_new = cu.calculate_normalized_group_area_over_time(num_cells, num_timepoints, storefile_path)
    timepoints = np.arange(normalized_areas.shape[0]) * T

    fig_A, ax_A = plt.subplots()

    ax_A.plot(timepoints, normalized_areas)
    ax_A.set_ylabel("$A/A_0$")
    ax_A.set_xlabel("t (min.)")

    # Put a legend to the right of the current axis
    ax_A.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    ax_A.grid(which="both")
    mean, deviation = cu.calculate_mean_and_deviation(normalized_areas)
    if save_dir != None:
        write_lines_to_file(
            os.path.join(save_dir, "group_area_data.txt"),
            ["group area (mean, deviation): {}, {}".format(mean, deviation)],
        )
    else:
        ax_A.set_title("mean = {}, deviation = {}".format(mean, deviation))
    add_to_general_data_structure(
        general_data_structure,
        [("group_area_mean", mean), ("group_area_deviation", deviation)],
    )

    fig_S, ax_S = plt.subplots()

    ax_S.plot(timepoints, normalized_cell_separations)
    ax_S.set_ylabel("$S/S_0$")
    ax_S.set_xlabel("t (min.)")
    mean, deviation = cu.calculate_mean_and_deviation(normalized_cell_separations)
    if save_dir != None:
        write_lines_to_file(
            os.path.join(save_dir, "group_separation_data.txt"),
            ["group area (mean, deviation): {}, {}".format(mean, deviation)],
        )
    else:
        ax_S.set_title("mean = {}, deviation = {}".format(mean, deviation))
    add_to_general_data_structure(
        general_data_structure,
        [("cell_separation_mean", mean), ("cell_separation_deviation", deviation)],
    )

    fig_R, ax_R = plt.subplots()

    ax_R.plot(timepoints, group_aspect_ratios)
    ax_R.set_ylabel("$R = W/H$")
    ax_R.set_xlabel("t (min.)")
    mean, deviation = cu.calculate_mean_and_deviation(group_aspect_ratios)
    if save_dir != None:
        write_lines_to_file(
            os.path.join(save_dir, "group_ratio_data.txt"),
            ["group area (mean, deviation): {}, {}".format(mean, deviation)],
        )
    else:
        ax_R.set_title("mean = {}, deviation = {}".format(mean, deviation))
    # ax_R.set_title("mean = {}, deviation = {}".format(mean, deviation))
    add_to_general_data_structure(
        general_data_structure,
        [("group_aspect_ratio_mean", mean), ("group_aspect_ratio_deviaion", deviation)],
    )

    if save_dir == None:
        plt.show()
    else:
        for save_name, fig, ax in [
            ("group_area", fig_A, ax_A),
            ("cell_separation", fig_S, ax_S),
            ("group_aspect_ratio", fig_R, ax_R),
        ]:
            show_or_save_fig(fig, (6, 4), save_dir, save_name, "")

    if (
        cell_subgroups != None
        and len(cell_subgroups) != 0
        and None not in cell_subgroups
    ):
        add_to_general_data_structure(
            general_data_structure, [("cell_subgroups", cell_subgroups)]
        )

    return general_data_structure


# =============================================================================


@nb.jit(nopython=True)
def find_approximate_transient_end(
    group_x_velocities, average_group_x_velocity, next_timesteps_window=60
):
    num_timepoints = group_x_velocities.shape[0]
    max_velocity = np.max(group_x_velocities)
    possible_endpoint = 0

    # possible_endpoints_found = 0
    for i in range(num_timepoints - 1):
        ti = i + 1
        velocities_until = group_x_velocities[:ti]
        average_velocities_until = np.sum(velocities_until) / velocities_until.shape[0]

        if (
            group_x_velocities[ti] < 0.1 * max_velocity
            and average_velocities_until < 0.1 * max_velocity
        ):
            possible_endpoint = ti

    return possible_endpoint


# =============================================================================


def graph_group_centroid_drift(
    T,
    time_unit,
    group_centroid_per_tstep,
    relative_all_cell_centroids_per_tstep,
    save_dir,
    save_name,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    timepoints = np.arange(group_centroid_per_tstep.shape[0]) * T
    relative_group_centroid_per_tstep = (
        group_centroid_per_tstep - group_centroid_per_tstep[0]
    )
    relative_group_centroid_x_coords = relative_group_centroid_per_tstep[:, 0]
    relative_group_centroid_y_coords = relative_group_centroid_per_tstep[:, 1]

    fig, ax = plt.subplots()
    fig_clean, ax_clean = plt.subplots()

    # A = np.vstack([timepoints, np.zeros(len(timepoints))]).T
    # delta_t = timepoints[1] - timepoints[0]
    group_x_velocities = (
        relative_group_centroid_x_coords[1:] - relative_group_centroid_x_coords[:-1]
    ) / (timepoints[1:] - timepoints[:-1])
    average_group_x_velocity = np.average(group_x_velocities)

    transient_end_index = find_approximate_transient_end(
        group_x_velocities, average_group_x_velocity, next_timesteps_window=60
    )
    # fit_group_x_velocity, c = np.linalg.lstsq(A, group_centroid_x_coords)[0]
    ax.plot(timepoints, relative_group_centroid_x_coords, label="x-coord", color="b")
    ax_clean.plot(timepoints, group_centroid_per_tstep[:, 0], label="x-coord")
    ax.plot(timepoints, relative_group_centroid_y_coords, label="y-coord", color="g")
    ax.plot(
        timepoints,
        average_group_x_velocity * timepoints,
        label="velocity ({} $\mu m$)".format(average_group_x_velocity),
        color="r",
    )
    write_lines_to_file(
        os.path.join(save_dir, "group_" + save_name),
        ["velocity ({} $\mu m$)\n".format(average_group_x_velocity)],
    )
    # ax.axvline(x=timepoints[transient_end_index], color='m', label='approximate transient end')
    ax.set_ylabel("$X_c$ ($\mu m$)")
    ax.set_xlabel("t (min.)")

    ax_clean.set_ylabel("$X_c$ ($\mu m$)")
    ax_clean.set_xlabel("t (min.)")

    # Put a legend to the right of the current axis
    # ax.legend(loc='best')
    ax.grid(which="both")

    if save_dir == None or save_name == None:
        plt.show()
    else:
        show_or_save_fig(fig, (6, 4), save_dir, "group_" + save_name, "")
        show_or_save_fig(fig_clean, (6, 4), save_dir, "group_" + save_name, "clean")

    group_velocities = cu.calculate_velocities(relative_group_centroid_per_tstep, T)
    group_x_speeds = np.abs(group_velocities[:, 0])
    group_y_speeds = np.abs(group_velocities[:, 0])
    group_speeds = np.linalg.norm(group_velocities, axis=1)

    add_to_general_data_structure(
        general_data_structure, [("group_speeds", group_speeds)]
    )
    add_to_general_data_structure(
        general_data_structure, [("average_group_x_speed", np.average(group_x_speeds))]
    )
    add_to_general_data_structure(
        general_data_structure, [("average_group_y_speed", np.average(group_y_speeds))]
    )
    add_to_general_data_structure(
        general_data_structure, [("fit_group_x_velocity", average_group_x_velocity)]
    )
    add_to_general_data_structure(
        general_data_structure, [("transient_end", timepoints[transient_end_index])]
    )

    if "cell_subgroups" in list(general_data_structure.keys()):
        cell_subgroups = general_data_structure["cell_subgroups"]
        if cell_subgroups != None:
            fig, ax = plt.subplots()

            multigroup_timesteps = []
            multigroup_data = []
            single_group_data = []
            group_split_connector_timesteps = []
            group_split_connector_data = []
            group_merge_connector_timesteps = []
            group_merge_connector_data = []
            last_num_subgroups = -1
            last_subgroup_centroids = []
            for ti, subgroups in enumerate(cell_subgroups):
                num_subgroups = len(subgroups)
                relevant_all_cell_centroids = relative_all_cell_centroids_per_tstep[
                    :, ti, :
                ]

                if num_subgroups > 1:
                    single_group_data.append(np.nan)

                    subgroup_centroids = []
                    for cell_indices in subgroups:
                        relevant_cell_centroids = relevant_all_cell_centroids[
                            cell_indices
                        ]
                        multigroup_timesteps.append(ti)
                        subgroup_centroid = geometry.calculate_cluster_centroid(
                            relevant_cell_centroids
                        )[0]
                        multigroup_data.append(subgroup_centroid)
                        subgroup_centroids.append(subgroup_centroid)
                        # multigroup_data.append(np.min(relevant_cell_centroids[:,0]))
                else:
                    group_centroid = geometry.calculate_cluster_centroid(
                        relevant_all_cell_centroids
                    )[0]
                    subgroup_centroids = [group_centroid]
                    single_group_data.append(group_centroid)
                    # single_group_data.append(np.min(relevant_all_cell_centroids[:,0]))
                    multigroup_timesteps.append(ti)
                    multigroup_data.append(np.nan)

                if ti > 0:
                    if last_num_subgroups != num_subgroups:
                        last_subgroups = cell_subgroups[ti - 1]
                        connected_subgroups = []
                        for i, cell_indices in enumerate(subgroups):
                            for j, last_cell_indices in enumerate(last_subgroups):
                                in_check = [
                                    (x in last_cell_indices) for x in cell_indices
                                ]
                                if np.all(in_check):
                                    connected_subgroups.append((i, j, "split"))
                                elif np.any(in_check):
                                    connected_subgroups.append((i, j, "merge"))

                        for p in connected_subgroups:
                            if p[2] == "split":
                                group_split_connector_timesteps += [
                                    timepoints[ti - 1],
                                    timepoints[ti],
                                    np.nan,
                                ]
                                group_split_connector_data += [
                                    last_subgroup_centroids[p[1]],
                                    subgroup_centroids[p[0]],
                                    np.nan,
                                ]
                            else:
                                group_merge_connector_timesteps += [
                                    timepoints[ti - 1],
                                    timepoints[ti],
                                    np.nan,
                                ]
                                group_merge_connector_data += [
                                    last_subgroup_centroids[p[1]],
                                    subgroup_centroids[p[0]],
                                    np.nan,
                                ]

                last_num_subgroups = num_subgroups
                last_subgroup_centroids = subgroup_centroids

            #            ax.plot(timepoints, single_group_data, color='r', rasterized=True)
            #            ax.plot(timepoints[multigroup_timesteps], multigroup_data, color='b', ls='', marker='.', markersize=1, rasterized=True)
            ax.plot(timepoints, single_group_data, color="r")
            ax.plot(
                timepoints[multigroup_timesteps],
                multigroup_data,
                color="b",
                ls="",
                marker=".",
                markersize=1,
            )
            #            ax.plot(group_split_connector_timesteps, group_split_connector_data, color='r')
            #            ax.plot(group_merge_connector_timesteps, group_merge_connector_data, color='g')
            ax.set_ylabel("$X_c$ ($\mu m$)")
            ax.set_xlabel("t (min.)")
            ax.grid(which="both")

            if save_dir == None or save_name == None:
                plt.show()
            else:
                show_or_save_fig(fig, (6, 4), save_dir, "group_split_" + save_name, "")
                show_or_save_fig(
                    fig, (6, 4), save_dir, "group_split_" + save_name, "", figtype="png"
                )

    return general_data_structure


# =============================================================================


def graph_centroid_related_data(
    skip_dynamics_flags,
    num_cells,
    num_timepoints,
    T,
    time_unit,
    cell_Ls,
    storefile_path,
    save_dir=None,
    save_name=None,
    max_tstep=None,
    make_group_centroid_drift_graph=True,
    fontsize=22,
    general_data_structure=None,
):
    # assuming that num_timepoints, T is same for all cells
    set_fontsize(fontsize)
    if max_tstep == None:
        max_tstep = num_timepoints

    all_cell_centroids_per_tstep = np.zeros((num_cells, max_tstep, 2), dtype=np.float64)

    # ------------------------

    for ci in range(num_cells):
        cell_centroids_per_tstep = (
            cu.calculate_cell_centroids_until_tstep(ci, max_tstep, storefile_path)
            * cell_Ls[ci]
        )

        all_cell_centroids_per_tstep[ci, :, :] = cell_centroids_per_tstep

    # ------------------------
    add_to_general_data_structure(
        general_data_structure,
        [("all_cell_centroids_per_tstep", all_cell_centroids_per_tstep)],
    )

    group_centroid_per_tstep = np.zeros((max_tstep, 2), dtype=np.float64)
    for tstep in range(max_tstep):
        cell_centroids = all_cell_centroids_per_tstep[:, tstep, :]
        group_centroid_per_tstep[tstep] = geometry.calculate_cluster_centroid(
            cell_centroids
        )
        
    group_velocity_per_tstep = group_centroid_per_tstep[1:] - group_centroid_per_tstep[:-1]
    group_direction_per_tstep = group_velocity_per_tstep/np.linalg.norm(group_velocity_per_tstep, axis=1)[:, np.newaxis]
    all_cell_velocities_per_tstep = np.array([cell_centroids_per_tstep[1:] - cell_centroids_per_tstep[:-1] for cell_centroids_per_tstep in all_cell_centroids_per_tstep])
    all_cell_directions_per_tstep = np.array([cell_velocities_per_tstep/np.linalg.norm(cell_velocities_per_tstep, axis=1)[:, np.newaxis] for cell_velocities_per_tstep in all_cell_velocities_per_tstep])
    all_cell_velocity_alignment_per_tstep = [[np.dot(cell_direction, group_direction) for cell_direction in all_cell_directions_per_tstep[:, t]] for t, group_direction in enumerate(group_direction_per_tstep)]
    
    add_to_general_data_structure(
        general_data_structure, [("group_centroid_per_tstep", group_centroid_per_tstep)]
    )
    add_to_general_data_structure(
        general_data_structure, [("all_cell_velocity_alignment_per_tstep", all_cell_velocity_alignment_per_tstep)]
    )
    init_group_centroid_per_tstep = group_centroid_per_tstep[0]
    relative_group_centroid_per_tstep = (
        group_centroid_per_tstep - init_group_centroid_per_tstep
    )
    relative_all_cell_centroids_per_tstep = (
        all_cell_centroids_per_tstep - init_group_centroid_per_tstep
    )

    group_centroid_displacements = (
        relative_group_centroid_per_tstep[1:] - relative_group_centroid_per_tstep[:-1]
    )
    all_cell_centroid_displacements = np.zeros(
        (num_cells, max_tstep - 1, 2), dtype=np.float64
    )
    for ci in range(num_cells):
        all_cell_centroid_displacements[ci, :, :] = (
            relative_all_cell_centroids_per_tstep[ci, 1:, :]
            - relative_all_cell_centroids_per_tstep[ci, :-1, :]
        )

    group_positive_ns, group_positive_das = cu.calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
        group_centroid_displacements
    )
    group_persistence_time, group_positive_ts = cu.estimate_persistence_time(
        T, group_positive_ns, group_positive_das
    )
    group_persistence_time = np.round(group_persistence_time, 0)

    add_to_general_data_structure(
        general_data_structure, [("group_persistence_time", group_persistence_time)]
    )

    positive_ts_per_cell = []
    positive_das_per_cell = []
    all_cell_persistence_times = []

    for ci in range(all_cell_centroid_displacements.shape[0]):
        this_cell_centroid_displacements = all_cell_centroid_displacements[ci, :, :]
        this_cell_positive_ns, this_cell_positive_das = cu.calculate_direction_autocorr_coeffs_for_persistence_time_parallel(
            this_cell_centroid_displacements
        )
        this_cell_persistence_time, this_cell_positive_ts = cu.estimate_persistence_time(
            T, this_cell_positive_ns, this_cell_positive_das
        )
        this_cell_persistence_time = np.round(this_cell_persistence_time, 0)

        positive_ts_per_cell.append(this_cell_positive_ts)
        positive_das_per_cell.append(this_cell_positive_das)
        all_cell_persistence_times.append(this_cell_persistence_time)

        if save_dir != None and skip_dynamics_flags[ci] == False:
            fig, ax = plt.subplots()

            graph_title = "persistence time: {} {}".format(
                np.round(this_cell_persistence_time, decimals=0), time_unit
            )
            ax.set_title(graph_title)
            ax.plot(
                this_cell_positive_ts, this_cell_positive_das, color="g", marker="."
            )
            ax.plot(
                this_cell_positive_ts,
                np.exp(-1 * this_cell_positive_ts / this_cell_persistence_time),
                color="r",
                marker=".",
            )

            this_cell_save_dir = os.path.join(save_dir, "cell_{}".format(ci))
            show_or_save_fig(
                fig, (12, 8), this_cell_save_dir, "persistence_time_estimation", ""
            )

    add_to_general_data_structure(
        general_data_structure,
        [("all_cell_persistence_times", all_cell_persistence_times)],
    )

    if save_dir != None:
        fig, ax = plt.subplots()
        ax.set_title(
            "persistence time: {}".format(np.round(group_persistence_time, decimals=0))
        )
        ax.plot(group_positive_ts, group_positive_das, color="g", marker=".")
        ax.plot(
            group_positive_ts,
            np.exp(-1 * group_positive_ts / group_persistence_time),
            color="r",
            marker=".",
        )

        show_or_save_fig(
            fig, (12, 8), save_dir, "group_persistence_time_estimation", ""
        )

    # ------------------------

    fig, ax = plt.subplots()

    # ------------------------

    min_x_data_lim = np.min(relative_all_cell_centroids_per_tstep[:, :, 0])
    max_x_data_lim = np.max(relative_all_cell_centroids_per_tstep[:, :, 0])
    delta_x = np.abs(min_x_data_lim - max_x_data_lim)
    max_y_data_lim = 1.2 * np.max(
        np.abs(relative_all_cell_centroids_per_tstep[:, :, 1])
    )
    ax.set_xlim(min_x_data_lim - 0.1 * delta_x, max_x_data_lim + 0.1 * delta_x)
    if 2 * max_y_data_lim < 0.25 * (1.2 * delta_x):
        max_y_data_lim = 0.5 * 1.2 * delta_x
    ax.set_ylim(-1 * max_y_data_lim, max_y_data_lim)
    ax.set_aspect("equal")

    group_net_displacement = (
        relative_group_centroid_per_tstep[-1] - relative_group_centroid_per_tstep[0]
    )
    group_net_displacement_mag = np.linalg.norm(group_net_displacement)
    group_net_distance = np.sum(
        np.linalg.norm(
            relative_group_centroid_per_tstep[1:]
            - relative_group_centroid_per_tstep[:-1],
            axis=1,
        )
    )
    group_persistence_ratio = np.round(
        group_net_displacement_mag / group_net_distance, 4
    )

    add_to_general_data_structure(
        general_data_structure, [("group_persistence_ratio", group_persistence_ratio)]
    )

    cell_persistence_ratios = []
    for ci in range(num_cells):
        ccs = relative_all_cell_centroids_per_tstep[ci, :, :]
        net_displacement = ccs[-1] - ccs[0]
        net_displacement_mag = np.linalg.norm(net_displacement)
        net_distance = np.sum(np.linalg.norm(ccs[1:] - ccs[:-1], axis=-1))
        persistence_ratio = net_displacement_mag / net_distance
        cell_persistence_ratios.append(persistence_ratio)

        ax.plot(ccs[:, 0], ccs[:, 1], marker=None, color=colors.color_list20[ci % 20])
        # ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list20[ci%20], label='cell {}, pers.={}'.format(ci, persistence))

    add_to_general_data_structure(
        general_data_structure,
        [("all_cell_persistence_ratios", cell_persistence_ratios)],
    )

    average_cell_persistence_ratio = np.round(
        np.average(cell_persistence_ratios), decimals=4
    )
    std_cell_persistence_ratio = np.round(np.std(cell_persistence_ratios), decimals=4)

    ax.plot(
        relative_group_centroid_per_tstep[:, 0],
        relative_group_centroid_per_tstep[:, 1],
        marker=None,
        label="group centroid",
        color="k",
        linewidth=2,
    )

    ax.set_ylabel("$\mu m$")
    ax.set_xlabel("$\mu m$")

    average_cell_persistence_time = np.round(
        np.average(all_cell_persistence_times), decimals=2
    )
    std_cell_persistence_time = np.round(np.std(all_cell_persistence_times), decimals=2)
    ax.set_title(
        "group pers_ratio = {} \n avg. cell pers_ratio = {} (std = {}) \n group pers_time = {} {},  avg. cell pers_time = {} {} (std = {} {})".format(
            group_persistence_ratio,
            average_cell_persistence_ratio,
            std_cell_persistence_ratio,
            group_persistence_time,
            time_unit,
            average_cell_persistence_time,
            time_unit,
            std_cell_persistence_time,
            time_unit,
        )
    )
    if save_dir != None:
        write_lines_to_file(
            os.path.join(save_dir, "persistence_data.txt"),
            [
                "group peristence ratio: {}\n".format(group_persistence_ratio),
                "(avg, std) cell persistence ratio: {}, {}\n".format(
                    average_cell_persistence_ratio, std_cell_persistence_ratio
                ),
                "(avg, std) cell persistence time: {}, {}".format(
                    average_cell_persistence_time, std_cell_persistence_time
                ),
            ],
        )

    # ------------------------

    show_or_save_fig(fig, (12, 10), save_dir, save_name, "")

    if make_group_centroid_drift_graph == True:
        general_data_structure = graph_group_centroid_drift(
            T,
            time_unit,
            group_centroid_per_tstep,
            relative_all_cell_centroids_per_tstep,
            save_dir,
            save_name,
            general_data_structure=general_data_structure,
        )

    return general_data_structure


# ==============================================================================


def graph_cell_speed_over_time(
    num_cells,
    T,
    cell_Ls,
    storefile_path,
    save_dir=None,
    save_name=None,
    max_tstep=None,
    time_to_average_over_in_minutes=1.0,
    fontsize=22,
    general_data_structure=None,
    convergence_test=False,
):
    set_fontsize(fontsize)
    fig_time, ax_time = plt.subplots()
    fig_box, ax_box = plt.subplots()

    average_speeds = []
    cell_full_speeds = []
    for ci in range(num_cells):
        L = cell_Ls[ci]
        #        num_timesteps_to_average_over = int(60.0*time_to_average_over_in_minutes/T)
        timepoints, cell_speeds = cu.calculate_cell_speeds_until_tstep(
            ci, max_tstep, storefile_path, T, L
        )

        #        chunky_timepoints = general.chunkify_numpy_array(timepoints, num_timesteps_to_average_over)
        #        chunky_cell_speeds = general.chunkify_numpy_array(cell_speeds, num_timesteps_to_average_over)
        #
        #        averaged_cell_speeds = np.average(chunky_cell_speeds, axis=1)

        #        resized_timepoints = np.arange(num_timesteps_to_average_over*chunky_timepoints.shape[0])
        #        corresponding_cell_speeds = np.repeat(averaged_cell_speeds, num_timesteps_to_average_over)
        ax_time.plot(timepoints, cell_speeds, color=colors.color_list20[ci % 20])
        average_speeds.append(np.average(cell_speeds))
        if convergence_test:
            cell_full_speeds.append(cell_speeds)

    add_to_general_data_structure(
        general_data_structure, [("all_cell_speeds", average_speeds)]
    )
    if convergence_test:
        add_to_general_data_structure(
            general_data_structure, [("cell_full_speeds", cell_full_speeds)]
        )

    # Shrink current axis by 20%
    # box = ax_time.get_position()
    # ax_time.set_position([box.x0, box.y0, box.width*0.8, box.height])

    ax_time.grid(which="both")

    ax_time.set_xlabel("t (min.)")
    ax_time.set_ylabel("$V_N$ ($\mu m$\min.)")

    ax_box.violinplot(average_speeds, showmedians=True, points=len(average_speeds))

    # adding horizontal grid lines
    ax_box.yaxis.grid(True)
    ax_box.set_xlabel("average $V_N$ ($\mu m$\min.)")
    ax_box.xaxis.set_ticks([])
    ax_box.xaxis.set_ticks_position("none")

    show_or_save_fig(fig_time, (12, 8), save_dir, save_name, "")
    show_or_save_fig(fig_box, (8, 8), save_dir, save_name, "box")

    return general_data_structure


# ==============================================================================


def graph_important_cell_variables_over_time(
    T,
    L,
    cell_index,
    storefile_path,
    polarity_scores=None,
    save_dir=None,
    save_name=None,
    max_tstep=None,
    fontsize=22,
    general_data_structure=None,
    convergence_test=False,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()

    # randomization_kicks = hardio.get_data_until_timestep(a_cell, max_tstep, 'randomization_event_occurred')
    # randomization_kicks = np.any(randomization_kicks, axis=1)

    # cell_index, max_tstep, data_label, storefile_path
    rac_mem_active = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rac_membrane_active", storefile_path
    )
    sum_rac_act_over_nodes = np.sum(rac_mem_active, axis=1)

    rac_mem_inactive = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rac_membrane_inactive", storefile_path
    )
    sum_rac_inact_over_nodes = np.sum(rac_mem_inactive, axis=1)

    rho_mem_active = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rho_membrane_active", storefile_path
    )
    sum_rho_act_over_nodes = np.sum(rho_mem_active, axis=1)

    rho_mem_inactive = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rho_membrane_inactive", storefile_path
    )
    sum_rho_inact_over_nodes = np.sum(rho_mem_inactive, axis=1)

    if convergence_test:
        node_coords = hardio.get_node_coords_for_all_tsteps(cell_index, storefile_path)
        edgeplus = np.array(
            [geometry.calculate_edgeplus_lengths(nc) for nc in node_coords]
        )
        edgeminus = np.array(
            [geometry.calculate_edgeplus_lengths(nc) for nc in node_coords]
        )
        average_edge_lengths = (
            np.array(
                [
                    [(ep + em) / 2.0 for ep, em in zip(eps, ems)]
                    for eps, ems in zip(edgeplus, edgeminus)
                ]
            )
            * L
        )

        conc_rac_mem_active = rac_mem_active / average_edge_lengths
        # add_to_general_data_structure(general_data_structure, [("rac_membrane_active_{}".format(cell_index), sum_rac_act_over_nodes)])
        add_to_general_data_structure(
            general_data_structure,
            [
                (
                    "avg_max_conc_rac_membrane_active_{}".format(cell_index),
                    np.max(conc_rac_mem_active, axis=1),
                )
            ],
        )

        conc_rac_mem_inactive = rac_mem_inactive / average_edge_lengths
        # add_to_general_data_structure(general_data_structure, [("rac_membrane_inactive_{}".format(cell_index), sum_rac_inact_over_nodes)])
        add_to_general_data_structure(
            general_data_structure,
            [
                (
                    "avg_max_conc_rac_membrane_inactive_{}".format(cell_index),
                    np.max(conc_rac_mem_inactive, axis=1),
                )
            ],
        )

        conc_rho_membrane_active = rho_mem_active / average_edge_lengths
        add_to_general_data_structure(
            general_data_structure,
            [
                (
                    "avg_max_conc_rho_membrane_active_{}".format(cell_index),
                    np.max(conc_rho_membrane_active, axis=1),
                )
            ],
        )
        # add_to_general_data_structure(general_data_structure, [("rho_membrane_active_{}".format(cell_index), sum_rho_act_over_nodes)])

        conc_rho_membrane_inactive = rho_mem_inactive / average_edge_lengths
        add_to_general_data_structure(
            general_data_structure,
            [
                (
                    "avg_max_conc_rho_membrane_inactive_{}".format(cell_index),
                    np.max(conc_rho_membrane_inactive, axis=1),
                )
            ],
        )
        # add_to_general_data_structure(general_data_structure, [("rho_membrane_inactive_{}".format(cell_index), sum_rho_inact_over_nodes)])

    rac_cyt_gdi = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rac_cytosolic_gdi_bound", storefile_path
    )[:, 0]
    rho_cyt_gdi = hardio.get_data_until_timestep(
        cell_index, max_tstep, "rho_cytosolic_gdi_bound", storefile_path
    )[:, 0]

    time_points = T * np.arange(rac_mem_active.shape[0])

    # for data_set, line_style, data_label in zip([randomization_kicks, sum_rac_act_over_nodes, sum_rho_act_over_nodes, sum_rac_inact_over_nodes, sum_rho_inact_over_nodes, rac_cyt_gdi, rho_cyt_gdi], ['k', 'b', 'r', 'b--', 'r--', 'c', 'm'], ['random kick', 'rac_active', 'rho_active', 'rac_inactive', 'rho_inactive', 'rac_gdi', 'rho_gdi'])

    for data_set, line_style, data_label in zip(
        [
            sum_rac_act_over_nodes,
            sum_rho_act_over_nodes,
            sum_rac_inact_over_nodes,
            sum_rho_inact_over_nodes,
            rac_cyt_gdi,
            rho_cyt_gdi,
        ],
        ["b", "r", "b--", "r--", "c", "m"],
        [
            "rac_active",
            "rho_active",
            "rac_inactive",
            "rho_inactive",
            "rac_gdi",
            "rho_gdi",
        ],
    ):
        ax.plot(time_points, data_set, line_style, label=data_label)

    if polarity_scores.shape[0] != 0:
        ax.plot(time_points, polarity_scores, ".", color="k", label="polarity_scores")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    ax.grid(which="both")

    ax.set_ylim([0, 1.1])
    ax.set_xlabel("time (min)")

    show_or_save_fig(fig, (12, 8), save_dir, save_name, "")

    return general_data_structure


# ==============================================================================


def graph_edge_and_areal_strains(
    T,
    cell_index,
    storefile_path,
    save_dir=None,
    save_name=None,
    max_tstep=None,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    fig_e, ax_e = plt.subplots()
    fig_a, ax_a = plt.subplots()

    avg_edge_strains = np.average(
        hardio.get_data_until_timestep(
            cell_index, max_tstep, "local_strains", storefile_path
        ),
        axis=1,
    )
    node_coordinates_per_tstep = hardio.get_node_coords_for_all_tsteps(
        cell_index, storefile_path
    )
    areas = np.array(
        [geometry.calculate_polygon_area(ncs) for ncs in node_coordinates_per_tstep]
    )
    areal_strains = (areas - areas[0]) / areas[0]

    if "all_cell_areal_strains" not in list(general_data_structure.keys()):
        add_to_general_data_structure(
            general_data_structure, [("all_cell_areal_strains", [areal_strains])]
        )
    else:
        general_data_structure["all_cell_areal_strains"].append(areal_strains)

    time_points = T * np.arange(avg_edge_strains.shape[0])

    ax_e.plot(time_points, avg_edge_strains, "k", label="avg_strains")

    # Shrink current axis by 20%
    box = ax_e.get_position()
    ax_e.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax_e.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    ax_e.grid(which="both")
    ax_e.set_xlabel("t (min.)")

    ax_a.plot(time_points, areal_strains, "k", label="area strain")

    # Shrink current axis by 20%
    box = ax_a.get_position()
    ax_a.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax_a.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    ax_a.grid(which="both")
    ax_a.set_xlabel("t (min.)")
    avg_areal_strain, std_areal_strain = (
        np.round(np.average(areal_strains), decimals=2),
        np.round(np.std(areal_strains), decimals=2),
    )
    ax_a.set_title("avg: {}, std: {}".format(avg_areal_strain, std_areal_strain))

    show_or_save_fig(fig_e, (12, 8), save_dir, save_name, "")
    show_or_save_fig(fig_a, (12, 8), save_dir, save_name, "areal")

    return general_data_structure


# ==============================================================================


def graph_rates(
    T,
    kgtp_rac_baseline,
    kgtp_rho_baseline,
    kdgtp_rac_baseline,
    kdgtp_rho_baseline,
    cell_index,
    storefile_path,
    save_dir=None,
    save_name=None,
    max_tstep=None,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()

    average_kgtp_rac = (
        np.average(
            hardio.get_data_until_timestep(
                cell_index, max_tstep, "kgtp_rac", storefile_path
            ),
            axis=1,
        )
        / kgtp_rac_baseline
    )
    avg_average_kgtp_rac = np.average(average_kgtp_rac)
    average_kgtp_rho = (
        np.average(
            hardio.get_data_until_timestep(
                cell_index, max_tstep, "kgtp_rho", storefile_path
            ),
            axis=1,
        )
        / kgtp_rho_baseline
    )
    avg_average_kgtp_rho = np.average(average_kgtp_rho)
    average_kdgtp_rac = (
        np.average(
            hardio.get_data_until_timestep(
                cell_index, max_tstep, "kdgtp_rac", storefile_path
            ),
            axis=1,
        )
        / kdgtp_rac_baseline
    )
    avg_average_kdgtp_rac = np.average(average_kdgtp_rac)
    average_kdgtp_rho = (
        np.average(
            hardio.get_data_until_timestep(
                cell_index, max_tstep, "kdgtp_rho", storefile_path
            ),
            axis=1,
        )
        / kdgtp_rho_baseline
    )
    avg_average_kdgtp_rho = np.average(average_kdgtp_rho)
    average_coa_signal = (
        np.average(
            hardio.get_data_until_timestep(
                cell_index, max_tstep, "coa_signal", storefile_path
            ),
            axis=1,
        )
        + 1.0
    )

    time_points = T * np.arange(average_kgtp_rac.shape[0]) / 60.0

    for data_set, line_style, data_label in zip(
        [
            average_kgtp_rac,
            average_kgtp_rho,
            average_kdgtp_rac,
            average_kdgtp_rho,
            average_coa_signal,
        ],
        ["b-.", "r-.", "c-.", "m-.", "b"],
        [
            "avg_kgtp_rac ({})".format(avg_average_kgtp_rac),
            "avg_kgtp_rho ({})".format(avg_average_kgtp_rho),
            "avg_kdgtp_rac ({})".format(avg_average_kdgtp_rac),
            "avg_kdgtp_rho ({})".format(avg_average_kdgtp_rho),
            "average_coa_signal",
        ],
    ):
        ax.plot(time_points, data_set, line_style, label=data_label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    ax.grid(which="both")
    ax.set_xlabel("t (min.)")

    show_or_save_fig(fig, (12, 8), save_dir, save_name, "")


# ============================================================================


def present_collated_single_cell_motion_data(
    centroids_persistences_speeds_per_repeat,
    experiment_dir,
    total_time_in_hours,
    time_unit,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    fig_box, ax_box = plt.subplots()

    max_data_lim = 0.0

    persistence_ratios = [x[1][0] for x in centroids_persistences_speeds_per_repeat]
    mean_persistence_ratio = np.round(np.average(persistence_ratios), 2)
    std_persistence_ratio = np.round(np.std(persistence_ratios), 2)
    persistence_times = [x[1][1] for x in centroids_persistences_speeds_per_repeat]
    mean_persistence_time = np.round(np.average(persistence_times), 0)
    std_persistence_time = np.round(np.std(persistence_times), 0)
    average_cell_speeds = [
        np.average(x[2]) for x in centroids_persistences_speeds_per_repeat
    ]

    for i, cps in enumerate(centroids_persistences_speeds_per_repeat):
        ccs = cps[0]
        ccs = ccs - ccs[0]

        # label='({}, {}), ps.={}'.format(si, rpt_number, np.round(persistences[i], decimals=3))
        this_max = np.max(np.abs(ccs))
        if this_max > max_data_lim:
            max_data_lim = this_max

        ax.plot(ccs[:, 0], ccs[:, 1], marker=None, color=colors.color_list300[i % 300])

    if experiment_dir != None:
        write_lines_to_file(
            os.path.join(experiment_dir, "single_cell_persistence_data.txt"),
            [
                "P_R over {} hours (mean: {}, std: {})\nP_T mean: {} {}, (std {} {})\ncell x:P_R, P_T".format(
                    total_time_in_hours,
                    mean_persistence_ratio,
                    std_persistence_ratio,
                    mean_persistence_time,
                    time_unit,
                    std_persistence_time,
                    time_unit,
                )
            ]
            + [
                "cell {}: {}, {}\n".format(ci, pr, pt)
                for ci, pr, pt in zip(
                    np.arange(len(persistence_times)),
                    persistence_ratios,
                    persistence_times,
                )
            ],
        )
    else:
        ax.set_title(
            "$P_R$ over {} hours (mean: {}, std: {}) \n $P_T$ mean: {} {}, (std {} {})".format(
                total_time_in_hours,
                mean_persistence_ratio,
                std_persistence_ratio,
                mean_persistence_time,
                time_unit,
                std_persistence_time,
                time_unit,
            )
        )

    ax.set_ylabel("$\mu m$")
    ax.set_xlabel("$\mu m$")

    ax.set_xlim(-1.1 * max_data_lim, 1.1 * max_data_lim)
    ax.set_ylim(-1.1 * max_data_lim, 1.1 * max_data_lim)
    ax.set_aspect("equal")
    ax.grid(which="both")

    violin = ax_box.violinplot(
        average_cell_speeds, showmedians=True, points=len(average_cell_speeds)
    )

    ax_box.yaxis.grid(True)
    ax_box.set_ylabel("$|V_N|$ ($\mu m$/min.)")
    ax_box.xaxis.set_ticks([])
    ax_box.xaxis.set_ticks_position("none")

    show_or_save_fig(
        fig, (3.825, 3.825), experiment_dir, "collated_single_cell_data", ""
    )
    show_or_save_fig(
        fig_box, (8, 8), experiment_dir, "collated_single_cell_data_speed_box", ""
    )


# =============================================================================

def present_collated_cell_motion_data(
    time_unit,
    all_cell_centroids_per_repeat,
    all_cell_persistence_ratios_per_repeat,
    all_cell_persistence_times_per_repeat,
    all_cell_speeds_per_repeat,
    all_cell_protrusion_lifetimes_and_directions_per_repeat,
    group_centroid_per_timestep_per_repeat,
    group_persistence_ratio_per_repeat,
    group_persistence_time_per_repeat,
    experiment_dir,
    total_time_in_hours,
    fontsize=22,
    chemoattraction_source_coords=None,
):
    set_fontsize(fontsize)
    max_x_data_lim = 0.0
    min_x_data_lim = 0.0
    max_y_data_lim = 0.0
    min_y_data_lim = 0.0

    all_protrusion_lifetimes_and_average_directions = np.zeros((0, 2), dtype=np.float64)

    fig_time, ax_time = plt.subplots()
    for rpti, all_cell_centroids in enumerate(all_cell_centroids_per_repeat):
        for ci, ccs in enumerate(all_cell_centroids):
            relative_ccs = ccs - ccs[0]

            this_max_x_data_lim = np.max(relative_ccs[:, 0])
            this_min_x_data_lim = np.min(relative_ccs[:, 0])
            this_max_y_data_lim = np.max(relative_ccs[:, 1])
            this_min_y_data_lim = np.min(relative_ccs[:, 1])

            if this_max_x_data_lim > max_x_data_lim:
                max_x_data_lim = this_max_x_data_lim
            if this_max_y_data_lim > max_y_data_lim:
                max_y_data_lim = this_max_y_data_lim
            if this_min_x_data_lim < min_x_data_lim:
                min_x_data_lim = this_min_x_data_lim
            if this_min_y_data_lim < min_y_data_lim:
                min_y_data_lim = this_min_y_data_lim

            all_protrusion_lifetimes_and_average_directions = np.append(
                all_protrusion_lifetimes_and_average_directions,
                np.array(
                    all_cell_protrusion_lifetimes_and_directions_per_repeat[rpti][ci]
                ),
                axis=0,
            )

#            print("Plotting centroids for cell {}...".format(ci))
#            ax_time.plot(
#                relative_ccs[:, 0],
#                relative_ccs[:, 1],
#                marker=None,
#                color=colors.color_list300[ci % 300],
#                alpha=0.5,
#            )

    for rpt_number in range(len(group_centroid_per_timestep_per_repeat)):
        group_centroid_per_timestep = group_centroid_per_timestep_per_repeat[rpt_number]
        relative_group_centroid_per_timestep = (
            group_centroid_per_timestep - group_centroid_per_timestep[0]
        )
        print("Plotting group centroid for rpt {}...".format(rpt_number))
        ax_time.plot(
            relative_group_centroid_per_timestep[:, 0],
            relative_group_centroid_per_timestep[:, 1],
            marker=None,
            color=colors.color_list300[rpt_number % 300],
            linewidth=2,
        )

    if type(chemoattraction_source_coords) != type(None):
        chemoattraction_source_coords = (
            chemoattraction_source_coords - group_centroid_per_timestep_per_repeat[0][0]
        )
        max_x_data_lim = np.max(
            [np.abs(chemoattraction_source_coords[0]), max_x_data_lim]
        )
        ax_time.plot(
            [chemoattraction_source_coords[0]],
            [chemoattraction_source_coords[1]],
            ls="",
            marker=".",
            color="g",
            markersize=20,
        )

    mean_persistence_ratio = np.round(
        np.average(all_cell_persistence_ratios_per_repeat), 2
    )
    std_persistence_ratio = np.round(np.std(all_cell_persistence_ratios_per_repeat), 2)
    mean_group_persistence_ratio = np.round(
        np.average(group_persistence_ratio_per_repeat), 2
    )
    mean_persistence_time = np.round(
        np.average(all_cell_persistence_times_per_repeat), 0
    )
    std_persistence_time = np.round(np.std(all_cell_persistence_times_per_repeat), 0)
    mean_group_persistence_time = np.round(
        np.average(group_persistence_time_per_repeat), 0
    )

    ax_time.set_title(
        "Experiment over {} hours \n Persistence ratio, cell mean: {} (std: {}), group mean: {} \n Persistence time cell mean: {} {}, (std {} {}), group mean: {} {}".format(
            total_time_in_hours,
            mean_persistence_ratio,
            std_persistence_ratio,
            mean_group_persistence_ratio,
            mean_persistence_time,
            time_unit,
            std_persistence_time,
            time_unit,
            mean_group_persistence_time,
            time_unit,
        )
    )

    ax_time.set_ylabel("$\mu m$")
    ax_time.set_xlabel("$\mu m$")

    if type(chemoattraction_source_coords) == type(None):
        y_lim = 1.1 * np.max([np.abs(min_y_data_lim), np.abs(max_y_data_lim)])
    else:
        y_lim = 1.1 * np.max(
            [
                np.abs(min_y_data_lim),
                np.abs(max_y_data_lim),
                np.abs(chemoattraction_source_coords[1]),
            ]
        )

    if max_x_data_lim > 0.0:
        max_x_data_lim = 1.1 * max_x_data_lim
    else:
        max_x_data_lim = -1 * 1.1 * np.abs(max_x_data_lim)

    if min_x_data_lim > 0.0:
        min_x_data_lim = 1.1 * min_x_data_lim
    else:
        min_x_data_lim = -1 * 1.1 * np.abs(min_x_data_lim)

    #ax_time.set_xlim(min_x_data_lim, max_x_data_lim)
    ax_time.set_xlim(-700, 700)

    if y_lim < 0.2 * (max_x_data_lim - min_x_data_lim):
        y_lim = 0.2 * (max_x_data_lim - min_x_data_lim)

    #ax_time.set_ylim(-1 * y_lim, y_lim)
    ax_time.set_ylim(-700, 700)
    ax_time.set_aspect("equal")

    ax_time.grid(which="both")

    fig_box, ax_box = plt.subplots()

    all_cell_average_speeds = np.ravel(all_cell_speeds_per_repeat)
    #    print "plotting cell speed violin..."
    #    violin = ax_box.violinplot(all_cell_average_speeds, showmedians=True, points=all_cell_average_speeds.shape[0])

    ax_box.yaxis.grid(True)
    ax_box.set_ylabel("$|V_N|$ ($\mu m$/min.)")
    ax_box.xaxis.set_ticks([])
    ax_box.xaxis.set_ticks_position("none")
    
    print("plotting protrusion directions radially...")
    graph_protrusion_directions_radially(
        all_protrusion_lifetimes_and_average_directions,
        12,
        total_time_in_hours * 60.0,
        cutoff_time_in_minutes=0.0,
        save_dir=experiment_dir,
        save_name="all_cells_protrusion_directions",
    )
    
    print("plotting protrusion directions radially...")
    graph_protrusion_directions_radially(
        all_protrusion_lifetimes_and_average_directions,
        12,
        total_time_in_hours * 60.0,
        cutoff_time_in_minutes=40.0,
        save_dir=experiment_dir,
        save_name="all_cells_protrusion_directions",
    )

    print("plotting protrusion lifetimes radially...")
    graph_protrusion_lifetimes_radially(
        all_protrusion_lifetimes_and_average_directions,
        12,
        total_time_in_hours * 60.0,
        save_dir=experiment_dir,
        save_name="all_cells_protrusion_life_dir",
    )

    print("showing/saving figures...")
    show_or_save_fig(fig_time, (12, 6), experiment_dir, "collated_cell_data", "")
    show_or_save_fig(
        fig_box, (8, 8), experiment_dir, "collated_cell_data_speed_box", ""
    )


# =============================================================================


def present_collated_group_centroid_drift_data(
    T,
    cell_diameter,
    min_x_centroid_per_tstep_per_repeat,
    max_x_centroid_per_tstep_per_repeat,
    group_x_centroid_per_tstep_per_repeat,
    fit_group_x_velocity_per_repeat,
    save_dir,
    total_time_in_hours,
    fontsize=22,
    general_data_structure=None,
    ax_simple_normalized=None,
    ax_full_normalized=None,
    ax_simple=None,
    ax_full=None,
    plot_speedbox=True,
    plot_if_no_axis_given=True,
    min_ylim=0.0,
    max_ylim=1500.0,
):
    set_fontsize(fontsize)
    timepoints = np.arange(group_x_centroid_per_tstep_per_repeat[0].shape[0]) * T / 60.0
    max_timepoint = int(((total_time_in_hours * 3600.0) / T) + 1)

    fig_simple_normalized, fig_simple, fig_full_normalized, fig_full, fig_box = (
        None,
        None,
        None,
        None,
        None,
    )
    if plot_if_no_axis_given:
        if ax_simple_normalized == None:
            fig_simple_normalized, ax_simple_normalized = plt.subplots()
        if ax_full_normalized == None:
            fig_full_normalized, ax_full_normalized = plt.subplots()
        if ax_simple == None:
            fig_simple, ax_simple = plt.subplots()
        if ax_full == None:
            fig_full, ax_full = plt.subplots()

    if plot_speedbox:
        fig_box, ax_box = plt.subplots()

    num_repeats = len(group_x_centroid_per_tstep_per_repeat)

    bar_step = int(0.01 * num_repeats * timepoints.shape[0])
    bar_offset = int(0.01 * timepoints.shape[0])

    for repeat_number in range(num_repeats):
        max_x_centroid_per_tstep = max_x_centroid_per_tstep_per_repeat[repeat_number]
        min_x_centroid_per_tstep = min_x_centroid_per_tstep_per_repeat[repeat_number]
        group_x_centroid_per_tstep = group_x_centroid_per_tstep_per_repeat[
            repeat_number
        ]

        relative_group_x_centroid_per_tstep = (
            group_x_centroid_per_tstep - min_x_centroid_per_tstep[0]
        )
        relative_max_x_centroid_per_tstep = (
            max_x_centroid_per_tstep - min_x_centroid_per_tstep[0]
        )
        relative_min_x_centroid_per_tstep = (
            min_x_centroid_per_tstep - min_x_centroid_per_tstep[0]
        )

        normalized_relative_group_centroid_x_coords = (
            relative_group_x_centroid_per_tstep / cell_diameter
        )
        normalized_relative_max_centroid_x_coords = (
            relative_max_x_centroid_per_tstep / cell_diameter
        )
        normalized_relative_min_centroid_x_coords = (
            relative_min_x_centroid_per_tstep / cell_diameter
        )

        bar_indices = np.arange(
            bar_offset * repeat_number,
            timepoints.shape[0] - bar_offset,
            bar_step,
            dtype=np.int64,
        )

        if repeat_number == 0:
            bar_indices = np.append(bar_indices, timepoints.shape[0] - 1)

        bar_timepoints = timepoints[bar_indices]

        if ax_simple_normalized != None:
            ax_simple_normalized.plot(
                timepoints[:max_timepoint],
                normalized_relative_group_centroid_x_coords[:max_timepoint],
                color=colors.color_list300[repeat_number % 300],
            )

        if ax_simple != None:
            ax_simple.plot(
                timepoints[:max_timepoint],
                relative_group_x_centroid_per_tstep[:max_timepoint],
                color=colors.color_list300[repeat_number % 300],
            )
            ax_simple.set_ylim([min_ylim, max_ylim])

        if ax_full_normalized != None:
            ax_full_normalized.plot(
                timepoints[:max_timepoint],
                normalized_relative_group_centroid_x_coords[:max_timepoint],
                color=colors.color_list300[repeat_number % 300],
            )
            bar_points = normalized_relative_group_centroid_x_coords[bar_indices]
            bar_min_points = normalized_relative_min_centroid_x_coords[bar_indices]
            bar_max_points = normalized_relative_max_centroid_x_coords[bar_indices]
            lower_bounds = np.abs(bar_points - bar_min_points)
            upper_bounds = np.abs(bar_points - bar_max_points)
            ax_full_normalized.errorbar(
                bar_timepoints,
                bar_points,
                yerr=[lower_bounds, upper_bounds],
                ls="",
                capsize=5,
                color=colors.color_list300[repeat_number % 300],
            )

        if ax_full != None:
            ax_full.plot(
                timepoints[:max_timepoint],
                relative_group_x_centroid_per_tstep[:max_timepoint],
                color=colors.color_list300[repeat_number % 300],
            )
            bar_points = relative_group_x_centroid_per_tstep[bar_indices]
            bar_min_points = relative_min_x_centroid_per_tstep[bar_indices]
            bar_max_points = relative_max_x_centroid_per_tstep[bar_indices]
            lower_bounds = np.abs(bar_points - bar_min_points)
            upper_bounds = np.abs(bar_points - bar_max_points)
            ax_full.errorbar(
                bar_timepoints,
                bar_points,
                yerr=[lower_bounds, upper_bounds],
                ls="",
                capsize=5,
                color=colors.color_list300[repeat_number % 300],
            )
            ax_full.set_ylim([min_ylim, max_ylim])

    if plot_if_no_axis_given:
        ax_simple_normalized.set_ylabel("$X_c/r$")
        ax_simple.set_ylabel("$X_c$ ($\mu m$)")
        ax_simple.set_xlabel("t (min.)")
        ax_simple_normalized.set_xlabel("t (min.)")
        ax_full_normalized.set_ylabel("$X_c$ \n (normalized by initial group width)")
        ax_full.set_ylabel("$X_c$ ($\mu m$)")
        ax_full_normalized.set_xlabel("t (min.)")
        ax_full.set_xlabel("t (min.)")
        ax_simple_normalized.grid(which="both")
        ax_simple.grid(which="both")
        ax_full_normalized.grid(which="both")
        ax_full.grid(which="both")

    if plot_speedbox:
        violin = ax_box.violinplot(
            fit_group_x_velocity_per_repeat,
            showmedians=True,
            points=len(fit_group_x_velocity_per_repeat),
        )
        ax_box.yaxis.grid(True)
        ax_box.set_ylabel("avg. $V_c$ ($\mu m$/min.)")
        ax_box.xaxis.set_ticks([])
        ax_box.xaxis.set_ticks_position("none")

    #    for fig, base_name, fig_size in zip([fig_simple_normalized, fig_simple, fig_full_normalized, fig_full, fig_box], ["collated_group_centroid_drift_simple_normalized", "collated_group_centroid_drift_simple", "collated_group_centroid_drift_full_normalized", "collated_group_centroid_drift_full", "collated_group_speed_box"], [(4, 4), (4, 4), (4, 4), (4, 4), (4, 4)]):
    #        if fig != None:
    #            show_or_save_fig(fig, fig_size, save_dir, base_name, "")

    for fig, base_name, fig_size in zip(
        [fig_simple_normalized, fig_simple, fig_full_normalized, fig_full, fig_box],
        [
            "collated_group_centroid_drift_simple_normalized",
            "collated_group_centroid_drift_simple",
            "collated_group_centroid_drift_full_normalized",
            "collated_group_centroid_drift_full",
            "collated_group_speed_box",
        ],
        [(7, 6), (7, 6), (7, 6), (7, 6), (7, 6)],
    ):
        if fig != None:
            show_or_save_fig(fig, fig_size, save_dir, base_name, "")
    #
    return ax_simple_normalized, ax_full_normalized, ax_simple, ax_full


# =============================================================================


def generate_theta_bins(num_bins):
    delta = 2 * np.pi / num_bins
    start = 0.5 * delta

    bin_bounds = []
    current = start
    for n in range(num_bins - 1):
        bin_bounds.append([current, current + delta])
        current += delta
    bin_bounds.append([2 * np.pi - 0.5 * delta, start])
    bin_bounds = np.array(bin_bounds)
    bin_mids = np.average(bin_bounds, axis=1)
    bin_mids[-1] = 0.0

    return bin_bounds, bin_mids, delta

# =============================================================================

def graph_protrusion_bias_vectors(
    protrusion_lifetime_and_average_directions_per_cell,
    num_polar_graph_bins,
    total_simulation_time_in_minutes,
    save_dir=None,
    save_name=None,
    fontsize=40,
    general_data_structure=None,
):
    num_cells = len(protrusion_lifetime_and_average_directions_per_cell)
    num_polar_graph_bins = 16
    set_fontsize(fontsize)

    bin_bounds, bin_midpoints, bin_size = generate_theta_bins(num_polar_graph_bins)
    binned_direction_data_per_cell = [[[] for x in range(num_polar_graph_bins)] for y in range(num_cells)]
    
    for ci, protrusion_lifetime_and_direction_data in enumerate(protrusion_lifetime_and_average_directions_per_cell):
        for protrusion_result in protrusion_lifetime_and_direction_data:
            lifetime, direction = protrusion_result
            lifetime = lifetime
    
            binned = False
            for n in range(num_polar_graph_bins - 1):
                a, b = bin_bounds[n]
                if a <= direction < b:
                    binned = True
                    binned_direction_data_per_cell[ci][n].append(lifetime)
                    break
    
            if binned == False:
                binned_direction_data_per_cell[ci][-1].append(lifetime)
                
    protrusion_bias_vectors_per_cell = []
    for binned_direction_data in binned_direction_data_per_cell:
        bin_bias_vectors = []
        for (bin_midpoint, bin_data) in zip(bin_midpoints, binned_direction_data):
            bin_sum = np.sum(bin_data)
            bin_bias_vectors.append([bin_sum*np.cos(bin_midpoint), bin_sum*np.sin(bin_midpoint)])
        protrusion_bias_vectors_per_cell.append(np.average(bin_bias_vectors, axis=0))

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    max_lim = np.max(np.abs(protrusion_bias_vectors_per_cell))
    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])
    for ci, bv in enumerate(protrusion_bias_vectors_per_cell):
        ax.plot([0.0, bv[0]], [0.0, bv[1]], color=colors.color_list300[int(ci%300)])

    show_or_save_fig(
        fig,
        (14, 12),
        save_dir,
        save_name,
        "B={}".format(num_polar_graph_bins),
        figtype="png",
        bbox_inches=None,
    )
    
# =============================================================================


def graph_protrusion_directions_radially(
    protrusion_lifetime_and_direction_data,
    num_polar_graph_bins,
    total_simulation_time_in_minutes,
    cutoff_time_in_minutes=40.0,
    save_dir=None,
    save_name=None,
    fontsize=40,
    general_data_structure=None,
):
    num_polar_graph_bins = 16
    set_fontsize(fontsize)

    bin_bounds, bin_midpoints, bin_size = generate_theta_bins(num_polar_graph_bins)
    binned_direction_data = [[] for x in range(num_polar_graph_bins)]
    for protrusion_result in protrusion_lifetime_and_direction_data:
        lifetime, direction = protrusion_result
        binned = False
        
        for n in range(num_polar_graph_bins - 1):
            a, b = bin_bounds[n]
            if a <= direction < b:
                binned = True
                if lifetime > cutoff_time_in_minutes:
                    binned_direction_data[n].append(1)
                break

        if binned == False and lifetime > cutoff_time_in_minutes:
            binned_direction_data[-1].append(1)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    
    summed_bin_direction_data = [np.sum(x) for x in binned_direction_data]

    bars = ax.bar(bin_midpoints, summed_bin_direction_data, width=bin_size, bottom=0.0)

    for bar in bars:
        bar.set_facecolor("green")
        bar.set_alpha(0.25)

    for bi in range(num_polar_graph_bins):
        ax.text(
            bin_midpoints[bi],
            summed_bin_direction_data[bi],
            "{}".format(len(binned_direction_data[bi])),
            fontdict={"size": 0.5 * fontsize},
        )

    max_y = np.max(summed_bin_direction_data)

    ax.yaxis.get_major_locator().base.set_params(nbins=5)
    # label_position=ax.get_rlabel_position()
    # ax.text(np.pi, ax.get_rmax()/2., 't (min.)', rotation=0.0,ha='center',va='center')

    show_or_save_fig(
        fig,
        (14, 12),
        save_dir,
        save_name,
        "CT={}_B={}".format(cutoff_time_in_minutes, num_polar_graph_bins),
        figtype="png",
        bbox_inches=None,
    )
    ax.set_yticklabels([])
    show_or_save_fig(
        fig,
        (14, 12),
        save_dir,
        save_name,
        "no_labels_CT={}_B={}".format(cutoff_time_in_minutes, num_polar_graph_bins),
        figtype="png",
        bbox_inches=None,
    )

# =============================================================================


def graph_protrusion_lifetimes_radially(
    protrusion_lifetime_and_direction_data,
    num_polar_graph_bins,
    total_simulation_time_in_minutes,
    save_dir=None,
    save_name=None,
    fontsize=40,
    general_data_structure=None,
):
    num_polar_graph_bins = 16
    set_fontsize(fontsize)

    bin_bounds, bin_midpoints, bin_size = generate_theta_bins(num_polar_graph_bins)
    binned_direction_data = [[] for x in range(num_polar_graph_bins)]
    for protrusion_result in protrusion_lifetime_and_direction_data:
        lifetime, direction = protrusion_result
        lifetime = lifetime

        binned = False
        for n in range(num_polar_graph_bins - 1):
            a, b = bin_bounds[n]
            if a <= direction < b:
                binned = True
                binned_direction_data[n].append(lifetime)
                break

        if binned == False:
            binned_direction_data[-1].append(lifetime)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    summed_x_direction_data = [
        np.sum(x) / total_simulation_time_in_minutes for x in binned_direction_data
    ]

    bars = ax.bar(bin_midpoints, summed_x_direction_data, width=bin_size, bottom=0.0)

    for bar in bars:
        bar.set_facecolor("green")
        bar.set_alpha(0.25)

    for bi in range(num_polar_graph_bins):
        ax.text(
            bin_midpoints[bi],
            summed_x_direction_data[bi],
            "{}".format(len(binned_direction_data[bi])),
            fontdict={"size": 0.5 * fontsize},
        )

    max_y = np.max(summed_x_direction_data)

    ax.yaxis.get_major_locator().base.set_params(nbins=5)
    # label_position=ax.get_rlabel_position()
    # ax.text(np.pi, ax.get_rmax()/2., 't (min.)', rotation=0.0,ha='center',va='center')

    show_or_save_fig(
        fig,
        (14, 12),
        save_dir,
        save_name,
        "B={}".format(num_polar_graph_bins),
        figtype="png",
        bbox_inches=None,
    )
    ax.set_yticklabels([])
    show_or_save_fig(
        fig,
        (14, 12),
        save_dir,
        save_name,
        "no_labels_B={}".format(num_polar_graph_bins),
        figtype="png",
        bbox_inches=None,
    )


# =============================================================================


def graph_protrusion_start_end_causes_radially(
    protrusion_lifetime_and_direction_data,
    protrusion_start_end_cause_data,
    num_polar_graph_bins,
    save_dir=None,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    bins, bin_midpoints, delta = generate_theta_bins(num_polar_graph_bins)

    start_cause_labels = ["coa", "randomization", "coa+rand"]
    end_cause_labels = ["cil", "other"]

    binned_start_cause_data = [
        np.zeros(3, dtype=np.int64) for x in range(num_polar_graph_bins)
    ]
    binned_end_cause_data = [
        np.zeros(2, dtype=np.int64) for x in range(num_polar_graph_bins)
    ]
    for cell_protrusion_data in zip(
        protrusion_lifetime_and_direction_data, protrusion_start_end_cause_data
    ):
        for protrusion_lifetime_direction_result, protrusion_start_end_cause in zip(
            cell_protrusion_data[0], cell_protrusion_data[1]
        ):
            _, direction = protrusion_lifetime_direction_result
            start_causes, end_causes = protrusion_start_end_cause

            bin_index = -1
            for n in range(num_polar_graph_bins - 1):
                a, b = bins[n]
                if a <= direction < b:
                    bin_index = n
                    break

            if bin_index == -1:
                bin_index = num_polar_graph_bins - 1

            if "coa" in start_causes and "cil" in start_causes:
                binned_start_cause_data[bin_index][2] += 1
            elif "coa" in start_causes:
                binned_start_cause_data[bin_index][0] += 1
            elif "rand" in start_causes:
                binned_start_cause_data[bin_index][1] += 1

            if "cil" in end_causes:
                binned_end_cause_data[bin_index][0] += 1
            else:
                binned_end_cause_data[bin_index][1] += 1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # ax.set_title('start causes given direction')
    for n, l in enumerate(start_cause_labels):
        # ax.bar(bin_midpoints, [x[n] for x in binned_start_cause_data], width=delta, bottom=0.0, color=colors.color_list20[n%20], label=l, alpha=0.5)
        # ax.plot(bin_midpoints, , label=l, ls='', marker=styles[n%3], markerfacecolor=colors.color_list20[n%20], color=colors.color_list20[n%20], markersize=30)
        thetas = np.zeros(0, dtype=np.float64)
        rs = np.zeros(0, dtype=np.float64)
        for bi in range(num_polar_graph_bins):
            a, b = bins[bi][0], bins[bi][1]
            if a > b:
                b = b + 2 * np.pi
                if a > b:
                    raise Exception("a is still greater than b!")
            thetas = np.append(thetas, np.linspace(a, b))
            rs = np.append(rs, 50 * [binned_start_cause_data[bi][n]])

        thetas = np.append(thetas, [thetas[-1] + 1e-6])
        rs = np.append(rs, [rs[0]])

        ax.plot(
            thetas, rs, label=l, color=colors.color_list20[n % 20], ls="", marker="."
        )

    ax.legend(loc="best", fontsize=fontsize)

    show_or_save_fig(fig, (8, 8), save_dir, "start_causes_given_direction", "")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    for n, l in enumerate(end_cause_labels):
        thetas = np.zeros(0, dtype=np.float64)
        rs = np.zeros(0, dtype=np.float64)
        for bi in range(num_polar_graph_bins):
            a, b = bins[bi][0], bins[bi][1]
            if a > b:
                b = b + 2 * np.pi
                if a > b:
                    raise Exception("a is still greater than b!")
            thetas = np.append(thetas, np.linspace(a, b))
            rs = np.append(rs, 50 * [binned_end_cause_data[bi][n]])

        thetas = np.append(thetas, [thetas[-1] + 1e-6])
        rs = np.append(rs, [rs[0]])

        ax.plot(
            thetas, rs, label=l, color=colors.color_list20[n % 20], ls="", marker="."
        )

    ax.legend(loc="best", fontsize=fontsize)

    show_or_save_fig(fig, (8, 8), save_dir, "end_causes_given_direction", "")


# ============================================================================


def graph_forward_backward_protrusions_per_timestep(
    max_tstep,
    protrusion_node_index_and_tpoint_start_ends,
    protrusion_lifetime_and_direction_data,
    T,
    forward_cones,
    backward_cones,
    num_nodes,
    save_dir=None,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    times = np.arange(max_tstep) * T / 60.0
    num_forward_protrusions = np.zeros(max_tstep, dtype=np.float)
    num_backward_protrusions = np.zeros(max_tstep, dtype=np.float)
    num_neither_protrusions = np.zeros(max_tstep, dtype=np.float64)

    for cell_protrusion_data in zip(
        protrusion_node_index_and_tpoint_start_ends,
        protrusion_lifetime_and_direction_data,
    ):
        for protrusion_start_end_info, protrusion_lifetime_direction_info in zip(
            cell_protrusion_data[0], cell_protrusion_data[1]
        ):
            ni, ti_start, ti_end = protrusion_start_end_info
            _, direction = protrusion_lifetime_direction_info

            direction_bin = None
            for lims in forward_cones:
                if lims[0] <= direction < lims[1]:
                    direction_bin = "f"
                    break
            if direction_bin != "f":
                for lims in backward_cones:
                    if lims[0] <= direction < lims[1]:
                        direction_bin = "b"
                        break

            if direction_bin == "f":
                num_forward_protrusions[ti_start:ti_end] += 1
            elif direction_bin == "b":
                num_backward_protrusions[ti_start:ti_end] += 1
            else:
                num_neither_protrusions[ti_start:ti_end] += 1

    fig, ax = plt.subplots()

    ax.plot(times, num_forward_protrusions, label="forward")
    ax.plot(times, num_backward_protrusions, label="backward")
    # ax.plot(times, other_cone, label='other')

    ax.legend(loc="best", fontsize=fontsize)
    ax.set_ylabel("number of protrusions")
    ax.set_xlabel("t (min.)")

    fig_with_other, ax_with_other = plt.subplots()

    ax_with_other.plot(times, num_forward_protrusions, label="forward")
    ax_with_other.plot(times, num_backward_protrusions, label="backward")
    ax_with_other.plot(times, num_neither_protrusions, label="other")
    # ax.plot(times, other_cone, label='other')

    ax_with_other.legend(loc="best", fontsize=fontsize)
    ax_with_other.set_ylabel("number of protrusions")
    ax_with_other.set_xlabel("t (min.)")

    fig_normalized, ax_normalized = plt.subplots()

    ax_normalized.plot(times, num_forward_protrusions / num_nodes, label="forward")
    ax_normalized.plot(times, num_backward_protrusions / num_nodes, label="backward")
    ax_normalized.plot(times, num_neither_protrusions / num_nodes, label="other")
    # ax.plot(times, other_cone, label='other')

    ax_normalized.legend(loc="best", fontsize=fontsize)
    ax_normalized.set_ylabel("number of protrusions/N per cell")
    ax_normalized.set_xlabel("t (min.)")

    fig_normalized_with_other, ax_normalized_with_other = plt.subplots()

    ax_normalized_with_other.plot(
        times, num_forward_protrusions / num_nodes, label="forward"
    )
    ax_normalized_with_other.plot(
        times, num_backward_protrusions / num_nodes, label="backward"
    )
    ax_normalized_with_other.plot(
        times, num_neither_protrusions / num_nodes, label="other"
    )
    # ax.plot(times, other_cone, label='other')

    ax_normalized_with_other.legend(loc="best", fontsize=fontsize)
    ax_normalized_with_other.set_ylabel("number of protrusions/N per cell")
    ax_normalized_with_other.set_xlabel("t (min.)")

    for f, fsize, base_name in zip(
        [fig, fig_with_other, fig_normalized, fig_normalized_with_other],
        [(12, 8)] * 4,
        [
            "num_forward_backward_protrusions_over_time",
            "num_fbo_protrusions_over_time",
            "normalized_forward_backward_protrusions_over_time",
            "normalized_fbo_protrusions_over_time",
        ],
    ):
        show_or_save_fig(f, fsize, save_dir, base_name, "")


# ============================================================================


def graph_forward_backward_cells_per_timestep(
    max_tstep,
    all_cell_speeds_and_directions,
    T,
    forward_cones,
    backward_cones,
    save_dir=None,
    fontsize=22,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    times = np.arange(max_tstep) * T / 60.0
    num_forward_cells = np.zeros(max_tstep, dtype=np.float64)
    num_backward_cells = np.zeros(max_tstep, dtype=np.float64)
    num_other_cells = np.zeros(max_tstep, dtype=np.float64)

    for cell_speed_direction_data in all_cell_speeds_and_directions:
        speeds, directions = cell_speed_direction_data
        for ti in range(max_tstep):
            if speeds[ti] > 0.5:  # speed has to be greater than 0.5 $\mu m$ per minute
                direction = directions[ti]
                direction_bin = None
                for lims in forward_cones:
                    if lims[0] <= direction < lims[1]:
                        direction_bin = "f"
                        break
                if direction_bin != "f":
                    for lims in backward_cones:
                        if lims[0] <= direction < lims[1]:
                            direction_bin = "b"
                            break

                if direction_bin == "f":
                    num_forward_cells[ti] += 1
                elif direction_bin == "b":
                    num_backward_cells[ti] += 1
                else:
                    num_other_cells[ti] += 1

    fig, ax = plt.subplots()

    ax.plot(times, num_forward_cells, label="forward")
    ax.plot(times, num_backward_cells, label="backward")

    ax.legend(loc="best", fontsize=fontsize)

    ax.set_ylabel("n")
    ax.set_xlabel("t (min.)")

    fig_with_other, ax_with_other = plt.subplots()

    ax_with_other.plot(times, num_forward_cells, label="forward")
    ax_with_other.plot(times, num_backward_cells, label="backward")
    ax_with_other.plot(times, num_other_cells, label="other")

    ax_with_other.legend(loc="best", fontsize=fontsize)

    ax_with_other.set_ylabel("n")
    ax_with_other.set_xlabel("t (min.)")

    show_or_save_fig(fig, (12, 8), save_dir, "num_forward_backward_cells_over_time", "")
    show_or_save_fig(fig_with_other, (12, 8), save_dir, "num_fbo_cells_over_time", "")


# =============================================================================


def graph_coa_variation_test_data_simple(
    sub_experiment_number,
    num_timepoints,
    T,
    num_cells_to_test,
    test_coas,
    average_cell_group_area_data,
    save_dir=None,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    timepoints = np.arange(num_timepoints) * T / 60.0

    for i, n in enumerate(num_cells_to_test):
        for j, tcoa in enumerate(test_coas):
            ax.plot(
                timepoints,
                average_cell_group_area_data[i][j],
                label="n={},".format(n) + "$M_{COA}$" + "={}".format(test_coas[j]),
            )
            ax.set_xlabel("t (min.)")
            ax.set_ylabel("$A(t)/A(0)$")

    show_or_save_fig(
        fig,
        (6, 2),
        save_dir,
        "coa_variation_results_{}".format(sub_experiment_number),
        "",
    )


# =========================================================================


def graph_confinement_data_persistence_ratios(
    sub_experiment_number,
    test_num_cells,
    test_heights,
    average_cell_persistence,
    save_dir=None,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()

    bin_boundaries = np.linspace(0.5, 1.0, num=100)
    cax = ax.imshow(
        average_cell_persistence, interpolation="none", cmap=plt.get_cmap("inferno")
    )
    cbar = fig.colorbar(
        cax, boundaries=bin_boundaries, ticks=np.linspace(0.5, 1.0, num=5)
    )
    cax.set_clim(0.5, 1.0)
    ax.set_yticks(np.arange(len(test_num_cells)))
    ax.set_xticks(np.arange(len(test_heights)))
    ax.set_yticklabels(test_num_cells)
    ax.set_xticklabels(test_heights)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    show_or_save_fig(
        fig,
        (12, 8),
        save_dir,
        "confinement_test_persistence_ratios_{}".format(sub_experiment_number),
        "",
    )


def graph_confinement_data_persistence_times(
    sub_experiment_number,
    test_num_cells,
    test_heights,
    average_cell_persistence,
    save_dir=None,
    general_data_structure=None,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()

    cax = ax.imshow(
        average_cell_persistence, interpolation="none", cmap=plt.get_cmap("inferno")
    )
    cbar = fig.colorbar(cax)

    ax.set_yticks(np.arange(len(test_num_cells)))
    ax.set_xticks(np.arange(len(test_heights)))
    ax.set_yticklabels(test_num_cells)
    ax.set_xticklabels(test_heights)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    show_or_save_fig(
        fig,
        (12, 8),
        save_dir,
        "confinement_test_graph_persistence_times_{}".format(sub_experiment_number),
        "",
    )


# ====================================================================


def calculate_statistics_and_write_into_text_file(
    sub_experiment_number,
    test_num_cells,
    test_heights,
    group_persistence_ratios,
    group_x_velocities,
    areal_strains,
    save_dir,
):
    lines = []

    lines.append(
        "(test num cells, test corridor widths): {}".format(
            str(list(zip(test_num_cells, test_heights)))
        )
    )
    lines.append("\n")

    lines.append("(average, median) group persistence ratios:\n")
    for i, gprs in enumerate(group_persistence_ratios):
        avg_pr = np.round(np.average(gprs), decimals=3)
        med_pr = np.round(np.median(gprs), decimals=3)
        lines.append("\tNC={}: {}, {}\n".format(test_num_cells[i], avg_pr, med_pr))
    lines.append("\n")

    lines.append("(average, median) group velocities\n")
    for tnc, gvs in enumerate(group_x_velocities):
        avg_v = np.round(np.average(gvs), decimals=3)
        med_v = np.round(np.median(gvs), decimals=3)
        lines.append("\tNC={}: {}, {}\n".format(tnc, avg_v, med_v))
    lines.append("\n")

    relevant_tnc_indices = [i for i, tnc in enumerate(test_num_cells) if not (tnc < 9)]
    lines.append(
        "(average, median) group velocities for tnc in {}\n".format(
            [test_num_cells[i] for i in relevant_tnc_indices]
        )
    )
    relevant_gvs = []
    for i in relevant_tnc_indices:
        relevant_gvs.append(group_x_velocities[i])
    avg_v = np.round(np.average(relevant_gvs), decimals=3)
    med_v = np.round(np.median(relevant_gvs), decimals=3)
    lines.append("\t{}, {}\n".format(avg_v, med_v))
    lines.append("\n")

    avg_areal_strain = np.round(np.average(areal_strains), decimals=2)
    std_areal_strain = np.round(np.std(areal_strains), decimals=2)
    lines.append(
        "(avg, std) areal strain: {}, {}".format(avg_areal_strain, std_areal_strain)
    )
    fp = os.path.join(save_dir, "cell_number_change_data.txt")

    with open(fp, "w+") as f:
        f.writelines(lines)


# ====================================================================


def calculate_statistics_and_write_into_general_file(
    text_file_name,
    labels,
    group_persistence_ratios,
    group_x_velocities,
    areal_strains,
    save_dir,
):
    lines = []

    lines.append("labels: {}".format(labels))
    lines.append("\n")

    lines.append("(average, median) group persistence ratios:\n")
    for i, gprs in enumerate(group_persistence_ratios):
        avg_pr = np.round(np.average(gprs), decimals=3)
        med_pr = np.round(np.median(gprs), decimals=3)
        lines.append("\tlabel={}: {}, {}\n".format(labels[i], avg_pr, med_pr))
    lines.append("\n")

    lines.append("(average, median) group velocities\n")
    for i, gvs in enumerate(group_x_velocities):
        avg_v = np.round(np.average(gvs), decimals=3)
        med_v = np.round(np.median(gvs), decimals=3)
        lines.append("\tlabel={}: {}, {}\n".format(labels[i], avg_v, med_v))
    lines.append("\n")

    fp = os.path.join(save_dir, "{}.txt".text_file_name)

    with open(fp, "w+") as f:
        f.writelines(lines)


# =======================================================================


def graph_cell_number_change_data(
    sub_experiment_number,
    test_num_cells,
    test_heights,
    graph_x_dimension,
    group_persistence_ratios,
    group_persistence_times,
    fit_group_x_velocities,
    cell_separations,
    areal_strains,
    experiment_set_label,
    save_dir=None,
    fontsize=22,
):

    set_fontsize(2 * fontsize)

    calculate_statistics_and_write_into_text_file(
        sub_experiment_number,
        test_num_cells,
        test_heights,
        group_persistence_ratios,
        fit_group_x_velocities,
        areal_strains,
        save_dir,
    )

    if graph_x_dimension == "test_heights":
        x_axis_positions = [i + 1 for i in range(len(test_heights))]
        x_axis_stuff = test_heights
        x_label = "corridor width (in cell diameters, N = {})".format(test_num_cells[0])
    elif graph_x_dimension == "test_num_cells":
        x_axis_positions = [i + 1 for i in range(len(test_num_cells))]
        x_axis_stuff = test_num_cells
        x_label = "n"
    else:
        raise Exception("Unknown graph_x_dimension given: {}".format(graph_x_dimension))

    # fit_group_x_velocities/(cell_separations*40)
    data_sets = [
        group_persistence_ratios,
        group_persistence_times,
        fit_group_x_velocities,
        cell_separations,
    ]
    data_labels = [
        "group persistence ratios",
        "group persistence times",
        "group X velocity",
        "average cell separation",
        "migration intensity",
        "migration characteristic distance",
        "migration number",
    ]
    data_symbols = ["$R_p$", "$T_p$", "$V_c$", "$S$", "$m_I$", "$m_D$", "$m$"]
    data_units = ["", "min.", "$\mu m$/min.", "", "1/min.", "$\mu m$", ""]
    ds_dicts = dict(list(zip(data_labels, data_sets)))
    ds_unit_dict = dict(list(zip(data_labels, data_units)))
    ds_symbol_dict = dict(list(zip(data_labels, data_symbols)))

    data_plot_order = [
        "average cell separation",
        "group persistence times",
        "group X velocity",
        "migration intensity",
    ]  # ["average cell separation", "group persistence ratios", "group persistence times", "group X velocity", "migration intensity"]

    num_rows = len(data_plot_order)
    for graph_type in ["box", "dot"]:
        fig_rows, axarr = plt.subplots(nrows=num_rows, sharex=True)
        last_index = num_rows - 1
        for i, ds_label in enumerate(data_plot_order):
            data = []
            if ds_label == "migration intensity":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                for v, a in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                ):
                    data.append(v / a)
            elif ds_label == "migration characteristic distance":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for j, v_t in enumerate(
                    zip(
                        group_velocities_per_experiment_per_repeat,
                        group_persistence_times_per_experiment_per_repeat,
                    )
                ):
                    v, t = v_t
                    if x_axis_stuff[j] == 1:
                        hide = np.nan
                    else:
                        hide = 1.0
                    data.append(hide * v * t)
            elif ds_label == "migration number":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for v, a, t in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                    group_persistence_times_per_experiment_per_repeat,
                ):
                    data.append(v * t / a)
            else:
                ds = ds_dicts[ds_label]
                data = [d for d in ds]

            if graph_type == "box":
                axarr[i].boxplot(data, showfliers=False)
                max_yticks = 3
                yloc = plt.MaxNLocator(max_yticks)
                axarr[i].yaxis.set_major_locator(yloc)
            else:
                axarr[i].errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )

            # axarr[i].set_title(ds_label)
            y_unit = ds_unit_dict[ds_label]
            if y_unit != "":
                axarr[i].set_ylabel("{} ({})".format(ds_symbol_dict[ds_label], y_unit))
            else:
                axarr[i].set_ylabel("{}".format(ds_symbol_dict[ds_label]))

            if i == last_index:
                if graph_type == "dot":
                    axarr[i].set_xticks(x_axis_positions)
                axarr[i].set_xlabel(x_label)
                axarr[i].set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label]
                + axarr[i].get_xticklabels()
                + axarr[i].get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            fig, ax = plt.subplots()
            if graph_type == "box":
                ax.boxplot(data, showfliers=False)
            else:
                ax.errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )
                ax.set_xticks(x_axis_positions)

            ax.set_title(ds_label)
            ax.set_ylabel(
                "{} ({})".format(ds_symbol_dict[ds_label], ds_unit_dict[ds_label])
            )
            ax.set_xlabel(x_label)
            ax.set_xlabel(x_label)
            ax.set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            show_or_save_fig(
                fig,
                (6, 6),
                save_dir,
                "cell_number_change_data",
                experiment_set_label + "_{}_{}".format(ds_label, graph_type),
            )

        show_or_save_fig(
            fig_rows,
            (12, 3 * len(data_plot_order)),
            save_dir,
            "cell_number_change_data",
            experiment_set_label + "_{}".format(graph_type),
        )


# =======================================================================


def graph_corridor_migration_parameter_test_data(
    labels,
    group_persistence_ratios,
    group_persistence_times,
    fit_group_x_velocities,
    cell_separations,
    areal_strains,
    experiment_set_label,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(2 * fontsize)
    # calculate_statistics_and_write_into_text_file(sub_experiment_number, test_num_cells, test_heights, group_persistence_ratios, fit_group_x_velocities, areal_strains, save_dir)

    x_axis_positions = [i + 1 for i in range(len(labels))]
    x_axis_stuff = labels
    # x_label = "corridor width (in cell diameters, N = {})".format(test_num_cells[0])

    # fit_group_x_velocities/(cell_separations*40)
    data_sets = [
        group_persistence_ratios,
        group_persistence_times,
        fit_group_x_velocities,
        cell_separations,
    ]
    data_labels = [
        "group persistence ratios",
        "group persistence times",
        "group X velocity",
        "average cell separation",
        "migration intensity",
        "migration characteristic distance",
        "migration number",
    ]
    data_symbols = ["$R_p$", "$T_p$", "$V_c$", "$S$", "$m_I$", "$m_D$", "$m$"]
    data_units = ["", "min.", "$\mu m$/min.", "", "1/min.", "$\mu m$", ""]
    ds_dicts = dict(list(zip(data_labels, data_sets)))
    ds_unit_dict = dict(list(zip(data_labels, data_units)))
    ds_symbol_dict = dict(list(zip(data_labels, data_symbols)))

    data_plot_order = [
        "average cell separation",
        "group persistence times",
        "group X velocity",
        "migration intensity",
    ]  # ["average cell separation", "group persistence ratios", "group persistence times", "group X velocity", "migration intensity"]

    num_rows = len(data_plot_order)
    for graph_type in ["box", "dot"]:
        fig_rows, axarr = plt.subplots(nrows=num_rows, sharex=True)
        last_index = num_rows - 1
        for i, ds_label in enumerate(data_plot_order):
            data = []
            if ds_label == "migration intensity":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                for v, a in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                ):
                    data.append(v / a)
            elif ds_label == "migration characteristic distance":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for j, v_t in enumerate(
                    zip(
                        group_velocities_per_experiment_per_repeat,
                        group_persistence_times_per_experiment_per_repeat,
                    )
                ):
                    v, t = v_t
                    if x_axis_stuff[j] == 1:
                        hide = np.nan
                    else:
                        hide = 1.0
                    data.append(hide * v * t)
            elif ds_label == "migration number":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for v, a, t in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                    group_persistence_times_per_experiment_per_repeat,
                ):
                    data.append(v * t / a)
            else:
                ds = ds_dicts[ds_label]
                data = [d for d in ds]

            if graph_type == "box":
                axarr[i].boxplot(data, showfliers=False)
                max_yticks = 3
                yloc = plt.MaxNLocator(max_yticks)
                axarr[i].yaxis.set_major_locator(yloc)
            else:
                axarr[i].errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )

            # axarr[i].set_title(ds_label)
            y_unit = ds_unit_dict[ds_label]
            if y_unit != "":
                axarr[i].set_ylabel("{} ({})".format(ds_symbol_dict[ds_label], y_unit))
            else:
                axarr[i].set_ylabel("{}".format(ds_symbol_dict[ds_label]))

            if i == last_index:
                if graph_type == "dot":
                    axarr[i].set_xticks(x_axis_positions)
                # axarr[i].set_xlabel(x_label)
                axarr[i].set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label]
                + axarr[i].get_xticklabels()
                + axarr[i].get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            fig, ax = plt.subplots()
            if graph_type == "box":
                ax.boxplot(data, showfliers=False)
            else:
                ax.errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )
                ax.set_xticks(x_axis_positions)

            ax.set_title(ds_label)
            ax.set_ylabel(
                "{} ({})".format(ds_symbol_dict[ds_label], ds_unit_dict[ds_label])
            )
            # ax.set_xlabel(x_label)
            # ax.set_xlabel(x_label)
            ax.set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            show_or_save_fig(
                fig,
                (6, 6),
                save_dir,
                "parameter_test_data",
                experiment_set_label + "_{}_{}".format(ds_label, graph_type),
            )

        show_or_save_fig(
            fig_rows,
            (12, 3 * len(data_plot_order)),
            save_dir,
            "parameter_test_data",
            experiment_set_label + "_{}".format(graph_type),
        )


# =======================================================================


def graph_coa_variation_test_data(
    sub_experiment_number,
    test_coas,
    default_cil,
    corridor_height,
    num_cells,
    group_persistence_ratios,
    group_persistence_times,
    fit_group_x_velocities,
    cell_separations,
    areal_strains,
    experiment_set_label,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(2 * fontsize)

    x_axis_positions = [i + 1 for i in range(len(test_coas))]
    x_axis_stuff = test_coas
    x_label = (
        "$M_{COA}$\n($M_{CIL}$"
        + "={}, corridor height = {} cell diam., {} cells)".format(
            default_cil, corridor_height, num_cells
        )
    )

    # fit_group_x_velocities/(cell_separations*40)
    data_sets = [
        group_persistence_ratios,
        group_persistence_times,
        fit_group_x_velocities,
        cell_separations,
    ]
    data_labels = [
        "group persistence ratios",
        "group persistence times",
        "group X velocity",
        "average cell separation",
        "migration intensity",
        "migration characteristic distance",
        "migration number",
    ]
    data_symbols = ["$R_p$", "$T_p$", "$V_c$", "$S$", "$m_I$", "$m_D$", "$m$"]
    data_units = ["", "min.", "$\mu m$/min.", "", "1/min.", "$\mu m$", ""]
    ds_dicts = dict(list(zip(data_labels, data_sets)))
    ds_unit_dict = dict(list(zip(data_labels, data_units)))
    ds_symbol_dict = dict(list(zip(data_labels, data_symbols)))

    data_plot_order = [
        "average cell separation",
        "group persistence times",
        "group X velocity",
        "migration intensity",
    ]  # ["average cell separation", "group persistence ratios", "group persistence times", "group X velocity", "migration intensity"]

    num_rows = len(data_plot_order)
    for graph_type in ["box", "dot"]:
        fig_rows, axarr = plt.subplots(nrows=num_rows, sharex=True)
        last_index = num_rows - 1
        for i, ds_label in enumerate(data_plot_order):
            data = []
            if ds_label == "migration intensity":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                for v, a in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                ):
                    data.append(v / a)
            elif ds_label == "migration characteristic distance":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for j, v_t in enumerate(
                    zip(
                        group_velocities_per_experiment_per_repeat,
                        group_persistence_times_per_experiment_per_repeat,
                    )
                ):
                    v, t = v_t
                    if x_axis_stuff[j] == 1:
                        hide = np.nan
                    else:
                        hide = 1.0
                    data.append(hide * v * t)
            elif ds_label == "migration number":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for v, a, t in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                    group_persistence_times_per_experiment_per_repeat,
                ):
                    data.append(v * t / a)
            else:
                ds = ds_dicts[ds_label]
                data = [d for d in ds]

            if graph_type == "box":
                axarr[i].boxplot(data, showfliers=False)
                max_yticks = 3
                yloc = plt.MaxNLocator(max_yticks)
                axarr[i].yaxis.set_major_locator(yloc)
            else:
                axarr[i].errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )

            # axarr[i].set_title(ds_label)
            y_unit = ds_unit_dict[ds_label]
            if y_unit != "":
                axarr[i].set_ylabel("{} ({})".format(ds_symbol_dict[ds_label], y_unit))
            else:
                axarr[i].set_ylabel("{}".format(ds_symbol_dict[ds_label]))

            if i == last_index:
                if graph_type == "dot":
                    axarr[i].set_xticks(x_axis_positions)
                axarr[i].set_xlabel(x_label)
                axarr[i].set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label]
                + axarr[i].get_xticklabels()
                + axarr[i].get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            fig, ax = plt.subplots()
            if graph_type == "box":
                ax.boxplot(data, showfliers=False)
            else:
                ax.errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )
                ax.set_xticks(x_axis_positions)

            ax.set_title(ds_label)
            ax.set_ylabel(
                "{} ({})".format(ds_symbol_dict[ds_label], ds_unit_dict[ds_label])
            )
            ax.set_xlabel(x_label)
            ax.set_xlabel(x_label)
            ax.set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            show_or_save_fig(
                fig,
                (6, 6),
                save_dir,
                "coa_variation_test",
                experiment_set_label + "_{}_{}".format(ds_label, graph_type),
            )

        show_or_save_fig(
            fig_rows,
            (12, 3 * len(data_plot_order)),
            save_dir,
            "coa_variation_test",
            experiment_set_label + "_{}".format(graph_type),
        )


# =======================================================================


def graph_cil_variation_test_data(
    sub_experiment_number,
    test_cils,
    default_coa,
    corridor_height,
    num_cells,
    group_persistence_ratios,
    group_persistence_times,
    fit_group_x_velocities,
    cell_separations,
    areal_strains,
    experiment_set_label,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(2 * fontsize)

    x_axis_positions = [i + 1 for i in range(len(test_cils))]
    x_axis_stuff = test_cils
    x_label = (
        "$M_{CIL}$\n($M_{COA}$"
        + "={}, corridor height = {} cell diam., {} cells)".format(
            default_coa, corridor_height, num_cells
        )
    )

    # fit_group_x_velocities/(cell_separations*40)
    data_sets = [
        group_persistence_ratios,
        group_persistence_times,
        fit_group_x_velocities,
        cell_separations,
    ]
    data_labels = [
        "group persistence ratios",
        "group persistence times",
        "group X velocity",
        "average cell separation",
        "migration intensity",
        "migration characteristic distance",
        "migration number",
    ]
    data_symbols = ["$R_p$", "$T_p$", "$V_c$", "$S$", "$m_I$", "$m_D$", "$m$"]
    data_units = ["", "min.", "$\mu m$/min.", "", "1/min.", "$\mu m$", ""]
    ds_dicts = dict(list(zip(data_labels, data_sets)))
    ds_unit_dict = dict(list(zip(data_labels, data_units)))
    ds_symbol_dict = dict(list(zip(data_labels, data_symbols)))

    data_plot_order = [
        "average cell separation",
        "group persistence times",
        "group X velocity",
        "migration intensity",
    ]  # ["average cell separation", "group persistence ratios", "group persistence times", "group X velocity", "migration intensity"]

    num_rows = len(data_plot_order)
    for graph_type in ["box", "dot"]:
        fig_rows, axarr = plt.subplots(nrows=num_rows, sharex=True)
        last_index = num_rows - 1
        for i, ds_label in enumerate(data_plot_order):
            data = []
            if ds_label == "migration intensity":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                for v, a in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                ):
                    data.append(v / a)
            elif ds_label == "migration characteristic distance":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for j, v_t in enumerate(
                    zip(
                        group_velocities_per_experiment_per_repeat,
                        group_persistence_times_per_experiment_per_repeat,
                    )
                ):
                    v, t = v_t
                    if x_axis_stuff[j] == 1:
                        hide = np.nan
                    else:
                        hide = 1.0
                    data.append(hide * v * t)
            elif ds_label == "migration number":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for v, a, t in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                    group_persistence_times_per_experiment_per_repeat,
                ):
                    data.append(v * t / a)
            else:
                ds = ds_dicts[ds_label]
                data = [d for d in ds]

            if graph_type == "box":
                axarr[i].boxplot(data, showfliers=False)
                max_yticks = 3
                yloc = plt.MaxNLocator(max_yticks)
                axarr[i].yaxis.set_major_locator(yloc)
            else:
                axarr[i].errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )

            # axarr[i].set_title(ds_label)
            y_unit = ds_unit_dict[ds_label]
            if y_unit != "":
                axarr[i].set_ylabel("{} ({})".format(ds_symbol_dict[ds_label], y_unit))
            else:
                axarr[i].set_ylabel("{}".format(ds_symbol_dict[ds_label]))

            if i == last_index:
                if graph_type == "dot":
                    axarr[i].set_xticks(x_axis_positions)
                axarr[i].set_xlabel(x_label)
                axarr[i].set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label]
                + axarr[i].get_xticklabels()
                + axarr[i].get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            fig, ax = plt.subplots()
            if graph_type == "box":
                ax.boxplot(data, showfliers=False)
            else:
                ax.errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )
                ax.set_xticks(x_axis_positions)

            ax.set_title(ds_label)
            ax.set_ylabel(
                "{} ({})".format(ds_symbol_dict[ds_label], ds_unit_dict[ds_label])
            )
            ax.set_xlabel(x_label)
            ax.set_xlabel(x_label)
            ax.set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            show_or_save_fig(
                fig,
                (6, 6),
                save_dir,
                "cil_variation_test",
                experiment_set_label + "_{}_{}".format(ds_label, graph_type),
            )

        show_or_save_fig(
            fig_rows,
            (12, 3 * len(data_plot_order)),
            save_dir,
            "cil_variation_test",
            experiment_set_label + "_{}".format(graph_type),
        )


# =======================================================================


def graph_vertex_choice_variation_test_data(
    test_vertex_choice_ratios_and_randomization_magnitudes,
    default_cil,
    default_coa,
    corridor_height,
    num_cells,
    group_persistence_ratios,
    group_persistence_times,
    fit_group_x_velocities,
    cell_separations,
    areal_strains,
    experiment_set_label,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(2 * fontsize)

    x_axis_positions = [
        i + 1
        for i in range(len(test_vertex_choice_ratios_and_randomization_magnitudes))
    ]
    x_axis_stuff = [
        "{}".format(tvr) + "\n$x_r$={}".format(trm)
        for tvr, trm in test_vertex_choice_ratios_and_randomization_magnitudes
    ]
    x_label = (
        "ratio of randomly selected vertices to total vertices\n$M_{CIL}$"
        + "={}".format(default_cil)
        + ",($M_{COA}$"
        + "={}, corridor height = {} cell diam., {} cells)".format(
            default_coa, corridor_height, num_cells
        )
    )

    # fit_group_x_velocities/(cell_separations*40)
    data_sets = [
        group_persistence_ratios,
        group_persistence_times,
        fit_group_x_velocities,
        cell_separations,
    ]
    data_labels = [
        "group persistence ratios",
        "group persistence times",
        "group X velocity",
        "average cell separation",
        "migration intensity",
        "migration characteristic distance",
        "migration number",
    ]
    data_symbols = ["$R_p$", "$T_p$", "$V_c$", "$S$", "$m_I$", "$m_D$", "$m$"]
    data_units = ["", "min.", "$\mu m$/min.", "", "1/min.", "$\mu m$", ""]
    ds_dicts = dict(list(zip(data_labels, data_sets)))
    ds_unit_dict = dict(list(zip(data_labels, data_units)))
    ds_symbol_dict = dict(list(zip(data_labels, data_symbols)))

    data_plot_order = [
        "average cell separation",
        "group persistence times",
        "group X velocity",
        "migration intensity",
    ]  # ["average cell separation", "group persistence ratios", "group persistence times", "group X velocity", "migration intensity"]

    num_rows = len(data_plot_order)
    for graph_type in ["box", "dot"]:
        fig_rows, axarr = plt.subplots(nrows=num_rows, sharex=True)
        last_index = num_rows - 1
        for i, ds_label in enumerate(data_plot_order):
            data = []
            if ds_label == "migration intensity":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                for v, a in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                ):
                    data.append(v / a)
            elif ds_label == "migration characteristic distance":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for j, v_t in enumerate(
                    zip(
                        group_velocities_per_experiment_per_repeat,
                        group_persistence_times_per_experiment_per_repeat,
                    )
                ):
                    v, t = v_t
                    if x_axis_stuff[j] == 1:
                        hide = np.nan
                    else:
                        hide = 1.0
                    data.append(hide * v * t)
            elif ds_label == "migration number":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for v, a, t in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                    group_persistence_times_per_experiment_per_repeat,
                ):
                    data.append(v * t / a)
            else:
                ds = ds_dicts[ds_label]
                data = [d for d in ds]

            if graph_type == "box":
                axarr[i].boxplot(data, showfliers=False)
                max_yticks = 3
                yloc = plt.MaxNLocator(max_yticks)
                axarr[i].yaxis.set_major_locator(yloc)
            else:
                axarr[i].errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )

            # axarr[i].set_title(ds_label)
            y_unit = ds_unit_dict[ds_label]
            if y_unit != "":
                axarr[i].set_ylabel("{} ({})".format(ds_symbol_dict[ds_label], y_unit))
            else:
                axarr[i].set_ylabel("{}".format(ds_symbol_dict[ds_label]))

            if i == last_index:
                if graph_type == "dot":
                    axarr[i].set_xticks(x_axis_positions)
                axarr[i].set_xlabel(x_label)
                axarr[i].set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label]
                + axarr[i].get_xticklabels()
                + axarr[i].get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            fig, ax = plt.subplots()
            if graph_type == "box":
                ax.boxplot(data, showfliers=False)
            else:
                ax.errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )
                ax.set_xticks(x_axis_positions)

            ax.set_title(ds_label)
            ax.set_ylabel(
                "{} ({})".format(ds_symbol_dict[ds_label], ds_unit_dict[ds_label])
            )
            ax.set_xlabel(x_label)
            ax.set_xlabel(x_label)
            ax.set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            show_or_save_fig(
                fig,
                (6, 6),
                save_dir,
                "vertex_choice_variation",
                experiment_set_label + "_{}_{}".format(ds_label, graph_type),
            )

        show_or_save_fig(
            fig_rows,
            (12, 3 * len(data_plot_order)),
            save_dir,
            "vertex_choice_variation",
            experiment_set_label + "_{}".format(graph_type),
        )


# =============================================================================


def draw_cell_arrangement(
    ax,
    origin,
    draw_space_factor,
    scale_factor,
    num_cells,
    box_height,
    box_width,
    corridor_height,
    box_y_placement_factor,
):
    bh = draw_space_factor * (box_height / scale_factor)
    ch = draw_space_factor * (corridor_height / scale_factor)
    bw = draw_space_factor * (box_width / scale_factor)
    cw = draw_space_factor * (1.2 * box_width / scale_factor)

    origin[0] = origin[0] - cw * 0.5

    corridor_boundary_coords = (
        np.array([[cw, 0.0], [0.0, 0.0], [0.0, ch], [cw, ch]], dtype=np.float64)
        + origin
    )

    corridor_boundary_patch = mpatch.Polygon(
        corridor_boundary_coords,
        closed=False,
        fill=False,
        color="r",
        ls="solid",
        clip_on=False,
    )
    ax.add_artist(corridor_boundary_patch)

    box_origin = origin + np.array([0.0, (ch - bh) * box_y_placement_factor])

    cell_radius = 0.5 * draw_space_factor * (1.0 / scale_factor)
    cell_placement_delta = cell_radius * 2
    y_delta = np.array([0.0, cell_placement_delta])
    x_delta = np.array([cell_placement_delta, 0.0])

    cell_origin = box_origin + np.array([cell_radius, cell_radius])
    y_delta_index = 0
    x_delta_index = 0

    for ci in range(num_cells):
        cell_patch = mpatch.Circle(
            cell_origin + y_delta_index * y_delta + x_delta_index * x_delta,
            radius=cell_radius,
            color="k",
            fill=False,
            ls="solid",
            clip_on=False,
        )
        ax.add_artist(cell_patch)

        if y_delta_index == box_height - 1:
            y_delta_index = 0
            x_delta_index += 1
        else:
            y_delta_index += 1


# ===================================================================


def graph_init_condition_change_data(
    sub_experiment_number,
    tests,
    group_persistence_ratios,
    group_persistence_times,
    fit_group_x_velocities,
    cell_separations,
    transient_end_times,
    experiment_set_label,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(2 * fontsize)

    x_axis_positions = [i + 1 for i in range(len(tests))]
    x_axis_stuff = []
    for nc_th_tw_ch_bpy_mpdf_ircpx in tests:
        nc, th, tw, ch, bpy, mpdf, ircpx = nc_th_tw_ch_bpy_mpdf_ircpx

        if th != "r":
            x_axis_stuff.append("regular: {}x{}".format(tw, th))
        else:
            x_axis_stuff.append("random: {} c.d.".format(mpdf))

    x_label = ""

    # fit_group_x_velocities/(cell_separations*40)
    data_sets = [
        group_persistence_ratios,
        group_persistence_times,
        fit_group_x_velocities,
        cell_separations,
    ]
    data_labels = [
        "group persistence ratios",
        "group persistence times",
        "group X velocity",
        "average cell separation",
        "migration intensity",
        "migration characteristic distance",
        "migration number",
    ]
    data_symbols = ["$R_p$", "$T_p$", "$V_c$", "$S$", "$m_I$", "$m_D$", "$m$"]
    data_units = ["", "min.", "$\mu m$/min.", "", "1/min.", "$\mu m$", ""]
    ds_dicts = dict(list(zip(data_labels, data_sets)))
    ds_unit_dict = dict(list(zip(data_labels, data_units)))
    ds_symbol_dict = dict(list(zip(data_labels, data_symbols)))

    data_plot_order = [
        "average cell separation",
        "group persistence times",
        "group X velocity",
        "migration intensity",
    ]  # ["average cell separation", "group persistence ratios", "group persistence times", "group X velocity", "migration intensity"]

    num_rows = len(data_plot_order)
    for graph_type in ["box", "dot"]:
        fig_rows, axarr = plt.subplots(nrows=num_rows, sharex=True)
        last_index = num_rows - 1
        for i, ds_label in enumerate(data_plot_order):
            data = []
            if ds_label == "migration intensity":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                for v, a in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                ):
                    data.append(v / a)
            elif ds_label == "migration characteristic distance":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for j, v_t in enumerate(
                    zip(
                        group_velocities_per_experiment_per_repeat,
                        group_persistence_times_per_experiment_per_repeat,
                    )
                ):
                    v, t = v_t
                    if x_axis_stuff[j] == 1:
                        hide = np.nan
                    else:
                        hide = 1.0
                    data.append(hide * v * t)
            elif ds_label == "migration number":
                group_velocities_per_experiment_per_repeat = ds_dicts[
                    "group X velocity"
                ]
                average_cell_separation_per_experiment_per_repeat = ds_dicts[
                    "average cell separation"
                ]
                group_persistence_times_per_experiment_per_repeat = ds_dicts[
                    "group persistence times"
                ]

                for v, a, t in zip(
                    group_velocities_per_experiment_per_repeat,
                    average_cell_separation_per_experiment_per_repeat,
                    group_persistence_times_per_experiment_per_repeat,
                ):
                    data.append(v * t / a)
            else:
                ds = ds_dicts[ds_label]
                data = [d for d in ds]

            if graph_type == "box":
                axarr[i].boxplot(data, showfliers=False)
                max_yticks = 3
                yloc = plt.MaxNLocator(max_yticks)
                axarr[i].yaxis.set_major_locator(yloc)
            else:
                axarr[i].errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )

            # axarr[i].set_title(ds_label)
            y_unit = ds_unit_dict[ds_label]
            if y_unit != "":
                axarr[i].set_ylabel("{} ({})".format(ds_symbol_dict[ds_label], y_unit))
            else:
                axarr[i].set_ylabel("{}".format(ds_symbol_dict[ds_label]))

            if i == last_index:
                if graph_type == "dot":
                    axarr[i].set_xticks(x_axis_positions)
                axarr[i].set_xlabel(x_label)
                axarr[i].set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label]
                + axarr[i].get_xticklabels()
                + axarr[i].get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            fig, ax = plt.subplots()
            if graph_type == "box":
                ax.boxplot(data, showfliers=False)
            else:
                ax.errorbar(
                    x_axis_positions,
                    [np.average(d) for d in data],
                    yerr=[
                        [abs(np.min(d) - np.average(d)) for d in data],
                        [abs(np.max(d) - np.average(d)) for d in data],
                    ],
                    marker="o",
                    ls="",
                )
                ax.set_xticks(x_axis_positions)

            ax.set_title(ds_label)
            ax.set_ylabel(
                "{} ({})".format(ds_symbol_dict[ds_label], ds_unit_dict[ds_label])
            )
            ax.set_xlabel(x_label)
            ax.set_xlabel(x_label)
            ax.set_xticklabels([str(j) for j in x_axis_stuff])

            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)

            show_or_save_fig(
                fig,
                (6, 6),
                save_dir,
                "init_conditions",
                experiment_set_label + "_{}_{}".format(ds_label, graph_type),
            )

        show_or_save_fig(
            fig_rows,
            (12, 3 * len(data_plot_order)),
            save_dir,
            "init_conditions_",
            experiment_set_label + "_{}".format(graph_type),
        )


# ==============================================================================


def plot_errorbar_graph(ax, x_data, y_data, marker, color, label, scheme):
    # avg_ys = np.average(y_data, axis=1)
    # avg_ys = np.average(y_data, axis=1)
    # min_ys = np.min(y_data, axis=1)
    # max_ys = np.max(y_data, axis=1)
    if scheme == "maximum":
        ax.plot(
            x_data,
            np.max(y_data, axis=1),
            marker=marker,
            color=color,
            label=label,
            ls="",
        )
        if marker == "o":
            average_marker = "*"
        else:
            average_marker = "+"
        ax.plot(
            x_data,
            np.average(y_data, axis=1),
            marker=average_marker,
            color=color,
            label=label,
            ls="",
        )
    elif scheme == "minimum":
        ax.plot(
            x_data,
            np.min(y_data, axis=1),
            marker=marker,
            color=color,
            label=label,
            ls="",
        )
    elif scheme == "average":
        ax.plot(
            x_data,
            np.average(y_data, axis=1),
            marker=marker,
            color=color,
            label=label,
            ls="",
        )
    else:
        raise Exception("Unknown scheme gotten: {}".format(scheme))

    # ax.errorbar(x_data, avg_ys, yerr=[np.abs(min_ys - avg_ys), np.abs(max_ys - avg_ys)], marker=marker, color=color, label=label)
    # ax.errorbar(x_data, avg_ys, yerr=np.std(y_data, axis=1), marker=marker, color=color, label=label, ls='')

    return ax


def graph_convergence_test_data(
    sub_experiment_number,
    test_num_nodes,
    cell_speeds,
    active_racs,
    active_rhos,
    inactive_racs,
    inactive_rhos,
    special_num_nodes=16,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(fontsize)
    for scheme in ["maximum", "minimum", "average"]:
        fig_speed, ax_speed = plt.subplots()
        fig_rgtp, ax_rgtp = plt.subplots()

        scheme_label = ""
        if scheme != "average":
            scheme_label = scheme[:3] + " "

        ax_speed = plot_errorbar_graph(
            ax_speed, test_num_nodes, cell_speeds, "o", "g", None, scheme
        )
        ax_speed.set_ylabel(
            "{}average cell speed ($\mu m$/min.)".format(scheme_label[:3])
        )
        ax_speed.set_xlabel("N")
        # ax_speed.axvline(x=special_num_nodes, color='m', label='N = {}'.format(special_num_nodes))
        ax_speed.grid(which="both")
        ax_speed.minorticks_on()

        ax_rgtp = plot_errorbar_graph(
            ax_rgtp, test_num_nodes, active_racs, "o", "b", "Rac1: active", scheme
        )
        # ax_rgtp = plot_errorbar_graph(ax_rgtp, test_num_nodes, inactive_racs, 'x', 'b', "Rac1: inactive", scheme)
        ax_rgtp = plot_errorbar_graph(
            ax_rgtp, test_num_nodes, active_rhos, "o", "r", "RhoA: active", scheme
        )
        # ax_rgtp = plot_errorbar_graph(ax_rgtp, test_num_nodes, inactive_rhos, 'x', 'r', "RhoA: inactive", scheme)
        ax_rgtp.set_ylabel("nodal concentration".format(scheme_label))
        ax_rgtp.set_xlabel("$N$")
        # ax_rgtp.axvline(x=special_num_nodes, color='m', label='N = {}'.format(special_num_nodes))
        ax_rgtp.grid(which="both")
        ax_rgtp.minorticks_on()

        for ax in [ax_speed, ax_rgtp]:
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontsize(fontsize)
            ax.legend(loc="best", fontsize=fontsize)

        show_or_save_fig(
            fig_speed, (12, 8), save_dir, "convergence_test", "speeds_{}".format(scheme)
        )
        show_or_save_fig(
            fig_rgtp,
            (12, 8),
            save_dir,
            "convergence_test",
            "rgtp_fractions_{}".format(scheme),
        )


def get_text_positions(text, x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = list(y_data)
    for index, (y, x) in enumerate(a):
        local_text_positions = [
            i
            for i in a
            if i[0] > (y - txt_height)
            and (abs(i[1] - x) < txt_width * 2)
            and i != (y, x)
        ]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height:  # True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height * 1.01
                for k, (j, m) in enumerate(differ):
                    # j is the vertical distance between words
                    if j > txt_height * 2:  # if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break

    return text_positions


def text_plotter(ax, text, x_data, y_data, text_positions, txt_width, txt_height):
    for z, x, y, t in zip(text, x_data, y_data, text_positions):
        ax.annotate(str(z), xy=(x - txt_width / 2, t), size=12)
        if y != t:
            ax.arrow(
                x,
                t,
                0,
                y - t,
                color="red",
                alpha=0.3,
                width=txt_width * 0.1,
                head_width=txt_width,
                head_length=txt_height * 0.5,
                zorder=0,
                length_includes_head=True,
            )


def graph_specific_convergence_test_data(
    num_timepoints,
    T,
    test_num_nodes,
    data_per_test_per_repeat_per_timepoint,
    fontsize,
    save_dir,
    data_name_and_unit,
):
    fig, ax = plt.subplots()
    set_fontsize(fontsize)
    timepoints = np.arange(num_timepoints) * T / 60.0
    text = []
    for i, nn in enumerate(test_num_nodes):
        data_per_cell_per_timepoint = data_per_test_per_repeat_per_timepoint[i][0]
        for j, marker in enumerate([".", "x"]):
            num_data_points = data_per_cell_per_timepoint[j].shape[0]
            ax.plot(
                timepoints[:num_data_points],
                data_per_cell_per_timepoint[j],
                marker=marker,
                ls="",
                color=colors.color_list300[i % 300],
            )
            ax.annotate(
                "NN={}".format(nn),
                xy=(
                    timepoints[:num_data_points][-1],
                    data_per_cell_per_timepoint[j][-1],
                ),
            )

    # txt_height = 0.0037*(ax.get_ylim()[1] - ax.get_ylim()[0])
    # txt_width = 0.018*(ax.get_xlim()[1] - ax.get_xlim()[0])

    # text_positions = get_text_positions(text, text_x, text_y, txt_width, txt_height)
    # text_plotter(ax, text, text_x, text_y, text_positions, txt_width, txt_height)

    name = ""
    if data_name_and_unit[1] != "":
        name, unit = data_name_and_unit
        ax.set_ylabel("{} ({})".format(name, unit))
    else:
        name, _ = data_name_and_unit
        ax.set_ylabel("{}".format(name))

    ax.set_xlabel("t (min.)")
    ax.grid(which="both")
    ax.minorticks_on()
    # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=fontsize)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)

    show_or_save_fig(fig, (14, 14), save_dir, "convergence_corridor_test", name)


# def graph_corridor_convergence_test_data(sub_experiment_number, num_timepoints, T, test_num_nodes, cell_positions_per_test_per_repeat_per_timepoint, cell_speeds_per_test_per_repeat_per_timepoint, active_racs_per_test_per_repeat_per_timepoint, active_rhos_per_test_per_repeat_per_timepoint, inactive_racs_per_test_per_repeat_per_timepoint, inactive_rhos_per_test_per_repeat_per_timepoint, special_num_nodes=16, save_dir=None, fontsize=22):
#
#    data_sets = [cell_positions_per_test_per_repeat_per_timepoint, cell_speeds_per_test_per_repeat_per_timepoint, active_racs_per_test_per_repeat_per_timepoint, active_rhos_per_test_per_repeat_per_timepoint, inactive_racs_per_test_per_repeat_per_timepoint, inactive_rhos_per_test_per_repeat_per_timepoint]
#    data_names_and_units = [("centroid positions", "$\mu m$"), ("speeds", "$\mu m$/min."), ("active Rac1 fraction", ""), ("active RhoA fraction", ""), ("inactive Rac1 fraction", ""), ("inactive RhoA fraction", "")]
#
##    for i, data_set in enumerate(data_sets):
##        data_name_and_unit = data_names_and_units[i]
##        graph_specific_convergence_test_data(num_timepoints, T, test_num_nodes, data_set, fontsize, save_dir, data_name_and_unit)
##
#    fig_speed, ax_speed = plt.subplots()
##    fig_rgtp, ax_rgtp = plt.subplots()
#
#    #(ax_speed, test_num_nodes, [np.average(cell_speeds_per_test_per_repeat_per_timepoint[:,0,k,:]) for k in range(2)], 'o', 'g', None, "average")
#    a = [np.average(cell_speeds_per_test_per_repeat_per_timepoint[n][0][0]) for n in range(len(test_num_nodes))]
#    b = [np.average(cell_speeds_per_test_per_repeat_per_timepoint[n][0][1]) for n in range(len(test_num_nodes))]
#    ax_speed.plot(test_num_nodes, [np.average([xa, xb]) for xa, xb in zip(a, b)], ls='', marker='.')
#    ax_speed.set_ylabel("average cell speed ($\mu m$/min.)")
#    ax_speed.set_xlabel("N")
#    #ax_speed.axvline(x=special_num_nodes, color='m', label='N = {}'.format(special_num_nodes))
#    ax_speed.grid(which=u'both')
#    ax_speed.minorticks_on()
#
#    show_or_save_fig(fig_speed, (12, 8), save_dir, "convergence_test", "speeds_{}".format("average"))


def graph_corridor_convergence_test_data(
    sub_experiment_number,
    test_num_nodes,
    cell_full_speeds,
    active_racs,
    active_rhos,
    inactive_racs,
    inactive_rhos,
    special_num_nodes=16,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(fontsize)
    fig_speed, ax_speed = plt.subplots()
    markers = ["o", "x"]
    cell_label = ["trailer", "leader"]
    for n in range(2):
        ax_speed.plot(
            test_num_nodes,
            [cf[n] for cf in cell_full_speeds],
            color="g",
            marker=markers[n],
            label="{}".format(cell_label[n]),
            ls="",
        )
    ax_speed.set_ylabel("avg. cell velocity ($\mu$m/min.)")
    ax_speed.set_xlabel("N")
    # ax_speed.axvline(x=special_num_nodes, color='m', label='N = {}'.format(special_num_nodes))
    ax_speed.grid(which="major")
    ax_speed.minorticks_on()
    ax_speed.legend(loc="best")

    fig_rgtpase, ax_rgtpase = plt.subplots()
    markers = ["o", "x"]
    for n in range(2):
        ax_rgtpase.plot(
            test_num_nodes,
            [cf[n] for cf in active_racs],
            color="b",
            marker=markers[n],
            label="active Rac1: {}".format(cell_label[n]),
            ls="",
        )
        ax_rgtpase.plot(
            test_num_nodes,
            [cf[n] for cf in active_rhos],
            color="r",
            marker=markers[n],
            label="active RhoA: {}".format(cell_label[n]),
            ls="",
        )
        # ax_rgtpase.plot(test_num_nodes, [cf[n] for cf in inactive_racs], color='c', marker=markers[n], label="inactive Rac1: cell {}".format(n), ls='')
        # ax_rgtpase.plot(test_num_nodes, [cf[n] for cf in inactive_rhos], color='m', marker=markers[n], label="inactive RhoA: cell {}".format(n), ls='')

    ax_rgtpase.set_ylabel("avg. RGTP concentration")
    ax_rgtpase.set_xlabel("N")
    # ax_speed.axvline(x=special_num_nodes, color='m', label='N = {}'.format(special_num_nodes))
    ax_rgtpase.grid(which="major")
    ax_rgtpase.minorticks_on()
    ax_rgtpase.legend(loc="best")

    for ax in [ax_speed, ax_rgtpase]:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(fontsize)

    show_or_save_fig(fig_speed, (12, 8), save_dir, "corridor_convergence", "speed")
    show_or_save_fig(fig_rgtpase, (12, 8), save_dir, "corridor_convergence", "rgtpase")


def graph_Tr_vs_Tp_test_data(
    sub_experiment_number,
    test_Trs,
    average_cell_persistence_times,
    save_dir=None,
    fontsize=22,
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    ax.plot(test_Trs, average_cell_persistence_times, marker="o", ls="")
    ax.set_xlabel("$T_r$")
    ax.set_ylabel("$T_p$")
    ax.grid(which="both")
    ax.minorticks_on()

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)

    show_or_save_fig(fig, (12, 8), save_dir, "Tr_vs_Tp", "line_plot")


# =============================================================================


def graph_combined_group_drifts(
    experiment_drift_args, base_name, experiment_set_label, save_dir=None
):
    num_experiments = len(experiment_drift_args)
    # s = np.sqrt(num_experiments)
    num_rows = num_experiments / 2
    num_cols = num_experiments / num_rows

    if num_experiments % 2 != 0:
        raise Exception(
            "Logic for dealing with uneven number of plots not yet complete!"
        )

    fig, axarr = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    max_ticks = 3
    yloc = plt.MaxNLocator(max_ticks)
    # xloc = plt.MaxNLocator(max_ticks)
    for i in range(num_rows):
        for j in range(num_cols):
            axarr[i, j].yaxis.set_major_locator(yloc)
            # axarr[i, j].xaxis.set_major_locator(xloc)
            axarr[i, j].grid(which="both")
    # ax = fig.add_subplot(111)
    #    colspan = 0
    #    if num_cols%2 == 0:
    #        gs = mgs.GridSpec(num_rows, num_cols)
    #        colspan = 1
    #        #sharex, sharey = True, True
    #    else:
    #        gs = mgs.GridSpec(num_rows, num_cols*2)
    #        colspan = 2
    #        #sharex, sharey = False, False
    #
    #
    #    axarr = []
    #    num_plots_placed = 0
    #    for i in range(num_rows):
    #        num_remaining_plots = num_experiments - num_plots_placed
    #        if  num_remaining_plots >= num_cols:
    #            for j in range(num_cols):
    #                axarr.append(plt.subplot(gs[i, j*colspan:(j + 1)*colspan]))
    #                num_plots_placed += 1
    #        else:
    #            num_side_cols = (num_cols*2 - num_remaining_plots*colspan)/2
    #            for k in range(num_remaining_plots):
    #                axarr.append(plt.subplots(gs[i, (num_side_cols + k*colspan):(num_side_cols + (k + 1)*colspan)]))
    #                num_plots_placed += 1
    #
    #    assert(num_experiments == num_plots_placed)

    fig = plt.gcf()
    # gs.tight_layout(fig)

    # present_collated_group_centroid_drift_data(T, cell_diameter, min_x_centroid_per_tstep_per_repeat, max_x_centroid_per_tstep_per_repeat, group_x_centroid_per_tstep_per_repeat, fit_group_x_velocity_per_repeat, save_dir, total_time_in_hours, fontsize=22, general_data_structure=None, ax_simple_normalized=None, ax_full_normalized=None, ax_simple=None, ax_full=None, plot_speedbox=True, plot_if_no_axis_given=True)

    # get the center position for all plots
    top = axarr[0, 0].get_position().y1
    bottom = axarr[-1, -1].get_position().y0
    left = axarr[0, 0].get_position().x1
    right = axarr[-1, -1].get_position().x0

    fig.canvas.draw()
    p = 0
    yy = 1
    xx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if p < num_experiments:
                present_collated_group_centroid_drift_data(
                    *experiment_drift_args[p],
                    ax_simple=axarr[i, j],
                    plot_speedbox=False,
                    plot_if_no_axis_given=False
                )
                axarr[i, j].set_xticks([0, 300, 600])
                axarr[i, j].set_xticklabels([0, 300, 600])
                p += 1
                bboxes_y, _ = axarr[i, j].yaxis.get_ticklabel_extents(
                    fig.canvas.renderer
                )
                bboxes_x, _ = axarr[i, j].xaxis.get_ticklabel_extents(
                    fig.canvas.renderer
                )

            for alabel, bboxes in zip(["x", "y"], [bboxes_x, bboxes_y]):
                bboxes = bboxes.inverse_transformed(fig.transFigure)

                if alabel == "y":
                    r = bboxes.x0
                    if r < yy:
                        yy = r
                else:
                    r = bboxes.y0
                    if r > xx:
                        xx = r

    tick_label_left = yy
    tick_label_down = xx

    ylabel = "$X_c$ ($\mu$m)"
    axarr[-1, -1].set_ylabel(ylabel)
    axarr[-1, -1].yaxis.set_label_coords(
        tick_label_left - 0.01, (bottom + top) / 2, transform=fig.transFigure
    )

    xlabel = "t (min.)"
    axarr[-1, -1].set_xlabel(xlabel)
    axarr[-1, -1].yaxis.set_label_coords(
        tick_label_down - 0.01, (left + right) / 2, transform=fig.transFigure
    )

    # ax.set_xlabel("t (min.)")
    # ax.set_ylabel("")

    show_or_save_fig(
        fig, (0.9 * 8.5, 8), save_dir, base_name, experiment_set_label + "_drifts"
    )


# =============================================================================


def graph_nonlin_to_lin_parameter_comparison(
    kgtp_rac_multipliers,
    kgtp_rho_multipliers,
    kgtp_rac_autoact_multipliers,
    kgtp_rho_autoact_multipliers,
    kdgtp_rac_multipliers,
    kdgtp_rho_multipliers,
    kdgtp_rho_mediated_rac_inhib_multipliers,
    kdgtp_rac_mediated_rho_inhib_multipliers,
    save_dir=None,
):
    num_sets = len(kgtp_rac_multipliers)
    fig, ax = plt.subplots()

    rac_rate_comparisons = []
    rho_rate_comparisons = []

    for (
        kgtp_rac,
        kgtp_rho,
        kgtp_rac_auto,
        kgtp_rho_auto,
        kdgtp_rac,
        kdgtp_rho,
        kdgtp_rho_on_rac,
        kdgtp_rac_on_rho,
    ) in zip(
        kgtp_rac_multipliers,
        kgtp_rho_multipliers,
        kgtp_rac_autoact_multipliers,
        kgtp_rho_autoact_multipliers,
        kdgtp_rac_multipliers,
        kdgtp_rho_multipliers,
        kdgtp_rho_mediated_rac_inhib_multipliers,
        kdgtp_rac_mediated_rho_inhib_multipliers,
    ):
        rac_rate_comparison = (kgtp_rac_auto) / (kgtp_rac)
        rho_rate_comparison = (kgtp_rho_auto) / (kgtp_rho)

        rac_rate_comparisons.append(rac_rate_comparison)
        rho_rate_comparisons.append(rho_rate_comparison)

    ax.plot(np.arange(num_sets), rac_rate_comparisons, color="b", ls="", marker=".")
    ax.plot(np.arange(num_sets), rho_rate_comparisons, color="r", ls="", marker=".")
    ax.set_ylim([0, 20])
    show_or_save_fig(fig, (6, 6), save_dir, "nonlin_vs_lin", "")


# ==========================================================================


def determine_intercellular_separations_after_first_collision(
    all_cell_centroids_per_repeat, cell_diameter, cutoff
):
    cell_intercellular_separations_after_first_collision = []

    for all_cell_centroids in all_cell_centroids_per_repeat:
        timestep_at_first_collision = -1

        ics_per_tstep = (
            np.abs(all_cell_centroids[0, :, 0] - all_cell_centroids[1, :, 0])
            / cell_diameter
        )

        for t, ics in enumerate(ics_per_tstep):
            if timestep_at_first_collision < 0:
                if ics < 1.0:
                    timestep_at_first_collision = t
                    break

        cell_intercellular_separations_after_first_collision.append(
            (
                timestep_at_first_collision,
                ics_per_tstep[
                    timestep_at_first_collision : (timestep_at_first_collision + cutoff)
                ]
                * cell_diameter,
            )
        )

    return cell_intercellular_separations_after_first_collision


def graph_intercellular_distance_after_first_collision(
    all_cell_centroids_per_repeat, T, cell_diameter, save_dir=None
):
    if all_cell_centroids_per_repeat[0].shape[0] != 2:
        raise Exception(
            "Logic for determining post-collision separation distances for more than 2 cells has not been implemented! all_cell_centroids_per_repeat shape: {}".format(
                all_cell_centroids_per_repeat.shape
            )
        )

    cell_intercellular_separations_after_first_collision = determine_intercellular_separations_after_first_collision(
        all_cell_centroids_per_repeat, cell_diameter, int(40.0 / T)
    )

    fig, ax = plt.subplots()
    for i, ics_after_first_collision in enumerate(
        cell_intercellular_separations_after_first_collision
    ):
        ts = np.arange(len(ics_after_first_collision[1])) * T
        ics = ics_after_first_collision[1]

        ax.plot(ts, ics, color=colors.color_list20[i % 20])

    ax.set_xlabel("t (min.)")
    ax.set_ylabel("centroid-to-centroid separation ($\mu$m)")

    show_or_save_fig(fig, (6, 6), save_dir, "post-collision-ics", "")

    fig, ax = plt.subplots()
    tpoint_of_interest = int(30.0 / T)
    cell_ics_at_tpoint_of_interest = [
        icinfo[1][tpoint_of_interest]
        for icinfo in cell_intercellular_separations_after_first_collision
    ]

    width = 0.35

    mean = np.average(cell_ics_at_tpoint_of_interest)
    std = np.std(cell_ics_at_tpoint_of_interest)
    rects1 = ax.bar(np.arange(1), mean, width, color="r", yerr=std)

    ax.set_ylabel("centroid-to-centroid separation\n30 min. after collision")
    ax.set_xticks(np.arange(1) + width / 2)
    ax.set_xticklabels((""))
    ax.set_title(
        "mean separation after 30 min.={} $\mu$m\nstd={} $\mu$m".format(
            np.round(mean, decimals=2), np.round(std, decimals=2)
        )
    )

    show_or_save_fig(fig, (6, 6), save_dir, "ics-at-30-min", "")

def generate_x_axis_titles_and_labels_for_varying_magnitudes(test_variants):
    x_labels = ["M={}".format(m) for m in test_variants]
    
    return ("chemotaxis magnitude ($M$)", x_labels)

def generate_x_axis_titles_and_labels_for_varying_parameters(test_variants, hide_scheme=True, hide_randomization_period_mean=False, hide_randomization_period_std=False, hide_randomization_magnitude=False, hide_randomization_node_percentage=True, hide_coa=False, hide_cil=False, hide_chemoattractant_magnitude_at_source=False):
    randomization_labels = ["scheme", "$T_R$", "SD($T_R$)", "R", "RVP"]
    other_labels = ["COA", "CIL", "M"]
    possible_data_labels = randomization_labels + other_labels
    possible_data_units = ["", " min.", "", "", "", "", "", ""]
    data_label_to_tag_dict = dict([(dl, tag) for dl, tag in zip(possible_data_labels, ["randomization_scheme", "randomization_time_mean", "randomization_time_variance_factor", "randomization_magnitude", "randomization_node_percentage", "coa_factor", "cil_factor", "chm"])])
    
    test_variants = [dict(tv) for tv in test_variants]
    
    data_labels_are_constant = []
    for data_label in possible_data_labels:
        data = []
        for tv in test_variants:
            x = None
            try:
                x = tv[data_label_to_tag_dict[data_label]]
            except:
                pass
            data.append(x)
            
        data_0 = data[0]
        constant = np.all([(d == data_0) for d in data])
        data_labels_are_constant.append(constant)
    
    data_labels_are_hidden = [hide_scheme, hide_randomization_period_mean, hide_randomization_period_std, hide_randomization_magnitude, hide_randomization_node_percentage, hide_coa, hide_cil, hide_chemoattractant_magnitude_at_source]
    
    x_axis_title_components = []
    x_labels_components = [[] for x in range(len(test_variants))]
    randomization_exists = [test_variants[x]["randomization_scheme"] != None for x in range(len(test_variants))]
    no_variants_with_randomization = not np.all(randomization_exists)
    
    for dl, dl_unit, constant, hidden in zip(possible_data_labels, possible_data_units, data_labels_are_constant, data_labels_are_hidden):
        if (dl == "scheme") and no_variants_with_randomization:
            x_axis_title_components.append("$-$rand")
        elif constant and not hidden:
            data = None
            try:
                data = test_variants[0][data_label_to_tag_dict[dl]]
            except:
                pass
            
            if data != None:
                x_axis_title_components.append("{}={}{}".format(dl, data, dl_unit))
        else:
            if not hidden:
                for vi in range(len(test_variants)):
                    if not no_variants_with_randomization and dl in randomization_labels and not randomization_exists[vi]:
                        if dl == "$T_R$":
                            x_labels_components[vi].append("$-$rand")
                        else:
                            continue
                    else:
                        try:
                            data = test_variants[vi][data_label_to_tag_dict[dl]]
                            x_labels_components[vi].append("{}={}".format(dl, data))
                        except:
                            pass
    
    x_axis_title = "(" + ", ".join(x_axis_title_components) + ")"
    x_labels = ["\n".join(xl_comps) if len(xl_comps) != 0 else "" for xl_comps in x_labels_components]
    
    return x_axis_title, x_labels

  
def generate_x_axis_titles_and_labels_for_varying_randomization_parameters(test_variants, hide_scheme=True, hide_node_percentage=True, hide_magnitude=False):
    possible_data_labels = ["scheme", "T", "SD(T)", "R", "NP"]
    data_label_to_tag_dict = dict([(dl, tag) for dl, tag in zip(possible_data_labels, ["randomization_scheme", "randomization_time_mean", "randomization_time_variance_factor", "randomization_magnitude", "randomization_node_percentage"])])
    
    test_variants = [dict(tv) for tv in test_variants]
    
    data_labels_are_constant = []
    for data_label in possible_data_labels:
        data = [tv[data_label_to_tag_dict[data_label]] for tv in test_variants]
        data_0 = data[0]
        constant = np.all([(d == data_0) for d in data])
        data_labels_are_constant.append(constant)
    
    data_labels_are_hidden = [hide_scheme, False, False, hide_magnitude, hide_node_percentage]
    
    x_axis_title_components = []
    x_labels_components = [[] for x in range(len(test_variants))]
    for dl, constant, hidden in zip(possible_data_labels, data_labels_are_constant, data_labels_are_hidden):
        if constant and not hidden:
            data = test_variants[0][data_label_to_tag_dict[dl]]
            x_axis_title_components.append("{}={}".format(dl, data))
        else:
            if not hidden:
                for vi in range(len(test_variants)):
                    data = test_variants[vi][data_label_to_tag_dict[dl]]
                    x_labels_components[vi].append("{}={}".format(dl, data))
    
    x_axis_title = "randomization parameters\n" + ", ".join(x_axis_title_components)
    x_labels = ["\n".join(xl_comps) if len(xl_comps) != 0 else "" for xl_comps in x_labels_components]
    
    return x_axis_title, x_labels

def graph_simple_chemotaxis_efficiency_data(
    test_variants,
    test_slope,
    chemotaxis_successes_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_widths,
    box_heights,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            chemotaxis_successes_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_widths,
            box_heights,
        )
    ):

        successes_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(successes_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], successes_per_variant, bar_width, label=label, color=requested_color)

    ax.set_ylabel("success:fail ratio")

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
        
    ax.set_xlabel(x_axis_title)

    ax.set_ylim([0.0, 1.05])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}simple_chemotaxis_efficiency_target_-{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_efficiency_data(
    test_variants,
    test_slope,
    chemotaxis_successes_per_variant_per_num_cells,
    chemotaxis_successes_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()

    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            chemotaxis_successes_per_variant_per_num_cells,
            chemotaxis_successes_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        successes_per_variant, successes_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(successes_per_variant))
            
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], successes_per_variant, bar_width, label=label, color=requested_color)


    ax.set_ylabel("success score")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)

    ax.set_ylim([0.0, 1.05])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_efficiency_target_-{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_group_persistence_time_data(
    test_variants,
    test_slope,
    chemotaxis_group_persistence_times_per_variant_per_num_cells,
    chemotaxis_group_persistence_times_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            chemotaxis_group_persistence_times_per_variant_per_num_cells,
            chemotaxis_group_persistence_times_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        gpt_per_variant, gpt_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(gpt_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], gpt_per_variant, bar_width, label=label, color=requested_color, yerr=gpt_std_per_variant)


    ax.set_ylabel("group persistence time (min.)")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    ax.set_ylim(bottom=0.0)
    
    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_group_persistence_time_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )

def graph_chemotaxis_cell_persistence_time_data(
    test_variants,
    test_slope,
    chemotaxis_cell_persistence_times_per_variant_per_num_cells,
    chemotaxis_cell_persistence_times_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            chemotaxis_cell_persistence_times_per_variant_per_num_cells,
            chemotaxis_cell_persistence_times_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        cpt_per_variant, cpt_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(cpt_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], cpt_per_variant, bar_width, label=label, color=requested_color, yerr=cpt_std_per_variant)

    ax.set_ylabel("cell persistence times (min.)")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    ax.set_ylim(bottom=0.0)
    
    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_cell_persistence_times_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_protrusion_lifetime_data(
    test_variants,
    test_slope,
    chemotaxis_protrusion_lifetime_per_variant_per_num_cells,
    chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            chemotaxis_protrusion_lifetime_per_variant_per_num_cells,
            chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        plt_per_variant, plt_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(plt_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], plt_per_variant, bar_width, label=label, color=requested_color, yerr=plt_std_per_variant)

    ax.set_ylabel("protrusion lifetimes (min.)")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_protrusion_lifetime_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_protrusion_lifetime_data_new(
    test_variants,
    test_slope,
    chemotaxis_protrusion_lifetime_per_variant_per_num_cells,
    chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    bar_width = 0.6
    
    ax.bar([0, 1], [chemotaxis_protrusion_lifetime_per_variant_per_num_cells[0][0], chemotaxis_protrusion_lifetime_per_variant_per_num_cells[1][0]], bar_width, label="nc={}".format(num_cells[0]), color='0.5', yerr=[chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells[0][0], chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells[1][0]])
    
    ax.bar([2, 3], [chemotaxis_protrusion_lifetime_per_variant_per_num_cells[0][1], chemotaxis_protrusion_lifetime_per_variant_per_num_cells[1][1]], bar_width, label="nc={}".format(num_cells[1]), color='0.0', yerr=[chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells[0][1], chemotaxis_protrusion_lifetime_std_per_variant_per_num_cells[1][1]])

    ax.set_ylabel("protrusion lifetimes (min.)")
    ax.set_xticks([0, 1, 2, 3])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    ax.set_xticklabels(["-\n(M=0.0)", "+\n(M=7.5)", "-\n(M=0.0)", "+\n(M=7.5)"])
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_protrusion_lifetime_new_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_group_x_speed_data(
    test_variants,
    test_slope,
    group_x_speed_per_variant_per_num_cells,
    group_x_speed_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            group_x_speed_per_variant_per_num_cells,
            group_x_speed_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        gxs_per_variant, gxs_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(gxs_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], gxs_per_variant, bar_width, label=label, color=requested_color, yerr=gxs_std_per_variant)

    ax.set_ylabel("group x speed ($\mu$m/min.)")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_group_x_speed_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_group_speed_data(
    test_variants,
    test_slope,
    group_speed_per_variant_per_num_cells,
    group_speed_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            group_speed_per_variant_per_num_cells,
            group_speed_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        gs_per_variant, gs_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(gs_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], gs_per_variant, bar_width, label=label, color=requested_color, yerr=gs_std_per_variant)

    ax.set_ylabel("group speed ($\mu$m/min.)")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_group_speed_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_speed_data_new(
    test_variants,
    test_slope,
    group_speed_per_num_cells,
    group_speed_std_per_num_cells,
    group_speed_x_per_num_cells,
    group_speed_x_std_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    #ax.plot(num_cells, group_speed_per_num_cells, marker=".", label="group speed")
    xticks = [x for x in range(len(num_cells))]
    ax.errorbar(xticks, group_speed_per_num_cells, yerr=group_speed_std_per_num_cells, label="group speed", markersize=10, linewidth=3, marker='o', capsize=10, capthick=3)
    #ax.plot(num_cells, group_speed_x_per_num_cells, marker=".", label="group speed (x-direction)")
    ax.errorbar(xticks, group_speed_x_per_num_cells, yerr=group_speed_x_std_per_num_cells, label="group speed (x-direction)", markersize=10, linewidth=3, marker='o', capsize=10, capthick=3)

    ax.set_ylabel("speed ($\mu$m/min.)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(num_cells)
    ax.set_xlabel("cluster size")
    
    ax.set_ylim(bottom=0.0)
    
    #ax.grid(b=True, which="major", axis="y")
    ax.legend(loc="best")

    current_time = datetime.datetime.now()

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    
    show_or_save_fig(
        fig,
        (11, 8),
        save_dir,
        "{}{}chemotaxis_group_speed_new_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second
        ),
        "",
    )
        
    with open(os.path.join(save_dir, "{}{}chemotaxis_group_speed_new_text_output_{}-{}-{}-{}-{}-{}.txt".format(
            info_tag, info_tag_dash, current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second
        )), 'w') as f:
        f.write("(number of cells, group speed): {}".format([e for e in zip(num_cells, [np.round(e, 2) for e in group_speed_per_num_cells])]) + "\n")
        f.write("(number of cells, group x-speed): {}".format([e for e in zip(num_cells, [np.round(e, 2) for e in group_speed_x_per_num_cells])]) + "\n")
        
def graph_chemotaxis_group_persistence_ratio_data(
    test_variants,
    test_slope,
    group_persistence_ratio_per_variant_per_num_cells,
    group_persistence_ratio_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            group_persistence_ratio_per_variant_per_num_cells,
            group_persistence_ratio_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        gpr_per_variant, gpr_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(gpr_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], gpr_per_variant, bar_width, label=label, color=requested_color, yerr=gpr_std_per_variant)

    ax.set_ylabel("group persistence ratio")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_group_persistence_ratio_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_cell_persistence_ratio_data(
    test_variants,
    test_slope,
    cell_persistence_ratio_per_variant_per_num_cells,
    cell_persistence_ratio_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            cell_persistence_ratio_per_variant_per_num_cells,
            cell_persistence_ratio_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        cpr_per_variant, cpr_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(cpr_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], cpr_per_variant, bar_width, label=label, color=requested_color, yerr=cpr_std_per_variant)

    ax.set_ylabel("cell persistence ratio")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_cell_persistence_ratio_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_velocity_alignment_data(
    test_variants,
    test_slope,
    velocity_alignment_per_variant_per_num_cells,
    velocity_alignment_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            velocity_alignment_per_variant_per_num_cells,
            velocity_alignment_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        va_per_variant, va_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(va_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], va_per_variant, bar_width, label=label, color=requested_color, yerr=va_std_per_variant)

    ax.set_ylabel("velocity alignment")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_velocity_alignment_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )
        
def graph_chemotaxis_protrusion_production_data(
    test_variants,
    test_slope,
    protrusion_production_per_variant_per_num_cells,
    protrusion_production_std_per_variant_per_num_cells,
    num_experiment_repeats,
    num_cells,
    box_width,
    box_height,
    save_dir=None,
    fontsize=22,
    info_tag="",
):
    set_fontsize(fontsize)
    fig, ax = plt.subplots()
    
    per_variant_space = 1.0
    
    if len(test_variants) == 1:
        between_variant_space = 0.0
    else:
        between_variant_space = 0.4
        
    num_cell_groups_per_variant = len(num_cells)
    within_variant_total_bar_space = 0.75*per_variant_space
    bar_width = within_variant_total_bar_space/num_cell_groups_per_variant
    between_bar_space = (per_variant_space - within_variant_total_bar_space)/(num_cell_groups_per_variant + 1)
    within_variant_initial_offset = between_bar_space + 0.5*bar_width

    for group_index, setup_data in enumerate(
        zip(
            protrusion_production_per_variant_per_num_cells,
            protrusion_production_std_per_variant_per_num_cells,
            num_experiment_repeats,
            num_cells,
            box_width,
            box_height,
        )
    ):

        pp_per_variant, pp_std_per_variant, nr, nc, bw, bh = setup_data

        requested_color = colors.color_list20[group_index]
        
        variant_indices = np.arange(len(pp_per_variant))
        
        this_group_initial_offset = 0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*group_index
        
        if len(test_variants) == 1:
            label = None
        else:
            label = "nc={}, nr={}".format(nc, nr)
            
        ax.bar([this_group_initial_offset + (per_variant_space + between_variant_space)*j for j in variant_indices], pp_per_variant, bar_width, label=label, color=requested_color, yerr=pp_std_per_variant)


    ax.set_ylabel("protrusion production (per cell*hour)")
    ax.set_xticks([0.5*between_variant_space + 0.5*per_variant_space + i*(per_variant_space + between_variant_space) for i in range(len(test_variants))])

    x_axis_title, x_labels = generate_x_axis_titles_and_labels_for_varying_parameters(test_variants)
    
    ax.set_ylim(bottom=0.0)
    
    if len(test_variants) == 1:
        ax.set_xticks([0.5*between_variant_space + within_variant_initial_offset + (bar_width + between_bar_space)*i for i in range(num_cell_groups_per_variant)])
        ax.set_xticklabels(["nc={}".format(nc) for nc in num_cells])
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_axis_title)
    
    ax.grid(b=True, which="major", axis="y")

    ax.legend(loc="upper center", ncol=int(len(num_cells)/2), fontsize=fontsize, bbox_to_anchor=(0.5, 1.25))

    info_tag_dash  = ""
    if info_tag != "":
        info_tag_dash = "-"
    x = datetime.datetime.now()
    show_or_save_fig(
        fig,
        (20, 10),
        save_dir,
        "{}{}chemotaxis_protrusion_production_{}-{}-{}-{}-{}-{}".format(
            info_tag, info_tag_dash, x.year, x.month, x.day, x.hour, x.minute, x.second
        ),
        "",
    )


def convert_rgb_a_to_rgb(rgb_bg, rgb_color, a):
    return [(1 - a) * rgb_bg[i] + a * rgb_color[i] for i in range(3)]
