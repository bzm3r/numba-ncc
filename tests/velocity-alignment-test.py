# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:58:18 2019

@author: Brian
"""

import core.geometry as geometry

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

# =================================================================
#@nb.jit(nopython=True)
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


# =================================================================
#@nb.jit(nopython=True)
def find_snapshot_index(unique_snapshot_timesteps, timestep):
    for i in range(unique_snapshot_timesteps.shape[0]):
        if unique_snapshot_timesteps[i] == timestep:
            return i


# =================================================================
#@nb.jit(nopython=True)
def calculate_velocity_alignments(
    time_deltas, distance_radii, all_cell_centroids_per_tstep, T, L
):
    num_drs = distance_radii.shape[0]
    num_tds = time_deltas.shape[0]
    num_timesteps = all_cell_centroids_per_tstep.shape[1]
    num_cells = all_cell_centroids_per_tstep.shape[0]

    num_unique_snapshots = 0
    unique_snapshot_timesteps = np.zeros(num_timesteps, dtype=np.int64)
    num_snapshots_per_td = np.zeros(num_tds, dtype=np.int64)
    tds = np.zeros(num_tds, dtype=np.int64)
    for xi in range(num_tds):
        td = int(np.round(time_deltas[xi] * 60.0 / T))
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
        (num_drs, num_tds, num_cells * num_cells * np.max(num_snapshots_per_td)),
        dtype=np.float64,
    )
    num_data_points_per_case = np.zeros((num_drs, num_tds), dtype=np.int64)

    for xi in range(num_drs):
        dr = distance_radii[xi] / L
        for yi in range(num_tds):
            num_snapshots = num_snapshots_per_td[yi]
            num_va_data = 0
            va_data = np.zeros(num_cells * num_cells * num_snapshots, dtype=np.float64)
            td = tds[yi]
            for si in range(num_snapshots - 1):
                et = int(td / 2) + (si + 1) * td
                cell_centroids_st = all_cell_centroids_per_tstep[:, et - td, :]
                cell_centroids_et = all_cell_centroids_per_tstep[:, et, :]
                cell_displacements = cell_centroids_et - cell_centroids_st
                cell_directions = geometry.multiply_vectors_by_scalars(
                    cell_displacements,
                    1.0 / geometry.calculate_2D_vector_mags(cell_displacements),
                )
                snapshot_index = find_snapshot_index(unique_snapshot_timesteps, et)
                relevant_intercellular_distance_matrix = intercellular_distance_matrices[
                    int(snapshot_index)
                ]

                for focus_ci in range(num_cells):
                    num_relevant_cells = 0
                    relevant_cell_indices = np.zeros(num_cells, dtype=np.int64)
                    relevant_intercellular_distances = relevant_intercellular_distance_matrix[
                        focus_ci
                    ]
                    focus_ci_direction = cell_directions[focus_ci]

                    for other_ci in range(num_cells):
                        if other_ci == focus_ci:
                            continue
                        else:
                            if relevant_intercellular_distances[other_ci] <= dr:
                                relevant_cell_indices[num_relevant_cells] = other_ci
                                num_relevant_cells += 1
                            else:
                                continue

                    for other_ci in relevant_cell_indices[:num_relevant_cells]:
                        va_data[num_va_data] = np.dot(
                            cell_directions[other_ci], focus_ci_direction
                        )
                        num_va_data += 1

            velocity_alignments[xi][yi][:num_va_data] = va_data[:num_va_data]
            num_data_points_per_case[xi][yi] = num_va_data

    return velocity_alignments, num_data_points_per_case


#@nb.jit(nopython=True)
def calculate_correlation_coefficient(xs, ys):
    avg_xs = np.sum(xs) / xs.shape[0]
    avg_ys = np.sum(ys) / ys.shape[0]

    delta_avg_xs = xs - avg_xs
    delta_avg_ys = ys - avg_ys
    delta_avg_product = delta_avg_xs * delta_avg_ys
    avg_delta_avg = np.sum(delta_avg_product) / delta_avg_product.shape[0]

    std_xs = np.sqrt(np.dot(delta_avg_xs, delta_avg_xs))
    std_ys = np.sqrt(np.dot(delta_avg_ys, delta_avg_ys))

    return avg_delta_avg / (std_xs * std_ys)


# #@nb.jit(nopython=True)
# def calculate_direction_correlations(time_deltas, distance_radii, all_cell_centroids_per_tstep, T, L):
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
#    intercellular_distance_matrices = np.zeros((num_unique_snapshots, num_cells, num_cells), dtype=np.float64)
#    for zi in range(num_unique_snapshots):
#        timestep = unique_snapshot_timesteps[zi]
#        cell_centroids_at_tstep = all_cell_centroids_per_tstep[:, timestep, :]
#        intercellular_distance_matrices[zi] = calculate_intercellular_distance_matrix(cell_centroids_at_tstep)
#
#    direction_correlations = np.zeros((num_drs, num_tds, np.max(num_snapshots_per_td)), dtype=np.float64)
#    num_data_points_per_case = np.zeros((num_drs, num_tds), dtype=np.int64)
#
#    for xi in range(num_drs):
#        dr = distance_radii[xi]/L
#        for yi in range(num_tds):
#            num_snapshots = num_snapshots_per_td[yi]
#            dc_data = np.zeros((num_snapshots, num_cells), dtype=np.float64)
#            td = tds[yi]
#
#            for si in range(num_snapshots - 1):
#                et = int(td/2) + (si + 1)*td
#                cell_centroids_st = all_cell_centroids_per_tstep[:, et - td, :]
#                cell_centroids_et = all_cell_centroids_per_tstep[:, et, :]
#                cell_displacements = cell_centroids_et - cell_centroids_st
#                cell_directions = geometry.calculate_2D_vector_directions(geometry.normalize_vectors(cell_displacements))
#                snapshot_index = find_snapshot_index(unique_snapshot_timesteps, et)
#                relevant_intercellular_distance_matrix = intercellular_distance_matrices[snapshot_index]
#
#                for focus_ci in range(num_cells):
#                    num_relevant_cells = 0
#                    relevant_cell_indices = np.zeros(num_cells, dtype=np.int64)
#
#                    relevant_intercellular_distances = relevant_intercellular_distance_matrix[focus_ci]
#
#                    for other_ci in range(num_cells):
#                        if other_ci == focus_ci or relevant_intercellular_distances[other_ci] <= dr:
#                            relevant_cell_indices[num_relevant_cells] = focus_ci
#                            num_relevant_cells += 1
#
#                    relevant_cell_directions = np.zeros(num_relevant_cells, dtype=np.float64)
#                    for xi in range(num_relevant_cells):
#                        relevant_cell_directions[xi] = cell_directions[relevant_cell_indices[xi]]
#
#                    dc_data[si][focus_ci] = calculate_correlation_coefficient()
#
#
#
#
#
#            velocity_alignments[xi][yi][:num_va_data] = va_data[:num_va_data]
#            num_data_points_per_case[xi][yi] = num_va_data
#
#
#    return velocity_alignments, num_data_points_per_case

# =================================================================
# #@nb.jit(nopython=True)
def find_index_of_max_element_in_velocity_alignment_matrix(
    num_x, num_y, va_matrix, num_va_data_points
):
    average_va_matrix = np.zeros((num_x, num_y), dtype=np.float64)
    for xi in range(num_x):
        for yi in range(num_y):
            average_va_matrix[xi][yi] = np.average(
                va_matrix[xi][yi][: (num_va_data_points[xi][yi])]
            )

    va_max = 0.0
    va_max_xi = 0
    va_max_yi = 0

    for xi in range(num_x):
        for yi in range(num_y):
            if average_va_matrix[xi][yi] > va_max:
                va_max_xi = xi
                va_max_yi = yi

    return va_max, va_max_xi, va_max_yi


# =================================================================
# #@nb.jit(nopython=True)
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


# =================================================================


def generate_random_direction_vector():
    return geometry.normalize_2D_vector(np.random.rand(2))


def dynamics_random_velocity_aligned_motion(
    last_positions,
    current_positions,
    velocity_alignment_factor,
    velocity_alignment_radius,
    initial=False,
):
    new_positions = np.zeros_like(current_positions)
    intercellular_distance_matrix = calculate_intercellular_distance_matrix(
        initial_centroids
    )
    velocities = np.zeros_like(current_positions)

    num_cells = current_positions.shape[0]
    for ci in range(num_cells):
        if initial:
            velocities[ci] = generate_random_direction_vector()
        else:
            velocities[ci] = geometry.normalize_2D_vector(
                current_positions[ci] - last_positions[ci]
            )

    for ci in range(num_cells):
        this_cell_velocity = velocities[ci]
        relevant_icds = intercellular_distance_matrix[ci]

        num_relevant_neigbours = 0
        relevant_neighbour_velocities = np.zeros((num_cells - 1, 2), dtype=np.float64)
        for other_ci in range(initial_centroids.shape[0]):
            if other_ci == ci:
                continue
            else:
                d_to_other = relevant_icds[other_ci]
                if d_to_other < velocity_alignment_radius:
                    relevant_neighbour_velocities[num_relevant_neigbours] = velocities[
                        other_ci
                    ]
                    num_relevant_neigbours += 1

        relevant_neighbour_velocities = relevant_neighbour_velocities[
            :num_relevant_neigbours
        ]

        if relevant_neighbour_velocities.shape[0] < 1:
            velocity_alignment_effect = np.zeros(2, dtype=np.float64)
        else:
            velocity_alignment_effect = velocity_alignment_factor * np.average(
                relevant_neighbour_velocities, axis=0
            )

        random_theta = np.random.rand() * 2 * np.pi
        new_velocity = geometry.normalize_2D_vector(
            velocity_alignment_effect
            + 0.7 * np.array([np.cos(random_theta), np.sin(random_theta)])
            + this_cell_velocity
        )
        if initial:
            new_positions[ci] = current_positions[ci] + velocities[ci]
        else:
            new_positions[ci] = current_positions[ci] + new_velocity

    return new_positions


def generate_cell_centroids_per_tstep(
    initial_centroids, tsteps, velocity_alignment_factor, velocity_alignment_radius
):
    cell_centroids_per_tstep = np.zeros(
        (initial_centroids.shape[0], tsteps, initial_centroids.shape[1]),
        dtype=np.float64,
    )

    for tstep in range(tsteps - 1):
        if tstep == 0:
            cell_centroids_per_tstep[
                :, tstep, :
            ] = dynamics_random_velocity_aligned_motion(
                initial_centroids,
                initial_centroids,
                velocity_alignment_factor,
                velocity_alignment_radius,
                initial=True,
            )
        else:
            cell_centroids_per_tstep[
                :, tstep + 1, :
            ] = dynamics_random_velocity_aligned_motion(
                cell_centroids_per_tstep[:, tstep - 1, :],
                cell_centroids_per_tstep[:, tstep, :],
                velocity_alignment_factor,
                velocity_alignment_radius,
            )

    return cell_centroids_per_tstep


side_length = 7
initial_centroids = np.reshape(
    np.array(
        [[[1.0 * i, 0.0 + j] for j in range(side_length)] for i in range(side_length)]
    ),
    (side_length * side_length, 2),
)
tsteps = 200

test_velocity_alignments = [1.0]  # [0.05, 0.25, 0.5, 0.75, 0.95]
cell_centroids_per_tstep_per_va = [
    generate_cell_centroids_per_tstep(initial_centroids, tsteps, 0.8, 10.0)
    for va in test_velocity_alignments
]

fig, ax = plt.subplots()
num_cells = initial_centroids.shape[0]
for ci in range(num_cells):
    ax.plot(
        cell_centroids_per_tstep_per_va[0][ci, :, 0],
        cell_centroids_per_tstep_per_va[0][ci, :, 1],
    )

ax.set_aspect("equal")

velocity_alignments, num_data_points_per_case = calculate_velocity_alignments(
    np.array([1.0]),
    np.array([1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 1000.0]),
    cell_centroids_per_tstep_per_va[0],
    60.0,
    1.0,
)

velocity_alignments_average = np.zeros(
    (velocity_alignments.shape[0], velocity_alignments.shape[1]), dtype=np.float64
)

for i in range(velocity_alignments.shape[0]):
    for j in range(velocity_alignments.shape[1]):
        num_data_points = num_data_points_per_case[i][j]

        if num_data_points > 20:
            velocity_alignments_average[i][j] = np.average(
                velocity_alignments[i][j][:num_data_points]
            )
        else:
            velocity_alignments_average[i][j] = 0.0

fig1, ax1 = plt.subplots()
ax1.plot(
    np.array([1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 1000.0]),
    velocity_alignments_average[:, 0],
)

plt.show()
