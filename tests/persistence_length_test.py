# -*- coding: utf-8 -*-
"""
Created on Tue May 09 17:00:45 2017

@author: Brian
"""

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


@nb.jit(nopython=True)
def calculate_cos_theta_for_direction_autocorr_coeffs(a, b):
    ax, ay = a
    bx, by = b

    norm_a = np.sqrt(ax * ax + ay * ay)
    norm_b = np.sqrt(bx * bx + by * by)

    if norm_a < 1e-6:
        a = np.random.rand(2)
        norm_a = np.sqrt(ax * ax + ay * ay)

    if norm_b < 1e-6:
        b = np.random.rand(2)
        norm_b = np.sqrt(bx * bx + by * by)

    ax_, ay_ = a / norm_a
    bx_, by_ = b / norm_b

    return ax_ * bx_ + ay_ * by_


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

        all_das[n] = da

    return all_das, all_das[: 2 * first_negative_n]


@nb.jit(nopython=True)
def generate_displacements(
    persistence_time,
    num_displacements,
    timestep_bw_displacements,
    avg_displacement_magnitude,
):
    displacements = np.zeros((num_displacements, 2), dtype=np.float64)

    theta = np.random.rand() * 2 * np.pi
    delta_t = 0.0
    for n in range(num_displacements):
        delta_t += timestep_bw_displacements
        this_step_size = avg_displacement_magnitude
        displacements[n][0] = this_step_size * np.cos(theta)
        displacements[n][1] = this_step_size * np.sin(theta)

        if np.random.rand() < 1.0 - np.exp(-delta_t / persistence_time):
            theta = np.random.rand() * 2 * np.pi
            delta_t = 0.0

    return displacements


@nb.jit(nopython=True)
def calculate_positions(displacements):
    num_displacements = displacements.shape[0]
    positions = np.zeros((num_displacements + 1, 2), dtype=np.float64)

    for x in range(num_displacements):
        prev_pos = positions[x]
        positions[x + 1] = prev_pos + displacements[x]

    return positions


def estimate_persistence_time(timestep, positive_das):
    ts = np.arange(positive_das.shape[0]) * timestep
    popt, pcov = sp.optimize.curve_fit(
        lambda t, pt: np.exp(-1.0 * t / pt), ts, positive_das
    )
    pt = popt[0]

    return pt, ts


persistence_time = 40.0
timestep_bw_displacements = 2.0 / 60.0
avg_displacement_magnitude = 3.0 * timestep_bw_displacements
num_displacements = 10000

displacements = generate_displacements(
    persistence_time,
    num_displacements,
    timestep_bw_displacements,
    avg_displacement_magnitude,
)
positions = calculate_positions(displacements)
all_das, positive_das = calculate_direction_autocorr_coeffs_for_persistence_time(
    displacements
)
pt, ts = estimate_persistence_time(timestep_bw_displacements, positive_das)

fig_traj, ax_traj = plt.subplots()
ax_traj.plot(positions[:, 0], positions[:, 1])
ax_traj.set_aspect("equal")
max_data_lim = 1.1 * np.max(np.abs(positions))
ax_traj.set_xlim(-1 * max_data_lim, 1 * max_data_lim)
ax_traj.set_ylim(-1 * max_data_lim, 1 * max_data_lim)


fig_das, ax_das = plt.subplots()
all_ts = np.arange(displacements.shape[0]) * timestep_bw_displacements

ax_das.plot(all_ts, all_das, color="b", marker=".", ls="")

fig_fit, ax_fit = plt.subplots()
ax_fit.plot(ts, positive_das, color="g", marker=".", ls="")
ax_fit.plot(ts, [np.exp(-1 * x / pt) for x in ts], color="r", marker=".", ls="")

plt.show()
print("pt: ", pt)
