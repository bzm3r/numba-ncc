# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 04:16:42 2017

@author: Brian
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_cell_arrangement(
    ax,
    cell_center_locations,
    cell_diameter,
    box_height,
    box_width,
    corridor_origin,
    corridor_height,
    box_y_placement_factor,
):

    corridor_width = 1.2 * box_width

    cw = cell_diameter * corridor_width
    ch = cell_diameter * corridor_height

    corridor_boundary_coords = np.array(
        [
            [corridor_origin[0] + cw, corridor_origin[1]],
            corridor_origin,
            [corridor_origin[0], corridor_origin[1] + ch],
            [corridor_origin[0] + cw, corridor_origin[1] + ch],
        ],
        dtype=np.float64,
    )

    corridor_boundary_patch = mpatches.Polygon(
        corridor_boundary_coords, closed=False, fill=False, color="r", ls="solid"
    )
    ax.add_artist(corridor_boundary_patch)

    box_origin = np.array(
        [
            corridor_origin[0],
            corridor_origin[1]
            + box_y_placement_factor * cell_diameter * (corridor_height - box_height),
        ]
    )
    bw, bh = cell_diameter * box_width, cell_diameter * box_height
    box_boundary_patch = mpatches.Polygon(
        [
            [box_origin[0] + bw, box_origin[1]],
            box_origin,
            [box_origin[0], box_origin[1] + bh],
            [box_origin[0] + bw, box_origin[1] + bh],
        ],
        closed=True,
        fill=False,
        color="b",
        ls="dashed",
    )
    ax.add_artist(box_boundary_patch)

    cell_radius = 0.5 * cell_diameter

    for cell_center in cell_center_locations:
        cell_patch = mpatches.Circle(
            cell_center, radius=cell_radius, color="k", fill=False, ls="solid"
        )
        ax.add_artist(cell_patch)


def place_init_cell_randomly(
    cell_centers,
    cell_diameter,
    corridor_origin,
    corridor_height,
    box_height,
    box_width,
    init_cell_x_placement,
):
    if init_cell_x_placement < 0.0:
        raise Exception(
            "negative init_cell_x_placement given: {}".format(init_cell_x_placement)
        )
    center_x = (
        corridor_origin[0]
        + 0.5 * cell_diameter
        + np.random.rand() * cell_diameter * init_cell_x_placement * (box_width - 1)
    )
    center_y = (
        corridor_origin[1]
        + 0.5 * cell_diameter
        + np.random.rand() * cell_diameter * (box_height - 1)
    )

    cell_centers[0] = np.array([center_x, center_y])

    return cell_centers


def generate_theta_bin_boundaries(num_bins):
    theta_bins = np.zeros((num_bins, 2), dtype=np.float64)
    delta_theta = 2 * np.pi / num_bins

    for n in range(num_bins):
        if n == 0:
            last_boundary = 0.0
        else:
            last_boundary = theta_bins[n - 1][1]

        theta_bins[n][0] = last_boundary
        theta_bins[n][1] = last_boundary + delta_theta

    return theta_bins


def generate_trial_theta(theta_bin_boundaries, theta_bin_probabilities):
    tbin_index = np.random.choice(
        np.arange(theta_bin_boundaries.shape[0]), p=theta_bin_probabilities
    )

    return theta_bin_boundaries[tbin_index][0] + np.random.rand() * (
        theta_bin_boundaries[tbin_index][1] - theta_bin_boundaries[tbin_index][0]
    )


def update_theta_bin_probabilities(target_bin_index, theta_bin_probabilities):
    num_bins = theta_bin_probabilities.shape[0]
    avg_p = np.average(theta_bin_probabilities)
    orig_p = theta_bin_probabilities[target_bin_index]
    new_p = 0.5 * orig_p
    theta_bin_probabilities[target_bin_index] = new_p

    interesting_bins = []
    for n in range(num_bins):
        if n == target_bin_index:
            continue
        else:
            if theta_bin_probabilities[n] < avg_p:
                continue
            else:
                interesting_bins.append(n)

    num_interesting_bins = len(interesting_bins)

    if num_interesting_bins == 0:
        interesting_bins = [n for n in range(num_bins) if n != target_bin_index]
        num_interesting_bins = len(interesting_bins)
    delta_p = new_p / num_interesting_bins

    for n in interesting_bins:
        theta_bin_probabilities[n] += delta_p

    if np.abs(1.0 - np.sum(theta_bin_probabilities)) > 1e-6:
        raise Exception(
            "theta_bin_probabilities: {}\nsum: {}\ndelta_p: {}\ninterestin_bins: {}".format(
                theta_bin_probabilities,
                np.sum(theta_bin_probabilities),
                delta_p,
                interesting_bins,
            )
        )

    return theta_bin_probabilities


def find_relevant_bin_index(theta, theta_bins):
    for n, tbin in enumerate(theta_bins):
        if tbin[0] <= theta < tbin[1]:
            return n

    raise Exception("could not find a bin for theta = {}! {}".format(theta, theta_bins))


def is_collision(
    last_placed_cell_index,
    cell_centers,
    cell_diameter,
    corridor_origin,
    corridor_height,
    center_x,
    center_y,
):

    if (
        center_y > corridor_origin[1] + (corridor_height - 0.5) * cell_diameter
        or center_y < corridor_origin[1] + 0.5 * cell_diameter
    ):
        return True
    if center_x < corridor_origin[0] + 0.5 * cell_diameter:
        return True

    for n in range(last_placed_cell_index + 1):
        other_x, other_y = cell_centers[n]

        if (
            np.sqrt((other_x - center_x) ** 2 + (other_y - center_y) ** 2)
            - cell_diameter
            < -1e-8
        ):
            return True

    return False


def try_placing_cell_randomly(
    last_successful_anchor_index,
    last_placed_cell_index,
    cell_centers,
    theta_bins,
    cell_diameter,
    corridor_origin,
    corridor_height,
    box_height,
    box_width,
    max_placement_dist,
):
    num_trials = 2 * theta_bins.shape[0]

    theta_bin_probabilities = (
        np.ones(theta_bins.shape[0], dtype=np.float64) / theta_bins.shape[0]
    )

    center_x, center_y = cell_centers[last_successful_anchor_index]

    for ti in range(num_trials):
        theta = generate_trial_theta(theta_bins, theta_bin_probabilities)
        placement_distance = cell_diameter * (
            (max_placement_dist - 1.0) * np.random.rand() + 1.0
        )
        dx, dy = placement_distance * np.cos(theta), placement_distance * np.sin(theta)
        test_x, test_y = center_x + dx, center_y + dy
        if is_collision(
            last_placed_cell_index,
            cell_centers,
            cell_diameter,
            corridor_origin,
            corridor_height,
            test_x,
            test_y,
        ):
            theta_bin_probabilities = update_theta_bin_probabilities(
                find_relevant_bin_index(theta, theta_bins), theta_bin_probabilities
            )
        else:
            cell_index = last_placed_cell_index + 1
            cell_centers[cell_index] = np.array([test_x, test_y])

            return cell_index, cell_centers

    return -1, cell_centers


def place_cells_randomly(
    num_cells,
    cell_diameter,
    corridor_origin,
    corridor_height,
    box_height,
    box_width,
    init_cell_x_placement,
    max_placement_distance,
    num_theta_bins=20,
):
    if max_placement_distance < 1.0:
        raise Exception(
            "max placement distance cannot be < 1.0! Given: {}".format(
                max_placement_distance
            )
        )
    cell_centers = np.nan * np.ones((num_cells, 2), dtype=np.float64)
    cell_centers = place_init_cell_randomly(
        cell_centers,
        cell_diameter,
        corridor_origin,
        corridor_height,
        box_height,
        box_width,
        init_cell_x_placement,
    )

    num_cells_placed = 1
    possible_anchor_indices = [0]
    trial_anchor_index = 0
    theta_bins = generate_theta_bin_boundaries(num_theta_bins)

    while num_cells_placed != num_cells:
        cell_index, cell_centers = try_placing_cell_randomly(
            possible_anchor_indices[trial_anchor_index],
            num_cells_placed - 1,
            cell_centers,
            theta_bins,
            cell_diameter,
            corridor_origin,
            corridor_height,
            box_height,
            box_width,
            max_placement_distance,
        )

        if cell_index != -1:
            num_cells_placed += 1
            possible_anchor_indices.append(cell_index)
        else:
            trial_anchor_index = (trial_anchor_index + 1) % len(possible_anchor_indices)

    return cell_centers


def generate_bounding_boxes_from_centers(radius, centers):
    bounding_boxes = np.zeros((centers.shape[0], 4), dtype=np.float64)

    for i, bb in enumerate(centers):
        x, y = centers[i]
        bounding_boxes[i] = np.array([x - radius, x + radius, y - radius, y + radius])

    return bounding_boxes


num_nodes = 16
cell_diameter = 40
num_cells = 16
box_width = 4
box_height = 4
box_y_placement_factor = 0.5
corridor_height = 4
init_cell_x_placement = 0.25
max_placement_distance = 1.0
corridor_origin = [0.0, 0.0]

# min_placement_distance = 0.5
# max_placement_distance = 1.5
# 53909, 38381, 68474, 84019

# min_placement_distance = 0.5
# max_placement_distance = 1.0
# 84446, 13830, 91705, 33743


seed = np.random.randint(0, 100000)
print("seed: ", seed)
np.random.seed(seed)
cell_centers = place_cells_randomly(
    num_cells,
    cell_diameter,
    corridor_origin,
    corridor_height,
    box_height,
    box_width,
    init_cell_x_placement,
    max_placement_distance,
)


fig, ax = plt.subplots()
ax.set_aspect("equal")


draw_cell_arrangement(
    ax,
    cell_centers,
    cell_diameter,
    box_height,
    box_width,
    corridor_origin,
    corridor_height,
    box_y_placement_factor,
)

ax.set_xlim(
    [
        corridor_origin[0] - cell_diameter * box_width * 1.5,
        np.max(cell_centers[:, 0]) + cell_diameter,
    ]
)
ax.set_ylim(
    [
        corridor_origin[1] - 0.5 * cell_diameter * corridor_height,
        corridor_origin[1] + 1.5 * cell_diameter * corridor_height,
    ]
)
plt.show()
