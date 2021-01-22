# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:27:54 2015
@author: Brian
"""


import numpy as np
import numba as nb


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def hill_function(exp, thresh, sig):
    pow_sig = sig ** exp
    pow_thresh = thresh ** exp

    return pow_sig / (pow_thresh + pow_sig)


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def generate_random_factor(distribution_width):
    random_delta_rac_factor = (np.random.rand() - 0.5) * distribution_width
    return random_delta_rac_factor


# ---------------------------------------------------------
@nb.jit(nopython=True)
def bell_function(x, centre, width, height, flatness):
    delta = -1 * (x - centre) ** flatness
    epsilon = 1.0 / (width ** 2)

    return height * np.exp(delta * epsilon)


# ---------------------------------------------------------
@nb.jit(nopython=True)
def reverse_bell_function(x, centre, width, depth, flatness):
    return_val = 1 - bell_function(x, centre, width, depth, flatness)
    return return_val


# ---------------------------------------------------------
@nb.jit(nopython=True)
def generate_randomization_width(
    rac_active,
    randomization_width_baseline,
    randomization_width_hf_exponent,
    randomization_width_halfmax_threshold,
    randomization_centre,
    randomization_width,
    randomization_depth,
    flatness,
    randomization_function_type,
):
    if randomization_function_type == 0:
        width = randomization_width_baseline * (
            1
            - hill_function(
                randomization_width_hf_exponent,
                randomization_width_halfmax_threshold,
                rac_active,
            )
        )
    elif randomization_function_type == 1:
        width = reverse_bell_function(
            rac_active,
            randomization_centre,
            randomization_width,
            randomization_depth,
            flatness,
        )
    else:
        width = -10

    return width

# -----------------------------------------------------------------
#@nb.jit(nopython=True)
def calculate_kgtp_rac(
    conc_rac_membrane_actives,
    exponent_rac_autoact,
    threshold_rac_autoact,
    kgtp_rac_baseline,
    kgtp_rac_autoact_baseline,
    coa_signals,
    randomization_factors,
    intercellular_contact_factors,
    close_point_smoothness_factors,
):
    num_vertices = conc_rac_membrane_actives.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)

    for i in range(num_vertices):
        i_plus1 = (i + 1) % num_vertices
        i_minus1 = (i - 1) % num_vertices

        cil_factor = (
            intercellular_contact_factors[i]
            + intercellular_contact_factors[i_plus1]
            + intercellular_contact_factors[i_minus1]
        ) / 3.0
        smooth_factor = np.max(close_point_smoothness_factors[i])
        coa_signal = coa_signals[i]

        if cil_factor > 0.0 or smooth_factor > 1e-6:
            coa_signal = 0.0

        rac_autoact_hill_effect = hill_function(
            exponent_rac_autoact, threshold_rac_autoact, conc_rac_membrane_actives[i]
        )
        kgtp_rac_autoact = (
            kgtp_rac_autoact_baseline
            * rac_autoact_hill_effect
        )

        if kgtp_rac_autoact > 1.25 * kgtp_rac_autoact_baseline:
            kgtp_rac_autoact = 1.25 * kgtp_rac_autoact_baseline

        kgtp_rac_base = (1.0 + randomization_factors[i] + coa_signal) * kgtp_rac_baseline
        result[i] = kgtp_rac_base + kgtp_rac_autoact

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_kgtp_rho(
    num_nodes,
    conc_rho_membrane_active,
    intercellular_contact_factors,
    exponent_rho_autoact,
    threshold_rho_autoact,
    kgtp_rho_baseline,
    kgtp_rho_autoact_baseline,
):

    result = np.empty(num_nodes)
    for i in range(num_nodes):
        kgtp_rho_autoact = kgtp_rho_autoact_baseline * hill_function(
            exponent_rho_autoact, threshold_rho_autoact, conc_rho_membrane_active[i]
        )

        i_plus1 = (i + 1) % num_nodes
        i_minus1 = (i - 1) % num_nodes

        cil_factor = (
            intercellular_contact_factors[i]
            + intercellular_contact_factors[i_plus1]
            + intercellular_contact_factors[i_minus1]
        ) / 3.0

        result[i] = (
            1.0 + cil_factor
        ) * kgtp_rho_baseline + kgtp_rho_autoact
    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_kdgtp_rac(
    num_nodes,
    conc_rho_membrane_actives,
    exponent_rho_mediated_rac_inhib,
    threshold_rho_mediated_rac_inhib,
    kdgtp_rac_baseline,
    kdgtp_rho_mediated_rac_inhib_baseline,
    intercellular_contact_factors,
    tension_mediated_rac_inhibition_half_strain,
    tension_mediated_rac_inhibition_magnitude,
        local_strains,
):
    result = np.empty(num_nodes, dtype=np.float64)

    global_tension = np.sum(local_strains) / num_nodes
    if global_tension < 0.0:
        global_tension = 0.0
    strain_inhibition = tension_mediated_rac_inhibition_magnitude * hill_function(
        3, tension_mediated_rac_inhibition_half_strain, global_tension
    )

    for i in range(num_nodes):
        kdgtp_rho_mediated_rac_inhib = (
            kdgtp_rho_mediated_rac_inhib_baseline
            * hill_function(
                exponent_rho_mediated_rac_inhib,
                threshold_rho_mediated_rac_inhib,
                conc_rho_membrane_actives[i],
            )
        )

        i_plus1 = (i + 1) % num_nodes
        i_minus1 = (i - 1) % num_nodes

        cil_factor = (
            intercellular_contact_factors[i]
            + intercellular_contact_factors[i_plus1]
            + intercellular_contact_factors[i_minus1]
        ) / 3.0

        result[i] = (
            1.0 + cil_factor + strain_inhibition
        ) * kdgtp_rac_baseline + kdgtp_rho_mediated_rac_inhib

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_kdgtp_rho(
    num_nodes,
    conc_rac_membrane_active,
    exponent_rac_mediated_rho_inhib,
    threshold_rac_mediated_rho_inhib,
    kdgtp_rho_baseline,
    kdgtp_rac_mediated_rho_inhib_baseline,
):

    result = np.empty(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        kdgtp_rac_mediated_rho_inhib = (
            kdgtp_rac_mediated_rho_inhib_baseline
            * hill_function(
                exponent_rac_mediated_rho_inhib,
                threshold_rac_mediated_rho_inhib,
                conc_rac_membrane_active[i],
            )
        )

        result[i] = kdgtp_rho_baseline + kdgtp_rac_mediated_rho_inhib

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_concentrations(num_nodes, species, avg_edge_lengths):
    result = np.empty(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        result[i] = species[i] / avg_edge_lengths[i]

    return result


# -----------------------------------------------------------------
#@nb.jit(nopython=True)
def calculate_flux_terms(
    num_nodes, concentrations, diffusion_constant, edgeplus_lengths
):
    result = np.empty(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        i_plus1_index = (i + 1) % num_nodes

        result[i] = (
            -diffusion_constant
            * (concentrations[i_plus1_index] - concentrations[i])
            / edgeplus_lengths[i]
        )

    return result


# -----------------------------------------------------------------
#@nb.jit(nopython=True)
def calculate_diffusion(
    num_nodes, concentrations, diffusion_constant, edgeplus_lengths
):
    result = np.empty(num_nodes, dtype=np.float64)

    fluxes = calculate_flux_terms(
        num_nodes,
        concentrations,
        diffusion_constant,
        edgeplus_lengths,
    )

    for i in range(num_nodes):
        i_minus1_index = (i - 1) % num_nodes

        result[i] = fluxes[i_minus1_index] - fluxes[i]

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_intercellular_contact_factors(
    this_cell_index,
    num_nodes,
    num_cells,
    intercellular_contact_factor_magnitudes,
        close_point_smoothness_factors,
):

    intercellular_contact_factors = np.zeros(num_nodes, dtype=np.float64)

    for other_ci in range(num_cells):
        if other_ci != this_cell_index:
            for ni in range(num_nodes):
                current_ic_mag = intercellular_contact_factors[ni]

                new_ic_mag = (
                    intercellular_contact_factor_magnitudes[other_ci]
                    * close_point_smoothness_factors[ni][other_ci]
                )

                if new_ic_mag > current_ic_mag:
                    intercellular_contact_factors[ni] = new_ic_mag

    return intercellular_contact_factors


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_coa_signals(
    this_cell_index,
    num_nodes,
        random_order_cell_indices,
    coa_distribution_exponent,
    cell_dependent_coa_signal_strengths,
    max_coa_signal,
    intercellular_dist_squared_matrix,
    line_segment_intersection_matrix,
        intersection_exponent,
):
    coa_signals = np.zeros(num_nodes, dtype=np.float64)
    too_close_dist_squared = 1e-6

    for ni in range(num_nodes):
        this_node_coa_signal = coa_signals[ni]

        this_node_relevant_line_seg_intersection_slice = line_segment_intersection_matrix[
            ni
        ]
        this_node_relevant_dist_squared_slice = intercellular_dist_squared_matrix[ni]

        for other_ci in random_order_cell_indices:
            if other_ci != this_cell_index:
                signal_strength = cell_dependent_coa_signal_strengths[other_ci]

                this_node_other_cell_relevant_line_seg_intersection_slice = this_node_relevant_line_seg_intersection_slice[
                    other_ci
                ]
                this_node_other_cell_relevant_dist_squared_slice = this_node_relevant_dist_squared_slice[
                    other_ci
                ]
                for other_ni in range(num_nodes):
                    line_segment_between_node_intersects_polygon = this_node_other_cell_relevant_line_seg_intersection_slice[
                        other_ni
                    ]
                    intersection_factor = (
                        1.0
                        / (line_segment_between_node_intersects_polygon + 1.0)
                        ** intersection_exponent
                    )

                    dist_squared_between_nodes = this_node_other_cell_relevant_dist_squared_slice[
                        other_ni
                    ]

                    coa_signal = 0.0
                    if max_coa_signal < 0:
                        if dist_squared_between_nodes > too_close_dist_squared:
                            coa_signal = (
                                np.exp(
                                    coa_distribution_exponent
                                    * np.sqrt(dist_squared_between_nodes)
                                )
                                * intersection_factor
                            )
                    else:
                        if dist_squared_between_nodes > too_close_dist_squared:
                            coa_signal = (
                                np.exp(
                                    coa_distribution_exponent
                                    * np.sqrt(dist_squared_between_nodes)
                                )
                                * intersection_factor
                            )

                    if max_coa_signal < 0.0:
                        this_node_coa_signal += coa_signal * signal_strength
                    else:
                        new_node_coa_signal = (
                            this_node_coa_signal + coa_signal * signal_strength
                        )

                        if new_node_coa_signal < max_coa_signal:
                            this_node_coa_signal = new_node_coa_signal
                        else:
                            this_node_coa_signal = max_coa_signal
                            break

        coa_signals[ni] = this_node_coa_signal

    return coa_signals


# -------------------------------------------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_chemoattractant_shielding_effect_factors(
    this_cell_index,
    num_nodes,
    num_cells,
    intercellular_dist_squared_matrix,
    line_segment_intersection_matrix,
    chemoattractant_shielding_effect_length,
):
    chemoattractant_shielding_effect_factors = np.zeros(num_nodes, dtype=np.float64)

    for ni in range(num_nodes):
        this_node_relevant_line_seg_intersection_slice = line_segment_intersection_matrix[
            ni
        ]
        this_node_relevant_dist_squared_slice = intercellular_dist_squared_matrix[ni]

        sum_weights = 0.0

        for other_ci in range(num_cells):
            if other_ci != this_cell_index:
                this_node_other_cell_relevant_line_seg_intersection_slice = this_node_relevant_line_seg_intersection_slice[
                    other_ci
                ]
                this_node_other_cell_relevant_dist_squared_slice = this_node_relevant_dist_squared_slice[
                    other_ci
                ]

                for other_ni in range(num_nodes):
                    if (
                        this_node_other_cell_relevant_line_seg_intersection_slice[
                            other_ni
                        ]
                        == 0
                    ):
                        ds = this_node_other_cell_relevant_dist_squared_slice[other_ni]
                        sum_weights += np.exp(
                            np.log(0.25)
                            * (ds / chemoattractant_shielding_effect_length)
                        )

        chemoattractant_shielding_effect_factors[ni] = 1.0 / (1.0 + sum_weights)

    return chemoattractant_shielding_effect_factors
