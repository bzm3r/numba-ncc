# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:27:54 2015
@author: Brian
"""


import numpy as np
import numba as nb

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_coa_sensitivity_exponent(coa_distribution_exponent, percent_drop_over_cell_diameter, cell_diameter):
    coa_sensitivity_exponent = np.log(percent_drop_over_cell_diameter)/(1 - np.exp(coa_distribution_exponent*cell_diameter))
    
    return coa_sensitivity_exponent

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def hill_function(exp, thresh, sig):
    pow_sig = sig**exp
    pow_thresh = thresh**exp
    
    return pow_sig/(pow_thresh + pow_sig)
          
# -----------------------------------------------------------------
@nb.jit(nopython=True)
def generate_random_factor(distribution_width):
    random_delta_rac_factor = (np.random.rand() - 0.5)*distribution_width
    return random_delta_rac_factor

# ---------------------------------------------------------
@nb.jit(nopython=True)
def bell_function(x, centre, width, height, flatness):
    delta = -1*(x - centre)**flatness
    epsilon = 1.0/(width**2)
    
    return height*np.exp(delta*epsilon)
    
# ---------------------------------------------------------
@nb.jit(nopython=True)  
def reverse_bell_function(x, centre, width, depth, flatness):
    return_val = 1 - bell_function(x, centre, width, depth, flatness) 
    return return_val
    
# ---------------------------------------------------------
@nb.jit(nopython=True)
def generate_randomization_width(rac_active, randomization_width_baseline, randomization_width_hf_exponent, randomization_width_halfmax_threshold, randomization_centre, randomization_width, randomization_depth, flatness, randomization_function_type):
    if randomization_function_type == 0:
        width = randomization_width_baseline*(1 - hill_function(randomization_width_hf_exponent, randomization_width_halfmax_threshold, rac_active))
    elif randomization_function_type == 1:
        width = reverse_bell_function(rac_active, randomization_centre, randomization_width, randomization_depth, flatness)
    else:
        width = -10
            
    return width
    
# -----------------------------------------------------------------
@nb.jit(nopython=True) 
def calculate_rac_randomization(cell_index, t, num_nodes, rac_actives, rac_inactives, randomization_width_baseline, randomization_width_halfmax_threshold, randomization_width_hf_exponent, randomization_centre, randomization_width, randomization_depth, randomization_function_type):
    flatness = 4
    
    randomized_rac_actives = np.zeros_like(rac_actives)
    randomized_rac_inactives = np.zeros_like(rac_inactives)

    randomization_factors_rac_active = np.zeros(num_nodes, dtype=np.float64)

    for ni in range(num_nodes):
        nodal_rac_active = rac_actives[ni]
        nodal_rac_inactive = rac_inactives[ni]
        
        uniform_distribution_width = generate_randomization_width(nodal_rac_active, randomization_width_baseline, randomization_width_hf_exponent, randomization_width_halfmax_threshold, randomization_centre, randomization_width, randomization_depth, flatness, randomization_function_type)
        
        randomization_factors_rac_active[ni] = generate_random_factor(uniform_distribution_width)
        
#    # ===========
#    print "rfs: ", randomization_factors_rac_active
#    # ===========
    
    for ni in range(num_nodes):
        nodal_rac_active = rac_actives[ni]
        nodal_rac_inactive = rac_inactives[ni]
        
        
        #randomization_factor_rac_inactive = generate_random_factor(randomization_width)
        randomization_factor_rac_active = randomization_factors_rac_active[ni]
        delta_rac_active = randomization_factor_rac_active*nodal_rac_active
        #delta_rac_inactive = randomization_factor_rac_inactive*nodal_rac_inactive
        
        if delta_rac_active > nodal_rac_inactive:
            delta_rac_active = nodal_rac_inactive
        
        randomized_rac_actives[ni] = nodal_rac_active + delta_rac_active #+ delta_rac_inactive
        randomized_rac_inactives[ni] = nodal_rac_inactive - delta_rac_active #- delta_rac_inactive
    
#    print "ini_rac_a: ", rac_actives
#    print "ini_rac_i: ", rac_inactives
#    
#    
#    print "fin_rac_a: ", randomized_rac_actives
#    print "fin_rac_i: ", randomized_rac_inactives
#    
#    
#    print "rac_check: ", np.abs((rac_actives + rac_inactives) - (randomized_rac_actives + randomized_rac_inactives)) < 1e-8 
    
    return randomized_rac_actives, randomized_rac_inactives
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_strain_mediated_rac_activation_reduction_using_neg_exp(strain, tension_mediated_rac_inhibition_exponent):
    return np.exp(-1*tension_mediated_rac_inhibition_exponent*strain)

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_strain_mediated_rac_activation_reduction_using_inv_fn(strain, tension_mediated_rac_inhibition_multiplier):
    return 1.0/(1 + tension_mediated_rac_inhibition_multiplier*strain)

# -----------------------------------------------------------------
@nb.jit(nopython=True)   
def calculate_strain_mediated_rac_activation_reduction_using_hill_fn(strain, tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain):
    return 1 - hill_function(tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain, strain)
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)        
def calculate_kgtp_rac(num_nodes, conc_rac_membrane_actives, migr_bdry_contact_factors, exponent_rac_autoact, threshold_rac_autoact, kgtp_rac_baseline, kgtp_rac_autoact_baseline, coa_signals, endocytosis_effect_factor_on_nodes, chemoattractant_signal_on_nodes, randomization_factors, intercellular_contact_factors, close_point_smoothness_factors):
    num_vertices = conc_rac_membrane_actives.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    kgtp_rac_autoact = 0.0
    
    for i in range(num_vertices):
        i_plus1 = (i + 1)%num_vertices
        i_minus1 = (i - 1)%num_vertices
        
        cil_factor = (intercellular_contact_factors[i] + intercellular_contact_factors[i_plus1] + intercellular_contact_factors[i_minus1])/3.0
        smooth_factor = np.max(close_point_smoothness_factors[i])
        coa_signal = coa_signals[i]#*(1.0 - smooth_factor)
        
        chemoattractant_signal_at_node = chemoattractant_signal_on_nodes[i]*endocytosis_effect_factor_on_nodes[i]
        
        if cil_factor > 0.0 or smooth_factor > 1e-6:
            coa_signal = 0.0

        rac_autoact_hill_function = hill_function(exponent_rac_autoact, threshold_rac_autoact, conc_rac_membrane_actives[i])
        kgtp_rac_autoact = kgtp_rac_autoact_baseline*rac_autoact_hill_function
        
        result[i] = (randomization_factors[i] + coa_signal)*kgtp_rac_baseline + kgtp_rac_autoact*(chemoattractant_signal_at_node + 1.0)
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True) 
def calculate_kgtp_rho(num_nodes, conc_rho_membrane_active, intercellular_contact_factors, migr_bdry_contact_factors, exponent_rho_autoact, threshold_rho_autoact, kgtp_rho_baseline, kgtp_rho_autoact_baseline):
    
    result = np.empty(num_nodes)
    for i in range(num_nodes):
        kgtp_rho_autoact = kgtp_rho_autoact_baseline*hill_function(exponent_rho_autoact, threshold_rho_autoact, conc_rho_membrane_active[i])
        
        i_plus1 = (i + 1)%num_nodes
        i_minus1 = (i - 1)%num_nodes
        
        cil_factor = (intercellular_contact_factors[i] + intercellular_contact_factors[i_plus1] + intercellular_contact_factors[i_minus1])/3.0
        
        migr_bdry_factor = (migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus1] + migr_bdry_contact_factors[i_minus1])/3.0
        
        result[i] =  (1.0 + migr_bdry_factor + cil_factor)*kgtp_rho_baseline + kgtp_rho_autoact#(migr_bdry_factor)*(cil_factor)*(kgtp_rho_autoact + kgtp_rho_baseline)
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)            
def calculate_kdgtp_rac(num_nodes, conc_rho_membrane_actives, exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, kdgtp_rac_baseline, kdgtp_rho_mediated_rac_inhib_baseline, intercellular_contact_factors, migr_bdry_contact_factors, tension_mediated_rac_inhibition_half_strain, tension_mediated_rac_inhibition_magnitude, strain_calculation_type, local_strains):
    result = np.empty(num_nodes, dtype=np.float64)
    
    global_tension = np.sum(local_strains)/num_nodes
    if global_tension < 0.0:
        global_tension = 0.0
    
    strain_inhibition = tension_mediated_rac_inhibition_magnitude*hill_function(3, tension_mediated_rac_inhibition_half_strain, global_tension)
    
    for i in range(num_nodes):        
        kdgtp_rho_mediated_rac_inhib = kdgtp_rho_mediated_rac_inhib_baseline*hill_function(exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, conc_rho_membrane_actives[i])
        
        i_plus1 = (i + 1)%num_nodes
        i_minus1 = (i - 1)%num_nodes

        
        cil_factor = (intercellular_contact_factors[i] + intercellular_contact_factors[i_plus1] + intercellular_contact_factors[i_minus1])/3.0
        
        migr_bdry_factor = (migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus1] + migr_bdry_contact_factors[i_minus1])/3.0
        
        result[i] = (1. + cil_factor + migr_bdry_factor + strain_inhibition)*kdgtp_rac_baseline + kdgtp_rho_mediated_rac_inhib
        
    return result
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_kdgtp_rho(num_nodes, conc_rac_membrane_active, exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, kdgtp_rho_baseline, kdgtp_rac_mediated_rho_inhib_baseline):
    
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        kdgtp_rac_mediated_rho_inhib = kdgtp_rac_mediated_rho_inhib_baseline*hill_function(exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, conc_rac_membrane_active[i])
        
        result[i] = kdgtp_rho_baseline + kdgtp_rac_mediated_rho_inhib
        
    return result
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_concentrations(num_nodes, species, avg_edge_lengths):
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        result[i] = species[i]/avg_edge_lengths[i]
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_flux_terms(num_nodes, concentrations, diffusion_constant, edgeplus_lengths, avg_edge_lengths):
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        i_plus1_index = (i + 1)%num_nodes
                                         
        result[i] = -diffusion_constant*(concentrations[i_plus1_index] - concentrations[i])/edgeplus_lengths[i]
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_diffusion(num_nodes, concentrations, diffusion_constant, edgeplus_lengths, avg_edge_lengths,):
    result = np.empty(num_nodes, dtype=np.float64)
    
    fluxes = calculate_flux_terms(num_nodes, concentrations, diffusion_constant, edgeplus_lengths, avg_edge_lengths)
    
    for i in range(num_nodes):
        i_minus1_index = (i - 1)%num_nodes

        
        result[i] = fluxes[i_minus1_index] - fluxes[i]
        
    return result
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)   
def calculate_intercellular_contact_factors(this_cell_index, num_nodes, num_cells, intercellular_contact_factor_magnitudes, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists, close_point_smoothness_factors):

    intercellular_contact_factors = np.zeros(num_nodes, dtype=np.float64)
    
    for other_ci in range(num_cells):
        if other_ci != this_cell_index:
            for ni in range(num_nodes):
                current_ic_mag = intercellular_contact_factors[ni]
                
                new_ic_mag = intercellular_contact_factor_magnitudes[other_ci]*close_point_smoothness_factors[ni][other_ci]
                
                if new_ic_mag > current_ic_mag:
                    intercellular_contact_factors[ni] = new_ic_mag
                    current_ic_mag = new_ic_mag
        
    return intercellular_contact_factors

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_coa_signals(this_cell_index, num_nodes, num_cells, random_order_cell_indices, coa_distribution_exponent,  cell_dependent_coa_signal_strengths, max_coa_signal,  intercellular_dist_squared_matrix, line_segment_intersection_matrix, closeness_dist_squared_criteria, intersection_exponent):
    coa_signals = np.zeros(num_nodes, dtype=np.float64)
    too_close_dist_squared = 1e-6
    
    for ni in range(num_nodes):
        this_node_coa_signal = coa_signals[ni]

        this_node_relevant_line_seg_intersection_slice = line_segment_intersection_matrix[ni]
        this_node_relevant_dist_squared_slice = intercellular_dist_squared_matrix[ni]
        
        for other_ci in random_order_cell_indices:
            if other_ci != this_cell_index:
                signal_strength = cell_dependent_coa_signal_strengths[other_ci]
                
                this_node_other_cell_relevant_line_seg_intersection_slice = this_node_relevant_line_seg_intersection_slice[other_ci]
                this_node_other_cell_relevant_dist_squared_slice = this_node_relevant_dist_squared_slice[other_ci]
                for other_ni in range(num_nodes):
                    line_segment_between_node_intersects_polygon = this_node_other_cell_relevant_line_seg_intersection_slice[other_ni]
                    intersection_factor = (1./(line_segment_between_node_intersects_polygon + 1.)**intersection_exponent)
                    
                    dist_squared_between_nodes = this_node_other_cell_relevant_dist_squared_slice[other_ni]
                    
                    coa_signal = 0.0
                    if max_coa_signal < 0:
                        if dist_squared_between_nodes > too_close_dist_squared:
                            coa_signal = np.exp(coa_distribution_exponent*np.sqrt(dist_squared_between_nodes))*intersection_factor
                    else:
                        if dist_squared_between_nodes > too_close_dist_squared:
                            coa_signal = np.exp(coa_distribution_exponent*np.sqrt(dist_squared_between_nodes))*intersection_factor
                    
                    if max_coa_signal < 0.0:
                        this_node_coa_signal += coa_signal*signal_strength
                    else:
                        new_node_coa_signal = this_node_coa_signal + coa_signal*signal_strength
                        
                        if (new_node_coa_signal < max_coa_signal):
                            this_node_coa_signal = new_node_coa_signal
                        else:
                            this_node_coa_signal = max_coa_signal
                            break
                        
        coa_signals[ni] = this_node_coa_signal

    return coa_signals

# -------------------------------------------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_endocytosis_effect_factors(this_cell_index, num_nodes, num_cells, intercellular_dist_squared_matrix, line_segment_intersection_matrix, endocytosis_effect_length):
    endocytosis_effect_factors = np.zeros(num_nodes, dtype=np.float64)

    for ni in range(num_nodes):
        this_node_relevant_line_seg_intersection_slice = line_segment_intersection_matrix[ni]
        this_node_relevant_dist_squared_slice = intercellular_dist_squared_matrix[ni]

        sum_weights = 0.0

        for other_ci in range(num_cells):
            if other_ci != this_cell_index:
                this_node_other_cell_relevant_line_seg_intersection_slice = \
                this_node_relevant_line_seg_intersection_slice[other_ci]
                this_node_other_cell_relevant_dist_squared_slice = this_node_relevant_dist_squared_slice[other_ci]

                for other_ni in range(num_nodes):
                    if this_node_other_cell_relevant_line_seg_intersection_slice[other_ni] == 0:
                        ds = this_node_other_cell_relevant_dist_squared_slice[other_ni]
                        sum_weights += np.exp(np.log(0.25)*(ds/endocytosis_effect_length))

        endocytosis_effect_factors[ni] = 1./(1.0 + sum_weights)

    return endocytosis_effect_factors