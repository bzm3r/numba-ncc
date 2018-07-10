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
def calculate_rac_randomization(cell_index, t, rac_actives, rac_inactives, randomization_width_baseline, randomization_width_halfmax_threshold, randomization_width_hf_exponent, randomization_centre, randomization_width, randomization_depth, randomization_function_type):
    num_vertices = rac_actives.shape[0]
    flatness = 4
    
    randomized_rac_actives = np.zeros_like(rac_actives)
    randomized_rac_inactives = np.zeros_like(rac_inactives)

    randomization_factors_rac_active = np.zeros(num_vertices, dtype=np.float64)

    for ni in range(num_vertices):
        nodal_rac_active = rac_actives[ni]
        nodal_rac_inactive = rac_inactives[ni]
        
        uniform_distribution_width = generate_randomization_width(nodal_rac_active, randomization_width_baseline, randomization_width_hf_exponent, randomization_width_halfmax_threshold, randomization_centre, randomization_width, randomization_depth, flatness, randomization_function_type)
        
        randomization_factors_rac_active[ni] = generate_random_factor(uniform_distribution_width)
        
#    # ===========
#    print "rfs: ", randomization_factors_rac_active
#    # ===========
    
    for ni in range(num_vertices):
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
def calculate_kgtp_rac(conc_rac_membrane_actives, migr_bdry_contact_factors, exponent_rac_autoact, threshold_rac_autoact, kgtp_rac_baseline, kgtp_rac_autoact_baseline, coa_signals, chemoattractant_gradient_on_nodes, randomization_factors, intercellular_contact_factors, close_point_smoothness_factors):
    num_vertices = conc_rac_membrane_actives.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    kgtp_rac_autoact = 0.0
    
    for i in range(num_vertices):
        i_plus1 = (i + 1)%num_vertices
        i_minus1 = (i - 1)%num_vertices
        
        cil_factor = (intercellular_contact_factors[i] + intercellular_contact_factors[i_plus1] + intercellular_contact_factors[i_minus1])/3.0
        smooth_factor = np.max(close_point_smoothness_factors[i])
        coa_signal = coa_signals[i]#*(1.0 - smooth_factor)
        
        chemoattractant_gradient_signal = hill_function(3, 0.5, chemoattractant_gradient_on_nodes[i])*hill_function(3, threshold_rac_autoact, conc_rac_membrane_actives[i])
        
        if cil_factor > 0.0 or smooth_factor > 1e-6:
            coa_signal = 0.0
            chemoattractant_gradient_signal = 0.0
        
        kgtp_rac_autoact = kgtp_rac_autoact_baseline*hill_function(exponent_rac_autoact, threshold_rac_autoact, conc_rac_membrane_actives[i])
        
        result[i] = (randomization_factors[i] + coa_signal)*kgtp_rac_baseline + (chemoattractant_gradient_signal + 1.0)*kgtp_rac_autoact
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True) 
def calculate_kgtp_rho(conc_rho_membrane_actives, intercellular_contact_factors, migr_bdry_contact_factors, exponent_rho_autoact, threshold_rho_autoact, kgtp_rho_baseline, kgtp_rho_autoact_baseline):
    num_vertices = conc_rho_membrane_actives.shape[0]
    
    result = np.empty(num_vertices)
    for i in range(num_vertices):
        kgtp_rho_autoact = kgtp_rho_autoact_baseline*hill_function(exponent_rho_autoact, threshold_rho_autoact, conc_rho_membrane_actives[i])
        
        i_plus1 = (i + 1)%num_vertices
        i_minus1 = (i - 1)%num_vertices
        
        cil_factor = (intercellular_contact_factors[i] + intercellular_contact_factors[i_plus1] + intercellular_contact_factors[i_minus1])/3.0
        
        migr_bdry_factor = (migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus1] + migr_bdry_contact_factors[i_minus1])/3.0
        
        result[i] =  (1.0 + migr_bdry_factor + cil_factor)*kgtp_rho_baseline + kgtp_rho_autoact#(migr_bdry_factor)*(cil_factor)*(kgtp_rho_autoact + kgtp_rho_baseline)
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)            
def calculate_kdgtp_rac(conc_rho_membrane_actives, exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, kdgtp_rac_baseline, kdgtp_rho_mediated_rac_inhib_baseline, intercellular_contact_factors, migr_bdry_contact_factors, tension_mediated_rac_inhibition_half_strain, tension_mediated_rac_inhibition_magnitude, global_strain):
    num_vertices = conc_rho_membrane_actives.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    
    strain_inhibition = tension_mediated_rac_inhibition_magnitude*hill_function(3, tension_mediated_rac_inhibition_half_strain, global_strain)
    
    for i in range(num_vertices):        
        kdgtp_rho_mediated_rac_inhib = kdgtp_rho_mediated_rac_inhib_baseline*hill_function(exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, conc_rho_membrane_actives[i])
        
        i_plus1 = (i + 1)%num_vertices
        i_minus1 = (i - 1)%num_vertices

        
        cil_factor = (intercellular_contact_factors[i] + intercellular_contact_factors[i_plus1] + intercellular_contact_factors[i_minus1])/3.0
        
        migr_bdry_factor = (migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus1] + migr_bdry_contact_factors[i_minus1])/3.0
        
        result[i] = (1. + cil_factor + migr_bdry_factor + strain_inhibition)*kdgtp_rac_baseline + kdgtp_rho_mediated_rac_inhib
        
    return result
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_kdgtp_rho(conc_rac_membrane_active, exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, kdgtp_rho_baseline, kdgtp_rac_mediated_rho_inhib_baseline):
    num_vertices = conc_rac_membrane_active.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    
    for i in range(num_vertices):
        kdgtp_rac_mediated_rho_inhib = kdgtp_rac_mediated_rho_inhib_baseline*hill_function(exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, conc_rac_membrane_active[i])
        
        result[i] = kdgtp_rho_baseline + kdgtp_rac_mediated_rho_inhib
        
    return result
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_concentrations(species, avg_edge_lengths):
    num_vertices = species.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    
    for i in range(num_vertices):
        result[i] = species[i]/avg_edge_lengths[i]
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_flux_terms(concentrations, diffusion_constant, edgeplus_lengths, avg_edge_lengths):
    num_vertices = concentrations.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    
    for i in range(num_vertices):
        i_plus1_index = (i + 1)%num_vertices
                                         
        result[i] = -diffusion_constant*(concentrations[i_plus1_index] - concentrations[i])/edgeplus_lengths[i]
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_diffusion(concentrations, diffusion_constant, edgeplus_lengths, avg_edge_lengths,):
    num_vertices = concentrations.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)
    
    fluxes = calculate_flux_terms(concentrations, diffusion_constant, edgeplus_lengths, avg_edge_lengths)
    
    for i in range(num_vertices):
        i_minus1_index = (i - 1)%num_vertices

        
        result[i] = fluxes[i_minus1_index] - fluxes[i]
        
    return result
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)   
def calculate_intercellular_contact_factors(this_cell_index, num_vertices, num_cells, intercellular_contact_factor_magnitudes, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists, close_point_smoothness_factors):

    intercellular_contact_factors = np.zeros(num_vertices, dtype=np.float64)
    
    for other_ci in range(num_cells):
        if other_ci != this_cell_index:
            for ni in range(num_vertices):
                current_ic_mag = intercellular_contact_factors[ni]
                
                new_ic_mag = intercellular_contact_factor_magnitudes[other_ci]*close_point_smoothness_factors[ni][other_ci]
                
                if new_ic_mag > current_ic_mag:
                    intercellular_contact_factors[ni] = new_ic_mag
                    current_ic_mag = new_ic_mag
        
    return intercellular_contact_factors

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_coa_signals(this_cell_index, num_vertices, num_cells, random_order_cell_indices, coa_distribution_exponent,  cell_dependent_coa_signal_strengths, max_coa_signal,  intercellular_dist_squared_matrix, line_segment_intersection_matrix, closeness_dist_squared_criteria, intersection_exponent):
    coa_signals = np.zeros(num_vertices, dtype=np.float64)
    too_close_dist_squared = 1e-6
    
    for ni in range(num_vertices):
        this_node_coa_signal = coa_signals[ni]

        this_node_relevant_line_seg_intersection_slice = line_segment_intersection_matrix[ni]
        this_node_relevant_dist_squared_slice = intercellular_dist_squared_matrix[ni]
        
        for other_ci in random_order_cell_indices:
            if other_ci != this_cell_index:
                signal_strength = cell_dependent_coa_signal_strengths[other_ci]
                
                this_node_other_cell_relevant_line_seg_intersection_slice = this_node_relevant_line_seg_intersection_slice[other_ci]
                this_node_other_cell_relevant_dist_squared_slice = this_node_relevant_dist_squared_slice[other_ci]
                for other_ni in range(num_vertices):
                    line_segment_between_node_intersects_polygon = this_node_other_cell_relevant_line_seg_intersection_slice[other_ni]
                    intersection_factor = (1./(line_segment_between_node_intersects_polygon + 1.)**intersection_exponent)
                    
                    dist_squared_between_nodes = this_node_other_cell_relevant_dist_squared_slice[other_ni]
                    
                    break_loop, this_node_coa_signal = calculate_coa_signal(max_coa_signal, too_close_dist_squared, coa_distribution_exponent, this_node_coa_signal, signal_strength, dist_squared_between_nodes, intersection_factor)
                    
                    if break_loop:
                        break
                        
        coa_signals[ni] = this_node_coa_signal

    return coa_signals

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_coa_signal(max_coa_signal, too_close_dist_squared, coa_distribution_exponent, current_coa_signal, signal_strength, dist_squared, intersection_factor):
    coa_signal = 0.0
    
    if dist_squared > too_close_dist_squared:
        coa_signal = np.exp(coa_distribution_exponent*np.sqrt(dist_squared))*intersection_factor
    
    if max_coa_signal < 0.0:
        return False, current_coa_signal + coa_signal*signal_strength
    else:
        new_node_coa_signal = current_coa_signal + coa_signal*signal_strength
        
        if (new_node_coa_signal < max_coa_signal):
            return False, new_node_coa_signal
        else:
            return True, max_coa_signal

# -----------------------------------------------------------------
            
def make_linear_gradient_function(source_x, source_y, max_value, slope):
    @nb.jit(nopython=True)
    def f(x):
        d = np.sqrt((x[0] - source_x)**2 + (x[1] - source_y)**2)
        calc_value = max_value - slope*d
        
        if calc_value > max_value:
            return max_value
        elif calc_value < 0.0:
            return 0.0
        else:
            return calc_value
            
    return f

# -----------------------------------------------------------------
    
def make_normal_gradient_function(source_x, source_y, gaussian_width, gaussian_height):    
    widthsq = gaussian_width*gaussian_width
    
    @nb.jit(nopython=True)
    def f(x):
        dsq = (x[0] - source_x)**2 + (x[1] - source_y)**2
        return gaussian_height*np.exp(-1*dsq/(2*widthsq))
    
    return f

# -----------------------------------------------------------------
    
def make_chemoattractant_gradient_function(source_type='', source_x=np.nan, source_y=np.nan, max_value=np.nan, slope=np.nan, gaussian_width=np.nan, gaussian_height=np.nan):
    if source_type == '':
        return lambda x: 0.0
    elif source_type == "normal":
        if not np.any(np.isnan([source_x, source_y, gaussian_width, gaussian_height])):
            return make_normal_gradient_function(source_x, source_y, gaussian_width, gaussian_height)
        else:
            raise Exception("Normal chemoattractant gradient function requested, but definition is not filled out properly!")
    elif source_type == "linear":
        if not np.any(np.isnan([source_x, source_y, max_value, slope])):
            return make_linear_gradient_function(source_x, source_y, max_value, slope)
        else:
            raise Exception("Linear chemoattractant gradient function requested, but definition is not filled out properly!")
        
