# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:27:54 2015
@author: Brian
"""

from __future__ import division
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
#    
#    if False and t%20 == 0 and cell_index == 0:
#        save_path = "A:\\cncell_output\\coa_fn_plots"
#            
#        test_xs = np.linspace(0, 1.0, num=100000)
#        if randomization_function_type == 1:
#            test_return_vals = [reverse_bell_function(x, randomization_centre, randomization_width, randomization_depth, flatness) for x in test_xs]
#            xs = rac_actives
#            return_vals = [reverse_bell_function(x, randomization_centre, randomization_width, randomization_depth, flatness) for x in xs]
#        else:
#            test_return_vals = [randomization_width_baseline*(1 - hill_function(randomization_width_hf_exponent, randomization_width_halfmax_threshold, x)) for x in test_xs]
#            xs = rac_actives
#            return_vals = [randomization_width_baseline*(1 - hill_function(randomization_width_hf_exponent, randomization_width_halfmax_threshold, x)) for x in xs]
#        
#        plt.plot(test_xs, test_return_vals)
#        plt.plot(xs, return_vals, 'r.')
#        plt.xlim(0, 2.0/num_nodes)
#        plt.ylim(0, 1.5)
#        plt.savefig(os.path.join(save_path, "graph_{}.png".format(t)))
#        plt.close()
    
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
def calculate_kgtp_rac(num_nodes, rac_membrane_active, migr_bdry_contact_factors, exponent_rac_autoact, threshold_rac_autoact, kgtp_rac_baseline, kgtp_rac_autoact_baseline, transduced_coa_signals, external_gradient_on_nodes, randomization_factors):
    result = np.empty(num_nodes, dtype=np.float64)
    rac_autoact = 0.0
    
    for i in range(num_nodes):
        if migr_bdry_contact_factors[i] > 1.0:
            rac_autoact = 0.0
        else:
            rac_autoact = kgtp_rac_autoact_baseline*hill_function(exponent_rac_autoact, threshold_rac_autoact, rac_membrane_active[i])
        
        result[i] = (transduced_coa_signals[i] + 1)*(randomization_factors[i]*kgtp_rac_baseline + rac_autoact*(external_gradient_on_nodes[i] + 1))
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True) 
def calculate_kgtp_rho(num_nodes, rho_membrane_active, intercellular_contact_factors, migr_bdry_contact_factors, exponent_rho_autoact, threshold_rho_autoact, kgtp_rho_baseline, kgtp_rho_autoact_baseline):
    
    result = np.empty(num_nodes)
    for i in range(num_nodes):
        kgtp_rho_autoact = kgtp_rho_autoact_baseline*hill_function(exponent_rho_autoact, threshold_rho_autoact, rho_membrane_active[i])
        
        #i_minus_1 = (i - 1)%num_nodes
        #i_plus_1 = (i + 1)%num_nodes
        
        #migr_bdry_contact_factor_average = (migr_bdry_contact_factors[i_minus_1] + migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus_1])/3.0
        
        #intercellular_contact_factor_average = (intercellular_contact_factors[i_minus_1] + intercellular_contact_factors[i] + intercellular_contact_factors[i_plus_1])/3.0
        
        result[i] = (migr_bdry_contact_factors[i])*(intercellular_contact_factors[i])*(kgtp_rho_autoact + kgtp_rho_baseline)
        
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)            
def calculate_kdgtp_rac(num_nodes, rho_membrane_active, exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, kdgtp_rac_baseline, kdgtp_rho_mediated_rac_inhib_baseline, intercellular_contact_factors, migr_bdry_contact_factors, tension_mediated_rac_inhibition_exponent, tension_mediated_rac_inhibition_multiplier, tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain, local_strains, tension_fn_type):
    result = np.empty(num_nodes, dtype=np.float64)
    
    global_tension = np.sum(local_strains)/num_nodes
    
    for i in range(num_nodes):        
        kdgtp_rho_mediated_rac_inhib = kdgtp_rho_mediated_rac_inhib_baseline*hill_function(exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, rho_membrane_active[i])
        
        #i_minus_1 = (i - 1)%num_nodes
        #i_plus_1 = (i + 1)%num_nodes
        
        #migr_bdry_contact_factor_average = (migr_bdry_contact_factors[i_minus_1] + migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus_1])/3.0
        
#        modification_multiplier = 2
#        this_modified_ic_factor = intercellular_contact_factors[i]
#        if this_modified_ic_factor > 1.0:
#            this_modified_ic_factor = modification_multiplier*this_modified_ic_factor
#            
#        this_plus1_modified_ic_factor = intercellular_contact_factors[i_plus_1]
#        if this_plus1_modified_ic_factor > 1.0:
#            this_plus1_modified_ic_factor = modification_multiplier*this_plus1_modified_ic_factor
#            
#        this_minus1_modified_ic_factor = intercellular_contact_factors[i_minus_1]
#        if this_minus1_modified_ic_factor > 1.0:
#            this_minus1_modified_ic_factor = modification_multiplier*this_minus1_modified_ic_factor
#            
#        modified_intercellular_contact_factor_average = (this_modified_ic_factor + this_plus1_modified_ic_factor + this_minus1_modified_ic_factor)/3.0
        
        #ic_factor = intercellular_contact_factors[i]
        
        strain_inhibition = 1.0
        if tension_fn_type == 5:
            exponent = np.log(2)/tension_mediated_rac_inhibition_half_strain
            strain_inhibition = np.exp(exponent*global_tension)
        elif tension_fn_type == 6:
            exponent = 2
            constant  = (2.0 - 1.0)/(tension_mediated_rac_inhibition_half_strain**exponent) 
            strain_inhibition = constant*(global_tension**exponent) + 1.0
        elif tension_fn_type == 7:
            exponent = 3
            constant  = (2.0 - 1.0)/(tension_mediated_rac_inhibition_half_strain**exponent) 
            strain_inhibition = constant*(global_tension**exponent) + 1.0
        elif tension_fn_type == 8:
            strain_inhibition = 1.0
        
        result[i] = (migr_bdry_contact_factors[i])*(intercellular_contact_factors[i])*(strain_inhibition)*(kdgtp_rac_baseline + kdgtp_rho_mediated_rac_inhib)
        
    return result
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_kdgtp_rho(num_nodes, rac_membrane_active, exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, kdgtp_rho_baseline, kdgtp_rac_mediated_rho_inhib_baseline):
    
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        kdgtp_rac_mediated_rho_inhib = kdgtp_rac_mediated_rho_inhib_baseline*hill_function(exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, rac_membrane_active[i])
        
        result[i] = 1.0*(kdgtp_rho_baseline + kdgtp_rac_mediated_rho_inhib)
        
    return result
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_diffusion(num_nodes, species, diffusion_constant, edgeplus_lengths):
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        i_minus_1_index = (i - 1)%num_nodes
        i_plus_1_index = (i + 1)%num_nodes
        
        length_edgeplus = edgeplus_lengths[i]
        length_edgeminus = edgeplus_lengths[i_minus_1_index]
        
        delta_i_plus_1 = (species[i_plus_1_index] - species[i])/(length_edgeplus*length_edgeplus)
        
        delta_i_minus_1 = (species[i_minus_1_index] - species[i])/(length_edgeminus*length_edgeminus)
        
        result[i] = diffusion_constant*(delta_i_plus_1 + delta_i_minus_1)
        
    return result
    
# -----------------------------------------------------------------------------
@nb.jit(nopython=True)   
def calculate_intercellular_contact_factors(this_cell_index, num_nodes, num_cells, intercellular_contact_factor_magnitudes, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists):

    intercellular_contact_factors = np.ones(num_nodes, dtype=np.float64)

    for other_ci in range(num_cells):
        if other_ci != this_cell_index:
            for ni in range(num_nodes):
                current_ic_mag = intercellular_contact_factors[ni]
                
                if are_nodes_inside_other_cells[ni][other_ci] == 1 or close_point_on_other_cells_to_each_node_exists[ni][other_ci] == 1:
                    
                    new_ic_mag = intercellular_contact_factor_magnitudes[other_ci]
                    
                    if new_ic_mag > current_ic_mag:
                        intercellular_contact_factors[ni] = new_ic_mag
                        current_ic_mag = new_ic_mag
        
    return intercellular_contact_factors

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_coa_signals(this_cell_index, num_nodes, num_cells, coa_distribution_exponent, coa_sensitivity_exponent, coa_belt_offset,  cell_dependent_coa_signal_strengths, intercellular_dist_squared_matrix, line_segment_intersection_matrix):
    coa_signals = np.zeros(num_nodes, dtype=np.float64)
    #coa_at_belt_offset = np.exp(coa_distribution_exponent*coa_belt_offset)
    for ni in range(num_nodes):
        this_node_coa_signal = 0.0

        this_node_relevant_line_seg_intersection_slice = line_segment_intersection_matrix[ni]
        this_node_relevant_dist_squared_slice = intercellular_dist_squared_matrix[ni]
        
        for other_ci in range(num_cells):
            if other_ci != this_cell_index:
                signal_strength = cell_dependent_coa_signal_strengths[other_ci]
                
                this_node_other_cell_relevant_line_seg_intersection_slice = this_node_relevant_line_seg_intersection_slice[other_ci]
                this_node_other_cell_relevant_dist_squared_slice = this_node_relevant_dist_squared_slice[other_ci]
                for other_ni in range(num_nodes):
                    line_segment_between_node_intersects_polygon = this_node_other_cell_relevant_line_seg_intersection_slice[other_ni]
                    if line_segment_between_node_intersects_polygon == 1:
                        transduced_coa_signal = 0.0
                    else:
                        dist_squared_between_nodes = this_node_other_cell_relevant_dist_squared_slice[other_ni]
                        
                        untransduced_coa_signal = 0.0
                        if dist_squared_between_nodes < 1e-6:
                            untransduced_coa_signal = np.exp(coa_distribution_exponent*1e-3)
                        else:
                            untransduced_coa_signal = np.exp(coa_distribution_exponent*np.sqrt(dist_squared_between_nodes))
                        
                        if coa_sensitivity_exponent > 0:
                            transduced_coa_signal = untransduced_coa_signal
                        else:
                            transduced_coa_signal = 0.0                         
                            if transduced_coa_signal > 1.0:
                                transduced_coa_signal = 0.0

                    if np.isnan(transduced_coa_signal):
                        raise StandardError("Caught a fish!")
                    this_node_coa_signal = this_node_coa_signal + transduced_coa_signal*signal_strength

                    
        coa_signals[ni] = this_node_coa_signal
        
    return coa_signals