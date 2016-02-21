# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:27:54 2015
@author: Brian
"""

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import shutil
import geometry

# -----------------------------------------------------------------

def calculate_coa_sensitivity_exponent(coa_distribution_exponent, percent_drop_over_cell_diameter, cell_diameter):
    coa_sensitivity_exponent = np.log(percent_drop_over_cell_diameter)/(1 - np.exp(coa_distribution_exponent*cell_diameter))
    
    return coa_sensitivity_exponent

# -----------------------------------------------------------------

def hill_function(exp, thresh, sig):
    pow_sig = sig**exp
    pow_thresh = thresh**exp
    
    return pow_sig/(pow_thresh + pow_sig)
          
# -----------------------------------------------------------------

def generate_random_factor(distribution_width):
    random_delta_rac_factor = (np.random.rand() - 0.5)*distribution_width
    return random_delta_rac_factor

# ---------------------------------------------------------

def bell_function(x, centre, width, height, flatness):
    delta = -1*(x - centre)**flatness
    epsilon = 1.0/(width**2)
    
    return height*np.exp(delta*epsilon)
    
# ---------------------------------------------------------
    
def reverse_bell_function(x, centre, width, depth, flatness):
    return_val = 1 - bell_function(x, centre, width, depth, flatness) 
    return return_val
    
# ---------------------------------------------------------

def generate_randomization_width(rac_active, randomization_width_baseline, randomization_width_hf_exponent, randomization_width_halfmax_threshold, randomization_centre, randomization_width, randomization_depth, flatness, randomization_function_type):
    if randomization_function_type == 0:
        width = randomization_width_baseline*(1 - hill_function(randomization_width_hf_exponent, randomization_width_halfmax_threshold, rac_active))
    elif randomization_function_type == 1:
        width = reverse_bell_function(rac_active, randomization_centre, randomization_width, randomization_depth, flatness)
    else:
        width = -10
            
    return width
    
# -----------------------------------------------------------------
   
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

def calculate_strain_mediated_rac_activation_reduction_using_neg_exp(strain, tension_mediated_rac_inhibition_exponent):
    return np.exp(-1*tension_mediated_rac_inhibition_exponent*strain)

# -----------------------------------------------------------------

def calculate_strain_mediated_rac_activation_reduction_using_inv_fn(strain, tension_mediated_rac_inhibition_multiplier):
    return 1.0/(1 + tension_mediated_rac_inhibition_multiplier*strain)

# -----------------------------------------------------------------
    
def calculate_strain_mediated_rac_activation_reduction_using_hill_fn(strain, tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain):
    return 1 - hill_function(tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain, strain)
    
# -----------------------------------------------------------------
         
def calculate_kgtp_rac(num_nodes, rac_membrane_active, migr_bdry_contact_factors, exponent_rac_autoact, threshold_rac_autoact, kgtp_rac_baseline, kgtp_rac_autoact_baseline, transduced_coa_signals):
    result = np.empty(num_nodes, dtype=np.float64)
    rac_autoact = 0.0
    
    for i in range(num_nodes):
        if migr_bdry_contact_factors[i] > 1.0:
            rac_autoact = 0.0
        else:
            rac_autoact = kgtp_rac_autoact_baseline*hill_function(exponent_rac_autoact, threshold_rac_autoact, rac_membrane_active[i])
        
        result[i] = (transduced_coa_signals[i] + 1)*(kgtp_rac_baseline + rac_autoact)
        
    return result

# -----------------------------------------------------------------
     
def calculate_kgtp_rho(num_nodes, rho_membrane_active, intercellular_contact_factors, migr_bdry_contact_factors, exponent_rho_autoact, threshold_rho_autoact, kgtp_rho_baseline, kgtp_rho_autoact_baseline):
    
    result = np.empty(num_nodes)
    for i in range(num_nodes):
        kgtp_rho_autoact = kgtp_rho_autoact_baseline*hill_function(exponent_rho_autoact, threshold_rho_autoact, rho_membrane_active[i])
        
        i_minus_1 = (i - 1)%num_nodes
        i_plus_1 = (i + 1)%num_nodes
        
        migr_bdry_contact_factor_average = (migr_bdry_contact_factors[i_minus_1] + migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus_1])/3.0
        
        intercellular_contact_factor_average = (intercellular_contact_factors[i_minus_1] + intercellular_contact_factors[i] + intercellular_contact_factors[i_plus_1])/3.0
        
        result[i] = (migr_bdry_contact_factor_average)*(intercellular_contact_factor_average)*(kgtp_rho_autoact + kgtp_rho_baseline)
        
    return result

# -----------------------------------------------------------------
             
def calculate_kdgtp_rac(num_nodes, rho_membrane_active, exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, kdgtp_rac_baseline, kdgtp_rho_mediated_rac_inhib_baseline, intercellular_contact_factors, migr_bdry_contact_factors, tension_mediated_rac_inhibition_exponent, tension_mediated_rac_inhibition_multiplier, tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain, local_strains, tension_fn_type):
    result = np.empty(num_nodes, dtype=np.float64)
    
    global_tension = np.sum(local_strains)/num_nodes
    
    for i in range(num_nodes):        
        kdgtp_rho_mediated_rac_inhib = kdgtp_rho_mediated_rac_inhib_baseline*hill_function(exponent_rho_mediated_rac_inhib, threshold_rho_mediated_rac_inhib, rho_membrane_active[i])
        
        i_minus_1 = (i - 1)%num_nodes
        i_plus_1 = (i + 1)%num_nodes
        
        migr_bdry_contact_factor_average = (migr_bdry_contact_factors[i_minus_1] + migr_bdry_contact_factors[i] + migr_bdry_contact_factors[i_plus_1])/3.0
        
        intercellular_contact_factor_average = (intercellular_contact_factors[i_minus_1] + intercellular_contact_factors[i] + intercellular_contact_factors[i_plus_1])/3.0
        
        strain_inhibition = 1.0
        if tension_fn_type == 0:
            strain_inhibition = 2 - calculate_strain_mediated_rac_activation_reduction_using_neg_exp(local_strains[i], tension_mediated_rac_inhibition_exponent)
        elif tension_fn_type == 1:
            strain_inhibition = 2 - calculate_strain_mediated_rac_activation_reduction_using_inv_fn(local_strains[i], tension_mediated_rac_inhibition_multiplier)
        elif tension_fn_type == 2:
            strain_inhibition = 2 - calculate_strain_mediated_rac_activation_reduction_using_hill_fn(local_strains[i], tension_mediated_rac_hill_exponent, tension_mediated_rac_inhibition_half_strain)
        elif tension_fn_type == 3:
            b = 1.0
            m = (2.0 - b)/tension_mediated_rac_inhibition_half_strain
            strain_inhibition = m*local_strains[i] + b
        elif tension_fn_type == 4:
            exponent = np.log(2)/tension_mediated_rac_inhibition_half_strain
            strain_inhibition = np.exp(exponent*local_strains[i])
        elif tension_fn_type == 5:
            exponent = np.log(2)/tension_mediated_rac_inhibition_half_strain
            strain_inhibition = np.exp(exponent*global_tension)
        
        result[i] = (migr_bdry_contact_factor_average)*(intercellular_contact_factor_average)*(strain_inhibition)*(kdgtp_rac_baseline + kdgtp_rho_mediated_rac_inhib)
        
    return result
        
# -----------------------------------------------------------------
     
def calculate_kdgtp_rho(num_nodes, rac_membrane_active, exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, kdgtp_rho_baseline, kdgtp_rac_mediated_rho_inhib_baseline):
    
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        kdgtp_rac_mediated_rho_inhib = kdgtp_rac_mediated_rho_inhib_baseline*hill_function(exponent_rac_mediated_rho_inhib, threshold_rac_mediated_rho_inhib, rac_membrane_active[i])
        
        result[i] = 1.0*(kdgtp_rho_baseline + kdgtp_rac_mediated_rho_inhib)
        
    return result
        
# -----------------------------------------------------------------

def calculate_diffusion(num_nodes, species, diffusion_constant, edgeplus_lengths):
    result = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        i_minus_1_index = (i - 1)%num_nodes
        i_plus_1_index = (i + 1)%num_nodes
        
        edgeplus_length_i = edgeplus_lengths[i]
        edgeplus_length_i_minus_1 = edgeplus_lengths[i_minus_1_index]
        
        delta_i_plus_1 = (species[i_plus_1_index] - species[i])/(edgeplus_length_i*edgeplus_length_i)
        
        delta_i_minus_1 = (species[i_minus_1_index] - species[i])/(edgeplus_length_i_minus_1*edgeplus_length_i_minus_1)
        
        result[i] = diffusion_constant*(delta_i_plus_1 + delta_i_minus_1)
        
    return result
    
# -----------------------------------------------------------------------------
    
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
    
# -----------------------------------------------------------------------------
    
def points_are_same_within_tolerance(a, b, tolerance):
    tolerance_squared = tolerance*tolerance

    ax, ay = a
    bx, by = b
    
    dx = ax - bx
    dy = ay - by
    
    if dx*dx + dy*dy < tolerance_squared:
        return True
    else:
        return False
        
# -----------------------------------------------------------------

def get_min_max_of_line_segment(a0, a1):
    a0x, a0y = a0
    a1x, a1y = a1
    
    if a0x < a1x or a0x == a1x:
        min_ax = a0x
        max_ax = a1x
    else:
        min_ax = a1x
        max_ax = a0x

    if a0y < a1y or a0y == a1y:
        min_ay = a0y
        max_ay = a1y
    else:
        min_ay = a1y
        max_ay = a0y
        
    return min_ax, max_ax, min_ay, max_ay
    
# -----------------------------------------------------------------

def check_if_numbers_are_same_within_tolerance(x, y, tolerance):
    if np.abs(x - y) < tolerance:
        return True
    else:
        return False
        
# -----------------------------------------------------------------
        
def check_if_line_segment_bounding_boxes_intersect(min_ax, max_ax, min_ay, max_ay, min_bx, max_bx, min_by, max_by, tolerance=1e-4):
    
    is_min_ax_same_as_min_bx_or_max_bx = check_if_numbers_are_same_within_tolerance(min_ax, min_bx, tolerance) or check_if_numbers_are_same_within_tolerance(min_ax, max_bx, tolerance)
    is_min_ay_same_as_min_by_or_max_by = check_if_numbers_are_same_within_tolerance(min_ay, min_by, tolerance) or check_if_numbers_are_same_within_tolerance(min_ay, max_by, tolerance)
    
    is_max_ax_same_as_min_bx_or_max_bx = check_if_numbers_are_same_within_tolerance(max_ax, min_bx, tolerance) or check_if_numbers_are_same_within_tolerance(max_ax, max_bx, tolerance)
    is_max_ay_same_as_min_by_or_max_by = check_if_numbers_are_same_within_tolerance(max_ay, min_by, tolerance) or check_if_numbers_are_same_within_tolerance(max_ay, max_by, tolerance)
    
    if (is_min_ax_same_as_min_bx_or_max_bx or min_bx < min_ax < max_bx) and (is_min_ay_same_as_min_by_or_max_by or min_by < min_ay < max_by):
        return True
    if (is_min_ax_same_as_min_bx_or_max_bx or min_bx < min_ax < max_bx) and (is_max_ay_same_as_min_by_or_max_by or min_by < max_ay < max_bx):
        return True
    if (is_max_ax_same_as_min_bx_or_max_bx or min_bx < max_ax < max_bx) and (is_max_ay_same_as_min_by_or_max_by or min_by < max_ay < max_bx):
        return True
    if (is_max_ax_same_as_min_bx_or_max_bx or min_bx < max_ax < max_bx) and (is_min_ay_same_as_min_by_or_max_by or min_by < min_ay < max_by):
        return True
        
    return False
    
# -----------------------------------------------------------------

def check_if_point_is_on_line_segment(px, py, min_ax, max_ax, min_ay, max_ay):
    if min_ax <= px <= max_ax and min_ay <= py <= max_ay:
        return True
    else:
        return False
        
# -----------------------------------------------------------------
        
      
def check_if_line_segments_intersect(a0, a1, b0, b1):
    min_ax, max_ax, min_ay, max_ay = get_min_max_of_line_segment(a0, a1)
    min_bx, max_bx, min_by, max_by = get_min_max_of_line_segment(b0, b1)
    
    #if check_if_line_segment_bounding_boxes_intersect(min_ax, max_ax, min_ay, max_ay, min_bx, max_bx, min_by, max_by) or check_if_line_segment_bounding_boxes_intersect(min_bx, max_bx, min_by, max_by, min_ax, max_ax, min_ay, max_ay):
    a0x, a0y = a0
    a1x, a1y = a1
    b0x, b0y = b0
    b1x, b1y = b1

    ma = 0.0
    if check_if_numbers_are_same_within_tolerance(a1x, a0x, 1e-6):
        ma = (a1y - a0y)/(1e-6)
    else:
        mb = (a1y - a0y)/(a1x - a0x)
        
    mb = 0.0
    if check_if_numbers_are_same_within_tolerance(a1x, a0x, 1e-6):
        ma = (b1y - b0y)/(1e-6)
    else:
        mb = (b1y - b0y)/(b1x - b0x)
    
    if check_if_numbers_are_same_within_tolerance(ma, mb, 1e-6):
        return False
    
    ca = a0y - ma*a0x
    cb = b0y - mb*b0x
    
    x_intersect = (cb - ca)/(ma - mb)
    y_intersect = ma*x_intersect + ca
    
    if check_if_point_is_on_line_segment(x_intersect, y_intersect, min_ax, max_ax, min_ay, max_ay) and check_if_point_is_on_line_segment(x_intersect, y_intersect, min_bx, max_bx, min_by, max_by):
        return True
    else:
        return False
        
# -----------------------------------------------------------------
      
def check_if_line_segment_from_polygon_vertex_goes_through_same_polygon(vi_a, coords_b, polygon_coords):
    num_vertices = polygon_coords.shape[0]
    coords_a = polygon_coords[vi_a]
    
    vi_a_minus_1 = (vi_a - 1)%num_vertices
    
    for vi in range(num_vertices):
        if vi == vi_a or vi == vi_a_minus_1:
            continue
        else:
            this_vertex_coords = polygon_coords[vi]
            this_vertex_plus_1 = (vi + 1)%num_vertices
            
#            if points_are_same_within_tolerance(this_vertex_coords, coords_a, 1e-4):
#                continue
            
            if check_if_line_segments_intersect(coords_a, coords_b, this_vertex_coords, polygon_coords[this_vertex_plus_1]):
                return True
    return False

# -----------------------------------------------------------------
 
def check_if_line_segment_goes_through_polygon(a, b, polygon_coords):
    num_vertices = polygon_coords.shape[0]
    for vi in range(num_vertices):
        if check_if_line_segments_intersect(a, b, polygon_coords[vi], polygon_coords[(vi + 1)%num_vertices]):
            return True
    return False
            
# -----------------------------------------------------------------
              
def check_if_line_segment_between_vertices_goes_through_any_polygon(pi_a, vi_a, pi_b, vi_b, all_polygon_coords):
    num_polygons = all_polygon_coords.shape[0]
    
    if pi_a == pi_b:
        return True
    else:
        coords_a = all_polygon_coords[pi_a][vi_a]
        coords_b = all_polygon_coords[pi_b][vi_b]
        
        if check_if_line_segment_from_polygon_vertex_goes_through_same_polygon(vi_a, coords_b, all_polygon_coords[pi_a]):
            return True
        elif check_if_line_segment_from_polygon_vertex_goes_through_same_polygon(vi_b, coords_a, all_polygon_coords[pi_b]):
            return True
        else:
            for pi in range(num_polygons):
                if pi == pi_a or pi == pi_b:
                    continue
                else:
                    this_poly_coords = all_polygon_coords[pi]
                    if check_if_line_segment_goes_through_polygon(coords_a, coords_b, this_poly_coords):
                        return True
                        
    return False
    
# -----------------------------------------------------------------

def calculate_coa_signals(this_cell_index, num_nodes, num_cells, coa_distribution_exponent, coa_sensitivity_exponent, coa_belt_offset,  cell_dependent_coa_signal_strengths, intercellular_dist_squared_matrix, cells_bounding_box_array, all_cells_node_coords):
    coa_signals = np.zeros(num_nodes, dtype=np.float64)
    #coa_at_belt_offset = np.exp(coa_distribution_exponent*coa_belt_offset)
    for ni in range(num_nodes):
        this_node_coa_signal = 0.0
        
        for other_ci in range(num_cells):
            if other_ci != this_cell_index:
                signal_strength = cell_dependent_coa_signal_strengths[other_ci]
                
                #cutoff_distance = np.log(0.01/signal_strength)/coa_distribution_exponent
                #cutoff_distance_squared = cutoff_distance*cutoff_distance
                
                for other_ni in range(num_nodes):
                    #if dist_squared_between_nodes < cutoff_distance_squared:
                    if geometry.check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(this_cell_index, ni, other_ci, other_ni, all_cells_node_coords, cells_bounding_box_array):
                        transduced_coa_signal = 0.0
                    else:
                        dist_squared_between_nodes = intercellular_dist_squared_matrix[ni][other_ci][other_ni]
                        
                        untransduced_coa_signal = 0.0
                        if dist_squared_between_nodes < 1e-6:
                            untransduced_coa_signal = np.exp(coa_distribution_exponent*1e-3)
                        else:
                            untransduced_coa_signal = np.exp(coa_distribution_exponent*np.sqrt(dist_squared_between_nodes))
                        
                        if coa_sensitivity_exponent > 0:
                            transduced_coa_signal = untransduced_coa_signal
                        else:
                            transduced_coa_signal = 0.0#np.exp(coa_sensitivity_exponent*(coa_at_belt_offset - untransduced_coa_signal))
                            
                            if transduced_coa_signal > 1.0:
                                transduced_coa_signal = 0.0
#                else:
#                    transduced_coa_signal = 0.0
                    
                    this_node_coa_signal = this_node_coa_signal + transduced_coa_signal*signal_strength

                    
        coa_signals[ni] = this_node_coa_signal
        
    return coa_signals