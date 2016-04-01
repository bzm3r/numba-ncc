# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:26:37 2015

@author: Brian
"""

import geometry
import numpy as np
import numba as nb

        
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def hill_function(exp, thresh, sig):
    pow_sig = sig**exp
    pow_thresh = thresh**exp
    
    return pow_sig/(pow_thresh + pow_sig)
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)         
def calculate_phys_space_bdry_contact_factors(num_nodes, node_coords, space_physical_bdry_polygons):
    
    if space_physical_bdry_polygons.size == 0:
        return np.zeros(len(node_coords))
    else:
        return geometry.are_points_inside_polygons(num_nodes, node_coords, space_physical_bdry_polygons)

# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_migr_bdry_contact_factors(num_nodes, node_coords, space_migratory_bdry_polygon, migr_bdry_contact_factor_mag):
    
    are_nodes_in_migr_space = geometry.are_points_inside_polygon(num_nodes, node_coords, space_migratory_bdry_polygon.shape[0], space_migratory_bdry_polygon)
    
    result = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        if are_nodes_in_migr_space[i] == 1:
            result[i] = 1.0
        else:
            result[i] = migr_bdry_contact_factor_mag  
    
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_local_strains(node_coords, length_edge_resting):
    average_edge_lengths = geometry.calculate_average_edge_lengths(node_coords)
    
    num_nodes = node_coords.shape[0]
    local_strains = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        local_strains[i] = (average_edge_lengths[i] - length_edge_resting)/length_edge_resting
        
    return local_strains

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_cytoplasmic_force(num_nodes, node_coords, area_resting, stiffness_cytoplasmic, unit_inside_pointing_vectors):
    current_area = abs(geometry.calculate_polygon_area(num_nodes, node_coords))
        
    if current_area < area_resting:
        area_strain = (current_area - area_resting)/area_resting
        force_mag = area_strain*stiffness_cytoplasmic/num_nodes
        
        return geometry.multiply_vectors_by_scalar(num_nodes, unit_inside_pointing_vectors, force_mag)
    else:
        return np.zeros((num_nodes, 2))
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_spring_edge_forces(num_nodes, node_coords, stiffness_edge, length_edge_resting):
    
    edge_vectors_to_plus = np.empty((num_nodes, 2), dtype=np.float64)
    edge_vectors_to_minus = np.empty((num_nodes, 2), dtype=np.float64)
    
    for i in range(num_nodes):
        i_plus_1 = (i + 1)%num_nodes
        i_minus_1 = (i - 1)%num_nodes
        edge_vector_to_plus = geometry.calculate_vector_from_p1_to_p2_given_vectors(node_coords[i], node_coords[i_plus_1])
        edge_vector_to_minus = geometry.calculate_vector_from_p1_to_p2_given_vectors(node_coords[i], node_coords[i_minus_1])
        
        edge_vectors_to_plus[i, 0] = edge_vector_to_plus[0]
        edge_vectors_to_plus[i, 1] = edge_vector_to_plus[1]
        
        edge_vectors_to_minus[i, 0] = edge_vector_to_minus[0]
        edge_vectors_to_minus[i, 1] = edge_vector_to_minus[1]
    
    plus_dirn_edge_length = geometry.calculate_2D_vector_mags(num_nodes, edge_vectors_to_plus)
    
    minus_dirn_edge_length = geometry.calculate_2D_vector_mags(num_nodes, edge_vectors_to_minus)
    
    edge_strains_plus = np.empty(num_nodes, dtype=np.float64)
    edge_strains_minus = np.empty(num_nodes, dtype=np.float64)
    local_average_strains = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        edge_strain_plus = (plus_dirn_edge_length[i] - length_edge_resting)/length_edge_resting
        edge_strain_minus = (minus_dirn_edge_length[i] - length_edge_resting)/length_edge_resting
        
        edge_strains_plus[i] = edge_strain_plus
        edge_strains_minus[i] = edge_strain_minus
        
        local_average_strains[i] = 0.5*edge_strain_plus + 0.5*edge_strain_minus
    
    unit_edge_disp_vecs_plus = geometry.normalize_vectors(num_nodes, edge_vectors_to_plus)
    unit_edge_disp_vecs_minus = geometry.normalize_vectors(num_nodes, edge_vectors_to_minus)
    
    EFplus_mags = np.zeros(num_nodes, dtype=np.float64)
    EFminus_mags = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        EFplus_mags[i] = edge_strains_plus[i]*stiffness_edge
        EFminus_mags[i] = edge_strains_minus[i]*stiffness_edge
        
    EFplus = geometry.multiply_vectors_by_scalars(num_nodes, unit_edge_disp_vecs_plus, EFplus_mags)
    
    EFminus = geometry.multiply_vectors_by_scalars(num_nodes, unit_edge_disp_vecs_minus, EFminus_mags)
    
    return local_average_strains, EFplus, EFminus

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_rgtpase_mediated_forces(num_nodes, node_coords, rac_membrane_actives, rho_membrane_actives, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, unit_inside_pointing_vectors):   
    rgtpase_mediated_force_mags = np.zeros(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        nodal_rac_activity = rac_membrane_actives[i]
        nodal_rho_activity = rho_membrane_actives[i]

        if nodal_rho_activity > nodal_rac_activity:
            rgtpase_mediated_force_mags[i] = hill_function(force_rho_exp, force_rho_threshold, nodal_rho_activity)*force_rho_max_mag
        else:
            rgtpase_mediated_force_mags[i] = -1*hill_function(force_rac_exp, force_rac_threshold, nodal_rac_activity)*force_rac_max_mag
    
    result = np.empty((num_nodes, 2), dtype=np.float64)
    result = geometry.multiply_vectors_by_scalars(num_nodes, unit_inside_pointing_vectors, rgtpase_mediated_force_mags)
    
    return result 
    
# ----------------------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_forces(num_nodes, node_coords, rac_membrane_actives, rho_membrane_actives, length_edge_resting, stiffness_edge, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, area_resting, stiffness_cytoplasmic):
    
    unit_inside_pointing_vectors = geometry.calculate_unit_inside_pointing_vecs(node_coords)
    
    rgtpase_mediated_forces = calculate_rgtpase_mediated_forces(num_nodes, node_coords, rac_membrane_actives, rho_membrane_actives, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, unit_inside_pointing_vectors)

    F_cytoplasmic = calculate_cytoplasmic_force(num_nodes, node_coords, area_resting, stiffness_cytoplasmic, unit_inside_pointing_vectors)
    
    local_strains, EFplus, EFminus = calculate_spring_edge_forces(num_nodes, node_coords, stiffness_edge, length_edge_resting) 
    
    F = np.empty((num_nodes, 2), dtype=np.float64)
    for i in range(num_nodes):
        rgtpase_mediated_force_x, rgtpase_mediated_force_y = rgtpase_mediated_forces[i]
        EFplus_x, EFplus_y = EFplus[i]
        EFminus_x, EFminus_y = EFminus[i]
        F_cytoplasmic_x, F_cytoplasmic_y = F_cytoplasmic[i]
        
        F[i, 0] = rgtpase_mediated_force_x + EFplus_x + EFminus_x + F_cytoplasmic_x
        F[i, 1] = rgtpase_mediated_force_y + EFplus_y + EFminus_y + F_cytoplasmic_y
    
    
    return F, EFplus, EFminus, rgtpase_mediated_forces, F_cytoplasmic, local_strains, unit_inside_pointing_vectors
    
# =============================================================================
