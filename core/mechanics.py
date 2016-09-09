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
def calculate_phys_space_bdry_contact_factors(num_nodes, this_cell_coords, space_physical_bdry_polygons):
    
    if space_physical_bdry_polygons.size == 0:
        return np.zeros(len(this_cell_coords))
    else:
        return geometry.are_points_inside_polygons(num_nodes, this_cell_coords, space_physical_bdry_polygons)

# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_migr_bdry_contact_factors(num_nodes, this_cell_coords, space_migratory_bdry_polygon, migr_bdry_contact_factor_mag):
    
    are_nodes_in_migr_space = geometry.are_points_inside_polygon(num_nodes, this_cell_coords, space_migratory_bdry_polygon.shape[0], space_migratory_bdry_polygon)
    
    result = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        if are_nodes_in_migr_space[i] == 1:
            result[i] = 1.0
        else:
            result[i] = migr_bdry_contact_factor_mag  
    
    return result

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_local_strains(this_cell_coords, length_edge_resting):
    average_edge_lengths = geometry.calculate_average_edge_lengths(this_cell_coords)
    
    num_nodes = this_cell_coords.shape[0]
    local_strains = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        local_strains[i] = (average_edge_lengths[i] - length_edge_resting)/length_edge_resting
        
    return local_strains

# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_cytoplasmic_force(num_nodes, this_cell_coords, area_resting, stiffness_cytoplasmic, unit_inside_pointing_vectors):
    current_area = abs(geometry.calculate_polygon_area(num_nodes, this_cell_coords))
        
    if current_area < area_resting:
        area_strain = (current_area - area_resting)/area_resting
        force_mag = area_strain*stiffness_cytoplasmic/num_nodes
        
        return geometry.multiply_vectors_by_scalar(num_nodes, unit_inside_pointing_vectors, force_mag)
    else:
        return np.zeros((num_nodes, 2))
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_spring_edge_forces(num_nodes, this_cell_coords, stiffness_edge, length_edge_resting):
    
    edge_vectors_to_plus = np.empty((num_nodes, 2), dtype=np.float64)
    edge_vectors_to_minus = np.empty((num_nodes, 2), dtype=np.float64)
    
    for i in range(num_nodes):
        i_plus_1 = (i + 1)%num_nodes
        i_minus_1 = (i - 1)%num_nodes
        edge_vector_to_plus = geometry.calculate_vector_from_p1_to_p2_given_vectors(this_cell_coords[i], this_cell_coords[i_plus_1])
        edge_vector_to_minus = geometry.calculate_vector_from_p1_to_p2_given_vectors(this_cell_coords[i], this_cell_coords[i_minus_1])
        
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
def determine_rac_rho_domination(rac_membrane_actives, rho_membrane_actives):
    num_nodes = rac_membrane_actives.shape[0]
    
    domination_array = np.empty(num_nodes, dtype=np.int64)
    
    for ni in range(num_nodes):
        if rac_membrane_actives[ni] < rho_membrane_actives[ni]:
            domination_array[ni] = 0
        else:
            domination_array[ni] = 1
            
    return domination_array

# -----------------------------------------------------------------
    
@nb.jit(nopython=True)  
def calculate_rgtpase_mediated_forces(num_nodes, this_cell_coords, rac_membrane_actives, rho_membrane_actives, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, unit_inside_pointing_vectors, squeeze_effect_threshold):   
    rgtpase_mediated_force_mags = np.zeros(num_nodes, dtype=np.float64)

    for ni in range(num_nodes):
        nodal_rac_activity = rac_membrane_actives[ni]
        nodal_rho_activity = rho_membrane_actives[ni]

        ni_coord = this_cell_coords[ni]
        ni_plus2_coord = this_cell_coords[(ni + 2)%num_nodes]
        ni_minus2_coord = this_cell_coords[(ni - 2)%num_nodes]
        
        if rac_membrane_actives[ni] < rho_membrane_actives[ni]:
            if geometry.calculate_2D_vector_mag(ni_coord - ni_plus2_coord) < squeeze_effect_threshold or geometry.calculate_2D_vector_mag(ni_coord - ni_minus2_coord) < squeeze_effect_threshold:
                rgtpase_mediated_force_mags[ni] = 0.0
            else:
                rgtpase_mediated_force_mags[ni] = hill_function(force_rho_exp, force_rho_threshold, nodal_rho_activity)*force_rho_max_mag
        else:
            rgtpase_mediated_force_mags[ni] = -1*hill_function(force_rac_exp, force_rac_threshold, nodal_rac_activity)*force_rac_max_mag
    
    result = np.empty((num_nodes, 2), dtype=np.float64)
    result = geometry.multiply_vectors_by_scalars(num_nodes, unit_inside_pointing_vectors, rgtpase_mediated_force_mags)
    
    return result 

# ----------------------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_adhesion_forces(num_nodes, num_cells, this_ci, this_cell_coords, this_cell_forces, force_adh_constant, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria, unit_inside_pointing_vectors):
    F_adh = np.zeros((num_nodes, 2), dtype=np.float64)
    close_point_force = np.zeros(2, dtype=np.float64)
    
    for ni in range(num_nodes):
        this_this_cell_coords = this_cell_coords[ni]
        this_node_force = this_cell_forces[ni]
        this_node_F_adh = np.zeros(2, dtype=np.float64)
        this_node_uiv = unit_inside_pointing_vectors[ni]
        for ci in range(num_cells):
            if ci != this_ci:
                if close_point_on_other_cells_to_each_node_exists[ni][ci] == 1:
                    dist = geometry.calculate_2D_vector_mag(close_point_on_other_cells_to_each_node[ni][ci] - this_this_cell_coords)
                    close_ni_a, close_ni_b = close_point_on_other_cells_to_each_node_indices[ni][ci]
                    
                    if close_ni_a == close_ni_b:
                        # closest point is another node
                        close_point_force = all_cells_node_forces[ci][close_ni_a]
                    else:
                        # closest point is on a line segment between two nodes
                        close_ni_a_force = all_cells_node_forces[ci][close_ni_a]
                        close_ni_b_force = all_cells_node_forces[ci][close_ni_b]
                        projection_factor = close_point_on_other_cells_to_each_node_projection_factors[ni][ci]
                        close_point_force = (1 - projection_factor)*close_ni_a_force + projection_factor*close_ni_b_force
                        
                    M_d = (1.0 - (dist/closeness_dist_criteria))
                    if M_d > 1.0 or M_d < 0.0:
                        M_d = 0

                    relative_force = close_point_force - this_node_force
                    #proj_this_force_on_uiv = geometry.calculate_projection_of_a_on_b(this_node_force, this_node_uiv)
                    #prof_other_force_on_uiv = geometry.calculate_projection_of_a_on_b(close_point_force, this_node_uiv)
                    
                    if geometry.calculate_projection_of_a_on_b(relative_force, this_node_uiv) > 0:
                        continue
                    else:
                        this_node_F_adh += force_adh_constant*relative_force*M_d
                    
        F_adh[ni] = this_node_F_adh
        
    return F_adh
    
# ----------------------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_forces(num_nodes, num_cells, this_ci, this_cell_coords, rac_membrane_actives, rho_membrane_actives, length_edge_resting, stiffness_edge, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, force_adh_constant, area_resting, stiffness_cytoplasmic, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria):
    
    unit_inside_pointing_vectors = geometry.calculate_unit_inside_pointing_vecs(this_cell_coords)
    
    rgtpase_mediated_forces = calculate_rgtpase_mediated_forces(num_nodes, this_cell_coords, rac_membrane_actives, rho_membrane_actives, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, unit_inside_pointing_vectors, closeness_dist_criteria)

    F_cytoplasmic = calculate_cytoplasmic_force(num_nodes, this_cell_coords, area_resting, stiffness_cytoplasmic, unit_inside_pointing_vectors)
    
    local_strains, EFplus, EFminus = calculate_spring_edge_forces(num_nodes, this_cell_coords, stiffness_edge, length_edge_resting) 
    
    F_non_adh = rgtpase_mediated_forces + EFplus + EFminus + F_cytoplasmic
    
    if force_adh_constant > 1e-6:
        F_adh = calculate_adhesion_forces(num_nodes, num_cells, this_ci, this_cell_coords, F_non_adh, force_adh_constant, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria, unit_inside_pointing_vectors)
    else:
        F_adh = np.zeros((num_nodes, 2), dtype=np.float64)
    
    F = F_non_adh + F_adh
    
    return F, EFplus, EFminus, rgtpase_mediated_forces, F_cytoplasmic, F_adh, local_strains, unit_inside_pointing_vectors
    
# =============================================================================
