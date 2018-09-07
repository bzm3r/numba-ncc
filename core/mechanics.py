# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:26:37 2015

@author: Brian
"""

from . import geometry
import numpy as np
import numba as nb

        
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def capped_linear_function(max_x, x):
    if x > max_x:
        return 1.0
    else:
        return (x/max_x)
    
# -----------------------------------------------------------------
@nb.jit(nopython=True)  
def linear_function(max_x, x):
    return (x/max_x)
        
# -----------------------------------------------------------------
@nb.jit(nopython=True)         
def calculate_phys_space_bdry_contact_factors(num_nodes, this_cell_coords, space_physical_bdry_polygons):
    
    if space_physical_bdry_polygons.size == 0:
        return np.zeros(len(this_cell_coords))
    else:
        return geometry.are_points_inside_polygons(this_cell_coords, space_physical_bdry_polygons)

# -----------------------------------------------------------------
@nb.jit(nopython=True)    
def calculate_migr_bdry_contact_factors(num_nodes, this_cell_coords, space_migratory_bdry_polygon, migr_bdry_contact_factor_mag):
    
    are_nodes_in_migr_space = geometry.are_points_inside_polygon(this_cell_coords, space_migratory_bdry_polygon)
    
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
    current_area = abs(geometry.calculate_polygon_area(this_cell_coords))
        
    if current_area < area_resting:
        area_strain = (current_area - area_resting)/area_resting
        force_mag = area_strain*stiffness_cytoplasmic/num_nodes
        
        return geometry.multiply_vectors_by_scalar(unit_inside_pointing_vectors, force_mag)
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
    
    plus_dirn_edge_length = geometry.calculate_2D_vector_mags(edge_vectors_to_plus)
    
    minus_dirn_edge_length = geometry.calculate_2D_vector_mags(edge_vectors_to_minus)
    
    edge_strains_plus = np.empty(num_nodes, dtype=np.float64)
    edge_strains_minus = np.empty(num_nodes, dtype=np.float64)
    local_average_strains = np.empty(num_nodes, dtype=np.float64)
    
    for i in range(num_nodes):
        edge_strain_plus = (plus_dirn_edge_length[i] - length_edge_resting)/length_edge_resting
        edge_strain_minus = (minus_dirn_edge_length[i] - length_edge_resting)/length_edge_resting
        
        edge_strains_plus[i] = edge_strain_plus
        edge_strains_minus[i] = edge_strain_minus
        
        local_average_strains[i] = 0.5*edge_strain_plus + 0.5*edge_strain_minus
    
    unit_edge_disp_vecs_plus = geometry.normalize_vectors(edge_vectors_to_plus)
    unit_edge_disp_vecs_minus = geometry.normalize_vectors(edge_vectors_to_minus)
    
    EFplus_mags = np.zeros(num_nodes, dtype=np.float64)
    EFminus_mags = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        EFplus_mags[i] = edge_strains_plus[i]*stiffness_edge
        EFminus_mags[i] = edge_strains_minus[i]*stiffness_edge
        
    EFplus = geometry.multiply_vectors_by_scalars(unit_edge_disp_vecs_plus, EFplus_mags)
    
    EFminus = geometry.multiply_vectors_by_scalars(unit_edge_disp_vecs_minus, EFminus_mags)
    
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
def calculate_rgtpase_mediated_forces(num_nodes, this_cell_coords, rac_membrane_actives, rho_membrane_actives, threshold_force_rac_activity, threshold_force_rho_activity, max_force_rac, max_force_rho, unit_inside_pointing_vectors):   
    rgtpase_mediated_force_mags = np.zeros(num_nodes, dtype=np.float64)
    
    for ni in range(num_nodes):
        force_mag = 0.0
        rac_activity = rac_membrane_actives[ni]
        rho_activity = rho_membrane_actives[ni]
        
        if rac_activity > rho_activity:
            force_mag = max_force_rac*capped_linear_function(2*threshold_force_rac_activity, rac_activity - rho_activity)
        else:
            force_mag = -1*max_force_rho*capped_linear_function(2*threshold_force_rho_activity, rho_activity - rac_activity)
            
        rgtpase_mediated_force_mags[ni] = -1*force_mag
            
    result = np.empty((num_nodes, 2), dtype=np.float64)
    result = geometry.multiply_vectors_by_scalars(unit_inside_pointing_vectors, rgtpase_mediated_force_mags)
    
    return result 

# ---------------------------------------------------------------------

@nb.jit(nopython=True)
def capped_linear_function_adhesion(x, normalizer):
    result = x/normalizer
    
    if result < 0.0 or result > 1.0:
        return 0.0
    else:
        return result

# -----------------------------------------------------------------------
        
@nb.jit(nopython=True)
def calculate_adhesion_forces(num_nodes, num_cells, this_ci, this_cell_coords, this_cell_forces, force_adh_constant, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria, unit_inside_pointing_vectors, max_force_adh):
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
                        
                    relative_force = close_point_force - this_node_force
                    #proj_this_force_on_uiv = geometry.calculate_projection_of_a_on_b(this_node_force, this_node_uiv)
                    #prof_other_force_on_uiv = geometry.calculate_projection_of_a_on_b(close_point_force, this_node_uiv)
                    
                    if abs(geometry.calculate_projection_of_a_on_b(relative_force, this_node_uiv)) < 1e-1:
                        continue
                    else:
                        force_adh = force_adh_constant*relative_force*capped_linear_function_adhesion(dist, 3*closeness_dist_criteria)
                        
                        if geometry.calculate_2D_vector_mag(force_adh) > max_force_adh:
                            force_adh = 0.0*force_adh
                            
                        this_node_F_adh += force_adh
                    
        F_adh[ni] = this_node_F_adh
        
    return F_adh

# -----------------------------------------------------------------------
        
@nb.jit(nopython=True)
def calculate_external_forces(num_nodes, num_cells, this_ci, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria, unit_inside_pointing_vectors):
    F_external = np.zeros((num_nodes, 2), dtype=np.float64)
    
    for ni in range(num_nodes):
        close_point_force = np.zeros(2, dtype=np.float64)
        uiv = unit_inside_pointing_vectors[ni]
        
        for ci in range(num_cells):
            if ci != this_ci:
                if close_point_on_other_cells_to_each_node_exists[ni][ci] == 1:
                    close_ni_a, close_ni_b = close_point_on_other_cells_to_each_node_indices[ni][ci]
                    
                    if close_ni_a == close_ni_b:
                        # closest point is another single node
                        close_point_force = all_cells_node_forces[ci][close_ni_a]
                        force_proj_mag = geometry.calculate_projection_of_a_on_b(close_point_force, uiv)
                        
                        if force_proj_mag < 0.0:
                            force_proj_mag = 0.0
                            
                        F_external[ni] += force_proj_mag*uiv
                    else:
                        # closest point is on a line segment between two nodes
                        close_ni_a_force = all_cells_node_forces[ci][close_ni_a]
                        close_ni_b_force = all_cells_node_forces[ci][close_ni_b]
                        
                        a_proj_mag = geometry.calculate_projection_of_a_on_b(close_ni_a_force, uiv)
                        b_proj_mag = geometry.calculate_projection_of_a_on_b(close_ni_b_force, uiv)
                        
                        if a_proj_mag < 0.0:
                            a_proj_mag = 0.0
                        if b_proj_mag < 0.0:
                            b_proj_mag = 0.0
                        
                        F_external[ni] += (a_proj_mag + b_proj_mag)*uiv
                    
    return F_external

# ----------------------------------------------------------------------------
@nb.jit(nopython=True)  
def calculate_forces(num_nodes, num_cells, this_ci, this_cell_coords, rac_membrane_actives, rho_membrane_actives, length_edge_resting, stiffness_edge, threhsold_force_rac_activity, threhsold_force_rho_activity, max_force_rac, max_force_rho, force_adh_constant, area_resting, stiffness_cytoplasmic, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria):
    
    unit_inside_pointing_vectors = geometry.calculate_unit_inside_pointing_vecs(this_cell_coords)
    
    rgtpase_mediated_forces = calculate_rgtpase_mediated_forces(num_nodes, this_cell_coords, rac_membrane_actives, rho_membrane_actives, threhsold_force_rac_activity, threhsold_force_rho_activity, max_force_rac, max_force_rho, unit_inside_pointing_vectors)

    F_cytoplasmic = calculate_cytoplasmic_force(num_nodes, this_cell_coords, area_resting, stiffness_cytoplasmic, unit_inside_pointing_vectors)
    
    local_strains, EFplus, EFminus = calculate_spring_edge_forces(num_nodes, this_cell_coords, stiffness_edge, length_edge_resting) 
    
    F_internal = rgtpase_mediated_forces + EFplus + EFminus + F_cytoplasmic
    
    #F_external = calculate_external_forces(num_nodes, num_cells, this_ci, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria, unit_inside_pointing_vectors)
    
    F = F_internal #+ F_external
    
    return F, EFplus, EFminus, rgtpase_mediated_forces, F_cytoplasmic, local_strains, unit_inside_pointing_vectors
    
# =============================================================================
