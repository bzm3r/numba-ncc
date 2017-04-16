# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:21:52 2015

@author: brian
"""

from __future__ import division
import numpy as np
import geometry as geometry
import hardio as hardio
import parameterorg as parameterorg
import general.moore_data_table as moore_data_table
import numba as nb


# ==============================================================================
@nb.jit(nopython=True)
def calculate_centroids_per_tstep(node_coords_per_tstep):
    num_tsteps = node_coords_per_tstep.shape[0]
    
    centroids_per_tstep = np.zeros((num_tsteps, 2), dtype=np.float64)
    
    for ti in range(num_tsteps):
        cx, cy = geometry.calculate_centroid(node_coords_per_tstep[ti])
        centroids_per_tstep[ti][0] = cx
        centroids_per_tstep[ti][1] = cy
        
    return centroids_per_tstep

# ============================================================================== 
def calculate_cell_centroids_for_all_time(cell_index, storefile_path):
    node_coords_per_tstep = hardio.get_node_coords_for_all_tsteps(cell_index, storefile_path)
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)
    
    return centroids_per_tstep

# ==============================================================================
def calculate_cell_centroids_until_tstep(cell_index, max_tstep, storefile_path):
    node_coords_per_tstep = hardio.get_node_coords_until_tstep(cell_index, max_tstep, storefile_path)
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)
    
    return centroids_per_tstep

# ==============================================================================
@nb.jit(nopython=True)   
def calculate_cell_centroids_for_given_times(cell_index, tsteps, storefile_path):
    node_coords_per_tstep = hardio.get_node_coords_for_given_tsteps(cell_index, tsteps, storefile_path)
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)
    
    return centroids_per_tstep

# ==============================================================================   
def calculate_cell_speeds_until_tstep(cell_index, max_tstep, storefile_path, T, L):
    node_coords_per_tstep = hardio.get_node_coords_until_tstep(cell_index, max_tstep, storefile_path)
    centroid_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)*L
    
    num_tsteps = node_coords_per_tstep.shape[0]
    
    timepoints = np.arange(num_tsteps - 1)*T # want times in minutes
    
    velocities = (centroid_per_tstep[2:] - centroid_per_tstep[:-2])/(2*T) # interested in micrometer/min
    
    speeds = geometry.calculate_2D_vector_mags(velocities.shape[0], velocities)
    
    return timepoints, speeds

# ==============================================================================
        
@nb.jit(nopython=True)   
def calculate_polarization_rating(rac_membrane_active, rho_membrane_active, num_nodes):
    
    max_rac = 0.0
    sum_rac = 0.0
    for i in range(num_nodes):
        this_node_rac_active = rac_membrane_active[i]
        sum_rac = sum_rac + this_node_rac_active
        if this_node_rac_active > max_rac:
            max_rac = this_node_rac_active
        
        
    sum_rho = 0.0
    for i in range(num_nodes):
        sum_rho = sum_rho + rho_membrane_active[i]
        
    avg_rac = sum_rac/num_nodes
    avg_rho = sum_rho/num_nodes
    
    if sum_rac > 0.4 or avg_rac < 1e-6:
        return 0.0
    
    if sum_rho > 0.4 or avg_rho < 1e-6:
        return 0.0
    
    significant_rac = np.zeros(num_nodes, dtype=np.int64)
    normalized_rac = rac_membrane_active/max_rac
    for i in range(num_nodes):
        if normalized_rac[i] > 0.2:
            significant_rac[i] = 1

    num_rac_fronts = 0
    front_starts = np.zeros(num_nodes, dtype=np.int64)
    for ni in range(num_nodes):
        ni_plus1 = (ni + 1)%num_nodes
        if significant_rac[ni] == 0 and significant_rac[ni_plus1] == 1:
            front_starts[num_rac_fronts] = ni_plus1
            num_rac_fronts += 1
    
    if num_rac_fronts == 0:
        return 0.0
      
    front_strengths = np.zeros(num_rac_fronts, dtype=np.float64)
    front_widths = np.zeros(num_rac_fronts, dtype=np.float64)
    for fi in range(num_rac_fronts):
        front_strength = 0.0
        front_width = 0.0
        i = front_starts[fi]
        
        while significant_rac[i] != 0:
            front_strength += normalized_rac[i]
            front_width += 1.0
            i = (i + 1)%num_nodes
            
        front_strengths[fi] = front_strength/front_width
        front_widths[fi] = front_width
            
    max_front_strength = np.max(front_strengths)
    normalized_front_strengths = front_strengths/max_front_strength
    
    rac_amount_score = 1.0
    if sum_rac > 0.3:
        rac_amount_score = np.exp((sum_rac - 0.3)*np.log(0.1)/0.1)
        
    rho_amount_score = 1.0
    if sum_rho > 0.2:
        rho_amount_score = np.exp((sum_rho - 0.2)*np.log(0.1)/0.1)
    
    if num_rac_fronts == 1:
        front_width_rating = front_widths[0]/(num_nodes/3.)
        
        if front_width_rating > 1.0:
            front_width_rating = 1.0 - score_function(1.0, 2.0, front_width_rating)
            
        return (front_width_rating)*rac_amount_score*rho_amount_score
    
    elif num_rac_fronts > 1:
        worst_distance_between_fronts = (num_nodes -np.sum(front_widths))/float(num_rac_fronts)
        distance_between_fronts_score = np.zeros(num_rac_fronts, dtype=np.float64)
        
        for fi in range(num_rac_fronts):
            this_si = front_starts[fi]
            this_width = front_widths[fi]
            fi_plus1 = (fi + 1)%(num_rac_fronts)
            next_si = front_starts[fi_plus1]
            if next_si < this_si:
                next_si = num_nodes + next_si
            dist_bw_fronts = next_si - (this_si + this_width)
            
            this_fs = normalized_front_strengths[fi]
            next_fs = normalized_front_strengths[fi_plus1]
            relevant_fs = 1.0
            if this_fs < next_fs:
                relevant_fs = this_fs
            else:
                relevant_fs = next_fs
                
            score = dist_bw_fronts*relevant_fs/worst_distance_between_fronts
            if score > 1.0:
                distance_between_fronts_score[fi] = 1.0
            else:
                distance_between_fronts_score[fi] = score
        
        combined_dist_bw_fronts_score = 1.0
        if num_rac_fronts == 2:
            combined_dist_bw_fronts_score = 1.0 - np.min(distance_between_fronts_score)
        else:
            combined_dist_bw_fronts_score = 1.0 - np.sum(distance_between_fronts_score)/num_rac_fronts
            
        total_front_width = (np.sum(front_widths) + 0.0)
        front_width_rating = total_front_width/(num_nodes/3.)
        
        if front_width_rating > 1.0:
            front_width_rating = 1.0 - score_function(1.0, 2.0, front_width_rating)
            
        return combined_dist_bw_fronts_score*front_width_rating*rac_amount_score*rho_amount_score
        
    return 0.0
# ==============================================================================

def calculate_rgtpase_polarity_score_from_rgtpase_data(rac_membrane_active_per_tstep, rho_membrane_active_per_tstep, significant_difference=0.1, weigh_by_timepoint=False):
    num_nodes = rac_membrane_active_per_tstep.shape[1]
    
    scores_per_tstep = np.array([calculate_polarization_rating(rac_membrane_active, rho_membrane_active, num_nodes) for rac_membrane_active, rho_membrane_active in zip(rac_membrane_active_per_tstep, rho_membrane_active_per_tstep)])

    averaged_score = np.average(scores_per_tstep)
        
    return averaged_score, scores_per_tstep

@nb.jit(nopython=True)
def score_function(min_cutoff, max_cutoff, x):
    if x > max_cutoff:
        return 1.0
    elif x < min_cutoff:
        return 0.0
    else:
        # 0.0 = m*min + b
        # 1.0 = m*max + b
        # 1.0 = m*max - m*min
        # 1.0/(max - min) = m
        # b = -m*min
        return (x - min_cutoff)/(max_cutoff - min_cutoff)
    
def calculate_parameter_exploration_score_from_cell(a_cell, significant_difference=0.1, num_data_points_from_end=None, weigh_by_timepoint=False):
    if num_data_points_from_end == None:
        rac_membrane_active_per_tstep = a_cell.system_history[:, :, parameterorg.rac_membrane_active_index]
        rho_membrane_active_per_tstep = a_cell.system_history[:, :, parameterorg.rho_membrane_active_index]
    else:
        rac_membrane_active_per_tstep = a_cell.system_history[-num_data_points_from_end:, :, parameterorg.rac_membrane_active_index]
        rho_membrane_active_per_tstep = a_cell.system_history[-num_data_points_from_end:, :, parameterorg.rho_membrane_active_index]
    
    polarity_score = score_function(0.0, 0.7, calculate_rgtpase_polarity_score_from_rgtpase_data(rac_membrane_active_per_tstep, rho_membrane_active_per_tstep)[0])
    
    persistence_score = 0.0
    velocity_score = 0.0

    node_coords_per_tstep = a_cell.system_history[:, :, [parameterorg.x_index, parameterorg.y_index]]
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)
    
    cell_centroids = centroids_per_tstep*a_cell.L/1e-6
    num_tsteps = cell_centroids.shape[0]
    
    rougher_cell_centroids = cell_centroids[::30]
    
    movement_steps = rougher_cell_centroids[1:] - rougher_cell_centroids[:-1]
    movement_step_mags = np.linalg.norm(movement_steps, axis=1)
    
    total_distance_travelled = np.sum(movement_step_mags)
        
    if total_distance_travelled < 10.0:
        velocity_score = 0.0
        persistence_score = 0.0
    else:
        net_displacement = cell_centroids[num_tsteps-1] - cell_centroids[0]
        net_displacement_mag = np.linalg.norm(net_displacement)
    
        distance_per_tstep = np.linalg.norm(cell_centroids[1:] - cell_centroids[:num_tsteps-1], axis=1)
        net_distance = np.sum(distance_per_tstep)
        
        persistence = net_displacement_mag/net_distance
        
        velocities = distance_per_tstep*(60.0/a_cell.T)
        average_velocity = np.average(velocities)
        
        persistence_score = 1.0 - score_function(0.5, 1.0, persistence)
        if velocity_score > 3.5:
            velocity_score = 1.0 - score_function(3.5, 5.0, velocity_score)
        else:
            velocity_score = score_function(0.0, 2.5, average_velocity)
    
    return polarity_score, persistence_score, velocity_score

def calculate_parameter_exploration_score_from_cell_no_randomization_variant(a_cell, should_be_polarized_by_in_hours=0.5):
    T = a_cell.T
    should_be_polarized_by_tstep = int((should_be_polarized_by_in_hours*3600.)/T)
       
    rac_membrane_active_per_tstep = a_cell.system_history[(should_be_polarized_by_tstep + 1):, :, parameterorg.rac_membrane_active_index]
    rho_membrane_active_per_tstep = a_cell.system_history[(should_be_polarized_by_tstep + 1):, :, parameterorg.rho_membrane_active_index]
    polarity_score_global = calculate_rgtpase_polarity_score_from_rgtpase_data(rac_membrane_active_per_tstep, rho_membrane_active_per_tstep)[0]
    
    polarity_score_at_SBPBP_tstep = polarity_score_global
    if T != 0:
        rac_membrane_active_around_SBPB_tstep = a_cell.system_history[(should_be_polarized_by_tstep - 2):(should_be_polarized_by_tstep + 2), :, parameterorg.rac_membrane_active_index]
        rho_membrane_active_around_SBPB_tstep = a_cell.system_history[(should_be_polarized_by_tstep - 2):(should_be_polarized_by_tstep + 2), :, parameterorg.rho_membrane_active_index]
        polarity_score_at_SBPBP_tstep = calculate_rgtpase_polarity_score_from_rgtpase_data(rac_membrane_active_around_SBPB_tstep, rho_membrane_active_around_SBPB_tstep)[0]
    
    velocity_score = 0.0

    node_coords_per_tstep = a_cell.system_history[:, :, [parameterorg.x_index, parameterorg.y_index]]
    centroids_per_tstep = calculate_centroids_per_tstep(node_coords_per_tstep)
    
    cell_centroids = centroids_per_tstep*a_cell.L/1e-6
    num_tsteps = cell_centroids.shape[0]
    
    rougher_cell_centroids = cell_centroids[::30]
    
    movement_steps = rougher_cell_centroids[1:] - rougher_cell_centroids[:-1]
    movement_step_mags = np.linalg.norm(movement_steps, axis=1)
    
    total_distance_travelled = np.sum(movement_step_mags)
        
    if total_distance_travelled < 10.0:
        velocity_score = 0.0
    else:
        distance_per_tstep = np.linalg.norm(cell_centroids[1:] - cell_centroids[:num_tsteps-1], axis=1)
        
        velocities = distance_per_tstep*(60.0/a_cell.T)
        average_velocity = np.average(velocities)
        
        if velocity_score > 3.5:
            velocity_score = 1.0 - score_function(3.5, 5.0, velocity_score)
        else:
            velocity_score = score_function(0.0, 2.5, average_velocity)
    
    return polarity_score_global, polarity_score_at_SBPBP_tstep, velocity_score
    
def calculate_rgtpase_polarity_score(cell_index, storefile_path, significant_difference=0.1, max_tstep=None, weigh_by_timepoint=False):
    rac_membrane_active_per_tstep = hardio.get_data_until_timestep(cell_index, max_tstep, "rac_membrane_active", storefile_path)
    rho_membrane_active_per_tstep = hardio.get_data_until_timestep(cell_index, max_tstep, "rho_membrane_active", storefile_path)
    
    return calculate_rgtpase_polarity_score_from_rgtpase_data(rac_membrane_active_per_tstep, rho_membrane_active_per_tstep, significant_difference=significant_difference, weigh_by_timepoint=weigh_by_timepoint)
        
    
# ==============================================================================

def calculate_average_rgtpase_activity(cell_index, storefile_path):
    rac_data = hardio.get_data(cell_index, None, 'rac_membrane_active', storefile_path)
    sum_rac_over_nodes = np.sum(rac_data, axis=1)
    avg_sum_rac_over_nodes = np.average(sum_rac_over_nodes)
    
    rho_data = hardio.get_data(cell_index, None, 'rho_membrane_active', storefile_path)
    sum_rho_over_nodes = np.sum(rho_data, axis=1)
    avg_sum_rho_over_nodes = np.average(sum_rho_over_nodes)
    
    return avg_sum_rac_over_nodes, avg_sum_rho_over_nodes

# ==============================================================================
    
def calculate_total_displacement(num_nodes, cell_index, storefile_path):
    init_node_coords = np.transpose(hardio.get_node_coords(cell_index, 0, storefile_path))
    final_node_coords = np.transpose(hardio.get_node_coords(cell_index, -1, storefile_path))
    
    init_centroid = geometry.calculate_centroid(num_nodes, init_node_coords)
    final_centroid = geometry.calculate_centroid(num_nodes, final_node_coords)
    
    return np.linalg.norm(init_centroid - final_centroid)

# ==============================================================================
   
def calculate_sum_displacement_per_interval(num_nodes, num_timesteps, cell_index, num_timesteps_to_pick, storefile_path):
    timepoints_of_interest = np.linspace(0, num_timesteps, num=num_timesteps_to_pick, dtype=np.int64)
    
    ncs_of_interest = [np.transpose(hardio.get_node_coords(cell_index, x, storefile_path))  for x in timepoints_of_interest]
    centroids_of_interest = np.array([geometry.calculate_centroid(num_nodes, x) for x in ncs_of_interest])

    return np.sum([np.linalg.norm(x) for x in centroids_of_interest[1:] - centroids_of_interest[:-1]])

# ==============================================================================
    
def calculate_migry_boundary_violation_score(num_nodes, num_timesteps, cell_index, storefile_path):
    migr_bdry_contact_data = hardio.get_data(cell_index, None, 'migr_bdry_contact', storefile_path)
    
    x = migr_bdry_contact_data - 1.0
    y = x > 1e-10
    y = np.array(y, dtype=np.int64)
    
    return np.sum(y)/(num_timesteps*num_nodes)

# ==============================================================================
   
def calculate_average_total_strain(num_nodes, cell_index, storefile_path):
    node_coords_per_timestep = hardio.get_node_coords_for_all_tsteps(cell_index, storefile_path)
    
    init_perimeter = geometry.calculate_perimeter(num_nodes, node_coords_per_timestep[0])
    total_strains = np.array([geometry.calculate_perimeter(num_nodes, x) for x in node_coords_per_timestep])/init_perimeter
    avg_total_strain = np.average(total_strains)
    
    return avg_total_strain - 1.0

# ==============================================================================

def calculate_acceleration(num_timepoints, num_nodes, L, T, cell_index, storefile_path):    
    init_point = 0
    mid_point = np.int(num_timepoints*0.5)
    final_point = -1
    
    init_node_coords = np.transpose(hardio.get_node_coords(cell_index, init_point, storefile_path))
    mid_node_coords = np.transpose(hardio.get_node_coords(cell_index, mid_point, storefile_path))
    final_node_coords = np.transpose(hardio.get_node_coords(cell_index, final_point, storefile_path))
    
    init_centroid = geometry.calculate_centroid(num_nodes, init_node_coords)
    mid_centroid = geometry.calculate_centroid(num_nodes, mid_node_coords)
    final_centroid = geometry.calculate_centroid(num_nodes, final_node_coords)
    
    acceleration = (np.linalg.norm((final_centroid - mid_centroid))*L*1e6 - np.linalg.norm((mid_centroid - init_centroid))*L*1e6)/(0.5*num_timepoints*T/60.0)**2
    
    return np.abs(acceleration)
    
        
# ==============================================================================
        
def score_distance_travelled(cell_index, storefile_path):
    xs = hardio.get_data(cell_index, None, 'x', storefile_path)
    ys = hardio.get_data(cell_index, None, 'y', storefile_path)
    
    x_disps = xs[1:] - xs[:-1]
    y_disps = ys[1:] - ys[:-1]
    
    dists = np.sqrt(x_disps*x_disps + y_disps*y_disps)
    
    total_dist_magnitude = np.sum(dists)
    
    sum_x_disps = np.sum(x_disps)
    sum_y_disps = np.sum(y_disps)
    
    total_disp_magnitude = np.sqrt(sum_x_disps*sum_x_disps + sum_y_disps*sum_y_disps)
    
    return total_dist_magnitude, total_disp_magnitude

# ==============================================================================

def get_event_tsteps(event_type, cell_index, storefile_path):
    relevant_data_per_tstep = None
    
    if event_type == "ic-contact":
        ic_mags = hardio.get_data(cell_index, None, "intercellular_contact_factor_magnitudes", storefile_path)
        relevant_data_per_tstep = np.any(ic_mags > 1, axis=1)
    elif event_type == "randomization":
        polarity_loss_occurred = hardio.get_data(cell_index, None, "polarity_loss_occurred", storefile_path)
        relevant_data_per_tstep =  np.any(polarity_loss_occurred, axis=1)
        
    if relevant_data_per_tstep == None:
        raise StandardError("Unknown event type given!")
        
    event_tsteps = [n for n in xrange(relevant_data_per_tstep.shape[0]) if relevant_data_per_tstep[n] == 1]
    
    return event_tsteps

# ==============================================================================

def determine_contact_start_end(T, contact_tsteps, min_tstep, max_tstep):
    contact_start_end_tuples = []

    current_start = None
    last_contact_tstep = None
    
    absolute_last_contact_tstep = contact_tsteps[-1]
    
    for contact_tstep in contact_tsteps:
        if current_start == None and contact_tstep != min_tstep and contact_tstep != max_tstep:
            current_start = contact_tstep
        
        if contact_tstep == absolute_last_contact_tstep:
            contact_start_end_tuples.append((current_start, last_contact_tstep))
            continue
                
        if last_contact_tstep != None:
            if contact_tstep - 1 != last_contact_tstep:
                if last_contact_tstep != max_tstep and last_contact_tstep != min_tstep:
                    contact_start_end_tuples.append((current_start, last_contact_tstep))
                current_start = contact_tstep
                
        last_contact_tstep = contact_tstep
        
    contact_start_end_tuples = [x for x in contact_start_end_tuples if (T*(x[1] - x[0]) > 30.0) and (x[1] != max_tstep)]
        
    return contact_start_end_tuples
    
# ==============================================================================
    
def calculate_kinematics(delta_t, centroid_pos_plus1s, centroid_pos_minus1s):
    delta_pos = centroid_pos_plus1s - centroid_pos_minus1s
    velocities = delta_pos/(2*delta_t)
    accelerations = delta_pos/(delta_t**2)
    
    return velocities, accelerations
    
# ==============================================================================

def are_all_elements_of_required_type(required_type, given_list):
    for element in given_list:
        if type(element) != required_type:
            return False
    
    return True
    
# ==============================================================================

def is_ascending(num_elements, test_list):
    for n in range(num_elements-1):
        if test_list[n] > test_list[n+1]:
            return False
    
    return True
    
# ==============================================================================
    
def determine_relevant_table_points(x, labels, type="row"):
    num_labels = len(labels)
    
    if not(is_ascending(num_labels, labels)):
        raise StandardError("Labels are not ascending!")
    
    lower_bound_index = None
    for n in range(num_labels - 1):
        if x > labels[n]:
            lower_bound_index = n
            break
    
    upper_bound_index = None
    if lower_bound_index != None:
        for n in range(lower_bound_index + 1, num_labels):
            if x < labels[n]:
                upper_bound_index = n
                break
    else:
        for n in range(num_labels):
            if x < labels[n]:
                upper_bound_index = n
                break
            
    if lower_bound_index == None and upper_bound_index != None:
        return upper_bound_index, lower_bound_index
    else:
        return lower_bound_index, upper_bound_index
            
# ==============================================================================

def determine_index_lb_ub_for_value_given_list(given_value, num_elements, given_list):
    ilb, iub = -1, -1
    
    for n, value in enumerate(given_list):
        if n == 0:
            if given_value < value:
                ilb, iub = n, n
            else:
                ilb, iub = n, n+1
            continue
        elif n == num_elements - 1:
            if given_value > value:
                ilb, iub = n, n
        else:
            if given_value == value:
                ilb, iub = n, n
            elif given_list[n - 1] < given_value < value:
                ilb, iub = n-1, n
                break
            
    if ilb == -1 or iub == -1 or ilb >= num_elements or iub >= num_elements:
        raise StandardError("Did not find one of ilb, iub!")
    else:
        return ilb, iub
            
# ==============================================================================
        
def determine_probability_given_N_Rstar(N, Rstar):
    data_dict = moore_data_table.moore_data_table_dict
    
    rlabels = moore_data_table.moore_row_labels
    num_rlabels = moore_data_table.num_row_labels
    
    clabels = moore_data_table.moore_col_labels
    num_clabels = moore_data_table.num_col_labels
    
    ylb, yub = determine_index_lb_ub_for_value_given_list(N, num_rlabels, rlabels)
    N_lb = rlabels[ylb]
    if ylb != ylb:
        N_ub = rlabels[yub]
    else:
        N_ub = N_lb
        
    rstars_lb = [data_dict[(N_lb, clabel)] for clabel in clabels]
    if yub != ylb:
        rstars_ub = [data_dict[(N_ub, clabel)] for clabel in clabels]
    else:
        rstars_ub = rstars_lb
        
    x_lb_lb, x_lb_ub = determine_index_lb_ub_for_value_given_list(Rstar, num_clabels, rstars_lb)
    if yub != ylb:
        x_ub_lb, x_ub_ub = determine_index_lb_ub_for_value_given_list(Rstar, num_clabels, rstars_ub)
    else:
        x_ub_lb, x_ub_ub = x_lb_lb, x_lb_ub
    
    if x_lb_lb != x_lb_ub:
        prob_lb = clabels[x_lb_lb] + ((clabels[x_lb_ub] - clabels[x_lb_lb])/(rstars_lb[x_lb_ub] - rstars_lb[x_lb_lb]))*(Rstar - rstars_lb[x_lb_lb])
    else:
        prob_lb = clabels[x_lb_lb]
        
    if x_ub_lb != x_ub_ub:
        prob_ub = clabels[x_ub_lb] + ((clabels[x_ub_ub] - clabels[x_ub_lb])/(rstars_ub[x_ub_ub] - rstars_ub[x_ub_lb]))*(Rstar - rstars_ub[x_ub_lb])
    else:
        prob_ub = clabels[x_ub_lb]
    
    if yub != ylb:
        prob = prob_lb + ((prob_ub - prob_lb)/(rlabels[yub] - rlabels[ylb]))*(N - rlabels[ylb])
    else:
        prob = prob_lb
        
    return prob
    
# ==============================================================================

def calculate_polar_velocities(velocities):
    polar_velocities = np.empty_like(velocities)
    
    num_velocities = velocities.shape[0]
    velocity_mags = geometry.calculate_2D_vector_mags(num_velocities, velocities)
    velocity_thetas = geometry.calculate_2D_vector_directions(num_velocities, velocities)
    
    polar_velocities[:,0] = velocity_mags
    polar_velocities[:,1] = velocity_thetas
        
    return polar_velocities

# ==============================================================================
 
def calculate_null_hypothesis_probability(velocities):
    N = velocities.shape[0]
    
    polar_velocities = calculate_polar_velocities(velocities)
    
    sorted_polar_velocities = np.array(sorted(polar_velocities, key=lambda x: x[0]))
    
    transformed_polar_velocities = np.array([[n + 1, pv[1]] for n, pv in enumerate(sorted_polar_velocities)])
    
    transformed_rs = transformed_polar_velocities[:, 0]
    thetas = transformed_polar_velocities[:, 1]
    
    X = np.dot(transformed_rs, np.cos(thetas))
    Y = np.dot(transformed_rs, np.sin(thetas))
    
    R = np.sqrt(X**2 + Y**2)
    Rstar = R/(N**(3./2.))
    
    return determine_probability_given_N_Rstar(N, Rstar)
    
# =================================================================================

def get_ic_contact_data(cell_index, storefile_path, max_tstep=None):
    if max_tstep == None:
        ic_contact_data = hardio.get_data(cell_index, None, "intercellular_contact_factor_magnitudes", storefile_path)
    else:
        ic_contact_data = hardio.get_data_until_timestep(cell_index, max_tstep, "intercellular_contact_factor_magnitudes", storefile_path)
    
    return np.array(np.any(ic_contact_data > 1, axis=1), dtype=np.int64)

# =================================================================================
 
def determine_contact_start_ends(ic_contact_data):
    num_ic_data = ic_contact_data.shape[0]
    contact_start_end_arrays = np.zeros((num_ic_data, 2), dtype=np.int64)
    num_contact_start_end_arrays = 0
    
    in_contact = False
    contact_start = -1
    
    for n in range(num_ic_data):
        if ic_contact_data[n] == 1:
            if in_contact == False:
                contact_start = n
                in_contact = True
            else:
                continue
        else:
            if in_contact == True:
                contact_start_end_arrays[num_contact_start_end_arrays][0] = contact_start
                contact_start_end_arrays[num_contact_start_end_arrays][1] = n
                num_contact_start_end_arrays += 1
                
                in_contact = False
            else:
                continue
            
    return contact_start_end_arrays[:num_contact_start_end_arrays]

# =================================================================================
   
def smoothen_contact_start_end_tuples(contact_start_end_arrays, min_tsteps_between_arrays=1):
    smoothened_contact_start_end_arrays = np.zeros_like(contact_start_end_arrays)
    num_start_end_arrays = contact_start_end_arrays.shape[0]
    
    num_smoothened_contact_start_end_arrays = 0
    for n in xrange(num_start_end_arrays-1):
        this_start, this_end = contact_start_end_arrays[n]
        next_start, next_end = contact_start_end_arrays[n+1]

        if (next_start - this_end) < min_tsteps_between_arrays:
            smoothened_contact_start_end_arrays[num_smoothened_contact_start_end_arrays][0] = this_start
            smoothened_contact_start_end_arrays[num_smoothened_contact_start_end_arrays][1] = next_end
            num_smoothened_contact_start_end_arrays += 1
        else:
            if n == 0:
                smoothened_contact_start_end_arrays[num_smoothened_contact_start_end_arrays][0] = this_start
                smoothened_contact_start_end_arrays[num_smoothened_contact_start_end_arrays][1] = this_end
                num_smoothened_contact_start_end_arrays += 1
            
            
            smoothened_contact_start_end_arrays[num_smoothened_contact_start_end_arrays][0] = next_start
            smoothened_contact_start_end_arrays[num_smoothened_contact_start_end_arrays][1] = next_end
            num_smoothened_contact_start_end_arrays += 1
    
    return smoothened_contact_start_end_arrays[:num_smoothened_contact_start_end_arrays]
    
# =================================================================================
    
def get_assessable_contact_start_end_tuples(smoothened_contact_start_end_arrays, data_max_tstep,  min_tsteps_needed_to_calculate_kinematics=2):
    
    
    num_smoothened_arrays = smoothened_contact_start_end_arrays.shape[0]
    
    if num_smoothened_arrays == 0:
        return np.zeros((0, 2), dtype=np.int64)
        
    assessable_contact_start_end_arrays = np.zeros_like(smoothened_contact_start_end_arrays)
    
    num_assessable_arrays = 0
    for n in range(num_smoothened_arrays):
        this_start, this_end = smoothened_contact_start_end_arrays[n]
        if n == 0:
            if this_start + 1 > min_tsteps_needed_to_calculate_kinematics:
                if n == num_smoothened_arrays - 1:
                    if (data_max_tstep - this_end) + 1 > min_tsteps_needed_to_calculate_kinematics:
                        assessable_contact_start_end_arrays[num_assessable_arrays][0] = this_start
                        assessable_contact_start_end_arrays[num_assessable_arrays][1] = this_end
                        num_assessable_arrays += 1
            continue
        else:
            last_start, last_end = smoothened_contact_start_end_arrays[n - 1]
            if n == num_smoothened_arrays - 1:
                if (data_max_tstep - this_end) + 1 > min_tsteps_needed_to_calculate_kinematics:
                    if (this_start - last_end) + 1 > min_tsteps_needed_to_calculate_kinematics:
                        assessable_contact_start_end_arrays[num_assessable_arrays][0] = this_start
                        assessable_contact_start_end_arrays[num_assessable_arrays][1] = this_end
                        num_assessable_arrays += 1
            else:
                next_start, next_end = smoothened_contact_start_end_arrays[n + 1]
                if (next_start - this_end) + 1 > min_tsteps_needed_to_calculate_kinematics:
                    if (this_start - last_end) + 1 > min_tsteps_needed_to_calculate_kinematics:
                        assessable_contact_start_end_arrays[num_assessable_arrays][0] = this_start
                        assessable_contact_start_end_arrays[num_assessable_arrays][1] = this_end
                        num_assessable_arrays += 1
                        
    return assessable_contact_start_end_arrays[:num_assessable_arrays]

# =================================================================================

def calculate_3_point_kinematics(last_centroid, next_centroid, delta_tsteps, tstep_length):
    delta_centroid = (next_centroid - last_centroid)
    
    delta_t = delta_tsteps*tstep_length/60.0
    
    acceleration = delta_centroid/(delta_t*delta_t)
    velocity = delta_centroid/(2*delta_t)
    
    return acceleration, velocity
    
# =================================================================================
    
def calculate_contact_pre_post_kinematics(assessable_contact_start_end_arrays, cell_centroids_per_tstep, delta_tsteps, tstep_length):
    num_start_end_tuples = assessable_contact_start_end_arrays.shape[0]
    
    pre_velocities = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    post_velocities = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    pre_accelerations = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    post_accelerations = np.zeros((num_start_end_tuples, 2), dtype=np.float64)
    
    for n in range(num_start_end_tuples):
        start_tstep, end_tstep = assessable_contact_start_end_arrays[n]
        
        delta_tsteps_doubled = int(delta_tsteps + delta_tsteps)
        pre_minus1_centroid = cell_centroids_per_tstep[start_tstep - delta_tsteps_doubled]
        pre_plus1_centroid = cell_centroids_per_tstep[start_tstep]
        
        pre_acceleration, pre_velocity = calculate_3_point_kinematics(pre_minus1_centroid, pre_plus1_centroid, delta_tsteps, tstep_length)
        pre_accelerations[n] = pre_acceleration
        pre_velocities[n] = pre_velocity
        
        post_plus1_centroid = cell_centroids_per_tstep[end_tstep + delta_tsteps_doubled]
        post_minus1_centroid = cell_centroids_per_tstep[end_tstep]
        
        post_acceleration, post_velocity = calculate_3_point_kinematics(post_minus1_centroid, post_plus1_centroid, delta_tsteps, tstep_length)
        
        post_accelerations[n] = post_acceleration
        post_velocities[n] = post_velocity
        
    return pre_velocities, post_velocities, pre_accelerations, post_accelerations
    
# =================================================================================
    
def rotate_contact_kinematics_data_st_pre_lies_along_given_and_post_maintains_angle_to_pre(pre_data, post_data, given_vector):
    
    num_elements = pre_data.shape[0]
    aligned_pre_data = np.zeros_like(pre_data)
    aligned_post_data = np.zeros_like(post_data)
    
    for n in range(num_elements):
        pre_datum = pre_data[n]
        post_datum = post_data[n]
        
        rot_mat = geometry.determine_rotation_matrix_to_rotate_vector1_to_lie_along_vector2(pre_datum, given_vector)
        
        aligned_pre_data[n] = np.dot(rot_mat, pre_datum)
        aligned_post_data[n] = np.dot(rot_mat, post_datum)
        
    return aligned_pre_data, aligned_post_data
    
# =================================================================================

# relevant_environment, storefile_path, si, rpt_number
def analyze_single_cell_motion(relevant_environment, storefile_path, subexperiment_index, rpt_number):
    # calculate centroid positions
    cell_centroids = calculate_cell_centroids_for_all_time(0, storefile_path)*relevant_environment.cells_in_environment[0].L/1e-6
    num_tsteps = cell_centroids.shape[0]
    
    net_displacement = cell_centroids[num_tsteps-1] - cell_centroids[0]
    net_displacement_mag = np.linalg.norm(net_displacement)

    distance_per_tstep = np.linalg.norm(cell_centroids[1:] - cell_centroids[:num_tsteps-1], axis=1)
    net_distance = np.sum(distance_per_tstep)
    
    persistence = net_displacement_mag/net_distance
    
    return (subexperiment_index, rpt_number, cell_centroids, persistence)

# ===========================================================================

def determine_run_and_tumble_periods(avg_strain_per_tstep, polarization_score_per_tstep, tumble_period_strain_threshold,  tumble_period_polarization_threshold):
    num_tsteps = polarization_score_per_tstep.shape[0]
    
    tumble_period_found = False
    associated_run_period_found = False
    
    tumble_info = -1*np.ones((int(num_tsteps/2), 3), dtype=np.int64)
    run_and_tumble_pair_index = 0
    
    for ti in range(num_tsteps):
        this_tstep_is_tumble = polarization_score_per_tstep[ti] <= tumble_period_polarization_threshold and avg_strain_per_tstep[ti] <= tumble_period_strain_threshold
        
        if tumble_period_found == False and this_tstep_is_tumble:
            tumble_period_found = True
            tumble_info[run_and_tumble_pair_index, 0] = ti
        else:
            if associated_run_period_found == False and (not this_tstep_is_tumble):
                associated_run_period_found = True
                tumble_info[run_and_tumble_pair_index, 1] = ti
            elif associated_run_period_found == True and this_tstep_is_tumble:
                tumble_info[run_and_tumble_pair_index, 2] = ti
                run_and_tumble_pair_index += 1
                tumble_info[run_and_tumble_pair_index, 0] = ti
                tumble_period_found = True
                associated_run_period_found = False

    num_run_and_tumble_pairs = run_and_tumble_pair_index + 1                
    for i in range(3):
        if tumble_info[run_and_tumble_pair_index, i] == -1:
            num_run_and_tumble_pairs -= 1
            break

    return_tumble_info = -1*np.ones((num_run_and_tumble_pairs, 3), dtype=np.int64)        
    for pi in range(num_run_and_tumble_pairs):
        tumble_info_tuple = tumble_info[pi]
        for i in range(3):
            return_tumble_info[pi, i] = tumble_info_tuple[i]
            
    return return_tumble_info

# ===========================================================================

def calculate_run_and_tumble_statistics(num_nodes, T, L, cell_index, storefile_path, cell_centroids = None, max_tstep=None, significant_difference=2.5e-2, tumble_period_strain_threshold=0.3, tumble_period_polarization_threshold=0.6):

    rac_membrane_active_per_tstep = hardio.get_data_until_timestep(cell_index, max_tstep, "rac_membrane_active", storefile_path)
    rho_membrane_active_per_tstep = hardio.get_data_until_timestep(cell_index, max_tstep, "rho_membrane_active", storefile_path)
    avg_strain_per_tstep = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, "local_strains", storefile_path), axis=1)
    polarization_score_per_tstep = np.array([calculate_polarization_rating(rac_membrane_active, rho_membrane_active, num_nodes, significant_difference=significant_difference) for rac_membrane_active, rho_membrane_active in zip(rac_membrane_active_per_tstep, rho_membrane_active_per_tstep)])
        
    tumble_periods_info = determine_run_and_tumble_periods(avg_strain_per_tstep, polarization_score_per_tstep, tumble_period_strain_threshold, tumble_period_polarization_threshold)
    
    tumble_periods = [(tpi[1] - tpi[0])*T for tpi in tumble_periods_info]
    run_periods = [(tpi[2] - tpi[1])*T for tpi in tumble_periods_info]
    
    if cell_centroids == None:
        cell_centroids = calculate_cell_centroids_for_all_time(cell_index, storefile_path)*L

    tumble_centroids = [cell_centroids[tpi[0]:tpi[1]] for tpi in tumble_periods_info]
    net_tumble_displacement_mags = [np.linalg.norm(tccs[-1] - tccs[0]) for tccs in tumble_centroids]
    mean_tumble_period_speeds = [np.average(np.linalg.norm((tccs[1:] - tccs[:-1])/T, axis=1)) for tccs in tumble_centroids]
    
    run_centroids = [cell_centroids[tpi[1]:tpi[2]] for tpi in tumble_periods_info]
    net_run_displacement_mags = [np.linalg.norm(rccs[-1] - rccs[0]) for rccs in run_centroids]
    mean_run_period_speeds = [np.average(np.linalg.norm((rccs[1:] - rccs[:-1])/T, axis=1)) for rccs in run_centroids]
    
    return (tumble_periods, run_periods, net_tumble_displacement_mags, mean_tumble_period_speeds, net_run_displacement_mags, mean_run_period_speeds)
    
def calculate_protrusion_lifetimes_from_existence_array(protrusions_existence):
    num_timesteps = protrusions_existence.shape[0]
    num_nodes = protrusions_existence.shape[1]
    
    num_protrusions_over_time_per_node = np.zeros(num_nodes, dtype=np.int64)
    protrusion_start_ends_per_node = np.zeros((num_nodes, num_timesteps, 2), dtype=np.int64)
    
    for ni in range(num_nodes):
        relevant_protrusions_existence_slice = protrusions_existence[:, ni]
        protrusion_index = 0
        protrusion_start_found = False
        protrusion_start_ends_for_this_node = np.zeros((num_timesteps, 2), dtype=np.int64)
        
        for ti in range(num_timesteps):
            if relevant_protrusions_existence_slice[ti] == 1:
                if protrusion_start_found == False:
                    protrusion_start_found = True
                    protrusion_start_ends_for_this_node[protrusion_index][0] = ti
                else:
                    continue
            else:
                if protrusion_start_found == False:
                    continue
                else:
                    protrusion_start_found = False
                    protrusion_start_ends_for_this_node[protrusion_index][1] = ti
                    protrusion_index += 1
                    
        protrusion_start_ends_per_node[ni] = protrusion_start_ends_for_this_node
        num_protrusions_over_time_per_node[ni] = protrusion_index

    return num_protrusions_over_time_per_node, protrusion_start_ends_per_node

def calculate_cell_protrusion_direction_lifetime(T, protrusions_existence, uivs_per_node_per_timestep, migration_axis):
    num_protrusions_over_time_per_node, protrusion_start_ends_per_node = calculate_protrusion_lifetimes_from_existence_array(protrusions_existence)
    
    num_timesteps = protrusions_existence.shape[0]
    num_nodes = protrusions_existence.shape[1]

    migration_axis_direction = geometry.calculate_2D_vector_direction(migration_axis)/(2*np.pi)
    protrusion_directions = np.zeros((num_nodes, num_timesteps, 3), dtype=np.float64)
    protrusion_lifetimes = np.zeros((num_nodes, num_timesteps), dtype=np.float64)
    
    for ni in range(num_nodes):
        num_protrusions_over_time = num_protrusions_over_time_per_node[ni]
        protrusion_start_ends = protrusion_start_ends_per_node[ni]
        this_node_uivs = uivs_per_node_per_timestep[:, ni]
        
        for pi in range(num_protrusions_over_time):
            start, end = protrusion_start_ends[pi]
            protrusion_dirn_vectors = -1*this_node_uivs[start:end]
            
            protrusion_direction_angles = np.round(geometry.calculate_2D_vector_directions(protrusion_dirn_vectors.shape[0], protrusion_dirn_vectors)/(2*np.pi), decimals=3)
            protrusion_directions_relative_to_migration_axis = np.mod(protrusion_direction_angles - migration_axis_direction, 1.0)
            protrusion_directions_relative_to_migration_axis = np.where(protrusion_directions_relative_to_migration_axis > 0.5, 0.5 - np.mod(protrusion_directions_relative_to_migration_axis, 0.5), protrusion_directions_relative_to_migration_axis)
            
            protrusion_directions[ni][pi][0] = np.min(protrusion_directions_relative_to_migration_axis)
            protrusion_directions[ni][pi][1] = np.max(protrusion_directions_relative_to_migration_axis)
            protrusion_directions[ni][pi][2] = np.average(protrusion_directions_relative_to_migration_axis)
            protrusion_lifetimes[ni][pi] = (end - start)*T
            
    return num_protrusions_over_time_per_node, protrusion_directions, protrusion_lifetimes
    
def calculate_cells_protrusion_direction_lifetime(num_cells, T, storefile_path, max_tstep=None, migration_axis=np.array([1, 0])):
    protrusion_directions_lifetimes_per_cell = []
    
    for ci in range(num_cells):
        rac_membrane_actives = hardio.get_data_until_timestep(ci, max_tstep, "rac_membrane_active", storefile_path)
        rho_membrane_actives = hardio.get_data_until_timestep(ci, max_tstep, "rho_membrane_active", storefile_path)
        
        protrusions_existence = np.where(rac_membrane_actives > rho_membrane_actives, 1, 0)
        
        uivs_per_node_per_timestep = hardio.get_vector_data_until_timestep(ci, max_tstep, "unit_in_vec", storefile_path)
        
        protrusion_directions_lifetimes = calculate_cell_protrusion_direction_lifetime(T, protrusions_existence, uivs_per_node_per_timestep, migration_axis)
        
        protrusion_directions_lifetimes_per_cell.append(protrusion_directions_lifetimes)
        
    return protrusion_directions_lifetimes_per_cell
    
def calculate_cells_protrusion_number_given_direction_per_timestep(num_cells, num_timepoints, num_nodes, storefile_path, max_tstep=None, migration_axis=np.array([1, 0])):
    
    if max_tstep == None:
        max_tstep = num_timepoints
        
    forward_cone = np.zeros(max_tstep, dtype=np.int64)
    backward_cone = np.zeros(max_tstep, dtype=np.int64)
    
    migration_axis_direction = geometry.calculate_2D_vector_direction(migration_axis)/(2*np.pi)
    
    for ci in range(num_cells):  
        rac_membrane_actives = hardio.get_data_until_timestep(ci, max_tstep, "rac_membrane_active", storefile_path)
        rho_membrane_actives = hardio.get_data_until_timestep(ci, max_tstep, "rho_membrane_active", storefile_path)
        protrusions_existence = np.where(rac_membrane_actives > rho_membrane_actives, 1, 0)
        
        uivs_per_node_per_timestep = hardio.get_vector_data_until_timestep(ci, max_tstep, "unit_in_vec", storefile_path)
        
        for ti in range(max_tstep):
            protrusions_exist_for_this_timepoint = protrusions_existence[ti]
            protrusion_dirn_vectors = -1*uivs_per_node_per_timestep[ti]
            
            protrusion_direction_angles = np.round(geometry.calculate_2D_vector_directions(protrusion_dirn_vectors.shape[0], protrusion_dirn_vectors)/(2*np.pi), decimals=3)
            protrusion_directions_relative_to_migration_axis = np.mod(protrusion_direction_angles - migration_axis_direction, 1.0)
            protrusion_directions_relative_to_migration_axis = np.where(protrusion_directions_relative_to_migration_axis > 0.5, 0.5 - np.mod(protrusion_directions_relative_to_migration_axis, 0.5), protrusion_directions_relative_to_migration_axis)
            
            for ni in range(num_nodes):
                if protrusions_exist_for_this_timepoint[ni] == 1:
                    protrusion_dirn = protrusion_directions_relative_to_migration_axis[ni]
                    
                    if protrusion_dirn < 0.25:
                        forward_cone[ti] += 1
                    else:
                        backward_cone[ti] += 1
        
    return forward_cone, backward_cone
    