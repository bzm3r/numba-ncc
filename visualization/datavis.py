# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:21:52 2015

@author: brian
"""

import numpy as np
import matplotlib.pyplot as plt
import core.utilities as cu
import os
import colors
import core.geometry as geometry
import core.hardio as hardio
from matplotlib import cm

# ====================================================================

def graph_group_area_over_time(num_cells, num_timepoints, T, storefile_path, save_dir=None, save_name=None):
    normalized_areas = cu.calculate_normalized_group_area_over_time(num_cells, num_timepoints, storefile_path)
    timepoints = np.arange(normalized_areas.shape[0])*T
    
    fig, ax = plt.subplots()
    
    ax.plot(timepoints, normalized_areas, label="normalized convex hull area")
    ax.set_ylabel("normalized convex hull area (dimensionless)")
    ax.set_xlabel("time (min.)")
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("alll")
        
# ====================================================================

def graph_avg_neighbour_distance_over_time():
    return 

# ====================================================================

def graph_group_centroid_drift(T, relative_group_centroid_per_tstep, save_dir, save_name):
    timepoints = np.arange(relative_group_centroid_per_tstep.shape[0])*T
    group_centroid_x_coords = relative_group_centroid_per_tstep[:,0]
    group_centroid_y_coords = relative_group_centroid_per_tstep[:,1]
    
    fig, ax = plt.subplots()
    
    ax.plot(timepoints, group_centroid_x_coords, label="x-coord", color='b')
    ax.plot(timepoints, group_centroid_y_coords, label="y-coord", color='g')
    ax.set_ylabel("group centroid position (micrometers)")
    ax.set_xlabel("time (min.)")
    
    # Put a legend to the right of the current axis
    ax.legend(loc='best')
    ax.grid(which=u'both')
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, "group_" + save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
    
# ====================================================================

def graph_centroid_related_data(num_cells, num_timepoints, T, cell_Ls, storefile_path, save_dir=None, save_name=None, max_tstep=None, make_group_centroid_drift_graph=True):    
    # assuming that num_timepoints, T is same for all cells
    if max_tstep == None:
        max_tstep = num_timepoints
        
    all_cell_centroids_per_tstep = np.zeros((max_tstep, num_cells, 2), dtype=np.float64)
    
    # ------------------------
    
    for ci in xrange(num_cells):
        cell_centroids_per_tstep = cu.calculate_cell_centroids_until_tstep(ci, max_tstep, storefile_path)*cell_Ls[ci]
        
        all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep
        
    # ------------------------
        
    group_centroid_per_tstep = np.array([geometry.calculate_cluster_centroid(cell_centroids) for cell_centroids in all_cell_centroids_per_tstep])
    
    init_group_centroid_per_tstep = group_centroid_per_tstep[0]
    relative_group_centroid_per_tstep = group_centroid_per_tstep - init_group_centroid_per_tstep
    relative_all_cell_centroids_per_tstep = all_cell_centroids_per_tstep - init_group_centroid_per_tstep

    # ------------------------
    
    fig, ax = plt.subplots()
    
    # ------------------------
    
    min_x_data_lim = np.min(relative_all_cell_centroids_per_tstep[:,:,0])
    max_x_data_lim = np.max(relative_all_cell_centroids_per_tstep[:,:,0])
    delta_x = np.abs(min_x_data_lim - max_x_data_lim)
    max_y_data_lim = np.max(np.abs(relative_all_cell_centroids_per_tstep[:,:,1]))
    ax.set_xlim(min_x_data_lim - 0.1*delta_x, max_x_data_lim + 0.1*delta_x)
    ax.set_ylim(-1.05*max_y_data_lim, 1.05*max_y_data_lim)
    ax.set_aspect('equal')
    
    
    group_net_displacement = relative_group_centroid_per_tstep[-1] - relative_group_centroid_per_tstep[0]
    group_net_displacement_mag = np.linalg.norm(group_net_displacement)
    group_net_distance = np.sum(np.linalg.norm(relative_group_centroid_per_tstep[1:] - relative_group_centroid_per_tstep[:-1], axis=-1))
    group_persistence = np.round(group_net_displacement_mag/group_net_distance, 3)
    
    ax.plot(relative_group_centroid_per_tstep[:,0], relative_group_centroid_per_tstep[:,1], '.', label="group centroid")
    
    cell_persistences = []
    for ci in xrange(num_cells):
        ccs = relative_all_cell_centroids_per_tstep[:,ci,:]
        net_displacement = ccs[-1] - ccs[0]
        net_displacement_mag = np.linalg.norm(net_displacement)
        net_distance = np.sum(np.linalg.norm(ccs[1:] - ccs[:-1], axis=-1))
        persistence = np.round(net_displacement_mag/net_distance)
        cell_persistences.append(persistence)
        
        ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list20[ci%20])
        #ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list20[ci%20], label='cell {}, pers.={}'.format(ci, persistence))
    average_cell_persistence = np.round(np.average(cell_persistences), decimals=3)
    std_cell_persistence = np.round(np.std(cell_persistences), decimals=3)
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")
    
    ax.set_title("cell and group centroid paths - group persistence = {}, avg. cell persistence = {} (std = {})".format(group_persistence, average_cell_persistence, std_cell_persistence))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')

    # ------------------------
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
        
    if make_group_centroid_drift_graph == True:
        graph_group_centroid_drift(T, relative_group_centroid_per_tstep, save_dir, save_name)
        
# ==============================================================================

def graph_cell_velocity_over_time(num_cells, T, cell_Ls, storefile_path, save_dir=None, save_name=None, max_tstep=None, time_to_average_over_in_minutes=1.0):
    fig, ax = plt.subplots()

    for ci in xrange(num_cells):
        L = cell_Ls[ci]
#        num_timesteps_to_average_over = int(60.0*time_to_average_over_in_minutes/T)
        timepoints, cell_speeds = cu.calculate_cell_speeds_until_tstep(ci, max_tstep, storefile_path, T, L)
        
#        chunky_timepoints = general.chunkify_numpy_array(timepoints, num_timesteps_to_average_over)
#        chunky_cell_speeds = general.chunkify_numpy_array(cell_speeds, num_timesteps_to_average_over)
#        
#        averaged_cell_speeds = np.average(chunky_cell_speeds, axis=1)
        
#        resized_timepoints = np.arange(num_timesteps_to_average_over*chunky_timepoints.shape[0])
#        corresponding_cell_speeds = np.repeat(averaged_cell_speeds, num_timesteps_to_average_over)
        
        ax.plot(timepoints[::30], cell_speeds[::30], color=colors.color_list20[ci%20], label="cell {}".format(ci))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    
    ax.set_xlabel("time ($min$)")
    ax.set_ylabel("speed ($\mu m/min$) -- avg. over {}min. intervals".format(time_to_average_over_in_minutes)) 
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
        
        
# ==============================================================================

def graph_important_cell_variables_over_time(T, cell_index, storefile_path, polarity_scores=None, save_dir=None, save_name=None, max_tstep=None):
    fig, ax = plt.subplots()
    
    #randomization_kicks = hardio.get_data_until_timestep(a_cell, max_tstep, 'randomization_event_occurred')
    #randomization_kicks = np.any(randomization_kicks, axis=1)
    
    # cell_index, max_tstep, data_label, storefile_path
    rac_mem_active = hardio.get_data_until_timestep(cell_index, max_tstep, 'rac_membrane_active', storefile_path)
    sum_rac_act_over_nodes = np.sum(rac_mem_active, axis=1)
    
    rac_mem_inactive = hardio.get_data_until_timestep(cell_index, max_tstep, 'rac_membrane_inactive', storefile_path)
    sum_rac_inact_over_nodes = np.sum(rac_mem_inactive, axis=1)
    
    rho_mem_active = hardio.get_data_until_timestep(cell_index, max_tstep, 'rho_membrane_active', storefile_path)
    sum_rho_act_over_nodes = np.sum(rho_mem_active, axis=1)
    
    rho_mem_inactive = hardio.get_data_until_timestep(cell_index, max_tstep, 'rho_membrane_inactive', storefile_path)
    sum_rho_inact_over_nodes = np.sum(rho_mem_inactive, axis=1)
    
    rac_cyt_gdi = hardio.get_data_until_timestep(cell_index, max_tstep, 'rac_cytosolic_gdi_bound', storefile_path)[:, 0]
    rho_cyt_gdi = hardio.get_data_until_timestep(cell_index, max_tstep, 'rho_cytosolic_gdi_bound', storefile_path)[:, 0]
    
    time_points = T*np.arange(rac_mem_active.shape[0])
    
    #for data_set, line_style, data_label in zip([randomization_kicks, sum_rac_act_over_nodes, sum_rho_act_over_nodes, sum_rac_inact_over_nodes, sum_rho_inact_over_nodes, rac_cyt_gdi, rho_cyt_gdi], ['k', 'b', 'r', 'b--', 'r--', 'c', 'm'], ['random kick', 'rac_active', 'rho_active', 'rac_inactive', 'rho_inactive', 'rac_gdi', 'rho_gdi'])
    
    for data_set, line_style, data_label in zip([sum_rac_act_over_nodes, sum_rho_act_over_nodes, sum_rac_inact_over_nodes, sum_rho_inact_over_nodes, rac_cyt_gdi, rho_cyt_gdi], ['b', 'r', 'b--', 'r--', 'c', 'm'], ['rac_active', 'rho_active', 'rac_inactive', 'rho_inactive', 'rac_gdi', 'rho_gdi']):
        ax.plot(time_points, data_set, line_style, label=data_label)
        
    if polarity_scores.shape[0] != 0:
        ax.plot(time_points, polarity_scores, '.', color='k', label='polarity_scores')
            
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("time (min)")
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
        
        
# ==============================================================================

def graph_strains(T, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None):
    fig, ax = plt.subplots()
    
    total_strains = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'local_strains', storefile_path), axis=1)
    time_points = T*np.arange(total_strains.shape[0])
    
    ax.plot(time_points, total_strains, 'k', label='avg_strains')
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    ax.set_xlabel("time (min.)")

    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'))
        plt.close(fig)
        plt.close("all")
        
# ==============================================================================
    
def graph_rates(T, kgtp_rac_baseline, kgtp_rho_baseline, kdgtp_rac_baseline, kdgtp_rho_baseline, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None):
    fig, ax = plt.subplots()
    
    average_kgtp_rac = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'kgtp_rac', storefile_path), axis=1)/kgtp_rac_baseline
    avg_average_kgtp_rac = np.average(average_kgtp_rac)
    average_kgtp_rho = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'kgtp_rho', storefile_path), axis=1)/kgtp_rho_baseline
    avg_average_kgtp_rho = np.average(average_kgtp_rho)
    average_kdgtp_rac = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'kdgtp_rac', storefile_path), axis=1)/kdgtp_rac_baseline
    avg_average_kdgtp_rac = np.average(average_kdgtp_rac)
    average_kdgtp_rho = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'kdgtp_rho', storefile_path), axis=1)/kdgtp_rho_baseline
    avg_average_kdgtp_rho = np.average(average_kdgtp_rho)
    average_coa_signal = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'coa_signal', storefile_path), axis=1) + 1.0
    
    time_points = T*np.arange(average_kgtp_rac.shape[0])/60.0
    
    for data_set, line_style, data_label in zip([average_kgtp_rac, average_kgtp_rho, average_kdgtp_rac, average_kdgtp_rho, average_coa_signal], ['b-.', 'r-.', 'c-.', 'm-.', 'b'], ['avg_kgtp_rac ({})'.format(avg_average_kgtp_rac), 'avg_kgtp_rho ({})'.format(avg_average_kgtp_rho), 'avg_kdgtp_rac ({})'.format(avg_average_kdgtp_rac), 'avg_kdgtp_rho ({})'.format(avg_average_kdgtp_rho), 'average_coa_signal']):
        ax.plot(time_points, data_set, line_style, label=data_label)
        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    ax.set_xlabel("time (min.)")

    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'))
        plt.close(fig)
        plt.close("all")
        
def graph_run_and_tumble_statistics(num_nodes, T, L, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None, significant_difference=0.2):
    tumble_periods, run_periods, net_tumble_displacement_mags, mean_tumble_period_speeds, net_run_displacement_mags, mean_run_period_speeds = cu.calculate_run_and_tumble_statistics(num_nodes, T, L, cell_index, storefile_path, significant_difference=significant_difference)
    
    num_run_and_tumble_periods = len(tumble_periods)
    
    mean_tumble_period = np.round(np.average(tumble_periods), decimals=1)
    std_tumble_period = np.round(np.std(tumble_periods), decimals=1)
    
    mean_run_period = np.round(np.average(run_periods), decimals=1)
    std_run_period = np.round(np.std(run_periods), decimals=1)
    
    N = 1
    indices = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects_tumble = ax.bar(indices, (mean_tumble_period,), width, color='r', yerr=(std_tumble_period,))
    rects_run = ax.bar(indices + width, (mean_run_period,), width, color='g', yerr=(std_run_period))

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Time (min.)')
    ax.set_title('Run & Tumble periods ({})'.format(num_run_and_tumble_periods))
    ax.set_xticks(indices + width)
    
    ax.legend((rects_tumble[0], rects_run[0]), ('Tumble', 'Run'))
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '_periods.png'))
        plt.close(fig)
        plt.close("all")
    
    mean_net_tumble_distance = np.round(np.average(net_tumble_displacement_mags), decimals=1)
    std_net_tumble_distance = np.round(np.std(net_tumble_displacement_mags), decimals=1)
    
    mean_net_run_distance = np.round(np.average(net_run_displacement_mags), decimals=1)
    std_net_run_distance = np.round(np.std(net_run_displacement_mags), decimals=1)
    
    N = 1
    indices = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects_tumble = ax.bar(indices, (mean_net_tumble_distance,), width, color='r', yerr=(std_net_tumble_distance,))
    rects_run = ax.bar(indices + width, (mean_net_run_distance,), width, color='g', yerr=(std_net_run_distance))

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Net distance (micrometers.)')
    ax.set_title('Run & Tumble periods ({}): net distance moved'.format(num_run_and_tumble_periods))
    ax.set_xticks(indices + width)
    
    ax.legend((rects_tumble[0], rects_run[0]), ('Tumble', 'Run'))
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '_dist.png'))
        plt.close(fig)
        plt.close("all")
        
    mean_tumble_speed = np.round(np.average(mean_tumble_period_speeds), decimals=1)
    std_mean_tumble_speed = np.round(np.std(mean_tumble_period_speeds), decimals=1)
    
    mean_run_speed = np.round(np.average(mean_run_period_speeds), decimals=1)
    std_mean_run_speed = np.round(np.std(mean_run_period_speeds), decimals=1)
    
    N = 1
    indices = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects_tumble = ax.bar(indices, (mean_tumble_speed,), width, color='r', yerr=(std_mean_tumble_speed,))
    rects_run = ax.bar(indices + width, (mean_run_speed,), width, color='g', yerr=(std_mean_run_speed))

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Speed (micrometers/min.)')
    ax.set_title('Run and Tumble periods ({}): mean speeds'.format(num_run_and_tumble_periods))
    ax.set_xticks(indices + width)
    
    ax.legend((rects_tumble[0], rects_run[0]), ('Tumble', 'Run'))
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '_speed.png'))
        plt.close(fig)
        plt.close("all")


# ==============================================================================

def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
        
# ==============================================================================
    
def graph_data_label_over_time(ax, a_cell, data_label):            
    data = hardio.get_data(a_cell, None, data_label)
    sum_data_over_nodes = np.sum(data, axis=1)
    
    ax.plot(sum_data_over_nodes, label=data_label)
    
    return ax
    
# ==============================================================================
    
def graph_data_label_average_node_over_time(ax, a_cell, data_label):            
    data = hardio.get_data(a_cell, None, data_label)
    average_data_over_nodes = np.average(data, axis=1)
    
    ax.plot(average_data_over_nodes, label=data_label)
    
    return ax
    
# ==============================================================================
    
def graph_data_labels_over_time(a_cell, data_labels, save_dir=None, save_name=None):
    fig, ax = plt.subplots()
    
    for data_label in data_labels:
        ax = graph_data_label_over_time(ax, a_cell, data_label)
        
    ax.legend(loc="best")
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.savefig(os.path.join(save_dir, save_name + '.png'))
        plt.close(fig)
        plt.close("all")
        
# ==============================================================================
    
def graph_data_labels_average_node_over_time(a_cell, data_labels, save_dir=None, save_name=None):
    fig, ax = plt.subplots()
    
    for data_label in data_labels:
        ax = graph_data_label_average_node_over_time(ax, a_cell, data_label)
        
    ax.legend(loc="best")
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")

# ==============================================================================

def graph_pre_post_contact_cell_kinematics(T, L, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None, timeperiod_in_seconds_over_which_to_calculate_kinematics=2.0):
    
    delta_tsteps = np.ceil(timeperiod_in_seconds_over_which_to_calculate_kinematics/T)
    min_tsteps_needed_to_calculate_kinematics = 2*delta_tsteps
    
    cell_centroids_per_tstep = cu.calculate_cell_centroids_until_tstep(cell_index, max_tstep, storefile_path)*L
    
    data_max_tstep = cell_centroids_per_tstep.shape[0] - 1
    
    ic_contact_data = cu.get_ic_contact_data(cell_index, storefile_path, max_tstep=max_tstep)
    
    contact_start_end_arrays = cu.determine_contact_start_ends(ic_contact_data)
    #print "contact_start_end_arrays: ", contact_start_end_arrays
    
    smoothened_contact_start_end_arrays = cu.smoothen_contact_start_end_tuples(contact_start_end_arrays, min_tsteps_between_arrays=1)
    #print "smoothened_contact_start_end_arrays: ", smoothened_contact_start_end_arrays
    
    assessable_contact_start_end_arrays = cu.get_assessable_contact_start_end_tuples(smoothened_contact_start_end_arrays, data_max_tstep, min_tsteps_needed_to_calculate_kinematics=min_tsteps_needed_to_calculate_kinematics)
    #print "assessable_contact_start_end_arrays: ", assessable_contact_start_end_arrays
    
    pre_velocities, post_velocities, pre_accelerations, post_accelerations = cu.calculate_contact_pre_post_kinematics(assessable_contact_start_end_arrays, cell_centroids_per_tstep, delta_tsteps, T)
    
    aligned_pre_velocities, aligned_post_velocities = cu.rotate_contact_kinematics_data_st_pre_lies_along_given_and_post_maintains_angle_to_pre(pre_velocities, post_velocities, np.array([1, 0]))
    
    aligned_pre_accelerations, aligned_post_accelerations = cu.rotate_contact_kinematics_data_st_pre_lies_along_given_and_post_maintains_angle_to_pre(pre_accelerations, post_accelerations, np.array([1, 0]))
    
    null_h_prob_velocities = 0
    null_h_prob_accelerations = 0
    max_data_lim_ax0 = 1
    max_data_lim_ax1 = 1
        
    if assessable_contact_start_end_arrays.shape[0] != 0:
        null_h_prob_velocities = np.round(cu.calculate_null_hypothesis_probability(aligned_post_velocities), decimals=3)
        
        null_h_prob_accelerations = np.round(cu.calculate_null_hypothesis_probability(aligned_post_accelerations), decimals=3)
    
        max_data_lim_ax0 = np.max([np.max(np.abs(aligned_pre_velocities)), np.max(np.abs(aligned_post_velocities))])
        max_data_lim_ax1 = np.max([np.max(np.abs(aligned_pre_accelerations)), np.max(np.abs(aligned_post_accelerations))])
    
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    
    ax0.set_title("Velocities pre-/post-contact (H0_prob = {})".format(null_h_prob_velocities))
    ax0.set_ylabel("micrometer/min.")
    ax0.set_xlabel("micrometer/min.")
    
    ax1.set_title("Accelerations pre-/post-contact (H0_prob = {})".format(null_h_prob_accelerations))
    ax1.set_xlabel("micrometer/min.^2")
    ax1.set_ylabel("micrometer/min.^2")
    
    ax0.set_xlim(-1.2*max_data_lim_ax0, 1.2*max_data_lim_ax0)
    ax0.set_ylim(-1.2*max_data_lim_ax0, 1.2*max_data_lim_ax0)
    
    ax1.set_xlim(-1.2*max_data_lim_ax1, 1.2*max_data_lim_ax1)
    ax1.set_ylim(-1.2*max_data_lim_ax1, 1.2*max_data_lim_ax1)
    
    ax0.plot([-2*max_data_lim_ax0, 2*max_data_lim_ax0], [0, 0], color='#808080', linewidth=1.5, alpha=0.5)
    ax0.plot([0, 0], [-2*max_data_lim_ax0, 2*max_data_lim_ax0], color='#808080', linewidth=1.5, alpha=0.5)
    
    ax1.plot([-2*max_data_lim_ax1, 2*max_data_lim_ax1], [0, 0], color='#808080', linewidth=1.5, alpha=0.5)
    ax1.plot([0, 0], [-2*max_data_lim_ax1, 2*max_data_lim_ax1], color='#808080', linewidth=1.5, alpha=0.5)
    
    ax0.set_aspect(u'equal')
    ax1.set_aspect(u'equal')
    
    ax0.grid(which=u'both')
    ax0.minorticks_on()
    
    ax1.grid(which=u'both')
    ax1.minorticks_on()
    
    for n in xrange(assessable_contact_start_end_arrays.shape[0]):
        start_time, end_time = assessable_contact_start_end_arrays[n]*(T/60.0)
        pre_velocity = aligned_pre_velocities[n]
        pre_acceleration = aligned_pre_accelerations[n]
        post_velocity = aligned_post_velocities[n]
        post_acceleration = aligned_post_accelerations[n]
        color = colors.color_list300[n%300]
        
        ax0.plot([0, pre_velocity[0]], [0, pre_velocity[1]], 'k', linewidth=2)
        ax0.plot([pre_velocity[0]], [pre_velocity[1]], color=color, markeredgecolor='k', markerfacecolor=color, ls='', marker="^", markersize=7.5, label="pre-c, {}min.".format(np.round(start_time, decimals=1)))
        
        ax0.plot([0, post_velocity[0]], [0, post_velocity[1]], 'k', linewidth=2)
        ax0.plot([post_velocity[0]], [post_velocity[1]], color=color, markeredgecolor='k', markerfacecolor=color, ls='', marker="o", markersize=7.5, label="post-c, {}min.".format(np.round(end_time, decimals=1)))
        
        ax1.plot([0, pre_acceleration[0]], [0, pre_acceleration[1]], 'k', linewidth=2)
        ax1.plot([pre_acceleration[0]], [pre_acceleration[1]], color=color, markeredgecolor='k', markerfacecolor=color, ls='', marker="^", markersize=7.5, label="pre-c, {}min.".format(np.round(start_time, decimals=1)))
        
        ax1.plot([0, post_acceleration[0]], [0, post_acceleration[1]], 'k', linewidth=2)
        ax1.plot([post_acceleration[0]], [post_acceleration[1]], color=color, markeredgecolor='k', markerfacecolor=color, ls='', marker="o", markersize=7.5, label="post-c, {}min.".format(np.round(end_time, decimals=1)))
    
    #ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        fig0.set_size_inches(8, 12)
        fig0.savefig(os.path.join(save_dir, save_name + "_vel.png"), forward=True)
        
        fig1.set_size_inches(8, 12)
        fig1.savefig(os.path.join(save_dir, save_name + "_acc.png"), forward=True)
        
        plt.close(fig0)
        plt.close(fig1)
        plt.close("all")
        
# ==========================================================================================
        
def present_collated_single_cell_motion_data(extracted_results, experiment_dir, time_in_hours):
    fig, ax = plt.subplots()
    
    max_data_lim = np.max([np.max(np.abs(x[2])) for x in extracted_results])
    
    persistences = [x[3] for x in extracted_results]
    mean_persistence = np.round(np.mean(persistences), 3)
    std_persistence = np.round(np.std(persistences), 3)
    
    for i, extracted_result in enumerate(extracted_results):
        si, rpt_number, ccs, persistence = extracted_result
        ccs = ccs - ccs[0]
        
        #label='({}, {}), ps.={}'.format(si, rpt_number, np.round(persistences[i], decimals=3))
        ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list300[i%300])

    ax.set_title("Persistence over {} hours (mean: {}, std: {})".format(time_in_hours, mean_persistence, std_persistence))
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")

    ax.set_xlim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_ylim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_aspect(u'equal')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    
    fig.set_size_inches(12, 8)
    save_path = os.path.join(experiment_dir, "collated_single_cell_data" + ".png")
    print "save_path: ", save_path
    fig.savefig(save_path, forward=True)
    plt.close(fig)
    plt.close("all")
    
# ==========================================================================================
        
def present_collated_cell_motion_data(extracted_results, experiment_dir, time_in_hours):
    fig, ax = plt.subplots()
    
    max_x_data_lim = 0.0
    min_x_data_lim = 0.0
    max_y_data_lim = 0.0
    min_y_data_lim = 0.0
    
    persistences = [x[1] for x in extracted_results]
    mean_persistence = np.round(np.mean(persistences), 3)
    std_persistence = np.round(np.std(persistences), 3)
    
    for i, extracted_result in enumerate(extracted_results):
        ccs, persistence = extracted_result
        normalized_ccs = ccs - ccs[0]
        this_max_x_data_lim = np.max(normalized_ccs[:,0])
        this_min_x_data_lim = np.min(normalized_ccs[:,0])
        this_max_y_data_lim = np.max(normalized_ccs[:,1])
        this_min_y_data_lim = np.min(normalized_ccs[:,1])
        
        if this_max_x_data_lim > max_x_data_lim:
            max_x_data_lim = this_max_x_data_lim
        if this_max_y_data_lim > max_y_data_lim:
            max_y_data_lim = this_max_y_data_lim
        if this_min_x_data_lim < min_x_data_lim:
            min_x_data_lim = this_min_x_data_lim
        if this_min_y_data_lim < min_y_data_lim:
            min_y_data_lim = this_min_y_data_lim
            
        ax.plot(normalized_ccs[:,0], normalized_ccs[:,1], marker=None, color=colors.color_list300[i%300])

    ax.set_title("Persistence over {} hours (mean: {}, std: {})".format(time_in_hours, mean_persistence, std_persistence))
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")

    y_lim = np.min([np.abs(min_y_data_lim), np.abs(max_y_data_lim)])
    
    ax.set_xlim(min_x_data_lim, 1.1*max_x_data_lim)
    ax.set_ylim(-1.1*y_lim, 1.1*y_lim)
    ax.set_aspect(u'equal')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    
    fig.set_size_inches(12, 8)
    save_path = os.path.join(experiment_dir, "collated_cell_data" + ".png")
    print "save_path: ", save_path
    fig.savefig(save_path, forward=True)
    plt.close(fig)
    plt.close("all")

#timestep_length, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_per_timestep_per_repeat, experiment_dir, total_time_in_hours, group_width=num_cells_width*cell_diameter
def present_collated_group_centroid_drift_data(T, min_x_centroid_per_tstep_per_repeat, max_x_centroid_per_tstep_per_repeat, group_centroid_per_tstep_per_repeat, save_dir, total_time_in_hours):
    timepoints = np.arange(group_centroid_per_tstep_per_repeat[0].shape[0])*T/60.0
    
    fig_simple, ax_simple = plt.subplots()
    fig_full, ax_full = plt.subplots()
    
    for repeat_number in range(len(group_centroid_per_tstep_per_repeat)):
        max_x_centroid_per_tstep = max_x_centroid_per_tstep_per_repeat[repeat_number]
        min_x_centroid_per_tstep = min_x_centroid_per_tstep_per_repeat[repeat_number]
        
        group_width = max_x_centroid_per_tstep[0][0] - min_x_centroid_per_tstep[0][0]
        
        group_centroid_per_tstep = group_centroid_per_tstep_per_repeat[repeat_number]

        relative_group_centroid_per_tstep = group_centroid_per_tstep - group_centroid_per_tstep[0]
        relative_max_centroid_per_tstep = max_x_centroid_per_tstep - group_centroid_per_tstep[0]
        relative_min_centroid_per_tstep = min_x_centroid_per_tstep - group_centroid_per_tstep[0]
        
        
        normalized_relative_group_centroid_x_coords = relative_group_centroid_per_tstep[:,0]/group_width
        normalized_relative_max_centroid_x_coords = relative_max_centroid_per_tstep[:,0]/group_width
        normalized_relative_min_centroid_x_coords = relative_min_centroid_per_tstep[:,0]/group_width
        
        ax_simple.plot(timepoints, normalized_relative_group_centroid_x_coords, color=colors.color_list300[repeat_number%300])
        ax_full.plot(timepoints, normalized_relative_group_centroid_x_coords, color=colors.color_list300[repeat_number%300])
        ax_full.plot(timepoints, normalized_relative_max_centroid_x_coords, color=colors.color_list300[repeat_number%300], alpha=0.2)
        ax_full.plot(timepoints, normalized_relative_min_centroid_x_coords, color=colors.color_list300[repeat_number%300], alpha=0.2)
        
    ax_simple.set_ylabel("group centroid position (normalized by initial group width)")
    ax_simple.set_xlabel("time (min.)")
    ax_full.set_ylabel("position (normalized by initial group width)")
    ax_full.set_xlabel("time (min.)")
    
    ax_simple.grid(which=u'both')
    ax_full.grid(which=u'both')
    
    if save_dir == None:
        plt.show()
    else:
        fig_simple.set_size_inches(12, 8)
        fig_simple.savefig(os.path.join(save_dir, "collated_group_centroid_drift_simple.png"), forward=True)
        plt.close(fig_simple)
        
        fig_full.set_size_inches(12, 8)
        fig_full.savefig(os.path.join(save_dir, "collated_group_centroid_drift_full.png"), forward=True)
        plt.close(fig_full)
        
        plt.close("all")
            
# ============================================================================

def generate_theta_bins(num_bins):
    delta = 2*np.pi/num_bins
    return np.array([[((2*n + 1)*delta)%(2*np.pi), (2*n + 3)*delta%(2*np.pi)] for n in range(num_bins)]), np.array([((2*n + 1)*delta)%(2*np.pi) for n in range(num_bins)]), delta

def graph_protrusion_lifetimes_radially(protrusion_lifetime_and_direction_data, num_polar_graph_bins, save_dir=None, mins_or_secs="mins"):
    bins, bin_midpoints, delta = generate_theta_bins(num_polar_graph_bins)

    binned_direction_data = [list() for x in range(num_polar_graph_bins)]    
    for cell_result in protrusion_lifetime_and_direction_data:
        for protrusion_result in cell_result:
            lifetime, direction = protrusion_result
            
            if mins_or_secs == "mins":
                lifetime = lifetime/60.0
            
            binned = False
            for n in range(num_polar_graph_bins - 1):
                a, b = bins[n]
                if a <= direction < b:
                    binned = True
                    binned_direction_data[n].append(lifetime)
                    break
                
            if binned == False:
                binned_direction_data[-1].append(lifetime)
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    
    average_lifetimes = [np.average(x) for x in binned_direction_data]
    ax.bar(bin_midpoints, average_lifetimes, width=delta, bottom=0.0)
    ax.set_ylabel('min.', labelpad=-100, rotation=0)
    ax.set_title('Average protrusion lifetime given direction')
        
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(8, 8)
        save_path = os.path.join(save_dir, "protrusion_lifetime_versus_direction" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        
def graph_protrusion_start_end_causes_radially(protrusion_lifetime_and_direction_data, protrusion_start_end_cause_data, num_polar_graph_bins, save_dir=None):
    bins, bin_midpoints, delta = generate_theta_bins(num_polar_graph_bins)

    start_cause_labels = ["coa", "randomization", "coa+rand"]
    end_cause_labels = ["cil", "other"]
    
    binned_start_cause_data = [np.zeros(3, dtype=np.int64) for x in range(num_polar_graph_bins)]
    binned_end_cause_data = [np.zeros(2, dtype=np.int64) for x in range(num_polar_graph_bins)] 
    for cell_protrusion_data in zip(protrusion_lifetime_and_direction_data, protrusion_start_end_cause_data):
        for protrusion_lifetime_direction_result, protrusion_start_end_cause in zip(cell_protrusion_data[0], cell_protrusion_data[1]):
            _, direction = protrusion_lifetime_direction_result
            start_causes, end_causes = protrusion_start_end_cause
            
            bin_index = -1
            for n in range(num_polar_graph_bins - 1):
                a, b = bins[n]
                if a <= direction < b:
                    bin_index = n
                    break
                
            if bin_index == -1:
                bin_index = num_polar_graph_bins - 1
            
            if "coa" in start_causes and "cil" in start_causes:
                binned_start_cause_data[bin_index][2] += 1
            elif "coa" in start_causes:
                binned_start_cause_data[bin_index][0] += 1
            elif "rand" in start_causes:
                binned_start_cause_data[bin_index][1] += 1
            
                
            if "cil" in end_causes:
                binned_end_cause_data[bin_index][0] += 1
            else:
                binned_end_cause_data[bin_index][1] += 1
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    
    #ax.set_title('start causes given direction')
    for n, l in enumerate(start_cause_labels):
        #ax.bar(bin_midpoints, [x[n] for x in binned_start_cause_data], width=delta, bottom=0.0, color=colors.color_list20[n%20], label=l, alpha=0.5)
        #ax.plot(bin_midpoints, , label=l, ls='', marker=styles[n%3], markerfacecolor=colors.color_list20[n%20], color=colors.color_list20[n%20], markersize=30)
        thetas = np.zeros(0, dtype=np.float64)
        rs = np.zeros(0, dtype=np.float64)
        for bi in range(num_polar_graph_bins):
            a, b = bins[bi][0], bins[bi][1]
            if a > b:
                b = b + 2*np.pi
                if a > b:
                    raise StandardError("a is still greater than b!")
            thetas = np.append(thetas, np.linspace(a, b))
            rs = np.append(rs, 50*[binned_start_cause_data[bi][n]])
            
        thetas = np.append(thetas, [thetas[-1] + 1e-6])
        rs = np.append(rs, [rs[0]])
            
        ax.plot(thetas, rs, label=l, color=colors.color_list20[n%20], ls='', marker='.')
        
    ax.legend(loc='best')
        
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(8, 8)
        save_path = os.path.join(save_dir, "start_causes_given_direction" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    
    #ax.set_title('end causes given direction')
    for n, l in enumerate(end_cause_labels):
        thetas = np.zeros(0, dtype=np.float64)
        rs = np.zeros(0, dtype=np.float64)
        for bi in range(num_polar_graph_bins):
            a, b = bins[bi][0], bins[bi][1]
            if a > b:
                b = b + 2*np.pi
                if a > b:
                    raise StandardError("a is still greater than b!")
            thetas = np.append(thetas, np.linspace(a, b))
            rs = np.append(rs, 50*[binned_end_cause_data[bi][n]])
            
        thetas = np.append(thetas, [thetas[-1] + 1e-6])
        rs = np.append(rs, [rs[0]])    
        
        ax.plot(thetas, rs, label=l, color=colors.color_list20[n%20], ls='', marker='.')
        
        
    ax.legend(loc='best')
        
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(8, 8)
        save_path = os.path.join(save_dir, "end_causes_given_direction" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        
# ============================================================================

def graph_forward_backward_protrusions_per_timestep(max_tstep, protrusion_node_index_and_tpoint_start_ends, protrusion_lifetime_and_direction_data, T, forward_cones, backward_cones, save_dir=None):
    times = np.arange(max_tstep)*T/60.0
    num_forward_protrusions = np.zeros(max_tstep, dtype=np.int64)
    num_backward_protrusions = np.zeros(max_tstep, dtype=np.int64)
    
    for cell_protrusion_data in zip(protrusion_node_index_and_tpoint_start_ends, protrusion_lifetime_and_direction_data):
        for protrusion_start_end_info, protrusion_lifetime_direction_info in zip(cell_protrusion_data[0], cell_protrusion_data[1]):
            ni, ti_start, ti_end = protrusion_start_end_info
            _, direction = protrusion_lifetime_direction_info
            
            direction_bin = None
            for lims in forward_cones:
                if lims[0] <= direction < lims[1]:
                    direction_bin = "f"
                    break
            if direction_bin != "f":
                for lims in backward_cones:
                    if lims[0] <= direction < lims[1]:
                        direction_bin = "b"
                        break
                        
            if direction_bin == "f":
                num_forward_protrusions[ti_start:ti_end] += 1
            elif direction_bin == "b":
                num_backward_protrusions[ti_start:ti_end] += 1
                
    fig, ax = plt.subplots()
    
    ax.plot(times, num_forward_protrusions, label='forward')
    ax.plot(times, num_backward_protrusions, label='backward')
    #ax.plot(times, other_cone, label='other')
    
    ax.legend(loc='best')
    
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "num_forward_backward_protrusions_over_time" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
    
# ============================================================================

def graph_forward_backward_cells_per_timestep(max_tstep, all_cell_speeds_and_directions, T, forward_cones, backward_cones, save_dir=None):
    times = np.arange(max_tstep)*T/60.0
    num_forward_cells = np.zeros(max_tstep, dtype=np.int64)
    num_backward_cells = np.zeros(max_tstep, dtype=np.int64)
    
    for cell_speed_direction_data in all_cell_speeds_and_directions:
        speeds, directions = cell_speed_direction_data
        for ti in range(max_tstep):
            if speeds[ti] > 0.5: #speed has to be greater than 0.5 micrometers per minute
                direction = directions[ti]
                direction_bin = None
                for lims in forward_cones:
                    if lims[0] <= direction < lims[1]:
                        direction_bin = "f"
                        break
                if direction_bin != "f":
                    for lims in backward_cones:
                        if lims[0] <= direction < lims[1]:
                            direction_bin = "b"
                            break
                                
                if direction_bin == "f":
                    num_forward_cells[ti] += 1
                elif direction_bin == "b":
                    num_backward_cells[ti] += 1
                
    fig, ax = plt.subplots()
    
    ax.plot(times, num_forward_cells, label='forward')
    ax.plot(times, num_backward_cells, label='backward')
    #ax.plot(times, other_cone, label='other')
    
    ax.legend(loc='best')
    
    ax.set_ylabel("number of cells")
    ax.set_xlabel("time (min.)")
    
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "num_forward_backward_cells_over_time" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
            
        
# =============================================================================

def graph_coa_variation_test_data(sub_experiment_number, num_cells_to_test, test_coas, average_cell_group_area_data, save_dir=None, max_normalized_group_area=3.0):
    
    fig, ax = plt.subplots()
    
    cax = ax.imshow(average_cell_group_area_data, interpolation='none', cmap=plt.get_cmap('viridis_r'))
    ax.set_yticks(np.arange(len(num_cells_to_test)))
    ax.set_xticks(np.arange(len(test_coas)))
    ax.set_yticklabels(num_cells_to_test)
    ax.set_xticklabels(test_coas)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, boundaries=np.linspace(1.0, max_normalized_group_area, num=100), ticks=np.linspace(1.0, max_normalized_group_area, num=5))
    cax.set_clim(1.0, max_normalized_group_area)

    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    

    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "coa_variation_results_{}".format(sub_experiment_number) + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        
        
def graph_confinement_data(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, save_dir=None):
    
    fig, ax = plt.subplots()
    
    bin_boundaries = np.linspace(0.5, 1.0, num=100)
    cax = ax.imshow(average_cell_persistence, interpolation='none', cmap=plt.get_cmap('viridis'))      
    cbar = fig.colorbar(cax, boundaries=bin_boundaries, ticks=np.linspace(0.5, 1.0, num=5))
    cax.set_clim(0.5, 1.0)
    ax.set_yticks(np.arange(len(test_num_cells)))
    ax.set_xticks(np.arange(len(test_heights)))
    ax.set_yticklabels(test_num_cells)
    ax.set_xticklabels(test_heights)
     
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "conefinement_test_graph_{}".format(sub_experiment_number) + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        

    
    

    
    
        
        