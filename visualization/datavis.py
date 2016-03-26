# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:21:52 2015

@author: brian
"""

import numpy as np
import matplotlib.pyplot as plt
import analysis.utilities as analysis_utils
import os
import colors
import scipy.spatial as space
import core.geometry as geometry
import core.hardio as hardio

# ==============================================================================

def graph_delaunay_triangulation_area_over_time(num_cells, num_timepoints, T, storefile_path, save_dir=None, save_name=None, max_tstep=None):
    # assuming that num_timepoints, T is same for all cells
    if max_tstep == None:
        max_tstep = num_timepoints
        
    all_cell_centroids_per_tstep = np.zeros((max_tstep, num_cells, 2), dtype=np.float64)
    
    # ------------------------
    
    for ci in xrange(num_cells):
        cell_centroids_per_tstep = analysis_utils.calculate_cell_centroids_until_tstep(ci, max_tstep, storefile_path)
        
        all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep
        
    # ------------------------
        
    delaunay_triangulations_per_tstep = [space.Delaunay(all_cell_centroids) for all_cell_centroids in all_cell_centroids_per_tstep]
    
    convex_hull_areas_per_tstep = []
    
    for dt, all_cell_centroids in zip(delaunay_triangulations_per_tstep, all_cell_centroids_per_tstep):
        simplices = all_cell_centroids[dt.simplices]
        simplex_areas = np.array([geometry.calculate_polygon_area(simplex.shape[0], simplex) for simplex in simplices])
        convex_hull_areas_per_tstep.append(np.round(np.sum(simplex_areas), decimals=3))
        
    convex_hull_areas_per_tstep = np.array(convex_hull_areas_per_tstep)
    init_area = convex_hull_areas_per_tstep[0]
    
    normalized_convex_hull_areas_per_tstep = convex_hull_areas_per_tstep/init_area
    timepoints = np.arange(normalized_convex_hull_areas_per_tstep.shape[0])*T/60.0
    
    fig, ax = plt.subplots()
    
    ax.plot(timepoints, normalized_convex_hull_areas_per_tstep, label="normalized convex hull area")
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
        
# ==============================================================================

def graph_avg_neighbour_distance_over_time():
    return 

# ==============================================================================

def graph_group_centroid_drift(T, relative_group_centroid_per_tstep, save_dir, save_name):
    timepoints = np.arange(relative_group_centroid_per_tstep.shape[0])*T/60.0
    group_centroid_x_coords = relative_group_centroid_per_tstep[:,0]
    
    fig, ax = plt.subplots()
    
    ax.plot(timepoints, group_centroid_x_coords, label="group centroid")
    ax.set_ylabel("x coordinate - micrometers")
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
        plt.close("alll")
    
        
# ==============================================================================

def graph_centroid_related_data(num_cells, num_timepoints, T, L, storefile_path, save_dir=None, save_name=None, max_tstep=None, make_group_centroid_drift_graph=True):    
    # assuming that num_timepoints, T is same for all cells
    if max_tstep == None:
        max_tstep = num_timepoints
        
    all_cell_centroids_per_tstep = np.zeros((max_tstep, num_cells, 2), dtype=np.float64)
    
    # ------------------------
    
    for ci in xrange(num_cells):
        cell_centroids_per_tstep = analysis_utils.calculate_cell_centroids_until_tstep(ci, max_tstep, storefile_path)
        
        all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep
        
    # ------------------------
    
    group_centroid_per_tstep = np.array([geometry.calculate_cluster_centroid(cell_centroids) for cell_centroids in all_cell_centroids_per_tstep])
    
    init_group_centroid_per_tstep = group_centroid_per_tstep[0]
    relative_group_centroid_per_tstep = group_centroid_per_tstep - init_group_centroid_per_tstep
    relative_all_cell_centroids_per_tstep = all_cell_centroids_per_tstep - init_group_centroid_per_tstep

    # ------------------------
    
    fig, ax = plt.subplots()
    
    # ------------------------
    
    max_data_lim = np.max(np.abs(relative_all_cell_centroids_per_tstep))
    ax.set_xlim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_ylim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_aspect(u'equal')
    
    ax.plot(relative_group_centroid_per_tstep[:,0], relative_group_centroid_per_tstep[:,1], '.', label="group centroid coordinates", color='k', markersize=0.5)
    
    for ci in xrange(num_cells):
        ccs = relative_all_cell_centroids_per_tstep[:,ci,:]
        num_ccs = ccs.shape[0]
        net_displacement = ccs[num_ccs-1] - ccs[0]
        net_displacement_mag = np.linalg.norm(net_displacement)
        net_distance = np.sum(np.linalg.norm(ccs[1:] - ccs[:num_ccs-1], axis=-1))
        persistence = net_displacement_mag/net_distance
        
        ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list20[ci%20], label='cell {}, pers.={}'.format(ci, persistence))
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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

def graph_cell_velocity_over_time(num_cells, T, L, storefile_path, save_dir=None, save_name=None, max_tstep=None, time_to_average_over_in_minutes=1.0):
    fig, ax = plt.subplots()

    for ci in xrange(num_cells):
#        num_timesteps_to_average_over = int(60.0*time_to_average_over_in_minutes/T)
        timepoints, cell_speeds = analysis_utils.calculate_cell_speeds_until_tstep(ci, max_tstep, storefile_path, T, L)
        
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
    
    average_strains = np.average(hardio.get_data_until_timestep(cell_index, max_tstep, 'local_strains', storefile_path), axis=1)*100
    time_points = T*np.arange(average_strains.shape[0])
    
    ax.plot(time_points, average_strains, 'k', label='avg_strains')
        
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
    tumble_periods, run_periods, net_tumble_displacement_mags, mean_tumble_period_speeds, net_run_displacement_mags, mean_run_period_speeds = analysis_utils.calculate_run_and_tumble_statistics(num_nodes, T, L, cell_index, storefile_path, significant_difference=significant_difference)
    
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
    
    cell_centroids_per_tstep = analysis_utils.calculate_cell_centroids_until_tstep(cell_index, max_tstep, storefile_path)*L
    
    data_max_tstep = cell_centroids_per_tstep.shape[0] - 1
    
    ic_contact_data = hardio.get_ic_contact_data(cell_index, max_tstep, storefile_path)
    
    contact_start_end_arrays = analysis_utils.determine_contact_start_ends(ic_contact_data)
    #print "contact_start_end_arrays: ", contact_start_end_arrays
    
    smoothened_contact_start_end_arrays = analysis_utils.smoothen_contact_start_end_tuples(contact_start_end_arrays, min_tsteps_between_arrays=1)
    #print "smoothened_contact_start_end_arrays: ", smoothened_contact_start_end_arrays
    
    assessable_contact_start_end_arrays = hardio.get_assessable_contact_start_end_tuples(smoothened_contact_start_end_arrays, data_max_tstep, min_tsteps_needed_to_calculate_kinematics=min_tsteps_needed_to_calculate_kinematics)
    #print "assessable_contact_start_end_arrays: ", assessable_contact_start_end_arrays
    
    pre_velocities, post_velocities, pre_accelerations, post_accelerations = analysis_utils.calculate_contact_pre_post_kinematics(assessable_contact_start_end_arrays, cell_centroids_per_tstep, delta_tsteps, T)
    
    aligned_pre_velocities, aligned_post_velocities = analysis_utils.rotate_contact_kinematics_data_st_pre_lies_along_given_and_post_maintains_angle_to_pre(pre_velocities, post_velocities, np.array([1, 0]))
    
    aligned_pre_accelerations, aligned_post_accelerations = analysis_utils.rotate_contact_kinematics_data_st_pre_lies_along_given_and_post_maintains_angle_to_pre(pre_accelerations, post_accelerations, np.array([1, 0]))
    
    null_h_prob_velocities = 0
    null_h_prob_accelerations = 0
    max_data_lim_ax0 = 1
    max_data_lim_ax1 = 1
        
    if assessable_contact_start_end_arrays.shape[0] != 0:
        null_h_prob_velocities = np.round(analysis_utils.calculate_null_hypothesis_probability(aligned_post_velocities), decimals=3)
        
        null_h_prob_accelerations = np.round(analysis_utils.calculate_null_hypothesis_probability(aligned_post_accelerations), decimals=3)
    
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
        
def present_collated_single_cell_motion_data(T, L, num_cells, storefile_path, experiment_dir):
    labels = []
    cell_centroid_data = []
    persistences = []
    
    for ci in xrange(num_cells):
        this_cell_centroid_data = analysis_utils.calculate_cell_centroids_for_all_time(ci, storefile_path)
        cell_centroid_data.append(this_cell_centroid_data)
        persistences.append(analysis_utils.calculate_persistence(this_cell_centroid_data))
    
    
    if len(persistences) > 0:
        mean_persistence = np.round(np.average(persistences), decimals=3)
        std_persistence = np.round(np.std(persistences), decimals=3)
    else:
        mean_persistence = None
        std_persistence = None
    
    fig, ax = plt.subplots()
    
    for i, label in enumerate(labels):
        ccs = cell_centroid_data[i]
        ccs = ccs - ccs[0]
        
        if i == 0:
            max_data_lim = np.max(np.abs(ccs))
        else:
            possible_max_data_lim = np.max(np.abs(ccs))
            if  possible_max_data_lim > max_data_lim:
                max_data_lim = possible_max_data_lim
        
        if len(persistences) > 0:
            ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list300[i%20], label='({}, {}), ps.={}'.format(label[0], label[1], np.round(persistences[i], decimals=3)))
        else:
            ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list300[i%20], label='({}, {}), ps.={}'.format(label[0], label[1], None))
        
    ax.set_title("persistence: {} +/- {}".format(mean_persistence, std_persistence))
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")

    ax.set_xlim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_ylim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_aspect(u'equal')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')
    
    if experiment_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(experiment_dir, "collated_single_cell_data" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
    