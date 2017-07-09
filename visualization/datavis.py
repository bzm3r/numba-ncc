# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:21:52 2015

@author: brian
"""

#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import core.utilities as cu
import os
import colors
import core.geometry as geometry
import core.hardio as hardio
from matplotlib import cm

def add_to_general_data_structure(general_data_structure, key_value_tuples):
    if general_data_structure != None:
        if type(general_data_structure) != dict:
            raise StandardError("general_data_structure is not dict, instead: {}".format(type(general_data_structure)))
        else:
            general_data_structure.update(key_value_tuples)
# ====================================================================

def graph_group_area_and_cell_separation_over_time(num_cells, num_timepoints, T, storefile_path, save_dir=None, fontsize=22, general_data_structure=None):
    normalized_areas, normalized_cell_separations = cu.calculate_normalized_group_area_and_average_cell_separation_over_time(num_cells, num_timepoints, storefile_path)
   # normalized_areas_new = cu.calculate_normalized_group_area_over_time(num_cells, num_timepoints, storefile_path)
    timepoints = np.arange(normalized_areas.shape[0])*T
    
    fig_A, ax_A = plt.subplots()
    
    ax_A.plot(timepoints, normalized_areas)
    ax_A.set_ylabel("$A/A_0$")
    ax_A.set_xlabel("time (min.)")
    
    # Put a legend to the right of the current axis
    ax_A.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_A.grid(which=u'both')
    mean, deviation = cu.calculate_mean_and_deviation(normalized_areas)
    
    add_to_general_data_structure(general_data_structure, [("group_area_mean", mean), ("group_area_deviation", deviation)])
    ax_A.set_title("mean = {}, deviation = {}".format(mean, deviation))
    
    fig_S, ax_S = plt.subplots()
    
    ax_S.plot(timepoints, normalized_cell_separations)
    ax_S.set_ylabel("$S/S_0$")
    ax_S.set_xlabel("time (min.)")
    mean, deviation = cu.calculate_mean_and_deviation(normalized_cell_separations)
    ax_S.set_title("mean = {}, deviation = {}".format(mean, deviation))

    add_to_general_data_structure(general_data_structure, [("cell_separation_mean", mean), ("cell_separation_deviation", deviation)])
    
    # Put a legend to the right of the current axis
    ax_S.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_S.grid(which=u'both')
    
    if save_dir == None:
        plt.show()
    else:
        for save_name, fig, ax in [("group_area", fig_A, ax_A), ("cell_separation", fig_S, ax_S)]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
                 ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
            
            fig.set_size_inches(12, 8)
            fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
            plt.close(fig)
        
        plt.close("all")
        
    return general_data_structure
        
# ====================================================================

def graph_avg_neighbour_distance_over_time(general_data_structure=None):
    return 

# ====================================================================

def graph_group_centroid_drift(T, time_unit, relative_group_centroid_per_tstep, save_dir, save_name, fontsize=22, general_data_structure=None):
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
    
    group_velocities = cu.calculate_velocities(relative_group_centroid_per_tstep, T)
    group_x_speeds = np.abs(group_velocities[:, 0])
    group_y_speeds = np.abs(group_velocities[:, 0])
    group_speeds = np.linalg.norm(group_velocities, axis=1)
    A = np.vstack([timepoints, np.ones(len(timepoints))]).T
    fit_group_x_velocity, c = np.linalg.lstsq(A, group_centroid_x_coords)[0]
    
    add_to_general_data_structure(general_data_structure, [("group_speeds", group_speeds)])
    add_to_general_data_structure(general_data_structure, [("average_group_x_speed", np.average(group_x_speeds))])
    add_to_general_data_structure(general_data_structure, [("average_group_y_speed", np.average(group_y_speeds))])
    add_to_general_data_structure(general_data_structure [("fit_group_x_velocity", fit_group_x_velocity)])
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, "group_" + save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
        
    return general_data_structure
    
# ====================================================================

def graph_centroid_related_data(num_cells, num_timepoints, T, time_unit, cell_Ls, storefile_path, save_dir=None, save_name=None, max_tstep=None, make_group_centroid_drift_graph=True, fontsize=22, general_data_structure=None):    
    # assuming that num_timepoints, T is same for all cells
    if max_tstep == None:
        max_tstep = num_timepoints
        
    all_cell_centroids_per_tstep = np.zeros((max_tstep, num_cells, 2), dtype=np.float64)
    
    # ------------------------
    
    for ci in xrange(num_cells):
        cell_centroids_per_tstep = cu.calculate_cell_centroids_until_tstep(ci, max_tstep, storefile_path)*cell_Ls[ci]
        
        all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep
        
    # ------------------------
    add_to_general_data_structure(general_data_structure, [("all_cell_centroids_per_tstep", all_cell_centroids_per_tstep)])
    
    group_centroid_per_tstep = np.array([geometry.calculate_cluster_centroid(cell_centroids) for cell_centroids in all_cell_centroids_per_tstep])
    
    init_group_centroid_per_tstep = group_centroid_per_tstep[0]
    add_to_general_data_structure(general_data_structure, [("group_centroid_per_tstep", group_centroid_per_tstep)])
    relative_group_centroid_per_tstep = group_centroid_per_tstep - init_group_centroid_per_tstep
    relative_all_cell_centroids_per_tstep = all_cell_centroids_per_tstep - init_group_centroid_per_tstep

    group_centroid_displacements = relative_group_centroid_per_tstep[1:] - relative_group_centroid_per_tstep[:-1]
    all_cell_centroid_displacements = np.array([x[1:] - x[:-1] for x in relative_all_cell_centroids_per_tstep])
    
    group_positive_ns, group_positive_das = cu.calculate_direction_autocorr_coeffs_for_persistence_time_parallel(group_centroid_displacements)
    group_persistence_time, group_positive_ts = cu.estimate_persistence_time(T, group_positive_ns, group_positive_das)
    group_persistence_time = np.round(group_persistence_time, 0)
    
    add_to_general_data_structure(general_data_structure, [("group_persistence_time", group_persistence_time)])
    
    positive_ts_per_cell = []
    positive_das_per_cell = []
    all_cell_persistence_times = []
    
    for ci in range(all_cell_centroid_displacements.shape[1]):
        this_cell_centroid_displacements = all_cell_centroid_displacements[:,ci,:]
        this_cell_positive_ns, this_cell_positive_das = cu.calculate_direction_autocorr_coeffs_for_persistence_time_parallel(this_cell_centroid_displacements)
        this_cell_persistence_time, this_cell_positive_ts = cu.estimate_persistence_time(T, this_cell_positive_ns, this_cell_positive_das)
        this_cell_persistence_time = np.round(this_cell_persistence_time, 0)
        
        positive_ts_per_cell.append(this_cell_positive_ts)
        positive_das_per_cell.append(this_cell_positive_das)
        all_cell_persistence_times.append(this_cell_persistence_time)
        
        if save_dir != None:
            fig, ax = plt.subplots()
            graph_title = "persistence time: {} {}".format(np.round(this_cell_persistence_time, decimals=0), time_unit)
            ax.set_title(graph_title)
            ax.plot(this_cell_positive_ts, this_cell_positive_das, color='g', marker='.')
            ax.plot(this_cell_positive_ts, np.exp(-1*this_cell_positive_ts/this_cell_persistence_time), color='r', marker='.')
            
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
                
            fig.set_size_inches(12, 8)
            this_cell_save_dir = os.path.join(save_dir, "cell_{}".format(ci))
            if not os.path.exists(this_cell_save_dir):
                os.makedirs(this_cell_save_dir)
            fig.savefig(os.path.join(this_cell_save_dir, 'persistence_time_estimation' + '.png'), forward=True)
            plt.close(fig)
    
    add_to_general_data_structure(general_data_structure, [("all_cell_persistence_times", all_cell_persistence_times)])
            
    if save_dir != None:
        fig, ax = plt.subplots()
        ax.set_title("persistence time: {}".format(np.round(group_persistence_time, decimals=0)))
        ax.plot(group_positive_ts, group_positive_das, color='g', marker='.')
        ax.plot(group_positive_ts, np.exp(-1*group_positive_ts/group_persistence_time), color='r', marker='.')
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
            
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, 'group_persistence_time_estimation' + '.png'), forward=True)
        plt.close(fig)
        
    # ------------------------
    
    fig, ax = plt.subplots()
    
    # ------------------------
    
    min_x_data_lim = np.min(relative_all_cell_centroids_per_tstep[:,:,0])
    max_x_data_lim = np.max(relative_all_cell_centroids_per_tstep[:,:,0])
    delta_x = np.abs(min_x_data_lim - max_x_data_lim)
    max_y_data_lim = 1.2*np.max(np.abs(relative_all_cell_centroids_per_tstep[:,:,1]))
    ax.set_xlim(min_x_data_lim - 0.1*delta_x, max_x_data_lim + 0.1*delta_x)
    if 2*max_y_data_lim < 0.25*(1.2*delta_x):
        max_y_data_lim = 0.5*1.2*delta_x
    ax.set_ylim(-1*max_y_data_lim, max_y_data_lim)
    ax.set_aspect('equal')
    
    
    group_net_displacement = relative_group_centroid_per_tstep[-1] - relative_group_centroid_per_tstep[0]
    group_net_displacement_mag = np.linalg.norm(group_net_displacement)
    group_net_distance = np.sum(np.linalg.norm(relative_group_centroid_per_tstep[1:] - relative_group_centroid_per_tstep[:-1], axis=1))
    group_persistence_ratio = np.round(group_net_displacement_mag/group_net_distance, 4)
    
    add_to_general_data_structure(general_data_structure, [("group_persistence_ratio", group_persistence_ratio)])
    
    cell_persistence_ratios = []
    for ci in xrange(num_cells):
        ccs = relative_all_cell_centroids_per_tstep[:,ci,:]
        net_displacement = ccs[-1] - ccs[0]
        net_displacement_mag = np.linalg.norm(net_displacement)
        net_distance = np.sum(np.linalg.norm(ccs[1:] - ccs[:-1], axis=-1))
        persistence_ratio = net_displacement_mag/net_distance
        cell_persistence_ratios.append(persistence_ratio)
        
        ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list20[ci%20])
        #ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list20[ci%20], label='cell {}, pers.={}'.format(ci, persistence))
        
    add_to_general_data_structure(general_data_structure, [("all_cell_persistence_ratios", cell_persistence_ratios)])
    
    average_cell_persistence_ratio = np.round(np.average(cell_persistence_ratios), decimals=4)
    std_cell_persistence_ratio = np.round(np.std(cell_persistence_ratios), decimals=4)
    
    ax.plot(relative_group_centroid_per_tstep[:,0], relative_group_centroid_per_tstep[:,1], marker=None, label="group centroid", color='k', linewidth=2)
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")
    
    average_cell_persistence_time = np.round(np.average(all_cell_persistence_times), decimals=2)
    std_cell_persistence_time = np.round(np.std(all_cell_persistence_times), decimals=2)
    ax.set_title("group pers_ratio = {} \n avg. cell pers_ratio = {} (std = {}) \n group pers_time = {} {},  avg. cell pers_time = {} {} (std = {} {})".format(group_persistence_ratio, average_cell_persistence_ratio, std_cell_persistence_ratio, group_persistence_time, time_unit, average_cell_persistence_time, time_unit, std_cell_persistence_time, time_unit))
    #ax.set_title("group pers_ratio = {} \n avg. cell pers_ratio = {} (std = {})".format(group_persistence_ratio, average_cell_persistence_ratio, std_cell_persistence_ratio))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(which=u'both')

    # ------------------------
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        
        fig.set_size_inches(12, 10)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
        
    if make_group_centroid_drift_graph == True:
        general_data_structure = graph_group_centroid_drift(T, time_unit, relative_group_centroid_per_tstep, save_dir, save_name, general_data_structure=general_data_structure)
        
    return general_data_structure
        
# ==============================================================================

def graph_cell_speed_over_time(num_cells, T, cell_Ls, storefile_path, save_dir=None, save_name=None, max_tstep=None, time_to_average_over_in_minutes=1.0, fontsize=22, general_data_structure=None):
    fig_time, ax_time = plt.subplots()
    fig_box, ax_box = plt.subplots()

    average_speeds = []
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
        
        ax_time.plot(timepoints, cell_speeds, color=colors.color_list20[ci%20])
        average_speeds.append(np.average(cell_speeds))
        
    add_to_general_data_structure(general_data_structure, [("all_cell_speeds", average_speeds)])

    # Shrink current axis by 20%
    #box = ax_time.get_position()
    #ax_time.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    #ax_time.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_time.grid(which=u'both')
    
    ax_time.set_xlabel("time (min.)")
    ax_time.set_ylabel("speed ($\mu m$\min.)") 
    
    
    ax_box.violinplot(average_speeds, showmedians=True, points=len(average_speeds))
    
    # adding horizontal grid lines
    ax_box.yaxis.grid(True)
    ax_box.set_xlabel('average cell speed ($\mu m$\min.)')
    ax_box.xaxis.set_ticks([]) 
    ax_box.xaxis.set_ticks_position('none') 
    
    if save_dir == None or save_name == None:
        plt.show()
    else:
        
        for item in ([ax_time.title, ax_time.xaxis.label, ax_time.yaxis.label]  +
             ax_time.get_xticklabels() + ax_time.get_yticklabels()):
            item.set_fontsize(fontsize)
            
        for item in ([ax_box.title, ax_box.xaxis.label, ax_box.yaxis.label]  +
             ax_box.get_xticklabels() + ax_box.get_yticklabels()):
            item.set_fontsize(fontsize)
        
        fig_time.set_size_inches(12, 8)
        fig_time.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        
        fig_box.set_size_inches(8, 8)
        fig_box.savefig(os.path.join(save_dir, save_name + '_box.png'), forward=True)
        
        plt.close(fig_time)
        plt.close(fig_box)
        plt.close("all")
        
    return general_data_structure
        
        
# ==============================================================================

def graph_important_cell_variables_over_time(T, cell_index, storefile_path, polarity_scores=None, save_dir=None, save_name=None, max_tstep=None, fontsize=22, general_data_structure=None):
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
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'), forward=True)
        plt.close(fig)
        plt.close("all")
        
        
# ==============================================================================

def graph_strains(T, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None, fontsize=22, general_data_structure=None):
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
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'))
        plt.close(fig)
        plt.close("all")
        
# ==============================================================================
    
def graph_rates(T, kgtp_rac_baseline, kgtp_rho_baseline, kdgtp_rac_baseline, kdgtp_rho_baseline, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None, fontsize=22, general_data_structure=None):
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
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
        
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, save_name + '.png'))
        plt.close(fig)
        plt.close("all")
        
def graph_run_and_tumble_statistics(num_nodes, T, L, cell_index, storefile_path, save_dir=None, save_name=None, max_tstep=None, significant_difference=0.2, general_data_structure=None):
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
    
    mean_net_tumble_distance = np.round(np.average(net_tumble_displacement_mags), decimals=2)
    std_net_tumble_distance = np.round(np.std(net_tumble_displacement_mags), decimals=2)
    
    mean_net_run_distance = np.round(np.average(net_run_displacement_mags), decimals=2)
    std_net_run_distance = np.round(np.std(net_run_displacement_mags), decimals=2)
    
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
        
def present_collated_single_cell_motion_data(centroids_persistences_speeds_per_repeat, experiment_dir, total_time_in_hours, time_unit, fontsize=22, general_data_structure=None):
    fig, ax = plt.subplots()
    fig_box, ax_box = plt.subplots()
    
    max_data_lim = 0.0
    
    persistence_ratios = [x[1][0] for x in centroids_persistences_speeds_per_repeat]
    mean_persistence_ratio = np.round(np.average(persistence_ratios), 2)
    std_persistence_ratio = np.round(np.std(persistence_ratios), 2)
    persistence_times = [x[1][1] for x in centroids_persistences_speeds_per_repeat]
    mean_persistence_time = np.round(np.average(persistence_times), 0)
    std_persistence_time = np.round(np.std(persistence_times), 0)
    average_cell_speeds = [np.average(x[2]) for x in centroids_persistences_speeds_per_repeat]
    
    for i, cps in enumerate(centroids_persistences_speeds_per_repeat):
        ccs = cps[0]
        ccs = ccs - ccs[0]
        
        #label='({}, {}), ps.={}'.format(si, rpt_number, np.round(persistences[i], decimals=3))
        this_max = np.max(np.abs(ccs))
        if  this_max > max_data_lim:
            max_data_lim = this_max
            
        ax.plot(ccs[:,0], ccs[:,1], marker=None, color=colors.color_list300[i%300])

    ax.set_title("$P_R$ over {} hours (mean: {}, std: {}) \n $P_T$ mean: {} {}, (std {} {})".format(total_time_in_hours, mean_persistence_ratio, std_persistence_ratio, mean_persistence_time, time_unit, std_persistence_time, time_unit))
    #ax.set_title("Persistence ratio over {} hours (mean: {}, std: {})".format(total_time_in_hours, mean_persistence_ratio, std_persistence_ratio))
    
    ax.set_ylabel("micrometers")
    ax.set_xlabel("micrometers")

    ax.set_xlim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_ylim(-1.1*max_data_lim, 1.1*max_data_lim)
    ax.set_aspect(u'equal')
    ax.grid(which=u'both')

    violin = ax_box.violinplot(average_cell_speeds, showmedians=True, points=len(average_cell_speeds))
    
    ax_box.yaxis.grid(True)
    ax_box.set_ylabel('average speed ($\mu m$/min.)')
    ax_box.xaxis.set_ticks([]) 
    ax_box.xaxis.set_ticks_position('none') 
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        
    for item in ([ax_box.title, ax_box.xaxis.label, ax_box.yaxis.label]  +
             ax_box.get_xticklabels() + ax_box.get_yticklabels()):
        item.set_fontsize(fontsize)
    
    fig.set_size_inches(12, 12)
    save_path = os.path.join(experiment_dir, "collated_single_cell_data" + ".png")
    print "save_path: ", save_path
    fig.savefig(save_path, forward=True)
    #plt.tight_layout()
    plt.close(fig)
    
    fig_box.set_size_inches(8, 8)
    save_path = os.path.join(experiment_dir, "collated_single_cell_data_speed_box" + ".png")
    print "save_path: ", save_path
    fig_box.savefig(save_path, forward=True)
    plt.close(fig_box)
    
    plt.close("all")
    
# =============================================================================
        
def present_collated_cell_motion_data(time_unit, all_cell_centroids_per_repeat, all_cell_persistence_ratios_per_repeat, all_cell_persistence_times_per_repeat, all_cell_speeds_per_repeat, all_cell_protrusion_lifetimes_and_directions_per_repeat, group_centroid_per_timestep_per_repeat, group_persistence_ratio_per_repeat, group_persistence_time_per_repeat, experiment_dir, total_time_in_hours, fontsize=22):
    max_x_data_lim = 0.0
    min_x_data_lim = 0.0
    max_y_data_lim = 0.0
    min_y_data_lim = 0.0
    
    all_protrusion_lifetimes_and_average_directions = np.zeros((0, 2), dtype=np.float64)
    
    fig_time, ax_time = plt.subplots()
    for rpti, all_cell_centroids in enumerate(all_cell_centroids_per_repeat):
        for ci in range(all_cell_centroids.shape[1]):
            ccs = all_cell_centroids[:,ci,:]
            relative_ccs = ccs - ccs[0]
                
            this_max_x_data_lim = np.max(relative_ccs[:,0])
            this_min_x_data_lim = np.min(relative_ccs[:,0])
            this_max_y_data_lim = np.max(relative_ccs[:,1])
            this_min_y_data_lim = np.min(relative_ccs[:,1])
            
            if this_max_x_data_lim > max_x_data_lim:
                max_x_data_lim = this_max_x_data_lim
            if this_max_y_data_lim > max_y_data_lim:
                max_y_data_lim = this_max_y_data_lim
            if this_min_x_data_lim < min_x_data_lim:
                min_x_data_lim = this_min_x_data_lim
            if this_min_y_data_lim < min_y_data_lim:
                min_y_data_lim = this_min_y_data_lim
            
            all_protrusion_lifetimes_and_average_directions = np.append(all_protrusion_lifetimes_and_average_directions, np.array(all_cell_protrusion_lifetimes_and_directions_per_repeat[rpti][ci]), axis=0)
                
            ax_time.plot(relative_ccs[:,0], relative_ccs[:,1], marker=None, color=colors.color_list300[ci%300], alpha=0.5)
            
    for rpt_number in range(len(group_centroid_per_timestep_per_repeat)):
        group_centroid_per_timestep = group_centroid_per_timestep_per_repeat[rpt_number]
        relative_group_centroid_per_timestep = group_centroid_per_timestep - group_centroid_per_timestep[0]
        ax_time.plot(relative_group_centroid_per_timestep[:,0], relative_group_centroid_per_timestep[:,1], marker=None, color=colors.color_list300[rpt_number%300], linewidth=2)
        
    mean_persistence_ratio = np.round(np.average(all_cell_persistence_ratios_per_repeat), 2)
    std_persistence_ratio = np.round(np.std(all_cell_persistence_ratios_per_repeat), 2)
    mean_group_persistence_ratio = np.round(np.average(group_persistence_ratio_per_repeat), 2)
    mean_persistence_time = np.round(np.average(all_cell_persistence_times_per_repeat), 0)
    std_persistence_time = np.round(np.std(all_cell_persistence_times_per_repeat), 0)
    mean_group_persistence_time = np.round(np.average(group_persistence_time_per_repeat), 0)

    ax_time.set_title("Experiment over {} hours \n Persistence ratio, cell mean: {} (std: {}), group mean: {} \n Persistence time cell mean: {} {}, (std {} {}), group mean: {} {}".format(total_time_in_hours, mean_persistence_ratio, std_persistence_ratio, mean_group_persistence_ratio, mean_persistence_time, time_unit, std_persistence_time, time_unit, mean_group_persistence_time, time_unit))
    #ax_time.set_title("Experiment over {} hours \n Persistence ratio, cell mean: {} (std: {}), group mean: {}".format(total_time_in_hours, mean_persistence_ratio, std_persistence_ratio, mean_group_persistence_ratio))
    
    ax_time.set_ylabel("micrometers")
    ax_time.set_xlabel("micrometers")

    y_lim = 1.1*np.max([np.abs(min_y_data_lim), np.abs(max_y_data_lim)])
    
    if max_x_data_lim > 0.0:
        max_x_data_lim = 1.1*max_x_data_lim
    else:
        max_x_data_lim = -1*1.1*np.abs(max_x_data_lim)
    
    if min_x_data_lim > 0.0:
        min_x_data_lim = 1.1*min_x_data_lim
    else:
        min_x_data_lim = -1*1.1*np.abs(min_x_data_lim)
        
    ax_time.set_xlim(min_x_data_lim, max_x_data_lim)
    
    if y_lim < 0.2*(max_x_data_lim - min_x_data_lim):
        y_lim = 0.2*(max_x_data_lim - min_x_data_lim) 
        
    ax_time.set_ylim(-1*y_lim, y_lim)    
    ax_time.set_aspect(u'equal')

    
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_time.grid(which=u'both')
    
    fig_box, ax_box = plt.subplots()
    
    all_cell_average_speeds = np.ravel(all_cell_speeds_per_repeat)
    violin = ax_box.violinplot(all_cell_average_speeds, showmedians=True, points=all_cell_average_speeds.shape[0])
    
    ax_box.yaxis.grid(True)
    ax_box.set_ylabel('average speed ($\mu m$/min.)')
    ax_box.xaxis.set_ticks([]) 
    ax_box.xaxis.set_ticks_position('none') 
    
    graph_protrusion_lifetimes_radially(all_protrusion_lifetimes_and_average_directions, 20, save_dir=experiment_dir, save_name="all_cells_protrusion_life_dir")
    
    for item in ([ax_box.title, ax_box.xaxis.label, ax_box.yaxis.label]  +
             ax_box.get_xticklabels() + ax_box.get_yticklabels()):
        item.set_fontsize(fontsize)
        
    for item in ([ax_box.title, ax_box.xaxis.label, ax_box.yaxis.label]  +
             ax_box.get_xticklabels() + ax_box.get_yticklabels()):
        item.set_fontsize(fontsize)
    
    fig_time.set_size_inches(12, 6)
    save_path = os.path.join(experiment_dir, "collated_cell_data" + ".png")
    print "save_path: ", save_path
    fig_time.savefig(save_path, forward=True)
    plt.close(fig_time)
    
    fig_box.set_size_inches(8, 8)
    save_path = os.path.join(experiment_dir, "collated_cell_data_speed_box" + ".png")
    print "save_path: ", save_path
    fig_box.savefig(save_path, forward=True)
    plt.close(fig_box)
    
    plt.close("all")
    
    

#timestep_length, min_x_centroid_per_timestep_per_repeat, max_x_centroid_per_timestep_per_repeat, group_centroid_per_timestep_per_repeat, experiment_dir, total_time_in_hours, group_width=num_cells_width*cell_diameter
def present_collated_group_centroid_drift_data(T, min_x_centroid_per_tstep_per_repeat, max_x_centroid_per_tstep_per_repeat, group_x_centroid_per_tstep_per_repeat, fit_group_x_velocity_per_repeat, save_dir, total_time_in_hours, fontsize=22, general_data_structure=None):
    timepoints = np.arange(group_x_centroid_per_tstep_per_repeat[0].shape[0])*T/60.0
    
    fig_simple_normalized, ax_simple_normalized = plt.subplots()
    fig_full_normalized, ax_full_normalized = plt.subplots()
    fig_simple, ax_simple = plt.subplots()
    fig_full, ax_full = plt.subplots()
    fig_box, ax_box = plt.subplots()
    
    num_repeats = len(group_x_centroid_per_tstep_per_repeat)
    
    bar_step = int(0.01*num_repeats*timepoints.shape[0])
    bar_offset = int(0.01*timepoints.shape[0])

    for repeat_number in range(num_repeats):
        max_x_centroid_per_tstep = max_x_centroid_per_tstep_per_repeat[repeat_number]
        min_x_centroid_per_tstep = min_x_centroid_per_tstep_per_repeat[repeat_number]
        group_x_centroid_per_tstep = group_x_centroid_per_tstep_per_repeat[repeat_number]
        
        group_width = max_x_centroid_per_tstep[0] - min_x_centroid_per_tstep[0]
        graph_upper_lower_bounds = True
        if np.isnan(group_width):
            group_width = 40.0
            max_x_centroid_per_tstep = group_x_centroid_per_tstep
            min_x_centroid_per_tstep = group_x_centroid_per_tstep
            graph_upper_lower_bounds = False

        relative_group_x_centroid_per_tstep = group_x_centroid_per_tstep - min_x_centroid_per_tstep[0]
        relative_max_x_centroid_per_tstep = max_x_centroid_per_tstep - min_x_centroid_per_tstep[0]
        relative_min_x_centroid_per_tstep = min_x_centroid_per_tstep - min_x_centroid_per_tstep[0]
            
        normalized_relative_group_centroid_x_coords = relative_group_x_centroid_per_tstep/group_width
        normalized_relative_max_centroid_x_coords = relative_max_x_centroid_per_tstep/group_width
        normalized_relative_min_centroid_x_coords = relative_min_x_centroid_per_tstep/group_width
        
        if graph_upper_lower_bounds:
            bar_indices = np.arange(bar_offset*repeat_number, timepoints.shape[0] - bar_offset, bar_step, dtype=np.int64)
            
            if repeat_number == 0:
                bar_indices = np.append(bar_indices, timepoints.shape[0] - 1)
            
            bar_timepoints = timepoints[bar_indices]
            
            ax_simple_normalized.plot(timepoints, normalized_relative_group_centroid_x_coords, color=colors.color_list300[repeat_number%300])
            ax_simple.plot(timepoints, relative_group_x_centroid_per_tstep, color=colors.color_list300[repeat_number%300])
            
            ax_full_normalized.plot(timepoints, normalized_relative_group_centroid_x_coords, color=colors.color_list300[repeat_number%300])
            bar_points = normalized_relative_group_centroid_x_coords[bar_indices]
            bar_min_points = normalized_relative_min_centroid_x_coords[bar_indices]
            bar_max_points = normalized_relative_max_centroid_x_coords[bar_indices]
            lower_bounds = np.abs(bar_points - bar_min_points)
            upper_bounds = np.abs(bar_points - bar_max_points)
            ax_full_normalized.errorbar(bar_timepoints, bar_points, yerr=[lower_bounds, upper_bounds], ls='', capsize=5, color=colors.color_list300[repeat_number%300])

            
            ax_full.plot(timepoints, relative_group_x_centroid_per_tstep, color=colors.color_list300[repeat_number%300])
            bar_points = relative_group_x_centroid_per_tstep[bar_indices]
            bar_min_points = relative_min_x_centroid_per_tstep[bar_indices]
            bar_max_points = relative_max_x_centroid_per_tstep[bar_indices]
            lower_bounds = np.abs(bar_points - bar_min_points)
            upper_bounds = np.abs(bar_points - bar_max_points)
            ax_full.errorbar(bar_timepoints, bar_points, yerr=[lower_bounds, upper_bounds], ls='', capsize=5, color=colors.color_list300[repeat_number%300])
        else:
            ax_simple_normalized.plot(timepoints, normalized_relative_group_centroid_x_coords, color=colors.color_list300[repeat_number%300])
            ax_simple.plot(timepoints, relative_group_x_centroid_per_tstep, color=colors.color_list300[repeat_number%300])
            
            ax_full_normalized.plot(timepoints, normalized_relative_group_centroid_x_coords, color=colors.color_list300[repeat_number%300])
            ax_full.plot(timepoints, relative_group_x_centroid_per_tstep, color=colors.color_list300[repeat_number%300])
        
        
    ax_simple_normalized.set_ylabel("position \n (normalized by initial group width)")
    ax_simple.set_ylabel("position ($\mu$m)")
    ax_simple.set_xlabel("time (min.)")
    ax_simple_normalized.set_xlabel("time (min.)")
    ax_full_normalized.set_ylabel("position \n (normalized by initial group width)")
    ax_full.set_ylabel("position ($\mu$m)")
    ax_full_normalized.set_xlabel("time (min.)")
    ax_full.set_xlabel("time (min.)")
    ax_simple_normalized.grid(which=u'both')
    ax_simple.grid(which=u'both')
    ax_full_normalized.grid(which=u'both')
    ax_full.grid(which=u'both')
    ax_simple.set_ylim([0, 1500])
    ax_full.set_ylim([0, 1500])

    violin = ax_box.violinplot(fit_group_x_velocity_per_repeat, showmedians=True, points=len(fit_group_x_velocity_per_repeat))
    ax_box.yaxis.grid(True)
    ax_box.set_ylabel('average group X velocity ($\mu m$/min.)')
    ax_box.xaxis.set_ticks([]) 
    ax_box.xaxis.set_ticks_position('none') 

    
    if save_dir == None:
        plt.show()
    else:
        for ax in [ax_simple, ax_simple_normalized, ax_full, ax_full_normalized, ax_box]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
                 ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
        
        fig_simple_normalized.set_size_inches(12, 8)
        fig_simple_normalized.savefig(os.path.join(save_dir, "collated_group_centroid_drift_simple_normalized.png"), forward=True)
        plt.close(fig_simple_normalized)
        
        fig_simple.set_size_inches(12, 8)
        fig_simple.savefig(os.path.join(save_dir, "collated_group_centroid_drift_simple.png"), forward=True)
        plt.close(fig_simple)
        
        fig_full_normalized.set_size_inches(12, 8)
        fig_full_normalized.savefig(os.path.join(save_dir, "collated_group_centroid_drift_full_normalized.png"), forward=True)
        plt.close(fig_full_normalized)
        
        fig_full.set_size_inches(12, 8)
        fig_full.savefig(os.path.join(save_dir, "collated_group_centroid_drift_full.png"), forward=True)
        plt.close(fig_full)
        
        fig_box.set_size_inches(8, 8)
        fig_box.savefig(os.path.join(save_dir, "collated_group_speed_box.png"), forward=True)
        plt.close(fig_box)
        
        plt.close("all")
            
# ============================================================================

def generate_theta_bins(num_bins):
    delta = 2*np.pi/num_bins
    start = 0.5*delta
    
    bin_bounds = []
    current = start
    for n in range(num_bins - 1):
        bin_bounds.append([current, current + delta])
        current += delta
    bin_bounds.append([2*np.pi - 0.5*delta, start])
    bin_bounds = np.array(bin_bounds)
    bin_mids = np.average(bin_bounds, axis=1)
    bin_mids[-1] = 0.0
    
    return bin_bounds, bin_mids, delta

def graph_protrusion_lifetimes_radially(protrusion_lifetime_and_direction_data, num_polar_graph_bins, save_dir=None, save_name=None, fontsize=22, general_data_structure=None):
    num_polar_graph_bins = 10
    bin_bounds, bin_midpoints, delta = generate_theta_bins(num_polar_graph_bins)
    binned_direction_data = [[] for x in range(num_polar_graph_bins)]    
    for protrusion_result in protrusion_lifetime_and_direction_data:
        lifetime, direction = protrusion_result
        lifetime = lifetime/60.0
        
        binned = False
        for n in range(num_polar_graph_bins - 1):
            a, b = bin_bounds[n]
            if a <= direction < b:
                binned = True
                binned_direction_data[n].append(lifetime)
                break
            
        if binned == False:
            binned_direction_data[-1].append(lifetime)
    
    fig_scaled = plt.figure(figsize=(10, 11))
    ax_scaled = fig_scaled.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    total_lifetimes = np.array([np.sum(x) for x in binned_direction_data])
    max_total_lifetime = np.max(total_lifetimes)
    scaled_lifetimes = total_lifetimes/max_total_lifetime
    
    ax_scaled.bar(bin_midpoints, scaled_lifetimes, width=delta, bottom=0.0)
    ax_scaled.set_title("Normalized total protrusion lifetime given direction \n (1.0 = {} min.)".format(np.round(max_total_lifetime, 1)))
    ax_scaled.title.set_position([.5, 1.1])
    
    fig_avg = plt.figure(figsize=(10, 11))
    ax_avg = fig_avg.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    #average_lifetimes = np.array([np.std(x) for x in binned_direction_data])
    #std_lifetimes = np.array([np.std(x) for x in binned_direction_data])
    bplot = ax_avg.boxplot(binned_direction_data, positions=bin_midpoints, showfliers=False, whis=0.0, showcaps=False, patch_artist=True)
    for pa in bplot['boxes']:
        pa.set_facecolor('lightblue')
        pa.set_alpha(0.25)
    theta_labels = ["{}$\pi$".format(theta) for theta in np.round(bin_midpoints/np.pi, decimals=2)]
    ax_avg.set_xticklabels(theta_labels)
    ax_avg.set_ylim([0, 60.0])
    #ax_avg.violinplot(binned_direction_data, bin_midpoints, widths=np.ones(num_polar_graph_bins, dtype=np.float64), showmeans=True, showextrema=False)
    #ax_avg.set_ylim([0, 60.0])
    #ax_avg.bar(bin_midpoints, average_lifetimes, width=delta, bottom=0.0)
    #ax_avg.errorbar(bin_midpoints, average_lifetimes, yerr=[np.min()], capsize=0, fmt="o", color='k')
    bin_sizes = [len(x) for x in binned_direction_data]
    min_bin_size = np.min(bin_sizes)
    max_bin_size = np.max(bin_sizes)
    ax_avg.set_title("Average protrusion lifetime given direction \n (min, max bin_size = {}, {})".format(min_bin_size, max_bin_size))
    ax_avg.title.set_position([.5, 1.1])

    
    if save_dir == None:
        plt.show()
    else:
        if save_name == None:
            save_name = "protrusion_lifetime_versus_direction"
        
        for fig_ax_save_prefix in [(fig_scaled, ax_scaled, "scaled_total_"), (fig_avg, ax_avg, "avg_")]:
            fig, ax, save_prefix = fig_ax_save_prefix
            for item in ([ax.title, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
            
            save_path = os.path.join(save_dir, save_prefix + save_name + ".png")
            print "save_path: ", save_path
            fig.savefig(save_path, forward=True)
            
        plt.close("all")        
        
def graph_protrusion_start_end_causes_radially(protrusion_lifetime_and_direction_data, protrusion_start_end_cause_data, num_polar_graph_bins, save_dir=None, general_data_structure=None):
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

def graph_forward_backward_protrusions_per_timestep(max_tstep, protrusion_node_index_and_tpoint_start_ends, protrusion_lifetime_and_direction_data, T, forward_cones, backward_cones, num_nodes, save_dir=None, general_data_structure=None):
    times = np.arange(max_tstep)*T/60.0
    num_forward_protrusions = np.zeros(max_tstep, dtype=np.float)
    num_backward_protrusions = np.zeros(max_tstep, dtype=np.float)
    num_neither_protrusions = np.zeros(max_tstep, dtype=np.float64)
    
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
            else:
                num_neither_protrusions[ti_start:ti_end] += 1
                
    fig, ax = plt.subplots()
    
    ax.plot(times, num_forward_protrusions, label='forward')
    ax.plot(times, num_backward_protrusions, label='backward')
    #ax.plot(times, other_cone, label='other')
    
    ax.legend(loc='best')
    ax.set_ylabel("number of protrusions")
    ax.set_xlabel("time (min.)")
    
    fig_with_other, ax_with_other = plt.subplots()
    
    ax_with_other.plot(times, num_forward_protrusions, label='forward')
    ax_with_other.plot(times, num_backward_protrusions, label='backward')
    ax_with_other.plot(times, num_neither_protrusions, label='other')
    #ax.plot(times, other_cone, label='other')
    
    ax_with_other.legend(loc='best')
    ax_with_other.set_ylabel("number of protrusions")
    ax_with_other.set_xlabel("time (min.)")
    
    
    fig_normalized, ax_normalized = plt.subplots()
    
    ax_normalized.plot(times, num_forward_protrusions/num_nodes, label='forward')
    ax_normalized.plot(times, num_backward_protrusions/num_nodes, label='backward')
    ax_normalized.plot(times, num_neither_protrusions/num_nodes, label='other')
    #ax.plot(times, other_cone, label='other')
    
    ax_normalized.legend(loc='best')
    ax_normalized.set_ylabel("number of protrusions/number of nodes per cell")
    ax_normalized.set_xlabel("time (min.)")
    
    
    fig_normalized_with_other, ax_normalized_with_other = plt.subplots()
    
    ax_normalized_with_other.plot(times, num_forward_protrusions/num_nodes, label='forward')
    ax_normalized_with_other.plot(times, num_backward_protrusions/num_nodes, label='backward')
    ax_normalized_with_other.plot(times, num_neither_protrusions/num_nodes, label='other')
    #ax.plot(times, other_cone, label='other')
    
    ax_normalized_with_other.legend(loc='best')
    ax_normalized_with_other.set_ylabel("number of protrusions/number of nodes per cell")
    ax_normalized_with_other.set_xlabel("time (min.)")
    
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "num_forward_backward_protrusions_over_time" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        
        fig_with_other.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "num_fbo_protrusions_over_time" + ".png")
        print "save_path: ", save_path
        fig_with_other.savefig(save_path, forward=True)
        plt.close(fig_with_other)
        
        fig_normalized.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "normalized_forward_backward_protrusions_over_time" + ".png")
        print "save_path: ", save_path
        fig_normalized.savefig(save_path, forward=True)
        plt.close(fig_normalized)
        
        fig_normalized_with_other.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "normalized_fbo_protrusions_over_time" + ".png")
        print "save_path: ", save_path
        fig_normalized_with_other.savefig(save_path, forward=True)
        plt.close(fig_normalized_with_other)
        
        plt.close("all")
    
# ============================================================================

def graph_forward_backward_cells_per_timestep(max_tstep, all_cell_speeds_and_directions, T, forward_cones, backward_cones, save_dir=None, general_data_structure=None):
    times = np.arange(max_tstep)*T/60.0
    num_forward_cells = np.zeros(max_tstep, dtype=np.float64)
    num_backward_cells = np.zeros(max_tstep, dtype=np.float64)
    num_other_cells = np.zeros(max_tstep, dtype=np.float64)
    
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
                else:
                    num_other_cells[ti] += 1
                
    fig, ax = plt.subplots()
    
    ax.plot(times, num_forward_cells, label='forward')
    ax.plot(times, num_backward_cells, label='backward')
    
    ax.legend(loc='best')
    
    ax.set_ylabel("number of cells")
    ax.set_xlabel("time (min.)")
    
    fig_with_other, ax_with_other = plt.subplots()
    
    ax_with_other.plot(times, num_forward_cells, label='forward')
    ax_with_other.plot(times, num_backward_cells, label='backward')
    ax_with_other.plot(times, num_other_cells, label='other')
    
    ax_with_other.legend(loc='best')
    
    ax_with_other.set_ylabel("number of cells")
    ax_with_other.set_xlabel("time (min.)")
    
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "num_forward_backward_cells_over_time" + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        
        fig_with_other.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "num_fbo_cells_over_time" + ".png")
        print "save_path: ", save_path
        fig_with_other.savefig(save_path, forward=True)
        plt.close(fig_with_other)
        
        
        plt.close("all")
            
        
# =============================================================================

def graph_coa_variation_test_data(sub_experiment_number, num_cells_to_test, test_coas, average_cell_group_area_data, save_dir=None, max_normalized_group_area=3.0, general_data_structure=None):
    
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
        
        
def graph_confinement_data_persistence_ratios(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, save_dir=None, general_data_structure=None):
    
    fig, ax = plt.subplots()
    
    bin_boundaries = np.linspace(0.5, 1.0, num=100)
    cax = ax.imshow(average_cell_persistence, interpolation='none', cmap=plt.get_cmap('inferno'))      
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
        save_path = os.path.join(save_dir, "confinement_test_persistence_ratios_{}".format(sub_experiment_number) + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")

def graph_confinement_data_persistence_times(sub_experiment_number, test_num_cells, test_heights, average_cell_persistence, save_dir=None, general_data_structure=None):
    
    fig, ax = plt.subplots()
    
    cax = ax.imshow(average_cell_persistence, interpolation='none', cmap=plt.get_cmap('inferno'))      
    cbar = fig.colorbar(cax)

    ax.set_yticks(np.arange(len(test_num_cells)))
    ax.set_xticks(np.arange(len(test_heights)))
    ax.set_yticklabels(test_num_cells)
    ax.set_xticklabels(test_heights)
     
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "confinement_test_graph_persistence_times_{}".format(sub_experiment_number) + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        
# =============================================================================

def graph_cell_number_change_data(sub_experiment_number, test_num_cells, group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations, aspect_ratio, save_dir=None):
    fig, axarr = plt.subplots(nrows=4, sharex=True)
    
    ds_dicts = dict(zip(["group persistence ratios", "group persistence times", "group X velocity", "average cell separation"], [group_persistence_ratios, group_persistence_times, fit_group_x_velocities, cell_separations]))
    ds_y_label_dict = dict(zip(["group persistence ratios", "group persistence times", "group X velocity", "average cell separation"], ["", "min.", "$\mu$m/min.", ""]))
    
    last_index = 3
    for i, ds_label in enumerate(["average cell separation", "group persistence ratios", "group persistence times", "group X velocity"]):
        ds = ds_dicts[ds_label]
        
        axarr[i].errorbar(test_num_cells, np.average(ds, axis=1), yerr=np.std(ds, axis=1), fmt='o')
        axarr[i].set_title(ds_label)
        axarr[i].set_ylabel(ds_y_label_dict[ds_label])
        
        if i == last_index:
            axarr[i].set_xlabel("number of cells in group")
            axarr[i].set_xticks(test_num_cells)
            axarr[i].set_xticklabels([str(j) for j in test_num_cells])
            
    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "cell_number_change_data_AR={}_{}".format(aspect_ratio, sub_experiment_number) + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")
        

    
    

    
    
        
        