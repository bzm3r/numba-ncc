# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 10:42:37 2016

@author: Brian
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import general.exec_utils as exec_utils
import analysis.utilities as analysis_utils
import os
import core.geometry as geometry
import scipy.spatial as space

ANALYSIS_NUMBER = 0
analysis_description = "Putting together "

BASE_OUTPUT_DIR = "A:\\numba-ncc\\output\\"
DATE_STR = "2016_APR_01"

analysis_dir = exec_utils.get_analysis_directory_path(BASE_OUTPUT_DIR, DATE_STR, ANALYSIS_NUMBER)

relevant_experiment_info = [("2016_MAR_28", 0, x) for x in [0, 1, 2]]

environment_dirs = exec_utils.get_environment_dirs_given_relevant_experiment_info(BASE_OUTPUT_DIR, relevant_experiment_info)

save_dir = os.path.join(BASE_OUTPUT_DIR, DATE_STR, "ANA_{}".format(ANALYSIS_NUMBER))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

        
def make_centroid_drift_and_delaunay_plots(environment_dirs, relevant_experiment_info, save_dir):
    x_scale_factor = 400
    
    group_centroids_and_tsteps_per_env = []
    normalized_convex_hull_areas_and_tsteps_per_env = []
    
    num_envs = len(environment_dirs)
    for i, environment_dir in enumerate(environment_dirs):
        print "processing env: {}, {}/{}".format(environment_dir, i+1, num_envs)
        print "retrieving pickle file..."
        stored_env = exec_utils.retrieve_environment(os.path.join(environment_dir, "environment.pkl"), False, False)
        print "gathering cell Ls..."
        storefile_path = os.path.join(environment_dir, "store.hdf5")                
        cell_Ls = [a_cell.L/1e-6 for a_cell in stored_env.cells_in_environment]
        num_cells = stored_env.num_cells
        T = stored_env.T/60.0
        max_tstep = stored_env.num_timepoints
        
        all_cell_centroids_per_tstep = np.zeros((max_tstep, num_cells, 2), dtype=np.float64)
        
        # ------------------------
        
        print "calculating cell centroids..."
        for ci in xrange(num_cells):
            cell_centroids_per_tstep = analysis_utils.calculate_cell_centroids_until_tstep(ci, max_tstep, storefile_path)*cell_Ls[ci]
            
            all_cell_centroids_per_tstep[:, ci, :] = cell_centroids_per_tstep
        
        # ------------------------
        
        print "calculating delaunay areas..."
        delaunay_triangulations_per_tstep = [space.Delaunay(all_cell_centroids) for all_cell_centroids in all_cell_centroids_per_tstep]
    
        convex_hull_areas = []
    
        for dt, all_cell_centroids in zip(delaunay_triangulations_per_tstep, all_cell_centroids_per_tstep):
            simplices = all_cell_centroids[dt.simplices]
            simplex_areas = np.array([geometry.calculate_polygon_area(simplex.shape[0], simplex) for simplex in simplices])
            convex_hull_areas.append(np.round(np.sum(simplex_areas), decimals=3))
            
        normalized_convex_hull_areas = np.array(convex_hull_areas/convex_hull_areas[0])
        timepoints = np.arange(normalized_convex_hull_areas.shape[0])*T
        normalized_convex_hull_areas_and_tsteps_per_env.append((timepoints, normalized_convex_hull_areas))
        # ------------------------
        
        print "preparing group centroid..."
        group_centroid_per_tstep = np.array([geometry.calculate_cluster_centroid(cell_centroids) for cell_centroids in all_cell_centroids_per_tstep])/x_scale_factor
        
        init_group_centroid_per_tstep = group_centroid_per_tstep[0]
        group_centroid_per_tstep = group_centroid_per_tstep - init_group_centroid_per_tstep
        timepoints = np.arange(group_centroid_per_tstep.shape[0])*T
        
        group_centroids_and_tsteps_per_env.append((timepoints, group_centroid_per_tstep[:,0]))
        
    
    fig, ax = plt.subplots()
    
    for exp_info, gcs_and_tsteps in zip(relevant_experiment_info, group_centroids_and_tsteps_per_env):
        date_str, exp_number, sub_exp_number = exp_info
        timesteps, gcs = gcs_and_tsteps
        
        ax.plot(timesteps, gcs, label="EXP {}-{}".format(exp_number, sub_exp_number))
    
        # Put a legend to the right of the current axis
    ax.legend(loc='best')
    ax.grid(which=u'both')
    
    ax.set_title("Group centroid position vs. time")
    ax.set_ylabel("x position (scaled by group width)")
    ax.set_xlabel("time (minutes)")
    
    if save_dir == None:
        print "No save_dir specified, showing graph..."
        plt.show()
    else:
        print "Saving graph at specified location..."
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, "collated_group_centroid" + ".png"), forward=True)
        plt.close(fig)
        plt.close("all")
        
    fig, ax = plt.subplots()
    
    for exp_info, nas_and_tsteps in zip(relevant_experiment_info, normalized_convex_hull_areas_and_tsteps_per_env):
        date_str, exp_number, sub_exp_number = exp_info
        timesteps, nas = nas_and_tsteps
        
        ax.plot(timesteps, nas, label="EXP {}-{}".format(exp_number, sub_exp_number))
    
        # Put a legend to the right of the current axis
    ax.legend(loc='best')
    ax.grid(which=u'both')
    
    ax.set_title("Group area vs. time")
    ax.set_ylabel("area (scaled by initial area)")
    ax.set_xlabel("time (minutes)")
    
    if save_dir == None:
        print "No save_dir specified, showing graph..."
        plt.show()
    else:
        print "Saving graph at specified location..."
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(save_dir, "collated_group_area" + ".png"), forward=True)
        plt.close(fig)
        plt.close("all")
        

make_centroid_drift_and_delaunay_plots(environment_dirs, relevant_experiment_info, save_dir)
        