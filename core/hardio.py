# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:58:28 2016

@author: Brian
"""

import h5py
import parameterorg
import numpy as np

# ============================================================================== 

def create_cell_dataset(cell_index, storefile_path, num_nodes, num_info_labels):
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset(str(cell_index), (0, num_nodes, num_info_labels), maxshape=(None, num_nodes, num_info_labels), chunks=True, shuffle=True)
        
    return
    
# ============================================================================== 

def append_cell_data_to_dataset(cell_index, cell_data, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        orig_index = dset.shape[0]
#        if orig_index == -1:
#            orig_index = 0
        
        dset.resize(dset.shape[0] + cell_data.shape[0], axis=0)
        dset[orig_index:,:,:] = cell_data  
        
# ============================================================================== 
        
def get_node_coords_for_all_tsteps(cell_index, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        x_coords = dset[:,:,parameterorg.x_index]
        y_coords = dset[:,:,parameterorg.y_index]
        
        node_coords = np.zeros((x_coords.shape[0], x_coords.shape[1], 2), dtype=np.float64)
    
        node_coords[:,:,0] = x_coords
        node_coords[:,:,1] = y_coords
        
        return node_coords
        
# ============================================================================== 

def get_node_coords_until_tstep(cell_index, max_tstep, storefile_path):
    if max_tstep == None:
        return get_node_coords_for_all_tsteps(cell_index, storefile_path)
    else:
        with h5py.File(storefile_path, "a") as f:
            dset = f[str(cell_index)]
            
            x_coords = dset[:max_tstep,:,parameterorg.x_index]
            y_coords = dset[:max_tstep,:,parameterorg.y_index]
            
            node_coords = np.zeros((x_coords.shape[0], x_coords.shape[1], 2), dtype=np.float64)
        
            node_coords[:,:,0] = x_coords
            node_coords[:,:,1] = y_coords
            
            return node_coords
    
# ==============================================================================
    
def get_node_coords_for_given_tsteps(cell_index, tsteps, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        x_coords = dset[tsteps,:,parameterorg.x_index]
        y_coords = dset[tsteps,:,parameterorg.y_index]
        
        node_coords = np.zeros((x_coords.shape[0], x_coords.shape[1], 2), dtype=np.float64)
        
        node_coords[:,:,0] = x_coords
        node_coords[:,:,1] = y_coords
        
        return node_coords
        
# ==============================================================================
    
def get_node_coords(cell_index, tstep, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        x_coords = dset[tstep,:,parameterorg.x_index]
        y_coords = dset[tstep,:,parameterorg.y_index]
        
        node_coords = np.zeros((x_coords.shape[0], 2), dtype=np.float64)
        
        node_coords[:,0] = x_coords
        node_coords[:,1] = y_coords
        
        return node_coords
    
# ==============================================================================
    
def get_data_for_tsteps(cell_index, tsteps, data_label, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        if type(tsteps) != type(np.array([])) and type(tsteps) != int:
            return dset[:,:,parameterorg.info_indices_dict[data_label]]
        else:
            return dset[tsteps,:,parameterorg.info_indices_dict[data_label]]
        
# ==============================================================================  
    
def get_data_until_timestep(cell_index, max_tstep, data_label, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        if max_tstep == None:
            return dset[:,:,parameterorg.info_indices_dict[data_label]]
        else:
            return dset[:max_tstep,:,parameterorg.info_indices_dict[data_label]]
            
# ============================================================================== 
            

