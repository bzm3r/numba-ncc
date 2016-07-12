# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:58:28 2016

@author: Brian
"""

import h5py
import parameterorg
import numpy as np

# ============================================================================== 

def create_exec_order_dataset(storefile_path, num_cells):
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset("exec_order", shape=(0, num_cells), maxshape=(None, num_cells), chunks=True, shuffle=True)

    return

# ============================================================================== 

    
def create_cell_dataset(cell_index, storefile_path, num_nodes, num_info_labels):
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset(str(cell_index), shape=(0, num_nodes, num_info_labels), maxshape=(None, num_nodes, num_info_labels), chunks=True, shuffle=True)
        
    return
    
# ============================================================================== 
    
def append_exec_orders_to_dataset(exec_orders, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f["exec_order"]
        
        orig_index = dset.shape[0]
        dset.resize(dset.shape[0] + exec_orders.shape[0], axis=0)
        dset[orig_index:,:] = exec_orders

# ============================================================================== 

def append_cell_data_to_dataset(cell_index, cell_data, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        orig_index = dset.shape[0]
        
        dset.resize(dset.shape[0] + cell_data.shape[0], axis=0)
        dset[orig_index:,:,:] = cell_data  

# ============================================================================== 

def get_storefile_tstep_range(num_cells, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        tstep_range = f[str(num_cells - 1)].shape[0]
        
        for ci in range(num_cells - 1):
            assert(tstep_range == f[str(ci)].shape[0])
            
        return tstep_range
# ============================================================================== 
    
def get_exec_order_for_tsteps(tsteps, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f["exec_order"]
        
        return dset[tsteps]
    
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


def check_if_data_label_is_vector_type(data_label):
    allowed_labels = ["F", "EFplus", "EFminus", "F_rgtpase", "F_cytoplasmic", "F_adhesion", "unit_in_vec"]
    
    if data_label not in allowed_labels:
        raise StandardError("get_vector_data_* functions are not meant to be used with given label: {}. Allowed labels are: {}".format(data_label, allowed_labels))
      
      
               
# ============================================================================== 
               
def get_vector_data_until_timestep(cell_index, max_tstep, data_label, storefile_path):
    check_if_data_label_is_vector_type(data_label)
    
    data = None
    for i, basis_string in enumerate(["_x", "_y"]):
        data_for_this_basis = get_data_until_timestep(cell_index, max_tstep, data_label + basis_string, storefile_path)
        
        if data == None:
            data = np.zeros((data_for_this_basis.shape[0], data_for_this_basis.shape[1], 2), dtype=data_for_this_basis.dtype)
        
        data[:, :, i] = data_for_this_basis
        
    return data
               
# ============================================================================== 
            
def get_data_for_all_timesteps(cell_index, data_label, storefile_path):
    return get_data_until_timestep(cell_index, None, data_label, storefile_path)


