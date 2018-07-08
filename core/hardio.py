# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:58:28 2016

@author: Brian
"""

import h5py
from . import parameterorg
import numpy as np
import os

# ==============================================================================
def create_coa_overlay_dataset(grid_points, unique_undrawn_timesteps, grid_points_data, storefile_path):
    if os.path.isfile(storefile_path):
        print("Erasing existing coa overlay data.")
        os.remove(storefile_path)
        
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset("timepoints", shape=(0,), maxshape=(None,))
        f.create_dataset("grid_points", shape=grid_points.shape, maxshape=(None, 2))
        dset = f['grid_points']
        dset[:,:] = grid_points
        
        f.create_dataset("grid_points_data", shape=grid_points_data.shape, maxshape=(None, None))

        
    append_coa_overlay_data(storefile_path, grid_points, unique_undrawn_timesteps, grid_points_data)
        
    print("Finished storing coa overlay data.")
        
    return

# ============================================================================== 
def create_parameter_exploration_dataset(storefile_path, length_results_axis1):
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset("exploration_results", shape=(0, length_results_axis1), maxshape=(None, length_results_axis1),  shuffle=True)
        f.create_dataset("last_executed_chunk_index", shape=(1,), maxshape=(1,))
        
    return
        
# ============================================================================== 

def create_exec_order_dataset(storefile_path, num_cells):
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset("exec_order", shape=(0, num_cells), maxshape=(None, num_cells))

    return

# ============================================================================== 

    
def create_cell_dataset(cell_index, storefile_path, num_nodes, num_info_labels):
    with h5py.File(storefile_path, "a") as f:
        f.create_dataset(str(cell_index), shape=(0, num_nodes, num_info_labels), maxshape=(None, num_nodes, num_info_labels),  shuffle=True)
        
    return
    
# ============================================================================== 
    
def append_coa_overlay_data(storefile_path, grid_points, unique_undrawn_timesteps, grid_points_data):
    timepoints = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f["timepoints"]
        timepoints = np.empty(dset.shape, dtype=np.int64)
        if dset.shape[0] > 0:
            dset.read_direct(timepoints)
        
    unstored_timepoints = []
    unstored_data_indices = []
    for unstored_data_index, tpoint in enumerate(unique_undrawn_timesteps):
        if tpoint not in timepoints:
            unstored_data_indices.append(unstored_data_index)
            unstored_timepoints.append(tpoint)
            
    unstored_timepoints = np.array(unstored_timepoints)
    unstored_data_indices = np.array(unstored_data_indices)
            
    with h5py.File(storefile_path, "a") as f:
        dset = f["timepoints"]
        orig_index = dset.shape[0]
        dset.resize(dset.shape[0] + unstored_timepoints.shape[0], axis=0)
        dset[orig_index:] = unstored_timepoints
        
        dset = f["grid_points_data"]
        orig_index = dset.shape[0]
        dset.resize(dset.shape[0] + unstored_timepoints.shape[0], axis=0)
        dset[orig_index:,:] = grid_points_data[unstored_data_indices]
        
    return

# ============================================================================== 
    
def append_exec_orders_to_dataset(exec_orders, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f["exec_order"]
        
        orig_index = dset.shape[0]
        dset.resize(dset.shape[0] + exec_orders.shape[0], axis=0)
        dset[orig_index:,:] = exec_orders
        
    return

# ============================================================================== 

def append_cell_data_to_dataset(cell_index, cell_data, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        
        orig_index = dset.shape[0]
        
        dset.resize(dset.shape[0] + cell_data.shape[0], axis=0)
        dset[orig_index:,:,:] = cell_data
        
    return
        
# ==============================================================================

def append_parameter_exploration_data_to_dataset(new_last_executed_chunk_index, parameter_exploration_results, storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f["exploration_results"]
        
        orig_index = dset.shape[0]
        dset.resize(dset.shape[0] + parameter_exploration_results.shape[0], axis=0)
        dset[orig_index:,:] = parameter_exploration_results
        
        dset = f["last_executed_chunk_index"]
        
        dset[0] = new_last_executed_chunk_index

    return
        
# ============================================================================== 
def get_coa_overlay_data(storefile_path, expected_grid_points):
    on_ram_grid_points = np.array([])
    on_ram_grid_point_data = np.array([])
    on_ram_timepoints = np.array([])
    sortorder = None
    
    if os.path.isfile(storefile_path):
        dset = None
        with h5py.File(storefile_path, "a") as f:
            try:
                dset = f["grid_points"]
            except:
                return False, None
            
            if type(dset) != type(None) and dset.shape == expected_grid_points.shape:
                on_ram_grid_points = np.empty(dset.shape, dtype=np.float64)
                dset.read_direct(on_ram_grid_points)
                
                if np.all(on_ram_grid_points == expected_grid_points):
                    dset = f["data_per_grid_point"]
                    on_ram_grid_point_data = np.empty(dset.shape, dtype=np.float64)
                    dset.read_direct(on_ram_grid_point_data)
                    
                    dset = f["timepoints"]
                    on_ram_timepoints = np.empty(dset.shape, dtype=np.int64)
                    dset.read_direct(on_ram_timepoints)
                    sortorder = np.argsort(on_ram_timepoints)
        
    if type(sortorder) != np.ndarray:
        return False, None
    else:
        return True, on_ram_grid_point_data[sortorder]
        
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
        exec_order = np.empty(dset.shape, dtype=dset.dtype)
        dset.read_direct(exec_order)
        return exec_order
    
# ============================================================================== 
    
def get_node_coords_for_all_tsteps(cell_index, storefile_path):
    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.copy(dset)
    
    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")
        
    x_coords = on_ram_dset[:,:,parameterorg.x_index]
    y_coords = on_ram_dset[:,:,parameterorg.y_index]
    
    node_coords = np.zeros((x_coords.shape[0], x_coords.shape[1], 2), dtype=np.float64)

    node_coords[:,:,0] = x_coords
    node_coords[:,:,1] = y_coords
        
    return node_coords
        
# ============================================================================== 

def get_node_coords_until_tstep(cell_index, max_tstep, storefile_path):
    if max_tstep == None:
        return get_node_coords_for_all_tsteps(cell_index, storefile_path)
    else:
        on_ram_dset = np.array([])
        with h5py.File(storefile_path, "a") as f:
            dset = f[str(cell_index)]
            on_ram_dset = np.copy(dset)
        
        if on_ram_dset.shape[0] == 0:
            raise Exception("on_ram_dset is empty!")

            
        x_coords = on_ram_dset[:max_tstep,:,parameterorg.x_index]
        y_coords = on_ram_dset[:max_tstep,:,parameterorg.y_index]
        
        node_coords = np.zeros((x_coords.shape[0], x_coords.shape[1], 2), dtype=np.float64)
    
        node_coords[:,:,0] = x_coords
        node_coords[:,:,1] = y_coords
            
        return node_coords
    
# ==============================================================================
    
def get_node_coords_for_given_tsteps(cell_index, tsteps, storefile_path):
    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.copy(dset)
        
    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")

    x_coords = on_ram_dset[tsteps,:,parameterorg.x_index]
    y_coords = on_ram_dset[tsteps,:,parameterorg.y_index]
        
    node_coords = np.zeros((x_coords.shape[0], x_coords.shape[1], 2), dtype=np.float64)
    
    node_coords[:,:,0] = x_coords
    node_coords[:,:,1] = y_coords
        
    return node_coords
        
# ==============================================================================
    
def get_node_coords(cell_index, tstep, storefile_path):
    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.empty(dset.shape, dtype=np.float64)
        dset.read_direct(on_ram_dset)
        
    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")
        
    x_coords = on_ram_dset[tstep,:,parameterorg.x_index]
    y_coords = on_ram_dset[tstep,:,parameterorg.y_index]
    
    node_coords = np.zeros((x_coords.shape[0], 2), dtype=np.float64)
    
    node_coords[:,0] = x_coords
    node_coords[:,1] = y_coords
    
    return node_coords
    
# ==============================================================================
    
def get_cell_data_for_tsteps(cell_index, tsteps, data_labels, storefile_path):
    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.empty(dset.shape, dtype=np.float64)
        dset.read_direct(on_ram_dset)

    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")
    
    fetched_data = []
    for data_label in data_labels:    
        if type(tsteps) != type(np.array([])) and type(tsteps) != int:
            fetched_data.append(on_ram_dset[:,:,parameterorg.info_indices_dict[data_label]])
        else:
            fetched_data.append(on_ram_dset[tsteps,:,parameterorg.info_indices_dict[data_label]])
        
    return fetched_data
        
# ==============================================================================
    
def get_data_for_tsteps(cell_index, tsteps, data_label, storefile_path):
    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.empty(dset.shape, dtype=np.float64)
        dset.read_direct(on_ram_dset)
        
    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")
        
    if type(tsteps) != type(np.array([])) and type(tsteps) != int:
        return on_ram_dset[:,:,parameterorg.info_indices_dict[data_label]]
    else:
        return on_ram_dset[tsteps,:,parameterorg.info_indices_dict[data_label]]
        
# ==============================================================================  
    
def get_data_until_timestep(cell_index, max_tstep, data_label, storefile_path):
    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.empty(dset.shape, dtype=np.float64)
        dset.read_direct(on_ram_dset)
        
    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")
    else:
        if max_tstep == None:
            return on_ram_dset[:,:,parameterorg.info_indices_dict[data_label]]
        else:
            return on_ram_dset[:max_tstep,:,parameterorg.info_indices_dict[data_label]]
        
def get_multiple_data_until_timestep(cell_index, max_tstep, data_labels, data_types, storefile_path):
    return_results = []

    on_ram_dset = np.array([])
    with h5py.File(storefile_path, "a") as f:
        dset = f[str(cell_index)]
        on_ram_dset = np.empty(dset.shape, dtype=np.float64)
        dset.read_direct(on_ram_dset)
        
    if on_ram_dset.shape[0] == 0:
        raise Exception("on_ram_dset is empty!")
        
    for data_label, data_type in zip(data_labels, data_types):
        if data_type == 'n':
            if max_tstep == None:
                return_results.append(on_ram_dset[:,:,parameterorg.info_indices_dict[data_label]])
            else:
                return_results.append(on_ram_dset[:max_tstep,:,parameterorg.info_indices_dict[data_label]])
        elif data_type == 'v':
            check_if_data_label_is_vector_type(data_label)

            data = None
            for i, basis_string in enumerate(["_x", "_y"]):
                if max_tstep == None:
                    data_for_this_basis = on_ram_dset[:,:,parameterorg.info_indices_dict[data_label + basis_string]]
                else:
                    data_for_this_basis = on_ram_dset[:max_tstep,:,parameterorg.info_indices_dict[data_label + basis_string]]
                
                if type(data) == type(None):
                    data = np.zeros((data_for_this_basis.shape[0], data_for_this_basis.shape[1], 2), dtype=data_for_this_basis.dtype)
                
                data[:, :, i] = data_for_this_basis
                
            return_results.append(data)
                
    return return_results


def check_if_data_label_is_vector_type(data_label):
    allowed_labels = ["F", "EFplus", "EFminus", "F_rgtpase", "F_cytoplasmic", "F_adhesion", "unit_in_vec"]
    
    if data_label not in allowed_labels:
        raise Exception("get_vector_data_* functions are not meant to be used with given label: {}. Allowed labels are: {}".format(data_label, allowed_labels))
      
      
               
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

# ==============================================================================

def get_last_executed_parameter_exploration_chunk_index(storefile_path):
    last_executed_chunk_index = -1
    with h5py.File(storefile_path, "a") as f:
        last_executed_chunk_index = int(f["last_executed_chunk_index"][0])
    
    return last_executed_chunk_index

# ==============================================================================

def get_parameter_exploration_results(storefile_path):
    with h5py.File(storefile_path, "a") as f:
        dset = f["exploration_results"]
        
        results = np.zeros(dset.shape, dtype=np.float64)
        results[:,:] = dset
        
    return results
        
    


