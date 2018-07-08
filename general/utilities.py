# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07 17:47:39 2015

@author: Brian
"""

import numpy as np
import numba as nb

# ==============================================================
def calculate_normalized_randomization_factors(size):
    rfs = np.random.random(size)
    return rfs/np.sum(rfs)

# ===============================================================

def is_numeric(s):
    try:
        float(s)
        return True
        
    except ValueError:
        return False
        
# ==============================================================
        
def chunkify(chunksize, input_array):
    arraysize = len(input_array)
    chunks = []
    
    if chunksize < arraysize:
        num_full_chunks = int(np.floor(arraysize/chunksize))
        
        for k in range(num_full_chunks + 1):
            chunk = input_array[k*chunksize:(k + 1)*chunksize]
            
            if len(chunk) > 0:
                chunks.append(chunk)
        
        if np.sum([len(x) for x in chunks]) == arraysize:
            return chunks
        else:
            raise Exception("failed to chunkify!")
        #assert(np.sum([len(x) for x in chunks]) == init_input_len)
    else:
        chunks.append(input_array)
        return chunks
    
# ==============================================================

@nb.jit(nopython=True)
def numba_arange(start, stop):
    result = np.empty(stop-start, dtype=np.int64)
    for i in range(start, stop):
        result[i] = i
        
    return result
    
# ==============================================================
    
@nb.jit(nopython=True)
def copy_1D_array(num_elements, array):
    new_array = np.empty(num_elements, dtype=np.float64)
    
    for i in range(num_elements):
        new_array[i] = array[i]
    
    return new_array
    
# ==============================================================
    
@nb.jit(nopython=True)
def insertion_sort(len_array, unsorted_array, unsorted_array_indices):
    for index in range(1, len_array):
    
        currentvalue = unsorted_array[index]
        current_index = unsorted_array_indices[index]
        
        position = index
        
        while position > 0 and unsorted_array[position-1] > currentvalue:
            unsorted_array[position] = unsorted_array[position-1]
            unsorted_array_indices[position] = unsorted_array_indices[position-1]
            position = position-1
            
            unsorted_array[position] = currentvalue
            unsorted_array_indices[position] = current_index
            
# ===============================================================

@nb.jit(nopython=True)
def make_node_coords_array_given_xs_and_ys(num_nodes, xs, ys):
    node_coords = np.empty((num_nodes, 2), dtype=np.float64)
    
    for i in range(num_nodes):
        node_coord = node_coords[i]
        node_coord[0] = xs[i]
        node_coord[1] = ys[i]
        
    return node_coords
