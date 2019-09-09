# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:42:04 2019

@author: Brian
"""

import numpy as np
import matplotlib.pyplot as plt

def determine_next_phase_change(currently_in_persistent_phase, persistent_phase_period, random_phase_period):
    next_phase_change = 0.0
    if currently_in_persistent_phase:
        next_phase_change = persistent_phase_period*(1.0 + np.random.normal(scale=0.25)) 
    else:
        next_phase_change = random_phase_period*(1.0 + np.random.normal(scale=0.25)) 
    if next_phase_change < 0.0:
        next_phase_change = 0.0
        
    return next_phase_change
    
def generate_cell_directions(num_timepoints, persistent_phase_period, random_phase_period):
    directions = np.empty(num_timepoints, dtype=np.float64)
    phases = np.empty(num_timepoints, dtype=np.bool)
    
    directions[0] = np.random.rand()
    
    total_period = random_phase_period + persistent_phase_period
    phases[0] = np.random.rand() < (persistent_phase_period/total_period)
    
    next_phase_change = np.random.rand()*determine_next_phase_change(phases[0], persistent_phase_period, random_phase_period)
    print("next phase change: ", next_phase_change)
    for i in range(1, num_timepoints):
        if i > next_phase_change:
            new_phase = not phases[i - 1]
            next_phase_change = i + determine_next_phase_change(new_phase, persistent_phase_period, random_phase_period)
            print("next phase change: ", next_phase_change)
            phases[i] = new_phase
        else:
            phases[i] = phases[i - 1]
    
        if phases[i]:
            directions[i] = (directions[i - 1] + np.random.normal(scale=0.01))%1.0
        else:
            directions[i] = (directions[i - 1] + np.random.normal(scale=0.25))%1.0
            
    return (directions, phases)

def generate_protrusions(num_nodes, directions):
    protrusion_existence_per_timepoint = np.zeros((directions.shape[0], num_nodes), dtype=np.bool)
    
    spacing_between_nodes = (1.0/num_nodes)
    node_locations = np.arange(num_nodes)*spacing_between_nodes
    num_nodes_in_protrusion = 6
    protrusion_front_size = (1.0/num_nodes)*num_nodes_in_protrusion
    
    for i in range(directions.shape[0]):
        leftmost_direction = (directions - 0.5*protrusion_front_size)%1.0
        
        for i in range(num_nodes_in_protrusion):
            
            
        
        
        

persistent_phase_period = 20.0
random_phase_period = 5.0
simulation_length = 500.0
num_timepoints = int(simulation_length)
timepoints = np.arange(num_timepoints)
num_nodes = 16

directions, phases = generate_cell_directions(num_timepoints, persistent_phase_period, random_phase_period)

#plt.plot(timepoints, directions, 'k.')
#plt.plot(timepoints, [1.2 if phases[i] else np.nan for i in timepoints], 'g.')
#plt.plot(timepoints, [1.2 if not phases[i] else np.nan for i in timepoints], 'r.')






