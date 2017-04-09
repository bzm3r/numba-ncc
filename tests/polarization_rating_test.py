# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 14:39:56 2017

@author: Brian
"""
from __future__ import division
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

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
        if normalized_rac[i] > 0.25:
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
    
    significant_fronts = np.zeros(num_rac_fronts, dtype=np.int64)
    for fi in range(num_rac_fronts):
        if normalized_front_strengths[fi] > 0.25:
            significant_fronts[fi] = 1
            
    num_significant_fronts = np.sum(significant_fronts)
    relevant_front_strengths = np.zeros(num_significant_fronts, dtype=np.float64)
    relevant_front_starts = np.zeros(num_significant_fronts, dtype=np.int64)
    relevant_front_widths = np.zeros(num_significant_fronts, dtype=np.int64)
    
    rfi = 0
    for fi in range(num_rac_fronts):
        if significant_fronts[fi] == 1:
            relevant_front_strengths[rfi] = normalized_front_strengths[fi]
            relevant_front_starts[rfi] = front_starts[fi]
            relevant_front_widths[rfi] = front_widths[fi]
            rfi += 1
            
    front_strengths = np.zeros(num_rac_fronts)
    
    rho_amount_penalizer = 1.0
    if sum_rho > 0.4:
        rho_amount_penalizer = np.exp((sum_rho - 0.4)*np.log(0.1)/0.1)
    rac_amount_penalizer = 1.0
    if sum_rac > 0.5:
        rac_amount_penalizer = np.exp((sum_rac - 0.4)*np.log(0.1)/0.1)
    
    if num_significant_fronts == 1:
        front_width_rating = relevant_front_widths[0]/(num_nodes/3.)
        if front_width_rating > 2.0:
            front_width_rating = 0.0
        elif front_width_rating > 1.0:
            front_width_rating = front_width_rating - 1.0

        return (front_width_rating)*rho_amount_penalizer*rac_amount_penalizer
    
    elif num_significant_fronts > 1:
        worst_distance_between_fronts = (num_nodes -np.sum(relevant_front_widths))/float(num_significant_fronts)
        distance_between_fronts_score = np.zeros(num_significant_fronts, dtype=np.float64)
        
        for fi in range(num_significant_fronts):
            this_si = relevant_front_starts[fi]
            this_width = relevant_front_widths[fi]
            fi_plus1 = (fi + 1)%(num_significant_fronts)
            next_si = relevant_front_starts[fi_plus1]
            if next_si < this_si:
                next_si = num_nodes + next_si
            dist_bw_fronts = next_si - (this_si + this_width)
            score = dist_bw_fronts/worst_distance_between_fronts
            if score > 1.0:
                distance_between_fronts_score[fi] = 1.0
            else:
                distance_between_fronts_score[fi] = score
        
        combined_dist_bw_fronts_score = 1.0
        if num_significant_fronts == 2:
            combined_dist_bw_fronts_score = 1.0 - np.min(distance_between_fronts_score)
        else:
            combined_dist_bw_fronts_score = 1.0 - np.sum(distance_between_fronts_score)/num_significant_fronts
        average_front_width = (np.sum(relevant_front_widths) + 0.0)/num_significant_fronts
        front_width_rating = average_front_width/(num_nodes/3.)
        
        if front_width_rating > 2.0:
            front_width_rating = 0.0
        elif front_width_rating > 1.0:
            front_width_rating = front_width_rating - 1.0
            
        return combined_dist_bw_fronts_score*front_width_rating*rho_amount_penalizer*rac_amount_penalizer
        
    return 0.0

def generate_rgtpase(num_nodes, num_fronts=2):
    front_widths_a = np.ceil(np.random.rand(num_fronts)*(num_nodes/float(num_fronts)))
    front_widths_b = np.zeros_like(front_widths_a, dtype=np.float64)
    total_front_widths = np.sum(front_widths_a)
    unoccupied_width = (num_nodes - total_front_widths)
    
    is_front = np.zeros(num_nodes, dtype=np.int64)
    num_fronts_placed = 0
    i = 0
    fronts_a = np.zeros((int(num_fronts), 2), dtype=np.int64)
    while i < num_nodes and num_fronts_placed < num_fronts:
        if is_front[i] == 0:
            if np.random.rand() < unoccupied_width/float(num_nodes):
                i += 1
                if unoccupied_width == 0.0:
                    continue
                else:
                    unoccupied_width -= 1.0
            else:
                front_width = int(front_widths_a[num_fronts_placed])
                is_front[i:i + front_width] = np.ones(front_width, dtype=np.int64)
                fronts_a[num_fronts_placed][0] = i
                fronts_a[num_fronts_placed][1] = i + front_width - 1
                
                num_fronts_placed += 1
                i += front_width
        else:
            i += 1
            
    fronts_b = np.zeros((int(num_fronts), 2), dtype=np.int64)
    if num_fronts > 1:
        for i, index_pair in enumerate(fronts_a):
            im1 = (i - 1)%int(num_fronts)
            s = (fronts_a[im1][1] + 1)%num_nodes
            e = (fronts_a[i][0] - 1)%num_nodes
            
            if e < s:
                e = num_nodes + e
                
            fronts_b[i][0] = s
            fronts_b[i][1] = e
            
            front_widths_b[i] = np.abs(fronts_b[i][1] - fronts_b[i][0]) + 1
    else:
        s = (fronts_a[0][1] + 1)%num_nodes
        e = (fronts_a[0][0] - 1)%num_nodes
        
        if e < s:
            e = num_nodes + e
            
        fronts_b[0][0] = s
        fronts_b[0][1] = e
        
        front_widths_b[0] = np.abs(fronts_b[0][1] - fronts_b[0][0]) + 1 
    
    rgtpase_a = np.zeros(num_nodes, dtype=np.float64)
    rgtpase_b = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_fronts):
        s, e = fronts_a[i]
        centre = int((s + e)/2.)
        width = front_widths_a[i]*0.2
        
        if width > 0:
            width_squared = width**2
            
            for x_ in range(s, e + 1):
                x = x_%num_nodes
                if x > centre:
                    y = x - centre
                else:
                    y = centre - x
                    
                rgtpase_a[x] = np.exp(-1*(y**2)/(2*width_squared))
        
            
        s, e = fronts_b[i]
        centre = int((s + e)/2.)
        width = front_widths_b[i]*0.2
        
        if width > 0:
            width_squared = width**2
            
            for x_ in range(s, e + 1):
                x = x_%num_nodes
                if x > centre:
                    y = x - centre
                else:
                    y = centre - x
                    
                rgtpase_b[x] = np.exp(-1*(y**2)/(2*width_squared))
        
    return rgtpase_a, rgtpase_b 

rac_membrane_actives = np.array([0.05625, 0.05625, 0.05625, 0.05625, 0.05625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
rho_membrane_actives = np.ones(16, dtype=np.float64)*(0.1/16.)

print "very good: ", calculate_polarization_rating(rac_membrane_actives, rho_membrane_actives, 16)

rac_membrane_actives = np.array([0.06*0.5, 0.06*0.5, 0.06*0.5, 0.06*0.5, 0.0, 0.0, 0.0, 0.0, 0.06*0.5, 0.06*0.5, 0.06*0.5, 0.06*0.5, 0.0, 0.0, 0.0, 0.0])
rho_membrane_actives = np.ones(16, dtype=np.float64)*(0.1/16.)

print "very bad: ", calculate_polarization_rating(rac_membrane_actives, rho_membrane_actives, 16)

rac_membrane_actives = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0])
rho_membrane_actives = np.ones(16, dtype=np.float64)*(0.1/16.)

print "very bad again: ", calculate_polarization_rating(rac_membrane_actives, rho_membrane_actives, 16)

rac_membrane_actives = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
rho_membrane_actives = np.ones(16, dtype=np.float64)*(0.1/16.)

print "in between: ", calculate_polarization_rating(rac_membrane_actives, rho_membrane_actives, 16)


