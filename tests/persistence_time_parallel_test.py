# -*- coding: utf-8 -*-
"""
Created on Tue May 09 17:00:45 2017

@author: Brian
"""

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scipio
import threading


@nb.jit(nopython=True)
def calculate_cos_theta_for_direction_autocorr_coeffs(a, b):
    ax, ay = a
    bx, by = b
    
    norm_a = np.sqrt(ax*ax + ay*ay)
    norm_b = np.sqrt(bx*bx + by*by)
    
    if norm_a < 1e-6:
        a = np.random.rand(2)
        ax, ay = a
        norm_a = np.sqrt(ax*ax + ay*ay)
        
        
    if norm_b < 1e-6:
        b = np.random.rand(2)
        bx, by = b
        norm_b = np.sqrt(bx*bx + by*by)

    ax_, ay_ = a/norm_a
    bx_, by_ = b/norm_b
    
    return ax_*bx_ + ay_*by_

@nb.jit(nopython=True)
def calculate_direction_autocorr_coeffs_for_persistence_time(displacements):
    N = displacements.shape[0]
    
    all_das = np.zeros(N, dtype=np.float64)
    first_negative_n = -1
    
    for n in range(N):
        sum_cos_thetas = 0.0
        m = 0.0
        
        i = 0
        while i + n < N:
            cos_theta = calculate_cos_theta_for_direction_autocorr_coeffs(displacements[i], displacements[i + n])
            sum_cos_thetas += cos_theta
            m += 1
            i += 1
        
        da = (1./m)*sum_cos_thetas 
        if da < 0.0 and first_negative_n == -1:
            first_negative_n = n
            break
        
        all_das[n] = da
        
    if first_negative_n == -1:
        first_negative_n = N
        
    return all_das[:first_negative_n]

@nb.jit(nopython=True, nogil=True)
def calculate_direction_autocorr_coeff_parallel_worker(N, ns, dacs, displacements):
    
    for n in ns:
        m = 0.0
        sum_cos_thetas = 0.0
        i = 0 
        
        while i + n < N:
            cos_theta = calculate_cos_theta_for_direction_autocorr_coeffs(displacements[i], displacements[i + n])
            sum_cos_thetas += cos_theta
            m += 1
            i += 1
        
        dacs[n] = (1./m)*sum_cos_thetas

@nb.jit(nopython=True)
def find_first_negative_n(dacs):
    for n, dac in enumerate(dacs):
        if n < 0.0:
            return n
    
def calculate_direction_autocorr_coeffs_for_persistence_time_parallel(displacements, num_threads=4):
    N = displacements.shape[0]
    dacs = np.empty(N, dtype=np.float64)
     
    task_indices = np.arange(N)
    chunklen = (N + num_threads - 1)//num_threads
    
    chunks = []
    for i in range(num_threads):
        chunk = [N, task_indices[i*chunklen:(i + 1)*chunklen], dacs, displacements]
        chunks.append(chunk)
            
    threads = [threading.Thread(target=calculate_direction_autocorr_coeff_parallel_worker, args=c) for c in chunks]
    
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    first_negative_n = find_first_negative_n(dacs)
    return dacs[:first_negative_n]

def estimate_persistence_time(timestep, positive_das):
    ts = np.arange(positive_das.shape[0])*timestep
#    A = np.zeros((ts.shape[0], 2), dtype=np.float64)
#    A[:, 0] = ts
#    pt = -1./(np.linalg.lstsq(A, np.log(positive_das))[0][0])
    try:
        popt, pcov = scipio.curve_fit(lambda t, pt: np.exp(-1.*t/pt), ts, positive_das)
        pt = popt[0]
    except:
        pt = np.nan
    
    return pt, ts
            

@nb.jit(nopython=True)
def generate_displacements(persistence_time, num_displacements, timestep_bw_displacements, avg_displacement_magnitude):
    displacements = np.zeros((num_displacements, 2), dtype=np.float64)
    
    theta = np.random.rand()*2*np.pi
    delta_t = 0.0
    for n in range(num_displacements):
        delta_t += timestep_bw_displacements
        this_step_size = avg_displacement_magnitude
        displacements[n][0] = this_step_size*np.cos(theta)
        displacements[n][1] = this_step_size*np.sin(theta)
        
        if delta_t > persistence_time:
            delta_theta = (np.random.rand()*np.pi)
            if np.random.rand() < 0.5:
                delta_theta = -1*delta_theta
            theta = (theta + delta_theta)%(2*np.pi)
            delta_t = 0.0

    return displacements

@nb.jit(nopython=True)
def calculate_positions(displacements):
    num_displacements = displacements.shape[0]
    positions = np.zeros((num_displacements + 1, 2), dtype=np.float64)
    
    for x in range(num_displacements):
        prev_pos = positions[x]
        positions[x + 1] = prev_pos + displacements[x]
        
    return positions

        
persistence_time = 40.0
timestep_bw_displacements = (2./60.0)
avg_displacement_magnitude = 3.*timestep_bw_displacements
num_displacements = 10000

displacements = generate_displacements(persistence_time, num_displacements, timestep_bw_displacements, avg_displacement_magnitude)
positions = calculate_positions(displacements)
positive_das = calculate_direction_autocorr_coeffs_for_persistence_time(displacements)
pt, ts = estimate_persistence_time(timestep_bw_displacements, positive_das)

fig_traj, ax_traj = plt.subplots()
ax_traj.plot(positions[:,0], positions[:,1])
ax_traj.set_aspect('equal')
max_data_lim = 1.1*np.max(np.abs(positions))
ax_traj.set_xlim(-1*max_data_lim, 1*max_data_lim)
ax_traj.set_ylim(-1*max_data_lim, 1*max_data_lim)


fig_das, ax_das = plt.subplots()
all_ts = np.arange(displacements.shape[0])*timestep_bw_displacements

#ax_das.plot(all_ts, all_das, color='b', marker='.', ls='')

fig_fit, ax_fit = plt.subplots()
ax_fit.plot(ts, positive_das, color='g', marker='.', ls='')
ax_fit.plot(ts, [np.exp(-1*x/pt) for x in ts], color='r', marker='.', ls='')

plt.show()
print "pt: ", pt




