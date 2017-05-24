# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:58:34 2017

@author: Brian
"""

import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
timepoints = np.arange(num_points)
velocities = np.random.choice([-1, 1], size=num_points-1)*np.random.rand(num_points - 1)
positions = np.zeros(num_points, dtype=np.float64)
for n in range(num_points - 1):
    positions[n + 1] = positions[n] + velocities[n]


max_value = np.max(np.abs(positions))
min_positions = positions - 0.5*(1.0 + np.random.rand(num_points))*max_value
max_positions = positions + 0.5*(1.0 + np.random.rand(num_points))*max_value

    
fig, ax = plt.subplots()

ax.plot(timepoints, positions, color='b')
ax.plot(timepoints, min_positions, color='r', alpha=0.2)
ax.plot(timepoints, max_positions, color='g', alpha=0.2)

step_size = 100

bar_timepoints = np.append(timepoints[::step_size], timepoints[-1])
bar_positions = np.append(positions[::step_size], positions[-1])
bar_min_positions = np.append(min_positions[::step_size], min_positions[-1])
bar_max_positions = np.append(max_positions[::step_size], max_positions[-1])
lower_bounds = np.abs(bar_positions - bar_min_positions)
upper_bounds = np.abs(bar_positions - bar_max_positions)
ax.errorbar(bar_timepoints, bar_positions, yerr=[lower_bounds, upper_bounds], color='b', ls='', capsize=2)
