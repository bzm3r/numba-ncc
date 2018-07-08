# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:05:55 2017

@author: Brian
"""


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots()

# generate some random test data
num_points = 10
cell_persistence_data = [np.random.rand(num_points) for n in range(4)]

# plot violin plot
violin = ax.violinplot(cell_persistence_data,
                   showmeans=True,
                   showmedians=True, points=num_points)
violin['cbars'].set_color('g')
violin['cmins'].set_color('g')
violin['cmaxes'].set_color('g')
violin['cmeans'].set_color('r')
violin['cmedians'].set_color('g')
[x.set_color('g') for x in violin['bodies']]

# generate some random test data
num_points = 10
cell_persistence_data = [np.random.rand(num_points) for n in range(4)]

# plot violin plot
violin = ax.violinplot(cell_persistence_data,
                   showmedians=True, points=num_points)
violin['cbars'].set_color('r')
violin['cmins'].set_color('r')
violin['cmaxes'].set_color('r')
violin['cmedians'].set_color('k')
[x.set_color('r') for x in violin['bodies']]


## plot box plot
#axes[1].boxplot(all_data)
#axes[1].set_title('box plot')

# adding horizontal grid lines
ax.yaxis.grid(True)
ax.set_xticks([y+1 for y in range(len(cell_persistence_data))])
ax.set_xlabel('COA')
ax.set_ylabel('persistence')



box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
black_patch = mpatches.Patch(color='k', label='CIL=120')
ax.legend(handles=[black_patch], loc='center left', bbox_to_anchor=(1, 0.5))




# add x-tick labels
plt.setp(ax, xticks=[y+1 for y in range(len(cell_persistence_data))],
         xticklabels=['4', '8', '12', '16'])
plt.show()