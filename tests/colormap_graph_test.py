# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:20:02 2017

@author: Brian
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# normal distribution center at x=0 and y=5
xs = [2, 4, 9, 16, 25, 49]
ys = [2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]

#data = 1.0 + 2.5*np.random.rand(len(xs), len(ys))
#data = np.where(data > 3.0, 3.0, data)

data = np.array([[ 1.13708228,  1.15775943,  1.09909963,  1.07444652,  1.15186194,
         1.24467625,  1.08018951,  1.09650866,  1.10843243],
       [ 1.4868342 ,  1.23499967,  1.27789272,  1.33407617,  1.21617012,
         1.35789445,  3.0,  1.4388506 ,  1.45982506],
       [ 1.2299272 ,  1.15364873,  1.16565969,  1.14922818,  1.21443923,
         1.25054165,  1.21660854,  1.22005011,  1.21774616],
       [ 1.138349  ,  1.10882325,  1.10547519,  1.16245244,  1.15801007,
         1.12624472,  1.16528356,  1.18256013,  1.19088628],
       [ 1.10217415,  1.08730538,  1.09861504,  1.10073579,  1.13955315,
         1.08479129,  1.11631074,  1.11863382,  1.14388193],
       [ 1.06222266,  1.07525101,  1.05918822,  1.07566021,  1.08318263,
         1.08461649,  1.08750276,  1.08947182,  1.10528072]])

def graph_coa_variation_test_data(sub_experiment_number, num_cells_to_test, test_coas, average_cell_group_area_data, save_dir=None, max_normalized_group_area=3.0):
    
    fig, ax = plt.subplots()
    
    cax = ax.imshow(average_cell_group_area_data, interpolation='none', cmap=plt.get_cmap('gist_heat_r'))
    ax.set_yticks(np.arange(len(num_cells_to_test)))
    ax.set_xticks(np.arange(len(test_coas)))
    ax.set_yticklabels(num_cells_to_test)
    ax.set_xticklabels(test_coas)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    upper_lim = np.min([np.max(average_cell_group_area_data), max_normalized_group_area])
    cbar = fig.colorbar(cax, boundaries=np.linspace(1.0, upper_lim, num=100), ticks=np.linspace(1.0, upper_lim, num=5))

    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    

    if save_dir == None:
        plt.show()
    else:
        fig.set_size_inches(12, 8)
        save_path = os.path.join(save_dir, "coa_variation_results_{}".format(sub_experiment_number) + ".png")
        print "save_path: ", save_path
        fig.savefig(save_path, forward=True)
        plt.close(fig)
        plt.close("all")

graph_coa_variation_test_data(0, xs, ys, data)
                              
#data[0, 0] = np.nan
#
#data_dict = {}
#
#for xi, x in enumerate(xs):
#    for yi, y in enumerate(ys):
#        data_dict[(x, y)] = data[xi, yi]
#
## Make plot with vertical (default) colorbar
#fig, ax = plt.subplots()
#
#
#cax = ax.imshow(data, interpolation='none', cmap=plt.get_cmap('gist_heat_r'))
#ax.set_yticks(np.arange(len(xs)))
#ax.set_xticks(np.arange(len(ys)))
#ax.set_yticklabels(xs)
#ax.set_xticklabels(ys)
## Add colorbar, make sure to specify tick locations to match desired ticklabels
#cbar = fig.colorbar(cax, boundaries=np.linspace(1.0, 3.0, num=100), ticks=np.linspace(1.0, 3.0, num=5))
#
#
#plt.show()



