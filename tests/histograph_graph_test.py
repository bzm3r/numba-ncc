# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:20:02 2017

@author: Brian
"""


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import numpy as np

# normal distribution center at x=0 and y=5
xs = np.arange(0, 5)
ys = np.linspace(0, 1, num=5)

data = np.random.rand(xs.shape[0], ys.shape[0])

data_dict = {}

for xi, x in enumerate(xs):
    for yi, y in enumerate(ys):
        data_dict[(x, y)] = data[xi, yi]

# Make plot with vertical (default) colorbar
fig, ax = plt.subplots()


cax = ax.imshow(data, interpolation="none", cmap=cm.coolwarm)
ax.set_xticks(np.arange(len(xs)))
ax.set_yticks(np.arange(len(ys)))
ax.set_xticklabels(xs)
ax.set_yticklabels(ys)
# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(
    cax, boundaries=np.linspace(0, 1, num=100), ticks=np.linspace(0, 1, num=5)
)


plt.show()
