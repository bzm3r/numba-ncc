# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:19:27 2017

@author: Brian
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# force square figure and square axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

N = 20
theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
bars = ax.bar(theta, radii, width=width, bottom=0.0)
for r, bar in zip(radii, bars):
    bar.set_facecolor(cm.jet(r / 10.0))
    bar.set_alpha(0.5)


plt.show()
