# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:02:20 2019

@author: Brian
"""

import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt

#data = np.random.rand(1, 5760000)
data = np.random.rand(1, 3)
#avg = np.average(data)
#fq, median, tq = np.quantile(data, [0.25, 0.5, 0.75])
#dmin, dmax = np.min(data), np.max(data)
#
#manual_stats = dict(zip(['label', 'mean', 'iqr', 'cilo', 'cihi', 'whishi', 'whislo', 'fliers', 'q1', 'med', 'q3'], ["test", avg, tq - fq, None, None, None, tq, median, fq]))
#
#stats = cbook.boxplot_stats(data)

fig, ax = plt.subplots()
ax.boxplot(data)

plt.show()