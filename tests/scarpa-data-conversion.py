# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:40:20 2017

@author: Brian
"""

import numpy as np

pixels_per_micrometer = 5.
pixels_per_minute = 14.

data_points_in_pixels = np.array([[37,164], [115,98], [190,83], [264,104], [338,146], [413,207], [492,247], [565,313]], dtype=np.float64)

xs = np.round(data_points_in_pixels[:,0]/pixels_per_minute, decimals=0)
ys = np.round(data_points_in_pixels[:,1]/pixels_per_micrometer + 25., decimals=0)