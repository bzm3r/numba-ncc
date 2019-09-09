# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:51:52 2017

@author: Brian
"""

import numpy as np
import matplotlib.pyplot as plt

# y = np.exp(-1*t/p)
# np.log(y) = -1*t/p
# np.log(y) = -1*t/p

p = 3.0
x = np.linspace(0.0, 3.0)
y = np.exp(-1 * x / p)
A = np.zeros((x.shape[0], 2), dtype=np.float64)
A[:, 0] = x
est_p = -1.0 / (np.linalg.lstsq(A, np.log(y))[0][0])
