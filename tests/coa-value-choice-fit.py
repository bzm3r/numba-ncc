# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 16:33:42 2017

@author: Brian
"""

import matplotlib.pyplot as plt
import numpy as np


coa_dict = {49: 8.0, 36: 10.0, 25: 12.0, 16: 14.0, 9: 16.0, 4: 24.0}

cell_numbers = sorted(coa_dict.keys())

plt.plot(
    [np.sqrt(k) for k in cell_numbers],
    [coa_dict[k] for k in cell_numbers],
    ls="",
    marker=".",
)
