# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 19:44:29 2016

@author: Brian
"""

from matplotlib import pyplot as plt
import numpy as np

test_strains = np.linspace(0, 1, num=50)

tension_mediated_rac_inhibition_half_strain = 0.2

for tension_fn_type in [6, 7]:
    if tension_fn_type == 5:
        exponent = np.log(2) / tension_mediated_rac_inhibition_half_strain
        strain_inhibitions = np.exp(exponent * test_strains)
        plt.plot(test_strains, strain_inhibitions, label="exp")
    elif tension_fn_type == 6:
        exponent = 2
        constant = (2.0 - 1.0) / (
            tension_mediated_rac_inhibition_half_strain ** exponent
        )
        strain_inhibitions = constant * (test_strains ** exponent) + 1.0
        plt.plot(test_strains, strain_inhibitions, label="2")
    elif tension_fn_type == 7:
        exponent = 3
        constant = (2.0 - 1.0) / (
            tension_mediated_rac_inhibition_half_strain ** exponent
        )
        strain_inhibitions = constant * (test_strains ** exponent) + 1.0
        plt.plot(test_strains, strain_inhibitions, label="3")
    elif tension_fn_type == 8:
        exponent = 4
        constant = (2.0 - 1.0) / (
            tension_mediated_rac_inhibition_half_strain ** exponent
        )
        strain_inhibitions = constant * (test_strains ** exponent) + 1.0
        plt.plot(test_strains, strain_inhibitions, label="4")

plt.legend()
