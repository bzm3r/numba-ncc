# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:47:52 2017

@author: Brian
"""

import os
import parse

experiment_set_dir = "C:\\Users\\bhmer\\Desktop\\numba-ncc\\output2019_FEB_26\\SET=1"
experiment_set_dir_contents = os.listdir(experiment_set_dir)

experiment_set_dir_subdirs = [
    x
    for x in experiment_set_dir_contents
    if os.path.isdir(os.path.join(experiment_set_dir, x))
]

for subdir in experiment_set_dir_subdirs:
    subdir_contents = os.listdir(os.path.join(experiment_set_dir, subdir))
    chemotaxis_data_fp = os.path.join(
        experiment_set_dir, subdir, "chemotaxis_success_per_repeat.np.npy"
    )
    if os.path.isfile(chemotaxis_data_fp):
        os.rename(
            chemotaxis_data_fp,
            os.path.join(
                experiment_set_dir, subdir, "chemotaxis_success_per_repeat.np"
            ),
        )
