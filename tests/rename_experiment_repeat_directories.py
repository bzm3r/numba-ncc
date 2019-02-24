# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:47:52 2017

@author: Brian
"""

import os
import parse

experiment_set_dir = "B:\\numba-ncc\\output\\2019_JAN_20\\SET=0"
experiment_set_dir_contents = os.listdir(experiment_set_dir)

experiment_set_dir_subdirs = [x for x in experiment_set_dir_contents if os.path.isdir(os.path.join(experiment_set_dir, x))]

parse_compilation = parse.compile("ch_{}_rand-{}_{}")

for old_dir_name in experiment_set_dir_subdirs:
    sen, randscheme, other = parse_compilation.parse(old_dir_name)
    new_dir_name = "ch_{}_{}-rand-{}".format(sen, other, randscheme)
    
    current_experiment_dir = os.path.join(experiment_set_dir, old_dir_name)
    new_experiment_dir = os.path.join(experiment_set_dir, new_dir_name)
    
    os.rename(current_experiment_dir, new_experiment_dir)
