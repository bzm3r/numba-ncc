# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:47:52 2017

@author: Brian
"""

import os
import parse

experiment_set_dir = os.getcwd()
experiment_set_dir_contents = os.listdir(experiment_set_dir)

experiment_set_dir_subdirs = [x for x in experiment_set_dir_contents if os.path.isdir(x)]

for x in experiment_set_dir_subdirs:
    current_experiment_dir = os.path.join(experiment_set_dir, x)
    experiment_dir_contents = os.listdir(current_experiment_dir)
    
    for y in experiment_dir_contents:
        if os.path.isdir(y):
            parse_result = parse.parse("{}RPT={}")
            if parse_result != None:
                repeat_number = parse_result[1]
                os.rename(os.path.join(current_experiment_dir, y), os.path.join(current_experiment_dir, "RPT={}".format(repeat_number)))