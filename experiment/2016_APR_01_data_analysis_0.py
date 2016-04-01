# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 10:42:37 2016

@author: Brian
"""

from __future__ import division
import numpy as np
import general.exec_utils as exec_utils

ANALYSIS_NUMBER = 0
analysis_description = "Putting together "

BASE_OUTPUT_DIR = "A:\\numba-ncc\\output\\"
DATE_STR = "2016_APR_01"

analysis_dir = analysis_exec.get_analysis_directory_path(BASE_OUTPUT_DIR, DATE_STR, ANALYSIS_NUMBER)

relevant_environment_info = [("2016_MAR_28", 0, [0, 1, 2])]

environment_dirs = get_environment_dirs_given_relevant_environment_info(relevant_environment_info)