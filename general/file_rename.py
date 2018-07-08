# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:10:19 2017

@author: Brian
"""

'''
Sometimes, many experiment files need to be renamed because of a change in how the experiment naming scheme works. 
'''

import os
import parse
import general.exec_utils as geu
import dill

def rename_experiment(sub_dir, current_dir, original_st, final_st, trial_run=False):
    current_sub_dir = os.path.join(current_dir, sub_dir)
    
    print("-----")
    print(("in sub dir: ", sub_dir))
    
    parse_result_obj = parse.parse(original_st, sub_dir)
    
    if parse_result_obj == None:
        raise Exception("parse result object was None!")
        
    sub_experiment_index, randomization_tag, total_num_cells, group_width, group_height, group_height, box_y_placement_factor, coa, cil, repeat_number = parse_result_obj.fixed
    
    new_sub_dir = final_st.format(sub_experiment_index, randomization_tag, total_num_cells, group_width, group_height, group_height, 0.0, coa, cil, repeat_number)
    
    new_dir = os.path.join(current_dir, new_sub_dir)
    
    empty_env_pickle_path = os.path.join(current_sub_dir, "environment.pkl")
    this_env = geu.load_empty_env(empty_env_pickle_path)
    
    if not trial_run:
        this_env.environment_dir = new_dir
        this_env.empty_self_pickle_path = os.path.join(new_dir, "environment.pkl")
        this_env.storefile_path = os.path.join(new_dir, "store.hdf5")
        os.rename(current_sub_dir, new_dir)
        
        with open(this_env.empty_self_pickle_path, 'wb') as f:
            dill.dump(this_env, f)
    else:
        
        print(("Environment in folder: ", new_dir))
        print(("Would update environment_dir to: ", new_dir))
        print(("Would update empty_self_pickle_path to: ", os.path.join(new_dir, "environment.pkl")))
        print(("Would update storefile_path to: ", os.path.join(current_sub_dir, "store.hdf5")))
        print(("Would rename this dir to: ", new_dir))
    
    print("-----")
    return

    
original_string_template = "cm_{}_{}_NC=({}, {}, {}, {}, {})_COA={}_CIL={}"
num_expected_blanks = len((parse.parse(original_string_template, original_string_template)).fixed)
final_string_template = "cm_{}_{}_NC=({}, {}, {}, {}, {})_COA={}_CIL={}"
    
    
folder_to_scan = "A:\\numba-ncc\\output\\2017_MAY_19\\SET=0"

sub_folders = os.listdir(folder_to_scan)

trial_run = False

for folder in sub_folders:
    parse_result_obj = parse.parse(original_string_template, folder)
    
    print(("processing: ", folder))
    if parse_result_obj != None:
        parse_result = parse_result_obj.fixed
        
        current_dir = os.path.join(folder_to_scan, folder)
        current_dir_items = os.listdir(current_dir)

        sub_experiment_index, randomization_tag, total_num_cells, group_width, group_height, group_height, box_y_placement_factor, coa, cil = parse_result 
        new_dir_name = final_string_template.format(sub_experiment_index, randomization_tag, total_num_cells, group_width, group_height, group_height, box_y_placement_factor, coa, cil)
        
        if not trial_run:
            new_dir = os.path.join(folder_to_scan, new_dir_name)
            os.rename(current_dir, new_dir)
            current_dir = new_dir
        else:
            print(("Would rename current directory to: ", os.path.join(folder_to_scan, new_dir_name)))
        
        for dir_item in current_dir_items:
            if not os.path.isdir(os.path.join(current_dir, dir_item)):
                continue
            
            sub_dir = dir_item
            
            rename_experiment(sub_dir, current_dir, original_string_template + "_RPT={}", final_string_template + "_RPT={}", trial_run=trial_run)
            
            current_name = os.path.join(current_dir, sub_dir)
            new_name = os.path.join(current_dir, sub_dir)
        
        print("==================")
    else:
        print("skipping!")
        print("==================")
        continue


    

