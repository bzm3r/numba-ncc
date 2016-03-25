# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 16:53:39 2015

@author: Brian
"""

from __future__ import division
import core.parameterorg as parameterorg
import numpy as np
import visualization.datavis as datavis
import os
import time
import copy
import multiprocessing as mp
import shutil
import gzip
import dill as pickling_package

def convert_parameter_override_dictionary_into_keyvalue_tuple_list(base_parameter_override_dict, other_parameter_override_dict):
    labels_in_base_dict = base_parameter_override_dict.keys()
    labels_in_other_dict = other_parameter_override_dict.keys()
    
    keyvalue_tuple_list = []    
    for label in labels_in_base_dict:
        if label in labels_in_other_dict:
            orig_value = base_parameter_override_dict[label]
            new_value = other_parameter_override_dict[label]
            if  orig_value != new_value:
                keyvalue_tuple_list.append((label, orig_value, new_value))
            
    return keyvalue_tuple_list
            
def make_space_polygons(make_migr_space_poly, make_phys_space_poly, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset, physical_bdry_polygon_extra=5):
    
    migr_space_poly = np.array([])
    phys_space_poly = np.array([])
    
    if make_migr_space_poly == True:
        migr_space_poly = np.array([[0 + corridor_x_offset, 0 + corridor_y_offset], [width_corridor + corridor_x_offset, 0 + corridor_y_offset], [width_corridor + corridor_x_offset, height_corridor + corridor_y_offset], [0 + corridor_x_offset, height_corridor + corridor_y_offset]], dtype=np.float64)*1e-6
        
        if make_phys_space_poly == True:
            phys_space_poly = np.array([[0 + corridor_x_offset - physical_bdry_polygon_extra, 0 + corridor_y_offset- physical_bdry_polygon_extra], [width_corridor + corridor_x_offset + physical_bdry_polygon_extra, 0 + corridor_y_offset - physical_bdry_polygon_extra], [width_corridor + corridor_x_offset + physical_bdry_polygon_extra, height_corridor + corridor_y_offset + physical_bdry_polygon_extra], [0 + corridor_x_offset - physical_bdry_polygon_extra, height_corridor + corridor_y_offset + physical_bdry_polygon_extra]], dtype=np.float64)*1e-6
    else:
        if make_phys_space_poly == True:
            phys_space_poly = np.array([[0 + corridor_x_offset, 0 + corridor_y_offset], [width_corridor + corridor_x_offset, 0 + corridor_y_offset], [width_corridor + corridor_x_offset, height_corridor + corridor_y_offset + physical_bdry_polygon_extra], [0 + corridor_x_offset, height_corridor + corridor_y_offset]], dtype=np.float64)*1e-6
        
            
    return migr_space_poly, phys_space_poly
    
# =============================================================================
    
def update_pd(pd, key, orig_value, new_value):
    if pd[key] != orig_value:
        raise StandardError("Key {} does not have orig_value {}, instead {}".format(key, orig_value, pd[key]))
    else:
        new_pd = copy.deepcopy(pd)
        new_pd.update([(key, new_value)])
        
        return new_pd

def update_pd_with_keyvalue_tuples(pd, keyvalue_tuples):
    new_pd = copy.deepcopy(pd)
    orig_keys = pd.keys()
    
    for keyvalue_tuple in keyvalue_tuples:
        key, orig_value, new_value = keyvalue_tuple
        
        if key == 'sigma_rac':
            pass
        
        if pd[key] != orig_value:
            raise StandardError("Key {} does not have orig_value {}, instead {}".format(key, orig_value, pd[key]))
        else:
            new_pd.update([(key, new_value)])
        
    return new_pd

# =============================================================================   
    
def make_experiment_description_file(experiment_description, environment_dir, environment_wide_variable_defns, user_cell_group_defns):
    notes_fp = os.path.join(environment_dir, 'experiment_notes.txt')
    notes_content = []

    notes_content.append("======= EXPERIMENT DESCRIPTION: {} =======\n\n")
    notes_content.append(experiment_description + '\n\n')
    notes_content.append("======= CELL GROUPS IN EXPERIMENT: {} =======\n\n")
    notes_content.append(repr(environment_wide_variable_defns) + '\n\n')
    
    num_cell_groups = len(user_cell_group_defns)
    notes_content.append("======= CELL GROUPS IN EXPERIMENT: {} =======\n\n".format(num_cell_groups))    
    notes_content.append(repr(user_cell_group_defns) + '\n\n')
            
    notes_content.append("======= VARIABLE SETTINGS =======\n\n")
    
    for n, cell_group_defn in enumerate(user_cell_group_defns):
        pd = cell_group_defn['parameter_override_dict']
        sorted_pd_keys = sorted(pd.keys())
        tuple_list = [(key, pd[key]) for key in sorted_pd_keys]
        notes_content.append('CELL_GROUP: {}\n'.format(n))
        notes_content.append(repr(tuple_list) + '\n\n')
    
    with open(notes_fp, 'w') as notes_file:
        notes_file.writelines(notes_content)
        
# =============================================================================

def form_base_environment_name_format_string(experiment_number, num_cells_total, total_time, num_timesteps, num_nodes_per_cell, height_corridor, width_corridor):
    
    base_env_name_format_string = "EXP{}".format(experiment_number) + "({},{})" + "_NC={}_TT={}_NT={}_NN={}".format(num_cells_total, total_time, num_timesteps, num_nodes_per_cell)
    
    if height_corridor == None or width_corridor == None:
        base_env_name_format_string += "_(None)"
    else:
        base_env_name_format_string += "_({}x{})".format(height_corridor, width_corridor)
    
    return base_env_name_format_string
    
# ========================================================================
    
def get_experiment_directory_path(base_output_dir, date_str, experiment_number):
    return os.path.join(base_output_dir, "{}/{}".format(date_str, 'EXP_{}'.format(experiment_number)))
    
def get_environment_directory_path(experiment_directory_path, environment_name):
    return os.path.join(experiment_directory_path, environment_name)

# ========================================================================

def run_experiments(experiment_directory, environment_name_format_strings, environment_wide_variable_defns, user_cell_group_defns_per_subexperiment, experiment_descriptions_per_subexperiment, external_gradient_fn_per_subexperiment, num_experiment_repeats=1, elapsed_timesteps_before_producing_intermediate_graphs=2500, elapsed_timesteps_before_producing_intermediate_animations=5000, animation_settings={}, produce_intermediate_visuals=True, produce_final_visuals=True, full_print=False, elapsed_timesteps_before_hard_saving_env=100, elapsed_timesteps_before_soft_saving_env=100, delete_and_rerun_experiments_without_stored_env=True):
    
    for repeat_number in xrange(num_experiment_repeats):
        for subexperiment_index, user_cell_group_defns in enumerate(user_cell_group_defns_per_subexperiment):
            environment_name_format_string = environment_name_format_strings[subexperiment_index]
            environment_name = environment_name_format_string.format(subexperiment_index, repeat_number)
            environment_dir = os.path.join(experiment_directory, environment_name)
            
            PO_set_string = "P0 SET {}, RPT {}".format(subexperiment_index, repeat_number)
            an_environment = None
            if os.path.exists(environment_dir):
                print PO_set_string + " directory exists."
                
                uncompressed_stored_env_path = os.path.join(environment_dir, environment_name + '.pkl')
                compressed_stored_env_path = os.path.join(environment_dir, environment_name + '.pkl.gz')
                
                if os.path.exists(compressed_stored_env_path) or os.path.exists(uncompressed_stored_env_path):
                    print PO_set_string + ' stored environment exists, checking to see if it has completed simulation...'
                    an_environment = retrieve_environment(environment_name, environment_dir)
                    if an_environment.simulation_complete() == True:
                        print "Simulation has been completed. Continuing..."
                        del an_environment
                        continue
                    else:
                        print "Simulation incomplete. Finishing..."
                else:
                    if delete_and_rerun_experiments_without_stored_env == True:
                        print PO_set_string + " directory exists, but stored environment missing -- deleting and re-running experiment."
                        shutil.rmtree(environment_dir)
                    else:
                        print PO_set_string + " directory exists, but stored environment missing. Continuing regardless."
                        continue
            
            print "RUNNING " + PO_set_string
            print "environment_dir: {}".format(environment_dir)
            
            if an_environment == None:
                os.makedirs(environment_dir)
                
                make_experiment_description_file(experiment_descriptions_per_subexperiment[subexperiment_index], environment_dir, environment_wide_variable_defns, user_cell_group_defns)    
                    
                print "Creating environment..."
                parameter_overrides = [x['parameter_override_dict'] for x in user_cell_group_defns]
                
                an_environment = parameterorg.make_environment_given_user_cell_group_defns(environment_name=environment_name, parameter_overrides=parameter_overrides, environment_filepath=environment_dir, user_cell_group_defns=user_cell_group_defns, external_gradient_fn=external_gradient_fn_per_subexperiment[subexperiment_index], **environment_wide_variable_defns)
                
            an_environment.full_print = full_print
            
            if elapsed_timesteps_before_soft_saving_env != None:
                print "Executing dynamics with soft_saving..."
                num_trials = 0
                while an_environment != None and an_environment.simulation_complete() == False:
                    num_trials += 1
                    print "num simulation trials: ", num_trials
                    
                    try:
                        simulation_time = an_environment.execute_system_dynamics(animation_settings,  produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, elapsed_timesteps_before_producing_intermediate_graphs=elapsed_timesteps_before_producing_intermediate_graphs, elapsed_timesteps_before_producing_intermediate_animations=elapsed_timesteps_before_producing_intermediate_animations, elapsed_timesteps_before_hard_saving_env=elapsed_timesteps_before_hard_saving_env, elapsed_timesteps_before_soft_saving_env=elapsed_timesteps_before_soft_saving_env)
                    except:
                        print "##############################"
                        print "Exception caught! -- trying to recover"
                        an_environment = an_environment.load_soft_save()
                        #an_environment.store()
                        if an_environment == None:
                            print "Error recovering from soft save -- retrieved environment was None."
                            break
                        else:
                            print "Recovery succesful."
                        print "##############################"
                        
                if an_environment != None and an_environment.simulation_complete() == True:
                    print "Completed running the simulation."
                else:
                    print "Simulation incomplete -- unknown issue. Num trials: {}".format(num_trials)
                    
            else:
                simulation_time = an_environment.execute_system_dynamics(animation_settings,  produce_intermediate_visuals=produce_intermediate_visuals, produce_final_visuals=produce_final_visuals, elapsed_timesteps_before_producing_intermediate_graphs=elapsed_timesteps_before_producing_intermediate_graphs, elapsed_timesteps_before_producing_intermediate_animations=elapsed_timesteps_before_producing_intermediate_animations, elapsed_timesteps_before_hard_saving_env=elapsed_timesteps_before_hard_saving_env, elapsed_timesteps_before_soft_saving_env=elapsed_timesteps_before_soft_saving_env)
                
            print "Simulation run time: {}s".format(simulation_time)
            
# =======================================================================
        
def retrieve_environment(environment_name, data_directory):
    if os.path.exists(data_directory):
        print "Data directory exists."
        uncompressed_stored_env_path = os.path.join(data_directory, environment_name + '.pkl')
        compressed_stored_env_path = os.path.join(data_directory, environment_name + '.pkl.gz')
        
        if os.path.exists(uncompressed_stored_env_path):
            print "Uncompressed stored environment exists, unpickling..."
            stored_env = None
            
            with open(uncompressed_stored_env_path, 'rb') as f_in:
                stored_env = pickling_package.load(f_in)
            return stored_env
            
        elif os.path.exists(compressed_stored_env_path):
            print "Stored environment exists, unpacking..."
            stored_env = None
            with gzip.open(compressed_stored_env_path, 'rb') as f_in:
                stored_env = pickling_package.load(f_in)
            return stored_env
        else:
            raise StandardError("Stored environment does not exist!")
    else:
        raise StandardError("Data directory does not exist!")
            
