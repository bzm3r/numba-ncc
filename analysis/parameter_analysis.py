from __future__ import division
from sklearn.utils.extmath import cartesian
import environment
import parameterorg
import numpy as np
import analysis
import general
import multiprocessing as multiproc
import time
import matplotlib.pyplot as plt

#------------------------------------------------------------------

def make_parameter_dictionary(task_index, parameter_overrides, num_nodes_per_cell, num_cells):

    cell_group_name = 'task_{}'.format(task_index)
    
    C_total = 3e6 # number of molecules
    H_total = 1.5e6  # number of molecule
    
    #-------------------------------------
    init_cell_radius = 12.5e-6 # micrometers    
    offset = 25
    cell_group_bounding_box = np.array([offset, 50 + offset, offset, 50 + offset])*1e-6 # micrometers
    
    #-------------------------------------
    
    integration_params = {}
    
    intercellular_contact_factors_magnitudes_defn = {cell_group_name: 2}
    cell_dependent_coa_factor_production_defn = {cell_group_name: 0}
    
    parameter_definition_dict = parameterorg.make_chem_mech_space_parameter_defn_dict(C_total=C_total, H_total=H_total, num_nodes=num_nodes_per_cell, init_cell_radius=init_cell_radius, cell_group_bounding_box=cell_group_bounding_box, **parameter_overrides)
        
    task_environment_cell_group_defn = {'cell_group_name': cell_group_name, 'num_cells': num_cells, 'init_cell_radius': init_cell_radius, 'C_total': C_total, 'H_total': H_total, 'cell_group_bounding_box': cell_group_bounding_box, 'chem_mech_space_defns': parameter_definition_dict, 'integration_params': integration_params, 'intercellular_contact_factor_magnitudes_defn': intercellular_contact_factors_magnitudes_defn, 'cell_dependent_coa_factor_production_defn': cell_dependent_coa_factor_production_defn}
    
    return task_environment_cell_group_defn

# -------------------------------------

def run_environment_dynamics(args):
    cell_group_defn_dict, num_timesteps = args
    
    an_environment = environment.Environment('worker', num_timesteps, cell_group_defns=[cell_group_defn_dict], environment_filepath=None, verbose=False, num_nodes_per_cell=15)

    an_environment.execute_system_dynamics_for_all_times()
    
    return an_environment.cells_in_environment[0]
    
# --------------------------------------------------------------------

global_weird_parameter_dicts = []

def parameter_explorer_asymmetry_criteria(parameter_exploration_program, num_timesteps, num_nodes_per_cell, num_cells, task_chunk_size=4, num_processes=4, sequential=False, min_symmetry_score=0.01, filters=[]):
    given_parameter_labels = []
    given_parameter_values = []
    
    num_estimated_combinations = 1
    
    for parameter_label, start_value, end_value, range_resolution in parameter_exploration_program:
        given_parameter_labels.append(parameter_label)
        given_parameter_values.append(np.linspace(start_value, end_value, num=range_resolution))
        num_estimated_combinations = num_estimated_combinations*range_resolution

    all_parameter_labels = parameterorg.standard_parameter_dictionary_keys
    if not np.all([(parameter_label in all_parameter_labels) for parameter_label in given_parameter_labels]):
        raise StandardError("Not all given parameter labels are in the standard parameter dictionary.")
    
    print "Estimated number of combinations: ", num_estimated_combinations
    task_value_arrays = cartesian(tuple(given_parameter_values))
    
    num_combinations = len(task_value_arrays)
    print "Number of parameter combinations: ", num_combinations
    
    chunky_task_value_array_indices = general.chunkify(np.arange(num_combinations), task_chunk_size)
    
    num_task_chunks = len(chunky_task_value_array_indices)
    
    results = []
    num_results = 0
    
    worker_pool = multiproc.Pool(processes=num_processes,maxtasksperchild=750)
    
    for chunk_index, task_index_chunk in enumerate(chunky_task_value_array_indices):
        st = time.time()
        print "Executing task chunk %d/%d..." %(chunk_index + 1, num_task_chunks)
        
        
        parameter_dicts = []
        task_chunk = []
        for task_index in task_index_chunk:
            task_value_array = task_value_arrays[task_index]
            update_dict = dict(zip(given_parameter_labels, task_value_array))
            parameter_dicts.append(update_dict)
            task_environment_cell_group_defn = make_parameter_dictionary(task_index, update_dict, num_nodes_per_cell, num_cells)
            task_chunk.append((task_environment_cell_group_defn, num_timesteps))
            
        loop_result_cells = []
        if sequential == True:
            loop_result_cells = []
            for task in task_chunk:
                loop_result_cells.append(run_environment_dynamics(task))
        else:
            loop_result_cells = worker_pool.map(run_environment_dynamics, task_chunk)
            
            
        symmetry_results = [analysis.score_symmetries(a_cell) for a_cell in loop_result_cells]
        
        for sym_result, parameter_dict in zip(symmetry_results, parameter_dicts):
            if np.all(np.array(sym_result.values()) > min_symmetry_score):
                results.append((sym_result, parameter_dict))
                num_results = num_results + 1
        
        et = time.time()
        print "Time: ", np.round(et - st, decimals=1)
        print "Results: ", num_results


    return results
    
# =====================================================================
    
def parameter_explorer_displacement_criteria(parameter_exploration_program, num_timesteps, num_nodes_per_cell, num_cells, task_chunk_size=4, num_processes=4, sequential=False, min_displacement=10):
    given_parameter_labels = []
    given_parameter_values = []
    
    num_estimated_combinations = 1
    
    for parameter_label, start_value, end_value, range_resolution in parameter_exploration_program:
        given_parameter_labels.append(parameter_label)
        given_parameter_values.append(np.linspace(start_value, end_value, num=range_resolution))
        num_estimated_combinations = num_estimated_combinations*range_resolution

    all_parameter_labels = parameterorg.standard_parameter_dictionary_keys
    if not np.all([(parameter_label in all_parameter_labels) for parameter_label in given_parameter_labels]):
        raise StandardError("Not all given parameter labels are in the standard parameter dictionary.")
    
    print "Estimated number of combinations: ", num_estimated_combinations
    task_value_arrays = cartesian(tuple(given_parameter_values))
    
    num_combinations = len(task_value_arrays)
    print "Number of parameter combinations: ", num_combinations
    
    chunky_task_value_array_indices = general.chunkify(np.arange(num_combinations), task_chunk_size)
    
    num_task_chunks = len(chunky_task_value_array_indices)
    
    results = []
    num_results = 0
    
    worker_pool = multiproc.Pool(processes=num_processes,maxtasksperchild=250)
    
    for chunk_index, task_index_chunk in enumerate(chunky_task_value_array_indices):
        st = time.time()
        print "Executing task chunk %d/%d..." %(chunk_index + 1, num_task_chunks)
        
        
        parameter_dicts = []
        task_chunk = []
        for task_index in task_index_chunk:
            task_value_array = task_value_arrays[task_index]
            update_dict = dict(zip(given_parameter_labels, task_value_array))
            parameter_dicts.append(update_dict)
            task_environment_cell_group_defn = make_parameter_dictionary(task_index, parameter_overrides=update_dict)
            task_chunk.append(task_environment_cell_group_defn)
            
        loop_result_cells = []
        if sequential == True:
            loop_result_cells = []
            for task in task_chunk:
                loop_result_cells.append(run_environment_dynamics(task))
        else:
            loop_result_cells = worker_pool.map(run_environment_dynamics, task_chunk)
            
            
        loop_distance_results = [analysis.score_distance_travelled(a_cell) for a_cell in loop_result_cells]
        
        for loop_distance_result, parameter_dict in zip(loop_distance_results, parameter_dicts):
            if loop_distance_result[1] > min_displacement:
                results.append((loop_distance_result, parameter_dict))
                num_results = num_results + 1
        
        et = time.time()
        print "Time: ", np.round(et - st, decimals=1)
        print "Results: ", num_results


    return results

# =====================================================================

if __name__ == '__main__':
    
    num_timesteps = 500
    num_nodes_list = [10, 25, 50, 75, 100]
    num_repetitions = 3
    num_cells = 1
    
    all_results = []
    
    for index, num_nodes in enumerate(num_nodes_list):
        results_for_this_num_nodes = []
        
        for i in range(num_repetitions):
            results = parameter_explorer_asymmetry_criteria([('kgtp_rac_multiplier', 50, 150, 5), ('kgtp_rho_multiplier', 50, 150, 5), ('kgtp_rho_autoact_multiplier', 100, 250, 5),('kgtp_rac_autoact_multiplier', 100, 250, 5), ('kdgtp_rac_multiplier', 20, 20, 1), ('kdgtp_rho_multiplier', 20, 20, 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 100, 250, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 100, 250, 5), ('randomize_rgtpase_distrib', 1, 1, 1), ('threshold_rac_autoact_multiplier', 0.5, 0.5, 1), ('threshold_rho_autoact_multiplier', 0.5, 0.5, 1), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5, 0.5, 1), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5, 0.5, 1), ('threshold_rac_autoact_dgdi_multiplier', 0.5, 0.5, 1), ('randomize_rgtpase_distrib', 1, 1, 1)], num_timesteps, num_nodes, num_cells, sequential=False, min_symmetry_score=-1)
        
        # [('kgtp_rac_multiplier', 75, 250, 5), ('kgtp_rho_multiplier', 75, 250, 5), ('kgtp_rho_autoact_multiplier', 75, 250, 5),('kgtp_rac_autoact_multiplier', 75, 250, 5), ('kdgtp_rac_multiplier', 10, 100, 5), ('kdgtp_rho_multiplier', 10, 100, 5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 75, 250, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 75, 250, 5), ('randomize_rgtpase_distrib', 1, 1, 1)]
        
            results_for_this_num_nodes.append(results)
            
        all_results.append(results_for_this_num_nodes)
    
#    robustness_results = check_robustness_of_parameter_sets([result[1] for result in interesting_results], {'kgtp_rac_multiplier': 300, 'kgtp_rho_multiplier': 300, 'kgtp_rho_autoact_multiplier': 1000, 'kgtp_rac_autoact_multiplier': 1000, 'kdgtp_rac_multiplier': 300, 'kdgtp_rho_multiplier': 300, 'kdgtp_rac_mediated_rho_inhib_multiplier': 1000, 'kdgtp_rho_mediated_rac_inhib_multiplier': 1000}, sequential=False)