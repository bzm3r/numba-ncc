from __future__ import division
from sklearn.utils.extmath import cartesian
import core.parameterorg as parameterorg
import numpy as np
import core.utilities as cu
import multiprocessing as multiproc
import time
import matplotlib.pyplot as plt
import general.utilities as general_utils
import general.experiment_templates as exptempls
import general.exec_utils as executils
import copy

# --------------------------------------------------------------------
STANDARD_PARAMETER_DICT = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 20), ('kgtp_rac_autoact_multiplier', 250), ('kdgtp_rac_multiplier', 15), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 20), ('kgtp_rho_autoact_multiplier', 120), ('kdgtp_rho_multiplier', 15), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.5), ('max_coa_signal', -1), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 10), ('closeness_dist_squared_criteria', 0.25e-12), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 1.), ('skip_dynamics', False)])

global_weird_parameter_dicts = []
global_results = []

def parameter_explorer_asymmetry_criteria(parameter_exploration_program, task_chunk_size=4, num_processes=4, sequential=False, min_polarity_score=0):
    given_parameter_labels = []
    given_parameter_values = []
    
    num_estimated_combinations = 1
    
    for parameter_label, start_value, end_value, range_resolution in parameter_exploration_program:
        given_parameter_labels.append(parameter_label)
        given_parameter_values.append(np.linspace(start_value, end_value, num=range_resolution))
        num_estimated_combinations = num_estimated_combinations*range_resolution

    all_parameter_labels = parameterorg.all_user_parameters_with_justifications.keys()
    for parameter_label in given_parameter_labels:
        if parameter_label not in all_parameter_labels:
            raise StandardError("Parameter label {} not in accepted parameter dictionary.".format(parameter_label))
    
    print "Estimated number of combinations: ", num_estimated_combinations
    task_value_arrays = cartesian(tuple(given_parameter_values))
    
    num_combinations = len(task_value_arrays)
    print "Number of parameter combinations: ", num_combinations
    
    chunky_task_value_array_indices = general_utils.chunkify(np.arange(num_combinations), task_chunk_size)
    
    num_task_chunks = len(chunky_task_value_array_indices)
    
    results = []
    num_results = 0
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    global global_results
    global STANDARD_PARAMETER_DICT
    best_pr = 0
    
    for chunk_index, task_index_chunk in enumerate(chunky_task_value_array_indices):
        st = time.time()
        print "Executing task chunk %d/%d..." %(chunk_index + 1, num_task_chunks)
        
        update_dicts = []
        task_chunk = []
        for task_index in task_index_chunk:
            task_value_array = task_value_arrays[task_index]
            update_dict = dict(zip(given_parameter_labels, task_value_array))
            parameter_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
            parameter_dict.update(update_dict)
            update_dicts.append(update_dict)
            task_environment_defn = exptempls.setup_polarization_experiment(parameter_dict)
            task_chunk.append(task_environment_defn)
            
        loop_result_cells = []
        if sequential == True:
            loop_result_cells = []
            for task in task_chunk:
                loop_result_cells.append(executils.run_simple_experiment_and_return_cell_worker(task))
        else:
            loop_result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_chunk)
            
            
        polarity_rating_tuples_per_loop_cell = [cu.calculate_rgtpase_polarity_score_from_cell(a_cell, significant_difference=0.2, weigh_by_timepoint=False) for a_cell in loop_result_cells]
        
        polarity_ratings_per_loop_cell = [x[0] for x in polarity_rating_tuples_per_loop_cell]
        print "polarity ratings: ", polarity_ratings_per_loop_cell
        for pr in polarity_ratings_per_loop_cell:
            if pr > best_pr:
                best_pr = pr
        
        for polarity_result, update_dict in zip(polarity_ratings_per_loop_cell, update_dicts):
            if polarity_result > min_polarity_score:
                results.append((polarity_result, update_dicts))
                num_results = num_results + 1
        
        global_results = results
        et = time.time()
        print "Time: ", np.round(et - st, decimals=1)
        print "best_pr: ", best_pr
        
        if best_pr > 0.25:
            break

    return results

# =====================================================================

if __name__ == '__main__':
    #results = parameter_explorer_asymmetry_criteria([('kgtp_rac_autoact_multiplier', 10, 250, 10), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1, 5, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1, 10, 5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.01, 0.4, 5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.01, 0.4, 5)], sequential=False, min_polarity_score=0, TOTAL_TIME=1000)

    results =  parameter_explorer_asymmetry_criteria([('kgtp_rac_multiplier', 1, 20, 10), ('kdgtp_rac_multiplier', 1, 20, 10), ('kgtp_rho_multiplier', 1, 20, 10), ('kdgtp_rho_multiplier', 1, 20, 10)], sequential=False, min_polarity_score=0)# parameter_explorer_asymmetry_criteria([('kgtp_rho_autoact_multiplier', 5, 500, 5),('kgtp_rac_autoact_multiplier', 5, 500, 5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 5, 1000, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 5, 4000, 5 ), ('threshold_rac_autoact_multiplier', 0.1, 0.6, 5), ('threshold_rho_autoact_multiplier', 0.1, 0.6, 5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.1, 0.6, 5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.1, 0.6, 5)], sequential=False, min_polarity_score=0, TOTAL_TIME=500)
    
    sorted_results = sorted(global_results, key = lambda x: x[0])
    
    print "Number of interesting results: ", len(sorted_results)
    