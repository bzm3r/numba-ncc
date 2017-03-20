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
import core.hardio as hardio
import copy
import os
import shutil
import dill

# --------------------------------------------------------------------
STANDARD_PARAMETER_DICT = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 10), ('closeness_dist_squared_criteria', 0.25e-12), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-6), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 1.), ('skip_dynamics', False)])

global_weird_parameter_dicts = []
global_results = []

def modify_pd_deterministically(parameter_dict, moddable_parameter_labels, mod_deltas, justifications, resolution):
    modded_pds_per_change = []
    
    for mpl, mod_delta, sorted_j in zip(moddable_parameter_labels, mod_deltas, justifications):
        old_value = parameter_dict[mpl]
        minj, maxj = sorted_j
        new_value = old_value + 1*resolution*mod_delta
        
        if new_value > maxj:
            new_value = maxj
        elif new_value < minj:
            new_value = minj
        
        new_pd = copy.deepcopy(parameter_dict)
        new_pd.update([(mpl, new_value)])
        modded_pds_per_change.append((new_pd, mpl, new_value))
        
        new_value = old_value + -1*resolution*mod_delta
        
        if new_value > maxj:
            new_value = maxj
        elif new_value < minj:
            new_value = minj
        
        new_pd = copy.deepcopy(parameter_dict)
        new_pd.update([(mpl, new_value)])
        modded_pds_per_change.append((new_pd, mpl, new_value))
        
    return modded_pds_per_change

def generate_random_initial_starting_pd(moddable_parameter_labels, mod_deltas, mod_justifications, pd):
    for mpl, delta, justification in zip(moddable_parameter_labels, mod_deltas, mod_justifications):
        pd.update([(mpl, justification[0] + np.random.rand()*delta)])
        
    return pd
    

def parameter_explorer_polarization_slope_follower(moddable_parameter_labels, required_polarization_score, num_processes=4, resolution=0.01, start_from_random_init_condition=True, num_experiment_repeats=3):
    num_new_dicts_to_generate = len(moddable_parameter_labels)
    
    mod_deltas = []
    mod_justifications = []
    for mpl in moddable_parameter_labels:
        justification = None
        try:
            justification = parameterorg.all_user_parameters_with_justifications[mpl]
        except:
            raise ValueError("Parameter label {} not found within accepted parameter list.".format(mpl))
            
        if justification == None:
            raise ValueError("Parameter label {} has no justified region to vary within!".format(mpl))
        
        delta_justification = max(justification) - min(justification)
        mod_deltas.append(delta_justification)
        mod_justifications.append(np.sort(justification))
        
    global STANDARD_PARAMETER_DICT
    
    if start_from_random_init_condition:
        current_best_parameter_dict = generate_random_initial_starting_pd(moddable_parameter_labels, mod_deltas, mod_justifications, copy.deepcopy(STANDARD_PARAMETER_DICT))
    else:
        current_best_parameter_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
        
    current_polarization_score = 0.0
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    stop_criterion_met = False
    
    while not stop_criterion_met:
        improvement = False
        print "current_polarization_score: ", current_polarization_score
        
        print "preparing modded dicts..."
        modded_parameter_dicts = modify_pd_randomly(num_new_dicts_to_generate, current_best_parameter_dict, moddable_parameter_labels, mod_deltas, mod_justifications, resolution)
        
        print "preparing tasks..."
        task_list = []
        for mpd in modded_parameter_dicts:
            for n in range(num_experiment_repeats):
                task_list.append(exptempls.setup_polarization_experiment(mpd))

        print "running tasks in parallel..."
        result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
        
        print "analyzing results..."
        i = 0
        result_cells_per_pd = []
        while i < len(result_cells):
            result_cells_per_pd.append(result_cells[i:i+num_experiment_repeats])
            i += num_experiment_repeats
            
        assert(len(result_cells_per_pd) == len(modded_parameter_dicts))
    
        delta_polarity_results_and_corresponding_pds = []        
        for result_cells_chunk, mpd in zip(result_cells_per_pd, modded_parameter_dicts):
            pr = 0
            
            for rc in result_cells_chunk:
                pr += cu.calculate_rgtpase_polarity_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False)[0]
            
            pr = pr/num_experiment_repeats
            delta_polarity_results_and_corresponding_pds.append((pr - current_polarization_score, mpd))
        
        best_delta_pr = 0.0
        for delta_pr, pd in delta_polarity_results_and_corresponding_pds:
            if delta_pr > best_delta_pr:
                print "best_delta_pr: ", delta_pr
                best_delta_pr = delta_pr
                current_polarization_score = best_delta_pr + current_polarization_score
                current_best_parameter_dict = copy.deepcopy(pd)
                improvement = True    
                
        if improvement:
            print "improvement seen!"
            num_tries_with_no_improvement = 0
        else:
            print "no improvement."
            num_tries_with_no_improvement += 1
            
        if current_polarization_score > required_polarization_score:
            print "Stop criterion met!"
            stop_criterion_met = True
        
        if num_tries_with_no_improvement >= 1:
            print "no improvement possible...retrying from new initial conditions!"
            current_best_parameter_dict = generate_random_initial_starting_pd(moddable_parameter_labels, mod_deltas, mod_justifications, copy.deepcopy(STANDARD_PARAMETER_DICT))
            current_polarization_score = cu.calculate_rgtpase_polarity_score_from_cell(executils.run_simple_experiment_and_return_cell_worker(exptempls.setup_polarization_experiment(current_best_parameter_dict)), significant_difference=0.2, weigh_by_timepoint=False)[0]
            print "initial polarization score: ", current_polarization_score
            
    return current_best_parameter_dict

def modify_pd_randomly(num_new_dicts_to_generate, parameter_dict, moddable_parameter_labels, mod_deltas, justifications, max_resolution):
    modded_pds = [copy.deepcopy(parameter_dict) for n in range(num_new_dicts_to_generate)]
    
    for mpl, mod_delta, sorted_j in zip(moddable_parameter_labels, mod_deltas, justifications):
        old_value = parameter_dict[mpl]
        minj, maxj = sorted_j
        for n in range(num_new_dicts_to_generate):
            new_value = old_value + (np.random.rand() - 0.5)*2*max_resolution*mod_delta
            
            if new_value > maxj:
                new_value = maxj
            elif new_value < minj:
                new_value = minj
                
            modded_pds[n].update([(mpl, new_value)])
            
    return modded_pds

def parameter_explorer_polarization_wanderer(moddable_parameter_labels, required_polarization_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, sequential=False, initial_resolution=0.01, max_resolution=1.0, start_from_random_init_condition=True, num_experiment_repeats=3):
    
    current_resolution = initial_resolution
    
    mod_deltas = []
    mod_justifications = []
    for mpl in moddable_parameter_labels:
        justification = None
        try:
            justification = parameterorg.all_user_parameters_with_justifications[mpl]
        except:
            raise ValueError("Parameter label {} not found within accepted parameter list.".format(mpl))
            
        if justification == None:
            raise ValueError("Parameter label {} has no justified region to vary within!".format(mpl))
        
        delta_justification = max(justification) - min(justification)
        mod_deltas.append(delta_justification)
        mod_justifications.append(np.sort(justification))
        
    global STANDARD_PARAMETER_DICT
    
    if start_from_random_init_condition:
        current_best_parameter_dict = generate_random_initial_starting_pd(moddable_parameter_labels, mod_deltas, mod_justifications, copy.deepcopy(STANDARD_PARAMETER_DICT))
    else:
        current_best_parameter_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
    
    current_polarization_score = cu.calculate_rgtpase_polarity_score_from_cell(executils.run_simple_experiment_and_return_cell_worker(exptempls.setup_polarization_experiment(current_best_parameter_dict)), significant_difference=0.2, weigh_by_timepoint=False)[0]
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    stop_criterion_met = False
    num_tries_with_no_improvement = 0
    
    while not stop_criterion_met:
        improvement = False
        print "current_polarization_score: ", current_polarization_score
        
        print "preparing modded dicts..."
        modded_parameter_dicts = modify_pd_randomly(num_new_dicts_to_generate, current_best_parameter_dict, moddable_parameter_labels, mod_deltas, mod_justifications, current_resolution)
        
        print "preparing tasks..."
        task_list = []
        for mpd in modded_parameter_dicts:
            for n in range(num_experiment_repeats):
                task_list.append(exptempls.setup_polarization_experiment(mpd))

        print "running tasks in parallel..."
        result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
        
        print "analyzing results..."
        i = 0
        result_cells_per_pd = []
        while i < len(result_cells):
            result_cells_per_pd.append(result_cells[i:i+num_experiment_repeats])
            i += num_experiment_repeats
            
        assert(len(result_cells_per_pd) == len(modded_parameter_dicts))
    
        polarity_results_and_corresponding_pds = []        
        for result_cells_chunk, mpd in zip(result_cells_per_pd, modded_parameter_dicts):
            pr = 0
            
            for rc in result_cells_chunk:
                pr += cu.calculate_rgtpase_polarity_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False)[0]
            
            pr = pr/num_experiment_repeats
            polarity_results_and_corresponding_pds.append((pr, mpd))
        
        for pr, pd in polarity_results_and_corresponding_pds:
            if pr > current_polarization_score:
                print "possible new score: ", pr
                current_polarization_score = pr
                current_best_parameter_dict = copy.deepcopy(pd)
                improvement = True    
                
        if improvement:
            print "improvement seen!"
            num_tries_with_no_improvement = 0
            current_resolution = initial_resolution
        else:
            print "no improvement."
            if num_tries_with_no_improvement > 2:
                current_resolution = 2*current_resolution
                if current_resolution > max_resolution:
                    current_resolution = max_resolution
                num_tries_with_no_improvement += 1
            
        if current_polarization_score > required_polarization_score:
            print "Stop criterion met!"
            stop_criterion_met = True
        
        if num_tries_with_no_improvement >= 50:
            print "no improvement seen for a while...retrying from new initial conditions!"
            current_best_parameter_dict = generate_random_initial_starting_pd(moddable_parameter_labels, mod_deltas, mod_justifications, copy.deepcopy(STANDARD_PARAMETER_DICT))
            current_polarization_score = cu.calculate_rgtpase_polarity_score_from_cell(executils.run_simple_experiment_and_return_cell_worker(exptempls.setup_polarization_experiment(current_best_parameter_dict)), significant_difference=0.2, weigh_by_timepoint=False)[0]
            print "initial polarization score: ", current_polarization_score
            
            
    return current_best_parameter_dict
    
def create_task_value_arrays(parameter_exploration_program, num_processes):
    given_parameter_labels = []
    given_parameter_values = []
    
    for parameter_label, start_value, end_value, range_resolution in parameter_exploration_program:
        given_parameter_labels.append(parameter_label)
        given_parameter_values.append(np.floor(np.linspace(start_value, end_value, num=range_resolution)))

    all_parameter_labels = parameterorg.all_user_parameters_with_justifications.keys()
    for parameter_label in given_parameter_labels:
        if parameter_label not in all_parameter_labels:
            raise StandardError("Parameter label {} not in accepted parameter dictionary.".format(parameter_label))
    
    task_value_arrays = cartesian(tuple(given_parameter_values))
    num_combinations = len(task_value_arrays)
    
    chunky_task_value_array_indices = general_utils.chunkify(np.arange(num_combinations), num_processes)
    
    return given_parameter_labels, given_parameter_values, task_value_arrays, chunky_task_value_array_indices
    
    
    
def parameter_explorer_asymmetry_criteria(parameter_exploration_name, parameter_exploration_program, sequential=False, result_storage_folder="A:\\numba-ncc\\output", overwrite=False, seed=36):
    num_processes = 4
    
    assert(type(parameter_exploration_name) == str)
    
    storedir = os.path.join(result_storage_folder, parameter_exploration_name)
    storefile_path = os.path.join(storedir, "dataset.h5py")
    program_path = os.path.join(storedir, "parameter_exploration_program.pkl")
    
    stored_data_exists = False
    if not os.path.exists(storedir):
        os.makedirs(storedir)
    else:
        if overwrite == True:
            shutil.rmtree(storedir)
            os.makedirs(storedir)
        else:
            does_stored_program_exist = os.path.isfile(program_path)
            does_storefile_exist = os.path.isfile(storefile_path)
            
            if not (does_storefile_exist and does_stored_program_exist):
                shutil.rmtree(storedir)
                os.makedirs(storedir)
            else:
                stored_data_exists = True
    
    if stored_data_exists:
        ci_offset = hardio.get_last_executed_parameter_exploration_chunk_index(storefile_path)
        with open(program_path, 'rb') as f:
            stored_parameter_exploration_program = dill.load(f)
        
        if stored_parameter_exploration_program != parameter_exploration_program:
            raise StandardError("Stored exploration program does not match given exploration program! stored: {} || given: {}".format(stored_parameter_exploration_program, parameter_exploration_program))
            
        given_parameter_labels, given_parameter_values, task_value_arrays, chunky_task_value_array_indices = create_task_value_arrays(parameter_exploration_program, num_processes)
    else:
        ci_offset = 0
            
        given_parameter_labels, given_parameter_values, task_value_arrays, chunky_task_value_array_indices = create_task_value_arrays(parameter_exploration_program, num_processes)
        
        with open(program_path, 'wb') as f:
            dill.dump(parameter_exploration_program, f)
        
        hardio.create_parameter_exploration_dataset(storefile_path, len(given_parameter_labels) + 1)
        
    
    
    num_task_chunks = len(chunky_task_value_array_indices)
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    global global_results
    global STANDARD_PARAMETER_DICT
    best_pr = 0
    
    
    for ci, task_index_chunk in enumerate(chunky_task_value_array_indices[ci_offset:]):
        chunk_index = ci + ci_offset
        st = time.time()
        print "Executing task chunk %d/%d..." %(chunk_index + 1, num_task_chunks)
        results = np.zeros((num_processes, len(given_parameter_labels) + 1), dtype=np.float64)
        
        update_dicts = []
        task_chunk = []
        for task_index in task_index_chunk:
            task_value_array = task_value_arrays[task_index]
            update_dict = dict(zip(given_parameter_labels, task_value_array))
            parameter_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
            parameter_dict.update(update_dict)
            update_dicts.append(update_dict)
            task_environment_defn = exptempls.setup_polarization_experiment(parameter_dict, seed=seed)
            task_chunk.append(task_environment_defn)
            
        loop_result_cells = []
        if sequential == True:
            loop_result_cells = []
            for task in task_chunk:
                loop_result_cells.append(executils.run_simple_experiment_and_return_cell_worker(task))
        else:
            loop_result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_chunk)
            
            
        results = np.array([np.append([cu.calculate_rgtpase_polarity_score_from_cell(a_cell, significant_difference=0.2, weigh_by_timepoint=False)[0]], task_value_arrays[ti]) for a_cell, ti in zip(loop_result_cells, task_index_chunk)])
        
        current_polarity_ratings = results[:,0]
        print "polarity ratings: ", current_polarity_ratings
        max_pr = np.max(current_polarity_ratings)
        if max_pr > best_pr:
            best_pr = max_pr
        
        et = time.time()
        print "Time: ", np.round(et - st, decimals=1)
        print "best_pr: ", best_pr
        
        print "Storing results..."
        hardio.append_parameter_exploration_data_to_dataset(ci + ci_offset, results, storefile_path)
        
    print "Storing last batch of results..."
    hardio.append_parameter_exploration_data_to_dataset(ci + ci_offset, results, storefile_path)

    return storefile_path

# =====================================================================

def get_result_as_dict_update(results, index, labels):
    return zip(labels, results[index][1:])
    
if __name__ == '__main__':
    pe_name = "2017_MAR_20_PE"
    
    exploration_program = [('kgtp_rac_multiplier', 1, 20, 5), ('kgtp_rho_multiplier', 1, 20, 5), ('kdgtp_rac_multiplier', 1, 20, 5), ('kdgtp_rho_multiplier', 1, 20, 5), ('kgtp_rho_autoact_multiplier', 100, 500, 5), ('kgtp_rac_autoact_multiplier', 100, 500, 5)]
    p_labels = [x[0] for x in exploration_program]
    storefile_path = parameter_explorer_asymmetry_criteria(pe_name, exploration_program, seed=36)
    
    results = hardio.get_parameter_exploration_results(storefile_path)
    sorted_results = results[results[:,0].argsort()]
    
    best_update = get_result_as_dict_update(sorted_results, -1, p_labels)
    
    

#    moddable_parameters = ['kgtp_rac_multiplier', 'kgtp_rho_multiplier', 'kdgtp_rac_multiplier', 'kdgtp_rho_multiplier', 'threshold_rac_activity_multiplier', 'threshold_rho_activity_multiplier', 'kgtp_rac_autoact_multiplier', 'kgtp_rho_autoact_multiplier', 'kdgtp_rac_mediated_rho_inhib_multiplier', 'kdgtp_rho_mediated_rac_inhib_multiplier']
#    #best_pd = parameter_explorer_polarization_wanderer(moddable_parameters, 0.5, num_new_dicts_to_generate=len(moddable_parameters), initial_resolution=0.1, max_resolution=1.0, num_experiment_repeats=2, num_processes=6)
#    best_pd = parameter_explorer_polarization_slope_follower(moddable_parameters, 0.6, resolution=0.01)
    