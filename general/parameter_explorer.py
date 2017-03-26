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
STANDARD_PARAMETER_DICT = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.025), ('max_coa_signal', -1), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 10), ('closeness_dist_squared_criteria', 0.25e-12), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-7), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 1.), ('skip_dynamics', False)])

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

def generate_random_starting_update(moddable_parameter_labels, mod_deltas, mod_min_max, pd):
    
    parameter_update = []
    
    for mpl, delta, mod_min_max in zip(moddable_parameter_labels, mod_deltas, mod_min_max):
        minval, maxval = mod_min_max
        value = np.random.choice(np.arange(mod_min_max[0], mod_min_max[1] + delta, step=delta))
        
        if value < minval:
            value = minval
        elif value > maxval:
            value = maxval
        
        parameter_update.append((mpl, value))
        
    return parameter_update
    

def generate_random_update_based_on_current_state(num_new_updates, parameter_dict, moddable_parameter_labels, mod_deltas, mod_min_max):
    parameter_updates = [[] for x in range(num_new_updates)]
    
    for mpl, delta, this_min_max in zip(moddable_parameter_labels, mod_deltas, mod_min_max):
        old_value = parameter_dict[mpl]
        minj, maxj = this_min_max
        
        if old_value == minj:
            sign_choice = [0, 1]
        elif old_value == maxj:
            sign_choice = [-1, 0]
        else:
            sign_choice = [-1, 0, 1]
            
        for n in range(num_new_updates):
            new_value = old_value + np.random.choice(sign_choice)*np.random.choice([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 6, 10, 100])*delta
            
            if new_value > maxj:
                new_value = maxj
            elif new_value < minj:
                new_value = minj
                
            parameter_updates[n].append((mpl, new_value))
            
    return parameter_updates

def parameter_explorer_polarization_wanderer(modification_program, required_polarization_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, sequential=False, num_experiment_repeats=3, total_time_in_hours=2.):
    
    mod_deltas = [x[3] for x in modification_program]
    mod_min_max = [x[1:3] for x in modification_program]
    mod_labels = [x[0] for x in modification_program]
    
    acceptable_labels = parameterorg.all_user_parameters_with_justifications.keys()
    
    for ml in mod_labels:
        if ml not in acceptable_labels:
            raise StandardError("{} not in acceptable parameter labels list.".format(ml))
        
    global STANDARD_PARAMETER_DICT
    current_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
    
    current_best_update = generate_random_starting_update(mod_labels, mod_deltas, mod_min_max, current_dict)
    current_dict.update(current_best_update)
    
    current_polarization_score = cu.calculate_rgtpase_polarity_score_from_cell(executils.run_simple_experiment_and_return_cell_worker(exptempls.setup_polarization_experiment(current_dict, total_time_in_hours=total_time_in_hours)), significant_difference=0.2, weigh_by_timepoint=False)[0]
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    stop_criterion_met = False
    num_tries_with_no_improvement = 0
    
    while not stop_criterion_met:
        improvement = False
        print "current_polarization_score: ", current_polarization_score
        
        print "preparing modded dicts..."
        trial_updates = generate_random_update_based_on_current_state(num_new_dicts_to_generate, current_dict, mod_labels, mod_deltas, mod_min_max)
        
        print "preparing tasks..."
        task_list = []
        for u in trial_updates:
            for n in range(num_experiment_repeats):
                trial_dict = copy.deepcopy(current_dict)
                trial_dict.update(u)
                task_list.append(exptempls.setup_polarization_experiment(trial_dict, total_time_in_hours=total_time_in_hours))

        print "running tasks in parallel..."
        result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
        
        print "analyzing results..."
        i = 0
        result_cells_per_pd = []
        while i < len(result_cells):
            result_cells_per_pd.append(result_cells[i:i+num_experiment_repeats])
            i += num_experiment_repeats
            
        assert(len(result_cells_per_pd) == len(trial_updates))
    
        polarity_results_and_corresponding_updates = []        
        for result_cells_chunk, u in zip(result_cells_per_pd, trial_updates):
            pr = 0
            
            for rc in result_cells_chunk:
                pr += cu.calculate_rgtpase_polarity_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False)[0]
            
            pr = pr/num_experiment_repeats
            print "possible pr: ", pr
            polarity_results_and_corresponding_updates.append((pr, u))
        
        for pr, u in polarity_results_and_corresponding_updates:
            if pr > current_polarization_score:
                print "possible new score: ", pr
                current_polarization_score = pr
                current_best_update = u
                current_dict = copy.deepcopy(current_dict)
                current_dict.update(u)
                improvement = True    
                
        if improvement:
            print "improvement seen!"
            num_tries_with_no_improvement = 0
        else:
            print "no improvement. ({})".format(num_tries_with_no_improvement)
            num_tries_with_no_improvement += 1
            if num_tries_with_no_improvement > 25:
                print "no improvement seen for a while...retrying from new initial conditions!"
                current_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
                current_best_update = generate_random_starting_update(mod_labels, mod_deltas, mod_min_max, current_dict)
                current_dict.update(current_best_update)
                current_polarization_score = cu.calculate_rgtpase_polarity_score_from_cell(executils.run_simple_experiment_and_return_cell_worker(exptempls.setup_polarization_experiment(current_dict)), significant_difference=0.2, weigh_by_timepoint=False)[0]
            
        if current_polarization_score > required_polarization_score:
            print "Success! Stop criterion met!"
            stop_criterion_met = True
        
        if num_tries_with_no_improvement >= 50:
            print "initial polarization score: ", current_polarization_score            
            
    return current_best_update

def bounded(value, limits):
    minv, maxv = limits
    if value < minv:
        return minv
    elif value > maxv:
        return maxv
    else:
        return value

def parameter_explorer_polarization_slope_follower(modification_program, required_polarization_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, sequential=False, num_experiment_repeats=3, total_time_in_hours=2.):
    
    mod_deltas = [x[3] for x in modification_program]
    mod_min_max = [x[1:3] for x in modification_program]
    mod_labels = [x[0] for x in modification_program]
    
    acceptable_labels = parameterorg.all_user_parameters_with_justifications.keys()
    
    for ml in mod_labels:
        if ml not in acceptable_labels:
            raise StandardError("{} not in acceptable parameter labels list.".format(ml))
        
    global STANDARD_PARAMETER_DICT
    current_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
    
    current_best_update = generate_random_starting_update(mod_labels, mod_deltas, mod_min_max, current_dict) #[(label, lims[0]) for label, lims in zip(mod_labels, mod_min_max)]
    current_dict.update(current_best_update)
    
    current_polarization_score = cu.calculate_rgtpase_polarity_score_from_cell(executils.run_simple_experiment_and_return_cell_worker(exptempls.setup_polarization_experiment(current_dict, total_time_in_hours=total_time_in_hours)), significant_difference=0.2, weigh_by_timepoint=False)[0]
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    stop_criterion_met = False
    
    while not stop_criterion_met:
        print "==========================="
        print "current_polarization_score: ", current_polarization_score
        
        dprs = []
        for label_index, loop_data in enumerate(zip(mod_labels, mod_min_max, mod_deltas)):
            label, this_min_max, this_delta = loop_data
            
            at_boundary = True
            minval, maxval = this_min_max
            current_value = current_dict[label]
            
            dv = this_delta
            
            if current_value == minval or current_value - dv < minval:
                signs = [0, 1]
            elif current_value == maxval or current_value + dv > maxval:
                signs = [-1, 0]
            else:
                signs = [-1, 1]
                at_boundary = False
                
            x = copy.deepcopy(current_dict)
            x.update([(label, current_value + dv*signs[0])])
            y = copy.deepcopy(current_dict)
            y.update([(label, current_value + dv*signs[1])])
            
            seeds = [int(np.round(np.random.rand(), decimals=4)*1000) for n in range(num_experiment_repeats)]
            task_list = [exptempls.setup_polarization_experiment(x, seed=seeds[n], total_time_in_hours=total_time_in_hours) for n in range(num_experiment_repeats)] + [exptempls.setup_polarization_experiment(y, seed=seeds[n], total_time_in_hours=total_time_in_hours) for n in range(num_experiment_repeats)]
            
            result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
            
            x_cells = result_cells[:num_experiment_repeats]
            y_cells = result_cells[num_experiment_repeats:]
            
            x_pr = np.array([cu.calculate_rgtpase_polarity_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False)[0] for rc in x_cells])
            y_pr = np.array([cu.calculate_rgtpase_polarity_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False)[0] for rc in y_cells])
            
            this_dpr = 0.0
            if at_boundary:
                this_dpr = np.average(y_pr - x_pr)/this_delta
            else:
                this_dpr = np.average(y_pr - x_pr)/(2*this_delta)
            
            dprs.append(this_dpr)
            
        dprs = np.array([bounded(v, [-1., 1.]) for v in dprs])
        
        for l, dv in zip(mod_labels, dprs):
            print "{}: {}".format(l, dv)
            
        old_update = copy.deepcopy(current_best_update)
        old_values = np.array([cbu[1] for cbu in current_best_update])
            
        new_values = np.array([bounded(v + w*d, lims) for v, w, d, lims in zip(old_values, dprs, mod_deltas, mod_min_max)])
        
        if np.all(new_values/old_values == 1.0):
            print "No improvement possible :("
            stop_criterion_met = True
            continue
        
        current_best_update = zip(mod_labels, new_values)
        
        for old_cbu, new_cbu in zip(old_update, current_best_update):
            print "{}: {}, {}".format(old_cbu[0], np.round(new_cbu[1]/old_cbu[1], decimals=3), new_cbu[1])
            
        current_dict.update(current_best_update)
        
        task_list = [exptempls.setup_polarization_experiment(current_dict) for n in range(num_experiment_repeats)]
        result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
        current_polarization_score = np.average([cu.calculate_rgtpase_polarity_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False)[0] for rc in result_cells])
            
        if current_polarization_score >= required_polarization_score:
            print "Success!"
            stop_criterion_met = True
            continue
            
    return current_best_update
    
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
    
    
    
def parameter_explorer_asymmetry_criteria(parameter_exploration_name, parameter_exploration_program, sequential=False, result_storage_folder="A:\\numba-ncc\\output", overwrite=False, seed=36, run=True, init_rho_gtpase_conditions=None):
    num_processes = 4
    
    assert(type(parameter_exploration_name) == str)
    
    storedir = os.path.join(result_storage_folder, parameter_exploration_name)
    storefile_path = os.path.join(storedir, "dataset.h5py")
    program_path = os.path.join(storedir, "parameter_exploration_program.pkl")
    
    if run == False:
        return storefile_path
    
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
            task_environment_defn = exptempls.setup_polarization_experiment(parameter_dict, seed=seed, init_rho_gtpase_conditions=init_rho_gtpase_conditions)
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
#    init_rho_gtpase_conditions = {'rac_membrane_active': np.array([  1.93310585e-02,   1.59639324e-02,   2.52662711e-02,
#         3.79913106e-03,   2.14234077e-02,   9.63795925e-03,
#         4.95031580e-03,   6.87662087e-03,   1.67108477e-02,
#         8.64009426e-04,   2.51349594e-02,   2.39962338e-02,
#         1.45977945e-02,   5.22764742e-03,   6.15702947e-03,
#         6.27815382e-05]), 'rac_membrane_inactive': np.array([ 0.00778487,  0.01292601,  0.01559152,  0.00847444,  0.01584577,
#        0.00724596,  0.02310397,  0.01555796,  0.02041079,  0.01126159,
#        0.00771375,  0.00742542,  0.00591246,  0.00078102,  0.02166378,
#        0.01830072]), 'rac_cytosolic_gdi_bound': np.array([ 0.6,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
#        0. ,  0. ,  0. ,  0. ,  0. ]), 'rho_cytosolic_gdi_bound': np.array([ 0.6,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
#        0. ,  0. ,  0. ,  0. ,  0. ]), 'rho_membrane_active': np.array([ 0.00759378,  0.01637452,  0.00543586,  0.01066832,  0.03114294,
#        0.00725198,  0.000793  ,  0.00470283,  0.01507525,  0.01103731,
#        0.01996065,  0.01308559,  0.00583466,  0.01097498,  0.01504583,
#        0.02502251]), 'rho_membrane_inactive': np.array([ 0.01155698,  0.01256599,  0.02504749,  0.00563587,  0.02070598,
#        0.01951374,  0.00589358,  0.00238163,  0.00594754,  0.02492706,
#        0.02056927,  0.01915446,  0.01148396,  0.00221784,  0.00662462,
#        0.005774  ]), 'coa_signal': np.zeros(16, dtype=np.float64)}
#    pe_name = "2017_MAR_20_PE_4"
    
#    exploration_program = [('kgtp_rac_multiplier', 1, 10, 3), ('kgtp_rho_multiplier', 1, 10, 3), ('kdgtp_rac_multiplier', 1, 10, 3), ('kdgtp_rho_multiplier', 1, 10, 3), ('kgtp_rho_autoact_multiplier', 100, 500, 3), ('kgtp_rac_autoact_multiplier', 100, 500, 3)]
#    p_labels = [x[0] for x in exploration_program]
#    storefile_path = parameter_explorer_asymmetry_criteria(pe_name, exploration_program, seed=36, run=False, init_rho_gtpase_conditions=init_rho_gtpase_conditions, sequential=False)
#    
#    results = hardio.get_parameter_exploration_results(storefile_path)
#    sorted_results = results[results[:,0].argsort()]
#    
#    best_update = get_result_as_dict_update(sorted_results, -1, p_labels)
    
    

    moddable_parameters = [('kgtp_rac_multiplier', 1., 20., 1.), ('kgtp_rho_multiplier', 1., 20., 1.), ('kdgtp_rac_multiplier', 1., 20., 1.), ('kdgtp_rho_multiplier', 1., 20., 1.), ('threshold_rac_activity_multiplier', 0.1, 0.8, 0.01), ('threshold_rho_activity_multiplier', 0.1, 0.8, 0.01), ('kgtp_rac_autoact_multiplier', 1., 200., 1.), ('kgtp_rho_autoact_multiplier', 1., 300., 1.), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1., 1000., 1.), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1., 1000., 1.), ('tension_mediated_rac_inhibition_half_strain', 0.01, 0.05, 0.005)]
    #best_update = parameter_explorer_polarization_wanderer(moddable_parameters, 0.7, num_new_dicts_to_generate=len(moddable_parameters), num_experiment_repeats=3, num_processes=6)
    best_update = parameter_explorer_polarization_wanderer(moddable_parameters, 0.7, num_new_dicts_to_generate=len(moddable_parameters), num_experiment_repeats=3, num_processes=6)
    #best_pd = parameter_explorer_polarization_slope_follower(moddable_parameters, 0.6, resolution=0.01)
    