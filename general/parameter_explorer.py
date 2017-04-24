from __future__ import division
import core.parameterorg as parameterorg
import numpy as np
import core.utilities as cu
import multiprocessing as multiproc
import general.experiment_templates as exptempls
import general.exec_utils as executils

import copy


# --------------------------------------------------------------------
BEST_UPDATES = []
closeness_dist_squared_criteria = (0.5e-6)**2
STANDARD_PARAMETER_DICT = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 20.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e100), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 9.0), ('randomization_node_percentage', 0.25), ('randomization_type', 'r'), ('coa_intersection_exponent', 2.0)])


def score_function(min_cutoff, max_cutoff, x):
    if x > max_cutoff:
        return 1.0
    elif x < min_cutoff:
        return 0.0
    else:
        return (x - min_cutoff)/(max_cutoff - min_cutoff)
    
#======================================================================

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

#=====================================================================
    

def generate_random_update_based_on_given_update(update, mod_deltas, mod_min_max):
    new_update = []
    
    for old_key_val, delta, this_min_max in zip(update, mod_deltas, mod_min_max):
        old_value = old_key_val[1]
        minj, maxj = this_min_max
        
        if old_value == minj:
            sign_choice = [0, 1]
        elif old_value == maxj:
            sign_choice = [-1, 0]
        else:
            sign_choice = [-1, 0, 1]
            
        new_value = old_value + np.random.choice(sign_choice)*np.random.choice([1, 1, 1, 1, 1, 2, 2, 2, 2., 5., 5., 5., 10., 10., 50.])*delta
        
        if new_value > maxj:
            new_value = maxj
        elif new_value < minj:
            new_value = minj
            
        new_update.append((old_key_val[0], new_value))
            
    return new_update

#=====================================================================

def generate_update_by_combining_two_updates(update0, update1, mod_deltas, mod_min_max, mutation_probability=-1.0):
    new_update = []
    
    for old_key_val0, old_key_val1, delta, this_min_max in zip(update0, update1, mod_deltas, mod_min_max):
        assert(old_key_val0[0] == old_key_val1[0])
        old_value0, old_value1 = old_key_val0[1], old_key_val1[1]
        
        minj, maxj = this_min_max
        
        new_value = 0.0
        combine_type = np.random.choice([0, 1])
        if combine_type == 0:
            new_value == old_value0
        elif combine_type == 1:
            new_value = old_value1
            
        if new_value > maxj:
            new_value = maxj
        elif new_value < minj:
            new_value = minj
                
        new_update.append((old_key_val0[0], new_value))
        
    if np.random.rand() < mutation_probability:
        new_update = generate_random_update_based_on_given_update(new_update, mod_deltas, mod_min_max)
        
            
    return new_update
    


#=====================================================================
    

def generate_random_updates_based_on_current_state(num_new_updates, parameter_dict, moddable_parameter_labels, mod_deltas, mod_min_max):
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
            new_value = old_value + np.random.choice(sign_choice)*np.random.choice([1, 1, 1, 1, 1, 2, 2, 2, 2., 5., 5., 5., 10., 10., 50.])*delta
            
            if new_value > maxj:
                new_value = maxj
            elif new_value < minj:
                new_value = minj
                
            parameter_updates[n].append((mpl, new_value))
            
    return parameter_updates

#=====================================================================
    

def generate_comprehensive_exploration_updates_based_on_current_given_update(given_update, moddable_parameter_labels, mod_deltas, mod_min_max, num_trials_to_generate_per_label=3):
    parameter_updates = []
    
    delta_factor_choice_list = [1, 2, 5, 10, 50]
    if num_trials_to_generate_per_label > len(delta_factor_choice_list):
        delta_factor_choice_list = int(np.ceil((num_trials_to_generate_per_label + 0.0)/len(delta_factor_choice_list)))*delta_factor_choice_list
    
    for index, zipped_data in enumerate(zip(moddable_parameter_labels, mod_deltas, mod_min_max)):
        mpl, delta, this_min_max = zipped_data
        
        assert(given_update[index][0] == mpl)
        old_value = given_update[index][1]
        minj, maxj = this_min_max
        
        delta_factor_choices = np.random.choice(delta_factor_choice_list, size=num_trials_to_generate_per_label, replace=False)
        if old_value <= minj:
            sign_choice = [1]
        elif old_value >= maxj:
            sign_choice = [-1]
        else:
            sign_choice = [-1, 1]
            
        for n in range(num_trials_to_generate_per_label):
            this_update = copy.deepcopy(given_update)
            sign = np.random.choice(sign_choice)
            new_value = old_value + sign*delta_factor_choices[n]*delta
            
            if new_value > maxj:
                new_value = maxj
            elif new_value < minj:
                new_value = minj
                
            this_update = this_update[:index] + [(mpl, new_value)] + this_update[(index + 1):]
            
            parameter_updates.append(this_update)
            
    return parameter_updates

#======================================================================

def run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, scores_and_updates_without_randomization=None, randomization=False, num_experiment_repeats=3, seeds=None, total_time_in_hours=3.):
    
    print "Preparing tasks (randomization={})...".format(randomization)
    task_list = []
    
    for i, u in enumerate(trial_updates):
        for n in range(num_experiment_repeats):
            if scores_and_updates_without_randomization != None:
                if scores_and_updates_without_randomization[i][0][0] > 0.0 and scores_and_updates_without_randomization[i][0][2] > 0.0:
                    trial_dict = copy.deepcopy(current_dict)
                    trial_dict.update(u)
                    if not randomization:
                        trial_dict.update([('randomization_scheme', None)])
                    task_list.append(exptempls.setup_polarization_experiment(trial_dict, total_time_in_hours=total_time_in_hours, seed=seeds[n]))
                else:
                    task_list.append(None)
            else:
                trial_dict = copy.deepcopy(current_dict)
                trial_dict.update(u)
                if not randomization:
                    trial_dict.update([('randomization_scheme', None)])
                task_list.append(exptempls.setup_polarization_experiment(trial_dict, total_time_in_hours=total_time_in_hours, seed=seeds[n]))
                
            

    if worker_pool != None:
        print "running tasks in parallel..."
        result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
    else:
        print "running tasks in sequence..."
        result_cells = [executils.run_simple_experiment_and_return_cell_worker(t) for t in task_list]
    
    print "analyzing results..."
    i = 0
    result_cells_per_pd = []
    while i < len(task_list):
        result_cells_per_pd.append(result_cells[i:i+num_experiment_repeats])
        i += num_experiment_repeats
        
    assert(len(result_cells_per_pd) == len(trial_updates))

    scores_and_updates = []        
    for result_cells_chunk, u in zip(result_cells_per_pd, trial_updates):
        scores = []
        for rc in result_cells_chunk:
            if rc != None:
                scores.append(cu.calculate_parameter_exploration_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False))
            else:
                scores.append([0.001, 0.001, 0.001])
        
        scores = np.average(scores, axis=0)
        scores_and_updates.append((scores, u))
        
    return scores_and_updates

#======================================================================

def run_and_score_trial_dicts_no_randomization_variant(worker_pool, current_dict, trial_updates, scores_and_updates_without_randomization=None, num_experiment_repeats=3, seeds=None, total_time_in_hours=3., should_be_polarized_by_in_hours=0.5):
    randomization = False
    print "Preparing tasks (randomization={})...".format(randomization)
    task_list = []
    
    for i, u in enumerate(trial_updates):
        for n in range(num_experiment_repeats):
            if scores_and_updates_without_randomization != None:
                if scores_and_updates_without_randomization[i][0][0] > 0.0 and scores_and_updates_without_randomization[i][0][2] > 0.0:
                    trial_dict = copy.deepcopy(current_dict)
                    trial_dict.update(u)
                    if not randomization:
                        trial_dict.update([('randomization_scheme', None)])
                    task_list.append(exptempls.setup_polarization_experiment(trial_dict, total_time_in_hours=total_time_in_hours, seed=seeds[n]))
                else:
                    task_list.append(None)
            else:
                trial_dict = copy.deepcopy(current_dict)
                trial_dict.update(u)
                if not randomization:
                    trial_dict.update([('randomization_scheme', None)])
                task_list.append(exptempls.setup_polarization_experiment(trial_dict, total_time_in_hours=total_time_in_hours, seed=seeds[n]))
                
            

    if worker_pool != None:
        print "running tasks in parallel..."
        result_cells = worker_pool.map(executils.run_simple_experiment_and_return_cell_worker, task_list)
    else:
        print "running tasks in sequence..."
        result_cells = [executils.run_simple_experiment_and_return_cell_worker(t) for t in task_list]
    
    print "analyzing results..."
    i = 0
    result_cells_per_pd = []
    while i < len(task_list):
        result_cells_per_pd.append(result_cells[i:i+num_experiment_repeats])
        i += num_experiment_repeats
        
    assert(len(result_cells_per_pd) == len(trial_updates))

    scores_and_updates = []        
    for result_cells_chunk, u in zip(result_cells_per_pd, trial_updates):
        scores = []
        for rc in result_cells_chunk:
            if rc != None:
                scores.append(cu.calculate_parameter_exploration_score_from_cell_no_randomization_variant(rc, should_be_polarized_by_in_hours=should_be_polarized_by_in_hours))
            else:
                scores.append([0.001, 0.001, 0.001])
        
        scores = np.average(scores, axis=0)
        scores_and_updates.append((scores, u))
        
    return scores_and_updates

#======================================================================

def rate_results_and_find_best_update(current_best_score, current_dict, current_best_update, scores_and_updates_no_randomization, best_updates, scores_and_updates_with_randomization=None):
    new_best_score = current_best_score
    new_best_update = current_best_update
    improvement = False
    
    for n in range(len(scores_and_updates_no_randomization)):
        if scores_and_updates_with_randomization == None:
            u = scores_and_updates_no_randomization[n][1]
        else:
            u = scores_and_updates_with_randomization[n][1]
            
        score_without_randomization = np.array(scores_and_updates_no_randomization[n][0])
        
        polarization_score_no_randomization, speed_score_no_randomization = score_without_randomization[0], score_without_randomization[2]
        
        polarization_score_no_randomization = score_function(0.0, 0.7, polarization_score_no_randomization)
        combined_score_no_randomization = polarization_score_no_randomization*speed_score_no_randomization
        
        combined_score = combined_score_no_randomization
        score_with_randomization = np.array([-1, -1, -1])
        if scores_and_updates_with_randomization != None:
            score_with_randomization = np.array(scores_and_updates_with_randomization[n][0])
            combined_score_with_randomization = score_with_randomization[0]*score_with_randomization[1]
            combined_score = combined_score*combined_score_with_randomization #np.average([combined_score_with_randomization, combined_score_no_randomization])
        
        if combined_score > new_best_score:
            print "possible new score: {}, {}, {}".format(combined_score, np.round(score_without_randomization, decimals=4), np.round(score_with_randomization, decimals=4))
            new_best_score = combined_score
            new_best_update = u
            current_dict.update(u)
            improvement = True
            best_updates += [(combined_score, new_best_update)]
            
    return new_best_score, new_best_update, current_dict, improvement, best_updates


#======================================================================

def rate_results_and_find_best_update_no_randomization_variant(current_best_score, current_dict, current_best_update, scores_and_updates_no_randomization, best_updates):
    new_best_score = current_best_score
    new_best_update = current_best_update
    improvement = False
    
    for n in range(len(scores_and_updates_no_randomization)):
        u = scores_and_updates_no_randomization[n][1]
        
        polarization_score_global, polarization_score_at_SBPBT, speed_score = scores_and_updates_no_randomization[n][0]
 
        #difference_between_pg_and_pat = polarization_score_global - polarization_score_at_SBPBT
        
        #difference_factor = 1. - score_function(0.0, 1.0, np.abs(difference_between_pg_and_pat))
        combined_score_no_randomization = polarization_score_global#difference_factor*polarization_score_global*speed_score
        
        if combined_score_no_randomization > new_best_score:
            print "possible new score: {}".format(combined_score_no_randomization)
            new_best_score = combined_score_no_randomization
            new_best_update = u
            current_dict.update(u)
            improvement = True
            best_updates += [(combined_score_no_randomization, new_best_update)]
            
    return new_best_score, new_best_update, current_dict, improvement, best_updates


#=====================================================================
    

def parameter_explorer_polarization_wanderer(modification_program, required_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=None, sequential=False, max_loops=100):
    
    global BEST_UPDATES
    
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
    
    default_dict_has_randomization_scheme = False
    if 'randomization_scheme' in current_dict:
        if current_dict['randomization_scheme'] != None:
            default_dict_has_randomization_scheme = True
            
    current_best_score = 0.0

    worker_pool = None
    if not sequential:
        worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
        
    stop_criterion_met = False
    num_tries_with_no_improvement = 0
    
    if type(seed) == int:
        num_experiment_repeats = 1
        seeds = [seed]*num_experiment_repeats
    elif seed == "auto-generate":
        seeds = np.random.random_integers(0, 10000, size=num_experiment_repeats)
    elif seed == None:
        seeds = [None]*num_experiment_repeats
    
    while not stop_criterion_met:
        improvement = False
        print "current best score: ", current_best_score
        
        print "preparing modded dicts..."
        trial_updates = generate_random_updates_based_on_current_state(num_new_dicts_to_generate, current_dict, mod_labels, mod_deltas, mod_min_max)
        
        scores_and_updates_no_randomization = run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=False, num_experiment_repeats=num_experiment_repeats_no_randomization, seeds=seeds, total_time_in_hours=total_time_in_hours)
        
        scores_and_updates_with_randomization = None
        if default_dict_has_randomization_scheme:
            scores_and_updates_with_randomization = run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=True, num_experiment_repeats=num_experiment_repeats, seeds=seeds, total_time_in_hours=total_time_in_hours, scores_and_updates_without_randomization=scores_and_updates_no_randomization)
            
        
        current_best_score, current_best_update, current_dict, improvement, BEST_UPDATES = rate_results_and_find_best_update(current_best_score, current_dict, current_best_update, scores_and_updates_no_randomization, BEST_UPDATES, scores_and_updates_with_randomization=scores_and_updates_with_randomization)
        
        if current_best_score > required_score:
            print "Success! Stop criterion met!"
            stop_criterion_met = True
            
        if improvement:
            print "improvement seen!"
            num_tries_with_no_improvement = 0
        else:
            print "no improvement. ({})".format(num_tries_with_no_improvement)
            num_tries_with_no_improvement += 1
            if num_tries_with_no_improvement > max_loops:
                print "no improvement seen for a while...retrying from new initial conditions!"
                print "Too many tries with no improvement. Stopping."
                stop_criterion_met = True
            
    return BEST_UPDATES


#=====================================================================
    

def parameter_explorer_polarization_wanderer_no_randomization_variant(modification_program, required_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, total_time_in_hours=2., seed=None, sequential=False, max_loops=50, should_be_polarized_by_in_hours=0.5):
    
    global BEST_UPDATES
    
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
    
    current_best_score = 0.0

    worker_pool = None
    if not sequential:
        worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
        
    stop_criterion_met = False
    num_tries_with_no_improvement = 0
    
    if type(seed) == int:
        num_experiment_repeats = 1
        seeds = [seed]*num_experiment_repeats
    elif seed == "auto-generate":
        seeds = np.random.random_integers(0, 10000, size=num_experiment_repeats)
    elif seed == None:
        seeds = [None]*num_experiment_repeats
    
    while not stop_criterion_met:
        improvement = False
        print "current best score: ", current_best_score
        
        print "preparing modded dicts..."
        trial_updates = generate_random_updates_based_on_current_state(num_new_dicts_to_generate, current_dict, mod_labels, mod_deltas, mod_min_max)
        
        scores_and_updates_no_randomization = run_and_score_trial_dicts_no_randomization_variant(worker_pool, current_dict, trial_updates, num_experiment_repeats=num_experiment_repeats, seeds=seeds, total_time_in_hours=total_time_in_hours, should_be_polarized_by_in_hours=should_be_polarized_by_in_hours)
        
        current_best_score, current_best_update, current_dict, improvement, BEST_UPDATES = rate_results_and_find_best_update_no_randomization_variant(current_best_score, current_dict, current_best_update, scores_and_updates_no_randomization, BEST_UPDATES)
        
        if current_best_score > required_score:
            print "Success! Stop criterion met!"
            stop_criterion_met = True
            
        if improvement:
            print "improvement seen!"
            num_tries_with_no_improvement = 0
        else:
            print "no improvement. ({})".format(num_tries_with_no_improvement)
            num_tries_with_no_improvement += 1
            if num_tries_with_no_improvement > max_loops:
                print "no improvement seen for a while...retrying from new initial conditions!"
                print "Too many tries with no improvement. Stopping."
                stop_criterion_met = True
            
    return BEST_UPDATES

# ====================================================================

def parameter_explorer_polarization_conservative_wanderer(modification_program, required_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=None, sequential=False):
    
    global BEST_UPDATES
    
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
    
    default_dict_has_randomization_scheme = False
    if 'randomization_scheme' in current_dict:
        if current_dict['randomization_scheme'] != None:
            default_dict_has_randomization_scheme = True
            
    current_best_score = 0.0

    worker_pool = None
    if not sequential:
        worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
        
    stop_criterion_met = False
    num_tries_with_no_improvement = 0
    
    if type(seed) == int:
        num_experiment_repeats = 1
        seeds = [seed]*num_experiment_repeats
    elif seed == "auto-generate":
        seeds = np.random.random_integers(0, 10000, size=num_experiment_repeats)
    elif seed == None:
        seeds = [None]*num_experiment_repeats
    
    while not stop_criterion_met:
        improvement = False
        print "current best score: ", current_best_score
        
        print "preparing modded dicts..."
        trial_updates = generate_comprehensive_exploration_updates_based_on_current_given_update(current_best_update, mod_labels, mod_deltas, mod_min_max)
        
        scores_and_updates_no_randomization = run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=False, num_experiment_repeats=num_experiment_repeats_no_randomization, seeds=seeds, total_time_in_hours=total_time_in_hours)
        
        scores_and_updates_with_randomization = None
        if default_dict_has_randomization_scheme:
            scores_and_updates_with_randomization = run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=True, num_experiment_repeats=num_experiment_repeats, seeds=seeds, total_time_in_hours=total_time_in_hours, scores_and_updates_without_randomization=scores_and_updates_no_randomization)
            
        
        current_best_score, current_best_update, current_dict, improvement, BEST_UPDATES = rate_results_and_find_best_update(current_best_score, current_dict, current_best_update, scores_and_updates_no_randomization, BEST_UPDATES, scores_and_updates_with_randomization=scores_and_updates_with_randomization)
        
        if current_best_score > required_score:
            print "Success! Stop criterion met!"
            stop_criterion_met = True
            
        if improvement:
            print "improvement seen!"
            num_tries_with_no_improvement = 0
        else:
            print "no improvement. ({})".format(num_tries_with_no_improvement)
            num_tries_with_no_improvement += 1
            if num_tries_with_no_improvement > 50:
                print "no improvement seen for a while...retrying from new initial conditions!"
                print "Too many tries with no improvement. Stopping."
                stop_criterion_met = True
            
    return BEST_UPDATES


#======================================================================

def rate_results_evolution(scores_and_updates_no_randomization, scores_and_updates_with_randomization=None):
    scores_and_updates = []
    for n in range(len(scores_and_updates_no_randomization)):
        if scores_and_updates_with_randomization == None:
            u = scores_and_updates_no_randomization[n][1]
        else:
            u = scores_and_updates_with_randomization[n][1]
            
        score_without_randomization = np.array(scores_and_updates_no_randomization[n][0])
        
        polarization_score_no_randomization, speed_score_no_randomization = score_without_randomization[0], score_without_randomization[2]
        
        if polarization_score_no_randomization > 0.8:
            polarization_score_no_randomization = 1.0
        combined_score_no_randomization = polarization_score_no_randomization*speed_score_no_randomization
        
        combined_score = combined_score_no_randomization
        score_with_randomization = np.array([-1, -1, -1])
        if scores_and_updates_with_randomization != None:
            score_with_randomization = np.array(scores_and_updates_with_randomization[n][0])
            combined_score_with_randomization = score_with_randomization[0]*score_with_randomization[1]
            combined_score = combined_score*combined_score_with_randomization #np.average([combined_score_with_randomization, combined_score_no_randomization])
        
        scores_and_updates.append((combined_score, u))
            
    return scores_and_updates

#======================================================================
        
def rate_overall_fitness(ordered_update_list):
    scores = [x[0] for x in ordered_update_list]
    return np.max(scores), np.median(scores), np.average(scores), np.min(scores)

def insert_into_ordered_update_list(score_and_update, ordered_update_list):
    this_score = score_and_update[0]
    
    insert_index = 0
    for x in ordered_update_list:
        if x[0] > this_score:
            insert_index += 1
            
    ordered_update_list.insert(insert_index, score_and_update)
    
    return ordered_update_list
            
def resize_ordered_update_list(ordered_update_list, max_size):
    ordered_update_list = ordered_update_list[:max_size]
    
    return ordered_update_list
    
def generate_trials_from_population(ordered_update_list, mod_deltas, mod_min_max, mutation_probability=-1.0):
    new_trial_updates = []
    
    population_size = len(ordered_update_list)
    indices = np.arange(population_size)
    scores = [x[0] for x in ordered_update_list]
    pass_probability = np.array(scores)/np.sum(scores)
    
    pairs = []
    for n in range(population_size):
        i0 = np.random.choice(indices, p=pass_probability)
        i1 = np.random.choice(indices, p=pass_probability)
        while i0 == i1:
            i1 = np.random.choice(indices, p=pass_probability)
        pairs.append((i0, i1))
    
    for pair in pairs:
        i0, i1 = pair
        new_trial_updates.append(generate_update_by_combining_two_updates(ordered_update_list[i0][1], ordered_update_list[i1][1], mod_deltas, mod_min_max, mutation_probability=mutation_probability))
        
    for n in indices:
        an_update = ordered_update_list[n][1]
        new_trial_updates.append(generate_random_update_based_on_given_update(an_update, mod_deltas, mod_min_max))
        
    return new_trial_updates
    

def parameter_explorer_polarization_evolution(modification_program, required_score, max_population_size=4, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=None, sequential=False, mutation_probability=-1.0, max_loops=10, init_population=None):
    
    global BEST_UPDATES
    
    mod_deltas = [x[3] for x in modification_program]
    mod_min_max = [x[1:3] for x in modification_program]
    mod_labels = [x[0] for x in modification_program]
    
    acceptable_labels = parameterorg.all_user_parameters_with_justifications.keys()
    
    for ml in mod_labels:
        if ml not in acceptable_labels:
            raise StandardError("{} not in acceptable parameter labels list.".format(ml))
        
    global STANDARD_PARAMETER_DICT
    standard_parameter_dict = copy.deepcopy(STANDARD_PARAMETER_DICT)
    ordered_update_list = []
    
    default_dict_has_randomization_scheme = False
    if 'randomization_scheme' in standard_parameter_dict:
        if standard_parameter_dict['randomization_scheme'] != None:
            default_dict_has_randomization_scheme = True
    
    worker_pool = None
    if not sequential:
        worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    
    if type(seed) == int:
        num_experiment_repeats = 1
        seeds = [seed]*num_experiment_repeats
    elif seed == "auto-generate":
        seeds = np.random.random_integers(0, 10000, size=num_experiment_repeats)
    elif seed == None:
        seeds = [None]*num_experiment_repeats
    
    #--------------------
    print "Preparing initial population..."
    init_updates = generate_random_updates_based_on_current_state(max_population_size, standard_parameter_dict, mod_labels, mod_deltas, mod_min_max)
    
    if init_population != None:
        init_updates += [x[1] for x in init_population]
            
    scores_and_updates_no_randomization = run_and_score_trial_dicts(worker_pool, standard_parameter_dict, init_updates, randomization=False, num_experiment_repeats=num_experiment_repeats_no_randomization, seeds=seeds, total_time_in_hours=total_time_in_hours)
    
    scores_and_updates_with_randomization = None
    if default_dict_has_randomization_scheme:
        scores_and_updates_with_randomization = run_and_score_trial_dicts(worker_pool, standard_parameter_dict, init_updates, randomization=True, num_experiment_repeats=num_experiment_repeats, seeds=seeds, total_time_in_hours=total_time_in_hours, scores_and_updates_without_randomization=scores_and_updates_no_randomization)
        
    scores_and_updates = rate_results_evolution(scores_and_updates_no_randomization, scores_and_updates_with_randomization=scores_and_updates_with_randomization)
    
    for su in scores_and_updates:
        ordered_update_list = insert_into_ordered_update_list(su, ordered_update_list)

    #--------------------
    
    max_pop_score, med_pop_score, avg_pop_score, min_pop_score = rate_overall_fitness(ordered_update_list)
    
    print "max, med, avg, min: {}, {}, {}, {}".format(np.round(max_pop_score, decimals=4), np.round(med_pop_score, decimals=4), np.round(avg_pop_score, decimals=4), np.round(min_pop_score, decimals=4))
    
    num_loops = 0
    num_loops_with_no_improvement = 0
    while max_pop_score < required_score and num_loops_with_no_improvement < 1000:
        print "======================================="
        print "Loop: {}".format(num_loops)
        print "no improvement: {}".format(num_loops_with_no_improvement)
        new_trial_updates = generate_trials_from_population(ordered_update_list, mod_deltas, mod_min_max, mutation_probability=mutation_probability)
        
        scores_and_updates_no_randomization = run_and_score_trial_dicts(worker_pool, standard_parameter_dict, new_trial_updates, randomization=False, num_experiment_repeats=num_experiment_repeats_no_randomization, seeds=seeds, total_time_in_hours=total_time_in_hours)
    
        scores_and_updates_with_randomization = None
        if default_dict_has_randomization_scheme:
            scores_and_updates_with_randomization = run_and_score_trial_dicts(worker_pool, standard_parameter_dict, new_trial_updates, randomization=True, num_experiment_repeats=num_experiment_repeats, seeds=seeds, total_time_in_hours=total_time_in_hours, scores_and_updates_without_randomization=scores_and_updates_no_randomization)
            
        scores_and_updates = rate_results_evolution(scores_and_updates_no_randomization, scores_and_updates_with_randomization=scores_and_updates_with_randomization)
    
        for su in scores_and_updates:
            ordered_update_list = insert_into_ordered_update_list(su, ordered_update_list)
        
        ordered_update_list = resize_ordered_update_list(ordered_update_list, max_population_size)
        
        new_max_pop_score, new_med_pop_score, new_avg_pop_score, new_min_pop_score = rate_overall_fitness(ordered_update_list)
        
        if new_max_pop_score > max_pop_score or new_avg_pop_score > avg_pop_score or new_min_pop_score > min_pop_score:
            num_loops_with_no_improvement = 0
        else:
            num_loops_with_no_improvement += 1
            
        max_pop_score, med_pop_score, avg_pop_score, min_pop_score = new_max_pop_score, new_med_pop_score, new_avg_pop_score, new_min_pop_score
            
        print "max, med, avg, min: {}, {}, {}, {}".format(np.round(max_pop_score, decimals=4), np.round(med_pop_score, decimals=4), np.round(avg_pop_score, decimals=4), np.round(min_pop_score, decimals=4))
        
        num_loops += 1
        
        BEST_UPDATES = copy.deepcopy(ordered_update_list)
        print "======================================="
        
    return ordered_update_list
        
    
if __name__ == '__main__':
    
    moddable_parameters = [('kgtp_rac_multiplier', 1.0, 40.0, 1.0),
 ('kgtp_rho_multiplier', 1.0, 40.0, 1.0),
 ('kdgtp_rac_multiplier', 1.0, 40.0, 1.0),
 ('kdgtp_rho_multiplier', 1.0, 40.0, 1.0),
 ('threshold_rac_activity_multiplier', 0.1, 0.8, 0.01),
 ('threshold_rho_activity_multiplier', 0.1, 0.8, 0.01),
 ('kgtp_rac_autoact_multiplier', 1.0, 300.0, 5.0),
 ('kgtp_rho_autoact_multiplier', 1.0, 300.0, 5.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 100., 2000., 100.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 100., 2000., 100.)]
    
#    moddable_parameters = [('kgtp_rac_multiplier', 1.0, 40.0, 1.0),
# ('kgtp_rho_multiplier', 1.0, 40.0, 1.0),
# ('kdgtp_rac_multiplier', 1.0, 40.0, 1.0),
# ('kdgtp_rho_multiplier', 1.0, 40.0, 1.0),
# ('threshold_rac_activity_multiplier', 0.1, 0.8, 0.05),
# ('threshold_rho_activity_multiplier', 0.1, 0.8, 0.05),
# ('kgtp_rac_autoact_multiplier', 1.0, 300.0, 5.0),
# ('kgtp_rho_autoact_multiplier', 1.0, 300.0, 5.0),
# ('kdgtp_rac_mediated_rho_inhib_multiplier', 100., 2000., 100.),
# ('kdgtp_rho_mediated_rac_inhib_multiplier', 100., 2000., 100.), 
# ('tension_mediated_rac_inhibition_half_strain', 0.01, 0.1, 0.005),
#  ('randomization_time_mean', 1.0, 40.0, 2.0),
#  ('randomization_time_variance_factor', 0.01, 0.5, 0.02),
#  ('randomization_magnitude', 2.0, 20.0, 1.0), ('stiffness_edge', 1000.0, 8000.0, 500.0), ('randomization_node_percentage', 0.25, 0.5, 0.05)]
    
    
    
    #BEST_UPDATES = parameter_explorer_polarization_evolution(moddable_parameters, 0.8, max_population_size=len(moddable_parameters), task_chunk_size=4, num_processes=4, num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, total_time_in_hours=3., seed=2836, sequential=False, mutation_probability=0.1, init_population=None)
    
    #BEST_UPDATES = parameter_explorer_polarization_multiwanderer(moddable_parameters, 0.8, max_population_size=3, task_chunk_size=4, num_processes=4, num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, total_time_in_hours=3., seed=2836, init_population=None)
    
    BEST_UPDATES = parameter_explorer_polarization_wanderer_no_randomization_variant(moddable_parameters, 0.8, num_new_dicts_to_generate=int(len(moddable_parameters)), num_experiment_repeats=1, num_processes=4, total_time_in_hours=1.5, seed=2836, sequential=False, should_be_polarized_by_in_hours=0.5)
    
    #BEST_UPDATES = parameter_explorer_polarization_conservative_wanderer(moddable_parameters, 0.8, num_new_dicts_to_generate=len(moddable_parameters), num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, num_processes=4, total_time_in_hours=3., seed=2836, sequential=False)
