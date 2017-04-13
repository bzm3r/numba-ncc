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
STANDARD_PARAMETER_DICT = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 20.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm'), ('randomization_time_mean', 40.0), ('randomization_time_variance_factor', 0.1), ('randomization_magnitude', 9.0)])


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
        combine_type = np.random.choice([0, 1, 2])
        if combine_type == 0:
            new_value == old_value0
        elif combine_type == 1:
            new_value = old_value1
        elif combine_type == 2:
            new_value = (old_value0 + old_value1)/2.
            
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
        
        polarization_score_no_randomization, velocity_score_no_randomization = score_without_randomization[0], score_without_randomization[2]
        
        polarization_score_no_randomization = score_function(0.0, 0.7, polarization_score_no_randomization)
        combined_score_no_randomization = polarization_score_no_randomization*velocity_score_no_randomization
        
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

#=====================================================================
    

def parameter_explorer_polarization_wanderer(modification_program, required_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=None, sequential=False):
    
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
            if num_tries_with_no_improvement > 50:
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
        
        polarization_score_no_randomization, velocity_score_no_randomization = score_without_randomization[0], score_without_randomization[2]
        
        if polarization_score_no_randomization > 0.8:
            polarization_score_no_randomization = 1.0
        combined_score_no_randomization = polarization_score_no_randomization*velocity_score_no_randomization
        
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
    pairs = []
    
    n = 0
    num_given_updates = len(ordered_update_list)
    while n < num_given_updates - 1:
        pairs.append((ordered_update_list[n], ordered_update_list[(n + 1)%num_given_updates]))
        n += 1
    
    for su in ordered_update_list:
        new_trial_updates.append(generate_random_update_based_on_given_update(su[1], mod_deltas, mod_min_max))
        
    for pair in pairs:
        for m in range(2):
            new_trial_updates.append(generate_update_by_combining_two_updates(pair[0][1], pair[1][1], mod_deltas, mod_min_max, mutation_probability=mutation_probability))
        
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


def parameter_exploration_polarization_multiwanderer(modification_program, required_score, population_size=3, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=None, sequential=False, mutation_probability=-1.0, max_loops=10, init_population=None):
    
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
    #moddable_parameters = [('kgtp_rac_multiplier', 1., 15., 1.), ('kgtp_rho_multiplier', 1., 15., 1.), ('kdgtp_rac_multiplier', 1., 25., 1.), ('kdgtp_rho_multiplier', 1., 25., 1.), ('threshold_rac_activity_multiplier', 0.1, 0.8, 0.01), ('threshold_rho_activity_multiplier', 0.1, 0.8, 0.01), ('kgtp_rac_autoact_multiplier', 1., 1000., 10.), ('kgtp_rho_autoact_multiplier', 1., 1000., 10.), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1., 1000., 10.), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1., 1000., 10.), ('tension_mediated_rac_inhibition_half_strain', 0.01, 0.1 , 0.005), ('stiffness_edge', 1000, 8000, 100), ('max_force_rac', 0.1*10e3, 2*10e3, 0.1*10e3), ('eta', 0.41*1e5, 1.6*1e5, 0.1*1e5), ('randomization_time_mean', 30.0, 40.0, 1.0), ('randomization_time_variance_factor', 0.1, 0.1, 0.1), ('randomization_magnitude', 2.0, 10.0, 1.0)]
    
    init_population = [(0.35519143418835003,
  [('kgtp_rac_multiplier', 6.5625),
   ('kgtp_rho_multiplier', 6.0),
   ('kdgtp_rac_multiplier', 1.5),
   ('kdgtp_rho_multiplier', 13.484375),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 95.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 7.5),
   ('stiffness_edge', 7750.0)]),
 (0.25446699043427751,
  [('kgtp_rac_multiplier', 5.0),
   ('kgtp_rho_multiplier', 7.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 11.5),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 50.0),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 9.0),
   ('stiffness_edge', 7500.0)]),
 (0.22536790910000806,
  [('kgtp_rac_multiplier', 8.125),
   ('kgtp_rho_multiplier', 1.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 13.46875),
   ('threshold_rac_activity_multiplier', 0.41875000000000007),
   ('threshold_rho_activity_multiplier', 0.29843750000000002),
   ('kgtp_rac_autoact_multiplier', 140.0),
   ('kgtp_rho_autoact_multiplier', 100.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 525.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.047500000000000001),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 8000.0)]),
 (0.22004538676502869,
  [('kgtp_rac_multiplier', 8.0),
   ('kgtp_rho_multiplier', 13.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 18.5),
   ('threshold_rac_activity_multiplier', 0.40000000000000002),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 165.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.065000000000000002),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 4750.0)]),
 (0.1985120419596057,
  [('kgtp_rac_multiplier', 5.0),
   ('kgtp_rho_multiplier', 6.0),
   ('kdgtp_rac_multiplier', 2.0),
   ('kdgtp_rho_multiplier', 13.5),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 50.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.075000000000000011),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 7500.0)]),
 (0.18253728676967537,
  [('kgtp_rac_multiplier', 10.125),
   ('kgtp_rho_multiplier', 5.25),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 11.46875),
   ('threshold_rac_activity_multiplier', 0.46875000000000006),
   ('threshold_rho_activity_multiplier', 0.34843750000000001),
   ('kgtp_rac_autoact_multiplier', 150.0),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 325.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.047500000000000001),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 7250.0)]),
 (0.18246727311386882,
  [('kgtp_rac_multiplier', 7.5),
   ('kgtp_rho_multiplier', 13.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 17.75),
   ('threshold_rac_activity_multiplier', 0.40000000000000002),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 165.0),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.060000000000000005),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 4750.0)]),
 (0.16947133461562713,
  [('kgtp_rac_multiplier', 9.0),
   ('kgtp_rho_multiplier', 7.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 17.5),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 200.0),
   ('kgtp_rho_autoact_multiplier', 90.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 1100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 800.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.070000000000000007),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 5000.0)]),
 (0.16944531236159341,
  [('kgtp_rac_multiplier', 12.25),
   ('kgtp_rho_multiplier', 6.5),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 6.4375),
   ('threshold_rac_activity_multiplier', 0.41875000000000007),
   ('threshold_rho_activity_multiplier', 0.546875),
   ('kgtp_rac_autoact_multiplier', 200.0),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 550.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.072500000000000009),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 7.5),
   ('stiffness_edge', 4750.0)]),
 (0.16932840659956006,
  [('kgtp_rac_multiplier', 7.0),
   ('kgtp_rho_multiplier', 9.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 17.0),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 150.0),
   ('kgtp_rho_autoact_multiplier', 95.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.070000000000000007),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 8.0),
   ('stiffness_edge', 5000.0)]),
 (0.16338991513978998,
  [('kgtp_rac_multiplier', 8.0),
   ('kgtp_rho_multiplier', 13.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 18.5),
   ('threshold_rac_activity_multiplier', 0.40000000000000002),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 180.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.060000000000000005),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 4500.0)]),
 (0.16250092507327238,
  [('kgtp_rac_multiplier', 6.5625),
   ('kgtp_rho_multiplier', 1.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 12.484375),
   ('threshold_rac_activity_multiplier', 0.41875000000000007),
   ('threshold_rho_activity_multiplier', 0.29843750000000002),
   ('kgtp_rac_autoact_multiplier', 140.0),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 525.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.047500000000000001),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 7.0),
   ('stiffness_edge', 7750.0)]),
 (0.15702684787426049,
  [('kgtp_rac_multiplier', 8.0),
   ('kgtp_rho_multiplier', 14.0),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 17.5),
   ('threshold_rac_activity_multiplier', 0.45000000000000001),
   ('threshold_rho_activity_multiplier', 0.30000000000000004),
   ('kgtp_rac_autoact_multiplier', 160.0),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.060000000000000005),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 9.0),
   ('stiffness_edge', 5000.0)]),
 (0.15464542569507655,
  [('kgtp_rac_multiplier', 8.0),
   ('kgtp_rho_multiplier', 9.0),
   ('kdgtp_rac_multiplier', 3.0),
   ('kdgtp_rho_multiplier', 16.5),
   ('threshold_rac_activity_multiplier', 0.1),
   ('threshold_rho_activity_multiplier', 0.25000000000000006),
   ('kgtp_rac_autoact_multiplier', 100.0),
   ('kgtp_rho_autoact_multiplier', 100.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 100.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.045000000000000005),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 8.0),
   ('stiffness_edge', 4000.0)]),
 (0.15380426321998153,
  [('kgtp_rac_multiplier', 12.25),
   ('kgtp_rho_multiplier', 1.5),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 5.4375),
   ('threshold_rac_activity_multiplier', 0.49375000000000002),
   ('threshold_rho_activity_multiplier', 0.44687499999999997),
   ('kgtp_rac_autoact_multiplier', 210.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 150.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 550.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.047500000000000001),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 7750.0)]),
 (0.15350624545206623,
  [('kgtp_rac_multiplier', 8.0),
   ('kgtp_rho_multiplier', 14.0),
   ('kdgtp_rac_multiplier', 2.0),
   ('kdgtp_rho_multiplier', 18.5),
   ('threshold_rac_activity_multiplier', 0.35000000000000003),
   ('threshold_rho_activity_multiplier', 0.30000000000000004),
   ('kgtp_rac_autoact_multiplier', 190.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 200.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.085000000000000006),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 5500.0)])]
    
    init_population2 = [(0.37536006104607722,
  [('kgtp_rac_multiplier', 12.2783203125),
   ('kgtp_rho_multiplier', 4.9375),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 5.0),
   ('threshold_rac_activity_multiplier', 0.43999023437499996),
   ('threshold_rho_activity_multiplier', 0.659912109375),
   ('kgtp_rac_autoact_multiplier', 101.6796875),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1430.078125),
   ('tension_mediated_rac_inhibition_half_strain', 0.051250000000000004),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 5.0),
   ('stiffness_edge', 7925.78125)]),
 (0.34624753110785872,
  [('kgtp_rac_multiplier', 12.68310546875),
   ('kgtp_rho_multiplier', 4.375),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 4.0),
   ('threshold_rac_activity_multiplier', 0.4403076171875),
   ('threshold_rho_activity_multiplier', 0.65312499999999996),
   ('kgtp_rac_autoact_multiplier', 50.0),
   ('kgtp_rho_autoact_multiplier', 150.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1403.90625),
   ('tension_mediated_rac_inhibition_half_strain', 0.051250000000000004),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 6.625),
   ('stiffness_edge', 7925.78125)]),
 (0.34599138328594631,
  [('kgtp_rac_multiplier', 12.68310546875),
   ('kgtp_rho_multiplier', 4.375),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 4.0),
   ('threshold_rac_activity_multiplier', 0.44034729003906248),
   ('threshold_rho_activity_multiplier', 0.65482177734374991),
   ('kgtp_rac_autoact_multiplier', 62.919921875),
   ('kgtp_rho_autoact_multiplier', 150.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1410.44921875),
   ('tension_mediated_rac_inhibition_half_strain', 0.051250000000000004),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 6.7265625),
   ('stiffness_edge', 7925.78125)]),
 (0.33924835934238395,
  [('kgtp_rac_multiplier', 12.480712890625),
   ('kgtp_rho_multiplier', 4.65625),
   ('kdgtp_rac_multiplier', 1.0),
   ('kdgtp_rho_multiplier', 4.5),
   ('threshold_rac_activity_multiplier', 0.44014892578124998),
   ('threshold_rho_activity_multiplier', 0.65651855468749998),
   ('kgtp_rac_autoact_multiplier', 75.83984375),
   ('kgtp_rho_autoact_multiplier', 150.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1416.9921875),
   ('tension_mediated_rac_inhibition_half_strain', 0.051250000000000004),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 6.625),
   ('stiffness_edge', 7925.78125)])]
    
    init_population3 = [(0.27619686490822898,
  [('kgtp_rac_multiplier', 12.5),
   ('kgtp_rho_multiplier', 18.0),
   ('kdgtp_rac_multiplier', 5.0),
   ('kdgtp_rho_multiplier', 20.0),
   ('threshold_rac_activity_multiplier', 0.5),
   ('threshold_rho_activity_multiplier', 0.67500000000000004),
   ('kgtp_rac_autoact_multiplier', 300.0),
   ('kgtp_rho_autoact_multiplier', 70.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.10000000000000001),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 7.0),
   ('stiffness_edge', 6500.0)]),
 (0.25638045769747858,
  [('kgtp_rac_multiplier', 9.25),
   ('kgtp_rho_multiplier', 16.0),
   ('kdgtp_rac_multiplier', 6.0),
   ('kdgtp_rho_multiplier', 16.0),
   ('threshold_rac_activity_multiplier', 0.5),
   ('threshold_rho_activity_multiplier', 0.70625000000000004),
   ('kgtp_rac_autoact_multiplier', 281.5625),
   ('kgtp_rho_autoact_multiplier', 120.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 150.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.099609375),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 8.25),
   ('stiffness_edge', 8000.0)]),
 (0.25115679673281049,
  [('kgtp_rac_multiplier', 16.0),
   ('kgtp_rho_multiplier', 19.5),
   ('kdgtp_rac_multiplier', 5.0),
   ('kdgtp_rho_multiplier', 20.0),
   ('threshold_rac_activity_multiplier', 0.52500000000000002),
   ('threshold_rho_activity_multiplier', 0.55000000000000004),
   ('kgtp_rac_autoact_multiplier', 285.0),
   ('kgtp_rho_autoact_multiplier', 127.5),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.093437500000000007),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 7.0),
   ('stiffness_edge', 6750.0)]),
 (0.25084390576519189,
  [('kgtp_rac_multiplier', 12.625),
   ('kgtp_rho_multiplier', 19.5),
   ('kdgtp_rac_multiplier', 5.5),
   ('kdgtp_rho_multiplier', 20.0),
   ('threshold_rac_activity_multiplier', 0.51249999999999996),
   ('threshold_rho_activity_multiplier', 0.62812500000000004),
   ('kgtp_rac_autoact_multiplier', 285.0),
   ('kgtp_rho_autoact_multiplier', 123.75),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.093437500000000007),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 7.625),
   ('stiffness_edge', 7375.0)]),
 (0.24691619485031557,
  [('kgtp_rac_multiplier', 15.0),
   ('kgtp_rho_multiplier', 19.5),
   ('kdgtp_rac_multiplier', 6.0),
   ('kdgtp_rho_multiplier', 19.0),
   ('threshold_rac_activity_multiplier', 0.5),
   ('threshold_rho_activity_multiplier', 0.55000000000000004),
   ('kgtp_rac_autoact_multiplier', 275.625),
   ('kgtp_rho_autoact_multiplier', 60.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 200.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.099218750000000008),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 6.5),
   ('stiffness_edge', 7000.0)]),
 (0.24637573595320181,
  [('kgtp_rac_multiplier', 9.25),
   ('kgtp_rho_multiplier', 16.0),
   ('kdgtp_rac_multiplier', 6.0),
   ('kdgtp_rho_multiplier', 16.0),
   ('threshold_rac_activity_multiplier', 0.5),
   ('threshold_rho_activity_multiplier', 0.70625000000000004),
   ('kgtp_rac_autoact_multiplier', 287.5),
   ('kgtp_rho_autoact_multiplier', 180.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.1),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.10000000000000001),
   ('randomization_magnitude', 10.0),
   ('stiffness_edge', 8000.0)]),
 (0.24323618788707754,
  [('kgtp_rac_multiplier', 13.5),
   ('kgtp_rho_multiplier', 18.0),
   ('kdgtp_rac_multiplier', 5.0),
   ('kdgtp_rho_multiplier', 19.5),
   ('threshold_rac_activity_multiplier', 0.5),
   ('threshold_rho_activity_multiplier', 0.625),
   ('kgtp_rac_autoact_multiplier', 267.5),
   ('kgtp_rho_autoact_multiplier', 50.0),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.087500000000000008),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 6.5),
   ('stiffness_edge', 8000.0)]),
 (0.24081601614500786,
  [('kgtp_rac_multiplier', 12.625),
   ('kgtp_rho_multiplier', 19.5),
   ('kdgtp_rac_multiplier', 5.25),
   ('kdgtp_rho_multiplier', 20.0),
   ('threshold_rac_activity_multiplier', 0.51249999999999996),
   ('threshold_rho_activity_multiplier', 0.58906250000000004),
   ('kgtp_rac_autoact_multiplier', 285.0),
   ('kgtp_rho_autoact_multiplier', 123.75),
   ('kdgtp_rac_mediated_rho_inhib_multiplier', 150.0),
   ('kdgtp_rho_mediated_rac_inhib_multiplier', 1900.0),
   ('tension_mediated_rac_inhibition_half_strain', 0.093437500000000007),
   ('randomization_time_mean', 40.0),
   ('randomization_time_variance_factor', 0.1),
   ('randomization_magnitude', 7.625),
   ('stiffness_edge', 7375.0)])]
    
    [('kgtp_rac_multiplier', 39.0),
  ('kgtp_rho_multiplier', 28.0),
  ('kdgtp_rac_multiplier', 36.0),
  ('kdgtp_rho_multiplier', 17.0),
  ('threshold_rac_activity_multiplier', 0.44999999999999996),
  ('threshold_rho_activity_multiplier', 0.44999999999999996),
  ('kgtp_rac_autoact_multiplier', 230.0),
  ('kgtp_rho_autoact_multiplier', 150.0),
  ('kdgtp_rac_mediated_rho_inhib_multiplier', 100.0),
  ('kdgtp_rho_mediated_rac_inhib_multiplier', 2000.0),
  ('tension_mediated_rac_inhibition_half_strain', 0.060000000000000005),
  ('randomization_time_mean', 35.0),
  ('randomization_time_variance_factor', 0.1),
  ('randomization_magnitude', 10.0),
  ('stiffness_edge', 8000.0)]
    
    moddable_parameters = [('kgtp_rac_multiplier', 1.0, 20.0, 1.0),
 ('kgtp_rho_multiplier', 1.0, 20.0, 1.0),
 ('kdgtp_rac_multiplier', 1.0, 40.0, 1.0),
 ('kdgtp_rho_multiplier', 1.0, 40.0, 1.0),
 ('threshold_rac_activity_multiplier', 0.1, 0.8, 0.05),
 ('threshold_rho_activity_multiplier', 0.1, 0.8, 0.05),
 ('kgtp_rac_autoact_multiplier', 5.0, 300.0, 5.0),
 ('kgtp_rho_autoact_multiplier', 5.0, 300.0, 5.0),
 ('kdgtp_rac_mediated_rho_inhib_multiplier', 100., 2000., 100.),
 ('kdgtp_rho_mediated_rac_inhib_multiplier', 100., 2000., 100.), 
 ('tension_mediated_rac_inhibition_half_strain', 0.01, 0.1, 0.005),
  ('randomization_time_mean', 40.0, 40.0, 5.0),
  ('randomization_time_variance_factor', 0.1, 0.1, 0.1),
  ('randomization_magnitude', 2.0, 10.0, 1.0), ('stiffness_edge', 1000.0, 8000.0, 500.0)]
    
    
    
    #BEST_UPDATES = parameter_explorer_polarization_evolution(moddable_parameters, 0.8, max_population_size=3, task_chunk_size=4, num_processes=4, num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=2836, sequential=True, mutation_probability=-1, init_population=None)
    
    #BEST_UPDATES = parameter_explorer_polarization_multiwanderer(moddable_parameters, 0.8, max_population_size=3, task_chunk_size=4, num_processes=4, num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, total_time_in_hours=3., seed=2836, init_population=None)
    
    BEST_UPDATES = parameter_explorer_polarization_wanderer(moddable_parameters, 0.8, num_new_dicts_to_generate=len(moddable_parameters), num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, num_processes=4, total_time_in_hours=3., seed=2836, sequential=False)
    
    #BEST_UPDATES = parameter_explorer_polarization_conservative_wanderer(moddable_parameters, 0.8, num_new_dicts_to_generate=len(moddable_parameters), num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, num_processes=4, total_time_in_hours=3., seed=2836, sequential=False)
