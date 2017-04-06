from __future__ import division
import core.parameterorg as parameterorg
import numpy as np
import core.utilities as cu
import multiprocessing as multiproc
import general.experiment_templates as exptempls
import general.exec_utils as executils

import copy


# --------------------------------------------------------------------
closeness_dist_squared_criteria = (0.5e-6)**2
STANDARD_PARAMETER_DICT = dict([('num_nodes', 16), ('init_cell_radius', 20e-6), ('C_total', 2.5e6), ('H_total', 1e6), ('init_rgtpase_cytosol_frac', 0.6), ('init_rgtpase_membrane_active_frac', 0.2), ('init_rgtpase_membrane_inactive_frac', 0.2), ('diffusion_const', 0.1e-12), ('kgdi_multiplier', 1), ('kdgdi_multiplier', 1), ('kgtp_rac_multiplier', 1.0), ('kgtp_rac_autoact_multiplier', 200), ('kdgtp_rac_multiplier', 5.0), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1000), ('threshold_rac_activity_multiplier', 0.4), ('kgtp_rho_multiplier', 10.0), ('kgtp_rho_autoact_multiplier', 100), ('kdgtp_rho_multiplier', 2.5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1000.0), ('threshold_rho_activity_multiplier', 0.4), ('hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('max_coa_signal', -1.0), ('coa_sensing_dist_at_value', 110e-6), ('coa_sensing_value_at_dist', 0.5), ('interaction_factor_migr_bdry_contact', 20.), ('closeness_dist_squared_criteria', closeness_dist_squared_criteria), ('length_3D_dimension', 10e-6), ('stiffness_edge', 5000), ('stiffness_cytoplasmic', 1e-5), ('eta', 1e5), ('max_force_rac', 10e3), ('force_rho_multiplier', 0.2), ('force_adh_const', 0.0), ('skip_dynamics', False), ('randomization_scheme', 'm')])

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

#======================================================================
    

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
            new_value = old_value + np.random.choice(sign_choice)*np.random.choice([1, 1, 1, 1, 1, 2, 2, 2, 2, 5, 5, 5, 10, 10, 100])*delta
            
            if new_value > maxj:
                new_value = maxj
            elif new_value < minj:
                new_value = minj
                
            parameter_updates[n].append((mpl, new_value))
            
    return parameter_updates

#======================================================================

def run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=False, num_experiment_repeats=3, seeds=None, total_time_in_hours=3.):
    
    print "Preparing tasks (randomization={})...".format(randomization)
    task_list = []
    
    for u in trial_updates:
        for n in range(num_experiment_repeats):
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
        scores = [cu.calculate_parameter_exploration_score_from_cell(rc, significant_difference=0.2, weigh_by_timepoint=False) for rc in result_cells_chunk]
        
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
        combined_score_no_randomization = score_without_randomization[0]
        
        combined_score = combined_score_no_randomization
        score_with_randomization = np.array([-1, -1, -1])
        if scores_and_updates_with_randomization != None:
            score_with_randomization = np.array(scores_and_updates_with_randomization[n][0])
            combined_score_with_randomization = score_with_randomization[0]*score_with_randomization[1]
            combined_score = np.average([combined_score_with_randomization, combined_score_no_randomization])
        
        if combined_score > new_best_score:
            print "possible new score: {}, {}, {}".format(combined_score, np.round(score_without_randomization, decimals=4), np.round(score_with_randomization, decimals=4))
            new_best_score = combined_score
            new_best_update = u
            current_dict.update(u)
            improvement = True
            best_updates += [(combined_score_no_randomization, combined_score_with_randomization, new_best_update)]
            
    return new_best_score, new_best_update, current_dict, improvement, best_updates


#======================================================================
    

def parameter_explorer_polarization_wanderer(modification_program, required_score, num_new_dicts_to_generate=8, task_chunk_size=4, num_processes=4, num_experiment_repeats=3, num_experiment_repeats_no_randomization=1, total_time_in_hours=2., seed=None, sequential=False):
    
    best_updates = []
    
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
        trial_updates = generate_random_update_based_on_current_state(num_new_dicts_to_generate, current_dict, mod_labels, mod_deltas, mod_min_max)
        
        scores_and_updates_no_randomization = run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=False, num_experiment_repeats=num_experiment_repeats_no_randomization, seeds=seeds, total_time_in_hours=total_time_in_hours)
        
        scores_and_updates_with_randomization = None
        if default_dict_has_randomization_scheme:
            scores_and_updates_with_randomization = run_and_score_trial_dicts(worker_pool, current_dict, trial_updates, randomization=True, num_experiment_repeats=num_experiment_repeats, seeds=seeds, total_time_in_hours=total_time_in_hours)
            
        
        current_best_score, current_best_update, current_dict, improvement, best_updates = rate_results_and_find_best_update(current_best_score, current_dict, current_best_update, scores_and_updates_no_randomization, best_updates, scores_and_updates_with_randomization=scores_and_updates_with_randomization)
        
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
            
    return best_updates

    
if __name__ == '__main__':
    moddable_parameters = [('kgtp_rac_multiplier', 1., 500., 1.), ('kgtp_rho_multiplier', 1., 500., 1.), ('kdgtp_rac_multiplier', 1., 500., 1.), ('kdgtp_rho_multiplier', 1., 500., 1.), ('threshold_rac_activity_multiplier', 0.1, 0.8, 0.01), ('threshold_rho_activity_multiplier', 0.1, 0.8, 0.01), ('kgtp_rac_autoact_multiplier', 1., 1000., 1.), ('kgtp_rho_autoact_multiplier', 1., 1000., 1.), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1., 2000., 1.), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1., 2000., 1.), ('tension_mediated_rac_inhibition_half_strain', 0.01, 0.5, 0.1), ('stiffness_edge', 1000, 8000, 500), ('max_force_rac', 0.5*10e3, 5*10e3, 0.1*10e3), ('eta', 0.25*1e5, 9*1e5, 0.1*1e5), ('randomization_time_mean', 1.0, 40.0, 2.0), ('randomization_time_variance_factor', 0.1, 0.5, 0.1), ('randomization_magnitude', 2.0, 20.0, 1.0)]
    
    best_updates = parameter_explorer_polarization_wanderer(moddable_parameters, 0.8, num_new_dicts_to_generate=len(moddable_parameters), num_experiment_repeats=1, num_experiment_repeats_no_randomization=1, num_processes=4, total_time_in_hours=3., seed=2836, sequential=False)
