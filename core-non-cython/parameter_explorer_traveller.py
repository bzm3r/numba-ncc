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
import expexec
import copy

#------------------------------------------------------------------

base_parameter_defn_dict = {'threshold_rho_mediated_rac_inhib_multiplier': 100000.0, 'sigma_rac': 2e-05, 'skip_dynamics': False, 'threshold_rho_autoact_multiplier': 0.1, 'randomization_width_baseline': 1.0, 'kdgdi_rac_estimate_multiplier': 0.2, 'kgtp_rac_autoact_multiplier': 10, 'halfmax_coa_sensing_dist_multiplier': 4.4, 'randomization_width_hf_exponent': 5, 'randomization_width_halfmax_threshold': 0.8, 'sigma_rho_multiplier': 0.2, 'kdgtp_rho_multiplier': 1, 'randomization_time_mean': 40, 'tension_fn_type': 1, 'randomization': False, 'randomization_function_type': 0, 'randomization_time_variance_factor': 0.25, 'randomization_centre': 0.5, 'tension_mediated_rac_inhibition_half_strain': 0.05, 'kgtp_rac_multiplier': 1, 'threshold_rac_autoact_multiplier': 0.075, 'stiffness_edge': 5e-10, 'threshold_rac_mediated_rho_inhib_multiplier': 100000.0, 'stiffness_cytoplasmic': 100, 'max_protrusive_node_velocity': 1e-06, 'kdgtp_rac_multiplier': 1, 'randomization_width': 5e-05, 'kgtp_rho_multiplier': 1, 'randomization_depth': 0.95, 'randomization_scheme': 0, 'kgtp_rho_autoact_multiplier': 20, 'cil_rac_inhibition': 0.2, 'kdgtp_rac_mediated_rho_inhib_multiplier': 1e-16, 'kdgtp_rho_mediated_rac_inhib_multiplier': 1e-16}

def make_environment_defn(TOTAL_TIME, task_index, parameter_override_dict={}):
    
    space_migratory_bdry_polygon, space_physical_bdry_polygon = expexec.make_space_polygons(False, False, 0, 0, 0, 0)
    NUM_CELLS = 1
    TIMESTEP_LENGTH = (1/0.5)
    NUM_TIMESTEPS = int(TOTAL_TIME/TIMESTEP_LENGTH)
    NUM_NODES_PER_CELL = 16
    CELL_DIAMETER = 40
    
    C_total = 3e6 # number of molecules
    H_total = 1.5e6  # number of molecule    

    cil_strength = 3
    coa_strength = 0.1
    
    global base_parameter_defn_dict
    parameter_defn_dict = copy.deepcopy(base_parameter_defn_dict)
    parameter_defn_dict.update(parameter_override_dict)

    
    cell_group_dict = {'cell_group_name': 'A', 'num_cells': NUM_CELLS, 'init_cell_radius': CELL_DIAMETER*0.5*1e-6, 'C_total': C_total, 'H_total': H_total, 'cell_group_bounding_box': np.array([0, CELL_DIAMETER, 0, CELL_DIAMETER])*1e-6, 'intercellular_contact_factor_magnitudes_defn': {'A': cil_strength}, 'cell_dependent_coa_signal_strengths_defn': {'A': coa_strength}, 'biased_rgtpase_distrib_defns': {'default': ['unbiased random', np.array([np.pi/4, -np.pi/4]), 0.5]}, 'parameter_override_dict': parameter_defn_dict} 
    
    cell_group_defns = [cell_group_dict]
    
    environment_wide_variable_defns = {'num_timesteps': NUM_TIMESTEPS, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': TIMESTEP_LENGTH, 'num_nodes_per_cell': NUM_NODES_PER_CELL, 'integration_params': {'atol': 1e-4, 'rtol': 1e-4}}
    
    parameter_overrides = [x['parameter_override_dict'] for x in cell_group_defns]
    
    task_defn_dict = {'environment_name': 'worker', 'parameter_explorer_run': True, 'user_cell_group_defns': cell_group_defns, 'parameter_overrides': parameter_overrides}
    
    task_defn_dict.update(environment_wide_variable_defns)
    
    return task_defn_dict

# -------------------------------------

def run_environment_dynamics(task_defn_dict):
    
    an_environment = parameterorg.make_environment_given_user_cell_group_defns(**task_defn_dict)

    an_environment.execute_system_dynamics({}, produce_intermediate_visuals=False, produce_final_visuals=False)
    
    return an_environment.cells_in_environment[0]
    
# --------------------------------------------------------------------

global_weird_parameter_dicts = []
global_results = []

def pick_random_point(given_parameter_num_values):
    result = [np.random.choice(np.arange(n)) for n in given_parameter_num_values]
    return result
    
def generate_parameter_update_dict(given_parameter_labels, given_parameter_values, pd_index):
    tuples = []
    for label, n in zip(given_parameter_labels, pd_index):
        tuples.append((label, n))
        
    return dict(tuples)
    
def generate_tweaks_from_given_point(given_parameter_num_values, pd_index, num_tweaks_to_generate):
    
    tweaked_pd_indices = []

    num_indices = len(pd_index)   
    for x in xrange(num_tweaks_to_generate):
        tweaked_pd_index = [0]*num_indices
        index_array = np.arange(num_indices)
        np.random.shuffle(index_array)
        
        for i, n in enumerate(index_array):
            if i == 0:
                delta = np.random.choice([-1, 1])
            else:
                delta = np.random.choice([-1, 0, 1])
            
            tweaked_pd_index[i] = (pd_index[i] + delta)%given_parameter_num_values[i]
            
        tweaked_pd_indices.append(tweaked_pd_index)
    
    return tweaked_pd_indices

# =======================================================================================

executed_pd_indices = []
executed_pds = []
executed_pd_scores = []

def parameter_explorer_traveller_using_asymmetry_criteria(parameter_exploration_program, num_tweaks_to_generate=11, num_processes=4, sequential=False, num_loops_before_giving_up=100, TOTAL_TIME=500):
    given_parameter_labels = []
    given_parameter_values = []
    given_parameter_num_values = []
    
    num_estimated_combinations = 1
    
    
    for parameter_label, start_value, end_value, range_resolution in parameter_exploration_program:
        given_parameter_labels.append(parameter_label)
        given_parameter_values.append(np.linspace(start_value, end_value, num=range_resolution))
        given_parameter_num_values.append(range_resolution)
        num_estimated_combinations = num_estimated_combinations*range_resolution

    all_parameter_labels = parameterorg.standard_parameter_dictionary_keys
    for parameter_label in given_parameter_labels:
        if parameter_label not in all_parameter_labels:
            raise StandardError("Parameter label {} not in standard parameter dictionary.".format(parameter_label))
    
    print "Number of parameter combinations: ", num_estimated_combinations
    
    # pick random starting point
    global executed_pd_indices
    global executed_pds
    global executed_pd_scores
    current_pd_index = pick_random_point(given_parameter_num_values)
    improved_score = False
    current_no_improvement_streak = 0
    current_best_score = -1.0
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    
    while current_no_improvement_streak < num_loops_before_giving_up:
        st = time.time()
        print "current best score: ", current_best_score
        print "current_no_improvement_streak: ", current_no_improvement_streak
        improved_score = False
        
        pd_indices = [current_pd_index] + [x for x in generate_tweaks_from_given_point(given_parameter_num_values, current_pd_index, num_tweaks_to_generate) if x not in executed_pd_indices]
        
        num_pd_indices_to_run = len(pd_indices)
        print "num_pd_indices_to_run: ", num_pd_indices_to_run
        
        if num_pd_indices_to_run < 0.5*(num_tweaks_to_generate + 1):
            print "re-randomizing current pd indices..."
            current_pd_index = pick_random_point(given_parameter_num_values)
            
        pds = [generate_parameter_update_dict(given_parameter_labels, given_parameter_values, pd_index) for pd_index in pd_indices]
 
        chunky_pd_indices = general.chunkify(pd_indices, num_processes)
        chunky_pds = general.chunkify(pds, num_processes)
        scores = []
        
        for n, pd_index_chunk in enumerate(chunky_pds):
            executed_pd_indices += chunky_pd_indices[n]
            executed_pds += pd_index_chunk
            task_args = [make_environment_defn(TOTAL_TIME, -1, parameter_override_dict=pd_index) for pd_index in pd_index_chunk]
            loop_result_cells = []
            
            print "running simulations..."
            if sequential == True:
                loop_result_cells = []
                for task in task_args:
                    loop_result_cells.append(run_environment_dynamics(task))
            else:
                loop_result_cells = worker_pool.map(run_environment_dynamics, task_args)
 
            scores = [analysis.score_symmetry(a_cell, significant_difference=0.2, weigh_by_timepoint=True) for a_cell in loop_result_cells]
            averaged_scores = [np.average(score_dict.values()) for score_dict in scores]
            print "scores: ", scores
            executed_pd_scores += scores
            
            for i, avg_score in enumerate(averaged_scores):
                if avg_score > current_best_score:
                    print "improved score!"
                    current_pd_index = pd_indices[n + i]
                    current_best_score = avg_score
                    improved_score = True
                
        if improved_score == True:
            current_no_improvement_streak = 0
        else:
            current_no_improvement_streak += 1
            
        et = time.time()
        print "time taken: ", np.round(et - st, decimals=1)
        print "-----------------------"

    return zip(executed_pd_scores, executed_pds)
  

# =================================================================
        
def make_experiment(num_timesteps, par_update_dict={}, environment_filepath=None, verbose=True, experiment_name=None):
    experiment_definition_dict = make_environment_defn(-1, task_name=experiment_name, **par_update_dict)
    an_environment = environment.Environment(num_timesteps, cell_group_defns=[experiment_definition_dict], environment_filepath=environment_filepath, verbose=verbose)
    
    return an_environment

# =====================================================================

if __name__ == '__main__':
    results = parameter_explorer_traveller_using_asymmetry_criteria([('kdgtp_rac_mediated_rho_inhib_multiplier', 1, 20, 10), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1, 20, 10), ('threshold_rac_mediated_rho_inhib_multiplier', 0.01, 0.8, 20), ('threshold_rho_mediated_rac_inhib_multiplier', 0.01, 0.8, 20)], sequential=False, TOTAL_TIME=2000, num_loops_before_giving_up=50, num_processes=4, num_tweaks_to_generate=11)

    
    sorted_results = sorted(results, key = lambda x: np.average(x[0].values()))
    
    print "Number of interesting results: ", len(sorted_results)
    
    
    plt.plot([x[0]['rac_membrane_active'] for x in sorted_results], [x[0]['rho_membrane_active'] for x in sorted_results], 'b.')
    plt.xlabel('rac_membrane_active')
    plt.ylabel('rho_membrane_active')