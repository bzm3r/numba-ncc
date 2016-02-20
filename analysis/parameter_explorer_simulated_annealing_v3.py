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
import copy

import numba as nb

global_num_timesteps = 500
global_num_nodes_per_cell = 11
global_num_cells = 1

#------------------------------------------------------------------

def make_parameter_dictionary(TOTAL_TIME, task_index, parameter_overrides={}):
    
    box_height = 25
    box_width = 25
    box_x_offset = 1000
    box_y_offset = 1000
    
    coa = 0
    coa_degr = 1
    cil = 2
    
    width_corridor = 1e5
    height_corridor = 1e5
    
    space_physical_bdry_polygon =  np.array([])
    space_migratory_bdry_polygon = np.array([[0 + 0, 0 + 0], [width_corridor + 0, 0 + 0], [width_corridor + 0, height_corridor + 0], [0 + 0, height_corridor + 0]], dtype=np.float64)*1e-6
    
    TIMESTEP_LENGTH = (1/0.5)
    NUM_TIMESTEPS = int(TOTAL_TIME/TIMESTEP_LENGTH)
    NUM_NODES_PER_CELL = 16
    
    C_total = 3e6 # number of molecules
    H_total = 1.5e6  # number of molecule
    
    init_cell_radius = 12.5e-6

    
    cell_group_defns = [{'cell_group_name': 'A', 'num_cells': 1, 
 'init_cell_radius': init_cell_radius,
 'C_total': C_total, 'H_total': H_total,
 'cell_group_bounding_box': np.array([box_x_offset, box_width + box_x_offset, box_y_offset, box_height+box_y_offset])*1e-6, 
 'intercellular_contact_factor_magnitudes_defn': {'A': cil},
 'cell_dependent_coa_factor_production_defn': {'A': coa}, 'max_coa_sensing_dist_multiplier': 3, 'coa_factor_degradation_rate_multiplier': coa_degr, 'biased_rgtpase_distrib_defns': {'default': ['unbiased random', np.array([np.pi/4, -np.pi/4]), 0.5]}}]
 

    parameter_overrides.update([('stiffness_edge', 3e-10)])
    
    environment_wide_variable_defns = {'num_timesteps': NUM_TIMESTEPS, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': TIMESTEP_LENGTH, 'num_nodes_per_cell': NUM_NODES_PER_CELL, 'close_dist': 75e-6, 'verbose': False, 'integration_params': {'atol': 1e-4, 'rtol': 1e-4}, 'force_rac_max': 1000e-12, 'force_rho_max': 200e-12, 'max_protrusive_node_velocity': 0.5e-6, 'closeness_dist_squared_criteria': (0.5e-6)**2}
    
    environment_creation_parameter_dict = {'environment_name': 'worker', 'parameter_overrides': [parameter_overrides], 'environment_filepath': None, 'user_cell_group_defns': cell_group_defns}
    
    environment_creation_parameter_dict.update(environment_wide_variable_defns)
    
    return environment_creation_parameter_dict

# -------------------------------------

def run_environment_dynamics(TOTAL_TIME, task_index, task_defn_dict):
    environment_creation_parameter_dict = make_parameter_dictionary(TOTAL_TIME, task_index, parameter_overrides=task_defn_dict)
    
    an_environment = parameterorg.make_environment_given_user_cell_group_defns(**environment_creation_parameter_dict)

    an_environment.execute_system_dynamics()
    
    return an_environment.cells_in_environment[0]
    
# -------------------------------------
    
def calculate_energy_using_asymmetry_criteria(TOTAL_TIME, task_index, task_defn_dict):
    cell = run_environment_dynamics(TOTAL_TIME, task_index, task_defn_dict)
    acceleration = analysis.calculate_acceleration(cell)
    
    score = 1.0/(acceleration)
        
    return acceleration, score
    
# -------------------------------------

def acceptance_probability_function(energy_alpha, task_defn_dict_beta, temperature, TOTAL_TIME, task_index=-1):
    symmetry_beta, energy_beta = calculate_energy_using_asymmetry_criteria(TOTAL_TIME, task_index, task_defn_dict_beta)
    
    if energy_beta < energy_alpha:
        return symmetry_beta, energy_beta, 1
    else:
        return symmetry_beta, energy_beta, np.exp(-(energy_beta - energy_alpha)/temperature)
        
# -------------------------------------
        
def generate_possible_transition(task_defn_dict, num_parameters, parameter_exploration_program):
    num_tweaks = 1
    
    #print "num_parameters_tweaked: ", num_tweaks
    if num_tweaks == 1:
        to_tweak_indices = [np.random.randint(0, high=num_parameters)]
    else:
        parameter_indices = np.arange(num_parameters)
        np.random.shuffle(parameter_indices)
        to_tweak_indices = parameter_indices[:num_tweaks]
        
    transition_dict = copy.deepcopy(task_defn_dict)
    
    for parameter_index in to_tweak_indices:
        label, min_val, max_val, delta = parameter_exploration_program[parameter_index]
        
        current_val = task_defn_dict[label]
        delta_sign = 0
        if current_val >= max_val:
            delta_sign = -1
        elif current_val <= min_val:
            delta_sign = 1
        else:
            check = np.random.randint(0, high=2)
            if check == 0:
                delta_sign = -1
            else:
                delta_sign = 1
        
        transition_dict[label] += delta_sign*delta
    
    return transition_dict
    
# --------------------------------------------------------------------

def parameter_explorer_asymmetry_criteria(parameter_exploration_program, TOTAL_TIME=500, cooldown_rate=1, max_tries=1000):
    given_parameter_labels = [x[0] for x in parameter_exploration_program]
    num_parameters = len(given_parameter_labels)

    all_parameter_labels = parameterorg.standard_parameter_dictionary_keys
    for parameter_label in given_parameter_labels:
        if parameter_label not in all_parameter_labels:
            raise StandardError("Parameter label {} not in standard parameter dictionary.".format(parameter_label))
    
    max_time = np.round(np.log(1e-2)/(-1*cooldown_rate), decimals=2)
    
    print "max_time: ", max_time
    
    print "setting up times and tempeartures..."
    times = np.linspace(0, max_time, num=max_tries)
    temperatures = np.exp(-1*cooldown_rate*times)
           
    # initialize task definition dictionary
    print "initializing task definition dictionary..."
    task_defn_dict = dict(zip(given_parameter_labels, [x[1] for x in parameter_exploration_program]))
    print "calculating initial asymmetry/energy rating..."
    current_symmetry, current_energy = calculate_energy_using_asymmetry_criteria(TOTAL_TIME, -1, task_defn_dict)

    trial_list = [(current_symmetry, task_defn_dict)]
    len_trial_list = 1
    time_takens = []
    symmetry_ratings = [current_symmetry]
    
    
    for try_number in range(max_tries):
        temperature = temperatures[try_number]
        print "-----TRY NUMBER: {}-------".format(try_number)
        print "temperature: ", np.round(temperature, decimals=2)
        
        possible_next_defn_dict = generate_possible_transition(task_defn_dict, num_parameters, parameter_exploration_program)
        
        st = time.time()
        symmetry_next, energy_next, acceptance_probability = acceptance_probability_function(current_energy, possible_next_defn_dict, temperature, TOTAL_TIME=TOTAL_TIME)
        
        print "current_energy: ", current_energy
        print "energy_next: ", energy_next
        
        time_takens.append(time.time() - st)
        print "acceptance_probability: ", np.round(acceptance_probability, decimals=1)
        
        
        if np.random.rand() <= acceptance_probability:
            print "Accepted."
            task_defn_dict = possible_next_defn_dict
            current_symmetry = symmetry_next
            current_energy = energy_next
            trial_list.append((current_symmetry, task_defn_dict))
            len_trial_list += 1
            symmetry_ratings.append(current_symmetry)
        else:
            print "Not accepted."
        
        print "current_symmetry: ", current_symmetry
        print "len_trial_list: ", len_trial_list
        print "--------------------------"
            
    return trial_list
  

# =================================================================
        
def make_experiment(num_timesteps, par_update_dict={}, environment_filepath=None, verbose=True, experiment_name=None):
    experiment_definition_dict = make_parameter_dictionary(-1, task_name=experiment_name, **par_update_dict)
    an_environment = environment.Environment(num_timesteps, cell_group_defns=[experiment_definition_dict], environment_filepath=environment_filepath, verbose=verbose)
    
    return an_environment

# =====================================================================

if __name__ == '__main__':
    # 2015_JUN_8: run 1
    # [('kgtp_rac_multiplier', 20, 200, 3), ('kgtp_rho_multiplier', 20, 200, 3), ('kgtp_rho_autoact_multiplier', 100, 1000, 3),('kgtp_rac_autoact_multiplier', 100, 1000, 3), ('kdgtp_rac_multiplier', 20, 200, 3), ('kdgtp_rho_multiplier', 20, 200, 3), ('kdgtp_rac_mediated_rho_inhib_multiplier', 100, 1000, 3), ('kdgtp_rho_mediated_rac_inhib_multiplier', 100, 1000, 3), ('randomization_exponent', 0, 0, 1), ('randomize_rgtpase_distrib', 1, 1, 1)]

    # 2015 JUN 8: run 2
    # 
#    results = parameter_explorer_asymmetry_criteria([('kgtp_rac_multiplier', 50, 150, 5), ('kgtp_rho_multiplier', 50, 150, 5), ('kgtp_rho_autoact_multiplier', 100, 250, 5),('kgtp_rac_autoact_multiplier', 100, 250, 5), ('kdgtp_rac_multiplier', 20, 20, 1), ('kdgtp_rho_multiplier', 20, 20, 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 100, 250, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 100, 250, 5), ('threshold_rac_autoact_multiplier', 0.5, 0.5, 1), ('threshold_rho_autoact_multiplier', 0.5, 0.5, 1), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5, 0.5, 1), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5, 0.5, 1), ('threshold_rac_autoact_dgdi_multiplier', 0.5, 0.5, 1), ('stiffness_edge', 3e-10, 3e-10, 1)], sequential=False, min_symmetry_score=0, TOTAL_TIME=500)

    results = parameter_explorer_asymmetry_criteria([('kgtp_rac_multiplier', 20, 75, 5), ('kgtp_rho_multiplier', 20, 100, 5), ('kgtp_rho_autoact_multiplier', 100, 250, 5),('kgtp_rac_autoact_multiplier', 100, 250, 5), ('kdgtp_rac_multiplier', 1, 40, 5), ('kdgtp_rho_multiplier', 1, 40, 5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 100, 250, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 100, 250, 5                                                                             ), ('threshold_rac_autoact_multiplier', 0.25, 1.25, 0.05), ('threshold_rho_autoact_multiplier', 0.25, 1.25, 0.05), ('threshold_rho_mediated_rac_inhib_multiplier', 0.25, 1.25, 0.05), ('threshold_rac_mediated_rho_inhib_multiplier', 0.25, 1.25, 0.05), ('threshold_rac_autoact_dgdi_multiplier', 0.25, 1.25, 0.05)], cooldown_rate=0.025, TOTAL_TIME=500, max_tries=1000)
    
    # [('kgtp_rac_multiplier', 75, 250, 5), ('kgtp_rho_multiplier', 75, 250, 5), ('kgtp_rho_autoact_multiplier', 75, 250, 5),('kgtp_rac_autoact_multiplier', 75, 250, 5), ('kdgtp_rac_multiplier', 10, 100, 5), ('kdgtp_rho_multiplier', 10, 100, 5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 75, 250, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 75, 250, 5), ('randomize_rgtpase_distrib', 1, 1, 1)]
    
    sorted_results = sorted(results, key = lambda x: x[0])
        
    plt.plot([x[0] for x in results], 'b.')
    
#    robustness_results = check_robustness_of_parameter_sets([result[1] for result in interesting_results], {'kgtp_rac_multiplier': 300, 'kgtp_rho_multiplier': 300, 'kgtp_rho_autoact_multiplier': 1000, 'kgtp_rac_autoact_multiplier': 1000, 'kdgtp_rac_multiplier': 300, 'kdgtp_rho_multiplier': 300, 'kdgtp_rac_mediated_rho_inhib_multiplier': 1000, 'kdgtp_rho_mediated_rac_inhib_multiplier': 1000}, sequential=False)