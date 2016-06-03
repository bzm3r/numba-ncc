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

base_parameter_dict = dict([('halfmax_coa_sensing_dist_multiplier', 4.4), ('kdgdi_rac_estimate_multiplier', 0.2), ('kdgdi_rho_estimate_multiplier', 0.2), ('kgdi_rac_estimate_multiplier', 1), ('kgdi_rho_estimate_multiplier', 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 500), ('kdgtp_rac_multiplier', 20), ('kdgtp_rho_mediated_rac_inhib_multiplier', 3000), ('kdgtp_rho_multiplier', 20), ('kgtp_rac_autoact_multiplier', 250), ('kgtp_rac_multiplier', 20), ('kgtp_rho_autoact_multiplier', 125), ('kgtp_rho_multiplier', 20), ('max_protrusive_node_velocity', 1e-06), ('randomization', False), ('randomization_centre', 0.15), ('randomization_depth', 1.0), ('randomization_function_type', 0), ('randomization_scheme', 0), ('randomization_time_mean', 20), ('randomization_time_variance_factor', 0.25), ('randomization_width', 10), ('randomization_width_baseline', 2), ('randomization_width_halfmax_threshold', 0.3), ('randomization_width_hf_exponent', 3), ('sigma_rac', 2e-05), ('sigma_rho_multiplier', 0.2), ('force_adh_constant', 1.0), ('force_rac_exp', 3), ('force_rho_exp', 3), ('force_rac_threshold_multiplier', 0.5), ('force_rho_threshold_multiplier', 0.5), ('skip_dynamics', False), ('stiffness_cytoplasmic', 100), ('stiffness_edge', 5e-10), ('tension_fn_type', 0), ('tension_mediated_rac_hill_exponent', 3), ('tension_mediated_rac_inhibition_half_strain', 0.05), ('threshold_rac_autoact_multiplier', 0.5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.5), ('threshold_rho_autoact_multiplier', 0.5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.5), ('coa_sensitivity_percent_drop_over_cell_diameter', 0.25), ('coa_belt_offset_multiplier', 1.5)])

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

    
    cell_group_dict = {'cell_group_name': 'A', 'num_cells': NUM_CELLS, 'init_cell_radius': CELL_DIAMETER*0.5*1e-6, 'C_total': C_total, 'H_total': H_total, 'cell_group_bounding_box': np.array([0, CELL_DIAMETER, 0, CELL_DIAMETER])*1e-6, 'intercellular_contact_factor_magnitudes_defn': {'A': cil_strength}, 'cell_dependent_coa_signal_strengths_defn': {'A': coa_strength}, 'biased_rgtpase_distrib_defns': {'default': ['unbiased random', np.array([np.pi/4, -np.pi/4]), 0.2]}, 'parameter_override_dict': parameter_defn_dict} 
    
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

def parameter_explorer_asymmetry_criteria(parameter_exploration_program, task_chunk_size=4, num_processes=4, sequential=False, min_polarity_score=0, filters=[], TOTAL_TIME=500):
    given_parameter_labels = []
    given_parameter_values = []
    
    num_estimated_combinations = 1
    
    for parameter_label, start_value, end_value, range_resolution in parameter_exploration_program:
        given_parameter_labels.append(parameter_label)
        given_parameter_values.append(np.linspace(start_value, end_value, num=range_resolution))
        num_estimated_combinations = num_estimated_combinations*range_resolution

    all_parameter_labels = parameterorg.standard_parameter_dictionary_keys
    for parameter_label in given_parameter_labels:
        if parameter_label not in all_parameter_labels:
            raise StandardError("Parameter label {} not in standard parameter dictionary.".format(parameter_label))
    
    print "Estimated number of combinations: ", num_estimated_combinations
    task_value_arrays = cartesian(tuple(given_parameter_values))
    
    num_combinations = len(task_value_arrays)
    print "Number of parameter combinations: ", num_combinations
    
    chunky_task_value_array_indices = general.chunkify(np.arange(num_combinations), task_chunk_size)
    
    num_task_chunks = len(chunky_task_value_array_indices)
    
    results = []
    num_results = 0
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    global global_results
    
    for chunk_index, task_index_chunk in enumerate(chunky_task_value_array_indices):
        st = time.time()
        print "Executing task chunk %d/%d..." %(chunk_index + 1, num_task_chunks)
        
        
        parameter_dicts = []
        task_chunk = []
        for task_index in task_index_chunk:
            task_value_array = task_value_arrays[task_index]
            update_dict = dict(zip(given_parameter_labels, task_value_array))
            parameter_dicts.append(update_dict)
            task_environment_defn = make_environment_defn(TOTAL_TIME, task_index, parameter_override_dict=update_dict)
            task_chunk.append(task_environment_defn)
            
        loop_result_cells = []
        if sequential == True:
            loop_result_cells = []
            for task in task_chunk:
                loop_result_cells.append(run_environment_dynamics(task))
        else:
            loop_result_cells = worker_pool.map(run_environment_dynamics, task_chunk)
            
            
        polarity_rating_tuples_per_loop_cell = [analysis.calculate_rgtpase_polarity_score(a_cell, significant_difference=0.2, weigh_by_timepoint=False) for a_cell in loop_result_cells]
        
        polarity_ratings_per_loop_cell = [x[0] for x in polarity_rating_tuples_per_loop_cell]
        print "polarity ratings: ", polarity_ratings_per_loop_cell
        
        for polarity_result, parameter_dict in zip(polarity_ratings_per_loop_cell, parameter_dicts):
            if polarity_result > min_polarity_score:
                results.append((polarity_result, parameter_dict))
                num_results = num_results + 1
        
        global_results = results
        et = time.time()
        print "Time: ", np.round(et - st, decimals=1)
        print "Results: ", num_results

    return results
  

# =================================================================
        
def make_experiment(num_timesteps, par_update_dict={}, environment_filepath=None, verbose=True, experiment_name=None):
    experiment_definition_dict = make_environment_defn(-1, task_name=experiment_name, **par_update_dict)
    an_environment = environment.Environment(num_timesteps, cell_group_defns=[experiment_definition_dict], environment_filepath=environment_filepath, verbose=verbose)
    
    return an_environment

# =====================================================================

if __name__ == '__main__':
    #results = parameter_explorer_asymmetry_criteria([('kgtp_rac_autoact_multiplier', 10, 250, 10), ('kdgtp_rac_mediated_rho_inhib_multiplier', 1, 5, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 1, 10, 5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.01, 0.4, 5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.01, 0.4, 5)], sequential=False, min_polarity_score=0, TOTAL_TIME=1000)

    results =  parameter_explorer_asymmetry_criteria([('kdgtp_rac_mediated_rho_inhib_multiplier', 5, 500, 5), ('threshold_rac_autoact_multiplier', 0.1, 0.6, 5), ('threshold_rho_autoact_multiplier', 0.1, 0.6, 5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.1, 0.6, 5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.1, 0.6, 5)], sequential=False, min_polarity_score=0, TOTAL_TIME=1200)# parameter_explorer_asymmetry_criteria([('kgtp_rho_autoact_multiplier', 5, 500, 5),('kgtp_rac_autoact_multiplier', 5, 500, 5), ('kdgtp_rac_mediated_rho_inhib_multiplier', 5, 1000, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 5, 4000, 5 ), ('threshold_rac_autoact_multiplier', 0.1, 0.6, 5), ('threshold_rho_autoact_multiplier', 0.1, 0.6, 5), ('threshold_rho_mediated_rac_inhib_multiplier', 0.1, 0.6, 5), ('threshold_rac_mediated_rho_inhib_multiplier', 0.1, 0.6, 5)], sequential=False, min_polarity_score=0, TOTAL_TIME=500)
    
    sorted_results = sorted(global_results, key = lambda x: x[0])
    
    print "Number of interesting results: ", len(sorted_results)
    
    
    plt.plot([x[0]['rac_membrane_active'] for x in sorted_results], [x[0]['rho_membrane_active'] for x in sorted_results], 'b.')
    plt.xlabel('rac_membrane_active')
    plt.ylabel('rho_membrane_active')