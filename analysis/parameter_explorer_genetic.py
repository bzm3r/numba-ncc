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

def make_parameter_dictionary(TOTAL_TIME, task_index, parameter_overrides={}, invariant_parameter_overrides={}):
    
    box_height = 50
    box_width = 50
    box_x_offset = 0
    box_y_offset = 0
    
    coa = 0
    coa_degr = 1
    cil = 2
    
    width_corridor = 200
    height_corridor = 75
    
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
 'cell_dependent_coa_signal_strengths_defn': {'A': coa}, 'biased_rgtpase_distrib_defns': {'default': ['uniform', np.array([np.pi/4, -np.pi/4]) + np.pi, 0.4]}}]
 

    parameter_overrides.update(invariant_parameter_overrides)
    
    environment_wide_variable_defns = {'num_timesteps': NUM_TIMESTEPS, 'space_physical_bdry_polygon': space_physical_bdry_polygon, 'space_migratory_bdry_polygon': space_migratory_bdry_polygon, 'T': TIMESTEP_LENGTH, 'num_nodes_per_cell': NUM_NODES_PER_CELL, 'verbose': False, 'integration_params': {'atol': 1e-4, 'rtol': 1e-4}, 'sigma_rac': 15e-4, 'closeness_dist_squared_criteria': (0.25e-6)**2}
    
    environment_creation_parameter_dict = {'environment_name': 'worker', 'parameter_overrides': [parameter_overrides], 'environment_filepath': None, 'user_cell_group_defns': cell_group_defns}
    
    environment_creation_parameter_dict.update(environment_wide_variable_defns)
    
    return environment_creation_parameter_dict

# -------------------------------------

def run_environment_dynamics(TOTAL_TIME, task_index, task_defn_dict, invariant_parameter_overrides):
    environment_creation_parameter_dict = make_parameter_dictionary(TOTAL_TIME, task_index, parameter_overrides=task_defn_dict, invariant_parameter_overrides=invariant_parameter_overrides)
    
    an_environment = parameterorg.make_environment_given_user_cell_group_defns(**environment_creation_parameter_dict)

    an_environment.execute_system_dynamics({},  produce_intermediate_visuals=False, produce_final_visuals=False, elapsed_timesteps_before_producing_intermediate_visuals=0, output_dir='')
       
    return an_environment.cells_in_environment[0]
    
# -------------------------------------

def calculate_fitness(TOTAL_TIME, task_index, task_defn_dict, invariant_parameter_overrides):
    cell = run_environment_dynamics(TOTAL_TIME, task_index, task_defn_dict, invariant_parameter_overrides=invariant_parameter_overrides)
    migr_bdry_violation_score = analysis.calculate_migry_boundary_violation_score(cell)
    average_strain = analysis.calculate_average_total_strain(cell)
    
    symmetry_value = analysis.score_symmetry(cell, significant_difference=0.2)['rac_membrane_active']
    
    if average_strain > 0.15 or symmetry_value < 0.05:
        strain_score = 0
    else:
        strain_score = 1
        
    score = symmetry_value*0.5 + strain_score*0.5
        
    return migr_bdry_violation_score, score
    
# -----------------------------------------------------------------
    
def mate_parameter_pair(parameter_exploration_program, p1_dict, p2_dict, mutation_probability):
    child = copy.deepcopy(p1_dict)
    
    for parameter_tuple in parameter_exploration_program:
        label, min_value, max_value, unused = parameter_tuple
        
        combine_type = np.random.randint(0, high=2)
        
        if combine_type == 0:
            child[label] = p1_dict[label]
        elif combine_type == 1:
            child[label] = p2_dict[label]
        else:
            child[label] = np.average([p1_dict[label], p2_dict[label]])
            
        if np.random.rand() < mutation_probability:
            print "MUTATION!"
            current_value = child[label]
            delta_sign_choice = np.random.randint(0, high=2)
            
            if current_value <= min_value:
                delta_sign = 1
                delta_mag = max_value - current_value
            elif current_value >= max_value:
                delta_sign = -1
                delta_mag = current_value - min_value
            else:
                if delta_sign_choice == 0:
                    delta_sign = 1
                    delta_mag = max_value - current_value
                else:
                    delta_sign = -1
                    delta_mag = current_value - min_value
                
            child[label] += delta_sign*delta_mag*np.random.rand()
            
    return child
            
#-----------------------------------------------------------------

class ParDict:
    def __init__(self, pardict, parents, TOTAL_TIME, invariant_parameter_overrides):
        self.pardict = pardict
        self.parents = parents
        
        self.mates = []
        self.children = []
        
        self.fitness = -1
        self.TOTAL_TIME = TOTAL_TIME
        self.invariant_parameter_overrides = invariant_parameter_overrides
        
    def mate_with(self, mate, num_children, parameter_exploration_program, mutation_probability):
        self.mates.append(mate)
        mate.mates.append(self)
        
        generated_children_pardicts = [mate_parameter_pair(parameter_exploration_program, self.pardict, mate.pardict, mutation_probability)for x in range(num_children)]
        
        generated_children = [ParDict(pd, [self, mate], self.TOTAL_TIME, self.invariant_parameter_overrides) for pd in generated_children_pardicts]
        
        self.children.append(generated_children)
        mate.children.append(generated_children)
        
        return generated_children
        
    def update_fitness(self):
        if self.fitness == -1:
            self.fitness = calculate_fitness(self.TOTAL_TIME, -1, self.pardict, self.invariant_parameter_overrides)
            
        return self.fitness
        
    def get_pardict(self):
        pd = copy.deepcopy(self.pardict)
        pd.update(self.invariant_parameter_overrides)
        
        return pd

#-----------------------------------------------------------------

def mp_update_fitness(pardict_obj):
    return pardict_obj.update_fitness()

#-----------------------------------------------------------------

def dechunkify(chunky_list):
    dechunked_list = []
    
    for chunk in chunky_list:
        dechunked_list += chunk
        
    return dechunked_list
            
#-----------------------------------------------------------------

@nb.jit(nopython=True)
def choose(n, m):
    if n < 0:
        return n/0
        
    result = 1
    
    for x in range(m):
        result = result*(n - x)
        
    return int(result/m)
        
@nb.jit(nopython=True)       
def determine_mating_pairs(num_mates):
    num_mating_pairs = int(choose(num_mates, 2))
    mating_pairs = -1*np.ones((num_mating_pairs, 2), dtype=np.int64)
    
    num_mating_pairs_instantiated = 0
    for x in range(num_mates):
        for y in range(x + 1, num_mates):
            mating_pairs[num_mating_pairs_instantiated][0] = x
            mating_pairs[num_mating_pairs_instantiated][1] = y
            num_mating_pairs_instantiated += 1
            
    return mating_pairs

g_generations = []

def parameter_explorer_genetic(parameter_exploration_program, invariant_parameter_overrides, task_chunk_size=4, num_processes=4, sequential=False, TOTAL_TIME=500, num_children=10, num_generations=100, mutation_probability=0.02, init_parents=[]):
    given_parameter_labels = [x[0] for x in parameter_exploration_program] 
    
    all_parameter_labels = parameterorg.standard_parameter_dictionary_keys
    for parameter_label in given_parameter_labels:
        if parameter_label not in all_parameter_labels:
            raise StandardError("Parameter label {} not in standard parameter dictionary.".format(parameter_label))
    
    if len(init_parents) != 0:
        init_p1_dict = init_parents.pop()
        init_p1 = ParDict(dict([(x[0], init_p1_dict[x[0]]) for x in parameter_exploration_program]), None, TOTAL_TIME, invariant_parameter_overrides)
    else:
        init_p1 = ParDict(dict([(x[0], x[1] + (x[2]-x[1])*np.random.rand()) for x in parameter_exploration_program]), None, TOTAL_TIME, invariant_parameter_overrides)
        
        
    if len(init_parents) != 0:
        init_p2_dict = init_parents.pop()
        init_p2 = ParDict(dict([(x[0], init_p2_dict[x[0]]) for x in parameter_exploration_program]), None, TOTAL_TIME, invariant_parameter_overrides)
    else:
        init_p2 = ParDict(dict([(x[0], x[1] + (x[2]-x[1])*np.random.rand()) for x in parameter_exploration_program]), None, TOTAL_TIME, invariant_parameter_overrides)
    
    worker_pool = multiproc.Pool(processes=num_processes, maxtasksperchild=750)
    
    global g_generations
    g_generations = [[init_p1, init_p2]]
    for generation_number in range(num_generations):
        print "------GENERATION {}--------".format(generation_number)
        
        st = time.time()
        current_generation = g_generations[generation_number]
        
        print "Num elements in current generation: ", len(current_generation)
        
        chunky_current_generation = general.chunkify(current_generation, task_chunk_size)
        
        num_chunks = len(chunky_current_generation)
        print "Num chunks: ", num_chunks
        
        for i, current_gen_chunk in enumerate(chunky_current_generation):
            print "Executing chunk {}/{}".format(i + 1, num_chunks)
#            fitness_results = []
#            for pdobj in current_gen_chunk:
#                fitness_results.append(pdobj.update_fitness())
            fitness_results = worker_pool.map(mp_update_fitness, current_gen_chunk)
            
            for fitness_result, cg_pdo in zip(fitness_results, current_gen_chunk):
                cg_pdo.fitness = fitness_result
        
        current_generation = sorted(current_generation, key=lambda x: x.fitness[1], reverse=True)
        g_generations[generation_number] = current_generation
        et = time.time()
        
        print "Multiproc calcs time: {}".format(np.round(et-st, decimals=2))
        fittest_in_generation = current_generation[:4]
        fitness_of_generation = np.array([x.fitness[1] for x in fittest_in_generation])
        print "generation_fitness: ", fitness_of_generation
        mating_pairs = determine_mating_pairs(len(fittest_in_generation))
        
#        if np.all(np.abs(np.average(fitness_of_generation) - fitness_of_generation) < 1e-8) == True:
#            print "Breaking because fittest_have_homogeneized..."
#            break
        
        if generation_number != num_generations - 1:
            print "Generating children..."
            next_generation = []
            for mating_pair in mating_pairs:
                mate1 = fittest_in_generation[mating_pair[0]]
                mate2 = fittest_in_generation[mating_pair[1]]
                
                children = mate1.mate_with(mate2, num_children, parameter_exploration_program, mutation_probability)
                next_generation += children
                
            print "Num elements in next gen: ", len(next_generation) + len(fittest_in_generation)
        
            g_generations.append(fittest_in_generation + next_generation)
            
        print "Done."
        print "---------------------------"
        
    return g_generations

# =================================================================
        
def make_experiment(num_timesteps, par_update_dict={}, environment_filepath=None, verbose=True, experiment_name=None):
    experiment_definition_dict = make_parameter_dictionary(-1, task_name=experiment_name, **par_update_dict)
    an_environment = environment.Environment(num_timesteps, cell_group_defns=[experiment_definition_dict], environment_filepath=environment_filepath, verbose=verbose)
    
    return an_environment

# =====================================================================

if __name__ == '__main__':

    results = parameter_explorer_genetic([('kgtp_rac_multiplier', 5, 500, 5), ('kgtp_rho_multiplier', 5, 500, 5), ('kgtp_rho_autoact_multiplier', 5, 500, 5),('kgtp_rac_autoact_multiplier', 5, 500, 5), ('kdgtp_rac_multiplier', 20, 20, 1), ('kdgtp_rho_multiplier', 20, 20, 1), ('kdgtp_rac_mediated_rho_inhib_multiplier', 5, 4000, 5), ('kdgtp_rho_mediated_rac_inhib_multiplier', 5, 4000, 5                                                                             ), ('threshold_rac_autoact_multiplier', 0.2, 0.8, 1), ('threshold_rho_autoact_multiplier', 0.2, 0.8, 1), ('threshold_rho_mediated_rac_inhib_multiplier', 0.2, 0.8, 1), ('threshold_rac_mediated_rho_inhib_multiplier', 0.2, 0.8, 1)], [('stiffness_edge', 3e-10), ('kdgtp_rho_multiplier', 20), ('kdgtp_rac_multiplier', 20), ('kdgdi_rac_estimate_multiplier', 0.02)], task_chunk_size=4, num_processes=4, sequential=False, TOTAL_TIME=1800, num_children=2, num_generations=30, mutation_probability=0.10, init_parents=[])
    
# {'threshold_rho_mediated_rac_inhib_multiplier': 0.4449310755436056, 'kdgtp_rac_multiplier': 20, 'kdgdi_rac_estimate_multiplier': 0.02, 'kgtp_rac_multiplier': 37.17547346975959, 'kgtp_rho_multiplier': 79.10034543123137, 'kdgtp_rho_multiplier': 20, 'stiffness_edge': 3e-10, 'space_at_node_factor_rho': 1000.0, 'threshold_rho_autoact_multiplier': 0.49256268354974697, 'kgtp_rho_autoact_multiplier': 252.67007762095602, 'threshold_rac_mediated_rho_inhib_multiplier': 0.4819644832750601, 'threshold_rac_autoact_multiplier': 0.4009947975234428, 'kdgtp_rac_mediated_rho_inhib_multiplier': 152.41654971330885, 'kgtp_rac_autoact_multiplier': 289.48214188456694, 'kdgtp_rho_mediated_rac_inhib_multiplier': 86.95543017237648, 'space_at_node_factor_rac': 1000.0}
    
