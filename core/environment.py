from __future__ import division
import numpy as np
import cell
from parameterorg import *
import geometry
import os
import visualization.datavis as datavis
import analysis.utilities as analysis_utils
import visualization.animator as animator
import cPickle as pickling_package
import gzip
import shutil

"""
Environment of cells.
"""

def custom_floor(fp_number, roundoff_distance):
    a = int(fp_number)
    b = a + 1
    
    if abs(fp_number - b) < roundoff_distance:
        return b
    else:
        return a 
# -----------------------------------------------------------------

def calc_dist_squared_bw_points(p1, p2):
    x_disp = p1[0] - p2[0]
    y_disp = p1[1] - p2[1]
    
    return x_disp*x_disp + y_disp*y_disp 

# -----------------------------------------------------------------
    
def calc_bounding_box(node_coords):
    return np.array([np.min(node_coords[0]), np.max(node_coords[0]), np.min(node_coords[1]), np.max(node_coords[1])])

# -----------------------------------------------------------------

    
def calc_bounding_box_centre(bb):
    x = bb[0] + (bb[1] - bb[0])*0.5
    y = bb[2] + (bb[3] - bb[2])*0.5
    
    return np.array([x, y])

# -----------------------------------------------------------------
    
def calculate_bounding_boxes(node_coords_per_cell):

    bounding_boxes = [calc_bounding_box(node_coords) for node_coords in node_coords_per_cell]
                    
    return bounding_boxes
                

# -----------------------------------------------------------------

class Environment():
    """Implementation of coupled map lattice model of a cell.
    """
    def __init__(self, environment_name='', num_timesteps=0, space_physical_bdry_polygon=np.array([], dtype=np.float64), space_migratory_bdry_polygon=np.array([], dtype=np.float64), cell_group_defns=None, environment_filepath=None, verbose=True, T=(1/0.5), num_nodes_per_cell=16, integration_params={}, full_print=False, persist=True, parameter_explorer_run=False): 

        self.parameter_explorer_run = parameter_explorer_run        
        if parameter_explorer_run == True:
            self.verbose = False
            self.persist = False
            self.environment_name = None
            self.environment_filepath = None
        else:
            self.verbose = verbose
            self.persist = persist
            self.environment_name = environment_name
            self.environment_filepath = environment_filepath
        
        self.space_physical_bdry_polygon = space_physical_bdry_polygon
        self.space_migratory_bdry_polygon = space_migratory_bdry_polygon
        
        self.cell_group_defns = cell_group_defns
        
        self.curr_t = 0
        self.T = T
        self.num_timesteps = num_timesteps
        self.num_timepoints = num_timesteps + 1
        self.timepoints = np.arange(self.curr_t, self.num_timepoints)
        
        self.integration_params = integration_params
        
        self.num_nodes_per_cell = num_nodes_per_cell
        
        self.micrometer = 1e-6
        
        self.num_cell_groups = len(cell_group_defns)
        self.num_cells = np.sum([cell_group_defn['num_cells'] for cell_group_defn in cell_group_defns])
        
        self.cells_in_environment = self.make_cells()
        self.full_output_dicts = [[] for cell in self.cells_in_environment]
        
        self.cell_indices = np.arange(self.num_cells)
        self.full_print = full_print

# -----------------------------------------------------------------

    def make_cells(self):
        cells_in_environment = []
        cell_bounding_boxes_wrt_time = []
    
        for cell_group_index, cell_group_defn in enumerate(self.cell_group_defns):
            cells_in_group, init_cell_bounding_boxes = self.create_cell_group(self.num_timesteps, cell_group_defn, cell_group_index)
            cells_in_environment += cells_in_group
            cell_bounding_boxes_wrt_time.append(init_cell_bounding_boxes)
    
        return np.array(cells_in_environment)
        
        
# -----------------------------------------------------------------

    def make_intercellular_contact_factor_magnitudes(self, cell_group_index, this_cell_group_defn):
        intercellular_contact_factor_magnitudes_defn = this_cell_group_defn['intercellular_contact_factor_magnitudes_defn']
        
        num_defns = len(intercellular_contact_factor_magnitudes_defn.keys())
        
        if num_defns != self.num_cell_groups:
            raise StandardError("Number of cell groups does not equal number of keys in intercellular_contact_factor_magnitudes_defn.")
        
        intercellular_contact_factor_magnitudes = []
        for cgi in range(self.num_cell_groups):
            cg = self.cell_group_defns[cgi]
            cg_name = cg['cell_group_name']
            intercellular_contact_factor_mag = intercellular_contact_factor_magnitudes_defn[cg_name]
            
            intercellular_contact_factor_magnitudes += (self.cell_group_defns[cgi]['num_cells'])*[intercellular_contact_factor_mag]
                
        return np.array(intercellular_contact_factor_magnitudes)
        
# -----------------------------------------------------------------

    def cell_dependent_coa_signal_strengths(self, cell_group_index, this_cell_group_defn):
        cell_dependent_coa_signal_strengths_defn = this_cell_group_defn['cell_dependent_coa_signal_strengths_defn']
        
        num_defns = len(cell_dependent_coa_signal_strengths_defn.keys())
        
        if num_defns != self.num_cell_groups:
            raise StandardError("Number of cell groups does not equal number of keys in intercellular_contact_factor_magnitudes_defn.")
        
        cell_dependent_coa_signal_strengths = []
        for cgi in range(self.num_cell_groups):
            cg = self.cell_group_defns[cgi]
            cg_name = cg['cell_group_name']
            cg_num_nodes = self.num_nodes_per_cell
            coa_signal_strength = cell_dependent_coa_signal_strengths_defn[cg_name]/cg_num_nodes
            
            cell_dependent_coa_signal_strengths += (self.cell_group_defns[cgi]['num_cells'])*[coa_signal_strength]
                
        return np.array(cell_dependent_coa_signal_strengths)
        
# -----------------------------------------------------------------
 
    def create_cell_group(self, num_timesteps, cell_group_defn, cell_group_index):
#        for variable_name in ['cell_group_name', 'num_cells', 'init_cell_radius', 'num_nodes_per_cell', 'C_total', 'H_total', 'cell_group_bounding_box', 'chem_mech_space_defns', 'integration_params']:
#            print(variable_name + ' = ' + "cell_group_defn['" + variable_name + "']")
        cell_group_name = cell_group_defn['cell_group_name']
        num_cells = cell_group_defn['num_cells']
        init_cell_radius = cell_group_defn['init_cell_radius']
        num_nodes = self.num_nodes_per_cell
        C_total = cell_group_defn['C_total']
        H_total = cell_group_defn['H_total']
        cell_group_bounding_box = cell_group_defn['cell_group_bounding_box']
        chem_mech_space_defns = cell_group_defn['chem_mech_space_defns']
        
        biased_rgtpase_distrib_defns = cell_group_defn['biased_rgtpase_distrib_defns']
        cells_with_bias_info = biased_rgtpase_distrib_defns.keys()    
            
        integration_params = self.integration_params
        intercellular_contact_factor_magnitudes = self.make_intercellular_contact_factor_magnitudes(cell_group_index, cell_group_defn)
        coa_factor_production_rates = self.cell_dependent_coa_signal_strengths(cell_group_index, cell_group_defn)

        init_cell_bounding_boxes = self.calculate_cell_bounding_boxes(num_cells, init_cell_radius, cell_group_bounding_box)
        
        cells_in_group = []
        
        for cell_number, bounding_box in enumerate(init_cell_bounding_boxes):
            bias_defn = []
            if len(cells_with_bias_info) > 0:
                if cell_number in cells_with_bias_info:
                    bias_defn = biased_rgtpase_distrib_defns[cell_number]
                else:
                    bias_defn = biased_rgtpase_distrib_defns["default"]
            else:
                raise StandardError("No default initial rGTPase distribution bias information provided!")
            init_node_coords, length_edge_resting, area_resting = self.create_default_init_cell_node_coords(bounding_box, init_cell_radius, num_nodes)
            
            cells_in_group.append(cell.Cell(str(cell_group_name) + '_' +  str(cell_number), cell_group_index, cell_number, integration_params, num_timesteps, self.T, C_total, H_total, init_node_coords, biased_rgtpase_distrib_defn=bias_defn, intercellular_contact_factor_magnitudes=intercellular_contact_factor_magnitudes, radius_resting=init_cell_radius, length_edge_resting=length_edge_resting, area_resting=area_resting, space_physical_bdry_polygon=self.space_physical_bdry_polygon, space_migratory_bdry_polygon=self.space_migratory_bdry_polygon, cell_dependent_coa_signal_strengths=coa_factor_production_rates, verbose=self.verbose, **chem_mech_space_defns))
            
        return cells_in_group, init_cell_bounding_boxes
# -----------------------------------------------------------------
    
    def calculate_cell_bounding_boxes(self, num_cells, init_cell_radius, cell_group_bounding_box):
        
        cell_bounding_boxes = np.zeros((num_cells, 4))
        xmin, xmax, ymin, ymax = cell_group_bounding_box
        x_length = xmax - xmin
        y_length = ymax - ymin
        
        cell_diameter = 2*init_cell_radius

        # check if cells can fit in given bounding box
        total_cell_group_area = num_cells*(np.pi*init_cell_radius**2)
        cell_group_bounding_box_area = abs(x_length*y_length)
        
        if total_cell_group_area > cell_group_bounding_box_area:
            raise StandardError("Cell group bounding box is not big enough to contain all cells given init_cell_radius constraint.")
        
        num_cells_along_x = custom_floor(np.abs(x_length/cell_diameter), 1e-4)
        num_cells_along_y = custom_floor(np.abs(y_length/cell_diameter), 1e-4)
        
        cell_x_coords = xmin + np.sign(x_length)*np.arange(num_cells_along_x)*cell_diameter
        cell_y_coords = ymin + np.sign(y_length)*np.arange(num_cells_along_y)*cell_diameter
        
        M = np.meshgrid(cell_x_coords, cell_y_coords)
        x_values_on_grid = M[0]
        y_values_on_grid = M[1]
        x_y = np.dstack((x_values_on_grid, y_values_on_grid))
        x_y = x_y.reshape((num_cells_along_x*num_cells_along_y, 2))
        
        x_step = np.sign(x_length)*cell_diameter
        y_step = np.sign(y_length)*cell_diameter
        for i, bb_lower_left_corner in enumerate(x_y):
            if i == num_cells:
                break
            
            x_coord, y_coord = bb_lower_left_corner
            cell_bounding_boxes[i] = [x_coord, x_coord + x_step, y_coord, y_coord + y_step]
                
        return cell_bounding_boxes
                

# -----------------------------------------------------------------
         
    def create_default_init_cell_node_coords(self, bounding_box, init_cell_radius, num_nodes):
        cell_centre = calc_bounding_box_centre(bounding_box)
        
        cell_node_thetas = np.pi*np.linspace(0, 2, endpoint=False, num=num_nodes)
        cell_node_coords = np.transpose(np.array([cell_centre[0] + init_cell_radius*np.cos(cell_node_thetas), cell_centre[1] + init_cell_radius*np.sin(cell_node_thetas)]))
        
        edge_vectors = geometry.calculate_edge_vectors(num_nodes, cell_node_coords)
        
        edge_lengths = geometry.calculate_2D_vector_mags(num_nodes, edge_vectors)
        
        length_edge_resting = np.average(edge_lengths)
        
        area_resting = geometry.calculate_polygon_area(num_nodes, cell_node_coords)
        if area_resting < 0:
            raise StandardError("Resting area was calculated to be negative.")
        return cell_node_coords, length_edge_resting, area_resting

# -----------------------------------------------------------------
            
    def execute_system_dynamics_in_random_sequence(self, t, cells_node_distance_matrix, cells_bounding_box_array, environment_cells_node_coords, environment_cells):        
        execution_sequence = self.cell_indices
        np.random.shuffle(execution_sequence)
        
        first_cell_index = execution_sequence[0]
        last_cell_index = execution_sequence[-1]
        
        for cell_index in execution_sequence:
            current_cell = environment_cells[cell_index]
            if self.verbose == True:
                if self.full_print:
                    if cell_index != first_cell_index:
                        print "-"*40
                    else:
                        print "="*40
                        print "Time step: {}/{}".format(t, self.num_timesteps)
                    
                    print "Executing dyanmics for cell: ", cell_index
                
            current_cell.execute_step(cell_index, self.num_nodes_per_cell, environment_cells_node_coords, cells_node_distance_matrix[cell_index], cells_bounding_box_array, be_talkative=self.full_print)
            
            if current_cell.skip_dynamics == False:
                this_cell_coords = current_cell.curr_node_coords*current_cell.L
                environment_cells_node_coords[cell_index] = this_cell_coords
                
                cells_node_distance_matrix = geometry.update_distance_squared_matrix(cell_index, self.num_cells, self.num_nodes_per_cell, environment_cells_node_coords, cells_node_distance_matrix)
                cells_bounding_box_array[cell_index] = geometry.calculate_bounding_box_polygon(this_cell_coords)
            
            if self.verbose == True:
                if self.full_print:
                    if cell_index == last_cell_index:
                        print "="*40
                
        return cells_node_distance_matrix, cells_bounding_box_array, environment_cells_node_coords
            
# -----------------------------------------------------------------
            
    def make_visuals(self, t, visuals_save_dir, animation_settings, animation_obj, produce_animations, produce_graphs):
        if produce_graphs:
            for cell_index, a_cell in enumerate(self.cells_in_environment):
                if a_cell.skip_dynamics == True:
                    continue
                
                save_dir_for_cell = os.path.join(visuals_save_dir, "cell_{}".format(cell_index))
                
                if not os.path.exists(save_dir_for_cell):
                    os.makedirs(save_dir_for_cell)
                
                averaged_score, scores_per_tstep = analysis_utils.calculate_rgtpase_polarity_score(a_cell, significant_difference=0.2, max_tstep=t)
        
                datavis.graph_important_cell_variables_over_time(a_cell, polarity_scores=scores_per_tstep, save_name='C={}'.format(cell_index) + '_important_cell_vars_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_rates(a_cell, save_name='C={}'.format(cell_index) + '_rates_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_strains(a_cell, save_name='C={}'.format(cell_index) + '_strain_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                #datavis.graph_run_and_tumble_statistics(a_cell, save_name='C={}'.format(cell_index) + '_r_and_t_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                #datavis.graph_pre_post_contact_cell_kinematics(a_cell, save_name='C={}'.format(cell_index) + '_pre_post_collision_kinematics_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                
            datavis.graph_cell_velocity_over_time(self.cells_in_environment, save_name='cell_velocities_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
            
            if self.num_cells > 2:
                datavis.graph_delaunay_triangulation_area_over_time(self.cells_in_environment, save_name='delaunay_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
            
            datavis.graph_centroid_related_data(self.cells_in_environment, save_name='centroid_data_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
        
        if produce_animations:
            animation_obj.create_animation_from_data(visuals_save_dir, num_timesteps=t)
            
        
# -----------------------------------------------------------------      
    
    def execute_system_dynamics(self, animation_settings,  produce_intermediate_visuals=True, produce_final_visuals=True, elapsed_timesteps_before_producing_intermediate_graphs=2500, elapsed_timesteps_before_producing_intermediate_animations=5000, given_pool_for_making_visuals=None):
        num_cells = self.num_cells
        num_nodes_per_cell = self.num_nodes_per_cell
        
        environment_cells = self.cells_in_environment
        environment_cells_node_coords = np.array([x.curr_node_coords*x.L for x in environment_cells])
        
        cell_node_distance_matrix = geometry.create_initial_distance_squared_matrix(num_cells, num_nodes_per_cell, environment_cells_node_coords)
        cells_bounding_box_array = geometry.create_initial_bounding_box_polygon_array(num_cells, num_nodes_per_cell, environment_cells_node_coords)
        
        if self.environment_filepath == None:
            animation_obj = None
            produce_intermediate_visuals = False
            produce_final_visuals = False
        else:
            animation_obj = animator.EnvironmentAnimation(os.path.join(self.environment_filepath), self, **animation_settings)
            
        if self.curr_t == 0 or self.curr_t < self.max_t:
            last_timestep_when_animations_made = 0
            last_timestep_when_graphs_made = 0
            
            for t in self.timepoints[:-1]:
                if produce_intermediate_visuals == True:
                    if t - last_timestep_when_animations_made >= elapsed_timesteps_before_producing_intermediate_animations:
                
                        last_timestep_when_animations_made = t
                        
                        visuals_save_dir = os.path.join(self.environment_filepath, 'T={}'.format(t))
                        if not os.path.exists(visuals_save_dir):
                            os.makedirs(visuals_save_dir)
                        
                        print "Making intermediate animations..."
                        self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, True, False)
                        
                    if t - last_timestep_when_graphs_made >= elapsed_timesteps_before_producing_intermediate_graphs:
                        
                        last_timestep_when_graphs_made = t
                        
                        visuals_save_dir = os.path.join(self.environment_filepath, 'T={}'.format(t))
                        if not os.path.exists(visuals_save_dir):
                            os.makedirs(visuals_save_dir)
                        
                        self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, False, True)
                    
                cell_node_distance_matrix, cells_bounding_box_array, environment_cells_node_coords = self.execute_system_dynamics_in_random_sequence(t, cell_node_distance_matrix, cells_bounding_box_array, environment_cells_node_coords, environment_cells)      
        else:
            raise StandardError("max_t has already been reached.")
        
        if self.environment_filepath != None and self.persist == True:
            print "Compressing and storing environment..."
            pkl_file_path = os.path.join(self.environment_filepath, "{}.pkl".format(self.environment_name))
            compressed_pkl_file_path = os.path.join(self.environment_filepath, "{}.pkl.gz".format(self.environment_name))
            
            with open(pkl_file_path, 'w') as f:
                pickling_package.dump(self, f)
            
            with open(pkl_file_path, 'r') as f_in, gzip.open(compressed_pkl_file_path, 'w') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
            os.remove(pkl_file_path)
            
        if produce_final_visuals == True:
            t = self.num_timepoints
            visuals_save_dir = os.path.join(self.environment_filepath, 'T={}'.format(t))
            
            if not os.path.exists(visuals_save_dir):
                os.makedirs(visuals_save_dir)
                
                print "Making final visuals..."
                self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, True, True)

# -----------------------------------------------------------------
    