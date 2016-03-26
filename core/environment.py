from __future__ import division
import numpy as np
import cell
from parameterorg import *
import geometry
import os
import visualization.datavis as datavis
import analysis.utilities as analysis_utils
import visualization.animator as animator
#import dill as pickling_package
#import gzip
#import copy
#import shutil
import numba as nb
import time
import core.hardio as hardio

"""
Environment of cells.
"""

@nb.jit(nopython=True)
def custom_floor(fp_number, roundoff_distance):
    a = int(fp_number)
    b = a + 1
    
    if abs(fp_number - b) < roundoff_distance:
        return b
    else:
        return a 
# -----------------------------------------------------------------
@nb.jit(nopython=True)
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
    def __init__(self, environment_name='', num_timesteps=0, space_physical_bdry_polygon=np.array([], dtype=np.float64), space_migratory_bdry_polygon=np.array([], dtype=np.float64), external_gradient_fn=lambda x: 0, cell_group_defns=None, environment_dir=None, verbose=True, T=(1/0.5), num_nodes=16, integration_params={}, full_print=False, persist=True, parameter_explorer_run=False, max_timepoints_on_ram=1000): 
        
        self.last_timestep_when_animations_made = None
        self.last_timestep_when_environment_hard_saved = None
        self.last_timestep_when_graphs_made = None
        
        self.parameter_explorer_run = parameter_explorer_run        
        if parameter_explorer_run == True:
            self.verbose = False
            self.persist = False
            self.environment_name = None
            self.environment_dir = None
        else:
            self.verbose = verbose
            self.persist = persist
            self.environment_name = environment_name
            self.environment_dir = environment_dir
            
        if environment_dir != None:
            self.storefile_path = os.path.join(environment_dir, "store.hdf5")
        else:
            self.storefile_path = None
        
        self.space_physical_bdry_polygon = space_physical_bdry_polygon
        self.space_migratory_bdry_polygon = space_migratory_bdry_polygon
        self.external_gradient_fn = external_gradient_fn
        self.cell_group_defns = cell_group_defns
        
        self.curr_t = 0
        self.T = T
        self.num_timesteps = num_timesteps
        self.num_timepoints = num_timesteps + 1
        self.timepoints = np.arange(self.curr_t, self.num_timepoints)
        self.max_t = num_timesteps
        
        self.integration_params = integration_params
        
        self.num_nodes = num_nodes
        
        self.micrometer = 1e-6
        
        self.num_cell_groups = len(cell_group_defns)
        self.num_cells = np.sum([cell_group_defn['num_cells'] for cell_group_defn in cell_group_defns])
        self.max_timepoints_on_ram = max_timepoints_on_ram
        self.cells_in_environment = self.make_cells()
        self.full_output_dicts = [[] for cell in self.cells_in_environment]
        
        self.cell_indices = np.arange(self.num_cells)
        self.full_print = full_print

# -----------------------------------------------------------------

    def make_cells(self):
        cells_in_environment = []
        cell_bounding_boxes_wrt_time = []
    
        cell_index_offset = 0
        for cell_group_index, cell_group_defn in enumerate(self.cell_group_defns):
            cells_in_group, init_cell_bounding_boxes = self.create_cell_group(self.num_timesteps, cell_group_defn, cell_group_index, cell_index_offset)
            cell_index_offset += len(cells_in_group)
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
            cg_num_nodes = self.num_nodes
            coa_signal_strength = cell_dependent_coa_signal_strengths_defn[cg_name]/cg_num_nodes
            
            cell_dependent_coa_signal_strengths += (self.cell_group_defns[cgi]['num_cells'])*[coa_signal_strength]
                
        return np.array(cell_dependent_coa_signal_strengths)
        
# -----------------------------------------------------------------
 
    def create_cell_group(self, num_timesteps, cell_group_defn, cell_group_index, cell_index_offset):
#        for variable_name in ['cell_group_name', 'num_cells', 'init_cell_radius', 'num_nodes', 'C_total', 'H_total', 'cell_group_bounding_box', 'chem_mech_space_defns', 'integration_params']:
#            print(variable_name + ' = ' + "cell_group_defn['" + variable_name + "']")
        cell_group_name = cell_group_defn['cell_group_name']
        num_cells = cell_group_defn['num_cells']
        init_cell_radius = cell_group_defn['init_cell_radius']
        num_nodes = self.num_nodes
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
            
            cell_index = cell_index_offset + cell_number
            
            cells_in_group.append(cell.Cell(str(cell_group_name) + '_' +  str(cell_index), cell_group_index, cell_index, integration_params, num_timesteps, self.T, C_total, H_total, init_node_coords, self.max_timepoints_on_ram, biased_rgtpase_distrib_defn=bias_defn, intercellular_contact_factor_magnitudes=intercellular_contact_factor_magnitudes, radius_resting=init_cell_radius, length_edge_resting=length_edge_resting, area_resting=area_resting, space_physical_bdry_polygon=self.space_physical_bdry_polygon, space_migratory_bdry_polygon=self.space_migratory_bdry_polygon, cell_dependent_coa_signal_strengths=coa_factor_production_rates, verbose=self.verbose, **chem_mech_space_defns))
            
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
            
    def execute_system_dynamics_in_random_sequence(self, t, cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells):        
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
            
            current_cell.execute_step(cell_index, self.num_nodes, environment_cells_node_coords, cells_node_distance_matrix[cell_index], cells_line_segment_intersection_matrix[cell_index], self.external_gradient_fn, be_talkative=self.full_print)
            
            if current_cell.skip_dynamics == False:
                this_cell_coords = current_cell.curr_node_coords*current_cell.L
                environment_cells_node_coords[cell_index] = this_cell_coords
                
                cells_bounding_box_array[cell_index] = geometry.calculate_polygon_bounding_box(this_cell_coords)
                cells_node_distance_matrix, cells_line_segment_intersection_matrix = geometry.update_line_segment_intersection_and_dist_squared_matrices(cell_index, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_bounding_box_array, cells_node_distance_matrix, cells_line_segment_intersection_matrix)
                #cells_node_distance_matrix = geometry.update_distance_squared_matrix(cell_index, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_node_distance_matrix)
                
            
            if self.verbose == True:
                if self.full_print:
                    if cell_index == last_cell_index:
                        print "="*40
                
        return cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords
            
# -----------------------------------------------------------------
            
    def make_visuals(self, t, visuals_save_dir, animation_settings, animation_obj, produce_animations, produce_graphs):
        if produce_graphs:
            for cell_index in xrange(self.num_cells):
                if self.cells_in_environment[cell_index].skip_dynamics == True:
                    continue
                
                save_dir_for_cell = os.path.join(visuals_save_dir, "cell_{}".format(cell_index))
                
                if not os.path.exists(save_dir_for_cell):
                    os.makedirs(save_dir_for_cell)
                
                averaged_score, scores_per_tstep = analysis_utils.calculate_rgtpase_polarity_score(cell_index, self.storefile_path, significant_difference=0.2, max_tstep=t)
        
                datavis.graph_important_cell_variables_over_time(cell_index, self.storefile_path,  polarity_scores=scores_per_tstep, save_name='C={}'.format(cell_index) + '_important_cell_vars_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_rates(cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_rates_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_strains(cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_strain_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                #datavis.graph_run_and_tumble_statistics(a_cell, save_name='C={}'.format(cell_index) + '_r_and_t_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                #datavis.graph_pre_post_contact_cell_kinematics(a_cell, save_name='C={}'.format(cell_index) + '_pre_post_collision_kinematics_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                
            datavis.graph_cell_velocity_over_time(self.storefile_path, save_name='cell_velocities_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
            
            if self.num_cells > 2:
                datavis.graph_delaunay_triangulation_area_over_time(self.storefile_path, save_name='delaunay_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
            
            datavis.graph_centroid_related_data(self.storefile_path, save_name='centroid_data_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
        
        if produce_animations:
            animation_obj.create_animation_from_data(visuals_save_dir, num_timesteps=t)
            
# -----------------------------------------------------------------    
    
    def get_empty_self_copy(self):
        empty_self = Environment()
    
        for attr_name, attr_value in inspect.getmembers(self):
            if attr_name != "cells_in_environment":
                setattr(empty_self, attr_name, attr_value)
        
        return empty_self
        
# ----------------------------------------------------------------- 
        
    def dump_cells_data(self, tpoint):
        for cell_index in xrange(self.num_cells):
            this_cell = self.cells_in_environment[cell_index]
            if this_cell.last_trim_timestep < 0:
                hardio.append_cell_data_to_dataset(cell_index, this_cell.system_info, self.storefile_path)
                this_cell.trim_system_info(tpoint)
            elif this_cell.last_trim_timestep < tpoint:
                hardio.append_cell_data_to_dataset(cell_index, this_cell.system_info[0:], self.storefile_path)
                this_cell.trim_system_info(tpoint)
            else:
                continue
            
    
# ----------------------------------------------------------------- 
        
    def execute_system_dynamics(self, animation_settings,  produce_intermediate_visuals=True, produce_final_visuals=True, elapsed_timesteps_before_producing_intermediate_graphs=2500, elapsed_timesteps_before_producing_intermediate_animations=5000, given_pool_for_making_visuals=None):
        simulation_st = time.time()
        num_cells = self.num_cells
        num_nodes = self.num_nodes
        
        environment_cells = self.cells_in_environment
        environment_cells_node_coords = np.array([x.curr_node_coords*x.L for x in environment_cells])
        
        cells_bounding_box_array = geometry.create_initial_bounding_box_polygon_array(num_cells, num_nodes, environment_cells_node_coords)
        cells_node_distance_matrix, cells_line_segment_intersection_matrix = geometry.create_initial_line_segment_intersection_and_dist_squared_matrices(num_cells, num_nodes, cells_bounding_box_array, environment_cells_node_coords)
            
        #cells_node_distance_matrix = geometry.create_initial_distance_squared_matrix(num_cells, num_nodes, environment_cells_node_coords)
    
        if self.environment_dir == None:
            animation_obj = None
            produce_intermediate_visuals = False
            produce_final_visuals = False
        else:
            animation_obj = animator.EnvironmentAnimation(self.environment_dir, self.get_empty_self_copy(), **animation_settings)
            
        if self.curr_t == 0 or self.curr_t < self.max_t:
            if self.last_timestep_when_animations_made == None:
                self.last_timestep_when_animations_made = self.curr_t
            if self.last_timestep_when_graphs_made == None:
                self.last_timestep_when_graphs_made = self.curr_t
            if self.last_timestep_when_environment_hard_saved == None:
                self.last_timestep_when_environment_hard_saved = self.curr_t
            
            for t in self.timepoints[self.curr_t:-1]:
                if produce_intermediate_visuals == True:
                    if t - self.last_timestep_when_animations_made >= elapsed_timesteps_before_producing_intermediate_animations:
                
                        self.last_timestep_when_animations_made = t
                        
                        visuals_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
                        if not os.path.exists(visuals_save_dir):
                            os.makedirs(visuals_save_dir)
                        
                        print "Making intermediate animations..."
                        self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, True, False)
                        
                    if t - self.last_timestep_when_graphs_made >= elapsed_timesteps_before_producing_intermediate_graphs:
                        
                        self.last_timestep_when_graphs_made = t
                        
                        visuals_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
                        if not os.path.exists(visuals_save_dir):
                            os.makedirs(visuals_save_dir)
                        
                        self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, False, True)
            
                if t - self.last_timestep_when_environment_hard_saved >= self.max_timepoints_on_ram:
                    self.last_timestep_when_environment_hard_saved = t
                    self.dump_cells_data(t)
                    
                cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords = self.execute_system_dynamics_in_random_sequence(t, cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells)
                self.curr_t += 1
        else:
            raise StandardError("max_t has already been reached.")
        
        simulation_et = time.time()
            
        if produce_final_visuals == True:
            t = self.num_timepoints
            visuals_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
            
            if not os.path.exists(visuals_save_dir):
                os.makedirs(visuals_save_dir)
                
                print "Making final visuals..."
                self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, True, True)
                
        if self.environment_dir != None and self.persist == True:
            self.dump_cells_data()
            
        simulation_time = np.round(simulation_et - simulation_st, decimals=2)
        print "Time taken to complete simulation: {}s".format(simulation_time)
                
        return simulation_time

# -----------------------------------------------------------------
    