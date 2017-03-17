from __future__ import division
import numpy as np
import cell
import parameterorg
import geometry
import os
import visualization.datavis as datavis
import analysis.utilities as analysis_utils
import visualization.animator as animator
import dill
import cPickle
import numba as nb
import time
import core.hardio as hardio
import copy
import warnings

"""
Environment of cells.
"""

MODE_EXECUTE = 0
MODE_OBSERVE = 1

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
    def __init__(self, environment_name='', num_timesteps=0, space_physical_bdry_polygon=np.array([], dtype=np.float64), space_migratory_bdry_polygon=np.array([], dtype=np.float64), external_gradient_fn=lambda x: 0.0, cell_group_defns=None, environment_dir=None, verbose=True, T=(1/0.5), integration_params={}, full_print=False, persist=True, parameter_explorer_run=False, max_timepoints_on_ram=1000, seed=None, allowed_drift_before_geometry_recalc=1.0): 
        
        self.last_timestep_when_animations_made = None
        self.last_timestep_when_environment_hard_saved = None
        self.last_timestep_when_graphs_made = None
        
        self.parameter_explorer_run = parameter_explorer_run        
        if parameter_explorer_run == True:
            self.verbose = False
            self.environment_name = None
            self.environment_dir = None
            self.full_print = False
            self.persist = False
            self.max_timepoints_on_ram = None
        else:
            self.verbose = verbose
            self.environment_name = environment_name
            self.environment_dir = environment_dir
            self.full_print = full_print
            self.persist = persist
            self.max_timepoints_on_ram = max_timepoints_on_ram
        
        if environment_dir != None:
            self.init_random_state(seed)
            self.storefile_path = os.path.join(environment_dir, "store.hdf5")
            self.empty_self_pickle_path = os.path.join(environment_dir, "environment.pkl")
        else:
            self.storefile_path = None
            self.empty_self_pickle_path = None
        
        self.space_physical_bdry_polygon = space_physical_bdry_polygon
        self.space_migratory_bdry_polygon = space_migratory_bdry_polygon
        self.external_gradient_fn = external_gradient_fn
        self.cell_group_defns = cell_group_defns
        
        self.curr_tpoint = 0
        self.timestep_offset_due_to_dumping = 0
        self.T = T
        self.num_timesteps = num_timesteps
        self.num_timepoints = num_timesteps + 1
        self.timepoints = np.arange(0, self.num_timepoints)
        
        self.integration_params = integration_params
        
        self.micrometer = 1e-6
        
        self.num_cell_groups = len(self.cell_group_defns)
        self.num_cells = np.sum([cell_group_defn['num_cells'] for cell_group_defn in self.cell_group_defns], dtype=np.int64)
        self.allowed_drift_before_geometry_recalc = allowed_drift_before_geometry_recalc*self.num_cells
        self.cells_in_environment = self.make_cells()
        num_nodes_per_cell = np.array([x.num_nodes for x in self.cells_in_environment], dtype=np.int64)
        self.num_nodes= num_nodes_per_cell[0]
        for n in num_nodes_per_cell[1:]:
            if n != self.num_nodes:
                raise StandardError("There exists a cell with number of nodes different from other cells!")
        self.full_output_dicts = [[] for cell in self.cells_in_environment]
        
        self.cell_indices = np.arange(self.num_cells)
        self.exec_orders = np.zeros((self.num_timepoints, self.num_cells), dtype=np.int64)
        self.full_print = full_print
        
        for cell_index in xrange(self.num_cells):
            if self.environment_dir != None:
                hardio.create_cell_dataset(cell_index, self.storefile_path, self.num_nodes, parameterorg.num_info_labels)
            
        self.mode = MODE_EXECUTE

# -----------------------------------------------------------------
    def write_random_state_file(self, random_state_fp):
        with open(random_state_fp, 'wb') as f:
            cPickle.dump(np.random.get_state(), f)
        
    def set_random_state_from_file(self, random_state_fp):
        with open(random_state_fp, 'rb') as f:
            np.random.set_state(cPickle.load(f))
        
    def init_random_state(self, seed):
        if self.environment_dir != None:
            random_state_fp = os.path.join(self.environment_dir, "random_state.pkl")
            if os.path.exists(random_state_fp):
                self.set_random_state_from_file(random_state_fp)
            else:
                np.random.seed(seed)
                self.write_random_state_file(random_state_fp)
                
# -----------------------------------------------------------------

    def get_system_history_index(self, tpoint):
        return tpoint - self.timestep_offset_due_to_dumping

    def extend_simulation_runtime(self, new_num_timesteps):
        self.num_timesteps = new_num_timesteps
        self.num_timepoints = self.num_timesteps + 1
        self.timepoints = np.arange(0, self.num_timepoints)
        
        for a_cell in self.cells_in_environment:
            a_cell.num_timepoints = self.num_timepoints
            
        old_exec_orders = np.copy(self.exec_orders)
        self.exec_orders = np.zeros((self.num_timepoints, self.num_cells), dtype=np.int64)
        self.exec_orders[:old_exec_orders.shape[0]] = old_exec_orders
        
        
    def simulation_complete(self):
        return self.num_timesteps == self.curr_tpoint
        
        
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
 
    def create_cell_group(self, num_timesteps, cell_group_defn, cell_group_index, cell_index_offset):
#        for variable_name in ['cell_group_name', 'num_cells', 'init_cell_radius', 'num_nodes', 'C_total', 'H_total', 'cell_group_bounding_box', 'chem_mech_space_defns', 'integration_params']:
#            print(variable_name + ' = ' + "cell_group_defn['" + variable_name + "']")
        cell_group_name = cell_group_defn['cell_group_name']
        num_cells = cell_group_defn['num_cells']
        cell_group_bounding_box = cell_group_defn['cell_group_bounding_box']
        
        cell_parameter_dict = copy.deepcopy(cell_group_defn['parameter_dict'])
        init_cell_radius = cell_parameter_dict['init_cell_radius']
        num_nodes = cell_parameter_dict['num_nodes']
        
        biased_rgtpase_distrib_defns = cell_group_defn['biased_rgtpase_distrib_defns']
        cells_with_bias_info = biased_rgtpase_distrib_defns.keys()    
            
        integration_params = self.integration_params

        init_cell_bounding_boxes = self.calculate_cell_bounding_boxes(num_cells, init_cell_radius, cell_group_bounding_box)
        
        cells_in_group = []
        
        for cell_number, bounding_box in enumerate(init_cell_bounding_boxes):
            bias_defn = biased_rgtpase_distrib_defns["default"]
            
            if cell_number in cells_with_bias_info:
                bias_defn = biased_rgtpase_distrib_defns[cell_number]
            
            init_node_coords, length_edge_resting, area_resting = self.create_default_init_cell_node_coords(bounding_box, init_cell_radius, num_nodes)
            
            cell_parameter_dict.update([('biased_rgtpase_distrib_defn', bias_defn), ('init_node_coords', init_node_coords), ('length_edge_resting', length_edge_resting), ('area_resting', area_resting)])
            
            cell_index = cell_index_offset + cell_number

            # cell_label, cell_group_index, cell_index, integration_params, num_timesteps, T, num_cells_in_environment, max_timepoints_on_ram, verbose, parameters_dict
            
            undefined_labels = parameterorg.find_undefined_labels(cell_parameter_dict)
            if len(undefined_labels) > 0:
                raise StandardError("The following labels are not yet defined: {}".format(undefined_labels))
            
            new_cell = cell.Cell(str(cell_group_name) + '_' +  str(cell_index), cell_group_index, cell_index, integration_params, num_timesteps, self.T, self.num_cells, self.max_timepoints_on_ram, self.verbose, cell_parameter_dict)
            
            cells_in_group.append(new_cell)
            
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
    def execute_system_dynamics_in_random_sequence(self, t, cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces, environment_cells, centroid_drift, recalc_geometry):        
        execution_sequence = self.cell_indices
        np.random.shuffle(execution_sequence)
        
        self.exec_orders[t] = np.copy(execution_sequence)
        
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
                    
                    print "centroid_drift: ", centroid_drift
                    if recalc_geometry:
                        print "**GEOMETRY RECALC IN PROGRESS**"
                    print "Time step: {}/{}".format(t, self.num_timesteps)
                    print "Executing dyanmics for cell: ", cell_index
            
            # this_cell_index, num_nodes, all_cells_node_coords, all_cells_node_forces, intercellular_squared_dist_array, line_segment_intersection_matrix, external_gradient_fn, be_talkative=False
            current_cell.execute_step(cell_index, self.num_nodes, environment_cells_node_coords, environment_cells_node_forces, cells_node_distance_matrix[cell_index], cells_line_segment_intersection_matrix[cell_index], self.external_gradient_fn, be_talkative=self.full_print)
            
            if current_cell.skip_dynamics == False:
                this_cell_coords = current_cell.curr_node_coords*current_cell.L
                this_cell_forces = current_cell.curr_node_forces*current_cell.ML_T2
                
                environment_cells_node_coords[cell_index] = this_cell_coords
                environment_cells_node_forces[cell_index] = this_cell_forces
                
                cells_bounding_box_array[cell_index] = geometry.calculate_polygon_bounding_box(this_cell_coords)
                if recalc_geometry:
                    cells_node_distance_matrix, cells_line_segment_intersection_matrix = geometry.update_line_segment_intersection_and_dist_squared_matrices(cell_index, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_bounding_box_array, cells_node_distance_matrix, cells_line_segment_intersection_matrix)
                else:
                    cells_node_distance_matrix = geometry.update_distance_squared_matrix(cell_index, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_node_distance_matrix)
                
            
            if self.verbose == True:
                if self.full_print:
                    if cell_index == last_cell_index:
                        print "="*40

        return cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces
            
# -----------------------------------------------------------------
            
    def make_visuals(self, t, visuals_save_dir, animation_settings, animation_obj, produce_animations, produce_graphs):
        if produce_graphs:
            for cell_index in xrange(self.num_cells):
                this_cell = self.cells_in_environment[cell_index]
                if this_cell.skip_dynamics == True:
                    continue
                
                save_dir_for_cell = os.path.join(visuals_save_dir, "cell_{}".format(cell_index))
                
                if not os.path.exists(save_dir_for_cell):
                    os.makedirs(save_dir_for_cell)
                
                averaged_score, scores_per_tstep = analysis_utils.calculate_rgtpase_polarity_score(cell_index, self.storefile_path, significant_difference=0.2, max_tstep=t)
        
                datavis.graph_important_cell_variables_over_time(self.T/60.0, cell_index, self.storefile_path,  polarity_scores=scores_per_tstep, save_name='C={}'.format(cell_index) + '_important_cell_vars_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_rates(self.T/60.0, this_cell.kgtp_rac_baseline, this_cell.kgtp_rho_baseline, this_cell.kdgtp_rac_baseline, this_cell.kdgtp_rho_baseline, cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_rates_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_strains(self.T/60.0, cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_strain_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_run_and_tumble_statistics(self.num_nodes, self.T/60.0, this_cell.L, cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_r_and_t_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                #datavis.graph_pre_post_contact_cell_kinematics(self.T/60.0, this_cell.L, cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_pre_post_collision_kinematics_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
            
            cell_Ls = np.array([a_cell.L for a_cell in self.cells_in_environment])/1e-6
            
            datavis.graph_cell_velocity_over_time(self.num_cells, self.T/60.0, cell_Ls, self.storefile_path, save_name='cell_velocities_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
            
            datavis.graph_protrusion_lifetimes(self.num_cells, self.T, self.storefile_path, save_dir=visuals_save_dir, max_tstep=t)
            datavis.graph_protrusion_number_given_direction_per_timestep(self.num_cells, self.num_timepoints, self.num_nodes, self.T, self.storefile_path, save_dir=visuals_save_dir, max_tstep=t)
            
            if self.num_cells > 2:
                datavis.graph_delaunay_triangulation_area_over_time(self.num_cells, self.num_timepoints, self.T/60.0, self.storefile_path, save_name='delaunay_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
            
            datavis.graph_centroid_related_data(self.num_cells, self.curr_tpoint, self.T/60.0, cell_Ls, self.storefile_path, save_name='centroid_data_T={}'.format(t-1), save_dir=visuals_save_dir, max_tstep=t)
        
        if produce_animations:
            animation_obj.create_animation_from_data(visuals_save_dir, timestep_to_draw_till=t)
            
# -----------------------------------------------------------------
    def get_empty_cell(self, cell_index):
        empty_cell = copy.deepcopy(self.cells_in_environment[cell_index])
        
        empty_cell.system_history = None
        
        return empty_cell
        
# -----------------------------------------------------------------
      
    def get_empty_cells(self):
        return [self.get_empty_cell(ci) for ci in xrange(self.num_cells)]
        
# -----------------------------------------------------------------
        
    def get_empty_self_copy(self):
        empty_self = copy.deepcopy(self)
        
        empty_self.cells_in_environment = self.get_empty_cells()
        
        return empty_self

# -----------------------------------------------------------------
        
    def is_empty(self):
        for a_cell in self.cells_in_environment:
            if a_cell.system_history != None:
                return False
        
        return True

# -----------------------------------------------------------------
        
    def empty_self(self, preserve_cell_structure=True):
        self.cells_in_environment = self.get_empty_cells()     
        
# ----------------------------------------------------------------- 
        
    def dump_to_store(self, tpoint):
        random_state_fp = os.path.join(self.environment_dir, "random_state.pkl")
        self.write_random_state_file(random_state_fp)
        
        access_index = self.get_system_history_index(self.curr_tpoint)
        
        for cell_index in xrange(self.num_cells):
            this_cell = self.cells_in_environment[cell_index]
            if this_cell.last_trim_timestep < 0:
                hardio.append_cell_data_to_dataset(cell_index, this_cell.system_history[:access_index+1], self.storefile_path)
                this_cell.trim_system_history(tpoint)
            elif this_cell.last_trim_timestep < tpoint:
                hardio.append_cell_data_to_dataset(cell_index, this_cell.system_history[1:access_index+1], self.storefile_path)
                this_cell.trim_system_history(tpoint)
            else:
                continue
            
        self.timestep_offset_due_to_dumping = self.curr_tpoint
        
        with open(self.empty_self_pickle_path, 'wb') as f:
            dill.dump(self.get_empty_self_copy(), f)
        
        
# ----------------------------------------------------------------- 
        
    def init_from_store(self, tpoint=None):
        self.init_random_state(None)
        if tpoint == None:
            tpoint = self.curr_tpoint
        
        for a_cell in self.cells_in_environment:
            a_cell.init_from_storefile(tpoint, self.storefile_path)
            
        self.last_timestep_when_environment_hard_saved = tpoint
            
# ----------------------------------------------------------------- 
    
    def load_curr_tpoint(self):
        self.mode = MODE_OBSERVE
        self.empty_self()
        self.init_from_store(tpoint=self.curr_tpoint)            
            
    def load_next_tpoint(self):
        self.mode = MODE_OBSERVE
        self.empty_self()
        self.orig_tpoint = self.curr_tpoint
        self.curr_tpoint += 1
        self.load_curr_tpoint()
        
    def load_last_tpoint(self):
        self.mode = MODE_OBSERVE
        self.empty_self()
        if self.curr_tpoint == 0:
            warnings.warn("Already at timestep 0!")
        else:
            self.orig_tpoint = self.curr_tpoint
            self.curr_tpoint -= 1
            self.load_curr_tpoint()
            
    def load_at_tpoint(self, tpoint):
        self.mode = MODE_OBSERVE
        self.empty_self()
        if tpoint < 0:
            warnings.warn("Given tstep is less than 0!")
        elif tpoint > self.get_storefile_tstep_range():
            warnings.warn("Given tstep is greater than range of data in storefile!")
        else:
            self.orig_tpoint = self.curr_tpoint 
            self.curr_tpoint = tpoint
            self.load_curr_tpoint()
            
    def get_storefile_tstep_range(self):
        return hardio.get_storefile_tstep_range(self.num_cells, self.storefile_path)
    
# ----------------------------------------------------------------- 
        
    def execute_system_dynamics(self, animation_settings,  produce_intermediate_visuals=True, produce_final_visuals=True, elapsed_timesteps_before_producing_intermediate_graphs=2500, elapsed_timesteps_before_producing_intermediate_animations=5000, given_pool_for_making_visuals=None):
        if self.mode == MODE_EXECUTE:
            allowed_drift_before_geometry_recalc = self.allowed_drift_before_geometry_recalc
            centroid_drift = allowed_drift_before_geometry_recalc*1.2
            simulation_st = time.time()
            num_cells = self.num_cells
            num_nodes = self.num_nodes
            
            environment_cells = self.cells_in_environment
            environment_cells_node_coords = np.array([x.curr_node_coords*x.L for x in environment_cells])
            environment_cells_node_forces = np.array([x.curr_node_forces*x.ML_T2 for x in environment_cells])
            
            cells_bounding_box_array = geometry.create_initial_bounding_box_polygon_array(num_cells, num_nodes, environment_cells_node_coords)
            cells_node_distance_matrix, cells_line_segment_intersection_matrix = geometry.create_initial_line_segment_intersection_and_dist_squared_matrices(num_cells, num_nodes, cells_bounding_box_array, environment_cells_node_coords)
                
            #cells_node_distance_matrix = geometry.create_initial_distance_squared_matrix(num_cells, num_nodes, environment_cells_node_coords)
        
            if self.environment_dir == None:
                animation_obj = None
                produce_intermediate_visuals = False
                produce_final_visuals = False
            else:
                cell_group_indices = []
                cell_Ls = []
                cell_etas = []
                cell_skip_dynamics = []
                
                for a_cell in self.cells_in_environment:
                    cell_group_indices.append(a_cell.cell_group_index)
                    cell_Ls.append(a_cell.L/1e-6)
                    cell_etas.append(a_cell.eta)
                    cell_skip_dynamics.append(a_cell.skip_dynamics)
                    
                animation_obj = animator.EnvironmentAnimation(self.environment_dir, self.environment_name, self.num_cells, self.num_nodes, self.num_timepoints, cell_group_indices, cell_Ls, cell_etas, cell_skip_dynamics, self.storefile_path, **animation_settings)
                
            if self.curr_tpoint == 0 or self.curr_tpoint < self.num_timesteps:
                if self.last_timestep_when_environment_hard_saved == None:
                    self.last_timestep_when_environment_hard_saved = self.curr_tpoint
                
                prev_centroids = geometry.calculate_centroids(environment_cells_node_coords)
                for t in self.timepoints[self.curr_tpoint:-1]:
                    if t - self.last_timestep_when_environment_hard_saved >= self.max_timepoints_on_ram:
                        self.last_timestep_when_environment_hard_saved = t
                        
                        if self.environment_dir != None:
                            self.dump_to_store(t)
                        
                    if produce_intermediate_visuals != False:
                        if t != 0 and t in produce_intermediate_visuals:
                            self.last_timestep_when_environment_hard_saved = t
                            
                            if self.environment_dir != None:
                                self.dump_to_store(t)
                                
                            visuals_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
                            if not os.path.exists(visuals_save_dir):
                                os.makedirs(visuals_save_dir)
                            
                            print "Making intermediate visuals..."
                            self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, True, True)                        

                    if allowed_drift_before_geometry_recalc == 0 or centroid_drift > allowed_drift_before_geometry_recalc:
                        recalc_geometry = True
                    else:
                        recalc_geometry = False
                        
                    cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces = self.execute_system_dynamics_in_random_sequence(t, cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces, environment_cells, centroid_drift, recalc_geometry)
                    self.curr_tpoint += 1
                    if allowed_drift_before_geometry_recalc > 0:
                        curr_centroids = geometry.calculate_centroids(environment_cells_node_coords)
                        if recalc_geometry:
                            centroid_drift = 0.0
                        else:
                            delta_drift = np.sum(geometry.calculate_centroid_dift(prev_centroids, curr_centroids))/1e-6
                            centroid_drift += delta_drift
                        prev_centroids = curr_centroids
            else:
                raise StandardError("max_t has already been reached.")
            
            simulation_et = time.time()
            
            if self.environment_dir != None:
                self.dump_to_store(self.curr_tpoint)
                
            if produce_final_visuals == True:
                t = self.num_timepoints
                visuals_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
                
                if not os.path.exists(visuals_save_dir):
                    os.makedirs(visuals_save_dir)
                    
                    print "Making final visuals..."
                    self.make_visuals(t, visuals_save_dir, animation_settings, animation_obj, True, True)
                
            simulation_time = np.round(simulation_et - simulation_st, decimals=2)
            print "Time taken to complete simulation: {}s".format(simulation_time)
                    
            return simulation_time
        else:
            return 0.0

# -----------------------------------------------------------------
    