from __future__ import division
import numpy as np
import cell
import parameterorg
import geometry
import os
import visualization.datavis as datavis
import utilities as cu
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
        
def place_init_cell_randomly(cell_centers, cell_diameter, corridor_origin, corridor_height, box_height, box_width, init_random_cell_placement_x_factor):
    if init_random_cell_placement_x_factor < 0.0:
        raise StandardError("negative init_random_cell_placement_x_factor given: {}".format(init_random_cell_placement_x_factor))
    center_x = corridor_origin[0] + 0.5*cell_diameter + np.random.rand()*init_random_cell_placement_x_factor*(box_width - cell_diameter)
    center_y = corridor_origin[1] + 0.5*cell_diameter + np.random.rand()*(box_height - cell_diameter)
    
    cell_centers[0] = np.array([center_x, center_y])
    
    return cell_centers

# -----------------------------------------------------------------

def generate_theta_bin_boundaries(num_bins):
    theta_bins = np.zeros((num_bins, 2), dtype=np.float64)
    delta_theta = 2*np.pi/num_bins
    
    for n in xrange(num_bins):
        if n == 0:
            last_boundary = 0.0
        else:
            last_boundary = theta_bins[n-1][1]
            
        theta_bins[n][0] = last_boundary
        theta_bins[n][1] = last_boundary + delta_theta

    return theta_bins

# -----------------------------------------------------------------
         
def generate_trial_theta(theta_bin_boundaries, theta_bin_probabilities):
    tbin_index = np.random.choice(np.arange(theta_bin_boundaries.shape[0]), p=theta_bin_probabilities)
    
    return theta_bin_boundaries[tbin_index][0] + np.random.rand()*(theta_bin_boundaries[tbin_index][1] -  theta_bin_boundaries[tbin_index][0])

# -----------------------------------------------------------------

def update_theta_bin_probabilities(target_bin_index, theta_bin_probabilities):
    num_bins = theta_bin_probabilities.shape[0]
    avg_p = np.average(theta_bin_probabilities)
    orig_p = theta_bin_probabilities[target_bin_index]
    new_p = 0.5*orig_p
    theta_bin_probabilities[target_bin_index] = new_p
    
    interesting_bins = []
    for n in range(num_bins):
        if n == target_bin_index:
            continue
        else:
            if theta_bin_probabilities[n] < avg_p:
                continue
            else:
                interesting_bins.append(n)
                
    num_interesting_bins = len(interesting_bins)
    
    if num_interesting_bins == 0:
        interesting_bins = [n for n in range(num_bins) if n != target_bin_index]
        num_interesting_bins = len(interesting_bins)
    delta_p = new_p/num_interesting_bins
    
    for n in interesting_bins:
        theta_bin_probabilities[n] += delta_p
        
    if np.abs(1.0 - np.sum(theta_bin_probabilities)) > 1e-6:
        raise StandardError("theta_bin_probabilities: {}\nsum: {}\ndelta_p: {}\ninterestin_bins: {}".format(theta_bin_probabilities, np.sum(theta_bin_probabilities), delta_p, interesting_bins))
    
    return theta_bin_probabilities

# -----------------------------------------------------------------

def find_relevant_bin_index(theta, theta_bins):
    for n, tbin in enumerate(theta_bins):
        if tbin[0] <= theta < tbin[1]:
            return n
        
    raise StandardError("could not find a bin for theta = {}! {}".format(theta, theta_bins))
    
# -----------------------------------------------------------------
    
def is_collision(last_placed_cell_index, cell_centers, cell_diameter, corridor_origin, corridor_height, center_x, center_y):
    
    if center_y > corridor_origin[1] + (corridor_height - 0.5*cell_diameter) or center_y < corridor_origin[1] + 0.5*cell_diameter:
        return True
    if center_x < corridor_origin[0] + 0.5*cell_diameter:
        return True

    for n in range(last_placed_cell_index + 1):
        other_x, other_y = cell_centers[n]
        
        if np.sqrt((other_x - center_x)**2 + (other_y - center_y)**2) - cell_diameter < -1e-8:
            return True
        
    return False
         
# -----------------------------------------------------------------    
    
def try_placing_cell_randomly(last_successful_anchor_index, last_placed_cell_index, cell_centers, theta_bins, cell_diameter, corridor_origin, corridor_height, box_height, box_width, max_placement_dist):
    num_trials = 2*theta_bins.shape[0]
    
    theta_bin_probabilities = np.ones(theta_bins.shape[0], dtype=np.float64)/theta_bins.shape[0]
    
    center_x, center_y = cell_centers[last_successful_anchor_index]

    for ti in xrange(num_trials):
        theta = generate_trial_theta(theta_bins, theta_bin_probabilities)
        placement_distance = cell_diameter*((max_placement_dist - 1.0)*np.random.rand() + 1.0)
        dx, dy = placement_distance*np.cos(theta), placement_distance*np.sin(theta)
        test_x, test_y = center_x + dx, center_y + dy
        if is_collision(last_placed_cell_index, cell_centers, cell_diameter, corridor_origin, corridor_height, test_x, test_y):
            theta_bin_probabilities = update_theta_bin_probabilities(find_relevant_bin_index(theta, theta_bins), theta_bin_probabilities)
        else:
            cell_index = last_placed_cell_index + 1
            cell_centers[cell_index] = np.array([test_x, test_y])
            
            return cell_index, cell_centers
        
    return -1, cell_centers

# -----------------------------------------------------------------       
        
def place_cells_randomly(num_cells, cell_diameter, corridor_origin, corridor_height, box_width, box_height, init_random_cell_placement_x_factor, max_placement_distance_factor, num_theta_bins = 20):
    if max_placement_distance_factor < 1.0:
        raise StandardError("max placement distance cannot be < 1.0! Given: {}".format(max_placement_distance_factor))
    cell_centers = np.nan*np.ones((num_cells, 2), dtype=np.float64)
    cell_centers = place_init_cell_randomly(cell_centers, cell_diameter, corridor_origin, corridor_height, box_height, box_width, init_random_cell_placement_x_factor)
    
    num_cells_placed = 1
    possible_anchor_indices = [0]
    trial_anchor_index = 0
    theta_bins = generate_theta_bin_boundaries(num_theta_bins)
    
    while num_cells_placed != num_cells:
        cell_index, cell_centers = try_placing_cell_randomly(possible_anchor_indices[trial_anchor_index], num_cells_placed - 1, cell_centers, theta_bins, cell_diameter, corridor_origin, corridor_height, box_height, box_width, max_placement_distance_factor)
        
        if cell_index != -1:
            num_cells_placed += 1
            possible_anchor_indices.append(cell_index)
        else:
            trial_anchor_index = (trial_anchor_index + 1)%len(possible_anchor_indices)
            
    return cell_centers

# -----------------------------------------------------------------

def generate_bounding_boxes_from_centers(radius, centers):
    bounding_boxes = np.zeros((centers.shape[0], 4), dtype=np.float64)
    
    for i, bb in enumerate(centers):
        x, y = centers[i]
        bounding_boxes[i] = np.array([x - radius, x + radius, y - radius, y + radius])
        
    return bounding_boxes

# -----------------------------------------------------------------

class Environment():
    """Implementation of coupled map lattice model of a cell.
    """
    def __init__(self, environment_name='', num_timesteps=0, space_physical_bdry_polygon=np.array([], dtype=np.float64), space_migratory_bdry_polygon=np.array([], dtype=np.float64), external_gradient_fn=lambda x: 0.0, cell_group_defns=None, environment_dir=None, verbose=True, T=(1/0.5), integration_params={}, full_print=False, persist=True, parameter_explorer_run=False, parameter_explorer_init_rho_gtpase_conditions=None, max_timepoints_on_ram=1000, seed=None, allowed_drift_before_geometry_recalc=1.0, max_geometry_recalc_skips=1000, cell_placement_method="", max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25, convergence_test=False, shell_environment=False): 
        
        self.convergence_test = convergence_test
        
        self.simulation_execution_enabled = True
        
        self.last_timestep_when_animations_made = None
        self.last_timestep_when_environment_hard_saved = None
        self.last_timestep_when_graphs_made = None
        
        self.parameter_explorer_run = parameter_explorer_run
        self.parameter_explorer_init_rho_gtpase_conditions = parameter_explorer_init_rho_gtpase_conditions
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
        
        if self.environment_dir != None:
            self.init_random_state(seed)
            self.storefile_path = os.path.join(self.environment_dir, "store.hdf5")
            self.empty_self_pickle_path = os.path.join(self.environment_dir, "environment.pkl")
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
        
        self.cell_placement_method = cell_placement_method
        self.max_placement_distance_factor = max_placement_distance_factor
        self.init_random_cell_placement_x_factor = init_random_cell_placement_x_factor
        
        self.micrometer = 1e-6
        
        if not shell_environment:
            self.num_cell_groups = len(self.cell_group_defns)
            self.num_cells = np.sum([cell_group_defn['num_cells'] for cell_group_defn in self.cell_group_defns], dtype=np.int64)
        else:
            self.num_cell_groups = 0
            self.num_cells = 0
            
        self.allowed_drift_before_geometry_recalc = allowed_drift_before_geometry_recalc
        self.max_geometry_recalc_skips = max_geometry_recalc_skips
        if not shell_environment:
            self.cells_in_environment = self.make_cells()
            num_nodes_per_cell = np.array([x.num_nodes for x in self.cells_in_environment], dtype=np.int64)
            self.num_nodes= num_nodes_per_cell[0]
            for n in num_nodes_per_cell[1:]:
                if n != self.num_nodes:
                    raise StandardError("There exists a cell with number of nodes different from other cells!")
        else:
            self.cells_in_environment = None
            self.num_nodes = 0
            
        
        
        if not shell_environment:
            self.full_output_dicts = [[] for cell in self.cells_in_environment]
        else:
            self.full_output_dicts = None
        
        self.cell_indices = np.arange(self.num_cells)
        self.exec_orders = np.zeros((self.num_timepoints, self.num_cells), dtype=np.int64)
        self.full_print = full_print
        
        for cell_index in xrange(self.num_cells):
            if self.environment_dir != None:
                hardio.create_cell_dataset(cell_index, self.storefile_path, self.num_nodes, parameterorg.num_info_labels)
        
        self.all_geometry_tasks = np.array(geometry.create_dist_and_line_segment_interesection_test_args(self.num_cells, self.num_nodes), dtype=np.int64)
        self.geometry_tasks_per_cell = np.array([geometry.create_dist_and_line_segment_interesection_test_args_relative_to_specific_cell(ci, self.num_cells, self.num_nodes) for ci in range(self.num_cells)], dtype=np.int64)
        
        self.mode = MODE_EXECUTE
        self.animation_settings = None

# -----------------------------------------------------------------
    def load_from_pickle(self, pickle_fp):
        env = None
        with open(pickle_fp, 'rb') as f:
            env = dill.load(f)
        
        if env == None:
            raise StandardError("Couldn't open pickled environment at: {}".format(pickle_fp))
        else:
            self.__dict__.update(copy.deepcopy(env.__dict__))
        del env

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
        else:
            np.random.seed(seed)
                
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
        cell_group_name = cell_group_defn['cell_group_name']
        num_cells = cell_group_defn['num_cells']
        cell_group_bounding_box = cell_group_defn['cell_group_bounding_box']
        
        cell_parameter_dict = copy.deepcopy(cell_group_defn['parameter_dict'])
        init_cell_radius = cell_parameter_dict['init_cell_radius']
        num_nodes = cell_parameter_dict['num_nodes']
        
        biased_rgtpase_distrib_defns = cell_group_defn['biased_rgtpase_distrib_defns']
        cells_with_bias_info = biased_rgtpase_distrib_defns.keys()    
            
        integration_params = self.integration_params

        init_cell_bounding_boxes = self.calculate_cell_bounding_boxes(num_cells, init_cell_radius, cell_group_bounding_box, cell_placement_method=self.cell_placement_method)
        
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
                
            if self.parameter_explorer_run and self.parameter_explorer_init_rho_gtpase_conditions != None:
                new_cell = cell.Cell(str(cell_group_name) + '_' +  str(cell_index), cell_group_index, cell_index, integration_params, num_timesteps, self.T, self.num_cells, self.max_timepoints_on_ram, self.verbose, cell_parameter_dict, init_rho_gtpase_conditions=self.parameter_explorer_init_rho_gtpase_conditions)
            else:
                new_cell = cell.Cell(str(cell_group_name) + '_' +  str(cell_index), cell_group_index, cell_index, integration_params, num_timesteps, self.T, self.num_cells, self.max_timepoints_on_ram, self.verbose, cell_parameter_dict)
            
            cells_in_group.append(new_cell)
            
        return cells_in_group, init_cell_bounding_boxes
# -----------------------------------------------------------------
    
    def calculate_cell_bounding_boxes(self, num_cells, init_cell_radius, cell_group_bounding_box, cell_placement_method=""):
        
        cell_bounding_boxes = np.zeros((num_cells, 4), dtype=np.float64)
        xmin, xmax, ymin, ymax = cell_group_bounding_box
        x_length = xmax - xmin
        y_length = ymax - ymin
        
        cell_diameter = 2*init_cell_radius

        # check if cells can fit in given bounding box
        total_cell_group_area = num_cells*(np.pi*init_cell_radius**2)
        cell_group_bounding_box_area = abs(x_length*y_length)
        
        if total_cell_group_area > cell_group_bounding_box_area:
            raise StandardError("Cell group bounding box is not big enough to contain all cells given init_cell_radius constraint.")
        
        if cell_placement_method == "":
            num_cells_along_x = custom_floor(x_length/cell_diameter, 1e-6)
            num_cells_along_y = custom_floor(y_length/cell_diameter, 1e-6)
            
            cell_x_coords = xmin + np.sign(x_length)*np.arange(num_cells_along_x)*cell_diameter
            cell_y_coords = ymin + np.sign(y_length)*np.arange(num_cells_along_y)*cell_diameter
            x_step = np.sign(x_length)*cell_diameter
            y_step = np.sign(y_length)*cell_diameter
    
            xi = 0
            yi = 0
            for ci in range(num_cells):
                cell_bounding_boxes[ci] = [cell_x_coords[xi], cell_x_coords[xi] + x_step, cell_y_coords[yi], cell_y_coords[yi] + y_step]
    
                
                if yi == (num_cells_along_y - 1):
                    yi = 0
                    xi += 1
                else:
                    yi += 1
        elif cell_placement_method == "r":
            if len(self.space_migratory_bdry_polygon) == 0:
                corridor_height = -1.0
            else:
                corridor_height = self.space_migratory_bdry_polygon[2][1] - self.space_migratory_bdry_polygon[0][1]
            
            box_width = cell_group_bounding_box[1] - cell_group_bounding_box[0]
            box_height = cell_group_bounding_box[3] - cell_group_bounding_box[2]
            
            cell_bounding_boxes = generate_bounding_boxes_from_centers(0.5*cell_diameter, place_cells_randomly(self.num_cells, cell_diameter, self.space_migratory_bdry_polygon[0], corridor_height, box_width, box_height, self.init_random_cell_placement_x_factor, self.max_placement_distance_factor))
                
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
    def execute_system_dynamics_in_random_sequence(self, t, cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces, environment_cells, centroid_drifts, recalc_geometry):        
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
                    
                    print "centroid_drift: ", centroid_drifts[cell_index]
                    if recalc_geometry[cell_index]:
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
                if recalc_geometry[cell_index]:
                    #cells_node_distance_matrix, cells_line_segment_intersection_matrix =  geometry.update_line_segment_intersection_and_dist_squared_matrices_old(cell_index, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_bounding_box_array, cells_node_distance_matrix, cells_line_segment_intersection_matrix)
                    geometry.update_line_segment_intersection_and_dist_squared_matrices(4, self.geometry_tasks_per_cell[cell_index], self.num_cells, self.num_nodes, environment_cells_node_coords, cells_bounding_box_array, cells_node_distance_matrix, cells_line_segment_intersection_matrix)
                else:
                    geometry.update_distance_squared_matrix(4, self.geometry_tasks_per_cell[cell_index], self.num_cells, self.num_nodes, environment_cells_node_coords, cells_node_distance_matrix)
                    #cells_node_distance_matrix = geometry.update_distance_squared_matrix_old(cell_index, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_node_distance_matrix)
                
            
            if self.verbose == True:
                if self.full_print:
                    if cell_index == last_cell_index:
                        print "="*40

        return cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces
            
# -----------------------------------------------------------------
            
    def do_data_analysis_and_make_visuals(self, t, save_dir, animation_settings, animation_obj, produce_animations, produce_graphs, num_polar_graph_bins=20):
        if self.environment_dir != None:
            data_dict = {}
        else:
            raise StandardError("self.environment_dir is None!")
        
        datavis.add_to_general_data_structure(data_dict, [("T", self.T)])
        
        if produce_graphs:
            data_dict = {}
            for cell_index in xrange(self.num_cells):
                this_cell = self.cells_in_environment[cell_index]
                if this_cell.skip_dynamics == True:
                    continue
                
                save_dir_for_cell = os.path.join(save_dir, "cell_{}".format(cell_index))
                
                if not os.path.exists(save_dir_for_cell):
                    os.makedirs(save_dir_for_cell)
                
                averaged_score, scores_per_tstep = cu.calculate_rgtpase_polarity_score(cell_index, self.storefile_path, significant_difference=0.2, max_tstep=t)
                
                cell_Ls = np.array([a_cell.L for a_cell in self.cells_in_environment])/1e-6
                
                data_dict = datavis.graph_important_cell_variables_over_time(self.T/60.0, cell_Ls[cell_index], cell_index, self.storefile_path, polarity_scores=scores_per_tstep, save_name='C={}'.format(cell_index) + '_important_cell_vars_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t, general_data_structure=data_dict, convergence_test=self.convergence_test)
                datavis.graph_rates(self.T/60.0, this_cell.kgtp_rac_baseline, this_cell.kgtp_rho_baseline, this_cell.kdgtp_rac_baseline, this_cell.kdgtp_rho_baseline, cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_rates_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
                datavis.graph_strains(self.T/60.0, cell_index, self.storefile_path, save_name='C={}'.format(cell_index) + '_strain_graph_T={}'.format(t-1), save_dir=save_dir_for_cell, max_tstep=t)
            
            data_dict = datavis.graph_cell_speed_over_time(self.num_cells, self.T/60.0, cell_Ls, self.storefile_path, save_name='cell_velocities_T={}'.format(t-1), save_dir=save_dir, max_tstep=t, general_data_structure=data_dict, convergence_test=self.convergence_test)
            
            data_dict = datavis.graph_group_area_and_cell_separation_over_time_and_determine_subgroups(self.num_cells, self.num_nodes, t, self.T/60.0, self.storefile_path, save_dir=save_dir, general_data_structure=data_dict)
            
            data_dict = datavis.graph_centroid_related_data(self.num_cells, self.num_timepoints, self.T/60.0, "min.", cell_Ls, self.storefile_path, save_name='centroid_data_T={}'.format(t-1), save_dir=save_dir, max_tstep=t, general_data_structure=data_dict)
            
            protrusion_data_per_cell = cu.collate_protrusion_data(self.num_cells, self.T, self.storefile_path, max_tstep=t)
            protrusion_lifetime_and_direction_data = [x[1] for x in protrusion_data_per_cell]
            datavis.add_to_general_data_structure(data_dict, [("all_cell_protrusion_lifetimes_and_directions", protrusion_lifetime_and_direction_data)])
            protrusion_start_end_cause_data = [x[2] for x in protrusion_data_per_cell]
            protrusion_lifetime_and_direction_data_compiled = np.zeros((0, 2), dtype=np.float64)
            for cell_data in protrusion_lifetime_and_direction_data:
                protrusion_lifetime_and_direction_data_compiled = np.append(protrusion_lifetime_and_direction_data_compiled, np.array(cell_data), axis=0)
                
            datavis.graph_protrusion_lifetimes_radially(protrusion_lifetime_and_direction_data_compiled, num_polar_graph_bins, save_dir=save_dir)
            datavis.graph_protrusion_start_end_causes_radially(protrusion_lifetime_and_direction_data, protrusion_start_end_cause_data, num_polar_graph_bins, save_dir=save_dir)
            
            forward_cones = [(7*np.pi/4, 2*np.pi), (0.0, np.pi/4)]
            backward_cones = [(3*np.pi/4, 5*np.pi/4)]
            protrusion_node_index_and_tpoint_start_ends = [x[0] for x in protrusion_data_per_cell]
            datavis.graph_forward_backward_protrusions_per_timestep(t, protrusion_node_index_and_tpoint_start_ends, protrusion_lifetime_and_direction_data, self.T, forward_cones, backward_cones, self.num_nodes, save_dir=save_dir)
            all_cell_speeds_and_directions = cu.calculate_all_cell_speeds_and_directions_until_tstep(self.num_cells, t, self.storefile_path, self.T/60.0, cell_Ls)
            datavis.graph_forward_backward_cells_per_timestep(t - 1, all_cell_speeds_and_directions, self.T, forward_cones, backward_cones, save_dir=save_dir)
        
            if self.environment_dir != None:
                data_dict_pickle_path = os.path.join(self.environment_dir, "general_data_dict.pkl")
                if os.path.isfile(data_dict_pickle_path):
                    os.remove(data_dict_pickle_path)
                    
                with open(data_dict_pickle_path, 'wb') as f:
                    dill.dump(data_dict, f)
            
        if produce_animations:
            animation_obj.create_animation_from_data(save_dir, "animation.mp4", timestep_to_draw_till=t)

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
        
    def init_from_store(self, tpoint=None, simulation_execution_enabled=True):
        self.init_random_state(None)
        if tpoint == None:
            tpoint = self.curr_tpoint
        
        self.simulation_execution_enabled = simulation_execution_enabled
        if simulation_execution_enabled:
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
        self.animation_settings = animation_settings
        if self.mode == MODE_EXECUTE and self.simulation_execution_enabled:
            allowed_drift_before_geometry_recalc = self.allowed_drift_before_geometry_recalc
            
            centroid_drifts = np.zeros(self.num_cells, dtype=np.float64)
            recalc_geometry = np.ones(self.num_cells, dtype=np.bool)
            
            simulation_st = time.time()
            num_cells = self.num_cells
            num_nodes = self.num_nodes
            
            environment_cells = self.cells_in_environment
            environment_cells_node_coords = np.array([x.curr_node_coords*x.L for x in environment_cells])
            environment_cells_node_forces = np.array([x.curr_node_forces*x.ML_T2 for x in environment_cells])
            
            curr_centroids = geometry.calculate_centroids(environment_cells_node_coords)
            
            cells_bounding_box_array = geometry.create_initial_bounding_box_polygon_array(num_cells, num_nodes, environment_cells_node_coords)
            cells_node_distance_matrix, cells_line_segment_intersection_matrix = geometry.create_initial_line_segment_intersection_and_dist_squared_matrices(4, self.all_geometry_tasks, num_cells, num_nodes, cells_bounding_box_array, environment_cells_node_coords)
                
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
                for t in self.timepoints[self.curr_tpoint:-1]:
                    if t - self.last_timestep_when_environment_hard_saved >= self.max_timepoints_on_ram:
                        self.last_timestep_when_environment_hard_saved = t
                        
                        if self.environment_dir != None:
                            self.dump_to_store(t)
                        
                    if type(produce_intermediate_visuals) == np.ndarray:
                        if t != 0 and t in produce_intermediate_visuals:
                            self.last_timestep_when_environment_hard_saved = t
                            
                            if self.environment_dir != None:
                                self.dump_to_store(t)
                                
                            data_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
                            if not os.path.exists(data_save_dir):
                                os.makedirs(data_save_dir)
                            
                            print "Doing intermediate analysis..."
                            self.do_data_analysis_and_make_visuals(t, data_save_dir, animation_settings, animation_obj, True, True)                        
                        
                    cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces = self.execute_system_dynamics_in_random_sequence(t, cells_node_distance_matrix, cells_bounding_box_array, cells_line_segment_intersection_matrix, environment_cells_node_coords, environment_cells_node_forces, environment_cells, centroid_drifts, recalc_geometry)
                    
                    prev_centroids = copy.deepcopy(curr_centroids)
                    curr_centroids = geometry.calculate_centroids(environment_cells_node_coords)
                    delta_drifts = geometry.calculate_centroid_dift(prev_centroids, curr_centroids)/1e-6
                    if np.all(recalc_geometry):
                        centroid_drifts = np.zeros_like(centroid_drifts)
                        recalc_geometry = np.zeros_like(recalc_geometry)
                    else:
                        centroid_drifts = centroid_drifts + delta_drifts
                        
                    if np.max(centroid_drifts) > allowed_drift_before_geometry_recalc:
                        recalc_geometry = np.ones_like(recalc_geometry)
#                    centroid_drifts = np.where(recalc_geometry, 0.0, centroid_drifts + delta_drifts)
#                    recalc_geometry = centroid_drifts > allowed_drift_before_geometry_recalc
                    
                    self.curr_tpoint += 1
            else:
                raise StandardError("max_t has already been reached.")
            
            simulation_et = time.time()
            
            if self.environment_dir != None:
                self.dump_to_store(self.curr_tpoint)
                
            if produce_final_visuals == True:
                t = self.num_timepoints
                data_save_dir = os.path.join(self.environment_dir, 'T={}'.format(t))
                
                if not os.path.exists(data_save_dir):
                    os.makedirs(data_save_dir)
                    
                    print "Doing final analysis..."
                    self.do_data_analysis_and_make_visuals(t, data_save_dir, animation_settings, animation_obj, True, True)
                
            simulation_time = np.round(simulation_et - simulation_st, decimals=2)
            
            if self.verbose == True:
                print "Time taken to complete simulation: {}s".format(simulation_time)
                    
            return simulation_time
        else:
            return 0.0

# -----------------------------------------------------------------
    