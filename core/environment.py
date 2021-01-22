import numpy as np
from . import cell
from . import parameterorg
from . import geometry
import numba as nb
import time
import copy

"""
Environment of cells.s
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


def calc_bounding_box_centre(bb):
    x = bb[0] + (bb[1] - bb[0]) * 0.5
    y = bb[2] + (bb[3] - bb[2]) * 0.5

    return np.array([x, y])

# -----------------------------------------------------------------


class Environment:
    """Implementation of coupled map lattice model of a cell.
    """

    def __init__(
        self,
        environment_name="",
        num_timesteps=0,
        cell_group_defns=None,
        environment_dir=None,
        integration_params=None,
        allowed_drift_before_geometry_recalc=1.0,
        max_geometry_recalc_skips=1000,
        cell_placement_method="",
        max_placement_distance_factor=1.0,
        shell_environment=False,
        T=0.0,
    ):

        if integration_params is None:
            integration_params = {}

        self.environment_name = environment_name
        self.environment_dir = environment_dir
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
        self.micrometer = 1e-6

        if not shell_environment:
            self.num_cell_groups = len(self.cell_group_defns)
            self.num_cells = np.sum(
                [
                    cell_group_defn["num_cells"]
                    for cell_group_defn in self.cell_group_defns
                ],
                dtype=np.int64,
            )
        else:
            self.num_cell_groups = 0
            self.num_cells = 0

        self.allowed_drift_before_geometry_recalc = allowed_drift_before_geometry_recalc
        self.max_geometry_recalc_skips = max_geometry_recalc_skips
        if not shell_environment:
            self.cells_in_environment = self.make_cells()
            num_nodes_per_cell = np.array(
                [x.num_nodes for x in self.cells_in_environment], dtype=np.int64
            )
            self.num_nodes = num_nodes_per_cell[0]
            for n in num_nodes_per_cell[1:]:
                if n != self.num_nodes:
                    raise Exception(
                        "There exists a cell with number of nodes different from other cells!"
                    )
        else:
            self.cells_in_environment = None
            self.num_nodes = 0

        if not shell_environment:
            self.full_output_dicts = [[] for _ in self.cells_in_environment]
        else:
            self.full_output_dicts = None

        self.cell_indices = np.arange(self.num_cells)
        self.exec_orders = np.zeros(
            (self.num_timepoints, self.num_cells), dtype=np.int64
        )

        self.all_geometry_tasks = np.array(
            geometry.create_dist_and_line_segment_interesection_test_args(
                self.num_cells, self.num_nodes
            ),
            dtype=np.int64,
        )
        self.geometry_tasks_per_cell = np.array(
            [
                geometry.create_dist_and_line_segment_interesection_test_args_relative_to_specific_cell(
                    ci, self.num_cells, self.num_nodes
                )
                for ci in range(self.num_cells)
            ],
            dtype=np.int64,
        )

        self.mode = MODE_EXECUTE
        self.animation_settings = None

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
        self.exec_orders = np.zeros(
            (self.num_timepoints, self.num_cells), dtype=np.int64
        )
        self.exec_orders[: old_exec_orders.shape[0]] = old_exec_orders

    def simulation_complete(self):
        return self.num_timesteps == self.curr_tpoint

    # -----------------------------------------------------------------

    def make_cells(self):
        cells_in_environment = []
        cell_bounding_boxes_wrt_time = []

        ci_offset = 0
        for cell_group_index, cell_group_defn in enumerate(self.cell_group_defns):
            cells_in_group, init_cell_bounding_boxes = self.create_cell_group(
                self.num_timesteps, cell_group_defn, cell_group_index, ci_offset
            )
            ci_offset += len(cells_in_group)
            cells_in_environment += cells_in_group
            cell_bounding_boxes_wrt_time.append(init_cell_bounding_boxes)

        return np.array(cells_in_environment)

    # -----------------------------------------------------------------

    def create_cell_group(
        self, num_timesteps, cell_group_defn, cell_group_index, ci_offset
    ):
        cell_group_name = cell_group_defn["cell_group_name"]
        num_cells = cell_group_defn["num_cells"]
        cell_group_bounding_box = cell_group_defn["cell_group_bounding_box"]

        cell_parameter_dict = copy.deepcopy(cell_group_defn["parameter_dict"])
        init_cell_radius = cell_parameter_dict["init_cell_radius"]
        num_nodes = cell_parameter_dict["num_nodes"]

        biased_rgtpase_distrib_defns = cell_group_defn["biased_rgtpase_distrib_defns"]
        cells_with_bias_info = list(biased_rgtpase_distrib_defns.keys())

        integration_params = self.integration_params

        init_cell_bounding_boxes = self.calculate_cell_bounding_boxes(
            num_cells,
            init_cell_radius,
            cell_group_bounding_box,
        )

        cells_in_group = []

        for cell_number, bounding_box in enumerate(init_cell_bounding_boxes):
            bias_defn = biased_rgtpase_distrib_defns["default"]

            if cell_number in cells_with_bias_info:
                bias_defn = biased_rgtpase_distrib_defns[cell_number]

            (
                init_node_coords,
                length_edge_resting,
                area_resting,
            ) = self.create_default_init_cell_node_coords(
                bounding_box, init_cell_radius, num_nodes
            )

            cell_parameter_dict.update(
                [
                    ("biased_rgtpase_distrib_defn", bias_defn),
                    ("init_node_coords", init_node_coords),
                    ("length_edge_resting", length_edge_resting),
                    ("area_resting", area_resting),
                ]
            )

            ci = ci_offset + cell_number

            undefined_labels = parameterorg.find_undefined_labels(cell_parameter_dict)
            if len(undefined_labels) > 0:
                raise Exception(
                    "The following labels are not yet defined: {}".format(
                        undefined_labels
                    )
                )

            new_cell = cell.Cell(
                str(cell_group_name) + "_" + str(ci),
                cell_group_index,
                ci,
                integration_params,
                num_timesteps,
                self.T,
                self.num_cells,
                cell_parameter_dict,
                )


            cells_in_group.append(new_cell)

        return cells_in_group, init_cell_bounding_boxes

    # -----------------------------------------------------------------

    @staticmethod
    def calculate_cell_bounding_boxes(
            num_cells,
        init_cell_radius,
        cell_group_bounding_box,
    ):

        cell_bounding_boxes = np.zeros((num_cells, 4), dtype=np.float64)
        xmin, xmax, ymin, ymax = cell_group_bounding_box
        x_length = xmax - xmin
        y_length = ymax - ymin

        cell_diameter = 2 * init_cell_radius

        # check if cells can fit in given bounding box
        total_cell_group_area = num_cells * (np.pi * init_cell_radius ** 2)
        cell_group_bounding_box_area = abs(x_length * y_length)

        if total_cell_group_area > cell_group_bounding_box_area:
            raise Exception(
                "Cell group bounding box is not big enough to contain all cells given init_cell_radius constraint."
            )
        num_cells_along_x = custom_floor(x_length / cell_diameter, 1e-6)
        num_cells_along_y = custom_floor(y_length / cell_diameter, 1e-6)

        cell_x_coords = (
            xmin + np.sign(x_length) * np.arange(num_cells_along_x) * cell_diameter
        )
        cell_y_coords = (
            ymin + np.sign(y_length) * np.arange(num_cells_along_y) * cell_diameter
        )
        x_step = np.sign(x_length) * cell_diameter
        y_step = np.sign(y_length) * cell_diameter

        xi = 0
        yi = 0
        for ci in range(num_cells):
            cell_bounding_boxes[ci] = [
                cell_x_coords[xi],
                cell_x_coords[xi] + x_step,
                cell_y_coords[yi],
                cell_y_coords[yi] + y_step,
            ]

            if yi == (num_cells_along_y - 1):
                yi = 0
                xi += 1
            else:
                yi += 1


        return cell_bounding_boxes

    # -----------------------------------------------------------------

    @staticmethod
    def create_default_init_cell_node_coords(
            bounding_box, init_cell_radius, num_nodes
    ):
        cell_centre = calc_bounding_box_centre(bounding_box)

        cell_node_thetas = np.pi * np.linspace(0, 2, endpoint=False, num=num_nodes)
        cell_node_coords = np.transpose(
            np.array(
                [
                    init_cell_radius * np.cos(cell_node_thetas),
                    init_cell_radius * np.sin(cell_node_thetas),
                ]
            )
        )

        # rotation_theta = np.random.rand()*2*np.pi
        # cell_node_coords = np.array([geometry.rotate_2D_vector_CCW_by_theta(rotation_theta, x) for x in cell_node_coords], dtype=np.float64)
        cell_node_coords = np.array(
            [[x + cell_centre[0], y + cell_centre[1]] for x, y in cell_node_coords],
            dtype=np.float64,
        )

        edge_vectors = geometry.calculate_edge_vectors(cell_node_coords)

        edge_lengths = geometry.calculate_2D_vector_mags(edge_vectors)

        length_edge_resting = np.average(edge_lengths)

        area_resting = geometry.calculate_polygon_area(cell_node_coords)
        if area_resting < 0:
            raise Exception("Resting area was calculated to be negative.")
        return cell_node_coords, length_edge_resting, area_resting

    # -----------------------------------------------------------------
    def execute_system_dynamics_in_random_sequence(
        self,
        t,
        cells_node_distance_matrix,
        cells_bounding_box_array,
        cells_line_segment_intersection_matrix,
        environment_cells_node_coords,
        environment_cells_node_forces,
        environment_cells,
        recalc_geometry,
    ):
        execution_sequence = self.cell_indices
        np.random.shuffle(execution_sequence)

        self.exec_orders[t] = np.copy(execution_sequence)

        for ci in execution_sequence:
            current_cell = environment_cells[ci]

            current_cell.execute_step(
                ci,
                self.num_nodes,
                environment_cells_node_coords,
                environment_cells_node_forces,
                cells_node_distance_matrix[ci],
                cells_line_segment_intersection_matrix[ci],
            )

            if not current_cell.skip_dynamics:
                this_cell_coords = current_cell.curr_node_coords * current_cell.L
                this_cell_forces = current_cell.curr_node_forces * current_cell.ML_T2

                environment_cells_node_coords[ci] = this_cell_coords
                environment_cells_node_forces[ci] = this_cell_forces

                cells_bounding_box_array[
                    ci
                ] = geometry.calculate_polygon_bounding_box(this_cell_coords)
                if recalc_geometry[ci]:
                    # cells_node_distance_matrix, cells_line_segment_intersection_matrix =  geometry.update_line_segment_intersection_and_dist_squared_matrices_old(ci, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_bounding_box_array, cells_node_distance_matrix, cells_line_segment_intersection_matrix)
                    geometry.update_line_segment_intersection_and_dist_squared_matrices(
                        4,
                        self.geometry_tasks_per_cell[ci],
                        environment_cells_node_coords,
                        cells_bounding_box_array,
                        cells_node_distance_matrix,
                        cells_line_segment_intersection_matrix,
                    )
                else:
                    geometry.update_distance_squared_matrix(
                        4,
                        self.geometry_tasks_per_cell[ci],
                        environment_cells_node_coords,
                        cells_node_distance_matrix,
                    )
                    # cells_node_distance_matrix = geometry.update_distance_squared_matrix_old(ci, self.num_cells, self.num_nodes, environment_cells_node_coords, cells_node_distance_matrix)

            # if self.verbose == True:
            #     if self.full_print:
            #         if ci == last_ci:
            #             #print("=" * 40)

        return (
            cells_node_distance_matrix,
            cells_bounding_box_array,
            cells_line_segment_intersection_matrix,
            environment_cells_node_coords,
            environment_cells_node_forces,
        )
    # -----------------------------------------------------------------

    def execute_system_dynamics(
        self,
    ):
        allowed_drift_before_geometry_recalc = (
            self.allowed_drift_before_geometry_recalc
        )

        centroid_drifts = np.zeros(self.num_cells, dtype=np.float64)
        recalc_geometry = np.ones(self.num_cells, dtype=np.bool)

        simulation_st = time.time()
        num_cells = self.num_cells
        num_nodes = self.num_nodes

        environment_cells = self.cells_in_environment
        environment_cells_node_coords = np.array(
            [x.curr_node_coords * x.L for x in environment_cells]
        )
        environment_cells_node_forces = np.array(
            [x.curr_node_forces * x.ML_T2 for x in environment_cells]
        )

        curr_centroids = geometry.calculate_centroids(environment_cells_node_coords)

        cells_bounding_box_array = geometry.create_initial_bounding_box_polygon_array(
            num_cells, environment_cells_node_coords
        )
        # num_cells, num_nodes_per_cell, init_cells_bounding_box_array, init_all_cells_node_coords
        (
            cells_node_distance_matrix,
            cells_line_segment_intersection_matrix,
        ) = geometry.create_initial_line_segment_intersection_and_dist_squared_matrices_old(
            num_cells,
            num_nodes,
            cells_bounding_box_array,
            environment_cells_node_coords,
        )

        # cells_node_distance_matrix = geometry.create_initial_distance_squared_matrix(num_cells, num_nodes, environment_cells_node_coords)

        cell_group_indices = []
        cell_Ls = []
        cell_Ts = []
        cell_etas = []
        cell_skip_dynamics = []

        for a_cell in self.cells_in_environment:
            cell_group_indices.append(a_cell.cell_group_index)
            cell_Ls.append(a_cell.L / 1e-6)
            cell_Ts.append(a_cell.T)
            cell_etas.append(a_cell.eta)
            cell_skip_dynamics.append(a_cell.skip_dynamics)

        if self.curr_tpoint == 0 or self.curr_tpoint < self.num_timesteps:
            for t in self.timepoints[self.curr_tpoint : -1]:

                (
                    cells_node_distance_matrix,
                    cells_bounding_box_array,
                    cells_line_segment_intersection_matrix,
                    environment_cells_node_coords,
                    environment_cells_node_forces,
                ) = self.execute_system_dynamics_in_random_sequence(
                    t,
                    cells_node_distance_matrix,
                    cells_bounding_box_array,
                    cells_line_segment_intersection_matrix,
                    environment_cells_node_coords,
                    environment_cells_node_forces,
                    environment_cells,
                    recalc_geometry,
                )

                prev_centroids = copy.deepcopy(curr_centroids)
                curr_centroids = geometry.calculate_centroids(
                    environment_cells_node_coords
                )
                delta_drifts = (
                    geometry.calculate_centroid_dift(prev_centroids, curr_centroids)
                    / 1e-6
                )
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
            raise Exception("max_t has already been reached.")

        simulation_et = time.time()

        simulation_time = np.round(simulation_et - simulation_st, decimals=2)

        print(
            ("Time taken to complete simulation: {}s".format(simulation_time))
        )



# -----------------------------------------------------------------
