import general.utilities as gu
import numba as nb
import numpy as np

from . import chemistry
from . import dynamics
from . import geometry
from . import mechanics
from . import parameterorg

"""
Cell.

"""


class NaNError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


# =============================================


@nb.jit(nopython=True)
def is_angle_between_range(a, b, angle):
    mod_a, mod_b = a % (2 * np.pi), b % (2 * np.pi)
    mod_angle = angle % (2 * np.pi)

    if mod_b < mod_a:
        if (0 <= mod_angle <= mod_b) or (mod_a <= mod_angle <= 2 * np.pi):
            return 1
        else:
            return 0
    else:
        if mod_a <= mod_angle <= mod_b:
            return 1
        else:
            return 0

# =============================================


def calculate_biased_distrib_factors(
    num_nodes, bias_range, bias_strength, bias_type, node_directions
):
    # index_directions = np.linspace(0, 2*np.pi, num=num_nodes)
    distrib_factors = np.zeros(num_nodes, dtype=np.float64)
    alpha, beta = bias_range

    biased_nodes = np.array(
        [is_angle_between_range(alpha, beta, angle) for angle in node_directions]
    )
    num_biased_nodes = int(np.sum(biased_nodes))
    num_unbiased_nodes = num_nodes - num_biased_nodes

    if bias_type == "random":
        biased_distrib_factors = (
            bias_strength
            * gu.calculate_normalized_randomization_factors(num_biased_nodes)
        )
        unbiased_distrib_factors = (
            1 - bias_strength
        ) * gu.calculate_normalized_randomization_factors(num_nodes - num_biased_nodes)
    elif bias_type == "uniform":
        biased_distrib_factors = (
            bias_strength
            * (1.0 / num_biased_nodes)
            * np.ones(num_biased_nodes, dtype=np.float64)
        )
        unbiased_distrib_factors = (
            (1 - bias_strength)
            * bias_strength
            * (1.0 / num_unbiased_nodes)
            * np.ones(num_unbiased_nodes, dtype=np.float64)
        )
    else:
        raise Exception("Got unknown bias type: {}.".format(bias_type))

    num_nodes_biased = 0
    num_nodes_unbiased = 0

    for ni, biased in enumerate(biased_nodes):
        if biased == 1:
            distrib_factors[ni] = biased_distrib_factors[num_nodes_biased]
            num_nodes_biased += 1
        else:
            distrib_factors[ni] = unbiased_distrib_factors[num_nodes_unbiased]
            num_nodes_unbiased += 1

    assert num_nodes_biased + num_nodes_unbiased == num_nodes

    return distrib_factors


# =============================================

# @nb.jit(nopython=True)
def generate_random_multipliers_fixed_number(num_nodes, threshold, magnitude):
    rfs = np.ones(num_nodes, dtype=np.float64)

    num_random_nodes = int(num_nodes * threshold)
    random_node_indices = np.random.choice(
        np.arange(num_nodes), size=num_random_nodes, replace=False
    )

    rfs[random_node_indices] = magnitude * np.ones(num_random_nodes, dtype=np.float64)

    return rfs


# =============================================


#@nb.jit(nopython=True)
def generate_random_multipliers_random_number(num_nodes, threshold, randoms, magnitude):
    rfs = np.ones(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        if randoms[i] < threshold:
            rfs[i] = magnitude
        else:
            continue

    return rfs


# ---------------------------------------------
def verify_parameter_completeness(parameters_dict):
    if type(parameters_dict) != dict:
        raise Exception(
            "parameters_dict is not a dict, instead: {}".type(parameters_dict)
        )

    given_keys = list(parameters_dict.keys())

    for key in given_keys:
        if key not in parameterorg.all_parameter_labels:
            raise Exception("Parameter {} not in standard parameter list!".format(key))

    for key in parameterorg.all_parameter_labels:
        if key not in given_keys:
            raise Exception("Parameter {} not in given keys!".format(key))

    return


# =============================================


# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
class Cell:
    # -----------------------------------------------------------------
    def __init__(
        self,
        cell_label,
        cell_group_index,
        cell_index,
        integration_params,
        num_timesteps,
        T,
        num_cells_in_environment,
        parameters_dict,
    ):

        self.cell_label = cell_label
        self.cell_group_index = cell_group_index
        self.cell_index = cell_index
        self.integration_params = integration_params

        self.num_timepoints = num_timesteps + 1
        self.num_timesteps = num_timesteps

        self.curr_tpoint = 0
        self.timestep_offset_due_to_dumping = 0

        self.num_nodes = parameters_dict["num_nodes"]
        self.num_cells_in_environment = num_cells_in_environment
        verify_parameter_completeness(parameters_dict)

        self.C_total = parameters_dict["C_total"]
        self.H_total = parameters_dict["H_total"]

        # initializing output arrays
        self.system_history = np.zeros(
            (
                self.num_timepoints + 1,
                self.num_nodes,
                len(parameterorg.output_info_labels),
            )
        )

        self.skip_dynamics = parameters_dict["skip_dynamics"]

        max_velocity_with_dimensions = parameters_dict["max_protrusive_nodal_velocity"]
        eta_with_dimensions = parameters_dict["eta"] / self.num_nodes
        self.L = 1e-6
        self.T = T
        self.ML_T2 = 1e-9

        self.max_protrusive_nodal_velocity = max_velocity_with_dimensions * (
            self.T / self.L
        )
        self.eta = eta_with_dimensions / (self.ML_T2 / (self.L / self.T))

        if parameters_dict["init_node_coords"].shape[0] != self.num_nodes:
            raise Exception(
                "Number of node coords given for initialization does not match number of nodes! Given: {}, required: {}.".format(
                    parameters_dict["init_node_coords"].shape[0], self.num_nodes
                )
            )

        self.curr_node_coords = parameters_dict["init_node_coords"] / self.L
        self.radius_resting = parameters_dict["init_cell_radius"] / self.L
        self.length_edge_resting = parameters_dict["length_edge_resting"] / self.L
        edgeplus_lengths = geometry.calculate_edgeplus_lengths(self.curr_node_coords)
        self.init_average_edge_lengths = np.average(
            geometry.calculate_average_edge_length_around_nodes(edgeplus_lengths)
        )
        self.area_resting = parameters_dict["area_resting"] / (self.L ** 2)

        self.diffusion_const_active = parameters_dict["diffusion_const_active"] * (
            self.T / (self.L ** 2)
        )
        self.diffusion_const_inactive = parameters_dict["diffusion_const_inactive"] * (
            self.T / (self.L ** 2)
        )

        self.stiffness_edge = (parameters_dict["stiffness_edge"]) / (
            self.ML_T2 / self.L
        )
        self.stiffness_cytoplasmic = parameters_dict["stiffness_cytoplasmic"] / (
            self.ML_T2
        )

        # ======================================================

        self.max_force_rac = parameters_dict["max_force_rac"] / self.ML_T2
        self.max_force_rho = parameters_dict["max_force_rho"] / self.ML_T2
        self.threshold_force_rac_activity = parameters_dict[
            "threshold_rac_activity"
        ] / (self.C_total * self.num_nodes)
        self.threshold_force_rho_activity = parameters_dict[
            "threshold_rho_activity"
        ] / (self.H_total * self.num_nodes)

        # ======================================================

        self.closeness_dist_squared_criteria = parameters_dict[
            "closeness_dist_squared_criteria"
        ] / (self.L ** 2)
        self.closeness_dist_squared_criteria_0_until = (
            self.closeness_dist_squared_criteria * 9
        )
        self.closeness_dist_squared_criteria_1_at = self.closeness_dist_squared_criteria
        self.closeness_dist_criteria = np.sqrt(self.closeness_dist_squared_criteria)
        self.force_adh_constant = parameters_dict["force_adh_const"]
            
        self.kgtp_rac_baseline = parameters_dict["kgtp_rac_baseline"] * self.T
        self.kgtp_rac_autoact_baseline = (
            parameters_dict["kgtp_rac_autoact_baseline"] * self.T
        )

        self.threshold_rac_autoact = parameters_dict["threshold_rac_activity"] / (
            self.C_total * self.num_nodes * self.init_average_edge_lengths
        )
        self.exponent_rac_autoact = parameters_dict["hill_exponent"]

        self.kdgtp_rac_baseline = parameters_dict["kdgtp_rac_baseline"] * self.T
        self.kdgtp_rho_mediated_rac_inhib_baseline = (
            parameters_dict["kdgtp_rho_mediated_rac_inhib_baseline"] * self.T
        )

        self.threshold_rho_mediated_rac_inhib = parameters_dict[
            "threshold_rho_activity"
        ] / (self.H_total * self.num_nodes * self.init_average_edge_lengths)
        self.exponent_rho_mediated_rac_inhib = parameters_dict["hill_exponent"]

        self.tension_mediated_rac_inhibition_half_strain = parameters_dict[
            "tension_mediated_rac_inhibition_half_strain"
        ]
        self.tension_mediated_rac_inhibition_magnitude = parameters_dict[
            "tension_mediated_rac_inhibition_magnitude"
        ]
        self.strain_calculation_type = parameters_dict["strain_calculation_type"]

        self.kgdi_rac = parameters_dict["kgdi_rac"] * self.T
        self.kdgdi_rac = parameters_dict["kdgdi_rac"] * self.T * (1.0 / self.num_nodes)

        # ======================================================

        self.kgtp_rho_baseline = parameters_dict["kgtp_rho_baseline"] * self.T
        self.kgtp_rho_autoact_baseline = (
            parameters_dict["kgtp_rho_autoact_baseline"] * self.T
        )

        self.threshold_rho_autoact = parameters_dict["threshold_rho_activity"] / (
            self.H_total * self.num_nodes * self.init_average_edge_lengths
        )
        self.exponent_rho_autoact = parameters_dict["hill_exponent"]

        self.kdgtp_rho_baseline = parameters_dict["kdgtp_rho_baseline"] * self.T
        self.kdgtp_rac_mediated_rho_inhib_baseline = (
            parameters_dict["kdgtp_rac_mediated_rho_inhib_baseline"] * self.T
        )

        self.threshold_rac_mediated_rho_inhib = parameters_dict[
            "threshold_rac_activity"
        ] / (self.C_total * self.num_nodes * self.init_average_edge_lengths)
        self.exponent_rac_mediated_rho_inhib = parameters_dict["hill_exponent"]

        self.kgdi_rho = parameters_dict["kgdi_rho"] * self.T
        self.kdgdi_rho = parameters_dict["kdgdi_rho"] * self.T * (1.0 / self.num_nodes)

        self.interaction_factors_intercellular_contact_per_celltype = parameters_dict[
            "interaction_factors_intercellular_contact_per_celltype"
        ]

        # ==============================================================
        self.interaction_factors_coa_per_celltype = parameters_dict[
            "interaction_factors_coa_per_celltype"
        ]

        self.coa_sensing_dist_at_value = (
            parameters_dict["coa_sensing_dist_at_value"] / self.L
        )

        self.coa_distribution_exponent = (
            np.log(parameters_dict["coa_sensing_value_at_dist"])
            / self.coa_sensing_dist_at_value
        )
        self.coa_intersection_exponent = parameters_dict["coa_intersection_exponent"]

        self.max_coa_signal = parameters_dict["max_coa_signal"]

        # =============================================================

        self.nodal_phase_var_indices = [
            parameterorg.rac_membrane_active_index,
            parameterorg.rac_membrane_inactive_index,
            parameterorg.rho_membrane_active_index,
            parameterorg.rho_membrane_inactive_index,
            parameterorg.x_index,
            parameterorg.y_index,
        ]
        self.num_nodal_phase_vars = len(self.nodal_phase_var_indices)
        self.total_num_nodal_phase_vars = self.num_nodal_phase_vars * self.num_nodes

        self.initialize_nodal_phase_var_indices()

        # =============================================================

        self.randomization_time_mean = int(
            parameters_dict["randomization_time_mean"] * 60.0 / T
        )
        self.randomization_time_variance_factor = parameters_dict[
            "randomization_time_variance_factor"
        ]
        self.next_randomization_event_tpoint = 1200
        self.randomization_magnitude = parameters_dict["randomization_magnitude"]
        self.randomization_rac_kgtp_multipliers = self.renew_randomization_rac_kgtp_multipliers(
                0
            )
        self.randomization_node_percentage = parameters_dict[
            "randomization_node_percentage"
        ]

        # =============================================================

        self.all_cellwide_phase_var_indices = [
            parameterorg.rac_cytosolic_gdi_bound_index,
            parameterorg.rho_cytosolic_gdi_bound_index,
        ]
        self.ode_cellwide_phase_var_indices = []
        self.num_all_cellwide_phase_vars = len(self.all_cellwide_phase_var_indices)
        self.num_ode_cellwide_phase_vars = len(self.ode_cellwide_phase_var_indices)

        self.initialize_all_cellwide_phase_var_indices()
        self.initialize_ode_cellwide_phase_var_indices()

        # =============================================================

        self.nodal_pars_indices = [
            parameterorg.kgtp_rac_index,
            parameterorg.kgtp_rho_index,
            parameterorg.kdgtp_rac_index,
            parameterorg.kdgtp_rho_index,
            parameterorg.kdgdi_rac_index,
            parameterorg.kdgdi_rho_index,
            parameterorg.local_strains_index,
            parameterorg.interaction_factors_intercellular_contact_per_celltype_index,
        ]

        self.initialize_nodal_pars_indices()

        # =============================================================

        self.init_rgtpase_cytosol_frac = parameters_dict["init_rgtpase_cytosol_frac"]
        self.init_rgtpase_membrane_inactive_frac = parameters_dict[
            "init_rgtpase_membrane_inactive_frac"
        ]
        self.init_rgtpase_membrane_active_frac = parameters_dict[
            "init_rgtpase_membrane_active_frac"
        ]
        self.biased_rgtpase_distrib_defn_for_randomization = [
            "unbiased random",
            0.0,
            0.0,
        ]
        self.initialize_cell(
            self.curr_node_coords,
            parameters_dict["biased_rgtpase_distrib_defn"],
            self.init_rgtpase_cytosol_frac,
            self.init_rgtpase_membrane_inactive_frac,
            self.init_rgtpase_membrane_active_frac,
        )

        self.last_trim_timestep = -1

    # -----------------------------------------------------------------
    def insert_state_array_into_system_history(self, state_array, tstep):
        nodal_phase_vars, ode_cellwide_phase_vars = dynamics.unpack_state_array(
            self.num_nodal_phase_vars, state_array
        )
        access_index = self.get_system_history_access_index(tstep)
        self.system_history[
            access_index, :, self.nodal_phase_var_indices
        ] = nodal_phase_vars
        self.system_history[
            access_index, 0, self.ode_cellwide_phase_var_indices
        ] = ode_cellwide_phase_vars

    # -----------------------------------------------------------------
    def initialize_cell(
        self,
        init_node_coords,
        biased_rgtpase_distrib_defn,
        init_rgtpase_cytosol_frac,
        init_rgtpase_membrane_inactive_frac,
        init_rgtpase_membrane_active_frac,
    ):

        init_tstep = 0
        access_index = self.get_system_history_access_index(init_tstep)

        # initializing geometry
        self.system_history[
            access_index, :, [parameterorg.x_index, parameterorg.y_index]
        ] = np.transpose(init_node_coords)
        self.system_history[
            access_index, :, parameterorg.edge_lengths_index
        ] = self.length_edge_resting * np.ones(self.num_nodes)

        node_coords = init_node_coords
        cell_centroid = geometry.calculate_cluster_centroid(init_node_coords)

        self.set_rgtpase_distribution(
            biased_rgtpase_distrib_defn,
            init_rgtpase_cytosol_frac,
            init_rgtpase_membrane_inactive_frac,
            init_rgtpase_membrane_active_frac,
            np.array([np.arctan2(y, x) for x, y in init_node_coords - cell_centroid]),
        )

        rac_membrane_actives = self.system_history[
            access_index, :, parameterorg.rac_membrane_active_index
        ]
        rho_membrane_actives = self.system_history[
            access_index, :, parameterorg.rho_membrane_active_index
        ]

        coa_signals = np.zeros(self.num_nodes, dtype=np.float64)
        self.system_history[
            access_index, :, parameterorg.coa_signal_index
        ] = coa_signals

        intercellular_contact_factors = np.zeros(self.num_nodes)

        np.zeros(
            (self.num_nodes, self.num_cells_in_environment), dtype=np.int64
        )
        np.zeros(
            (self.num_nodes, self.num_cells_in_environment, 2), dtype=np.float64
        )
        np.zeros(
            (self.num_nodes, self.num_cells_in_environment, 2), dtype=np.int64
        )
        np.zeros(
            (self.num_nodes, self.num_cells_in_environment), dtype=np.int64
        )
        close_point_smoothness_factors = np.zeros(
            (self.num_nodes, self.num_cells_in_environment), dtype=np.float64
        )
        np.zeros(
            (self.num_cells_in_environment, 2), dtype=np.float64
        )
        np.zeros(
            (self.num_cells_in_environment, self.num_nodes, 2), dtype=np.float64
        )

        F, EFplus, EFminus, F_rgtpase, F_cytoplasmic, local_strains, unit_inside_pointing_vectors = mechanics.calculate_forces(
            self.num_nodes,
            node_coords,
            rac_membrane_actives,
            rho_membrane_actives,
            self.length_edge_resting,
            self.stiffness_edge,
            self.threshold_force_rac_activity,
            self.threshold_force_rho_activity,
            self.max_force_rac,
            self.max_force_rho,
            self.area_resting,
            self.stiffness_cytoplasmic,
        )

        self.system_history[
            access_index, :, parameterorg.local_strains_index
        ] = local_strains

        self.system_history[
            access_index, :, parameterorg.kdgdi_rac_index
        ] = self.kdgdi_rac * np.ones(self.num_nodes, dtype=np.float64)
        self.system_history[
            access_index, :, parameterorg.kdgdi_rho_index
        ] = self.kdgdi_rho * np.ones(self.num_nodes, dtype=np.float64)

        edgeplus_lengths = geometry.calculate_edgeplus_lengths(node_coords)
        avg_edge_lengths = geometry.calculate_average_edge_length_around_nodes(
            edgeplus_lengths
        )

        conc_rac_membrane_actives = chemistry.calculate_concentrations(
            self.num_nodes, rac_membrane_actives, avg_edge_lengths
        )

        conc_rho_membrane_actives = chemistry.calculate_concentrations(
            self.num_nodes, rho_membrane_actives, avg_edge_lengths
        )

        self.system_history[
            access_index, :, parameterorg.randomization_rac_kgtp_multipliers_index
        ] = self.randomization_rac_kgtp_multipliers

        self.system_history[
            access_index, :, parameterorg.kgtp_rac_index
        ] = chemistry.calculate_kgtp_rac(
            conc_rac_membrane_actives,
            self.exponent_rac_autoact,
            self.threshold_rac_autoact,
            self.kgtp_rac_baseline,
            self.kgtp_rac_autoact_baseline,
            coa_signals,
            self.randomization_rac_kgtp_multipliers,
            intercellular_contact_factors,
            close_point_smoothness_factors,
        )

        self.system_history[
            access_index, :, parameterorg.kgtp_rho_index
        ] = chemistry.calculate_kgtp_rho(
            self.num_nodes,
            conc_rho_membrane_actives,
            intercellular_contact_factors,
            self.exponent_rho_autoact,
            self.threshold_rho_autoact,
            self.kgtp_rho_baseline,
            self.kgtp_rho_autoact_baseline,
        )

        self.system_history[
            access_index, :, parameterorg.kdgtp_rac_index
        ] = chemistry.calculate_kdgtp_rac(
            self.num_nodes,
            conc_rho_membrane_actives,
            self.exponent_rho_mediated_rac_inhib,
            self.threshold_rho_mediated_rac_inhib,
            self.kdgtp_rac_baseline,
            self.kdgtp_rho_mediated_rac_inhib_baseline,
            intercellular_contact_factors,
            self.tension_mediated_rac_inhibition_half_strain,
            self.tension_mediated_rac_inhibition_magnitude,
            np.array([ls if ls > 0 else 0.0 for ls in local_strains]),
        )

        self.system_history[
            access_index, :, parameterorg.kdgtp_rho_index
        ] = chemistry.calculate_kdgtp_rho(
            self.num_nodes,
            conc_rac_membrane_actives,
            self.exponent_rac_mediated_rho_inhib,
            self.threshold_rac_mediated_rho_inhib,
            self.kdgtp_rho_baseline,
            self.kdgtp_rac_mediated_rho_inhib_baseline,
        )

        # update mechanics parameters
        self.system_history[
            access_index, :, [parameterorg.F_x_index, parameterorg.F_y_index]
        ] = np.transpose(F)
        self.system_history[
            access_index, :, [parameterorg.EFplus_x_index, parameterorg.EFplus_y_index]
        ] = np.transpose(EFplus)
        self.system_history[
            access_index,
            :,
            [parameterorg.EFminus_x_index, parameterorg.EFminus_y_index],
        ] = np.transpose(EFminus)
        self.system_history[
            access_index,
            :,
            [parameterorg.F_rgtpase_x_index, parameterorg.F_rgtpase_y_index],
        ] = np.transpose(F_rgtpase)
        self.system_history[
            access_index,
            :,
            [parameterorg.F_cytoplasmic_x_index, parameterorg.F_cytoplasmic_y_index],
        ] = np.transpose(F_cytoplasmic)

        self.system_history[
            access_index,
            :,
            [parameterorg.unit_in_vec_x_index, parameterorg.unit_in_vec_y_index],
        ] = np.transpose(unit_inside_pointing_vectors)

        self.system_history[
            access_index,
            :,
            parameterorg.interaction_factors_intercellular_contact_per_celltype_index,
        ] = intercellular_contact_factors

        self.curr_node_coords = node_coords
        self.curr_node_forces = F  # - F_adhesion

    # -----------------------------------------------------------------
    def initialize_nodal_phase_var_indices(self):
        for index, sys_info_index in enumerate(self.nodal_phase_var_indices):
            label = parameterorg.output_info_labels[sys_info_index]
            setattr(self, "nodal_" + label + "_index", index)

    # -----------------------------------------------------------------
    def initialize_nodal_pars_indices(self):
        for index, sys_info_index in enumerate(self.nodal_pars_indices):
            label = parameterorg.output_info_labels[sys_info_index]
            setattr(self, "nodal_" + label + "_index", index)

    # -----------------------------------------------------------------
    def initialize_ode_cellwide_phase_var_indices(self):
        for index, sys_info_index in enumerate(self.ode_cellwide_phase_var_indices):
            label = parameterorg.output_info_labels[sys_info_index]
            setattr(self, "cellwide_" + label + "_index", index)

    # -----------------------------------------------------------------
    def initialize_all_cellwide_phase_var_indices(self):
        for index, sys_info_index in enumerate(self.all_cellwide_phase_var_indices):
            label = parameterorg.output_info_labels[sys_info_index]
            setattr(self, "cellwide_" + label + "_index", index)

    # -----------------------------------------------------------------
    def calculate_when_randomization_event_occurs(self):
        return self.curr_tpoint + 1200

    # -----------------------------------------------------------------
    def set_rgtpase_distribution(
        self,
        biased_rgtpase_distrib_defn,
        init_rgtpase_cytosol_frac,
        init_rgtpase_membrane_inactive_frac,
        init_rgtpase_membrane_active_frac,
        node_directions,
        tpoint=0,
    ):
        rgtpase_distrib = []
        distrib_type, bias_direction_range, bias_strength = biased_rgtpase_distrib_defn

        cellwide_distrib_factors = np.zeros(self.num_nodes)
        cellwide_distrib_factors[0] = 1

        access_index = self.get_system_history_access_index(tpoint)

        for rgtpase_label in ["rac_", "rho_"]:
            for label in parameterorg.output_chem_labels[:7]:
                if rgtpase_label in label:
                    if "_membrane_" in label:
                        if "_inactive" in label:
                            frac_factor = init_rgtpase_membrane_inactive_frac
                        else:
                            frac_factor = init_rgtpase_membrane_active_frac

                        if distrib_type == "unbiased random":
                            rgtpase_distrib = (
                                frac_factor
                                * gu.calculate_normalized_randomization_factors(
                                    self.num_nodes
                                )
                            )
                        elif distrib_type == "biased random":
                            if rgtpase_label == "rac_":
                                rgtpase_distrib = (
                                    frac_factor
                                    * calculate_biased_distrib_factors(
                                        self.num_nodes,
                                        bias_direction_range,
                                        bias_strength,
                                        "random",
                                        node_directions,
                                    )
                                )
                            elif rgtpase_label == "rho_":
                                rgtpase_distrib = (
                                    frac_factor
                                    * gu.calculate_normalized_randomization_factors(
                                        self.num_nodes
                                    )
                                )
                                # rgtpase_distrib = frac_factor*calculate_biased_distrib_factors(self.num_nodes, bias_direction_range + np.pi, bias_strength, 'random')
                        elif distrib_type == "unbiased uniform":
                            rgtpase_distrib = (
                                frac_factor * np.ones(self.num_nodes) / self.num_nodes
                            )
                        elif distrib_type == "biased uniform":
                            if rgtpase_label == "rac_":
                                rgtpase_distrib = (
                                    frac_factor
                                    * calculate_biased_distrib_factors(
                                        self.num_nodes,
                                        bias_direction_range,
                                        bias_strength,
                                        "uniform",
                                        node_directions,
                                    )
                                )
                            elif rgtpase_label == "rho_":
                                # rgtpase_distrib = frac_factor*gu.calculate_normalized_randomization_factors(self.num_nodes)
                                rgtpase_distrib = (
                                    frac_factor
                                    * calculate_biased_distrib_factors(
                                        self.num_nodes,
                                        np.array(
                                            [
                                                bias_direction_range[0] + np.pi,
                                                bias_direction_range[1] + np.pi,
                                            ]
                                        ),
                                        bias_strength,
                                        "uniform",
                                        node_directions,
                                    )
                                )
                            elif rgtpase_label == "rho_":
                                # rgtpase_distrib = frac_factor*gu.calculate_normalized_randomization_factors(self.num_nodes)
                                rgtpase_distrib = 1e-5 * np.ones(
                                    self.num_nodes, dtype=np.float64
                                )
                                rgtpase_distrib[int(self.num_nodes / 2)] = 1.0
                                if (self.num_nodes % 2) == 1:
                                    rgtpase_distrib[int(self.num_nodes / 2) + 1] = 1.0
                                    if self.num_nodes > 3:
                                        rgtpase_distrib[
                                            int(self.num_nodes / 2) - 1
                                        ] = 1.0
                                rgtpase_distrib = (
                                    frac_factor
                                    * rgtpase_distrib
                                    / np.sum(rgtpase_distrib)
                                )
                        elif distrib_type == "biased nodes":
                            biased_nodes = bias_direction_range
                            distrib_factors = np.array([1.0 if x in biased_nodes else 0.0 for x in range(self.num_nodes)])
                            distrib_factors = distrib_factors / np.sum(distrib_factors)
                            if rgtpase_label == "rac_":
                                rgtpase_distrib = frac_factor * distrib_factors
                            elif rgtpase_label == "rho_":
                                # rgtpase_distrib = frac_factor*gu.calculate_normalized_randomization_factors(self.num_nodes)
                                rgtpase_distrib = frac_factor * np.roll(distrib_factors, 8)
                        else:
                            raise Exception(
                                "Invalid initial rGTPase distribution type provided! ({})".format(
                                    distrib_type
                                )
                            )

                    elif "_cytosolic_" in label:
                        frac_factor = init_rgtpase_cytosol_frac
                        rgtpase_distrib = frac_factor * cellwide_distrib_factors
                    else:
                        continue

                    self.system_history[
                        access_index, :, eval("parameterorg." + label + "_index")
                    ] = rgtpase_distrib

    # -----------------------------------------------------------------

    def renew_randomization_rac_kgtp_multipliers(self, tstep):
        possible_rfs = np.array(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
        )
        ix = int((tstep/1200) % 6)
        return possible_rfs[ix] * self.randomization_magnitude

    # -----------------------------------------------------------------
    def set_next_state(
        self,
        next_state_array,
        this_cell_index,
        num_cells,
        intercellular_squared_dist_array,
        line_segment_intersection_matrix,
        all_cells_node_coords,
            close_point_smoothness_factors,
    ):

        new_tpoint = self.curr_tpoint + 1
        next_tstep_system_history_access_index = self.get_system_history_access_index(
            new_tpoint
        )
        assert next_tstep_system_history_access_index > -1

        num_nodes = self.num_nodes

        assert new_tpoint < self.num_timepoints

        self.insert_state_array_into_system_history(next_state_array, new_tpoint)

        node_coords = np.transpose(
            self.system_history[
                next_tstep_system_history_access_index,
                :,
                [parameterorg.x_index, parameterorg.y_index],
            ]
        )

        edge_displacement_vectors_plus = geometry.calculate_edge_vectors(node_coords)
        edge_lengths = geometry.calculate_2D_vector_mags(edge_displacement_vectors_plus)

        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.edge_lengths_index
        ] = edge_lengths

        geometry.calculate_centroids(all_cells_node_coords)

        intercellular_contact_factors = chemistry.calculate_intercellular_contact_factors(
            this_cell_index,
            num_nodes,
            num_cells,
            self.interaction_factors_intercellular_contact_per_celltype,
            close_point_smoothness_factors,
        )

        # ==================================
        if self.curr_tpoint == self.next_randomization_event_tpoint:
            self.next_randomization_event_tpoint += 1200

            # randomization event has occurred, so renew Rac kgtp rate multipliers
            self.randomization_rac_kgtp_multipliers = self.renew_randomization_rac_kgtp_multipliers(
                self.curr_tpoint
            )

        # store the Rac randomization factors for this timestep
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.randomization_rac_kgtp_multipliers_index,
        ] = self.randomization_rac_kgtp_multipliers

        # ==================================
        rac_membrane_actives = self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.rac_membrane_active_index,
        ]
        rho_membrane_actives = self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.rho_membrane_active_index,
        ]
        random_order_cell_indices = np.arange(num_cells)

        coa_signals = chemistry.calculate_coa_signals(
            this_cell_index,
            num_nodes,
            random_order_cell_indices,
            self.coa_distribution_exponent,
            self.interaction_factors_coa_per_celltype,
            self.max_coa_signal,
            intercellular_squared_dist_array,
            line_segment_intersection_matrix,
            self.coa_intersection_exponent,
        )

        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.coa_signal_index
        ] = coa_signals
        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.cil_signal_index
        ] = intercellular_contact_factors

        rac_cytosolic_gdi_bound = (
            1
            - np.sum(rac_membrane_actives)
            - np.sum(
                self.system_history[
                    next_tstep_system_history_access_index,
                    :,
                    parameterorg.rac_membrane_inactive_index,
                ]
            )
        )
        rho_cytosolic_gdi_bound = (
            1
            - np.sum(rho_membrane_actives)
            - np.sum(
                self.system_history[
                    next_tstep_system_history_access_index,
                    :,
                    parameterorg.rho_membrane_inactive_index,
                ]
            )
        )

        insertion_array = np.zeros(self.num_nodes)
        insertion_array[0] = 1

        self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.rac_cytosolic_gdi_bound_index,
        ] = (rac_cytosolic_gdi_bound * insertion_array)
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.rho_cytosolic_gdi_bound_index,
        ] = (rho_cytosolic_gdi_bound * insertion_array)

        F, EFplus, EFminus, F_rgtpase, F_cytoplasmic, local_strains, unit_inside_pointing_vecs = mechanics.calculate_forces(
            self.num_nodes,
            node_coords,
            rac_membrane_actives,
            rho_membrane_actives,
            self.length_edge_resting,
            self.stiffness_edge,
            self.threshold_force_rac_activity,
            self.threshold_force_rho_activity,
            self.max_force_rac,
            self.max_force_rho,
            self.area_resting,
            self.stiffness_cytoplasmic,
        )

        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.local_strains_index
        ] = local_strains

        # update chemistry parameters
        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.kdgdi_rac_index
        ] = self.kdgdi_rac * np.ones(num_nodes, dtype=np.float64)
        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.kdgdi_rho_index
        ] = self.kdgdi_rho * np.ones(num_nodes, dtype=np.float64)

        edgeplus_lengths = geometry.calculate_edgeplus_lengths(node_coords)
        avg_edge_lengths = geometry.calculate_average_edge_length_around_nodes(
            edgeplus_lengths
        )

        conc_rac_membrane_actives = chemistry.calculate_concentrations(
            self.num_nodes, rac_membrane_actives, avg_edge_lengths
        )

        conc_rho_membrane_actives = chemistry.calculate_concentrations(
            self.num_nodes, rho_membrane_actives, avg_edge_lengths
        )

        self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.randomization_rac_kgtp_multipliers_index,
        ] = self.randomization_rac_kgtp_multipliers

        kgtp_rac_per_node = chemistry.calculate_kgtp_rac(
            conc_rac_membrane_actives,
            self.exponent_rac_autoact,
            self.threshold_rac_autoact,
            self.kgtp_rac_baseline,
            self.kgtp_rac_autoact_baseline,
            coa_signals,
            self.randomization_rac_kgtp_multipliers,
            intercellular_contact_factors,
            close_point_smoothness_factors,
        )

        kdgtps_rac = chemistry.calculate_kdgtp_rac(
            self.num_nodes,
            conc_rho_membrane_actives,
            self.exponent_rho_mediated_rac_inhib,
            self.threshold_rho_mediated_rac_inhib,
            self.kdgtp_rac_baseline,
            self.kdgtp_rho_mediated_rac_inhib_baseline,
            intercellular_contact_factors,
            self.tension_mediated_rac_inhibition_half_strain,
            self.tension_mediated_rac_inhibition_magnitude,
            np.array([ls if ls > 0 else 0.0 for ls in local_strains]),
        )

        kdgtps_rho = chemistry.calculate_kdgtp_rho(
            self.num_nodes,
            conc_rac_membrane_actives,
            self.exponent_rac_mediated_rho_inhib,
            self.threshold_rac_mediated_rho_inhib,
            self.kdgtp_rho_baseline,
            self.kdgtp_rac_mediated_rho_inhib_baseline,
        )

        kgtp_rho_per_node = chemistry.calculate_kgtp_rho(
            self.num_nodes,
            conc_rho_membrane_actives,
            intercellular_contact_factors,
            self.exponent_rho_autoact,
            self.threshold_rho_autoact,
            self.kgtp_rho_baseline,
            self.kgtp_rho_autoact_baseline,
        )

        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.kgtp_rac_index
        ] = kgtp_rac_per_node
        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.kgtp_rho_index
        ] = kgtp_rho_per_node

        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.kdgtp_rac_index
        ] = kdgtps_rac

        self.system_history[
            next_tstep_system_history_access_index, :, parameterorg.kdgtp_rho_index
        ] = kdgtps_rho

        # update mechanics parameters
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            [parameterorg.F_x_index, parameterorg.F_y_index],
        ] = np.transpose(F)
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            [parameterorg.EFplus_x_index, parameterorg.EFplus_y_index],
        ] = np.transpose(EFplus)
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            [parameterorg.EFminus_x_index, parameterorg.EFminus_y_index],
        ] = np.transpose(EFminus)
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            [parameterorg.F_rgtpase_x_index, parameterorg.F_rgtpase_y_index],
        ] = np.transpose(F_rgtpase)
        self.system_history[
            next_tstep_system_history_access_index,
            :,
            [parameterorg.F_cytoplasmic_x_index, parameterorg.F_cytoplasmic_y_index],
        ] = np.transpose(F_cytoplasmic)

        self.system_history[
            next_tstep_system_history_access_index,
            :,
            [parameterorg.unit_in_vec_x_index, parameterorg.unit_in_vec_y_index],
        ] = np.transpose(unit_inside_pointing_vecs)

        self.system_history[
            next_tstep_system_history_access_index,
            :,
            parameterorg.interaction_factors_intercellular_contact_per_celltype_index,
        ] = intercellular_contact_factors

        self.curr_tpoint = new_tpoint
        self.curr_node_coords = node_coords
        self.curr_node_forces = F  # - F_adhesion

    # -----------------------------------------------------------------
    def pack_rhs_arguments(
        self,
        t,
        this_cell_index,
        all_cells_node_coords,
        close_point_smoothness_factors,
    ):
        access_index = self.get_system_history_access_index(t)
        num_cells = all_cells_node_coords.shape[0]
        num_nodes = self.num_nodes
        all_cells_centres = geometry.calculate_centroids(all_cells_node_coords)

        intercellular_contact_factors = chemistry.calculate_intercellular_contact_factors(
            this_cell_index,
            num_nodes,
            num_cells,
            self.interaction_factors_intercellular_contact_per_celltype,
            close_point_smoothness_factors,
        )

        transduced_coa_signals = self.system_history[
            access_index, :, parameterorg.coa_signal_index
        ]

        return (
            this_cell_index,
            self.num_nodes,
            self.num_nodal_phase_vars,
            self.nodal_rac_membrane_active_index,
            self.length_edge_resting,
            self.nodal_rac_membrane_inactive_index,
            self.nodal_rho_membrane_active_index,
            self.nodal_rho_membrane_inactive_index,
            self.nodal_x_index,
            self.nodal_y_index,
            self.kgtp_rac_baseline,
            self.kdgtp_rac_baseline,
            self.kgtp_rho_baseline,
            self.kdgtp_rho_baseline,
            self.kgtp_rac_autoact_baseline,
            self.kgtp_rho_autoact_baseline,
            self.kdgtp_rho_mediated_rac_inhib_baseline,
            self.kdgtp_rac_mediated_rho_inhib_baseline,
            self.kgdi_rac,
            self.kdgdi_rac,
            self.kgdi_rho,
            self.kdgdi_rho,
            self.threshold_rac_autoact,
            self.threshold_rho_autoact,
            self.threshold_rho_mediated_rac_inhib,
            self.threshold_rac_mediated_rho_inhib,
            self.exponent_rac_autoact,
            self.exponent_rho_autoact,
            self.exponent_rho_mediated_rac_inhib,
            self.exponent_rac_mediated_rho_inhib,
            self.diffusion_const_active,
            self.diffusion_const_inactive,
            self.eta,
            self.stiffness_edge,
            self.threshold_force_rac_activity,
            self.threshold_force_rho_activity,
            self.max_force_rac,
            self.max_force_rho,
            self.area_resting,
            self.stiffness_cytoplasmic,
            transduced_coa_signals,
            close_point_smoothness_factors,
            intercellular_contact_factors,
            self.tension_mediated_rac_inhibition_half_strain,
            self.tension_mediated_rac_inhibition_magnitude,
            self.randomization_rac_kgtp_multipliers,
        )

    # -----------------------------------------------------------------
    def trim_system_history(self, environment_tpoint):

        assert environment_tpoint == self.curr_tpoint

        trimmed_system_history = np.empty_like(self.system_history)
        trimmed_system_history[0] = self.system_history[
            self.get_system_history_access_index(self.curr_tpoint)
        ]
        self.system_history = trimmed_system_history

        self.last_trim_timestep = self.curr_tpoint

        self.timestep_offset_due_to_dumping = self.curr_tpoint

    # -----------------------------------------------------------------

    def get_system_history_access_index(self, tpoint):
        return tpoint - self.timestep_offset_due_to_dumping

    # -----------------------------------------------------------------
    def execute_step(
        self,
        this_cell_index,
        num_nodes,
        all_cells_node_coords,
        all_cells_node_forces,
        intercellular_squared_dist_array,
        line_segment_intersection_matrix,
    ):
        dynamics.print_var = True

        if not self.skip_dynamics:
            L_squared = self.L ** 2
            intercellular_squared_dist_array = (
                intercellular_squared_dist_array / L_squared
            )
            all_cells_node_coords = all_cells_node_coords / self.L
            all_cells_node_forces = all_cells_node_forces / self.ML_T2

            num_cells = all_cells_node_coords.shape[0]

            are_nodes_inside_other_cells = geometry.check_if_nodes_inside_other_cells(
                this_cell_index, num_nodes, num_cells, all_cells_node_coords
            )

            close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, close_point_smoothness_factors = geometry.do_close_points_to_each_node_on_other_cells_exist(
                num_cells,
                num_nodes,
                this_cell_index,
                all_cells_node_coords[this_cell_index],
                intercellular_squared_dist_array,
                self.closeness_dist_squared_criteria_0_until,
                self.closeness_dist_squared_criteria_1_at,
                all_cells_node_coords,
                are_nodes_inside_other_cells,
            )

            state_array = dynamics.pack_state_array_from_system_history(
                self.nodal_phase_var_indices,
                self.ode_cellwide_phase_var_indices,
                self.system_history,
                self.get_system_history_access_index(self.curr_tpoint),
            )

            rhs_args = self.pack_rhs_arguments(
                self.curr_tpoint,
                this_cell_index,
                all_cells_node_coords,
                close_point_smoothness_factors,
            )

            output_array = dynamics.eulerint(dynamics.cell_dynamics, state_array, np.array([0, 1]), rhs_args, 10)

            next_state_array = output_array[1]

            self.set_next_state(
                next_state_array,
                this_cell_index,
                num_cells,
                intercellular_squared_dist_array,
                line_segment_intersection_matrix,
                all_cells_node_coords,
                close_point_smoothness_factors,
            )

        else:
            self.system_history[
                self.get_system_history_access_index(self.curr_tpoint + 1)
            ] = self.system_history[
                self.get_system_history_access_index(self.curr_tpoint)
            ]
            self.curr_tpoint += 1


# ===============================================================
