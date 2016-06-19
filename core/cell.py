from __future__ import division
import numpy as np
import scipy.integrate as scint
import parameterorg

import numba as nb
import geometry
import chemistry
import mechanics
import dynamics
import general.utilities as general_utils
import analysis.utilities as analysis_utils
import core.hardio as hardio

"""
Cell.

"""

#mech_labels = ['x', 'y', 'edge_lengths', 'F_x', 'F_y', 'EFplus_x', 'EFplus_y', 'EFminus_x', 'EFminus_y', 'F_rgtpase_x', 'F_rgtpase_y', 'F_cytoplasmic_x', 'F_cytoplasmic_y', 'local_strains', 'intercellular_contact_factor_magnitudes', 'migr_bdry_contact', 'unit_in_vec_x', 'unit_in_vec_y']
#
#chem_labels = ['rac_membrane_active', 'rac_membrane_inactive', 'rac_cytosolic_gdi_bound', 'rho_membrane_active', 'rho_membrane_inactive', 'rho_cytosolic_gdi_bound', 'coa_signal', 'kdgdi_rac', 'kdgdi_rho', 'kgtp_rac', 'kgtp_rho', 'kdgtp_rac', 'kdgtp_rho', 'migr_bdry_contact_factor_mag', 'randomization_event_occurred', 'external_gradient_on_nodes']

class NaNError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
        
    
# =============================================

@nb.jit(nopython=True)
def is_angle_between_range(alpha, beta, angle):
    range1 = (alpha - beta)%(2*np.pi)
    range2 = (beta - alpha)%(2*np.pi)
    
    min_range = 0
    max_range = 2*np.pi
    range_size = 2*np.pi
    
    if range1 < range2:
        min_range = beta
        max_range = alpha
        range_size =  range1
    else:
        min_range = alpha
        max_range = beta
        range_size = range2
        
    range3 = (angle - min_range)%(2*np.pi) + (max_range - angle)%(2*np.pi)
    
    if range3 <= range_size:
        return 1
    else:
        return 0

# =============================================
   
def calculate_biased_distrib_factors(bias_range, bias_strength, size, bias_type):
    index_directions = np.linspace(0, 2*np.pi, num=size)
    distrib_factors = np.zeros(size, dtype=np.float64)
    alpha, beta = bias_range
    
    biased_nodes = np.array([is_angle_between_range(alpha, beta, index_dir) for index_dir in index_directions])
    num_biased_nodes = np.sum(biased_nodes)
    num_unbiased_nodes = size - num_biased_nodes
    
    if bias_type == 'random':
        biased_distrib_factors = bias_strength*general_utils.calculate_normalized_randomization_factors(num_biased_nodes)
        unbiased_distrib_factors = (1 - bias_strength)*general_utils.calculate_normalized_randomization_factors(size - num_biased_nodes)
    elif bias_type == 'uniform':
        biased_distrib_factors = bias_strength*(1.0/num_biased_nodes)*np.ones(num_biased_nodes, dtype=np.float64)
        unbiased_distrib_factors = (1 - bias_strength)*bias_strength*(1.0/num_unbiased_nodes)*np.ones(num_unbiased_nodes, dtype=np.float64)
    else:
        raise StandardError("Got unknown bias type: {}.".format(bias_type))
        
    num_nodes_biased = 0
    num_nodes_unbiased = 0
    
    for ni, biased in enumerate(biased_nodes):
        if biased == 1:
            distrib_factors[ni] = biased_distrib_factors[num_nodes_biased]
            num_nodes_biased += 1
        else:
            distrib_factors[ni] = unbiased_distrib_factors[num_nodes_unbiased]
            num_nodes_unbiased += 1
            
    return distrib_factors
    
# =============================================

class Cell():
    """Implementation of coupled vertex model of a cell.
    """
# -----------------------------------------------------------------
    def __init__(self, cell_label, cell_group_index, cell_index,  
                 integration_params, 
                 num_timesteps, T, C_total, H_total, num_cells_in_environment, 
                 init_node_coords,
                 max_timepoints_on_ram,
                 radius_resting=None,
                 length_edge_resting=None,
                 area_resting=None,
                 biased_rgtpase_distrib_defn=["uniform", 0, 0],
                 init_rgtpase_cytosol_gdi_bound_frac=None,
                 init_rgtpase_membrane_inactive_frac=None,
                 init_rgtpase_membrane_active_frac=None,
                 kgtp_rac_baseline=None, kdgtp_rac_baseline=None,
                 kgtp_rho_baseline=None, kdgtp_rho_baseline=None,
                 kgtp_rac_autoact_baseline=None, 
                 kgtp_rho_autoact_baseline=None, 
                 kdgtp_rho_mediated_rac_inhib_baseline=None,
                 kdgtp_rac_mediated_rho_inhib_baseline=None, 
                 kgdi_rac=None, kdgdi_rac=None, 
                 kgdi_rho=None, kdgdi_rho=None,
                 threshold_rac_autoact=None,
                 threshold_rho_autoact=None,
                 threshold_rho_mediated_rac_inhib=None,
                 threshold_rac_mediated_rho_inhib=None,
                 exponent_rac_autoact=None,
                 exponent_rho_autoact=None,
                 exponent_rho_mediated_rac_inhib=None,
                 exponent_rac_mediated_rho_inhib=None,
                 diffusion_const_active=None,
                 diffusion_const_inactive=None,
                 space_physical_bdry_polygon=None,
                 space_migratory_bdry_polygon=None,
                 stiffness_edge=None,
                 sigma_rac=None,
                 sigma_rho_multiplier=None,
                 force_rac_exp=None,
                 force_rho_exp=None,
                 force_rac_threshold=None,
                 force_rho_threshold=None,
                 force_adh_constant=None,
                 eta=None, 
                 factor_migr_bdry_contact=None,
                 space_at_node_factor_rac=None,
                 space_at_node_factor_rho=None,
                 stiffness_cytoplasmic=None,
                 migr_bdry_contact_factor_mag=None,
                 intercellular_contact_factor_magnitudes=None,
                 closeness_dist_squared_criteria=None,
                 cell_dependent_coa_signal_strengths=None, halfmax_coa_sensing_dist=None, coa_sensitivity_percent_drop_over_cell_diameter=None, coa_belt_offset=None, randomization=None, randomization_scheme=None, randomization_time_mean=None, randomization_time_variance_factor=None, randomization_magnitude=None, skip_dynamics=None, randomization_rgtpase_distrib_strength=None, tension_mediated_rac_inhibition_half_strain=None, tension_fn_type=None, tension_mediated_rac_hill_exponent=None, verbose=False):
        """Constructor for Cell object.
        """
        
        self.verbose = verbose
        
        self.cell_label = cell_label
        self.cell_group_index = cell_group_index
        self.cell_index = cell_index
        self.integration_params = integration_params
        
        self.num_timepoints = num_timesteps + 1
        self.num_timesteps = num_timesteps
        
        self.curr_tpoint = 0
        self.timestep_offset_due_to_dumping = 0
        
        self.C_total = C_total
        self.H_total = H_total
        
        #self.C_total = 0.5*C_total/self.num_nodes
        #self.H_total = 0.5*H_total/self.num_nodes
        
        self.num_nodes = init_node_coords.shape[0]
        self.num_cells_in_environment = num_cells_in_environment
        
        parameterorg.info_labels = parameterorg.mech_labels + parameterorg.chem_labels
        self.max_timepoints_on_ram = max_timepoints_on_ram
        # initializing system information holders
        self.system_info = np.zeros((self.max_timepoints_on_ram + 1, self.num_nodes,  len(parameterorg.info_labels)))
        
        # L is node dependent; should we make it node independent?
        # self.L = length_edge_resting -> self.L = L
        # No, I do not think there is a need to do so because

        self.L = 0.1*(sigma_rac*length_edge_resting/eta)*T
        self.T = T
        self.ML_T2 = sigma_rac*length_edge_resting#stiffness_edge
      
#        print "*********************************"
#        print "cell_index: {}".format(cell_index)
#        print "L: {}".format(self.L)
#        print "T: {}".format(self.T)
#        print "ML_T2: {}".format(self.ML_T2)
#        print "*********************************"
      
        self.skip_dynamics = skip_dynamics
        
        # ======================================================
        
        self.curr_node_coords = init_node_coords/self.L
        self.radius_resting = radius_resting/self.L
        self.length_edge_resting = length_edge_resting/self.L
        self.area_resting = area_resting/(self.L**2)
        self.diffusion_const_active = diffusion_const_active*(self.T/(self.L**2))
        self.diffusion_const_inactive = diffusion_const_inactive*(self.T/(self.L**2))
        self.stiffness_edge = stiffness_edge/(self.ML_T2)
        self.stiffness_cytoplasmic = (1e10*self.ML_T2)*self.stiffness_edge#stiffness_cytoplasmic/(self.ML_T2)
        
        # ======================================================
        
        self.force_rac_max_mag = (sigma_rac*length_edge_resting)/(self.ML_T2)
        self.force_rac_exp = force_rac_exp
        self.force_rac_threshold = force_rac_threshold/(self.C_total*self.num_nodes)
        
        self.force_rho_max_mag = self.force_rac_max_mag*sigma_rho_multiplier
        self.force_rho_exp = force_rho_exp
        self.force_rho_threshold = force_rho_threshold/(self.H_total*self.num_nodes)

        self.closeness_dist_squared_criteria = closeness_dist_squared_criteria/(self.L**2)
        self.closeness_dist_criteria = np.sqrt(self.closeness_dist_squared_criteria)        
        self.force_adh_constant = force_adh_constant
        self.eta = (eta/self.num_nodes)/(self.ML_T2/(self.L/self.T))
                
        # ======================================================
        
        self.kgtp_rac_baseline = kgtp_rac_baseline*self.T
        self.kgtp_rac_autoact_baseline = kgtp_rac_autoact_baseline*self.T
        
        self.threshold_rac_autoact = threshold_rac_autoact/(self.C_total*self.num_nodes)
        self.exponent_rac_autoact = exponent_rac_autoact
        
        self.kdgtp_rac_baseline = kdgtp_rac_baseline*self.T
        self.kdgtp_rho_mediated_rac_inhib_baseline = kdgtp_rho_mediated_rac_inhib_baseline*self.T
        
        self.threshold_rho_mediated_rac_inhib = threshold_rho_mediated_rac_inhib/(self.H_total*self.num_nodes)
        self.exponent_rho_mediated_rac_inhib = exponent_rho_mediated_rac_inhib
        
        self.tension_mediated_rac_inhibition_exponent = np.log(0.5)/(-1*tension_mediated_rac_inhibition_half_strain)
        self.tension_mediated_rac_inhibition_multiplier = ((1/0.5) - 1)/(tension_mediated_rac_inhibition_half_strain**1)
        self.tension_mediated_rac_hill_exponent = tension_mediated_rac_hill_exponent
        self.tension_mediated_rac_inhibition_half_strain = tension_mediated_rac_inhibition_half_strain
        self.tension_fn_type = tension_fn_type
        self.kgdi_rac = kgdi_rac*self.T
        self.kdgdi_rac = kdgdi_rac*self.T*(1.0/self.num_nodes)
        
        # ======================================================
        
        self.kgtp_rho_baseline = kgtp_rho_baseline*self.T
        self.kgtp_rho_autoact_baseline = kgtp_rho_autoact_baseline*self.T
        
        self.threshold_rho_autoact = threshold_rho_autoact/(self.H_total*self.num_nodes)
        self.exponent_rho_autoact = exponent_rho_autoact
        
        self.kdgtp_rho_baseline = kdgtp_rho_baseline*self.T
        self.kdgtp_rac_mediated_rho_inhib_baseline = kdgtp_rac_mediated_rho_inhib_baseline*self.T
        
        self.threshold_rac_mediated_rho_inhib = threshold_rac_mediated_rho_inhib/(self.C_total*self.num_nodes)
        self.exponent_rac_mediated_rho_inhib = exponent_rac_mediated_rho_inhib
        
        self.kgdi_rho = kgdi_rho*self.T
        self.kdgdi_rho = kdgdi_rho*self.T*(1.0/self.num_nodes)
        
        # ======================================================
        
        self.exists_space_physical_bdry_polygon = 0
        self.space_physical_bdry_polygon = np.ones((4, 2), dtype=np.float64)
        if space_physical_bdry_polygon.shape[0] > 0:
            self.exists_space_physical_bdry_polygon = 1
            self.space_physical_bdry_polygon = space_physical_bdry_polygon/self.L
            
        self.space_migratory_bdry_polygon = space_migratory_bdry_polygon/self.L
        
        self.factor_migr_bdry_contact = None
        
        self.space_at_node_factor_rac = space_at_node_factor_rac/self.num_nodes
        self.space_at_node_factor_rho = space_at_node_factor_rac/self.num_nodes
        
        self.migr_bdry_contact_factor_mag = migr_bdry_contact_factor_mag
        self.intercellular_contact_factor_magnitudes = intercellular_contact_factor_magnitudes
        
        #==============================================================
        self.cell_dependent_coa_signal_strengths = cell_dependent_coa_signal_strengths
        
        self.halfmax_coa_sensing_dist = halfmax_coa_sensing_dist/self.L
        
        self.coa_distribution_exponent = np.log(0.5)/self.halfmax_coa_sensing_dist
        if coa_sensitivity_percent_drop_over_cell_diameter <= 0.0:
            self.coa_sensitivity_exponent = 1
        else:
            self.coa_sensitivity_exponent = chemistry.calculate_coa_sensitivity_exponent(self.coa_distribution_exponent, coa_sensitivity_percent_drop_over_cell_diameter, 2*self.radius_resting)
            
        self.coa_belt_offset = coa_belt_offset/self.L
        # =============================================================
        
        self.nodal_phase_var_indices = [parameterorg.rac_membrane_active_index, parameterorg.rac_membrane_inactive_index, parameterorg.rho_membrane_active_index, parameterorg.rho_membrane_inactive_index, parameterorg.x_index, parameterorg.y_index]
        self.num_nodal_phase_vars = len(self.nodal_phase_var_indices)
        self.total_num_nodal_phase_vars = self.num_nodal_phase_vars*self.num_nodes
        
        self.initialize_nodal_phase_var_indices()
        
        # =============================================================
        
        self.randomization = randomization
        
        self.randomization_time_mean = int(randomization_time_mean*60.0/T)
        self.randomization_time_variance_factor = randomization_time_variance_factor
        self.next_randomization_event_tstep = None
        self.randomization_magnitude = randomization_magnitude
        self.randomization_rac_kgtp_multipliers = np.empty(self.num_nodes, dtype=np.float64)
        
            
        if randomization_scheme == "wipeout":
            self.randomization_scheme = 0
        elif randomization_scheme == "kgtp_rac_multipliers":
            self.randomization_scheme = 1
            self.renew_randomization_rac_kgtp_multipliers()
        else:
            raise StandardError("Unknown randomization scheme given: {}.".format(randomization_scheme))
        
        # =============================================================
        
        self.all_cellwide_phase_var_indices = [parameterorg.rac_cytosolic_gdi_bound_index, parameterorg.rho_cytosolic_gdi_bound_index]
        self.ode_cellwide_phase_var_indices = []
        self.num_all_cellwide_phase_vars = len(self.all_cellwide_phase_var_indices)
        self.num_ode_cellwide_phase_vars = len(self.ode_cellwide_phase_var_indices)
        
        self.initialize_all_cellwide_phase_var_indices()
        self.initialize_ode_cellwide_phase_var_indices()
        
        # =============================================================
        
        self.nodal_pars_indices = [parameterorg.kgtp_rac_index, parameterorg.kgtp_rho_index, parameterorg.kdgtp_rac_index, parameterorg.kdgtp_rho_index, parameterorg.kdgdi_rac_index, parameterorg.kdgdi_rho_index, parameterorg.local_strains_index, parameterorg.intercellular_contact_factor_magnitudes_index, parameterorg.migr_bdry_contact_index]
        
        self.initialize_nodal_pars_indices()
        
        # =============================================================
        
        self.init_rgtpase_cytosol_gdi_bound_frac = init_rgtpase_cytosol_gdi_bound_frac
        self.init_rgtpase_membrane_inactive_frac = init_rgtpase_membrane_inactive_frac
        self.init_rgtpase_membrane_active_frac = init_rgtpase_membrane_active_frac
        self.biased_rgtpase_distrib_defn_for_randomization = ['unbiased random', biased_rgtpase_distrib_defn[1], biased_rgtpase_distrib_defn[2]]
        self.initialize_cell(init_node_coords/self.L, biased_rgtpase_distrib_defn, init_rgtpase_cytosol_gdi_bound_frac, init_rgtpase_membrane_inactive_frac, init_rgtpase_membrane_active_frac)
        
        self.last_trim_timestep = -1
        
        
        
# -----------------------------------------------------------------
    def insert_state_array_into_system_info(self, state_array, tstep):
        nodal_phase_vars, ode_cellwide_phase_vars = dynamics.unpack_state_array(self.num_nodal_phase_vars, self.num_nodes, state_array)
        access_index = self.get_system_info_access_index(tstep)
        self.system_info[access_index, :, self.nodal_phase_var_indices] = nodal_phase_vars
        self.system_info[access_index, 0, self.ode_cellwide_phase_var_indices] = ode_cellwide_phase_vars
        
# -----------------------------------------------------------------
    def initialize_cell(self, init_node_coords, biased_rgtpase_distrib_defn, init_rgtpase_cytosol_gdi_bound_frac, init_rgtpase_membrane_inactive_frac, init_rgtpase_membrane_active_frac):
        
        init_tstep = 0
        access_index = self.get_system_info_access_index(init_tstep)
        
        # initializing geometry
        self.system_info[access_index, :, [parameterorg.x_index, parameterorg.y_index]] = np.transpose(init_node_coords)
        self.system_info[access_index, :, parameterorg.edge_lengths_index] = self.length_edge_resting*np.ones(self.num_nodes)
        
        node_coords = init_node_coords
                
        self.set_rgtpase_distribution(biased_rgtpase_distrib_defn, init_rgtpase_cytosol_gdi_bound_frac, init_rgtpase_membrane_inactive_frac, init_rgtpase_membrane_active_frac)
        
        rac_membrane_actives = self.system_info[access_index, :, parameterorg.rac_membrane_active_index]
        rho_membrane_actives = self.system_info[access_index, :, parameterorg.rho_membrane_active_index]
        
        transduced_coa_signals = np.zeros(self.num_nodes, dtype=np.float64)
        self.system_info[access_index, :, parameterorg.coa_signal_index] = transduced_coa_signals
        external_gradient_on_nodes = np.zeros(self.num_nodes, dtype=np.float64)
        self.system_info[access_index, :, parameterorg.external_gradient_on_nodes_index] = np.zeros(self.num_nodes, dtype=np.float64)
        
        intercellular_contact_factors = np.ones(self.num_nodes)
        migr_bdry_contact_factors = np.ones(self.num_nodes)
        
        # num_nodes, num_cells, this_ci, this_cell_coords, rac_membrane_actives, rho_membrane_actives, length_edge_resting, stiffness_edge, force_rac_exp, force_rac_threshold, force_rac_max_mag, force_rho_exp, force_rho_threshold, force_rho_max_mag, force_adh_constant, area_resting, stiffness_cytoplasmic, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria
    
        close_point_on_other_cells_to_each_node_exists = np.zeros((self.num_nodes, self.num_cells_in_environment), dtype=np.int64)
        close_point_on_other_cells_to_each_node = np.zeros((self.num_nodes, self.num_cells_in_environment, 2), dtype=np.float64)
        close_point_on_other_cells_to_each_node_indices = np.zeros((self.num_nodes, self.num_cells_in_environment, 2), dtype=np.int64)
        close_point_on_other_cells_to_each_node_projection_factors = np.zeros((self.num_nodes, self.num_cells_in_environment), dtype=np.int64)
        all_cells_centres = np.zeros((self.num_cells_in_environment, 2), dtype=np.float64)
        all_cells_node_forces = np.zeros((self.num_cells_in_environment, self.num_nodes, 2), dtype=np.float64)
        
        F, EFplus, EFminus, F_rgtpase, F_cytoplasmic, F_adhesion, local_strains, unit_inside_pointing_vecs = mechanics.calculate_forces(self.num_nodes, self.num_cells_in_environment, self.cell_index, node_coords, rac_membrane_actives, rho_membrane_actives, self.length_edge_resting, self.stiffness_edge, self.force_rac_exp, self.force_rac_threshold, self.force_rac_max_mag, self.force_rho_exp, self.force_rho_threshold, self.force_rho_max_mag, self.force_adh_constant, self.area_resting, self.stiffness_cytoplasmic, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, self.closeness_dist_criteria)
        
        self.system_info[access_index, :, parameterorg.local_strains_index] = local_strains
        
        local_tension_strains = np.where(local_strains < 0, 0, local_strains)
        
        # update chemistry parameters
        self.system_info[access_index, :, parameterorg.kdgdi_rac_index] = self.kdgdi_rac*np.ones(self.num_nodes, dtype=np.float64)
        self.system_info[access_index, :, parameterorg.kdgdi_rho_index] = self.kdgdi_rho*np.ones(self.num_nodes, dtype=np.float64)
        
        self.system_info[access_index, :, parameterorg.kgtp_rac_index] = chemistry.calculate_kgtp_rac(self.num_nodes, rac_membrane_actives, migr_bdry_contact_factors, self.exponent_rac_autoact, self.threshold_rac_autoact, self.kgtp_rac_baseline, self.kgtp_rac_autoact_baseline, transduced_coa_signals, external_gradient_on_nodes, self.randomization_rac_kgtp_multipliers)
        
        self.system_info[access_index, :, parameterorg.kgtp_rho_index] = chemistry.calculate_kgtp_rho(self.num_nodes, rho_membrane_actives, intercellular_contact_factors, migr_bdry_contact_factors, self.exponent_rho_autoact, self.threshold_rho_autoact, self.kgtp_rho_baseline, self.kgtp_rho_autoact_baseline)
        
        self.system_info[access_index, :, parameterorg.kdgtp_rac_index] = chemistry.calculate_kdgtp_rac(self.num_nodes, rho_membrane_actives, self.exponent_rho_mediated_rac_inhib, self.threshold_rho_mediated_rac_inhib, self.kdgtp_rac_baseline, self.kdgtp_rho_mediated_rac_inhib_baseline, intercellular_contact_factors, migr_bdry_contact_factors, self.tension_mediated_rac_inhibition_exponent, self.tension_mediated_rac_inhibition_multiplier, self.tension_mediated_rac_hill_exponent, self.tension_mediated_rac_inhibition_half_strain, local_tension_strains, self.tension_fn_type)
        
        self.system_info[access_index, :, parameterorg.kdgtp_rho_index] = chemistry.calculate_kdgtp_rho(self.num_nodes, rac_membrane_actives, self.exponent_rac_mediated_rho_inhib, self.threshold_rac_mediated_rho_inhib, self.kdgtp_rho_baseline, self.kdgtp_rac_mediated_rho_inhib_baseline)
        
        # update mechanics parameters
        self.system_info[access_index, :, [parameterorg.F_x_index, parameterorg.F_y_index]] = np.transpose(F)
        self.system_info[access_index, :, [parameterorg.EFplus_x_index, parameterorg.EFplus_y_index]] = np.transpose(EFplus)
        self.system_info[access_index, :, [parameterorg.EFminus_x_index, parameterorg.EFminus_y_index]] = np.transpose(EFminus)
        self.system_info[access_index, :, [parameterorg.F_rgtpase_x_index, parameterorg.F_rgtpase_y_index]] = np.transpose(F_rgtpase)
        self.system_info[access_index, :, [parameterorg.F_cytoplasmic_x_index, parameterorg.F_cytoplasmic_y_index]] = np.transpose(F_cytoplasmic)
        self.system_info[access_index, :, [parameterorg.F_adhesion_x_index, parameterorg.F_adhesion_y_index]] = np.transpose(F_adhesion)
        
        self.system_info[access_index, :, parameterorg.intercellular_contact_factor_magnitudes_index] = intercellular_contact_factors
        self.system_info[access_index, :, parameterorg.migr_bdry_contact_index] = migr_bdry_contact_factors
        
        self.curr_node_coords = node_coords
        self.curr_node_forces = F - F_adhesion

# -----------------------------------------------------------------
    def initialize_nodal_phase_var_indices(self):
        for index, sys_info_index in enumerate(self.nodal_phase_var_indices):
            label = parameterorg.info_labels[sys_info_index]
            setattr(self, 'nodal_' + label + '_index', index)

# -----------------------------------------------------------------
    def initialize_nodal_pars_indices(self):
        for index, sys_info_index in enumerate(self.nodal_pars_indices):
            label = parameterorg.info_labels[sys_info_index]
            setattr(self, 'nodal_' + label + '_index', index)
            
# -----------------------------------------------------------------
    def initialize_ode_cellwide_phase_var_indices(self):
        for index, sys_info_index in enumerate(self.ode_cellwide_phase_var_indices):
            label = parameterorg.info_labels[sys_info_index]
            setattr(self, 'cellwide_' + label + '_index', index)
            
# -----------------------------------------------------------------
    def initialize_all_cellwide_phase_var_indices(self):
        for index, sys_info_index in enumerate(self.all_cellwide_phase_var_indices):
            label = parameterorg.info_labels[sys_info_index]
            setattr(self, 'cellwide_' + label + '_index', index)
        
# -----------------------------------------------------------------
            
    def calculate_when_randomization_event_occurs(self, mean=None, variance=None):
        if mean == None:
            mean = self.randomization_time_mean
        if variance == None:
            variance = self.randomization_time_variance_factor*mean
            
        step_shift = max(1, np.int(np.abs(np.random.normal(loc=mean, scale=variance))))
            
        next_shift_step = self.curr_tpoint + step_shift
        
        return next_shift_step

# -----------------------------------------------------------------
     
    def check_if_randomization_criteria_met(self, t):
        access_index = self.get_system_info_access_index(t)
        
        local_strains = self.system_info[access_index, :, parameterorg.local_strains_index]
        rac_membrane_active = self.system_info[access_index, :, parameterorg.rac_membrane_active_index]
        rho_membrane_active = self.system_info[access_index, :, parameterorg.rho_membrane_active_index]
        
        polarization_score = analysis_utils.calculate_polarization_rating(rac_membrane_active, rho_membrane_active, self.num_nodes, significant_difference=.2)
        avg_strain = np.average(local_strains)
        
        return avg_strain > 0.03 or polarization_score > 0.6
        
# -----------------------------------------------------------------
    def set_rgtpase_distribution(self, biased_rgtpase_distrib_defn, init_rgtpase_cytosol_gdi_bound_frac, init_rgtpase_membrane_inactive_frac, init_rgtpase_membrane_active_frac, tpoint=0):
        
        distrib_type, bias_direction_range, bias_strength = biased_rgtpase_distrib_defn

        cellwide_distrib_factors = np.zeros(self.num_nodes)
        cellwide_distrib_factors[0] = 1
        
        access_index = self.get_system_info_access_index(tpoint)
        
        for rgtpase_label in ['rac_', 'rho_']:
            for label in parameterorg.chem_labels:
                if rgtpase_label in label:
                    if '_membrane_' in label:
                        if '_inactive' in label:
                            frac_factor = init_rgtpase_membrane_inactive_frac
                        else:
                            frac_factor = init_rgtpase_membrane_active_frac
                        
                        if distrib_type == "unbiased random":
                            rgtpase_distrib = frac_factor*general_utils.calculate_normalized_randomization_factors(self.num_nodes)
                        elif distrib_type == "biased random":
                            if rgtpase_label == "rac_":
                                rgtpase_distrib = frac_factor*calculate_biased_distrib_factors(bias_direction_range, bias_strength, self.num_nodes, 'random')
                            elif rgtpase_label == "rho_":
                                rgtpase_distrib = frac_factor*calculate_biased_distrib_factors(bias_direction_range + np.pi, bias_strength, self.num_nodes, 'random')
                        elif distrib_type == "unbiased uniform":
                            rgtpase_distrib = frac_factor*np.ones(self.num_nodes)/self.num_nodes
                        elif distrib_type == "biased uniform":
                            if rgtpase_label == "rac_":
                                rgtpase_distrib = frac_factor*calculate_biased_distrib_factors(bias_direction_range, bias_strength, self.num_nodes, 'uniform')
                            elif rgtpase_label == "rho_":
                                rgtpase_distrib = frac_factor*calculate_biased_distrib_factors(bias_direction_range + np.pi, bias_strength, self.num_nodes, 'uniform')
                        else:
                            raise StandardError("Invalid initial rGTPase distribution type provided! ({})".format(distrib_type))
                            
                    elif '_cytosolic_' in label:
                        # every node contains the same data regarding 
                        # cytosolic species
                        frac_factor = init_rgtpase_cytosol_gdi_bound_frac
    
                            
                        rgtpase_distrib = frac_factor*cellwide_distrib_factors
                    else:
                        continue
                    
                    self.system_info[access_index, :, eval("parameterorg." + label + "_index")] = rgtpase_distrib

# -----------------------------------------------------------------           
            
    def rotate_rgtpase_distribution(self, num_nodes, tpoint, current_rac_membrane_actives, current_rac_membrane_inactives, current_rho_membrane_actives, current_rho_membrane_inactives):
        random_shift = np.random.choice([-1, 1])
        
        access_index = self.get_system_info_access_index(tpoint)
        
        self.system_info[access_index, :, parameterorg.rac_membrane_active_index] = np.roll(current_rac_membrane_actives, random_shift)
        self.system_info[access_index, :, parameterorg.rac_membrane_inactive_index] = np.roll(current_rac_membrane_inactives, random_shift)
        self.system_info[access_index, :, parameterorg.rho_membrane_active_index] = np.roll(current_rho_membrane_actives, random_shift)
        self.system_info[access_index, :, parameterorg.rho_membrane_inactive_index] = np.roll(current_rho_membrane_inactives, random_shift)
            
        return random_shift
        
    def renew_randomization_rac_kgtp_multipliers(self):
        rfs = np.random.random(self.num_nodes)
        rfs = rfs/np.sum(rfs)
        
        self.randomization_rac_kgtp_multipliers = (self.randomization_magnitude*rfs + 1.0)
        
# -----------------------------------------------------------------
    def set_next_state(self, next_state_array, this_cell_index, num_cells, intercellular_squared_dist_array, line_segment_intersection_matrix, all_cells_node_coords, all_cells_node_forces, are_nodes_inside_other_cells, external_gradient_on_nodes, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors):
        
        new_tpoint = self.curr_tpoint + 1
        next_tstep_system_info_access_index = self.get_system_info_access_index(new_tpoint)
        assert(next_tstep_system_info_access_index > -1)
        
        num_nodes = self.num_nodes
        
        assert(new_tpoint < self.num_timepoints)
      
        self.insert_state_array_into_system_info(next_state_array, new_tpoint)
        
        node_coords = np.transpose(self.system_info[next_tstep_system_info_access_index, :, [parameterorg.x_index, parameterorg.y_index]])

        edge_displacement_vectors_plus = geometry.calculate_edge_vectors(num_nodes, node_coords)
        edge_lengths = geometry.calculate_2D_vector_mags(num_nodes, edge_displacement_vectors_plus)
        
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.edge_lengths_index] = edge_lengths
        
        all_cells_centres = geometry.calculate_centroids(all_cells_node_coords)
        
        intercellular_contact_factors = chemistry.calculate_intercellular_contact_factors(this_cell_index, num_nodes, num_cells, self.intercellular_contact_factor_magnitudes, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists)
            
        if self.space_migratory_bdry_polygon.size == 0:
            migr_bdry_contact_factors = np.ones(num_nodes, dtype=np.int64)
        else:
            migr_bdry_contact_factors = mechanics.calculate_migr_bdry_contact_factors(num_nodes, node_coords, self.space_migratory_bdry_polygon, self.migr_bdry_contact_factor_mag)
        
        rac_membrane_actives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rac_membrane_active_index]
        rho_membrane_actives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rho_membrane_active_index]
        rac_membrane_inactives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rac_membrane_inactive_index]
        rho_membrane_inactives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rho_membrane_inactive_index]
        
        transduced_coa_signals = chemistry.calculate_coa_signals(this_cell_index, num_nodes, num_cells, self.coa_distribution_exponent, self.coa_sensitivity_exponent, self.coa_belt_offset, self.cell_dependent_coa_signal_strengths, intercellular_squared_dist_array, line_segment_intersection_matrix)
        
        #print "transduced_coa_signals: ", transduced_coa_signals
        
        max_coa = np.max(transduced_coa_signals)
        min_coa = np.min(transduced_coa_signals)
        
        if np.isnan(max_coa) or np.isnan(min_coa):
            raise StandardError("Caught a nan!")
            
        print "max_coa: ", np.max(transduced_coa_signals)
        print "min_coa: ", np.min(transduced_coa_signals)
        print "max_ext: ", np.max(external_gradient_on_nodes)
        print "min_ext: ", np.min(external_gradient_on_nodes)
        #print "ic: ", (1.0/3.0)*(intercellular_contact_factors + np.roll(intercellular_contact_factors, 1) + np.roll(intercellular_contact_factors, -1))
        
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.coa_signal_index] = transduced_coa_signals
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.external_gradient_on_nodes_index] = external_gradient_on_nodes
        kgtp_rac_per_node = chemistry.calculate_kgtp_rac(self.num_nodes, rac_membrane_actives, migr_bdry_contact_factors, self.exponent_rac_autoact, self.threshold_rac_autoact, self.kgtp_rac_baseline, self.kgtp_rac_autoact_baseline, transduced_coa_signals, external_gradient_on_nodes, self.randomization_rac_kgtp_multipliers)
        
        kgtp_rho_per_node = chemistry.calculate_kgtp_rho(self.num_nodes, rho_membrane_actives, intercellular_contact_factors, migr_bdry_contact_factors, self.exponent_rho_autoact, self.threshold_rho_autoact, self.kgtp_rho_baseline, self.kgtp_rho_autoact_baseline)
        
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.kgtp_rac_index] = kgtp_rac_per_node
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.kgtp_rho_index] = kgtp_rho_per_node
        
        # ==================================
        # RANDOMIZATION
        if self.randomization == True:                
            if self.randomization_scheme == 0:
                if self.next_randomization_event_tstep == None:
                    if self.check_if_randomization_criteria_met(self.curr_tpoint):
                        self.next_randomization_event_tstep = self.calculate_when_randomization_event_occurs()
                        
                if new_tpoint == self.next_randomization_event_tstep:
                    self.next_randomization_event_tstep = None
                    
                    # run and tumble: cell loses polarity approximately after T seconds
                    self.set_rgtpase_distribution(self.biased_rgtpase_distrib_defn_for_randomization, self.init_rgtpase_cytosol_gdi_bound_frac, self.init_rgtpase_membrane_inactive_frac, self.init_rgtpase_membrane_active_frac, tpoint=new_tpoint)
                    
                    self.system_info[next_tstep_system_info_access_index, 0, parameterorg.randomization_event_occurred_index] = 1
                    rac_membrane_actives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rac_membrane_active_index]
                    rho_membrane_actives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rho_membrane_active_index]
                    rac_membrane_inactives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rac_membrane_inactive_index]
                    rho_membrane_inactives = self.system_info[next_tstep_system_info_access_index, :, parameterorg.rho_membrane_inactive_index]
                
                    
            elif self.randomization_scheme == 1:
                if self.next_randomization_event_tstep == None:
                    self.next_randomization_event_tstep = self.calculate_when_randomization_event_occurs()
                    
                if new_tpoint == self.next_randomization_event_tstep:
                    self.next_randomization_event_tstep = None
                    
                    # renew Rac kgtp rate multipliers
                    self.renew_randomization_rac_kgtp_multipliers()                  
                    self.system_info[next_tstep_system_info_access_index, 0, parameterorg.randomization_event_occurred_index] = 1
                    
            if self.verbose == True:
                print "next_randomization_event_tstep: ", self.next_randomization_event_tstep

            
        # ==================================
        
        rac_cytosolic_gdi_bound = 1 - np.sum(rac_membrane_actives) - np.sum(self.system_info[next_tstep_system_info_access_index, :, parameterorg.rac_membrane_inactive_index])
        rho_cytosolic_gdi_bound = 1 - np.sum(rho_membrane_actives) - np.sum(self.system_info[next_tstep_system_info_access_index, :, parameterorg.rho_membrane_inactive_index])
        
        insertion_array = np.zeros(self.num_nodes)
        insertion_array[0] = 1
        
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.rac_cytosolic_gdi_bound_index] = rac_cytosolic_gdi_bound*insertion_array
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.rho_cytosolic_gdi_bound_index] = rho_cytosolic_gdi_bound*insertion_array
        
#        print "num_nodes: ", self.num_nodes
#        print "node_coords: ", node_coords
#        print "rac_membrane_actives: ", rac_membrane_actives
#        print "rho_membrane_actives: ", rho_membrane_actives
#        print "length_edge_resting: ", self.length_edge_resting
#        print "stiffness_edge: ", self.stiffness_edge
        print "force_rac_mag: ", self.force_rac_max_mag
        print "force_rho_mag: ", self.force_rho_max_mag
        print "force_adh_constant: ", self.force_adh_constant
#        print "area_resting: ", self.area_resting
#        print "stiffness_cytoplasmic: ", self.stiffness_cytoplasmic
        
        
        F, EFplus, EFminus, F_rgtpase, F_cytoplasmic, F_adhesion, local_strains, unit_inside_pointing_vecs = mechanics.calculate_forces(num_nodes, self.num_cells_in_environment, this_cell_index, node_coords, rac_membrane_actives, rho_membrane_actives, self.length_edge_resting, self.stiffness_edge, self.force_rac_exp, self.force_rac_threshold, self.force_rac_max_mag, self.force_rho_exp, self.force_rho_threshold, self.force_rho_max_mag, self.force_adh_constant, self.area_resting, self.stiffness_cytoplasmic, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, self.closeness_dist_criteria)
        
#        printing_efplus = np.round(np.linalg.norm(EFplus, axis=1)*1e6, decimals=2)
#        printing_efminus = np.round(np.linalg.norm(EFminus, axis=1)*1e6, decimals=2)
        printing_frgtpase = np.round(np.linalg.norm(F_rgtpase, axis=1)*1e6, decimals=2)
        printing_fadh = np.round(np.linalg.norm(F_adhesion, axis=1)*1e6, decimals=2)
#        printing_fcyto = np.round(np.linalg.norm(F_cytoplasmic, axis=1)*1e6, decimals=2)
#        
#        current_area = abs(geometry.calculate_polygon_area(num_nodes, node_coords))
#        area_strain = (current_area - self.area_resting)/self.area_resting
        
#        print "edge_forces_plus: ", np.min(printing_efplus), np.max(printing_efplus)
#        print "edge_forces_minus: ", np.min(printing_efminus), np.max(printing_efminus) 
        print "F_rgtpase: ", printing_frgtpase
        print "F_adh: ", printing_fadh
        print "F_rgtpase/F_adh: ", np.max(printing_frgtpase)/np.max(printing_fadh)
#        print "area_strain: ", area_strain
#        print "stiff_cyto: ", self.stiffness_cytoplasmic
#        print "F_cytoplasmic: ", np.min(printing_fcyto), np.max(printing_fcyto) 
        
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.local_strains_index] = local_strains
        
        # update chemistry parameters
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.kdgdi_rac_index] = self.kdgdi_rac*np.ones(num_nodes, dtype=np.float64)
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.kdgdi_rho_index] = self.kdgdi_rho*np.ones(num_nodes, dtype=np.float64)
        
#        if self.verbose == True:
#            print "local_strains: ", local_strains
            
        local_tension_strains = np.where(local_strains < 0, 0, local_strains)

        self.system_info[next_tstep_system_info_access_index, :, parameterorg.kdgtp_rac_index] = chemistry.calculate_kdgtp_rac(self.num_nodes, rho_membrane_actives, self.exponent_rho_mediated_rac_inhib, self.threshold_rho_mediated_rac_inhib, self.kdgtp_rac_baseline, self.kdgtp_rho_mediated_rac_inhib_baseline, intercellular_contact_factors, migr_bdry_contact_factors, self.tension_mediated_rac_inhibition_exponent, self.tension_mediated_rac_inhibition_multiplier, self.tension_mediated_rac_hill_exponent, self.tension_mediated_rac_inhibition_half_strain, local_tension_strains, self.tension_fn_type)

        self.system_info[next_tstep_system_info_access_index, :, parameterorg.kdgtp_rho_index] = chemistry.calculate_kdgtp_rho(self.num_nodes, rac_membrane_actives, self.exponent_rac_mediated_rho_inhib, self.threshold_rac_mediated_rho_inhib, self.kdgtp_rho_baseline, self.kdgtp_rac_mediated_rho_inhib_baseline)
        
        # update mechanics parameters
        self.system_info[next_tstep_system_info_access_index, :, [parameterorg.F_x_index, parameterorg.F_y_index]] = np.transpose(F)
        self.system_info[next_tstep_system_info_access_index, :, [parameterorg.EFplus_x_index, parameterorg.EFplus_y_index]] = np.transpose(EFplus)
        self.system_info[next_tstep_system_info_access_index, :, [parameterorg.EFminus_x_index, parameterorg.EFminus_y_index]] = np.transpose(EFminus)
        self.system_info[next_tstep_system_info_access_index, :, [parameterorg.F_rgtpase_x_index, parameterorg.F_rgtpase_y_index]] = np.transpose(F_rgtpase)
        self.system_info[next_tstep_system_info_access_index, :, [parameterorg.F_cytoplasmic_x_index, parameterorg.F_cytoplasmic_y_index]] = np.transpose(F_cytoplasmic)
        self.system_info[next_tstep_system_info_access_index, :, [parameterorg.F_adhesion_x_index, parameterorg.F_adhesion_y_index]] = np.transpose(F_adhesion)
        
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.intercellular_contact_factor_magnitudes_index] = intercellular_contact_factors
        self.system_info[next_tstep_system_info_access_index, :, parameterorg.migr_bdry_contact_index] = migr_bdry_contact_factors
        
        self.curr_tpoint = new_tpoint
        self.curr_node_coords = node_coords
        self.curr_node_forces = F - F_adhesion

# -----------------------------------------------------------------
    def pack_rhs_arguments(self, t, this_cell_index, all_cells_node_coords, all_cells_node_forces, intercellular_squared_dist_array, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_points_on_other_cells_to_each_node_indices, close_points_on_other_cells_to_each_node_projection_factors, external_gradient_on_nodes):
        access_index = self.get_system_info_access_index(t)
        
        state_parameters = self.system_info[access_index, :, self.nodal_pars_indices]
        
        num_cells = all_cells_node_coords.shape[0]
        num_nodes = self.num_nodes
        all_cells_centres = geometry.calculate_centroids(all_cells_node_coords)
        
        intercellular_contact_factors = chemistry.calculate_intercellular_contact_factors(this_cell_index, num_nodes, num_cells, self.intercellular_contact_factor_magnitudes, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists)
        
        transduced_coa_signals = self.system_info[access_index, :, parameterorg.coa_signal_index]
        
        return state_parameters, this_cell_index, self.num_nodes, self.num_nodal_phase_vars, self.num_ode_cellwide_phase_vars, self.nodal_rac_membrane_active_index, self.length_edge_resting, self.nodal_rac_membrane_inactive_index, self.nodal_rho_membrane_active_index, self.nodal_rho_membrane_inactive_index, self.nodal_x_index, self.nodal_y_index, self.kgtp_rac_baseline, self.kdgtp_rac_baseline, self.kgtp_rho_baseline, self.kdgtp_rho_baseline, self.kgtp_rac_autoact_baseline, self.kgtp_rho_autoact_baseline, self.kdgtp_rho_mediated_rac_inhib_baseline, self.kdgtp_rac_mediated_rho_inhib_baseline, self.kgdi_rac, self.kdgdi_rac, self.kgdi_rho, self.kdgdi_rho, self.threshold_rac_autoact, self.threshold_rho_autoact, self.threshold_rho_mediated_rac_inhib, self.threshold_rac_mediated_rho_inhib, self.exponent_rac_autoact, self.exponent_rho_autoact, self.exponent_rho_mediated_rac_inhib, self.exponent_rac_mediated_rho_inhib, self.diffusion_const_active, self.diffusion_const_inactive, self.nodal_intercellular_contact_factor_magnitudes_index, self.nodal_migr_bdry_contact_index, self.space_at_node_factor_rac, self.space_at_node_factor_rho, self.eta, num_cells, all_cells_node_coords, all_cells_node_forces, all_cells_centres, intercellular_squared_dist_array, self.stiffness_edge, self.force_rac_exp, self.force_rac_threshold, self.force_rac_max_mag, self.force_rho_exp, self.force_rho_threshold, self.force_rho_max_mag, self.force_adh_constant, self.closeness_dist_criteria, self.area_resting, self.stiffness_cytoplasmic, transduced_coa_signals, self.space_physical_bdry_polygon, self.exists_space_physical_bdry_polygon, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_points_on_other_cells_to_each_node_indices, close_points_on_other_cells_to_each_node_projection_factors, intercellular_contact_factors, self.tension_mediated_rac_inhibition_exponent, self.tension_mediated_rac_inhibition_multiplier, self.tension_mediated_rac_hill_exponent, self.tension_mediated_rac_inhibition_half_strain, self.tension_fn_type, external_gradient_on_nodes, self.intercellular_contact_factor_magnitudes, self.randomization_rac_kgtp_multipliers

# -----------------------------------------------------------------
    def trim_system_info(self, environment_tpoint):
        
        assert(environment_tpoint == self.curr_tpoint)
        
        trimmed_system_info = np.empty_like(self.system_info)
        trimmed_system_info[0] = self.system_info[self.get_system_info_access_index(self.curr_tpoint)]
        self.system_info = trimmed_system_info
        
        self.last_trim_timestep = self.curr_tpoint
        
        self.timestep_offset_due_to_dumping = self.curr_tpoint

# -----------------------------------------------------------------
    
    def get_system_info_access_index(self, tpoint):
        return tpoint - self.timestep_offset_due_to_dumping
        
 # -----------------------------------------------------------------
        
    def init_from_storefile(self, environment_tpoint, storefile_path):
        assert(self.system_info == None)
        
        self.system_info = np.zeros((self.max_timepoints_on_ram + 1, self.num_nodes,  len(parameterorg.info_labels)))
        for info_label in parameterorg.info_labels:
            self.system_info[0,:,parameterorg.info_indices_dict[info_label]] = hardio.get_data_for_tsteps(self.cell_index, environment_tpoint, info_label, storefile_path)
        
# -----------------------------------------------------------------
    def execute_step(self, this_cell_index, num_nodes, all_cells_node_coords, all_cells_node_forces, intercellular_squared_dist_array, line_segment_intersection_matrix, external_gradient_fn, be_talkative=False):
        dynamics.print_var = True
        
        if self.skip_dynamics == False:            
            intercellular_squared_dist_array = intercellular_squared_dist_array/(self.L**2)
            all_cells_node_coords = all_cells_node_coords/self.L
            all_cells_node_forces = all_cells_node_forces/self.ML_T2
            
            num_cells = all_cells_node_coords.shape[0]
            
            are_nodes_inside_other_cells = geometry.check_if_nodes_inside_other_cells(this_cell_index, num_nodes, num_cells, all_cells_node_coords)

            close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors = geometry.do_close_points_to_each_node_on_other_cells_exist(num_cells, num_nodes, this_cell_index, all_cells_node_coords[this_cell_index], intercellular_squared_dist_array, self.closeness_dist_squared_criteria, all_cells_node_coords, are_nodes_inside_other_cells)
             
            state_array = dynamics.pack_state_array_from_system_info(self.nodal_phase_var_indices, self.ode_cellwide_phase_var_indices, self.system_info, self.get_system_info_access_index(self.curr_tpoint))
            
            this_cell_node_x_coords = all_cells_node_coords[this_cell_index, :, 0]
            external_gradient_on_nodes = np.where(np.array([external_gradient_fn(x*self.L/1e-6) for x in this_cell_node_x_coords]) < 0, np.zeros(num_nodes), np.array([external_gradient_fn(x*self.L/1e-6) for x in this_cell_node_x_coords]))
            
            rhs_args = self.pack_rhs_arguments(self.curr_tpoint, this_cell_index, all_cells_node_coords, all_cells_node_forces, intercellular_squared_dist_array, are_nodes_inside_other_cells, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, external_gradient_on_nodes)
            
            output_array = scint.odeint(dynamics.cell_dynamics, state_array, [0, 1], args=rhs_args, **self.integration_params)
            
            next_state_array = output_array[1]
            
            self.set_next_state(next_state_array, this_cell_index, num_cells, intercellular_squared_dist_array, line_segment_intersection_matrix, all_cells_node_coords, all_cells_node_forces, are_nodes_inside_other_cells, external_gradient_on_nodes, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors)
            
        else:
            self.system_info[self.get_system_info_access_index(self.curr_tpoint + 1)] = self.system_info[self.get_system_info_access_index(self.curr_tpoint)]
            self.curr_tpoint += 1
# ===============================================================