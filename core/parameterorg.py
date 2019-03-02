# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:10:15 2015

@author: brian
"""

import numpy as np
from . import environment
from . import geometry
import copy

#g_cell_autonit_ignored_args = ['randomize_rgtpase_distrib', 'init_rgtpase_cytosol_frac', 'init_rgtpase_membrane_active_frac', 'init_rgtpase_membrane_inactive_frac']


output_mech_labels = ['x', 'y', 'edge_lengths', 'F_x', 'F_y', 'EFplus_x', 'EFplus_y', 'EFminus_x', 'EFminus_y', 'F_rgtpase_x', 'F_rgtpase_y', 'F_cytoplasmic_x', 'F_cytoplasmic_y', 'F_adhesion_x', 'F_adhesion_y', 'local_strains', 'interaction_factors_intercellular_contact_per_celltype', 'migr_bdry_contact', 'unit_in_vec_x', 'unit_in_vec_y']

output_chem_labels = ['rac_membrane_active', 'rac_membrane_inactive', 'rac_cytosolic_gdi_bound', 'rho_membrane_active', 'rho_membrane_inactive', 'rho_cytosolic_gdi_bound', 'coa_signal', 'cil_signal', 'kdgdi_rac', 'kdgdi_rho', 'kgtp_rac', 'kgtp_rho', 'kdgtp_rac', 'kdgtp_rho', 'migr_bdry_contact_factor_mag', 'randomization_event_occurred', 'randomization_rac_kgtp_multipliers', 'chemoattractant_signal_on_nodes', 'chemoattractant_shielding_effect_factor_on_nodes']

output_info_labels = output_mech_labels + output_chem_labels

for index, label in enumerate(output_info_labels):
    exec("{}_index = {}".format(label, repr(index)))
    
num_info_labels = len(output_mech_labels + output_chem_labels)

info_indices_dict = {x: i for i, x in enumerate(output_mech_labels + output_chem_labels)}

g_rate_labels = ['kgtp', 'kdgtp', 'kgdi', 'kdgdi']

g_var_labels = ['exponent', 'threshold', 'diffusion', 'space', 'eta', 'length', 'stiffness', 'force', 'factor']



#-----------------------------------------------------------------
rho_gtpase_parameter_labels = ['C_total', 'H_total', 'init_rgtpase_cytosol_frac', 'init_rgtpase_membrane_active_frac', 'init_rgtpase_membrane_inactive_frac', 'diffusion_const_active', 'diffusion_const_inactive', 'kgtp_rac_baseline', 'kdgtp_rac_baseline', 'kgtp_rho_baseline', 'kdgtp_rho_baseline', 'kgtp_rac_autoact_baseline', 'kgtp_rho_autoact_baseline', 'kdgtp_rho_mediated_rac_inhib_baseline', 'kdgtp_rac_mediated_rho_inhib_baseline', 'kgdi_rac', 'kdgdi_rac', 'kgdi_rho', 'kdgdi_rho', 'threshold_rac_activity', 'threshold_rho_activity', 'hill_exponent', 'tension_mediated_rac_inhibition_half_strain', 'tension_mediated_rac_inhibition_magnitude', 'strain_calculation_type']

mech_parameter_labels = ['length_edge_resting', 'area_resting', 'stiffness_edge', 'stiffness_cytoplasmic', 'eta', 'max_protrusive_nodal_velocity', 'max_force_rac', 'max_force_rho', 'threshold_force_rac_activity', 'threshold_force_rho_activity', 'force_adh_const']

space_parameter_labels = ['space_physical_bdry_polygon', 'space_migratory_bdry_polygon']

interaction_parameter_labels = ['interaction_factor_migr_bdry_contact', 'interaction_factors_intercellular_contact_per_celltype', 'interaction_factors_coa_per_celltype', 'closeness_dist_squared_criteria', 'coa_intersection_exponent', 'max_coa_signal', 'coa_sensing_dist_at_value', 'coa_sensing_value_at_dist', 'chemoattractant_mediated_coa_dampening_factor', 'chemoattractant_mediated_coa_production_factor', 'max_chemoattractant_signal', 'enable_chemoattractant_shielding_effect']

randomization_parameter_labels = ['randomization_scheme', 'randomization_time_mean', 'randomization_time_variance_factor', 'randomization_magnitude', 'randomization_node_percentage', 'randomization_type']

model_run_parameter_labels = ['num_nodes', 'skip_dynamics', 'biased_rgtpase_distrib_defn', 'init_node_coords', 'init_cell_radius']

all_parameter_labels = rho_gtpase_parameter_labels + mech_parameter_labels + space_parameter_labels + interaction_parameter_labels + randomization_parameter_labels + model_run_parameter_labels


#-----------------------------------------------------------------

polygon_model_parameters = {'init_cell_radius': 2e-05, 'num_nodes': None}

user_rho_gtpase_biochemistry_parameters = {'kdgdi_multiplier': [1, 2], 'init_rgtpase_membrane_active_frac': [0, 1], 'coa_sensing_value_at_dist': 0.5, 'threshold_rho_activity_multiplier': [0.01, 1], 'kgtp_rac_autoact_multiplier': [1, 1000], 'C_total': [2e6, 3e6], 'kdgtp_rho_multiplier': [1, 2000], 'coa_sensing_dist_at_value': 0.00011, 'tension_mediated_rac_inhibition_half_strain': [0.01, 0.99], 'tension_mediated_rac_inhibition_magnitude': [1.0, 100.0], 'strain_calculation_type': None, 'init_rgtpase_cytosol_frac': [0, 1], 'hill_exponent': 3, 'kgtp_rac_multiplier': [1, 500], 'max_coa_signal': [-1, 10], 'H_total': [0.5e6, 1.5e6], 'diffusion_const': [2e-14, 4.5e-13], 'kdgtp_rac_multiplier': [1, 2000], 'kgtp_rho_multiplier': [1, 500], 'kgdi_multiplier': [1, 2], 'kgtp_rho_autoact_multiplier': [1, 1000], 'init_rgtpase_membrane_inactive_frac': [0, 1], 'kdgtp_rac_mediated_rho_inhib_multiplier': [1, 2000], 'kdgtp_rho_mediated_rac_inhib_multiplier': [1, 2000],
                                           'threshold_rac_activity_multiplier': [0.01, 1], 'max_chemoattractant_signal': None, 'chemoattractant_mediated_coa_dampening_factor': [0.0, 1.0], 'chemoattractant_mediated_coa_production_factor': [0.0, np.inf]}


user_interaction_parameters = {'interaction_factors_intercellular_contact_per_celltype': None, 'interaction_factor_migr_bdry_contact': None, 'interaction_factors_coa_per_celltype': None, 'closeness_dist_squared_criteria': [(0.25e-6)**2, (5e-6)**2], 'coa_intersection_exponent': [0.0, 1000.0], 'chemoattractant_shielding_effect_length_squared': None, 'enable_chemoattractant_shielding_effect': None}

user_space_parameters = {'space_physical_bdry_polygon': None, 'space_migratory_bdry_polygon': None}

user_mechanical_parameters = {'stiffness_cytoplasmic': None, 'length_3D_dimension': 1e-05, 'skip_dynamics': None, 'max_force_rac': [0.001*10e3, 5*10e3], 'force_adh_const': [0, 100], 'stiffness_edge': [1, 8000], 'force_rho_multiplier': [0, 1], 'eta': [0.01*1e5, 1e100]}

user_randomization_parameters = {'randomization_magnitude': None, 'randomization_scheme': None, 'randomization_time_mean': None, 'randomization_time_variance_factor': None, 'randomization_node_percentage': [0.01, 0.5], 'randomization_type': None}
    
user_model_run_parameters = {'skip_dynamics': None, 'biased_rgtpase_distrib_defn': None, 'num_nodes': [3, 100]}

all_user_parameters_with_justifications = {}
for parameter_dict in [polygon_model_parameters, user_rho_gtpase_biochemistry_parameters, user_interaction_parameters, user_space_parameters, user_mechanical_parameters, user_randomization_parameters, user_model_run_parameters]:
    all_user_parameters_with_justifications.update(parameter_dict)

#-----------------------------------------------------------------
def verify_user_parameters(justify_parameters, user_parameter_dict):
    global all_user_parameters_with_justifications
    
    for key in list(user_parameter_dict.keys()):
        try:
            justification = all_user_parameters_with_justifications[key]
        except:
            raise Exception("Unknown parameter given: {}".format(key))

        if justify_parameters and justification != None:
            value = user_parameter_dict[key]
            
            if type(justification) == list:
                assert(len(justification) == 2)
                if not (justification[0] <= value <= justification[1]):
                    raise Exception("Parameter {} violates justification ({}) with value {}".format(key, justification, value))
            elif value != justification:
                raise Exception("Parameter {} violates justification ({}) with value {}".format(key, justification, value))
    
#-----------------------------------------------------------------


def make_cell_group_parameter_dict(justify_parameters, user_parameter_dict):
    verify_user_parameters(justify_parameters, user_parameter_dict)
    cell_parameter_dict = {}
    
    kgtp_rac_unmodified = 2e-4 # per second
    kdgtp_rac_unmodified = 2e-4 # per second
    
    kgtp_rho_unmodified = 2e-4 # per second    
    kdgtp_rho_unmodified = 2e-4 # per second
    
    cell_parameter_dict['num_nodes'] = user_parameter_dict['num_nodes']
    cell_parameter_dict['skip_dynamics'] = user_parameter_dict['skip_dynamics']
     
    C_total, H_total = user_parameter_dict['C_total'], user_parameter_dict['H_total']
    
    cell_parameter_dict['C_total'] = C_total
    cell_parameter_dict['H_total'] = H_total
    
    assert(user_parameter_dict['init_rgtpase_cytosol_frac'] + user_parameter_dict['init_rgtpase_membrane_inactive_frac'] + user_parameter_dict['init_rgtpase_membrane_active_frac'] == 1)
    
    cell_parameter_dict['init_rgtpase_cytosol_frac'] = user_parameter_dict['init_rgtpase_cytosol_frac']
    cell_parameter_dict['init_rgtpase_membrane_inactive_frac'] = user_parameter_dict['init_rgtpase_membrane_inactive_frac']
    cell_parameter_dict['init_rgtpase_membrane_active_frac'] = user_parameter_dict['init_rgtpase_membrane_active_frac']
    
    #--------------   
    cell_parameter_dict['kgtp_rac_baseline'] = kgtp_rac_unmodified*user_parameter_dict['kgtp_rac_multiplier'] # per second
    cell_parameter_dict['kdgtp_rac_baseline'] = kdgtp_rac_unmodified*user_parameter_dict['kdgtp_rac_multiplier'] # per second
    #--------------
    cell_parameter_dict['kgtp_rho_baseline'] = kgtp_rho_unmodified*user_parameter_dict['kgtp_rho_multiplier'] # per second
    cell_parameter_dict['kdgtp_rho_baseline'] = kdgtp_rho_unmodified*user_parameter_dict['kdgtp_rho_multiplier']# per second
    #--------------
    cell_parameter_dict['kgtp_rac_autoact_baseline'] = user_parameter_dict['kgtp_rac_autoact_multiplier']*kgtp_rac_unmodified # per second
    cell_parameter_dict['kgtp_rho_autoact_baseline'] = user_parameter_dict['kgtp_rho_autoact_multiplier']*kgtp_rho_unmodified # per second
    #--------------
    cell_parameter_dict['kdgtp_rho_mediated_rac_inhib_baseline'] = kdgtp_rac_unmodified*user_parameter_dict['kdgtp_rho_mediated_rac_inhib_multiplier'] # per second
    cell_parameter_dict['kdgtp_rac_mediated_rho_inhib_baseline'] = kdgtp_rho_unmodified*user_parameter_dict['kdgtp_rac_mediated_rho_inhib_multiplier'] # per second
    #--------------
    kgdi = 0.15
    kdgdi = 0.02
    cell_parameter_dict['kgdi_rac'] = kgdi*user_parameter_dict['kgdi_multiplier'] # per second
    cell_parameter_dict['kdgdi_rac'] = kdgdi*user_parameter_dict['kdgdi_multiplier'] # per second
    #--------------
    cell_parameter_dict['kgdi_rho'] = kgdi*user_parameter_dict['kgdi_multiplier'] # per second
    cell_parameter_dict['kdgdi_rho'] = kdgdi*user_parameter_dict['kdgdi_multiplier'] # per second
    #--------------
    cell_parameter_dict['threshold_rac_activity'] = user_parameter_dict['threshold_rac_activity_multiplier']*C_total
    cell_parameter_dict['threshold_rho_activity'] = user_parameter_dict['threshold_rho_activity_multiplier']*H_total
    #--------------
    cell_parameter_dict['diffusion_const_active'] = user_parameter_dict['diffusion_const'] # micrometers squared per second
    cell_parameter_dict['diffusion_const_inactive'] = user_parameter_dict['diffusion_const'] # micrometers squared per second
    #--------------
    cell_parameter_dict['hill_exponent'] = user_parameter_dict['hill_exponent']
    cell_parameter_dict['tension_mediated_rac_inhibition_half_strain'] = user_parameter_dict['tension_mediated_rac_inhibition_half_strain']
    cell_parameter_dict['tension_mediated_rac_inhibition_magnitude'] = user_parameter_dict['tension_mediated_rac_inhibition_magnitude']
    cell_parameter_dict['strain_calculation_type'] = user_parameter_dict['strain_calculation_type']
    cell_parameter_dict['max_coa_signal'] = user_parameter_dict['max_coa_signal']
    cell_parameter_dict['coa_sensing_dist_at_value'] = user_parameter_dict['coa_sensing_dist_at_value']
    cell_parameter_dict['coa_sensing_value_at_dist'] = user_parameter_dict['coa_sensing_value_at_dist']
    cell_parameter_dict['coa_intersection_exponent'] = user_parameter_dict['coa_intersection_exponent']
    cell_parameter_dict['chemoattractant_mediated_coa_dampening_factor'] = user_parameter_dict['chemoattractant_mediated_coa_dampening_factor']
    cell_parameter_dict['chemoattractant_mediated_coa_production_factor'] = user_parameter_dict['chemoattractant_mediated_coa_production_factor']
    cell_parameter_dict['max_chemoattractant_signal'] = user_parameter_dict['max_chemoattractant_signal']
    cell_parameter_dict['enable_chemoattractant_shielding_effect'] = user_parameter_dict['enable_chemoattractant_shielding_effect']
    
    num_nodes, init_cell_radius = user_parameter_dict['num_nodes'], user_parameter_dict['init_cell_radius']
    cell_parameter_dict['num_nodes'], cell_parameter_dict['init_cell_radius'] = num_nodes, init_cell_radius
    
    cell_node_thetas = np.pi*np.linspace(0, 2, endpoint=False, num=num_nodes)
    cell_node_coords = np.transpose(np.array([init_cell_radius*np.cos(cell_node_thetas), init_cell_radius*np.sin(cell_node_thetas)]))
    edge_vectors = geometry.calculate_edge_vectors(cell_node_coords)
    edge_lengths = geometry.calculate_2D_vector_mags(edge_vectors)
        
    length_edge_resting = np.average(edge_lengths)

    length_3D_dimension = user_parameter_dict['length_3D_dimension']
    cell_parameter_dict['eta'] = user_parameter_dict['eta']*length_3D_dimension
    cell_parameter_dict['stiffness_edge'] = user_parameter_dict['stiffness_edge']*length_3D_dimension
    cell_parameter_dict['stiffness_cytoplasmic'] = user_parameter_dict['stiffness_cytoplasmic']
    cell_parameter_dict['length_edge_resting'] = length_edge_resting
    cell_parameter_dict['max_force_rac'] = user_parameter_dict['max_force_rac']*length_edge_resting*200e-9
    cell_parameter_dict['max_force_rho'] = user_parameter_dict['force_rho_multiplier']*cell_parameter_dict['max_force_rac']
    cell_parameter_dict['max_protrusive_nodal_velocity'] = cell_parameter_dict['max_force_rac']/cell_parameter_dict['eta']
    cell_parameter_dict['threshold_force_rac_activity'] = user_parameter_dict['threshold_rac_activity_multiplier']*C_total
    cell_parameter_dict['threshold_force_rho_activity'] = user_parameter_dict['threshold_rho_activity_multiplier']*H_total
    cell_parameter_dict['force_adh_const'] = user_parameter_dict['force_adh_const']
    #--------------
    cell_parameter_dict['closeness_dist_squared_criteria'] = user_parameter_dict['closeness_dist_squared_criteria']
    cell_parameter_dict['interaction_factor_migr_bdry_contact'] = user_parameter_dict['interaction_factor_migr_bdry_contact']
    cell_parameter_dict['interaction_factors_intercellular_contact_per_celltype'] = user_parameter_dict['interaction_factors_intercellular_contact_per_celltype']
    cell_parameter_dict['interaction_factors_coa_per_celltype'] = user_parameter_dict['interaction_factors_coa_per_celltype']
    #--------------
    cell_parameter_dict['space_physical_bdry_polygon'] = user_parameter_dict['space_physical_bdry_polygon']
    cell_parameter_dict['space_migratory_bdry_polygon'] = user_parameter_dict['space_migratory_bdry_polygon']
    #--------------
    randomization_scheme = user_parameter_dict['randomization_scheme']
    cell_parameter_dict['randomization_scheme'] = randomization_scheme
    
    cell_parameter_dict['randomization_time_mean'] = user_parameter_dict['randomization_time_mean']
    cell_parameter_dict['randomization_time_variance_factor'] = user_parameter_dict['randomization_time_variance_factor']
    cell_parameter_dict['randomization_magnitude'] = user_parameter_dict['randomization_magnitude']        
    cell_parameter_dict['randomization_node_percentage'] = user_parameter_dict['randomization_node_percentage']
    cell_parameter_dict['randomization_type'] = user_parameter_dict['randomization_type']
    
    return cell_parameter_dict

# ==============================================================

def expand_interaction_factors_intercellular_contact_per_celltype_array(num_cell_groups, cell_group_defns, this_cell_group_defn):
        interaction_factors_intercellular_contact_per_celltype_defn = this_cell_group_defn['interaction_factors_intercellular_contact_per_celltype']
        
        num_defns = len(list(interaction_factors_intercellular_contact_per_celltype_defn.keys()))
        
        if num_defns != num_cell_groups:
            raise Exception("Number of cell groups does not equal number of keys in interaction_factors_intercellular_contact_per_celltype_defn.")
        
        interaction_factors_intercellular_contact_per_celltype = []
        for cgi in range(num_cell_groups):
            cg = cell_group_defns[cgi]
            cg_name = cg['cell_group_name']
            intercellular_contact_factor_mag = interaction_factors_intercellular_contact_per_celltype_defn[cg_name]
            
            interaction_factors_intercellular_contact_per_celltype += (cell_group_defns[cgi]['num_cells'])*[intercellular_contact_factor_mag]
                
        return np.array(interaction_factors_intercellular_contact_per_celltype)
    
# ==============================================================

def expand_interaction_factors_coa_per_celltype_array(num_cell_groups, cell_group_defns, this_cell_group_defn):
        interaction_factors_coa_per_celltype_defn = this_cell_group_defn['interaction_factors_coa_per_celltype']
        
        num_defns = len(list(interaction_factors_coa_per_celltype_defn.keys()))
        
        if num_defns != num_cell_groups:
            raise Exception("Number of cell groups does not equal number of keys in interaction_factors_intercellular_contact_per_celltype_defn.")
        
        interaction_factors_coa_per_celltype = []
        for cgi in range(num_cell_groups):
            cg = cell_group_defns[cgi]
            cg_name = cg['cell_group_name']
            cg_num_nodes = this_cell_group_defn['parameter_dict']['num_nodes']
            coa_signal_strength = interaction_factors_coa_per_celltype_defn[cg_name]/cg_num_nodes
            
            interaction_factors_coa_per_celltype += (cell_group_defns[cgi]['num_cells'])*[coa_signal_strength]
                
        return np.array(interaction_factors_coa_per_celltype)
    
# ==============================================================

def find_undefined_labels(cell_group_parameter_dict):
    given_labels = list(cell_group_parameter_dict.keys())
    undefined_labels = []
    global all_parameter_labels
    
    for label in all_parameter_labels:
        if label not in given_labels:
            undefined_labels.append(label)
            
    return undefined_labels
        
    
# ==============================================================

def make_environment_given_user_cell_group_defns(animation_settings, environment_name='', num_timesteps=0, user_cell_group_defns=[], space_physical_bdry_polygon=np.array([]), space_migratory_bdry_polygon=np.array([]), chemoattractant_shielding_effect_length_squared=0.0, chemoattractant_signal_fn=lambda x: 0, verbose=False, environment_dir="B:\\numba-ncc\\output", T=(1/0.5), integration_params={}, persist=True, parameter_explorer_run=False, max_timepoints_on_ram=1000, seed=None, allowed_drift_before_geometry_recalc=1.0, parameter_explorer_init_rho_gtpase_conditions=None, justify_parameters=True, cell_placement_method='r', max_placement_distance_factor=1.0, init_random_cell_placement_x_factor=0.25, convergence_test=False, graph_group_centroid_splits=False):
    
    num_cell_groups = len(user_cell_group_defns)
        
    for cell_group_defn_index, user_cell_group_defn in enumerate(user_cell_group_defns):
        user_cell_group_parameter_dict = copy.deepcopy(user_cell_group_defn['parameter_dict'])
        
        user_cell_group_parameter_dict['interaction_factors_intercellular_contact_per_celltype'] = expand_interaction_factors_intercellular_contact_per_celltype_array(num_cell_groups, user_cell_group_defns, user_cell_group_defn)
        user_cell_group_parameter_dict['interaction_factors_coa_per_celltype'] = expand_interaction_factors_coa_per_celltype_array(num_cell_groups, user_cell_group_defns, user_cell_group_defn)
        cell_group_parameter_dict = make_cell_group_parameter_dict(justify_parameters, user_cell_group_parameter_dict)
            
        user_cell_group_defn.update([('parameter_dict', cell_group_parameter_dict)])
        
        
    
    the_environment = environment.Environment(environment_name=environment_name, num_timesteps=num_timesteps, cell_group_defns=user_cell_group_defns, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, environment_dir=environment_dir, verbose=verbose, T=T, integration_params=integration_params, persist=persist, parameter_explorer_run=parameter_explorer_run,
      chemoattractant_shielding_effect_length_squared=chemoattractant_shielding_effect_length_squared,
      chemoattractant_signal_fn=chemoattractant_signal_fn, max_timepoints_on_ram=max_timepoints_on_ram, seed=seed, allowed_drift_before_geometry_recalc=allowed_drift_before_geometry_recalc, parameter_explorer_init_rho_gtpase_conditions=parameter_explorer_init_rho_gtpase_conditions, cell_placement_method=cell_placement_method, max_placement_distance_factor=max_placement_distance_factor, init_random_cell_placement_x_factor=init_random_cell_placement_x_factor, convergence_test=convergence_test, graph_group_centroid_splits=graph_group_centroid_splits)
    
    return the_environment