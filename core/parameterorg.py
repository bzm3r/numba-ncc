# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:10:15 2015

@author: brian
"""

import numpy as np
import environment
import geometry

#g_cell_autonit_ignored_args = ['randomize_rgtpase_distrib', 'init_rgtpase_cytosol_gdi_bound_frac', 'init_rgtpase_membrane_active_frac', 'init_rgtpase_membrane_inactive_frac']


mech_labels = ['x', 'y', 'edge_lengths', 'F_x', 'F_y', 'EFplus_x', 'EFplus_y', 'EFminus_x', 'EFminus_y', 'F_rgtpase_x', 'F_rgtpase_y', 'F_cytoplasmic_x', 'F_cytoplasmic_y', 'F_adhesion_x', 'F_adhesion_y', 'local_strains', 'intercellular_contact_factor_magnitudes', 'migr_bdry_contact', 'unit_in_vec_x', 'unit_in_vec_y']

chem_labels = ['rac_membrane_active', 'rac_membrane_inactive', 'rac_cytosolic_gdi_bound', 'rho_membrane_active', 'rho_membrane_inactive', 'rho_cytosolic_gdi_bound', 'coa_signal', 'kdgdi_rac', 'kdgdi_rho', 'kgtp_rac', 'kgtp_rho', 'kdgtp_rac', 'kdgtp_rho', 'migr_bdry_contact_factor_mag', 'randomization_event_occurred', 'external_gradient_on_nodes']

info_labels = mech_labels + chem_labels

for index, label in enumerate(info_labels):
    exec("{}_index = {}".format(label, repr(index)))
    
num_info_labels = len(mech_labels + chem_labels)

info_indices_dict = {x: i for i, x in enumerate(mech_labels + chem_labels)}

g_rate_labels = ['kgtp', 'kdgtp', 'kgdi', 'kdgdi']

g_var_labels = ['exponent', 'threshold', 'diffusion', 'space', 'eta', 'length', 'stiffness', 'force', 'factor']



#-----------------------------------------------------------------

standard_chem_mech_space_parameter_labels = ['init_rgtpase_cytosol_gdi_bound_frac', 'init_rgtpase_membrane_active_frac', 'init_rgtpase_membrane_inactive_frac', 'kgtp_rac_baseline', 'kdgtp_rac_baseline', 'kgtp_rho_baseline', 'kdgtp_rho_baseline', 'kgtp_rac_autoact_baseline', 'kgtp_rho_autoact_baseline', 'kdgtp_rho_mediated_rac_inhib_baseline', 'kdgtp_rac_mediated_rho_inhib_baseline', 'kgdi_rac', 'kdgdi_rac', 'kgdi_rho', 'kdgdi_rho', 'threshold_rac_autoact', 'threshold_rho_autoact', 'threshold_rac_mediated_rho_inhib', 'threshold_rho_mediated_rac_inhib', 'exponent_rac_autoact', 'exponent_rho_autoact', 'exponent_rho_mediated_rac_inhib', 'exponent_rac_mediated_rho_inhib','stiffness_edge', 'space_physical_bdry_polygon', 'space_migratory_bdry_polygon', 'eta', 'sigma_rac', 'sigma_rho_multiplier', 'force_adh_constant', 'force_rac_exp', 'force_rho_exp', 'force_rac_threshold', 'force_rho_threshold', 'factor_migr_bdry_contact', 'diffusion_const_active', 'diffusion_const_inactive', 'space_at_node_factor_rac', 'space_at_node_factor_rho', 'stiffness_cytoplasmic', 'migr_bdry_contact_factor_mag', 'intercellular_contact_factor_magnitudes', 'closeness_dist_squared_criteria', 'cell_dependent_coa_signal_strengths', 'halfmax_coa_sensing_dist', 'randomization', 'randomization_scheme', 'randomization_time_mean', 'randomization_time_variance_factor', 'randomization_magnitude', 'skip_dynamics', 'tension_mediated_rac_inhibition_half_strain', 'tension_mediated_rac_hill_exponent', 'tension_fn_type', 'coa_sensitivity_percent_drop_over_cell_diameter', 'coa_belt_offset']


#-----------------------------------------------------------------

standard_parameter_dictionary_keys = ['C_total', 'H_total', 'num_nodes', 'num_cells', 'init_cell_radius', 'cell_group_bounding_box', 'biased_rgtpase_distrib_defn', 'stiffness_edge', 'stiffness_cytoplasmic', 'kgtp_rac_multiplier', 'kgtp_rac_autoact_multiplier', 'kdgtp_rac_multiplier', 'kdgtp_rho_mediated_rac_inhib_multiplier', 'kgtp_rho_multiplier', 'kgtp_rho_autoact_multiplier', 'kdgtp_rho_multiplier', 'kdgtp_rac_mediated_rho_inhib_multiplier', 'kdgdi_rac_estimate_multiplier', 'kdgdi_rho_estimate_multiplier', 'kgdi_rac_estimate_multiplier', 'kgdi_rho_estimate_multiplier', 'threshold_tension_mediated_rac_inhib', 'exponent_tension_mediated_rac_inhib', 'threshold_rac_autoact_multiplier', 'threshold_rho_autoact_multiplier', 'threshold_rho_mediated_rac_inhib_multiplier', 'threshold_rac_mediated_rho_inhib_multiplier', 'space_at_node_factor_rac', 'space_at_node_factor_rho', 'migr_bdry_contact_factor_mag', 'init_rgtpase_cytosol_gdi_bound_frac', 'init_rgtpase_membrane_active_frac', 'init_rgtpase_membrane_inactive_frac', 'halfmax_coa_sensing_dist_multiplier', 'exponent_rac_autoact', 'exponent_rho_autoact', 'exponent_rho_mediated_rac_inhib', 'exponent_rac_mediated_rho_inhib', 'sigma_rac', 'force_rac_exp', 'force_rho_exp', 'force_rac_threshold_multiplier', 'force_rho_threshold_multiplier', 'sigma_rho_multiplier', 'max_protrusive_node_velocity', 'randomization_scheme', 'randomization_time_mean', 'randomization_time_variance_factor', 'randomization_magnitude', 'randomization', 'tension_mediated_rac_inhibition_half_strain', 'tension_mediated_rac_hill_exponent', 'tension_fn_type', 'coa_sensitivity_percent_drop_over_cell_diameter', 'coa_belt_offset_multiplier']

#-----------------------------------------------------------------

def make_chem_mech_space_parameter_defn_dict(C_total=3e6, H_total=1.5e6, num_nodes=15, num_cells=1, init_cell_radius=12.5e-6, cell_group_bounding_box=np.array([0, 50, 0, 50])*1e-6, stiffness_edge=2e-10, stiffness_cytoplasmic=100, kgtp_rac_multiplier=40, kgtp_rac_autoact_multiplier=1000, kdgtp_rac_multiplier=17, kdgtp_rho_mediated_rac_inhib_multiplier=30, kgtp_rho_multiplier=200, kgtp_rho_autoact_multiplier=350, kdgtp_rho_multiplier=250, kdgtp_rac_mediated_rho_inhib_multiplier=67, kdgdi_rac_estimate_multiplier=1, kdgdi_rho_estimate_multiplier=1, kgdi_rac_estimate_multiplier=1, kgdi_rho_estimate_multiplier=1, threshold_tension_mediated_rac_inhib=0.2*0.75, exponent_tension_mediated_rac_inhib=3, biased_rgtpase_distrib_defn=None, threshold_rac_autoact_multiplier=0.5, threshold_rho_autoact_multiplier=0.5, threshold_rho_mediated_rac_inhib_multiplier=0.5, threshold_rac_mediated_rho_inhib_multiplier=0.5, space_at_node_factor_rac=1, space_at_node_factor_rho=1, migr_bdry_contact_factor_mag=2, init_rgtpase_cytosol_gdi_bound_frac = 0.8, init_rgtpase_membrane_inactive_frac = 0.1, init_rgtpase_membrane_active_frac = 0.1, intercellular_contact_factor_magnitudes=np.array([2]), halfmax_coa_sensing_dist_multiplier=3, max_protrusive_node_velocity=0.25e-6, sigma_rac=2e-4, force_adh_constant=1.0, sigma_rho_multiplier=0.2, force_rac_exp=3, force_rho_exp=3, force_rac_threshold_multiplier=0.5, force_rho_threshold_multiplier=0.5, closeness_dist_squared_criteria=(0.5e-6)**2, exponent_rac_autoact=3, exponent_rho_autoact=3, exponent_rho_mediated_rac_inhib=3, exponent_rac_mediated_rho_inhib=3, randomization_scheme='wipeout', randomization_time_mean=20, randomization_time_variance_factor=0.25, randomization_magnitude=1.0, randomization=False, skip_dynamics=False, tension_mediated_rac_inhibition_half_strain=0.025, tension_mediated_rac_hill_exponent=3, tension_fn_type=0, coa_sensitivity_percent_drop_over_cell_diameter=0.25, coa_belt_offset_multiplier=1.0):
    
    kgtp_rac_factor = 1.5e-4 # per second
    kdgtp_rac_factor = 1.8e-4 # per second
    
    kgtp_rho_factor = 1.5e-4 # per second    
    kdgtp_rho_factor = 3.5e-4 # per second

    kdgdi_rac_estimate = 0.1*kdgdi_rac_estimate_multiplier # per second
    kgdi_rac_estimate = 0.1*kgdi_rac_estimate_multiplier # per second
    
    kdgdi_rho_estimate = 0.1*kdgdi_rho_estimate_multiplier # per second
    kgdi_rho_estimate = 0.1*kgdi_rho_estimate_multiplier # per second
    
    assert(init_rgtpase_cytosol_gdi_bound_frac + init_rgtpase_membrane_inactive_frac + init_rgtpase_membrane_active_frac == 1)
    
    #--------------   
    kgtp_rac_baseline = kgtp_rac_factor*(kgtp_rac_multiplier + 1) # per second
    kdgtp_rac_baseline = kdgtp_rac_factor*(kdgtp_rac_multiplier + 1) # per second
    #--------------
    kgtp_rho_baseline = kgtp_rho_factor*(kgtp_rho_multiplier + 1) # per second
    kdgtp_rho_baseline = kdgtp_rho_factor*(kdgtp_rho_multiplier + 1) # per second
    #--------------
    kgtp_rac_autoact_baseline = kgtp_rac_autoact_multiplier*kgtp_rac_factor # per second
    kgtp_rho_autoact_baseline = kgtp_rho_autoact_multiplier*kgtp_rho_factor # per second
    #--------------
    kdgtp_rho_mediated_rac_inhib_baseline = kdgtp_rac_factor*kdgtp_rho_mediated_rac_inhib_multiplier # per second
    kdgtp_rac_mediated_rho_inhib_baseline = kdgtp_rho_factor*kdgtp_rac_mediated_rho_inhib_multiplier # per second
    #--------------
    kgdi_rac = kgdi_rac_estimate # per second
    kdgdi_rac = kdgdi_rac_estimate # per second
    #--------------
    kgdi_rho = kgdi_rho_estimate # per second
    kdgdi_rho = kdgdi_rho_estimate # per second
    #--------------
    threshold_rac_autoact = threshold_rac_autoact_multiplier*C_total
    threshold_rho_autoact = threshold_rho_autoact_multiplier*H_total
    #--------------
    threshold_rho_mediated_rac_inhib = threshold_rho_mediated_rac_inhib_multiplier*H_total
    threshold_rac_mediated_rho_inhib = threshold_rac_mediated_rho_inhib_multiplier*C_total
    #--------------
    force_rac_threshold = force_rac_threshold_multiplier*C_total
    force_rho_threshold = force_rho_threshold_multiplier*H_total
    #--------------
    diffusion_const = 0.15*1e-12
    diffusion_const_active = diffusion_const # micrometers squared per second
    diffusion_const_inactive = diffusion_const # micrometers squared per second
    #--------------
    
    cell_node_thetas = np.pi*np.linspace(0, 2, endpoint=False, num=num_nodes)
    cell_node_coords = np.transpose(np.array([init_cell_radius*np.cos(cell_node_thetas), init_cell_radius*np.sin(cell_node_thetas)]))
    edge_vectors = geometry.calculate_edge_vectors(num_nodes, cell_node_coords)
    edge_lengths = geometry.calculate_2D_vector_mags(num_nodes, edge_vectors)
        
    length_edge_resting = np.average(edge_lengths)
        
    eta = (sigma_rac*length_edge_resting)/max_protrusive_node_velocity
    #--------------
    factor_migr_bdry_contact = 10
    #--------------
    stiffness_cytoplasmic = stiffness_cytoplasmic#0.00001 # Newtons
    #--------------
    halfmax_coa_sensing_dist = halfmax_coa_sensing_dist_multiplier*25e-6
    coa_belt_offset = coa_belt_offset_multiplier*2*init_cell_radius

    ignore_list = ['intercellular_contact_factor_magnitudes', 'space_physical_bdry_polygon', 'space_migratory_bdry_polygon', 'cell_dependent_coa_signal_strengths']
    relevant_labels = [label for label in standard_chem_mech_space_parameter_labels if label not in ignore_list]
    parameter_definition_dict = dict(zip(relevant_labels, [eval(label) for label in relevant_labels]))
    
    return parameter_definition_dict

# ==============================================================

def make_environment_given_user_cell_group_defns(environment_name='', num_timesteps=0, user_cell_group_defns=[], space_physical_bdry_polygon=np.array([]), space_migratory_bdry_polygon=np.array([]), external_gradient_fn=lambda x: 0, verbose=False, environment_dir="A:\\cncell\\experiment-storage\\", parameter_overrides=[], num_nodes=15, T=(1/0.5), integration_params={}, closeness_dist_squared_criteria=(0.5e-6)**2, persist=True, parameter_explorer_run=False):
    
    for cell_group_defn_index, user_cell_group_defn in enumerate(user_cell_group_defns):
        C_total = user_cell_group_defn['C_total']
        H_total = user_cell_group_defn['H_total']
        num_nodes = num_nodes
        num_cells = user_cell_group_defn['num_cells']
        init_cell_radius = user_cell_group_defn['init_cell_radius']
        cell_group_bounding_box = user_cell_group_defn['cell_group_bounding_box']
        
        parameter_dict = make_chem_mech_space_parameter_defn_dict(C_total=C_total, H_total=H_total, num_nodes=num_nodes, num_cells=num_cells, init_cell_radius=init_cell_radius, cell_group_bounding_box=cell_group_bounding_box, closeness_dist_squared_criteria=closeness_dist_squared_criteria, **(parameter_overrides[cell_group_defn_index]))
        
        user_cell_group_defn.update([('chem_mech_space_defns', parameter_dict)])
    
    the_environment = environment.Environment(environment_name=environment_name, num_timesteps=num_timesteps, cell_group_defns=user_cell_group_defns, space_physical_bdry_polygon=space_physical_bdry_polygon, space_migratory_bdry_polygon=space_migratory_bdry_polygon, environment_dir=environment_dir, verbose=verbose, num_nodes=num_nodes, T=T, integration_params=integration_params, persist=persist, parameter_explorer_run=parameter_explorer_run, external_gradient_fn=external_gradient_fn)
    
    return the_environment