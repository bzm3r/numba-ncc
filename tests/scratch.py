# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:47:10 2017

@author: Brian
"""

polygon_model_parameters = ['num_nodes', 'init_cell_radius']
polygon_model_parameter_justifications = [None, 20e-6]
if len(polygon_model_parameters) != len(polygon_model_parameter_justifications):
    raise StandardError("Not enough justifications provided for polygon_model_parameters!") 
print dict(zip(polygon_model_parameters, polygon_model_parameter_justifications))
    
user_rho_gtpase_biochemistry_parameters = ['C_total', 'H_total', 'init_rgtpase_cytosol_frac', 'init_rgtpase_membrane_active_frac', 'init_rgtpase_membrane_inactive_frac', 'diffusion_const', 'kgdi_multiplier', 'kdgdi_multiplier', 'kgtp_rac_multiplier', 'kgtp_rac_autoact_multiplier', 'kdgtp_rac_multiplier', 'kdgtp_rho_mediated_rac_inhib_multiplier', 'threshold_rac_activity_multiplier', 'kgtp_rho_multiplier', 'kgtp_rho_autoact_multiplier', 'kdgtp_rho_multiplier', 'kdgtp_rac_mediated_rho_inhib_multiplier', 'threshold_rho_activity_multiplier', 'hill_exponent', 'tension_mediated_rac_inhibition_half_strain', 'max_coa_signal', 'coa_sensing_dist_at_value', 'coa_sensing_value_at_dist']
rho_gtpase_biochemistry_parameter_justifications = [[2e6, 3e6], [0.5e6, 1.5e6], [0, 1], [0, 1], [0, 1], [0.02e-12, 0.45e-12], [1, 2], [1, 2], [1, 500], [1, 500], [1, 2000], [1, 2000], [0.25, 0.5], [1, 500], [1, 500], [1, 2000], [1, 2000], [0.25, 0.5], 3, [0.01, 0.99], [-1, 10], 110e-6, 0.5]
if len(user_rho_gtpase_biochemistry_parameters) != len(rho_gtpase_biochemistry_parameter_justifications):
    raise StandardError("Not enough justifications provided for user_rho_gtpase_biochemistry_parameters!")
print dict(zip(user_rho_gtpase_biochemistry_parameters, rho_gtpase_biochemistry_parameter_justifications))

user_interaction_parameters = ['interaction_factor_migr_bdry_contact', 'interaction_factors_intercellular_contact_per_celltype', 'interaction_factors_coa_per_celltype', 'closeness_dist_squared_criteria']
interaction_parameter_justifications = [None, None, None, 0.25e-12]
if len(user_interaction_parameters) != len(interaction_parameter_justifications):
    raise StandardError("Not enough justifications provided for user_interaction_parameters!")
print dict(zip(user_interaction_parameters, interaction_parameter_justifications))

user_space_parameters = ['space_physical_bdry_polygon', 'space_migratory_bdry_polygon']
space_parameter_justifications = [None, None]
if len(user_space_parameters) != len(space_parameter_justifications):
    raise StandardError("Not enough justifications provided for user_space_parameters!")
print dict(zip(user_space_parameters, space_parameter_justifications))

user_mechanical_parameters = ['stiffness_edge', 'stiffness_cytoplasmic', 'eta', 'max_force_rac', 'force_rho_multiplier', 'length_3D_dimension', 'force_adh_const', 'skip_dynamics']
mechanical_parameter_justifications = [[1000, 8000], None, [1e5, 9e5], [0.5*10e3, 2*10e3], [0, 1], 10e-6, [0, 100], None]
if len(user_mechanical_parameters) != len(mechanical_parameter_justifications):
    raise StandardError("Not enough justifications provided for user_mechanical_parameters!")
print dict(zip(user_mechanical_parameters, mechanical_parameter_justifications))

user_randomization_parameters = ['randomization_scheme', 'randomization_time_mean', 'randomization_time_variance_factor', 'randomization_magnitude']
randomization_parameter_justifications = [None, None, None, None]
if len(user_randomization_parameters) != len(randomization_parameter_justifications):
    raise StandardError("Not enough justifications provided for user_randomization_parameters!")
print dict(zip(user_randomization_parameters, randomization_parameter_justifications))
    
user_model_run_parameters = ['num_nodes', 'skip_dynamics', 'biased_rgtpase_distrib_defn']
model_run_parameter_justifications = [[3, 100], None, None]
if len(user_model_run_parameters) != len(model_run_parameter_justifications):
    raise StandardError("Not enough justifications provided for user_model_run_parameters!")
print dict(zip(user_model_run_parameters, model_run_parameter_justifications))
    
