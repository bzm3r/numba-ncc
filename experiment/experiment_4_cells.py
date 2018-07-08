
import parameterorg
import animator
import numpy as np
import analysis
import datavis
import os

cell_group_defn_labels = ['cell_group_name', 'num_cells', 'init_cell_radius', 'num_nodes_per_cell', 'C_total', 'H_total', 'cell_group_bounding_box', 'chem_mech_space_defns', 'integration_params', 'intercellular_contact_factor_magnitudes_defn']

user_cell_group_defns = [
{'cell_group_name': 'A', 'num_cells': 4, 
 'init_cell_radius': 12.5e-6,
 'num_nodes_per_cell': 15,
 'C_total': 3e6, 'H_total': 1e6,
 'cell_group_bounding_box': np.array([0, 50, 0, 50])*1e-6,
 'integration_params': {}, 
 'intercellular_contact_factor_magnitudes_defn': {'A': 1000, 'B': 2}},
{'cell_group_name': 'B', 'num_cells': 0, 
 'init_cell_radius': 12.5e-6,
 'num_nodes_per_cell': 15,
 'C_total': 3e6, 'H_total': 1e6,
 'cell_group_bounding_box': np.array([0, 50, 50, 100])*1e-6,
 'integration_params': {}, 
 'intercellular_contact_factor_magnitudes_defn': {'A': 2, 'B': 2}}]
 
# ========================================================================

all_po_sets = [{'randomize_rgtpase_distrib': 0.0, 'kdgtp_rac_multiplier': 20.0, 'kgtp_rac_multiplier': 77.171717171717162, 'kgtp_rho_multiplier': 123.23232323232322, 'kdgtp_rho_multiplier': 20.0, 'kgtp_rho_autoact_multiplier': 500.0, 'randomization_exponent': 1.0, 'kdgtp_rac_mediated_rho_inhib_multiplier': 500.0, 'kgtp_rac_autoact_multiplier': 500.0, 'kdgtp_rho_mediated_rac_inhib_multiplier': 500.0}]

for po_set_index, candidate_po_set in enumerate(all_po_sets):    
    base_environment_name = 'fourcell_pset_{}'.format(po_set_index)
    
    environment_dir = "A:\\cncell\\notes\\2015_JUN_8\\{}".format(base_environment_name)
    if not os.path.exists(environment_dir):
        os.makedirs(environment_dir)
        
    datavis.graph_parameter_overrides(base_environment_name, candidate_po_set, save_name=base_environment_name + '_parameter_vis', save_dir=environment_dir)
    
    NUM_TIMESTEPS = 1500
    NUM_EXPERIMENT_REPEATS = 3
    
    for x in range(NUM_EXPERIMENT_REPEATS):
        environment_name = base_environment_name + '_NT={}_RPT={}'.format(NUM_TIMESTEPS, x+1)
        
        corridor_height = 50e-6
        corridor_length = 50e-6
        
        space_physical_bdry_polygon = np.array([])
        
        print("Creating environment...")
        an_environment = parameterorg.make_environment_given_user_cell_group_defns(environment_name, NUM_TIMESTEPS, user_cell_group_defns, space_physical_bdry_polygon=space_physical_bdry_polygon, parameter_overrides=candidate_po_set, verbose=True, environment_filepath=environment_dir)
        
        print("Executing dynamics...")
        an_environment.execute_system_dynamics_for_all_times()
        
        a_cell = an_environment.cells_in_environment[0]
        
        symmetry_output_labels, symmetry_output_values, symmetry_output_arrays = analysis.calculate_rgtpase_symmetries(a_cell)
        symmetry_results = dict(list(zip(symmetry_output_labels, symmetry_output_values)))
        
        datavis.graph_important_cell_variables_over_time(a_cell, symmetry_ratings_dict=symmetry_results, save_name=environment_name + '_important_cell_vars_graph', save_dir=environment_dir)
        
        datavis.graph_rates(a_cell, save_name=environment_name + '_strains_and_rates_graph', save_dir=environment_dir)
        
        print("Making animation...")
        height_in_pixels = 400
        width_in_pixels = 400
        origin_offset_in_micrometers = 5
        
        scale = 2.5
        
        animation_obj = animator.EnvironmentAnimation(height_in_pixels, width_in_pixels, an_environment, origin_offset_in_micrometers, cartesian_to_pixels_scale=scale, rgtpase_info_scale=scale*15, velocity_info_scale=scale, plot_rgtpase_lines=True, plot_forces=False)
        
        animation_obj.create_animation_from_data(duration=5)
        
        print("Done.")