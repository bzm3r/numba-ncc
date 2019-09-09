import numpy as np
import os
import shutil
import time
from expexec import *

experiment_number = "EXP100"
env_dir_format_string = "A:\\cncell\\notes\\{}\\{}"

TOTAL_TIME = 2500 * 8 * 2
TIMESTEP_LENGTH = 1 / 0.5
NUM_TIMESTEPS = int(TOTAL_TIME / TIMESTEP_LENGTH)
NUM_NODES_PER_CELL = 16

box_height = 25
box_width = 25

corridor_x_offset = 10
corridor_y_offset = 10
physical_bdry_polygon_extra = 5

space_between_box_A_and_box_B = 15

box_x_offset = corridor_x_offset
box_y_offset = corridor_y_offset
other_box_x_offset = box_width + space_between_box_A_and_box_B + corridor_x_offset


width_corridor = box_width + space_between_box_A_and_box_B + 25
height_corridor = box_height

space_migratory_bdry_polygon, space_physical_bdry_polygon = make_space_polygons(
    True, False, width_corridor, height_corridor, corridor_x_offset, corridor_y_offset
)

environment_wide_variable_defns = {
    "num_timesteps": NUM_TIMESTEPS,
    "space_physical_bdry_polygon": space_physical_bdry_polygon,
    "space_migratory_bdry_polygon": space_migratory_bdry_polygon,
    "T": TIMESTEP_LENGTH,
    "num_nodes_per_cell": NUM_NODES_PER_CELL,
    "close_dist": 75e-6,
    "verbose": True,
    "integration_params": {"atol": 1e-4, "rtol": 1e-4},
    "force_rac_max": 1000e-12,
    "force_rho_max": 200e-12,
    "closeness_dist_squared_criteria": (0.25e-6) ** 2,
}

user_cell_group_defns = [
    {
        "cell_group_name": "A",
        "num_cells": 1,
        "init_cell_radius": 12.5e-6,
        "C_total": 3e6,
        "H_total": 1.5e6,
        "cell_group_bounding_box": np.array(
            [
                box_x_offset,
                box_width + box_x_offset,
                box_y_offset,
                box_height + box_y_offset,
            ]
        )
        * 1e-6,
        "intercellular_contact_factor_magnitudes_defn": {"A": cil, "B": cil},
        "cell_dependent_coa_factor_production_defn": {"A": coa, "B": coa},
        "max_coa_sensing_dist_multiplier": 3,
        "coa_factor_degradation_rate_multiplier": coa_degr,
        "biased_rgtpase_distrib_defns": {
            "default": [
                "unbiased random",
                np.array([np.pi / 4, -np.pi / 4]) + np.pi,
                0.5,
            ]
        },
    },
    {
        "cell_group_name": "B",
        "num_cells": 1,
        "init_cell_radius": 12.5e-6,
        "C_total": 3e6,
        "H_total": 1.5e6,
        "cell_group_bounding_box": np.array(
            [
                other_box_x_offset,
                box_width + other_box_x_offset,
                box_y_offset,
                box_height + box_y_offset,
            ]
        )
        * 1e-6,
        "intercellular_contact_factor_magnitudes_defn": {"A": cil, "B": cil},
        "cell_dependent_coa_factor_production_defn": {"A": coa, "B": coa},
        "biased_rgtpase_distrib_defns": {
            "default": [
                "unbiased random",
                np.array([np.pi / 4, -np.pi / 4]) + np.pi,
                0.5,
            ]
        },
    },
]


base_parameter_dict = {
    "threshold_rho_mediated_rac_inhib_multiplier": 0.5,
    "kgtp_rac_multiplier": 35,
    "stiffness_edge": 3e-10,
    "threshold_rho_autoact_multiplier": 0.5,
    "threshold_rac_mediated_rho_inhib_multiplier": 0.5,
    "kgtp_rac_autoact_multiplier": 150,
    "kdgtp_rac_multiplier": 20,
    "kgtp_rho_multiplier": 80,
    "kdgtp_rho_multiplier": 20,
    "kgtp_rho_autoact_multiplier": 175,
    "threshold_rac_autoact_multiplier": 0.5,
    "kdgtp_rac_mediated_rho_inhib_multiplier": 125,
    "kdgtp_rho_mediated_rac_inhib_multiplier": 125,
    "rac_inhibition_cil": 0.2,
    "space_at_node_factor_rac": 2,
    "space_at_node_factor_rho": 2,
}

paramter_override_sets = []

animation_settings = dict(
    [
        ("height_in_pixels", 400),
        ("width_in_pixels", 600),
        ("global_scale", 20),
        ("velocity_scale", 1),
        ("rgtpase_scale", 1250),
        ("show_velocities", False),
        ("show_rgtpase", True),
        ("show_centroid_trail", True),
        ("only_show_cells", []),
        ("polygon_width", 2),
        ("space_physical_bdry_polygon", space_physical_bdry_polygon),
        ("space_migratory_bdry_polygon", space_migratory_bdry_polygon),
    ]
)

"""
height_in_pixels = 400
width_in_pixels = 800
origin_offset_in_micrometers = 6

scale = 20

animation_obj = animator.EnvironmentAnimation(environment_dir, height_in_pixels, width_in_pixels, an_environment, global_scale=scale, rgtpase_scale=1250, velocity_scale=1, show_velocities=False, show_rgtpase=True, show_centroid_trail=True, polygon_width=2, rgtpase_line_width=1, velocity_line_width=1, space_physical_bdry_polygon=space_physical_bdry_polygon                                                                               , space_migratory_bdry_polygon=space_migratory_bdry_polygon)

if TOTAL_TIME < 2000:
    animation_obj.create_animation_from_data(duration=(NUM_TIMESTEPS/(TOTAL_TIME/TIMESTEP_LENGTH))*5.0, timestep_length=TIMESTEP_LENGTH)
else:
    animation_obj.create_animation_from_data(duration=(NUM_TIMESTEPS/(2000.0/TIMESTEP_LENGTH))*5.0, timestep_length=TIMESTEP_LENGTH)
    
"""
