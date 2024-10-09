from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

import trimesh
from omni.isaac.lab.terrains.trimesh import mesh_terrains_cfg

if TYPE_CHECKING:
    from omni.isaac.lab.terrains.height_field import hf_terrains_cfg
    from . import custom_terrains_cfg

def custom_box_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HurdleNoiseTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve the terrain configuration
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    total_height = box_height
    if cfg.double_box:
        total_height *= 2.0
    # constants for terrain generation
    terrain_height = 1.0
    box_2_ratio = 0.6

    # Generate the top box
    dim = (cfg.platform_width[0], cfg.platform_width[1], terrain_height + total_height)
    pos = (0.5 * cfg.size[0] + cfg.box_position[0], 0.5 * cfg.size[1] + cfg.box_position[1], (total_height - terrain_height) / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)
    # Generate the lower box
    if cfg.double_box:
        # calculate the size of the lower box
        outer_box_x = cfg.platform_width[0] + (cfg.size[0] - cfg.platform_width[0]) * box_2_ratio
        outer_box_y = cfg.platform_width[1] + (cfg.size[1] - cfg.platform_width[1]) * box_2_ratio
        # create the lower box
        dim = (outer_box_x, outer_box_y, terrain_height + total_height / 2)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], (total_height - terrain_height) / 2 - total_height / 4)
        box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box_mesh)
    
    # # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin

def hurdle_noise_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HurdleNoiseTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve the terrain configuration
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    total_height = box_height
    if cfg.double_box:
        total_height *= 2.0
    # constants for terrain generation
    terrain_height = 1.0
    box_2_ratio = 0.6

    # Generate the top box
    dim = (cfg.platform_width[0], cfg.platform_width[1], terrain_height + total_height)
    pos = (0.5 * cfg.size[0] + cfg.box_position[0], 0.5 * cfg.size[1] + cfg.box_position[1], (total_height - terrain_height) / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)
    # Generate the lower box
    if cfg.double_box:
        # calculate the size of the lower box
        outer_box_x = cfg.platform_width[0] + (cfg.size[0] - cfg.platform_width[0]) * box_2_ratio
        outer_box_y = cfg.platform_width[1] + (cfg.size[1] - cfg.platform_width[1]) * box_2_ratio
        # create the lower box
        dim = (outer_box_x, outer_box_y, terrain_height + total_height / 2)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], (total_height - terrain_height) / 2 - total_height / 4)
        box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box_mesh)
    
    #! Deprecated
    # # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    #! Add uniform ground mesh
    from omni.isaac.lab.terrains.height_field.hf_terrains import random_uniform_terrain
    uniform_ground_mesh, _ = random_uniform_terrain(0.5, cfg.random_uniform_terrain_cfg)
    meshes_list.append(uniform_ground_mesh[0])

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin


