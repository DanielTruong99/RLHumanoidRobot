from omni.isaac.lab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from dataclasses import MISSING
from typing import Literal

import omni.isaac.lab.terrains.trimesh.mesh_terrains as mesh_terrains
import omni.isaac.lab.terrains.trimesh.utils as mesh_utils_terrains
from omni.isaac.lab.utils import configclass

from . import custom_terrains

@configclass
class HurdleNoiseTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = custom_terrains.hurdle_noise_terrain #type: ignore

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""

    #! Add the x and y dimensions of the box
    platform_width: tuple[float, float] = MISSING
    """The width of the rectangular platform. Defaults to (1.0, 1.0)."""

    #! Add box position respect to the center of the terrain
    box_position: tuple[float, float] = MISSING
    """The position of the rectangular which respect to the center of the terrain. Defaults to (0.0, 0.0)."""

    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""

    #! Add unifrom random terrain cfg
    random_uniform_terrain_cfg: HfRandomUniformTerrainCfg = MISSING #type: ignore

@configclass
class CustomBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = custom_terrains.custom_box_terrain #type: ignore

    box_height_range: tuple[float, float] = MISSING 
    """The minimum and maximum height of the box (in m)."""

    #! Add the x and y dimensions of the box
    platform_width: tuple[float, float] = MISSING 
    """The width of the rectangular platform. Defaults to (1.0, 1.0)."""

    #! Add box position respect to the center of the terrain
    box_position: tuple[float, float] = MISSING 
    """The position of the rectangular which respect to the center of the terrain. Defaults to (0.0, 0.0)."""

    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""

