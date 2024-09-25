from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .leg_planar_walk import LegPlanarWalkEnv 

from .leg_leap_cfg import LegLeapEnvCfg

class LegLeapEnv(LegPlanarWalkEnv):
    cfg: LegLeapEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    
