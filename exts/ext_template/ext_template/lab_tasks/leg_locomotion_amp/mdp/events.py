from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.envs import ManagerBasedEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm
    from ..leg_robot import LegRobotEnv

def reset_phase(
    env: LegRobotEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the phase of the clock."""

    env.phase[env_ids, 0] = torch.rand(
        (torch.numel(env_ids),), requires_grad=False, device=env.device
    )
 
