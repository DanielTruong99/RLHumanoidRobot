
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

def my_bad_orientation(
    env: ManagerBasedRLEnv, limit_gx: float, limit_gy: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the projected gravity x, y component exceeded the limit gx, gy.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    is_exceed_gx = torch.abs(asset.data.projected_gravity_b[:, 0]) > limit_gx
    is_exceed_gy = torch.abs(asset.data.projected_gravity_b[:, 1]) > limit_gy
    return torch.logical_or(is_exceed_gx, is_exceed_gy)

def norm_base_lin_vel_out_of_limit(
    env: ManagerBasedRLEnv, max_norm: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the norm of the base linear velocity exceeded the limit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_b
    return torch.norm(base_lin_vel, dim=1) > max_norm

def norm_base_ang_vel_out_of_limit(
    env: ManagerBasedRLEnv, max_norm: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the norm of the base angular velocity exceeded the limit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_ang_vel = asset.data.root_ang_vel_b
    return torch.norm(base_ang_vel, dim=1) > max_norm

