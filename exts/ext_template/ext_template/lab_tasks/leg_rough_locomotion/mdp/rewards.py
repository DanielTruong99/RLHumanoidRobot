from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from ..leg_robot import LegRobotEnv

def action_rate_1st_order(env: LegRobotEnv) -> torch.Tensor:
    qk = env.action_manager.action
    qk_1 = env.action_manager.prev_action
    finite_diff = (qk - qk_1) / env.step_dt
    return torch.sum(torch.square(finite_diff), dim=1)

def action_rate_2nd_order(env: LegRobotEnv) -> torch.Tensor:
    qk = env.action_manager.action
    qk_1 = env.action_manager.prev_action
    qk_2 = env.action_manager.prev2_action
    finite_diff = (qk - 2.0 * qk_1 + qk_2) / env.step_dt
    return torch.sum(torch.square(finite_diff), dim=1)

def joint_regulization_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    R_hip_joint_index = asset.find_joints("R_hip_joint")[0][0]
    R_hip2_joint_index = asset.find_joints("R_hip2_joint")[0][0]
    R_thigh_joint_index = asset.find_joints("R_thigh_joint")[0][0]
    L_hip_joint_index = asset.find_joints("L_hip_joint")[0][0]
    L_hip2_joint_index = asset.find_joints("L_hip2_joint")[0][0]
    L_thigh_joint_index = asset.find_joints("L_thigh_joint")[0][0]

    error_R_yaw = torch.square(joint_pos[:, R_hip_joint_index])
    error_L_yaw = torch.square(joint_pos[:, L_hip_joint_index])
    error_hip2 = torch.square(joint_pos[:, R_hip2_joint_index] - joint_pos[:, L_hip2_joint_index])
    error_thigh = torch.square(joint_pos[:, R_thigh_joint_index] + joint_pos[:, L_thigh_joint_index])

    rewards = torch.exp(-error_R_yaw/std**2) + torch.exp(-error_L_yaw/std**2) + torch.exp(-error_hip2/std**2) + torch.exp(-error_thigh/std**2)
    rewards = rewards / 4.0
    return rewards

def pb_joint_regulization_exp(env: LegRobotEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    delta_phi = ~env.reset_buf * (joint_regulization_exp(env, std, asset_cfg) - env.rwd_jointRegPrev)    
    return delta_phi / env.step_dt

def base_height_exp(
    env: ManagerBasedRLEnv, target_height: float, std: float,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize asset height from its target using exponential-kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO: Fix this for rough-terrain.
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-error/std**2)

def pb_base_height_exp(env: LegRobotEnv, target_height: float, std: float,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    delta_phi = ~env.reset_buf * (base_height_exp(env, target_height, std, asset_cfg) - env.rwd_baseHeightPrev)    
    return delta_phi / env.step_dt

def flat_orientation_exp(env: ManagerBasedRLEnv, std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ) -> torch.Tensor:
    """Penalize non-flat base orientation using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-error/std**2)

def pb_flat_orientation_exp(env: LegRobotEnv, std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ) -> torch.Tensor:
    delta_phi = ~env.reset_buf * (flat_orientation_exp(env, std, asset_cfg) - env.rwd_oriPrev)    
    return delta_phi / env.step_dt


