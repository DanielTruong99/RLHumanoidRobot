from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

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

