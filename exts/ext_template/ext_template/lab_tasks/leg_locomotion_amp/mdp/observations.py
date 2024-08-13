
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils import math

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm
    from ...leg_locomotion_parkour.leg_robot import LegRobotEnv

def binary_foot_contact_state(
    env: ManagerBasedRLEnv, contact_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the binary foot contact state.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name] # type: ignore
    foot_contact_force = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, 2] # type: ignore
    in_contact = torch.gt(foot_contact_force, 0.0).float()
    return in_contact

def foot_positions_in_base_frame(
    env: LegRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the foot positions in the base frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name] # type: ignore
    root_postion = asset.data.root_pos_w
    right_foot_pos = asset.data.body_pos_w[:, asset.find_bodies("R_toe")[0][0], :] 
    left_foot_pos = asset.data.body_pos_w[:, asset.find_bodies("L_toe")[0][0], :]
    right_foot_pos_from_base = right_foot_pos - root_postion; left_foot_pos_from_base = left_foot_pos - root_postion

    base_frame_right_foot_pos_from_base = math.quat_rotate_inverse(asset.data.root_quat_w, right_foot_pos_from_base)
    base_frame_left_foot_pos_from_base = math.quat_rotate_inverse(asset.data.root_quat_w, left_foot_pos_from_base)

    return torch.cat((base_frame_right_foot_pos_from_base, base_frame_left_foot_pos_from_base), dim=-1)

def clock_phase(env: LegRobotEnv) -> torch.Tensor:
    """Return the clock phase.
    """
    p = 2.0 * torch.pi * env.phase * env.phase_freq
    smooth_wave = torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + env.eps**2.)) + 1./2.

    clock_phase = torch.concat((
        smooth_wave,
        torch.sin(2.0 * torch.pi * env.phase),
        torch.cos(2.0 * torch.pi * env.phase),
    ),dim=-1)
    return clock_phase
