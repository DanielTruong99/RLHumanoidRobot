
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm
    from ..leg_robot import LegRobotEnv

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
