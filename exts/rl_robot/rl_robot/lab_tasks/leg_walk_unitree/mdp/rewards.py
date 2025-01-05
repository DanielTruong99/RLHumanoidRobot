import torch

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.managers import SceneEntityCfg

from ..walking_robot import WalkingRobotEnv

def feet_schedule_contact(env: WalkingRobotEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    result = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history

    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0 # type: ignore
    result = result + ~(is_contact[:, 0] ^ (env.phase_left < 0.55)) + ~(is_contact[:, 1] ^ (env.phase_right < 0.55))
    return result

def feet_height(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0 # type: ignore
    asset = env.scene[asset_cfg.name]

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    base_pos_w = asset.data.root_pos_w
    feet_height = feet_pos_w - base_pos_w.unsqueeze(1) 

    result = ~is_contact * torch.square(feet_height[:, :, 2] - (-0.7405))
    return torch.sum(result, dim=1)
