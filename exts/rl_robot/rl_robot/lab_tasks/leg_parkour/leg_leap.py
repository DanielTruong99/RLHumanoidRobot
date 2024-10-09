from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat, normalize

from .leg_planar_walk import LegPlanarWalkEnv 

from .leg_leap_cfg import LegLeapEnvCfg

class LegLeapEnv(LegPlanarWalkEnv):
    cfg: LegLeapEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #* adjust the joint limit of the thigh joint [-1.7, 0] -> [-1.7, 0.78]
        thigh_joint_ids = self._robot.find_joints(".*_thigh_joint")[0]
        limits = torch.zeros_like(self._robot.data.joint_limits[:, thigh_joint_ids, :])
        limits[:, :, 0] = -1.7; limits[:, :, 1] = 0.78
        self._robot.write_joint_limits_to_sim(limits, thigh_joint_ids)

        #* set terrain level into 0
        self._terrain.terrain_levels = torch.zeros_like(self._terrain.terrain_levels)
        self._terrain.env_origins = self._terrain.terrain_origins[self._terrain.terrain_levels, self._terrain.terrain_types]

    def _init_buffers(self):
        super()._init_buffers()
        
        #* position command in world frame and body frame
        self._pos_command_w = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self._heading_command_w = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        #* add log
        self._episode_sums["feet_stumble"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._episode_sums["position_tracking"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._episode_sums["heading_tracking"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._episode_sums["joint_velocity_penalty"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        #* log terrain levels
        self.extras["log"].update({
            "Curiculum/terrain_levels": torch.mean(self._terrain.terrain_levels.float()),
        })

    def _get_rewards(self):
        baseline_rewards = super()._get_rewards()

        #* feet stumble reward
        feet_stumble = torch.norm(
            self._contact_sensor.data.net_forces_w[:, self._feet_ids, 0:2], dim=-1 # type: ignore
        ) > 2.0 * torch.norm(
            self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2:], dim=-1 # type: ignore
        )  # type: ignore
        feet_stumble = torch.any(feet_stumble, dim=-1).float()

        #* positive tracking reward
        p_goal = quat_rotate_inverse(yaw_quat(self._robot.data.root_quat_w), self._pos_command_w - self._robot.data.root_pos_w[:, :3])
        p_goal = p_goal[:, :2]
        is_p_goal_smaller_than_threshold = torch.norm(p_goal, dim=-1) < 0.15
        position_tracking_reward = ~is_p_goal_smaller_than_threshold * torch.sum(self._robot.data.root_lin_vel_b[:, :2] * normalize(p_goal), dim=-1) 
        position_tracking_reward += is_p_goal_smaller_than_threshold * 1.5

        #* heading reward
        heading_error = self._heading_command_w - self._robot.data.heading_w
        heading_tracking_reward = torch.exp(-10.0 * torch.square(heading_error)) * torch.exp(-4.0 * torch.sum(torch.square(p_goal), dim=-1))

        #* joint velocity penalty
        joint_velocity_penalty = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

        rewards = {
            "baseline_rewards": baseline_rewards,
            "feet_stumble": feet_stumble * self.cfg.feet_stumble_reward_scale * self.step_dt,
            "position_tracking": position_tracking_reward * self.cfg.position_tracking_reward_scale * self.step_dt,
            "heading_tracking": heading_tracking_reward * self.cfg.heading_tracking_reward_scale * self.step_dt,
            "joint_velocity_penalty": joint_velocity_penalty * self.cfg.joint_velocity_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        #* Logging
        for key, value in rewards.items():
            if "baseline_rewards" in key: continue
            self._episode_sums[key] += value
        
        return reward

    def _compute_curriculum(self, env_ids: torch.Tensor):
        """Compute the curriculum for the given environment ids."""
        #* compute the position and heading error
        position_error = torch.norm(self._robot.data.root_pos_w[env_ids, :2] - self._pos_command_w[env_ids, :2], dim=1)
        heading_error = torch.abs(wrap_to_pi(self._robot.data.heading_w[env_ids] - self._heading_command_w[env_ids]))
        
        #* if the robot reach the target earlier than the time limit, move up the terrain
        #* position_error < 1e-2 
        #* heading_error < 10 degree
        #* time_left > 0.0
        move_up = (position_error < 1e-2) & (heading_error < (10.0*3.14/180.0)) & (self._time_left[env_ids] > 0.0)
        
        #* update terrain levels
        self._terrain.update_env_origins(env_ids, move_up, ~move_up)

    def _command_compute(self, dt: float):
        self._time_left -= dt
        resample_env_ids = (self._time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._time_left[resample_env_ids] = self._time_left[resample_env_ids].uniform_(*self.cfg.commands.resampling_time_range)
            self._resample(resample_env_ids)
        self._post_process_commands()

    def _resample_cmds(self, env_ids: torch.Tensor):
        """resample the commands for the given environment ids
        """
        #* randomize the position command around the environment origin
        self._pos_command_w[env_ids] = self.scene.env_origins[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        self._pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.commands.ranges_pos_x)
        self._pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.commands.ranges_pos_y)
        self._heading_command_w[env_ids] = r.uniform_(*self.cfg.commands.ranges_heading)


    def _post_process_commands(self):
        """Re-target the position command to the current root state.
            _commands: [x, y, heading] in body frame
        """
        target_vec = self._pos_command_w - self._robot.data.root_pos_w[:, :3]
        self._commands[:, :2] = quat_rotate_inverse(yaw_quat(self._robot.data.root_quat_w), target_vec)[:,:2]
        self._commands[:, 2] = wrap_to_pi(self._heading_command_w - self._robot.data.heading_w)
 








    
