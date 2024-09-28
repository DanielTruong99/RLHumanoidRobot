from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster

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

        #* add heading target to the command
        self._heading_target = torch.zeros(self.num_envs, device=self.device)

        #* set terrain level into 0
        self._terrain.terrain_levels = torch.zeros_like(self._terrain.terrain_levels)
        self._terrain.env_origins = self._terrain.terrain_origins[self._terrain.terrain_levels, self._terrain.terrain_types]

    def _compute_curriculum(self, env_ids: torch.Tensor):
        """Compute the curriculum for the given environment ids."""
        #* compute the distance the robot walked
        distance = torch.norm(self._robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        
        #* robots that walked far enough progress to harder terrains
        move_up = distance > 1.7
        
        #* robots that walked less than half of their required distance go to simpler terrains
        move_down = distance < torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down *= ~move_up
        
        #* update terrain levels
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _resample_cmds(self, env_ids: torch.Tensor):
        """resample the commands for the given environment ids
            
            #! the modification is sample commands without heading velocity
            #! the heading velocity will be adjusted in the post process based on the heading error
        """
        r = torch.empty(len(env_ids), device=self.device)

        #* linear velocity - x direction
        self._commands[env_ids, 0] = r.uniform_(*self.cfg.commands.ranges_lin_vel_x)

        #* linear velocity - y direction
        self._commands[env_ids, 1] = r.uniform_(*self.cfg.commands.ranges_lin_vel_y)


    def _post_process_commands(self):
        """adjust the heading velocity command
        """
        if self.cfg.commands.heading_control_stiffness:
            #* adjust the heading velocity command
            heading_error = math_utils.wrap_to_pi(self._heading_target - self._robot.data.heading_w)
            self._commands[:, 2] = torch.clamp(
                heading_error * self.cfg.commands.heading_control_stiffness,
                -self.cfg.commands.ranges_ang_vel_z[1],
                self.cfg.commands.ranges_ang_vel_z[1],
            )

            #* adjust the velocity command based on the heading error
            large_heading_error_ids = torch.abs(heading_error) > 3.14 / 2
            self._commands[large_heading_error_ids, 0] = 0.0
            self._commands[large_heading_error_ids, 1] = 0.0








    
