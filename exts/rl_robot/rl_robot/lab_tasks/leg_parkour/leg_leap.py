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
        self._episode_sums["joint_acc_penalty"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._episode_sums["stand_still_penalty"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._episode_sums["stand_still_collision_penalty"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        #* log terrain levels
        self.extras["log"].update({
            "Curiculum/terrain_levels": torch.mean(self._terrain.terrain_levels.float()),
        })

    def _get_rewards(self):
        (
            height_error_mapped,
            flat_orientation_mapped,
            first_order_action_rate,
            second_order_action_rate,
            joint_regularization,
            undesired_contacts,
            applied_torque_penalty,
            terminated_penalty,
            feet_stumble,
            position_tracking_reward,
            heading_tracking_reward,
            joint_velocity_penalty,
            joint_acc_penalty,
            stand_still_penalty,
            stand_still_collision_penalty
        ) = compute_rewards(
            pos_command_w=self._pos_command_w,
            heading_command_w=self._heading_command_w,
            heading_w=self._robot.data.heading_w,
            root_lin_vel_b=self._robot.data.root_lin_vel_b,
            root_pos_w=self._robot.data.root_pos_w,
            root_quat_w=self._robot.data.root_quat_w,
            height_scanner_pos_w=self._height_scanner.data.pos_w,
            height_scanner_ray_hits_w=self._height_scanner.data.ray_hits_w,
            base_height_target=self.cfg.base_height_target,
            projected_gravity_b=self._robot.data.projected_gravity_b,
            actions=self.actions,
            previous_actions=self._previous_actions,
            previous_actions_2=self._previous_actions_2,
            applied_torque=self._robot.data.applied_torque,
            joint_vel=self._robot.data.joint_vel,
            joint_acc=self._robot.data.joint_acc,
            joint_pos=self._robot.data.joint_pos,
            default_joint_pos=self._robot.data.default_joint_pos,
            R_hip_joint_index=self._R_hip_joint_index,
            L_hip_joint_index=self._L_hip_joint_index,
            R_hip2_joint_index= self._R_hip2_joint_index,
            L_hip2_joint_index= self._L_hip2_joint_index,
            contact_sensor_net_forces_w_history=self._contact_sensor.data.net_forces_w_history,
            contact_sensor_net_forces_w=self._contact_sensor.data.net_forces_w,
            underisred_contact_body_ids=self._underisred_contact_body_ids,
            feet_ids=self._feet_ids,
            reset_terminated=self.reset_terminated,
        )

        rewards = {
            "base_height_exp": height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "flat_orientation": flat_orientation_mapped * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "first_order_action_rate": first_order_action_rate * self.cfg.first_order_action_rate_reward_scale * self.step_dt,
            "second_order_action_rate": second_order_action_rate * self.cfg.second_order_action_rate_reward_scale * self.step_dt,
            "joint_regularization": joint_regularization * self.cfg.joint_regularization_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contacts_reward_scale * self.step_dt,
            "applied_torque": applied_torque_penalty * self.cfg.applied_torque_reward_scale * self.step_dt,
            "terminated_penalty": terminated_penalty * self.cfg.terminated_penalty_reward_scale * self.step_dt,
            "feet_stumble": feet_stumble * self.cfg.feet_stumble_reward_scale * self.step_dt,
            "position_tracking": position_tracking_reward * self.cfg.position_tracking_reward_scale * self.step_dt,
            "heading_tracking": heading_tracking_reward * self.cfg.heading_tracking_reward_scale * self.step_dt,
            "joint_velocity_penalty": joint_velocity_penalty * self.cfg.joint_velocity_reward_scale * self.step_dt,
            "joint_acc_penalty": joint_acc_penalty * self.cfg.joint_acc_reward_scale * self.step_dt,
            "stand_still_penalty": stand_still_penalty * self.cfg.stand_still_reward_scale * self.step_dt,
            "stand_still_collision_penalty": stand_still_collision_penalty * self.cfg.stand_still_collision_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        #* Logging
        for key, value in rewards.items():
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
        move_up = (position_error < 12.0/100.0) & (heading_error < (10.0*3.14/180.0)) & (self._time_left[env_ids] > 0.0)
        
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

@torch.jit.script
def compute_rewards(
    pos_command_w: torch.Tensor,
    heading_command_w: torch.Tensor,
    heading_w: torch.Tensor,
    root_lin_vel_b: torch.Tensor, 
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    height_scanner_pos_w: torch.Tensor, 
    height_scanner_ray_hits_w: torch.Tensor, 
    base_height_target: float,
    projected_gravity_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_actions_2: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    R_hip_joint_index: list[int],
    L_hip_joint_index: list[int],
    R_hip2_joint_index: list[int],
    L_hip2_joint_index: list[int],
    contact_sensor_net_forces_w_history,
    contact_sensor_net_forces_w,
    underisred_contact_body_ids: list[int],
    feet_ids: list[int],
    reset_terminated: torch.Tensor,
):
    #* base height
    base_height = torch.mean(height_scanner_pos_w[:, 2].unsqueeze(1) - height_scanner_ray_hits_w[..., 2], dim=1)
    height_error = torch.square(base_height - base_height_target)
    height_error_mapped = torch.exp(-height_error / 0.25)

    #* flat orientation
    flat_orientation = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    flat_orientation_mapped = torch.exp(-flat_orientation / 0.25)

    #* 1st order action rate
    finite_diff_1st_order = (actions - previous_actions)
    first_order_action_rate = torch.sum(torch.square(finite_diff_1st_order), dim=1)

    #* 2nd order action rate
    finite_diff_2nd_order = (actions - 2.0 * previous_actions + previous_actions_2)
    second_order_action_rate = torch.sum(torch.square(finite_diff_2nd_order), dim=1)

    #* joint regularization
    error_R_yaw = torch.square(joint_pos[:, R_hip_joint_index[0]])
    error_L_yaw = torch.square(joint_pos[:, L_hip_joint_index[0]])
    error_hip2 = torch.square(joint_pos[:, R_hip2_joint_index[0]] - joint_pos[:, L_hip2_joint_index[0]])
    # error_thigh = torch.square(joint_pos[:, R_thigh_joint_index[0]] + joint_pos[:, L_thigh_joint_index[0]])
    joint_regularization = (torch.exp(-error_R_yaw / 0.25) + torch.exp(-error_L_yaw / 0.25) + torch.exp(-error_hip2 / 0.25))/ 3.0
                          

    #* undesired contacts
    is_contact = torch.max(torch.norm(contact_sensor_net_forces_w_history[:, :, underisred_contact_body_ids], dim=-1), dim=1)[0] > 1.0
    undesired_contacts = torch.sum(is_contact, dim=1)

    #* applied torque
    applied_torque_penalty = torch.sum(torch.square(applied_torque), dim=1)

    #* terminated penalty
    terminated_penalty = reset_terminated.float()

    #* feet stumble reward
    feet_stumble = torch.norm(
        contact_sensor_net_forces_w[:, feet_ids, 0:2], dim=-1 # type: ignore
    ) > 2.0 * torch.norm(
        contact_sensor_net_forces_w[:, feet_ids, 2:], dim=-1 # type: ignore
    )  # type: ignore
    feet_stumble = torch.any(feet_stumble, dim=-1).float()

    #* positive tracking reward
    p_goal = quat_rotate_inverse(root_quat_w, pos_command_w - root_pos_w[:, :3])
    p_goal = p_goal[:, :2]
    is_p_goal_smaller_than_threshold = torch.norm(p_goal, dim=-1) < 0.15
    position_tracking_reward = ~is_p_goal_smaller_than_threshold * torch.sum(root_lin_vel_b[:, :2] * normalize(p_goal), dim=-1) 
    position_tracking_reward += is_p_goal_smaller_than_threshold * 1.5

    #* heading reward
    heading_error = heading_command_w - heading_w
    heading_tracking_reward = torch.exp(-10.0 * torch.square(heading_error)) * torch.exp(-7.0 * torch.sum(torch.square(p_goal), dim=-1))

    #* joint velocity, acceleration penalty
    joint_velocity_penalty = torch.sum(torch.square(joint_vel), dim=1)    
    joint_acc_penalty = torch.sum(torch.square(joint_acc), dim=1)

    #* stand still - force the robot move closer to the target
    stand_still_penalty = is_p_goal_smaller_than_threshold * torch.sum(torch.abs(actions - default_joint_pos), dim=1)

    #* stand still collision - force the robot put 2 feet on the ground
    is_feet_contacts = torch.max(torch.norm(contact_sensor_net_forces_w_history[:, :, feet_ids], dim=-1), dim=1)[0] > 1.0
    feet_contacts = torch.sum(~is_feet_contacts, dim=1)
    stand_still_collision_penalty = is_p_goal_smaller_than_threshold * feet_contacts

    return (
        height_error_mapped,
        flat_orientation_mapped,
        first_order_action_rate,
        second_order_action_rate,
        joint_regularization,
        undesired_contacts,
        applied_torque_penalty,
        terminated_penalty,
        feet_stumble,
        position_tracking_reward,
        heading_tracking_reward,
        joint_velocity_penalty,
        joint_acc_penalty,
        stand_still_penalty,
        stand_still_collision_penalty
    )
 








    
