from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .simple_walking_robot_cfg import SimpleWalkingRobotEnvCfg
from ..leg_parkour.leg_planar_walk import LegPlanarWalkEnv

class SimpleWalkingRobot(LegPlanarWalkEnv):
    cfg: SimpleWalkingRobotEnvCfg

    def _init_buffers(self):
        super()._init_buffers()

        #* Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for key in [
                "lin_vel",
                "yaw_rate",
                "base_height_exp",
                "flat_orientation",
                "first_order_action_rate",
                "second_order_action_rate",
                "joint_regularization",
                "undesired_contacts",
                "applied_torque",
                "terminated_penalty",
                "feet_stumble",
                "joint_velocity_penalty",
                "joint_acc_penalty",
            ]
        }

    def _setup_scene(self):
        """Setup the scene for the environment."""
        #* Add the robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        #* Add the contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        #* Add the terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene._terrain = self._terrain

        #* clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        #* add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """Get the observations from the environment.
            v: base linear velocity (, 3)
            w: base angular velocity (, 3)
            g: projected gravity (, 3)
            c: commands (, 3)
            p: joint positions (, 10)
            p_dot: joint velocities (, 10)
            a: last actions (, 10)
            foot_contact_state: binary foot contact state (, 2)
            height_data: height data from the height scanner (, 1)
        """
        self._previous_actions_2 = self._previous_actions
        self._previous_actions = self.actions.clone()
        self._previous_applied_torque = self._robot.data.applied_torque.clone()

        height_data = self._robot.data.root_pos_w[:, 2].unsqueeze(-1)

        foot_contact_force = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2] # type: ignore
        foot_contact_state = torch.gt(foot_contact_force, 0.0).float()

        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos,
                self._robot.data.joint_vel,
                self.actions,
                foot_contact_state,
                height_data
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self):
        (
            lin_vel_error_mapped,
            yaw_rate_error_mapped,
            height_error_mapped,
            flat_orientation_mapped,
            first_order_action_rate,
            second_order_action_rate,
            joint_regularization,
            undesired_contacts,
            applied_torque_penalty,
            terminated_penalty,
            feet_stumble,
            joint_velocity_penalty,
            joint_acc_penalty,
        ) = compute_rewards(
            root_ang_vel_b=self._robot.data.root_ang_vel_b,
            commands=self._commands,
            root_lin_vel_b=self._robot.data.root_lin_vel_b,
            base_height=self._robot.data.root_pos_w[:, 2],
            base_height_target=self.cfg.base_height_target,
            projected_gravity_b=self._robot.data.projected_gravity_b,
            actions=self.actions,
            previous_actions=self._previous_actions,
            previous_actions_2=self._previous_actions_2,
            applied_torque=self._robot.data.applied_torque,
            joint_vel=self._robot.data.joint_vel,
            joint_acc=self._robot.data.joint_acc,
            joint_pos=self._robot.data.joint_pos,
            R_hip_joint_index=self._R_hip_joint_index,
            L_hip_joint_index=self._L_hip_joint_index,
            R_hip2_joint_index= self._R_hip2_joint_index,
            L_hip2_joint_index= self._L_hip2_joint_index,
            R_thigh_joint_index=self._R_thigh_joint_index,
            L_thigh_joint_index=self._L_thigh_joint_index,
            contact_sensor_net_forces_w_history=self._contact_sensor.data.net_forces_w_history,
            contact_sensor_net_forces_w=self._contact_sensor.data.net_forces_w,
            underisred_contact_body_ids=self._underisred_contact_body_ids,
            feet_ids=self._feet_ids,
            reset_terminated=self.reset_terminated,
        )

        rewards = {
            "lin_vel": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "yaw_rate": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "base_height_exp": height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "flat_orientation": flat_orientation_mapped * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "first_order_action_rate": first_order_action_rate * self.cfg.first_order_action_rate_reward_scale * self.step_dt,
            "second_order_action_rate": second_order_action_rate * self.cfg.second_order_action_rate_reward_scale * self.step_dt,
            "joint_regularization": joint_regularization * self.cfg.joint_regularization_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contacts_reward_scale * self.step_dt,
            "applied_torque": applied_torque_penalty * self.cfg.applied_torque_reward_scale * self.step_dt,
            "terminated_penalty": terminated_penalty * self.cfg.terminated_penalty_reward_scale * self.step_dt,
            "feet_stumble": feet_stumble * self.cfg.feet_stumble_reward_scale * self.step_dt,
            "joint_velocity_penalty": joint_velocity_penalty * self.cfg.joint_velocity_reward_scale * self.step_dt,
            "joint_acc_penalty": joint_acc_penalty * self.cfg.joint_acc_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        #* Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward
    
@torch.jit.script
def compute_rewards(
    root_lin_vel_b: torch.Tensor, 
    root_ang_vel_b: torch.Tensor,
    commands: torch.Tensor,
    base_height: torch.Tensor,
    base_height_target: float,
    projected_gravity_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_actions_2: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_pos: torch.Tensor,
    R_hip_joint_index: list[int],
    L_hip_joint_index: list[int],
    R_hip2_joint_index: list[int],
    L_hip2_joint_index: list[int],
    R_thigh_joint_index: list[int],
    L_thigh_joint_index: list[int],
    contact_sensor_net_forces_w_history,
    contact_sensor_net_forces_w,
    underisred_contact_body_ids: list[int],
    feet_ids: list[int],
    reset_terminated: torch.Tensor,
):
    # #* linear velocity tracking
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
    
    # #* angular velocity tracking
    yaw_rate_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

    #* base height
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
    error_thigh = torch.square(joint_pos[:, R_thigh_joint_index[0]] + joint_pos[:, L_thigh_joint_index[0]])
    joint_regularization = (torch.exp(-error_R_yaw / 0.25) + torch.exp(-error_L_yaw / 0.25) + torch.exp(-error_hip2 / 0.25) + torch.exp(-error_thigh / 0.25))/ 4.0
                          

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

    #* joint velocity, acceleration penalty
    joint_velocity_penalty = torch.sum(torch.square(joint_vel), dim=1)    
    joint_acc_penalty = torch.sum(torch.square(joint_acc), dim=1)


    return (
        lin_vel_error_mapped,
        yaw_rate_error_mapped,
        height_error_mapped,
        flat_orientation_mapped,
        first_order_action_rate,
        second_order_action_rate,
        joint_regularization,
        undesired_contacts,
        applied_torque_penalty,
        terminated_penalty,
        feet_stumble,
        joint_velocity_penalty,
        joint_acc_penalty,
    )