from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .leg_planar_walk_cfg import LegPlanarWalkEnvCfg

class LegPlanarWalkEnv(DirectRLEnv):
    cfg: LegPlanarWalkEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """DirectRLEnv initialization

        this function initializes the DirectRLEnv performing these inheritable functions:
            #! 1. _setup_scene()
            #! 2. _configure_gym_env_spaces()
        """
        super().__init__(cfg, render_mode, **kwargs)

        #! super.init() has created the self.actions
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions_2 = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        #* vx, vy, wz commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        #* previous applied torque
        self._previous_applied_torque = torch.zeros_like(self._robot.data.applied_torque, device=self.device)

        #* undersired contact body ids
        self._underisred_contact_body_ids, _ = self._contact_sensor.find_bodies([".*_thigh", ".*_calf"])
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_toe")

        #* joint ids
        self._R_hip_joint_index, _ = self._robot.find_joints("R_hip_joint")
        self._R_hip2_joint_index, _ = self._robot.find_joints("R_hip2_joint")
        self._R_thigh_joint_index, _ = self._robot.find_joints("R_thigh_joint")
        self._L_hip_joint_index, _ = self._robot.find_joints("L_hip_joint")
        self._L_hip2_joint_index, _ = self._robot.find_joints("L_hip2_joint")
        self._L_thigh_joint_index, _ = self._robot.find_joints("L_thigh_joint")

        #* Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "base_height_exp",
                "flat_orientation",
                "first_order_action_rate",
                "second_order_action_rate",
                "energy_consumption",
                "joint_regularization",
                "undersired_contacts",
                "is_alive",
                "applied_torque",
                "applied_torque_rate",
                "terminated_penalty"
            ]
        }

    def step(self, action: torch.Tensor):
        #* add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action.clone())
        
        #* pre physics step
        self._pre_physics_step(action)

        #* check if rendering is enabled
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        #* run the simulation
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1 
            self._apply_action() # set processed action to the buffer
            self.scene.write_data_to_sim() # calculate the torque and apply it to the robot
            self.sim.step(render=False) # run the simulation

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
   
            self.scene.update(dt=self.physics_dt) # update simulation data to the internal buffer
 
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        #* check if the episode is terminated or timed out
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        #* calculate rewards
        self.reward_buf = self._get_rewards()

        #* do reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids) # type: ignore

        #* apply events
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        #* calculate next step observations s(t + 1) from the retrieved simulation data scene.update()
        self.obs_buf = self._get_observations()

        #* add observation noise
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"]) #type: ignore

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
        
    def _setup_scene(self):
        """Setup the scene for the environment."""
        #* Add the robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        #* Add the contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        #* Add the height scanner (if available)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        #* Add the terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        #* clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        #* add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self.actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

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
            estimated_height: estimated height from the height scanner (, 1)
            height_data: height data from the height scanner (, 209)
            #//clock_phase: (, 3)
        """
        self._previous_actions_2 = self._previous_actions.clone()
        self._previous_actions = self.actions.clone()
        self._previous_applied_torque = self._robot.data.applied_torque.clone()

        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        )
        estimated_height = torch.mean(self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2], dim=1)

        foot_contact_force = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2] # type: ignore
        foot_contact_state = torch.gt(foot_contact_force, 0.0).float()

        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self.actions,
                foot_contact_state,
                estimated_height,
                height_data
            ),
            dim=-1,
        )

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
            Get the rewards from the environment.

            Returns:
                torch.Tensor: The computed reward tensor.
        """
        (
            lin_vel_error_mapped,
            yaw_rate_error_mapped,
            height_error_mapped,
            flat_orientation_mapped,
            first_order_action_rate,
            second_order_action_rate,
            energy_consumption,
            joint_regularization,
            undesired_contacts,
            is_alive,
            applied_torque,
            applied_torque_rate,
            terminated_penalty
        ) = compute_rewards(
            commands=self._commands,
            root_lin_vel_b=self._robot.data.root_lin_vel_b,
            root_ang_vel_b=self._robot.data.root_ang_vel_b,
            height_scanner_pos_w=self._height_scanner.data.pos_w,
            height_scanner_ray_hits_w=self._height_scanner.data.ray_hits_w,
            base_height_target=self.cfg.base_height_target,
            projected_gravity_b=self._robot.data.projected_gravity_b,
            actions=self.actions,
            previous_actions=self._previous_actions,
            previous_actions_2=self._previous_actions_2,
            step_dt=self.step_dt,
            applied_torque=self._robot.data.applied_torque,
            joint_vel=self._robot.data.joint_vel,
            joint_pos=self._robot.data.joint_pos,
            R_hip_joint_index=self._R_hip_joint_index,
            L_hip_joint_index=self._L_hip_joint_index,
            R_hip2_joint_index=self._R_hip2_joint_index,
            L_hip2_joint_index=self._L_hip2_joint_index,
            R_thigh_joint_index=self._R_thigh_joint_index,
            L_thigh_joint_index=self._L_thigh_joint_index,
            contact_sensor_net_forces_w_history=self._contact_sensor.data.net_forces_w_history,
            underisred_contact_body_ids=self._underisred_contact_body_ids,
            reset_terminated=self.reset_terminated,
            previous_applied_torque=self._previous_applied_torque
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "base_height_exp": height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "flat_orientation": flat_orientation_mapped * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "first_order_action_rate": first_order_action_rate * self.cfg.first_order_action_rate_reward_scale * self.step_dt,
            "second_order_action_rate": second_order_action_rate * self.cfg.second_order_action_rate_reward_scale * self.step_dt,
            "energy_consumption": energy_consumption * self.cfg.energy_consumption_reward_scale * self.step_dt,
            "joint_regularization": joint_regularization * self.cfg.joint_regularization_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contacts_reward_scale * self.step_dt,
            "is_alive": is_alive * self.cfg.is_alive_reward_scale * self.step_dt,
            "applied_torque": applied_torque * self.cfg.applied_torque_reward_scale * self.step_dt,
            "applied_torque_rate": applied_torque_rate * self.cfg.applied_torque_rate_reward_scale * self.step_dt,
            "terminated_penalty": terminated_penalty * self.cfg.terminated_penalty_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        #* Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward
    


@torch.jit.script
def compute_rewards(
    commands: torch.Tensor, 
    root_lin_vel_b: torch.Tensor, 
    root_ang_vel_b: torch.Tensor, 
    height_scanner_pos_w: torch.Tensor, 
    height_scanner_ray_hits_w: torch.Tensor, 
    base_height_target: float,
    projected_gravity_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_actions_2: torch.Tensor,
    step_dt: float,
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    R_hip_joint_index,
    L_hip_joint_index,
    R_hip2_joint_index,
    L_hip2_joint_index,
    R_thigh_joint_index,
    L_thigh_joint_index,
    contact_sensor_net_forces_w_history,
    underisred_contact_body_ids,
    reset_terminated: torch.Tensor,
    previous_applied_torque: torch.Tensor
):
    #* linear velocity TRACKING
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
    
    #* angular velocity TRACKING
    yaw_rate_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

    #* base height
    base_height = torch.mean(height_scanner_pos_w[:, 2].unsqueeze(1) - height_scanner_ray_hits_w[..., 2], dim=1)
    height_error = torch.square(base_height - base_height_target)
    height_error_mapped = torch.exp(-height_error / 0.25)

    #* flat orientation
    flat_orientation = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    flat_orientation_mapped = torch.exp(-flat_orientation / 0.25)

    #* 1st order action rate
    finite_diff_1st_order = (actions - previous_actions) / step_dt
    first_order_action_rate = torch.sum(torch.square(finite_diff_1st_order), dim=1)

    #* 2nd order action rate
    finite_diff_2nd_order = (actions - 2.0 * previous_actions + previous_actions_2) / step_dt
    second_order_action_rate = torch.sum(torch.square(finite_diff_2nd_order), dim=1)

    #* save energy
    energy_consumption = torch.sum(torch.square(applied_torque) * torch.square(joint_vel), dim=1)

    #* joint regularization
    error_R_yaw = torch.square(joint_pos[:, R_hip_joint_index])
    error_L_yaw = torch.square(joint_pos[:, L_hip_joint_index])
    error_hip2 = torch.square(joint_pos[:, R_hip2_joint_index] - joint_pos[:, L_hip2_joint_index])
    error_thigh = torch.square(joint_pos[:, R_thigh_joint_index] + joint_pos[:, L_thigh_joint_index])
    joint_regularization = (torch.exp(-error_R_yaw / 0.25) + torch.exp(-error_L_yaw / 0.25) + 
                            torch.exp(-error_hip2 / 0.25) + torch.exp(-error_thigh / 0.25)) / 4.0

    #* undesired contacts
    is_contact = torch.max(torch.norm(contact_sensor_net_forces_w_history[:, :, underisred_contact_body_ids], dim=-1), dim=1)[0] > 1.0
    undesired_contacts = torch.sum(is_contact, dim=1)

    #* is alive
    is_alive = (~reset_terminated).float()

    #* applied torque
    applied_torque = torch.sum(torch.square(applied_torque), dim=1)

    #* applied torque rate
    applied_torque_rate = torch.sum(torch.square(applied_torque - previous_applied_torque), dim=1)

    #* terminated penalty
    terminated_penalty = reset_terminated.float()

    return (
        lin_vel_error_mapped,
        yaw_rate_error_mapped,
        height_error_mapped,
        flat_orientation_mapped,
        first_order_action_rate,
        second_order_action_rate,
        energy_consumption,
        joint_regularization,
        undesired_contacts,
        is_alive,
        applied_torque,
        applied_torque_rate,
        terminated_penalty
    )

    
