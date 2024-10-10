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
        self._init_buffers()


    def _init_buffers(self):
        #! super.init() has created the self.actions
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device, requires_grad=False)
        self._previous_actions_2 = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device, requires_grad=False)

        #* minimal velocity command manager
        self._time_left = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        #* vx, vy, wz commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        #* previous applied torque
        self._previous_applied_torque = torch.zeros_like(self._robot.data.applied_torque, device=self.device, requires_grad=False)

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
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for key in [
                # "track_lin_vel_xy_exp",
                # "track_ang_vel_z_exp",
                "base_height_exp",
                "flat_orientation",
                "first_order_action_rate",
                "second_order_action_rate",
                # "energy_consumption",
                "joint_regularization",
                "undesired_contacts",
                # "is_alive",
                "applied_torque",
                # "applied_torque_rate",
                "terminated_penalty",
                # "joint_pos_limit"
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

        #* resample the command if necessary
        if self.cfg.commands:
            self._command_compute(self.step_dt)
        
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
    
    def _command_compute(self, dt: float):
        self._time_left -= dt
        resample_env_ids = (self._time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._time_left[resample_env_ids] = self._time_left[resample_env_ids].uniform_(*self.cfg.commands.resampling_time_range)
            self._resample(resample_env_ids)
        self._post_process_commands()
        
    def _post_process_commands(self):
        pass

    def _resample(self, env_ids):
        """Resample the commands if the time left is less than 0."""
        if len(env_ids) != 0:
            self._time_left[env_ids] = self._time_left[env_ids].uniform_(*self.cfg.commands.resampling_time_range)
            self._resample_cmds(env_ids)

    def _resample_cmds(self, env_ids: torch.Tensor):
        r = torch.empty(len(env_ids), device=self.device)

        #* linear velocity - x direction
        self._commands[env_ids, 0] = r.uniform_(*self.cfg.commands.ranges_lin_vel_x)

        #* linear velocity - y direction
        self._commands[env_ids, 1] = r.uniform_(*self.cfg.commands.ranges_lin_vel_y)

        #* ang vel yaw - rotation around z
        self._commands[env_ids, 2] = r.uniform_(*self.cfg.commands.ranges_ang_vel_z)

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
        self.scene._terrain = self._terrain

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
            height_data: height data from the height scanner (, 209)
        """
        self._previous_actions_2 = self._previous_actions
        self._previous_actions = self.actions.clone()
        self._previous_applied_torque = self._robot.data.applied_torque.clone()

        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2]
        ).clip(-3.0, 3.0)

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

    def _compute_curriculum(self, env_ids: torch.Tensor):
        """Compute the curriculum for the given environment ids."""
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES  # type: ignore

        #* reset the robot
        self._robot.reset(env_ids) #type: ignore

        #* compute the curriculum if enabled
        if self.cfg.curriculum:
            self._compute_curriculum(env_ids)

        #* reset the scene and noise model
        #* event manager also trigger "reset" event
        super()._reset_idx(env_ids) #type: ignore
        if self.cfg.events:
            self.event_manager.reset(env_ids) #type: ignore

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        #* reset user buffers
        self._previous_actions[env_ids] = 0.0
        self._previous_actions_2[env_ids] = 0.0
        self._previous_applied_torque[env_ids] = 0.0

        #* sample new commands
        if self.cfg.commands:
            self._resample(env_ids)
    
        #* Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        #* check if the base is in contact with the ground
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1) # type: ignore

        # #* check if the base's pitch and roll eceeded the limits
        # is_exceed_gx = torch.abs(self._robot.data.projected_gravity_b[:, 0]) > 0.85
        # is_exceed_gy = torch.abs(self._robot.data.projected_gravity_b[:, 1]) > 0.85
        # # died |=  is_exceed_gx; died |= is_exceed_gy

        # #* check if the base's linear velocity exceeded the limit
        # is_exceed_v = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1) > 11.0
        # # died |= is_exceed_v

        # #* check if the base's angular velocity exceeded the limit
        # is_exceed_w = torch.norm(self._robot.data.root_ang_vel_b, dim=1) > 7.0
        # # died |= is_exceed_w

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        """
            Get the rewards from the environment.

            Returns:
                torch.Tensor: The computed reward tensor.
        """
        (
            # lin_vel_error_mapped,
            # yaw_rate_error_mapped,
            height_error_mapped,
            flat_orientation_mapped,
            first_order_action_rate,
            second_order_action_rate,
            # energy_consumption,
            joint_regularization,
            undesired_contacts,
            # is_alive,
            applied_torque,
            # applied_torque_rate,
            terminated_penalty,
            # joint_pos_limit
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
            soft_joint_pos_limits=self._robot.data.soft_joint_pos_limits,
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
            # "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "base_height_exp": height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "flat_orientation": flat_orientation_mapped * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "first_order_action_rate": first_order_action_rate * self.cfg.first_order_action_rate_reward_scale * self.step_dt,
            "second_order_action_rate": second_order_action_rate * self.cfg.second_order_action_rate_reward_scale * self.step_dt,
            # "energy_consumption": energy_consumption * self.cfg.energy_consumption_reward_scale * self.step_dt,
            "joint_regularization": joint_regularization * self.cfg.joint_regularization_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contacts_reward_scale * self.step_dt,
            # "is_alive": is_alive * self.cfg.is_alive_reward_scale * self.step_dt,
            "applied_torque": applied_torque * self.cfg.applied_torque_reward_scale * self.step_dt,
            # "applied_torque_rate": applied_torque_rate * self.cfg.applied_torque_rate_reward_scale * self.step_dt,
            "terminated_penalty": terminated_penalty * self.cfg.terminated_penalty_reward_scale * self.step_dt,
            # "joint_pos_limit": joint_pos_limit * self.cfg.joint_pos_limit_reward_scale * self.step_dt
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
    soft_joint_pos_limits: torch.Tensor,
    R_hip_joint_index: list[int],
    L_hip_joint_index: list[int],
    R_hip2_joint_index: list[int],
    L_hip2_joint_index: list[int],
    R_thigh_joint_index: list[int],
    L_thigh_joint_index: list[int],
    contact_sensor_net_forces_w_history,
    underisred_contact_body_ids: list[int],
    reset_terminated: torch.Tensor,
    previous_applied_torque: torch.Tensor
):
    # #* linear velocity tracking
    # lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
    
    # #* angular velocity tracking
    # yaw_rate_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

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

    #* save energy
    # energy_consumption = torch.sum(torch.square(applied_torque * joint_vel), dim=1)

    #* joint regularization
    error_R_yaw = torch.square(joint_pos[:, R_hip_joint_index[0]])
    error_L_yaw = torch.square(joint_pos[:, L_hip_joint_index[0]])
    # error_hip2 = torch.square(joint_pos[:, R_hip2_joint_index[0]] - joint_pos[:, L_hip2_joint_index[0]])
    # error_thigh = torch.square(joint_pos[:, R_thigh_joint_index[0]] + joint_pos[:, L_thigh_joint_index[0]])
    joint_regularization = (torch.exp(-error_R_yaw / 0.25) + torch.exp(-error_L_yaw / 0.25)) / 2.0
                          

    #* undesired contacts
    is_contact = torch.max(torch.norm(contact_sensor_net_forces_w_history[:, :, underisred_contact_body_ids], dim=-1), dim=1)[0] > 1.0
    undesired_contacts = torch.sum(is_contact, dim=1)

    # #* is alive
    # is_alive = (~reset_terminated).float()

    #* applied torque
    applied_torque_penalty = torch.sum(torch.square(applied_torque), dim=1)

    #* applied torque rate
    # applied_torque_rate = torch.sum(torch.square(applied_torque - previous_applied_torque), dim=1)

    #* terminated penalty
    terminated_penalty = reset_terminated.float()

    # #* joint position limit
    # out_of_limits = -(
    #     joint_pos - soft_joint_pos_limits[:, :, 0]
    # ).clip(max=0.0)
    # out_of_limits += (
    #     joint_pos - soft_joint_pos_limits[:, :, 1]
    # ).clip(min=0.0)
    # joint_pos_limit =  torch.sum(out_of_limits, dim=1)

    #* stumble 
    # stumble_feet_0 = torch.norm(contact_sensor_net_forces_w_history[:, :, feet_ids[0]], dim=-1)

    return (
        # lin_vel_error_mapped,
        # yaw_rate_error_mapped,
        height_error_mapped,
        flat_orientation_mapped,
        first_order_action_rate,
        second_order_action_rate,
        # energy_consumption,
        joint_regularization,
        undesired_contacts,
        # is_alive,
        applied_torque_penalty,
        # applied_torque_rate,
        terminated_penalty,
        # joint_pos_limit
    )

    
