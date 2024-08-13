import torch

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionManager
from typing import TYPE_CHECKING
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.common import VecEnvStepReturn
from . import mdp as custom_mdp
from omni.isaac.lab.managers.reward_manager import RewardTermCfg

class CustomActionManager(ActionManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._prev2_action = torch.zeros_like(self._action)

    def process_action(self, action: torch.Tensor):
        self._prev2_action[:] = self._prev_action
        super().process_action(action)

    def reset(self, env_ids) -> dict[str, torch.Tensor]:
        self._prev2_action[env_ids] = 0.0
        return super().reset(env_ids)

    @property
    def prev2_action(self) -> torch.Tensor:
        """The previous previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev2_action
        

class LegRobotEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        #! Custom attributes for clock phase
        self.phase = torch.zeros(
            cfg.scene.num_envs, 1, dtype=torch.float,
            device=cfg.sim.device, requires_grad=False
        )
        self.phase_freq = 1.0
        self.eps = 0.2        
        
        super().__init__(cfg, render_mode, **kwargs)



    def load_managers(self):
        super().load_managers()

        self.action_manager = CustomActionManager(self.cfg.actions, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # ! Custom pre physic step callback
        # * Cache the reward for potential-based 
        # self.reward_manager.get_term_cfg("pb_flat_orientation_exp)")
        pb_flat_cfg = self.reward_manager.get_term_cfg("pb_orientation").params
        pb_base_cfg = self.reward_manager.get_term_cfg("pb_base_height").params
        pb_joint_regulization_cfg = self.reward_manager.get_term_cfg("pb_joint_regularization").params
        self.rwd_oriPrev = custom_mdp.flat_orientation_exp(env=self, **pb_flat_cfg) #type: ignore
        self.rwd_baseHeightPrev = custom_mdp.base_height_exp(env=self, **pb_base_cfg) #type: ignore
        self.rwd_jointRegPrev = custom_mdp.joint_regulization_exp(env=self, **pb_joint_regulization_cfg) #type: ignore

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        # ! Custom post physic step callback
        # * Update the phase
        self.phase = torch.fmod(self.phase + self.physics_dt, 1.0)

        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids) #type: ignore
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras