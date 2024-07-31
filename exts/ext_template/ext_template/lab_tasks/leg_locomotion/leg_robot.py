import torch

from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionManager
from typing import TYPE_CHECKING
from omni.isaac.lab.envs import ManagerBasedEnv

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
    def load_managers(self):
        super().load_managers()

        self.action_manager = CustomActionManager(self.cfg.actions, self)