from __future__ import annotations
from rsl_rl.runners import OnPolicyRunner


import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from ..modules import CustomActorCritic
from rsl_rl.utils import store_code_state



class CustomOnPolicyRunner(OnPolicyRunner):

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent | CustomActorCritic = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        #! Add strict=False
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        # if load_optimizer:
        #     self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]