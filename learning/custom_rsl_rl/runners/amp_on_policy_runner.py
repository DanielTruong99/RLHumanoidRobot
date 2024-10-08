# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

import numpy as np
import torch

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from ..algorithms import AMPPPO, AMPDiscriminator
from ..datasets import AMPLoader
from ..utils import Normalizer
from rsl_rl.runners import OnPolicyRunner

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state

class AMPOnPolicyRunner(OnPolicyRunner):
    
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
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        """
            #TODO: nedd to add to the config file:
                amp_num_preload_transitions
                amp_motion_files
                amp_reward_coef
                amp_discr_hidden_dims
                amp_task_reward_lerp
                min_normalized_std
        """
        #! Initialize AMPLoader
        amp_data = AMPLoader(
            device, 
            time_between_frames=self.env.env.step_dt, #type: ignore
            preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        

        #! Initialize discriminator network
        discriminator = AMPDiscriminator(
        amp_data.observation_dim * 2,
        self.cfg['amp_reward_coef'],
        self.cfg['amp_discr_hidden_dims'], device,
        self.cfg['amp_task_reward_lerp']).to(self.device)

        #! Initialize normalizer for normalization observations
        amp_normalizer = Normalizer(amp_data.observation_dim)
        

        #! Initialize AMPPPO algorithm
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.env.scene["robot"].data.joint_limits[0, :, 1] - self.env.env.scene["robot"].data.joint_limits[0, :, 0]))) #type: ignore
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: AMPPPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)


        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg) #type: ignore
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg) #type: ignore
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        
        #! Add AMP observations
        amp_obs = extras["observations"].get("amp", obs)
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, rewards, dones, infos = self.env.step(actions) #type: ignore
                    obs = self.obs_normalizer(obs)
                    
                    #! Add amp observations
                    next_amp_obs = infos["observations"]["amp"]
                    
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs

                    #! Add amp observations
                    obs, critic_obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )
                    next_amp_obs_with_term = torch.clone(next_amp_obs)

                    #! Combine task and style rewards
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)

                    #! Add amp observations
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt")) #type: ignore
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path) #type: ignore

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")) #type: ignore


    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs)

        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])


    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration) #type: ignore

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        super().train_mode()
        self.alg.discriminator.train()