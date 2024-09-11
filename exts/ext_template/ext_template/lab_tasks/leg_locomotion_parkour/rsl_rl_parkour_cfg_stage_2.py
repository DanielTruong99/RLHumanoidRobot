# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LegRobotParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    resume = True
    # logs/rsl_rl/leg_robot_parkour/2024-08-28_11-46-06/model_9200.pt
    load_run = "2024-08-28_11-46-06"
    load_checkpoint = "model_9200.pt"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 50
    experiment_name = "leg_robot_parkour"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="CustomActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# @configclass
# class H1FlatPPORunnerCfg(H1RoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 1000
#         self.experiment_name = "h1_flat"
#         self.policy.actor_hidden_dims = [128, 128, 128]
#         self.policy.critic_hidden_dims = [128, 128, 128]
