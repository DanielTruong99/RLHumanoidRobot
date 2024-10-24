from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class CustomRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name="ActorCritic"
    init_noise_std=0.5
    actor_hidden_dims=[256, 256, 256]
    critic_hidden_dims=[256, 256, 256]
    activation="elu"

@configclass
class SimpleWalkingRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "simple_walking_robot"
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_3200.pt"
    # load_run = "2024-09-25_17-14-00"

    policy = CustomRslRlPpoActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
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

@configclass
class SimpleWalkingRobotPPOPlayRunnerCfg(SimpleWalkingRobotPPORunnerCfg):
    resume = False
    # load_checkpoint = "model_2850.pt"
    # load_run = "2024-10-23_18-21-54"