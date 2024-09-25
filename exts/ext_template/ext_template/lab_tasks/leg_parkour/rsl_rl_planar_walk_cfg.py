from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class RslRlPpoEncoderActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name="EncoderActorCritic"
    init_noise_std=0.7
    actor_hidden_dims=[256, 256, 256]
    critic_hidden_dims=[256, 256, 256]
    activation="celu"

    memory_cfg: dict = {
        "type": "gru",
        "num_layers": 1,
        "hidden_size": 256,
        "input_dim": 45 + 32, # proprioception + latent height
    }

    encoder_cfg: dict = {
        "input_dim": 220,
        "hidden_dims": [128, 64],
        "activation": "celu",
        "output_dim": 32,
    }


@configclass
class LegPlanarWalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "leg_planar_walk"
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_3200.pt"
    # load_run = "2024-09-25_17-14-00"

    policy = RslRlPpoEncoderActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="CustomPPO",
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