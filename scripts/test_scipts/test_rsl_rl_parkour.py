from learning.rsl_rl_parkour.modules.all_mixer import EncoderStateAcRecurrent
from collections import OrderedDict

policy = EncoderStateAcRecurrent(
    num_actor_obs=42, 
    num_critic_obs=42, 
    num_actions=12,
    estimator_obs_components = [
        "ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos",
        "dof_vel",
        "last_actions",
    ],
    estimator_target_components = ["lin_vel"],
    replace_state_prob = 1.0,
    estimator_kwargs={
        'hidden_sizes': [128, 64],
        'nonlinearity': "CELU",
    },
    encoder_component_names = ["height_measurements"],
    encoder_class_name = "MlpModel",
    encoder_kwargs = {
        'hidden_sizes' : [128, 64],
        'nonlinearity' : "CELU",
    },
    encoder_output_size = 32,
    critic_encoder_component_names = ["height_measurements"],
    init_noise_std = 0.5,
    rnn_type = 'gru',
    mu_activation = None,
    obs_segments = OrderedDict([
        ('lin_vel', (3, )),
        ('ang_vel', (3, )),
        ('projected_gravity', (3, )),
        ('commands', (4, )),
        ('dof_pos', (12, )),
        ('dof_vel', (12, )),
        ('last_actions', (12, )),
        ('height_measurements', (1, 21, 11)),
    ])
)