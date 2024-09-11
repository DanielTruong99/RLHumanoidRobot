from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torchsummary import summary

from rsl_rl.modules import ActorCritic

class CustomActorCritic(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs - 187
        mlp_input_dim_c = num_critic_obs - 187
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        self.actor_input_layer = nn.Sequential(*list(self.actor.children())[0:1])
        self.remaining_actor_layers = nn.Sequential(*list(self.actor.children())[1:])

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        self.critic_input_layer = nn.Sequential(*list(self.critic.children())[0:1])
        self.remaining_critic_layers = nn.Sequential(*list(self.critic.children())[1:])

        #! Scandots enconder
        #! Temporaly fixed the architecture
        #* 187 is the dimension of scandots vector
        encoder_layers = [
            nn.Linear(187, 64),
            nn.ELU(),

            nn.Linear(64, 64),
            nn.ELU(),

            nn.Linear(64, 32),
            nn.ELU(),

            nn.Linear(32, 16),
        ] 
        self.scandots_encoder = nn.Sequential(*encoder_layers)
        self.scandot_combine_layer_actor = nn.Sequential(nn.Linear(16, actor_hidden_dims[0]))
        self.scandot_combine_layer_critic = nn.Sequential(nn.Linear(16, critic_hidden_dims[0]))


        print(f"Actor MLP: {self.actor}")
        self.actor.cuda()
        summary(self.actor, (mlp_input_dim_a, ))

        print(f"Critic MLP: {self.critic}")
        self.critic.cuda()
        summary(self.critic, (mlp_input_dim_c, ))

        print(f"Scandots Encoder MLP: {self.scandots_encoder}")
        self.scandots_encoder.cuda()
        summary(self.scandots_encoder, (187, ))

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    #! Edit this function to accompany the encoder
    def update_distribution(self, observations):
        #! Suppose the first 187 element of the observations is the scandots
        scan_dots = observations[:, :187]
        latent_scandots = self.scandots_encoder(scan_dots)
        
        new_observations = self.actor_input_layer(observations[:, 187:]) + self.scandot_combine_layer_actor(latent_scandots)

        mean = self.remaining_actor_layers(new_observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    #! Edit this function to accompany the encoder
    def act_inference(self, observations):
        #! Suppose the first 187 element of the observations is the scandots
        scan_dots = observations[:, :187]
        latent_scandots = self.scandots_encoder(scan_dots)
        
        new_observations = self.actor_input_layer(observations[:, 187:]) + self.scandot_combine_layer_actor(latent_scandots)

        actions_mean = self.remaining_actor_layers(new_observations)

        # actions_mean = self.actor(observations)
        return actions_mean

    #! Edit this function to accompany the encoder
    def evaluate(self, critic_observations, **kwargs):
        scan_dots = critic_observations[:, :187]
        latent_scandots = self.scandots_encoder(scan_dots)
        
        new_observations = self.actor_input_layer(critic_observations[:, 187:]) + self.scandot_combine_layer_critic(latent_scandots)

        value = self.remaining_critic_layers(new_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU() # type: ignore
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None