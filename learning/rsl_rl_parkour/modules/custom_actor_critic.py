import torch
import torch.nn as nn
import sys

from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.modules.actor_critic import get_activation as get_activation_base

def get_activation(act_name):
    if act_name == "celu":
        return nn.CELU()
    else: 
        return get_activation_base(act_name)
    
sys.modules['rsl_rl.modules.actor_critic'].get_activation = get_activation # type: ignore

class EncoderActorCritic(ActorCriticRecurrent):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        #* get encoder configuration
        encoder_cfg = kwargs["encoder_cfg"]
        activation = get_activation(encoder_cfg["activation"])

        #* get memory configuration
        memory_cfg = kwargs["memory_cfg"]
        num_actor_obs = memory_cfg["input_dim"]; num_critic_obs = memory_cfg["input_dim"]

        super().__init__(
            num_actor_obs, 
            num_critic_obs, 
            num_actions, 
            rnn_type=memory_cfg["type"],
            rnn_hidden_size=memory_cfg["hidden_size"],
            rnn_num_layers=memory_cfg["num_layers"],
            **kwargs
        )

        #* build height map encoder from configuration
        encoder_layers = []
        encoder_layers.append(nn.Linear(encoder_cfg["input_dim"], encoder_cfg["hidden_dims"][0]))
        encoder_layers.append(activation)
        for layer_index in range(len(encoder_cfg["hidden_dims"])):
            if layer_index == len(encoder_cfg["hidden_dims"]) - 1:
                encoder_layers.append(nn.Linear(encoder_cfg["hidden_dims"][layer_index], encoder_cfg["output_dim"]))
            
            else:
                encoder_layers.append(nn.Linear(encoder_cfg["hidden_dims"][layer_index], encoder_cfg["hidden_dims"][layer_index + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"encoder mlp: {self.encoder}")

        self.tanh_layer = nn.Tanh()

    def act(self, observations, masks=None, hidden_states=None):
        #! split proprioception and height map data based on the configuration (preferred to be done in the environment configuration)
        #* proprioception: 3 + 3 + 3 + 3 + 10 + 10 + 10 + 2
        #* height_data: 220
        proprioception, height_data = torch.split(observations, [44, 220], dim= -1)
        latent_height = self.encoder(height_data)
        policy_input = torch.cat([proprioception, latent_height], dim= -1)
        policy_output = super().act(policy_input, masks, hidden_states)
        # return self.tanh_layer(policy_output)
        return policy_output
    
    def act_inference(self, observations):
        #! split proprioception and height map data based on the configuration (preferred to be done in the environment configuration)
        #* proprioception: 3 + 3 + 3 + 3 + 10 + 10 + 10 + 2 
        #* height_data: 220
        proprioception, height_data = torch.split(observations, [44, 220], dim= -1)
        latent_height = self.encoder(height_data)
        policy_input = torch.cat([proprioception, latent_height], dim= -1)
        policy_output = super().act_inference(policy_input)
        # return self.tanh_layer(policy_output)
        return policy_output
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        #! split proprioception and height map data based on the configuration (preferred to be done in the environment configuration)
        #* proprioception: 3 + 3 + 3 + 3 + 10 + 10 + 10 + 2
        #* height_data: 220
        proprioception, height_data = torch.split(critic_observations, [44, 220], dim= -1)
        latent_height = self.encoder(height_data)
        critic_input = torch.cat([proprioception, latent_height], dim= -1)
        return super().evaluate(critic_input, masks, hidden_states)


    @torch.no_grad()
    def clip_std(self, min= None, max= None):
        self.std.copy_(self.std.clip(min= min, max= max))
    
globals()["EncoderActorCritic"] = EncoderActorCritic