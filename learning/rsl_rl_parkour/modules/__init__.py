
from .all_mixers import EncoderStateAcRecurrent

def build_actor_critic(env, policy_class_name, policy_cfg):
    """ NOTE: This method allows to hack the policy kwargs by adding the env attributes to the policy_cfg. """
    actor_critic_class = globals()[policy_class_name] # EncoderStateAcRecurrent
    policy_cfg = policy_cfg.copy()

    #* Check if use obs difference than the policy for the critic 
    num_critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs

    if hasattr(env, "obs_segments") and "obs_segments" not in policy_cfg:
        policy_cfg["obs_segments"] = env.obs_segments
    if hasattr(env, "privileged_obs_segments") and "privileged_obs_segments" not in policy_cfg:
        policy_cfg["privileged_obs_segments"] = env.privileged_obs_segments
    if not "num_actor_obs" in policy_cfg:
        policy_cfg["num_actor_obs"] = env.num_obs
    if not "num_critic_obs" in policy_cfg:
        policy_cfg["num_critic_obs"] = num_critic_obs
    if not "num_actions" in policy_cfg:
        policy_cfg["num_actions"] = env.num_actions
    
    actor_critic: EncoderStateAcRecurrent = actor_critic_class(**policy_cfg)

    return actor_critic