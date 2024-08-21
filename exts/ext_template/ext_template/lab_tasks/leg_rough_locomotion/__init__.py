import gymnasium as gym
from . import agents, rough_env_cfg

gym.register(
    id="Isaac-Velocity-Rough-Leg-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.LegRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LegRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Leg-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.LegRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LegRoughPPORunnerCfg,
    },
)