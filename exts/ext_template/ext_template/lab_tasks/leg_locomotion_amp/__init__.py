import gymnasium as gym
from . import rough_amp_env_cfg
from . import rsl_rl_amp_cfg

gym.register(
    id="Isaac-Velocity-Rough-AMP-LegRobot-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_amp.leg_robot:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_amp_env_cfg.LegRobotRoughAMPEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_amp_cfg.LegRobotRoughAMPPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Play-AMP-LegRobot-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_amp.leg_robot:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_amp_env_cfg.LegRobotRoughAMPEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_amp_cfg.LegRobotRoughAMPPPORunnerCfg,
    },
)