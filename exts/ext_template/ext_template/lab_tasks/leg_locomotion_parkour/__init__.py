import gymnasium as gym
from . import parkour_env_cfg
from . import rsl_rl_parkour_cfg

gym.register(
    id="Isaac-Parkour-LegRobot-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot_parkour:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.LegRobotParkourEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg.LegRobotParkourPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Play-AMP-LegRobot-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.LegRobotParkourEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg.LegRobotParkourPPORunnerCfg,
    },
)