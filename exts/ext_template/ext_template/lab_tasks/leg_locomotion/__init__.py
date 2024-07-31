import gymnasium as gym
from . import rough_env_cfg
from . import rsl_rl_cfg


gym.register(
    id="Isaac-Velocity-Rough-LegRobot-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion.leg_robot:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.LegRobotRoughEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_cfg.LegRobotRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Play-LegRobot-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion.leg_robot:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.LegRobotRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_cfg.LegRobotRoughPPORunnerCfg,
    },
)
