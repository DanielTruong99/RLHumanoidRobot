import gymnasium as gym
from .leg_planar_walk import LegPlanarWalkEnv
from .leg_planar_walk_cfg import LegPlanarWalkEnvCfg, LegPlanarWalkPlayEnvCfg
from .rsl_rl_planar_walk_cfg import LegPlanarWalkPPORunnerCfg

from .leg_leap import LegLeapEnv
from .leg_leap_cfg import LegLeapEnvCfg, LegLeapPlayEnvCfg
from .rsl_rl_leap_cfg import LegLeapPPORunnerCfg, LegLeapPPOPlayRunnerCfg


'''
    Register the LegRobot-planar-walk-v0 and LegRobot-planar-walk-play-v0 environments
'''
gym.register(
    id="LegRobot-planar-walk-v1",
    entry_point="exts.rl_robot.rl_robot.lab_tasks.leg_parkour.leg_planar_walk:LegPlanarWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LegPlanarWalkEnvCfg,
        "rsl_rl_cfg_entry_point": LegPlanarWalkPPORunnerCfg,
    },
)

gym.register(
    id="LegRobot-planar-walk-play-v1",
    entry_point="exts.rl_robot.rl_robot.lab_tasks.leg_parkour.leg_planar_walk:LegPlanarWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LegPlanarWalkPlayEnvCfg,
        "rsl_rl_cfg_entry_point": LegPlanarWalkPPORunnerCfg,
    },
)

'''
    Register the 
'''
gym.register(
    id="LegRobot-leap-v1",
    entry_point="exts.rl_robot.rl_robot.lab_tasks.leg_parkour.leg_leap:LegLeapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LegLeapEnvCfg,
        "rsl_rl_cfg_entry_point": LegLeapPPORunnerCfg,
    },
)

gym.register(
    id="LegRobot-leap-play-v1",
    entry_point="exts.rl_robot.rl_robot.lab_tasks.leg_parkour.leg_leap:LegLeapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LegLeapPlayEnvCfg,
        "rsl_rl_cfg_entry_point": LegLeapPPOPlayRunnerCfg,
    },
)
