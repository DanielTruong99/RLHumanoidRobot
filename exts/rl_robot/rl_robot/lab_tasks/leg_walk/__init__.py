import gymnasium as gym

from .simple_walking_robot_cfg import SimpleWalkingRobotEnvCfg, SimpleWalkingRobotPlayEnvCfg
from .rsl_rl_simple_walking_cfg import SimpleWalkingRobotPPORunnerCfg, SimpleWalkingRobotPPOPlayRunnerCfg


'''
    Register the LegRobot-planar-walk-v2 and LegRobot-planar-walk-play-v2 environments
'''
gym.register(
    id="LegRobot-planar-walk-v2",
    entry_point="exts.rl_robot.rl_robot.lab_tasks.leg_walk.simple_walking_robot:SimpleWalkingRobot",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimpleWalkingRobotEnvCfg,
        "rsl_rl_cfg_entry_point": SimpleWalkingRobotPPORunnerCfg,
    },
)

gym.register(
    id="LegRobot-planar-walk-play-v2",
    entry_point="exts.rl_robot.rl_robot.lab_tasks.leg_walk.simple_walking_robot:SimpleWalkingRobot",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimpleWalkingRobotPlayEnvCfg,
        "rsl_rl_cfg_entry_point": SimpleWalkingRobotPPOPlayRunnerCfg,
    },
)

