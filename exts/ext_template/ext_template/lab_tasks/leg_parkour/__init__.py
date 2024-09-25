import gymnasium as gym
from .leg_planar_walk import LegPlanarWalkEnv
from .leg_planar_walk_cfg import LegPlanarWalkEnvCfg, LegPlanarWalkPlayEnvCfg
from .rsl_rl_planar_walk_cfg import LegPlanarWalkPPORunnerCfg


gym.register(
    id="LegRobot-planar-walk-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_parkour.leg_planar_walk:LegPlanarWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LegPlanarWalkEnvCfg,
        "rsl_rl_cfg_entry_point": LegPlanarWalkPPORunnerCfg,
    },
)

gym.register(
    id="LegRobot-planar-walk-play-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_parkour.leg_planar_walk:LegPlanarWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LegPlanarWalkPlayEnvCfg,
        "rsl_rl_cfg_entry_point": LegPlanarWalkPPORunnerCfg,
    },
)