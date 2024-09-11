import gymnasium as gym
from . import parkour_env_cfg
from . import rsl_rl_parkour_cfg
from . import rsl_rl_parkour_cfg_stage_2
from . import parkour_env_cfg_stage_1
from . import parkour_env_cfg_stage_2

gym.register(
    id="Isaac-Parkour-LegRobot-Stage2-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot_parkour_stage_2:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg_stage_2.LegRobotParkourEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg_stage_2.LegRobotParkourPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Parkour-LegRobot-Stage1-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot_parkour_stage_1:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg_stage_1.LegRobotParkourEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg.LegRobotParkourPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Parkour-LegRobot-Stage1-Play-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot_parkour_stage_1:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg_stage_1.LegRobotParkourEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg.LegRobotParkourPPORunnerCfg,
    },
)

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
    id="Isaac-Parkour-LegRobot-Play-v0",
    entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot_parkour:LegRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.LegRobotParkourEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg.LegRobotParkourPPORunnerCfg,
    },
)