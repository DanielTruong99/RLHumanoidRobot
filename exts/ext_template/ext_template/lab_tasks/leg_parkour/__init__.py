import gymnasium as gym


# gym.register(
#     id="Isaac-Parkour-LegRobot-Stage2-v0",
#     entry_point="exts.ext_template.ext_template.lab_tasks.leg_locomotion_parkour.leg_robot_parkour_stage_2:LegRobotEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": parkour_env_cfg_stage_2.LegRobotParkourEnvCfg,
#         "rsl_rl_cfg_entry_point": rsl_rl_parkour_cfg_stage_2.LegRobotParkourPPORunnerCfg,
#     },
# )