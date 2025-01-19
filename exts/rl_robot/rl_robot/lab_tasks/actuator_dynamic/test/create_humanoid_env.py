# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a humanoid environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from exts.rl_robot.rl_robot.lab_tasks.leg_walk_unitree.walking_robot import WalkingRobotEnv
from exts.rl_robot.rl_robot.lab_tasks.leg_walk_unitree.walking_robot_cfg import WalkingRobotEnvPLayCfg


def main():
    """Main function."""
    # setup base environment
    env_cfg = WalkingRobotEnvPLayCfg()
    env_cfg.scene.num_envs = args_cli.num_envs  
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True #type: ignore
    env_cfg.scene.terrain.terrain_type = "usd"
    env_cfg.scene.terrain.usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"

    # create environment
    env = WalkingRobotEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # infer action
            action = torch.zeros(env_cfg.scene.num_envs, 10, device=env_cfg.sim.device)
            
            # step env
            obs, *_ = env.step(action)
            
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
