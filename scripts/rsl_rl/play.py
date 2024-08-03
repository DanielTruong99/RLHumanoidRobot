"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
# import ext_template.tasks  # noqa: F401
import exts.ext_template.ext_template.lab_tasks

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

from omni.isaac.core.loggers.data_logger import DataLogger

DOF_NAMES = [
    'R_hip_joint',
    'R_hip2_joint',
    'R_thigh_joint',
    'R_calf_joint',  
    'R_toe_joint',
    'L_hip_joint',
    'L_hip2_joint',
    'L_thigh_joint',
    'L_calf_joint',  
    'L_toe_joint'
]

import numpy as np 
import pandas as pd
from collections import defaultdict
class Logger:
    def __init__(self):
        self.state_log = defaultdict(list)
    
    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def save_log(self, path):
        pd.DataFrame(self.state_log).to_csv(path, index=False)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    #! Create a logger
    data_logger = Logger()

    # reset environment 
    obs, _ = env.get_observations()

    policy_counter = 0
    policy_step_dt = env.env.step_dt
    stop_state_log_s = 10.0
    # stop_state_log = env.env.max_episode_length
    stop_state_log = int(stop_state_log_s / policy_step_dt)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)

            obs[:, 9] = 0.3
            obs[:, 10] = 0.0
            obs[:, 11] = 0.0

            policy_counter += 1

            if policy_counter < stop_state_log:
                data_frame = {
                    'time_step': policy_counter*policy_step_dt,
                    'base_x': env.env.scene["robot"].data.root_pos_w[0, 0].item(),
                    'base_y': env.env.scene["robot"].data.root_pos_w[0, 1].item(),
                    'base_z': env.env.scene["robot"].data.root_pos_w[0, 2].item(),
                    'base_vx': env.env.scene["robot"].data.root_lin_vel_b[0, 0].item(),
                    'base_vy': env.env.scene["robot"].data.root_lin_vel_b[0, 1].item(),
                    'base_vz': env.env.scene["robot"].data.root_lin_vel_b[0, 2].item(),
                    'base_wx': env.env.scene["robot"].data.root_ang_vel_b[0, 0].item(),
                    'base_wy': env.env.scene["robot"].data.root_ang_vel_b[0, 1].item(),
                    'base_wz': env.env.scene["robot"].data.root_ang_vel_b[0, 2].item(),
                    **{'pos_' + key : env.env.scene["robot"].data.joint_pos[0, index].item() for index, key in enumerate(DOF_NAMES)},
                    **{'vel_' + key : env.env.scene["robot"].data.joint_vel[0, index].item() for index, key in enumerate(DOF_NAMES)},
                    **{'torque_' + key : env.env.scene["robot"].data.applied_torque[0, index].item() for index, key in enumerate(DOF_NAMES)},
                }

                data_logger.log_states(data_frame)
            else:
                data_logger.save_log('analysis/data/state_log.csv')
                


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
