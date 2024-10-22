"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner
from learning.rsl_rl_parkour.runners import CustomOnPolicyRunner

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
import exts.rl_robot.rl_robot.lab_tasks
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

# Import extensions to set up environment tasks
import rl_robot.lab_tasks  # noqa: F401

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
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    #! Create a logger
    data_logger = Logger()
    

    #! create marker
    # goal_visualizer = VisualizationMarkers(VisualizationMarkersCfg(
    #     prim_path="/World/visualization",
    #     markers={
    #         "sphere": sim_utils.SphereCfg(
    #             radius=0.1,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #         ),
    #     }
    # ))

    

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    policy_counter = 0
    policy_step_dt = env.unwrapped.step_dt
    stop_state_log_s = 5.0
    # stop_state_log = env.env.max_episode_length
    stop_state_log = int(stop_state_log_s / policy_step_dt)    # simulate environment



    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # #* command [x, y, heading]
            # translation = env.unwrapped._pos_command_w.clone() #type: ignore
            # translation[:, 2] += 0.75
            # goal_visualizer.visualize(translation) #type: ignore
            env.unwrapped._commands[:, 0] = 0.5
            env.unwrapped._commands[:, 1] = 0.0
            env.unwrapped._commands[:, 2] = 0.0

            policy_counter += 1

            if policy_counter < stop_state_log:
                data_frame = {
                    'time_step': policy_counter*policy_step_dt,
                    'c_x': env.unwrapped._commands[0, 0].item(),
                    'c_y': env.unwrapped._commands[0, 1].item(),
                    'c_z': env.unwrapped._commands[0, 2].item(),
                    'base_x': env.env.scene["robot"].data.root_pos_w[0, 0].item(),
                    'base_y': env.env.scene["robot"].data.root_pos_w[0, 1].item(),
                    'base_z': env.env.scene["robot"].data.root_pos_w[0, 2].item(),
                    'base_vx': env.env.scene["robot"].data.root_lin_vel_b[0, 0].item(),
                    'base_vy': env.env.scene["robot"].data.root_lin_vel_b[0, 1].item(),
                    'base_vz': env.env.scene["robot"].data.root_lin_vel_b[0, 2].item(),
                    'base_wx': env.env.scene["robot"].data.root_ang_vel_b[0, 0].item(),
                    'base_wy': env.env.scene["robot"].data.root_ang_vel_b[0, 1].item(),
                    'base_wz': env.env.scene["robot"].data.root_ang_vel_b[0, 2].item(),
                    **{'pos_' + key : env.env.scene["robot"].data.joint_pos[0, env.env.scene["robot"].find_joints(key)[0]].item() for index, key in enumerate(DOF_NAMES)},
                    **{'vel_' + key : env.env.scene["robot"].data.joint_vel[0, env.env.scene["robot"].find_joints(key)[0]].item() for index, key in enumerate(DOF_NAMES)},
                    **{'torque_' + key : env.env.scene["robot"].data.applied_torque[0, index].item() for index, key in enumerate(DOF_NAMES)},
                }

                data_logger.log_states(data_frame)
            else:
                data_logger.save_log('analysis/data/state_log.csv')

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
