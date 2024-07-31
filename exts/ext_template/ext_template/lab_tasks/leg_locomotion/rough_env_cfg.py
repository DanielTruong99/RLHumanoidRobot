# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from omni.isaac.lab.envs.common import ViewerCfg

from . import mdp as custom_mdp

##
# Pre-defined configs
##
from ...lab_assets import LEG10_CFG

@configclass
class RewardsCfg:
    """#! Baseline rewards
        Linear velocity: 10 
        Angular velocity: 5
        1st order action rate: -1e-3
        2nd order action rate: -1e-4
        Torque: -1e-4
        Torque limit: -0.01
        Joint limit: -10
        Termination: -100
    """
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )

    #TODO: Override action_rate_l2 by action_rate_1st_order
    action_rate_1st_order = RewTerm(func=custom_mdp.action_rate_1st_order, weight=-1e-3)

    action_rate_2nd_order = RewTerm(func=custom_mdp.action_rate_2nd_order, weight=-1e-4)

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        ".*_hip_joint",
                        ".*_hip2_joint",
                        ".*_thigh_joint",
                        ".*_calf_joint",
                        ".*_toe_joint"
                    ]
                )            
        }
    )   

    dof_torques_limit = RewTerm(
        func=mdp.applied_torque_limits, 
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        ".*_hip_joint",
                        ".*_hip2_joint",
                        ".*_thigh_joint",
                        ".*_calf_joint",
                        ".*_toe_joint"
                    ]
                ),
        }
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, 
        params={
            "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        ".*_hip_joint",
                        ".*_hip2_joint",
                        ".*_thigh_joint",
                        ".*_calf_joint",
                        ".*_toe_joint"
                    ]
                )
        }
    )    

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)





@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base"), "threshold": 1.0},
    )


@configclass
class LegRobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Scene
        self.scene.robot = LEG10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Randomization
        self.events.push_robot = None  # type: ignore
        self.events.add_base_mass = None # type: ignore
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*base"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        #! Just for debug purposes
        #! Should be commented out when actually trainning
        # self.scene.height_scanner.debug_vis = True
        # self.viewer = ViewerCfg(
        #     eye=(3, 3, 3),
        #     origin_type='asset_root',
        #     asset_name='robot',
        #     env_index=0,
        # )



@configclass
class LegRobotRoughEnvCfg_PLAY(LegRobotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None # type: ignore
        self.events.push_robot = None # type: ignore

        #! Just for debug purposes
        #! Should be commented out when actually trainning
        self.scene.height_scanner.debug_vis = True
        self.viewer = ViewerCfg(
            eye=(3, 3, 3),
            origin_type='asset_root',
            asset_name='robot',
            env_index=0,
        )

        self.sim.dt = 0.005
        # self.sim.render_interval = self.decimation * 3
        # self.scene.contact_forces.debug_vis = True
 
