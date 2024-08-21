from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets.articulation import ArticulationCfg
import omni.isaac.lab.sim as sim_utils

import math
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from . import mdp as custom_mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from ...lab_assets import LEG10_CFG, LEG10_USD_PATH

@configclass
class Rewards(RewardsCfg):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R_toe", "L_toe"]),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["R_toe", "L_toe"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["R_toe", "L_toe"]),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint"])},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
 


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

@configclass
class LegRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Rewards = Rewards()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # TODO: Override simulation settings
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # TODO: Override scene configurations
        #! Robot
        self.scene.robot = LEG10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=LEG10_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
        ),
        self.scene.robot.replace(spawn=spawn_cfg) # type: ignore
        init_state_cfg = ArticulationCfg.InitialStateCfg( # type: ignore
            pos=(0.0, 0.0, 0.892),
            joint_pos={
                'R_hip_joint': 0.0,
                'R_hip2_joint': 0.0,
                'R_thigh_joint': -0.2,
                'R_calf_joint': 0.25,  
                'R_toe_joint': 0.0,
                'L_hip_joint': 0.0,
                'L_hip2_joint': 0.0,
                'L_thigh_joint': -0.2,
                'L_calf_joint': 0.25,  
                'L_toe_joint': 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        self.scene.robot.replace(init_state=init_state_cfg) # type: ignore
        self.scene.robot.actuators["legs"].effort_limit = 300.0
        self.scene.robot.actuators["legs"].stiffness = {
            ".*_hip_joint": 150.0,
            ".*_hip2_joint": 150.0,
            ".*_thigh_joint": 200.0,
            ".*_calf_joint": 200.0,
        }
        self.scene.robot.actuators["legs"].damping = {
            ".*_hip_joint": 5.0,
            ".*_hip2_joint": 5.0,
            ".*_thigh_joint": 5.0,
            ".*_calf_joint": 5.0,
        }
        self.scene.robot.actuators["feet"].stiffness = {".*_toe_joint": 20.0}
        self.scene.robot.actuators["feet"].damping = {".*_toe_joint": 2.0}

        #! Height Scanner
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # TODO: Override randomization configurations
        self.events.push_robot = None # type: ignore
        self.events.add_base_mass = None # type: ignore
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
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

        # TODO: Override reward configurations
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None # type: ignore
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint", ".*_toe_joint"]
        )

        # TODO: Override random command velocity range
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class LegRoughEnvCfg_PLAY(LegRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.scene.height_scanner.debug_vis = True
        