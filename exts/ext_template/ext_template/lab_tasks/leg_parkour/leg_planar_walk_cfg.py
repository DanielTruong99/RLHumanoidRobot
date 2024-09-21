from __future__ import annotations

import torch
import math

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as UNoiseCfg
from omni.isaac.lab.utils.noise import NoiseModelCfg
from omni.isaac.lab.managers import EventTermCfg

##
#! user Custom configs
##
from . import mdp as custom_mdp

##
#! Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from exts.ext_template.ext_template.lab_assets import LEGPARKOUR_CFG

@configclass
class EventCfg:
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0), 
                "y": (-0.0, 0.0),  
                "roll": (-math.pi/10, math.pi/10),
                "pitch": (-math.pi/10, math.pi/10),
                "yaw": (-math.pi/10, math.pi/10)
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },  
        },
    )

    reset_robot_joints = EventTermCfg(
        func=custom_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": {
                'R_hip_joint': (-0.1, 0.1),
                'R_hip2_joint': (-0.2, 0.2),
                'R_thigh_joint': (-0.2, 0.2),
                'R_calf_joint': (0.0, 0.2),
                'R_toe_joint': (-0.3, 0.3),
                'L_hip_joint': (-0.1, 0.1),
                'L_hip2_joint': (-0.2, 0.2),
                'L_thigh_joint': (-0.2, 0.2),
                'L_calf_joint': (0.0, 0.2),
                'L_toe_joint': (-0.3, 0.3),
            },
            "velocity_range": {
                'R_hip_joint': (-0.1, 0.1),
                'R_hip2_joint': (-0.1, 0.1),
                'R_thigh_joint': (-0.1, 0.1),
                'R_calf_joint': (-0.1, 0.1),
                'R_toe_joint': (-0.1, 0.1),
                'L_hip_joint': (-0.1, 0.1),
                'L_hip2_joint': (-0.1, 0.1),
                'L_thigh_joint': (-0.1, 0.1),
                'L_calf_joint': (-0.1, 0.1),
                'L_toe_joint': (-0.1, 0.1),
            }
        },
    )

    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 2.5),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class CommandCfg:
    resampling_time_range = (5.0, 5.0)
    ranges_lin_vel_x = (0.0, 1.0)
    ranges_lin_vel_y = (-0.3, 0.3)
    ranges_ang_vel_z = (-1.0, 1.0)

@configclass
class LegPlanarWalkEnvCfg(DirectRLEnvCfg):
    """configuration class for the LegPlanarWalkEnv environment
        0. event manager configuration
        1. env configuration
        2. simulation configuration
        3. scene configuration
        4. robot configuration
        5. contact sensor configuration
        6. terrain configuration
        7. height scanner configuration
        8. observation noise model configuration
    """
    #* command configuration
    commands = CommandCfg()

    #* event manager configuration
    events = EventCfg()

    #* env
    episode_length_s = 20.0
    decimation = 2
    action_scale = 1.0
    num_actions = 10
    num_observations = 3 + 3 + 3 + 3 + 10 + 10 + 10 + 2 + 1 + 209 #! NEED TO BE CHANGED
    num_states = 0

    #* simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    #* scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    #* robot configuration
    #! replace the ImplicitActuatorCfg like continuous pd controller
    robot: ArticulationCfg = LEGPARKOUR_CFG.replace( #type: ignore
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness={
                    ".*_hip_joint": 150.0,
                    ".*_hip2_joint": 150.0,
                    ".*_thigh_joint": 200.0,
                    ".*_calf_joint": 200.0,
                },
                damping={
                    ".*_hip_joint": 5.0,
                    ".*_hip2_joint": 5.0,
                    ".*_thigh_joint": 5.0,
                    ".*_calf_joint": 5.0,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_toe_joint"],
                effort_limit=30.0,
                velocity_limit=50.0,
                stiffness={".*_toe_joint": 30.0},
                damping={".*_toe_joint": 5.0},
            ),
        },
    ) 
    
    #* contact sensor configuration for the feet
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )

    #* terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    #* height scanner configuration
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.93, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.9, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    #* observation noise model configuration
    observation_noise_model = NoiseModelCfg(
        noise_cfg=UNoiseCfg(
            n_min=torch.cat([
                torch.tensor([-0.1] * 3),   # v: base linear velocity (3,)
                torch.tensor([-0.2] * 3),   # w: base angular velocity (3,)
                torch.tensor([-0.05] * 3),  # g: projected gravity (3,)
                torch.tensor([0.0] * 3),    # c: commands (3,)
                torch.tensor([-0.01] * 10), # p: joint positions (10,)
                torch.tensor([-1.5] * 10),  # p_dot: joint velocities (10,)
                torch.tensor([0.0] * 10),   # a: last actions (10,)
                torch.tensor([0.0] * 2),  # foot_contact_state: foot_contact_state (2,)
                torch.tensor([0.0] * 1), # estimated_height: estimated_height (1,)
                torch.tensor([-0.1] * 209)  # height_data: height scanner (209,)
            ]), 
            n_max=torch.cat([
                torch.tensor([0.1] * 3),   # v: base linear velocity (3,)
                torch.tensor([0.2] * 3),   # w: base angular velocity (3,)
                torch.tensor([0.05] * 3),  # g: projected gravity (3,)
                torch.tensor([0.0] * 3),   # c: commands (3,)
                torch.tensor([0.01] * 10), # p: joint positions (10,)
                torch.tensor([1.5] * 10),  # p_dot: joint velocities (10,)
                torch.tensor([0.0] * 10),  # a: last actions (10,)
                torch.tensor([0.0] * 2),  # foot_contact_state: foot_contact_state (2,)
                torch.tensor([0.0] * 1), # estimated_height: estimated_height (1,)
                torch.tensor([0.1] * 209)  # height_data: height scanner (209,)
            ])
        )
    )

    #* reward configuration
    #! encourage reward 
    base_height_target = 0.74
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 1.5
    base_height_reward_scale = 0.5
    flat_orientation_reward_scale = 0.5
    joint_regularization_reward_scale = 0.2
    is_alive_reward_scale = 0.7

    #! penalty reward
    first_order_action_rate_reward_scale = -1e-3
    second_order_action_rate_reward_scale = -1e-4
    energy_consumption_reward_scale = -2.5e-7
    undesired_contacts_reward_scale = -3e-3
    applied_torque_reward_scale = -1e-4
    applied_torque_rate_reward_scale = -1e-7

    #! terminated penalty reward
    terminated_penalty_reward_scale = -10.0
