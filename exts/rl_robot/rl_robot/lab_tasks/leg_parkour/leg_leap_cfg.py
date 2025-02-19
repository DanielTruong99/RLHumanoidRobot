import math

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.terrains as terrain_gen
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import EventTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs import ViewerCfg

###
#! User defined imports
###
from . import terrain
from .leg_planar_walk_cfg import LegPlanarWalkEnvCfg, EventCfg
from . import mdp as custom_mdp

#* terrain configurations
PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=0.0,
    num_rows=10,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "hurdle_noise": terrain.HurdleNoiseTerrainCfg(
            box_height_range=(0.2, 0.5), platform_width=(0.1, 20.0), box_position=(1.5, 0.0),
            random_uniform_terrain_cfg=terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.00, 0.00), noise_step=0.01, border_width=0.0,
                size=(10.0, 10.0),
            )
        )
    }, #type: ignore
    curriculum=True,
)    

@configclass
class LeapEventsCfg(EventCfg):
    # reset_base = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-0.0, 1.5), 
    #             "y": (-1.0, 1.0),  
    #             "roll": (-math.pi/10, math.pi/10),
    #             "pitch": (-math.pi/10, math.pi/10),
    #             "yaw": (-math.pi/10, math.pi/10)
    #         },
    #         "velocity_range": {
    #             "x": (-1.0, 1.0),
    #             "y": (-1.0, 1.0),
    #             "z": (-1.0, 1.0),
    #             "roll": (-0.7, 0.7),
    #             "pitch": (-0.7, 0.7),
    #             "yaw": (-0.7, 0.7),
    #         },  
    #     },
    # )

    # reset_robot_joints = EventTermCfg(
    #     func=custom_mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": {
    #             'R_hip_joint': (-0.1, 0.1),
    #             'R_hip2_joint': (-0.2, 0.2),
    #             'R_thigh_joint': (-0.5, 0.5),
    #             'R_calf_joint': (0.0, 0.5),
    #             'R_toe_joint': (-0.3, 0.3),
    #             'L_hip_joint': (-0.1, 0.1),
    #             'L_hip2_joint': (-0.2, 0.2),
    #             'L_thigh_joint': (-0.5, 0.5),
    #             'L_calf_joint': (0.0, 0.5),
    #             'L_toe_joint': (-0.3, 0.3),
    #         },
    #         "velocity_range": {
    #             'R_hip_joint': (-0.15, 0.15),
    #             'R_hip2_joint': (-0.15, 0.15),
    #             'R_thigh_joint': (-0.15, 0.15),
    #             'R_calf_joint': (-0.15, 0.15),
    #             'R_toe_joint': (-0.15, 0.15),
    #             'L_hip_joint': (-0.15, 0.15),
    #             'L_hip2_joint': (-0.15, 0.15),
    #             'L_thigh_joint': (-0.15, 0.15),
    #             'L_calf_joint': (-0.15, 0.15),
    #             'L_toe_joint': (-0.15, 0.15),
    #         }
    #     },
    # )

    push_robot = None


@configclass
class CommandCfg:
    resampling_time_range = (5.0, 5.0)
    ranges_lin_vel_x = (0.0, 1.5)
    ranges_lin_vel_y = (-0.0, 0.0)
    ranges_ang_vel_z = (-0.5, 0.5)
    heading_control_stiffness = 0.15 #* added heading stiffness

@configclass
class LegLeapEnvCfg(LegPlanarWalkEnvCfg):
    """configurations for the LegLeapEnv inherits from LegPlanarWalkEnvCfg
        
        the modifications are as follows:
        1. event configurations 
            change the random range of thigh joint from previous (-1.7, 0) to (-1.7, 0.78)
            eliminate the push robot event

        2. command configurations - added the heading stiffness
        3. terrain configurations - added the sub_terrains for the leap task
        4. scene configurations - change the environment spacing to 0.0
    """
    curriculum = True

    #* environment configurations
    episode_length_s = 10.0

    #* event configurations
    events = LeapEventsCfg()

    #* command configurations
    commands = CommandCfg()


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
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

    #* scene configurations
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__() #type: ignore
        
        self.observation_noise_model = None
        self.robot.soft_joint_pos_limit_factor = 1.0

        #* reward configuration
        #! encourage reward 
        self.base_height_target = 0.81
        self.lin_vel_reward_scale = 15.0
        self.yaw_rate_reward_scale = 10.0
        self.is_alive_reward_scale = 0.0
        self.flat_orientation_reward_scale = 7.0
        self.base_height_reward_scale = 7.0
        self.joint_regularization_reward_scale = 9.0

        #! penalty reward
        self.first_order_action_rate_reward_scale = -5e-5
        self.second_order_action_rate_reward_scale = -5e-5
        self.energy_consumption_reward_scale = 0.0
        self.undesired_contacts_reward_scale = -20.0
        self.applied_torque_reward_scale = -1e-6
        self.applied_torque_rate_reward_scale = 0.0
        self.joint_pos_limit_reward_scale = 0.0
        self.feet_stumble_reward_scale = -25.0

        #! terminated penalty reward
        self.terminated_penalty_reward_scale = -100.0

@configclass
class LeapEventsPlayCfg(EventCfg):

    push_robot = None

@configclass
class LegLeapPlayEnvCfg(LegLeapEnvCfg):
    episode_length_s = 10.0
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        eye=(-2.5, 2.5, 1.5),
        lookat=(0.0, 0.0, 0.0),
    )
    
    events = LeapEventsPlayCfg()

    def __post_init__(self):
        super().__post_init__() #type: ignore
        # self.commands.heading_control_stiffness = 0.15

