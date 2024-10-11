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
        "hurdle_noise": terrain.CustomBoxTerrainCfg(
            box_height_range=(0.2, 0.5), platform_width=(0.1, 20.0), box_position=(5.0, 0.0),
        )
    }, #type: ignore
    curriculum=True,
)    

@configclass
class CommandCfg:
    resampling_time_range = (10.0, 10.0)
    ranges_pos_x = (6.5, 9.5)
    ranges_pos_y = (-3.0, 3.0)
    ranges_heading= (-3.14/2, 3.14/2)

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
    #* enable the curriculum
    curriculum = True

    #* environment configurations
    episode_length_s = 10.0

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__() #type: ignore
        
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.commands.resampling_time_range = (self.episode_length_s, self.episode_length_s) #type: ignore

        # self.events.push_robot = None #type: ignore 
        self.observation_noise_model = None
        self.robot.soft_joint_pos_limit_factor = 1.0

        #* reward configuration
        #! encourage reward 
        self.base_height_target = 0.75
        self.position_tracking_reward_scale = 15.0
        self.heading_tracking_reward_scale = 10.0
        self.lin_vel_reward_scale = 0.0
        self.yaw_rate_reward_scale = 0.0
        self.is_alive_reward_scale = 0.0
        self.flat_orientation_reward_scale = 5.0
        self.base_height_reward_scale = 5.0
        self.joint_regularization_reward_scale = 7.0

        #! penalty reward
        self.joint_velocity_reward_scale = -1e-3
        self.joint_acc_reward_scale = -1e-7
        self.first_order_action_rate_reward_scale = -1e-4
        self.second_order_action_rate_reward_scale = -1e-5
        self.energy_consumption_reward_scale = 0.0
        self.undesired_contacts_reward_scale = -1.0
        self.applied_torque_reward_scale = -1e-5
        self.applied_torque_rate_reward_scale = 0.0
        self.joint_pos_limit_reward_scale = 0.0
        self.feet_stumble_reward_scale = -1.0
        self.stand_still_reward_scale = -0.5
        self.stand_still_collision_reward_scale = -1.0

        #! terminated penalty reward
        self.terminated_penalty_reward_scale = -200.0


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

    def __post_init__(self):
        super().__post_init__() #type: ignore
        # self.commands.heading_control_stiffness = 0.15

        self.events.push_robot = None #type: ignore



