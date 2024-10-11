from omni.isaac.lab.utils import configclass
from .rsl_rl_planar_walk_cfg import LegPlanarWalkPPORunnerCfg

@configclass
class LegLeapPPORunnerCfg(LegPlanarWalkPPORunnerCfg):
    experiment_name = "leg_leap_6"
    num_steps_per_env = 24
    max_iterations = 30000
    # resume = True
    # load_checkpoint = "model_4500.pt"
    # load_run = "2024-10-10_02-32-57"

@configclass
class LegLeapPPOPlayRunnerCfg(LegLeapPPORunnerCfg):
    resume = False
    # resume = True
    # load_checkpoint = "model_21050.pt"
    # load_run = "2024-10-10_02-32-57"
