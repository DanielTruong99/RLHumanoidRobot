from omni.isaac.lab.utils import configclass
from .rsl_rl_planar_walk_cfg import LegPlanarWalkPPORunnerCfg

@configclass
class LegLeapPPORunnerCfg(LegPlanarWalkPPORunnerCfg):
    experiment_name = "leg_leap_6"
    num_steps_per_env = 48
    max_iterations = 30000
    # resume = True
    # load_checkpoint = "model_30550.pt"
    # load_run = "2024-10-04_22-42-29"

@configclass
class LegLeapPPOPlayRunnerCfg(LegLeapPPORunnerCfg):
    resume = False
