from omni.isaac.lab.utils import configclass
from .rsl_rl_planar_walk_cfg import LegPlanarWalkPPORunnerCfg

@configclass
class LegLeapPPORunnerCfg(LegPlanarWalkPPORunnerCfg):
    experiment_name = "leg_leap"
    resume = True
    load_checkpoint = "model_13199.pt"
    load_run = "2024-09-25_18-29-54"

@configclass
class LegLeapPPOPlayRunnerCfg(LegLeapPPORunnerCfg):
    resume = False
