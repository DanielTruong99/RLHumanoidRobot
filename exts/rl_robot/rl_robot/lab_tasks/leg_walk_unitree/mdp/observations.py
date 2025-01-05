import torch
import numpy as np

from ..walking_robot import WalkingRobotEnv

def get_phase(env: WalkingRobotEnv) -> torch.Tensor:
    sin_phase = torch.sin(2 * np.pi * env.phase ).unsqueeze(1)
    cos_phase = torch.cos(2 * np.pi * env.phase ).unsqueeze(1)
    return torch.cat([sin_phase, cos_phase], dim=-1)