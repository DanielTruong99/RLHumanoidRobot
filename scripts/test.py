from learning.custom_rsl_rl.datasets.motion_loader import AMPLoader

motion_loader = AMPLoader(device='cuda:0', time_between_frames=0.01)