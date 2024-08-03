import glob 
import torch
import numpy as np
from rsl_rl.utils import utils

from __future__ import annotations
from typing import TYPE_CHECKING

from ..amp.poselib.poselib.skeleton.skeleton3d import CustomSkeletonMotion
from omni.isaac.lab.utils import math

class AMPLoader:
    """
        !base_quat, base_lin_vel, base_ang_vel, joint_pos, joint_vel, z_pos, foot_pos
    """
    POS_SIZE = 1 #! just get the z_pos
    ROT_SIZE = 4
    JOINT_POS_SIZE = 10
    TAR_TOE_POS_LOCAL_SIZE = 6
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 10
    TAR_TOE_VEL_LOCAL_SIZE = 6

    # TODO: base_quat index
    ROOT_ROT_START_IDX = 0
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    # TODO: base_lin_vel index
    LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    # TODO: base_ang_vel index
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    # TODO: joint_pos index
    JOINT_POSE_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    # TODO: join_vel index
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    # TODO: z_pos index
    ROOT_POS_START_IDX = JOINT_POSE_END_IDX
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + 1

    # TODO: foot_pos index
    TAR_TOE_POS_LOCAL_START_IDX = ROOT_POS_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    def __init__(
            self, 
            device,
            time_between_frames, 
            data_dir='', 
            preload_transitions=False, 
            num_preload_transitions=1000000, 
            motion_files=glob.glob('learning/custom_rsl_rl/datasets/mocap_motions/*')):
    
        self.device = device
        self.time_between_frames = time_between_frames
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for index, motion_file in enumerate(motion_files):
            # TODO: get trajectory name
            self.trajectory_names.append(motion_file.split('.')[0])

            # TODO: read motion data from file
            current_motion = CustomSkeletonMotion.from_file(motion_file)

            ''' 
                # TODO: create motion_data from SkeletonMotion
                * motion_data: np.array of shape (num_frames, num_features)
                ! motion_data features: 
                  base_quat, base_lin_vel, base_ang_vel, joint_pos, joint_vel, z_pos, foot_pos
            '''
            pelvis_index = current_motion.skeleton_tree._node_indices['pelvis']
            leftfoot_index = current_motion.skeleton_tree._node_indices['left_foot']
            rightfoot_index = current_motion.skeleton_tree._node_indices['right_foot']
            
            base_quat = current_motion.global_rotation[:, pelvis_index, ...]
            
            #* convert to base frame
            base_lin_vel = current_motion.global_root_velocity
            base_lin_vel = math.quat_rotate_inverse(base_quat, base_lin_vel)

            #* convert to base frame
            base_ang_vel = current_motion.global_root_angular_velocity
            base_ang_vel = math.quat_rotate_inverse(base_quat, base_ang_vel)

            z_pose = current_motion.global_translation[:, pelvis_index, 2:3]

            #* calculate the relative foot position and convert to base frame
            leftfoot_pos = current_motion.global_translation[:, leftfoot_index, ...] - current_motion.global_translation[:, pelvis_index, ...]
            rightfoot_pos = current_motion.global_translation[:, rightfoot_index, ...] - current_motion.global_translation[:, pelvis_index, ...]
            leftfoot_pos = math.quat_rotate_inverse(base_quat, leftfoot_pos)
            rightfoot_pos = math.quat_rotate_inverse(base_quat, rightfoot_pos)

            # TODO: add dof_pos data into motion file
            joint_pos = current_motion.dof_pos
            joint_vel = current_motion.dof_vel

            # TODO: formulate the motion_data and append to the tracjectories list
            motion_data = torch.cat((
                base_quat.to(dtype=torch.float32, device=device), 
                base_lin_vel.to(dtype=torch.float32, device=device), 
                base_ang_vel.to(dtype=torch.float32, device=device), 
                joint_pos.to(dtype=torch.float32, device=device), 
                joint_vel.to(dtype=torch.float32, device=device), 
                z_pose.to(dtype=torch.float32, device=device), 
                rightfoot_pos.to(dtype=torch.float32, device=device),
                leftfoot_pos.to(dtype=torch.float32, device=device)
            ), dim=-1)
            self.trajectories.append(motion_data)
            self.trajectories_full.append(motion_data)
            self.trajectory_idxs.append(index)
            self.trajectory_weights.append(1.0)
            self.trajectory_num_frames.append(float(motion_data.shape[0]))
            
            frame_duration = 1.0 / current_motion.fps
            self.trajectory_frame_durations.append(frame_duration)

            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)

            #* Just for logging
            print(f"Loaded {traj_len}s. motion from {motion_file}.")