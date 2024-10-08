from __future__ import annotations
import glob 
import torch
import numpy as np
from rsl_rl.utils import utils

from typing import TYPE_CHECKING

from ..amp.poselib.poselib.skeleton.skeleton3d import CustomSkeletonMotion
from omni.isaac.lab.utils import math
from ..utils import utils

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
            pelvis_index = current_motion.skeleton_tree._node_indices['pelvis'] #type: ignore
            leftfoot_index = current_motion.skeleton_tree._node_indices['left_foot'] #type: ignore
            rightfoot_index = current_motion.skeleton_tree._node_indices['right_foot'] #type: ignore
            
            base_quat = current_motion.global_rotation[:, pelvis_index, ...] #type: ignore
            base_quat = base_quat[:, [3, 0, 1, 2]]
            
            #* convert to base frame
            base_lin_vel = current_motion.global_root_velocity #type: ignore
            base_lin_vel = math.quat_rotate_inverse(base_quat, base_lin_vel)

            #* convert to base frame
            base_ang_vel = current_motion.global_root_angular_velocity #type: ignore
            base_ang_vel = math.quat_rotate_inverse(base_quat, base_ang_vel)

            z_pose = current_motion.global_translation[:, pelvis_index, 2:3] #type: ignore

            #* calculate the relative foot position and convert to base frame
            leftfoot_pos = current_motion.global_translation[:, leftfoot_index, ...] - current_motion.global_translation[:, pelvis_index, ...] #type: ignore
            rightfoot_pos = current_motion.global_translation[:, rightfoot_index, ...] - current_motion.global_translation[:, pelvis_index, ...] #type: ignore
            leftfoot_pos = math.quat_rotate_inverse(base_quat, leftfoot_pos)
            rightfoot_pos = math.quat_rotate_inverse(base_quat, rightfoot_pos)

            # TODO: add dof_pos data into motion file
            joint_pos = current_motion.dof_pos #type: ignore
            joint_vel = current_motion.dof_vel #type: ignore

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
            
            frame_duration = 1.0 / current_motion.fps #type: ignore
            self.trajectory_frame_durations.append(frame_duration)

            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)

            #* Just for logging
            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        rot_blend = utils.quaternion_slerp(
            frame_start[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX],
            frame_end[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX],
            blend
        )
        amp_blend = self.slerp(
            frame_start[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX], 
            frame_end[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX], 
            blend
        )
        return torch.cat([rot_blend, amp_blend], dim=-1)        

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        return self.get_frame_at_time(traj_idx, time)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(int), np.ceil(p * n).astype(int)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), 33, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  33, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_rot_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]
            if torch.isnan(all_frame_rot_starts).any():
                print(f'Found NaN rotation in trajectory {traj_idx}. Skipping...')
            all_frame_rot_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.ROOT_ROT_END_IDX:]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.ROOT_ROT_END_IDX:]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)


    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, :]
                s_next = self.preloaded_s_next[idxs, :]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)