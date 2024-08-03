import torch
import numpy as np
import time
import pybullet
import pybullet_data as pd
from gpugym.envs.leg_amp.amp.poselib.poselib.core.rotation3d import *
from gpugym.envs.leg_amp.amp.poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from gpugym.envs.leg_amp.amp.poselib.poselib.visualization.common import plot_skeleton_state

class CustomSkeletonMotion(SkeletonMotion):
    def __init__(self, tensor_backend, skeleton_tree, is_local, *args, **kwargs):
        super().__init__(tensor_backend, skeleton_tree, is_local, *args, **kwargs)
        self._dof_pos = None 
        self._dof_vel = None 

    @property
    def dof_pos(self):
        return self._dof_pos
    
    @property
    def dof_vel(self):
        return self._dof_vel
    
    def set_dof_state(self, dof_pos):
        self._dof_pos = dof_pos
        self._compute_dof_vel()

    def _compute_dof_vel(self):
        if self._dof_pos is None:
            raise ValueError("dof_pos is not set")
        
        time_delta = 1.0 / self.fps
        theta = self._dof_pos
        self._dof_vel = torch.from_numpy(
            filters.gaussian_filter1d(
                np.gradient(theta, axis=0), 2, axis=0, mode="nearest"
            )
            / time_delta,
        )

''' Pybullet setup'''
p = pybullet
pybullet.connect(pybullet.GUI, options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
pybullet.setAdditionalSearchPath(pd.getDataPath())
pybullet.resetSimulation()
pybullet.setGravity(0, 0, 0)

#* load assets
ground = pybullet.loadURDF('plane.urdf')
leg_robot = pybullet.loadURDF('../../../../assets/urdf/leg_robot/leg.urdf', [0, 0, 0.0], [0, 0, 0, 1])

'''Motion capture setup'''
leg_robot_motion = SkeletonMotion.from_file("data/07_01_cmu_amp.npy")
pelvis_index = leg_robot_motion.skeleton_tree._node_indices['pelvis']
leftfoot_index = leg_robot_motion.skeleton_tree._node_indices['left_foot']
rightfoot_index = leg_robot_motion.skeleton_tree._node_indices['right_foot']


'''Simulation loop'''
num_frames = leg_robot_motion.root_translation.shape[0]
# time_step = 1.0/leg_robot_motion.fps
time_step = 1.0/60
for frame in range(num_frames):
    #* get kinematic information from the motion
    pelvis_pos = leg_robot_motion.global_translation[frame, pelvis_index, ...]
    pelvis_rot = leg_robot_motion.global_root_rotation[frame, ...]
    leftfoot_pos = leg_robot_motion.global_translation[frame, leftfoot_index, ...]
    leftfoot_rot = leg_robot_motion.global_rotation[frame, leftfoot_index, ...]
    rightfoot_pos = leg_robot_motion.global_translation[frame, rightfoot_index, ...]
    rightfoot_rot = leg_robot_motion.global_rotation[frame, rightfoot_index, ...]
    
    #* solve inverse kinematics for the left foot and right foot
    joint_damping_list = [0.1] * pybullet.getNumJoints(leg_robot) 
    joint_lim_low = []; joint_lim_high = []
    for index in range(pybullet.getNumJoints(leg_robot)):
        link_info = pybullet.getLinkState(
            leg_robot,
            index,
            computeLinkVelocity=0,
            computeForwardKinematics=1
        )
        joint_info = pybullet.getJointInfo(leg_robot, index)
        joint_lim_low.append(joint_info[8])
        joint_lim_high.append(joint_info[9])
        
    default_joint_pose = [0.0] * pybullet.getNumJoints(leg_robot)

    #* calculate the left and right toe position and rotation
    pybullet.resetBasePositionAndOrientation(leg_robot, pelvis_pos, pelvis_rot)

    end_effector_index_list = [4, 9] # right toe, left toe
    # relative_leftfoot_pos = leftfoot_pos - pelvis_pos
    leftfoot_pos_target = leftfoot_pos
    # relative_rightfoot_pos = rightfoot_pos - pelvis_pos
    rightfoot_pos_target = rightfoot_pos
    leftfoot_rot_target = leftfoot_rot
    rightfoot_rot_target = rightfoot_rot
    target_pos_list = [rightfoot_pos_target, leftfoot_pos_target]
    target_rot_list = [rightfoot_rot_target, leftfoot_rot_target]
    dof_pos_solution = [0.0] * pybullet.getNumJoints(leg_robot)
    n = 5
    for index, toe_index in enumerate(end_effector_index_list):
        joint_solution = pybullet.calculateInverseKinematics(   
            leg_robot, 
            toe_index,
            targetPosition=target_pos_list[index],
            targetOrientation=target_rot_list[index],
            jointDamping=joint_damping_list,
            lowerLimits=joint_lim_low,
            upperLimits=joint_lim_high,
            restPoses=dof_pos_solution,
            maxNumIterations=100,  
            solver=pybullet.IK_DLS,
        )
        dof_pos_solution[index * n : (index + 1) * n] = list(joint_solution[index * n : (index + 1) * n])
    
    #* set the kinematic information to the robot
    # dof_pos_solution = [0, 0, -torch.pi/3, 0.0, 0.0, 0, 0, -torch.pi/3, 0.0, 0.0]
    for joint_index in range(pybullet.getNumJoints(leg_robot)):
        pybullet.resetJointState(
            bodyUniqueId=leg_robot,
            jointIndex=joint_index,
            targetValue=dof_pos_solution[joint_index]
        )

    #* watting for the time to match the motion
    time.sleep(time_step)


'''End simulation'''
pybullet.disconnect()
