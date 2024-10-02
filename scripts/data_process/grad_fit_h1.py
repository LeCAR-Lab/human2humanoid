import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from torch.autograd import Variable
from tqdm import tqdm

device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

h1_rotation_axis = torch.tensor([[
    [0, 0, 1], # l_hip_yaw
    [1, 0, 0], # l_hip_roll
    [0, 1, 0], # l_hip_pitch
    
    [0, 1, 0], # kneel
    [0, 1, 0], # ankle
    
    [0, 0, 1], # r_hip_yaw
    [1, 0, 0], # r_hip_roll
    [0, 1, 0], # r_hip_pitch
    
    [0, 1, 0], # kneel
    [0, 1, 0], # ankle
    
    [0, 0, 1], # torso
    
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_roll_pitch
    [0, 0, 1], # l_yaw_pitch
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_roll_pitch
    [0, 0, 1], # r_yaw_pitch
    
    [0, 1, 0], # r_elbow
]]).to(device)

h1_joint_names = [ 'pelvis', 
                   'left_hip_yaw_link', 'left_hip_roll_link','left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
                   'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
                   'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
                  'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link']

h1_joint_names_augment = h1_joint_names + ["left_hand_link", "right_hand_link"]
h1_joint_pick = ['pelvis', "left_knee_link", "left_ankle_link",  'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link",]
smpl_joint_pick = ["Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
h1_joint_pick_idx = [ h1_joint_names_augment.index(j) for j in h1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
smpl_parser_n.to(device)

amass_data = joblib.load('amass_copycat_take6_train.pkl') # From PHC
shape_new = joblib.load("data/h1/shape_optimized_v1.pkl").to(device)



h1_fk = Humanoid_Batch(device = device)
data_dump = {}
pbar = tqdm(amass_data.keys())
for data_key in pbar:
    trans = torch.from_numpy(amass_data[data_key]['trans']).float().to(device)
    N = trans.shape[0]
    pose_aa_walk = torch.from_numpy(np.concatenate((amass_data[data_key]['pose_aa'][:, :66], np.zeros((N, 6))), axis = -1)).float().to(device)


    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset


    pose_aa_h1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), N, axis = 1)
    pose_aa_h1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
    pose_aa_h1 = torch.from_numpy(pose_aa_h1).float().to(device)
    gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

    dof_pos = torch.zeros((1, N, 19, 1)).to(device)

    dof_pos_new = Variable(dof_pos, requires_grad=True)
    optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)

    for iteration in range(500):
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
        fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
        
        diff = fk_return['global_translation'][:, :, h1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        loss_g = diff.norm(dim = -1).mean() 
        loss = loss_g
        
        if iteration % 50 == 0:
            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

        optimizer_pose.zero_grad()
        loss.backward()
        optimizer_pose.step()
        
        dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
        
    dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
    pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
    fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])

    root_trans_offset_dump = root_trans_offset.clone()

    root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08

    
    data_dump[data_key]={
            "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().cpu().detach().numpy(),   
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            }

import ipdb; ipdb.set_trace()
joblib.load("data/h1/amass_test.pkl")