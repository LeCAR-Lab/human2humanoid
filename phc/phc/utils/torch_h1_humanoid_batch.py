import torch 
import numpy as np
import phc.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot

H1_ROTATION_AXIS = torch.tensor([[
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
]])


class Humanoid_Batch:

    def __init__(self, mjcf_file = f"resources/robots/h1/h1.xml", extend_hand = True, extend_head = False, device = torch.device("cpu")):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = extend_hand
        self.extend_head = extend_head
        if extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([15, 19]))).to(device) # Adding the hands joints
            arm_length = 0.3
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim = 0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim = 0)[None, ].to(device)
            self._remove_idx = 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
            
        if extend_head:
            self._remove_idx = 3
            self.model_names = self.model_names + ["head_link"]
            self._parents = torch.cat((self._parents, torch.tensor([0]).to(device))).to(device) # Adding the hands joints
            head_length = 0.75
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim = 1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim = 1).to(device)
            
        
        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        
    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range))
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30): 
        """
        Forward kinematics for batch of poses and translations.
        Args:
            pose (torch.Tensor): (B, seq_len, J, 3, 3)
            trans (torch.Tensor): (B, seq_len, 3)
            convert_to_mat (bool): if True, convert pose to rotation matrix
            return_full (bool): if True, return extended model with hands and head
            dt (float): time step
        Returns:
            dict: with keys:
                global_translation (torch.Tensor): (B, seq_len, J, 3)
                global_rotation_mat (torch.Tensor): (B, seq_len, J, 3, 3)
                global_rotation (torch.Tensor): (B, seq_len, J, 4)
                global_translation_extend (torch.Tensor): (B, seq_len, J+2, 3)
                global_rotation_mat_extend (torch.Tensor): (B, seq_len, J+2, 3, 3)
                global_rotation_extend (torch.Tensor): (B, seq_len, J+2, 4)
                global_root_velocity (torch.Tensor): (B, seq_len, 3)
                global_angular_velocity (torch.Tensor): (B, seq_len, 3)
                local_rotation (torch.Tensor): (B, seq_len, J, 4)
        """.
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2] # 切片获取前两个元素的形状,sequence[start(默认是0):stop(不包含该索引):step(默认是1)]，即 batch 大小和序列长度
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        if self.extend_hand and self.extend_head and pose.shape[-2] == 22:
            pose = torch.cat([pose, torch.zeros(B, seq_len, 1, 3).to(device).type(dtype)], dim = -2) # adding hand and head joints

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
        if pose_mat.shape != 5: # 确保 pose_mat 的形状是 (B, seq_len, J, 3, 3)
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        if self.extend_hand:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            wbody_pos = wbody_pos[..., :-self._remove_idx, :] # 全部关节的位置，如果包含手和头，移除它们的数据
            wbody_mat = wbody_mat[..., :-self._remove_idx, :, :] # 旋转矩阵
            wbody_rot = wbody_rot[..., :-self._remove_idx, :] # 四元数
        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
            
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            if self.extend_hand or self.extend_head:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:][..., :-self._remove_idx] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            else:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim = 1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        # print(expanded_offsets.shape, J)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]]) # 将当前关节的局部偏移量转换为世界坐标系下的位置，然后加上父关节的世界坐标位置，最后得到当前关节的世界坐标位置
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :])) # 将当前关节的旋转角度首先与当前关节的局部旋转矩阵相乘，然后再与父关节的旋转矩阵相乘，得到当前关节的最终的世界旋转矩阵
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2) # 堆叠所有关节的位置信息，在第二个维度上堆叠，从0开始计算,shape[1,1,23,3]
        rotations_world = torch.cat(rotations_world, dim=2) # 在第二个维度上堆叠旋转信息，从0开始计算,shape[1,1,23,3,3]
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        """
        计算速度向量
        
        Args:
            p: 位置数据
            time_delta: 时间间隔
            guassian_filter: 是否应用高斯滤波，默认为True
            
        Returns:
            速度向量
            
        Note:
            该函数用于计算物体在连续时间点之间的速度变化率，
            可选的高斯滤波用于平滑速度计算结果
        """

        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        """
        计算角速度
        
        Args:
            r: 旋转矩阵或四元数表示的姿态变化
            time_delta (float): 时间间隔
            guassian_filter (bool, optional): 是否应用高斯滤波，默认为True
            
        Returns:
            角速度向量 (弧度/秒)
        """

        # assume the second last dimension is the time axis
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity  
