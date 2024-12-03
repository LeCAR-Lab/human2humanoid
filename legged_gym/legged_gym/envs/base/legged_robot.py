from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from phc.utils import torch_utils
from isaacgym import gymtorch, gymapi, gymutil
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple, Dict
import copy
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.transform import apply_rotation_to_quat_z
from .legged_robot_config import LeggedRobotCfg
from .lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch

from phc.utils.motion_lib_h1 import MotionLibH1
from phc.learning.network_loader import load_mcp_mlp
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from termcolor import colored
from rl_games.algos_torch import torch_ext
from rsl_rl.modules import VelocityEstimator, VelocityEstimatorGRU
from easydict import EasyDict
from legged_gym.utils import  task_registry
from phc.learning.network_loader import load_mlp
from typing import OrderedDict
import torch.optim as optim
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.viewer.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.self_obs_size = 0
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
    
        self._init_buffers()
        if self.cfg.domain_rand.motion_package_loss:
            offset=(self.env_origins + self.env_origins_init_3Doffset)
            motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
            self.freeze_motion_res = motion_res.copy()
        self._prepare_reward_function()
        self.init_done = True
        self.trajectories = torch.zeros(self.num_envs, 63 * 100).to(self.device) # 19dof + 19dofvel + 3angular velocity + 4projectedgravity + 19lastaction
        self.trajectories_with_linvel = torch.zeros(self.num_envs, 66 * 100).to(self.device) # 19dof + 19dofvel + 3angular velocity + 4projectedgravity + 19lastaction
        if self.cfg.train_velocity_estimation:
            # self.velocity_estimator = VelocityEstimator(63, 512, 256, 3, 25).to(self.device)
            self.velocity_estimator = VelocityEstimatorGRU(63, 512, 3).to(self.device)
            
            
            
            self.velocity_optimizer = optim.Adam(self.velocity_estimator.parameters(), lr=0.00001)

        if self.cfg.use_velocity_estimation:
            load_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/velocity_orand", "velocity_estimator_33000.pt")
            self.velocity_estimator = VelocityEstimator(63, 512, 256, 3, 25).to(self.device)
            self.velocity_estimator.load_state_dict(torch.load(load_path))

        self.prioritize_closing = torch.zeros(self.num_envs)

        # init low pass filter
        if self.cfg.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.num_envs*self.num_actions),
                                                        highcut=np.ones(self.num_envs*self.num_actions) * self.cfg.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.num_envs * self.num_actions, 
                                                        device=self.device)
            
        if self.cfg.motion.teleop:
            self.extend_body_parent_ids = [15, 19]
            self._track_bodies_id = [self._body_list.index(body_name) for body_name in self.cfg.motion.teleop_selected_keypoints_names]
            self._track_bodies_extend_id = self._track_bodies_id + [len(self._body_list), len(self._body_list) + 1]
            self.extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0]]).repeat(self.num_envs, 1, 1).to(self.device)
            if self.cfg.motion.extend_head:
                self.extend_body_parent_ids += [0]
                self._track_bodies_id += [len(self._body_list)]
                self._track_bodies_extend_id += [len(self._body_list) + 2]
                self.extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]]).repeat(self.num_envs, 1, 1).to(self.device)
        self.num_compute_average_epl = self.cfg.rewards.num_compute_average_epl
        self.average_episode_length = 0. # num_compute_average_epl last termination episode length

        self.reset_idx(torch.arange(self.num_envs).to(self.device))
        self.compute_observations() # compute initial obs vuffer. 
        self.start_idx = 0

        if self.cfg.train.distill:
            self.load_expert()
        
        if self.viewer != None:
            self._init_camera()
        self.setup_kin_info()
            
    def setup_kin_info(self):
        if self.cfg.train.distill:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
            "gt_action": torch.zeros([self.num_envs, self.num_actions]).to(self.device),
            }) # current root pos + root for future aggergration
            
            if self.cfg.train.algorithm.get("save_z_noise", False):
                self.kin_dict["z_noise"] = torch.zeros([self.num_envs, self.cfg.train.policy.embedding_size]).to(self.device)

    def load_expert(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.train.runner.resume = True
        cfg.motion.teleop_obs_version = self.cfg.train.distill_model_config.obs_v
        cfg.motion.future_tracks = self.cfg.train.distill_model_config.future_tracks
        cfg.motion.num_traj_samples = self.cfg.train.distill_model_config.num_traj_samples


        load_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", cfg.task, cfg.train.dagger.load_run_dagger, f"model_{cfg.train.dagger.checkpoint_dagger}.pt")
        print("[EXPERT] loading expert policy: ", load_path)
        model_dict = torch.load(load_path)
        activation = "elu"
        actvation_func = torch_utils.activation_facotry(activation)
        model_key = "model_state_dict"
        net_key_name = "actor"
        loading_keys = [k for k in model_dict[model_key].keys() if k.startswith(net_key_name)]
        self.expert_policy = load_mlp(loading_keys, model_dict, actvation_func, model_key = model_key)
        self.expert_policy.to(self.device)
        

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        if self.cfg.train.distill and not self.cfg.env.test:
        # if self.cfg.train.distill :
            if  "expert_policy" in self.__dict__:
                temp_obs_v, temp_future_tracks, temp_num_traj_samples, teleop_selected_keypoints_names, temp_num_privileged_obs, temp_num_observations, temp_add_noise, temp_track_bodies_extend_id\
                    = self.cfg.motion.teleop_obs_version, self.cfg.motion.future_tracks, self.cfg.motion.num_traj_samples, self.cfg.motion.teleop_selected_keypoints_names \
                    , self.cfg.env.num_privileged_obs, self.cfg.env.num_observations, self.add_noise, self._track_bodies_extend_id
                
                self.cfg.motion.teleop_obs_version = self.cfg.train.distill_model_config.obs_v
                self.cfg.motion.future_tracks = self.cfg.train.distill_model_config.future_tracks
                self.cfg.motion.num_traj_samples = self.cfg.train.distill_model_config.num_traj_samples
                self.cfg.motion.teleop_selected_keypoints_names = self.cfg.train.distill_model_config.teleop_selected_keypoints_names
                self.cfg.env.num_privileged_obs = self.cfg.train.distill_model_config.num_privileged_obs
                self.cfg.env.num_observations = self.cfg.train.distill_model_config.num_observations

                
                _track_bodies_extend_id_new = [self._body_list.index(body_name) for body_name in self.cfg.motion.teleop_selected_keypoints_names] + [len(self._body_list), len(self._body_list) + 1]
                _track_bodies_extend_id_new += [len(self._body_list) + 2]
                self._track_bodies_extend_id = _track_bodies_extend_id_new
                self.add_noise = False

                full_obs, full_privilaged_obs = self.compute_self_and_task_obs()
                gt_actions = self.expert_policy(full_obs)
                self.cfg.motion.teleop_obs_version, self.cfg.motion.future_tracks, self.cfg.motion.num_traj_samples , self.cfg.motion.teleop_selected_keypoints_names, self.cfg.env.num_privileged_obs, self.cfg.env.num_observations, self.add_noise,  self._track_bodies_extend_id\
                    = temp_obs_v, temp_future_tracks, temp_num_traj_samples, teleop_selected_keypoints_names, temp_num_privileged_obs, temp_num_observations, temp_add_noise, temp_track_bodies_extend_id 

                self.kin_dict['gt_action'] = gt_actions.clone()



        clip_actions = self.cfg.normalization.clip_actions
        
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        actions = self.actions.clone()
        # actions_copy = actions.clone()

        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(self.num_envs), self.action_delay].clone()
            # import pdb; pdb.set_trace()
        if self.cfg.control.action_filt:
            actions = self.action_filter.filter(actions.reshape(self.num_envs * self.num_actions)).reshape(self.num_envs, self.num_actions)
        else:
            actions = actions.clone()


        self.render()
        # self.actions = actions.clone()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
        if self.cfg.train.distill: # this needs to happen BEFORE the next time-step observation is computed, to collect the "current time-step target"
            self.extras['kin_dict'] = self.kin_dict
        

        

        # ===== UPDATE self.trajectories =====
        dof = self.dof_pos[:]
        dof_vel = self.dof_vel[:]
        base_ang_vel = self.base_ang_vel
        base_gravity = self.projected_gravity
        current_obs_a = torch.cat((dof, dof_vel, base_ang_vel, base_gravity, actions), dim=1)
        self.trajectories[:, 1 * 63 :] = self.trajectories[:, :-1 * 63].clone()
        self.trajectories[:, 0 * 63 : 1 * 63] = current_obs_a.clone()

        lin_vel = self.base_lin_vel
        current_obs_a_with_linvel = torch.cat((dof, dof_vel, lin_vel, base_ang_vel, base_gravity, actions), dim=1)
        self.trajectories_with_linvel[:, 1 * 66 :] = self.trajectories_with_linvel[:, :-1 * 66].clone()
        self.trajectories_with_linvel[:, 0 * 66 : 1 * 66] = current_obs_a_with_linvel.clone()
        if self.cfg.train_velocity_estimation:

            velocity = self.base_lin_vel


            self.ready_for_train_indices = self.episode_length_buf > 25

            # MLP
            train_input = self.trajectories[self.ready_for_train_indices]

            # GRU
            # Reshape A into the desired shape (num_envs, 25, 63)
            # B_reshaped = train_input.reshape(train_input.shape[0], 25, 63)
            B_reshaped = train_input.reshape(train_input.shape[0], 25, 63)

            # Transpose the reshaped array to match the desired rearrangement of axes
            # B_transposed = B_reshaped.transpose(0, 2, 1)

            # Assign the values of the transposed array back to B
            # train_input = current_obs_a.unsqueeze(0).clone() # [batch_size, 63]
            train_input = torch.flip(B_reshaped,dims=[1])
            

            

            if train_input.shape[0] > 0:
                # import ipdb; ipdb.set_trace()
                # train_input = current_obs_a[self.ready_for_train_indices].to(self.device)
                train_target = velocity[self.ready_for_train_indices].to(self.device)

                model_output = self.velocity_estimator(train_input)

                loss = F.mse_loss(model_output, train_target)
                self.velocity_optimizer.zero_grad()
                loss.backward()
                self.velocity_optimizer.step()
                print("Velocity Estimation Loss: ", loss.item())
            # save model
            if self.common_step_counter % 5000 == 0:
                model_dir = "logs/velocity_25_lr0.00001_gru_1"
                if not os.path.exists(os.path.join(LEGGED_GYM_ROOT_DIR, model_dir)):
                    os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, model_dir))
                load_path = os.path.join(LEGGED_GYM_ROOT_DIR, model_dir, "velocity_estimator_" + str(self.common_step_counter) + ".pt")
                torch.save(self.velocity_estimator.state_dict(), load_path)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    




    def _refresh_sim_tensors(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        return
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.last_episode_length_buf = self.episode_length_buf.clone()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.cfg.motion.teleop:
            self._update_recovery_count()
            self._update_package_loss_count()

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])

        self.base_ang_vel[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self._rigid_body_ang_vel[:, 11, :])
        

        self.projected_gravity[:] = quat_rotate_inverse(self._rigid_body_rot[:, 11, :], self.gravity_vec)

        self._post_physics_step_callback()
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_root_pos[:] = self.root_states[:, 0:3]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            
        if self.cfg.env.im_eval:
            offset = self.env_origins + self.env_origins_init_3Doffset
            time = (self.episode_length_buf) * self.dt + self.motion_start_times 
            # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, time, offset)
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, time, offset= offset)

            ref_body_pos_extend = motion_res['rg_pos_t']
            
            body_rot = self._rigid_body_rot
            body_pos = self._rigid_body_pos
        
            extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
            body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

            diff_global_body_pos = ref_body_pos_extend - body_pos_extend
            
            self.extras['mpjpe'] = (diff_global_body_pos).norm(dim=-1).mean(dim=-1)
            self.extras['body_pos'] = body_pos_extend.cpu().numpy()
            self.extras['body_pos_gt'] = ref_body_pos_extend.cpu().numpy()

        
    def begin_seq_motion_samples(self):
        # For evaluation
        self.start_idx = 0
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
        self.reset()            
        
            
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        
        # Termination for knee distance too close
        if self.cfg.asset.terminate_by_knee_distance and self.knee_distance.shape:
            # print("terminate_by knee_distance")
            self.reset_buf |= torch.any(self.knee_distance < self.cfg.asset.termination_scales.min_knee_distance, dim=1)
            #print("Terminated by knee distance: ", torch.sum(self.reset_buf).item())
                    
        # Termination for velocities
        if self.cfg.asset.terminate_by_lin_vel:
            # print("terminate_by lin_vel")
            self.reset_buf |= torch.any(torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > self.cfg.asset.termination_scales.base_vel, dim=1)
            #print("Terminated by lin vel: ", torch.sum(self.reset_buf).item())
        # print(self.reset_buf)

        # Termination for angular velocities
        if self.cfg.asset.terminate_by_ang_vel:
            self.reset_buf |= torch.any(torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > self.cfg.asset.termination_scales.base_ang_vel, dim=1)

        # Termination for gravity in x-direction
        if self.cfg.asset.terminate_by_gravity:
            # print("terminate_by gravity")
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > self.cfg.asset.termination_scales.gravity_x, dim=1)
            
            # Termination for gravity in y-direction
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > self.cfg.asset.termination_scales.gravity_y, dim=1)

        
        # Termination for low height
        if self.cfg.asset.terminate_by_low_height:
            # print("terminate_by low_height")
            self.reset_buf |= torch.any(self.root_states[:, 2:3] < self.cfg.asset.termination_scales.base_height, dim=1)

        if self.cfg.motion.teleop:
            if self.cfg.asset.terminate_by_ref_motion_distance:
                termination_distance = self.cfg.asset.termination_scales.max_ref_motion_distance

                offset = self.env_origins + self.env_origins_init_3Doffset
                time = (self.episode_length_buf) * self.dt + self.motion_start_times 

                motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, time, offset= offset)

                ref_body_pos = motion_res["rg_pos"]
                
                if self.cfg.asset.local_upper_reward:
                    diff =  ref_body_pos[:, [0]] - self._rigid_body_pos[:, [0]]
                    ref_body_pos[:, 11:] -= diff
                             

                if self.cfg.env.test or self.cfg.env.im_eval:
                    reset_buf_teleop = torch.any(torch.norm(self._rigid_body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance, dim=-1)

                else:
                    reset_buf_teleop = torch.any(torch.norm(self._rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1)
                    # self.reset_buf |= torch.any(torch.norm(self._rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1)  # using average, same as UHC"s termination condition
                if self.cfg.motion.teleop: 
                    is_recovery = self._recovery_counter > 0 # give pushed robot time to recover
                    reset_buf_teleop[is_recovery] = 0
                self.reset_buf |= reset_buf_teleop
                
            if self.cfg.asset.terminate_by_1time_motion:
                time = (self.episode_length_buf) * self.dt + self.motion_start_times 
                self.time_out_by_1time_motion = time > self.motion_len # no terminal reward for time-outs
                self.time_out_buf = self.time_out_by_1time_motion
        else:
            self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        self.reset_buf |= self.time_out_buf
        # if self.cfg.motion.teleop: 
        #     is_recovery = self._recovery_counter > 0 # give pushed robot time to recover
        #     self.reset_buf[is_recovery] = 0
     

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # if self.cfg.env.test and len(env_ids) > 0 and not self.cfg.env.im_eval:
            # print("Terminated", self.episode_length_buf[env_ids].numpy())
        
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        if self.cfg.motion.teleop:
            self._recovery_counter[env_ids] = 0
            self._package_loss_counter[env_ids] = 0
            
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        self.update_average_episode_length(env_ids)
        if self.cfg.rewards.sigma_curriculum:
            self._update_sigma_curriculum()
        if self.cfg.rewards.penalty_curriculum:
            self._update_penalty_curriculum()
        if self.cfg.domain_rand.born_offset_curriculum:
            self._update_born_offset_curriculum()
        if self.cfg.domain_rand.born_heading_curriculum:
            self._update_born_heading_curriculum()
        # reset motion selection
        if self.cfg.motion.teleop:
            if self.cfg.motion.curriculum:
                self._update_teleop_curriculum(env_ids)
            self._resample_motion_times(env_ids) #need to resample before reset root states
            # self._update_motion_reference()
        

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        
        

        
        self._episodic_domain_randomization(env_ids)
        #TODO: reset action filter for the env ids  ( n * 19 joint)
        if self.cfg.control.action_filt:

            filter_action_ids_torch = torch.concat([torch.arange(self.num_actions,dtype=torch.int32, device=self.device) + env_id * self.num_actions for env_id in env_ids])
            self.action_filter.reset_hist(filter_action_ids_torch)


        self._resample_commands(env_ids)
        
        if self.cfg.motion.teleop:
            self.base_pos_init[env_ids] = self.root_states[env_ids, :3]

            
            
        
        self.extras['cost'] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.motion.teleop and self.cfg.motion.curriculum:
            self.extras["episode"]["teleop_level"] = torch.mean(self.teleop_levels.float())
        # send timeout info to the algorithm
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0], 
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)
        self._refresh_sim_tensors()


        self.trajectories[env_ids] *= 0
        self.trajectories_with_linvel[env_ids] *= 0
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.cfg.rewards.penalty_reward_names:

                rew *= self.cfg.rewards.penalty_scale

            if self.cfg.asset.zero_out_far and "teleop" in name:
                if name != "teleop_body_position_extend" or name!= "teleop_body_position_extend_upper_0dot5sigma" or name!="teleop_body_position_extend_upper" or name!="teleop_body_position_vr_3keypoints":
                    rew_ignore = ~self.prioritize_closing # prioritize_closing = 1 if distance > close_distance, then ignore dof reward
                    rew *= rew_ignore

            
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        

        

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def update_freeze_ref(self, motion_res, index):
        self.freeze_motion_res["rg_pos"][index]  = motion_res["rg_pos"][index] 
        self.freeze_motion_res["rg_pos_t"][index]  = motion_res["rg_pos_t"][index] 
        self.freeze_motion_res["body_vel"][index]  = motion_res["body_vel"][index]  # [num_envs, num_markers, 3]
        self.ref_body_vel = self.freeze_motion_res["body_vel"][index] 
        self.freeze_motion_res["body_vel_t"][index]  = motion_res["body_vel_t"][index]  # [num_envs, num_markers, 3]
        self.freeze_motion_res["rb_rot"][index]  = motion_res["rb_rot"][index]  # [num_envs, num_markers, 4]
        self.freeze_motion_res["body_ang_vel"][index]  = motion_res["body_ang_vel"][index]  # [num_envs, num_markers, 3]
        self.freeze_motion_res["dof_pos"][index]  = motion_res["dof_pos"][index]  # [num_envs, num_dofs]
        self.freeze_motion_res["dof_vel"][index]  = motion_res["dof_vel"][index]  # [num_envs, num_dofs]

    
    def compute_observations(self):
        self.obs_buf, self.privileged_obs_buf = self.compute_self_and_task_obs()


    def compute_self_and_task_obs(self, ):
        """ Computes observations
        """
        # import ipdb; ipdb.set_trace()
        # print("self.episode_length_buf: ", self.episode_length_buf)
        if self.cfg.motion.teleop:
            offset = self.env_origins + self.env_origins_init_3Doffset
            if self.cfg.motion.future_tracks:
                time_steps = self.cfg.motion.num_traj_samples
                B = self.motion_ids.shape[0]
                time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) / self.cfg.motion.traj_sample_timestep_inv
                motion_times_steps = ((self.episode_length_buf[self.motion_ids, None] + 1) * self.dt + time_internals + self.motion_start_times[self.motion_ids, None]).flatten()  # Next frame, so +1
                env_ids_steps = self.motion_ids.repeat_interleave(time_steps)
                motion_res = self._get_state_from_motionlib_cache_trimesh(env_ids_steps, motion_times_steps, offset= offset.repeat_interleave(time_steps, dim=0).view(-1, 3))
            else:
                B = self.motion_ids.shape[0]
                motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
                motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
            # import ipdb; ipdb.set_trace()
            if self.cfg.domain_rand.motion_package_loss:
                freeze_idx = self._package_loss_counter > 0
                self.update_freeze_ref(motion_res=motion_res, index=~freeze_idx)
                # self.freeze_motion_res[~freeze_idx] = motion_res[~freeze_idx]
                # motion_res[freeze_idx] = self.freeze_motion_res[freeze_idx]
                ref_body_pos = self.freeze_motion_res["rg_pos"] 
                ref_body_pos_extend = self.freeze_motion_res["rg_pos_t"]
                ref_body_vel_subset = self.freeze_motion_res["body_vel"] # [num_envs, num_markers, 3]
                self.ref_body_vel = ref_body_vel_subset
                ref_body_vel_extend = self.freeze_motion_res["body_vel_t"] # [num_envs, num_markers, 3]
                ref_body_rot = self.freeze_motion_res["rb_rot"] # [num_envs, num_markers, 4]
                ref_body_ang_vel = self.freeze_motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
                ref_joint_pos = self.freeze_motion_res["dof_pos"] # [num_envs, num_dofs]
                ref_joint_vel = self.freeze_motion_res["dof_vel"] # [num_envs, num_dofs]
            else:
                ref_body_pos = motion_res["rg_pos"] 
                ref_body_pos_extend = motion_res["rg_pos_t"]
                ref_body_vel_subset = motion_res["body_vel"] # [num_envs, num_markers, 3]
                ref_body_vel = ref_body_vel_subset
                ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
                ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
                ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
                ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
                ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
                ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
                ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]

            
            if self.cfg.asset.local_upper_reward:
                ref_body_pos_extend = ref_body_pos_extend.clone()
                diff =  ref_body_pos_extend[:, [0]] - self._rigid_body_pos[:, [0]]
                ref_body_pos_extend[:, 11:] -= diff
            
            self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)
            
        if self.cfg.motion.teleop:
            if self.cfg.motion.teleop_obs_version == 'v1':
                with torch.no_grad():
                    body_pos = self._rigid_body_pos
                    body_rot = self._rigid_body_rot
                    body_vel = self._rigid_body_vel
                    body_ang_vel = self._rigid_body_ang_vel
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    
                    root_pos = body_pos[..., 0, :]
                    root_rot = body_rot[..., 0, :]
                    root_vel = body_vel[:, 0, :]
                    root_ang_vel = body_ang_vel[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, True) # 122
                    task_obs = compute_imitation_observations(root_pos, root_rot, body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, ref_body_pos, ref_body_rot, ref_root_vel, ref_root_ang_vel,  ref_joint_pos, ref_joint_vel, 1)
                    obs = torch.cat([self_obs, task_obs, self.projected_gravity, self.actions], dim = -1)
            elif self.cfg.motion.teleop_obs_version == 'v-min':
                with torch.no_grad():
                    
                    # robot
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    base_vel = self.base_lin_vel
                    base_ang_vel = self.base_ang_vel
                    base_gravity = self.projected_gravity
                    delta_root_pos = ref_body_pos[:, 0, :] - self.base_pos
                    delta_base_pos = quat_rotate_inverse(self.base_quat, delta_root_pos)[:, :2]
                    
                    forward = quat_apply(self.base_quat, self.forward_vec)
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    
                    ref_forward = quat_apply(ref_body_rot[:, 0, :], self.forward_vec)
                    ref_heading = torch.atan2(ref_forward[:, 1], ref_forward[:, 0])
                    delta_heading = wrap_to_pi(ref_heading - heading).unsqueeze(1)
                    
                    
                    
                    # ref
                    ref_root_rot = ref_body_rot[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    
                    ref_dof_pos = ref_joint_pos
                    ref_dof_vel = ref_joint_vel
                    ref_base_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel)
                    ref_base_ang_vel = quat_rotate_inverse(ref_root_rot, ref_root_ang_vel)
                    ref_base_gravity = quat_rotate_inverse(ref_root_rot, self.gravity_vec)
                    
 
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading,
                                              ref_dof_pos, ref_dof_vel, ref_base_vel, ref_base_ang_vel,ref_base_gravity,
                                              self.actions], dim = -1)
            elif self.cfg.motion.teleop_obs_version == 'v-min2':
                with torch.no_grad():
                    # robot
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    base_vel = self.base_lin_vel
                    base_ang_vel = self.base_ang_vel
                    base_gravity = self.projected_gravity
                    delta_root_pos = ref_body_pos[:, 0, :] - self.base_pos
                    delta_base_pos = quat_rotate_inverse(self.base_quat, delta_root_pos)[:, :2]
                    
                    forward = quat_apply(self.base_quat, self.forward_vec)
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    
                    ref_forward = quat_apply(ref_body_rot[:, 0, :], self.forward_vec)
                    ref_heading = torch.atan2(ref_forward[:, 1], ref_forward[:, 0])
                    delta_heading = wrap_to_pi(ref_heading - heading).unsqueeze(1)
                    
                    
                    
                    # ref
                    ref_root_rot = ref_body_rot[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    
                    ref_dof_pos = ref_joint_pos
                    # ref_dof_vel = ref_joint_vel
                    # ref_base_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel)
                    # ref_base_ang_vel = quat_rotate_inverse(ref_root_rot, ref_root_ang_vel)
                    # ref_base_gravity = quat_rotate_inverse(ref_root_rot, self.gravity_vec)
                    
 
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading,
                                              ref_dof_pos, self.actions], dim = -1)
                    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop':
                with torch.no_grad():
                    
                    # robot
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    base_vel = self.base_lin_vel
                    base_ang_vel = self.base_ang_vel
                    base_gravity = self.projected_gravity

                    delta_root_pos = ref_body_pos[:, 0, :] - self.base_pos
                    delta_base_pos = quat_rotate_inverse(self.base_quat, delta_root_pos)[:, :2]
                    
                    forward = quat_apply(self.base_quat, self.forward_vec)
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    
                    ref_forward = quat_apply(ref_body_rot[:, 0, :], self.forward_vec)
                    ref_heading = torch.atan2(ref_forward[:, 1], ref_forward[:, 0])
                    delta_heading = wrap_to_pi(ref_heading - heading).unsqueeze(1)
                    
                    
                    
                    # ref
                    # ref_root_rot = ref_body_rot[:, 0, :]
                    # ref_root_vel = ref_body_vel[:, 0, :]
                    # ref_base_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel)

                    ########################### BEGIN: compute keypoint pos diff in robot base frame ###########################
                    # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 24dim
                    selected_keypoints_idx = [self._body_list.index(body_name) for body_name in self.cfg.motion.teleop_selected_keypoints_names]

                    body_pos = self._rigid_body_pos
                    body_rot = self._rigid_body_rot
                    body_vel = self._rigid_body_vel
                    body_ang_vel = self._rigid_body_ang_vel
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    
                    root_pos = body_pos[..., 0, :]
                    root_rot = body_rot[..., 0, :]
                    root_vel = body_vel[:, 0, :]
                    root_ang_vel = body_ang_vel[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    
                    task_obs = compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos[:, selected_keypoints_idx, :], ref_body_pos[:, selected_keypoints_idx, :],  1)

                    ####################### END: compute keypoint pos diff in robot base frame ###########################
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, # 19dim + 19dim + 3dim + 3dim + 3dim + 2dim + 1dim
                                             task_obs,  # 3xselected_dim = 18dim
                                              self.actions], dim = -1) # 19dim
                    
            
                    

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-clean':
                with torch.no_grad():
                    
                    # robot
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    base_vel = self.base_lin_vel
                    base_ang_vel = self.base_ang_vel
                    base_gravity = self.projected_gravity

                    # ref
                    # ref_root_rot = ref_body_rot[:, 0, :]
                    # ref_root_vel = ref_body_vel[:, 0, :]
                    # ref_base_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel)

                    ########################### BEGIN: compute keypoint pos diff in robot base frame ###########################
                    # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                    selected_keypoints_idx = [self._body_list.index(body_name) for body_name in self.cfg.motion.teleop_selected_keypoints_names]

                    body_pos = self._rigid_body_pos
                    body_rot = self._rigid_body_rot
                    body_vel = self._rigid_body_vel
                    body_ang_vel = self._rigid_body_ang_vel
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    
                    root_pos = body_pos[..., 0, :]
                    root_rot = body_rot[..., 0, :]
                    root_vel = body_vel[:, 0, :]
                    root_ang_vel = body_ang_vel[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    
                    task_obs = compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos[:, selected_keypoints_idx, :], ref_body_pos[:, selected_keypoints_idx, :],  1)

                    ####################### END: compute keypoint pos diff in robot base frame ###########################
                    
                    
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                                             task_obs,  # 3xselected_dim = 18dim
                                              self.actions], dim = -1) # 19dim
                    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-superclean':
                with torch.no_grad():
                    
                    # robot
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel


                    # ref
                    # ref_root_rot = ref_body_rot[:, 0, :]
                    # ref_root_vel = ref_body_vel[:, 0, :]
                    # ref_base_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel)

                    ########################### BEGIN: compute keypoint pos diff in robot base frame ###########################
                    # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                    selected_keypoints_idx = [self._body_list.index(body_name) for body_name in self.cfg.motion.teleop_selected_keypoints_names]

                    body_pos = self._rigid_body_pos
                    body_rot = self._rigid_body_rot
                    body_vel = self._rigid_body_vel
                    body_ang_vel = self._rigid_body_ang_vel
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    
                    root_pos = body_pos[..., 0, :]
                    root_rot = body_rot[..., 0, :]
                    root_vel = body_vel[:, 0, :]
                    root_ang_vel = body_ang_vel[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    
                    task_obs = compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos[:, selected_keypoints_idx, :], ref_body_pos[:, selected_keypoints_idx, :],  1)

                    ####################### END: compute keypoint pos diff in robot base frame ###########################
                    
                    
                    obs = torch.cat([dof_pos, dof_vel,   # 19dim + 19dim 
                                             task_obs,  # 3xselected_dim = 18dim
                                              self.actions], dim = -1) # 19dim
                    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-clean-nolastaction':
                with torch.no_grad():
                    
                    # robot
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    base_vel = self.base_lin_vel
                    base_ang_vel = self.base_ang_vel
                    base_gravity = self.projected_gravity
                    # ref_root_rot = ref_body_rot[:, 0, :]
                    # ref_root_vel = ref_body_vel[:, 0, :]
                    # ref_base_vel = quat_rotate_inverse(ref_root_rot, ref_root_vel)

                    ########################### BEGIN: compute keypoint pos diff in robot base frame ###########################
                    # ref_keypoint_pos_baseframe including 8 keypoints:  elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                    selected_keypoints_idx = [self._body_list.index(body_name) for body_name in self.cfg.motion.teleop_selected_keypoints_names]

                    body_pos = self._rigid_body_pos
                    body_rot = self._rigid_body_rot
                    body_vel = self._rigid_body_vel
                    body_ang_vel = self._rigid_body_ang_vel
                    dof_pos = self.dof_pos
                    dof_vel = self.dof_vel
                    
                    root_pos = body_pos[..., 0, :]
                    root_rot = body_rot[..., 0, :]
                    root_vel = body_vel[:, 0, :]
                    root_ang_vel = body_ang_vel[:, 0, :]
                    ref_root_vel = ref_body_vel_subset[:, 0, :]
                    ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                    
                    task_obs = compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos[:, selected_keypoints_idx, :], ref_body_pos[:, selected_keypoints_idx, :],  1)

                    ####################### END: compute keypoint pos diff in robot base frame ###########################
                    
                    
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                                             task_obs,  # 3xselected_dim = 18dim
                                             ], dim = -1) 
                    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend':
                
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                ########################### BEGIN: compute keypoint pos diff in robot base frame ###########################
                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_vel = ref_body_vel_subset[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
 
                task_obs = compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos_extend[:, self._track_bodies_extend_id, :], ref_rb_pos_subset,  1)
                
                ####################### END: compute keypoint pos diff in robot base frame ###########################
                
                
                obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                                            task_obs,  # 3xselected_dim = 18dim
                                            self.actions], dim = -1) # 19dim
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-nolinvel':
                
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                ########################### BEGIN: compute keypoint pos diff in robot base frame ###########################
                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_vel = ref_body_vel_subset[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                
                task_obs = compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos_extend[:, self._track_bodies_extend_id, :], ref_rb_pos_subset,  1)
                
                ####################### END: compute keypoint pos diff in robot base frame ###########################
                
                
                obs = torch.cat([dof_pos, dof_vel,  base_ang_vel, base_gravity,  # 19dim + 19dim  + 3dim + 3dim 
                                            task_obs,  # 3xselected_dim = 24dim
                                            self.actions], dim = -1) # 19dim
            
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                
                if self.cfg.asset.zero_out_far:
                    close_distance = self.cfg.asset.close_distance    
                    distance = torch.norm(root_pos - ref_body_pos_extend[..., 0, :], dim=-1)
                    zeros_subset = distance > close_distance
                    self.prioritize_closing = zeros_subset
                    if self.cfg.asset.zero_out_far_change_obs:
                        ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                        ref_body_vel_subset[zeros_subset, :] = base_vel
                        self.point_goal= distance
                        far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                        vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                        ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]
                        
                    
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset, ref_rb_pos_subset, ref_body_vel_subset,  1, ref_episodic_offset = self.ref_episodic_offset)
                obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                                            task_obs,  # 
                                            self.actions], dim = -1) # 19dim
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max_no_vel':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                
                if self.cfg.asset.zero_out_far:
                    close_distance = self.cfg.asset.close_distance    
                    distance = torch.norm(root_pos - ref_body_pos_extend[..., 0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset

                    ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                    ref_body_vel_subset[zeros_subset, :] = base_vel
                    self.point_goal= distance
                    far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                    vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                    ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]
                    
                    
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset, ref_rb_pos_subset, ref_body_vel_subset,  1, ref_episodic_offset = self.ref_episodic_offset, ref_vel_in_task_obs = False)
                obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                                            task_obs,  # 
                                            self.actions], dim = -1) # 19dim
                if self.cfg.env.add_short_history:
                    assert self.cfg.env.short_history_length > 0
                    history_to_be_append = self.trajectories[:, 0:self.cfg.env.short_history_length*63]
                    obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions,
                                                history_to_be_append], dim = -1) # 19dim
                
                else:
                    obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions], dim = -1) # 19dim
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]

                if self.cfg.asset.clip_motion_goal:
                    # import ipdb; ipdb.set_trace()
                    ref_head = ref_rb_pos_subset[:, 2]
                    body_xyz = self.root_states[:, :3]
                    direction_to_body = body_xyz - ref_head
                    xy_direction = direction_to_body[:,:2]
                    distance = torch.norm(xy_direction, dim=1)
                    # import ipdb; ipdb.set_trace()
                    far = distance > self.cfg.asset.clip_motion_goal_distance
                    direction_to_body_norm = F.normalize(direction_to_body[:,:2], p = 2, dim=1)
                    # direction_to_body_norm = xy_direction / 
                    ref_rb_pos_subset[far, 2, :2] = self.root_states[far, :2] - direction_to_body_norm[far] * self.cfg.asset.clip_motion_goal_distance
                
                if self.cfg.asset.zero_out_far: # ref_rb_pos_subset[0], ref_rb_pos_subset[1], head ref_rb_pos_subset[2]
                    close_distance = self.cfg.asset.close_distance  
                    distance = torch.norm(root_pos - ref_body_pos_extend[0::self.cfg.motion.num_traj_samples,0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset
                    if self.cfg.asset.zero_out_far_change_obs:
                        
                        if self.cfg.motion.future_tracks:
                            n = self.cfg.motion.num_traj_samples
                            # import ipdb; ipdb.set_trace()
                            zeros_set_future = zeros_subset.repeat_interleave(n) # zeros_set_future[n * i: n * i + n] = zeros_subset
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            body_pos = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_rb_pos_subset[zeros_set_future, :2] = body_pos.repeat_interleave(n,dim=0)
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            root_vel_ = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            ref_body_vel_subset[zeros_set_future, :] = root_vel_.repeat_interleave(n,dim=0)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            vector_zero_subset_future = torch.zeros(n * len(vector_zero_subset), dtype=torch.bool)
                            vector_zero_subset_future[::n] = vector_zero_subset # vector_zero_subset_future[n * i] = vector_zero_subset[i]
                            vector_zero_subset_future2 = vector_zero_subset.repeat_interleave(n) # vector_zero_subset_future[n * i : n * i + n] = vector_zero_subset[i]
                            dis_new = ((ref_rb_pos_subset[vector_zero_subset_future, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                            ref_rb_pos_subset[vector_zero_subset_future2, 2] = dis_new.repeat_interleave(n,dim=0)
                        
                        else:
                            ref_rb_pos_subset[zeros_subset, :2] = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_body_vel_subset[zeros_subset, :] = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            ref_rb_pos_subset[vector_zero_subset, 2] = ((ref_rb_pos_subset[vector_zero_subset, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                        
                
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                # import ipdb; ipdb.set_trace()
                if self.cfg.motion.realtime_vr_keypoints:
                    ref_rb_pos_subset = self.realtime_vr_keypoints_pos
                    ref_body_vel_subset = self.realtime_vr_keypoints_vel
                    assert self.cfg.motion.num_traj_samples == 1
                
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset, ref_rb_pos_subset, ref_body_vel_subset,  self.cfg.motion.num_traj_samples , ref_episodic_offset = self.ref_episodic_offset)
                

                
                # obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                #                             task_obs,  # 
                #                             self.actions], dim = -1) # 19dim
                if self.cfg.env.add_short_history:
                    assert self.cfg.env.short_history_length > 0
                    history_to_be_append = self.trajectories_with_linvel[:, 0:self.cfg.env.short_history_length*66]
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions,
                                                history_to_be_append], dim = -1) # 19dim
                
                else:
                    obs = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions], dim = -1) # 19dim
                    

                if self.cfg.use_velocity_estimation:
                    self.ready_for_train_indices = self.episode_length_buf > 25
                    current_obs_a = self.trajectories[self.ready_for_train_indices, 0]
                    if current_obs_a.shape[0] > 0:
                        estimate_velocity = self.velocity_estimator(self.trajectories[self.ready_for_train_indices])
                        obs[self.ready_for_train_indices,38:41] = estimate_velocity

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max-nolinvel':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]

                if self.cfg.asset.clip_motion_goal:
                    # import ipdb; ipdb.set_trace()
                    ref_head = ref_rb_pos_subset[:, 2]
                    body_xyz = self.root_states[:, :3]
                    direction_to_body = body_xyz - ref_head
                    xy_direction = direction_to_body[:,:2]
                    distance = torch.norm(xy_direction, dim=1)
                    # import ipdb; ipdb.set_trace()
                    far = distance > self.cfg.asset.clip_motion_goal_distance
                    direction_to_body_norm = F.normalize(direction_to_body[:,:2], p = 2, dim=1)
                    # direction_to_body_norm = xy_direction / 
                    ref_rb_pos_subset[far, 2, :2] = self.root_states[far, :2] - direction_to_body_norm[far] * self.cfg.asset.clip_motion_goal_distance
                
                if self.cfg.asset.zero_out_far: # ref_rb_pos_subset[0], ref_rb_pos_subset[1], head ref_rb_pos_subset[2]
                    close_distance = self.cfg.asset.close_distance  
                    distance = torch.norm(root_pos - ref_body_pos_extend[0::self.cfg.motion.num_traj_samples,0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset
                    if self.cfg.asset.zero_out_far_change_obs:
                        
                        if self.cfg.motion.future_tracks:
                            n = self.cfg.motion.num_traj_samples
                            # import ipdb; ipdb.set_trace()
                            zeros_set_future = zeros_subset.repeat_interleave(n) # zeros_set_future[n * i: n * i + n] = zeros_subset
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            body_pos = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_rb_pos_subset[zeros_set_future, :2] = body_pos.repeat_interleave(n,dim=0)
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            root_vel_ = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            ref_body_vel_subset[zeros_set_future, :] = root_vel_.repeat_interleave(n,dim=0)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            vector_zero_subset_future = torch.zeros(n * len(vector_zero_subset), dtype=torch.bool)
                            vector_zero_subset_future[::n] = vector_zero_subset # vector_zero_subset_future[n * i] = vector_zero_subset[i]
                            vector_zero_subset_future2 = vector_zero_subset.repeat_interleave(n) # vector_zero_subset_future[n * i : n * i + n] = vector_zero_subset[i]
                            dis_new = ((ref_rb_pos_subset[vector_zero_subset_future, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                            ref_rb_pos_subset[vector_zero_subset_future2, 2] = dis_new.repeat_interleave(n,dim=0)
                        
                        else:
                            ref_rb_pos_subset[zeros_subset, :2] = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_body_vel_subset[zeros_subset, :] = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            ref_rb_pos_subset[vector_zero_subset, 2] = ((ref_rb_pos_subset[vector_zero_subset, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                        
                
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                # import ipdb; ipdb.set_trace()
                if self.cfg.motion.realtime_vr_keypoints:
                    ref_rb_pos_subset = self.realtime_vr_keypoints_pos
                    ref_body_vel_subset = self.realtime_vr_keypoints_vel
                    assert self.cfg.motion.num_traj_samples == 1
                
                
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset, ref_rb_pos_subset, ref_body_vel_subset,  self.cfg.motion.num_traj_samples , ref_episodic_offset = self.ref_episodic_offset)
                

                

                if self.cfg.env.add_short_history:
                    assert self.cfg.env.short_history_length > 0
                    history_to_be_append = self.trajectories[:, 0:self.cfg.env.short_history_length*63]
                    obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions,
                                                history_to_be_append], dim = -1) # 19dim
                
                else:
                    obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions], dim = -1) # 19dim
                    
                if self.cfg.use_velocity_estimation:
                    self.ready_for_train_indices = self.episode_length_buf > 25
                    current_obs_a = self.trajectories[self.ready_for_train_indices, :63]
                    if current_obs_a.shape[0] > 0:
                        raise NotImplementedError
                        estimate_velocity = self.velocity_estimator(self.trajectories[self.ready_for_train_indices])
                        obs[self.ready_for_train_indices,38:41] = estimate_velocity
                        
                        
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max-nolinvel-heading':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                body_rot_extend = torch.cat([body_rot, body_rot[:, self.extend_body_parent_ids]], dim=1)
                body_rot_subset = body_rot_extend[:, self._track_bodies_extend_id, :]
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_rb_rot_subset = ref_body_rot_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]

                if self.cfg.asset.clip_motion_goal:
                    # import ipdb; ipdb.set_trace()
                    ref_head = ref_rb_pos_subset[:, 2]
                    body_xyz = self.root_states[:, :3]
                    direction_to_body = body_xyz - ref_head
                    xy_direction = direction_to_body[:,:2]
                    distance = torch.norm(xy_direction, dim=1)
                    # import ipdb; ipdb.set_trace()
                    far = distance > self.cfg.asset.clip_motion_goal_distance
                    direction_to_body_norm = F.normalize(direction_to_body[:,:2], p = 2, dim=1)
                    # direction_to_body_norm = xy_direction / 
                    ref_rb_pos_subset[far, 2, :2] = self.root_states[far, :2] - direction_to_body_norm[far] * self.cfg.asset.clip_motion_goal_distance
                
                if self.cfg.asset.zero_out_far: # ref_rb_pos_subset[0], ref_rb_pos_subset[1], head ref_rb_pos_subset[2]
                    close_distance = self.cfg.asset.close_distance  
                    distance = torch.norm(root_pos - ref_body_pos_extend[0::self.cfg.motion.num_traj_samples,0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset
                    if self.cfg.asset.zero_out_far_change_obs:
                        
                        if self.cfg.motion.future_tracks:
                            n = self.cfg.motion.num_traj_samples
                            # import ipdb; ipdb.set_trace()
                            zeros_set_future = zeros_subset.repeat_interleave(n) # zeros_set_future[n * i: n * i + n] = zeros_subset
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            body_pos = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_rb_pos_subset[zeros_set_future, :2] = body_pos.repeat_interleave(n,dim=0)
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            root_vel_ = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            ref_body_vel_subset[zeros_set_future, :] = root_vel_.repeat_interleave(n,dim=0)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            vector_zero_subset_future = torch.zeros(n * len(vector_zero_subset), dtype=torch.bool)
                            vector_zero_subset_future[::n] = vector_zero_subset # vector_zero_subset_future[n * i] = vector_zero_subset[i]
                            vector_zero_subset_future2 = vector_zero_subset.repeat_interleave(n) # vector_zero_subset_future[n * i : n * i + n] = vector_zero_subset[i]
                            dis_new = ((ref_rb_pos_subset[vector_zero_subset_future, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                            ref_rb_pos_subset[vector_zero_subset_future2, 2] = dis_new.repeat_interleave(n,dim=0)
                        
                        else:
                            ref_rb_pos_subset[zeros_subset, :2] = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_body_vel_subset[zeros_subset, :] = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            ref_rb_pos_subset[vector_zero_subset, 2] = ((ref_rb_pos_subset[vector_zero_subset, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                        
                
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                # import ipdb; ipdb.set_trace()
                if self.cfg.motion.realtime_vr_keypoints:
                    ref_rb_pos_subset = self.realtime_vr_keypoints_pos
                    ref_body_vel_subset = self.realtime_vr_keypoints_vel
                    assert self.cfg.motion.num_traj_samples == 1
                
                task_obs = compute_imitation_observations_teleop_max_heading(root_pos, root_rot, body_pos_subset, body_rot_subset[:, -1, :], ref_rb_pos_subset, ref_rb_rot_subset[:, -1, :], ref_body_vel_subset,  self.cfg.motion.num_traj_samples , ref_episodic_offset = self.ref_episodic_offset)
                

                

                if self.cfg.env.add_short_history:
                    assert self.cfg.env.short_history_length > 0
                    history_to_be_append = self.trajectories[:, 0:self.cfg.env.short_history_length*63]
                    obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions,
                                                history_to_be_append], dim = -1) # 19dim
                
                else:
                    obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim 
                                                task_obs,  # 
                                                self.actions], dim = -1) # 19dim
                    
                if self.cfg.use_velocity_estimation:
                    self.ready_for_train_indices = self.episode_length_buf > 25
                    current_obs_a = self.trajectories[self.ready_for_train_indices, :63]
                    if current_obs_a.shape[0] > 0:
                        raise NotImplementedError
                        estimate_velocity = self.velocity_estimator(self.trajectories[self.ready_for_train_indices])
                        obs[self.ready_for_train_indices,38:41] = estimate_velocity

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-full':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                # print(f"body_pos_extend.shape: {body_pos_extend.shape}")
                # print(f"self._track_bodies_extend_id: {self._track_bodies_extend_id}")
                # track_bodies_extend_id_tensor = torch.tensor(self._track_bodies_extend_id, device=body_pos_extend.device)

                # # Ensure the indices are valid
                # assert torch.all((track_bodies_extend_id_tensor >= 0)), "Index out of bounds1"
                # if not torch.all(track_bodies_extend_id_tensor < body_pos_extend.shape[1]):
                #     print("Error: Index out of bounds in the second dimension")
                #     print(f"track_bodies_extend_id_tensor: {track_bodies_extend_id_tensor}")
                #     print(f"body_pos_extend.shape[1]: {body_pos_extend.shape[1]}")
                #     out_of_bounds_indices = track_bodies_extend_id_tensor[track_bodies_extend_id_tensor >= body_pos_extend.shape[1]]
                #     print(f"Out of bounds indices: {out_of_bounds_indices}")
                #     raise ValueError("Index out of bounds in the second dimension")

                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]

                # print(f"body_rot.shape: {body_rot.shape}")
                # print(f"self.extend_body_parent_ids: {self.extend_body_parent_ids}")
                extend_curr_rot = body_rot[:, self.extend_body_parent_ids].clone()
                # extend_curr_rot = body_rot[:, self.extend_body_parent_ids]
                body_rot_extend = torch.cat([body_rot, extend_curr_rot], dim=1)
                body_rot_subset = body_rot_extend[:, self._track_bodies_extend_id, :]

                body_vel_extend = torch.cat([body_vel, body_vel[:, self.extend_body_parent_ids].clone()], dim=1)
                body_vel_subset = body_vel_extend[:, self._track_bodies_extend_id, :]

                body_ang_vel_extend = torch.cat([body_ang_vel, body_ang_vel[:, self.extend_body_parent_ids].clone()], dim=1)
                body_ang_vel_subset = body_ang_vel_extend[:, self._track_bodies_extend_id, :]

                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_rb_rot_subset = ref_body_rot_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                ref_body_ang_vel_subset = ref_body_ang_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]


                
                if self.cfg.asset.zero_out_far: # ref_rb_pos_subset[0], ref_rb_pos_subset[1], head ref_rb_pos_subset[2]
                    close_distance = self.cfg.asset.close_distance  
                    # import ipdb; ipdb.set_trace()  

                    # distance = torch.norm(root_pos - ref_body_pos_extend[..., 0, :], dim=-1)
                    
                    distance = torch.norm(root_pos - ref_body_pos_extend[0::self.cfg.motion.num_traj_samples,0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset
                    if self.cfg.asset.zero_out_far_change_obs:
                        
                        if self.cfg.motion.future_tracks:
                            n = self.cfg.motion.num_traj_samples
                            # import ipdb; ipdb.set_trace()
                            zeros_set_future = zeros_subset.repeat_interleave(n) # zeros_set_future[n * i: n * i + n] = zeros_subset
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            body_pos = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_rb_pos_subset[zeros_set_future, :2] = body_pos.repeat_interleave(n,dim=0)
                            # print(ref_rb_pos_subset[zeros_set_future, :2])
                            root_vel_ = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            ref_body_vel_subset[zeros_set_future, :] = root_vel_.repeat_interleave(n,dim=0)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            vector_zero_subset_future = torch.zeros(n * len(vector_zero_subset), dtype=torch.bool)
                            vector_zero_subset_future[::n] = vector_zero_subset # vector_zero_subset_future[n * i] = vector_zero_subset[i]
                            vector_zero_subset_future2 = vector_zero_subset.repeat_interleave(n) # vector_zero_subset_future[n * i : n * i + n] = vector_zero_subset[i]
                            dis_new = ((ref_rb_pos_subset[vector_zero_subset_future, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                            ref_rb_pos_subset[vector_zero_subset_future2, 2] = dis_new.repeat_interleave(n,dim=0)
                        
                        else:
                            ref_rb_pos_subset[zeros_subset, :2] = body_pos_subset[zeros_subset, :2] # two hands\
                            ref_body_vel_subset[zeros_subset, :] = root_vel[zeros_subset].unsqueeze(1).repeat(1, 3, 1)
                            self.point_goal= distance
                            far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                            vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                            ref_rb_pos_subset[vector_zero_subset, 2] = ((ref_rb_pos_subset[vector_zero_subset, 2] - body_pos_subset[vector_zero_subset, 2]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 2]
                        
                
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                # import ipdb; ipdb.set_trace()
                if self.cfg.motion.realtime_vr_keypoints:
                    ref_rb_pos_subset = self.realtime_vr_keypoints_pos
                    ref_body_vel_subset = self.realtime_vr_keypoints_vel
                    assert self.cfg.motion.num_traj_samples == 1

                
                self_obs = compute_humanoid_observations_max_full(body_pos_extend, body_rot_extend, body_vel_extend, body_ang_vel_extend, True, False) # 342
                # 22 * 3 + 23 * 6 + 23 * 3 + 23 * 3  = 342 | pos, rot, vel, ang_vel
                task_obs = compute_imitation_observations_max_full(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset,  ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset,   \
                                                                   self.cfg.motion.num_traj_samples, ref_episodic_offset = self.ref_episodic_offset)
                 # 23 * 3 + 23 * 6 + 23 * 3 + 23 * 3 + 23 * 3 + 23 * 6  = 552 diff pos, rot, vel, ang_vel | pos, rot
                
                obs = torch.cat([ self_obs, 
                                            task_obs,  # 
                                            self.actions], dim = -1) # 342 + 552 + 19 = 913
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-nolinvel':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                
                if self.cfg.asset.zero_out_far:
                    close_distance = self.cfg.asset.close_distance    
                    distance = torch.norm(root_pos - ref_body_pos_extend[..., 0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset

                    ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                    ref_body_vel_subset[zeros_subset, :] = base_vel
                    self.point_goal= distance
                    far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                    vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                    ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]
                    
                    
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset, ref_rb_pos_subset, ref_body_vel_subset,  1)
                obs = torch.cat([dof_pos, dof_vel, base_ang_vel, base_gravity,  # 19dim + 19dim + (no 3dim) + 3dim + 3dim 
                                            task_obs,  # 
                                            self.actions], dim = -1) # 19dim
                

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-acc':
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                
                extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                body_pos_subset = body_pos_extend[:, self._track_bodies_extend_id, :]
                
                
                ref_rb_pos_subset = ref_body_pos_extend[:, self._track_bodies_extend_id]
                ref_body_vel_subset = ref_body_vel_extend[:, self._track_bodies_extend_id]
                
                # robot
                dof_pos = self.dof_pos
                dof_vel = self.dof_vel
                base_vel = self.base_lin_vel
                last_base_vel = self.last_base_lin_vel
                base_acc = (base_vel - last_base_vel) / self.dt
                base_ang_vel = self.base_ang_vel
                base_gravity = self.projected_gravity

                # ref_keypoint_pos_baseframe including 8 keypoints: handx2, elbowx2, shoulderx2, anklex2, 3dimx8keypoints = 18dim
                root_pos = body_pos[..., 0, :]
                root_rot = body_rot[..., 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                ref_root_ang_vel = ref_body_ang_vel[:, 0, :]
                if self.cfg.asset.zero_out_far:
                    close_distance = self.cfg.asset.close_distance    
                    distance = torch.norm(root_pos - ref_body_pos_extend[..., 0, :], dim=-1)
                    zeros_subset = distance > close_distance

                    self.prioritize_closing = zeros_subset

                    ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                    ref_body_vel_subset[zeros_subset, :] = base_vel
                    self.point_goal= distance
                    far_distance = self.cfg.asset.far_distance  # does not seem to need this in particular...
                    vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                    ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]
                    
                    
                # self_obs = compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, True, False) # 222
                task_obs = compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos_subset, ref_rb_pos_subset, ref_body_vel_subset,  1)
                obs = torch.cat([dof_pos, dof_vel, base_acc, base_ang_vel, base_gravity,  # 19dim + 19dim + 3dim + 3dim + 3dim 
                                            task_obs,  # 
                                            self.actions], dim = -1) # 19dim                
                    
                

            else:
                raise NotImplementedError
        else:
            obs = torch.cat((  
                                    # self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        

        
        obs_buf_denoise = obs.clone()
        
        # add noise if needed
        if self.add_noise:
            # print("Adding noise")
            # print("Noise scale: ", self.noise_scale_vec)
            # print("before: ", self.obs_buf)
            noise_rescale = torch.ones(self.num_envs, 1, device=self.device)
            if self.cfg.motion.teleop:
                if self.cfg.motion.curriculum and self.cfg.motion.obs_noise_by_curriculum:
                    noise_rescale = self.teleop_levels.unsqueeze(1) / 10.
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec * noise_rescale


        
        if self.cfg.env.num_privileged_obs is not None:
            # privileged obs
            self.privileged_info = torch.cat([
                self._base_com_bias,
                self._ground_friction_values[:, self.feet_indices],
                self._link_mass_scale,
                self._kp_scale,
                self._kd_scale,
                self._rfi_lim_scale,
                self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, 6),
                torch.clamp_max(self._recovery_counter.unsqueeze(1), 1),
            ], dim=1)
            privileged_obs_buf = torch.cat([obs_buf_denoise, self.privileged_info], dim=1)

        return obs, privileged_obs_buf
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _init_height_points(self):
       """ Returns points at which the height measurments are sampled (in base frame)


       Returns:
           [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
       """
       y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
       x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
       grid_x, grid_y = torch.meshgrid(x, y)


       self.num_height_points = grid_x.numel()
       points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
       points[:, :, 0] = grid_x.flatten()
       points[:, :, 1] = grid_y.flatten()
       return points



    def _get_heights(self, position = None, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw


        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.


        Raises:
            NameError: [description]


        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
        #    points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
        #                            self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            points = (quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + position.view(self.num_envs, -1, 3))
        else:
        #    points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
        #    self.root_states[:, :3]).unsqueeze(1)
            points = (quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (position).view(self.num_envs, -1, 3)) 


        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)


        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)


        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _update_realtime_vr_keypoints(self, keypoints_pos, keypoints_vel):
        """ Updates the keypoints in the simulation based on the VR input
        """
        self.realtime_vr_keypoints_pos = torch.Tensor(keypoints_pos).to(self.device)
        self.realtime_vr_keypoints_vel = torch.Tensor(keypoints_vel).to(self.device)


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """   
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # import ipdb; ipdb.set_trace()
        for env_id in range(self.num_envs):
            if self.cfg.motion.teleop:
                if self.cfg.motion.realtime_vr_keypoints:
                    for vr_keypoint_idx in range(self.realtime_vr_keypoints_pos.shape[0]):
                        color_inner = [1, 0.651, 0]
                        color_inner = tuple(color_inner)
                        sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04, 20, 20, None, color=color_inner)
                        sphere_pose = gymapi.Transform(gymapi.Vec3(self.realtime_vr_keypoints_pos[vr_keypoint_idx, 0], self.realtime_vr_keypoints_pos[vr_keypoint_idx, 1], self.realtime_vr_keypoints_pos[vr_keypoint_idx, 2]), r=None)
                        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)
                else:
                    for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                        
                        color_inner = (0.3, 0.3, 0.3) if not self.cfg.motion.visualize_config.customize_color \
                                                        else self.cfg.motion.visualize_config.marker_joint_colors[pos_id % len(self.cfg.motion.visualize_config.marker_joint_colors)]
                        color_inner = tuple(color_inner)
                        sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04, 20, 20, None, color=color_inner)
                        # sphere_geom3 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1., 1.0, 1.0))
                        if self.cfg.domain_rand.randomize_motion_ref_xyz:
                            # import ipdb; ipdb.set_trace()
                            # pos_joint[0] += self.ref_episodic_offset[env_id][0]
                            # pos_joint[1] += self.ref_episodic_offset[env_id][1]
                            # pos_joint[2] += self.ref_episodic_offset[env_id][2]
                            # import ipdb; ipdb.set_trace()
                            if pos_id == 22:
                                pos_joint += self.ref_episodic_offset[env_id]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
                        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose) 
                        # gymutil.draw_lines(sphere_geom3, self.gym, self.viewer, self.envs[env_id], sphere_pose) 

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                # import pdb; pdb.set_trace()
                self._ground_friction_values[env_id, s] += self.friction_coeffs[env_id].squeeze()
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (before randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (before randomization)")
        # print()
        
        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            torso_index = self._body_list.index("torso_link")
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.x[0], self.cfg.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.y[0], self.cfg.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.z[0], self.cfg.domain_rand.base_com_range.z[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i, body_name in enumerate(self.cfg.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.cfg.domain_rand.link_mass_range[0], self.cfg.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            raise Exception("index 0 is for world, 13 is for torso!")
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (after randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (afters randomization)")
        # print()

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        # if self.cfg.motion.teleop:
            # self.motion_times += self.dt # TODO: align with motion_dt. ZL: don't need that, motion lib will handle it. 
            # self._update_motion_reference()
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            if self.cfg.motion.curriculum and self.cfg.motion.push_robot_by_curriculum:
                mean_teleop_level = (self.teleop_levels/10.).mean()
                if torch.rand(1).to('cuda') < mean_teleop_level:
                    self._push_robots()
            else:
                self._push_robots()
        if self.cfg.domain_rand.motion_package_loss and  (self.common_step_counter % self.cfg.domain_rand.package_loss_interval == 0):
            self._freeze_ref_motion()
        if self.cfg.motion.teleop and (self.common_step_counter % self.cfg.motion.resample_motions_for_envs_interval == 0):
            if self.cfg.motion.resample_motions_for_envs:
                print("Resampling motions for envs")
                print("common_step_counter: ", self.common_step_counter)
                self.resample_motion()
            

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self._kd_scale * self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.dof_vel) - self._kd_scale * self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.cfg.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.cfg.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        if self.cfg.motion.teleop:
            
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins + self.env_origins_init_3Doffset

            # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
            
            self.dof_pos[env_ids] = motion_res['dof_pos'][env_ids]
            self.dof_vel[env_ids] = motion_res['dof_vel'][env_ids]
            
        else:
            self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # env_ids_int32 = torch.cat([env_ids_int32 * (self._num_teleop_markers+1) + _actor for _actor in range(self._num_teleop_markers+1)], dim=0).to(dtype=torch.int32)
        if self.cfg.motion.teleop and self.cfg.motion.visualize:
            env_ids_int32 *= (self.cfg.motion.num_markers+1)
                
        # print("before reset dof"); import pdb; pdb.set_trace()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        
        # print("after reset dof"); import pdb; pdb.set_trace()
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # trimesh
            if self.cfg.motion.teleop:

                motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
                offset = self.env_origins + self.env_origins_init_3Doffset
                # import ipdb; ipdb.set_trace()
                # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
                motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
                
                self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
                # print("root",motion_res['root_pos'][env_ids])
                # self.root_states[env_ids, 2] += 0.03 # in case under the terrain
                self.root_states[env_ids, 2] += 0.04 # in case under the terrain

                # self.root_states[env_ids, 0] += 5.0 # in case under the terrain
                # self.root_states[env_ids, 1] += 5.0 # in case under the terrain
                if self.cfg.domain_rand.born_offset:
                    rand_num = np.random.rand()
                    if rand_num < self.cfg.domain_rand.born_offset_possibility:
                        randomize_distance = torch_rand_float(-self.cfg.domain_rand.born_distance, self.cfg.domain_rand.born_distance, (len(env_ids), 2), device=self.device)
                        # import ipdb; ipdb.set_trace()
                        # randomize_distance = torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
                        self.root_states[env_ids, :2] += randomize_distance
                        # self.root_states[env_ids, :2] += torch_rand_float(self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1], (len(env_ids), 2), device=self.device)
                self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids] # ZL: use random velicty initation should be more robust? 
                self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]


                if self.cfg.domain_rand.born_heading_randomization:
                    random_angles_rad_axis = torch.zeros(len(env_ids),3, device=self.device)
                    random_angles = (torch.rand((len(env_ids),), device=self.device) * (2 * self.cfg.domain_rand.born_heading_degree) - self.cfg.domain_rand.born_heading_degree)
                    # random_angles = torch_rand_float(-self.cfg.domain_rand.born_heading_degree, self.cfg.domain_rand.born_heading_degree, (len(env_ids),1), device=self.device).squeeze(-1)
                    # random_angles = torch.rand(len(env_ids)) * 0 - 180
                    # import ipdb; ipdb.set_trace()
                    random_angles_rad = torch.deg2rad(random_angles)
                    # random_angles_rad = random_angles * 0
                    # print("random_angles_rad_axis shape", random_angles_rad_axis.shape)
                    # print("env_ids= ", env_ids)
                    
                    random_angles_rad_axis[:, 0] = random_angles_rad
                    # random_angles_rad_axis[env_ids, 0] = 2.7
                    
                    self.root_states[env_ids, 3:7] = apply_rotation_to_quat_z(self.root_states[env_ids, 3:7], random_angles_rad_axis)
                # import ipdb; ipdb.set_trace()
                # self.measured_heights = self._get_heights().reshape((self.num_envs))
                # delta_height = self.measured_heights[env_ids] - offset[env_ids, 2]
                # self.root_states[env_ids, 2] += delta_height
                # motion_res['root_pos'][env_ids,2] += delta_height
                
                self._rigid_body_pos[env_ids] = motion_res['rg_pos'][env_ids]
                self._rigid_body_rot[env_ids] = motion_res['rb_rot'][env_ids]
                self._rigid_body_vel[env_ids] =   motion_res['body_vel'][env_ids]
                self._rigid_body_ang_vel[env_ids] = motion_res['body_ang_vel'][env_ids]
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
                self.root_states[env_ids, 7:13].uniform_(-0.5, 0.5) # random base twist
        else:
            if self.cfg.motion.teleop:
                # import pdb; pdb.set_trace()
                motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
                offset = self.env_origins + self.env_origins_init_3Doffset
                # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
                motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
                
                
                self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
                self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids] # ZL: use random velicty initation should be more robust? 
                self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
                
                self._rigid_body_pos[env_ids] = motion_res['rg_pos'][env_ids]
                self._rigid_body_rot[env_ids] = motion_res['rb_rot'][env_ids]
                self._rigid_body_vel[env_ids] =   motion_res['body_vel'][env_ids]
                self._rigid_body_ang_vel[env_ids] = motion_res['body_ang_vel'][env_ids]
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, 7:13].uniform_(-0.5, 0.5) # random base twist
            
        # base velocities
        
        # import pdb; pdb.set_trace()
        # if self.cfg.motion.teleop:
        #     assert len(env_ids) != 0
        #     self.root_states[env_ids, 3:7] += self.ref_base_rot_init[env_ids]
        # self.root_states[env_ids, 7:10] = torch_rand_float(-self.cfg.init_state.max_linvel, self.cfg.init_state.max_linvel, (len(env_ids), 3), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # self.root_states[env_ids, 10:13] = torch_rand_float(-self.cfg.init_state.max_angvel, self.cfg.init_state.max_angvel, (len(env_ids), 3), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        
        # env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).cuda()
        env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).to(self.device)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        if self.cfg.motion.teleop: self._recovery_counter[:] = 60 # 60 steps for the robot to stabilize
    
    def _freeze_ref_motion(self):
        if self.cfg.motion.teleop:
            # import ipdb; ipdb.set_trace()
            package_loss_random_time = np.random.randint(self.cfg.domain_rand.package_loss_range[0], self.cfg.domain_rand.package_loss_range[1] + 1)
            self._package_loss_counter[:] = package_loss_random_time 


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if self.cfg.motion.teleop:
            # import ipdb; ipdb.set_trace()
            teleop_distance = torch.norm(self._rigid_body_pos[env_ids] - ref_body_pos[env_ids], dim=-1).mean(dim=-1) # shape [num_envs]
            move_up = teleop_distance < self.cfg.motion.terrain_level_down_distance / 5 
            move_down = teleop_distance > self.cfg.motion.terrain_level_down_distance
            # import ipdb; ipdb.set_trace()
        else:
            distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            # robots that walked far enough progress to harder terains
            move_up = distance > self.terrain.env_length / 2
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up

        #import ipdb; ipdb.set_trace()
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def update_average_episode_length(self, env_ids):
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def _update_sigma_curriculum(self):
        # import ipdb; ipdb.set_trace()
        if self.average_episode_length < self.cfg.rewards.reward_position_sigma_level_up_threshold:
            self.cfg.rewards.teleop_body_pos_upperbody_sigma *= (1 + self.cfg.rewards.level_degree)
        elif self.average_episode_length > self.cfg.rewards.reward_position_sigma_level_down_threshold:
            self.cfg.rewards.teleop_body_pos_upperbody_sigma *= (1 - self.cfg.rewards.level_degree)
        self.cfg.rewards.teleop_body_pos_upperbody_sigma = np.clip(self.cfg.rewards.teleop_body_pos_upperbody_sigma, self.cfg.rewards.teleop_body_pos_upperbody_sigma_range[0], self.cfg.rewards.teleop_body_pos_upperbody_sigma_range[1])
    
    def _update_penalty_curriculum(self):
        if self.average_episode_length < self.cfg.rewards.penalty_level_down_threshold:
            self.cfg.rewards.penalty_scale *= (1 - self.cfg.rewards.level_degree)
        elif self.average_episode_length > self.cfg.rewards.penalty_level_up_threshold:
            self.cfg.rewards.penalty_scale *= (1 + self.cfg.rewards.level_degree)
        self.cfg.rewards.penalty_scale = np.clip(self.cfg.rewards.penalty_scale, self.cfg.rewards.penalty_scale_range[0], self.cfg.rewards.penalty_scale_range[1])
    
    def _update_born_offset_curriculum(self):
        if self.average_episode_length < self.cfg.domain_rand.born_offset_level_down_threshold:
            self.cfg.domain_rand.born_distance *= (1 - self.cfg.domain_rand.level_degree)
        elif self.average_episode_length > self.cfg.domain_rand.born_offset_level_up_threshold:
            self.cfg.domain_rand.born_distance *= (1 + self.cfg.domain_rand.level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        self.cfg.domain_rand.born_distance = np.clip(self.cfg.domain_rand.born_distance, self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
    
    def _update_born_heading_curriculum(self):
        if self.average_episode_length < self.cfg.domain_rand.born_heading_level_down_threshold:
            self.cfg.domain_rand.born_heading_degree *= (1 - self.cfg.domain_rand.born_heading_level_degree)
        elif self.average_episode_length > self.cfg.domain_rand.born_heading_level_up_threshold:
            self.cfg.domain_rand.born_heading_degree *= (1 + self.cfg.domain_rand.born_heading_level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        self.cfg.domain_rand.born_heading_degree = np.clip(self.cfg.domain_rand.born_heading_degree, self.cfg.domain_rand.born_heading_range[0], self.cfg.domain_rand.born_heading_range[1])
    
    def _update_teleop_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        if self.cfg.motion.teleop:
            episode_length_buf = self.last_episode_length_buf[env_ids]
            move_up = episode_length_buf > self.cfg.motion.teleop_level_up_episode_length
            move_down = episode_length_buf < self.cfg.motion.teleop_level_down_episode_length
        else:
            raise NotImplementedError
        #import ipdb; ipdb.set_trace()
        self.teleop_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.teleop_levels[env_ids] = torch.where(self.teleop_levels[env_ids]>=10, # (the maximum level is nine)
                                                   torch.randint_like(self.teleop_levels[env_ids], 10), # (the maximum level is nine)
                                                   torch.clip(self.teleop_levels[env_ids], 0)) # (the minumum level is zero)

    

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        if self.cfg.motion.teleop:
            # noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            # noise_vec[3:6] = noise_scales.gravity * noise_level
            # noise_vec[6                       :   6+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[6+  self.num_actions    :   6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            # noise_vec[6+2*self.num_actions    :   6+3*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[6+3*self.num_actions    :   6+4*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel 
            # noise_vec[6+4*self.num_actions    :                       ] = 0. # previous actions, commands
            if self.cfg.motion.teleop_obs_version == 'v1':

                # SELF_OBSERVATION
                root_height_obs_end_idx = 1
                noise_vec[0: root_height_obs_end_idx] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements # [0: 1]

                body_pos_end_idx = 1 + self.num_actions*3 # 3x19 -> 57dim
                noise_vec[0: body_pos_end_idx] = noise_scales.body_pos * noise_level * self.obs_scales.body_pos # [1: 58]

                body_rot_end_idx = body_pos_end_idx + self.num_actions*6 + 6 # 6x20 -> 120dim  
                noise_vec[body_pos_end_idx: body_rot_end_idx] = noise_scales.body_rot * noise_level * self.obs_scales.body_rot # [58: 178]

                root_lin_vel_end_idx = body_rot_end_idx + 3 # 3dim
                noise_vec[body_rot_end_idx: root_lin_vel_end_idx] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel # [178:181]

                root_ang_vel_end_idx = root_lin_vel_end_idx + 3 # 3dim
                noise_vec[root_lin_vel_end_idx: root_ang_vel_end_idx] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # [181:184]

                dof_pos_end_idx = root_ang_vel_end_idx + self.num_actions # 1x19 -> 19dim
                noise_vec[root_ang_vel_end_idx: dof_pos_end_idx] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # [184:203]

                dof_vel_end_idx = dof_pos_end_idx + self.num_actions # 1x19 -> 19dim
                noise_vec[dof_pos_end_idx: dof_vel_end_idx] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # [203:222]

                # TAKS OBSERVATION
                diff_local_body_pos_flat_end_idx = dof_vel_end_idx + self.num_actions*3 + 3 # 3x20 -> 60dim
                noise_vec[dof_vel_end_idx: diff_local_body_pos_flat_end_idx] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos # [222:282]

                diff_local_body_rot_flat_end_idx = diff_local_body_pos_flat_end_idx + self.num_actions*6 + 6 # 6x20 -> 120dim
                noise_vec[diff_local_body_pos_flat_end_idx: diff_local_body_rot_flat_end_idx] = noise_scales.ref_body_rot * noise_level * self.obs_scales.body_rot # [282:402]

                diff_local_lin_vel_end_idx = diff_local_body_rot_flat_end_idx + 3 # 3dim
                noise_vec[diff_local_body_rot_flat_end_idx: diff_local_lin_vel_end_idx] = noise_scales.ref_lin_vel * noise_level * self.obs_scales.lin_vel # [402:405]

                diff_local_ang_vel_end_idx = diff_local_lin_vel_end_idx + 3 # 3dim
                noise_vec[diff_local_lin_vel_end_idx: diff_local_ang_vel_end_idx] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.ang_vel # [405:408]

                diff_local_ref_body_pos_root_end_idx = diff_local_ang_vel_end_idx + self.num_actions*3 + 3 # 3x20 -> 60dim
                noise_vec[diff_local_ang_vel_end_idx: diff_local_ref_body_pos_root_end_idx] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos # [408:468]

                diff_local_ref_body_rot_root_end_idx = diff_local_ref_body_pos_root_end_idx + self.num_actions*6 + 6 # 6x20 -> 120dim
                noise_vec[diff_local_ref_body_pos_root_end_idx: diff_local_ref_body_rot_root_end_idx] = noise_scales.ref_body_rot * noise_level * self.obs_scales.body_rot # [468:588]

                diff_dof_pos_end_idx = diff_local_ref_body_rot_root_end_idx + self.num_actions # 1x19 -> 19dim
                noise_vec[diff_local_ref_body_rot_root_end_idx: diff_dof_pos_end_idx] = noise_scales.ref_dof_pos * noise_level * self.obs_scales.dof_pos # [588:607]

                diff_dof_vel_end_idx = diff_dof_pos_end_idx + self.num_actions # 1x19 -> 19dim
                noise_vec[diff_dof_pos_end_idx: diff_dof_vel_end_idx] = noise_scales.ref_dof_vel * noise_level * self.obs_scales.dof_vel # [607:626]

                # PROJECTED GRAVITY
                projected_gravity_end_idx = diff_dof_vel_end_idx + 3 # 3dim
                noise_vec[diff_dof_vel_end_idx: projected_gravity_end_idx] = noise_scales.gravity * noise_level # [626:629]
                #import ipdb; ipdb.set_trace()
                # LAST ACTION
                last_action_end_idx = projected_gravity_end_idx + self.num_actions
                noise_vec[projected_gravity_end_idx: last_action_end_idx] = 0. # [629:648]
            elif self.cfg.motion.teleop_obs_version == 'v-min':
                # self.obs_buf = torch.cat([dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading,
                #             ref_dof_pos, ref_dof_vel, ref_base_vel, ref_base_ang_vel,ref_base_gravity,
                #             self.actions], dim = -1)
                # raise NotImplementedError
                # print(colored("Not Implemented", "red"))
                # dof_pos
                noise_vec[0                   : self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # x y heading targets
                noise_vec[2*self.num_dof + 9  : 2*self.num_dof + 12] = 0.
                
                # ref dof pos
                noise_vec[2*self.num_dof + 12 : 3*self.num_dof + 12] = noise_scales.ref_dof_pos * noise_level * self.obs_scales.dof_pos
                # ref dof vel
                noise_vec[3*self.num_dof + 12 : 4*self.num_dof + 12] = noise_scales.ref_dof_vel * noise_level * self.obs_scales.dof_vel
                # ref base vel
                noise_vec[4*self.num_dof + 12 : 4*self.num_dof + 15] = noise_scales.ref_lin_vel * noise_level * self.obs_scales.lin_vel
                # ref base ang vel
                noise_vec[4*self.num_dof + 15 : 4*self.num_dof + 18] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.ang_vel
                # ref base gravity
                noise_vec[4*self.num_dof + 18 : 4*self.num_dof + 21] = noise_scales.ref_gravity * noise_level
                
                # self.actions
                noise_vec[4*self.num_dof + 21 : ] = 0.                
            elif self.cfg.motion.teleop_obs_version == 'v-min2':
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_dof_pos, self.actions   
                # dof_pos
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # delta base pos dim2
                noise_vec[2*self.num_dof + 9  : 2*self.num_dof + 11] = noise_scales.delta_base_pos * noise_level * self.obs_scales.delta_base_pos
                # delta heading dim=1
                noise_vec[2*self.num_dof + 11 : 2*self.num_dof + 12] = noise_scales.delta_heading * noise_level * self.obs_scales.delta_heading
                # ref dof pos
                noise_vec[2*self.num_dof + 12 : 3*self.num_dof + 12] = noise_scales.ref_dof_pos * noise_level * self.obs_scales.dof_pos
                # last actions
                noise_vec[3*self.num_dof + 12 : 4*self.num_dof + 12] = noise_scales.last_action * noise_level
            elif self.cfg.motion.teleop_obs_version == 'v-teleop':
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_local_selected_body_pos, self.actions   
                # dof_pos
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # delta base pos dim2
                noise_vec[2*self.num_dof + 9  : 2*self.num_dof + 11] = noise_scales.delta_base_pos * noise_level * self.obs_scales.delta_base_pos
                # delta heading dim=1
                noise_vec[2*self.num_dof + 11 : 2*self.num_dof + 12] = noise_scales.delta_heading * noise_level * self.obs_scales.delta_heading
                # ref dof pos
                noise_vec[2*self.num_dof + 12 : 2*self.num_dof + 12 + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos
                # last actions
                self._rigid_body_pos
                noise_vec[2*self.num_dof + 12 + len(self.cfg.motion.teleop_selected_keypoints_names)*3 : 3*self.num_dof + 12 + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.last_action * noise_level
                
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-clean':
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_local_selected_body_pos, self.actions   
                # dof_pos
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos
                # last actions
                noise_vec[2*self.num_dof + 9 + len(self.cfg.motion.teleop_selected_keypoints_names)*3 : 3*self.num_dof + 9 + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.last_action * noise_level
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-superclean':
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_local_selected_body_pos, self.actions   
                # dof_pos
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # ref dof pos
                noise_vec[2*self.num_dof : 2*self.num_dof + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos
                # last actions
                noise_vec[2*self.num_dof + len(self.cfg.motion.teleop_selected_keypoints_names)*3 : 3*self.num_dof + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.last_action * noise_level
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-clean-nolastaction':
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_local_selected_body_pos, self.actions   
                # dof_pos
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + len(self.cfg.motion.teleop_selected_keypoints_names)*3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend':
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_local_selected_body_pos, self.actions   
                # dof_pos
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2)*3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-nolinvel':
                
                # dof_pos, dof_vel, base_vel, base_ang_vel, base_gravity, delta_base_pos, delta_heading, ref_local_selected_body_pos, self.actions   
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # angular velocity
                noise_vec[2*self.num_dof  : 2*self.num_dof + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.gravity * noise_level
                # ref body pos
                noise_vec[2*self.num_dof + 6 : 2*self.num_dof + 6 + (len(self.cfg.motion.teleop_selected_keypoints_names)+2)*3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
                # last actions
                noise_vec[2*self.num_dof + 6 + (len(self.cfg.motion.teleop_selected_keypoints_names)+2)*3 : ] = noise_scales.last_action * noise_level
                
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-full':
                
                max_num_bodies = len(self.cfg.motion.teleop_selected_keypoints_names) + 3
                curr_obs_len = 0
                # body_pos
                noise_vec[0                   : (max_num_bodies - 1) * 3      ] = noise_scales.body_pos * noise_level * self.obs_scales.dof_pos
                curr_obs_len += (max_num_bodies - 1) * 3

                # body_rot
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 6    ] = noise_scales.body_rot * noise_level * self.obs_scales.dof_vel
                curr_obs_len += max_num_bodies * 6

                # body vel
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 3] = noise_scales.body_lin_vel * noise_level * self.obs_scales.lin_vel
                curr_obs_len += max_num_bodies * 3

                # body ang vel
                noise_vec[curr_obs_len        : curr_obs_len + max_num_bodies * 3] = noise_scales.body_ang_vel * noise_level * self.obs_scales.ang_vel
                self.self_obs_size = curr_obs_len
                curr_obs_len += max_num_bodies * 3
                
                
                # ref body_pos diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_rot diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 6] = noise_scales.ref_body_rot * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 6

                # ref lin vel diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_lin_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref ang vel diff
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_pos
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 3] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 3

                # ref body_rot
                noise_vec[curr_obs_len: curr_obs_len + max_num_bodies * 6] = noise_scales.ref_ang_vel * noise_level * self.obs_scales.body_pos  
                curr_obs_len += max_num_bodies * 6

              
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max_no_vel':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  

            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                
                self.self_obs_size = 2*self.num_dof + 9
                # ref dof pos
                if self.cfg.motion.future_tracks:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) *3 * 3 * self.cfg.motion.num_traj_samples ] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos 
                else:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) * 3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-vr-max-nolinvel':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base ang vel
                noise_vec[2*self.num_dof   : 2*self.num_dof + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.gravity * noise_level
                
                self.self_obs_size = 2*self.num_dof + 6
                # ref dof pos
                if self.cfg.motion.future_tracks:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) *3 * 3 * self.cfg.motion.num_traj_samples ] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos 
                else:
                    noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 3) * 3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-nolinvel':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                # noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # base ang vel
                noise_vec[2*self.num_dof + 3 - 3 : 2*self.num_dof + 6 - 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6 - 3 : 2*self.num_dof + 9 - 3] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 - 3 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3 - 3 ] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos    
            elif self.cfg.motion.teleop_obs_version == 'v-teleop-extend-max-acc':
                # local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape
                # local_body_pos 3x19
                noise_vec[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                # dof vel
                noise_vec[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                # base vel
                noise_vec[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_acc * noise_level * self.obs_scales.lin_acc # need to modify
                # base ang vel
                noise_vec[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                # base gravity
                noise_vec[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
                # ref dof pos
                noise_vec[2*self.num_dof + 9 : 2*self.num_dof + 9 + (len(self.cfg.motion.teleop_selected_keypoints_names) + 2) *3 * 3] = noise_scales.ref_body_pos * noise_level * self.obs_scales.body_pos  
            else:
                raise NotImplementedError
        else:
            # noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            # noise_vec[6:9] = noise_scales.gravity * noise_level
            # noise_vec[9:12] = 0.                                             # commands
            # noise_vec[12                       :   12+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # noise_vec[12+  self.num_actions    :   12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            # noise_vec[12+2*self.num_actions    :                       ] = 0. # previous actions
            noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.gravity * noise_level
            noise_vec[6:9] = 0.01                                             # commands
            noise_vec[9                       :   9+  self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[9+  self.num_actions    :   9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[9+2*self.num_actions    :                       ] = 0.01 # previous actions
            assert len(noise_vec) == 9 + 3 * self.num_actions
        return noise_vec

    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        
        if self.cfg.domain_rand.randomize_pd_gain:

            self._kp_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
    
            if self.cfg.motion.curriculum and self.cfg.motion.kpkd_by_curriculum: # update based on teleop level
                kp_scale_offset_from_1 = self._kp_scale[env_ids] - 1
                self._kp_scale[env_ids] -= kp_scale_offset_from_1 * (1 - self.teleop_levels[env_ids].unsqueeze(-1)/10.)
                kd_scale_offset_from_1 = self._kd_scale[env_ids] - 1
                self._kd_scale[env_ids] -= kd_scale_offset_from_1 *  (1 - self.teleop_levels[env_ids].unsqueeze(-1)/10.)
                

        if self.cfg.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.rfi_lim_range[0], self.cfg.domain_rand.rfi_lim_range[1], (len(env_ids), self.num_actions), device=self.device)
        
            if self.cfg.motion.curriculum and self.cfg.motion.rfi_by_curriculum:
                rfi_lim_scale_offset_from_lowerlimit = self._rfi_lim_scale[env_ids] - self.cfg.domain_rand.rfi_lim_range[0]
                self._rfi_lim_scale[env_ids] -= rfi_lim_scale_offset_from_lowerlimit * (1 - self.teleop_levels[env_ids].unsqueeze(-1)/10.)
        # print(self._kp_scale[env_ids[0]])

        if self.cfg.domain_rand.randomize_motion_ref_xyz:
            # print(self.ref_episodic_offset[env_ids], " before")
            self.ref_episodic_offset[env_ids,0] = torch_rand_float(self.cfg.domain_rand.motion_ref_xyz_range[0][0], self.cfg.domain_rand.motion_ref_xyz_range[0][1], (len(env_ids),1), device=self.device).squeeze(1)
            self.ref_episodic_offset[env_ids,1] = torch_rand_float(self.cfg.domain_rand.motion_ref_xyz_range[1][0], self.cfg.domain_rand.motion_ref_xyz_range[1][1], (len(env_ids),1), device=self.device).squeeze(1)
            self.ref_episodic_offset[env_ids,2] = torch_rand_float(self.cfg.domain_rand.motion_ref_xyz_range[2][0], self.cfg.domain_rand.motion_ref_xyz_range[2][1], (len(env_ids),1), device=self.device).squeeze(1)
            # print(self.ref_episodic_offset[env_ids], " after")

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        

        # create some wrapper tensors for different slices
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # if (not self.headless) and self.cfg.motion.visualize and self.cfg.motion.teleop:
        #     self.root_states = self.root_states_all[0::(self._num_teleop_markers+1)]
        #     self.marker_states = [self.root_states_all[marker_i::(self._num_teleop_markers + 1)] for marker_i in range(1,self._num_teleop_markers + 1)]
        #     # self.root_states = self.root_states_all[0::(self.cfg.motion.num_markers+1)]
        #     # self.marker_states = [self.root_states_all[marker_i::(self.cfg.motion.num_markers + 1)] for marker_i in range(1,self.cfg.motion.num_markers + 1)]
        # else:
        #     self.root_states = self.root_states_all
            
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # init rigid body state
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        
        # self._rigid_body_pos = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 0:3]
        # self._rigid_body_rot = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 3:7]
        # self._rigid_body_vel = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 7:10]
        # self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., 1:self.num_bodies, 10:13]
        
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_root_pos = torch.zeros_like(self.root_states[:, 0:3])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # normalization
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # import ipdb; ipdb.set_trace()
        if self.cfg.terrain.measure_heights:
           self.height_points = self._init_height_points()
        self.measured_heights = 0
    
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel) # different from self.last_root_vel


        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        # Init for motion reference
        if self.cfg.motion.teleop:
            self.ref_motion_cache = {}
            self._load_motion()
            self.marker_coords = torch.zeros(self.num_envs, (self.num_dofs + (4 if self.cfg.motion.extend_head else 3)) * self.cfg.motion.num_traj_samples, 3, dtype=torch.float, device=self.device, requires_grad=False) # extend
            self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
            self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
            self.motion_ids = torch.arange(self.num_envs).to(self.device)
            self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
            self.base_pos_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            if self.cfg.motion.teleop: 
                self._recovery_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                self._package_loss_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

            self.ref_base_pos_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_base_rot_init = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_base_vel_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_base_ang_vel_init = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

            self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

            self.env_origins_init_3Doffset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self._resample_motion_times(env_ids) #need to resample before reset root states
            # self._update_motion_reference()
            self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
            
            if self.cfg.motion.curriculum:
                self.teleop_levels = torch.randint(0, 10+1, (self.num_envs,), device=self.device)
        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0], 
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)
            
    def _init_domain_params(self):
        # init params for domain randomization
        # init 0 for values
        # init 1 for scales
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._ground_friction_values = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)        
        self._link_mass_scale = torch.ones(self.num_envs, len(self.cfg.domain_rand.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    
    
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset["driveMode"] = gymapi.DOF_MODE_EFFORT
        
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self._init_domain_params()

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # import pdb; pdb.set_trace()
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        # import ipdb; ipdb.set_trace()
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel

        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        
        

        self.actor_handles = []
        self.envs = []
        self.marker_handles = []
        
        if (not self.headless) and self.cfg.motion.visualize and self.cfg.motion.teleop:
            self._num_teleop_markers = self.cfg.motion.num_markers * self.num_envs
            self._load_marker_asset()
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self._body_list = self.gym.get_actor_rigid_body_names(env_handle, actor_handle)
            dof_props = self._process_dof_props(dof_props_asset, i)
            # if self.cfg.asset.set_dof_properties:
                # dof_props['stiffness'] = self.cfg.asset.default_dof_prop_stiffness
                # dof_props['damping'] = self.cfg.asset.default_dof_prop_damping
                # dof_props['friction'] = self.cfg.asset.default_dof_prop_friction
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            if (not self.headless) and self.cfg.motion.visualize and self.cfg.motion.teleop:
                assert self.cfg.motion.num_markers == self.num_dofs, "we visualize all joints"
                # self.marker_handles.append(list())
                
                for marker_i in range(self.cfg.motion.num_markers):
                    start_pose_obj = gymapi.Transform()
                    pos_obj = torch.zeros((3,), device=self.device)
                    start_pose_obj.p = gymapi.Vec3(*pos_obj)
                    marker_handle = self.gym.create_actor(env_handle, self._marker_asset, start_pose_obj, 'marker', i, 1, 0)
                    self.marker_handles.append(marker_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        print("terminate by", termination_contact_names)
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales

        if isinstance(self.cfg.rewards.scales, EasyDict):
            self.reward_scales = {k: eval(v) if isinstance(v, str) else v for k, v in self.cfg.rewards.scales.items()}
            self.command_ranges = self.cfg.commands.ranges
        else:
            self.reward_scales = class_to_dict(self.cfg.rewards.scales)
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        

        
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False

        self.max_episode_length_s = self.cfg.env.episode_length_s
        # import pdb; pdb.set_trace()
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.package_loss_interval = np.ceil(self.cfg.domain_rand.package_loss_interval_s / self.dt)
        self.cfg.motion.resample_motions_for_envs_interval = np.ceil(self.cfg.motion.resample_motions_for_envs_interval_s / self.dt)

    #-------------- Reference Motion ---------------
    def _load_motion(self):
        motion_path = self.cfg.motion.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        skeleton_path = self.cfg.motion.skeleton_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._motion_lib = MotionLibH1(motion_file=motion_path, device=self.device, masterfoot_conifg=None, fix_height=False,multi_thread=False,mjcf_file=skeleton_path, extend_head=self.cfg.motion.extend_head) #multi_thread=True doesn't work
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        
        self.skeleton_trees = [sk_tree] * self.num_envs
        if self.cfg.env.test:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False)
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        self.motion_dt = self._motion_lib._motion_dt

    def resample_motion(self):
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)

    def forward_motion_samples(self):
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)

        
    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        # self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        # self.motion_ids[env_ids] = torch.randint(0, self._motion_lib._num_unique_motions, (len(env_ids),), device=self.device)
        # print(self.motion_ids[:10])
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        # self.env_origins_init_3Doffset[env_ids, :2] = torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        if self.cfg.env.test:
            self.motion_start_times[env_ids] = 0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        offset=(self.env_origins + self.env_origins_init_3Doffset)
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        
        self.ref_base_pos_init[env_ids] = motion_res["root_pos"][env_ids]
        self.ref_base_rot_init[env_ids] = motion_res["root_rot"][env_ids]
        self.ref_base_vel_init[env_ids] = motion_res["root_vel"][env_ids]
        self.ref_base_ang_vel_init[env_ids] = motion_res["root_ang_vel"][env_ids]

        
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        # import ipdb; ipdb.set_trace()
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            # import ipdb; ipdb.set_trace()
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        # import ipdb; ipdb.set_trace()
        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache
    
    def _get_state_from_motionlib_cache_trimesh(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        # import ipdb; ipdb.set_trace()
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)

        # import ipdb; ipdb.set_trace()
        # self.root_states[:,:2] = motion_res['root_pos'][:, :2]
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(position=motion_res['root_pos'][:, :3]).flatten()
            delta_height = self.measured_heights[:] - offset[:, 2]
            # self.root_states[:, 2] += delta_height
            motion_res['root_pos'][:, 2] += delta_height
            # import ipdb; ipdb.set_trace()
            if "rg_pos" in motion_res:
                motion_res['rg_pos'][:, :, 2] += delta_height.unsqueeze(1)
            if "rg_pos_t" in motion_res:
                motion_res['rg_pos_t'][:, :, 2] += delta_height.unsqueeze(1)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

        
    # def _update_motion_reference(self,):
    #     motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times)
    #     self.ref_body_pos = motion_res["rg_pos"] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
    #     ref_body_pos_extend = motion_res["rg_pos_t"] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
    #     ref_body_vel = motion_res["body_vel"] # [num_envs, num_markers, 3]
    #     ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
    #     ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
    #     ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
    #     ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
    #     ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
    #     self.marker_coords[:] = motion_res["rg_pos"][:, 1:,] + self.env_origins[:, None] + self.env_origins_init_3Doffset[:, None]
        
        
    def _load_marker_asset(self):
        asset_path = self.cfg.motion.marker_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        marker_asset_options = gymapi.AssetOptions()
        marker_asset_options.angular_damping = 0.0
        marker_asset_options.linear_damping = 0.0
        marker_asset_options.max_angular_velocity = 0.0
        marker_asset_options.density = 0
        marker_asset_options.fix_base_link = True
        marker_asset_options.thickness = 0.0
        marker_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # set no collision
        marker_asset_options.disable_gravity = True
        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, marker_asset_options)
        return
    
    #------------ helper functions ---------------
    def _get_rigid_body_pos(self, body_name):
        body_list = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        # assert len(body_list) == 21
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        return gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[body_list.index(body_name)::len(body_list),:3]
    
    @property
    def knee_distance(self):
        left_knee_pos = self._get_rigid_body_pos("left_knee_link")
        right_knee_pos = self._get_rigid_body_pos("right_knee_link")
        # print(f"left knee pos: {left_knee_pos}")
        dist_knee = torch.norm(left_knee_pos - right_knee_pos, dim=-1, keepdim=True)
        # print("dist knee shape", dist_knee.shape)
        return dist_knee
  
    @property
    def feet_distance(self):
        left_foot_pos = self._get_rigid_body_pos("left_ankle_link")
        right_foot_pos = self._get_rigid_body_pos("right_ankle_link")
        dist_feet = torch.norm(left_foot_pos - right_foot_pos, dim=-1, keepdim=True)
        return dist_feet
    
    #------------ reward functions----------------
    def _reward_closing(self):
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos = motion_res['root_pos']
        last_dis = ref_body_pos[:, 0:2] - self.last_root_pos[:, 0:2]
        # import ipdb; ipdb.set_trace()
        last_dis_norm = torch.norm(last_dis, dim=1)
        current_dis = ref_body_pos[:, 0:2] - self.root_states[:, 0:2]
        current_dis_norm = torch.norm(current_dis, dim=1)
        reward = torch.clamp(last_dis_norm - current_dis_norm, max=1 / 20)
        return reward

    def _reward_in_the_air(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        # import ipdb; ipdb.set_trace()
        return reward
    

    def _reward_stable_lower_when_vrclose(self):
        assert self.cfg.motion.extend_head, "This reward is only for the extended model"

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        ref_body_vel = motion_res['body_vel']
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend

        diff_global_body_pos_vr = diff_global_body_pos[:, 20:22] # left hand, right hand
        far_enough = torch.norm(diff_global_body_pos_vr, dim=-1) > self.cfg.rewards.vrclose_threshold
        far_enough_any = far_enough.any(dim=-1)
        close_enough = ~far_enough_any

        # penalize the lower for lifting up legs if the upper is close
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        both_feet_contact = first_foot_contact & second_foot_contact
        not_both_feet_contact = ~both_feet_contact

        

        reward = close_enough & not_both_feet_contact
        reward *= torch.norm(ref_body_vel[:, 0, :2], dim=1) < self.cfg.rewards.ref_stable_velocity_threshold #no reward for low ref motion velocity (root xy velocity)
        # import ipdb; ipdb.set_trace()
        return reward
    
    def _reward_stable_lower_when_vrclose_positive(self):
        assert self.cfg.motion.extend_head, "This reward is only for the extended model"

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        ref_body_vel = motion_res['body_vel']
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend

        diff_global_body_pos_vr = diff_global_body_pos[:, 20:22] # left hand, right hand
        far_enough = torch.norm(diff_global_body_pos_vr, dim=-1) > self.cfg.rewards.vrclose_threshold
        far_enough_any = far_enough.any(dim=-1)
        close_enough = ~far_enough_any

        # penalize the lower for lifting up legs if the upper is close
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        both_feet_contact = first_foot_contact & second_foot_contact
        # not_both_feet_contact = ~both_feet_contact

        

        reward = close_enough & both_feet_contact
        reward *= torch.norm(ref_body_vel[:, 0, :2], dim=1) < self.cfg.rewards.ref_stable_velocity_threshold #no reward for low ref motion velocity (root xy velocity)
        # import ipdb; ipdb.set_trace()
        return reward

    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_feet_ori(self):
        left_quat = self._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        dif = torch.abs(base_height - self.cfg.rewards.base_height_target)
        return torch.clip(dif - 0.15, min=0.)

    def _reward_feet_height(self):
        # Penalize base height away from target
        feet_height = self._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.cfg.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_lower_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :11] - self.actions[:, :11]), dim=1)
    
    def _reward_upper_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, 11:] - self.actions[:, 11:]), dim=1)
        

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_close_feet(self):
        # returns 1 if two feet are too close
        return (self.feet_distance < 0.24).squeeze(-1) * 1.0
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1) ** 0.5
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
            # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_freeze_upper_body(self):
        return torch.mean(torch.square(self.dof_pos[:, 10:] - self.default_dof_pos[:, 10:]), dim=1)
        
    def _reward_tracking_dof_vel(self):
        # Tracking of dof velocity commands
        dof_vel_error = torch.sum(torch.square(self.joint_vel_reference - self.dof_vel), dim=1)
        return torch.exp(-dof_vel_error/self.cfg.rewards.tracking_sigma)    
    
    def _reward_teleop_joint_position_lower(self):
        dof_pos = self.dof_pos
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        
        ref_dof_pos = motion_res['dof_pos']
        
        
        diff_dof_pos = ref_dof_pos - dof_pos
        diff_dof_pos = diff_dof_pos[:, :10] # lower, not including torso
        diff_dof_pos_dist = torch.mean(torch.square(diff_dof_pos), dim=1)
        r_dof_pos = torch.exp(-diff_dof_pos_dist / self.cfg.rewards.teleop_joint_pos_sigma)
        return r_dof_pos

    def _reward_teleop_joint_position_upper(self):
        dof_pos = self.dof_pos
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        
        ref_dof_pos = motion_res['dof_pos']
        
        
        diff_dof_pos = ref_dof_pos - dof_pos
        diff_dof_pos = diff_dof_pos[:, 10:] # upper

        diff_dof_pos_dist = torch.mean(torch.square(diff_dof_pos), dim=1)
        r_dof_pos = torch.exp(-diff_dof_pos_dist / self.cfg.rewards.teleop_joint_pos_sigma)
        return r_dof_pos
    
    def _reward_teleop_selected_joint_position(self):
        dof_pos = self.dof_pos
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_dof_pos = motion_res['dof_pos']
        
        diff_dof_pos = ref_dof_pos - dof_pos
        # scale the diff by self.cfg.rewards.teleop_joint_pos_selection
        for joint_name, scale in self.cfg.rewards.teleop_joint_pos_selection.items():
            joint_index = self.dof_names.index(joint_name)
            assert joint_index >= 0, f"Joint {joint_name} not found in the robot"
            
            diff_dof_pos[:, joint_index] *= scale **.5
        diff_dof_pos_dist = torch.mean(torch.square(diff_dof_pos), dim=1)
        r_dof_pos = torch.exp(-diff_dof_pos_dist / self.cfg.rewards.teleop_joint_pos_sigma)
        return r_dof_pos
        
    def _reward_teleop_joint_vel_lower(self):
        dof_vel = self.dof_vel
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_dof_vel = motion_res['dof_vel']
        
        diff_dof_vel = ref_dof_vel - dof_vel
        diff_dof_vel = diff_dof_vel[:, :10] # lower, not including torso
        diff_dof_vel_dist = torch.mean(torch.square(diff_dof_vel), dim=1)
        r_dof_vel = torch.exp(-diff_dof_vel_dist / self.cfg.rewards.teleop_joint_vel_sigma)
        return r_dof_vel

    def _reward_teleop_joint_vel_upper(self):
        dof_vel = self.dof_vel
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_dof_vel = motion_res['dof_vel']
        
        diff_dof_vel = ref_dof_vel - dof_vel
        diff_dof_vel = diff_dof_vel[:, 10:] # upper
        diff_dof_vel_dist = torch.mean(torch.square(diff_dof_vel), dim=1)
        r_dof_vel = torch.exp(-diff_dof_vel_dist / self.cfg.rewards.teleop_joint_vel_sigma)
        return r_dof_vel
        
    def _reward_teleop_selected_joint_vel(self):
        dof_vel = self.dof_vel
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_dof_vel = motion_res['dof_vel']
        
        diff_dof_vel = ref_dof_vel - dof_vel
        # scale the diff by self.cfg.rewards.teleop_joint_pos_selection
        for joint_name, scale in self.cfg.rewards.teleop_joint_pos_selection.items():
            joint_index = self.dof_names.index(joint_name)
            assert joint_index >= 0, f"Joint {joint_name} not found in the robot"
            diff_dof_vel[:, joint_index] *= scale **.5
        diff_dof_vel_dist = torch.mean(torch.square(diff_dof_vel), dim=1)
        r_dof_vel = torch.exp(-diff_dof_vel_dist / self.cfg.rewards.teleop_joint_vel_sigma)
        return r_dof_vel
        

    def _reward_teleop_body_position(self):
        body_pos = self._rigid_body_pos
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos = motion_res['rg_pos']
        # body_pos[:, :, :2] -= self.base_pos_init[:, :2].unsqueeze(1)
        # ref_body_pos[:, :, :2] -= self.ref_base_pos_init[:, :2].unsqueeze(1)
        diff_global_body_pos = ref_body_pos - body_pos
        diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(-diff_body_pos_dist / self.cfg.rewards.teleop_body_pos_sigma)
        return r_body_pos
    
    # def _reward_teleop_body_position_extend(self):
    #     body_pos = self._rigid_body_pos
    #     body_rot = self._rigid_body_rot
        
    #     offset = self.env_origins + self.env_origins_init_3Doffset
    #     motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
    #     motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
    #     ref_body_pos_extend = motion_res['rg_pos_t']
        
    #     if self.cfg.asset.local_upper_reward:
    #         diff =  ref_body_pos_extend[:, [0]] - body_pos[:, [0]]
    #         ref_body_pos_extend[:, 11:] -= diff
        
    #     extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_ids]
    #     body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

    #     diff_global_body_pos = ref_body_pos_extend - body_pos_extend
    #     diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    #     r_body_pos = torch.exp(-diff_body_pos_dist / self.cfg.rewards.teleop_body_pos_sigma)
        
    #     return r_body_pos
    
    def _reward_teleop_body_position_extend_small_sigma(self):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(-diff_body_pos_dist / self.cfg.rewards.teleop_body_pos_small_sigma)
        
        return r_body_pos
    

    def _reward_teleop_body_position_extend(self):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        
        if self.cfg.asset.local_upper_reward:
            diff =  ref_body_pos_extend[:, [0]] - body_pos[:, [0]]
            ref_body_pos_extend[:, 11:] -= diff
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_lower = diff_global_body_pos[:, :11]
        diff_global_body_pos_upper = diff_global_body_pos[:, 11:]
        diff_body_pos_dist_lower = (diff_global_body_pos_lower**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_upper = (diff_global_body_pos_upper**2).mean(dim=-1).mean(dim=-1)
        
        diff_body_pos_dist_lower = diff_body_pos_dist_lower
        diff_body_pos_dist_upper = diff_body_pos_dist_upper
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.cfg.rewards.teleop_body_pos_lowerbody_sigma)
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.teleop_body_pos_upperbody_sigma)
        
        r_body_pos = r_body_pos_lower * self.cfg.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.cfg.rewards.teleop_body_pos_upperbody_weight
        
        return r_body_pos
    

    def _reward_teleop_body_position_extend_lower(self):

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        
        if self.cfg.asset.local_upper_reward:
            diff =  ref_body_pos_extend[:, [0]] - body_pos[:, [0]]
            ref_body_pos_extend[:, 11:] -= diff
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_lower = diff_global_body_pos[:, :11]
        diff_body_pos_dist_lower = (diff_global_body_pos_lower**2).mean(dim=-1).mean(dim=-1)
        
        diff_body_pos_dist_lower = diff_body_pos_dist_lower
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.cfg.rewards.teleop_body_pos_lowerbody_sigma)
        
        r_body_pos = r_body_pos_lower
        
        return r_body_pos
    
    def _reward_teleop_body_position_extend_upper(self):

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        
        if self.cfg.asset.local_upper_reward:
            diff =  ref_body_pos_extend[:, [0]] - body_pos[:, [0]]
            ref_body_pos_extend[:, 11:] -= diff
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_upper = diff_global_body_pos[:, 11:]
        diff_body_pos_dist_upper = (diff_global_body_pos_upper**2).mean(dim=-1).mean(dim=-1)
        
        diff_body_pos_dist_upper = diff_body_pos_dist_upper
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.teleop_body_pos_upperbody_sigma)
        
        r_body_pos = r_body_pos_upper 
        
        return r_body_pos

    def _reward_teleop_body_position_extend_upper_0dot5sigma(self):

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        
        if self.cfg.asset.local_upper_reward:
            diff =  ref_body_pos_extend[:, [0]] - body_pos[:, [0]]
            ref_body_pos_extend[:, 11:] -= diff
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_upper = diff_global_body_pos[:, 11:]
        diff_body_pos_dist_upper = (diff_global_body_pos_upper**2).mean(dim=-1).mean(dim=-1)
        
        diff_body_pos_dist_upper = diff_body_pos_dist_upper
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.teleop_body_pos_0dot5sigma)
        
        r_body_pos = r_body_pos_upper 
        
        return r_body_pos
    
    def _reward_teleop_body_position_vr_3keypoints(self):

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        

        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
        
        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_global_body_pos_vr_3keypoints = diff_global_body_pos[:, -3:]
        diff_body_pos_dist_vr_3keypoints = (diff_global_body_pos_vr_3keypoints**2).mean(dim=-1).mean(dim=-1)

        diff_body_pos_dist_vr_3keypoints= diff_body_pos_dist_vr_3keypoints

        r_body_pos_vr_3keypoints = torch.exp(-diff_body_pos_dist_vr_3keypoints / self.cfg.rewards.teleop_body_pos_vr_3keypoints_sigma)
        
    
        
        return r_body_pos_vr_3keypoints
    

    def _reward_teleop_body_position_extend_small_sigma(self):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_pos_extend = motion_res['rg_pos_t']
        
        extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos[:, ].reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
        body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)

        diff_global_body_pos = ref_body_pos_extend - body_pos_extend
        diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(-diff_body_pos_dist / self.cfg.rewards.teleop_body_pos_small_sigma)
        
        return r_body_pos
    

    def _reward_teleop_body_rotation(self): # wholebody 
        body_rot = self._rigid_body_rot

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_rot = motion_res['rb_rot']

        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_global_body_angle_dist / self.cfg.rewards.teleop_body_rot_sigma)
        return r_body_rot
    
        
    def _reward_teleop_body_rotation_lower(self):
        body_rot = self._rigid_body_rot

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_rot = motion_res['rb_rot']

        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle = diff_global_body_angle[:, :11] # lower
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_global_body_angle_dist / self.cfg.rewards.teleop_body_rot_sigma)
        return r_body_rot
    

    def _reward_teleop_body_rotation_upper(self):
        body_rot = self._rigid_body_rot

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        ref_body_rot = motion_res['rb_rot']
        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle = diff_global_body_angle[:, 11:] # upper
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_global_body_angle_dist / self.cfg.rewards.teleop_body_rot_sigma)
        return r_body_rot
    
    def _reward_teleop_selected_body_rotation(self):
        raise NotImplementedError
        body_rot = self._rigid_body_rot
        ref_body_rot = ref_body_rot
        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        track_rot_body_indices = [self._body_list.index(body_name) for body_name in self.cfg.rewards.teleop_body_rot_selection]
        diff_global_body_angle_dist = (diff_global_body_angle[:, track_rot_body_indices]**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_global_body_angle_dist / self.cfg.rewards.teleop_body_rot_sigma)
        return r_body_rot
    
    def _reward_teleop_body_vel(self):
        body_vel = self._rigid_body_vel

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_vel = motion_res['body_vel']

        diff_global_vel = ref_body_vel - body_vel
        diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
        
        r_vel = torch.exp(-diff_global_vel_dist / self.cfg.rewards.teleop_body_vel_sigma)
        return r_vel

    def _reward_teleop_body_vel_lower(self):
        body_vel = self._rigid_body_vel

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_vel = motion_res['body_vel']

        diff_global_vel = ref_body_vel - body_vel
        diff_global_vel = diff_global_vel[:, :11]
        diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
        
        r_vel = torch.exp(-diff_global_vel_dist / self.cfg.rewards.teleop_body_vel_sigma)
        return r_vel
        
    def _reward_teleop_body_vel_upper(self):
        body_vel = self._rigid_body_vel

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_vel = motion_res['body_vel']

        diff_global_vel = ref_body_vel - body_vel
        diff_global_vel = diff_global_vel[:, 11:]
        diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
        
        r_vel = torch.exp(-diff_global_vel_dist / self.cfg.rewards.teleop_body_vel_sigma)
        return r_vel
    

    
    def _reward_teleop_selected_body_vel(self):
        raise NotImplementedError
        body_vel = self._rigid_body_vel
        ref_body_vel = ref_body_vel
        diff_global_vel = ref_body_vel - body_vel
        track_vel_body_indices = [self._body_list.index(body_name) for body_name in self.cfg.rewards.teleop_body_vel_selection]
        diff_global_vel_dist = (diff_global_vel[:, track_vel_body_indices]**2).mean(dim=-1).mean(dim=-1)
        r_vel = torch.exp(-diff_global_vel_dist / self.cfg.rewards.teleop_body_vel_sigma)
        return r_vel
        

    def _reward_teleop_body_ang_vel(self):
        body_ang_vel = self._rigid_body_ang_vel

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_ang_vel = motion_res['body_ang_vel']

        diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
        diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
        r_ang_vel = torch.exp(-diff_global_ang_vel_dist / self.cfg.rewards.teleop_body_ang_vel_sigma)
        return r_ang_vel
    
    def _reward_teleop_body_ang_vel_lower(self):
        body_ang_vel = self._rigid_body_ang_vel

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_ang_vel = motion_res['body_ang_vel']

        diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
        diff_global_ang_vel = diff_global_ang_vel[:, :11] # lower
        diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
        r_ang_vel = torch.exp(-diff_global_ang_vel_dist / self.cfg.rewards.teleop_body_ang_vel_sigma)
        return r_ang_vel
    

    def _reward_teleop_body_ang_vel_upper(self):
        body_ang_vel = self._rigid_body_ang_vel

        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_ang_vel = motion_res['body_ang_vel']


        diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
        diff_global_ang_vel = diff_global_ang_vel[:, 11:] # upper
        diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
        r_ang_vel = torch.exp(-diff_global_ang_vel_dist / self.cfg.rewards.teleop_body_ang_vel_sigma)
        return r_ang_vel
    
    def _reward_teleop_selected_body_ang_vel(self):
        raise NotImplementedError
        body_ang_vel = self._rigid_body_ang_vel
        ref_body_ang_vel = ref_body_ang_vel
        diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
        track_ang_vel_body_indices = [self._body_list.index(body_name) for body_name in self.cfg.rewards.teleop_body_ang_vel_selection]
        diff_global_ang_vel_dist = (diff_global_ang_vel[:, track_ang_vel_body_indices]**2).mean(dim=-1).mean(dim=-1)
        r_ang_vel = torch.exp(-diff_global_ang_vel_dist / self.cfg.rewards.teleop_body_ang_vel_sigma)
        return r_ang_vel

    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt

        self.feet_air_max_height = torch.max(self.feet_air_max_height, self._rigid_body_pos[:, self.feet_indices, 2])
        
        rew_feet_max_height = torch.sum((torch.clamp_min(self.cfg.rewards.desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_max_height *= ~contact_filt
        return rew_feet_max_height
        
    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime
    
    def _reward_feet_air_time_teleop(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        offset = self.env_origins + self.env_origins_init_3Doffset
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset=offset)
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

        ref_body_vel = motion_res['body_vel']
        
        
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(ref_body_vel[:, 0, :2], dim=1) > 0.1 #no reward for low ref motion velocity (root xy velocity)
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_slippage(self):
        assert self._rigid_body_vel.shape[1] == 20
        foot_vel = self._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    def _reward_alive(self):
        return 1.0 - 1.0 * (self.reset_buf * ~self.time_out_buf) # 1 - termination
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_move_or_not(self):
        should_move = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, self.commands[:, 2] > 0.2)
        is_moving = self.dof_vel.norm(dim=1) > 0.2
        return (2.* is_moving -1) * (2.*should_move - 1.)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_freeze_arms(self):
        ## hardcode zc, soft on torso
        return torch.sum(torch.square(self.dof_pos[:, 11:]-self.default_dof_pos[:, 11:]), dim=1) + 0.1*torch.abs(self.dof_pos[:, 10] - self.default_dof_pos[:, 10])
    
    def render(self, sync_frame_time=False):
        # if self.viewer:
            # self._update_camera()

        super().render(sync_frame_time)
        return
    
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return
    
    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)

        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        # self.gym.set_camera_location(self.recorder_camera_handle, self.envs[0], new_cam_pos, new_cam_target)

        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def next_task(self):
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
        self.reset()
        
        
    def _update_recovery_count(self):
        assert self.cfg.motion.teleop
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return
    
    def _update_package_loss_count(self):
        assert self.cfg.motion.teleop
        # import ipdb; ipdb.set_trace()
        self._package_loss_counter -= 1
        self._package_loss_counter = torch.clamp_min(self._package_loss_counter, 0)
        return
    


@torch.jit.script
def compute_imitation_observations(root_pos, root_rot, body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, ref_body_pos, ref_body_rot, ref_root_vel, ref_root_ang_vel, ref_dof_pos, ref_dof_vel, time_steps):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor, Tensor, int) -> Tensor
    # V7 with the addition of head position. 
    obs = []
    B, J, _ = body_pos.shape

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences for root
    diff_global_root_vel = ref_root_vel.view(B, time_steps,3) - root_vel.view(B, 1, 3)
    diff_local_root_vel = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), diff_global_root_vel.view(-1, 3))
    
    diff_global_root_ang_vel = ref_root_ang_vel.view(B, time_steps, 3) - root_ang_vel.view(B, 1, 3)
    diff_local_root_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), diff_global_root_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    ###### Dof difference
    dof_diff = ref_dof_pos.view(B, time_steps, -1) - dof_pos.view(B, 1, -1)
    dof_vel_diff = ref_dof_vel.view(B, time_steps, -1) - dof_vel.view(B, 1, -1)


    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * J * 6
    obs.append(diff_local_root_vel.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(diff_local_root_ang_vel.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * J * 6
    obs.append(dof_diff.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(dof_vel_diff.view(B, time_steps, -1))  # timestep  * J * 3

    
    # print(obs[0].shape, obs[1].shape, obs[2].shape, obs[3].shape, obs[4].shape, obs[5].shape, obs[6].shape, obs[7].shape)
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

# @torch.jit.script
def compute_imitation_observations_teleop(root_pos, root_rot, root_vel, body_pos, ref_body_pos, time_steps):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tensor
    # V7 with the addition of head position. 
    obs = []
    B, J, _ = body_pos.shape

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    


    # ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
    # local_ref_body_vel = ref_body_vel.view(B, time_steps, J, 3) 
    # local_ref_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_vel.view(-1, 3))


    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3

    # print(obs[0].shape, obs[1].shape, obs[2].shape, obs[3].shape, obs[4].shape, obs[5].shape, obs[6].shape, obs[7].shape)
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

# @torch.jit.script
def compute_imitation_observations_teleop_max(root_pos, root_rot, body_pos,   ref_body_pos, ref_body_vel, time_steps,  ref_episodic_offset = None, ref_vel_in_task_obs = True):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,  int, bool, bool) -> Tensor
    #  Teleop version
    obs = []
    B, J, _ = body_pos.shape

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    
    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)) # 

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
    
    local_ref_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

    if ref_episodic_offset is not None:
        # import ipdb; ipdb.set_trace()
        diff_global_body_pos_offset= ref_episodic_offset.unsqueeze(1).unsqueeze(2).expand(-1, 1, J, -1)
        # diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset.view(-1, 3)
        diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset
        local_ref_body_pos_offset = ref_episodic_offset.repeat(J,1)[:J * ref_episodic_offset.shape[0], :]
        local_ref_body_pos[2::3] += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)[2::3]
        # local_ref_body_pos += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
    if ref_vel_in_task_obs:
        obs.append(local_ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3

    obs = torch.cat(obs, dim=-1).view(B, -1)
    
    return obs

# @torch.jit.script
def compute_imitation_observations_teleop_max_heading(root_pos, root_rot, body_pos, head_rot, ref_body_pos, ref_head_rot, ref_body_vel, time_steps,  ref_episodic_offset = None, ref_vel_in_task_obs = True):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,  int, bool, bool) -> Tensor
    #  Teleop version
    obs = []
    B, J, _ = body_pos.shape

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    
    ###### Heading rot 
    diff_global_body_rot = torch_utils.quat_mul(ref_head_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(head_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    # diff_local_heading_rot_flat = torch_utils.quat_to_tan_norm(torch_utils.calc_heading_quat(diff_local_body_rot_flat))
    diff_local_heading_rot_flat = torch_utils.calc_heading(diff_local_body_rot_flat)
    
    
    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)) # 

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
    
    local_ref_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

    if ref_episodic_offset is not None:
        # import ipdb; ipdb.set_trace()
        diff_global_body_pos_offset= ref_episodic_offset.unsqueeze(1).unsqueeze(2).expand(-1, 1, J, -1)
        # diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset.view(-1, 3)
        diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset
        local_ref_body_pos_offset = ref_episodic_offset.repeat(J,1)[:J * ref_episodic_offset.shape[0], :]
        local_ref_body_pos[2::3] += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)[2::3]
        # local_ref_body_pos += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(diff_local_heading_rot_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
    if ref_vel_in_task_obs:
        obs.append(local_ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3

    obs = torch.cat(obs, dim=-1).view(B, -1)
    
    return obs




# @torch.jit.script
def compute_humanoid_observations(body_pos, body_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_expanded_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_expanded_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_expanded_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    local_body_vel = torch_utils.my_quat_rotate(heading_rot_inv, root_vel) #  1x3, root
    local_body_ang_vel = torch_utils.my_quat_rotate(heading_rot_inv, root_ang_vel) # 1x3, root

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_pos, dof_vel] # [19x3, 20x6, 1x3, 1x3, 19x1, 19x1]
    #print(local_body_pos.shape, local_body_rot_obs.shape, local_body_vel.shape, local_body_ang_vel.shape, dof_pos.shape, dof_vel.shape)
    obs = torch.cat(obs_list, dim=-1)
    return obs

# @torch.jit.script
def compute_humanoid_observations_max_full(body_pos, body_rot, body_vel, body_ang_vel,  local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool,  bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    

    obs = torch.cat(obs_list, dim=-1)
    return obs



# @torch.jit.script
def compute_imitation_observations_max_full(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, ref_episodic_offset = None, ref_vel_in_task_obs = True):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, Tensor, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape


    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    if ref_episodic_offset is not None:
        # import ipdb; ipdb.set_trace()
        diff_global_body_pos_offset= ref_episodic_offset.unsqueeze(1).unsqueeze(2).expand(-1, 1, J, -1)
        diff_local_body_pos_flat = diff_local_body_pos_flat.view(B, 1, J, 3) + diff_global_body_pos_offset
        local_ref_body_pos_offset = ref_episodic_offset.repeat(J,1)[:J * ref_episodic_offset.shape[0], :]
        local_ref_body_pos += local_ref_body_pos_offset.repeat_interleave(time_steps, 0)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * J * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * J * 6
    obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * J * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

