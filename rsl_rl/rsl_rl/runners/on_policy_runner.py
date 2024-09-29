# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticPULSE
from rsl_rl.env import VecEnv
from tqdm import tqdm

import numpy as np
from phc.smpllib.smpl_eval import compute_metrics_lite
import gc
import joblib

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.dagger_only = train_cfg.dagger.dagger_only
        self.dagger_anneal = train_cfg.dagger.dagger_anneal


        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic

        self.policy_cfg['self_obs_size'] = self.env.self_obs_size
        
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # init storage and model
        if self.env.cfg.train.distill:
            self.kin_dict_info = {k: (v.shape, v.reshape(v.shape[0], -1).shape) for k, v in env.kin_dict.items()}
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], self.kin_dict_info)
        else:
            self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], None)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        
        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        costbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions.detach())
                    
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    
                    if self.alg_cfg.get("save_z_noise", False):
                        infos['kin_dict']['z_noise'] = self.alg.actor_critic.z_noise.clone()
                        
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_cost_sum += infos['cost']
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        costbuffer.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_cost_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            if self.cfg.get("has_eval", False) and it > 0 and it % self.cfg.eval_interval == 0:
                with torch.inference_mode():
                    eval_info = self.eval()

            teleop_body_pos_upperbody_sigma = self.env.cfg.rewards.teleop_body_pos_upperbody_sigma
            penalty_scale = self.env.cfg.rewards.penalty_scale
            average_episode_length_for_reward_curriculum = self.env.average_episode_length
            born_distance = self.env.cfg.domain_rand.born_distance
            born_heading = self.env.cfg.domain_rand.born_heading_degree

            
            mean_value_loss, mean_surrogate_loss, mean_action_smoothness_loss, mean_kin_loss = self.alg.update(epoch = it, dagger_anneal=self.dagger_anneal, dagger_only=self.dagger_only)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))


    def update_training_data(self, failed_keys):
        humanoid_env = self.env
        humanoid_env._motion_lib.update_soft_sampling_weight(failed_keys)
        # joblib.dump({"failed_keys": failed_keys, "termination_history": humanoid_env._motion_lib._termination_history.clone()}, osp.join(self.network_path, f"failed_{self.epoch_num:010d}.pkl"))
        
        
    def eval(self):
        info = self.run_eval_loop()
        if self.cfg.auto_negative_samping:
            self.update_training_data(info['failed_keys'])
        del self.terminate_state, self.terminate_memory, self.mpjpe, self.mpjpe_all
        return info["eval_info"]

    def run_eval_loop(self):
        print("############################ Evaluation ############################")
        

        self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        self.success_rate = 0
        self.pbar = tqdm(
            range(self.env._motion_lib._num_unique_motions // self.env.num_envs)
        )
        self.pbar.set_description("")
        temp_max_distance = self.env.cfg.asset.termination_scales.max_ref_motion_distance
        self.env.cfg.env.test=True
        self.env.cfg.env.im_eval=True
        self.env.begin_seq_motion_samples()
        self.env.cfg.asset.termination_scales.max_ref_motion_distance = 0.5

        policy = self.get_inference_policy(device=self.device)
        obs, privileged_obs = self.env.reset()
        batch_size = self.env.num_envs

        done_indices = []
        
        while True:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = self.env.step(actions.detach())
            
            done, info = self._post_step_eval(infos, dones.clone())

            if dones.sum() == self.env.num_envs:
                obs, privileged_obs = self.env.reset()
                
            if info['end']:
                break
        
        self.env.cfg.env.test=False
        self.env.cfg.env.im_eval=False
        self.env.cfg.asset.termination_scales.max_ref_motion_distance = temp_max_distance
        self.env.reset()  # Reset ALL environments, go back to training mode.

        torch.cuda.empty_cache()
        gc.collect()
        
        return info
    
    def _post_step_eval(self, info, done):
        end = False
        eval_info = {}
        # modify done such that games will exit and reset.
        humanoid_env = self.env
        termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, done) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
        # termination_state = info["terminate"]
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
        if (~self.terminate_state).sum() > 0:
            max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
            curr_ids = humanoid_env._motion_lib._curr_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][
                        ~self.terminate_state[:bound]
                    ].max()
                else:
                    curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()
                
            if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
        else:
            curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

        
        self.mpjpe.append(info["mpjpe"])
        self.gt_pos.append(info["body_pos_gt"])
        self.pred_pos.append(info["body_pos"])
        self.curr_stpes += 1

        if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
            self.curr_stpes = 0
            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.success_rate = (1- np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
            all_mpjpe = [all_mpjpe[:(i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_pred = [all_body_pos_pred[:(i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            all_body_pos_gt = np.stack(self.gt_pos)
            all_body_pos_gt = [all_body_pos_gt[:(i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]


            self.mpjpe_all.append(all_mpjpe)
            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt
            

            if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                self.pbar.clear()
                terminate_hist = np.concatenate(self.terminate_memory)
                succ_idxes = np.flatnonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]).tolist()

                pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]


                # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                # humanoid_env._motion_lib.get_motion_num_steps().sum()

                failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])

                metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
                metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                metrics_all_print = {m: np.mean(v) for m, v in metrics_all.items()}
                metrics_succ_print = {m: np.mean(v) for m, v in metrics_succ.items()}
                
                if len(metrics_succ_print) == 0:
                    print("No success!!!")
                    metrics_succ_print = metrics_all_print
                    
                print("------------------------------------------")
                print(f"Success Rate: {self.success_rate:.10f}")
                print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
                print("Failed keys: ", len(failed_keys), failed_keys)
                
                end = True
                
                eval_info = {
                    "eval_success_rate": self.success_rate,
                    "eval_mpjpe_all": metrics_all_print['mpjpe_g'],
                    "eval_mpjpe_succ": metrics_succ_print['mpjpe_g'],
                    "accel_dist": metrics_succ_print['accel_dist'], 
                    "vel_dist": metrics_succ_print['vel_dist'], 
                    "mpjpel_all": metrics_all_print['mpjpe_l'],
                    "mpjpel_succ": metrics_succ_print['mpjpe_l'],
                    "mpjpe_pa": metrics_succ_print['mpjpe_pa'], 
                }
                # failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[:humanoid_env._motion_lib._num_unique_motions]]
                # success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[:humanoid_env._motion_lib._num_unique_motions]]
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                # joblib.dump(failed_keys, "output/dgx/smpl_im_shape_long_1/failed_1.pkl")
                # joblib.dump(success_keys, "output/dgx/smpl_im_fit_3_1/long_succ.pkl")
                # print("....")
                return done, {"end": end, "eval_info": eval_info, "failed_keys": failed_keys,  "success_keys": success_keys}

            done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

            humanoid_env.forward_motion_samples()
            self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)

            self.pbar.update(1)
            self.pbar.refresh()
            self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []


        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
        self.pbar.set_description(update_str)

        return done, {"end": end, "eval_info": eval_info, "failed_keys": [],  "success_keys": []}


    def log(self, locs, width=80, pad=35):

        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Episode/Average_episode_length_for_reward_curriculum', locs['average_episode_length_for_reward_curriculum'], locs['it'])
        self.writer.add_scalar('Episode/Penalty_scale', locs['penalty_scale'], locs['it'])
        self.writer.add_scalar('Episode/Teleop_body_pos_upperbody_sigma', locs['teleop_body_pos_upperbody_sigma'], locs['it'])
        self.writer.add_scalar('Episode/Born_distance', locs['born_distance'], locs['it'])
        self.writer.add_scalar('Episode/Born_heading', locs['born_heading'], locs['it'])

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/action_smoothness', locs['mean_action_smoothness_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_cost', statistics.mean(locs['costbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_cost/time', statistics.mean(locs['costbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        if "eval_info" in locs:
            self.writer.add_scalar('Eval/Success_rate', locs['eval_info']['eval_success_rate'], locs['it'])
            self.writer.add_scalar('Eval/Mpjpe_all', locs['eval_info']['eval_mpjpe_all'], locs['it'])
            self.writer.add_scalar('Eval/Mpjpe_succ', locs['eval_info']['eval_mpjpe_succ'], locs['it'])
            self.writer.add_scalar('Eval/Accel_dist', locs['eval_info']['accel_dist'], locs['it'])
            self.writer.add_scalar('Eval/Vel_dist', locs['eval_info']['vel_dist'], locs['it'])
            self.writer.add_scalar('Eval/Mpjpel_all', locs['eval_info']['mpjpel_all'], locs['it'])
            self.writer.add_scalar('Eval/Mpjpel_succ', locs['eval_info']['mpjpel_succ'], locs['it'])
            self.writer.add_scalar('Eval/Mpjpe_pa', locs['eval_info']['mpjpe_pa'], locs['it'])


        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Action smoothness loss:':>{pad}} {locs['mean_action_smoothness_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean cost:':>{pad}} {statistics.mean(locs['costbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Average_episode_length_for_reward_curriculum:':>{pad}} {locs['average_episode_length_for_reward_curriculum']:.6f}\n"""
                          f"""{'Born_distance:':>{pad}} {locs['born_distance']:.6f}\n"""
                          f"""{'Born_heading:':>{pad}} {locs['born_heading']:.6f}\n"""
                          f"""{'Penalty_scale:':>{pad}} {locs['penalty_scale']:.6f}\n"""
                          f"""{'Teleop_body_pos_upperbody_sigma:':>{pad}} {locs['teleop_body_pos_upperbody_sigma']:.6f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Action smoothness loss:':>{pad}} {locs['mean_action_smoothness_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        if "mean_kin_loss" in locs:
            log_string += (f"""{'-' * width}\n"""
                        f"""{'Mean kin loss:':>{pad}} {locs['mean_kin_loss']:.3f}\n""")
            
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        
        
        log_string += f"""path: {self.log_dir}\n"""
        print("\r " + log_string, end='')

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
