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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules import VelocityEstimator
from rsl_rl.storage import RolloutStorage
from phc.learning.loss_functions import kl_multi

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 action_smoothness_coef = 0.1,
                 **kwargs
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.action_smoothness_coef = action_smoothness_coef
        self.kin_dict_info = None
        
        self.kin_only = kwargs.get("kin_only", False)
        self.z_type = kwargs.get("z_type", None)
        self.kld_coefficient = kwargs.get("kld_coefficient_max", 1.0)
        self.kld_coefficient_max = kwargs.get("kld_coefficient_max", 1.0)
        self.kld_coefficient_min = kwargs.get("kld_coefficient_min", 0.0)

        self.dagger_coefficient = kwargs.get("dagger_coefficient_max", 1.0)
        self.dagger_coefficient_max = kwargs.get("dagger_coefficient_max", 1.0)
        self.dagger_coefficient_min = kwargs.get("dagger_coefficient_min", 0.0)
        # self.velocity_estimator = VelocityEstimator(63, 512, 256, 3, 25)
        # self.trajectory = torch.zeros()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, kin_dict_info = None):
        self.kin_dict_info = kin_dict_info
        if not kin_dict_info is None:
            self.kin_optimizer = optim.Adam(self.actor_critic.parameters(), lr=5e-4)
            
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, kin_dict_info, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            
        if "kin_dict" in infos:
            self.transition.kin_dict = torch.cat([v.reshape(v.shape[0], -1) for k, v in infos['kin_dict'].items()], dim = -1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, epoch = 0, dagger_anneal = False, dagger_only=False):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_action_smoothness_loss = 0
        mean_kin_loss = 0
        
        anneal_start_epoch = 5000
        anneal_end_epoch = 10000
        min_val = self.kld_coefficient_min
        if epoch > anneal_start_epoch:
            self.kld_coefficient = (self.kld_coefficient_max - self.kld_coefficient_min) * max((anneal_end_epoch - epoch) / (anneal_end_epoch - anneal_start_epoch), 0) + min_val
        

        dagger_anneal_start_epoch = 0
        dagger_anneal_end_epoch = 2500
        dagger_min_val = self.dagger_coefficient_min
        if dagger_anneal and epoch > dagger_anneal_start_epoch and not self.kin_dict_info is None:
            self.dagger_coefficient = (self.dagger_coefficient_max - self.dagger_coefficient_min) * max((dagger_anneal_end_epoch - epoch) / (dagger_anneal_end_epoch - dagger_anneal_start_epoch), 0) + dagger_min_val
            print("current dagger coefficient: ", self.dagger_coefficient)
            print("current epoch: {}      dagger_anneal_start_epoch: {}      dagger_anneal_end_epoch: {}".format(epoch, dagger_anneal_start_epoch, dagger_anneal_end_epoch))
            
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
            
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, obs_squential, kin_dict_batch in generator:
                
                
                if self.kin_only:
                    batch_dict = {
                            'obs_batch': obs_batch,
                            'kin_dict_batch': kin_dict_batch,
                    }
                    kin_loss_info = self._optimize_kin(batch_dict)
                    mean_kin_loss += kin_loss_info['kin_loss'].item()
                    
                else:
                    # self.actor_critic.act(obs_squential, masks=masks_batch, hidden_states=hid_states_batch[0])
                    # mu_squential = self.actor_critic.action_mean

                    # mu_squential_t_lower = mu_squential[0:-1][:, :11]
                    # mu_squential_t_lower = mu_squential_t_lower.detach()
                    # mu_squential_tp1_lower = mu_squential[1:][:, :11]
                    # action_smoothness_loss_lower = torch.mean(torch.sum(torch.square(mu_squential_t_lower - mu_squential_tp1_lower), dim=-1))

                    # mu_squential_t_upper = mu_squential[0:-1][:, 11:]
                    # mu_squential_t_upper = mu_squential_t_upper.detach()
                    # mu_squential_tp1_upper = mu_squential[1:][:, 11:]
                    # action_smoothness_loss_upper = torch.mean(torch.sum(torch.square(mu_squential_t_upper - mu_squential_tp1_upper), dim=-1))

                    # action_smoothness_loss = (action_smoothness_loss_lower + action_smoothness_loss_upper * 0.1) * self.action_smoothness_coef
                    # action_smoothness_loss = action_smoothness_loss_lower * self.action_smoothness_coef
                    action_smoothness_loss = 0
                    

                    self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                    mu_batch = self.actor_critic.action_mean
                    sigma_batch = self.actor_critic.action_std
                    entropy_batch = self.actor_critic.entropy
                    
                    

                    # KL
                    if self.desired_kl != None and self.schedule == 'adaptive':
                        with torch.inference_mode():
                            kl = torch.sum(
                                torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                            kl_mean = torch.mean(kl)

                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                            
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.learning_rate

                    
                    

                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                    1.0 + self.clip_param)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()

                    
                    if dagger_anneal and not self.kin_dict_info is None:
                        surrogate_loss = surrogate_loss * (1 - self.dagger_coefficient)
                        value_loss = value_loss * (1 - self.dagger_coefficient)
                        action_smoothness_loss = action_smoothness_loss * (1 - self.dagger_coefficient)
                        loss = surrogate_loss + self.value_loss_coef * value_loss # no entropy loss
                    else:
                        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                    loss += action_smoothness_loss 
                    # Gradient step
                    self.optimizer.zero_grad()
                    if not dagger_only:
                        loss.backward()
                    else:
                        print("Dagger only, ignoring RL losses")
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()
                    mean_action_smoothness_loss += 0 # action_smoothness_loss.item()
                    
                    
                    if not self.kin_dict_info is None:
                        batch_dict = {
                            'obs_batch': obs_batch,
                            'kin_dict_batch': kin_dict_batch,
                            # 'action_batch': actions_batch.flatten(0, 1)
                        }
                        # import ipdb; ipdb.set_trace()
                        kin_loss_info = self._optimize_kin(batch_dict)
                        mean_kin_loss += kin_loss_info['kin_loss'].item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_action_smoothness_loss /= num_updates
        mean_kin_loss = mean_kin_loss / num_updates if not self.kin_dict_info is None else 0

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_action_smoothness_loss, mean_kin_loss
    
    def _optimize_kin(self, batch_dict):
        info_dict = {}
        kin_dict_batch = batch_dict['kin_dict_batch']
        obs_batch = batch_dict['obs_batch']
        # import ipdb; ipdb.set_trace()
        if len(kin_dict_batch.shape) == 3 and len(obs_batch.shape) == 3:
            obs_batch = obs_batch.flatten(0, 1)
            kin_dict_batch = kin_dict_batch.flatten(0, 1)
            kin_dict = self._assamble_kin_dict(kin_dict_batch)
        else: 
            kin_dict = self._assamble_kin_dict(kin_dict_batch)
        
        
        
        if self.z_type == "vae":
            pred_action, extra_dict = self.actor_critic.act_train(obs_batch, **kin_dict)
            gt_action = kin_dict['gt_action']
            kin_action_loss = torch.norm(pred_action - gt_action, dim=-1).mean()
            vae_mu, vae_log_var = extra_dict['vae_mu'], extra_dict['vae_log_var']
            prior_mu, prior_log_var = self.actor_critic.compute_prior(obs_batch)
            KLD = kl_multi(vae_mu, vae_log_var, prior_mu, prior_log_var).mean()
            
            # if humanoid_env.use_ar1_prior:
            #     time_zs = vae_mu.view(self.minibatch_size // self.horizon_length, self.horizon_length, -1)
            #     phi = 0.99
                
            #     error = time_zs[:, 1:] - time_zs[:, :-1] * phi
                
            #     idxes = kin_dict['progress_buf'].view(self.minibatch_size // self.horizon_length, self.horizon_length, -1)
                
            #     not_consecs = ((idxes[:, 1:] - idxes[:, :-1]) != 1).view(-1)
            #     error = error.view(-1, error.shape[-1])
            #     error[not_consecs] = 0
                
            #     starteres = ((idxes <= 2)[:, 1:] + (idxes <= 2)[:, :-1]).view(-1) # make sure the "drop" is not affected. 
            #     error[starteres] = 0
                
            #     ar1_prior = torch.norm(error, dim=-1).mean() 
            #     info_dict["kin_ar1"] = ar1_prior
            
            kin_loss = kin_action_loss +  KLD * self.kld_coefficient 
            
            self.kin_optimizer.zero_grad()
            kin_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.kin_optimizer.step()
        else:

            print("Teacher student training")
            # obs_batch = obs_batch[:10].clone()
            # gt_action = kin_dict['gt_action'][:10].clone()
            if self.actor_critic.is_recurrent:
                print("Using RNN (RNN/GRU/LSTM) action batch for Dagger")
                pred_action = self.actor_critic.action_mean.flatten(0, 1)
            else:
                print("Using MLP action batch for Dagger")
                pred_action = self.actor_critic.act_inference(obs_batch)
            gt_action = kin_dict['gt_action']
            # import ipdb; ipdb.set_trace()
            kin_loss = torch.norm(pred_action - gt_action, dim=-1).mean()  ## RMSE
            kin_loss = kin_loss * self.dagger_coefficient
            self.kin_optimizer.zero_grad()
            kin_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.kin_optimizer.step()
        info_dict.update({"kin_loss": kin_loss})
        return info_dict

    def _assamble_kin_dict(self, kin_dict_flat):
        B = kin_dict_flat.shape[0]
        len_acc = 0
        kin_dict = {}
        for k, v in self.kin_dict_info.items():
            kin_dict[k] = kin_dict_flat[:, len_acc:(len_acc + v[1][-1])].view(B, *v[0][1:])
            len_acc += v[1][-1]
        return kin_dict