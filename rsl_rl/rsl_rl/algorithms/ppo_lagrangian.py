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

from rsl_rl.modules import ActorCriticCost
from rsl_rl.storage import RolloutStorageExtend

class PPOLagrangian:
    actor_critic: ActorCriticCost
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 cost_gamma=0.998,
                 cost_lam=0.95,
                 value_loss_coef=1.0,
                 cost_value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 penalty_lr=5e-2,
                 max_grad_norm=1.0,
                 cost_limit=0,
                 use_clipped_value_loss=True,
                 use_clipped_cost_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
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
        self.transition = RolloutStorageExtend.Transition()

        # penalty params
        self.penalty_param = torch.tensor(1.0,requires_grad=True).float()
        self.penalty_optimizer = optim.Adam([self.penalty_param], lr=penalty_lr)
        self.cost_limit = cost_limit

        # PPO-Lagragian parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_clipped_cost_value_loss = use_clipped_cost_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
        self.storage = RolloutStorageExtend(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            raise NotImplementedError
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.cost_evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.cost_rewards = infos['cost']
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.cost_rewards += self.cost_gamma * torch.squeeze(self.transition.cost_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        last_cost_values = self.actor_critic.cost_evaluate(last_critic_obs).detach()
        self.storage.compute_returns_and_costs(last_values, last_cost_values, self.gamma, self.lam, self.cost_gamma, self.cost_lam)

    def update(self, **args):
        current_cost = args.get('current_cost')
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_surrogate_loss = 0
        mean_cost_surrogate_loss = 0.
        mean_penalty_loss = 0.
        mean_penalty_param = 0.

        penalty_param = self.penalty_param
        # Penalty Loss
        # penalty_loss = -penalty_param * (current_cost - self.cost_limit).mean()
        penalty_loss = -penalty_param * (current_cost - self.cost_limit)
        
        self.penalty_optimizer.zero_grad()
        penalty_loss.backward()
        self.penalty_optimizer.step()
        mean_penalty_param = penalty_param

        
        if self.actor_critic.is_recurrent:
            raise NotImplementedError
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, \
                target_values_batch, advantages_batch, returns_batch, \
                target_cost_values_batch, cost_advantages_batch, cost_returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                cost_value_batch = self.actor_critic.cost_evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                
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


                # Cost Surrogate Loss
                ## not clip the cost
                cost_surrogate = torch.squeeze(cost_advantages_batch) * ratio
                cost_surrogate_loss = cost_surrogate.mean()
                ## clip the cost
                # cost_surrogate_clipped = torch.squeeze(cost_advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                            # 1.0 + self.clip_param)
                # cost_surrogate_loss = (torch.min(cost_surrogate, cost_surrogate_clipped)).mean()





                # Cost Value loss
                if self.use_clipped_cost_value_loss:
                    cost_value_clipped = target_cost_values_batch + (cost_value_batch - target_cost_values_batch).clamp(-self.clip_param,
                                                                                                                        self.clip_param)
                    cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                    cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                    cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
                else:
                    cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()
                

                # Penalty Item
                penalty_item = torch.nn.functional.softplus(penalty_param).item()

                loss = (surrogate_loss + cost_surrogate_loss*penalty_item)/(1+penalty_item) \
                        + self.value_loss_coef * value_loss \
                        + self.cost_value_loss_coef * cost_value_loss \
                        - self.entropy_coef * entropy_batch.mean() \
                        # + penalty_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_cost_surrogate_loss += cost_surrogate_loss.item()
                mean_penalty_loss += penalty_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_cost_surrogate_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_penalty_loss /= num_updates


        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_cost_value_loss, mean_cost_surrogate_loss, mean_penalty_loss, mean_penalty_param
