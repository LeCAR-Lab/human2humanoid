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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.actor_critic import get_activation, ActorCritic


class ActorCriticPULSE(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        embedding_size = kwargs['embedding_size']
        use_vae_prior = kwargs['use_vae_prior']
        self.use_vae_clamped_prior = kwargs['use_vae_clamped_prior']
        self.vae_var_clamp_max = kwargs['vae_var_clamp_max']
        self.self_obs_size = kwargs['self_obs_size']


        # Encoder
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(activation)
        self.encoder = nn.Sequential(*actor_layers)
        self.encoder_mu = nn.Linear(actor_hidden_dims[-1], embedding_size)
        self.encoder_logvar = nn.Linear(actor_hidden_dims[-1], embedding_size)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(embedding_size, actor_hidden_dims[0]))
        decoder_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                decoder_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                decoder_layers.append(activation)
        self.decoder = nn.Sequential(*decoder_layers)

        if use_vae_prior:
            # Prior
            prior_layers = []
            prior_layers.append(nn.Linear(self.self_obs_size, actor_hidden_dims[0]))
            prior_layers.append(activation)
            for l in range(len(actor_hidden_dims)- 1):
                prior_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                prior_layers.append(activation)
            self.prior = nn.Sequential(*prior_layers)
            self.prior_mu = nn.Linear(actor_hidden_dims[-1], embedding_size)
            self.prior_logvar = nn.Linear(actor_hidden_dims[-1], embedding_size)


        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Decoder MLP: {self.decoder}")
        print(f"Prior MLP: {self.prior}")

        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.std.requires_grad = True
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def form_embedding(self, observations, obs_dict = None):
        extra_dict = {}
        task_out_z = self.encoder(observations)
        self.vae_mu = vae_mu = self.encoder_mu(task_out_z)
        self.vae_log_var = vae_log_var = self.encoder_logvar(task_out_z)
        
        if self.use_vae_clamped_prior:
            self.vae_log_var = vae_log_var = torch.clamp(vae_log_var, min = -5, max = self.vae_var_clamp_max)
        
        if "z_noise"  in obs_dict and self.training: # bypass reparatzation and use the noise sampled during training. 
            task_out_proj = vae_mu + torch.exp(0.5*vae_log_var) * obs_dict['z_noise']
        else:
            task_out_proj, self.z_noise = self.reparameterize(vae_mu, vae_log_var)
                    
        extra_dict = {"vae_mu": vae_mu, "vae_log_var": vae_log_var, "noise": self.z_noise}
        return task_out_proj, extra_dict


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, eps
    
    def act(self, observations, **kwargs):
        actions_mean, extra_dict = self.act_train(observations, **kwargs)
        self.distribution = Normal(actions_mean, actions_mean*0. + self.std)
        return actions_mean
    
    def act_train(self, observations, **kwargs):
        z_out, extra_dict = self.form_embedding(observations=observations, obs_dict=kwargs)
        actions_mean = self.decoder(z_out)
        return actions_mean, extra_dict
    
    def act_inference(self, observations, **kwargs):
        z_out, extra_dict = self.form_embedding(observations=observations, obs_dict=kwargs)
        actions_mean = self.decoder(z_out)
        return actions_mean
    
    def compute_prior(self, observations):
        self_obs = observations[:, :self.self_obs_size]
        
        prior_latent = self.prior(self_obs)
        prior_mu = self.prior_mu(prior_latent)
        prior_logvar = self.prior_logvar(prior_latent)
        if self.use_vae_clamped_prior:
            prior_logvar = torch.clamp(prior_logvar, min = -5, max = self.vae_var_clamp_max)
        return prior_mu, prior_logvar
        