import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch
from termcolor import colored

command_state = {
    'vel_forward': 0.0,
    'vel_side': 0.0,
    'orientation': 0.0,
}

override = False

EXPORT_ONNX = True

def play(args):


    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.env.num_envs = 2
    env_cfg.viewer.debug_viz = True
    env_cfg.motion.visualize = False
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.mesh_type = 'plane'
    if env_cfg.terrain.mesh_type == 'trimesh':
        env_cfg.terrain.terrain_types = ['flat', 'rough', 'low_obst', 'smooth_slope', 'rough_slope']  # do not duplicate!
        # env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.5, 0.5]
        # env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 20
    env_cfg.domain_rand.randomize_rfi_lim = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.ctrl_delay_step_range = [2, 2]
    env_cfg.asset.termination_scales.max_ref_motion_distance = 1

    env_cfg.env.test = True
    if args.joystick:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        from pynput import keyboard
        from legged_gym.utils import key_response_fn

    # prepare environment
    # import ipdb; ipdb.set_trace()
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 3 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards


    obs = env.get_observations()
    #import ipdb; ipdb.set_trace()
    # obs[:, 9:12] = torch.Tensor([0.5, 0, 0])
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    exported_policy_name = str(task_registry.loaded_policy_path.split('/')[-2]) + str(task_registry.loaded_policy_path.split('/')[-1])
    print('Loaded policy from: ', task_registry.loaded_policy_path)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, exported_policy_name)
        print('Exported policy as jit script to: ', os.path.join(path, exported_policy_name))
    if EXPORT_ONNX:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_onnx(ppo_runner.alg.actor_critic, path, exported_policy_name)
        print('Exported policy as onnx to: ', os.path.join(path, exported_policy_name))

    if args.joystick:
        print(colored("joystick on", "green"))
        key_response = key_response_fn(mode='vel')
        def on_press(key):
            global command_state
            try:
                # print(key.char)
                key_response(key, command_state, env)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    for i in range(10*int(env.max_episode_length)):
        # obs[:, -19:] = 0 # will destroy the performance
        actions = policy(obs.detach())
        # print(torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1))
        obs, _, rews, dones, infos = env.step(actions.detach())
        # print("obs = ", obs)
        # print("actions = ", actions)
        # print()
        # exit()
        if override: 
            obs[:,9] = 0.5
            obs[:,10] = 0.0
            obs[:,11] = 0.0
        
        # overwrite linear velocity - z and angular velocity - xy
        # obs[:, 40] = 0.
        # obs[: 41:43] = 0.

        
        

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    # 'dof_pos_target': env.actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            pass
            # logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)