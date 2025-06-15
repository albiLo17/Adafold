import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from matplotlib import pyplot as plt

import os.path as osp

from assistive_gym.utils.utils import save_numpy_as_gif

from half_folding.arguments import get_argparse

from assistive_gym.envs.half_folding import HalfFoldEnv
from utils.normalized_env import normalize

import gym

from PPO import PPO
import pybullet as p


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "HalfFoldEnv"
    save_video_dir = './data/'
    has_continuous_action_space = True
    max_ep_len = 50 #1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1# 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = normalize(HalfFoldEnv(frame_skip=10, hz=50, action_mult=0.02))
    if render:
        env.render()
    # env.render(width=640, height=480)
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n


    # preTrained weights directory
    args = get_argparse()
    args.K = 1
    args.dyn_conditioning = 0
    args.batch_norm = 0

    random_seed = 1             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 3      #### set this to load a particular checkpoint num

    exp_name = 'pointcloud'
    directory = "PPO_preTrained" + '/' + env_name + '/' + exp_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim,
                    action_dim,
                    lr_actor,
                    lr_critic,
                    gamma,
                    K_epochs,
                    eps_clip,
                    has_continuous_action_space,
                    action_std_init=action_std,
                    args=args,
                    use_pointnet=True,
                    different_encoders=True)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        img_size = 720
        frames = []
        rewards = []

        if render:
            env.render()
            time.sleep(frame_delay)

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)
            print(action)
            ep_reward += reward

            # _, _, img, _, _ = p.getCameraImage(width=640, height=480,
            #                                viewMatrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, -0.5, 1.2], distance=1., yaw=180.0, pitch=-0.3, roll=0, upAxisIndex=2),
            #                                # viewMatrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.0, -0.5, 1.5], distance=1., yaw=90.0, pitch=-90., roll=0, upAxisIndex=2),
            #                                projectionMatrix=p.computeProjectionMatrixFOV(fov=60, aspect=float(640)/480, nearVal=0.01, farVal=100.0))
            _, _, img, _, _ = p.getCameraImage(width=640, height=480,
                                               viewMatrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.0, -0.0, 0.0], distance=0.45, yaw=-45., pitch=-40, roll=0, upAxisIndex=2),
                                               # viewMatrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.0, -0.5, 1.5], distance=1., yaw=90.0, pitch=-90., roll=0, upAxisIndex=2),
                                               projectionMatrix=p.computeProjectionMatrixFOV(fov=60, aspect=float(640) / 480, nearVal=0.01, farVal=100.0))

            frames.append(np.asarray(img).reshape((480, 640, 4))[:, :, :].astype(np.uint8))
            rewards.append(reward)

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    # Plot reward
    fig = plt.figure()
    plt.plot(rewards)
    plt.title('Episode rewards')
    plt.show()

    if save_video_dir is not None:
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        save_name = osp.join(save_video_dir, env_name + f'_PPO_{run_num_pretrained}.gif')
        save_numpy_as_gif(np.array(frames[:]), save_name, fps=5)
        print('Video generated and save to {}'.format(save_name))

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()