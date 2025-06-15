import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add needed environmental paths
import os
# os.environ['PYFLEXROOT'] = os.environ['PWD'] + "/PyFlex"
# os.environ['LD_LIBRARY_PATH'] = os.environ['PYFLEXROOT'] + "/external/SDL2-2.0.4/lib/x64"
import argparse

from assistive_gym.envs.half_folding import HalfFoldEnv
from utils.normalized_env import normalize
import pybullet as p

import warnings
warnings.filterwarnings('ignore', message="From .* import .* is deprecated")

from PPO import PPO
from half_folding.arguments import get_argparse

# def get_env():
#     parser = argparse.ArgumentParser(description='Process some integers.')
#     # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFoldElas', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
#     parser.add_argument('--env_name', type=str, default='ClothFoldElas')
#     parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
#     parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
#     parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
#     parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
#     parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
#     parser.add_argument('--save_data', type=bool, default=False, help='save trajectory in a folder')
#
#     args = parser.parse_args()
#
#     env_kwargs = env_arg_dict[args.env_name]
#
#     # Generate and save the initial states for running this environment for the first time
#     env_kwargs['use_cached_states'] = False
#     env_kwargs['observation_mode '] = 'point_cloud'  # cam_rgb, point_cloud, key_point
#     env_kwargs['save_cached_states'] = False
#     env_kwargs['num_variations'] = args.num_variations
#     env_kwargs['render'] = True
#     env_kwargs['headless'] = args.headless
#
#     if not env_kwargs['use_cached_states']:
#         print('Waiting to generate environment variations. May take 1 minute for each variation...')
#
#     if not env_kwargs['use_cached_states']:
#         print('Waiting to generate environment variations. May take 1 minute for each variation...')
#
#     env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
#
#     return env


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "HalfFoldEnv"


    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 30 #1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    # print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    print_freq = max_ep_len * 2  # print avg reward in the interval (in num timesteps)

    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = 1000 # int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 25000 #int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    # update_timestep = max_ep_len * 4  # update policy every n timesteps
    # update_timestep = max_ep_len * 2  # update policy every n timesteps
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update
    K_epochs = 40  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    # random_seed = 2  # set random seed if required (0 = no random seed)
    # random_seed = 3  # set random seed if required (0 = no random seed)
    random_seed = 1  # set random seed if required (0 = no random seed)

    ######## PARAMS FOR DEBUG ############
    # max_ep_len = 5
    # random_seed = 9999
    # update_timestep = max_ep_len


    ##################### ENVIRONMENT ################################

    print("training environment name : " + env_name)

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFoldElas', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    # parser.add_argument('--env_name', type=str, default='ClothFoldElas')
    # parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    # parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    # parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    # parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    # parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    # parser.add_argument('--save_data', type=bool, default=False, help='save trajectory in a folder')
    #
    # args = parser.parse_args()
    #
    # env_kwargs = env_arg_dict[args.env_name]
    #
    # # Generate and save the initial states for running this environment for the first time
    # env_kwargs['use_cached_states'] = False
    # env_kwargs['observation_mode '] = 'point_cloud'  # cam_rgb, point_cloud, key_point
    # env_kwargs['save_cached_states'] = False
    # env_kwargs['num_variations'] = args.num_variations
    # env_kwargs['render'] = True
    # env_kwargs['headless'] = args.headless
    #
    # if not env_kwargs['use_cached_states']:
    #     print('Waiting to generate environment variations. May take 1 minute for each variation...')
    #
    # if not env_kwargs['use_cached_states']:
    #     print('Waiting to generate environment variations. May take 1 minute for each variation...')
    #
    # env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env = normalize(HalfFoldEnv(frame_skip=10, hz=50, action_mult=0.02))
    # TODO: debug this


    # env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    exp_name = 'pointcloud'

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_exp_" + str(exp_name) + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################


    run_name = f"{env_name}__exp_name:{exp_name}__run_num{run_num}"
    writer = SummaryWriter(f"PPO_logs/logs/{run_name}")

    ################### checkpointing ###################
    run_num_pretrained = run_num  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/' + exp_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    args = get_argparse()
    args.K = 1
    args.dyn_conditioning = 0
    args.batch_norm = 0

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

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        elas, bend, damp, frict = 20.0, 20., 1.5, 1.5
        state = env.reset(stiffness=[elas, bend, damp], friction=frict)

        current_ep_reward = 0
        episode_steps = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            episode_steps += 1

            # update PPO agent
            if time_step % update_timestep == 0:
                policy_loss, value_loss, dist_entropy, values, expl_var = ppo_agent.update()

                writer.add_scalar("Losses/policy_loss", policy_loss, i_episode + 1)
                writer.add_scalar("Losses/value_loss", value_loss, i_episode + 1)

                writer.add_scalar("Policy/entropy", dist_entropy, i_episode + 1)
                writer.add_scalar("Policy/actor_learning_rate", ppo_agent.optimizer.param_groups[0]["lr"],
                                  i_episode + 1)
                writer.add_scalar("Policy/critic_learning_rate", ppo_agent.optimizer.param_groups[1]["lr"],
                                  i_episode + 1)
                writer.add_scalar("Policy/value_estimate", values.mean(), i_episode + 1)
                writer.add_scalar("Policy/explained_variance", expl_var, i_episode + 1)


            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        # update the policy every episode, might need to decrease it
        # policy_loss, value_loss, dist_entropy, values, expl_var = ppo_agent.update()
        if writer is not None:
            writer.add_scalar("Env/cumulative_reward", current_ep_reward, i_episode + 1)
            writer.add_scalar("Env/final_reward", reward, i_episode + 1)
            writer.add_scalar("Env/episode_length", episode_steps, i_episode + 1)

            # writer.add_scalar("Losses/policy_loss", policy_loss, i_episode + 1)
            # writer.add_scalar("Losses/value_loss", value_loss, i_episode + 1)
            # writer.add_scalar("Policy/entropy", dist_entropy, i_episode + 1)
            # writer.add_scalar("Policy/actor_learning_rate", ppo_agent.optimizer.param_groups[0]["lr"],
            #                   i_episode + 1)
            # writer.add_scalar("Policy/critic_learning_rate", ppo_agent.optimizer.param_groups[1]["lr"],
            #                   i_episode + 1)
            # writer.add_scalar("Policy/value_estimate", values.mean(), i_episode + 1)
            # writer.add_scalar("Policy/explained_variance", expl_var, i_episode + 1)




        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()






