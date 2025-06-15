import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import random
from Adafold.mpc.mpc_planner import MPC
from Adafold.mpc.mppi_planner import MPPI
from Adafold.mpc.Q_planner import Q_planner
from Adafold.model.model import RMA_MB
from Adafold.experiments_RAL.utils import get_fig_rewards, get_stored_params
from Adafold.model.CQL import CQL


def parallel_rollout(env_idx,
                     traj_idx,
                     max_traj_len,
                     num_trajs,
                     args,
                     mod,
                     H,
                     A,
                     sigma_noise,
                     lambda_param,
                     device,
                     dir_model,
                     exp_name,
                     cumulative,
                     w_cumul,
                     cost_coefficients,
                     shared_final_rewards,
                     shared_cumulative_rewards,
                     save_data,
                     dataset_path,
                     env_params=[[40, 60, 0.1, 4, 0]],
                     ):

    # set a new random seed using the current time and the process ID
    seed_value = int(time.time()) + os.getpid()
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

    params = env_params[env_idx]

    dir_results = f'./data/mpc_results/{exp_name}/H{H}_A{A}/{mod}/e_{env_idx}/t_{traj_idx}'
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    np.savetxt(dir_results + "/params.txt", params)

    model = RMA_MB(args, device).to(device)
    model.load_dict(model_path=dir_model + '/full_dict_model_val.pt', load_full=True, device=device)
    print('loaded model from path')
    model.eval()

    planner = MPPI(
        args,
        noise_mu=0.0,
        noise_sigma=sigma_noise,
        lambda_param=lambda_param,
        A=A,
        H=H,
        mod=mod,         # ['mpc', 'rand', 'fixed', ]
        env_idx=env_idx,
        load_model=True,
        model=model,
        cumulative=cumulative,
        cost_coefficients=cost_coefficients,
        w_cumul=w_cumul,
        dir_model=dir_model,
        dir_results=dir_results,
        save_datapoints=save_data,
        downsample=False,
        save_pcd=True,
        verbose=False,
        path_dataset=dataset_path,
        env_params=params,
        device=device
    )

    planning_results = planner.control(iterations=1, params=params)
    cumulative_reward, final_reward, frames = planning_results

    cumulative_reward = np.asarray(cumulative_reward)
    shared_cumulative_rewards[(num_trajs*env_idx) + traj_idx*max_traj_len: (num_trajs*env_idx) + traj_idx*max_traj_len + cumulative_reward.shape[-1]] = cumulative_reward
    shared_final_rewards[(num_trajs*env_idx) + traj_idx] = final_reward

    result_final_reward = f'Final reward: {final_reward}'
    # Writing both results to a file
    with open(dir_results + "/final_reward.txt", 'w') as file:
        file.write(result_final_reward + '\n')

    episode_reward_fig = get_fig_rewards(args.reward, mod, cumulative_reward)
    episode_reward_fig.savefig(dir_results + "/episode_reward.png")

    return cumulative_reward, final_reward, frames


def full_rollout(args,
                 num_envs, #TODO: to add to everyone!
                 num_trajs,
                 methods,
                 H,
                 A,
                 exp_name,
                 dir_models,
                 fusions,
                 cumulative,
                 w_cumul,
                 sigma_noise=0.1,
                 lambda_param=0.01,
                 cost_coefficients=[1., 1., 0.01, 1.],
                 num_processes=1,
                 save_data=False,
                 parallelize=True,
                 device='cpu'):

    for mod in methods:
        dataset_path = None
        if save_data:
            # Define dataset name and folder
            dataset_path = f'./data/dataset/{exp_name}/H{H}_A{A}/{mod}'
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

        if 'mpc_NoAda' in mod:
            args.dyn_conditioning = 0
            args.fusion = 0
        if 'mpc_Ada' in mod:
            args.dyn_conditioning = 3
            args.fusion = fusions[mod]

        print(f'Rollout for experiment: {mod}')

        max_traj_len = 25
        default_value = -1
        shared_final_rewards = mp.Array('f', num_trajs * num_envs)
        shared_cumulative_rewards = mp.Array('f', [default_value] * num_trajs * num_envs * max_traj_len)

        if num_envs == 1:
            env_params = [[40, 60, 0.1, 4, 0]]
        else:
            # env_params = [[40, 60, 0.1, 4, 0]]
            env_params = get_stored_params(id_config='240113').tolist()
            while len(env_params) < num_envs:
                print('Adding new params to the list')
                # sample elast in the range 40-100
                elast = np.random.randint(40, 100)
                bend = np.random.randint(40, 100)
                env_params.append([elast, bend, 0.1, 4, 0])

        if not parallelize:
            for env_idx in range(num_envs):
                for traj_idx in range(num_trajs):
                    parallel_rollout(env_idx,
                                     traj_idx,
                                     max_traj_len,
                                     num_trajs,
                                     args,
                                     mod,
                                     H,
                                     A,
                                     sigma_noise,
                                     lambda_param,
                                     device,
                                     dir_models[mod],
                                     exp_name,
                                     cumulative,
                                     w_cumul,
                                     cost_coefficients,
                                     shared_final_rewards,
                                     shared_cumulative_rewards,
                                     save_data,
                                     dataset_path,
                                     env_params=env_params)
        else:
            for env_idx in range(num_envs):
                for i in range(num_trajs // num_processes + 1):
                    processes = []
                    for j in range(num_processes):
                        traj_idx = num_processes * i + j
                        if traj_idx >= num_trajs:
                            break
                        processes.append(mp.Process(target=parallel_rollout,
                                                    args=(env_idx,
                                                          traj_idx,
                                                          max_traj_len,
                                                          num_trajs,
                                                          args,
                                                          mod,
                                                          H,
                                                          A,
                                                          sigma_noise,
                                                          lambda_param,
                                                          device,
                                                          dir_models[mod],
                                                          exp_name,
                                                          cumulative,
                                                          w_cumul,
                                                          cost_coefficients,
                                                          shared_final_rewards,
                                                          shared_cumulative_rewards,
                                                          save_data,
                                                          dataset_path,
                                                          env_params)))

                    # Start the processes
                    for process in processes:
                        process.start()
                    # Wait for the processes to complete
                    for process in processes:
                        process.join()

        final_rewards = np.asarray(list(shared_final_rewards)).reshape(num_envs, num_trajs, 1)
        cumulative_rewards = np.asarray(list(shared_cumulative_rewards)).reshape(num_envs, num_trajs, max_traj_len)
        cumulative_rewards = np.asarray([cumulative_rewards[l, cumulative_rewards[0] > default_value] for l in range(len(cumulative_rewards))])

        # TODO: improve results folder
        results_folder = f'./data/results/{exp_name}/H{H}_A{A}/{mod}'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        np.save(results_folder + f'/cumulative_reward.npy', cumulative_rewards)
        np.save(results_folder + f'/final_reward.npy', final_rewards)





