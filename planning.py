import numpy as np
from Adafold.mpc.mpc_planner import MPC
import matplotlib.pyplot as plt
import copy
import torch
import copy
from Adafold.viz.viz_mpc import plot_vectors_3d, plot_action_selection, plot_prediction_and_state
from Adafold.trajectory.trajectory import Action_Sampler_Simple
from Adafold.mpc.mppi_planner import MPPI

from Adafold.args.arguments import get_argparse
import os


if __name__ == '__main__':
    args = get_argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and torch.cuda.get_device_name(torch.cuda.current_device()) != 'NVIDIA GeForce GTX 1080' and not args.train_cluster:
        torch.cuda.set_device(1)

    args.render = 0
    args.obs = 'mesh'       # [mesh, full_pcd]
    args.loss = 'MSE'
    args.reward = 'IoU'       # ['IoU', IoU_Gr]
    args.lr = 0.0001


    # Setting to evaluate multiple models if needed
    dir_models = {
        'mpc_mppi_Adafold':f'./data/checkpoint/D=100_obs={args.obs}_loss={args.loss}_K={args.K}_H={args.H}_zDim={args.z_dim}_mode=3_fusion=1_seed=1234',
    }
    
    conditionings = {'mpc_mppi_Adafold':3, }
    fusions = {'mpc_mppi_Adafold':1,}
    all_models = ['mpc_mppi_Adafold']
    exp_name = 'eval'

    for key in dir_models.keys():
        dir_results = f'./data/results/{exp_name}/{key}'
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)

    save_data = False
    dataset_path = None
    if save_data:
        # TODO: iterate multiple time for each environment!
        # create new folder on top of the already existing dataset
        dataset_path = f'./data/bootstrap_0'
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    mod = 'mpc_mppi_rand'       # sampling distribution initialization
    model = 'mpc_mppi_Adafold'
    args.dyn_conditioning = conditionings[model]
    args.fusion = fusions[model]
    A = 100
    H = 2
    init_rand = True
    args.reward = 'IoU_Gr'  # ['IoU', IoU_Gr]
    planner = MPPI(
        args,
        noise_mu=0.0,
        noise_sigma=0.01,
        lambda_param=2.0,
        A=A,
        H=H,
        mod=mod,  # ['mpc', 'rand', 'fixed']
        env_idx=0,
        load_model=True,
        init_rand=init_rand,
        dir_model=dir_models[model],
        dir_results=dir_results,
        save_datapoints=save_data,
        downsample=False,
        cumulative=True,
        save_pcd=False,
        verbose=False,
        path_dataset=dataset_path,
        env_params=[40, 60, 0.1, 4, 0],
        device=device
    )

    #close all figures
    plt.close('all')
    
    cumulative_reward, final_reward, frames = planner.control(iterations=1)
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_reward)
    plt.title(f"Model: {model}, H= {H}")
    plt.tight_layout()
    plt.show()

