import torch
import numpy as np
import random
import os
import sys
sys.path.append(os.getcwd())

from Adafold.args.arguments import get_argparse
from Adafold.model.model import RMA_MB
from Adafold.dataloader.dataloader_half import PointcloudDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Adafold.viz.viz_mpc import plot_prediction_and_state_singleview
from Adafold.utils.logger import wandb_logger
from Adafold.utils.utils_main import load_datasets


def main(args=None):

    # args = wandb.config
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    if torch.cuda.is_available() and torch.cuda.get_device_name(torch.cuda.current_device()) != 'NVIDIA GeForce GTX 1080' and not args.train_cluster:
        torch.cuda.set_device(1)
    torch.cuda.empty_cache()

    logger = wandb_logger(args, mode="dryrun")    #  "run", "dryrun"
    num_folders_train = args.train_folders
    num_trajs = args.train_trajs
    num_folders_val = 0
    if args.validation:
        num_folders_val = 100# num_folders_train # last add 30/11

    print('*****************************')
    print(f"Observations: {args.obs}")
    print(f"Loss: {args.loss}")
    print(f"Epochs: {args.epochs}")
    print(f"Lr: {args.lr}")
    print()
    print(f"Horizon: {args.H}")
    print(f"Conditioning: {args.dyn_conditioning}")
    print(f"Fusion: {args.fusion}")
    print('*****************************')

    # Dataset Folders
    dataset_train, dataset_test, dataset_val = load_datasets(args=args,
                                                             dataload=PointcloudDataset,
                                                             checkpoint_folder=logger.checkpoint_folder,
                                                             num_folders_train=num_folders_train,
                                                             num_folders_val=num_folders_val,
                                                             num_envs=-1,
                                                             num_trajectories=num_trajs,)

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    if args.validation:
        val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = RMA_MB(args, device).to(device)
    logger.watch(model)

    best_tot_loss = 10000
    best_tot_loss_val = 1000

    for e in tqdm(range(args.epochs)):

        train_loss_info, train_image_info = model.train_epoch(train_dataloader)
        logger_dict = train_loss_info

        if args.validation:
            val_loss_info, val_image_info = model.val_epoch(val_dataloader)
            logger_dict.update(val_loss_info)

        # Get images and log them
        if e % int(args.epochs / 10) or e == args.epochs - 1:
            train_fig = plot_prediction_and_state_singleview(
                points=train_image_info['label'].detach().cpu().numpy(),
                predicted_points=train_image_info['pred'].detach().cpu().numpy(),
                title=None,
                view='top')
            # logger_dict.update({'Train/Img': train_fig})

            if args.validation:
                val_fig = plot_prediction_and_state_singleview(
                    points=val_image_info['label'].detach().cpu().numpy(),
                    predicted_points=val_image_info['pred'].detach().cpu().numpy(),
                    title=None,
                    view='top')

                # logger_dict.update({'Val/Img': val_fig})

        logger.log(logger_dict)

        if train_loss_info['Train/Loss'] < best_tot_loss:
            # Save
            best_tot_loss = train_loss_info['Train/Loss']
            torch.save(model.state_dict(), logger.checkpoint_folder + 'full_dict_model_train.pt')
            print()

        if args.validation:
            if val_loss_info['Val/Loss'] < best_tot_loss_val:
                # Save
                best_tot_loss_val = train_loss_info['Val/Loss']
                torch.save(model.state_dict(), logger.checkpoint_folder + 'full_dict_model_val.pt')



if __name__ == '__main__':
    args = get_argparse()
    # args.H = 2
    args.epochs = 800
    # args.batch_size = 64
    main(args)
