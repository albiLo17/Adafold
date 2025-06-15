import os
import matplotlib.pyplot as plt
import os
from matplotlib import collections
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# WandB – Import the wandb library
import wandb
from Adafold.utils import utils_main
import torch
import numpy as np

class wandb_logger():
    def __init__(self, args, run_name=None, mode='dryrun', original_loss=None, single_env=False, data_name=None):

        self.run_name = run_name
        if self.run_name is None:
            self.run_name = f'D={args.train_trajs}_obs={args.obs}_loss={args.loss}_K={args.K}_H={args.H}_zDim={args.z_dim}_mode={args.dyn_conditioning}_fusion={args.fusion}_seed={args.seed}'
        if args.batch_size != 32:
            self.run_name += '_bs=' + str(args.batch_size)
        if args.l2_reg != 0:
            self.run_name += '_l2'
        if args.dropout != 0:
            self.run_name += '_dropout=0.2'

        # WANDB_MODE=online to enable cloud syncing
        os.environ["WANDB_MODE"] = mode

        wandb_dir = './wandb_dir_RAL'
        if not os.path.exists(os.path.join(wandb_dir, 'wandb')):
            os.makedirs(os.path.join(wandb_dir, 'wandb'), exist_ok=True)

        self.run = wandb.init(entity="albilo", project="AdaFold_Online2", name=self.run_name, dir=wandb_dir)

        print(f'Starting experiment: {self.run_name}')
        wandb.config.update(args)

        self.checkpoint_folder = args.checkpoint + '/' + self.run_name + '/'
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        # if not os.path.exists(self.checkpoint_folder):
        #     utils_main.make_dir(self.checkpoint_folder)

    def watch(self, model):
        wandb.watch(model)

    def log(self, info):
        wandb.log(info)

    def close(self):
        self.run.finish()
