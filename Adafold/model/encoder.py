import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from Adafold.model.base_models import PointNetHFE, PointNetDecoder
from Adafold.model.utils import mlp, orthogonal_init, mse, l1, ema
from Adafold.viz.viz_mpc import plot_state_singleview, plot_state_rec_singleview
import argparse
from Adafold.dataloader.dataloader_half_encoder import PointcloudAEDataset
from Adafold.utils.utils_main import load_datasets
from torch_geometric.loader import DataLoader
import random
import os
import wandb
from Adafold.model.utils import count_parameters
import time

class PointAE(nn.Module):

    def __init__(self, args, device, front_encoder=False):
        super().__init__()
        self.args = args
        self.front_encoder = front_encoder

        self._encoder = PointNetHFE(args=args,
                                    node_feat_dim=3,    # only the positions of the points
                                    globalSA=True)

        self._encoder_front = PointNetHFE(args=args,
                                          node_feat_dim=3,    # only the positions of the points
                                          globalSA=True)

        self._decoder = PointNetDecoder(args=args,
                                        init_input_dim=3,
                                        skip=args.skip_connections,
                                        classification=False)

    def h(self, obs, intermediate=False):
        """Encodes an observation into its latent representation (h).
         obs: should be a batch of poitnclouds to be encoded """
        # if intermediate is True, this is sa0_out, sa1_out, sa2_out, sa3_out, else only sa3_out
        sa0_out, sa1_out, sa2_out, sa3_out = self._encoder(obs, intermediate=True)
        if intermediate:
            return (sa0_out, sa1_out, sa2_out, sa3_out)
        else:
            return sa3_out

    def h_front(self, obs, intermediate=False):
        """Encodes an observation into its latent representation (h).
         obs: should be a batch of poitnclouds to be encoded """
        # if intermediate is True, this is sa0_out, sa1_out, sa2_out, sa3_out, else only sa3_out
        HF_h = self._encoder_front(obs, intermediate=intermediate)
        return HF_h

    def d(self, sa0_out, sa1_out, sa2_out, sa3_out):
        hat_obs = self._decoder(sa0_out, sa1_out, sa2_out, sa3_out)
        return hat_obs

    def forward(self, x, x_front=None):
        sa0_out, sa1_out, sa2_out, sa3_out = self.h(x, intermediate=True)

        if self.front_encoder and x_front is not None:
            sa3_out_front = self.h_front(x, intermediate=False)
            x_hat = self.d(sa0_out, sa1_out, sa2_out, sa3_out_front)
            h_front = sa3_out_front[0]
            h = sa3_out[0]
            return x_hat, h_front, h

        x_hat = self.d(sa0_out, sa1_out, sa2_out, sa3_out)
        return x_hat


    def save_model(self, path, modules=None):
        """
        Saves the specified modules of the model.
        If no modules are specified, saves the entire model.
        """
        if modules is None:
            torch.save(self.state_dict(), path)
        else:
            torch.save({name: module.state_dict() for name, module in self.named_children() if name in modules}, path)

    def load_model(self, path, modules=None):
        """
        Loads the specified modules of the model.
        If no modules are specified, loads the entire model.
        """
        state_dict = torch.load(path)
        if modules is None:
            self.load_state_dict(state_dict)
        else:
            for name, module in self.named_children():
                if name in modules:
                    module.load_state_dict(state_dict[name])

    def freeze_modules(self, modules):
        """
        Freezes the specified modules in the model.
        """
        for name, module in self.named_children():
            if name in modules:
                for param in module.parameters():
                    param.requires_grad = False


def train_full_encoder(model, train_loader, device, optimizer):
    model.train()
    loss = nn.MSELoss()

    example_fig = None
    total_loss = 0
    for i, data in enumerate(train_loader):
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        reconstruction_loss = loss(out, data.y)
        reconstruction_loss.backward()
        optimizer.step()
        total_loss += reconstruction_loss.item()

        if example_fig is None:
            points = data.pos[data.batch==0].clone().cpu()
            reconstruction = out[data.batch==0].clone().detach().cpu()
            example_fig = plot_state_rec_singleview(points=points, full_points=reconstruction, view='top')

    train_loss = total_loss / i
    return (train_loss, example_fig)

@torch.no_grad()
def test_full_encoder(model, test_loader, device):
    model.eval()
    loss = nn.MSELoss()

    example_fig = None
    total_loss = 0
    for i, data in enumerate(test_loader):
        data = data[0]
        data = data.to(device)
        out = model(data)

        reconstruction_loss = loss(out, data.y)
        total_loss += reconstruction_loss.item()

        if example_fig is None:
            points = data[0].pos.clone().cpu()
            reconstruction = out[data.batch == 0].clone().cpu()
            example_fig = plot_state_rec_singleview(points=points, full_points=reconstruction, view='top')

    test_loss = total_loss / i
    return (test_loss, example_fig)

def full_encoder_trainer(args, device):
    model = PointAE(args, device, front_encoder=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    print(f'The encoder model:\n{model}')
    print(f'Parameters: {count_parameters(model)}.\n')
    start = time.time()

    best_test_loss = 10000

    # Train and test!
    for epoch in range(1, args.epochs + 1):

        train_loss, train_fig = train_full_encoder(model, train_loader, device, optimizer)
        test_loss, test_fig = test_full_encoder(model, test_loader, device)
        if scheduler is not None:
            scheduler.step()

        print(f'Epoch: {epoch:02d}, TrLoss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        wandb_dict = {
            'epoch': epoch,
            'elapsed_t': time.time() - start,
            'train/loss': train_loss,
            'train/img': train_fig,
            'test/img': test_fig,
            'test/loss': test_loss,
        }

        if wandb_dict['test/loss'] < best_test_loss:
            # Save
            best_test_loss = wandb_dict['test/loss']
            torch.save(model.state_dict(), checkpoint_folder + 'full_dict_model_val.pt')

        wandb.log(wandb_dict)


def train_front_encoder(model, train_loader, device, optimizer):
    model.train()
    loss = nn.MSELoss()

    example_fig = None
    tot_rec_loss = 0
    total_loss = 0
    for i, data in enumerate(train_loader):
        data_full = data[0].to(device)
        data_front = data[1].to(device)
        optimizer.zero_grad()

        x_hat, h_front, h = model(data_full, x_front=data_front)

        reconstruction_loss = loss(x_hat, data_full.y)
        latent_loss = loss(h_front, h)
        latent_loss.backward()
        optimizer.step()
        tot_rec_loss += reconstruction_loss.item()
        total_loss += latent_loss.item()

        if example_fig is None:
            points = data_full.pos[data_full.batch==0].clone().cpu()
            reconstruction = x_hat[data_full.batch==0].clone().detach().cpu()
            example_fig = plot_state_rec_singleview(points=points, full_points=reconstruction, view='top')

    train_rec_loss = tot_rec_loss / i
    train_loss = total_loss / i
    return (train_loss, train_rec_loss, example_fig)



@torch.no_grad()
def test_front_encoder(model, test_loader, device):
    model.eval()
    loss = nn.MSELoss()

    example_fig = None
    tot_rec_loss = 0
    total_loss = 0
    for i, data in enumerate(test_loader):
        data_full = data[0].to(device)
        data_front = data[1].to(device)
        x_hat, h_front, h = model(data_full, x_front=data_front)

        reconstruction_loss = loss(x_hat, data_full.y)
        latent_loss = loss(h_front, h)
        tot_rec_loss += reconstruction_loss.item()
        total_loss += latent_loss.item()


        if example_fig is None:
            points = data_full.pos[data_full.batch==0].clone().cpu()
            reconstruction = x_hat[data_full.batch==0].clone().detach().cpu()
            example_fig = plot_state_rec_singleview(points=points, full_points=reconstruction, view='top')

    test_rec_loss = tot_rec_loss / i
    test_loss = total_loss / i
    return (test_loss, test_rec_loss, example_fig)


def front_encoder_trainer(args, device, model_path=None):
    model = PointAE(args, device, front_encoder=True).to(device)

    modules = ['_encoder', '_decoder']
    model.load_state_dict(torch.load(model_path))
    model.freeze_modules(modules)
    optimizer = torch.optim.Adam(model._encoder_front.parameters(), lr=args.lr)
    scheduler = None

    print(f'The encoder model:\n{model}')
    print(f'Parameters: {count_parameters(model)}.\n')
    start = time.time()

    best_test_loss = 10000

    # Train and test!
    for epoch in range(1, args.epochs + 1):

        train_latent_loss, train_rec_loss, train_fig = train_front_encoder(model, train_loader, device, optimizer)
        test_latent_loss, test_rec_loss, test_fig = test_front_encoder(model, test_loader, device)
        if scheduler is not None:
            scheduler.step()

        print(f'Epoch: {epoch:02d}, Tr Rec Loss: {train_rec_loss:.4f}, Test Rec Loss: {test_rec_loss:.4f}')
        wandb_dict = {
            'epoch': epoch,
            'elapsed_t': time.time() - start,
            'train/loss': train_rec_loss,
            'train/latent_loss': train_latent_loss,
            'train/img': train_fig,
            'test/img': test_fig,
            'test/loss': test_rec_loss,
            'test/latent_loss': test_latent_loss,
        }

        if wandb_dict['test/loss'] < best_test_loss:
            # Save
            best_test_loss = wandb_dict['test/loss']
            torch.save(model.state_dict(), checkpoint_folder + 'full_dict_model_val.pt')

        wandb.log(wandb_dict)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='modelnet10')
    p.add_argument('--model', type=str, default='pointnet2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--validation', type=bool, default=True)
    p.add_argument('--dataset_path', default='../data/datasets', type=str, help="Path to the dataset")
    p.add_argument('--dataset_name', default='/oct31_1s_train', type=str, help="Name of the dataset")
    p.add_argument('--data_aug', default=1, type=int, help="whether to perform data augmentation or not")
    p.add_argument('--z_dim', default=32, type=int, help="dimension of the latent space")
    p.add_argument('--dropout', default=0, type=int, help="Use (1) or not (0) dropout normalization.")

    p.add_argument('--batch_norm', default=0, type=int, help="Use (1) or not (0) batch normalization.")
    p.add_argument('--dyn_conditioning', default=3, type=int, help="Type of conditioning of dynamic model:"
                                                                        "0: baseline, no conditioning"
                                                                        "1: condition on GT PI"
                                                                        "2: condition on encoded PI"
                                                                        "3: condition on z - EDO-net")

    p.add_argument('--fusion', default=1, type=int, help="Type of conditioning of dynamic model:"
                                                                "0: concatenation + MPL"
                                                                "1: RNN"
                                                                "2: GRU"
                                                                "3: Attention")
    p.add_argument('--pcd_dim', default=625, type=int, help="Dimension of the state observations")
    p.add_argument('--pcd_scale', default=2, type=int, help="Downscaling ratio of the point cloud")
    p.add_argument('--action_dim', default=3, type=int, help="Dimension of the actions")
    p.add_argument('--pi_dim', default=5, type=int, help="Number of privileged information")
    p.add_argument('--zero_center', default=0, type=int, help="zero center the observations if 1, otherwise 0")
    p.add_argument('--obs', default='mesh', type=str,
                        help="Type of the state observations [mesh, pcd, full_pcd]")
    p.add_argument('--inv_dyn', default=0, type=int, help="Use inverse dynamics instead of forward.")
    p.add_argument('--HFE_SA_r', default=[0.025, 0.05, 0.1], help="radius of SA layer")
    p.add_argument('--HFE_SA_ratio', default=[0.5, 0.25, 0.25], help="Sampling ration SA layer")
    p.add_argument('--seg_FP_k', default=[1, 3, 3], help="kNN for upsapmpling in forward propagation layer")
    p.add_argument('--train_cluster', default=1, type=int, help="Train on cluster of personal desktop")

    p.add_argument('--skip_connections', type=bool, default=False)


    args = p.parse_args()

    # Bells and whistles.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and torch.cuda.get_device_name(
            torch.cuda.current_device()) != 'NVIDIA GeForce GTX 1080' and not args.train_cluster:
        torch.cuda.set_device(1)
    torch.cuda.empty_cache()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_folders_train = 100
    num_folders_val = 0
    if args.validation:
        num_folders_val = num_folders_train

    # run_name = f'mask={1 * args.add_mask}_action={1 * args.add_action}'
    run_name = 'full_encoder'
    # run_name = 'front_encoder'
    if args.skip_connections == 0:
        run_name += '_noSkip'
    checkpoint_folder = f'../data/rec_checkpoint/{run_name}/'
    os.makedirs(checkpoint_folder, exist_ok=True)
    dataset_train, dataset_test, dataset_val = load_datasets(args=args,
                                                             dataload=PointcloudAEDataset,
                                                             checkpoint_folder=checkpoint_folder,
                                                             num_folders_train=num_folders_train,
                                                             num_folders_val=num_folders_val,
                                                             num_envs=-1,
                                                             num_trajectories=-1, )

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    if args.validation:
        test_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0)


    os.environ["WANDB_MODE"] = "run"    #  "run", "dryrun"
    wandb_dir = './wandb_dir_rec'
    if not os.path.exists(os.path.join(wandb_dir, 'wandb')):
        os.makedirs(os.path.join(wandb_dir, 'wandb'), exist_ok=True)

    wandb.init(project="point-cloud-rec", entity="albilo", name=run_name)
    wandb.config.update(args)

    if 'full' in run_name:
        full_encoder_trainer(args, device)
    if 'front' in run_name:
        model_path = checkpoint_folder.replace(run_name, 'full_encoder') + '/full_dict_model_val.pt'
        front_encoder_trainer(args, device, model_path=model_path)


    print(f'Done!')