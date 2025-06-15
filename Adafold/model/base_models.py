import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
from torch.nn import Linear as Lin
from .layers import MLPEncoder, MLPDecoder, SAModule, MLP, GlobalSAModule, FPModule
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import SamplePoints
import os.path as osp
import numpy as np
from torch_geometric.datasets import ModelNet


class AutoEncoder(nn.Module):

    def __init__(self, args, hidden_size=8):
        super(AutoEncoder, self).__init__()

        self.input_size = args.pi_dim
        self.latent_size = args.z_dim
        self.hidden_size = hidden_size
        self.output_size = args.pi_dim

        self.encoder =MLPEncoder(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=self.latent_size)
        self.decoder = MLPDecoder(input_size=self.latent_size,
                                  hidden_size=self.hidden_size,
                                  output_size=self.output_size)

    def forward(self, x, z=None):
        if z is None:
            z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return z, x_reconstructed



####################################

class PointNetHFE(torch.nn.Module):
    def __init__(self, args, node_feat_dim=6, state_feat_dim=3, globalSA=False, only_action=False):
        super(PointNetHFE, self).__init__()

        self.latent_size = args.z_dim
        self.hidden_size = 64
        # self.output_size = args.z_dim
        self.node_feat_dim = node_feat_dim

        self.batch_norm = True*args.batch_norm

        self.conditioning = args.dyn_conditioning
        self.inv_dyn = args.inv_dyn

        # # in this case encode the ground truth PI
        # if self.conditioning == 1:
        #     self.output_size = args.pi_dim

        self.SA_r = args.HFE_SA_r
        self.SA_ratio = args.HFE_SA_ratio

        # Input channels account for both `pos` and node features, where nodes features are position and action.
        self.sa1_module = SAModule(ratio=self.SA_ratio[0], r=self.SA_r[0], nn=MLP([self.node_feat_dim + 3, 64, 64, 128], batch_norm=self.batch_norm))
        self.sa2_module = SAModule(ratio=self.SA_ratio[1], r=self.SA_r[1], nn=MLP([128 + 3, 128, 128, 256], batch_norm=self.batch_norm))
        if globalSA:
            self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024], batch_norm=self.batch_norm))
        else:
            self.sa3_module = SAModule(ratio=self.SA_ratio[2], r=self.SA_r[2], nn=MLP([256 + 3, 256, 256, 512], batch_norm=self.batch_norm))

    def forward(self, batch, intermediate=False):

        sa0_out = (batch.x, batch.pos, batch.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        if intermediate:
            return sa0_out, sa1_out, sa2_out, sa3_out

        return sa3_out


class PointNetDecoder(torch.nn.Module):
    def __init__(self, args, num_classes=2, classification=False, skip=True, init_input_dim=10):
        super(PointNetDecoder, self).__init__()
        # init_input_dim = self.node_feat_dim + self.encoding_dim[self.conditioning]
        self.FP_k = args.seg_FP_k
        self.classification = classification
        self.num_classes = num_classes

        self.batch_norm = args.batch_norm
        self.skip = skip

        self.fp3_module = FPModule(k=self.FP_k[0], nn=MLP([1024 + 256, 256, 256], batch_norm=self.batch_norm))
        self.fp2_module = FPModule(k=self.FP_k[1], nn=MLP([256 + 128, 256, 128], batch_norm=self.batch_norm))
        self.fp1_module = FPModule(k=self.FP_k[2],
                                   nn=MLP([128 + init_input_dim, 128, 128, 128],
                                          batch_norm=self.batch_norm))

        self.lin1 = torch.nn.Linear(128, 128)
        # self.lin2 = torch.nn.Linear(128, 128)
        if not classification:
            # decode 3d positions
            self.lin2 = torch.nn.Linear(128, 3)
        else:
            self.lin2 = torch.nn.Linear(128, self.num_classes)

    def forward(self, sa0_out, sa1_out, sa2_out, sa3_out):
        # Reconstruct from a latent code
        if not self.skip:
            # TODO: remember that you removed skip connections by myltipling by 0
            fp3_out = self.fp3_module(*sa3_out, sa2_out[0]*0., sa2_out[1], sa2_out[2])
            fp2_out = self.fp2_module(*fp3_out, sa1_out[0]*0., sa1_out[1], sa1_out[2])
            x, _, _ = self.fp1_module(*fp2_out, sa0_out[0]*0., sa0_out[1], sa0_out[2])

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        if self.classification:
            # TODO: debug the dimension of the log
            return x.log_softmax(dim=-1)

        return x




class PointNet2Cls(torch.nn.Module):
    def __init__(self, args, only_action=False, node_feat_dim=9, K=None):
        super(PointNet2Cls, self).__init__()

        self.latent_size = args.z_dim
        self.hidden_size = 64
        self.output_size = args.z_dim
        if K is None:
            self.K = args.K
        else:
            self.K = K

        self.batch_norm = True*args.batch_norm
        self.dropout = True*args.dropout

        self.conditioning = args.dyn_conditioning

        # in this case encode the ground truth PI
        if self.conditioning == 1:
            self.output_size = args.pi_dim

        self.feat_extract = PointNetHFE(args=args,
                                        node_feat_dim=node_feat_dim,
                                        globalSA=True)    # pt, at, xt

        # Classification Branch
        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 256)

        self.fusion_type = args.fusion
        if self.fusion_type == 0:
            self.fusion = MLP([256*(self.K+1), 512, 512, 256],
                                          batch_norm=self.batch_norm)

        if self.fusion_type == 1:
            node_feat_dim = 256  # Input size
            hidden_size = 256  # Hidden layer size
            self.recurrent = nn.RNN(node_feat_dim, hidden_size, num_layers=1, batch_first=True)
            # self.linear_fusion = nn.Linear(hidden_size, 256)

        if self.fusion_type == 2:
            node_feat_dim = 256  # Input size
            hidden_size = 256  # Hidden layer size
            self.recurrent = nn.GRU(node_feat_dim, hidden_size, num_layers=1, batch_first=True)
            # self.linear_fusion = nn.Linear(hidden_size, 256)

        # if fusion_type == 4 use and LSTM
        if self.fusion_type == 4:
            node_feat_dim = 256  # Input size
            hidden_size = 256  # Hidden layer size
            self.recurrent = nn.LSTM(node_feat_dim, hidden_size, num_layers=1, batch_first=True)


        # final projection
        self.lin4 = Lin(256, 128)
        self.lin5 = Lin(128, self.output_size)




    def forward(self, input):

        # Classification branch
        sa3_out = self.feat_extract(input)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)

        # Fusion

        # bring back the K pointcloud per batch
        n_batch = int((input.batch.max() + 1) / (self.K+1))
        if self.fusion_type == 0:
            # concatenate for all the latent vectors
            x = x.view(n_batch, -1)
            x = self.fusion(x)
        else:
            x = x.view(n_batch, self.K+1, -1)
            x, _ = self.recurrent(x)
            # select the last hidden state for the linear layer
            x = x[:, -1, :]

        x = F.relu(self.lin4(x))
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin5(x)

        return x


class PointNet2Seg(torch.nn.Module):
    def __init__(self, args):
        super(PointNet2Seg, self).__init__()

        self.hidden_size = 64
        self.action_dim = args.action_dim
        self.node_feat_dim = 3 + 3 + 3  #*args.H       # pcd, gripper, actions*(num actions)

        self.batch_norm = args.batch_norm
        self.dropout = args.dropout

        self.flow = args.flow
        self.output_size = args.z_dim
        self.conditioning = args.dyn_conditioning
        self.inv_dyn = args.inv_dyn
        self.latent_encoding_size = args.z_dim
        self.pi_dim = args.pi_dim

        # Different conditioning modalities: NoC, PI, e(PI), e(obs)
        self.encoding_dim = [0, self.pi_dim, self.latent_encoding_size, self.latent_encoding_size]

        self.FP_k = args.seg_FP_k

        self.feat_extract = PointNetHFE(args=args, node_feat_dim=self.node_feat_dim + self.encoding_dim[self.conditioning], globalSA=True) # pt, at, xt

        self.fp3_module = FPModule(k=self.FP_k[0], nn=MLP([1024 + 256, 256, 256], batch_norm=self.batch_norm))
        self.fp2_module = FPModule(k=self.FP_k[1], nn=MLP([256 + 128, 256, 128], batch_norm=self.batch_norm))
        self.fp1_module = FPModule(k=self.FP_k[2], nn=MLP([128 + self.node_feat_dim + self.encoding_dim[self.conditioning], 128, 128, 128], batch_norm=self.batch_norm))


        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 3)

    def forward(self, input, encoding=None):
        
        # I want to concatenate the encoding to the input but without modifying the input, knowing that input might not be a graph leave
        if encoding is not None and self.conditioning > 0:
            input.x = torch.cat([input.x, torch.zeros(input.x.shape[0], encoding.shape[-1]).to(input.x.device)], dim=1)
            for i in range(encoding.shape[0]):
                batch_val = input.batch == i
                input.x[batch_val, -encoding.shape[-1]:] = encoding[i].unsqueeze(dim=0).repeat(torch.sum(batch_val), 1)

        # Feature extraction
        sa0_out, sa1_out, sa2_out, sa3_out = self.feat_extract(input, intermediate=True)

        # debugging plots
        # import copy
        # from Adafold.viz.viz_pcd import plot_pcd
        # plot_pcd(copy.deepcopy(input.pos[input.batch == 1]).cpu(), azim=0, z_lim=[0, 0.3], x_lim=[-0.25, 0.25], y_lim=[-0.25, 0.25])
        # plot_pcd(copy.deepcopy(sa0_out[1])[sa0_out[2] == 1].cpu(), azim=0, z_lim=[0, 0.3], x_lim=[-0.25, 0.25], y_lim=[-0.25, 0.25])
        # plot_pcd(copy.deepcopy(sa1_out[1])[sa1_out[2] == 1].cpu(), azim=0, z_lim=[0, 0.3], x_lim=[-0.25, 0.25], y_lim=[-0.25, 0.25])
        # plot_pcd(copy.deepcopy(sa2_out[1])[sa2_out[2] == 1].cpu(), azim=0, z_lim=[0, 0.3], x_lim=[-0.25, 0.25], y_lim=[-0.25, 0.25])
        # plot_pcd(copy.deepcopy(sa3_out[1])[sa3_out[2] == 1].cpu(), azim=0, z_lim=[0, 0.3], x_lim=[-0.25, 0.25], y_lim=[-0.25, 0.25])


        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        # if torch.any(torch.isnan(fp2_out[0])):
        #     print()
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        if self.dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)

        # learn residual of it
        if self.flow and self.inv_dyn == 0:
            x += input.pos


        return x

    def half_batch(self, sa_out):
        x, pos, batch = sa_out
        pos = pos[batch % 2 == 0]
        x = x[batch % 2 == 0]
        batch = (batch[batch % 2 == 0]/2).long()

        return [x, pos, batch]




def train(model, loader):
    model.train()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# Visualization functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_mesh(pos, face=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    if face is not None:
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=face.t(), antialiased=False)
    else:
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], antialiased=False)
    plt.show()


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()

