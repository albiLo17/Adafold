import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_cluster import knn_graph
from torch_geometric.nn import fps, radius, global_max_pool, knn_interpolate

# Enable two different versions of Pytorch Geometric
try:
    from torch_geometric.nn import PointConv
except:
    from torch_geometric.nn.conv import PointConv

class MLPEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLPEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        x = self.encoder(x)

        return x

class MLPDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLPDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        x = self.encoder(x)

        return x


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index)     # one or both pcds?

        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    # layer responsible for the conversion of the abstracted point set to a single vector.
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))      # position not needed anymore
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # input: features, pos, upsampled pos, batch, upsampled batch, num neigh.
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        # if torch.any(torch.isnan(x)):
        #     print()
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        # if torch.any(torch.isnan(x)):
        #     print()
        return x, pos_skip, batch_skip

def MLP(channels, batch_norm=True):
    if batch_norm:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ])

    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

