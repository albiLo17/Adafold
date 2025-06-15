import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__REDUCE__ = lambda b: 'mean' if b else 'none'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]), act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        nn.Linear(mlp_dim[1], out_dim))


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)

def chamfer(predictions, labels, bidirectional=False, squared=False):
    # a and b don't need to have the same number of points\
    """"
    Chamfer loss:
        predictions \in batch x num pcd x num_nodes x dim
        labels \in batch x num pcd x num_nodes x dim
        bidirectional \modality of chamfer loss
    """
    if not squared:
        c_dist = torch.cdist(predictions, labels, p=2)
    else:
        c_dist = torch.cdist(predictions, labels, p=2) ** 2
    dist = c_dist.min(dim=-2)[0].mean(-1)       # Keep only distance from g_t to predicted, the other way around leads to collapse
    if bidirectional:
        dist += c_dist.min(dim=-1)[0].mean(-1)

    # Batch average
    chamfer_dist = dist.mean()
    return chamfer_dist

# ---------------------------------------------------------------------------------- #
# From tool flow net
# ---------------------------------------------------------------------------------- #


def random_crop_pc(obs, action, max_x, min_x, max_y, min_y, max_z, min_z):
    """From Yufei, might be useful for data augmentation."""
    gripper_pos = obs.pos[obs.x[:, 2] == 1]

    gripper_x_min, gripper_x_max = torch.min(gripper_pos[:, 0]).item(), torch.max(gripper_pos[:, 0]).item()
    gripper_y_min, gripper_y_max = torch.min(gripper_pos[:, 1]).item(), torch.max(gripper_pos[:, 1]).item()
    gripper_z_min, gripper_z_max = torch.min(gripper_pos[:, 2]).item(), torch.max(gripper_pos[:, 2]).item()

    x_start = np.random.rand() * (gripper_x_min - min_x) + min_x
    y_start = np.random.rand() * (gripper_y_min - min_y) + min_y
    z_start = np.random.rand() * (gripper_z_min - min_z) + min_z

    x_end = x_start + (max_x - min_x) * 0.75
    y_end = y_start + (max_y - min_y) * 0.75
    z_end = z_start + (max_z - min_z) * 0.75

    x_end = max(x_end, gripper_x_max)
    y_end = max(y_end, gripper_y_max)
    z_end = max(z_end, gripper_z_max)

    mask = (obs.pos[:, 0] <= x_end) & (obs.pos[:, 0] >= x_start) & \
            (obs.pos[:, 1] <= y_end) & (obs.pos[:, 1] >= y_start) & \
            (obs.pos[:, 2] <= z_end) & (obs.pos[:, 2] >= z_start)

    obs.pos = obs.pos[mask]
    obs.x = obs.x[mask]
    obs.batch = obs.batch[mask]

    return obs, action[mask]


def rotate_pc(obs, angles=None, device=None, return_rot=False):
    """From Yufei, might be useful for data augmentation.
    Note: `obs` should be just positions, e.g., from a PCL.
    Tested on the real datasets, need to test more in sim.
    """
    if angles is None:
        angles = np.random.uniform(-np.pi, np.pi, size=3)
    Rx = np.array([[1,0,0],
                [0,np.cos(angles[0]),-np.sin(angles[0])],
                [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                [0,1,0],
                [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                [np.sin(angles[2]),np.cos(angles[2]),0],
                [0,0,1]])

    R = np.dot(Rz, np.dot(Ry,Rx))
    if device is not None:
        R = torch.from_numpy(R).to(device).float()
    obs2 = obs @ R   # (N,3) x (3x3)
    if return_rot:
        return (obs, R)
    else:
        return obs2 - obs



