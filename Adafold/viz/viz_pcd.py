import torch
import matplotlib.pyplot as plt

def plot_pcd(pcd,
             elev=30,
             azim=0,
             x_lim=[-0.25, 0.25],
             y_lim=[-0.25, 0.25],
             z_lim=[0, 0.3],
             alpha_value=1,
             remove_zeros=False,
             title=None,
             return_fig=False):



    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(projection='3d', elev=elev, azim=azim)
    # rotate by 90, invert
    if remove_zeros:
        if torch.is_tensor(pcd[0]):
            p0 = torch.stack([pt for pt in pcd if not (pt == torch.zeros(3)).all()])
        else:
            p0 = torch.Tensor([pt for pt in pcd if not (pt == [0,0,0]).all()])

        img = ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], alpha=alpha_value)
    else:
        img = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], alpha=alpha_value)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

    if title is not None:
        plt.title(title)

    if return_fig:
        return fig

    plt.show()


def plot_pcd_list(pcd,
                  elev=30,
                  azim=0,
                  x_lim=[-0.25, 0.25],
                  y_lim=[-0.25, 0.25],
                  z_lim=[0, 0.3],
                  alpha_value=1,
                  title=None,
                  remove_zeros=False,
                  return_fig=False):

    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(projection='3d', elev=elev, azim=azim)

    for p in pcd:
        img = ax.scatter(p[:, 0], p[:, 1], p[:, 2], alpha=alpha_value)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    if title is not None:
        plt.title(title)

    if return_fig:
        return fig
    plt.show()