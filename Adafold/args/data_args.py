import argparse


def get_argparse(notebook=False, p=False):
    parser = argparse.ArgumentParser()


    parser.add_argument('--rw', default=0, type=int, help="whether dataset comes from the RW or not")

    parser.add_argument('--obs', default='mesh', type=str,
                        help="Type of the state observations [mesh, pcd, full_pcd]")


    parser.add_argument('--pcd_dim', default=625, type=int, help="Dimension of the state observations")
    parser.add_argument('--pcd_scale', default=2, type=int, help="Downscaling ratio of the point cloud")
    parser.add_argument('--action_dim', default=3, type=int, help="Dimension of the actions")
    parser.add_argument('--pi_dim', default=5, type=int, help="Number of privileged information")
    parser.add_argument('--zero_center', default=0, type=int, help="zero center the observations if 1, otherwise 0")











    if notebook:
        return parser.parse_args(args=[])

    return parser.parse_args()