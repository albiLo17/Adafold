import argparse

def get_argparse_dataset(notebook=False, p=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--action_norm', default=0.03, type=float, help="Norm of each action")
    parser.add_argument('--action_dim', default=3, type=int, help="Action dimension, 3D displacement")
    parser.add_argument('--action_mult', default=1, type=int, help="Multiplier each action")
    parser.add_argument('--num_actions', default=13, type=int, help="Number of actions per trajectory")

    parser.add_argument('--c_threshold', default=0.3, type=float, help="Cosine similarity threshold")



    if notebook:
        return parser.parse_args(args=[])

    return parser.parse_args()
