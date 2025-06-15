import argparse


def get_argparse(notebook=False, p=False):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dyn_conditioning', default=0, type=int, help="Type of conditioning of dynamic model:"
    #                                                                     "0: baseline, no conditioning"
    #                                                                     "1: condition on GT PI"
    #                                                                     "2: condition on encoded PI"
    #                                                                     "3: condition on z - EDO-net")

    # parser.add_argument('--fusion', default=1, type=int, help="Type of conditioning of dynamic model:"
    #                                                             "0: concatenation + MPL"
    #                                                             "1: RNN"
    #                                                             "2: GRU"
    #                                                             "3: Attention")
    # Model params
    parser.add_argument('--HFE_SA_r', default=[0.025, 0.05, 0.1], help="radius of SA layer")
    parser.add_argument('--HFE_SA_ratio', default=[0.5, 0.25, 0.25], help="Sampling ration SA layer")
    parser.add_argument('--seg_FP_k', default=[1, 3, 3], help="kNN for upsapmpling in forward propagation layer")

    parser.add_argument('--K', default=3, type=int, help="Number of past observation to encode")
    parser.add_argument('--H', default=1, type=int, help="Prediction Horizon")
    parser.add_argument('--z_dim', default=32, type=int, help="dimension of the latent space")

    parser.add_argument('--batch_norm', default=0, type=int, help="Use (1) or not (0) batch normalization.")
    parser.add_argument('--dropout', default=0, type=int, help="Use (1) or not (0) dropout normalization.")
    parser.add_argument('--flow', default=1, type=int, help="Flag to set the prediction with or without flow.")

    # Needed?
    parser.add_argument('--inv_dyn', default=0, type=int, help="Use inverse dynamics instead of forward.")
    parser.add_argument('--fusion_cost', default=1, type=int, help="Type of conditioning of dynamic model:"
                                                                "0: concatenation + MPL"
                                                                "1: RNN"
                                                                "2: GRU"
                                                                "3: Attention")














    if notebook:
        return parser.parse_args(args=[])

    return parser.parse_args()