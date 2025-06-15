import argparse


def get_argparse(notebook=False, p=False):
    parser = argparse.ArgumentParser()


    parser.add_argument('--seed', default=1234, type=int, help="Random seed")
    parser.add_argument('--finetune', default=0, type=int, help="If 1, use recostruction loss for the reward module.")
    parser.add_argument('--validation', default=1, type=int, help="If 1, use validation dataset.")

    parser.add_argument('--data_aug', default=1, type=int, help="whether to perform data augmentation or not")

    parser.add_argument('--epochs', default=400, type=int, help="Training epochs")
    parser.add_argument('--batch_size', default=32, type=int, help="Training batch size")

    parser.add_argument('--train_folders', default=-1, type=int, help="Number of different cloths")
    # parser.add_argument('--train_trajs', default=100, type=int, help="Number of trajectories per cloth")


    parser.add_argument('--lr', default=1e-4, type=float, help="Training learning rate")
    parser.add_argument('--lr_schedule', default=0, type=int, help="Use learning rate scheduler if 1")
    parser.add_argument('--scheduler', default='cosine', type=str, help="type of lr scheduler, ['cosine', 'lambda']")
    parser.add_argument('--RMA_schedule', default=1, type=int, help="RMA scheduler for training encoder")
    parser.add_argument('--early_stop_pi', default=0, type=int, help="Stop training the pi encoder")


    parser.add_argument('--loss', default='MSE', type=str, help="Choices: [MAE, MSE, chamfer]")
    parser.add_argument('--alpha', default=1., type=float, help="multiply coeff. for prediction loss")
    parser.add_argument('--beta', default=0.1, type=float, help="multiply coeff. for adaptation loss")

    # NEEDED?
    parser.add_argument('--chamfer_loss', default=0, type=int, help="use chamfer loss if 1")
    parser.add_argument('--single_env', default=0, type=int, help="Use only one environment for training")
    parser.add_argument('--pretrained_enc', default=0, type=int, help="If 1, load pretrained encoders.")
    parser.add_argument('--corner_feat', default=0, type=int, help="Whether to use corner features or not")
    parser.add_argument('--reconstruction', default=0, type=int, help="Whether to use reconstuction loss or not")
    parser.add_argument('--multi_step', default=True, type=bool, help="Whether to train on multiple action steps or only"
                                                                       "on the mid-waypoints.")











    if notebook:
        return parser.parse_args(args=[])

    return parser.parse_args()