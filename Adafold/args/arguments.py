import argparse
from Adafold.args.controller_args import get_argparse as c_args
from Adafold.args.data_args import get_argparse as d_args
from Adafold.args.model_args import get_argparse as m_args
from Adafold.args.training_args import get_argparse as t_args

def get_argparse(notebook=False, p=False):
    parser = argparse.ArgumentParser()

    ########## CLUSTER VARIABLES ######################
    parser.add_argument('--train_trajs', default=100, type=int, help="Number of trajectories per cloth")
    parser.add_argument('--dyn_conditioning', default=3, type=int, help="Type of conditioning of dynamic model:"
                                                                        "0: baseline, no conditioning"
                                                                        "1: condition on GT PI"
                                                                        "2: condition on encoded PI"
                                                                        "3: condition on z - EDO-net")

    parser.add_argument('--fusion', default=1, type=int, help="Type of conditioning of dynamic model:"
                                                              "0: concatenation + MPL"
                                                              "1: RNN"
                                                              "2: GRU"
                                                              "3: Attention"
                                                              "4: LSTM")
    parser.add_argument('--train_cluster', default=0, type=int, help="Train on cluster of personal desktop")

    ###################################### FOLDERS #########################

    parser.add_argument('--project', default='ADA', type=str, help="Name of the wand project")    # Folders
    parser.add_argument('--dataset_path', default='./data/datasets', type=str, help="Path to the dataset")
    parser.add_argument('--dataset_name', default='/folding_dataset', type=str, help="Name of the dataset")

    parser.add_argument('--checkpoint', default='./data/checkpoint', type=str, help="Name of the checkpoint folder")
    parser.add_argument('--cluster', default='./logs_cluster', type=str, help="Name of the checkpoint folder")

    ###################################### TRAINING #########################

    parser.add_argument('--seed', default=1234, type=int, help="Random seed")
    parser.add_argument('--finetune', default=0, type=int, help="If 1, use recostruction loss for the reward module.")
    parser.add_argument('--validation', default=1, type=int, help="If 1, use validation dataset.")

    parser.add_argument('--data_aug', default=1, type=int, help="whether to perform data augmentation or not")

    parser.add_argument('--epochs', default=400, type=int, help="Training epochs")
    parser.add_argument('--batch_size', default=32, type=int, help="Training batch size")

    parser.add_argument('--train_folders', default=-1, type=int, help="Number of different cloths")
    # parser.add_argument('--train_trajs', default=100, type=int, help="Number of trajectories per cloth")

    parser.add_argument('--lr', default=1e-4, type=float, help="Training learning rate")
    parser.add_argument('--l2_reg', default=0, type=int, help="use or not l2 regularization")
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
    parser.add_argument('--multi_step', default=True, type=bool,
                        help="Whether to train on multiple action steps or only"
                             "on the mid-waypoints.")

    ###################################### DATA #########################
    parser.add_argument('--rw', default=0, type=int, help="whether dataset comes from the RW or not")

    parser.add_argument('--obs', default='mesh', type=str,
                        help="Type of the state observations [mesh, pcd, full_pcd]")


    parser.add_argument('--pcd_dim', default=625, type=int, help="Dimension of the state observations")
    parser.add_argument('--pcd_scale', default=2, type=int, help="Downscaling ratio of the point cloud")
    parser.add_argument('--action_dim', default=3, type=int, help="Dimension of the actions")
    parser.add_argument('--pi_dim', default=5, type=int, help="Number of privileged information")
    parser.add_argument('--zero_center', default=0, type=int, help="zero center the observations if 1, otherwise 0")

    ###################################### MODEL #########################

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

    # TOLD
    parser.add_argument('--latent_dim', default=64, type=int, help="dimension of the latent space")
    parser.add_argument('--mlp_dim', default=512, type=int, help="dimension of mpl")
    parser.add_argument('--rho', default=1., type=float, help="scaling factor for the reward")
    parser.add_argument('--consistency_coef', default=1., type=float, help="scaling coefficient for the loss of the dyn prediction")
    parser.add_argument('--reward_coef', default=1., type=float, help="scaling coefficient for the loss of the reward prediction")
    parser.add_argument('--grad_clip_norm', default=10., type=float, help="Value to clip the gradient.")
    parser.add_argument('--update_freq', default=2, type=int, help="frequency of update of the ")
    parser.add_argument('--tau', default=1., type=float, help="coefficient for moving average of the model")


    ##################################### PERCEPTION MASK ######################
    parser.add_argument('--add_action', type=bool, default=True, help="condition the mask extractor to the previous action")
    parser.add_argument('--add_mask', type=bool, default=True, help="condition the mask extractor to the approximated mask")

    ###################################### CONTROLLERS #########################
    parser.add_argument('--render', type=int, default=0, help="If 1, will open a gui for rendering and dump gifs.")
    # parser.add_argument('--data_save_path', type=str, default='./data/store', help="path to store the data.")
    parser.add_argument('--save_gif', type=int, default=1, help="save gif of the simulation")
    parser.add_argument('--save_data', type=int, default=1, help="save dataset simulations")
    parser.add_argument('--save_data_path', type=str, default='./data/datasets', help="path where to save dataset ")
    parser.add_argument('--gif_path', type=str, default='./data/gif',
                        help="save directory of the gif of the simulation")

    parser.add_argument('--num_variations', type=int, default=1000,
                        help="number of different simulations that could be run")

    parser.add_argument('--len_traj', type=int, default=30,
                        help="number of steps per trajectory")

    # Table params
    parser.add_argument('--urdf_table_path', type=str, default="table/table.urdf",
                        help="path to a urdf file that represent the object the cloth is pulling downwards to. NOTE: the actual file path is assistive_gym/envs/assets/dinnerware/sphere.urdf. If you wanna use a new .urdf file, put it in the directory assistive_gym/envs/assets/")
    parser.add_argument('--urdf_table_collision_path', type=str, default="table/table.obj",
                        help="path to a urdf file that represent the object the cloth is pulling downwards to. NOTE: the actual file path is assistive_gym/envs/assets/dinnerware/sphere.urdf. If you wanna use a new .urdf file, put it in the directory assistive_gym/envs/assets/")

    parser.add_argument('--urdf_file_path', type=str, default="dinnerware/sphere.urdf",
                        help="path to a urdf file that represent the object the cloth is pulling downwards to. NOTE: the actual file path is assistive_gym/envs/assets/dinnerware/sphere.urdf. If you wanna use a new .urdf file, put it in the directory assistive_gym/envs/assets/")
    # parser.add_argument('--urdf_file_path', type=str, default="None", help="path to a urdf file that represent the object the cloth is pulling downwards to. NOTE: the actual file path is assistive_gym/envs/assets/dinnerware/sphere.urdf. If you wanna use a new .urdf file, put it in the directory assistive_gym/envs/assets/")
    parser.add_argument('--obj_visual_file_path', type=str, default=None,
                        help="Alternatively, you can provide a .obj file that describes the object. E.g., dinnerware/plastic_coffee_cup.obj. NOTE: the actual file path is assistive_gym/envs/assets/dinnerware/plastic_coffee_cup.obj. If you wanna use a new .urdf file, put it in the directory assistive_gym/envs/assets/")
    parser.add_argument('--obj_collision_file_path', type=str, default=None,
                        help="For .obj file description of the object, need to provide both a visual file and a collision file. E.g., dinnerware/plastic_coffee_cup_vhacd.obj. NOTE: the actual file path is assistive_gym/envs/assets/dinnerware/plastic_coffee_cup.obj")

    parser.add_argument('--obj_scale', type=list, default=[0.2, 0.2, 0.2],
                        help="the scaling of the .obj file along 3 axis")
    parser.add_argument('--urdf_scale', type=float, default=2, help="the scaling of the .urdf file")

    parser.add_argument('--cloth_obj_file_path', type=str, default='clothing/cloth_ordered.obj',
                        help="path to a .obj file that describes the cloth mesh.")


    if notebook:
            return parser.parse_args(args=[])

    return parser.parse_args()


def load_all_args(notebook=False):
    general_args = get_argparse(notebook)
    training_args = t_args(notebook)
    model_args = m_args(notebook)
    data_args = d_args(notebook)
    controller_args = c_args(notebook)

    # Combine arguments
    combined_args = {**vars(general_args), **vars(training_args), **vars(model_args), **vars(data_args), **vars(controller_args)}
    combined_args_namespace = argparse.Namespace(**combined_args)

    return combined_args_namespace


if __name__=='__main__':
    print()
