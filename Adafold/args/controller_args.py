import argparse


def get_argparse(notebook=False, p=False):
    parser = argparse.ArgumentParser()


    #################### MPC parameters
    parser.add_argument('--P', default=1, type=int, help="Prediction steps")
    parser.add_argument('--A', default=100, type=int, help="Num. sampled actions")
    parser.add_argument('--use_adapt', type=int, default=1, help="If 1, will use the adaptation module for the prediction")

    # NEEDED?
    parser.add_argument('--closed_loop', default=0, type=int, help="update latent representation in a closed loop if 1")
    parser.add_argument('--reward', default='corner', type=str, help="Type of reward [corner, shape]")

    ############ CONTROLLERS ############

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