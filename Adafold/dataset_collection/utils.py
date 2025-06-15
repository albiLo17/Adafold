from Adafold.dataset_collection.camera_utils import Cameras
import h5py
import numpy as np
from Adafold.trajectory.trajectory import Trajectory, Action_Sampler
import os.path as osp
import imageio
from Adafold.viz.viz_pcd import plot_pcd, plot_pcd_list
def reset_env(env, elas, bend, scale):
    damp, frict = 1.5, 1.50
    env.reset(stiffness=[elas, bend, damp], friction=frict, cloth_scale=scale, cloth_mass=0.5)  # Elas, bend, damp

    env.camera = Cameras(id=env.id)
    env.camera.setup_camera(camera_eye=[0, -0.60, 0.23], camera_target=[0., 0., 0.], camera_width=720,
                            camera_height=720)
    env.camera.setup_camera(camera_eye=[0.4, 0.2, 0.23], camera_target=[0., 0., 0.], camera_width=720,
                            camera_height=720)

    env.setup_camera(camera_eye=[0., 0., 0.65], camera_target=[0., 0.0, 0.], camera_width=720,
                     camera_height=720)



def get_states(args, traj_type, place_pos, gripper_pos, num_actions, action_norm, action_mult):
    # August25 dataset
    if traj_type == 'fixed':
        mid_w = (place_pos + gripper_pos) / 2
        mid_w[2] = 0.08
        waypoints = np.asarray([gripper_pos, mid_w, place_pos])

        controller = Trajectory(args=args,
                                waypoints=waypoints,
                                vel=action_norm,  # as it will be rescaled when processed
                                interpole=True,
                                action_scale=action_mult,
                                constraint=False,
                                rw=False)
        states = controller.traj_points
    else:
        sampler = Action_Sampler(
            N=num_actions,  # trajectory length
            action_len=action_norm,
            c_threshold=0.3,
            pp_dir=place_pos - gripper_pos,
            starting_point=gripper_pos,
            sampling_mean=None,
            rw=False)

        states = sampler.sample_trajectory()

    return states

def store_data_by_name(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()

def create_gif(image_list, gif_name, duration):
    """
    Creates a GIF from a list of images.
    :param image_list: List of NumPy arrays representing the images.
    :param gif_name: Name of the output GIF file.
    :param duration: Duration of each frame in the GIF in seconds.
    """
    with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
        for img in image_list:
            writer.append_data(img)



def save_datapoint(data_dict, steps, data_save_path, save_pcd=True):
    save_name = "data_{:06}".format(steps)

    if save_pcd:
        store_data_by_name(
            ['pcd_pos', 'params', 'action', 'done', 'gripper_pos', 'back_pcd', 'front_pcd', 'pick'],
            # s_t, e, a_t, done
            [data_dict['past_pcd_pos'], data_dict['params'], data_dict['action'], data_dict['done'],
             data_dict['past_gripper_pos'], data_dict['past_back_pcd'], data_dict['past_front_pcd'], data_dict['pick']],
            osp.join(data_save_path, save_name))

        data_dict['past_pcd_pos'] = data_dict['pcd_pos']
        data_dict['past_gripper_pos'] = data_dict['gripper_pos']
        data_dict['past_back_pcd'] = data_dict['back_pcd']
        data_dict['past_front_pcd'] = data_dict['front_pcd']

    else:
        store_data_by_name(
            ['pcd_pos', 'params', 'action', 'done', 'gripper_pos', 'pick'],
            # s_t, e, a_t, done
            [data_dict['past_pcd_pos'], data_dict['params'], data_dict['action'], data_dict['done'],
             data_dict['past_gripper_pos'], data_dict['pick']],
            osp.join(data_save_path, save_name))

        data_dict['past_pcd_pos'] = data_dict['pcd_pos']
        data_dict['past_gripper_pos'] = data_dict['gripper_pos']
        data_dict['past_back_pcd'] = data_dict['back_pcd']
        data_dict['past_front_pcd'] = data_dict['front_pcd']
