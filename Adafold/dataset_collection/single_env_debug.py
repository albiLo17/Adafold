
from assistive_gym.envs.half_folding_former import HalfFoldEnv

import copy
import time
from Adafold.args.arguments import get_argparse
import numpy as np
import pybullet as p
from Adafold.dataset_collection.dataset_args import get_argparse_dataset
from Adafold.dataset_collection.utils import reset_env, get_states, save_datapoint, create_gif
import os
import matplotlib.pyplot as plt

def collect_trajectory(args,
                       args_dataset,
                       env_idx,
                       elas,
                       bend,
                       scale,
                       frame_skip,
                       side,
                       task='fold',
                       traj_type='fixed',
                       save_pcd=False,
                       return_frames=False,
                       states=None,):
    # TODO: add specific task

    data_dict = {'pcd_pos': [], 'back_pcd': [], 'front_pcd': [], 'gripper_pos': [],
                 'past_pcd_pos': [], 'past_back_pcd': [], 'past_front_pcd': [], 'past_gripper_pos': [],
                 'params': [], 'action': [], 'done': [], 'pick': []}

    # args.render = 1
    if args.save_data:
        idx = "{:06}".format(env_idx)
        # TODO: change naming
        data_save_path = f'{args.dataset_path}{args.dataset_name}/env_e={elas}_b={bend}_sc={scale}_f={frame_skip}_sd={side}/{idx}'
        os.makedirs(data_save_path, exist_ok=True)

    env = HalfFoldEnv(frame_skip=frame_skip,
                      hz=100,
                      action_mult=args_dataset.action_mult,
                      obs=args.obs,
                      side=side,
                      reward=args.reward)

    # TODO: add specific task
    if args.render:
        env.render(width=640, height=480)

    reset_env(env, elas, bend, scale)
    # env.setup_camera(camera_eye=[0.65, 0., 0.65], camera_target=[0., 0.0, 0.], camera_width=720,
    #                  camera_height=720)
    env.setup_camera_rpy(camera_target=[-0., 0, 0.], distance=0.55, rpy=[0, -30, 45])

    data_dict['params'] = [elas, bend, scale, frame_skip, side]

    # pick and place among corners and middle edges
    positions = env.get_corners()

    # try:
    pick_pos = positions[0]
    env.pick(pick_pos)
    data_dict['pick'] = 1

    # start from the same position the rw starts after grasping (or assuming pregrasped)
    z_offset = 0.03/0.1*scale
    rw_action = np.zeros_like(pick_pos)
    rw_action[-1] += z_offset

    if task == 'fold':
        place_pos = positions[1]
        place_pos[-1] = z_offset
    elif task == 'lift':
        z_lift = 0.3
        place_pos = copy.deepcopy(pick_pos)
        place_pos[2] += z_lift
    elif task == 'drag':
        y_drag = 0.3
        place_pos = copy.deepcopy(pick_pos)
        place_pos[1] += y_drag
        place_pos[-1] = z_offset


    obs, reward, done, info = env.step(action=rw_action)
    data_dict['gripper_pos'] = env.sphere_ee.get_base_pos_orient()[0]
    data_dict['past_gripper_pos'] = copy.deepcopy(data_dict['gripper_pos'])

    data_dict['pcd_pos'] = obs.reshape(-1, 3)
    data_dict['past_pcd_pos'] = copy.deepcopy(data_dict['pcd_pos'])
    if save_pcd:
        data_dict['front_pcd'], data_dict['back_pcd'] = env.process_pointclouds(RGB_threshold=[0.78, 0.78, 0.78],
                                                                        voxel_size=0.005)
        data_dict['past_front_pcd'], data_dict['past_back_pcd'] = copy.deepcopy(data_dict['front_pcd']), copy.deepcopy(data_dict['back_pcd'])

    valid = True
    steps = 0

    # set a new random seed using the current time and the process ID
    seed_value = int(time.time()) + os.getpid()
    np.random.seed(seed_value)

    # init sampling distribution
    ########################
    if states is None:
        states = get_states(args, traj_type, place_pos,
                            gripper_pos=data_dict['gripper_pos'],
                            num_actions=args_dataset.num_actions,
                            action_norm=args_dataset.action_norm,
                            action_mult=args_dataset.action_mult)

    next_states = states[1:]

    frames_gif = []
    for i, pos in enumerate(next_states):
        if i > 0:
            past_obs = obs

        current_pos = env.sphere_ee.get_base_pos_orient()[0]
        data_dict['action'] = pos - current_pos
        obs, reward, done, info = env.step(action=data_dict['action'])

        rgb, depth = env.get_camera_image_depth(shadow=True)
        rgb = rgb.astype(np.uint8)
        frames_gif.append(rgb)

        data_dict['pcd_pos'] = obs.reshape(-1, 3)
        data_dict['gripper_pos'] = env.sphere_ee.get_base_pos_orient()[0]
        if save_pcd:
            data_dict['front_pcd'], data_dict['back_pcd'] = env.process_pointclouds(RGB_threshold=[0.78, 0.78, 0.78],
                                                             voxel_size=0.005)

        # Stability check in case the simulation explodes
        if i > 0:
            if np.linalg.norm(past_obs - obs) > 2.:
                print(f'############# Instability detected env_e={elas}_b={bend}_sc={scale}_f={frame_skip}_sd={side}')
                valid = False
                break

        if args.save_data and valid:
            data_dict['done'] = False
            save_datapoint(data_dict, steps, data_save_path, save_pcd=save_pcd)

        steps += 1

    # Place and collect last observations
    data_dict['action'] = np.zeros(args_dataset.action_dim)
    data_dict['pick'] = 0
    obs, reward, done, info = env.place()
    data_dict['pcd_pos'] = obs.reshape(-1, 3)
    data_dict['gripper_pos'] = env.sphere_ee.get_base_pos_orient()[0]

    if save_pcd:
        data_dict['front_pcd'], data_dict['back_pcd'] = env.process_pointclouds(RGB_threshold=[0.78, 0.78, 0.78],
                                                         voxel_size=0.005)


    # rgb, depth = env.render_image()
    # save last state but grasped
    if args.save_data and valid:
        data_dict['done'] = False
        save_datapoint(data_dict, steps, data_save_path, save_pcd=save_pcd)


    # save last state not grasped
    data_dict['action'] = np.zeros(args_dataset.action_dim)
    obs, reward, done, info = env.step(action=data_dict['action'])

    data_dict['pcd_pos'] = obs.reshape(-1, 3)
    data_dict['gripper_pos'] = env.sphere_ee.get_base_pos_orient()[0]
    if save_pcd:
        data_dict['front_pcd'], data_dict['back_pcd'] = env.process_pointclouds(RGB_threshold=[0.78, 0.78, 0.78],
                                                             voxel_size=0.005)
    if args.save_data and valid:
        data_dict['done'] = True
        save_datapoint(data_dict, steps, data_save_path, save_pcd=save_pcd)


    p.disconnect()

    if return_frames:
        return frames_gif



if __name__ == "__main__":
    args = get_argparse()
    args.reward = 'IoU'
    args_dataset = get_argparse_dataset()

    args.render = 1
    elas, bend, scale, frame_skip, side = 40, 60, 0.1, 4, 0
    # elas, bend, scale, frame_skip, side = 40, 60, 0.1, 2, 1  # side 1 is rigid
    elas, bend, scale, frame_skip, side = 100, 100, 0.1, 4, 0  # side 1 is rigid
    # elas, bend, scale, frame_skip, side = [40, 40., 0.1, 4, 0]
    # #
    # main(args, elas, bend, scale, frame_skip, side, velocity=0.03)
    # test_traj(args, elas, bend, scale, frame_skip, side, velocity=0.02)

    # Record Gif
    # elas, bend, scale, frame_skip, side = 20, 20, 0.1, 4, 0
    elas, bend, scale, frame_skip, side = 50, 50, 0.1, 4, 0
    # elas, bend, scale, frame_skip, side = 100, 100, 0.1, 4, 0
    # params = [40, 60, 0.1, 4, 0]
    # params = [100, 100, 0.1, 4, 0]
    env_idx = 0
    args.save_data = False
    frames_gif = collect_trajectory(args,
                       args_dataset,
                       env_idx,
                       elas,
                       bend,
                       scale,
                       frame_skip,
                       side,
                       task='fold',
                       traj_type='fixed',
                       return_frames=True,
                       save_pcd=False,
                       states=None,)
    gif_path = './data/gif/output_gif_50_50.gif'
    create_gif(frames_gif, gif_path, 0.5)
    plt.imshow(frames_gif[2])
    plt.axis(False)
    plt.savefig('./data/gif/output_img_50_50.png')



    # May4 dataset
    params = [40, 60, 0.1, 4, 0]
    env_idx = 0
    collect_trajectory(args,
                       args_dataset,
                       env_idx,
                       elas,
                       bend,
                       scale,
                       frame_skip,
                       side,
                       task='fold',
                       traj_type='fixed',
                       save_pcd=False,
                       states=None,)



