import copy
import time
from Adafold.args.arguments import get_argparse
from Adafold.trajectory.trajectory import Action_Sampler_Simple
import multiprocessing
import numpy as np
import pybullet as p
from Adafold.dataset_collection.dataset_args import get_argparse_dataset
from Adafold.dataset_collection.utils import reset_env, save_datapoint
import os
from assistive_gym.envs.half_folding_former import HalfFoldEnv
from tqdm import tqdm

def collect_trajectory(args,
                       args_dataset,
                       env_idx,
                       traj_idx,
                       elas,
                       bend,
                       scale,
                       frame_skip,
                       side,
                       save_pcd=False,
                       states=None,):


    data_dict = {'pcd_pos': [], 'back_pcd': [], 'front_pcd': [], 'gripper_pos': [],
                 'past_pcd_pos': [], 'past_back_pcd': [], 'past_front_pcd': [], 'past_gripper_pos': [],
                 'params': [], 'action': [], 'done': [], 'pick': []}

    # args.render = 1
    if args.save_data:
        t_idx = "{:06}".format(traj_idx)
        data_save_path = f'{args.dataset_path}{args.dataset_name}/env_e={elas}_b={bend}_sc={scale}_f={frame_skip}_sd={side}/{t_idx}'
        os.makedirs(data_save_path, exist_ok=True)

    env = HalfFoldEnv(frame_skip=frame_skip,
                      hz=100,
                      action_mult=args_dataset.action_mult,
                      obs=args.obs,
                      side=side,
                      reward=args.reward)

    if args.render:
        env.render(width=640, height=480)

    reset_env(env, elas, bend, scale)

    data_dict['params'] = [elas, bend, scale, frame_skip, side]

    # pick and place among corners and middle edges
    positions = env.get_corners()

    # try:
    pick_pos = positions[0]
    env.pick(pick_pos)
    data_dict['pick'] = 1

    # start from the same position the rw starts after grasping (or assuming pregrasped)
    z_offset = 0.01
    rw_action = np.zeros_like(pick_pos)
    rw_action[-1] += z_offset


    place_pos = positions[1]
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

    next_states = states[1:]

    frames_gif = []
    for i, pos in enumerate(tqdm(next_states, desc=f"Collecting traj {traj_idx}")):

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
        
        print(f"Saved trajectory {traj_idx} for env_e={elas}_b={bend}_sc={scale}_f={frame_skip}_sd={side} at {data_save_path}")

        p.disconnect()


def collect_in_parallel(args):
    arg, args_dataset, env_idx, traj_idx, elas, bend, scale, frame_skip, side, save_pcd, states = args
    collect_trajectory(arg,
                       args_dataset,
                       env_idx,
                       traj_idx,
                       elas,
                       bend,
                       scale,
                       frame_skip,
                       side,
                       save_pcd=save_pcd,
                       states=states)


def mutliprocess_collection(
        arg,
        args_dataset,
        env_idx,
        save_pcd,
        states_list,
        num_processes=10,
        params=[40, 60, 0.1, 4, 0],
):
    elas, bend, scale, frame_skip, side = params
    with multiprocessing.Pool(num_processes) as pool:

        args_list = [
            (arg, args_dataset, env_idx, i, elas, bend, scale, frame_skip, side, save_pcd, states) for
            i, states in enumerate(states_list)]
        pool.map(collect_in_parallel, args_list)

    print("All processes have finished.")



if __name__ == "__main__":
    args = get_argparse()
    args.dataset_name = '/oct31_1s'
    args_dataset = get_argparse_dataset()
    args.render = 0
    args.reward = 'IoU'

    # Dataset params dataset
    params = [40, 60, 0.1, 4, 0]
    env_idx = 0
    save_pcd = True

    pick_pos = np.asarray([-0.1, 0.1, 0.01])
    place_pos = np.asarray([-0.1, -0.1, 0.01])

    state_sampler = Action_Sampler_Simple(
        N=12,              # trajectory length
        action_len=0.03,
        c_threshold=0.,
        grid_size=0.04,
        pp_dir=place_pos -pick_pos,
        starting_point=pick_pos,
        sampling_mean=None,
        fixed_trajectory=None)

    # Oct31
    states_list = state_sampler.generate_dataset(num_trajectories=2000, starting_point=pick_pos, target_point=place_pos, prob=False)

    mutliprocess_collection(
        args,
        args_dataset,
        env_idx,
        save_pcd,
        states_list,
        num_processes=1,
        params=[40, 60, 0.1, 4, 0],
    )
