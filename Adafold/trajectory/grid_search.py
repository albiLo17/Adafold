from scipy.spatial import cKDTree
from assistive_gym.envs.half_folding_former import HalfFoldEnv
import imageio
import copy
from Adafold.trajectory.trajectory import TrajectorySimple
from Adafold.viz.viz_pcd import plot_pcd
import multiprocessing
import numpy as np
import pybullet as p
from Adafold.dataset_collection.utils import reset_env
import os
import matplotlib.pyplot as plt

def get_linear_states(gripper_pos, place_pos, h):
    mid_w1 = copy.deepcopy(gripper_pos)
    mid_w1[-1] = h
    mid_w2 = copy.deepcopy(place_pos)
    mid_w2[-1] = h
    waypoints = np.asarray([gripper_pos, mid_w1, mid_w2, place_pos])

    controller = TrajectorySimple(waypoints=waypoints,
                            vel=0.03,  # as it will be rescaled when processed
                            interpole=True,
                            action_scale=1,)
    states = controller.traj_points
    return states

def get_trapezoidal_states(gripper_pos, place_pos, h, alpha):
    mid_w1 = alpha*copy.deepcopy(gripper_pos) + (1 - alpha)*copy.deepcopy(place_pos)
    mid_w1[-1] = h
    mid_w2 = (1 - alpha)*copy.deepcopy(gripper_pos) + alpha*copy.deepcopy(place_pos)
    mid_w2[-1] = h
    waypoints = np.asarray([gripper_pos, mid_w1, mid_w2, place_pos])

    controller = TrajectorySimple(waypoints=waypoints,
                            vel=0.03,  # as it will be rescaled when processed
                            interpole=True,
                            action_scale=1)
    states = controller.traj_points
    return states

def get_3d_states(gripper_pos, place_pos, h, alpha, beta):
    mid_w1 = alpha*copy.deepcopy(gripper_pos) + (1 - alpha)*copy.deepcopy(place_pos)
    mid_w1[0] += beta
    mid_w1[-1] = h
    mid_w2 = (1 - alpha)*copy.deepcopy(gripper_pos) + alpha*copy.deepcopy(place_pos)
    mid_w2[0] += beta
    mid_w2[-1] = h
    waypoints = np.asarray([gripper_pos, mid_w1, mid_w2, place_pos])

    controller = TrajectorySimple(waypoints=waypoints,
                            vel=0.03,  # as it will be rescaled when processed
                            interpole=True,
                            action_scale=1)
    states = controller.traj_points
    return states

# Function to filter points in the second set based on distance to the nearest point in the first set
def filter_points(set1, set2):
    # Distance threshold
    D_max = 0.005

    # Build a KDTree for the first set of points
    tree = cKDTree(set1)

    filtered_set = []
    for point in set2:
        # Find the distance to the nearest point in the first set
        distance, _ = tree.query(point)
        # Keep the point if the distance is less than or equal to D_max
        if distance <= D_max:
            filtered_set.append(point)
    return np.array(filtered_set)

def compute_2d_iou_and_plot_occupancy(set1, set2, grid_size=0.05, plot=False, save=False):
    # Project 3D points onto 2D by ignoring the Z dimension
    set1_2d = set1[:, :2]
    set2_2d = set2[:, :2]

    # Compute axis-aligned bounding boxes (AABB)
    min_bound = np.minimum(np.min(set1_2d, axis=0), np.min(set2_2d, axis=0))
    max_bound = np.maximum(np.max(set1_2d, axis=0), np.max(set2_2d, axis=0))

    # Create occupancy grids
    x_grid = np.arange(min_bound[0], max_bound[0], grid_size)
    y_grid = np.arange(min_bound[1], max_bound[1], grid_size)
    grid1 = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
    grid2 = np.zeros((len(x_grid), len(y_grid)), dtype=bool)

    # Mark occupied cells in the grids
    for x in set1_2d:
        i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
        grid1[i, j] = True
    for x in set2_2d:
        i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
        grid2[i, j] = True

    # Compute intersection and union
    intersection = np.logical_and(grid1, grid2)
    union = np.logical_or(grid1, grid2)
    inter_area = np.sum(intersection)
    union_area = np.sum(union)

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    # Plotting
    if plot:
        fig, ax = plt.subplots()
        ax.set_title(f"2D IoU: {iou:.2f}")

        # Plot occupancy grids
        ax.imshow(grid1.T, extent=(min_bound[0], max_bound[0], min_bound[1], max_bound[1]), origin='lower', cmap='Blues', alpha=0.5)
        ax.imshow(grid2.T, extent=(min_bound[0], max_bound[0], min_bound[1], max_bound[1]), origin='lower', cmap='Greens', alpha=0.5)

        if not save:
            plt.show()
        else:
            return iou, fig

    return iou, None

def execute_trajectory(h=0.03, alpha=1., beta=0., traj_type='linear', render=0, params=[ 40, 60, 0.1, 4, 0]):
    elas, bend, scale, frame_skip, side = params
    env = HalfFoldEnv(frame_skip=frame_skip,
                      hz=100,
                      action_mult=1,
                      obs='pcd',
                      side=side,
                      reward='corner')
    if render:
        env.render(width=640, height=480)
    reset_env(env, elas, bend, scale)

    # pick and place among corners and middle edges
    positions = env.get_corners()

    # try:
    pick_pos = positions[0]
    env.pick(pick_pos)

    # start from the same position the rw starts after grasping (or assuming pregrasped)
    z_offset = 0.01
    lift_action = np.zeros_like(pick_pos)
    lift_action[-1] += z_offset

    env.step(action=lift_action)

    place_pos = positions[1]
    place_pos[-1] = z_offset

    gripper_pos = env.sphere_ee.get_base_pos_orient()[0]

    if traj_type == 'linear':
        states = get_linear_states(gripper_pos, place_pos, h)
    if traj_type == 'trapezoidal':
        states = get_trapezoidal_states(gripper_pos, place_pos, h, alpha)
    if traj_type == '3d':
        states = get_3d_states(gripper_pos, place_pos, h, alpha, beta)

    next_states = states[1:]
    obs_list = [env._get_obs().reshape(-1,3)]

    frames_gif = []
    rgb, depth = env.get_camera_image_depth(shadow=True)
    rgb = rgb.astype(np.uint8)
    frames_gif.append(rgb)

    steps = 0
    for i, pos in enumerate(next_states):
        current_pos = env.sphere_ee.get_base_pos_orient()[0]
        action = pos - current_pos
        obs, reward, done, info = env.step(action=action)
        obs_list.append(obs.reshape(-1,3))
        rgb, depth = env.get_camera_image_depth(shadow=True)
        rgb = rgb.astype(np.uint8)
        frames_gif.append(rgb)

        steps += 1

    p.disconnect()

    return obs_list, frames_gif, states


def evaluate_trajectory(h=0.04, alpha=1., beta=0., traj_type='linear', save_dir=None, save=False, render=0, params=[ 40, 60, 0.1, 4, 0]):
    obs_list, frames_gif, states = execute_trajectory(h=h, alpha=alpha, beta=beta, traj_type=traj_type, render=render, params=params)

    init_obs = obs_list[0]
    indeces_half = np.where(init_obs[:, 1] <= 0.)
    set1 = init_obs[indeces_half]
    indeces_moving_half = np.where(init_obs[:, 1] > 0.)

    IoU_list = []
    for obs in obs_list:
        set2 = obs[indeces_moving_half]
        iou_2d, _ = compute_2d_iou_and_plot_occupancy(set1, set2, grid_size=0.01)
        IoU_list.append(iou_2d)

    iou_2d, fig = compute_2d_iou_and_plot_occupancy(set1, set2, grid_size=0.01, plot=True, save=save)

    if save:
        save_dir += f'_IoU:{round(iou_2d, 2)}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'occupancy.png'))
        np.save(os.path.join(save_dir, 'iou.npy'), np.asarray([iou_2d]))
    print("2D IoU:", iou_2d)

    # plot reward
    plt.figure()
    plt.plot(IoU_list)
    plt.title("IoU per timestep")
    plt.ylabel('IoU')
    plt.xlabel('time step')
    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, 'occupancy_over_time.png'))

    # plot visited states
    fig = plot_pcd(np.asarray(states), return_fig=True)
    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, 'visited_states.png'))

        gif_path = os.path.join(save_dir, 'traj.gif')
        imageio.mimsave(gif_path, frames_gif, fps=3, subrectangles=True)


def evaluate_in_parallel(args):
    h, alpha, beta, traj_type, save_dir, save, params = args
    evaluate_trajectory(h=h, alpha=alpha, beta=beta, traj_type=traj_type, save_dir=save_dir, save=save, params=params)

def mutliprocess_evaluation(num_processes=10,
                            params=[40, 60, 0.1, 4, 0],
                            save=False):

    cloth_folder = f'e={params[0]}_b={params[1]}_s={params[2]}'

    # Using multiprocessing pool to manage parallel execution
    with multiprocessing.Pool(num_processes) as pool:
        # First set of evaluations
        traj_type = 'linear'
        args_list = [(h, None, None, traj_type, os.path.join(os.getcwd(), 'exps', cloth_folder, traj_type, str(h)), save, params) for h in [0.04, 0.05, 0.06, 0.07, 0.08]]
        pool.map(evaluate_in_parallel, args_list)

        # Second set of evaluations
        traj_type = 'trapezoidal'
        args_list = [(h, alpha, None, traj_type, os.path.join(os.getcwd(), 'exps', cloth_folder, traj_type, str(h) + '_' + str(alpha)), save, params)
                     for h in [0.04, 0.05, 0.06, 0.07, 0.08] for alpha in [0.7, 0.8, 0.9]]
        pool.map(evaluate_in_parallel, args_list)

        # Third set of evaluations
        traj_type = '3d'
        args_list = [(h, alpha, beta, traj_type, os.path.join(os.getcwd(), 'exps', cloth_folder, traj_type, str(h) + '_' + str(alpha) + '_' + str(beta)), save, params)
                     for h in [0.04, 0.05, 0.06, 0.07, 0.08] for alpha in [0.7, 0.8, 0.9] for beta in [0.03, 0.05, 0.07]]
        pool.map(evaluate_in_parallel, args_list)

    print("All processes have finished.")


if __name__=='__main__':
    save = True
    params = [40, 60, 0.1, 4, 0]    # [elas, bend, scale, frame_skip, side]
    mutliprocess_evaluation(num_processes=10,
                            params=params,
                            save=save)

    # multi_params = [[40, 60, 0.1, 4, 0], [60, 60, 0.1, 4, 0], [80, 60, 0.1, 4, 0], [100, 60, 0.1, 4, 0],
    #                 [40, 80, 0.1, 4, 0], [60, 80, 0.1, 4, 0], [80, 80, 0.1, 4, 0], [100, 80, 0.1, 4, 0],
    #                 [40, 100, 0.1, 4, 0], [60, 100, 0.1, 4, 0], [80, 100, 0.1, 4, 0], [100, 100, 0.1, 4, 0],
    #
    #                 [40, 60, 0.15, 4, 0], [60, 60, 0.15, 4, 0], [80, 60, 0.15, 4, 0], [100, 60, 0.15, 4, 0],
    #                 [40, 80, 0.15, 4, 0], [60, 80, 0.15, 4, 0], [80, 80, 0.15, 4, 0], [100, 80, 0.15, 4, 0],
    #                 [40, 100, 0.15, 4, 0], [60, 100, 0.15, 4, 0], [80, 100, 0.15, 4, 0], [100, 100, 0.15, 4, 0],
    #
    #                 [40, 60, 0.07, 4, 0], [60, 60, 0.07, 4, 0], [80, 60, 0.07, 4, 0], [100, 60, 0.07, 4, 0],
    #                 [40, 80, 0.07, 4, 0], [60, 80, 0.07, 4, 0], [80, 80, 0.07, 4, 0], [100, 80, 0.07, 4, 0],
    #                 [40, 100, 0.07, 4, 0], [60, 100, 0.07, 4, 0], [80, 100, 0.07, 4, 0], [100, 100, 0.07, 4, 0],
    #                 ]
    #
    # for params in multi_params:
    #     print(f'Processing params: {params}')
    #     mutliprocess_evaluation(num_processes=10,
    #                             params=params,
    #                             save=save)

    # cloth_folder = f'e={params[0]}_b={params[1]}_s={params[2]}'
    # traj_type = 'linear'
    # for h in [0.05, 0.06, 0.07, 0.08]:
    #     save_dir = os.path.join(os.getcwd(), 'exps', traj_type, str(h))
    #     evaluate_trajectory(h=h, save_dir=save_dir, save=save, params=params)
    #
    # traj_type = 'trapezoidal'
    # for h in [0.05, 0.06, 0.07, 0.08]:
    #     for alpha in [0.7, 0.8, 0.9]:
    #         save_dir = os.path.join(os.getcwd(), 'exps', traj_type, str(h) + '_' + str(alpha))
    #         evaluate_trajectory(h=h, alpha=alpha, traj_type=traj_type, save_dir=save_dir, save=save, params=params)
    #
    # traj_type = '3d'
    # for h in [0.05, 0.06, 0.07, 0.08]:
    #     for alpha in [0.7, 0.8, 0.9]:       # should be between (0.5 to 1)
    #         for beta in [0.03, 0.05, 0.07]:
    #             save_dir = os.path.join(os.getcwd(), 'exps', traj_type, str(h) + '_' + str(alpha)+ '_' + str(beta))
    #             evaluate_trajectory(h=h, alpha=alpha, beta=beta, traj_type=traj_type, save_dir=save_dir, save=save, params=params)


    print()





