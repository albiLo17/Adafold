import time
from assistive_gym.envs.half_folding_former import HalfFoldEnv
from Adafold.trajectory.trajectory import Trajectory, Action_Sampler
from Adafold.args.arguments import get_argparse
from Adafold.utils.utils_datacollection import *
from Adafold.viz.viz_pcd import plot_pcd_list


def generate(args, frame_skip, side, elas, bend, scale, traj_type, num_actions, action_norm, num_generations=1):
    action_mult = 1
    env = HalfFoldEnv(frame_skip=frame_skip,
                      hz=100,
                      action_mult=action_mult,
                      obs=args.obs,
                      side=side,
                      reward=args.reward)
    if args.render:
        env.render(width=640, height=480)

    reset_env(env, elas, bend, scale)

    # pick and place among corners and middle edges
    positions = env.get_corners()

    # try:
    pick_pos = positions[0]
    place_pos = positions[1]
    env.pick(pick_pos)


    # start from the same position the rw starts after grasping (or assuming pregrasped)
    z_offset = 0.03 /0.1 *scale
    rw_action = np.zeros_like(pick_pos)
    rw_action[-1] += z_offset
    place_pos[-1] = z_offset

    obs, reward, done, info = env.step(action=rw_action)
    gripper_pos = env.sphere_ee.get_base_pos_orient()[0]

    # set a new random seed using the current time and the process ID
    seed_value = int(time.time()) + os.getpid()
    np.random.seed(seed_value)

    all_states = []
    for i in range(num_generations):
        if traj_type == 'fixed':
            mid_w = (place_pos + gripper_pos) / 2
            mid_w[2] = 0.08
            waypoints = np.asarray([gripper_pos, mid_w, place_pos])


            controller = Trajectory(args=args,
                                    waypoints=waypoints,
                                    vel=action_norm, # as it will be rescaled when processed
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

        all_states.append(np.asarray(states))

    return all_states


if __name__=='__main__':
    args = get_argparse()
    args.render = 0
    elas, bend, scale, frame_skip, side = [40, 60, 0.1, 4, 0]
    traj_type = 'fixed'
    num_actions = 10
    action_norm = 0.03
    states_fixed = generate(args, frame_skip, side, elas, bend, scale, 'fixed', num_actions, action_norm)
    states_random = generate(args, frame_skip, side, elas, bend, scale, 'random', num_actions, action_norm, num_generations=10)
    plot_pcd_list(states_fixed + states_random)

    # np.save('../data/generated_traj/fixed_3008.npy', np.asarray(states_fixed))
    # np.save('../data/generated_traj/random_3008.npy', np.asarray(states_random)
    # states_random = generate(args, frame_skip, side, elas, bend, scale, 'random', num_actions, action_norm)
    # plot_pcd_list()