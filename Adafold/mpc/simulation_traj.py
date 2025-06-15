import numpy as np
from assistive_gym.envs.half_folding_former import HalfFoldEnv

def read_sim_data():
    import json
    with open('env_data.json', 'r') as f:
        data = json.load(f)
    return data

def simulate_trajectory_test(_):
    print('*********************')

def simulate_trajectory(traj_actions):
    print('*********************')
    data = read_sim_data()
    elas = data["elas"]
    bend = data["bend"]
    scale = data["scale"]
    frame_skip = data["frame_skip"]
    side = data["side"]
    action_mult = data["action_mult"]
    obs = data["obs"]
    gripper_attractor_tr = data["gripper_attractor_tr"]
    grid_res = data["grid_res"]
    reward_env_type = data["reward_env_type"]
    half_mesh_indeces = data["half_mesh_indeces"]

    env = HalfFoldEnv(frame_skip=frame_skip,
                           hz=100,
                           action_mult=action_mult,
                           obs=obs,
                           side=side,
                           gripper_attractor_tr=gripper_attractor_tr,
                           grid_res=grid_res,
                           reward=reward_env_type)  # the reward of the env is only the alignment!

    damp, frict = 1.5, 1.50
    raw_obs = env.reset(stiffness=[elas, bend, damp], friction=frict, cloth_scale=scale,
                                  cloth_mass=0.5)  # Elas, bend, damp

    positions = env.get_corners()

    # try:
    x_pick = positions[0]
    env.pick(x_pick)

    z_offset = 0.01
    action = np.zeros_like(x_pick)
    action[-1] += z_offset

    x_place = positions[1]
    x_place[-1] = z_offset

    obs, reward, done, info = env.step(action=action)

    for i, a in enumerate(traj_actions):
        raw_obs, reward, done, info = env.step(action=a)

    mesh = raw_obs.reshape(-1, 3)
    half_mesh = mesh[half_mesh_indeces]

    return half_mesh

if __name__ == '__main__':
    simulate_trajectory_test()
    simulate_trajectory(np.zeros((1, 4)))

