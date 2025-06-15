import numpy as np
# import pybullet as p
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
# from assistive_gym_fem.assistive_gym.envs.half_folding_BO import HalfFoldEnv
# from assistive_gym_fem.assistive_gym.utils.normalized_env import normalize
# from folding.arguments import get_argparse
import os
import copy
from Adafold.viz.viz_pcd import plot_pcd, plot_pcd_list
# from assistive_gym_fem.Adafold.viz.viz_pcd import plot_pcd, plot_pcd_list
import matplotlib.pyplot as plt



class Action_Sampler():
    def __init__(self,
                 N=14,              # trajectory length
                 action_len=0.02,
                 c_threshold=0.3,
                 pp_dir=None,
                 starting_point=None,
                 sampling_mean=None,
                 fixed_trajectory=None,
                 rw=False):

        self.action_dim = 3
        self.action_len = action_len

        self.real_world = rw
        self.scale_x = (0.18/(0.1*2))
        # self.scale_y = (0.25/(0.1*2))
        self.scale_y = (0.22/(0.1*2))

        # constraint parameters: cosine similarity and workspace limits
        self.c_threshold = c_threshold
        # TODO: make sure that the cloth is not too big
        self.box_min = np.asarray([-0.3, -0.3, 0.02])
        self.box_max = np.asarray([0.3, 0.3, 0.3])


        # set the pick and place direction to constrain the action sampling
        self.pp_dir = pp_dir

        # this is the mean of the sampling trajectory
        # if not porvided will be set equal to zero
        self.N = N
        self.traj_mean = np.zeros((N, 3))
        if sampling_mean is not None:
            self.traj_mean = sampling_mean

        # use it to kick start the planning (if None)
        self.fixed_trajectory = fixed_trajectory

        # this is the starting point of the trajectory
        # if not porvided will be set equal to zero
        self.starting_point = np.zeros((3))
        if starting_point is not None:
            self.starting_point = starting_point


    def sample_trajectory(self, starting_point=None, return_actions=False):
        """"
        Sample a random trajectory that does not go outside the workspace
        """
        if starting_point is None:
            starting_point = self.starting_point

        actions = []
        traj = [starting_point]
        if self.fixed_trajectory is not None:
            traj = [self.fixed_trajectory[i] for i in range(self.fixed_trajectory.shape[0])]
        for i in range(self.N - len(traj) + 1):
            count = 0
            action = self.sample_constrained_action(i)
            next_state = traj[-1] + action
            while(self.check_workspace_constraint(next_state) and count < 1000):
                count += 1
                action = self.sample_constrained_action(i)
                next_state = traj[-1] + action
                if count >= 1000:
                    print('########### Cannot generate action - add zero action ###########')
                    # os.system("taskkill /im make.exe")
                    action = np.zeros_like(action)
                    next_state = traj[-1] + action
                    continue
            # if count >= 1000:
            #     exit
            traj.append(traj[-1] + action)
            actions.append(action)

        if return_actions:
            return traj, actions

        return traj

    def sample_constrained_action(self, i):
        """"
        sample an action that satisfy the cosine similarity with the
        direction of pick and place.
        The mean of the sampling distribution comes from the predefined mean
        which is initialized as zero but could be varied.
        """
        mean = self.traj_mean[i]
        var = np.ones_like(mean)*0.1#0.01
        action = np.random.normal(mean, var)
        # past_a = action
        # apply the constraint
        while not self.cosine_similarity(action, self.pp_dir) >= self.c_threshold:
            # past_a = copy.deepcopy(action)
            action = np.random.normal(mean, var)
            # plot_pcd_list([action.reshape(1, 3), self.pp_dir.reshape(1, 3)], elev=90, title=f'{self.cosine_similarity(action, self.pp_dir)}')
            # print(self.cosine_similarity(action, self.pp_dir))

        # print(f'action {i}')
        if self.real_world:
            action = action / np.linalg.norm(action) * self.action_len
        else:
            # as the action will be post processed and we want them to be the same as in the real world
            direction = action / np.linalg.norm(action)
            direction[0] *= self.scale_x
            direction[1] *= self.scale_y
            action = direction / np.linalg.norm(direction) * self.action_len
            action[0] /= self.scale_x
            action[1] /= self.scale_y

        return action


    def cosine_similarity(self, a, b):
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cosine

    def check_workspace_constraint(self, next_pos):
        # Check if each dimension of next_pos exceeds the box bounds
        exceed_min = next_pos - self.box_min
        exceed_max = self.box_max - next_pos

        # Determine whether each dimension exceeds the box bounds or not
        exceeds_limits = np.logical_or(np.any(exceed_min < 0), np.any(exceed_max < 0))

        return exceeds_limits


class Action_Sampler_Simple():
    def __init__(self,
                 N=14,  # trajectory length
                 action_len=0.03,
                 c_threshold=0.,
                 noise_sigma=0.01,
                 cloth_half_size_x=0.1,
                 cloth_half_size_y=0.1,
                 pp_dir=None,
                 place=None,
                 starting_point=None,
                 grid_size=0.01,
                 sampling_mean=None,
                 fixed_trajectory=None):

        self.action_dim = 3
        self.action_len = action_len
        self.grid_size = grid_size
        self.cell_visit_counts = {}

        # constraint parameters: cosine similarity and workspace limits
        self.c_threshold = c_threshold

        # TODO: make sure that the cloth is not too big
        # self.box_min = np.asarray([-cloth_half_size_x - 0.1, -cloth_half_size_y - 0.1, 0.01])
        # self.box_max = np.asarray([cloth_half_size_x, cloth_half_size_y + 0.1, 0.3])

        # New box constraints to avoid actions that go ouside the cloth as well
        # self.box_min = np.asarray([-cloth_half_size_x - 0.03, -cloth_half_size_y - 0.1, 0.01])        # New tighter constraints
        self.box_min = np.asarray([-cloth_half_size_x - 0.1, -cloth_half_size_y - 0.1, 0.01])
        self.box_max = np.asarray([cloth_half_size_x, cloth_half_size_y + 0.1, 0.3])


        # set the pick and place direction to constrain the action sampling
        self.pp_dir = pp_dir
        self.place = place

        # this is the mean of the sampling trajectory
        # if not porvided will be set equal to zero
        self.N = N
        self.action_means = np.zeros((N, 3))
        if sampling_mean is not None:
            self.action_means = sampling_mean

        # use it to kick start the planning (if None)
        self.fixed_trajectory = fixed_trajectory

        # this is the starting point of the trajectory
        # if not porvided will be set equal to zero
        self.starting_point = np.zeros((3))
        self.noise_sigma = noise_sigma
        if starting_point is not None:
            self.starting_point = starting_point

        self.reference_traj = None

    def traj_from_actions(self, action_means=None):
        if action_means is None:
            action_means = np.asarray(self.action_means)

        traj = [self.starting_point]

        # Append the subsequent points based on action_means
        for i in range(len(action_means)):
            traj.append(traj[-1] + action_means[i])
        # traj = [traj[-1] + action_means[i - 1] for i in range(action_means.shape[0] + 1) if i !=0 else self.starting_point]

        return np.asarray(traj)

    def sample_trajectory(self, starting_point=None, target_point=None, return_actions=False):
        """"
        Sample a random trajectory that does not go outside the workspace
        """
        if starting_point is None:
            starting_point = self.starting_point

        actions = []
        traj = [starting_point]
        if self.fixed_trajectory is not None:
            traj = [self.fixed_trajectory[i] for i in range(self.fixed_trajectory.shape[0])]
        # if self.N == 0:
        #     print()

        for i in range(min(self.N, self.action_means.shape[0]) - len(traj) + 1):
            count = 0

            action = self.get_action(i, traj[-1])
            next_state = traj[-1] + action
            while (self.check_exceed_workspace_constraint(next_state) and count < 1000):
                count += 1
                action = self.get_action(i, traj[-1])
                next_state = traj[-1] + action
                if count >= 1000:
                    print('########### Cannot generate action - add zero action ###########')
                    # os.system("taskkill /im make.exe")
                    action = np.zeros_like(action)
                    next_state = traj[-1] + action
                    continue
            # if count >= 1000:
            #     exit
            traj.append(traj[-1] + action)
            actions.append(action)

        # if target point provided, pad actions that bring closer to the target point to ensure that the trajectories
        # lead to the place positions
        if target_point is not None:
            while np.linalg.norm(np.array(traj[-1]) - np.array(target_point)) > self.action_len * 0.8:
                direction = np.array(target_point) - np.array(traj[-1])
                norm = np.linalg.norm(direction)
                action = direction / norm * min(self.action_len, norm)
                traj.append(traj[-1] + action)
                actions.append(action)

        if return_actions:
            return traj, actions

        return traj

    def get_action(self, i, last_state):
        if self.place is not None:
            if self.cosine_similarity(self.pp_dir, (last_state - self.place)) > 0:
                # if the action surpassed the place position there is no need to constrain the sampling
                action = self.sample_action(i)
            else:
                action = self.sample_constrained_action(i)
        else:
            action = self.sample_constrained_action(i)
        return action

    def sample_action(self, i):
        action_mean = self.action_means[i]
        noise_mean = np.zeros(self.action_dim)
        var = np.ones_like(noise_mean) * self.noise_sigma
        action_noise = np.random.normal(noise_mean, var)
        action = action_mean + action_noise
        action = action / np.linalg.norm(action) * self.action_len

        return action

    def sample_constrained_action(self, i):
        """"
        sample an action that satisfy the cosine similarity with the
        direction of pick and place.
        The mean of the sampling distribution comes from the predefined mean
        which is initialized as zero but could be varied.
        """
        action_mean = self.action_means[i]
        noise_mean = np.zeros(self.action_dim)
        var = np.ones_like(noise_mean) * self.noise_sigma
        action_noise = np.random.normal(noise_mean, var)
        # past_a = action
        # apply the constraint

        while not self.cosine_similarity(action_noise + action_mean, self.pp_dir) >= self.c_threshold:
            # past_a = copy.deepcopy(action)
            action_noise = np.random.normal(noise_mean, var)
            # plot_pcd_list([action.reshape(1, 3), self.pp_dir.reshape(1, 3)], elev=90, title=f'{self.cosine_similarity(action, self.pp_dir)}')
            # print(self.cosine_similarity(action, self.pp_dir))

        # print(f'action {i}')
        perturbed_action = action_mean + action_noise
        action = perturbed_action / np.linalg.norm(perturbed_action) * self.action_len

        return action

    def cosine_similarity(self, a, b):
        if a.shape[0] != 3 or b.shape[0] != 3:
            print()
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cosine


    def check_exceed_workspace_constraint(self, next_pos):
        # Check if each dimension of next_pos exceeds the box bounds
        exceed_min = next_pos - self.box_min
        exceed_max = self.box_max - next_pos

        # Determine whether each dimension exceeds the box bounds or not
        exceeds_limits = np.logical_or(np.any(exceed_min < 0), np.any(exceed_max < 0))

        return exceeds_limits

    def get_cell_visit_probability(self, cell):
        """Get the probability of visiting a cell based on its visit count."""
        visit_count = self.cell_visit_counts.get(cell, 0)
        # Adjust the formula as needed to control the exploration-exploitation balance
        return 1 / (1 + visit_count)

    def sample_trajectory_with_prob(self, starting_point=None, return_actions=False):
        if starting_point is None:
            starting_point = self.starting_point

        traj = [starting_point]
        actions = []
        i = 0
        init_len = len(traj)
        while i < self.N - init_len + 1:
            # for i in rane(self.N - len(traj) + 1):
            action = self.sample_constrained_action(i)
            next_state = traj[-1] + action

            if not self.check_exceed_workspace_constraint(next_state):
                cell = tuple((next_state // self.grid_size).astype(int))
                visit_prob = self.get_cell_visit_probability(cell)

                if np.random.rand() < visit_prob:
                    self.cell_visit_counts[cell] = self.cell_visit_counts.get(cell, 0) + 1
                    traj.append(next_state.tolist())
                    # print(len(traj))
                    actions.append(action.tolist())
                    i += 1

        # print(i)
        if return_actions:
            return traj, actions

        return traj

    def generate_dataset(self, num_trajectories, starting_point=None, target_point=None, prob=False):
        dataset = []
        max_length = 0
        for _ in range(num_trajectories):
            if prob:
                traj = self.sample_trajectory_with_prob(starting_point=starting_point, return_actions=False)
            else:
                traj = self.sample_trajectory(starting_point=starting_point, target_point=target_point, return_actions=False)
            dataset.append(np.asarray(traj))
            max_length = max(max_length, len(traj))

        # Pad trajectories to have the same length
        for i in range(len(dataset)):
            length_diff = max_length - len(dataset[i])
            if length_diff > 0:
                last_value = dataset[i][-1]
                padding = np.tile(last_value, (length_diff, 1))
                dataset[i] = np.vstack((dataset[i], padding))


        return dataset

class Trajectory():
    def __init__(self,
                 args=None,
                 waypoints=None,
                 vel=1.,
                 interpole=True,
                 multi_traj=False,
                 action_scale=0.02,
                 rw=True,
                 constraint=False):

        self.action_scale = action_scale

        self.real_world = rw
        self.scale_x = (0.18/(0.1*2))
        # self.scale_y = (0.25/(0.1*2))
        self.scale_y = (0.22/(0.1*2))

        self.vertices = [24, 624]

        if args is not None:
            self.action_dim = args.action_dim
        else:
            self.action_dim = 3

        self.waypoints = waypoints
        self.vel = vel

        if not multi_traj:
            self.traj_points = self.interpol_waypoints(interpole)
            # self.dense_traj_points = self.cubic_spline()
        else:
            self.traj_points = []
            self.lengths = []
            for waypoints in self.waypoints:
                try:
                    traj = self.interpol_waypoints(interpole, waypoints)
                    self.traj_points.append(traj)
                    # self.cubic_spline(traj)
                    self.lengths.append(len(traj))
                except:
                    print('NO way')

            # Pad with zero actions to get to the longest possible trajectory
            max_len = max(self.lengths)
            #TODO: find a better strategy to deal with uneven action lengths
            self.actions = []
            for i, traj in enumerate(self.traj_points):
                last_element = traj[-1]
                while len(traj) < max_len:
                    traj.append(last_element)
                self.traj_points[i] = traj
                self.actions.append(np.asarray(traj[1:]) - np.asarray(traj[:-1]))
            # print()

    def interpol_waypoints(self, interpole, waypoints=None):
        self.waypoints_idx = []
        if waypoints is None:
            waypoints = self.waypoints
        if interpole:

            interpol = []
            for i in range(waypoints.shape[0] - 1):
                direction = waypoints[i+1] - waypoints[i]
                norm = np.linalg.norm(direction)

                if self.real_world:
                    action_norm = (self.vel * self.action_scale)
                    # action = action / np.linalg.norm(action) * self.action_len
                else:
                    # scale as real world
                    norm_direction = direction / np.linalg.norm(direction)
                    norm_direction[0] *= self.scale_x
                    norm_direction[1] *= self.scale_y
                    # force rw scale to be of norm 0.3
                    rw_direction = norm_direction / np.linalg.norm(norm_direction) * (self.vel * self.action_scale)
                    # downscale to sim and compute the norm of the action sim
                    rw_direction[0] /= self.scale_x
                    rw_direction[1] /= self.scale_y
                    action_norm = np.linalg.norm(rw_direction)


                interpol_points = int(norm / action_norm)
                self.waypoints_idx.append(len(interpol))
                for t in range(interpol_points):
                    w = waypoints[i] + ((t / interpol_points)) * (waypoints[i + 1] - waypoints[i])
                    interpol.append(w)
            interpol.append(waypoints[-1])
            return interpol
        else:
            return waypoints

    def cubic_spline(self, traj_points=None):
        if traj_points is None:
            traj_points = self.traj_points
        t = np.arange(len(traj_points))
        x = [point[0] for point in traj_points]
        y = [point[1] for point in traj_points]
        z = [point[2] for point in traj_points]
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        t = np.linspace(0, len(traj_points) - 1, num=1000)
        trajectory_points = np.array([(cs_x(ti), cs_y(ti), cs_z(ti)) for ti in t])

        return trajectory_points

    def get_single_variation(self, pos, action=None):
        # pos = p.getMeshData(self.env.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.env.id)[1][
        #     self.vertices[0]]

        if action is None:
            distances = cdist([pos], self.traj_points)
            idx = np.argmin(distances)
            # lookahead
            if idx < len(self.traj_points) - 1:
                idx += 1

            desired_pos = self.traj_points[idx]

            e = desired_pos - pos
            kp = 1/(self.action_scale)

            action = kp*e
            action_norm = np.linalg.norm(action)

        e = np.random.normal(0, 1, self.action_dim)

        e = e / np.linalg.norm(e)*action_norm

        if self.constraint:
            while not self.cosine_similarity(e, action) > self.c_thresh or not self.distance(pos+e/kp) < self.d_thresh:
                e = np.random.normal(0, 1, self.action_dim)
                e = e / np.linalg.norm(e)*action_norm

        return e

    # def get_demo_action(self, data=None):
    #     if data is None:
    #         data = p.getMeshData(self.env.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.env.id)
    #     # gripper position
    #     pos = np.asarray(data[1][self.vertices[0]])
    #
    #     distances = cdist([pos], self.traj_points)
    #     idx = np.argmin(distances)
    #     # lookahead
    #     if idx < len(self.traj_points) - 1:
    #         idx += 1
    #
    #     desired_pos = self.traj_points[idx]
    #
    #     e = desired_pos - pos
    #     kp = 1 / (self.action_scale)
    #
    #     action = kp * e
    #
    #     return action
    def cosine_similarity(self, A, B):
        cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        return cosine

    def distance(self, point):
        distances = cdist([point], self.dense_traj_points)
        return np.min(distances)

    # def get_constrained_variation(self, action=None, num_variations=1, num_steps=1):
    #     data = p.getMeshData(self.env.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.env.id)
    #     # gripper position
    #     pos = np.asarray(data[1][self.vertices[0]])
    #
    #     if num_variations == 1:
    #         return self.get_single_variation(pos, action)
    #
    #     else:
    #         a = np.zeros((num_variations, num_steps, self.action_dim))
    #         # TODO: debug when there is more than 1 action to sample
    #         for s in range(num_steps):
    #             for i in range(num_variations):
    #                 if s == 0:
    #                     a[i, s, :] = self.get_single_variation(pos, action)
    #                 else:
    #                     a[i, s, :] = self.get_single_variation(pos + a[i, s-1, :], action)
    #
    #         return a


class TrajectorySimple():
    def __init__(self,
                 waypoints=None,
                 vel=0.03,
                 interpole=True,
                 action_scale=1.,):

        self.action_scale = action_scale

        self.vertices = [24, 624]

        self.action_dim = 3

        self.waypoints = waypoints
        self.vel = vel

        self.traj_points = self.interpol_waypoints(interpole)

    def interpol_waypoints(self, interpole, waypoints=None):
        self.waypoints_idx = []
        if waypoints is None:
            waypoints = self.waypoints
        if interpole:

            interpol = []
            for i in range(waypoints.shape[0] - 1):
                direction = waypoints[i+1] - waypoints[i]
                norm = np.linalg.norm(direction)
                action_norm = (self.vel * self.action_scale)

                interpol_points = int(norm / action_norm)
                self.waypoints_idx.append(len(interpol))
                for t in range(interpol_points):
                    w = waypoints[i] + ((t / interpol_points)) * (waypoints[i + 1] - waypoints[i])
                    interpol.append(w)
            interpol.append(waypoints[-1])
            return interpol
        else:
            return waypoints

    def generate_perturbed_trajectory(self, epsilon=0.1):
        """
        Generate a trajectory with random perturbations.
        The start and end points remain the same, but the path in between can vary.
        """
        perturbed_traj = [self.traj_points[0]]  # Start with the initial point
        for i in range(1, len(self.traj_points) - 1):
            if np.random.rand() < epsilon:
                # Introduce a random perturbation
                random_direction = np.random.randn(self.action_dim)
                random_direction /= np.linalg.norm(random_direction)  # Normalize
                random_perturbation = random_direction * (self.vel * self.action_scale)
                new_point = perturbed_traj[-1] + random_perturbation
            else:
                # Follow the original trajectory
                new_point = self.traj_points[i]

            perturbed_traj.append(new_point)

        perturbed_traj.append(self.traj_points[-1])  # Ensure the end point is the same
        return perturbed_traj



# if __name__ == '__main__':
#     # Test for Action Sampler
#     sampler = Action_Sampler(
#         N=25,              # trajectory length
#         action_len=0.02,
#         c_threshold=0.3,
#         pp_dir=np.asarray([-0.1, 0.1, 0.015]) - np.asarray([-0.1, -0.1, 0.015]),
#         starting_point=np.asarray([-0.1, -0.1, 0.015]),
#         sampling_mean=None)
#     t = []
#     for i in range(100):
#         t.append(np.asarray(sampler.sample_trajectory()))
#     plot_pcd_list(t, elev=90)
#     print()
#
#     # Test for Trajectory
#     args = get_argparse()
#
#     args.render = 1
#     args.K = 3
#
#     frame_skip = 2
#     action_mult = 1
#     env = normalize(HalfFoldEnv(frame_skip=frame_skip, hz=100, action_mult=action_mult, obs=args.obs, reward=args.reward))
#
#     velocity = 0.02
#     waypoints = np.asarray([[[0,0,0], [0.5,0,1], [1,0,1]],
#                             [[0,0,0], [0.5,0.5,1], [1,0,1]]])
#
#     controller = Trajectory(env=env,
#                             args=args,
#                             waypoints=waypoints,
#                             vel=velocity,
#                             interpole=True,
#                             multi_traj=True,
#                             action_scale=0.02,
#                             constraint=False)