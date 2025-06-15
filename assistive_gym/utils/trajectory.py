import numpy as np
import pybullet as p
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
# from assistive_gym.envs.half_folding_BO import HalfFoldEnv
from assistive_gym.utils.normalized_env import normalize
# from arguments import get_argparse
class Trajectory():
    def __init__(self,
                 waypoints,
                 vel=1.,
                 interpole=True,
                 multi_traj=False,
                 action_scale=1,):

        self.action_scale = action_scale

        self.vertices = [24, 624]

        self.action_dim = 3

        self.waypoints = waypoints
        self.vel = vel


        if not multi_traj:
            self.traj_points = self.interpol_waypoints(interpole)
            self.dense_traj_points = self.cubic_spline()
            self.actions = np.asarray(self.traj_points[1:]) - np.asarray(self.traj_points[:-1])
        else:
            self.traj_points = []
            self.lengths = []
            for waypoints in self.waypoints:
                traj = self.interpol_waypoints(interpole, waypoints)
                self.traj_points.append(traj)
                self.lengths.append(len(traj))

            # Pad with zero actions to get to the longest possible trajectory
            max_len = max(self.lengths)
            self.actions = []
            for i, traj in enumerate(self.traj_points):
                last_element = traj[-1]
                while len(traj) < max_len:
                    traj.append(last_element)
                self.traj_points[i] = traj
                self.actions.append(np.asarray(traj[1:]) - np.asarray(traj[:-1]))
            # print()

    def interpol_waypoints(self, interpole, waypoints=None):
        if waypoints is None:
            waypoints = self.waypoints
        if interpole:
            norm = np.linalg.norm(waypoints[1] - waypoints[0])
            interpol_points = int(norm / (self.vel * self.action_scale))

            interpol = []
            for i in range(waypoints.shape[0] - 1):
                for t in range(interpol_points):
                    w = waypoints[i] + ((t / interpol_points)) * (waypoints[i + 1] - waypoints[i])
                    interpol.append(w)
            interpol.append(waypoints[-1])
            return interpol
        else:
            return waypoints

    def cubic_spline(self):
        t = np.arange(len(self.traj_points))
        x = [point[0] for point in self.traj_points]
        y = [point[1] for point in self.traj_points]
        z = [point[2] for point in self.traj_points]
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        t = np.linspace(0, len(self.traj_points) - 1, num=1000)
        trajectory_points = np.array([(cs_x(ti), cs_y(ti), cs_z(ti)) for ti in t])

        return trajectory_points

    def get_single_variation(self, pos, action=None):

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
        else:
            action_norm = np.linalg.norm(action)

        e = np.random.normal(0, 1, self.action_dim)

        e = e / np.linalg.norm(e)*action_norm

        return e

    def cosine_similarity(self, A, B):
        cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        return cosine

    def distance(self, point):
        distances = cdist([point], self.dense_traj_points)
        return np.min(distances)

    def get_constrained_variation(self, gripper_pos, action=None, num_variations=1, num_steps=1):

        if num_variations == 1:
            return self.get_single_variation(gripper_pos, action)

        else:
            a = np.zeros((num_variations, num_steps, self.action_dim))
            # TODO: debug when there is more than 1 action to sample
            for s in range(num_steps):
                for i in range(num_variations):
                    if s == 0:
                        a[i, s, :] = self.get_single_variation(gripper_pos, action)
                    else:
                        a[i, s, :] = self.get_single_variation(gripper_pos + a[i, s-1, :], action)

            return a



# if __name__ == '__main__':
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
