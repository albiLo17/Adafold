import pdb

import numpy as np
from Adafold.dataset_collection.utils import reset_env, save_datapoint
from assistive_gym.envs.half_folding_former import HalfFoldEnv
import pybullet as p
import os
# from folding.utils import Cameras
from Adafold.utils.utils_folding import Cameras, make_dir
import imageio
import multiprocessing
import matplotlib.pyplot as plt
from Adafold.trajectory.trajectory import Trajectory, Action_Sampler_Simple

from torch_geometric.data import Data, Batch
from Adafold.utils.utils_planning import filter_half_pointcloud, compute_iou, normalized_wasserstein_distance
from Adafold.model.model import RMA_MB
from Adafold.trajectory.trajectory import Action_Sampler_Simple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="'(type|1)type'")

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

import torch
import copy
import time
from Adafold.viz.viz_mpc import *

def print_cuda_memory():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()

    print(f"Memory Allocated: {allocated / 1024**3:.2f} GB")
    print(f"Memory Cached: {cached / 1024**3:.2f} GB")

def model_memory_usage(model,):
    # Push the model to GPU
    model.to('cuda')

    # Calculate Parameters Size
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()

    # Calculate Buffers Size
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Calculate Total Size
    total_size = param_size + buffer_size

    # Convert to Megabytes
    total_size_MB = total_size / (1024 ** 2)

    print(f"Parameter number: {num_params}")
    print(f"Model Parameters Memory: {param_size / (1024 ** 2):.2f} MB")
    print(f"Model Buffers Memory: {buffer_size / (1024 ** 2):.2f} MB")
    print(f"Total Model Memory: {total_size_MB:.2f} MB")

class MPC():

    def __init__(self,
                 args,
                 A=100,
                 H=1,
                 mod='mpc',         # ['mpc', 'rand', 'fixed']
                 env_idx=0,
                 load_model=True,
                 model=None,
                 dir_model=None,
                 dir_results=None,
                 save_datapoints=False,
                 traj_type='triangular',
                 downsample=False,
                 save_pcd=False,
                 cumulative=False,
                 verbose=False,
                 w_cumul=1,
                 path_dataset=None,
                 env_params=[40, 60, 0.1, 4, 0],
                 cost_coefficients=[1., 1., 0.01, 1.],
                 gripper_attractor_tr=0.06,
                 smooth_threshold=0.8,
                 device='cpu'):

        self.device = device
        self.args = args

        self.A = A          # number of candidate actions
        self.K = args.K     # N_SAMPLES
        self.H = H          # Prediction horizon
        self.sampler = None     # Action_Sampler_Simple
        self.reward_env_type = 'HalfIoU'
        self.reward_type = args.reward# 'HalfIoU'
        self.traj_type = traj_type      # ['inward', 'rect', 'triangular'] shape of the fixed trajectory
        self.terminate_reward = 0.98        # it's folded well enough!
        self.cumulative = cumulative
        self.w_cumul = w_cumul
        self.cost_coefficients = cost_coefficients
        self.gripper_attractor_tr = gripper_attractor_tr
        self.smooth_threshold = smooth_threshold

        self.z_offset_pick = 0.01
        self.velocity = 0.03

        self.env_params = env_params
        self.obs_type = args.obs
        self.downsample = downsample
        self.grid_res = 0.01        # Needed to compute the occupancy grid for the IoU

        self.mod = mod
        self.model = None
        self.dyn_conditioning = args.dyn_conditioning

        if ('mpc' in mod or 'mppi' in mod) and 'sim' not in mod:
            if model is None:
                self.load_model(dir_model=dir_model)
            else:
                self.model = model

        print(f"Cost: {self.reward_type}, cumulative:{self.cumulative} and {self.w_cumul}, coeff: {self.cost_coefficients}")

        # save params
        self.save_datapoints = save_datapoints
        self.data_dict = None
        self.path_dataset = path_dataset
        self.save_pcd = False
        if self.save_datapoints and self.path_dataset is not None:
            idx = "{:06}".format(env_idx)
            self.path_dataset = f'{self.path_dataset}/env_{idx}'
            make_dir(self.path_dataset)

        self.dir_results = dir_results
        self.env_idx = env_idx
        self.save_gif = args.save_gif
        self.gif_path = args.gif_path
        self.verbose = verbose
        self.frames = []

    def update_dict(self, obs=None, keys=None):
        " Updated but not tested"
        if self.data_dict is None:
            self.data_dict = {'pcd_pos': [], 'back_pcd': [], 'front_pcd': [], 'gripper_pos': [], 'past_pcd_pos': [],
                              'past_back_pcd': [], 'past_front_pcd': [], 'past_gripper_pos': [],
                              'params': self.env_params, 'action': [], 'done': [], 'pick': []}

        if keys is not None:
            for i, k in enumerate(keys):
                self.data_dict[k] = obs[i]


    def load_model(self, dir_model):
        " Updated "
        " Function to load the model. Can be used at initialization or to load finetuned models"
        if self.model is None:
            self.model = RMA_MB(self.args, self.device).to(self.device)

        self.model.load_dict(model_path=dir_model + '/full_dict_model_val.pt', load_full=True, device=self.device)
        print(f"Model loaded from {dir_model}")
        self.model.eval()

    def process_obs(self, mesh, gripper_pos, pcd_obs=None, type='mesh'):
        " Updated "
        mesh = mesh.reshape(-1, 3)
        half_mesh = mesh[self.half_mesh_indeces]
        if type == 'mesh':
            obs = half_mesh
            # No downsampling for this guy

        if 'pcd' in type and pcd_obs is not None:
            obs = filter_half_pointcloud(half_mesh, pcd_obs, D_max=0.005)
            if self.downsample:
                print("NOT IMPLEMENTED YET")

        obs = torch.from_numpy(obs).float().to(self.device) # add batch and num_obs (B, K, Nodes, 3)
        gripper_pos = torch.from_numpy(gripper_pos).float().to(self.device)

        return obs, gripper_pos

    def reset_env(self, elas, bend, scale):
        " Updated "
        self.damp, self.frict = 1.5, 1.50
        self.raw_obs = self.env.reset(stiffness=[elas, bend, self.damp], friction=self.frict, cloth_scale=scale, cloth_mass=0.5)  # Elas, bend, damp

        self.env.camera = Cameras(id=self.env.id)
        self.env.camera.setup_camera(camera_eye=[0.60, 0., 0.17], camera_target=[0., 0., 0.17], camera_width=720,
                                camera_height=720)
        self.env.camera.setup_camera(camera_eye=[-0.15, 0.2, 0.17], camera_target=[0., 0., 0.17], camera_width=720,
                                camera_height=720)

        self.env.setup_camera(camera_eye=[0., 0., 0.65], camera_target=[0., 0.0, 0.], camera_width=720,
                         camera_height=720)

        self.camera_params = self.env.camera_params

    def reset(self,):
        " Updated "
        elas, bend, scale, frame_skip, side = self.env_params
        elas = int(elas)
        bend = int(bend)
        frame_skip = int(frame_skip)
        action_mult = 1
        self.env = HalfFoldEnv(frame_skip=frame_skip,
                               hz=100,
                               action_mult=action_mult,
                               obs=self.args.obs,
                               side=side,
                               gripper_attractor_tr=self.gripper_attractor_tr,
                               grid_res=self.grid_res,
                               reward=self.reward_env_type) # the reward of the env is only the alignment!

        if self.args.render:
            self.env.render(width=640, height=480)
        self.reset_env(elas, bend, scale)

        # Define shape goal and mesh indeces to process observations
        self.fixed_half = self.env.fixed_half
        self.half_mesh_indeces = self.env.moving_half_idx

        # Initialize observations
        self.raw_gripper_pos = self._get_gripper()
        if self.obs_type == 'mesh':
            self.obs, self.gripper_pos = self.process_obs(self.raw_obs, gripper_pos=self.raw_gripper_pos, type='mesh')
        self.raw_seg_front_pcd, self.raw_seg_back_pcd = None, None
        if self.obs_type == 'full_pcd' or self.save_pcd:
            self.raw_seg_front_pcd, self.raw_seg_back_pcd = self.env.process_pointclouds(RGB_threshold=[0.78, 0.78, 0.78], voxel_size=0.005)
            self.obs, self.gripper_pos = self.process_obs(self.raw_obs, pcd_obs=np.concatenate([self.raw_seg_front_pcd, self.raw_seg_back_pcd], 0), gripper_pos=self.raw_gripper_pos, type='full_pcd')

        self.init_state = copy.deepcopy(self.raw_obs.reshape(-1, 3 ))
        self.past_a = None
        self.reward = 0
        self.ref_reward = 0
        self.done = False

        self.candidate_actions = None

        self.update_dict()

    def get_pick_place(self, ):
        " Updated "
        pick_pos = self.env.get_corners()[0]
        place_pos = self.env.get_corners()[1]

        return pick_pos, place_pos

    def _get_cost(self, data, gripper_pos, next_action=None):
        " Updated, but should be debugged"
        l = len(data)
        cost = torch.zeros(l)
        next_pos = None

        if 'IoU' in self.reward_type:
            w1 = self.cost_coefficients[0]
            # TODO: to be better parallelized
            for i in range(l):
                set1 = self.fixed_half
                set2 = data[i]
                iou = compute_iou(set1, set2, grid_size=self.grid_res)

                cost[i] -= w1*iou     # as max value is when there is perfect matching

        if 'Gr' in self.reward_type and l > 1:
            w2 = self.cost_coefficients[1]
            # attractor for the gripper when close to the place
            g_2Ddist = np.linalg.norm((self.raw_gripper_pos[:2] - self.x_place[:2]))
            if g_2Ddist < self.gripper_attractor_tr:
                if next_pos is None:
                    next_pos = torch.from_numpy(copy.deepcopy(self.candidate_actions[:, 0]))
                    next_pos += gripper_pos
                g_dist = torch.norm(next_pos - self.x_place, dim=1)
                cost += w2*g_dist

        if 'out' in self.reward_type:
            # compute cosine similarity b
            if self.candidate_actions is not None and l > 1:
                if next_pos is None:
                    next_pos = torch.from_numpy(copy.deepcopy(self.candidate_actions[:, 0])).to(self.device)
                    next_pos += gripper_pos
                    
                gripper_dir = (next_pos - torch.from_numpy(self.x_pick).to(self.device))
                # Compute the cross product for the batch
                repeated_ppdir = torch.from_numpy(self.pp_dir).unsqueeze(0).repeat(gripper_dir.shape[0], 1).to(self.device)

                scalar_products = gripper_dir[:, 0] * repeated_ppdir[:, 1] - gripper_dir[:, 1] * repeated_ppdir[:, 0]

                # Determine the relative position, positive sign means that we are outside the boudaries
                signs = torch.sign(scalar_products).cpu()
                w4 = self.cost_coefficients[3]
                outside_penalty = w4 * (signs + 1)/2
                cost += outside_penalty
                # print()


        if 'EMD' in self.reward_type:
            for i in range(l):
                set1 = self.fixed_half[:, :2]
                set2 = data[i][:, :2]
                emd = normalized_wasserstein_distance(set1, set2, d_max=0.05)
                cost[i] -= emd

        if 'smooth' in self.reward_type:
            w3 = self.cost_coefficients[2]
            cos_threshold = 0.3#self.gripper_attractor_tr
            # if past action is not none, then compute the cosine similarity between the current action and the past one
            if self.past_a is not None:
                cs = torch.cosine_similarity(torch.Tensor(self.past_a), torch.Tensor(next_action), dim=-1)
                # if cosine similarity is below threshold, then add cost as (1 - cosine similarity)
                if l > 1:
                    cost[cs < cos_threshold] += w3*(1 - cs[cs < cos_threshold])
                elif cs < cos_threshold:
                    cost += w3*(1 - cs)


        return cost

    def pick(self):
        "Updated"
        self.env.pick(self.x_pick)
        # start from the same position the rw starts after grasping (or assuming pregrasped)
        z_offset = self.z_offset_pick
        action = np.zeros_like(self.x_pick)
        action[-1] += z_offset
        tau_pcd, tau_x, tau_a = self.execute_action(action)

        return tau_pcd, tau_x, tau_a

    def update_history(self, tau_pcd=None, tau_x=None, tau_a=None, action=None):
        # If tau_x is None, it is the first iteration so we need to repeat it for K times
        if tau_x is None:
            tau_pcd = []
            tau_x = []
            tau_a = []
            # plus two as at the end of the function we remove the first observation to mimic a sliding window
            for i in range(self.args.K + 2):
                tau_pcd.append(self.obs)
                tau_x.append(self.gripper_pos)
                # set zero as action as it is the first observation at all
                tau_a.append(torch.zeros(3).float().to(self.device))
        else:
            tau_pcd.append(self.obs)
            tau_x.append(self.gripper_pos)
            tau_a.append(torch.from_numpy(action).float().to(self.device))

        return tau_pcd[1:], tau_x[1:], tau_a[1:]


    def execute_action(self, action, tau_pcd=None, tau_x=None, tau_a=None):
        "Updated"
        self.raw_obs, self.ref_reward, self.done, self.info = self.env.step(action=action)
        self.raw_gripper_pos = self._get_gripper()
        # mesh = self.raw_obs.reshape(-1, 3)
        # half_mesh = mesh[self.half_mesh_indeces]
        # plot_pcd_list([half_mesh, self.env._get_obs().reshape(-1, 3)[self.half_mesh_indeces]], alpha_value=0.3)

        if self.obs_type == 'mesh':
            self.obs, self.gripper_pos = self.process_obs(mesh=copy.deepcopy(self.raw_obs),
                                                          gripper_pos=copy.deepcopy(self.raw_gripper_pos),
                                                          type='mesh')

        if self.obs_type == 'full_pcd':
            self.raw_seg_front_pcd , self.raw_seg_back_pcd = self.env.process_pointclouds(RGB_threshold=[0.78, 0.78, 0.78],
                                                                                          voxel_size=0.005)
            self.obs, self.gripper_pos = self.process_obs(mesh=copy.deepcopy(self.raw_obs),
                                                          pcd_obs=np.concatenate([copy.deepcopy(self.raw_seg_front_pcd), copy.deepcopy(self.raw_seg_back_pcd)], 0),
                                                          gripper_pos=copy.deepcopy(self.raw_gripper_pos),
                                                          type='full_pcd')

        self.reward = -self._get_cost(copy.deepcopy(self.obs).unsqueeze(0).cpu().numpy(), self.raw_gripper_pos, next_action=action)[0].item()

        tau_pcd, tau_x, tau_a = self.update_history(tau_pcd=tau_pcd, tau_x=tau_x, tau_a=tau_a, action=action)
        return tau_pcd, tau_x, tau_a

    def track_frames(self,):
        " Updated "
        _, _, img, _, _ = p.getCameraImage(width=640, height=480,
                                           viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                                               cameraTargetPosition=[0.0, 0., 0.0], distance=0.5, yaw=-0,
                                               pitch=-45, roll=0, upAxisIndex=2),
                                           projectionMatrix=p.computeProjectionMatrixFOV(fov=60,
                                                                                         aspect=float(640) / 480,
                                                                                         nearVal=0.01, farVal=100.0))

        self.frames.append(np.asarray(img).reshape((480, 640, 4))[:, :, :].astype(np.uint8))


    def save_data(self, steps, action, done=False, pick_action=False, place_action=False):
        "   Updated but to be finished"
        if self.save_datapoints:
            # This is the first one and when we start recording the observations
            if pick_action:
                self.update_dict(obs=[self.raw_gripper_pos, self.raw_gripper_pos, self.raw_obs, self.raw_obs,
                                      self.raw_seg_front_pcd, self.raw_seg_front_pcd, self.raw_seg_back_pcd, self.raw_seg_back_pcd,
                                      1, action, done],
                                 keys=['gripper_pos', 'past_gripper_pos', 'pcd_pos', 'past_pcd_pos',
                                       'front_pcd', 'past_front_pcd', 'back_pcd', 'past_back_pcd',
                                       'pick', 'action', 'done'])
            elif place_action:
                self.update_dict(obs=[self.raw_gripper_pos, self.raw_obs, self.raw_seg_front_pcd, self.raw_seg_back_pcd,
                                      0, action, done],
                                 keys=['gripper_pos', 'pcd_pos',
                                       'front_pcd', 'back_pcd',
                                       'pick', 'action', 'done'])
                save_datapoint(self.data_dict, steps, self.path_dataset, save_pcd=self.save_pcd)
            else:
                self.update_dict(obs=[self.raw_gripper_pos, self.raw_obs, self.raw_seg_front_pcd, self.raw_seg_back_pcd,
                                      1, action, done],
                                 keys=['gripper_pos', 'pcd_pos',
                                       'front_pcd', 'back_pcd',
                                       'pick', 'action', 'done'])
                save_datapoint(self.data_dict, steps, self.path_dataset, save_pcd=self.save_pcd)

    def initialize_fixed_traj(self):
        # generate fixed trajectory
        gripper = self._get_gripper()
        if 'triangular' in self.traj_type:
            mid_w = (self.x_place + gripper) / 2
            mid_w[2] = 0.08
            waypoints = np.asarray([gripper, mid_w, self.x_place])
        elif 'rect' in self.traj_type: # straight line
            final = self.x_place
            final[-1] = gripper[-1]     # same height of pick
            mid_w = (final + gripper) / 2
            waypoints = np.asarray([gripper, mid_w, final])
        elif 'inward' in self.traj_type: # triangular inward
            mid_w = (self.x_place + gripper) / 2
            mid_w[0] = 0.
            mid_w[2] = 0.08
            waypoints = np.asarray([gripper, mid_w, self.x_place])

        self.planner_fixed = Trajectory(args=self.args,
                                  waypoints=waypoints,
                                  vel=self.velocity,
                                  interpole=True,
                                  action_scale=1,
                                  constraint=False)

        self.fixed_traj = np.asarray(self.planner_fixed.traj_points)[1:] - np.asarray(self.planner_fixed.traj_points[:-1])

    def _get_gripper(self):
        return self.env.sphere_ee.get_base_pos_orient()[0]

    def init_candidates(self,):
        " Updated but not debugged "
        gripper = self._get_gripper()
        self.initialize_fixed_traj()

        # generate sampler for candidate trajectory
        self.sampler =Action_Sampler_Simple(
            N=12,              # trajectory length
            action_len=self.velocity,
            c_threshold=0.,
            grid_size=0.04,
            pp_dir=self.x_place - self.x_pick,
            place=self.x_place,
            starting_point=self.x_pick,
            sampling_mean=None,
            fixed_trajectory=None)


        self.trajectories = np.asarray(self.sampler.generate_dataset(num_trajectories=self.A, starting_point=gripper, target_point=self.x_place, prob=False))
        self.candidate_actions_traj = self.trajectories[:, 1:] - self.trajectories[:, :-1]

        # Debug plots
        if self.verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for t in self.trajectories:
                ax.scatter(np.asarray(self.planner_fixed.traj_points)[:, 0], np.asarray(self.planner_fixed.traj_points)[:, 1], np.asarray(self.planner_fixed.traj_points)[:, 2])
            plt.title("Demo trajectory")
            plt.show()


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for t in self.trajectories:
                ax.scatter(t[:, 0], t[:, 1], t[:, 2])
            plt.title("Sampling distribution")
            plt.show()

    def update_candidates(self):
        if self.sampler.N > 0:
            self.sampler.N -= 1

        self.trajectories = np.asarray(
            self.sampler.generate_dataset(num_trajectories=self.A, starting_point=self.raw_gripper_pos, target_point=self.x_place,
                                          prob=False))
        self.candidate_actions_traj = self.trajectories[:, 1:] - self.trajectories[:, :-1]

        # consider the actions already performed plus the remaining ones (need a -1???)
        self.len_traj = len(self.actions) + self.trajectories.shape[1]


        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for t in self.trajectories:
        #     ax.scatter(t[:, 0], t[:, 1], t[:, 2])
        # plt.title("Sampling distribution")
        # plt.show()


    def get_z(self, tau_pcd, tau_x, tau_a):
        " Updated, not debugged but shouldn't be needed. The processing must be like the dataloader"
        # get z only for MPC predictions
        if self.dyn_conditioning != 0 and ('mpc' in self.mod or 'mppi' in self.mod or 'CQL' in self.mod) and 'sim' not in self.mod:
            pi = self.env_params
            datas = []
            for i in range(len(tau_pcd)):
                pcd_pos_t = tau_pcd[i]
                # pcd_pos_t1 = tau_pcd[i+1]
                gripper_pos = tau_x[i]
                a = tau_a[i]
                if i == len(tau_pcd) - 1:
                    a = torch.zeros_like(a).float().to(a.device)
                grip_action = torch.cat([gripper_pos, a], dim=-1).repeat(pcd_pos_t.shape[0], 1)
                pcd_feat = torch.cat([pcd_pos_t, grip_action], dim=-1)
                datas.append(Data(x=pcd_feat, pos=pcd_pos_t))

            batch_probe = Batch.from_data_list(datas)

            z, _ = self.model.get_z(pi, batch_probe, test_phase=True)
            return z

        return None

    @torch.no_grad()
    def _unroll_model(self, all_costs):
        " Updated function "
        num_actions = self.candidate_actions.shape[1]
        init_states = self.obs.repeat((self.A, 1, 1))
        gripper_poses = self.gripper_pos.repeat((self.A, 1))

        tot_cost = torch.zeros((self.A))
        # print_cuda_memory()

        z = None
        if num_actions == 0:
            print()
        if self.z is not None:
            z = self.z.repeat((self.A, 1))

        for t in range(num_actions):
            # select action of interest
            future_a_numpy = self.candidate_actions[:, t, :]
            future_a = torch.from_numpy(future_a_numpy).float().to(self.device)

            x_pos = gripper_poses
            grip_action = torch.cat([x_pos, future_a], dim=-1).unsqueeze(1).repeat(1, init_states.shape[1], 1)
            x_pcd_feat = torch.cat([init_states, grip_action], dim=-1)

            # we need to iterate to create the list of Data datapoints for the Batch class
            datas = [Data(x=x_pcd_feat[i], pos=init_states[i], y=init_states[i]) for i in range(len(init_states))]

            batch_forward = Batch.from_data_list(datas)
            batch = [[], batch_forward, []]
            # print_cuda_memory()
            with torch.no_grad():
                pcd_pred, _, _, _ = self.model(batch, z=z, get_adapt=False)
                # updated init state with the current prediction
                init_states = torch.cat([pcd_pred[batch_forward.batch == i].unsqueeze(0) for i in range(batch_forward.batch.max() + 1)], dim=0)                # update gripper pose and remove mean
                gripper_poses += future_a

                final_preds = [copy.deepcopy(pcd_pred[batch_forward.batch == i]).cpu().numpy() for i in
                               range(batch_forward.batch.max() + 1)]
                if self.cumulative:
                    tot_cost += self.w_cumul**(num_actions - 1 - t) * self._get_cost(final_preds, gripper_poses[0], next_action=self.candidate_actions[:, 0])
            # print_cuda_memory()

        if not self.cumulative:
            tot_cost = self._get_cost(final_preds, gripper_poses[0], next_action=self.candidate_actions[:, 0])  # [M, A]
        all_costs += tot_cost.unsqueeze(-1).numpy()
        return final_preds


    def _compute_total_cost(self, step):
        " Updated function "

        shared_all_cost = multiprocessing.Array(np.ctypeslib.ctypes.c_double, self.A * 1)
        shared_all_cost = np.ctypeslib.as_array(shared_all_cost.get_obj())
        shared_all_cost = shared_all_cost.reshape(self.A, 1)


        model_predictions = self._unroll_model(shared_all_cost)
        self.cost_total = shared_all_cost
        return model_predictions


    def get_best_action(self, iteration=None, step=None):
        " Updated but not debugged"
        start_time = time.time()

        # This is for the plot later on
        obs = self.obs.clone()
        if obs.is_cuda:
            obs = obs.cpu()


        num_actions = min(self.H, self.len_traj - step)
        self.candidate_actions = self.candidate_actions_traj[:][:, :num_actions]
        
        if 'mpc' in self.mod:
            # in open loop evaluate all trajectrory
            if 'ol' in self.mod:
                self.candidate_actions = self.candidate_actions_traj
                model_predictions = self._compute_total_cost(step=step)
            else:
                model_predictions = self._compute_total_cost(step=step)

            action_index = np.argmin(self.cost_total)
            action_sequence = self.candidate_actions[action_index]
            self.final_model_prediction = model_predictions[action_index]

            action_fig = plot_action_selection(vectors=self.candidate_actions, action=action_sequence,
                points=obs, full_points=self.raw_obs.reshape(-1,3), start_point=self.raw_gripper_pos)
            plt.savefig(self.dir_results + f'/action_selection_t{step}.png')
            plt.clf()

            action_fig = plot_action_selection_with_values(
                vectors=self.candidate_actions,
                action=action_sequence,
                points=obs.detach().cpu().numpy(), full_points=self.raw_obs.reshape(-1, 3),
                start_point=self.raw_gripper_pos
                , Q_values=self.cost_total)
            plt.savefig(self.dir_results + f'/values_action_selection_t{step}.png')
            plt.clf()




            fig = plot_vectors_3d(
                action=action_sequence,
                points=obs,
                predicted_points=self.final_model_prediction,
                gripper_pos=self.raw_gripper_pos)
            plt.savefig(self.dir_results + f'/model_prediction_t{step}.png')
            plt.clf()

            # TODO: uncomment for debug
            # if iteration is not None:
            #     print(f'Iteration {iteration}, step {step} - {self.device} - time={time.time() - start_time}')
            #     print(f'action_index - {action_index}, cost - {self.cost_total[action_index]}')

            final_actions = action_sequence


        else:

            if self.mod == 'Rand':
                action_index = np.random.randint(0, self.A)
                action_sequence = self.candidate_actions[action_index]

                action_fig = plot_action_selection(vectors=self.candidate_actions, action=action_sequence,
                                                   points=obs, full_points=self.raw_obs.reshape(-1,3), start_point=self.raw_gripper_pos)
                plt.savefig(self.dir_results + '/action_selection.png')
                plt.clf()
                final_actions = action_sequence

            if self.mod == 'NoAda':
                action_sequence = [self.fixed_traj[step]]

                action_fig = plot_action_selection(vectors=self.candidate_actions, action=action_sequence,
                                                   points=obs, full_points=self.raw_obs.reshape(-1,3), start_point=self.raw_gripper_pos)
                plt.savefig(self.dir_results + '/action_selection.png')
                plt.clf()
                final_actions = action_sequence
                action_index = 0


        return final_actions, action_index


    def control(self, iterations=10000, params=None):

        for iteration in range(iterations):
            # sample params
            if params is not None:
                self.env_params = params


            print(f'evaluation env: elas{self.env_params[0]}, bend{self.env_params[1]}, side{self.env_params[4]}, frame{self.env_params[3]}')
            self.reset()

            self.x_pick, self.x_place = self.get_pick_place()
            self.pp_dir = self.x_place - self.x_pick

            tau_pcd, tau_x, tau_a = self.pick()
            self.save_data(steps=0, action=np.zeros(3), done=False, pick_action=True, place_action=False)
            # track frames of only the first interaction with the environment
            if iteration == 0:
                self.track_frames()

            self.init_candidates()
            self.len_traj = self.candidate_actions_traj.shape[1]        # TODO: set properly the lenght of the trajectory
            if self.mod == 'NoAda':
                self.len_traj = self.fixed_traj.shape[0]

            # self.len_traj = 2
            self.actions = []
            states = [self.obs]
            rewards = []
            rewards_ref = []
            t = 0
            done = False
            ol_actions = None       # filled once planned for the first time
            while t < self.len_traj and not done:# REMOVED but might be addedd and self.sampler.action_means.shape[0] > 0:

            # while t < self.sampler.action_means.shape[0] and not done:
                self.z = None
                self.z = self.get_z(tau_pcd, tau_x, tau_a)

                # select best action, enters the if every time if not ol and only once if ol
                if ol_actions is None:
                    final_actions, action_index = self.get_best_action(iteration, step=t)
                    
                # execute best action
                if 'ol' in self.mod:
                    if ol_actions is None:
                        ol_actions = final_actions

                    if t < len(ol_actions):
                        best_action = ol_actions[t]
                    else:
                        best_action = np.zeros(3)
                else:
                    best_action = final_actions[0]
                    if np.linalg.norm(best_action) > 0.0302:
                        print()

                self.actions.append(best_action)
                tau_pcd, tau_x, tau_a = self.execute_action(best_action, tau_pcd, tau_x, tau_a)
                self.past_a = best_action
                self.save_data(steps=t, action=best_action, done=done, pick_action=False, place_action=False)
                states.append(self.obs)
                rewards.append(self.reward)
                rewards_ref.append(self.ref_reward)

                self.update_candidates()

                # termination condition
                if np.linalg.norm(self.raw_gripper_pos - self.x_place) < 0.03:
                    done = True

                # termination condition when the gripper is close to the
                if self.ref_reward >= self.terminate_reward:
                    self.reward += 1.
                    done = True

                t += 1

                if iteration == 0:
                    self.track_frames()

            self.env.place()
            self.execute_action(np.zeros(3), tau_pcd, tau_x, tau_a)
            self.save_data(steps=t, action=np.zeros(3), done=True, pick_action=False, place_action=True)

            if iteration == 0:
                self.track_frames()

            if 'mpc' in self.mod or 'mppi' in self.mod:
                if 'DDPG' not in self.mod:
                    self.cost_total[:] = 0

            obs = self.obs.clone()
            full_state = self.raw_obs
            if obs.is_cuda:
                obs = obs.cpu()

            if 'mpc' in self.mod or 'mppi' in self.mod:
                fig = plot_prediction_and_state(
                    points=obs,
                    predicted_points=self.final_model_prediction,
                    title=None)

                plt.savefig(self.dir_results + '/finalVSpred.png')
                plt.clf()

                # info = f'Predicted cost = {pred_cost}, real cost = {r}'
                # np.savetxt(self.dit_results + '/info.txt', info)

                # text_file = open(self.dir_results + '/info.txt', "w")
                # n = text_file.write(f'Predicted cost = {pred_cost}, real cost = {r}\n')
                # n = text_file.write(f'Candidates waypoints mean cost = {mean_cost}, std cost = {std_cost}\n')
                # text_file.close()
            else:
                fig = plot_prediction_and_state(
                    points=obs,
                    predicted_points=None,
                    title=None)

                plt.savefig(self.dir_results + '/final.png')
                plt.clf()

            fig = plot_state_singleview(
                points=obs, full_points=full_state.reshape(-1, 3), view='top')

            plt.savefig(self.dir_results + '/final_Top_view.png')
            plt.clf()

            # plot final planned trajectory
            action_fig = plot_action_selection(vectors=None, action=np.asarray(self.actions),
                points=obs, full_points=self.init_state, start_point=self.x_pick)
            plt.savefig(self.dir_results + f'/planned_traj.png')
            plt.clf()

            states.append(self.obs)
            rewards.append(self.reward)
            rewards_ref.append(self.ref_reward)

            # save states and rewards and actions in self.dir_results
            # check if state is tensor or numpy
            if states[0].is_cuda:
                np.save(self.dir_results + f'/states_{iteration}.npy', np.stack([state.cpu().numpy() for state in states]))
            else:
                np.save(self.dir_results + f'/states_{iteration}.npy', np.stack([state.numpy() for state in states]))
            np.save(self.dir_results + f'/rewards_{iteration}.npy', np.asarray(rewards))
            np.save(self.dir_results + f'/rewards_ref_{iteration}.npy', np.asarray(rewards_ref))
            np.save(self.dir_results + f'/actions_{iteration}.npy', np.asarray(self.actions))



            print(f'End iteration {iteration} with final reward ({self.reward_type}): {self.reward} - reference reward: {self.ref_reward}')
            if self.save_gif:
                gif_path = self.dir_results + f'/fold_elas{self.env_params[0]}_bend{self.env_params[1]}_scale{self.env_params[-1]}.gif'
                imageio.mimsave(gif_path, self.frames, fps=3, subrectangles=True)


        r = self.reward
        self.env._get_reward()

        r_ref = self.ref_reward
        p.disconnect()

        return rewards, r, self.frames

if __name__=='__main__':
    from Adafold.args.arguments import get_argparse
    from Adafold.model.model import RMA_MB
    # from rollout_experiments import evaluation_params
    import random
    from Adafold.utils.load_configs import get_configs
    from Adafold.utils.setup_model import setup_model

    args = get_argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and torch.cuda.get_device_name(torch.cuda.current_device()) != 'NVIDIA GeForce GTX 1080' and not args.train_cluster:
        torch.cuda.set_device(1)

    args.render = 1
    args.obs = 'mesh'       # [mesh, full_pcd]
    args.loss = 'MSE'
    args.reward = 'IoU'       # ['IoU', IoU_Gr]
    args.lr = 0.0001

    dir_models = {
        'mpc_NoAda_100': f'../data/trained_models/D=100_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=0_fusion=0',
        'mpc_Ada_f0_100': f'../data/trained_models/D=100_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=0',
        'mpc_Ada_f1_100': f'../data/trained_models/D=100_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=1',
        'mpc_NoAda_500': f'../data/trained_models/D=500_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=0_fusion=0',
        'mpc_Ada_f0_500': f'../data/trained_models/D=500_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=0',
        'mpc_Ada_f1_500': f'../data/trained_models/D=500_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=1',
        'mpc_NoAda_1000': f'../data/trained_models/D=1000_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=0_fusion=0',
    }

    conditionings = {'mpc_NoAda_100': 0, 'mpc_Ada_f0_100':3, 'mpc_Ada_f1_100':3, 'mpc_NoAda_500': 0, 'mpc_NoAda_1000': 0, 'mpc_Ada_f0_500':3, 'mpc_Ada_f1_500':3}
    fusions = {'mpc_NoAda_100': 0, 'mpc_Ada_f0_100':0, 'mpc_Ada_f1_100':1, 'mpc_NoAda_500': 0, 'mpc_NoAda_1000': 0, 'mpc_Ada_f0_500':0, 'mpc_Ada_f1_500':1}
    all_models = ['mpc_NoAda_100', 'mpc_Ada_f0_100', 'mpc_Ada_f1_100', 'mpc_NoAda_500', 'mpc_NoAda_1000', 'mpc_Ada_f0_500', 'mpc_Ada_f1_500']
    exp_name = '0211'

    dir_results = './tmp'

    save_data = True
    dataset_path = None
    if save_data:
        # TODO: iterate multiple time for each environment!
        # create new folder on top of the already existing dataset
        dataset_path = f'./data/bootstrap_0'
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    mod = 'mpc_RMA'
    mod = 'mpc'
    model = 'mpc_Ada_f1_500'
    args.dyn_conditioning = conditionings[model]
    args.fusion = fusions[model]
    A = 100
    H = 2
    args.reward = 'IoU_Gr'  # ['IoU', IoU_Gr]
    planner = MPC(
        args,
        A=A,
        H=H,
        mod='mpc',  # ['mpc', 'rand', 'fixed']
        env_idx=0,
        load_model=True,
        dir_model=dir_models[model],
        dir_results=dir_results,
        save_datapoints=save_data,
        downsample=False,
        cumulative=True,
        save_pcd=True,
        verbose=True,
        path_dataset=dataset_path,
        env_params=[40, 60, 0.1, 4, 0],
        device=device
    )

    cumulative_reward, final_reward, frames = planner.control(iterations=1)
    plt.plot(cumulative_reward)
    plt.title(f"Model: {model}, H= {H}")
    plt.tight_layout()
    plt.show()

    print()
