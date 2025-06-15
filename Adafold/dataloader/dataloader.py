import copy

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import glob
import h5py
# import open3d as o3d
import os
import random
from Adafold.args.arguments import get_argparse
from Adafold.model.utils import rotate_pc
from Adafold.viz.viz_mpc import plot_vectors_3d
import matplotlib.pyplot as plt


def plot_pcd(pcd, pcd2=None, elev=30, azim=-180):
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(projection='3d', elev=elev, azim=azim)
    # rotate by 90, invert
    img = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2])
    if pcd2 is not None:
        img = ax.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(0, 0.2)


    plt.show()
    
class PointcloudDataset(Dataset):

    "Version of the dataset that hanldes pointlcoud of different sizes"
    def __init__(self, dataset_folders, args, train=True, downsample=True, downsampling="scale"):
        super(PointcloudDataset).__init__()

        self.data_folders = dataset_folders

        self.train = train
        self.aug = args.data_aug
        self.downsample = downsample
        self.downsampling = downsampling
        self.downsampling_scale = args.pcd_scale
        self.pcd_dim = args.pcd_dim
        self.multi_step = args.multi_step

        self.obs = args.obs
        self.data_aug = args.data_aug
        self.zero_c = args.zero_center

        # TODO: integrate this in a proper way
        self.pred_probe = True

        self.num_past = args.K      # number of past observations
        self.num_pred = args.H      # number of future predcitions

        self.max_points = 0
        self.fixed_half = None

        # augmentation
        print(f'Loading dataset: {os.path.dirname(dataset_folders[0])}')
        self.pcd_noise = 0.001



        # Stats of param obtained from the data collection
        # TODO: update parameters statistics
        self.params_stats = {'params': {'mean': [50, 50, 0., 3, 0.5], 'std': [22.36, 22.36, 1., 1., 0.5]}}
        self.normalize = True

        self.look_up = []
        self.data = self.load_data()
        self.update_info()

        # print()
    def norm_transform(self, x, type='params'):        # available types: 'actions', 'params', 'pcds'
        if self.normalize:
            normalized = (copy.deepcopy(x) - self.params_stats[type]['mean']) / self.params_stats[type]['std']
            return normalized
        else:
            return x

    def update_info(self):
        self.num_datapoints = len(self.look_up)
        self.num_samples = len(self.data_folders)
        self.pcd_dim = self.max_points

    def denormalize(self, x, type='pcds'):  # available types: 'actions', 'params', 'pcds'
        if self.normalize == 1:
            return copy.deepcopy(x) * self.dataset_stats[type]['std'] + self.dataset_stats[type]['mean']
        return x

    def load_data(self):
        '''
        This function initialize the buffer and load the dataset contained in
        the specified folders
        :return:
        :rtype:
        '''
        self.params = []
        self.pcds = []
        self.pcds_means = []
        self.grippers = []
        self.actions = []
        self.dones = []
        self.waypoints = []
        self.env = 0

        folders = self.data_folders
        if not isinstance(self.data_folders, list):
            folders = [self.data_folders]

        for folder in tqdm(folders):
            self.upload_traj(folder)

    def upload_traj(self, folder):
        try:
            param, pcd, grippers, means, action, done, waypoints = self.load_pcd(folder)
            # TODO: store flow
            self.params.append(param)  # elas, bend, damp, frict
            self.pcds.append(pcd)
            self.pcds_means.append(means)  # Needed for plot in case we want to invert the zero centering
            self.grippers.append(grippers)
            self.actions.append(action)
            self.dones.append(done)
            self.waypoints.append(waypoints)

            # Divide the dataset by K that is the length of the past prediction
            for traj in range(len(pcd)):
                len_traj = pcd[traj].shape[0]
                K = self.num_past
                H = self.num_pred

                # The best option would probably to have probe slice (0, K+1) and to skip the last element (used only for prediction)
                # TODO: remove -1?
                for t in range(len_traj - K - 1 - H):
                    # Clip the slide for the probing action as we want to predict intitial state with recent past
                    # But at the same time we want to predict intermediate state with only probing observations
                    # TODO: debug waypoint
                    # slide = np.clip(t, 0, K)
                    self.look_up.append({'env': self.env,
                                         'traj': traj,
                                         'z': slice(t, K + t + 1, 1),  # +1 as we want to include also the element K+t
                                         'x': slice(K + t, K + t + self.num_pred, 1),
                                         # start, end, step, +1 needed to include the stop index
                                         'y': slice(K + 1 + t, K + 1 + t + self.num_pred, 1),
                                         'waypoint': self.env}
                                        )

            self.env += 1

        except RuntimeError as e:
            print(f'Folder unstable: {folder}')


    def downsample_pcd(self, pcd):
        if self.downsampling == 'random':
            indices = random.sample(range(pcd.shape[0]), self.pcd_dim)
            downsampled_pcd = pcd[indices, :]

        elif self.downsampling == 'scale':
            if self.obs == 'mesh':
                num_nodes = pcd.shape[0]
                downsampled_pcd = pcd.reshape(int(np.sqrt(num_nodes)), int(np.sqrt(num_nodes)), 3)[::2, ::2, :].reshape((-1, 3))
            else:
                # This corresponds more or less to a voxelization of 0.02 like in the real world
                # pc = o3d.geometry.PointCloud()
                # pc.points = o3d.utility.Vector3dVector(pcd)
                # downsampled_pcd = voxelize_pcd(pc, voxel_size=0.02)
                downsampled_pcd = pcd[::5, :]
        # elif self.downsampling == 'voxel':
            # This corresponds more or less to a voxelization of 0.02 like in the real world
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(pcd)
            # downsampled_pcd = voxelize_pcd(pc, voxel_size=0.02)


        return downsampled_pcd


    def load_pcd(self, path):
        samples = glob.glob(path + '/data*')
        samples.sort()

        params = None
        trajs_pcds = []
        trajs_grippers = []
        trajs_actions = []
        trajs_dones = []
        trajs_means = []
        trajs_flow = []

        pcds = []
        grippers = []
        actions = []
        dones = []
        means = []
        waypoints = []
        flows = []

        first_done = True

        for s in samples:
            f = h5py.File(s, 'r')
            if params is None:
                # [elas, bend, scale, frame_skip, side]
                params = np.asarray(f.get('params'))
                params = self.norm_transform(params, type='params')

            action = np.asarray(f.get('action'))
            if self.obs == 'mesh':
                pcd_pos = np.asarray(f.get('pcd_pos'))
            if self.obs == 'pcd':
                pcd_pos = np.asarray(f.get('part_pcd'))
            if self.obs == 'full_pcd':
                pcd_pos = np.asarray(f.get('full_pcd'))
            # mean = pcd_pos.mean(axis=0)
            # Zero center the pcd
            # pcd_pos -= mean
            # gripper_pos = pcd_pos[24, :]
            gripper_pos = np.asarray(f.get('gripper_pos')) #- mean
            done = np.asarray(f.get('done')).tolist()

            waypoint_idx = np.asarray(f.get('waypoint_idx'))

            if self.downsample:
                downsampled_pcd_pos = self.downsample_pcd(pcd_pos)
                # append gripper position as last in order to append to it the action
                # downsampled_pcd_pos = np.concatenate((downsampled_pcd_pos, np.expand_dims(gripper_pos, 0)), axis=0)
                mean = 0
                if self.zero_c:
                    mean = downsampled_pcd_pos.mean(axis=0)
                    downsampled_pcd_pos -= mean
                    gripper_pos -= mean
                pcds.append(downsampled_pcd_pos)
                grippers.append(gripper_pos)
                if downsampled_pcd_pos.shape[0] > self.max_points:
                    self.max_points = downsampled_pcd_pos.shape[0]
            else:
            # not downsampled pcd
            #     pcd_pos = np.concatenate((pcd_pos, np.expand_dims(gripper_pos, 0)), axis=0)
                mean = 0
                if self.zero_c:
                    mean = pcd_pos.mean(axis=0)
                    pcd_pos -= mean
                    gripper_pos -= mean
                pcds.append(pcd_pos)
                grippers.append(gripper_pos)
                if pcd_pos.shape[0] > self.max_points:
                    self.max_points = pcd_pos.shape[0]

            actions.append(action)
            dones.append(done)
            means.append(mean)
            f.close()

            if done:
                # # TODO: compute the flow of each particle
                # TODO: use this part later to learn longer horizon predictions
                # reduced_pcds, reduced_actions, reduced_dones, reduced_means, waypoint = self.get_reduced_traj(pcds, actions,
                #                                                                                     dones, means,
                #                                                                                     idx_mid=waypoint_idx)

                # Repeat the first state K times and add zero actions K times
                for i in range(self.num_past):
                    pcds.insert(0, pcds[0])
                    grippers.insert(0, grippers[0])
                    actions.insert(0, np.zeros_like(actions[0]))
                    dones.insert(0, dones[0])
                    means.insert(0, means[0])
                #
                #     reduced_pcds.insert(0, reduced_pcds[0])
                #     reduced_actions.insert(0, np.zeros_like(reduced_actions[0]))
                #     reduced_dones.insert(0, reduced_dones[0])
                #     reduced_means.insert(0, reduced_means[0])

                # Full trajectory
                # if self.multi_step:
                trajs_pcds.append(np.asarray(pcds))
                trajs_grippers.append(np.asarray(grippers))
                trajs_actions.append(np.asarray(actions))
                trajs_dones.append(dones)
                trajs_means.append(means)

                # trajs_pcds.append(np.asarray(reduced_pcds))
                # trajs_actions.append(np.asarray(reduced_actions))
                # trajs_dones.append(reduced_dones)
                # trajs_means.append(reduced_means)
                # waypoints.append(waypoint)

                pcds = []
                grippers = []
                actions = []
                dones = []
                means = []
                waypoints = []
                done = False


        return params, trajs_pcds, trajs_grippers, trajs_means, trajs_actions, trajs_dones, waypoints


    def get_reduced_traj(self, pcds, actions, dones, means, idx_mid=None):
        a = np.asarray(actions)
        K = self.num_past
        temp_pcds = pcds[:K]
        temp_actions = actions[:K]
        temp_dones = dones[:K]
        temp_means = means[:K]

        # Probe to mid
        after_probe = a[K:, :]
        if idx_mid is None:
            try:
                # this shoudl work as far as the first action goes up
                idx_mid = np.where((-1 * after_probe + after_probe[0]) > 0.00001)[0][0]
            except:
                print()
        # we want the voxel position representing the waypoint
        waypoint = pcds[K + idx_mid][-1, :]

        temp_actions.append(after_probe[:idx_mid, :].sum(0))
        temp_pcds.append(pcds[K])
        temp_dones.append(dones[K])
        temp_means.append(means[K])

        # Mid to end
        temp_actions.append(after_probe[idx_mid:, :].sum(0))
        temp_pcds.append(pcds[K + idx_mid])
        temp_dones.append(dones[K + idx_mid])
        temp_means.append(means[K + idx_mid])

        # End
        temp_actions.append(actions[-1])
        temp_pcds.append(pcds[-1])
        temp_dones.append(dones[-1])
        temp_means.append(means[-1])

        # fig = plt.figure()
        # points = temp_pcds[5]
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', alpha=0.2, s=30, label='State')
        # ax.set_xlim([-0.15, 0.15])
        # ax.set_ylim([-0.15, 0.15])
        # ax.set_zlim([-0.1, 0.2])
        # plt.show()

        # for i in range(len(pcds)):
            # fig = plt.figure()
            # points = pcds[i]
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', alpha=0.2, s=30, label='State')
            # ax.set_xlim([-0.15, 0.15])
            # ax.set_ylim([-0.15, 0.15])
            # ax.set_zlim([-0.1, 0.2])
            # plt.title(f'{i}')
            # plt.show()

        return temp_pcds, temp_actions, temp_dones, temp_means, waypoint


    # def visualize_points(self, pos, edge_index=None, index=None):
    #     fig = plt.figure(figsize=(4, 4))
    #     if edge_index is not None:
    #         for (src, dst) in edge_index.t().tolist():
    #             src = pos[src].tolist()
    #             dst = pos[dst].tolist()
    #             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    #     if index is None:
    #         plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    #     else:
    #         mask = torch.zeros(pos.size(0), dtype=torch.bool)
    #         mask[index] = True
    #         plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
    #         plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    #     plt.axis('off')
    #     plt.show()

    def _get_datapoint(self, idx):
        env = self.look_up[idx]['env']
        traj = self.look_up[idx]['traj']
        range_z = self.look_up[idx]['z']
        range_x = self.look_up[idx]['x']
        range_y = self.look_up[idx]['y']

        # (p_t-past, ..., p_t-1, p_t, a_t-past, ..., a_t-1)

        # try:
        params = copy.deepcopy(torch.from_numpy(self.params[env]).float())
        z_pcds = copy.deepcopy(self.pcds[env][traj][range_z])
        z_grippers = copy.deepcopy(self.grippers[env][traj][range_z])
        z_actions = copy.deepcopy(self.actions[env][traj][range_z])

        x_pcds = copy.deepcopy(self.pcds[env][traj][range_x])[0:1]
        x_grippers = copy.deepcopy(self.grippers[env][traj][range_x])[0:1]
        x_actions = copy.deepcopy(self.actions[env][traj][range_x])     # TODO: concatenate along last dim?
        y_pcds = copy.deepcopy(self.pcds[env][traj][range_y])[-1:]

        # plot_vectors_3d(action=x_actions, points=x_pcds[0], predicted_points=y_pcds[0], gripper_pos=x_grippers[0])
        # plt.show()

        if self.data_aug and self.train:
            set_pcds, params = self.augment_pcd([copy.deepcopy(z_pcds),
                                                 copy.deepcopy(z_grippers),
                                                 copy.deepcopy(z_actions),
                                                 copy.deepcopy(x_pcds),
                                                 copy.deepcopy(x_grippers),
                                                 copy.deepcopy(x_actions),
                                                 copy.deepcopy(y_pcds)],
                                                params=params,
                                                augmentations=['noise', 'scale', 'rot', 't'],
                                                # augmentations=['noise',],
                                                types=['pc', 'gripper', 'action',
                                                       'pc', 'gripper', 'action', 'pc'])

            z_pcds, z_grippers, z_actions, x_pcds, x_grippers, x_actions, y_pcds = set_pcds

        # the order of the features is pt (3), pt1(3), gripper (3), action(3)
        datas = []
        for i in range(len(z_pcds)):
            if i < len(z_pcds):
                pcd_pos_t = torch.from_numpy(z_pcds[i]).float()
                # pcd_pos_t1 = torch.from_numpy(z_pcds[i+1]).float()
                gripper_pos = torch.from_numpy(z_grippers[i]).float()
                # pad pcdt, pcdt+1, gripper pos and action
                a = torch.from_numpy(z_actions[i]).float()
                if i == len(z_pcds) - 1:
                    a = torch.zeros_like(a).float()
                grip_action = torch.cat([gripper_pos, a], dim=-1).repeat(pcd_pos_t.shape[0], 1)
                # a[-1, :] = torch.from_numpy(z_actions[i]).float()  # append actions to the gripper
                # pcd_feat = torch.cat([pcd_pos_t, pcd_pos_t1, grip_action], dim=-1)
                pcd_feat = torch.cat([pcd_pos_t, grip_action], dim=-1)
                datas.append(Data(x=pcd_feat, pos=pcd_pos_t))

            # else:
            #     pcd_pos = torch.from_numpy(z_pcds[i]).float()
            #     # pad pcd and actions
            #     a = torch.zeros_like(pcd_pos)
            #     pcd_feat = torch.cat([pcd_pos, a], dim=-1)
            #     datas.append(Data(x=pcd_feat, pos=pcd_pos))

        batch_z = Batch.from_data_list(datas)

        datas = []
        # No need for multiple pointclouds as input as we only give the one at time t even if we predict t+H
        # for i in range(len(x_pcds)):
        x_pcd_pos = torch.from_numpy(x_pcds[0]).float()
        x_gripper_pos = torch.from_numpy(x_grippers[0]).float()
        # pad pcd and actions
        # TODO: concatenate all the actions
        a = torch.cat([torch.from_numpy(x_actions[i]).float() for i in range(len(x_actions))], dim=-1)
        grip_action = torch.cat([x_gripper_pos, a], dim=-1).repeat(x_pcd_pos.shape[0], 1) # append actions to the gripper
        x_pcd_feat = torch.cat([x_pcd_pos, grip_action], dim=-1)
        y_pcd_pos = torch.from_numpy(y_pcds[0]).float()

        datas.append(Data(x=x_pcd_feat, pos=x_pcd_pos, y=y_pcd_pos))

        batch_forward = Batch.from_data_list(datas)
        # except:
        #     print("not able to compute the datapoint")

        return [batch_z, batch_forward, params]

    def get_sample_trajecotry(self, env, traj):
        traj_z, traj_forward, traj_params = [], [], []
        for idx in range(len(self.look_up)):
            if self.look_up[idx]['env'] == env and self.look_up[idx]['traj'] == traj:
                batch_z, batch_forward, params = self._get_datapoint(
                    idx)
                traj_z.append(batch_z)
                traj_forward.append(batch_forward)
                traj_params.append(params)

                # if probe_pcd is None:
                #     probe_pcd = probe_pcd_idx.unsqueeze(0)
                #     probe_a = probe_a_idx.unsqueeze(0)
                #     x_pcd = x_pcd_idx.unsqueeze(0)
                #     x_action = x_action_idx.unsqueeze(0)
                #     y_pcd = y_pcd_idx.unsqueeze(0)
                #     params = params_idx.unsqueeze(0)
                # else :
                #     probe_pcd = torch.cat((probe_pcd ,probe_pcd_idx.unsqueeze(0)), 0)
                #     probe_a = torch.cat((probe_a ,probe_a_idx.unsqueeze(0)), 0)
                #     x_pcd = torch.cat((x_pcd ,x_pcd_idx.unsqueeze(0)), 0)
                #     x_action = torch.cat((x_action, x_action_idx.unsqueeze(0)), 0)
                #     y_pcd = torch.cat(( y_pcd,y_pcd_idx.unsqueeze(0)), 0)
                #     params = torch.cat(( params,params_idx.unsqueeze(0)), 0)

        return [traj_z, traj_forward, traj_params]

    # TODO: fix reshaping of the scale and remove rescaling of z
    def augment_pcd(self, set_pcds,
                    params,
                    augmentations=['noise', 'scale', 'rot', 't'],
                    types = []
                    ):
        """

        :param pcd: Numpy tensor, zero centered pointcloud
        :param augmentation: list containing the types of augmentation to perform
        :return: Numpy tensor of augmented pointcloud
        """
        if 'noise' in augmentations:
            noise_var = self.pcd_noise
            for i, pcds in enumerate(set_pcds):
                for pcd in pcds:
                    pcd += noise_var * (torch.randn(pcd.shape)).numpy()

        if 'scale' in augmentations:
            # fixed augmentation to make it as big as the real world: np.asarray([0.85, 1.1, 1]
            # scale_low = 0.8
            # scale_high = 1.2
            scale_low = 0.1
            scale_high = 0.1
            s_x = np.random.uniform(scale_low, scale_high)
            s_y = np.random.uniform(scale_low, scale_high)
            for i, pcds in enumerate(set_pcds):
                for pcd in pcds:
                    if len(pcd.shape) > 1:
                        pcd[:, 0] *= (0.85 + s_x)
                        pcd[:, 1] *= (1.1 + s_y)
                    else:
                        pcd[0] *= (0.85 + s_x)
                        pcd[1] *= (1.1 + s_y)

            params[2] *= (1.1 +s_y)         # Folding direction, but we should consider also the other one

            # TODO: add scaling of the scale param.

        if 't' in augmentations:
            t_x = np.random.uniform(-0.05, -0.05)
            t_y = np.random.uniform(-0.05,  -0.05)
            t_z = np.random.uniform(-0.01, 0.01)
            for i, pcds in enumerate(set_pcds):
                if types[i] != 'action':
                    for j, pcd in enumerate(pcds):
                        pcd += np.asarray([t_x, t_y, t_z])
                        # if i < len(set_pcds) - 1:
                        #     set_pcds[i+1][j] -= pcd.mean(0)

        if 'rot' in augmentations:
            # At the moment allows for rotations of at most 180 degrees
            angles = np.random.uniform(-np.pi*10/180, np.pi*10/180, size=3)
            # angles = np.random.uniform(-np.pi, np.pi, size=3)
            for pcds in set_pcds:
                for pcd in pcds:
                    if len(pcd.shape) > 1:
                        pcd += rotate_pc(pcd, angles=angles, device=None, return_rot=False)
                    else:
                        pcd += rotate_pc(pcd, angles=angles, device=None, return_rot=False)

        # Zero center again
        if self.zero_c:
            for i, pcds in enumerate(set_pcds):
                mean = 0
                if types[i] != 'action' and types[i] != 'gripper':
                    for j, pcd in enumerate(pcds):
                        pcd -= pcd.mean(0)
                        if i < len(set_pcds) - 1:
                            set_pcds[i+1][j] -= pcd.mean(0)

        return set_pcds, params


    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
         return self.num_datapoints

class ClassifierDataset(PointcloudDataset):
    def __init__(self, dataset_folders, args, train=True, downsample=True, downsampling="scale", mlp=False):
        super(ClassifierDataset, self).__init__(dataset_folders, args, train=train, downsample=downsample, downsampling=downsampling)

        self.mlp = args.mlp
        self.rnn = args.rnn

    def _get_datapoint(self, idx):

        env = self.look_up[idx]['env']
        traj = self.look_up[idx]['traj']
        range_z = self.look_up[idx]['z']
        range_x = self.look_up[idx]['x']

        # (p_t-past, ..., p_t-1, p_t, a_t-past, ..., a_t-1)

        # try:
        params = torch.from_numpy(self.params[env]).float()
        x_pcds = copy.deepcopy(self.pcds[env][traj][range_x])
        z_pcds = copy.deepcopy(self.pcds[env][traj][range_z])

        # if self.data_aug and self.train:
        #     z_pcds, z_grippers, z_actions, x_pcds, x_grippers, x_actions, y_pcds = self.augment_pcd([copy.deepcopy(z_pcds),
        #                                                                                              copy.deepcopy(z_grippers),
        #                                                                                              copy.deepcopy(z_actions),
        #                                                                                              copy.deepcopy(x_pcds),
        #                                                                                              copy.deepcopy(x_grippers),
        #                                                                                              copy.deepcopy(x_actions),
        #                                                                                              copy.deepcopy(y_pcds)],
        #                                                                                             augmentations=['noise', 'scale','rot'],
        #                                                                                             # augmentations=['noise',],
        #                                                                                             types=['pc', 'gripper', 'action',
        #                                                                                                    'pc', 'gripper', 'action', 'pc'])
        if self.mlp:
            pcd = torch.cat([torch.from_numpy(z_pcds[i].flatten()).float() for i in range(len(z_pcds))])
            return [[], pcd, params]

        if self.rnn:
            pcd = torch.from_numpy(z_pcds.reshape(z_pcds.shape[0], -1)).float()
            return [[], pcd, params]

        else:

            batch_z = []
            datas = []
            for i in range(len(z_pcds)):
                # x_pcd_feat = torch.cat([torch.from_numpy(z_pcds[0]).float(),
                #                         torch.from_numpy(z_pcds[1]).float()],
                #                        dim=-1)
                x_pcd_feat = torch.from_numpy(z_pcds[i]).float()
                x_pcd_pos = torch.from_numpy(x_pcds[0]).float()
                datas.append(Data(x=x_pcd_feat, pos=x_pcd_pos))
            # Debug plot
            # plot_pcd(x_pcd_feat, x_pcd_pos)

            batch_forward = Batch.from_data_list(datas)
            # except:
            #     print("not able to compute the datapoint")

            return [batch_z, batch_forward, params]


if __name__ == '__main__':
    # Test function for the dataloader
    path = '../data/datasets/june16_1s_train/env*'
    paths = glob.glob(path)
    paths.sort()

    args = get_argparse()
    args.obs = 'pcd'       # [mesh, pcd, full_pcd]
    dataset = PointcloudDataset(paths[:5], args=args, downsample=False, downsampling="scale")
    # dataset = PointcloudDatasetInv(paths[:5], args=args)
    # dataset = PCDDataset(paths[:10], args=args)
    datapoint = dataset._get_datapoint(3)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    b = next(iter(dataloader))
    #################### VISUALIZATIONS ###########
    # show the trajectory for the sample id
    id = 2
    plot_vectors_3d(action=dataset.actions[id][0], points=dataset.pcds[id][0][0], gripper_pos=dataset.grippers[id][0][0])
    plt.show()


    # visualize pointclouds, action and gripper position
    plot_pcd(b[0].x[b[0].batch == 0])
    # next state
    plot_vectors_3d(points=b[0].x[b[0].batch == 4], predicted_points=b[0].x[b[0].batch == 4][:, 3:])
    plt.show()
    # gripper position
    plot_vectors_3d(points=b[0].x[b[0].batch == 3], predicted_points=b[0].x[b[0].batch == 3][:, 6:])
    plt.show()
    # action
    plot_vectors_3d(points=b[0].x[b[0].batch == 4], predicted_points=b[0].x[b[0].batch == 4][:, 3:], action=[(b[0].x[b[0].batch == 4][0, 9:]).numpy()])
    plt.show()
    import torch.nn as nn
    loss = nn.MSELoss()
    loss(b[0].pos[b[0].batch == 0], b[0].pos[b[0].batch == 1])
    print()
    # Work on handling the list of batches
