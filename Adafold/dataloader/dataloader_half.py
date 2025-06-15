import copy
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import glob
import h5py
# import open3d as o3d
import shapely
import os
from Adafold.args.arguments import get_argparse
from Adafold.model.utils import rotate_pc
from Adafold.viz.viz_mpc import plot_vectors_3d
from Adafold.viz.viz_pcd import plot_pcd_list
import matplotlib.pyplot as plt
from Adafold.utils.utils_planning import normalized_wasserstein_distance

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
    def __init__(self, dataset_folders, args, train=True, downsample=True, downsampling="voxel", voxel_size=0.02):
        super(PointcloudDataset).__init__()

        self.data_folders = dataset_folders

        self.train = train
        self.aug = args.data_aug
        self.downsample = downsample
        self.downsampling = downsampling
        self.downsampling_scale = args.pcd_scale
        self.pcd_dim = args.pcd_dim
        self.multi_step = args.multi_step
        self.voxel_size = voxel_size

        self.obs = args.obs
        self.loss = args.loss
        self.data_aug = args.data_aug
        self.zero_c = args.zero_center
        # back camera adjustment
        self.transform = np.eye(4)

        self.num_past = args.K      # number of past observations
        self.num_pred = args.H      # number of future predcitions

        self.max_points = 0
        self.fixed_half = None

        # augmentation
        print(f'Loading dataset: {os.path.dirname(dataset_folders[0])}')
        self.pcd_noise = 0.001

        # Stats of param obtained from the data collection
        self.params_stats = {'params': {'mean': [70., 70.,  0.1,  3.,  0.5], 'std': [22.36, 22.36, 1., 1., 0.5]}}
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
        self.full_states = []
        self.pcds = []
        self.full_pcds = []
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
            param, full_states, pcd, full_pcd, grippers, means, action, done, waypoints = self.load_pcd(folder)

            self.params.append(param)  # elas, bend, damp, frict
            self.full_states.append(full_states)
            self.pcds.append(pcd)
            self.full_pcds.append(full_pcd)
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
                for t in range(len_traj - K - H):
                    # Clip the slide for the probing action as we want to predict intitial state with recent past
                    # But at the same time we want to predict intermediate state with only probing observations
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


    def load_pcd(self, path):
        # in this case the dataloader has been provided with the path to the sample and not to the trajectory
        # so it needs to iterate over the trajectories folders
        traj_samples = glob.glob(path + '/*')
        traj_samples.sort()
        if 'data_' in traj_samples[0]:
            traj_samples = [path]

        params = None
        trajs_full_states = []
        trajs_pcds = []
        trajs_full_pcds = []
        trajs_grippers = []
        trajs_actions = []
        trajs_dones = []
        trajs_means = []

        for traj_sample in traj_samples:
            samples = glob.glob(traj_sample + '/data*')
            samples.sort()

            full_states = []
            pcds = []
            full_pcds = []
            grippers = []
            actions = []
            dones = []
            means = []
            waypoints = []

            # This works only in simulation!
            # for the real world we need to perform some test and decide which is the best strategy
            half_mesh_indeces = None
            for s in samples:
                f = h5py.File(s, 'r')
                if params is None:
                    # [elas, bend, scale, frame_skip, side]
                    params = np.asarray(f.get('params'))
                    params = self.norm_transform(params, type='params')

                action = np.asarray(f.get('action'))
                mesh = np.asarray(f.get('pcd_pos'))
                if self.fixed_half is None:
                    self.fixed_half = mesh[np.where(mesh[:, 1] <= 0.)]
                if half_mesh_indeces is None:
                    half_mesh_indeces = np.where(mesh[:, 1] > 0.)
                half_mesh = mesh[half_mesh_indeces]

                if self.obs == 'mesh':
                    full_state = mesh
                    full_pcd_pos = half_mesh
                    pcd_pos = half_mesh

                if 'pcd' in self.obs:
                    front_pcd_pos = np.asarray(f.get('front_pcd'))
                    pcd_pos = self.filter_half_pointcloud(self, half_mesh, front_pcd_pos, D_max=0.005)

                    # add back observation
                    back_pcd_pos = np.asarray(f.get('back_pcd'))
                    half_back_pcd_pos = self.filter_half_pointcloud(self, half_mesh, back_pcd_pos, D_max=0.005)
                    full_state = self.merge_pcds(front_pcd_pos, back_pcd_pos, self.transform)
                    full_pcd_pos = self.merge_pcds(pcd_pos, half_back_pcd_pos, self.transform)

                if self.obs == 'full_pcd':
                    pcd_pos = full_pcd_pos

                gripper_pos = np.asarray(f.get('gripper_pos')) #- mean
                done = np.asarray(f.get('done')).tolist()

                if self.downsample and self.obs!= 'mesh':
                    downsampled_pcd_pos = self.downsample_pcd(pcd_pos, voxel_size=self.voxel_size)
                    downsampled_full_pcd_pos = self.downsample_pcd(full_pcd_pos, obs_type='pcd', voxel_size=self.voxel_size)

                    mean = 0
                    if self.zero_c:
                        mean = downsampled_pcd_pos.mean(axis=0)
                        downsampled_pcd_pos -= mean
                        downsampled_full_pcd_pos -= mean
                        gripper_pos -= mean
                    full_states.append(full_state)
                    pcds.append(downsampled_pcd_pos)
                    full_pcds.append(downsampled_full_pcd_pos)
                    grippers.append(gripper_pos)
                    if downsampled_pcd_pos.shape[0] > self.max_points:
                        self.max_points = downsampled_pcd_pos.shape[0]
                else:
                    mean = 0
                    if self.zero_c:
                        mean = pcd_pos.mean(axis=0)
                        pcd_pos -= mean
                        full_pcd_pos -= mean
                        gripper_pos -= mean
                    full_states.append(full_state)
                    pcds.append(pcd_pos)
                    full_pcds.append(full_pcd_pos)  # Full pcd is needed for a better chamfer loss
                    grippers.append(gripper_pos)
                    if pcd_pos.shape[0] > self.max_points:
                        self.max_points = pcd_pos.shape[0]

                actions.append(action)
                dones.append(done)
                means.append(mean)
                f.close()

                if done:
                    # Repeat the first state K times and add zero actions K times
                    for i in range(self.num_past):
                        full_states.insert(0, full_states[0])
                        pcds.insert(0, pcds[0])
                        full_pcds.insert(0, full_pcds[0])
                        grippers.insert(0, grippers[0])
                        actions.insert(0, np.zeros_like(actions[0]))
                        dones.insert(0, dones[0])
                        means.insert(0, means[0])

                    # Full trajectory
                    # if self.multi_step:
                    trajs_full_states.append(np.asarray(full_states))
                    trajs_pcds.append(np.asarray(pcds))
                    trajs_full_pcds.append(np.asarray(full_pcds))
                    trajs_grippers.append(np.asarray(grippers))
                    trajs_actions.append(np.asarray(actions))
                    trajs_dones.append(dones)
                    trajs_means.append(means)

                    full_states = []
                    pcds = []
                    full_pcds = []
                    grippers = []
                    actions = []
                    dones = []
                    means = []
                    waypoints = []
                    done = False


        return params, trajs_full_states, trajs_pcds, trajs_full_pcds, trajs_grippers, trajs_means, trajs_actions, trajs_dones, waypoints

    def filter_half_pointcloud(self, half_mesh, pointcloud, D_max=0.005):
        tree = cKDTree(half_mesh)
        filtered_set = []
        for point in pointcloud:
            # Find the distance to the nearest point in the first set
            distance, _ = tree.query(point)
            # Keep the point if the distance is less than or equal to D_max
            if distance <= D_max:
                filtered_set.append(point)
        return np.array(filtered_set)

    def merge_pcds(self, pcd1, pcd2, transform):
        # Convert to homogeneous coordinates
        num_points = pcd2.shape[0]
        points_homogeneous = np.ones((num_points, 4))
        points_homogeneous[:, :3] = pcd2

        # Apply transformation
        transformed_points_homogeneous = np.dot(transform, points_homogeneous.T).T

        # Convert back to 3D
        transformed_pcd2 = transformed_points_homogeneous[:, :3]

        merged_pcd = np.concatenate([pcd1, pcd2], 0)
        return merged_pcd


    def downsample_pcd(self, pcd, obs_type=None, voxel_size=0.02):
        if self.loss != 'MSE':
            downsampled_pcd = self.voxelize_downsample(pcd, voxel_size=voxel_size)
        else:
            if self.obs == 'mesh':
                num_nodes = pcd.shape[0]
                downsampled_pcd = pcd.reshape(int(np.sqrt(num_nodes)), int(np.sqrt(num_nodes)), 3)[::2, ::2, :].reshape((-1, 3))


        return downsampled_pcd


    def voxelize_downsample(self, point_cloud, voxel_size):
        """
        Downsample a point cloud using voxelization, mimicking Open3D's approach.

        Parameters:
        - point_cloud: A (N, 3) numpy array of points.
        - voxel_size: The size of each voxel.

        Returns:
        - A numpy array representing the downsampled point cloud.
        """

        # Compute voxel indices for each point
        voxel_indices = np.floor(point_cloud / voxel_size).astype(np.int64)

        # Convert voxel indices to unique strings to identify each voxel
        voxel_ids = ['_'.join(map(str, coord)) for coord in voxel_indices]

        # Use a dictionary to group points by their voxel identifier
        voxel_dict = {}
        for i, voxel_id in enumerate(voxel_ids):
            if voxel_id not in voxel_dict:
                voxel_dict[voxel_id] = []
            voxel_dict[voxel_id].append(point_cloud[i])

        # Compute the centroid for each voxel
        downsampled_points = np.array([np.mean(voxel_dict[voxel_id], axis=0) for voxel_id in voxel_dict])

        return downsampled_points


    def voxelize_to_grid(self, point_cloud, voxel_size):
        """
        Voxelize a point cloud.

        Parameters:
        - point_cloud: A (N, 3) numpy array of points.
        - voxel_size: The size of each voxel.

        Returns:
        - A 3D numpy array representing the voxelized point cloud.
        """

        # Determine the bounds of the point cloud
        min_bounds = point_cloud.min(axis=0)
        max_bounds = point_cloud.max(axis=0)

        # Compute the dimensions of the voxel grid
        grid_dim = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)

        # Initialize an empty voxel grid
        voxel_grid = np.zeros(grid_dim, dtype=np.uint8)

        # For each point, determine its voxel coordinates
        voxel_coords = ((point_cloud - min_bounds) / voxel_size).astype(int)

        # Set the corresponding voxel to 1
        for coord in voxel_coords:
            voxel_grid[tuple(coord)] = 1

        return voxel_grid


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

        x_pcds = copy.deepcopy(self.pcds[env][traj][range_x])
        x_grippers = copy.deepcopy(self.grippers[env][traj][range_x])
        x_actions = copy.deepcopy(self.actions[env][traj][range_x])

        y_pcds = copy.deepcopy(self.pcds[env][traj][range_y])
        # y_actions = copy.deepcopy(self.actions[env][traj][range_y])
        # y_grippers = copy.deepcopy(self.grippers[env][traj][range_y])

        # plot_vectors_3d(action=x_actions, points=x_pcds[0], predicted_points=y_pcds[0], gripper_pos=x_grippers[0])
        # plt.show()

        if self.data_aug and self.train:
            set_pcds, params = self.augment_pcd([copy.deepcopy(z_pcds),
                                                 copy.deepcopy(z_grippers),
                                                 copy.deepcopy(z_actions),
                                                 copy.deepcopy(x_pcds),
                                                 copy.deepcopy(x_grippers),
                                                 copy.deepcopy(x_actions),
                                                 copy.deepcopy(y_pcds),
                                                 # copy.deepcopy(y_grippers),
                                                 # copy.deepcopy(y_actions)
                                                 ],
                                                params=params,
                                                augmentations=['noise', 'scale', 'rot', 't'],
                                                # augmentations=['rot', 't'],
                                                # augmentations=['noise',],
                                                types=['pc', 'gripper', 'action',
                                                       'pc', 'gripper', 'action', 'pc', #'gripper', 'action'
                                                       ])

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


        batch_z = Batch.from_data_list(datas)

        # Load data for the first forward pass

        datas = []

        x_pcd_pos = torch.from_numpy(x_pcds[0]).float()
        for i in range(len(x_pcds)):
            x_gripper_pos = torch.from_numpy(x_grippers[i]).float()
            a = torch.from_numpy(x_actions[i]).float()
            grip_action = torch.cat([x_gripper_pos, a], dim=-1).repeat(x_pcd_pos.shape[0], 1)

            x_pcd_feat = torch.cat([x_pcd_pos, grip_action], dim=-1)
            y_pcd_pos = torch.from_numpy(y_pcds[i]).float()
            datas.append(Data(x=x_pcd_feat, pos=x_pcd_pos, y=y_pcd_pos))

        batch_forward = datas

        return [batch_z, batch_forward, params]

    def get_sample_trajectory(self, env, traj):
        # avoid augmentations
        train = copy.deepcopy(self.train)
        self.train = False

        traj_z, traj_forward, traj_params = [], [], []
        for idx in range(len(self.look_up)):
            if self.look_up[idx]['env'] == env and self.look_up[idx]['traj'] == traj:
                batch_z, batch_forward, params = self._get_datapoint(
                    idx)
                traj_z.append(batch_z)
                # traj_forward.append(batch_forward)
                batch_forward = Batch.from_data_list(batch_forward)
                traj_forward.append(batch_forward)
                traj_params.append(params)

        self.train = train

        return [traj_z, traj_forward, traj_params]


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
            scale_low = 0.8
            scale_high = 1.2
            # scale_low = 0.1
            # scale_high = 0.1
            s_x = np.random.uniform(scale_low, scale_high)
            s_y = np.random.uniform(scale_low, scale_high)
            for i, pcds in enumerate(set_pcds):
                for pcd in pcds:
                    if len(pcd.shape) > 1:
                        pcd[:, 0] *= (s_x)
                        pcd[:, 1] *= (s_y)
                    else:
                        pcd[0] *= (s_x)
                        pcd[1] *= (s_y)

            params[2] *= (1.1 + s_y)         # Folding direction, but we should consider also the other one

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

        return set_pcds, params


    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
         return self.num_datapoints


if __name__ == '__main__':
    # Test function for the dataloader
    path = '../data/datasets/oct31_1s_train/env*'
    paths = glob.glob(path)
    # go inside first folder
    paths = glob.glob(os.path.join(paths[0], '*'))
    paths.sort()

    args = get_argparse()
    args.obs = 'mesh'       # [mesh, pcd, full_pcd]
    dataset = PointcloudDataset(paths[:5], args=args, downsample=False, downsampling="voxel")
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
    # gripper position
    plot_vectors_3d(points=b[0].x[b[0].batch == 4], predicted_points=b[0].x[b[0].batch == 4][:, 3:])
    plt.show()
    # next state
    plot_vectors_3d(points=b[0].x[b[0].batch == 2], predicted_points=b[0].x[b[0].batch == 3])
    plt.show()
    # action
    plot_vectors_3d(points=b[0].x[b[0].batch == 4], predicted_points=b[0].x[b[0].batch == 4][:, 3:], action=[(b[0].x[b[0].batch == 4][0, 6:]).numpy()])
    plt.show()
    import torch.nn as nn
    loss = nn.MSELoss()
    loss(b[0].pos[b[0].batch == 0], b[0].pos[b[0].batch == 1])
    print()
    # Work on handling the list of batches
