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
import os
from Adafold.args.arguments import get_argparse
from Adafold.model.utils import rotate_pc
from Adafold.viz.viz_mpc import plot_vectors_3d
from Adafold.viz.viz_pcd import plot_pcd_list
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from Adafold.utils.utils_planning import measure_area, filter_half_pointcloud, compute_iou


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


class PointcloudAEDataset(Dataset):

    "Version of the dataset that hanldes pointlcoud of different sizes"
    def __init__(self, dataset_folders, args, train=True, downsample=True, downsampling="voxel", voxel_size=0.02):
        super(PointcloudAEDataset).__init__()

        self.data_folders = dataset_folders

        self.train = train
        self.aug = args.data_aug
        self.downsample = downsample
        self.downsampling = downsampling
        self.downsampling_scale = args.pcd_scale
        self.pcd_dim = args.pcd_dim
        self.voxel_size = voxel_size

        self.obs = args.obs
        self.data_aug = args.data_aug
        self.zero_c = args.zero_center
        # back camera adjustment
        self.transform = np.eye(4)

        self.grid_res = 0.01

        self.max_points = 0
        self.fixed_half = None
        self.num_classes = 2
        self.categories = ["bottom half", "top half"]


        # augmentation
        print(f'Loading dataset: {os.path.dirname(dataset_folders[0])}')
        self.pcd_noise = 0.001

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
        self.full_states = []
        self.pcds = []
        self.full_pcds = []
        self.front_pcds = []
        self.grippers = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.nn_masks = []
        self.env = 0

        folders = self.data_folders
        if not isinstance(self.data_folders, list):
            folders = [self.data_folders]

        for folder in tqdm(folders):
            self.upload_traj(folder)

    def upload_traj(self, folder):
        try:
            full_states, pcd, full_pcd, front_pcd, grippers, action, reward, mask, nn_mask = self.load_pcd(folder)

            self.full_states.append(full_states)
            self.pcds.append(pcd)
            self.full_pcds.append(full_pcd)
            self.front_pcds.append(front_pcd)
            self.grippers.append(grippers)
            self.actions.append(action)
            self.rewards.append(reward)
            self.masks.append(mask)
            self.nn_masks.append(nn_mask)

            # Divide the dataset by K that is the length of the past prediction
            for traj in range(len(pcd)):
                len_traj = pcd[traj].shape[0]

                # The best option would probably to have probe slice (0, K+1) and to skip the last element (used only for prediction)
                for t in range(len_traj):
                    # Clip the slide for the probing action as we want to predict intitial state with recent past
                    # But at the same time we want to predict intermediate state with only probing observations
                    # slide = np.clip(t, 0, K)
                    self.look_up.append({'env': self.env,
                                         'traj': traj,
                                         'x': slice(t, t + 1, 1),
                                         'x-1': slice(max(t-1, 0), max(t, 1), 1),}
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
        trajs_front_pcds = []
        trajs_grippers = []
        trajs_actions = []
        trajs_rewards = []
        trajs_masks = []
        trajs_nn_masks = []

        for traj_sample in traj_samples:
            samples = glob.glob(traj_sample + '/data*')
            samples.sort()

            full_states = []
            pcds = []
            full_pcds = []
            front_pcds = []
            grippers = []
            actions = []
            rewards = []
            masks = []
            nn_masks = []


            # This works only in simulation!
            # for the real world we need to perform some test and decide which is the best strategy
            init_obs = None
            half_mesh_indeces = None
            for s in samples:
                f = h5py.File(s, 'r')

                action = np.asarray(f.get('action'))
                mesh = np.asarray(f.get('pcd_pos'))
                if self.fixed_half is None:
                    self.fixed_half = mesh[np.where(mesh[:, 1] <= 0.)]
                if half_mesh_indeces is None:
                    half_mesh_indeces = np.where(mesh[:, 1] > 0.)
                half_mesh = mesh[half_mesh_indeces]
                iou = compute_iou(half_mesh, self.fixed_half, grid_size=self.grid_res)

                if self.obs == 'mesh':
                    full_state = mesh
                    full_pcd_pos = half_mesh
                    pcd_pos = half_mesh
                    # TODO: debug if needed [0]
                    mask = np.zeros((full_state.shape[0], 1))
                    mask[half_mesh_indeces] = 1


                    ## get antway the front pcd
                    front_pcd_pos = np.asarray(f.get('front_pcd'))
                    front_pcd, mask_front = self.filter_half_pointcloud(half_mesh, front_pcd_pos, D_max=0.005)

                if 'pcd' in self.obs:
                    front_pcd_pos = np.asarray(f.get('front_pcd'))
                    pcd_pos, mask_front = self.filter_half_pointcloud(half_mesh, front_pcd_pos, D_max=0.005)
                    front_pcd = copy.deepcopy(pcd_pos)
                    mask = mask_front

                    # add back observation
                    back_pcd_pos = np.asarray(f.get('back_pcd'))
                    half_back_pcd_pos, mask_back = self.filter_half_pointcloud(half_mesh, back_pcd_pos, D_max=0.005)
                    full_state = self.merge_pcds(front_pcd_pos, back_pcd_pos, self.transform)
                    full_pcd_pos = self.merge_pcds(pcd_pos, half_back_pcd_pos, self.transform)

                if self.obs == 'full_pcd':
                    pcd_pos = full_pcd_pos
                    # TODO: debug
                    mask = mask_front or mask_back

                gripper_pos = np.asarray(f.get('gripper_pos')) #- mean
                done = np.asarray(f.get('done')).tolist()

                if self.downsample and self.obs!= 'mesh':
                    downsampled_pcd_pos = self.downsample_pcd(pcd_pos, voxel_size=self.voxel_size)
                    downsampled_full_pcd_pos = self.downsample_pcd(full_pcd_pos, obs_type='pcd', voxel_size=self.voxel_size)
                    downsampled_front_pcd = self.downsample_pcd(front_pcd, obs_type='pcd', voxel_size=self.voxel_size)

                    mean = 0
                    if self.zero_c:
                        mean = downsampled_pcd_pos.mean(axis=0)
                        downsampled_pcd_pos -= mean
                        downsampled_full_pcd_pos -= mean
                        downsampled_front_pcd -= mean
                        gripper_pos -= mean
                    full_states.append(full_state)
                    pcds.append(downsampled_pcd_pos)
                    full_pcds.append(downsampled_full_pcd_pos)
                    front_pcds.append(front_pcd)
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
                    front_pcds.append(front_pcd)
                    grippers.append(gripper_pos)
                    if pcd_pos.shape[0] > self.max_points:
                        self.max_points = pcd_pos.shape[0]

                rewards.append(iou)
                actions.append(action)
                masks.append(mask)
                if len(masks) > 1:
                    nn_masks.append(self.initialize_mask(full_states[-2], masks[-2], full_state))
                else:
                    nn_masks.append(np.zeros((full_state.shape[0], 1)))

                # plot_pcd_list([full_state[(mask == 1)[:, 0]], full_state, ], alpha_value= 0.4)

                f.close()

                if done:
                    # Repeat the first state K times and add zero actions K times
                    # for i in range(self.num_past):
                    #     full_states.insert(0, full_states[0])
                    #     pcds.insert(0, pcds[0])
                    #     full_pcds.insert(0, full_pcds[0])
                    #     grippers.insert(0, grippers[0])
                    #     actions.insert(0, np.zeros_like(actions[0]))
                    #     masks.insert(0, masks[0])
                    #     nn_masks.insert(0, nn_masks[0])

                    # Full trajectory
                    # if self.multi_step:
                    trajs_full_states.append(np.asarray(full_states))
                    trajs_pcds.append(np.asarray(pcds))
                    trajs_full_pcds.append(np.asarray(full_pcds))
                    trajs_front_pcds.append(np.asarray(front_pcds))
                    trajs_grippers.append(np.asarray(grippers))
                    trajs_actions.append(np.asarray(actions))
                    trajs_rewards.append(np.asarray(rewards))
                    trajs_masks.append(masks)
                    trajs_nn_masks.append(nn_masks)

                    full_states = []
                    pcds = []
                    front_pcds = []
                    full_pcds = []
                    grippers = []
                    rewards = []
                    actions = []
                    masks = []
                    nn_masks = []



        return trajs_full_states, trajs_pcds, trajs_full_pcds, trajs_front_pcds, trajs_grippers, trajs_actions, trajs_rewards, trajs_masks, trajs_nn_masks

    def filter_half_pointcloud(self, half_mesh, pointcloud, D_max=0.005):
        tree = cKDTree(half_mesh)
        filtered_set = []
        mask = np.zeros((pointcloud.shape[0], 1))
        for idx, point in enumerate(pointcloud):
            # Find the distance to the nearest point in the first set
            distance, _ = tree.query(point)
            # Keep the point if the distance is less than or equal to D_max
            if distance <= D_max:
                filtered_set.append(point)
                mask[idx] = 1
        return np.array(filtered_set), mask

    def initialize_mask(self, P1, labels_P1, P2):
        """
        Assign labels to points in P2 based on the nearest neighbors in P1.

        Parameters:
        P1 (array): A numpy array of shape (n_points, n_dimensions) representing point cloud P1.
        labels_P1 (array): A numpy array of shape (n_points,) representing labels for each point in P1.
        P2 (array): A numpy array of shape (m_points, n_dimensions) representing point cloud P2.

        Returns:
        numpy array: Array of labels for P2.
        """
        # Build a k-d tree for P1
        tree = KDTree(P1)

        # Find the nearest neighbor in P1 for each point in P2
        _, indices = tree.query(P2)

        # Assign labels from P1 to P2 based on nearest neighbors
        return labels_P1[indices]

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
        range_x = self.look_up[idx]['x']
        # range_x_1 = self.look_up[idx]['x-1']


        x_pcds = copy.deepcopy(self.pcds[env][traj][range_x])[0:1]
        x_front_pcds = copy.deepcopy(self.front_pcds[env][traj][range_x])[0:1]
        reward = copy.deepcopy(self.rewards[env][traj][range_x])[0:1]

        # plot_vectors_3d(action=x_actions, points=x_pcds[0], predicted_points=y_pcds[0], gripper_pos=x_grippers[0])
        # plt.show()

        if self.data_aug and self.train:
            set_pcds, _ = self.augment_pcd([
                                                 copy.deepcopy(x_pcds),
                                                 copy.deepcopy(x_front_pcds),
                                                 ],
                                                augmentations=['noise', 'scale', 'rot', 't'],
                                                # augmentations=['rot', 't'],
                                                # augmentations=['noise',],
                                                types=['pc', 'pc'])

            x_pcds, x_front_pcds = set_pcds



        datas = []
        # No need for multiple pointclouds as input as we only give the one at time t even if we predict t+H
        # for i in range(len(x_pcds)):
        x_pcd_pos = torch.from_numpy(x_pcds[0]).float()
        x_front_pcds_pos = torch.from_numpy(x_front_pcds[0]).float()
        reward = torch.tensor(reward[0], dtype=torch.float)


        # pad pcd and actions
        # a = torch.cat([torch.from_numpy(x_actions[i]).float() for i in range(len(x_actions))], dim=-1)
        # grip_action = torch.cat([x_gripper_pos, x_actions], dim=-1).repeat(x_pcd_pos.shape[0], 1) # append actions to the gripper
        # x_pcd_feat = torch.cat([x_pcd_pos, grip_action, x_nn_mask], dim=-1)

        data = Data(x=x_pcd_pos, pos=x_pcd_pos, y=x_pcd_pos)
        # datas.append(data)
        # #
        # datas = Batch.from_data_list(datas)
        data_front = Data(x=x_front_pcds_pos, pos=x_front_pcds_pos)


        return data, data_front, reward

    def get_sample_trajectory(self, env, traj):
        # avoid augmentations
        train = copy.deepcopy(self.train)
        self.train = False

        traj_z, traj_forward, traj_params = [], [], []
        for idx in range(len(self.look_up)):
            if self.look_up[idx]['env'] == env and self.look_up[idx]['traj'] == traj:
                batch_z, batch_forward, params, batch_y = self._get_datapoint(
                    idx)
                traj_z.append(batch_z)
                traj_forward.append(batch_forward)
                traj_params.append(params)

        self.train = train

        return [traj_z, traj_forward, traj_params]


    def augment_pcd(self, set_pcds,
                params=None,
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
            if params is not None:
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
    # path = '../data/datasets/june16_1s_train/env*'
    path = '../data/datasets/oct31_1s_train/env*'
    paths = glob.glob(path)
    # go inside first folder
    paths = glob.glob(os.path.join(paths[0], '*'))
    paths.sort()

    args = get_argparse()
    args.obs = 'mesh'       # [mesh, pcd, full_pcd]
    dataset = PointcloudAEDataset(paths[:5], args=args, downsample=False, downsampling="voxel")
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
