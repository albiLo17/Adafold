import numpy as np
import h5py
import os
import torch
import matplotlib
import open3d as o3d
import matplotlib.pyplot as plt
import pybullet as p
import cv2
from scipy.spatial import Delaunay
import threading

def chamfer(predictions, labels, bidirectional=True):
    # a and b don't need to have the same number of points\
    # TODO: fix description
    """"
    Chamfer loss:
        predictions \in batch x num pcd x num_nodes x dim
        labels \in batch x num pcd x num_nodes x dim
        bidirectional \modality of chamfer loss
    """
    c_dist = torch.cdist(predictions, labels, p=2) ** 2
    dist = c_dist.min(dim=-2)[0].mean(-1)       # Keep only distance from g_t to predicted, the other way around leads to collapse
    if bidirectional:
        dist += c_dist.min(dim=-1)[0].mean(-1)

    # Batch average
    chamfer_dist = dist.mean()
    return chamfer_dist


def plot_predictions(pct_fut, pred, metric='euc'):      # possible metrics: euc - cham
    fig = plt.figure()
    # axes = fig.add_subplot(projection='3d', elev=20, azim=45)
    axes = fig.add_subplot(projection='3d', elev=20, azim=20)

    # , axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5), projection='3d', elev=0, azim=-45)

    rr = 0.7
    process_tensor = lambda t: t.detach().cpu().data.numpy()

    def visualize_state(ax, state, predicted_clusters=None, fix_lim=False, title=''):

        if torch.is_tensor(state):
            state = process_tensor(state)
        if fix_lim:
            # ax.set_xlim(0.9, 1.2)
            # ax.set_ylim(0.2, 0.5)
            # axes.set_zlim(-0.2, -0.5)
            # ax.set_xlim(-0.35, -0.15)
            # ax.set_ylim(-0.35, -0.15)
            # axes.set_zlim(-0.25, -0.15)
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            axes.set_zlim(-0.1, 0.1)
        ax.set_facecolor('white')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        plt.axis('off')

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        if state is not None:
            if torch.is_tensor(state):
                state = process_tensor(state)
            ax.scatter(state[:, 0], state[:, 1], state[:, 2], color='black', s=1, marker='o', label='barycenter',
                       alpha=0.5, )

        if predicted_clusters is not None:
            if torch.is_tensor(predicted_clusters):
                predicted_clusters = process_tensor(predicted_clusters)
            # C = [[1, 0, 0]]*predicted_clusters.shape[0]

            if metric == 'euc':
                diff = state - predicted_clusters
                # distance =np.linalg.norm( (diff - np.min(diff, 0)) / (np.max(diff, 0) - np.min(diff, 0)), axis=1)
                distance =np.linalg.norm( diff, axis=1)


            if metric == 'cham':
                if torch.is_tensor(state):
                    predicted_clusters = process_tensor(predicted_clusters)
                c_dist = torch.cdist(torch.from_numpy(predicted_clusters), torch.from_numpy(state), p=2)
                distance =c_dist.min(dim=-1)[0].numpy()

            # distance = (distance - (distance.min() - distance.min()*0.01 )) / (distance.max() - (distance.min() - distance.min()*0.1))
            distance = (distance - (distance.min())) / (distance.max() - (distance.min()))
            # print(torch.is_tensor(distance[0]))
            C = [[(d), 0, 0] for d in distance]



            # ax.scatter(predicted_clusters[:, 2], predicted_clusters[:, 0], predicted_clusters[:, 1], color='darkred', s=6, marker='X',
            #            alpha=0.9, label='prediction')
            ax.scatter(predicted_clusters[:, 0], predicted_clusters[:, 1], predicted_clusters[:, 2], c=C , s=6, marker='X',
                       alpha=0.9, label='prediction')

            C.sort(key=lambda color: color[0])
            cmap = matplotlib.colors.ListedColormap([[r/100, 0, 0] for r in range(0, 100, 1)])
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            # plt.colorbar(sm, shrink=0.5)



    # visualize_state(axes[0], pcd_pre, fix_lim=True, title='first state')  #
    visualize_state(axes, pct_fut, predicted_clusters=pred, fix_lim=True, title='second state')

    return fig



def voxelized_mid_waypoints(voxel_size=0.01, max_length=0.2, min_length=0.01, x_coord=0.1, z_offset=0, z_min=0.02, along_x=True):
    # this funciton assume that we are folding along the x axis,
    # if this is not the case then we need to apply a transformation before and after
    if not along_x:
        print("Not implemented yet for pick-place actions that are not along the x axis")
        exit()

    # Determine the number of discrete points in each dimension
    num_points = int(np.ceil(max_length / voxel_size)) + 1

    # Generate a grid of coordinates
    y_coords = np.linspace(0, max_length, num_points)
    z_coords = np.linspace(0, max_length, num_points)
    yy, zz = np.meshgrid(y_coords, z_coords)

    # Convert the grid of coordinates to a list of points
    points = []
    for i in range(num_points):
        for j in range(num_points):
            p = [yy[i, j], zz[i, j]]
            if zz[i, j] >= z_min:
                if np.linalg.norm(p) <= max_length and  np.linalg.norm(p) >= min_length:
                    points.append(p)

    points = np.asarray(points)
    final_waypoints = np.ones((points.shape[0], 3)) * x_coord
    final_waypoints[:, 1] = points[:, 0]        # yy
    final_waypoints[:, 2] = points[:, 1] + z_offset      # zz

    return final_waypoints

# Comment for the cluster
class Cameras():
    def __init__(self, id):
        self.cameras = []
        self.camera_width = []
        self.camera_height = []
        self.view_matrix = []
        self.projection_matrix = []

        self.id = id

        self.roi = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0.02, -0.2, -0.66), max_bound=(0.2, -0.1, -0.48))
        # cropped_pcd_rw = pcd_rw.crop(roi)

        # TODO: add bounding boxes?

    def get_pointclouds(self):
        # Collect point clouds from each camera
        intrinsic = []
        extrinsic = []
        point_clouds = []
        for i, camera in enumerate(self.cameras):
            _, _, depth, _, _, _ = p.getCameraImage(
                width=self.camera_width[i],
                height=self.camera_height[i],
                viewMatrix=camera[2],
                projectionMatrix=camera[3],
                shadow=1,
                lightDirection=[1, 1, 1],
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
                width=640,
                height=480,
                fx=camera[3][0][0],
                fy=camera[3][1][1],
                cx=camera[3][0][2],
                cy=camera[3][1][2]
            )
            extrinsic_matrix = camera[2]
            rgbd_image = o3d.geometry.RGBDImage.create_from_depth_image(
                o3d.geometry.Image(depth),
                intrinsic_matrix,
                extrinsic_matrix
            )
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsic_matrix
            )
            intrinsic.append(intrinsic_matrix)
            extrinsic.append(extrinsic_matrix)
            point_clouds.append(point_cloud)
        return intrinsic, extrinsic, point_clouds

    def get_scene(self):
        intrinsic, extrinsic, point_clouds = self.get_pointclouds()

        # Merge point clouds
        fusion = o3d.pipelines.integration.TSDFVolumeIntegrator(
            voxel_length=0.01,
            sdf_trunc=0.03,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for i, point_cloud in enumerate(point_clouds):
            fusion.integrate(
                point_cloud,
                intrinsic[i],
                extrinsic[i]
            )

        # Extract mesh from fused point cloud
        mesh = fusion.extract_triangle_mesh()
        o3d.visualization.draw_geometries([mesh])

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920 // 4,
                     camera_height=1080 // 4):

        self.camera_width.append(camera_width)
        self.camera_height.append(camera_height)
        self.view_matrix.append(p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id))
        self.projection_matrix.append(p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100,
                                                              physicsClientId=self.id))

    def visualize_all_cameras(self):
        pcds = []
        for i in range(len(self.camera_height)):
            pcd, colors = self.get_point_cloud(i)

            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(pcd)
            pointcloud.colors = o3d.utility.Vector3dVector(colors)
            pcds.append(pointcloud)
        o3d.visualization.draw_geometries(pcds)


    def segment_pointcloud(self, points, colors, RGB_threshold=[0.78, 0.78, 0.78]):
        # Define threshold for color segmentation (WHITE)
        red_threshold = RGB_threshold[0]
        green_threshold = RGB_threshold[1]
        blue_threshold = RGB_threshold[2]

        # Create boolean mask for color segmentation
        mask = (colors[:, 0] > red_threshold) & (colors[:, 1] > green_threshold) & (
                    colors[:, 2] > blue_threshold)

        # Extract segmented point cloud
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        pcd_segmented = pointcloud.select_by_index(np.where(mask)[0])
        # Extract segmented point cloud
        points_segmented = np.asarray(pcd_segmented.points)
        colors_segmented = np.asarray(pcd_segmented.colors)

        return points_segmented, colors_segmented

    def get_segmented_pointclouds(self, RGB_threshold=[0.78, 0.78, 0.78]):
        pcds = []
        for i in range(len(self.camera_height)):
            pcd, colors = self.get_point_cloud(i)
            pcd_seg, colors_seg = self.segment_pointcloud(pcd, colors, RGB_threshold=RGB_threshold)
            pcds.append(pcd_seg)

        return pcds


    def voxelize_pcd(self, points, voxel_size=0.001):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        # Get all the voxels in the voxel grid
        voxels_all = voxel_grid.get_voxels()
        # get all the centers and colors from the voxels in the voxel grid
        all_centers = [voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels_all]
        downsampled_pcd = np.asarray(all_centers)

        return downsampled_pcd

    def get_point_cloud(self, camera_id=0, light_pos=[0, -3, 1], shadow=False, ambient=0.8, diffuse=0.3, specular=0.1, object_id=None):

        # get a depth image
        # "infinite" depths will have a value close to 1
        image_arr = p.getCameraImage(self.camera_width[camera_id], self.camera_height[camera_id],
            self.view_matrix[camera_id], self.projection_matrix[camera_id],
            shadow=shadow,
            lightDirection=light_pos, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.id)

        depth = np.reshape(image_arr[3], (image_arr[1], image_arr[0]))

        # zfar = 1000.
        # znear = 0.01
        # mydepth = depth.copy()
        # mydepth = (zfar + znear - (2. * mydepth - 1.) * (zfar - znear))
        # mydepth = (2. * znear * zfar) / mydepth

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix[camera_id]).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix[camera_id]).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.camera_height[camera_id], -1:1:2 / self.camera_width[camera_id]]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        rgb = np.reshape(image_arr[2], (image_arr[1], image_arr[0], 4)).reshape(-1, 4)[:, :-1].astype(np.float) / 255.0
        colors = rgb[z < 0.99]

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points, colors




def get_all_envs(elast=np.arange(20, 70, 5),
                              bend=np.arange(20, 50, 5),
                              scale=np.arange(7,12)*0.01,
                              frame_skip=np.asarray([2, 5, 8])):
    params = []
    for e in elast:
        for b in bend:
            for s in scale:
                for f in frame_skip:
                    params.append([e, b, s, f])
    return params


def store_data_by_name(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(path):
    f = h5py.File(path, 'r')
    obs = np.asarray(f.get('obs'))
    f.close()
    return obs


def make_dir(path):
    tot_path = ''
    for folder in path.split('/'):
        if not folder == '.' and not folder == '':
            tot_path = tot_path + folder + '/'
            if not os.path.exists(tot_path):
                os.mkdir(tot_path)
                # print(tot_path)
        else:
            if folder == '.':
                tot_path = tot_path + folder + '/'


def get_grid_mask(points, grid_size=100):
    # TODO: not working
    # Compute the Delaunay triangulation of the points
    tri = Delaunay(points*grid_size)

    # fig, ax = plt.subplots()
    # ax.triplot(points[:, 0], points[:, 1], tri.simplices)
    # ax.plot(points[:, 0], points[:, 1], 'o')
    # ax.set_aspect('equal')
    # plt.show()

    # Calculate the bounding box of the points
    min_x, min_y = -0.2*grid_size, -0.2*grid_size
    max_x, max_y = 0.2*grid_size, 0.2*grid_size

    # Calculate the dimensions of the grid
    grid_width = int(np.ceil((max_x - min_x) / grid_size))
    grid_height = int(np.ceil((max_y - min_y) / grid_size))

    # Create a mask with the same dimensions as the grid
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Iterate over the triangles and set the pixels inside to 1
    for triangle in tri.simplices:
        pts = tri.points[triangle]
        rect = cv2.boundingRect(pts.astype(np.float32))
        rect_pts = ((pts - rect[:2]) / grid_size).astype(int)
        cv2.fillConvexPoly(mask, rect_pts, 1)

    return mask


def get_spatial_occupancy_mask(points_list):

    def get_mask(y_cells, x_cells, x_range, y_range, points):
        # Create empty occupancy grid
        occupancy_grid = np.zeros((y_cells, x_cells))
        # Loop over each cell in the grid
        for i in range(y_cells):
            for j in range(x_cells):
                # Calculate the boundaries of the current cell
                x_min = j * cell_size + x_range[0]
                x_max = (j + 1) * cell_size + x_range[0]
                y_min = i * cell_size + y_range[0]
                y_max = (i + 1) * cell_size + y_range[0]
                # Check if any of the points fall within the cell boundaries
                for point in points:
                    if x_min <= point[0] < x_max and y_min <= point[1] < y_max:
                        # Mark cell as occupied if at least one point falls within it
                        occupancy_grid[i][j] = 1
                        break
        # plt.imshow(occupancy_grid, cmap='gray')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()
        return occupancy_grid

    x_range = [-0.2, 0.2]
    y_range = [-0.2, 0.2]
    cell_size = 0.02
    x_cells = int(np.ceil((x_range[1] - x_range[0]) / cell_size))
    y_cells = int(np.ceil((y_range[1] - y_range[0]) / cell_size))

    # Compute the number of grid points in each dimension
    masks = np.zeros((len(points_list), x_cells, y_cells), dtype=np.uint8)

    for p, points in enumerate(points_list):

        # Initialize the mask with zeros
        mask = get_mask(y_cells, x_cells, x_range, y_range, points)
        masks[p] = mask

    return masks