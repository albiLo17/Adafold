import numpy as np
import os
import h5py

from Adafold.utils.visual import get_pixel_coord_from_world, get_world_coord_from_pixel
# from folding.utils import Cameras

# class Cameras():
#     def __init__(self, id):
#         self.cameras = []
#         self.camera_width = []
#         self.camera_height = []
#         self.view_matrix = []
#         self.projection_matrix = []

#         self.id = id

#         self.roi = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0.02, -0.2, -0.66), max_bound=(0.2, -0.1, -0.48))


def voxel_grid_from_mask(mask, camera_params, z_min=0.02, z_max=0.08, voxel_size=0.04, gripper_pos=None, place=None, env=None):
    # TODO: remove the assumption that the cloth is not rotated
    rows, cols = np.where(mask != 0)
    # Compute the bounding box coordinates
    # min_row, min_col = np.min(rows), np.min(cols)
    # max_row, max_col = np.max(rows), np.max(cols)
    # env.create_sphere(pos=(get_world_coord_from_pixel([min_row, min_col], mask, env.camera_params)))

    bottom = get_world_coord_from_pixel([np.min(rows), np.min(cols)], mask, camera_params)
    top = get_world_coord_from_pixel([np.max(rows), np.max(cols)], mask, camera_params)
    x_min, y_min = bottom[0], bottom[1]
    x_max, y_max = top[0], top[1]
    num_voxels_x = int(np.abs((x_max - x_min)) / voxel_size) + 1
    num_voxels_y = int(np.abs((y_max - y_min)) / voxel_size) + 1
    num_voxels_z = int(np.abs((z_min - z_max)) / voxel_size) + 1
    x = np.linspace(min(x_min, x_max), max(x_min, x_max), num=num_voxels_x)
    y = np.linspace(min(y_min, y_max), max(y_min, y_max), num=num_voxels_y)
    z = np.linspace(z_min, z_max, num=num_voxels_z)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Combine the xx, yy, and zz arrays into a single 3D vector
    voxel_grid = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(xx, yy, zz)

    if gripper_pos is not None and place is not None:
        voxel_grid = filter_voxel_grid(grid=voxel_grid, gripper_pos=gripper_pos, place=place)
    return voxel_grid

def filter_voxel_grid(grid, gripper_pos, place):
    # Function that filters the grid by removing too far waypoints and waypoints that go back
    filtered_voxel = grid[grid[:, 1] <= gripper_pos[1]]
    filtered_voxel = filtered_voxel[np.linalg.norm(-1 * filtered_voxel + place, axis=1) < np.linalg.norm(place - gripper_pos)]
    filtered_voxel = filtered_voxel[np.linalg.norm(filtered_voxel - gripper_pos, axis=1) < np.linalg.norm(place - gripper_pos)]

    filtered_voxel = filtered_voxel[
        np.linalg.norm(-1 * filtered_voxel + place, axis=1) > np.linalg.norm(place - gripper_pos)/10]
    filtered_voxel = filtered_voxel[
        np.linalg.norm(filtered_voxel - gripper_pos, axis=1) > np.linalg.norm(place - gripper_pos)/10]

    return filtered_voxel

def reset_env(env, elas, bend, scale):
    damp, frict = 1.5, 1.50
    obs = env.reset(stiffness=[elas, bend, damp], friction=frict, cloth_scale=scale, cloth_mass=0.5)  # Elas, bend, damp

    env.camera = Cameras(id=env.id)
    # The position of the camera matches the RW
    env.camera.setup_camera(camera_eye=[0.0, -0.72, 0.27], camera_target=[0., 0., 0.], camera_width=720,
                            camera_height=720)

    # This second camera is used to eventually obtain the full pointcloud
    env.camera.setup_camera(camera_eye=[0.28, 0.55, 0.28], camera_target=[0., 0., 0.], camera_width=720,
                            camera_height=720)

    env.setup_camera(camera_eye=[0., 0., 0.65], camera_target=[0., 0.0, 0.], camera_width=720,
                     camera_height=720)



def get_mask(depth, threshold=0.64705):
    mask = depth.copy()
    mask[mask > threshold] = 0
    mask[mask != 0] = 1
    return mask


def preprocess(depth, threshold=0.64705):
    mask = get_mask(depth, threshold=threshold)
    depth = depth * mask
    return depth




def get_all_envs(elast=np.arange(20, 70, 5),
                 bend=np.arange(20, 50, 5),
                 scale=np.arange(7,12)*0.01,
                 frame_skip=np.asarray([2, 5, 8]),
                 side=np.asarray([0,1])):
    params = []
    for e in elast:
        for b in bend:
            for sc in scale:
                for f in frame_skip:
                    for sd in side:
                        params.append([e, b, sc, f, sd])
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


