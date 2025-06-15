import numpy as np
import os
import h5py
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
# Function to calculate the IoU between two polygons
def calculate_iou_poly(point_cloud1, point_cloud2):
    # Convert point clouds to polygons
    polygon1 = Polygon(point_cloud1)
    polygon2 = Polygon(point_cloud2)

    # Calculate IoU
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    iou = intersection_area / union_area
    return iou

def visualize_polygon(polygon):
    # Extract the x and y coordinates of the polygon's exterior
    x, y = polygon.exterior.xy

    # Create a plot
    plt.figure()
    plt.plot(x, y, color='blue', linewidth=2)
    plt.fill(x, y, color='lightblue', alpha=0.6)

    # Set axis limits
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon Visualization')

    # Show the plot
    plt.show()


import torch


def soft_iou(point_cloud1, point_cloud2, image_size, smooth=1.0):
    # Convert point clouds to soft masks
    mask1 = create_soft_mask(point_cloud1, image_size)
    mask2 = create_soft_mask(point_cloud2, image_size)

    # Calculate soft IoU
    intersection = torch.sum(torch.min(mask1, mask2))
    union = torch.sum(torch.max(mask1, mask2))
    iou = (intersection + smooth) / (union + smooth)
    return iou


def create_soft_mask(point_cloud, image_size, sigma=1.0):
    mask = torch.zeros(image_size, dtype=torch.float32)

    # Scale the point cloud to match the image size
    scale_factor_x = (image_size[1] - 1) / (point_cloud.max(0)[0][0] - point_cloud.min(0)[0][0])
    scale_factor_y = (image_size[0] - 1) / (point_cloud.max(0)[0][1] - point_cloud.min(0)[0][1])
    scaled_point_cloud = (point_cloud - point_cloud.min(0)[0]) * torch.tensor([scale_factor_x, scale_factor_y])
    scaled_point_cloud = scaled_point_cloud.round().long()

    for point in scaled_point_cloud:
        y, x = point
        if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
            mask[y, x] = 1

    return mask


def normalized_wasserstein_distance(set1, set2, d_max=0.05):
    # Flatten the point sets and compute the distance
    set1_flattened = set1.flatten()
    set2_flattened = set2.flatten()
    distance = wasserstein_distance(set1_flattened, set2_flattened)

    # normalized such that 1 corresponds to perfect alignment and 0 corresponds to worst alignment
    normalized = (1 - distance / d_max)
    return normalized


# Example usage
# point_cloud1 = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float32)
# point_cloud2 = torch.tensor([[2, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.float32)
# image_size = (10, 10)
#
# iou = soft_iou(point_cloud1, point_cloud2, image_size)
# print("Soft IoU:", iou.item())

def measure_area(pointcloud, grid_resolution=0.01):
    # Project the 3D points onto the 2D plane
    projected_points = pointcloud[:, :2]
    # Map the projected points onto the occupancy grid
    # min_x = np.min(projected_points[:, 0])
    # max_x = np.max(projected_points[:, 0])
    # min_y = np.min(projected_points[:, 1])
    # max_y = np.max(projected_points[:, 1])
    min_x = -0.2
    max_x = 0.2
    min_y = -0.2
    max_y = 0.2
    grid_width = int(np.ceil((max_x - min_x) / grid_resolution))
    grid_height = int(np.ceil((max_y - min_y) / grid_resolution))
    occupancy_grid = np.zeros((grid_height, grid_width))
    depth = np.zeros((grid_height, grid_width))
    for i, point in enumerate(projected_points):
        x = int(np.floor((point[0] - min_x) / grid_resolution))
        y = int(np.floor((point[1] - min_y) / grid_resolution))
        occupancy_grid[y, x] = 1
        if pointcloud[i][2] > depth[y, x]:
            depth[y, x] = pointcloud[i][2]
    # Sum the binary occupancy map to obtain the total area covered
    area = np.sum(occupancy_grid)
    return area, occupancy_grid, depth


def compute_iou(set1, set2, grid_size=0.01):
    # Project 3D points onto 2D by ignoring the Z dimension
    set1_2d = set1[:, :2]
    set2_2d = set2[:, :2]

    # Compute axis-aligned bounding boxes (AABB)
    min_bound = np.minimum(np.min(set1_2d, axis=0), np.min(set2_2d, axis=0))
    max_bound = np.maximum(np.max(set1_2d, axis=0), np.max(set2_2d, axis=0))

    # Create occupancy grids
    x_grid = np.arange(min_bound[0], max_bound[0], grid_size)
    y_grid = np.arange(min_bound[1], max_bound[1], grid_size)
    grid1 = np.zeros((len(x_grid), len(y_grid)), dtype=bool)
    grid2 = np.zeros((len(x_grid), len(y_grid)), dtype=bool)

    # Mark occupied cells in the grids
    for x in set1_2d:
        i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
        grid1[i, j] = True
    for x in set2_2d:
        i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
        grid2[i, j] = True

    # Compute intersection and union
    intersection = np.logical_and(grid1, grid2)
    union = np.logical_or(grid1, grid2)
    inter_area = np.sum(intersection)
    union_area = np.sum(union)

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

# def compute_iou(set1, set2, grid_size=0.01):
#     # Check the type of the inputs
#     if isinstance(set1, torch.Tensor):
#         module = torch
#     elif isinstance(set1, np.ndarray):
#         module = np
#     else:
#         raise ValueError("Unsupported input type: {}".format(type(set1)))
#
#     # Project 3D points onto 2D by ignoring the Z dimension
#     set1_2d = set1[:, :2]
#     set2_2d = set2[:, :2]
#
#     # Compute axis-aligned bounding boxes (AABB)
#     min_bound = module.min(module.min(set1_2d, axis=0), module.min(set2_2d, axis=0))
#     max_bound = module.max(module.max(set1_2d, axis=0), module.max(set2_2d, axis=0))
#
#     # Create occupancy grids
#     x_grid = module.arange(min_bound[0], max_bound[0], grid_size)
#     y_grid = module.arange(min_bound[1], max_bound[1], grid_size)
#     grid1 = module.zeros((len(x_grid), len(y_grid)), dtype=bool)
#     grid2 = module.zeros((len(x_grid), len(y_grid)), dtype=bool)
#
#     # Mark occupied cells in the grids
#     for x in set1_2d:
#         i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
#         grid1[i, j] = True
#     for x in set2_2d:
#         i, j = int((x[0] - min_bound[0]) / grid_size), int((x[1] - min_bound[1]) / grid_size)
#         grid2[i, j] = True
#
#     # Compute intersection and union
#     intersection = module.logical_and(grid1, grid2)
#     union = module.logical_or(grid1, grid2)
#     inter_area = module.sum(intersection)
#     union_area = module.sum(union)
#
#     # Compute the IoU
#     iou = inter_area / union_area if union_area != 0 else 0
#
#     return iou

def filter_half_pointcloud(half_mesh, pointcloud, D_max=0.005):
    tree = cKDTree(half_mesh)
    filtered_set = []
    for point in pointcloud:
        # Find the distance to the nearest point in the first set
        distance, _ = tree.query(point)
        # Keep the point if the distance is less than or equal to D_max
        if distance <= D_max:
            filtered_set.append(point)
    return np.array(filtered_set)

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
