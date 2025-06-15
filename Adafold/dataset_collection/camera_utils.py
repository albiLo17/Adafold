import numpy as np
import open3d as o3d
import pybullet as p

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

