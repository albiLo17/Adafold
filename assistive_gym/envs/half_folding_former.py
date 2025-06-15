import copy
import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from gym import spaces
import scipy
from assistive_gym.utils.trajectory import Trajectory
from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import furniture
from .agents.furniture import Furniture
import cv2
from scipy.spatial.distance import cdist
import time
from Adafold.utils.utils_planning import measure_area, compute_iou, filter_half_pointcloud

class HalfFoldEnv(AssistiveEnv):
    def __init__(self,
                 robot=None,
                 human=None,
                 use_ik=False,
                 frame_skip=10,
                 hz=480,
                 obs='mesh',
                 reward='corner',
                 grid_res=0.02,
                 use_mesh=False,
                 gripper_attractor_tr=0.04,
                 action_mult=0.01,
                 side=0,
                 debug_reward=False):
        assert human is None, "For now just consider no human!"
        if robot is None:
            super(HalfFoldEnv, self).__init__(robot=None, human=human, task='folding', obs_robot_len=12, obs_human_len=0, frame_skip=frame_skip, time_step=0.01, deformable=True)
            self.use_mesh = use_mesh
        else:
            super(HalfFoldEnv, self).__init__(robot=robot, human=None, task='folding',
            obs_robot_len=(16 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)),
            obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)),
            frame_skip=frame_skip, time_step=1./480, deformable=True)

            self.use_ik = use_ik
            self.use_mesh = (human is None)

        # Divide by frame skip to be sure that the simulation time is always the same independently on the number of steps
        self.time_step = 1.0 / hz
        # divide by frame skip that is the number of substeps
        p.setTimeStep(self.time_step/self.frame_skip)
        self.side = side
        # self.time_step = self.time_step/self.frame_skip
        # p.setTimeStep(self.time_step)

        # TODO: this might be variable at some point
        self.obs_clot_len = 625 * 3
        self.observation_space = spaces.Box(low=np.array([-1000000000.0]*(self.obs_clot_len), dtype=np.float32), high=np.array([1000000000.0]*(self.obs_clot_len), dtype=np.float32), dtype=np.float32)
        self.action_len = 3
        self.action_space.__init__(low=-np.ones(self.action_len, dtype=np.float32),
                                   high=np.ones(self.action_len, dtype=np.float32), dtype=np.float32)

        self.action_multiplier = action_mult
        self.apply_rw_offset = False        # Set to True if we want it to be like in the real world
        self.use_table = False

        self.x_workspace = [-0.5, 0.5]
        self.y_workspace = [-0.5, 0.5]
        self.z_workspace = [0.01, 2.]

        self.camera = None
        self.image_dim = 224
        self.reward = reward
        self.cost_coeff = [1., 1/200, 10., -1]        # [corner, foldErr, chamfer, IoU]
        self.obs_type = obs
        self.velocity = 0.02
        self.grid_res = grid_res       # For occupancy grid
        self.gripper_attractor_tresh = gripper_attractor_tr

        self.debug_rewards = debug_reward

        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS
        # self.reset(stiffness=[40, 20., 1.5], friction=1.2, cloth_scale=0.1, cloth_mass=0.5)

    def process_pointclouds(self, RGB_threshold=[0.78, 0.78, 0.78], voxel_size=0.005):
        seg_pcds = self.camera.get_segmented_pointclouds(RGB_threshold=RGB_threshold)
        front_pcd = seg_pcds[0]
        back_pcd = seg_pcds[1]
        partial_front_voxelized = self.camera.voxelize_pcd(front_pcd, voxel_size=voxel_size)
        partial_back_voxelized = self.camera.voxelize_pcd(back_pcd, voxel_size=voxel_size)

        return partial_front_voxelized, partial_back_voxelized
    def step(self, action):

        self.take_step_sphere(action, action_multiplier=1, num_sphere_steps=1)        # TODO: decide the action multiplier
        # self.take_step_joint_sphere(action, action_multiplier=1, num_sphere_steps=1)
        # self.take_step(action, action_multiplier=0.003)

        # print(f'Total step time : {time.time() - start_time}')

        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_cloth_matching = self._get_reward()

        reward = reward_cloth_matching

        obs = self._get_obs()

        info = {}

        done = self.iteration >= 100

        if not done:
            done = self.check_done()

        return obs, reward, done, info

    def take_step_sphere(self, action, action_multiplier=1., num_sphere_steps=1):
        # start_time = time.time()
        # action is 7 dimensional so lets use the 3 components
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        final_pos = current_pos + action[:3] * action_multiplier
        final_pos[2] = np.clip(final_pos[2], self.z_workspace[0], self.z_workspace[1])

        action = final_pos - current_pos

        # past_pose = copy.deepcopy(current_pos)
        # current_pos += action
        # self.sphere_ee.set_base_pos_orient(current_pos, np.array([0, 0, 0]))
        # p.stepSimulation(physicsClientId=self.id)
        #
        # # Let the cloth settle
        # for _ in range(self.frame_skip):
        #     p.stepSimulation(physicsClientId=self.id)
        # self.iteration += 1

        # Num sphere steps to get to the position in 1s
        augment_time = 50
        # augment_time = 20
        num_substeps = int(1 / (self.time_step*augment_time))
        # num_substeps = self.frame_skip
        sub_action = action / (num_substeps)

        for i in range(num_substeps):
            current_pos += sub_action
            self.sphere_ee.set_base_pos_orient(current_pos, np.array([0, 0, 0]))
            # start_time = time.time()
            # p.stepSimulation(physicsClientId=self.id)

            for i in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
            # print(f'step with action: {time.time() - start_time}')
            #
            # start_time = time.time()
            # p.stepSimulation(physicsClientId=self.id)
            # print(f'step without action: {time.time() - start_time}')

        self.iteration += 1
        # print(f'Moved by {self.sphere_ee.get_base_pos_orient()[0] - past_pose}')
    def take_step_joint_sphere(self, action, action_multiplier=1., num_sphere_steps=1):
        # action is 7 dimensional so lets use the 3 components
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        final_pos = current_pos + action[:3] * action_multiplier
        final_pos[2] = np.clip(final_pos[2], self.z_workspace[0], self.z_workspace[1])


        # Define the time step and the initial position and velocity of the sphere
        dt = self.time_step
        vx = self.velocity


        # Compute the error between the current and target positions
        error = final_pos - current_pos
        max_force = 10
        kp = 50
        raw_force = kp * error
        force = np.clip(raw_force, -1.0 * max_force, max_force)
        t = 0
        # Set the target position of the floating joint
        # while np.linalg.norm(error) > 0.01:
        # for i in range(self.frame_skip):
        p.changeConstraint(self.sphere_ee_joint, final_pos, maxForce=10)
        # Step the simulation
        p.stepSimulation()

        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        error = final_pos - current_pos
        time.sleep(dt)
        # Update the time
        t += dt


    def check_done(self):
        # TODO: look into this
        # the grasped corner is close to the final one
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])

        threshold = 0.01
        # print(f'distance: {np.linalg.norm(data[24] - data[624])}')
        if np.linalg.norm(data[24] - data[624]) < threshold:
            return True
        return False

    def _get_obs(self, agent=None):
        cloth_mesh = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]).flatten()
        return cloth_mesh

    def get_mesh(self, data=None):
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))
        return data

    def _get_reward(self):
        reward = 0
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])

        # Corner matching
        if 'corner' in self.reward or self.debug_rewards:
            # get corner positions
            left_corners = np.asarray([data[self.corners[0]], data[self.corners[2]]])
            right_corners = np.asarray([data[self.corners[1]], data[self.corners[3]]])
            norm = np.linalg.norm(left_corners - right_corners)

            if self.debug_rewards:
                reward = {'corner': -self.cost_coeff[0]*norm}
            else:
                reward -= self.cost_coeff[0]*norm

        # Folding error from Benchmariking
        if 'foldErr' in self.reward or self.debug_rewards:
            area = np.asarray( [measure_area(data, grid_resolution=self.grid_res)[0]])
            folding_error = np.abs((area[0] / self.init_area - 0.5) * 100 / 0.5)

            # for debugging purposes
            # area, grid, depth = measure_area(data, grid_resolution=self.grid_res)
            # plt.figure()
            # plt.imshow(grid, vmin=0., vmax=0.03)
            # plt.show()
            if self.debug_rewards:
                reward.update({'foldErr': -self.cost_coeff[1]*folding_error})
            else:
                reward -= self.cost_coeff[1]*folding_error

                # Shape matching
        if 'chamfer' in self.reward or self.debug_rewards:
            def chamfer(set_a, set_b):
                distances_a_to_b = cdist(set_a, set_b, 'minkowski', p=2)  # compute distances from points in set_a to set_b
                distances_b_to_a = cdist(set_b, set_a, 'minkowski', p=2)  # compute distances from points in set_b to set_a
                min_distances_a_to_b = distances_a_to_b.min(
                    axis=1)  # find the minimum distance from each point in set_a to set_b
                min_distances_b_to_a = distances_b_to_a.min(
                    axis=1)  # find the minimum distance from each point in set_b to set_a
                return np.mean(min_distances_a_to_b) + np.mean(
                    min_distances_b_to_a)  # return the mean of the minimum distances in both directions

            # Get desired shape
            desired = self.init_pos_half.reshape(-1, 3)
            desired_second = copy.deepcopy(desired)
            desired_second[:, 2] += 0.01
            full_cloth = np.concatenate((desired_second, desired), axis=0)
            desired_shape = full_cloth
            # Get current shape
            current_shape = data       # take only x, y

            dist = chamfer(desired_shape, current_shape)
            # compute chamfer distance
            # c_dist = scipy.spatial.distance.cdist(current_shape, desired_shape, metric='euclidean') ** 2
            # dist = c_dist.min(0).mean()
            # dist += c_dist.min(1).mean()

            if self.debug_rewards:
                reward.update({'chamfer': -self.cost_coeff[2]*dist})
            else:
                reward -= self.cost_coeff[2]*dist

        if 'IoU' in self.reward or self.debug_rewards:
            set1 = self.fixed_half    #
            set2 = data.reshape(-1,3)[self.moving_half_idx]
            iou = compute_iou(set1, set2, grid_size=self.grid_res)
            # grids = measure_area(data, grid_resolution=self.grid_res)[1]
            # goal_grids = self.desired_grid
            # intersection = np.sum(np.logical_and(grids, goal_grids).reshape(1, -1), axis=1)
            # union = np.sum(np.logical_or(grids, goal_grids).reshape(1, -1), axis=1)
            # iou = intersection / union

            # if self.debug_rewards:
            #     reward.update({'IoU': (iou + self.cost_coeff[3])})
            # else:
            #     reward += (iou + self.cost_coeff[3])

            if self.debug_rewards:
                reward.update({'IoU': (iou )})
            else:
                reward += iou

        if 'Gr' in self.reward:
            # attractor for the gripper when close to the place
            g_dist = np.linalg.norm((self.sphere_ee.get_base_pos_orient()[0] - self.pick))
            if g_dist < self.gripper_attractor_tresh:
                reward += -g_dist

        # visualize the z penalty
        if self.debug_rewards:
            z = data[:, 2]
            flatness = z.sum()
            reward.update({'flat': -flatness})


        return reward


    def reset(self, stiffness=[20., 20., 1.5], friction=1.2, cloth_scale=0.1, cloth_mass=0.5):     # Siffness = [elas, bend, damp]
        super(HalfFoldEnv, self).reset()

        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')


        # self.cloth_attachment = self.create_sphere(radius=0.02, mass=0, pos=[0.4, -0.35, 1.05], visual=True, collision=False, rgba=[0, 0, 0, 1], maximal_coordinates=False)
        self.cloth = p.loadSoftBody(
            # os.path.join(self.directory, 'clothing', 'gown_696v.obj'),
            os.path.join(self.directory, 'clothing', 'bl_cloth_25_cuts.obj'),
            # os.path.join(self.directory, 'clothing', 'square25.obj'),
            # scale=1.0,
            scale=cloth_scale,
            mass=cloth_mass,
            useBendingSprings=1,
            useMassSpring=1,
            # springElasticStiffness=5,
            springElasticStiffness=stiffness[0],
            # springDampingStiffness=0.01,
            springDampingStiffness=stiffness[2],
            # springDampingAllDirections=1,
            springDampingAllDirections=0,
            springBendingStiffness=stiffness[1],
            useNeoHookean=0,
            useSelfCollision=1,
            collisionMargin=0.0001,
            # frictionCoeff=0.1,
            frictionCoeff=friction,
            useFaceContact=1,
            physicsClientId=self.id)
        # p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.5], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 1], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        # p.setPhysicsEngineParameter(numSubSteps=8, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=16, physicsClientId=self.id)

        axis_angle = [np.pi / 2, 0, -np.pi / 2]
        if self.side == 1:
            axis_angle = [np.pi / 2, 0, np.pi]
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]
        offset = np.zeros(3)
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
                                          physicsClientId=self.id)

        # vertex_index = 680
        # vertex_index = 624
        vertex_index = 24 # 624-25
        self.corners = [0, 24, 600, 624]
        if self.side == 1:
            self.corners = [24, 624, 0, 600]

        # axis_angle = [np.pi / 2, 0, 0]
        # # offset = np.asarray([0.35069006, -0.3483985,  1.05122447])        # Previous version
        # self.z_offset = 0.02
        # self.z_workspace[0] = self.z_offset
        # offset = np.asarray([0., 0., self.z_offset])
        # p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
        #                                   physicsClientId=self.id)

        # Place grasped corner in the correct position
        axis_angle = [np.pi / 2, 0, -np.pi / 2]
        if self.side == 1:
            axis_angle = [np.pi / 2, 0, np.pi]
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        current_corner_pos = np.array(data[1][vertex_index])

        if self.apply_rw_offset:
            grasped_corner_position = [0.3, 0, 0.15]         # TODO: this is the grasped position in the real world
        else:
            grasped_corner_position = current_corner_pos + np.asarray([0., 0, 0.1])

        offset = np.asarray(grasped_corner_position) - current_corner_pos
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
                                          physicsClientId=self.id)

        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)


        # * spawn sphere manipulator
        # data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # vertex_position = np.array(data[1][vertex_index])
        # position = vertex_position
        # self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos=position, visual=True, collision=True,
        #                                     rgba=[0, 0, 1, 1])
        # self.sphere_ee.set_base_pos_orient(vertex_position, np.array([0, 0, 0]))
        #
        # p.createSoftBodyAnchor(self.cloth, vertex_index, self.sphere_ee.body, -1, [0, 0, 0],
        #                        physicsClientId=self.id)
        #

        ########## PICK ##########
        # position_gripper_rw = grasped_corner_position
        self.gripper_init_pos = np.asarray([0.2, 0.2, 0.2])
        # OLD
        self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos=self.gripper_init_pos, visual=True, collision=True,
                                            rgba=[0, 0, 1, 1])

        # self.sphere_ee, self.sphere_ee_joint = self.create_joint_sphere(radius=0.01, mass=0.10, pos=self.gripper_init_pos, visual=True,
        #                                     collision=True,
        #                                     rgba=[0, 0, 1, 1])

        # self.sphere_ee.set_base_pos_orient(position_gripper_rw, np.array([0, 0, 0]))
        #
        # p.createSoftBodyAnchor(self.cloth, vertex_index, self.sphere_ee.body, -1, [0, 0, 0],
        #                        physicsClientId=self.id)


        # * initialize env variables
        from gym import spaces
        # * update observation and action spaces
        obs_len = len(self._get_obs())
        self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32) * 1000000000,
                                        high=np.ones(obs_len, dtype=np.float32) * 1000000000, dtype=np.float32)

        self.action_space.__init__(low=-np.ones(self.action_len, dtype=np.float32),
                                   high=np.ones(self.action_len, dtype=np.float32), dtype=np.float32)
        # * Define action/obs lengths
        self.action_robot_len = 3
        self.action_space_robot = spaces.Box(low=np.array([-1.0] * self.action_robot_len, dtype=np.float32),
                                             high=np.array([1.0] * self.action_robot_len, dtype=np.float32),
                                             dtype=np.float32)


        # Create a table
        if self.use_table:
            self.table = Furniture()
            self.table.init('table', self.directory, self.id, self.np_random, rw=False)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.id)

        # Initial position half cloth to make sure its not moving:
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))

        # Get half shapes
        half1 = data[:int(data.shape[1] / 2), :, :]
        half2 = data[int(data.shape[1] / 2):, :, :]
        self.moving_half_idx = np.where(data.reshape(-1, 3)[:, 1] > 0.)
        self.fixed_half_idx = np.where(data.reshape(-1, 3)[:, 1] <= 0.)
        self.fixed_half = data.reshape(-1, 3)[self.fixed_half_idx]
        self.init_pos_half = half2
        # TODO: make the code agnostic to the goal, right now it cannot be changed
        # rotate only for the zero pos
        if self.side == 0:
            theta = np.radians(90)
            # Define the rotation matrix
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.init_pos_half = np.matmul(self.init_pos_half, R)
        self.desired_area, self.desired_grid, self.desired_depth = measure_area(self.init_pos_half.reshape(-1, 3),
                                                                       grid_resolution=self.grid_res)
        self.init_area, self.init_grid, self.init_depth = measure_area(self.get_mesh().reshape(-1,3), grid_resolution=self.grid_res)


        self.time = time.time()

        self.iteration = 0

        return self._get_obs()

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=45, camera_width=1920//4, camera_height=1080//4):
        cameraDistance = camera_eye[2] - camera_target[2]
        # -90 does not visualize the correct posi
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=camera_target, physicsClientId=self.id)
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
                                                       cameraTargetPosition=camera_target, distance=cameraDistance, yaw=0,
                                                       pitch=-90, roll=0, upAxisIndex=2)

        # self.create_sphere()
            #p.computeViewMatrix(camera_eye, camera_target, [0, 1, 0], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)
        # TODO: add camera config
        # self.camera_params = {'default_camera': {'pos': np.asarray([0., 0.65, 0.]), 'angle': np.asarray([0., -1.57079633, 0.]), 'width': 720, 'height': 720}}
        self.camera_params = {'default_camera': {'pos': np.asarray([0, 0, 0.65]), 'angle': np.asarray([0, 0, 1.57079633]), 'width': 720, 'height': 720}}


    def render_image(self, light_pos=[0, -3, 1], shadow=False, ambient=0.8, diffuse=0.3, specular=0.1):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, rgb, depth, segmask = p.getCameraImage(self.camera_width, self.camera_height,
                                                     self.view_matrix, self.projection_matrix,
                                                     shadow=shadow,
                                                     lightDirection=light_pos, lightAmbientCoeff=ambient,
                                                     lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.id)
        rgb = np.reshape(rgb, (h, w, 4)).astype(np.uint8)
        assert h == 720
        depth = np.reshape(depth, (h, w))
        depth = 100 * 0.01 / (100 - (100 - 0.01) * depth)

        # print(depth[np.where(segmask == self.sphere_ee.body)[0][0], np.where(segmask == self.sphere_ee.body)[1][0]])
        # print(depth[np.where(segmask == self.cloth)[0][0], np.where(segmask == self.cloth)[1][0]])
        # print(depth[np.where(segmask != self.cloth)[0][0], np.where(segmask != self.cloth)[1][0]])

        rgb = rgb.reshape((720, 720, 4))[:, :, :3]#[::-1, :, :3]
        depth = depth.reshape((720, 720))#[::-1]
        rgb = cv2.resize(rgb.astype('float32'), (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth.astype('float32'), (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)

        return rgb, depth


    def get_corners(self):
        corner_ids = np.array([0, 24, 600, 624])
        if self.side == 1:
            corner_ids = np.array([24, 624, 0, 600])
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        corners = data[corner_ids, :3]
        return corners


    def get_center(self):
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        corners = data[312, :3]
        return corners


    def get_edge_middles(self):
        corner_ids = np.array([12, 300, 324, 612])
        if self.side == 1:
            corner_ids = np.array([300, 612, 12, 324])
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        edge_middle_pos = data[corner_ids, :3]
        return edge_middle_pos


    def pick(self, pick):
        # find closest corner to pick (currently not needed:TODO)
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        distances = np.linalg.norm(data - pick, axis=1)
        vertex_index = np.argmin(distances)
        # pick
        self.sphere_ee.set_base_pos_orient(pick, np.array([0, 0, 0]))
        self.anchor_id = p.createSoftBodyAnchor(self.cloth, vertex_index, self.sphere_ee.body, -1, [0, 0, 0],
                               physicsClientId=self.id)

    def place(self):
        p.removeConstraint(self.anchor_id)
        # back to original position
        self.sphere_ee.set_base_pos_orient(self.gripper_init_pos, np.array([0, 0, 0]))

        # let the cloth settle
        for _ in range(20):
            p.stepSimulation(self.id)

        reward_cloth_matching = self._get_reward()

        reward = reward_cloth_matching

        obs = self._get_obs()

        # if self.gui:
        #     print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)

        info = {}

        done = True

        return obs, reward, done, info
        #
    def pick_and_place(self, pick, place, midpoint=None, K=3):
        # find closest corner to pick (currently not needed:TODO)
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        distances = np.linalg.norm(data - pick, axis=1)
        vertex_index = np.argmin(distances)
        # pick
        self.sphere_ee.set_base_pos_orient(pick, np.array([0, 0, 0]))
        anchor_id = p.createSoftBodyAnchor(self.cloth, vertex_index, self.sphere_ee.body, -1, [0, 0, 0],
                               physicsClientId=self.id)

        # probe
        z_offset = np.linalg.norm(place-pick)/2
        pick_place_dir = place - pick
        probe_action_dir = np.asarray([pick_place_dir[0], pick_place_dir[1], z_offset])
        probe_action = probe_action_dir / np.linalg.norm([probe_action_dir]) * self.velocity * 2

        for i in range(K):
            action = probe_action / K
            obs, reward, done, info = self.step(action=action)
            gripper_pos = self.sphere_ee.get_base_pos_orient()[0]
            # TODO: store trajectory at some point if needed


        # get midpoint
        if midpoint is None:
            midway = (np.asarray(pick) + np.asarray(place))/2
            midpoint = np.asarray([midway[0], midway[1], z_offset])

        # We assume that we always probe
        waypoints = np.asarray([gripper_pos, midpoint, place])

        # this function is defined in utils of assistive gym
        trajectory = Trajectory(
                 waypoints,
                 vel=self.velocity,
                 interpole=True,
                 multi_traj=False,
                 action_scale=1,)

        # go to midpoint
        # go to place
        # Currently these two steps are executed together but we might need to split it for the data collection
        for action in trajectory.actions:
            obs, reward, done, info = self.step(action)
        # release pick
        # p.changeConstraint(anchor_id, maxForce=0)
        p.removeConstraint(anchor_id)
        # back to original position
        self.sphere_ee.set_base_pos_orient(self.gripper_init_pos, np.array([0, 0, 0]))
        #
        print()



from .agents.pr2 import PR2
# robot_arm = 'left'


# class HalfFoldEnvPR2Env(HalfFoldEnv):
#     def __init__(self, frame_skip=10):
#         super(HalfFoldEnvPR2Env, self).__init__(robot=PR2(robot_arm), human=None, use_ik=True, frame_skip=frame_skip)

