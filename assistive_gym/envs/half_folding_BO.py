import copy
import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from gym import spaces
import scipy

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import furniture
from .agents.furniture import Furniture

import time

class HalfFoldEnv(AssistiveEnv):
    def __init__(self, robot=None, human=None, use_ik=False, frame_skip=10, hz=480, obs='mesh', reward='corner', use_mesh=False, action_mult=0.01):
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

        self.time_step = 1.0 / hz
        p.setTimeStep(self.time_step)

        # TODO: this might be variable at some point
        self.obs_clot_len = 625 * 3
        self.observation_space = spaces.Box(low=np.array([-1000000000.0]*(self.obs_clot_len), dtype=np.float32), high=np.array([1000000000.0]*(self.obs_clot_len), dtype=np.float32), dtype=np.float32)
        self.action_len = 3
        self.action_space.__init__(low=-np.ones(self.action_len, dtype=np.float32),
                                   high=np.ones(self.action_len, dtype=np.float32), dtype=np.float32)

        self.action_multiplier = action_mult

        self.x_workspace = [-0.5, 0.5]
        self.y_workspace = [-0.5, 0.5]
        self.z_workspace = [0.01, 2.]

        self.camera = None
        self.reward = reward
        self.obs_type = obs

        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS
        # self.reset(stiffness=[40, 20., 1.5], friction=1.2, cloth_scale=0.1, cloth_mass=0.5)

    def process_pointclouds(self, RGB_threshold=[0.78, 0.78, 0.78], voxel_size=0.005):
        seg_pcds = self.camera.get_segmented_pointclouds(RGB_threshold=RGB_threshold)
        front_pcd = seg_pcds[0]
        back_pcd = seg_pcds[1]
        partial_voxelized = self.camera.voxelize_pcd(front_pcd, voxel_size=voxel_size)
        full_voxelized = self.camera.voxelize_pcd(np.concatenate((front_pcd, back_pcd), 0), voxel_size=voxel_size)

        return partial_voxelized, full_voxelized
    def step(self, action):
        # start_time = time.time()
        # print(f'step {self.iteration}')
        # print(f'Pybullet timestep: {p.getPhysicsEngineParameters()["fixedTimeStep"]}')
        if self.robot is not None:
            if self.use_ik:
                t = time.time()
                self.take_step(action, action_multiplier=0.05, ik=True)
                # print(f'take_step execution time: {time.time()-t}')
            else:
                self.take_step(action, action_multiplier=0.003)
        else:
            self.take_step_sphere(action, action_multiplier=1, num_sphere_steps=1)        # TODO: decide the action multiplier
            # self.take_step(action, action_multiplier=0.003)

        # print(f'Total step time : {time.time() - start_time}')

        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_cloth_matching = self._get_reward()

        reward = reward_cloth_matching

        obs = self._get_obs()

        # if self.gui:
        #     print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)

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

        past_pose = copy.deepcopy(current_pos)
        current_pos += action
        self.sphere_ee.set_base_pos_orient(current_pos, np.array([0, 0, 0]))
        p.stepSimulation(physicsClientId=self.id)

        # Let the cloth settle
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.id)
        self.iteration += 1

        # print(f'Moved by {self.sphere_ee.get_base_pos_orient()[0] - past_pose}')


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
    def _get_reward_old(self):
        # TODO: we have two different types of reward, corner matching and shape matching
        data = self.get_mesh()

        # Get half shapes
        half1 = data[:int(data.shape[1]/2), :, :]
        half2 = data[int(data.shape[1] / 2):, :, :]

        reward = 0

        # Compute distances
        c1 = 1.
        # half2_flip = np.flip(half2, axis=0)      # Mirror one dimension to parallelize the computation
        half2_flip = half2[::-1, :, :]
        norm = np.linalg.norm(half1 - half2_flip[:half1.shape[0], :, :], axis=-1)
        reward -= np.sum(norm)*c1


        # compute displacement second half
        c2 = 0.
        norm = np.linalg.norm(half2 - self.init_pos_half, axis=-1)
        reward -= np.sum(norm)*c2

        # compute gripper distance to opposite corner
        c3 = 1
        gripper = half1.reshape(-1, 3)[24]
        opposite_corner = half2.reshape(-1, 3)[-1]
        norm = np.linalg.norm(gripper - opposite_corner)
        reward -= norm * c3

        return reward

    def _get_reward(self):
        reward = 0
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])

        # Corner matching
        if self.reward == 'corner':
            # get corner positions
            left_corners = np.asarray([data[self.corners[0]], data[self.corners[1]]])
            right_corners = np.asarray([data[self.corners[2]], data[self.corners[3]]])
            norm = np.linalg.norm(left_corners - right_corners)
            reward -= norm

        # compute distances corresponding corners

        # Shape matching
        if self.reward == 'shape':
            # Get desired shape (2D)
            desired_shape = self.init_pos_half.reshape(-1, 3)[:, :2]    # take only x,y
            # Get current shape (2D)
            current_shape = data[:, :2]         # take only x, y
            # compute chamfer distance
            c_dist = scipy.spatial.distance.cdist(current_shape, desired_shape, metric='euclidean') ** 2
            dist = c_dist.min(0).mean()
            dist += c_dist.min(1).mean()

            reward -= dist


        return reward


    def reset(self, stiffness=[20., 20., 1.5], friction=1.2, cloth_scale=0.1, cloth_mass=0.5):     # Siffness = [elas, bend, damp]
        super(HalfFoldEnv, self).reset()

        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')

        if self.robot is not None:
            # Update robot motor gains
            self.robot.motor_gains = 0.05
            self.robot.motor_forces = 100.0

            base_pos = [0.87448014, 0.40101663, 0]
            base_orient = [0, 0, 0.99178843, -0.12788948]
            joint_angles = [1.14317203, 0.18115511, 1.36771027, -0.77079951, 0.28547459, -0.6480673, -1.58786233]
            self.robot.reset_joints()
            self.robot.set_base_pos_orient(base_pos, base_orient)
            self.robot.set_joint_angles(self.robot.controllable_joint_indices, joint_angles)


        # self.cloth_attachment = self.create_sphere(radius=0.02, mass=0, pos=[0.4, -0.35, 1.05], visual=True, collision=False, rgba=[0, 0, 0, 1], maximal_coordinates=False)
        self.cloth = p.loadSoftBody(
            # os.path.join(self.directory, 'clothing', 'gown_696v.obj'),
            os.path.join(self.directory, 'clothing', 'bl_cloth_25_cuts.obj'),
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
        # p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=8, physicsClientId=self.id)

        axis_angle = [np.pi / 2, 0, -np.pi / 2]
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        offset = np.zeros(3)
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
                                          physicsClientId=self.id)

        # vertex_index = 680
        # vertex_index = 624
        vertex_index = 24 # 624-25
        self.corners = [0, 24, 600, 624]
        # anchor_vertices = [307, 300, 603, 43, 641, 571]
        # anchor_vertices = [624, 623, 624 - 25, 623 - 25] #624 - 25, 623 np.pi, np.pi-25]
        # if vertex_index == 0:
        #     anchor_vertices = [0, 1, 25, 26] #624 - 25, 623 -25]
        # if vertex_index == 624:
        #     anchor_vertices = [624, 623, 624 - 25, 623 - 25] #624 - 25, 623 -25]

        # self.triangle1_point_indices = [80, 439, 398]
        # self.triangle2_point_indices = [245, 686, 355]

        if self.robot is not None:
            # Move cloth grasping vertex into robot end effectors
            axis_angle = [-np.pi / 2, 0, 0]
            p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion(axis_angle),
                                              physicsClientId=self.id)
            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            vertex_position = np.array(data[1][vertex_index])
            offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
            p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
                                              physicsClientId=self.id)
            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            new_vertex_position = np.array(data[1][vertex_index])

            # NOTE: Create anchors between cloth and robot end effector
            p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0],
                                   physicsClientId=self.id)

            # for i in anchor_vertices:
            #     pos_diff = np.array(data[1][i]) - new_vertex_position
            #     p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

            self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)

            # Disable collisions between robot and cloth
            for i in [-1] + self.robot.all_joint_indices:
                p.setCollisionFilterPair(self.robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)
            self.robot.set_gravity(0, 0, 0)

        if self.robot is None:
            # axis_angle = [np.pi / 2, 0, 0]
            # # offset = np.asarray([0.35069006, -0.3483985,  1.05122447])        # Previous version
            # self.z_offset = 0.02
            # self.z_workspace[0] = self.z_offset
            # offset = np.asarray([0., 0., self.z_offset])
            # p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
            #                                   physicsClientId=self.id)

            # Place grasped corner in the correct position
            grasped_corner_position = [0.3, 0, 0.15]

            # TODO: define orientation
            axis_angle = [np.pi / 2, 0, -np.pi / 2]
            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            current_corner_pos = np.array(data[1][vertex_index])
            offset = np.asarray(grasped_corner_position) - current_corner_pos
            p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
                                              physicsClientId=self.id)

            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            new_corner_pos = np.array(data[1][vertex_index])
            # print(f'Poisition grasped corner: {new_corner_pos}')

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

            position_gripper_rw = [0.3, 0, 0.15]
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos=position_gripper_rw, visual=True, collision=True,
                                                rgba=[0, 0, 1, 1])
            self.sphere_ee.set_base_pos_orient(position_gripper_rw, np.array([0, 0, 0]))

            p.createSoftBodyAnchor(self.cloth, vertex_index, self.sphere_ee.body, -1, [0, 0, 0],
                                   physicsClientId=self.id)



            # self.sphere_origin = self.create_sphere(radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True,
            #                                     rgba=[0, 0, 0, 1])




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
            # self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            # self.obs_robot_len = len(self._get_obs('robot'))  # 1
            # self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0] * self.action_robot_len, dtype=np.float32),
                                                 high=np.array([1.0] * self.action_robot_len, dtype=np.float32),
                                                 dtype=np.float32)
            # self.action_space_human = spaces.Box(low=np.array([-1.0] * self.action_human_len, dtype=np.float32),
            #                                      high=np.array([1.0] * self.action_human_len, dtype=np.float32),
            #                                      dtype=np.float32)
            # self.observation_space_robot = spaces.Box(
            #     low=np.array([-1000000000.0] * self.obs_robot_len, dtype=np.float32),
            #     high=np.array([1000000000.0] * self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            # self.observation_space_human = spaces.Box(
            #     low=np.array([-1000000000.0] * self.obs_human_len, dtype=np.float32),
            #     high=np.array([1000000000.0] * self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()

        # Create a table
        self.table = Furniture()
        self.table.init('table', self.directory, self.id, self.np_random)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        # for _ in range(10):
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.id)

        # Initial position half cloth to make sure its not moving:
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))

        # Get half shapes
        half1 = data[:int(data.shape[1] / 2), :, :]
        half2 = data[int(data.shape[1] / 2):, :, :]
        self.init_pos_half = half2

        self.time = time.time()

        self.iteration = 0

        return self._get_obs()



from .agents.pr2 import PR2
# robot_arm = 'left'


# class HalfFoldEnvPR2Env(HalfFoldEnv):
#     def __init__(self, frame_skip=10):
#         super(HalfFoldEnvPR2Env, self).__init__(robot=PR2(robot_arm), human=None, use_ik=True, frame_skip=frame_skip)

