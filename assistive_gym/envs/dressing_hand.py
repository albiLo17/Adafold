import copy
import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from gym import spaces

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import furniture
from .agents.furniture import Furniture

import time

class DressingHandEnv(AssistiveEnv):
    def __init__(self, robot=None, human=None, use_ik=False, frame_skip=10, hz=480, use_mesh=False, action_mult=0.01):
        assert human is None, "For now just consider no human!"
        if robot is None:
            super(DressingHandEnv, self).__init__(robot=None, human=human, task='folding', obs_robot_len=12, obs_human_len=0, frame_skip=frame_skip, time_step=0.01, deformable=True)
            self.use_mesh = use_mesh
        else:
            super(DressingHandEnv, self).__init__(robot=robot, human=None, task='folding',
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

        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS
        # self.reset(stiffness=[40, 20., 1.5], friction=1.2, cloth_scale=0.1, cloth_mass=0.5)

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
            self.take_step_sphere(action, action_multiplier=self.action_multiplier, num_sphere_steps=1)        # TODO: decide the action multiplier
            # self.take_step(action, action_multiplier=0.003)

        # print(f'Total step time : {time.time() - start_time}')

        reward_action = -np.linalg.norm(action) # Penalize actions
        # reward_cloth_matching = self._get_reward()

        # reward = reward_cloth_matching
        reward = 0

        obs = self._get_obs()

        # if self.gui:
        #     print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)

        info = {}

        done = self.iteration >= 100

        # if not done:
        #     done = self.check_done()

        return obs, reward, done, info

    def take_step_sphere(self, action, action_multiplier=1., num_sphere_steps=1):
        # start_time = time.time()
        # action is 7 dimensional so lets use the 3 components
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        final_pos = current_pos + action[:3] * action_multiplier
        final_pos[2] = np.clip(final_pos[2], self.z_workspace[0], self.z_workspace[1])

        action = final_pos - current_pos
        # print(action)

        # error = action[:3] * action_multiplier
        # for i in range(num_sphere_steps):
            # p.setJointMotorControl2(bodyUniqueId=self.sphere_ee.body,
            #                         jointIndex=0,
            #                         controlMode=p.POSITION_CONTROL,
            #                         targetVelocity=-4.0 * error,
            #                         force=500000,
            #                         physicsClientId=self.id)

            # Gains and limits for a simple controller for the anchors.
            # CTRL_MAX_FORCE = 10  # 10
            # CTRL_PD_KD = 50.0  # 50
            #
            # anc_linvel, _ = p.getBaseVelocity(self.sphere_ee.body)
            # print(f'vel - {anc_linvel}')
            # vel_diff = error - np.array(anc_linvel)
            # raw_force = CTRL_PD_KD * vel_diff
            # force = np.clip(raw_force, -1.0 * CTRL_MAX_FORCE, CTRL_MAX_FORCE)
            # print(f'force - {force}')
            # # p.setRealTimeSimulation(0)
            # p.applyExternalForce(
            #     self.sphere_ee.body, -1, force.tolist(), [0, 0, 0], p.LINK_FRAME)



        # for i in range(num_sphere_steps):
        # while np.linalg.norm(current_pos -final_pos) > 0.01:
        past_pose = copy.deepcopy(current_pos)
        current_pos += action
        # print(f'Total time before step: {time.time() - start_time}')
        # start_time = time.time()
        self.sphere_ee.set_base_pos_orient(current_pos, np.array([0, 0, 0]))
        p.stepSimulation(physicsClientId=self.id)

        # print(f'Total time step1: {time.time() - start_time}')
        # start_time = time.time()
        # Let the cloth settle
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.id)
        # print(f'Total time after last step: {time.time() - start_time}')
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

        # TODO: get depth images and convert them into pointcloud
        cloth_mesh = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]).flatten()


        return cloth_mesh

    def get_mesh(self, data=None):
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        # data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))

        return data
    def _get_reward(self):
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


    def reset(self, stiffness=[20., 20., 1.5], friction=1.2, cloth_scale=0.1, cloth_mass=0.5):     # Siffness = [elas, bend, damp]
        super(DressingHandEnv, self).reset()

        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')

        # if self.robot is not None:
        #     # Update robot motor gains
        #     self.robot.motor_gains = 0.05
        #     self.robot.motor_forces = 100.0
        #
        #     base_pos = [0.87448014, 0.40101663, 0]
        #     base_orient = [0, 0, 0.99178843, -0.12788948]
        #     joint_angles = [1.14317203, 0.18115511, 1.36771027, -0.77079951, 0.28547459, -0.6480673, -1.58786233]
        #     self.robot.reset_joints()
        #     self.robot.set_base_pos_orient(base_pos, base_orient)
        #     self.robot.set_joint_angles(self.robot.controllable_joint_indices, joint_angles)


        # self.cloth_attachment = self.create_sphere(radius=0.02, mass=0, pos=[0.4, -0.35, 1.05], visual=True, collision=False, rgba=[0, 0, 0, 1], maximal_coordinates=False)
        self.cloth = p.loadSoftBody(
            # os.path.join(self.directory, 'clothing', 'gown_696v.obj'),
            os.path.join(self.directory, 'clothing', 'sleeve_585v.obj'),
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
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.7], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)
        # vertex_index = 680
        # vertex_index = 624
        vertex_index = 24 # 624-25



        # anchor_vertices = [307, 300, 603, 43, 641, 571]
        # anchor_vertices = [624, 623, 624 - 25, 623 - 25] #624 - 25, 623 np.pi, np.pi-25]
        # if vertex_index == 0:
        #     anchor_vertices = [0, 1, 25, 26] #624 - 25, 623 -25]
        # if vertex_index == 624:
        #     anchor_vertices = [624, 623, 624 - 25, 623 - 25] #624 - 25, 623 -25]

        # self.triangle1_point_indices = [80, 439, 398]
        # self.triangle2_point_indices = [245, 686, 355]


        if self.robot is None:
            axis_angle = [np.pi / 2, 0, 0]
            offset = np.asarray([0., 0,  1.05122447])        # Previous version
            # self.z_offset = 0.02
            # self.z_workspace[0] = self.z_offset
            # offset = np.asarray([0., 0., self.z_offset])
            p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle),
                                              physicsClientId=self.id)
            # * spawn sphere manipulator
            data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
            vertex_position = np.array(data[1][vertex_index])
            position = vertex_position
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos=position, visual=True, collision=True,
                                                rgba=[0, 0, 1, 1])
            self.sphere_ee.set_base_pos_orient(vertex_position, np.array([0, 0, 0]))

            # p.createSoftBodyAnchor(self.cloth, vertex_index, self.sphere_ee.body, -1, [0, 0, 0],
            #                        physicsClientId=self.id)
            anchor_verteces = [vertex_index - 2, vertex_index - 1,
                               vertex_index, vertex_index + 1, vertex_index + 2,]
            for av in anchor_verteces:
                p.createSoftBodyAnchor(self.cloth, av, self.sphere_ee.body, -1, [0, 0, 0],
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

        # # Create a table
        # self.table = Furniture()
        # self.table.init('table', self.directory, self.id, self.np_random)

        self.create_sphere(radius=0.1, pos=[0., -0., 1.2])

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.id)

        # Initial position half cloth to make sure its not moving:
        # data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        # data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))
        #
        # # Get half shapes
        # half1 = data[:int(data.shape[1] / 2), :, :]
        # half2 = data[int(data.shape[1] / 2):, :, :]
        # self.init_pos_half = half2

        self.time = time.time()

        self.iteration = 0

        return self._get_obs()





from .agents.pr2 import PR2
# robot_arm = 'left'


# class HalfFoldEnvPR2Env(HalfFoldEnv):
#     def __init__(self, frame_skip=10):
#         super(HalfFoldEnvPR2Env, self).__init__(robot=PR2(robot_arm), human=None, use_ik=True, frame_skip=frame_skip)

