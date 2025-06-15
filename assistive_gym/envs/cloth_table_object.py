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

class ClothTableFoldEnv(AssistiveEnv):
    def __init__(self, robot, human=None, use_ik=False, frame_skip=10):
        assert human is None, "For now just consider no human!"
        super(ClothTableFoldEnv, self).__init__(robot=robot, human=None, task='dressing',
            obs_robot_len=(16 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), 
            obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)), 
            frame_skip=frame_skip, time_step=1./480, deformable=True)

        self.use_ik = use_ik
        self.use_mesh = (human is None)
        hz=480
        p.setTimeStep(1.0 / hz)

        # TODO: this might be variable at some point
        self.obs_clot_len = 625 * 3
        self.observation_space = spaces.Box(low=np.array([-1000000000.0]*(self.obs_clot_len), dtype=np.float32), high=np.array([1000000000.0]*(self.obs_clot_len), dtype=np.float32), dtype=np.float32)


        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS

    def step(self, action):
        if self.use_ik:
            t = time.time()
            self.take_step(action, action_multiplier=0.05, ik=True)
            # print(f'take_step execution time: {time.time()-t}')
        else:
            self.take_step(action, action_multiplier=0.003)


        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_cloth_matching = self._get_reward()

        reward = reward_cloth_matching

        # TODO: get pointcloud or mesh
        obs = self._get_obs()

        # if self.gui:
        #     print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)

        info = {     }

        done = self.iteration >= 200

        if not done:
            done = self.check_done()

        return obs, reward, done, info

    def check_done(self):
        # the grasped corner is close to the final one
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])

        threshold = 0.01
        if np.linalg.norm(data[24] - data[624]) < threshold:
            return True
        return False

    def _get_obs(self, agent=None):
        # UNCOMMENT this in case we want robot positions as observations
        # end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        # end_effector_pos_real, end_effector_orient_real = self.robot.convert_to_realworld(end_effector_pos, end_effector_orient)
        # robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # # Fix joint angles to be in [-pi, pi]
        # robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        # if self.robot.mobile:
        #     # Don't include joint angles for the wheels
        #     robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        # robot_obs = np.concatenate([
        #     end_effector_pos_real, end_effector_orient_real, robot_joint_angles
        # ]).ravel()

        # self.cloth_force_sum = np.sum(np.linalg.norm(self.cloth_forces, axis=-1))
        # self.robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        # self.total_force_on_human = self.robot_force_on_human + self.cloth_force_sum



        # robot_force_on_cloth = np.sum(self.robot.get_contact_points(self.cloth)[-1])
        # print("robot force on cloth: ", robot_force_on_cloth)

        # ret = p.getContactPoints(self.sphere.body, self.cloth)
        # print("getContactPoints between self.robot and self.cloth: ", ret)
        # input("press enter to continue")

        # TODO: get depth images and convert them into pointcloud
        cloth_mesh = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1]).flatten()


        return cloth_mesh

    def get_mesh(self, data=None):
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))
        return data
    def _get_reward(self):
        data = self.get_mesh()

        # Get half shapes
        half1 = data[:int(data.shape[1]/2), :, :]
        half2 = data[int(data.shape[1] / 2):, :, :]

        reward = 0

        # Compute distances
        c1 = 0.1
        # half2_flip = np.flip(half2, axis=0)      # Mirror one dimension to parallelize the computation
        half2_flip = half2[::-1, :, :]
        norm = np.linalg.norm(half1 - half2_flip[:half1.shape[0], :, :], axis=-1)
        reward = - np.sum(norm)*c1


        # compute displacement second half
        c2 = 0.1
        norm = np.linalg.norm(half2 - self.init_pos_half, axis=-1)
        reward -= np.sum(norm)*c2


        return reward


    def reset(self, stiffness=[40, 100., 1.5], friction=1.):     # Siffness = [elas, bend, damp]
        super(ClothTableFoldEnv, self).reset()
        # TODO: no need to build a human there
        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')

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
            scale=0.1,
            mass=0.5, 
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
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.3], flags=0, physicsClientId=self.id)
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

        self.triangle1_point_indices = [80, 439, 398]
        self.triangle2_point_indices = [245, 686, 355]

        # Move cloth grasping vertex into robot end effectors
        axis_angle = [np.pi / 2, 0, 0]
        p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        new_vertex_position = np.array(data[1][vertex_index])

        # NOTE: Create anchors between cloth and robot end effector
        p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)
        # for i in anchor_vertices:
        #     pos_diff = np.array(data[1][i]) - new_vertex_position
        #     p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

        self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)

        # Disable collisions between robot and cloth
        # for i in [-1] + self.robot.all_joint_indices:
        #     p.setCollisionFilterPair(self.robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)
        self.robot.set_gravity(0, 0, 0)

        # Create a table
        self.table = Furniture()
        self.table.init('table', self.directory, self.id, self.np_random)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)

        # Initial position half cloth to make sure its not moving:
        data = np.asarray(p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)[1])
        data = data.reshape((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0])), 3))

        # Get half shapes
        half1 = data[:int(data.shape[1] / 2), :, :]
        half2 = data[int(data.shape[1] / 2):, :, :]
        self.init_pos_half = half2

        self.time = time.time()
        self.init_env_variables()
        return self._get_obs()


class ClothTableObjectsEnv(AssistiveEnv):
    # TODO:
    # 1. remove -- done
    # 1.1. remove wheelchair -- done
    # 2. add table & bowl & cup - DONE
    # 3. collect force applied to the object
    # 4. collect robot observation
    # 5. design a heuristic strategy for moving cloth along the table (how to control the robot?)

    def __init__(self, robot, human=None, use_ik=False):
        assert human is None, "For now just consider no human!"
        super(ClothTableObjectsEnv, self).__init__(robot=robot, human=None, task='dressing',
                                                   obs_robot_len=(16 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)),
                                                   obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)),
                                                   frame_skip=5, time_step=1. / 480, deformable=True)

        self.use_ik = use_ik
        self.use_mesh = (human is None)
        hz = 480
        p.setTimeStep(1.0 / hz)

        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS

    def step(self, action):
        if self.human is not None:
            if self.human.controllable:
                action = np.concatenate([action['robot'], action['human']])
        if self.use_ik:
            t = time.time()
            self.take_step(action, action_multiplier=0.05, ik=True)
            # print(f'take_step execution time: {time.time()-t}')
        else:
            self.take_step(action, action_multiplier=0.003)

        # NOTE yufei: somehow left_should is no longer defined in human, so I remove it for now
        if self.human is not None:
            # shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
            # elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
            # wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]
            shoulder_pos = np.zeros(3)
            elbow_pos = np.zeros(3)
            wrist_pos = np.zeros(3)

        # Get cloth data
        # x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        # mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        # forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        # contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)

        # NOTE: Uncomment this to visualize contact points between the cloth and the human body
        # x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        # mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        # forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        # contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        # total_force = 0
        # i = 0
        # zero_contact_point = 0
        # non_zero_count = 0
        # if forces.shape[0] > 0:
        #     max_force = np.max(np.linalg.norm(forces, axis=1))
        #     print("max_force is: ", max_force)
        #     for cp, f in zip(contact_positions, forces):
        #         if i >= len(self.points):
        #             break
        #         # if not np.array_equal(f, np.zeros(3)):
        #         self.points[i].set_base_pos_orient(cp, [0, 0, 0, 1])
        #         color = plt.cm.jet(min(np.linalg.norm(f)/0.5, 1.0)) - np.array([0, 0, 0, 0.5])
        #         # print(np.linalg.norm(f))
        #         # print(max_force)
        #         c = np.linalg.norm(f)/max_force
        #         # color = plt.cm.jet(np.array([c, c, c, c]))
        #         if np.array_equal(f, np.zeros(3)):
        #             zero_contact_point += 1
        #             color = np.array([0, 0, 1, 0.2])
        #         else:
        #             color = np.array([1, 0, 0, 1])
        #             non_zero_count += 1
        #
        #         p.changeVisualShape(self.points[i].body, -1, rgbaColor=color, flags=0, physicsClientId=self.id)
        #         if np.linalg.norm(f) > 0:
        #             total_force += np.linalg.norm(f)
        #
        #         i += 1
        #         # print(plt.cm.jet(0.0), plt.cm.jet(1.0))
        #
        #     print("there are {} contact points out of {} cloth points, {} contact points have zero force".format(
        #         i, len(mesh_points), zero_contact_point
        #     ))
        #     if non_zero_count > 0:
        #         print('average non-zero contact force is: ', total_force / non_zero_count)
        #     # input("press enter to continue")

        # print('Time:', time.time() - self.time, 'Force:', total_force)
        # self.time = time.time()
        # for j in range(i, len(self.points)):
        #     self.points[j].set_base_pos_orient([100, 100+j, 100], [0, 0, 0, 1])

        # end_effector_joint_state = p.getJointState(self.robot.body, self.robot.left_end_effector-1)
        # print("left end-effector joint state is: ", end_effector_joint_state)

        # Get 3D points for two triangles around the sleeve to detect if the sleeve is around the arm
        # if self.human is not None:
        #     triangle1_points = mesh_points[self.triangle1_point_indices]
        #     triangle2_points = mesh_points[self.triangle2_point_indices]
        #     forearm_in_sleeve, upperarm_in_sleeve, distance_along_forearm, distance_along_upperarm, distance_to_hand, distance_to_elbow, distance_to_shoulder, forearm_length, upperarm_length = self.util.sleeve_on_arm_reward(triangle1_points, triangle2_points, shoulder_pos, elbow_pos, wrist_pos, self.human.hand_radius, self.human.elbow_radius, self.human.shoulder_radius)
        #     self.forearm_in_sleeve = forearm_in_sleeve
        #     self.upperarm_in_sleeve = upperarm_in_sleeve

        reward_action = -np.linalg.norm(action)  # Penalize actions

        # if self.human is not None:
        #     if self.upperarm_in_sleeve:
        #         reward_dressing = forearm_length
        #         if distance_along_upperarm < upperarm_length:
        #             reward_dressing += distance_along_upperarm
        #     elif self.forearm_in_sleeve and distance_along_forearm < forearm_length:
        #         reward_dressing = distance_along_forearm
        #     else:
        #         reward_dressing = -distance_to_hand
        #     if reward_dressing > self.task_success:
        #         self.task_success = reward_dressing

        reward = 0
        # reward = self.config('dressing_reward_weight')*reward_dressing + self.config('action_weight')*reward_action

        obs = self._get_obs()

        # if self.gui:
        #     print('Task success:', self.task_success, 'Dressing reward:', reward_dressing)

        if self.human is not None:
            info = {
                'total_force_on_human': 0,
                'task_success': min(self.task_success / self.config('task_success_threshold'), 1),
                'action_robot_len': self.action_robot_len,
                'action_human_len': self.action_human_len,
                'obs_robot_len': self.obs_robot_len,
                'obs_human_len': self.obs_human_len
            }
        else:
            info = {

            }

        done = self.iteration >= 200

        if self.human is not None:
            if not self.human.controllable:
                return obs, reward, done, info
            else:
                # Co-optimization with both human and robot controllable
                return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}
        else:
            return obs, reward, done, info

    def _get_obs(self, agent=None):
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        end_effector_pos_real, end_effector_orient_real = self.robot.convert_to_realworld(end_effector_pos, end_effector_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]

        if self.human is not None:
            # NOTE yufei: somehow left_shoulder is no longer defined, so I will just remove it
            # shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
            # elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
            # wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]
            shoulder_pos = np.zeros(3)
            elbow_pos = np.zeros(3)
            wrist_pos = np.zeros(3)

            shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
            elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
            wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)

        # self.cloth_force_sum = np.sum(np.linalg.norm(self.cloth_forces, axis=-1))
        # self.robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        # self.total_force_on_human = self.robot_force_on_human + self.cloth_force_sum
        if self.human is not None:
            robot_obs = np.concatenate([
                end_effector_pos_real, end_effector_orient_real, robot_joint_angles,
                shoulder_pos_real, elbow_pos_real, wrist_pos_real
            ]).ravel()
        else:
            robot_obs = np.concatenate([
                end_effector_pos_real, end_effector_orient_real, robot_joint_angles
            ]).ravel()

        robot_force_on_cloth = np.sum(self.robot.get_contact_points(self.cloth)[-1])
        print("robot force on cloth: ", robot_force_on_cloth)

        # ret = p.getContactPoints(self.sphere.body, self.cloth)
        # print("getContactPoints between self.robot and self.cloth: ", ret)
        # input("press enter to continue")

        if agent == 'robot':
            return robot_obs

        if self.human is not None:
            if self.human.controllable:
                human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
                end_effector_pos_human, end_effector_orient_human = self.human.convert_to_realworld(end_effector_pos, end_effector_orient)
                shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
                elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
                wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
                human_obs = np.concatenate([end_effector_pos_human, end_effector_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human]).ravel()
                if agent == 'human':
                    return human_obs
                # Co-optimization with both human and robot controllable
                return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self, stiffness=[40, 1., 0], friction=1.):  # Siffness = [elas, bend, damp]
        super(ClothTableObjectsEnv, self).reset()
        # TODO: no need to build a human there
        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, np.pi / 2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = 0.05
        self.robot.motor_forces = 100.0

        if self.human is not None:
            if self.use_mesh:
                self.human = HumanMesh()
                joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
                body_shape = np.zeros((1, 10))
                gender = 'female'  # 'random'
                self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

                chair_seat_position = np.array([0, 0.1, 0.55])
                self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])
            else:
                joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_shoulder_x, -80), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
                self.human.set_joint_angles([j for j, _ in joints_positions], [np.deg2rad(j_angle) for _, j_angle in joints_positions])
                self.human.target_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
                self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)

                # shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        # elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        # wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]

        # target_ee_pos = np.array([0.45, -0.45, 1.05])# + self.np_random.uniform(-0.05, 0.05, size=3)
        # target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][0])
        # target_ee_orient_shoulder = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task][-1])
        # offset = np.array([0, 0, 0.1])
        # self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos+offset, target_ee_orient_shoulder), (elbow_pos+offset, target_ee_orient), (wrist_pos+offset, target_ee_orient)], arm='left', tools=[], collision_objects=[self.human, self.furniture], right_side=False)
        # print(self.robot.get_base_pos_orient())
        # print(self.robot.get_joint_angles(self.robot.controllable_joint_indices))
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
            scale=0.1,
            mass=0.5,
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
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.3], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)
        # vertex_index = 680
        # vertex_index = 624
        vertex_index = 24  # 624-25
        # anchor_vertices = [307, 300, 603, 43, 641, 571]
        # anchor_vertices = [624, 623, 624 - 25, 623 - 25] #624 - 25, 623 np.pi, np.pi-25]
        # if vertex_index == 0:
        #     anchor_vertices = [0, 1, 25, 26] #624 - 25, 623 -25]
        # if vertex_index == 624:
        #     anchor_vertices = [624, 623, 624 - 25, 623 - 25] #624 - 25, 623 -25]

        self.triangle1_point_indices = [80, 439, 398]
        self.triangle2_point_indices = [245, 686, 355]

        # Move cloth grasping vertex into robot end effectors
        axis_angle = [np.pi / 2, 0, 0]
        p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        new_vertex_position = np.array(data[1][vertex_index])

        # NOTE: Create anchors between cloth and robot end effector
        p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)
        # for i in anchor_vertices:
        #     pos_diff = np.array(data[1][i]) - new_vertex_position
        #     p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

        self.robot.enable_force_torque_sensor(self.robot.left_end_effector - 1)

        # Disable collisions between robot and cloth
        # for i in [-1] + self.robot.all_joint_indices:
        #     p.setCollisionFilterPair(self.robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)
        if self.human is not None:
            # p.setCollisionFilterPair(self.furniture.body, self.cloth, -1, -1, 0, physicsClientId=self.id)
            # Disable collision between chair and human
            for i in [-1] + self.human.all_joint_indices:
                p.setCollisionFilterPair(self.human.body, self.furniture.body, i, -1, 0, physicsClientId=self.id)

        # NOTE: Uncomment this to visualize contact points between cloth and human body. Uncomment code in step() function too.
        batch_positions = []
        for i in range(500):
            batch_positions.append(np.array([100, 100 + i, 100]))
        self.points = self.create_spheres(radius=0.01 / 2, mass=0, batch_positions=batch_positions, visual=True, collision=False, rgba=[1, 1, 1, 1])

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        if self.human is not None:
            self.human.set_gravity(0, 0, 0)

        # Create a table
        self.table = Furniture()
        self.table.init('table', self.directory, self.id, self.np_random)

        # Place a bowl on a table
        # self.bowl = Furniture()
        # self.bowl.init('bowl', self.directory, self.id, self.np_random)

        # self.sphere = Furniture()
        # self.sphere.init('smooth_sphere', self.directory, self.id, self.np_random)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)

        self.time = time.time()
        self.init_env_variables()
        return self._get_obs()


from .agents.pr2 import PR2
robot_arm = 'left'
class ClothTableObjectsPR2Env(ClothTableObjectsEnv):
    def __init__(self):
        super(ClothTableObjectsPR2Env, self).__init__(robot=PR2(robot_arm), human=None, use_ik=True)


class ClothTableFoldEnvPR2Env(ClothTableFoldEnv):
    def __init__(self, frame_skip=10):
        super(ClothTableFoldEnvPR2Env, self).__init__(robot=PR2(robot_arm), human=None, use_ik=True, frame_skip=frame_skip)

