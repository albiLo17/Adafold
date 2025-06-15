import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import furniture
from .agents.furniture import Furniture

# scale_factor = 5

class ClothObjectEnv(AssistiveEnv):
    def __init__(self, robot, second_robot, human=None, use_ik=True, delta_action=False, scale_factor=1, context_task="task", *args, **kwargs):
        
        assert human is None, "No human is needed!"
        time_step = 0.02
        super(ClothObjectEnv, self).__init__(robot=robot, second_robot=second_robot, human=None, task='dressing', 
            frame_skip=5, time_step=time_step, deformable=True, *args, **kwargs)

        self.context_task = context_task
        self.delta_action = delta_action
        self.scale_factor = scale_factor
        self.use_ik = use_ik
        self.use_mesh = (human is None)
        hz=int(1/time_step)
        p.setTimeStep(1.0 / hz)

    def init_position(self, action):

        # #apply the action on the base blocks
        # print("jjjjjjj")
        for j in range(p.getNumJoints(self.robot.body, physicsClientId=self.id)):
            joint_info = p.getJointInfo(self.robot.body, j, physicsClientId=self.id)
            # print(joint_info)

        # p.setJointMotorControl2(bodyUniqueId=self.robot.body,
        #                         jointIndex=0,
        #                         controlMode=p.VELOCITY_CONTROL,
        #                         targetVelocity=-action[0],
        #                         force=500000,
        #                         physicsClientId=self.id)
        # p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
        #                         jointIndex=0,
        #                         controlMode=p.VELOCITY_CONTROL,
        #                         targetVelocity=-action[0],
        #                         force=500000,
        #                         physicsClientId=self.id)

        # p.setJointMotorControl2(bodyUniqueId=self.robot.body,
        #                          jointIndex=1,
        #                          controlMode=p.VELOCITY_CONTROL,
        #                          targetVelocity=-action[1],
        #                          force=500000,
        #                          physicsClientId=self.id)
        # p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
        #                          jointIndex=1,
        #                          controlMode=p.VELOCITY_CONTROL,
        #                          targetVelocity=-action[1],
        #                          force=500000,
        #                          physicsClientId=self.id)

        p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                jointIndex=0,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-action[0],
                                force=500000,
                                physicsClientId=self.id)
        p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                jointIndex=0,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-action[0],
                                force=500000,
                                physicsClientId=self.id)

        p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-action[1],
                                force=500000,
                                physicsClientId=self.id)
        p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-action[1],
                                force=500000,
                                physicsClientId=self.id)
        print("--")
        p.stepSimulation(physicsClientId=self.id)
        # print(p.getJointState(self.robot.body,1)[2])

        # if self.use_ik:
        #     self.take_step(action, action_multiplier=0.05, ik=True, delta_action=self.delta_action)
        # else:
        #     self.take_step(action, action_multiplier=0.003)

        obs = self._get_obs()
        return obs, None, None, None

    def step(self, action):

        # #apply the action on the base blocks    
        # print("jjjjjjj")
        for j in range(p.getNumJoints(self.robot.body, physicsClientId=self.id)):
            joint_info = p.getJointInfo(self.robot.body, j, physicsClientId=self.id)
            # print(joint_info)

        if self.context_task == "context":
            p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                    jointIndex=0,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity = -action[0],
                                    force = 500000,
                                    physicsClientId=self.id)
            p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                    jointIndex=0,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity = -action[0],
                                    force = 500000,
                                    physicsClientId=self.id)

            p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                    jointIndex=1,
                                    controlMode=p.POSITION_CONTROL,
                                    targetVelocity=-action[1],
                                    force=500000,
                                    physicsClientId=self.id)
            p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                    jointIndex=1,
                                    controlMode=p.POSITION_CONTROL,
                                    targetVelocity=-action[1],
                                    force=500000,
                                    physicsClientId=self.id)

        else:

            p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                    jointIndex=0,
                                    controlMode=p.POSITION_CONTROL,
                                    targetVelocity=-action[0],
                                    force=500000,
                                    physicsClientId=self.id)
            p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                    jointIndex=0,
                                    controlMode=p.POSITION_CONTROL,
                                    targetVelocity=-action[0],
                                    force=500000,
                                    physicsClientId=self.id)

            p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                    jointIndex=1,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity = -action[1],
                                    force = 500000,
                                    physicsClientId=self.id)
            p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                    jointIndex=1,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity = -action[1],
                                    force = 500000,
                                    physicsClientId=self.id)

        #print("--")        
        p.stepSimulation(physicsClientId=self.id)
        #print(p.getJointState(self.robot.body,1)[2])


        # if self.use_ik:
        #     self.take_step(action, action_multiplier=0.05, ik=True, delta_action=self.delta_action)
        # else:
        #     self.take_step(action, action_multiplier=0.003)

        obs = self._get_obs()
        return obs, None, None, None

    def load_sphere(self, urdf_file_path, urdf_scale):
        # load a sphere into the scene using urdf files:
        if urdf_file_path is not None:
            robot_base_pos, _ = self.robot.get_base_pos_orient()
            second_robot_base_pos, _ = self.second_robot.get_base_pos_orient()
            sphere_pos = (robot_base_pos + second_robot_base_pos) / 2 + np.array(
                [0.0 * self.scale_factor, 0.0 * self.scale_factor, 0.75 * self.scale_factor])
            furniture = p.loadURDF(os.path.join(self.directory, urdf_file_path),
                                   basePosition=sphere_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=self.id,
                                   useFixedBase=0, globalScaling=urdf_scale * 0.3)

    def _get_obs(self):
        # get cloth vertices positions
        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)

        # get force-torque readings from both robots
        first_force_torque_reading = self.robot.get_force_torque_sensor(2) # the joint we want is 0
        #print(first_force_torque_reading)
        second_force_torque_reading = self.second_robot.get_force_torque_sensor(2)

        # get eef pos/orient from both robots
        first_linkstate = p.getLinkState(self.robot.body, 1, physicsClientId=self.id)
        second_linkstate = p.getLinkState(self.second_robot.body, 1, physicsClientId=self.id)
        # first_eef_pos, first_eef_orient = self.robot.get_pos_orient(1)
        # first_eef_pos_world, first_eef_orient_world = self.robot.convert_to_realworld(first_eef_pos, first_eef_orient)
        # second_eef_pos, second_eef_orient = self.second_robot.get_pos_orient(1)
        # second_eef_pos_world, second_eef_orient_world = self.second_robot.convert_to_realworld(second_eef_pos, second_eef_orient)
        return [mesh_points, first_force_torque_reading, second_force_torque_reading, first_linkstate[0], first_linkstate[1], second_linkstate[0], second_linkstate[0]]

        #return [mesh_points, first_force_torque_reading, second_force_torque_reading, first_eef_pos_world, first_eef_orient_world, second_eef_pos_world, second_eef_orient_world]

    def reset(
        self,
        spring_elastic_stiffness=40,
        spring_damping_stiffness=0.1,
        spring_bending_stiffness=0,
        obj_visual_file_path=None, # e.g., 'dinnerware/plastic_coffee_cup.obj',
        obj_collision_file_path=None, # e.g., 'dinnerware/plastic_coffee_cup_vhacd.obj',
        obj_scale=[0.2, 0.2, 0.2],
        urdf_file_path = 'dinnerware/sphere.urdf',
        urdf_scale=2,
        cloth_obj_file_path='clothing/bl_cloth_25_cuts.obj',
        robot_init_x=0.2,
        delta_x=0.9,
    ):
        """
        If using .obj file for the object, a visual file and a collision file needs to be passed in.
        If using .urdf file for the object, only a single urdf file needs to be passed in.
        """

        super(ClothObjectEnv, self).reset()
        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')

        # Update robot motor gains
        # for robot in [self.robot, self.second_robot]:
        #     robot.motor_gains = 0.05
        #     robot.motor_forces = 100.0

        # Set robot base position & orientation, and joint angles
        # first set a random joint angle
        base_pos_1 = [robot_init_x*self.scale_factor, 0.0*self.scale_factor, 1.0*self.scale_factor]
        base_orient_1 = [0, 0, 1, 0]
        #joint_angles = [0,  -np.pi/2,  0.0, -np.pi/2*1.5,  0, np.pi/2,  np.pi/2]
        joint_angles = [0]
        self.robot.reset_joints()
        self.robot.set_base_pos_orient(base_pos_1, base_orient_1)
        self.robot.set_joint_angles(self.robot.controllable_joint_indices, joint_angles)

        base_pos_2 = [-robot_init_x*self.scale_factor, 0.0*self.scale_factor, 1.0*self.scale_factor]
        base_orient_2 = [0, 0, 0, 1]
        #joint_angles = [0,  -np.pi/2,  0.0, -np.pi/2*1.5,  0, np.pi/2,  np.pi/2]
        joint_angles = [0]
        self.second_robot.reset_joints()
        self.second_robot.set_base_pos_orient(base_pos_2, base_orient_2)
        self.second_robot.set_joint_angles(self.second_robot.controllable_joint_indices, joint_angles)

        # use IK to adjust the joint angle to find a more feasiable initial configuration
        # delta_poses = [[-delta_x, 0, 0.4], [delta_x, 0, 0.4]]
        # for robot, delta_pos  in zip([self.robot, self.second_robot], delta_poses):
        #     joint = robot.right_end_effector if 'right' in robot.controllable_joints else robot.left_end_effector
        #     print(joint)
        #     ik_indices = robot.right_arm_ik_indices if 'right' in robot.controllable_joints else robot.left_arm_ik_indices
        #     pos, orient = robot.get_pos_orient(joint)
        #     pos += delta_pos
        #     robot_joint_angles = robot.ik(joint, pos, orient, ik_indices, max_iterations=200, use_current_as_rest=True)
        #     robot.set_joint_angles(robot.controllable_joint_indices, robot_joint_angles)

        for j in range(p.getNumJoints(self.robot.body, physicsClientId=self.id)):
            joint_info = p.getJointInfo(self.robot.body, j, physicsClientId=self.id)
            # print(joint_info)

        # p.setJointMotorControl2(bodyUniqueId=self.robot.body,
        #                         jointIndex=0,
        #                         controlMode=p.VELOCITY_CONTROL,
        #                         targetVelocity=-action[0],
        #                         force=500000,
        #                         physicsClientId=self.id)
        # p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
        #                         jointIndex=0,
        #                         controlMode=p.VELOCITY_CONTROL,
        #                         targetVelocity=-action[0],
        #                         force=500000,
        #                         physicsClientId=self.id)

        # p.setJointMotorControl2(bodyUniqueId=self.robot.body,
        #                          jointIndex=1,
        #                          controlMode=p.VELOCITY_CONTROL,
        #                          targetVelocity=-action[1],
        #                          force=500000,
        #                          physicsClientId=self.id)
        # p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
        #                          jointIndex=1,
        #                          controlMode=p.VELOCITY_CONTROL,
        #                          targetVelocity=-action[1],
        #                          force=500000,
        #                          physicsClientId=self.id)

        p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                jointIndex=0,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-4.0*self.scale_factor,
                                force=500000,
                                physicsClientId=self.id)
        p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                jointIndex=0,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-4.0*self.scale_factor,
                                force=500000,
                                physicsClientId=self.id)

        p.setJointMotorControl2(bodyUniqueId=self.robot.body,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-0.0*self.scale_factor,
                                force=500000,
                                physicsClientId=self.id)
        p.setJointMotorControl2(bodyUniqueId=self.second_robot.body,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                targetVelocity=-0.0*self.scale_factor,
                                force=500000,
                                physicsClientId=self.id)
        print("--")
        p.stepSimulation(physicsClientId=self.id)

        # load a table into the scene using urdf files:
        if self.context_task == "task":
            robot_base_pos, _ = self.robot.get_base_pos_orient()
            second_robot_base_pos, _ = self.second_robot.get_base_pos_orient()
            table_pos = (robot_base_pos + second_robot_base_pos) / 2 + np.array(
                [0.0 * self.scale_factor, 0.0 * self.scale_factor, 0.4 * self.scale_factor])
            furniture = p.loadURDF(os.path.join(self.directory, 'table/table.urdf'),
                                   basePosition=table_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=self.id,
                                   useFixedBase=1, globalScaling=urdf_scale * 0.25)

        if self.context_task == "context_table":
            robot_base_pos, _ = self.robot.get_base_pos_orient()
            second_robot_base_pos, _ = self.second_robot.get_base_pos_orient()
            table_pos = (robot_base_pos + second_robot_base_pos) / 2 + np.array(
                [0.07 * self.scale_factor, -0.0 * self.scale_factor, 0.4 * self.scale_factor])
            furniture = p.loadURDF(os.path.join(self.directory, 'table/table.urdf'),
                                   basePosition=table_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=self.id,
                                   useFixedBase=1, globalScaling=urdf_scale * 0.22)


        # Load cloth from a .obj file
        self.cloth = p.loadSoftBody(
            os.path.join(self.directory, cloth_obj_file_path),
            scale=0.3*self.scale_factor, ### scale of cloth
            mass=0.5*self.scale_factor,  ### originally 0.5
            useBendingSprings=1,
            useMassSpring=1,
            springElasticStiffness=spring_elastic_stiffness, # default: 40
            springDampingStiffness=spring_damping_stiffness, 
            springDampingAllDirections=0,
            springBendingStiffness=spring_bending_stiffness, 
            useNeoHookean=0,
            useSelfCollision=1, 
            collisionMargin=0.0001,
            frictionCoeff=1.0, 
            useFaceContact=1, 
            physicsClientId=self.id,
        )

        # Set cloth visual apperances
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 0.5], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)

        # Cloth points to be attached to the robot gripper. Pin the whole edge
        vertex_index = 25 // 2 
        anchor_vertices = [i for i in range(25)]

        # Move cloth grasping vertex into first robot end effectors
        axis_angle = [np.pi / 2, 0, np.pi / 2]
        p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        offset = self.robot.get_pos_orient(self.robot.left_end_effector)[0] - vertex_position
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        new_vertex_position = np.array(data[1][vertex_index])

        # Create anchors between cloth and first robot end effector
        p.createSoftBodyAnchor(self.cloth, vertex_index, self.robot.body, self.robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)
        for i in anchor_vertices:
            pos_diff = np.array(data[1][i]) - new_vertex_position
            p.createSoftBodyAnchor(self.cloth, i, self.robot.body, self.robot.left_end_effector, pos_diff, physicsClientId=self.id)

        # Pin the other whole edge to the second robot
        vertex_index = 25 * 24 + 25 // 2 
        anchor_vertices = [25 * 24 + i for i in range(25)]

        # Move cloth grasping vertex into second robot end effectors
        p.resetBasePositionAndOrientation(self.cloth, [0, 0, 0], self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        vertex_position = np.array(data[1][vertex_index])
        offset = self.second_robot.get_pos_orient(self.second_robot.left_end_effector)[0] - vertex_position
        p.resetBasePositionAndOrientation(self.cloth, offset, self.get_quaternion(axis_angle), physicsClientId=self.id)
        data = p.getMeshData(self.cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        new_vertex_position = np.array(data[1][vertex_index])

        # Create anchors between cloth and second robot end effector
        p.createSoftBodyAnchor(self.cloth, vertex_index, self.second_robot.body, self.second_robot.left_end_effector, [0, 0, 0], physicsClientId=self.id)
        for i in anchor_vertices:
            pos_diff = np.array(data[1][i]) - new_vertex_position
            p.createSoftBodyAnchor(self.cloth, i, self.second_robot.body, self.second_robot.left_end_effector, pos_diff, physicsClientId=self.id)

        # Enable force torque sensors
        # self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)
        # self.second_robot.enable_force_torque_sensor(self.second_robot.left_end_effector-1)
        self.robot.enable_force_torque_sensor(2) #also joint 0 nown
        self.second_robot.enable_force_torque_sensor(2)


        # Disable collisions between robot and cloth
        for i in [-1] + self.robot.all_joint_indices:
            p.setCollisionFilterPair(self.robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)
        for i in [-1] + self.second_robot.all_joint_indices:
            p.setCollisionFilterPair(self.second_robot.body, self.cloth, i, -1, 0, physicsClientId=self.id)

        # gravity compensation
        self.robot.set_gravity(0, 0, 0)
        self.second_robot.set_gravity(0, 0, 0)

        if urdf_file_path == 'None':
            urdf_file_path = None
        if obj_visual_file_path == 'None':
            obj_visual = None
        if obj_collision_file_path == 'None':
            obj_collision_file_path = None
            
        # # load a sphere into the scene using urdf files:
        # if urdf_file_path is not None:
        #     robot_base_pos, _ = self.robot.get_base_pos_orient()
        #     second_robot_base_pos, _ = self.second_robot.get_base_pos_orient()
        #     sphere_pos = (robot_base_pos + second_robot_base_pos) / 2 + np.array([0.0*self.scale_factor, 0.0*self.scale_factor, 1.05*self.scale_factor])
        #     furniture = p.loadURDF(os.path.join(self.directory, urdf_file_path),
        #             basePosition=sphere_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=self.id, useFixedBase=0, globalScaling=urdf_scale*0.3)

        # or, load other objects described by .obj files
        elif obj_visual_file_path is not None and obj_collision_file_path is not None:
            visual_filename = os.path.join(self.directory, obj_visual_file_path)
            collision_filename = os.path.join(self.directory, obj_collision_file_path)
            obj_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=obj_scale, rgbaColor=[1, 1, 1, 1], physicsClientId=self.id)
            obj_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=obj_scale, physicsClientId=self.id)
            robot_base_pos, _ = self.robot.get_base_pos_orient()
            second_robot_base_pos, _ = self.second_robot.get_base_pos_orient()
            obj_pos = (robot_base_pos + second_robot_base_pos) / 2 + np.array([0, 0, 0.6]) 
            obj_orient = [0, 0, 0, 1]
            obj = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=obj_collision, baseVisualShapeIndex=obj_visual, basePosition=obj_pos, baseOrientation=obj_orient, useMaximalCoordinates=False, physicsClientId=self.id)

        # Enable debug rendering
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        self.time = time.time()
        self.init_env_variables()
        return self._get_obs()


from .agents.rail_grippers import RailGripper
class ClothObjectRailGripperEnv(ClothObjectEnv):
    def __init__(self, *args, **kwargs):
        super(ClothObjectRailGripperEnv, self).__init__(
            robot=RailGripper(),
            second_robot=RailGripper(),
            human=None, 
            use_ik=True,
            *args, **kwargs
        )


