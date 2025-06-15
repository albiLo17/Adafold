import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import furniture
from .agents.furniture import Furniture

class SimpleTestEnv(AssistiveEnv):
    # TODO: 
    # 1. remove -- done
    # 1.1. remove wheelchair -- done
    # 2. add table & bowl & cup - DONE
    # 3. collect force applied to the object
    # 4. collect robot observation
    # 5. design a heuristic strategy for moving cloth along the table (how to control the robot?)

    def __init__(self, robot=None, human=None, use_ik=False):
        assert human is None, "For now just consider no human!"
        assert robot is None, "For simple test, no robot is needed!"
        super(SimpleTestEnv, self).__init__(robot=robot, human=None, task='dressing', 
            obs_robot_len=1, 
            obs_human_len=0, 
            frame_skip=5, time_step=1. / 480, deformable=True)

        self.use_ik = use_ik
        self.use_mesh = (human is None)
        hz=480
        p.setTimeStep(1.0 / hz)

        # 1 skip, 0.1 step, 50 substep, springElasticStiffness=100 # ? FPS
        # 1 skip, 0.1 step, 20 substep, springElasticStiffness=10 # 4.75 FPS
        # 5 skip, 0.02 step, 4 substep, springElasticStiffness=5 # 5.75 FPS
        # 10 skip, 0.01 step, 2 substep, springElasticStiffness=5 # 4.5 FPS

    def step(self, action):
        # if self.use_ik:
        #     self.take_step(action, action_multiplier=0.05, ik=True)
        # else:
        #     self.take_step(action, action_multiplier=0.003)
        p.stepSimulation(physicsClientId=self.id)

        if self.iteration >= 0:
            # NOTE: Uncomment this to visualize contact points between the cloth and the human body
            x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
            mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
            forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
            contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
            total_force = 0
            i = 0
            zero_contact_point = 0
            if forces.shape[0] > 0:
                max_force = np.max(np.linalg.norm(forces, axis=1))
                for cp, f in zip(contact_positions, forces):
                    if i >= len(self.points):
                        break
                    self.points[i].set_base_pos_orient(cp, [0, 0, 0, 1])
                    color = plt.cm.jet(min(np.linalg.norm(f)/0.5, 1.0)) - np.array([0, 0, 0, 0.5])
                    # c = np.linalg.norm(f)/max_force
                    if np.array_equal(f, np.zeros(3)):
                        # print("zero contact force!")
                        zero_contact_point += 1
                        color = np.array([0, 0, 1, 0.2])
                    else:
                        color = np.array([1, 0, 0, 1])
                    p.changeVisualShape(self.points[i].body, -1, rgbaColor=color, flags=0, physicsClientId=self.id)
                    total_force += np.linalg.norm(f)
                    
                    i += 1
                print("there are {} contact points out of {} cloth points, {} contact points have zero force".format(
                    i, len(mesh_points), zero_contact_point
                ))


            print('Time:', time.time() - self.time, 'Force:', total_force)
            self.time = time.time()
            for j in range(i, len(self.points)):
                self.points[j].set_base_pos_orient([100, 100+j, 100], [0, 0, 0, 1])
        
        reward = 0
        obs = self._get_obs()
        info = {}
        done = self.iteration >= 200

        return obs, reward, done, info

    def _get_obs(self, agent=None):
        return 0

    def reset(self):
        super(SimpleTestEnv, self).reset()
        self.build_assistive_env(furniture_type=None, gender='female', human_impairment='none')
        # self.cloth = p.loadSoftBody(
        #     # os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_2000tri.obj'), 
        #     os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj'), 
        #     # os.path.join(self.directory, 'clothing', 'gown_696v.obj'), 
        #     scale=1.0, 
        #     mass=0.15, 
        #     useBendingSprings=1, 
        #     useMassSpring=1, 
        #     springElasticStiffness=5, 
        #     springDampingStiffness=0.01, 
        #     springDampingAllDirections=1, 
        #     springBendingStiffness=0, 
        #     # useNeoHookean=0,
        #     useSelfCollision=1, 
        #     collisionMargin=0.001, 
        #     frictionCoeff=0.1, 
        #     useFaceContact=1, 
        #     physicsClientId=self.id)
        self.cloth = p.loadSoftBody(
            # os.path.join(self.directory, 'clothing', 'gown_696v.obj'), 
            os.path.join(self.directory, 'clothing', 'bl_cloth_25_cuts.obj'), 
            # scale=1.0, 
            scale=0.3, 
            mass=0.5, 
            useBendingSprings=1, 
            useMassSpring=1, 
            # springElasticStiffness=5, 
            springElasticStiffness=40, 
            # springDampingStiffness=0.01, 
            springDampingStiffness=0.1, 
            # springDampingAllDirections=1, 
            springDampingAllDirections=0, 
            springBendingStiffness=0, 
            useNeoHookean=0,
            useSelfCollision=1, 
            collisionMargin=0.0001, 
            # frictionCoeff=0.1, 
            frictionCoeff=1.0, 
            useFaceContact=1, 
            physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced_2000tri.obj'), scale=1.0, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        # self.cloth = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'fullgown.obj'), scale=1.0, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, rgbaColor=[1, 1, 1, 1], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.cloth, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)

        # Move cloth grasping vertex into robot end effector
        # p.resetBasePositionAndOrientation(self.cloth, [0, -0.05, 0.62], self.get_quaternion([0, 0, np.pi]), physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.cloth, [0, -0.05, 0.6], self.get_quaternion([np.pi / 2, 0, 0]), physicsClientId=self.id)
        p.stepSimulation()

        # NOTE: Uncomment this to visualize contact points between cloth and human body. Uncomment code in step() function too.
        batch_positions = []
        for i in range(8000):
            batch_positions.append(np.array([100, 100+i, 100]))
        self.points = self.create_spheres(radius=0.01/2, mass=0, batch_positions=batch_positions, visual=True, collision=False, rgba=[1, 1, 1, 1])

        self.sphere = Furniture()
        self.sphere.init('large_sphere', self.directory, self.id, self.np_random)
        # TODO: the above furniture thing might be able to be simplified to just the following two lines.
        # sphere_pos = np.array([0.1, -0.5, 0.825]) # + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
        # furniture = p.loadURDF(os.path.join(directory, 'dinnerware', 'sphere.urdf'), basePosition=sphere_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)

        self.time = time.time()
        self.init_env_variables()
        return self._get_obs()



