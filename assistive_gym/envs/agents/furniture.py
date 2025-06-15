import os
import pybullet as p
import numpy as np
from .agent import Agent

class Furniture(Agent):
    def __init__(self):
        super(Furniture, self).__init__()

    def init(self, furniture_type, directory, id, np_random, wheelchair_mounted=False, rw=True):
        if 'wheelchair' in furniture_type:
            left = False
            if 'left' in furniture_type:
                furniture_type = 'wheelchair'
                left = True
            furniture = p.loadURDF(os.path.join(directory, furniture_type, 'wheelchair.urdf' if not wheelchair_mounted else ('wheelchair_jaco.urdf' if not left else 'wheelchair_jaco_left.urdf')), basePosition=[0, 0, 0.06], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'bed':
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'bed.urdf'), basePosition=[-0.1, 0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)

            # mesh_scale = [1.1]*3
            # bed_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced.obj'), rgbaColor=[1, 1, 1, 1], specularColor=[0.2, 0.2, 0.2], meshScale=mesh_scale, physicsClientId=self.id)
            # bed_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced_vhacd.obj'), meshScale=mesh_scale, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.id)
            # furniture = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=self.id)
            # # Initialize bed position
            # p.resetBasePositionAndOrientation(furniture, [-0.1, 0, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        elif furniture_type == 'hospital_bed':
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'hospital_bed.urdf'), basePosition=[0, 0.2, 0.43], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
            self.controllable_joint_indices = [1]
            super(Furniture, self).init(furniture, id, np_random)
        elif furniture_type == 'table':
            # furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
            # furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[-0., -0.5, 0.32], baseOrientation=[0, 0, 0.7071068, 0.7071068], physicsClientId=id)        # This is for half folding with PR2
            # furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0.3, -0.5, 0.32], baseOrientation=[0, 0, 0.7071068, 0.7071068], physicsClientId=id)

            # TABLE HALF FOLDING
            # furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0.0,  -0., -0.72], baseOrientation=[0, 0, 0.7071068, 0.7071068], physicsClientId=id)
            # TABLE HALF FOLDING RW
            if rw:
                furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0.3,  -0., -0.6],
                                       baseOrientation=[0, 0, 0., 1.], physicsClientId=id)
            else:
                # TABLE HALF FOLDING FOLDSFORMER
                furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0., -0., -0.7],
                                       baseOrientation=[0, 0, 0., 1.], physicsClientId=id)

        elif furniture_type == 'bowl':
            # bowl_pos = np.array([-0.15, -0.65, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            bowl_pos = np.array([0.1, -0.5, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            furniture = p.loadURDF(os.path.join(directory, 'dinnerware', 'bowl.urdf'), basePosition=bowl_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'sphere':
            # bowl_pos = np.array([-0.15, -0.65, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            sphere_pos = np.array([0.1, -0.5, 0.825]) # + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            furniture = p.loadURDF(os.path.join(directory, 'dinnerware', 'sphere.urdf'), basePosition=sphere_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'large_sphere':
            # bowl_pos = np.array([-0.15, -0.65, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            sphere_pos = np.array([0, -0.4, 0.25]) # + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            furniture = p.loadURDF(os.path.join(directory, 'tinkercad', 'large_sphere.urdf'), 
                globalScaling=0.005,
                basePosition=sphere_pos, 
                baseOrientation=[0, 0, 0, 1], 
                useFixedBase=1,
                physicsClientId=id)
        elif furniture_type == 'smooth_sphere':
            # bowl_pos = np.array([-0.15, -0.65, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            sphere_pos = np.array([0.1, -0.6, 0.9]) # + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            furniture = p.loadURDF(os.path.join(directory, 'tinkercad', 'large_sphere.urdf'), 
                globalScaling=0.002,
                basePosition=sphere_pos, 
                baseOrientation=[0, 0, 0, 1], 
                useFixedBase=0,
                physicsClientId=id)
        

        elif furniture_type == 'nightstand':
            furniture = p.loadURDF(os.path.join(directory, 'nightstand', 'nightstand.urdf'), basePosition=np.array([-0.9, 0.7, 0]), baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == "deformable_bed":
            # load the modified version of the hospital bed
            #   modified the hospital bed so there is no rigid mattress (using frame only)
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'hospital_bed_modified.urdf'), basePosition=[0, 0.2, 0.43], baseOrientation=[0, 0, 0, 1], physicsClientId=id)

            # load in deformable mattress
            #   properties of the mattress have not been tuned well - may not interact will with other deformables
            #   vtk of soft mattress generated by first converting stl/obj of mattress to tetrahedral mesh (msh file) using tetwild (https://github.com/Yixin-Hu/TetWild)
            #   msh then converted to vtk file using to gmsh (https://gmsh.info/)
            mattress = p.loadSoftBody(os.path.join(directory, 'bed', 'soft_mattress.vtk'), basePosition = [0.45,-1,0.49], baseOrientation = [0, 0, 1, 1], scale = 1, mass = 800, useNeoHookean = 1, NeoHookeanMu = 4500, NeoHookeanLambda = 2500, NeoHookeanDamping = 50, useSelfCollision = 1, frictionCoeff = 100, collisionMargin = 0.004, useFaceContact=1, repulsionStiffness = 1000, physicsClientId=id)
            p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=id)
        else:
            furniture = None

        super(Furniture, self).init(furniture, id, np_random, indices=-1)

