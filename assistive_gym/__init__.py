from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='DressingPR2Mesh-v1',
    entry_point='assistive_gym.envs:DressingPR2MeshEnv',
    max_episode_steps=200,
)
register(
    id='DressingPR2IKMesh-v1',
    entry_point='assistive_gym.envs:DressingPR2IKMeshEnv',
    max_episode_steps=200,
)
register(
    id='DressingPR2IK-v1',
    entry_point='assistive_gym.envs:DressingPR2IKEnv',
    max_episode_steps=200,
)

register(
    id='DressingPandaIKMesh-v1',
    entry_point='assistive_gym.envs:DressingPandaIKMeshEnv',
    max_episode_steps=200,
)

register(
    id='BedPosePR2-v1',
    entry_point='assistive_gym.envs:BedPosePR2Env',
    max_episode_steps=200,
)
register(
    id='BedPosePR2Mesh-v1',
    entry_point='assistive_gym.envs:BedPosePR2MeshEnv',
    max_episode_steps=200,
)
register(
    id='BedPoseStretch-v1',
    entry_point='assistive_gym.envs:BedPoseStretchEnv',
    max_episode_steps=200,
)
register(
    id='BedPoseStretchMesh-v1',
    entry_point='assistive_gym.envs:BedPoseStretchMeshEnv',
    max_episode_steps=200,
)


register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)

register(
    id='ViewClothVertices-v1',
    entry_point='assistive_gym.envs:ViewClothVerticesEnv',
    max_episode_steps=1000000,
)

#! ADDED FOR BEDDDING MANIPULATION
register(
    id='BeddingManipulationSphere-v1',
    entry_point='assistive_gym.envs:BeddingManipulationSphereEnv',
    max_episode_steps=1,
)


register(
    id='BeddingManipulationStretch-v1',
    entry_point='assistive_gym.envs:BeddingManipulationStretchEnv',
    max_episode_steps=200,
)

register(
    id='RemoveContactSphere-v1',
    entry_point='assistive_gym.envs:RemoveContactSphereEnv',
    max_episode_steps=1,
)


register(
    id='HumanSMPLXTesting-v1',
    entry_point='assistive_gym.envs:HumanSMPLXTestingEnv',
    max_episode_steps=200,
)

# Added by yufei
register(
    id='ClothTableObject-v1',
    entry_point='assistive_gym.envs:ClothTableObjectsPR2Env',
    max_episode_steps=200,
)

register(
    id='SimpleTest-v1',
    entry_point='assistive_gym.envs:SimpleTestEnv',
    max_episode_steps=200,
)

register(
    id='ClothObjectPandaEnv-v1',
    entry_point='assistive_gym.envs:ClothObjectPandaEnv',
    max_episode_steps=1000,
)

register(
    id='HalfFoldingEnv-v1',
    entry_point='assistive_gym.envs:HalfFoldEnv',
    max_episode_steps=200,
)
