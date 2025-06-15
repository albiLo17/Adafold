from setuptools import setup, find_packages
import sys, os.path

base_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(base_dir, 'assistive_gym'))
sys.path.insert(0, os.path.join(base_dir, 'Adafold'))

# with open("README.md", "r") as f:
#     long_description = f.read()

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive_gym', 'envs', 'assets')
data_files = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive_gym', 'config.ini')]

for root, dirs, files in os.walk(directory):
    for fn in files:
        data_files.append(os.path.join(root, fn))

setup(name='adafold',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[
        'gym>=0.2.3',
        'numpy==1.24',
        'pybullet @ git+https://github.com/Zackory/bullet3.git@pybullet_3_0_9#egg=pybullet',
        'opencv-python',
        'moviepy',
        'numpy',
        'smplx',
        'trimesh',
        'numpngw',
        'matplotlib',
        'h5py',
        'ray[rllib]',
        'open3d',
        'gymnasium',
        'dm_tree',
        'shapely',
        'wandb',
    ] + ['screeninfo==0.6.1' if sys.version_info >= (3, 6) else 'screeninfo==0.2'],
)
