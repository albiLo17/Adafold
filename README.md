
# Panda Grasping Cloth based on Assistive Gym 
## Install
We encourage installing the repo with a conda environment.
```bash
conda create -n assistive-gym python=3.6
git clone git@github.com:Moes96/assistive-gym-fem.git
cd assistive-gym
conda activate assistive-gym
pip install -e .
```

## Collect data
To run the two panda gripper, run
```bash
python panda_cloth/run_close_lift.py --urdf_file_path None
``` 

Please refer to `panda_cloth/run.py` for the available arguments for changing the cloth parameteres and loading different objects.

The main file that implements the simulation environment is `assistive_gym/envs/panda_cloth_env.py`.
Please refer to the comments in that file to see how the simulation environment is built.
