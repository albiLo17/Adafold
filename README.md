
# Adafold
Original implementation of the paper [Adafold: Adapting folding trajectories of cloths via feedback-loop manipulation](https://arxiv.org/pdf/2403.06210).
## Install
We encourage installing the repo with a conda environment.
```bash
conda create -n adafold python=3.9
git clone git@github.com:albiLo17/Adafold.git
cd Adafold
conda activate adafold
```

We use `python3.9` and `cuda 12.1` for our experiments.You can use the following commands to install the required cuda version and torch dependencies.
```
conda install cuda -c nvidia/label/cuda-12.1.0
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```
For the installation of the `torch_geometric` dependencies, for more information refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```
For the remaining pip dependencies, you can install them using :
```
pip install -e .
```

## Collect Folding Dataset
To run the data collection for the folding task, run:
```bash
python data_collection.py
``` 

Please refer to `Adafold/dataset_collection/dataset_args.py` for the available arguments for the data collection.

## Training Dynamics Model
To train the dynamics model on the collected dataset, run:
```bash
python train_dynamics.py
``` 

## Planning 
An example of how to plan with the learned model can be seen by running:
```bash
python planning.py
``` 

Please refer to the paper for more implementation details.
