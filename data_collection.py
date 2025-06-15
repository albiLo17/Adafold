
from Adafold.args.arguments import get_argparse
from Adafold.trajectory.trajectory import Action_Sampler_Simple
import numpy as np
from Adafold.dataset_collection.dataset_args import get_argparse_dataset
from Adafold.dataset_collection.multiprocess_dc import mutliprocess_collection

if __name__ == "__main__":
    args = get_argparse()
    args.dataset_path = './data/datasets/'
    args.dataset_name = 'folding_dataset' #/oct31_1s'
    args.reward = 'IoU'
    args_dataset = get_argparse_dataset()
    args.render = 0

    # Dataset params dataset
    params = [40, 60, 0.1, 4, 0]
    env_idx = 0
    save_pcd = False

    pick_pos = np.asarray([-0.1, 0.1, 0.01])
    place_pos = np.asarray([-0.1, -0.1, 0.01])

    state_sampler = Action_Sampler_Simple(
        N=12,              # trajectory length
        action_len=0.03,
        c_threshold=0.,
        grid_size=0.04,
        pp_dir=place_pos -pick_pos,
        starting_point=pick_pos,
        sampling_mean=None,
        fixed_trajectory=None)

    # Select number of trajectories to collect
    num_trajectories = 2000
    states_list = state_sampler.generate_dataset(num_trajectories=num_trajectories, starting_point=pick_pos, target_point=place_pos, prob=False)

    mutliprocess_collection(
        args,
        args_dataset,
        env_idx,
        save_pcd,
        states_list,
        num_processes=10,
        params=[40, 60, 0.1, 4, 0],
    )
