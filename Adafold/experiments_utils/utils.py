import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import os
import glob

def plot_results(categories, means, stds, y_label='final_reward', num_trajs=5, title=None, save_path=None):
    fig, ax = plt.subplots()
    x_pos = np.arange(len(categories))
    # Plot the bars with error bars showing the mean and variance
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)

    # Add labels and title
    ax.set_xlabel('Method')
    ax.set_ylabel(y_label)
    if title is None:
        ax.set_title(f'Mean and variance across {num_trajs} trajs')
    else:
        ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    plt.tight_layout()
    # TODO: save figure in the right folder somewhere and with the right name
    if save_path is not None:
        plt.savefig(save_path + '/model_comparisons.png')
    else:
        # Show the plot
        plt.show()

def print_results_table(methods, means, stds, save_path=None):
    # Create a PrettyTable instance
    table = PrettyTable()

    # Add columns
    table.field_names = ["Method", "Mean", "Standard Deviation"]

    for category, mean, std in zip(methods, means, stds):
        formatted_mean = '{:.3g}'.format(mean)
        formatted_std = '{:.3g}'.format(std)
        table.add_row([category, formatted_mean, formatted_std])

    # Print the table
    print(table)

    if save_path:
        with open(save_path + '/table_results.txt', 'w') as file:
            file.write(str(table))

    return table


def get_fig_rewards(reward_type, method, cumulative_reward):
    fig = plt.figure()
    cr = cumulative_reward[cumulative_reward != 0.0]
    plt.plot(cr, label=f'{reward_type}')
    plt.title(f"Episode reward ({reward_type}), method:{method}")
    plt.legend()
    plt.tight_layout()
    return fig


def process_results(methods, exp_name, H, A, num_trajs, num_envs, save_dir=None):
    cumulative_rewards = []
    final_rewards = []
    naming = []

    if save_dir is None:
        save_dir = f'./data/results/{exp_name}/H{H}_A{A}'

    # make the plot with confrontation against no adaptation
    for mod in methods:
        results_folder = os.path.join(save_dir, f'{mod}')# f'./data/results/{exp_name}/H{H}_A{A}/{mod}'
        cumulative_rewards.append(np.load(
            results_folder + f'/cumulative_reward.npy',
            allow_pickle=True)[:, :])
        final_rewards.append(np.load(
            results_folder + f'/final_reward.npy',
            allow_pickle=True)[:, :].reshape(-1))

        naming.append(f'{mod}')

    means = np.asarray([r.mean() for r in final_rewards])
    stds = np.asarray([r.std() for r in final_rewards])

    plot_results(naming, means, stds, y_label='final_reward', num_trajs=num_trajs,
                 title=f'Mean and variance across {num_envs} envs and {num_trajs} trajs per env.',
                 save_path=save_dir)#f'./data/results/{exp_name}/H{H}')

    table = print_results_table(naming, means, stds, save_path=save_dir)#f'./data/results/{exp_name}/H{H}_A{A}')

    print()


def process_reward_results(mod, exp_name, H, A, num_trajs, num_envs, reward, W=[0, 0.3, 0.5], W2=[0.1, 1, 0], W3=[0.01, 0.1, 0], W4=[0, 0.01, 0.03], save_dir=None):
    cumulative_rewards = []
    final_rewards = []
    naming = []

    if save_dir is None:
        save_dir = f'./data/results/{exp_name}'

    for num_envs in [num_envs]:#[1, 10]:
        # exp_name = f'240112_paper_e={num_envs}_c={args.reward}'
        # exp_name = f'240113_paper_e={num_envs}_c={args.reward}'

        # cumulative cost
        for w in W:#[0.4, 0.5, 0.7, 0.8, 0.9, 0]:
            # gripper attractor
            for w2 in W2:#[0.1, 0.5, 1., 10., 100.]:
                # Smooth penalty
                for w3 in W3:
                    # outside penalty cost
                    for w4 in W4:#[0.1, 0.5, 1., 10., 100.]:
                        folder_name = f'w={w}_w2={w2}_w3={w3}_w4={w4}/H{H}_A{A}/{mod}'


                        results_folder = os.path.join(save_dir, f'{folder_name}')# f'./data/results/{exp_name}/H{H}_A{A}/{mod}'
                        cumulative_rewards.append(np.load(
                            results_folder + f'/cumulative_reward.npy',
                            allow_pickle=True)[:, :])
                        final_rewards.append(np.load(
                            results_folder + f'/final_reward.npy',
                            allow_pickle=True)[:, :].reshape(-1))

                        naming.append(f'e={num_envs}_c={reward}_w={w}_w2={w2}_w3={w3}_w4={w4}')

    means = np.asarray([r.mean() for r in final_rewards])
    stds = np.asarray([r.std() for r in final_rewards])

    plot_results(naming, means, stds, y_label='final_reward', num_trajs=num_trajs,
                 title=f'Mean and variance across {num_envs} envs and {num_trajs} trajs per env.',
                 save_path=save_dir)#f'./data/results/{exp_name}/H{H}')

    table = print_results_table(naming, means, stds, save_path=save_dir)#f'./data/results/{exp_name}/H{H}_A{A}')

    return table


def process_MPPI_results(mod, exp_name, H, A, num_trajs, num_envs, reward, W=[0, 0.3, 0.5], W2=[0.1, 1, 0], W3=[0.01, 0.1, 0], W4=[0, 0.01, 0.03], save_dir=None):
    cumulative_rewards = []
    final_rewards = []
    naming = []

    if save_dir is None:
        save_dir = f'./data/results/{exp_name}'

    for num_envs in [num_envs]:#[1, 10]:
        # exp_name = f'240112_paper_e={num_envs}_c={args.reward}'
        # exp_name = f'240113_paper_e={num_envs}_c={args.reward}'

        # cumulative cost
        for w in W:#[0.4, 0.5, 0.7, 0.8, 0.9, 0]:
            # gripper attractor
            for w2 in W2:#[0.1, 0.5, 1., 10., 100.]:
                # Smooth penalty
                for w3 in W3:
                    # outside penalty cost
                    for w4 in W4:#[0.1, 0.5, 1., 10., 100.]:
                        folder_name = f'w={w}_w2={w2}_w3={w3}_w4={w4}/H{H}_A{A}/{mod}'


                        results_folder = os.path.join(save_dir, f'{folder_name}')# f'./data/results/{exp_name}/H{H}_A{A}/{mod}'
                        cumulative_rewards.append(np.load(
                            results_folder + f'/cumulative_reward.npy',
                            allow_pickle=True)[:, :])
                        final_rewards.append(np.load(
                            results_folder + f'/final_reward.npy',
                            allow_pickle=True)[:, :].reshape(-1))

                        naming.append(f'e={num_envs}_c={reward}_w={w}_w2={w2}_w3={w3}_w4={w4}')

    means = np.asarray([r.mean() for r in final_rewards])
    stds = np.asarray([r.std() for r in final_rewards])

    plot_results(naming, means, stds, y_label='final_reward', num_trajs=num_trajs,
                 title=f'Mean and variance across {num_envs} envs and {num_trajs} trajs per env.',
                 save_path=save_dir)#f'./data/results/{exp_name}/H{H}')

    table = print_results_table(naming, means, stds, save_path=save_dir)#f'./data/results/{exp_name}/H{H}_A{A}')

    return table



def read_params(path):
    base_dir = "/home/alberta/Pycharm/assistive-gym-fem/Adafold/experiments_RAL/data/mpc_results/240113_paper_e=10_c=IoU/H_13/mpc_mppi_zero_Ada_f1_1000_3_1234"

    all_params = []
    # Loop over all directories from e_0 to e_9
    for i in range(10):
        # Construct the directory path
        dir_path = os.path.join(base_dir, f"e_{i}")
        paths = glob.glob(os.path.join(dir_path, "t_*", "params.txt"))
        file_path = paths[0]     # just need to check one trajectory
        with open(file_path, 'r') as file:
            s = file.read()
            params = np.fromstring(s, sep=' ')
            all_params.append(params)
            print(f"Parameters in e_{i}:\n{params}\n")

    # np.savetxt('/home/alberta/Pycharm/assistive-gym-fem/Adafold/experiments_RAL/config/params_240113.txt', all_params)


def get_stored_params(id_config='240113'):
    base_dir = '/home/alberta/Pycharm/assistive-gym-fem/Adafold/experiments_RAL/config'
    # Read the file
    params = np.loadtxt(os.path.join(base_dir, f'params_{id_config}.txt') )
    # convert to int only element 0,1, 3
    # params[:, [0, 1, 3]] = params[:, [0, 1, 3]].astype(int)
    # params = params.astype(int)
    return params


if __name__=='__main__':

    # Print reward results
    mod='mpc_mppi_zero_Ada_f1_1000_3_1234'
    H = 13
    A=100
    num_envs=1
    num_trajs = 20
    reward = 'IoU_out_Gr_smooth'
    exp_name=f'245118_paper_reward_e={num_envs}_c={reward}'
    process_reward_results(mod, exp_name, H, A, num_trajs, num_envs, reward, save_dir=None)
    print()