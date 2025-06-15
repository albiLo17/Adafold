import numpy as np
from Adafold.mpc.mpc_planner import MPC
import matplotlib.pyplot as plt
import copy
import torch
import copy
from Adafold.viz.viz_mpc import plot_vectors_3d, plot_action_selection, plot_prediction_and_state
from Adafold.trajectory.trajectory import Action_Sampler_Simple

from Adafold.args.arguments import get_argparse
import os

class MPPI(MPC):
    def __init__(self,
                 args,
                 noise_mu=0.0,
                 noise_sigma=0.01,   # 0.01 is a bit low
                 lambda_param=1.0,
                 A=100,
                 H=1,
                 mod='mpc',         # ['mpc', 'rand', 'fixed', 'mpc_mppi']
                 model=None,
                 env_idx=0,
                 load_model=True,
                 init_rand=True,
                 dir_model=None,
                 dir_results=None,
                 cumulative=False,
                 w_cumul=0,
                 cost_coefficients=[1., 1., 0.01, 1.],
                 save_datapoints=False,
                 downsample=False,
                 save_pcd=False,
                 verbose=False,
                 candidate_types=None,
                 path_dataset=None,
                 env_params=[40, 60, 0.1, 4, 0],
                 device='cpu'):

        super(MPPI, self).__init__(
            args,
            A=A,
            H=H,
            mod=mod,         # ['mpc', 'rand', 'fixed']
            env_idx=env_idx,
            model=model,
            load_model=load_model,
            dir_model=dir_model,
            dir_results=dir_results,
            cumulative=cumulative,
            w_cumul=w_cumul,
            cost_coefficients=cost_coefficients,
            save_datapoints=save_datapoints,
            downsample=downsample,
            save_pcd=save_pcd,
            verbose=verbose,
            path_dataset=path_dataset,
            env_params=env_params,
            device=device)

        # MPPI-specific parameters
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.num_samples = A  # Number of noise samples
        self.lambda_ = lambda_param  # Temperature parameter for the softmax

        # flag that set random the initialization of the sampling trajectory
        self.init_rand = init_rand

        if 'rand' in self.mod:
            mod = 'rand'
        elif 'zero' in self.mod:
            mod = 'zero'
        else:
            mod = 'fixed'
        print(f'MPPI params: mu init={mod}, noise_sigma={self.noise_sigma}, lambda={self.lambda_}')

    def init_candidates(self, N=12):
        gripper = self.env.sphere_ee.get_base_pos_orient()[0]

        # generate sampler for candidate trajectory
        self.sampler = Action_Sampler_Simple(
            N=N,  # trajectory length
            action_len=self.velocity,
            c_threshold=0.,
            grid_size=0.04,
            noise_sigma=self.noise_sigma,
            pp_dir=self.x_place - self.x_pick,
            place=self.x_place,
            starting_point=self.x_pick,
            sampling_mean=np.zeros((N, 3)),
            fixed_trajectory=None)

        self.initialize_fixed_traj()

        if 'rand' in self.mod:
            # random initialization of the sampling trajectory
            initial_traj, initial_action_means = self.sampler.sample_trajectory(starting_point=self.x_pick,
                                                                                  target_point=self.x_place,
                                                                                  return_actions=True)

        elif 'zero' in self.mod:
            # This is zero initialization of the candidates
            initial_traj = np.asarray([gripper])
            while len(initial_traj) < N:
                initial_traj = np.concatenate([initial_traj, initial_traj[-1].reshape(1, -1)])
            initial_action_means = initial_traj[1:] - initial_traj[:-1]

        else:
            initial_traj = self.planner_fixed.traj_points
            while len(initial_traj) < N:
                initial_traj = np.concatenate([initial_traj, initial_traj[-1].reshape(1, -1)])
            initial_action_means = initial_traj[1:] - initial_traj[:-1]


        self.sampler.reference_traj = np.asarray(initial_traj)
        self.sampler.action_means = np.asarray(initial_action_means)

        self.trajectories = np.asarray(
            self.sampler.generate_dataset(num_trajectories=self.A, starting_point=gripper, target_point=self.x_place,
                                          prob=False))
        self.candidate_actions_traj = self.trajectories[:, 1:] - self.trajectories[:, :-1]

        # Debug plots
        if self.verbose:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for t in self.trajectories:
                ax.scatter(t[:, 0], t[:, 1], t[:, 2])
            plt.title("Sampling distribution")
            x_lim = [-0.25, 0.25]
            y_lim = [-0.25, 0.25]
            z_lim = [0, 0.3]
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_zlim(*z_lim)
            plt.show()
    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def update_candidates(self):
        if self.sampler.N > 0:
            self.sampler.N -= 1
            self.sampler.action_means = self.updated_actions[1:]

        self.sampler.starting_point = self.raw_gripper_pos

        self.trajectories = np.asarray(
            self.sampler.generate_dataset(num_trajectories=self.A, starting_point=self.raw_gripper_pos,
                                          target_point=self.x_place,
                                          prob=False))
        self.candidate_actions_traj = self.trajectories[:, 1:] - self.trajectories[:, :-1]

        # consider the actions already performed plus the remaining ones (need a -1???)
        self.len_traj = len(self.actions) + self.trajectories.shape[1]

        # Debug
        action_means = np.asarray(self.sampler.action_means)
        # if action_means.shape[1] > self.candidate_actions_traj[:, :action_means.shape[1]].shape[1]:
        #     print()
        # if np.linalg.norm(self.trajectories[0, 1] - self.trajectories[0, 0]) > 0.0302:
        #     print()

        # action_fig = plot_action_selection(vectors=self.candidate_actions, action=self.candidate_actions[0],
        #                                    points=self.obs, full_points=self.raw_obs.reshape(-1, 3),
        #                                    start_point=self.raw_gripper_pos)
        # plt.show()

    def heuristic_pad_action(self, action_sequence):
        start_point = self.raw_gripper_pos
        # create the trajectory from the starting point with the action sequence
        traj = [start_point]
        padded_actions = []
        for action in action_sequence:
            traj.append(traj[-1] + action)
            padded_actions.append(action)

        # pad the trajectory to reach the target point
        target_point = self.x_place
        while np.linalg.norm(np.array(traj[-1]) - np.array(target_point)) > self.sampler.action_len * 0.8:
            direction = np.array(target_point) - np.array(traj[-1])
            norm = np.linalg.norm(direction)
            action = direction / norm * min(self.sampler.action_len, norm)
            traj.append(traj[-1] + action)
            padded_actions.append(action)

        return padded_actions

    def get_best_action(self, iteration=None, step=None):
        # This is for the plot later on
        obs = self.obs.clone()
        if obs.is_cuda:
            obs = obs.cpu()

        # make sure there are enough actions to plan
        num_actions = min(self.H, self.candidate_actions_traj.shape[1])
        self.candidate_actions = self.candidate_actions_traj[:][:, :num_actions]

        # in open loop evaluate all trajectory
        if 'ol' in self.mod:
            self.candidate_actions = copy.deepcopy(self.candidate_actions_traj)
            model_predictions = self._compute_total_cost(step=step)
        else:
            model_predictions = self._compute_total_cost(step=step)
        costs = self.cost_total

        # plan only if there are enough actions in the planner, otherwise follow the heuristic
        if self.sampler.N > 0:
            beta = np.min(costs)
            cost_total_non_zero = self._ensure_non_zero(cost=costs, beta=beta, factor=1 / self.lambda_)
            eta = np.sum(cost_total_non_zero)
            omega = 1 / eta * cost_total_non_zero

            # evaluate the noise of the actions of the sampler
            action_means = np.asarray(self.sampler.action_means)
            action_means = action_means.reshape(1, *action_means.shape).repeat(self.candidate_actions_traj.shape[0], 0)
            noise = np.zeros_like(action_means)
            min_len = min(action_means.shape[1], self.candidate_actions_traj[:, :action_means.shape[1]].shape[1])
            noise[:, :min_len] = self.candidate_actions_traj[:, :min_len] - action_means[:, :min_len]

            # update action mean with weighted sum
            omega_ext = omega.reshape(-1, 1).repeat(3, 1)
            action_mean_update = np.asarray([np.sum(omega_ext * noise[:, t, :], axis=0) for t in range(action_means.shape[1])])
            self.updated_actions = copy.deepcopy(action_means[0])

            # with normalization
            for i in range(action_mean_update.shape[0]):
                updated_action = action_means[0][i] + action_mean_update[i]
                self.updated_actions[i] = updated_action / np.linalg.norm(updated_action) * self.sampler.action_len

            # take the first action of the updated actions
            action_index = 0
            action_sequence = self.updated_actions
            # pad to the action sequence the rest of the trajectory with the heuristic
            action_sequence = self.heuristic_pad_action(action_sequence)
            # return the model prediction that gave the best cost
            self.final_model_prediction = model_predictions[np.argmin(self.cost_total)]

            # check if any of the candidate actions norm is bigger than 0.03
            norms = np.linalg.norm(self.candidate_actions, axis=2)

        else:
            action_index = 0
            action_sequence = self.candidate_actions[action_index]

            # pad to the action sequence the rest of the
            # return the model prediction that gave the best cost
            self.final_model_prediction = model_predictions[np.argmin(self.cost_total)]

        # ################ OLD CODE ################
        # if self.sampler.N > 0:
        #     num_actions = min(self.H, self.sampler.action_means.shape[0])
        # else:
        #     # the expectation is that all the candidates are the same so no need to plan but just take the first action
        #     num_actions = min(self.H, self.candidate_actions_traj[0].shape[1])
        # self.candidate_actions = self.candidate_actions_traj[:][:, :num_actions]
        #
        # # in open loop evaluate all trajectory
        # if 'ol' in self.mod:
        #     self.candidate_actions = copy.deepcopy(self.candidate_actions_traj)
        #     model_predictions = self._compute_total_cost(step=step)
        # else:
        #     model_predictions = self._compute_total_cost(step=step)
        # costs = self.cost_total
        #
        # beta = np.min(costs)
        # cost_total_non_zero = self._ensure_non_zero(cost=costs, beta=beta, factor=1 / self.lambda_)
        # eta = np.sum(cost_total_non_zero)
        # omega = 1 / eta * cost_total_non_zero
        #
        # # evaluate the noise of the actions
        # action_means = np.asarray(self.sampler.action_means)
        # action_means = action_means.reshape(1, *action_means.shape).repeat(self.candidate_actions_traj.shape[0], 0)
        # noise = np.zeros_like(self.candidate_actions_traj)
        # # check if the shape is correct
        # if action_means.shape[1] > self.candidate_actions_traj[:, :action_means.shape[1]].shape[1]:
        #     print()
        # min_shape = min(action_means.shape[1], self.candidate_actions_traj.shape[1])
        # noise[:, :min_shape] = self.candidate_actions_traj[:, :min_shape] - action_means[:, :min_shape]
        #
        # # update action mean with weighted sum
        # omega_ext = omega.reshape(-1, 1).repeat(3, 1)
        # action_mean_update = np.asarray([np.sum(omega_ext * noise[:, t, :], axis=0) for t in range(num_actions)])
        # self.updated_actions = copy.deepcopy(action_means[0])
        #
        # # with normalization
        # for i in range(action_mean_update.shape[0]):
        #     updated_action = action_means[0][i] + action_mean_update[i]
        #     self.updated_actions[i] = updated_action / np.linalg.norm(updated_action) * self.sampler.action_len
        #
        # # without normalization
        # # updated_actions = action_means[0][:action_mean_update.shape[0]] +  action_mean_update
        #
        # # take the first action of the updated actions
        # action_index = 0
        # action_sequence = self.updated_actions
        # # return the model prediction that gave the best cost
        # self.final_model_prediction = model_predictions[np.argmin(self.cost_total)]
        #
        # # check if any of the candidate actions norm is bigger than 0.03
        # norms = np.linalg.norm(self.candidate_actions, axis=2)
        # # if np.any(norms > 0.031):
        # #     print()
        #
        # #############################################

        action_fig = plot_action_selection(vectors=self.candidate_actions, action=action_sequence,
                                           points=obs, full_points=self.raw_obs.reshape(-1,3), start_point=self.raw_gripper_pos)
        plt.savefig(self.dir_results + f'/action_selection_t{step}.png')
        plt.clf()

        fig = plot_vectors_3d(
            action=action_sequence,
            points=obs,
            predicted_points=self.final_model_prediction,
            gripper_pos=self.raw_gripper_pos)
        plt.savefig(self.dir_results + f'/model_prediction_t{step}.png')
        plt.clf()

        # TODO: uncomment for debug
        # if iteration is not None:
        #     print(f'Iteration {iteration}, step {step} - {self.device} - time={time.time() - start_time}')
        #     print(f'action_index - {action_index}, cost - {self.cost_total[action_index]}')

        final_actions = action_sequence
        return final_actions, action_index

if __name__ == '__main__':

    args = get_argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and torch.cuda.get_device_name(torch.cuda.current_device()) != 'NVIDIA GeForce GTX 1080' and not args.train_cluster:
        torch.cuda.set_device(1)

    args.render = 1
    args.obs = 'mesh'       # [mesh, full_pcd]
    args.loss = 'MSE'
    args.reward = 'IoU'       # ['IoU', IoU_Gr]
    args.lr = 0.0001

    dir_models = {
        'mpc_NoAda_100': f'../data/trained_models/D=100_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=0_fusion=0',
        'mpc_Ada_f0_100': f'../data/trained_models/D=100_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=0',
        'mpc_Ada_f1_100': f'../data/trained_models/D=100_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=1',
        'mpc_NoAda_500': f'../data/trained_models/D=500_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=0_fusion=0',
        'mpc_Ada_f0_500': f'../data/trained_models/D=500_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=0',
        'mpc_Ada_f1_500': f'../data/trained_models/D=500_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=3_fusion=1',
        'mpc_NoAda_1000': f'../data/trained_models/D=1000_obs=mesh_loss={args.loss}_K={args.K}_zDim={args.z_dim}_mode=0_fusion=0',
    }

    conditionings = {'mpc_NoAda_100': 0, 'mpc_Ada_f0_100':3, 'mpc_Ada_f1_100':3, 'mpc_NoAda_500': 0, 'mpc_NoAda_1000': 0, 'mpc_Ada_f0_500':3, 'mpc_Ada_f1_500':3}
    fusions = {'mpc_NoAda_100': 0, 'mpc_Ada_f0_100':0, 'mpc_Ada_f1_100':1, 'mpc_NoAda_500': 0, 'mpc_NoAda_1000': 0, 'mpc_Ada_f0_500':0, 'mpc_Ada_f1_500':1}
    all_models = ['mpc_NoAda_100', 'mpc_Ada_f0_100', 'mpc_Ada_f1_100', 'mpc_NoAda_500', 'mpc_NoAda_1000', 'mpc_Ada_f0_500', 'mpc_Ada_f1_500']
    exp_name = '0211'

    dir_results = './tmp'

    save_data = False
    dataset_path = None
    if save_data:
        # TODO: iterate multiple time for each environment!
        # create new folder on top of the already existing dataset
        dataset_path = f'./data/bootstrap_0'
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    mod = 'mpc_RMA'
    mod = 'mpc_mppi_rand'
    model = 'mpc_Ada_f1_500'
    args.dyn_conditioning = conditionings[model]
    args.fusion = fusions[model]
    A = 100
    H = 2
    init_rand = True
    args.reward = 'IoU_Gr'  # ['IoU', IoU_Gr]
    planner = MPPI(
        args,
        noise_mu=0.0,
        noise_sigma=0.01,
        lambda_param=2.0,
        A=A,
        H=H,
        mod=mod,  # ['mpc', 'rand', 'fixed']
        env_idx=0,
        load_model=True,
        init_rand=init_rand,
        dir_model=dir_models[model],
        dir_results=dir_results,
        save_datapoints=save_data,
        downsample=False,
        cumulative=True,
        save_pcd=True,
        verbose=True,
        path_dataset=dataset_path,
        env_params=[40, 60, 0.1, 4, 0],
        device=device
    )

    cumulative_reward, final_reward, frames = planner.control(iterations=1)
    plt.plot(cumulative_reward)
    plt.title(f"Model: {model}, H= {H}")
    plt.tight_layout()
    plt.show()

