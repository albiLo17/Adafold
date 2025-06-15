import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from half_folding.models.base_models import AutoEncoder, PointNet2Cls
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.PI_encoder = AutoEncoder(args)
        self.adaptation_model = PointNet2Cls(args)

        self.action_dim = args.action_dim

    def forward(self, state, action=None, pi=None, z=None, get_adapt=True):
        # TODO: modify input class
        pi_rec = None
        if z is None:
            z, pi_rec = self.get_z(pi)

        z_adapt = None
        if get_adapt:
            z_adapt = self.get_adapt(state, action)

        return pi_rec, z, z_adapt

    def get_adapt(self, state, action=None):
        if action is None:
            action = torch.zeros(self.action_dim)
        return self.adaptation_model(state, action)
    def get_z(self, pi):
        # NC baseline
        if self.conditioning == 0:
            return None, None
        # PI baseline
        if self.conditioning == 1:
            return pi, None

        if self.conditioning == 2:
            return self.PI_encoder(pi)

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self, ):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def load(self, states=None, actions=None, logprobs=None, rewards=None, is_terminals=None):
        if states is not None:
            self.states = torch.cat([self.states, states.unsqueeze(1)], 1) if not isinstance(self.states, list) else states.unsqueeze(1)
        if actions is not None:
            self.actions = torch.cat([self.actions, actions.unsqueeze(1)], 1) if not isinstance(self.actions, list) else actions.unsqueeze(1)
        if logprobs is not None:
            self.logprobs = torch.cat([self.logprobs, logprobs.unsqueeze(1)], 1) if not isinstance(self.logprobs, list) else logprobs.unsqueeze(1)
        if rewards is not None:
            self.rewards = np.concatenate((self.rewards, np.expand_dims(rewards, 1)), 1) if not isinstance(self.rewards, list) else np.expand_dims(rewards, 1)
        if is_terminals is not None:
            self.is_terminals = np.concatenate((self.is_terminals, np.expand_dims(is_terminals, 1)), 1) if not isinstance(self.is_terminals, list) else np.expand_dims(is_terminals, 1)
    def clear(self):
        del self.actions
        del self.states
        del self.logprobs
        del self.rewards
        del self.is_terminals

        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, args=None, use_pointnet=False, different_encoders=False):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.use_pointnet = use_pointnet
        self.different_encoders = different_encoders
        if self.use_pointnet:
            self.encoder = PointNet2Cls(args, only_action=True)
            if different_encoders:
                self.encoder_critic = PointNet2Cls(args, only_action=True)
            state_dim = args.z_dim

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, past_actions=None):
        if self.use_pointnet:
            if past_actions is None:
                past_actions = torch.zeros(state.shape[0], self.action_dim).to(state.device)
            state = self.encoder(state.view(state.shape[0], -1, 3).unsqueeze(1), past_actions.unsqueeze(1))
            # state = state.squeeze(0)
        # TODO: add demonstration - waypoints (update)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).repeat(state.shape[0], 1, 1)
            # cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, past_actions=None):
        if self.use_pointnet:
            if past_actions is None:
                past_actions = torch.zeros(state.shape[0], self.action_dim).to(state.device)
            state_actor = self.encoder(state.view(state.shape[0], -1, 3).unsqueeze(1), past_actions.unsqueeze(1))
            # state = state.squeeze(0)

        if self.has_continuous_action_space:
            action_mean = self.actor(state_actor)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state_actor)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Modify input of the critic to give him privileged information
        mask = torch.zeros_like(state).reshape(state.shape[0], -1, 3)
        mask[:, 24, :] = torch.ones_like(mask[0, 24, :])        # Mask everything except gripper position
        mask = mask.reshape(mask.shape[0], -1)
        state = state * mask
        if self.use_pointnet:
            if self.different_encoders:
                del state_actor
                state = self.encoder_critic(state.view(state.shape[0], -1, 3).unsqueeze(1), past_actions.unsqueeze(1))
            else:
                state = state_actor

        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim,
                 action_dim,
                 lr_actor,
                 lr_critic,
                 gamma,
                 K_epochs,
                 eps_clip,
                 has_continuous_action_space,
                 args=None,
                 action_std_init=0.6,
                 use_pointnet=False,
                 different_encoders=False):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.use_pointnet = use_pointnet

        if not self.use_pointnet:
            self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': lr_critic}
            ])
            self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(
                device)

        else:
            self.policy = ActorCritic(args.z_dim, action_dim, has_continuous_action_space, action_std_init, args, use_pointnet=True, different_encoders=different_encoders).to(device)
            if not different_encoders:
                self.optimizer = torch.optim.Adam([
                    {'params': self.policy.encoder.parameters(), 'lr': lr_actor},
                    {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                    {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                ])
            else:
                self.optimizer = torch.optim.Adam([
                    {'params': self.policy.encoder.parameters(), 'lr': lr_actor},
                    {'params': self.policy.encoder_critic.parameters(), 'lr': lr_critic},
                    {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                    {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                ])

            self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, args, use_pointnet=True, different_encoders=different_encoders).to(
                device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, pi=None):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
                action_logprob = action_logprob.reshape(state.shape[0], -1)

            self.buffer.load(states=state, actions=action, logprobs=action_logprob)
            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards.flatten()), reversed(self.buffer.is_terminals.flatten())):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = self.buffer.states.view(-1, self.buffer.states.shape[-1]).detach().to(device)
        old_actions = self.buffer.actions.view(-1, self.buffer.actions.shape[-1]).detach().to(device)
        old_logprobs = self.buffer.logprobs.view(-1).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            loss = policy_loss + value_loss - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        expl_var = self.explained_variance(rewards, state_values)

        return policy_loss.mean(), value_loss, dist_entropy.mean(), state_values, expl_var

    def explained_variance(self, rewards, state_values):
        """
         Computes fraction of variance that ypred explains about y.
         Returns 1 - Var[y-ypred] / Var[y]
         interpretation:
             ev=0  =>  might as well have predicted zero
             ev=1  =>  perfect prediction
             ev<0  =>  worse than just predicting zero
         """
        # TODO: parallelize
        assert rewards.ndim == 1 and state_values.ndim == 1
        vary = torch.var(rewards)
        return torch.nan if vary == 0 else 1 - torch.var(rewards - state_values) / vary


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




