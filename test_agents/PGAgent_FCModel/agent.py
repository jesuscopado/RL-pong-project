import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import discount_rewards


class PolicyFC(torch.nn.Module):
    def __init__(self, action_space, hidden, input_dimension):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dimension, hidden)
        self.fc2 = torch.nn.Linear(hidden, action_space)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x_probs = F.softmax(x, dim=-1)
        dist = Categorical(x_probs)
        return dist


class Agent(object):
    def __init__(self, train_device="cuda"):
        self.name = "PGAgent_FCModel"
        self.train_device = train_device
        self.input_dimension = 100 * 100  # downsampled 100x100 grid
        self.action_space = 2
        self.policy = PolicyFC(self.action_space, 256, self.input_dimension).to(self.train_device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.batch_size = 200
        self.prev_obs = None
        self.states = []
        self.log_act_probs = []
        self.rewards = []

    def get_action(self, observation, evaluation=False):
        x = self.preprocess(observation).to(self.train_device)
        dist = self.policy.forward(x)

        if evaluation:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()

        # Calculate the log probability of the action
        log_act_prob = -dist.log_prob(action)  # negative in order to perform gradient ascent

        action = action.item() + 1 if self.action_space == 2 else action.item()
        return action, log_act_prob

    def episode_finished(self, episode_number):
        log_act_probs = torch.stack(self.log_act_probs, dim=0)\
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0)\
            .to(self.train_device).squeeze(-1)
        self.states, self.log_act_probs, self.rewards = [], [], []

        # Compute discounted rewards and normalize it to zero mean and unit variance
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        weighted_probs = log_act_probs * discounted_rewards
        loss = torch.mean(weighted_probs)
        loss.backward()

        self.reset()
        if episode_number % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return self.name

    def load_model(self):
        weights = torch.load("{}.mdl".format(self.name))
        self.policy.load_state_dict(weights, strict=False)

    def save_model(self):
        torch.save(self.policy.state_dict(), "{}.mdl".format(self.name))
        # TODO: is it enough saving just the state dict? What about the optimizer?
        # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

    def preprocess(self, observation):
        observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = np.concatenate((self.prev_obs, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        stack_ob = stack_ob.transpose(1, 3)
        self.prev_obs = observation
        return stack_ob

    def preprocess(self, obs):
        # Preprocess the observation and set input to network to be difference image
        obs = obs[::2, ::2, 0]  # downsample by factor of 2
        obs[obs == 43] = 0  # erase background (background type 1)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
        cur_obs = obs.astype(np.float).ravel()
        obs = cur_obs - self.prev_obs if self.prev_obs is not None else np.zeros(self.input_dimension)
        self.prev_obs = cur_obs
        obs = torch.from_numpy(obs).float()
        return obs

    def store_outcome(self, observation, log_act_prob, action_taken, reward, done):
        self.states.append(observation)
        self.log_act_probs.append(log_act_prob)
        self.rewards.append(torch.tensor([float(reward)]))
