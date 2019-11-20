import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import sys
import os.path
import data.utils2 as utils2


class Policy(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(2, 32, 3, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128*11*11
        self.fc1_actor = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc1_critic = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.reshape(-1, self.reshaped_size)
        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        x_mean = self.fc2_mean(x_ac)

        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)

        return dist, value, x_mean




class Agent(object):
    def __init__(self):
        self.train_device = "cuda"
        self.policy = Policy(3, 128).to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.prev_obs = None
        self.policy.eval()
        self.name = "BombaPong *_*"

    def replace_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, stack_ob):
        x = stack_ob.to(self.train_device)
        dist, value, _ = self.policy.forward(x)
        action = torch.argmax(dist.probs)
        action_prob = torch.max(dist.probs)
        return action, action_prob, value

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return self.name

    def load_model(self, network_name):
        if os.path.isfile(network_name):
            print("Network Loaded!")
            self.policy.load_state_dict(torch.load(network_name), strict=False)
            self.policy.eval()
            # TODO: load also the optimizer file
    '''
    def save_model(self, number_episodes, n):
        torch.save(self.policy.state_dict(), "data/params" + str(number_episodes) + ".ckpt")
        utils2.save_mean_value2(v_reward_sum_running_avg, n)
        utils2.save_rew2(v_reward_sum, n)
    '''

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

