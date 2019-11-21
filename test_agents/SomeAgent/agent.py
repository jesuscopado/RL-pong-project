import torch
import numpy as np
from torch.distributions import Categorical
from models import PolicyConv


class Agent(object):
    def __init__(self):
        self.train_device = "cuda"
        self.policy = PolicyConv(3, 128).to(self.train_device)
        self.prev_obs = None
        self.policy.eval()

    def replace_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, observation):
        x = self.preprocess(observation).to(self.train_device)
        dist, value = self.policy.forward(x)
        action = torch.argmax(dist.probs)
        return action

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return "Some agent"

    def load_model(self):
        weights = torch.load("model.mdl")
        self.policy.load_state_dict(weights, strict=False)

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

