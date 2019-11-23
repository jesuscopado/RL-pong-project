import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import discount_rewards, ReplayMemory


class PolicyConv(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(2, 32, 3, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128 * 11 * 11
        self.fc1_actor = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc1_critic = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif type(m) is torch.nn.Conv2d:
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)

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

        return dist, value


class Agent(object):
    def __init__(self, train_device="cuda", gamma=0.99, batch_size=100, memory_size=5000):
        self.train_device = train_device
        self.policy = PolicyConv(3, 128).to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = gamma
        self.batch_size = batch_size
        self.prev_obs = None
        self.memory = ReplayMemory(memory_size)
        self.states = []
        self.log_act_probs = []
        self.rewards = []
        self.state_value_preds = []

    def replace_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, observation, evaluation=False):
        x = self.preprocess(observation).to(self.train_device)
        dist, value = self.policy.forward(x)

        if evaluation:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()

        # Calculate the log probability of the action
        log_act_prob = -dist.log_prob(action)  # negative in order to perform gradient ascent

        return action.item(), log_act_prob, value

    def episode_finished(self, episode_number):
        log_act_probs = torch.stack(self.log_act_probs, dim=0)\
            .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0)\
            .to(self.train_device).squeeze(-1)
        state_value_preds = torch.stack(self.state_value_preds, dim=0)\
            .to(self.train_device).squeeze(-1)
        self.states, self.log_act_probs, self.rewards, self.state_value_preds = [], [], [], []

        # Compute critic loss and advantages
        next_state_value_preds = torch.cat((state_value_preds[1:], torch.tensor([0.0]).to(self.train_device)))
        next_state_value_preds_nograd = next_state_value_preds.detach()
        advantages = (rewards + self.gamma * next_state_value_preds_nograd) - state_value_preds

        ''' Monte Carlo approach, not completed
        # Compute discounted rewards and normalize it to zero mean and unit variance
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        advantages = discounted_rewards
        '''

        critic_loss = torch.mean(advantages**2)
        # critic_loss = torch.sum(advantages)  # TODO: check if it's faster and still correct
        
        # Compute the optimization term, i.e. the loss
        advantages_stopgrad = advantages.detach()
        actor_loss = torch.mean(log_act_probs * advantages_stopgrad)

        # Compute the gradients of loss w.r.t. network parameters
        total_loss = critic_loss + actor_loss
        total_loss.backward()

        self.reset()
        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return "ACAgent"

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

    def store_outcome(self, observation, log_act_prob, reward, state_value_pred):
        self.states.append(observation)
        self.log_act_probs.append(log_act_prob)
        self.rewards.append(torch.tensor([float(reward)]))
        self.state_value_preds.append(state_value_pred.squeeze(-1))

    def store_outcome_in_memory(self, *args):
        self.memory.push(*args)
