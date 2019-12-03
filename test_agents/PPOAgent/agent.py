import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(torch.nn.Module):
    def __init__(self, action_space, input_dimension):
        super().__init__()
        self.hidden = 512
        self.fc1 = torch.nn.Linear(input_dimension*2, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space)
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Policy3FC(torch.nn.Module):
    def __init__(self, action_space, input_dimension):
        super().__init__()
        self.hidden1 = 512
        self.hidden2 = 64
        self.fc1 = torch.nn.Linear(input_dimension*2, self.hidden1)
        self.fc2 = torch.nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = torch.nn.Linear(self.hidden2, action_space)
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class PolicyConv(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden  # TODO: 64 or 128?
        self.conv1 = torch.nn.Conv2d(2, 32, 3, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128 * 11 * 11
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, action_space)
        # self.init_weights()

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
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class Agent(object):
    def __init__(self, train_device="cpu"):
        self.train_device = train_device
        self.input_dimension = 100 * 100  # downsampled by 2 -> 100x100 grid
        self.action_space = 2
        self.policy = Policy(self.action_space, self.input_dimension).to(self.train_device)
        # self.policy = PolicyConv(self.action_space, 128).to(self.train_device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.eps_clip = 0.1
        self.batch_size = 200
        self.prev_obs = None
        self.perc_minibatch = 0.7
        self.name = "PPOAgent_{}".format(type(self.policy).__name__)

    def get_action(self, obs, evaluation=False):
        stack_obs = self.preprocess(obs)
        logits = self.policy.forward(stack_obs)

        if evaluation:
            action = int(torch.argmax(logits[0]).detach().cpu().numpy())
            action_prob = 1.0
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().cpu().numpy()[0])
            action_prob = float(dist.probs[0, action].detach().cpu().numpy())

        return action, action_prob, stack_obs

    def convert_action(self, action):
        return action + 1 if self.action_space == 2 else action

    def preprocess(self, obs):
        if "Conv" not in type(self.policy).__name__:
            obs = obs[::2, ::2, 0]  # downsample by factor of 2
            obs[obs == 43] = 0  # erase background (background type 1)
            obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
            obs = torch.from_numpy(obs.astype(np.float32).ravel()).unsqueeze(0)
            if self.prev_obs is None:
                self.prev_obs = obs
            stack_obs = torch.cat([obs, self.prev_obs], dim=1)
        else:
            obs = obs[::2, ::2].mean(axis=-1)
            obs = np.expand_dims(obs, axis=-1)
            if self.prev_obs is None:
                self.prev_obs = obs
            stack_obs = np.concatenate((self.prev_obs, obs), axis=-1)
            stack_obs = torch.from_numpy(stack_obs).float().unsqueeze(0)
            stack_obs = stack_obs.transpose(1, 3)

        self.prev_obs = obs
        return stack_obs.to(self.train_device)

    def discount_rewards(self, reward_history):
        R = 0
        discounted_rewards = []
        for r in reward_history[::-1]:
            if r != 0:
                R = 0  # scored/lost a point in pong, so reset reward sum
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        return torch.FloatTensor(discounted_rewards)

    def update_policy(self, d_obs_history, action_history, action_prob_history, reward_history):
        # Compute discounted rewards and normalize
        discounted_rewards = self.discount_rewards(reward_history)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        for _ in range(5):  # TODO: check ideal number of updates
            n_batch = int(self.perc_minibatch * len(action_history))  # TODO: check ideal batch size
            idxs = random.sample(range(len(action_history)), n_batch)
            d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0).to(self.train_device)
            action_batch = torch.LongTensor([action_history[idx] for idx in idxs]).to(self.train_device)
            action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs]).to(self.train_device)
            advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs]).to(self.train_device)
            # advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()

            self.optimizer.zero_grad()  # TODO: check if we get better results without it (full-batch?)
            vs = np.array([[1., 0.], [0., 1.]])
            ts = torch.FloatTensor(vs[action_batch.cpu().numpy()]).to(self.train_device)
            logits = self.policy.forward(d_obs_batch)
            r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob_batch
            loss1 = r * advantage_batch
            loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_batch
            loss = -torch.min(loss1, loss2)
            loss = torch.mean(loss)
            loss.backward()
            self.optimizer.step()

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return self.name

    def load_model(self, name=None, evaluation=False):
        name_file = "{}.mdl".format(self.name if name is None else name)
        weights = torch.load(name_file, map_location=torch.device(self.train_device))
        self.policy.load_state_dict(weights, strict=False)
        if evaluation:
            self.policy.eval()

    def save_model(self, iteration=-1):
        hundreds_iterations = (iteration // 100) * 100
        torch.save(self.policy.state_dict(), "{}_{}.mdl".format(self.name, hundreds_iterations))
        # TODO: is it enough saving just the state dict? What about the optimizer?
        # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
