import random
from collections import namedtuple

import matplotlib.pyplot as plt
import torch


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def save_plot(win_ratio_history, average_win_ratio_history, agent_name, episodes_batch=200, it=-1):
    plt.figure()
    plt.clf()
    plt.plot(win_ratio_history)
    plt.plot(average_win_ratio_history)
    plt.title("Training: %s" % agent_name)
    plt.xlabel("{} episodes batch".format(episodes_batch))
    plt.ylabel("Win ratio (%)")
    plt.grid(True)
    plt.savefig("Training_{}_{}.png".format(agent_name, it))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
