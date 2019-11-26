import time

import gym
import matplotlib.pyplot as plt
import torch
import numpy as np

from test_agents.PGAgent.agent import Agent as PGAgent
from test_agents.PGAgent_FCModel.agent import Agent as PGAgent_FCModel
from utils import save_plot

import wimblepong

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
player = PGAgent_FCModel(device)


def train(episodes_per_game=100, train_episodes=500000, render=False, resume=False):
    if resume:
        player.load_model()

    print("Training for {} started!".format(player.get_name()))
    win_ratio_history, average_win_ratio_history = [], []
    wins = 0
    start_time = time.time()
    for episode_number in range(1, train_episodes+1):
        done = False
        obs1 = env.reset()
        rew1 = 1
        while not done:
            if render:
                env.render()

            # Get action from the agent
            action1, log_act_prob = player.get_action(obs1)
            prev_obs1 = obs1

            # Perform the action on the environment, get new state and reward
            obs1, rew1, done, info = env.step(action1)

            # Store action's outcome (so that the agent can improve its policy)
            player.store_outcome(prev_obs1, log_act_prob, action1, rew1, done)

        player.episode_finished(episode_number)
        wins = wins + 1 if rew1 == 10 else wins

        if episode_number % 5 == 0:
            env.switch_sides()

        if episode_number % episodes_per_game == 0:
            win_ratio = int((wins / episodes_per_game) * 100)
            print("Episode {} over. Win ratio: {}%".format(episode_number, win_ratio))
            wins = 0

            # Bookkeeping (mainly for generating plots)
            win_ratio_history.append(win_ratio)
            if episode_number > 100:
                avg = np.mean(win_ratio_history[-100:])
            else:
                avg = np.mean(win_ratio_history)
            average_win_ratio_history.append(avg)

        if episode_number % 10000 == 0:
            player.save_model()
            save_plot(win_ratio_history, average_win_ratio_history, player.get_name())

    elapsed_time_min = round((time.time() - start_time) / 60, 2)
    print("Training finished in %f minutes." % elapsed_time_min)
    save_plot(win_ratio_history, average_win_ratio_history, player.get_name())


if __name__ == "__main__":
    train()
