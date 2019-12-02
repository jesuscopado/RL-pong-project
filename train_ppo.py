import time

import gym
import matplotlib.pyplot as plt
import torch
import numpy as np

from test_agents.PPOAgent.agent import Agent as PPOAgent
from utils import save_plot

import wimblepong

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")

# Set up the player here. We used the SimpleAI that does not take actions for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
player = PPOAgent(device)


def train(episodes_per_game=100, episodes_per_iteration=200, iterations=100000, max_timesteps=190000,
          render=False, resume=False):

    if resume:
        player.load_model()

    print("Training for {} started!".format(player.get_name()))

    win_ratio_history, average_win_ratio_history = [], []
    wins, total_episodes = 0, 0
    start_time = time.time()
    reward_sum_running_avg = None
    for it in range(iterations):
        stack_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
        for ep in range(episodes_per_iteration):  # TODO: which number here? 10? 30? 200?
            obs1 = env.reset()
            player.reset()
            for t in range(max_timesteps):
                if render:
                    env.render()

                stack_obs = player.preprocess(obs1)
                with torch.no_grad():
                    action1, action_prob1 = player.get_action(stack_obs)

                obs1, reward1, done, info = env.step(player.convert_action(action1))

                stack_obs_history.append(stack_obs)
                action_history.append(action1)
                action_prob_history.append(action_prob1)
                reward_history.append(reward1)

                if done:
                    total_episodes += 1
                    wins = wins + 1 if reward1 == 10 else wins

                    reward_sum = sum(reward_history[-t:])
                    reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
                    print('Iteration %d, Episode %d (%d timesteps) - '
                          'last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' %
                          (it, ep, t, action1, action_prob1, reward_sum, reward_sum_running_avg))

                    if total_episodes % episodes_per_game == 0:
                        win_ratio = int((wins / episodes_per_game) * 100)
                        wins = 0

                        # Bookkeeping (mainly for generating plots)
                        win_ratio_history.append(win_ratio)
                        if total_episodes > 100:
                            avg = np.mean(win_ratio_history[-100:])
                        else:
                            avg = np.mean(win_ratio_history)
                        average_win_ratio_history.append(avg)
                        print("Total episodes: {}. Win ratio (last 100 episodes): {}. Average win ratio: {}".format(
                            total_episodes, win_ratio, avg))

                    break

        discounted_rewards = player.discounted_rewards(reward_history)
        for _ in range(5):
            loss = player.update_policy(stack_obs_history, action_history, action_prob_history, discounted_rewards)
            print('Iteration %d -- Loss: %.3f' % (it, loss))

        if it % 10 == 0:
            player.save_model()
            save_plot(win_ratio_history, average_win_ratio_history, player.get_name())

    elapsed_time_min = round((time.time() - start_time) / 60, 2)
    print("Training finished in %f minutes." % elapsed_time_min)
    save_plot(win_ratio_history, average_win_ratio_history, player.get_name())


if __name__ == "__main__":
    train()
