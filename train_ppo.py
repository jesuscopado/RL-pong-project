import argparse
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


def train(episodes_batch=200, iterations=100000, max_timesteps=190000,
          render=False, resume=False, full_print=True):

    if resume:
        player.load_model()

    print("Training for {} started!".format(player.get_name()))

    win_ratio_history, average_win_ratio_history = [], []  # within the batch of episodes
    wins, total_episodes = 0, 0
    start_time = time.time()
    reward_sum_running_avg = None
    for it in range(iterations):
        stack_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
        for ep in range(episodes_batch):  # TODO: which number here? 10? 30? 200?
            obs1 = env.reset()
            player.reset()
            total_episodes += 1
            for t in range(max_timesteps):
                if render:
                    env.render()

                with torch.no_grad():
                    action1, action_prob1, stack_obs = player.get_action(obs1, evaluation=False)

                obs1, reward1, done, info = env.step(action1)

                stack_obs_history.append(stack_obs)
                action_history.append(player.revert_action_convertion(action1))
                action_prob_history.append(action_prob1)
                reward_history.append(reward1)

                if done:
                    wins = wins + 1 if reward1 == 10 else wins
                    if full_print:
                        reward_sum = sum(reward_history[-t:])
                        if reward_sum_running_avg:
                            reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum
                        else:
                            reward_sum_running_avg = reward_sum
                        print('Iteration %d, Episode %d (%d timesteps) - '
                              'last_action: %d, last_action_prob: %.2f, result: %s, running_avg: %.2f' %
                              (it, ep, t, action1, action_prob1,
                               "¡¡VICTORY!!" if reward1 == 10 else "defeat", reward_sum_running_avg))
                    break

        player.update_policy(stack_obs_history, action_history, action_prob_history, reward_history)

        # Bookkeeping (mainly for generating plots)
        win_ratio = int((wins / episodes_batch) * 100)
        win_ratio_history.append(win_ratio)
        avg = np.mean(win_ratio_history[-100:])
        average_win_ratio_history.append(avg)
        print("Total episodes: {}. Win ratio (last episodes batch): {}%. Average win ratio: {}%".format(
            total_episodes, win_ratio, round(float(avg), 2)))
        wins = 0

        if it % 10 == 0:
            player.save_model(it)
            save_plot(win_ratio_history, average_win_ratio_history, player.get_name(), episodes_batch, it)

    elapsed_time_min = round((time.time() - start_time) / 60, 2)
    player.save_model(it)
    save_plot(win_ratio_history, average_win_ratio_history, player.get_name(), episodes_batch)
    print("Training finished in %f minutes." % elapsed_time_min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume training.")
    args = parser.parse_args()
    train(render=args.render, resume=args.resume)
