""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

import gym
import numpy as np
import time
import wimblepong

from test_agents.Karpathy.agent import Agent as KarpathyAgent
from utils import save_plot

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
train_episodes = 500000
episodes_per_game = 100

# Define the player
player_id = 1
# Set up the player here
player = KarpathyAgent()

resume = False  # resume from previous checkpoint?
render = False

if resume:
    player.load_model()

# Arrays to keep track of wins (rewards)
win_ratio_history, average_win_ratio_history = [], []
# timestep_history = []
wins = 0
start_time = time.time()
for episode_number in range(1, train_episodes+1):
    reward_sum = 0
    observation = env.reset()
    player.reset()
    # timesteps = 0
    done = False
    reward = 0
    while not done:
        if render:
            env.render()

        observation = player.preprocess_obs(observation)
        action, aprob, hidden_state = player.get_action(observation)  # and store outcome
        prev_observation = observation

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        player.store_outcome(prev_observation, hidden_state, action, aprob, reward)
        # timesteps += 1

    player.episode_finished(episode_number)  # update policy
    # timestep_history.append(timesteps)
    wins = wins+1 if reward == 10 else wins

    if (episode_number + 1) % 5 == 0:
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
print("Training finished in %f minutes" % elapsed_time_min)
save_plot(win_ratio_history, average_win_ratio_history, player.get_name())
