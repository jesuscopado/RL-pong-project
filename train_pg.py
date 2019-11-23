import argparse

import gym
import matplotlib.pyplot as plt
import wimblepong
import time

from test_agents.PGAgent.agent import Agent as PGAgent

parser = argparse.ArgumentParser()
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
# Number of episodes/games to train
episodes = 10000

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
player = PGAgent("cpu")


def train(print_things=True, episodes_per_match=100, train_episodes=10000, render=False):
    states = []
    win1 = 0
    start_time = time.time()
    cum_reward = 0
    for episode_number in range(0, train_episodes):
        done = False
        obs1 = env.reset()
        while not done:
            # Get action from the agent
            action1, log_act_prob = player.get_action(obs1)
            prev_obs1 = obs1

            # Perform the action on the environment, get new state and reward
            obs1, rew1, done, info = env.step(action1)

            # Store action's outcome (so that the agent can improve its policy)
            player.store_outcome(prev_obs1, log_act_prob, action1, rew1, done)

            cum_reward += rew1

            if render:  # if episode_number % 50 == 0:
                env.render()

            if done:
                # Count the wins
                if rew1 == 10:
                    win1 += 1

                plt.close()  # Hides game window
                player.episode_finished(episode_number)

                if (episode_number+1) % 5 == 0:
                    env.switch_sides()

                if (episode_number+1) % episodes_per_match == 0:
                    if print_things:
                        print("Episode {} over. {} wins out of {}: ".format(
                            episode_number+1, win1, episodes_per_match))
                        print("Cumulative reward: {}".format(cum_reward))
                    win1 = 0
                    cum_reward = 0

                obs1 = env.reset()
    elapsed_time_min = round((time.time() - start_time) / 60, 2)


if __name__ == "__main__":
    train()
