import time

import gym
import matplotlib.pyplot as plt
import torch

from test_agents.ACAgent.agent import Agent as ACAgent
import wimblepong

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
player = ACAgent(device)


def train(print_things=True, episodes_per_game=100, train_episodes=100000, render=False):
    states = []
    win1 = 0
    start_time = time.time()
    for episode_number in range(0, train_episodes):
        done = False
        obs1 = env.reset()
        while not done:
            # Get action from the agent
            action1, log_act_prob, state_value_pred = player.get_action(obs1)
            prev_obs1 = obs1

            # Perform the action on the environment, get new state and reward
            obs1, rew1, done, info = env.step(action1)

            # Store action's outcome (so that the agent can improve its policy)
            player.store_outcome(prev_obs1, log_act_prob, rew1, state_value_pred)

            if render:  # if episode_number % 50 == 0:
                env.render()

            if done:
                # Count the wins
                if rew1 == 10:
                    win1 += 1

                plt.close()  # Hides game window
                player.episode_finished(episode_number)

                if (episode_number + 1) % 5 == 0:
                    env.switch_sides()

                if (episode_number + 1) % episodes_per_game == 0:
                    if print_things:
                        print("Episode {} over. Win ratio: {}%".format(
                            episode_number + 1, int((win1 / episodes_per_game) * 100)))
                    win1 = 0

                obs1 = env.reset()
    elapsed_time_min = round((time.time() - start_time) / 60, 2)

if __name__ == "__main__":
    train()
