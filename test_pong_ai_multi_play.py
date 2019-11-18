"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import torch

import numpy as np
import argparse
import wimblepong
from PIL import Image
from agent import Agent, Policy
import random
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
#player = wimblepong.SimpleAi(env, player_id)

policy = Policy().to(device)
policy.load_state_dict(torch.load('params_2.ckpt'))
policy.eval()
myplayer = Agent(env, policy)

# Set the names for both SimpleAIs
env.set_names(myplayer.get_name(), opponent.get_name())
reward_sum_running_avg = None
win1 = 0
for it in range(0,episodes):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for ep in range(21):
        done = False

        (ob1, ob2) = env.reset()
        previous_obs = None
        t = 0
        while not done:

            # Get the actions from both SimpleAIs

            diff_obs = myplayer.elaborate_frame(ob1, previous_obs)
            #np.set_printoptions(threshold=sys.maxsize)
            
            #print(diff_obs.numpy())
            #exit()
            # "with" assure that the resources will be cleaned up,
            # even if there will be exceptions during the code execution
            with torch.no_grad():
                action1, action1_prob = policy(diff_obs.to(device))
            #if action1 == 0:   # TODO: change it!
            #    action1 = 2
            #print(action1)
            #action1 = myplayer.get_action(diff_obs)
            action2 = opponent.get_action()
            previous_obs = ob1
            # Step the environment and get the rewards and new observations
            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
            
            

            # Count the wins
            if rew1 == 10:
                win1 += 1
            if not args.headless:
                env.render()
            if done:
                print('Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f' % (it, ep, t, action1, action1_prob))    
                #observation= env.reset()
                #print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            t = t + 1
env.close()