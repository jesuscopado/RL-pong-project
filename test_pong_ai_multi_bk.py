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
import utils2


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
myplayer = Agent(env, policy)
opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
# Set the names for both SimpleAIs
env.set_names(myplayer.get_name(), opponent.get_name())
reward_sum_running_avg = None
win1 = 0
for it in range(0,episodes):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for ep in range(10):
        
        t = 0
        for part in range(21):
            
            (ob1, ob2) = env.reset()
            previous_obs = None
            done = False
            while not done:

                # Get the actions from both SimpleAIs

                diff_obs = myplayer.elaborate_frame(ob1, previous_obs)
                #np.set_printoptions(threshold=sys.maxsize)
                
                #print(diff_obs.numpy())
                #exit()
                # "with" assure that the resources will be cleaned up,
                # even if there will be exceptions during the code execution
                with torch.no_grad():
                    action1, action1_prob = policy.forward2(diff_obs.to(device))
                #if action1 == 0:   # TODO: change it!
                #    action1 = 2
                #print(action1)
                #action1 = myplayer.get_action(diff_obs)
                action2 = opponent.get_action()
                previous_obs = ob1
                # Step the environment and get the rewards and new observations
                (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
                

                # Statistics
                d_obs_history.append(diff_obs)
                action_history.append(action1-1)
                action_prob_history.append(action1_prob)
                reward_history.append(rew1)

                # Count the wins
                if rew1 == 10:
                    win1 += 1
                if not args.headless:
                    env.render()
                t = t + 1
        
        reward_sum = sum(reward_history[-t:])
        reward_sum_running_avg = 0.99*reward_sum_running_avg + 0.01*reward_sum if reward_sum_running_avg else reward_sum
        print('Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (it, ep, t, action1, action1_prob, reward_sum, reward_sum_running_avg))    
        utils2.save_mean_value(reward_sum_running_avg)
        utils2.save_rew(reward_sum)
            #observation= env.reset()
            #print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            
    print(len(action_history))        
    R = 0
    discounted_rewards = []
    for r in reward_history[::-1]:
        if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum
        R = r + policy.gamma * R
        discounted_rewards.insert(0, R)

    #print(discounted_rewards[:5])

    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
    
    # update policy
    for _ in range(5):
        n_batch = int(len(action_history) / 3)
        idxs = random.sample(range(len(action_history)), n_batch)
        d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0).to(device)
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs]).to(device)
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs]).to(device)
        advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs]).to(device)
        #advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()
            
        opt.zero_grad()
        loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        loss.backward()
        opt.step()
    
        print('Iteration %d -- Loss: %.3f' % (it, loss))
    if it % 5 == 0:
        torch.save(policy.state_dict(), 'params.ckpt')
env.close()