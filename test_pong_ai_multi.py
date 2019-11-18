"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import torch
import torch.optim as optim

import numpy as np
import argparse
import wimblepong
from PIL import Image
from agent import Agent, Policy
import random
import sys
import data.utils2 as utils2


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--net", type=str, help="Path of the saved network", default="")
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



loss_computator = torch.nn.BCELoss(reduction='none') 
optimizer = optim.RMSprop(policy.parameters(), lr=1e-4)
running_reward = None
optimizer.zero_grad()
episode_number = 0
batch_size = 4
#opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
# Set the names for both SimpleAIs
env.set_names(myplayer.get_name(), opponent.get_name())
p_ups, fake_labels, rewards = [], [], []
rew_partial = []
win1 = 0
try:
    utils2.init_utils2()
except:
    print("Unable to remove old data!")

import os.path
if os.path.isfile(args.net):
    print("Network Loaded!")
    policy.load_state_dict(torch.load(args.net))
    policy.eval()

for it in range(1,episodes):
    i = 0
    (ob1, ob2) = env.reset()
    previous_obs = None #prev_f = torch.zeros(100, 100)
    done = False
    while not done:
        i += 1
        ob1_1 = myplayer.elaborate_frame(ob1)
        img = Image.fromarray(ob1[:,:,0])   # Method 1:
        img.save("ob0_"+str(i)+".png")
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        np.savetxt("data0_"+str(i)+".csv", ob1[:,:,0], delimiter=',')

        img = Image.fromarray(ob1[:,:,1])   # Method 1:
        img.save("ob1_"+str(i)+".png")
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        np.savetxt("data1_"+str(i)+".csv", ob1[:,:,1], delimiter=',')

        img = Image.fromarray(ob1[:,:,2])   # Method 1:
        img.save("ob2_"+str(i)+".png")
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        np.savetxt("data2_"+str(i)+".csv", ob1[:,:,2], delimiter=',')
        #exit()
        #action1, action1_prob = policy.forward2(diff_obs.to(device))
        action2 = opponent.get_action()
        obs = ob1_1 - previous_obs if previous_obs is not None else np.zeros_like(ob1_1)
        #np.set_printoptions(threshold=sys.maxsize)
        #print(obs)
        #exit()
        previous_obs = ob1_1
        
        obs = torch.from_numpy(obs).float().to(device)
        p_up = policy(obs)
        p_ups.append(p_up)
        action1 = 1 if np.random.uniform() < p_up.data[0] else 2
        y = 1.0 if action1 == 1 else 0.0 # fake label
        fake_labels.append(y)

        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        rewards.append(rew1/10)
        #if(it % 21 == 0):
        #    rewards.append(rew_partial)
        #    rewards = rewards[0]
            #print(rewards)
            #exit()

        if done and (it % 21 == 0):
            episode_number += 1
            eprewards = np.vstack(rewards)
            discounted_epr = myplayer.discount_rewards(eprewards)
            
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            lx = torch.stack(p_ups).float().to(device).squeeze(1)
            ly = torch.tensor(fake_labels).float().to(device)
            losses = loss_computator(lx, ly)
            t_discounted_epr = torch.from_numpy(discounted_epr).squeeze(1).float().to(device)
            losses *=  t_discounted_epr                
            loss = torch.mean(losses)         
            reward_sum = sum(rewards)
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            if episode_number % 10 == 1: 
                print("EPNUM: {0}, LOSS: {1}. REWARDS: {2} RUNNING_REWARDS: {3}".format(episode_number, loss, reward_sum, running_reward))
            loss.backward(torch.tensor(1.0/batch_size).to(device))
            if episode_number % batch_size == 0 and episode_number > 0:                        
                optimizer.step()
                optimizer.zero_grad()
            
            utils2.save_mean_value(reward_sum)
            utils2.save_rew(running_reward)
            p_ups, fake_labels, rewards = [], [], [] # reset
            observation = env.reset() # reset env
            prev_x = None
            print("episode {} over. Broken WR: {:.3f}, Timesteps: {},".format(episode_number, win1/(it+1), i))
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        
    if episode_number % 100 == 0:
        a = "params" + str(episode_number) + ".ckpt"
        torch.save(policy.state_dict(), a)
env.close()