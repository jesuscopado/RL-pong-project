"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import torch
import torch.nn.functional as F

import numpy as np
import argparse
import wimblepong

from bombaAgent import Agent
import random
import data.utils2 as utils2


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--net", type=str, help="Scale of the rendered game", default="")
parser.add_argument("--number", type=int, help="Number of time of the same training", default=0)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
myplayer = Agent(device)

# Set the names for both SimpleAIs
env.set_names(myplayer.get_name(), opponent.get_name())
reward_sum_running_avg = None
win1 = 0

utils2.init_utils2()

v_reward_sum_running_avg, v_reward_sum = [], []

for it in range(0, episodes):
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for ep in range(10):    # Number of epochs
        game_end = False
        win1 = 0
        win2 = 0
        t = 0
        for ep_numb in range(10):    # TODO: try different episode_numbers

            myplayer.reset()    # Init the previous frame
            (ob1, ob2) = env.reset()
            done = False

            while not done:

                stack_ob = myplayer.preprocess(ob1)
                with torch.no_grad():
                    action1, action1_prob = myplayer.get_action(stack_ob)

                action2 = opponent.get_action()
                # Step the environment and get the rewards and new observations
                (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

                # Statistics
                d_obs_history.append(stack_ob)
                action_history.append(action1)
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
        v_reward_sum.append(reward_sum)
        v_reward_sum_running_avg.append(reward_sum_running_avg)
        print('Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (it, ep, t, action1, action1_prob, reward_sum, reward_sum_running_avg))

    R = 0
    discounted_rewards = []
    for r in reward_history[::-1]:
        if r != 0:
            R = 0  # scored/lost a point in pong, so reset reward sum
        R = r + myplayer.gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.FloatTensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
    
    print(len(action_history))
    # update policy
    for _ in range(5):  # TODO: Check if it this number is equal to the number of epochs
        n_batch = 2144  # 24576
        idxs = random.sample(range(len(action_history)), n_batch)
        d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0).to(device)
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs]).to(device)
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs]).to(device)
        advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs]).to(device)
        #advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()
            
        myplayer.policy.zero_grad()
        #loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        
        #PPO
        vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) # 3x3 Identity matrix
        action_batch_np = action_batch.cpu().numpy()
        ts = torch.FloatTensor(vs[action_batch_np]) # [n_actions, 3]
        ts_np = ts.cpu().numpy()

        logits = myplayer.policy.forward(d_obs_batch)
        r = torch.sum(F.softmax(logits, dim=-1) * ts.to(device), dim=1) / action_prob_batch
        loss1 = r * advantage_batch
        loss2 = torch.clamp(r, 1-myplayer.eps_clip, 1+myplayer.eps_clip) * advantage_batch
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)
        
        loss.backward()
        myplayer.optimizer.step()
    
        print('Iteration %d -- Loss: %.3f' % (it, loss))
    if it % 100 == 0:
        torch.save(myplayer.policy.state_dict(), "./data/params" + str(it) + ".ckpt")
        utils2.save_mean_value2(v_reward_sum_running_avg, args.number)
        utils2.save_rew2(v_reward_sum, args.number)
        v_reward_sum_running_avg, v_reward_sum = [], []

env.close()
