from wimblepong import Wimblepong
import torch
import sys
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import pickle

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100*100, 256)
        self.fc2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logp = self.fc2(x)
        return torch.sigmoid(logp)



class Agent(object):
    def __init__(self, env, policy, player_id=2):
        self.env = env
        self.player_id = player_id
        #self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_device = torch.device("cpu")
        self.policy = policy.to(self.train_device)
        #self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.name = "BombaPong *_*"
        self.saved_agent = "./agent.pkl" # Name of the file to save the model on
        self.model = [] # The model trained from our network
        self.gamma = 0.99

    def get_name(self):
        return self.name

    def load_model(self):
        filename = self.saved_agent
        #self.model = pickle.load(open(filename, 'rb'))
        if(os.path.exists(filename)):
            net.load_state_dict(torch.load(filename))

    def reset(self):
        return 0

    
    def elaborate_frame(self, f):
        f = f[::2,::2,0]
        f[f == 43] = 0
        f[f == 195] = 0.7    # TODO: check if the values are correct
        f[f == 120] = 0.7
        f[f == 255] = 1
        #torch.from_numpy(prev_f.astype(np.float32).ravel()).unsqueeze(0)
        return f.astype(np.float).ravel()
        # DEBUG
        #img = Image.fromarray(f)   # Method 1:
        #img.save("ob2.png")
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        #print(f)
        #exit()
        #img = Image.fromarray(prev_f)   # Method 1:
        #img.save("ob1.png")
        # From numpy to tensor - TODO: try method 2
        #f = torch.from_numpy(f.astype(np.float32))#.  reshape(1,1,100,100))
        # ff = torch.from_numpy(f).float() # method 2
        #print(prev_f.size())
        #print(f.size())
        #exit()

        
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        #print(torch.cat([f, prev_f], dim=1).shape)
        #exit()
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def get_action(self, frame):
        #print(frame.shape)
        # Convert the image in grayscale
        #frame = tv.transforms.Grayscale(frame)
            # Keep only the first layer
        #195 120 43
        frame = self.elaborate_frame(frame)
        #print(frame.shape)
        #img = Image.fromarray(frame)
        #img.save("ob2.png")
        
        #print(frame)
        #print(frame.toTensor)
        
        #print(frame.shape)
        #exit()
        values = self.policy.forward(frame.to(self.train_device))
        print(values)
        return np.random.randint(1, high=3)

    def save_model(self):
        # Function to call at the end of the training 
        model = self.model
        filename = self.saved_agent
        torch.save(net.state_dict(), filename)