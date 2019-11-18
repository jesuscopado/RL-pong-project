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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.gamma = 0.99
        self.eps_clip = 0.1

        self.layers = nn.Sequential(

            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            self.view()
            nn.Linear(6*6*64, 512), nn.ReLU(),
            nn.Linear(512, 2),
        )

    
    def forward2(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        # TODO: to debug and modify all
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits = logits)
                    #print(logits)
                    #exit()  
                    action = int(c.sample().cpu().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
            return action+1, action_prob
        # PPO
        vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])
        
        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts.to(device), dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss

class Agent(object):
    def __init__(self, env, policy, player_id=2):
        self.env = env
        self.player_id = player_id
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.train_device)
        #self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.name = "BombaPong *_*"
        self.saved_agent = "./agent.pkl" # Name of the file to save the model on
        self.model = [] # The model trained from our network

    def get_name(self):
        return self.name

    def load_model(self):
        filename = self.saved_agent
        #self.model = pickle.load(open(filename, 'rb'))
        if(os.path.exists(filename)):
            net.load_state_dict(torch.load(filename))

    def reset(self):
        return 0

    
    def elaborate_frame(self, f, prev_f):
        # TODO: prev_f has to be modified (temporary implementation)
        if prev_f is None:
            prev_f = torch.zeros(100, 100)
        else:
            prev_f = prev_f[::2,::2,0]
            prev_f[prev_f == 43] = 0
            prev_f[prev_f == 195] = 0.7
            prev_f[prev_f == 120] = 0.7
            prev_f[prev_f == 255] = 1
            prev_f = torch.from_numpy(prev_f.astype(np.float32))#.reshape(1,1,100,100))
        

        f = f[::2,::2,0]
        f[f == 43] = 0
        f[f == 195] = 0.7    # TODO: check if the values are correct
        f[f == 120] = 0.7
        f[f == 255] = 1

        # DEBUG
        #img = Image.fromarray(f)   # Method 1:
        #img.save("ob2.png")
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        #print(f)
        #exit()
        #img = Image.fromarray(prev_f)   # Method 1:
        #img.save("ob1.png")
        # From numpy to tensor - TODO: try method 2
        f = torch.from_numpy(f.astype(np.float32))#.  reshape(1,1,100,100))
        # ff = torch.from_numpy(f).float() # method 2
        #print(prev_f.size())
        #print(f.size())
        #exit()

        
        #np.set_printoptions(threshold=sys.maxsize) # Method 2:
        #print(torch.cat([f, prev_f], dim=1).shape)
        #exit()

        a = torch.stack((f, prev_f)).unsqueeze(0)
        #print(a.shape)
        #exit()
        return a

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