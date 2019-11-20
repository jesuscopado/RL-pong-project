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

#openai/baseline
#convolutional layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 0.99
        self.eps_clip = 0.1

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 128, kernel_size=3, stride=2)
        # Missing pooling layer (from matrix to vector)
        self.fc1 = nn.Linear(self.conv3.out_channels*3*3, 128)
        self.fc2 = nn.Linear(128, 3)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif type(m) is torch.nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight)

    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        self.prev_frame = None

    def get_name(self):
        return self.name

    def load_model(self):
        filename = self.saved_agent
        #self.model = pickle.load(open(filename, 'rb'))
        if(os.path.exists(filename)):
            net.load_state_dict(torch.load(filename))

    def reset(self):
        self.prev_frame = np.zeros((100, 100))
        return 0

    def elaborate_frame(self, f):
        f = f[::2,::2,2]
        f[f==58]=0  # Background set to 0
        f[f==97]=255    # Paddle 1 set to white
        f[f==84]=255    # Paddle 2 set to white

        return f

    def get_diff_frame(self, f):

        f_fin = self.elaborate_frame(f)

        diff_frame = f_fin - (self.prev_frame / 2)
        diff_frame[diff_frame==0] = -255 # Background set to black

        diff_frame = (diff_frame+255)/510 # Normalization

        self.prev_frame = f_fin    # Update of the previous frame

        return torch.from_numpy(diff_frame.astype(np.float32).ravel())
    '''
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
    '''
    def get_action(self, d_obs):
        with torch.no_grad():
            logits = self.layers(d_obs) # This line might give an error
            c = torch.distributions.Categorical(logits=logits)
            action = int(c.sample().cpu().numpy()[0])
            action_prob = float(c.probs[0, action].detach().cpu().numpy())
        return action, action_prob
    
        '''
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
        '''

    def save_model(self):
        # Function to call at the end of the training 
        model = self.model
        filename = self.saved_agent
        torch.save(net.state_dict(), filename)