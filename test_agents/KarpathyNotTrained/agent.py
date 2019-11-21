import numpy as np
import pickle


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 43] = 0 # erase background (background type 1)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent(object):
    def __init__(self):
        self.H = 200  # number of hidden layer neurons
        self.D = 100 * 100  # input dimensionality: 100x100 grid
        self.prev_x = None
        self.model = {}
        self.init_model()
        self.name = "KarpathyRaw"
        self.rewards = []
        self.model_file = "save.p"

    def get_name(self):
        return self.name

    def load_model(self):
        self.model = pickle.load(open(self.model_file, 'rb'))

    def init_model(self):
        self.model.clear()
        self.model['W1'] = np.random.randn(self.H, self.D) / np.sqrt(self.D)
        self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def get_action(self, observation):
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, _ = self.policy_forward(x)
        action = 1 if np.random.uniform() < aprob else 2  # roll the dice!
        return action

    def reset(self):
        # Reset previous observation
        self.prev_x = None

