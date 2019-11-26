import numpy as np
import pickle


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 43] = 0  # erase background (background type 1)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Agent(object):
    def __init__(self):
        # hyperparameters
        self.H = 200  # number of hidden layer neurons
        self.D = 100 * 100  # input dimensionality: 100x100 grid
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.batch_size = 200  # (10 * 21 games) every how many episodes to do a param update?
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

        self.prev_x = None
        self.model = {}
        self.init_model()
        self.name = "Karpathy"
        self.rewards = []
        self.model_file = "PG_karpathy.p"
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []

        self.grad_buffer = {k: np.zeros_like(v) for k, v in
                            self.model.items()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

    def get_name(self):
        return self.name

    def load_model(self):
        self.model = pickle.load(open(self.model_file, 'rb'))

    def save_model(self):
        pickle.dump(self.model, open(self.model_file, 'wb'))

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

    def policy_backward(self, eph, epdlogp, epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        grad = {'W1': dW1, 'W2': dW2}
        for k in self.model:
            self.grad_buffer[k] += grad[k]  # accumulate grad over batch
        return {'W1': dW1, 'W2': dW2}

    def preprocess_obs(self, observation):
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x
        return x

    def get_action(self, x):
        # forward the policy network and sample an action from the returned probability
        aprob, h = self.policy_forward(x)
        action = 1 if np.random.uniform() < aprob else 2  # roll the dice!
        return action, aprob, h

    def episode_finished(self, episode_number):
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(self.xs)  # observations
        eph = np.vstack(self.hs)  # hidden states
        epdlogp = np.vstack(self.dlogps)  # action gradients
        epr = np.vstack(self.drs)  # rewards
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr, self.gamma).astype(np.float64)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        self.policy_backward(eph, epdlogp, epx)

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.batch_size == 0:
            for k, v in self.model.items():
                g = self.grad_buffer[k]  # gradient
                self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g ** 2
                self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

    def reset(self):
        # Reset previous observation
        self.prev_x = None

    def store_outcome(self, x, h, action, aprob, reward):
        # record various intermediates (needed later for backprop)
        self.xs.append(x)  # observation
        self.hs.append(h)  # hidden state

        y = 1 if action == 2 else 0  # a "fake label"
        self.dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        self.drs.append(reward)
