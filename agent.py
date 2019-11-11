from wimblepong import Wimblepong
import pickle

class Agent(object):
    def __init__(self, env, player_id=2):
        self.env = env
        self.player_id = player_id
        self.name = "BombaPong =_="
        self.saved_agent = "./agent.mdl" # Name of the file to save the model on
        self.model = [] # The model trained from our network

    def get_name(self):
        return self.name

    def load_model(self):
        filename = self.saved_agent
        self.model = pickle.load(open(filename, 'rb'))

    def reset(self):
        return 0

    def get_action(self):
        return 0

    def save_model(self):
        # Function to call at the end of the training 
        model = self.model
        filename = self.saved_agent
        pickle.dump(model, open(filename, 'wb'))