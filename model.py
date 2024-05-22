from torch import from_numpy
from net import Net
from numpy import zeros,sum,squeeze,array
from numpy.random import choice


class Model(object):
    """model0 of agents"""

    def __init__(self, env_id, device):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = Net().to(device)  # neural network

    def eval(self, observation, vector, valid_action, input_state,num_agent):
        """using neural network in training for prediction"""
        observation = from_numpy(observation).to(self.device)
        vector = from_numpy(vector).to(self.device)
        ps, _,  _, output_state,_ = self.network(observation, vector, input_state)
        actions = []
        ps = squeeze(ps.cpu().detach().numpy())
        for i in range(num_agent):
            try:
                valid_dist = array([ps[i, valid_action[i]]])
                valid_dist /= sum(valid_dist)
                actions.append(valid_action[i][choice(range(valid_dist.shape[1]), p=valid_dist.ravel())])
            except:
                actions.append(valid_action[i][choice(range(len(valid_action[i])))])
        return actions,output_state

    def restep(self, observation, vector, valid_action, input_state,num_agent):
        """using neural network in training for prediction"""
        observation = from_numpy(observation).to(self.device)
        vector = from_numpy(vector).to(self.device)
        ps, _,  _, output_state,_ = self.network(observation, vector, input_state)
        actions = zeros(num_agent)
        ps = squeeze(ps.cpu().detach().numpy(),axis=0)
        for i in range(num_agent):
            valid_dist = array([ps[i, valid_action[i]]])
            valid_dist /= sum(valid_dist)
            actions[i] = valid_action[i][choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
        return actions,output_state

    def set_weights(self, weights):
        """load global weights to local models"""
        self.network.load_state_dict(weights)
