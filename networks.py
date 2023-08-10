import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.normal import Normal

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dims, fc1_dims, fc2_dims, action_dims, 
                 max_action, name, chkpt_dir='checkpoint'):
        super(ActorNetwork, self).__init__()
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.max_action = max_action
        self.name = name
        self.reparam_noise = 1e-6

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        
        mu = self.mu(action)
        sigma = self.sigma(action)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma
    
    def sample_normal(self, state):
        mu, sigma = self.forward(state)
        dist = T.distributions.Normal(mu, sigma)

        action = dist.rsample()
        bounded_action = F.tanh(action)*T.tensor(self.max_action).to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = log_prob - \
            T.log(1-bounded_action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(1, keepdim=True)

        return bounded_action, log_prob 

    def save_checkpoint(self):
        print('... saving checkppoint ...', self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...', self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dims, fc1_dims, fc2_dims, action_dims,
                 name, chkpt_dir='checkpoint'):
        super(CriticNetwork, self).__init__()
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims
        self.name = name

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.state_dims + self.action_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action_value = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        action_value = F.relu(self.fc2(action_value))
        
        action_value = self.action_value(action_value)
        
        return action_value
    
    def save_checkpoint(self):
        print('... saving checkpoint ...', self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...', self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))
