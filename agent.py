import os
import numpy as np
import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, state_dims, action_dims, tau, env, 
                 gamma=0.99, max_size=1000000, layer1_size=256, 
                 layer2_size=256, batch_size=256, entropy_coef=0.2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_size, state_dims, action_dims)

        self.actor = ActorNetwork(alpha, state_dims, layer1_size, layer2_size,
                                  action_dims, env.action_space.high, 
                                  'actor')

        self.beh_critic_1 = CriticNetwork(beta, state_dims, layer1_size, layer2_size, 
                                          action_dims, 'beh_critic_1')
        
        self.beh_critic_2 = CriticNetwork(beta, state_dims, layer1_size, layer2_size,
                                          action_dims, 'beh_critic_2')
        
        self.tar_critic_1 = CriticNetwork(beta, state_dims, layer1_size, layer2_size, 
                                          action_dims, 'tar_critic_1')
        
        self.tar_critic_2 = CriticNetwork(beta, state_dims, layer1_size, layer2_size,
                                          action_dims, 'tar_critic_2')

        self.entropy_coeficient = entropy_coef
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        action, _ = self.actor.forward(T.tensor(np.array([observation]), 
                                    dtype=T.float).to(self.actor.device))

        return action.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states, dtype=T.float).to(self.beh_critic_1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.beh_critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.beh_critic_1.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.beh_critic_1.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.beh_critic_1.device)

        sample_beh_action_value_1 = self.beh_critic_1.forward(states, actions).view(-1)
        sample_beh_action_value_2 = self.beh_critic_2.forward(states, actions).view(-1)

        next_actions, log_probs = self.actor.sample_normal(next_states)

        tar_next_action_value_1 = self.tar_critic_1.forward(next_states, next_actions)
        tar_next_action_value_2 = self.tar_critic_2.forward(next_states, next_actions)

        tar_next_action_value = T.min(tar_next_action_value_1, tar_next_action_value_2)
        tar_next_action_value = tar_next_action_value.view(-1)
        tar_next_action_value[dones] = 0.0

        target = rewards + self.gamma * \
            (tar_next_action_value - self.entropy_coeficient * log_probs.view(-1))
        
        self.beh_critic_1.optimizer.zero_grad()
        self.beh_critic_2.optimizer.zero_grad()
        loss_critic_1 = F.mse_loss(sample_beh_action_value_1, target)
        loss_critic_2 = F.mse_loss(sample_beh_action_value_2, target)
        loss_critic = loss_critic_1 + loss_critic_2
        loss_critic.backward()
        self.beh_critic_1.optimizer.step()
        self.beh_critic_2.optimizer.step()

        actor_action, log_probs = self.actor.forward(states)
        actor_action_noise = actor_action - self.entropy_coeficient * log_probs

        actor_beh_action_value_1 = self.beh_critic_1.forward(states, actor_action_noise)
        actor_beh_action_value_2 = self.beh_critic_2.forward(states, actor_action_noise)

        actor_beh_action_value = T.min(actor_beh_action_value_1, actor_beh_action_value_2)

        self.actor.optimizer.zero_grad()
        loss_actor = self.entropy_coeficient * log_probs \
            - actor_beh_action_value
        loss_actor = T.mean(loss_actor)
        loss_actor.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau == None:
            tau = self.tau

        beh_critic_1_params = self.beh_critic_1.named_parameters()
        beh_critic_2_params = self.beh_critic_2.named_parameters()
        tar_critic_1_params = self.tar_critic_1.named_parameters()
        tar_critic_2_params = self.tar_critic_2.named_parameters()

        beh_critic_1_state_dict = dict(beh_critic_1_params)
        beh_critic_2_state_dict = dict(beh_critic_2_params)
        tar_critic_1_state_dict = dict(tar_critic_1_params)
        tar_critic_2_state_dict = dict(tar_critic_2_params)

        for name in tar_critic_1_state_dict:
            tar_critic_1_state_dict[name] = tau * tar_critic_1_state_dict[name] +\
                                        (1 - tau) * beh_critic_1_state_dict[name]

        for name in tar_critic_2_state_dict:
            tar_critic_2_state_dict[name] = tau * tar_critic_2_state_dict[name] +\
                                        (1 - tau) * beh_critic_2_state_dict[name]
            
        self.tar_critic_1.load_state_dict(tar_critic_1_state_dict)
        self.tar_critic_2.load_state_dict(tar_critic_2_state_dict)

    def save_model(self):
        self.actor.save_checkpoint()
        self.beh_critic_1.save_checkpoint()
        self.beh_critic_2.save_checkpoint()
        self.tar_critic_1.save_checkpoint()
        self.tar_critic_2.save_checkpoint()
        self.memory.save_buffer()

    def load_model(self):
        self.actor.load_checkpoint()
        self.beh_critic_1.load_checkpoint()
        self.beh_critic_2.load_checkpoint()
        self.tar_critic_1.load_checkpoint()
        self.tar_critic_2.load_checkpoint()
        self.memory.load_buffer()
    
