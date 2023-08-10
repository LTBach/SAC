import os
import numpy as np
import joblib

class ReplayBuffer:
    def __init__(self, max_size, state_dims, action_dims, chkpt_dir='checkpoint'):

        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.state_memory = np.zeros((self.mem_size, state_dims))
        self.action_memory = np.zeros((self.mem_size, action_dims))
        self.reward_memory = np.zeros((self.mem_size))
        self.next_state_memory = np.zeros((self.mem_size, state_dims))
        self.done_memory = np.zeros((self.mem_size), dtype=np.bool_)
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'buffer')

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, next_states, dones
    
    def save_buffer(self):
        print('... saving buffer ...')
        joblib.dump(self, self.checkpoint_file)
    
    def load_buffer(self):
        print('... loading buffer ...')
        self = joblib.load(self.checkpoint_file)
