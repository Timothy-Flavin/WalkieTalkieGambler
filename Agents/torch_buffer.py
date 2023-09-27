import torch
import numpy as np


# Format following Machine learning with phil memory buffer 
class memory_buffer():
  def __init__(self, num_steps, action_size, state_size, device):
    self.device = device
    self.mem_size = num_steps
    self.states = torch.zeros((num_steps,state_size), device=device,dtype=torch.float64)
    self.states_ = torch.zeros((num_steps,state_size), device=device,dtype=torch.float64)
    self.actions = torch.zeros((num_steps,action_size), device=device,dtype=torch.float64)
    self.rewards = torch.zeros((num_steps), device=device,dtype=torch.float64)
    self.done = torch.zeros((num_steps), device=device,dtype=torch.float64)
    self.idx = 0
  
  def save_transition(self, state, action, reward, state_, done):
    idx = self.idx % self.mem_size
    self.states[idx] = torch.from_numpy(state).to(self.device)
    self.states_[idx] = torch.from_numpy(state_).to(self.device)
    self.actions[idx] = torch.from_numpy(action).to(self.device)
    self.rewards[idx] = reward
    self.idx+=1 
    self.done[idx] = done

  def sample_memory(self, batch_size):
    size = min(self.idx, batch_size)
    idx = torch.from_numpy(np.random.choice(self.idx%self.mem_size, size, replace=False)).to(self.device)
    return self.states[idx],self.actions[idx],self.rewards[idx],self.states_[idx], self.done[idx]
  
  