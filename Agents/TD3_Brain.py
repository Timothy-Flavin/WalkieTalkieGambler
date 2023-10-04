import numpy as np
import torch
from brain import brain
from halsarc.Game.game import sar_env
import os
import numpy as np
from TD3 import TD3
# define policy network

class td3_brain(brain):
  def __init__(self, anum,a, sar, state, vectorizer = sar_env.vectorize_state, update_every=32, fname='ppo_brain',action_dim=2,max_act=1):
    super(td3_brain,self).__init__('ppo_brain',anum)
    self.fname= fname
    self.update_every = update_every
    self.frame = 0
    self.num_ac = 0
    self.num_up = 0
    self.a_type = a
    self.sar = sar
    self.vectorizer = vectorizer
    nn_state = self.vectorizer(state,anum,True)
    self.td3 = TD3(nn_state.shape[0],action_dim=action_dim,max_action=max_act,discount=0.997,
                   tau=0.005,policy_noise=0.2,noise_clip=0.5,policy_freq=2)
    self.buffer = ReplayBuffer(nn_state.shape[0],action_dim,200000)

  def action(self,state,anum):
    self.num_ac+=1
    act = np.zeros((1,14+self.sar.max_agents))
    act[0,0:2] = self.td3.select_action(self.vectorizer(state,anum,True))
    return act
  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    self.frame+=1
    self.num_up+=1
    self.buffer.add(self.vectorizer(state,anum,True),action[0:2],self.vectorizer(state_,anum,True),rewards,(terminated or truncated))
        
    if self.frame>=self.update_every:
      self.td3.train(self.buffer,256)
      self.frame=0
  # for algorithms that can only update after an episode
  def update_end_of_episode(self):
    pass
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self):
    self.td3.save(f"./{self.fname}/{self.a_type}/policy")
  # load this model from a checkpoint
  def load(self):
    self.td3.load(f"./{self.fname}/{self.a_type}/policy")
    
if __name__ == "__main__":
  print()


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)