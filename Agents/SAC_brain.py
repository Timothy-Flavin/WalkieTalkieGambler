import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Normal
import copy

import numpy as np
import matplotlib.pyplot as plt

from brain import brain

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action, max_std,device):
    super(Actor, self).__init__()
    self.l1 = nn.Linear(state_dim, 256)
    self.l2 = nn.Linear(256, 256)
    self.l3 = nn.Linear(256, action_dim)
    self.l4 = nn.Linear(256, action_dim)
    self.max_action = max_action
    self.max_std = max_std
    self.to(device)
    self.device=device

  def forward(self, state):
    a = F.relu(self.l1(state))
    a = F.relu(self.l2(a))
    return self.l3(a), self.l4(a)


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim, device):
    super(Critic, self).__init__()
    # Q1 architecture
    self.l1 = nn.Linear(state_dim + action_dim, 256)
    self.l2 = nn.Linear(256, 256)
    self.l3 = nn.Linear(256, 1)
    # Q2 architecture
    self.l4 = nn.Linear(state_dim + action_dim, 256)
    self.l5 = nn.Linear(256, 256)
    self.l6 = nn.Linear(256, 1)
    self.to(device)
    self.device=device
  
  def forward(self, state, action):
    sa = torch.cat([state, action], 1)

    q1 = F.relu(self.l1(sa))
    q1 = F.relu(self.l2(q1))
    q1 = self.l3(q1)

    q2 = F.relu(self.l4(sa))
    q2 = F.relu(self.l5(q2))
    q2 = self.l6(q2)
    return q1, q2


  def Q1(self, state, action):
    sa = torch.cat([state, action], 1)

    q1 = F.relu(self.l1(sa))
    q1 = F.relu(self.l2(q1))
    q1 = self.l3(q1)
    return q1

class SAC_Brain(brain):
  def __init__(self, name, anum, state_dim, action_dim, vec, device='cpu', trade_off=0.1, max_action=1, max_std=1, update_every=1,gamma=0.99, tau=0.995, filepath="sac"):
    super(SAC_Brain,self).__init__(name,anum)
    print(state_dim)
    self.file_name = filepath
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.vec = vec
    self.trade_off = trade_off
    self.max_action = max_action
    self.max_std = max_std
    self.up_num = 0
    self.gamma = gamma
    self.update_every = update_every
    self.LOG_STD_MAX = 2
    self.LOG_STD_MIN = -20
    self.tau=tau
    self.device = device
    
    self.actor = Actor(state_dim,action_dim,max_action,max_std,device=device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
    
    self.critic = Critic(state_dim,action_dim,device=self.device)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    self.critic_target = copy.deepcopy(self.critic)
    
    
    self.buffer = ReplayBuffer(state_dim,action_dim, 500000)
    self.norm = Normal(torch.zeros(self.action_dim),torch.ones(self.action_dim))

  def action(self,state,anum):
    vec_state = self.vec(state,self.anum,True)
    with torch.no_grad():
      #print(torch.FloatTensor(vec_state)[None,:].to(self.device))
      prob_means,prob_log_stds = self.actor(torch.FloatTensor(vec_state)[None,:].to(self.device))
      prob_log_stds = torch.clamp(prob_log_stds,self.LOG_STD_MIN,self.LOG_STD_MAX)
      prob_stds = torch.exp(prob_log_stds)

      #var_mat = torch.square(prob_stds)#.unsqueeze(dim=0)
      dist = Normal(prob_means, prob_stds)
      action = (self.max_action * torch.tanh(dist.sample())).to('cpu').numpy() #TODO make this not be so ugly
      act = np.zeros((1,14+3))
      act[0,0:2] = action
    return act
  
  def q_loss(self, states,actions,states_,rewards,dones):
    Q1,Q2 = self.critic(states,actions)
    with torch.no_grad():
      #Get the distribution
      actions_, act_log_prob = self.__action__(states_)
      target_Q1, target_Q2 = self.critic_target(states_, actions_)
      target_Q = torch.min(target_Q1, target_Q2)
      #target_Q = reward + not_done * self.discount * target_Q
      target = rewards+self.gamma*(1-dones)*(target_Q-self.trade_off*act_log_prob)
    critic_loss = ((Q2 - target)**2).mean() + ((Q1 - target)**2).mean()
    return critic_loss

  def a_loss(self, states, actions, states_, rewards, dones):
    actions,log_probs = self.__action__(states)
    Q1,Q2 = self.critic(states,actions)
    Q = torch.min(Q1,Q2)
    loss = (self.trade_off*log_probs - Q).mean()
    return loss

  def __action__(self,state):
    prob_means,prob_log_stds = self.actor(state)
    prob_log_stds = torch.clamp(prob_log_stds,self.LOG_STD_MIN,self.LOG_STD_MAX)
    std = torch.exp(prob_log_stds)

    #var_mat = torch.square(prob_stds)#.unsqueeze(dim=0)
    dist = Normal(prob_means, std)
    action = dist.rsample()

    logp_pi = dist.log_prob(action).sum(axis=-1)
    logp_pi -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
    return self.max_action*torch.tanh(action), logp_pi
  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    self.up_num +=1
    self.vec(state,anum,True)
    self.buffer.add(self.vec(state,anum,True),action[0:self.action_dim],self.vec(state_,anum,True),rewards,int((terminated or truncated)))
    
    if self.up_num%self.update_every==0:
      states,actions,states_,rewards,dones = self.buffer.sample(256)
      self.up_num = 0
      
      critic_loss = self.q_loss(states,actions,states_,rewards,dones)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      #update the critic
      loss = self.a_loss(states,actions,states_,rewards,dones)
      self.actor_optimizer.zero_grad()
      loss.backward()
      self.actor_optimizer.step()

      # spinning up
      with torch.no_grad():
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
          # NB: We use an in-place operations "mul_", "add_" to update target
          # params, as opposed to "mul" and "add", which would make new tensors.
          p_targ.data.mul_(self.tau)
          p_targ.data.add_((1 - self.tau) * p.data)

  # for algorithms that can only update after an episode
  def update_end_of_episode(self):
    pass
  
  def save(self, filename):
    torch.save(self.critic.state_dict(), filename + "_critic")
    torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    torch.save(self.actor.state_dict(), filename + "_actor")
    torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


  def _load(self, filename):
    self.critic.load_state_dict(torch.load(filename + "_critic"))
    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    self.critic_target = copy.deepcopy(self.critic)

    self.actor.load_state_dict(torch.load(filename + "_actor"))
    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    #self.actor_target = copy.deepcopy(self.actor)

  # load this model from a checkpoint
  def load(self):
    self._load(f'./{self.file_name}/{self.name}')
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self):
    self.save(f'./{self.file_name}/{self.name}')

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)