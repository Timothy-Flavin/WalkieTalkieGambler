# import dependencies
# Code Adapted from github https://github.com/lbarazza/VPG-PyTorch/tree/master
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from halsarc.Game.game import sar_env
from halsarc.Game.controllers import player_controller
# define policy network
class policy_net(nn.Module):
  def __init__(self, nS, nH, nA): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
    super(policy_net, self).__init__()
    self.h = nn.Linear(nS, nH)
    self.h2 = nn.Linear(nH, nH)
    self.out = nn.Linear(nH, nA)
  # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
  def forward(self, x):
    x = F.relu(self.h(x))
    x = F.relu(self.h2(x))
    x = F.softmax(self.out(x), dim=1)
    return x

def class_to_action(action):
  if action == 0: # none
    return np.array([0,0])
  if action == 1: # w
    return np.array([0,-1])
  if action == 2: # wd
    return np.array([1,-1])
  if action == 3: # d
    return np.array([1,0])
  if action == 4: # ds
    return np.array([1,1])
  if action == 5: # s
    return np.array([0,1])
  if action == 6: # sa
    return np.array([-1,1])
  if action == 7: # a
    return np.array([-1,0])
  if action == 8: # aw
    return np.array([-1,-1])
# create environment

def svectorize_state(state):
  return np.concatenate(
    (
      state['view'][0].flatten(),
      np.concatenate(
        (
          state['object_state'][0]["a_state"].flatten(),
          state['object_state'][0]["s_state"].flatten(),
          state['object_state'][0]["p_state"].flatten(),
         )
      ),
      state['memory'][0].flatten()
     )
    )

def vectorize_state(state,anum,radio=False):
  a1 = state['view'][anum].flatten()
  a2 = np.concatenate(
        (
          state['object_state'][anum]["a_state"].flatten(),
          state['object_state'][anum]["s_state"].flatten(),
          state['object_state'][anum]["p_state"].flatten(),
        )
      )
  a3 = state['memory'][anum].flatten()
  a4 = np.concatenate(
        (
          state['radio']["message"].flatten(),
          state['radio']["message_legality"][anum].flatten(),
          state['radio']["queue_status"][anum].flatten(),
          state['radio']['sender'][0],
          state['radio']['sender'][1]
        )
      )
  if not radio:
    a4 = np.zeros(a4.shape)
  return np.concatenate((a1,a2,a3,a4))

import sys
#sys.path.insert(1, '../Game') # lets us import game from another folder
agents = ["Human","RoboDog","Drone"]
max_agents = len(agents)
pois = ["Child", "Child", "Adult"]
premade_map = np.load("../LevelGen/Island/Map.npy")
env = sar_env(display=True, tile_map=premade_map, agent_names=agents, poi_names=pois,seed=random.randint(0,10000),player=0,explore_multiplier=0.005)
      #sar_env(display=True, tile_map=premade_map, agent_names=agents, poi_names=pois,player=0)
state, info = env.start()
print(state)

nn_state = vectorize_state(state,0)
print(state)
print(nn_state.shape)
terminated = False
# instantiate the policy
policies = {}
optimizers = {}

for a in agents:
  policies[a] = policy_net(nn_state.shape[0], 256, 9)
  try:
    pa = torch.load(f"./nets/{a}")
    print(pa())
    policies[a].load_state_dict(pa())
  except:
    print(f"Failed to load net for: {a}")
  optimizers[a] = torch.optim.Adam(policies[a].parameters(), lr=0.001)

# create an optimizer
# initialize gamma and stats
gamma=0.99
n_episode = 0
returns = deque(maxlen=100)
render_rate = 10 # render every render_rate episodes
while True:
  rewards = []
  actions = []
  states  = []
  # reset environment
  state, info = env.start()
  nn_state=[]

  for i,a in enumerate(agents):
    nn_state.append(
      vectorize_state(state,i)
    )
  nn_state = np.array(nn_state)
  while True:
    # render episode every render_rate epsiodes
    if n_episode%render_rate==0:
        env.display=True
    else:
        env.display=False
    # calculate probabilities of taking each action
    agent_actions = []
    np_actions = []
    for i,a in enumerate(agents):
      probs = policies[a](torch.tensor(nn_state[i,:]).unsqueeze(0).float())
      # sample an action from that set of probs
      sampler = Categorical(probs)
      agent_actions.append(sampler.sample())
      np_actions.append(agent_actions[-1].item())

    # use that action in the environment
    env_actions = []
    for act in np.array(np_actions):
      clact = np.zeros(14+max_agents)
      clact[0:2] = class_to_action(act)
      env_actions.append(clact)
    env_actions = np.array(env_actions)
    new_state, reward, done, _, info = env.step(env_actions)
    new_nn_state=[]
    for i,a in enumerate(agents):
      new_nn_state.append(
        vectorize_state(new_state,i)
      )# store state, action and reward
    new_nn_state = np.array(new_nn_state)
    states.append(new_nn_state)
    actions.append(agent_actions)
    rewards.append(reward)

    nn_state = new_nn_state
    if done:
      break

  for ai,a in enumerate(agents):
    # preprocess rewards
    rewardsi = np.array(rewards[ai])
    # calculate rewards to go for less variance
    R = torch.tensor([np.sum(rewardsi[i:]*(gamma**np.array(range(i, len(rewardsi))))) for i in range(len(rewardsi))])
    # or uncomment following line for normal rewards
    #R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    statesi = torch.tensor(states[ai]).float()
    actionsi = torch.tensor(actions[ai])

    # calculate gradient
    probs = policies[a](statesi)
    sampler = Categorical(probs)
    log_probs = -sampler.log_prob(actionsi)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
    pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
    # update policy weights
    optimizers[a].zero_grad()
    pseudo_loss.backward()
    optimizers[a].step()

    # calculate average return and print it out
    returns.append(np.sum(rewardsi))
    print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(n_episode, np.mean(returns)))
  n_episode += 1

# close environment
env.close()