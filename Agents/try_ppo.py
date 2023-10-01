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
from ppo_git import PPO
# define policy network

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

nn_state = sar_env.vectorize_state(state,0,True)
print(state)
print(nn_state.shape)
terminated = False
# instantiate the policy
policies = {}
optimizers = {}

for a in agents:
  policies[a] = PPO(nn_state.shape[0],2,0.01,0.01,0.99,10,0.01,True,0.6)

# create an optimizer
# initialize gamma and stats
gamma=0.99
n_episode = 0
returns = deque(maxlen=100)
render_rate = 100 # render every render_rate episodes
update_every = 100
rewards = []
while True:
  ts = 0
  state, info = env.start()
  nn_state=[]

  for i,a in enumerate(agents):
    nn_state.append(
      sar_env.vectorize_state(state,i,True)
    )
  nn_state = np.array(nn_state)
  ep_rew = np.zeros(3)
  while True:
    ts+=1
    # render episode every render_rate epsiodes
    if n_episode%render_rate==0:
        env.display=True
        for i,a in enumerate(agents):
          policies[a].save("./ppo/Run1")
    else:
        env.display=False
    # calculate probabilities of taking each action
    agent_actions = []
    np_actions = []
    for i,a in enumerate(agents):
      action = policies[a].select_action(torch.tensor(nn_state[i,:]).unsqueeze(0).float())
      #print(action)
      np_actions.append(action)

    # use that action in the environment
    env_actions = []
    for act in np_actions:
      #print(act)
      clact = np.zeros(14+max_agents)
      clact[0:2] = act
      env_actions.append(clact)
    env_actions = np.array(env_actions)
    #print(env_actions)
    #print(env_actions)
    new_state, reward, done, _, info = env.step(env_actions)
    #print(env_actions)
    #input()
    for i,a in enumerate(agents):
      policies[a].buffer.rewards.append(reward[i])
      policies[a].buffer.is_terminals.append(done)
      if ts % update_every == 0:
        policies[a].update()
        policies[a].decay_action_std(0.001, 0.1)
    ep_rew += reward
    if done or _:
      break
  n_episode+=1
  rewards.append(np.mean(ep_rew))
  print(f"Mean rew: {rewards[-1]}")   
  if len(rewards)>10:
    print(f"smooth mean: {np.mean(np.array(rewards[-10:]))}") 
    np.save("./ppo/rewards.npy",np.array(rewards))
# close environment
env.close()