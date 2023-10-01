import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from halsarc.Game.game import sar_env
from memory_buffer import memory_buffer
import os
import time
import matplotlib.pyplot as plt

# define policy network
class policy_net(nn.Module):
  def __init__(self, state_size, hidden_dims=[256,256], max_agents=5, m_types=8, max_instruction=5, device='cpu'): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
    super(policy_net, self).__init__()
    self.max_in = max_instruction
    self.hidden_layers = []
    self.hidden_layers.append(nn.Linear(state_size,hidden_dims[0]))
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
    
    self.xy = nn.Linear(hidden_dims[-1],2)
    self.send = nn.Linear(hidden_dims[-1],1)
    self.dxy = nn.Linear(hidden_dims[-1],2)
    self.mag = nn.Linear(hidden_dims[-1],1)
    self.message_type = nn.Linear(hidden_dims[-1],m_types)
    self.message_target = nn.Linear(hidden_dims[-1],max_agents)
    self.hidden_layers = nn.ModuleList(self.hidden_layers)
    self.double()
    self.to(device)
  # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
  def forward(self, x):
    #print(x.shape)
    for l in self.hidden_layers:
      #print(l(x).shape)
      x = nn.functional.relu(l(x))
    out = torch.cat((
      nn.functional.tanh(self.xy(x)),
      nn.functional.sigmoid(self.send(x)),
      nn.functional.tanh(self.dxy(x)),
      torch.mul(nn.functional.sigmoid(self.mag(x)),self.max_in),
      nn.functional.softmax(self.message_type(x),1),
      nn.functional.softmax(self.message_target(x),1)),1
    )
    return out
  
class value_net(nn.Module):
  def __init__(self, state_size, hidden_dims=[256,256], max_agents=5, m_types=8,max_instructions=5, device='cpu'): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
    super(value_net, self).__init__()
    self.input_size = state_size+2+1+2+1+m_types+max_agents
    self.max_in = 5
    self.hidden_layers = []
    self.hidden_layers.append(nn.Linear(self.input_size,hidden_dims[0]))
    for i in range(len(hidden_dims)-1):
      self.hidden_layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
    self.hidden_layers.append(nn.Linear(hidden_dims[-1],1))
    self.hidden_layers = nn.ModuleList(self.hidden_layers)
    self.double()
    self.to(device)
  # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
  def forward(self, x):
    for i,l in enumerate(self.hidden_layers):
      if i<len(self.hidden_layers)-1:
        x = nn.functional.relu(l(x))
      else:
        #print(f"Before last layer {x.shape}")
        x = l(x)
        #print(f"After last layer {x.shape}")
    return x.flatten()
  
class cnn(nn.Module):
  def __init__(self, dims=[13,13], device='cpu'): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
    super(cnn, self).__init__()
    self.conv1 = nn.Conv2d(3,out_channels=16,kernel_size=5,stride=2)
    self.conv2 = nn.Conv2d(16,out_channels=32,kernel_size=5,stride=2)
    self.conv3 = nn.Conv2d(32,out_channels=48,kernel_size=3,stride=1)

  # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    print(x.shape)
    return x.flatten()
class ddpgCNN():

  def vectorize_state(state,anum,radio=False):
    a2 = np.concatenate(
          (
            state['object_state'][anum]["a_state"].flatten(),
            state['object_state'][anum]["s_state"].flatten(),
            state['object_state'][anum]["p_state"].flatten(),
          )
        )
    #a3 = state['memory'][anum].flatten()
    a3 = np.concatenate(
          (
            state['radio']["message"].flatten(),
            state['radio']["message_legality"][anum].flatten(),
            state['radio']["queue_status"][anum].flatten(),
            state['radio']['sender'][0],
            state['radio']['sender'][1]
          )
        )
    if not radio:
      a3 = np.zeros(a3.shape)
    return np.concatenate((a2,a3)).astype(np.double)

  def __set_networks__(self,state_size,hidden_dims,max_agents,m_types,max_instruction,try_load=False):
    #policy
    self.policy = policy_net(state_size, hidden_dims,max_agents,m_types,max_instruction,self.device)
    try:
      pa = torch.load(os.path.join(self.dir,"policy.pkl"))
      #print(pa())
      self.policy=pa
    except Exception as e:
      print(e)
      print(f"Failed to load net for: {self.dir}policy.pkl")
    self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

    self.target_policy = policy_net(state_size, hidden_dims,max_agents,m_types,max_instruction,self.device)
    try:
      pa = torch.load(os.path.join(self.dir,"target_policy.pkl"))
      #print(pa)
      self.target_policy=pa
    except:
      self.__move_to_target__(self.policy,self.target_policy,1)
      print(f"Failed to load net for: {self.dir}target_policy.pkl")

    self.value = value_net(state_size,hidden_dims,max_agents,m_types,max_instruction,self.device)
    try:
      pa = torch.load(os.path.join(self.dir,"value.pkl"))
      #print(pa())
      self.value=pa
    except:
      print(f"Failed to load net for: {self.dir}value.pkl")
    self.value_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

    self.target_value = value_net(state_size,hidden_dims,max_agents,m_types,max_instruction,self.device)
    try:
      pa = torch.load(os.path.join(self.dir,"target_value.pkl"))
      #print(pa())
      self.target_value=pa
    except:
      self.__move_to_target__(self.value,self.target_value,1)
      print(f"Failed to load net for: {self.dir}target_value.pkl")

  def __init__(self, state_size, mv, max_agents=5, m_types=8, 
               max_instruction=5, hidden_dims=[256,256], 
               tau=0.95, directory="./ddpg/", eps=0.9,
               eps_decay=0.995, max_mem=30000,
               batch_size = 32, gamma=0.99, update_every=1000):
    self.max_view = mv
    self.dead = False
    self.gamma = gamma
    self.state_size = state_size
    self.hidden_dims = hidden_dims
    self.max_agents=max_agents
    self.m_types = m_types
    self.max_instruction = max_instruction
    self.update_num = 0

    print("making ddpg")
    if torch.cuda.is_available():
      self.device = torch.device('cuda:0')
    else:
      self.device = torch.device('cpu')
    self.net_names = ["policy","value","value_target"]
    self.tau = tau
    self.dir = directory
    self.networks = {}
    self.optimizers = {}
    self.eps = eps
    self.eps_decay = eps_decay
    self.action_size = 2+1+2+1+m_types+max_agents
    self.state = None
    self.batch_size = batch_size
    self.__set_networks__(state_size,hidden_dims,max_agents,m_types,max_instruction)
    self.cnn = cnn([mv,mv],self.device)
    self.memory = memory_buffer(max_mem,self.action_size,state_size, self.device)
    self.update_every = update_every

  def __rand_action__(self):
    mtype = np.zeros(8)
    mtype[random.randint(0,7)] = 1
    targ = np.zeros(self.max_agents)
    targ[random.randint(0,self.max_agents-1)] = 1
    ar = np.zeros((1,self.action_size))
    ar[0,0]=random.random()*2-1
    ar[0,1]=random.random()*2-1
    ar[0,2]=random.random()
    ar[0,3]=random.random()*2-1
    ar[0,4]=random.random()*2-1
    ar[0,5]=random.random()*self.max_instruction
    ar[0,6:14]=mtype
    ar[0,14:14+self.max_agents]=targ
    #print(ar.shape)
    return ar
  # This is for the environment to get an action from ddpg
  def action(self, state, anum):
    act=0
    if random.random()<self.eps:
      return self.__rand_action__()
    
    view = torch.from_numpy(state['view'][anum])[None,:,:,:].to(self.device)
    agent_state = torch.from_numpy(self.vectorize_state(state,anum,True))[None,:].to(self.device)
    v = self.cnn(view)
    print(f"view shape: {view.shape}")
    print(f"encoded shape: {v.shape}")
    nn_state = torch.cat((
      agent_state,
      v
    ),1)
    with torch.no_grad():
      return self.policy(nn_state).detach().cpu().numpy()

  def load(self):
    self.__set_networks__(self.state_size,self.hidden_dims,self.max_agents,self.m_types,self.max_instruction)
  
  def __move_to_target__(self, net, target_net, beta):
   #The interpolation parameter    
    target_params = target_net.state_dict().items()
    net_params = net.state_dict()
    for name1, param1 in target_params:
        if name1 in net_params:
            net_params[name1].data.copy_(beta*param1.data + (1-beta)*net_params[name1].data)

    target_net.load_state_dict(net_params)

  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    self.update_num+=1
    if not self.dead or terminated or truncated:
      st = sar_env.vectorize_state(state,anum,True)
      st_ = sar_env.vectorize_state(state_,anum,True)
      rw = rewards
      done = int(terminated or truncated)
      self.memory.save_transition(st,action,rw,st_,done)

    if not self.update_num % self.update_every==0:
      return
    self.eps*=self.eps_decay
    states,actions,rewards,states_,dones = self.memory.sample_memory(self.batch_size)
    states = torch.from_numpy(states).to(device=self.device)
    #print(f"States.shape: {states.shape}")
    #print(f"Actions.shape: {actions.shape}")
    states_ = torch.from_numpy(states_).to(device=self.device)
    dones = torch.from_numpy(dones).to(device=self.device)
    rewards = torch.from_numpy(rewards).to(device=self.device)
    actions = torch.from_numpy(actions).to(device=self.device)

    self.value.zero_grad()
    #print(f"gamma: {self.gamma}")
    tnet = self.gamma*(1-dones)*self.target_value(
        torch.cat(
          (states_,self.target_policy(states_)),1
        )
      )
    #print(f"tnet: {tnet.shape}")
    value_target = (
      rewards
      + tnet
    )

    #print(states.shape)
    #print(actions.shape)
    #print(value_target.shape)
    #lf = torch.nn.MSELoss().to(self.device)
    loss = torch.square(self.value(
      torch.cat(
        (states,actions),1
      )
    )- value_target)
    #print(loss.shape)
    #input()
    loss.sum().backward()
    self.value_optimizer.step()
    self.value.zero_grad()
    self.policy.zero_grad()

    policy_loss = - self.value(torch.cat((states,self.policy(states)),1))
    policy_loss.sum().backward()
    self.policy_optimizer.step()

    self.__move_to_target__(self.policy,self.target_policy,self.tau)
    self.__move_to_target__(self.value,self.target_value,self.tau)
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self, agent):
    dir = f"./ddpg/{agent}/"
    if not os.path.exists(dir):
      os.makedirs(dir)
    torch.save(self.policy,f"{dir}policy.pkl")
    torch.save(self.target_policy,f"{dir}target_policy.pkl")
    torch.save(self.value,f"{dir}value.pkl")
    torch.save(self.target_value,f"{dir}target_value.pkl")

if __name__ == "__main__":
  player_num = 1
  agents = ["Human","RoboDog","Drone"]
  max_agents = len(agents)
  pois = ["Child", "Child", "Adult"]
  premade_map = np.load("../LevelGen/Island/Map.npy")
  env = sar_env(display=True, tile_map=premade_map, agent_names=agents, poi_names=pois,seed=random.randint(0,10000),player=player_num,explore_multiplier=0.005)
  state, info = env.start()
  print(f"Message shape: {state['radio']['message'].shape}")
  st = sar_env.vectorize_state(state,0,True)
  print(f"state shape: {st.shape}")

  terminated = False
  # instantiate the policy
  brains = {}
  for a in agents:
    brains[a] = ddpg(st.shape[0],5,8,5,[128,128],0.95,f"./ddpg/{a}/",batch_size=32,update_every=64,eps=0.90,eps_decay=0.99995)

  # create an optimizer
  # initialize gamma and stats
  gamma=0.99
  n_episode = 0
  render_rate = 500 # render every render_rate episodes
  envrew = []
  try:
    envrew = list(np.load("./ddpg/rewards.npy"))
  except:
    envrew = []
  
  while True:
    first_start = time.time()
    n_episode+=1
    if n_episode%render_rate==0:
      #plt.plot(envrew)
      #plt.title("rewards over time")
      #plt.show()
      np.save("./ddpg/rewards.npy",np.array(envrew))
      for a in agents:
        print("Checkpoint")
        brains[a].checkpoint(a)
      env.display=True
    else:
      env.display=False
    action_time = 0
    env_time = 0
    update_time = 0
    tot_r = np.zeros(3)
    # reset environment
    state, info = env.start()
    while True:
      # render episode every render_rate epsiodes
      
      # calculate probabilities of taking each action
      start = time.time()
      np_actions = []
      for i,a in enumerate(agents):
        brains[a].dead = info.agents[i].destroyed
        if brains[a].dead:
          np_actions.append(np.zeros((1,brains[a].action_size)))
          #print(np_actions[-1].shape)
        else:
          np_actions.append(brains[a].action(state,i))
          #print(np_actions[-1].shape)
          
      np_actions = np.array(np_actions)
      #print(np.max(np_actions[player_num]))
      #print(np_actions[player_num][0:2])
      
      np_actions = np_actions.reshape((np_actions.shape[0],np_actions.shape[2]))
      #print(np_actions[0])
      #input()
      action_time += time.time()-start
      #print(f"action shape: {np_actions.shape}")
      # use that action in the environment
      start = time.time()
      new_state, reward, done, _, info = env.step(np_actions)
      tot_r += reward
      env_time+=time.time()-start

      start = time.time()
      new_nn_state=[]
      for i,a in enumerate(agents):
        new_nn_state.append(
          sar_env.vectorize_state(new_state,i,True)
        )# store state, action and reward
        #print(f"new state shape: {new_nn_state[0].shape}")
        brains[a].update(i,state,np_actions[i],reward[i],new_state,done,_,env)
      update_time += time.time()-start
      if done:
        break
    tot = time.time() - first_start
    print(brains[agents[0]].eps)
    envrew.append(np.mean(tot_r))
    print(f"rewards {tot_r}, env_time {100*env_time/tot:.2f}%, {env_time:.2f}s, action_time: {100*action_time/tot:.2f}%, {action_time:.2f}s, update_time: {100*update_time/tot:.2f}%, {update_time:.2f}s other: {100*(tot-env_time-update_time-action_time)/tot:.2f}%, total: {tot}")

