import torch
import torch.nn as nn
import torch.nn.functional as F
from brain import brain
from halsarc.Game.game import sar_env
import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
# define policy network
class sup_policy(nn.Module):
  def __init__(self, state_size, hidden_dims=[256,256], max_agents=5, m_types=8, max_instruction=5, device='cpu'): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
    super(sup_policy, self).__init__()
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
    
    xy = nn.functional.tanh(self.xy(x))
    send = nn.functional.sigmoid(self.send(x))
    dxy = nn.functional.tanh(self.dxy(x))
    mag = torch.mul(nn.functional.sigmoid(self.mag(x)),self.max_in)
    tp = nn.functional.softmax(self.message_type(x),1)
    targ = nn.functional.softmax(self.message_target(x),1)
    
    return xy,send,dxy,mag,tp,targ
  

class t_sup(brain):
  def __init__(self, state_size, anum,max_agents=3, m_types=8, 
               max_instruction=5, hidden_dims=[256,256], 
               directory="./torch_sup/", data_dir="./",
               batch_size = 32):
    
    super(t_sup,self).__init__('t_sup',anum)
    self.action_size = 2+1+2+1+m_types+max_agents
    if torch.cuda.is_available():
      self.device = torch.device('cuda:0')
    else:
      self.device = torch.device('cpu')
    self.policy = sup_policy(state_size,hidden_dims,max_agents,m_types,max_instruction,self.device)
    self.dir = directory
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.state_size = state_size
    self.hidden_dims = hidden_dims
    self.max_agents = max_agents
    self.m_types = m_types
    self.max_instruction = max_instruction
    self.policy_optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.00001)
    self.load()
  
  def action(self,state,anum):
    act = torch.cat(
      self.policy(torch.from_numpy(sar_env.vectorize_state(state,anum,True)).to(self.device)[None,:])
    ,1)
    return act.detach().cpu().numpy()
  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    pass
  # for algorithms that can only update after an episode
  def update_end_of_episode(self):
    pass
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self):
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)
    torch.save(self.policy,self.dir+f"policy.pkl")
  # load this model from a checkpoint
  def load(self):
    try:
      pa = torch.load(self.dir+"policy.pkl")
      #print(pa)
      self.policy=pa
    except Exception as e:
      print(e)
      print(f"Failed to load net for: {self.dir}policy.pkl")
    

  def train(self, n_epochs):
    al = np.zeros((1,self.action_size))
    try:
      al = np.load(self.data_dir+"_action_record.npy")
    except:
      print("No action record found for: " + self.data_dir)

    fileanum = 0
    try:
      fileanum = np.load(self.data_dir+"_anum.npy")[0]
    except:
      print("No anum found for: " + self.data_dir)

    rw = np.zeros((1,1))
    try:
      rw = np.load(self.data_dir+"_reward_record.npy")
    except:
      print("No reward record found for: " + self.data_dir)

    oldstate = []
    try:
      filehandler = open(self.data_dir+"_state_record.pkl", 'rb') 
      oldstate = pickle.load(filehandler)
    except:
      print(f"no old state dump found at {self.data_dir}")
      Exception("Could not load training data")

    print(f"reward shape: {rw.shape}")
    print(f"action shape: {al.shape}")
    print(f"num states: {len(oldstate)}")
    print(f"File agent num: {fileanum}")
    np.set_printoptions(precision=2)
    #for i in oldstate:
      #print(i['view'][fileanum])
      #input()
    idx = np.arange(len(oldstate))
    np.random.shuffle(idx)
    s1 = sar_env.vectorize_state(oldstate[0],fileanum,True)
    
    st = np.zeros((len(oldstate),s1.shape[0]))
    for i in range(len(oldstate)):
      st[i] = sar_env.vectorize_state(oldstate[i],fileanum,True)
    print(idx)

    al = torch.from_numpy(al).to(self.device)
    rw = torch.from_numpy(rw).to(self.device)
    st = torch.from_numpy(st).to(self.device)
    mv_losses = []
    send_losses=[]
    dmv_losses = []
    mag_losses = []
    mtp_losses=[]
    targ_losses = []
    for epoch in range(n_epochs):
      idx = np.arange(len(oldstate))
      np.random.shuffle(idx)
      for i in range(0,len(idx),self.batch_size):
        self.policy.zero_grad()
        sampis = idx[i:min(i+self.batch_size,len(oldstate)-1)]
        xy,send,dxy,mag,tp,targ = self.policy(st[sampis])
        
        deb = False
        if deb:
          print("\nBefore: ")
          print(f"xy: {xy[0].to('cpu').detach().numpy()}, \
  send: {send[0].to('cpu').detach().numpy()}, \
  dxy: {dxy[0].to('cpu').detach().numpy()}, \
  mag: {mag[0].to('cpu').detach().numpy()}, \
  tp: {tp[0].to('cpu').detach().numpy()}, \
  targ: {targ[0].to('cpu').detach().numpy()}")
          print(al[sampis].to('cpu').detach().numpy()[0])
          print(xy[0])

        #print(al[sampis])
        torch.square(xy-al[sampis,0:2]).sum().backward(retain_graph=True)
        torch.square(send-al[sampis,2]).sum().backward(retain_graph=True)
        torch.square(dxy-al[sampis,3:5]).sum().backward(retain_graph=True)
        torch.square(mag-al[sampis,5]).sum().backward(retain_graph=True)
        F.cross_entropy(tp,al[sampis,6:14]).sum().backward(retain_graph=True)
        F.cross_entropy(targ,al[sampis,14:14+self.max_agents]).sum().backward()
        self.policy_optimizer.step()

        xy,send,dxy,mag,tp,targ = self.policy(st[sampis])

        if deb:
          print("After: ")
          print(f"xy: {xy[0].to('cpu').detach().numpy()}, \
  send: {send[0].to('cpu').detach().numpy()}, \
  dxy: {dxy[0].to('cpu').detach().numpy()}, \
  mag: {mag[0].to('cpu').detach().numpy()}, \
  tp: {tp[0].to('cpu').detach().numpy()}, \
  targ: {targ[0].to('cpu').detach().numpy()}")
          print(al[sampis].to('cpu').detach().numpy()[0])
          print(xy[0])

      self.policy.zero_grad()
      sampis = idx[0:self.batch_size]
      xy,send,dxy,mag,tp,targ = self.policy(st[sampis])

      #print(xy[0])
      #print(al[sampis])
      mv_losses.append(torch.square(xy-al[sampis,0:2]).sum().to('cpu').item()/self.batch_size)
      send_losses.append(torch.square(send-al[sampis,2]).sum().to('cpu').item()/self.batch_size)
      dmv_losses.append(torch.square(dxy-al[sampis,3:5]).sum().to('cpu').item()/self.batch_size)
      mag_losses.append(torch.square(mag-al[sampis,5]).sum().to('cpu').item()/self.batch_size)
      mtp_losses.append(F.cross_entropy(tp,al[sampis,6:14]).sum().to('cpu').item()/self.batch_size)
      targ_losses.append(F.cross_entropy(targ,al[sampis,14:14+self.max_agents]).to('cpu').item()/self.batch_size)
      #self.policy()
    #np.random.choice(len(oldstate),len(oldstate),replace=False)
    plt.plot(mv_losses, label="XY")
    plt.plot(send_losses, label="Send")
    plt.plot(dmv_losses, label="d XY")
    #plt.plot(mag_losses, label="mag")
    plt.plot(mtp_losses, label="m_type")
    plt.plot(targ_losses, label="target")
    plt.grid()
    plt.legend()
    plt.title(f"loss over time for {self.data_dir}")
    plt.show()

if __name__=="__main__":
  agents = ["RoboDog","Human","Drone"]
  fname = "./recorded_data/"+agents[0]+"_state_record.pkl"
  state_shape = 0
  try:
    filehandler = open(fname, 'rb') 
    print(f"found {fname}")
    states=(pickle.load(filehandler))
    state_shape = sar_env.vectorize_state(states[0],0,True).shape[0]
  except:
    print(f"Could not load agent at '{fname}'")
  print(state_shape)
  for i,a in enumerate(agents):
    tsup = t_sup(anum = i,state_size=state_shape,data_dir=f"./recorded_data/{a}",directory=f"./torch_sup/{a}/")
    tsup.train(100)
    tsup.checkpoint()