import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
    x = torch.sigmoid(self.out(x)) # leads to numbers between 0 and 1
    return x
  





import numpy as np
import pickle
import sys

print(sys.argv)

agent=sys.argv[1]

har = np.load("../Game/"+agent+"_action_record.npy")
hrr = np.load("../Game/"+agent+"_reward_record.npy")
filename = "../Game/"+agent+"_state_record.pkl"
hsr=pickle.load(open('../Game/'+agent+'_state_record.pkl','rb'))
agents = pickle.load(open('../Game/agents.pkl','rb'))

# torch usually uses a 1-hot coding so like class = 3 would be [0,0,0,1,0,0,0,0,0]
# but we want a probability distribution where 3 is more likely than the rest but
# we want the distribution to be pretty random still so that the model explores 

def action_to_class(action):
  w = 0.3
  o = (1-0.3)/8
  if action[0]==0 and action[1]==0:
    return np.array([w,o,o,o,o,o,o,o,o]) # none
  if action[0] == 0 and action[1] < 0:
    return np.array([o,w,o,o,o,o,o,o,o]) # w
  if action[0] > 0 and action[1] < 0:
    return np.array([o,o,w,o,o,o,o,o,o]) # wd
  if action[0] > 0 and action[1] == 0:
    return np.array([o,o,o,w,o,o,o,o,o]) # d
  if action[0] > 0 and action[1] > 0:
    return np.array([o,o,o,o,w,o,o,o,o]) # ws
  if action[0] == 0 and action[1] > 0:
    return np.array([o,o,o,o,o,w,o,o,o]) # s
  if action[0] < 0 and action[1] > 0:
    return np.array([o,o,o,o,o,o,w,o,o]) # as
  if action[0] < 0 and action[1] == 0:
    return np.array([o,o,o,o,o,o,o,w,o]) # a
  if action[0] < 0 and action[1] < 0:
    return np.array([o,o,o,o,o,o,o,o,w]) # aw


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

print("entering loop")
data = {}

actions = np.array([action_to_class(x) for x in har],dtype=np.float32) # all the actions for agent a
print(actions.shape)
states = np.array([np.concatenate((h['view'].flatten(),h['object_state'].flatten(),h['memory'].flatten())) for h in hsr],dtype=np.float32) # all the states agent a saw
rewards = hrr 
print(rewards.shape)

data = { 'actions':actions[rewards!=0], 'states':states[rewards!=0,:] }
print(f"Actions Shape: {data['actions'].shape}")
print(f"States Shape: {data['states'].shape}")
#input()


net = policy_net(data["states"].shape[1], 32, 9).to(device)

print(f"Model structure: {net}\n\n")
#for name, param in net.named_parameters():
#  print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

train = torch.utils.data.TensorDataset(torch.from_numpy(data['states']), torch.from_numpy(data['actions']))
train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

loss_arr = []
n_epochs = 10
for e in range(n_epochs):
  print(f"Epoch: {e} ")
  for i, (train_x, train_y) in enumerate(train_loader):
    pred = net.forward(train_x.to(device))
    loss = loss_fn(pred,train_y.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    l, current = loss.item(), (i + 1) * len(train_x)
    loss_arr.append(l)
    print(f"loss: {l:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")


torch.save(net.state_dict, f"./nets/{agent}")


import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()