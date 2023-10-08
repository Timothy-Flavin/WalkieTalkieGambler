import gymnasium as gym
from swig import *
import torch
from TD3_Brain import td3_brain
from ppo_agent import ppo_brain
from SAC_brain import SAC_Brain
import numpy as np
import matplotlib.pyplot as plt
import random
env = gym.make('CartPole-v1')
print(env.action_space.shape)
print(env.observation_space.shape)
observation, info = env.reset()
class sa():
    def __init__(self,ma):
        self.max_agents=ma

def vec(a,b,c):
    return a

def smooth(arr):
    l = int(len(arr)/10)
    arr = np.array(arr)
    narr = []
    for i in range(0,len(arr)-l):
        narr.append(np.mean(arr[i:i+l]))
    return narr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#brain = td3_brain(0,"drone",sa(-12),observation,vec,1,"./sanity", action_dim=2,max_act=2, eps=0,eps_decay=0.99)
#brain = ppo_brain(0,'drone',sa(-13),observation,vec,124,"./sanity2",action_dim=1)
brain = SAC_Brain(name="sac",anum=0,state_dim=4,action_dim=2,vec=vec,device=device,max_action=1,update_every=2)
ep_r = 0
tot_r = []

nep = 0
exp = 0.1
while nep < 100000:
    exp = (100000-nep)/2/100000
    #print(brain.action(observation,0)[0])
    nn_act = brain.action(observation,0)
    action = np.argmax(nn_act) # agent policy that uses the observation and info
    if random.random()<exp:
        action = env.action_space.sample()
    #print(action)
    observation_, reward, terminated, truncated, info = env.step(action)
    ep_r+=reward
    brain.update(0,observation,nn_act,reward,observation_,terminated,truncated,None)
    observation = observation_
    if terminated or truncated:
        print(nep)
        nep+=1
        tot_r.append(ep_r)
        print(ep_r)
        ep_r=0
        
        env.close()
        if nep %5000==0:
            env = gym.make('CartPole-v1',render_mode='human')
            plt.plot(smooth(tot_r))
            plt.show()
        else:
            env = gym.make('CartPole-v1')
        observation, info = env.reset()

env.close()